"""OpenCode agent: writes opencode.json with two Databricks-backed providers."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Mapping
from typing import cast

from ucode.agent_updates import available_npm_package_update
from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_json_safe,
    write_json_file,
)
from ucode.databricks import (
    TOKEN_REFRESH_INTERVAL_SECONDS,
    build_opencode_base_urls,
    get_databricks_token,
)
from ucode.policies import endpoint_vanity_name, resolve_policy_default_model
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version

OPENCODE_XDG_CONFIG_HOME = APP_DIR / "opencode-xdg"
OPENCODE_CONFIG_DIR = OPENCODE_XDG_CONFIG_HOME / "opencode"
OPENCODE_CONFIG_PATH = OPENCODE_CONFIG_DIR / "opencode.json"
OPENCODE_BACKUP_PATH = APP_DIR / "opencode-config.backup.json"
# OpenCode reads stored credentials from `$XDG_DATA_HOME/opencode/auth.json`.
# We point XDG_DATA_HOME at a ucode-managed dir (build_runtime_env) so we own
# that file too.
OPENCODE_XDG_DATA_HOME = APP_DIR / "opencode-xdg-data"
OPENCODE_AUTH_PATH = OPENCODE_XDG_DATA_HOME / "opencode" / "auth.json"
OPENCODE_MCP_AUTH_HEADER_VALUE = "Bearer {env:OAUTH_TOKEN}"
OPENCODE_OPEN_RESPONSES_PLUGIN_MARKER = "ucode-managed-open-responses-plugin"

SPEC: ToolSpec = {
    "binary": "opencode",
    "package": "opencode-ai",
    "display": "OpenCode",
    "config_path": OPENCODE_CONFIG_PATH,
    "backup_path": OPENCODE_BACKUP_PATH,
}

PROVIDER_KEYS: list[list[str]] = [
    ["provider", "databricks-anthropic"],
    ["provider", "databricks-google"],
]


def _usage_plugin_path() -> str:
    return str(OPENCODE_CONFIG_PATH.parent / "plugins" / "ucode-usage.mjs")


def _is_ucode_usage_plugin(value: object) -> bool:
    usage_plugin_path = _usage_plugin_path()
    if isinstance(value, str):
        return value == usage_plugin_path
    if isinstance(value, list) and value:
        return value[0] == usage_plugin_path
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        raw = mapping.get("path") or mapping.get("module")
        return raw == usage_plugin_path
    return False


def _remove_usage_plugin(doc: dict) -> None:
    # The usage plugin used to spawn a blocking `ucode` subprocess on a timer and
    # before every tool call, which badly slowed OpenCode down. We no longer track
    # OpenCode spend via a plugin (budget-status reads its DB directly instead), so
    # drop any previously-installed plugin from the config and delete the file.
    plugins = doc.get("plugin")
    if isinstance(plugins, list):
        doc["plugin"] = [plugin for plugin in plugins if not _is_ucode_usage_plugin(plugin)]
    stale = OPENCODE_CONFIG_PATH.parent / "plugins" / "ucode-usage.mjs"
    stale.unlink(missing_ok=True)


def _open_responses_plugin_path() -> str:
    return str(OPENCODE_CONFIG_PATH.parent / "plugins" / "ucode-open-responses.mjs")


def _open_responses_plugin_source() -> str:
    # @ai-sdk/openai (Responses API) always POSTs to `${baseURL}/responses`, but
    # Databricks serves it at `${baseURL}/open-responses`. A plugin auth.loader
    # is the only place we can inject a custom `fetch` (JSON config can't carry a
    # function), so we use it to rewrite the trailing `/responses` to
    # `/open-responses`. The apiKey already lives in the provider `options`
    # (refreshed on the token-refresh loop), so the loader only overrides fetch.
    return f"""// {OPENCODE_OPEN_RESPONSES_PLUGIN_MARKER}
export default async function UcodeOpenResponsesPlugin() {{
  return {{
    auth: {{
      provider: "databricks-open-responses",
      methods: [{{ label: "Databricks PAT", type: "api" }}],
      async loader(getAuth) {{
        const auth = await getAuth();
        const apiKey =
          auth?.type === "api" ? auth.key : process.env.OAUTH_TOKEN || process.env.DATABRICKS_TOKEN;
        return {{
          apiKey,
          async fetch(input, init) {{
            const raw =
              typeof input === "string"
                ? input
                : input instanceof URL
                  ? input.href
                  : input.url;
            const url = new URL(raw);
            url.pathname = url.pathname.replace(/\\/responses$/, "/open-responses");
            // Kimi reasoning OFF by default.
            let opts = init;
            try {{
              if (init && typeof init.body === "string") {{
                const body = JSON.parse(init.body);
                body.reasoning = {{ effort: "none" }};
                opts = {{ ...init, body: JSON.stringify(body) }};
              }}
            }} catch (e) {{}}
            return fetch(url, opts);
          }},
        }};
      }},
    }},
  }};
}}
"""


def _write_open_responses_plugin() -> None:
    path = OPENCODE_CONFIG_PATH.parent / "plugins" / "ucode-open-responses.mjs"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_open_responses_plugin_source(), encoding="utf-8")


def _is_open_responses_plugin(value: object) -> bool:
    plugin_path = _open_responses_plugin_path()
    if isinstance(value, str):
        return value == plugin_path
    if isinstance(value, list) and value:
        return value[0] == plugin_path
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        raw = mapping.get("path") or mapping.get("module")
        return raw == plugin_path
    return False


def _upsert_open_responses_plugin(doc: dict) -> None:
    plugins = doc.get("plugin")
    if not isinstance(plugins, list):
        plugins = []
    plugins = [plugin for plugin in plugins if not _is_open_responses_plugin(plugin)]
    plugins.append(_open_responses_plugin_path())
    doc["plugin"] = plugins


def _remove_open_responses_plugin(doc: dict) -> None:
    plugins = doc.get("plugin")
    if not isinstance(plugins, list):
        return
    doc["plugin"] = [plugin for plugin in plugins if not _is_open_responses_plugin(plugin)]


def _write_open_responses_auth(token: str) -> None:
    # OpenCode only runs a provider's plugin `auth.loader` when a stored
    # credential exists for that provider (it skips the loader otherwise, which
    # leaves @ai-sdk/openai with no apiKey -> "OpenAI API key is missing").
    # Writing an `api` entry here satisfies that gate so the loader runs; the
    # loader reads this key, while the live token used for requests comes from
    # OAUTH_TOKEN. We keep the token in sync anyway so the value is never stale.
    OPENCODE_AUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    auth = read_json_safe(OPENCODE_AUTH_PATH)
    auth["databricks-open-responses"] = {"type": "api", "key": token}
    write_json_file(OPENCODE_AUTH_PATH, auth)


def _remove_open_responses_auth() -> None:
    auth = read_json_safe(OPENCODE_AUTH_PATH)
    if auth.pop("databricks-open-responses", None) is not None:
        write_json_file(OPENCODE_AUTH_PATH, auth)


def is_update_available() -> tuple[str, str] | None:
    return available_npm_package_update(SPEC["package"])


def _resolve_model_selector(model: str, opencode_models: dict[str, list[str]]) -> str:
    """Return an OpenCode model selector in provider/model form when possible."""
    if (
        model.startswith("databricks-anthropic/")
        or model.startswith("databricks-google/")
        or model.startswith("databricks-open-responses/")
    ):
        return model

    anthropic_models = opencode_models.get("anthropic") or []
    if model in anthropic_models:
        return f"databricks-anthropic/{model}"

    gemini_models = opencode_models.get("gemini") or []
    if model in gemini_models:
        return f"databricks-google/{model}"

    open_responses_models = opencode_models.get("open-responses") or []
    if model in open_responses_models:
        return f"databricks-open-responses/{model}"

    return model


def _models_with_names(model_ids: list[str], base: dict) -> dict:
    """Map each model id to its own copy of ``base``, adding a vanity ``name``
    (OpenCode's model-picker label) whenever one differs from the raw id."""
    entries: dict[str, dict] = {}
    for mid in model_ids:
        entry = dict(base)
        vanity = endpoint_vanity_name(mid)
        if vanity != mid:
            entry["name"] = vanity
        entries[mid] = entry
    return entries


def render_overlay(
    model: str,
    token: str,
    opencode_base_urls: dict[str, str],
    opencode_models: dict[str, list[str]],
) -> tuple[dict, list[list[str]]]:
    """Return (overlay, managed_key_paths) for opencode.json."""
    auth_headers = {"Authorization": f"Bearer {token}"}
    # OpenCode hardcodes `User-Agent: opencode/<ver>` in session/llm.ts for
    # every provider, after the AI SDK's combineHeaders. The provider-level
    # `headers` are clobbered by that injection, but per-model `headers` are
    # merged AFTER and win — so the UA must live on each model entry.
    ua_header = {
        "User-Agent": f"ucode/{ucode_version()} opencode/{agent_version('opencode')}",
    }

    anthropic_models = opencode_models.get("anthropic") or []
    gemini_models = opencode_models.get("gemini") or []
    open_responses_models = opencode_models.get("open-responses") or []

    providers: dict = {}
    keys: list[list[str]] = [["model"]]
    if anthropic_models:
        # @ai-sdk/anthropic injects `eager_input_streaming: true` on tool defs;
        # the Databricks gateway's strict validator rejects it. opencode's
        # auto-disable in transform.ts skips models whose id contains "claude",
        # so we opt out per-model. The setting lives in per-call providerOptions,
        # which opencode reads from `models.<m>.options`, not provider `options`.
        anthropic_model_overlay = {
            "headers": ua_header,
            "options": {"toolStreaming": False},
        }
        providers["databricks-anthropic"] = {
            "npm": "@ai-sdk/anthropic",
            "options": {
                "baseURL": opencode_base_urls["anthropic"],
                "apiKey": token,
                "headers": auth_headers,
            },
            "models": _models_with_names(anthropic_models, anthropic_model_overlay),
        }
        keys.append(["provider", "databricks-anthropic"])
    if gemini_models:
        providers["databricks-google"] = {
            "npm": "@ai-sdk/google",
            "options": {
                "baseURL": opencode_base_urls["gemini"],
                "apiKey": token,
                "headers": auth_headers,
            },
            "models": _models_with_names(gemini_models, {"headers": ua_header}),
        }
        keys.append(["provider", "databricks-google"])
    if open_responses_models:
        # Kimi speaks the OpenAI Responses API at the serving-endpoints path.
        # @ai-sdk/openai POSTs to `${baseURL}/responses`; the rewrite plugin
        # (see _write_open_responses_plugin) maps that to `/open-responses`.
        # Per-model `headers` carry the UA (same reason as anthropic).
        #
        # `X-Axon-Mode: ROUTER` is intentionally NOT sent: the gateway 502s on
        # the `-colo` endpoint when that header is present. `X-Axon-LB-Mode`
        # remains to keep load-balancer routing.
        #
        # Crucially, `apiKey` is NOT set here: OpenCode only invokes a provider's
        # plugin `auth.loader` when it must resolve credentials itself. Supplying
        # `apiKey` in `options` short-circuits that, so the loader — and its
        # custom `fetch` that does the /responses -> /open-responses rewrite —
        # never runs, and requests hit the wrong path. The loader sources the
        # token from OAUTH_TOKEN (set by build_runtime_env) instead.
        providers["databricks-open-responses"] = {
            "npm": "@ai-sdk/openai",
            "name": "Databricks",
            "options": {
                "baseURL": opencode_base_urls["open-responses"],
                "headers": {
                    **auth_headers,
                    "X-Axon-LB-Mode": "DICER",
                },
            },
            "models": _models_with_names(open_responses_models, {"headers": ua_header}),
        }
        keys.append(["provider", "databricks-open-responses"])

    overlay: dict = {"model": _resolve_model_selector(model, opencode_models)}
    if providers:
        overlay["provider"] = providers
    return overlay, keys


def write_tool_config(
    state: dict,
    model: str,
    token: str | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[dict, str]:
    backup_existing_file(OPENCODE_CONFIG_PATH, OPENCODE_BACKUP_PATH)
    if token is None:
        token = get_databricks_token(
            state["workspace"], state.get("profile"), force_refresh=force_refresh
        )
    opencode_base_urls = state.get("base_urls", {}).get("opencode") or build_opencode_base_urls(
        state["workspace"]
    )
    overlay, managed_keys = render_overlay(
        model,
        token,
        opencode_base_urls,
        state.get("opencode_models") or {},
    )
    existing = read_json_safe(OPENCODE_CONFIG_PATH)
    providers = existing.get("provider")
    if isinstance(providers, dict):
        for stale in (
            "databricks-anthropic",
            "databricks-google",
            "databricks-openai",
            "databricks-open-responses",
        ):
            providers.pop(stale, None)
    merged = deep_merge_dict(existing, overlay)
    _remove_usage_plugin(merged)
    # The open-responses rewrite plugin is only needed when the Kimi provider is
    # configured; register it then and prune it otherwise so a removed endpoint
    # leaves no dangling plugin path.
    if "databricks-open-responses" in overlay.get("provider", {}):
        _write_open_responses_plugin()
        _upsert_open_responses_plugin(merged)
        _write_open_responses_auth(token)
    else:
        _remove_open_responses_plugin(merged)
        _remove_open_responses_auth()
    managed_keys = managed_keys + [["plugin"]]
    write_json_file(OPENCODE_CONFIG_PATH, merged)
    state = mark_tool_managed(state, "opencode", managed_keys)
    save_state(state)
    return state, token


def build_mcp_server_entry(url: str) -> dict:
    return {
        "type": "remote",
        "url": url,
        "enabled": True,
        "headers": {
            "Authorization": OPENCODE_MCP_AUTH_HEADER_VALUE,
        },
    }


def write_mcp_server_config(name: str, url: str) -> bool:
    backup_existing_file(OPENCODE_CONFIG_PATH, OPENCODE_BACKUP_PATH)
    existing = read_json_safe(OPENCODE_CONFIG_PATH)
    mcp_servers = existing.get("mcp")
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
    removed = name in mcp_servers
    mcp_servers[name] = build_mcp_server_entry(url)
    existing["mcp"] = mcp_servers
    write_json_file(OPENCODE_CONFIG_PATH, existing)
    return removed


def remove_mcp_server_config(name: str) -> bool:
    existing = read_json_safe(OPENCODE_CONFIG_PATH)
    mcp_servers = existing.get("mcp")
    if not isinstance(mcp_servers, dict) or name not in mcp_servers:
        return False
    mcp_servers.pop(name)
    existing["mcp"] = mcp_servers
    write_json_file(OPENCODE_CONFIG_PATH, existing)
    return True


def default_model(state: dict) -> str | None:
    opencode_models = state.get("opencode_models") or {}
    anthropic = opencode_models.get("anthropic") or []
    if anthropic:
        return anthropic[0]
    gemini = opencode_models.get("gemini") or []
    if gemini:
        return gemini[0]
    open_responses = opencode_models.get("open-responses") or []
    return open_responses[0] if open_responses else None


def _configured_model(state: dict) -> str | None:
    """The model already selected in opencode.json, if any.

    The launch path writes the chosen model here (e.g. the cheaper Haiku model
    picked from the budget-warn selector), so the token-refresh loop must reuse
    it rather than resetting to the default and clobbering the selection. The
    stored value is already a selector (``provider/model``), which
    ``_resolve_model_selector`` passes through unchanged."""
    existing = read_json_safe(OPENCODE_CONFIG_PATH)
    model = existing.get("model")
    return model if isinstance(model, str) and model else None


def _policy_resolved_default(state: dict) -> str | None:
    """Default model with admin policies applied. ``None`` if no model is configured."""
    base = default_model(state)
    if not base:
        return None
    return resolve_policy_default_model(state, "opencode", base)


def _refresh_token_once(state: dict, *, force_refresh: bool = False) -> str:
    # Prefer the model already configured so a token refresh never overrides a
    # deliberately-selected model; fall back to the policy default on first run.
    model = _configured_model(state) or _policy_resolved_default(state)
    if not model:
        raise RuntimeError("No OpenCode model is configured.")
    _, token = write_tool_config(state, model, force_refresh=force_refresh)
    return token


def _refresh_forever(state: dict, stop_event: threading.Event) -> None:
    while not stop_event.wait(TOKEN_REFRESH_INTERVAL_SECONDS):
        try:
            _refresh_token_once(state, force_refresh=True)
        except RuntimeError:
            continue


def build_runtime_env(token: str, state: dict | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["OAUTH_TOKEN"] = token
    env["XDG_CONFIG_HOME"] = str(OPENCODE_XDG_CONFIG_HOME)
    env["XDG_DATA_HOME"] = str(OPENCODE_XDG_DATA_HOME)
    return env


def launch(state: dict, tool_args: list[str]) -> None:
    """Launch opencode with background token refresh (same pattern as Gemini)."""
    token = _refresh_token_once(state)
    env = build_runtime_env(token, state)

    stop_event = threading.Event()
    refresher = threading.Thread(
        target=_refresh_forever,
        args=(state, stop_event),
        daemon=True,
    )
    refresher.start()

    proc = subprocess.Popen([SPEC["binary"], *tool_args], env=env)
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        returncode = proc.wait()
    finally:
        stop_event.set()
        refresher.join(timeout=1)

    raise SystemExit(returncode)


def validate_cmd(binary: str) -> list[str]:
    return [binary, "run", "say hi in 5 words or less"]


def validate_env(state: dict) -> dict[str, str]:
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured.")
    return build_runtime_env(get_databricks_token(workspace, state.get("profile")), state)
