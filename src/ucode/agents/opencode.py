"""OpenCode agent: writes opencode.json with Databricks-backed providers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading

from ucode.agent_updates import available_npm_package_update
from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_json_safe,
    write_json_file,
    write_text_file,
)
from ucode.databricks import (
    TOKEN_REFRESH_INTERVAL_SECONDS,
    build_auth_token_argv,
    build_opencode_base_urls,
    get_databricks_token,
    model_token_limits,
)
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version

OPENCODE_XDG_CONFIG_HOME = APP_DIR / "opencode-xdg"
OPENCODE_CONFIG_DIR = OPENCODE_XDG_CONFIG_HOME / "opencode"
OPENCODE_CONFIG_PATH = OPENCODE_CONFIG_DIR / "opencode.json"
OPENCODE_BACKUP_PATH = APP_DIR / "opencode-config.backup.json"
OPENCODE_MCP_AUTH_HEADER_VALUE = "Bearer {env:OAUTH_TOKEN}"

# Axis B of #190: opencode bakes provider.options.headers.Authorization into
# opencode.json once at `ucode configure`/refresh time and never re-reads it
# for the life of the process, so a long-lived session keeps sending a
# bearer token that may since have been rotated or revoked. This
# ucode-managed plugin's `chat.headers` hook overrides that static header on
# every chat request with a freshly fetched (and short-TTL-cached) token.
# The static opencode.json header stays in place as the bootstrap/fallback
# the plugin fails open to if the refresh call errors.
OPENCODE_PLUGIN_DIR = OPENCODE_CONFIG_DIR / "plugin"
OPENCODE_PLUGIN_PATH = OPENCODE_PLUGIN_DIR / "ucode-databricks-auth.js"
AUTH_PLUGIN_TOKEN_TTL_SECONDS = 60

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
    ["provider", "databricks-oss"],
]


def is_update_available() -> tuple[str, str] | None:
    return available_npm_package_update(SPEC["package"])


def _resolve_model_selector(model: str, opencode_models: dict[str, list[str]]) -> str:
    """Return an OpenCode model selector in provider/model form when possible."""
    if model.startswith(("databricks-anthropic/", "databricks-google/", "databricks-oss/")):
        return model

    anthropic_models = opencode_models.get("anthropic") or []
    if model in anthropic_models:
        return f"databricks-anthropic/{model}"

    gemini_models = opencode_models.get("gemini") or []
    if model in gemini_models:
        return f"databricks-google/{model}"

    oss_models = opencode_models.get("oss") or []
    if model in oss_models:
        return f"databricks-oss/{model}"

    return model


def _oss_model_overlay(model: str, ua_header: dict[str, str]) -> dict:
    """Per-model overlay for an OSS model entry.

    All OSS models carry the User-Agent header; models with known token limits
    also pin `limit` (context + output) so OpenCode clamps `max_tokens` to a
    value the gateway accepts. OpenCode's schema requires both fields together,
    so the limits table always supplies both."""
    overlay: dict = {"headers": ua_header}
    limits = model_token_limits(model)
    if limits is not None:
        overlay["limit"] = limits
    return overlay


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
    oss_models = opencode_models.get("oss") or []

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
            "models": dict.fromkeys(anthropic_models, anthropic_model_overlay),
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
            "models": {m: {"headers": ua_header} for m in gemini_models},
        }
        keys.append(["provider", "databricks-google"])
    if oss_models:
        providers["databricks-oss"] = {
            "npm": "@ai-sdk/openai",
            "options": {
                "baseURL": opencode_base_urls["oss"],
                "apiKey": token,
                "headers": auth_headers,
            },
            "models": {m: _oss_model_overlay(m, ua_header) for m in oss_models},
        }
        keys.append(["provider", "databricks-oss"])

    overlay: dict = {"model": _resolve_model_selector(model, opencode_models)}
    if providers:
        overlay["provider"] = providers
    return overlay, keys


def _managed_provider_ids(managed_keys: list[list[str]]) -> list[str]:
    """Provider ids actually configured, derived from render_overlay's managed_keys.

    `managed_keys` entries are key paths like `["provider", "databricks-anthropic"]`
    for each provider render_overlay populated (only providers with at least one
    configured model are included); this just picks those out."""
    return [key[1] for key in managed_keys if len(key) == 2 and key[0] == "provider"]


def render_auth_plugin(workspace: str, profile: str | None, provider_ids: list[str]) -> str:
    """Return the JS source for the ucode-managed opencode `chat.headers` plugin.

    On every chat request, the plugin sets a freshly fetched
    `Authorization: Bearer <token>` header for our Databricks providers
    (identified by `provider_ids`), overriding the static header baked into
    opencode.json. The token comes from `ucode auth-token` -- the same
    cross-platform helper used elsewhere (see `build_auth_token_argv`) -- but
    is cached in-process for `AUTH_PLUGIN_TOKEN_TTL_SECONDS` so it doesn't
    shell out on every request; a single in-flight promise coalesces
    concurrent requests within the process. On any error, the hook leaves
    `output.headers` untouched (fail open to the static bootstrap token)."""
    argv = build_auth_token_argv(workspace, profile)
    argv_literal = json.dumps(argv)
    provider_ids_literal = json.dumps(sorted(provider_ids))
    ttl_ms = AUTH_PLUGIN_TOKEN_TTL_SECONDS * 1000

    return f"""\
// Generated and managed by ucode -- do not edit by hand. Overwritten on every
// `ucode configure` / token refresh.
//
// Fixes #190: opencode reads provider.options.headers.Authorization from
// opencode.json once at startup and never again, so a long-lived session
// keeps sending a token that may since have been rotated or revoked. This
// `chat.headers` hook sets a fresh Authorization header on every chat
// request for our Databricks providers, overriding the static header.

import {{ execFile }} from "node:child_process";
import {{ promisify }} from "node:util";

const execFileAsync = promisify(execFile);

const UCODE_AUTH_TOKEN_ARGV = {argv_literal};
const MANAGED_PROVIDER_IDS = new Set({provider_ids_literal});
const TTL_MS = {ttl_ms};

let cachedToken = null;
let cachedAt = 0;
let inflight = null;

async function fetchToken() {{
  const [cmd, ...args] = UCODE_AUTH_TOKEN_ARGV;
  const {{ stdout }} = await execFileAsync(cmd, args);
  const token = stdout.trim();
  if (!token) {{
    // `ucode auth-token` exited 0 but printed nothing usable -- treat as a
    // failure so the caller's catch block fails open to the static
    // bootstrap token instead of caching/sending an empty bearer.
    throw new Error("ucode auth-token returned empty output");
  }}
  return token;
}}

async function getToken() {{
  const now = Date.now();
  if (cachedToken && now - cachedAt < TTL_MS) {{
    return cachedToken;
  }}
  if (inflight) {{
    return inflight;
  }}
  inflight = fetchToken()
    .then((token) => {{
      cachedToken = token;
      cachedAt = Date.now();
      return token;
    }})
    .finally(() => {{
      inflight = null;
    }});
  return inflight;
}}

export const UcodeDatabricksAuth = async (_ctx) => ({{
  "chat.headers": async (input, output) => {{
    // NOTE: the hook's `input.provider` is the runtime Provider record
    // itself (id/name/env/options/source/models), not the ProviderContext
    // shape ({{source, info, options}}) some type surfaces suggest -- `id`
    // lives directly on `input.provider`, not `input.provider.info.id`.
    // Confirmed empirically against opencode 1.17.10; verify against a real
    // hook invocation before changing this path again (#190 follow-up).
    const providerId = input.provider?.id;
    if (!providerId || !MANAGED_PROVIDER_IDS.has(providerId)) {{
      return;
    }}
    try {{
      const token = await getToken();
      // Defensive: ensure `output.headers` exists even if this opencode
      // version ever invokes the hook without a pre-populated headers
      // object, so the header still gets set rather than throwing (which
      // would be caught below and silently no-op the refresh).
      output.headers = output.headers || {{}};
      output.headers["Authorization"] = "Bearer " + token;
    }} catch (err) {{
      console.error("[ucode] failed to refresh Databricks token:", err);
      // Fail open: leave the static bootstrap token from opencode.json.
    }}
  }},
}});
"""


def write_auth_plugin(state: dict, provider_ids: list[str]) -> None:
    """Write the ucode-managed opencode auth plugin (see `render_auth_plugin`).

    Always writes -- including with an empty `provider_ids` -- so a
    workspace/model change that drops the last provider still overwrites a
    stale plugin rather than leaving one referencing removed provider ids."""
    plugin_js = render_auth_plugin(state["workspace"], state.get("profile"), provider_ids)
    write_text_file(OPENCODE_PLUGIN_PATH, plugin_js)


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
            "databricks-oss",
        ):
            providers.pop(stale, None)
    merged = deep_merge_dict(existing, overlay)
    write_json_file(OPENCODE_CONFIG_PATH, merged)
    # Plugin is the live/per-request override; opencode.json (above) remains
    # the static bootstrap/fallback the plugin fails open to. See #190.
    write_auth_plugin(state, _managed_provider_ids(managed_keys))
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
    oss = opencode_models.get("oss") or []
    return oss[0] if oss else None


def _refresh_token_once(state: dict, *, force_refresh: bool = False) -> str:
    model = default_model(state)
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
