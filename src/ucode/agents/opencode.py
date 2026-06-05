"""OpenCode agent: writes opencode.json with two Databricks-backed providers."""

from __future__ import annotations

import json
import os
import shutil
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
from ucode.policies import resolve_policy_default_model
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version

OPENCODE_XDG_CONFIG_HOME = APP_DIR / "opencode-xdg"
OPENCODE_CONFIG_DIR = OPENCODE_XDG_CONFIG_HOME / "opencode"
OPENCODE_CONFIG_PATH = OPENCODE_CONFIG_DIR / "opencode.json"
OPENCODE_BACKUP_PATH = APP_DIR / "opencode-config.backup.json"
OPENCODE_MCP_AUTH_HEADER_VALUE = "Bearer {env:OAUTH_TOKEN}"
OPENCODE_USAGE_PLUGIN_MARKER = "ucode-managed-usage-plugin"

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


def _ucode_command_prefix() -> list[str]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if os.path.exists(os.path.join(repo_root, "pyproject.toml")) and os.path.exists(
        os.path.join(repo_root, "src", "ucode")
    ):
        return ["uv", "run", "--project", repo_root, "ucode"]
    return [shutil.which("ucode") or "ucode"]


def _usage_plugin_path() -> str:
    return str(OPENCODE_CONFIG_PATH.parent / "plugins" / "ucode-usage.mjs")


def _usage_plugin_source(workspace: str, model: str) -> str:
    command_json = json.dumps(_ucode_command_prefix())
    workspace_json = json.dumps(workspace)
    model_json = json.dumps(model)
    return f"""// {OPENCODE_USAGE_PLUGIN_MARKER}
import {{ spawnSync }} from "node:child_process";

const UCODE = {command_json};
const WORKSPACE = {workspace_json};
const MODEL = {model_json};
const POLL_INTERVAL_MS = 2000;

function hook(event, sessionID, extra = {{}}) {{
  const args = [
    ...UCODE.slice(1),
    "usage",
    "hook",
    "opencode",
    event,
    "--model",
    MODEL,
    "--workspace",
    WORKSPACE,
  ];
  const input = JSON.stringify(sessionID ? {{ sessionID, ...extra }} : extra);
  const result = spawnSync(UCODE[0], args, {{
    input,
    encoding: "utf8",
    stdio: ["pipe", "pipe", "ignore"],
  }});
  if (result.error || result.status !== 0) return {{}};
  try {{
    return JSON.parse(result.stdout || "{{}}");
  }} catch {{
    return {{}};
  }}
}}

function message(response) {{
  return response?.reason
    || response?.systemMessage
    || response?.hookSpecificOutput?.additionalContext
    || "";
}}

async function enforce(client, event, sessionID) {{
  const response = hook(event, sessionID);
  const text = message(response);
  if (response?.decision === "block" || response?.continue === false) {{
    throw new Error(text || "Daily budget exceeded.");
  }}
  if (text) {{
    await client.tui.showToast({{
      title: "ucode budget",
      message: text,
      variant: "warning",
      duration: 10000,
    }}).catch(() => undefined);
  }}
}}

export default async function UcodeUsagePlugin({{ client }}) {{
  const timer = setInterval(() => {{
    hook("event");
  }}, POLL_INTERVAL_MS);
  timer.unref?.();
  return {{
    async event(input) {{
      const event = input?.event;
      const type = event?.type;
      const sessionID = event?.properties?.sessionID;
      if (!type) {{
        return;
      }}
      if (
        type === "message.updated"
        || type === "session.updated"
        || type === "session.idle"
        || type.startsWith("session.next.")
      ) {{
        hook("event", sessionID);
      }}
      if (type === "session.next.step.ended" && sessionID) {{
        hook("step-ended", sessionID, {{ tokens: event?.properties?.tokens }});
      }}
    }},
    async "chat.params"(input, _output) {{
      await enforce(client, "chat-params", input?.sessionID);
    }},
    async "tool.execute.before"(input, _output) {{
      await enforce(client, "tool-execute-before", input?.sessionID);
    }},
  }};
}}
"""


def _write_usage_plugin(workspace: str, model: str) -> None:
    path = OPENCODE_CONFIG_PATH.parent / "plugins" / "ucode-usage.mjs"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_usage_plugin_source(workspace, model), encoding="utf-8")


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


def _upsert_usage_plugin(doc: dict) -> None:
    plugins = doc.get("plugin")
    if not isinstance(plugins, list):
        plugins = []
    plugins = [plugin for plugin in plugins if not _is_ucode_usage_plugin(plugin)]
    plugins.append(_usage_plugin_path())
    doc["plugin"] = plugins


def is_update_available() -> tuple[str, str] | None:
    return available_npm_package_update(SPEC["package"])


def _resolve_model_selector(model: str, opencode_models: dict[str, list[str]]) -> str:
    """Return an OpenCode model selector in provider/model form when possible."""
    if model.startswith("databricks-anthropic/") or model.startswith("databricks-google/"):
        return model

    anthropic_models = opencode_models.get("anthropic") or []
    if model in anthropic_models:
        return f"databricks-anthropic/{model}"

    gemini_models = opencode_models.get("gemini") or []
    if model in gemini_models:
        return f"databricks-google/{model}"

    return model


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
        for stale in ("databricks-anthropic", "databricks-google", "databricks-openai"):
            providers.pop(stale, None)
    merged = deep_merge_dict(existing, overlay)
    _write_usage_plugin(state["workspace"], str(overlay["model"]))
    _upsert_usage_plugin(merged)
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
    return gemini[0] if gemini else None


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
