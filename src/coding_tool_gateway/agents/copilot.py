"""GitHub Copilot CLI agent: writes ~/.copilot/.env and injects BYOK env vars at launch.

Copilot CLI's BYOK config (COPILOT_PROVIDER_*) is documented as env-var only —
the CLI does not auto-load ~/.copilot/.env. We still write the file so users can
inspect what's configured (`cat ~/.copilot/.env`) and to give `revert` something
to clean up; the values are also injected directly into the child process's
environment at launch.

We point Copilot CLI's `openai` provider at the Databricks MLflow chat-completions
gateway, which serves Claude and codex (gpt-5) models. Gemini is intentionally
excluded — Databricks' Gemini translation layer rejects the `stream_options`
field that Copilot CLI sends, so Gemini models 400 on every request.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from pathlib import Path

from coding_tool_gateway.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    parse_dotenv,
    read_json_safe,
    write_dotenv,
    write_json_file,
)
from coding_tool_gateway.databricks import (
    TOKEN_REFRESH_INTERVAL_SECONDS,
    build_copilot_base_url,
    get_databricks_token,
)
from coding_tool_gateway.state import mark_tool_managed, save_state

COPILOT_CONFIG_DIR = Path.home() / ".copilot"
COPILOT_ENV_PATH = COPILOT_CONFIG_DIR / ".env"
COPILOT_MCP_CONFIG_PATH = COPILOT_CONFIG_DIR / "mcp-config.json"
COPILOT_BACKUP_PATH = APP_DIR / "copilot-env.backup"
COPILOT_MCP_BACKUP_PATH = APP_DIR / "copilot-mcp-config.backup.json"

SPEC: ToolSpec = {
    "binary": "copilot",
    "package": "@github/copilot",
    "display": "GitHub Copilot CLI",
    "config_path": COPILOT_ENV_PATH,
    "backup_path": COPILOT_BACKUP_PATH,
}

MANAGED_KEYS: list[str] = [
    "COPILOT_PROVIDER_TYPE",
    "COPILOT_PROVIDER_BASE_URL",
    "COPILOT_MODEL",
    "COPILOT_PROVIDER_BEARER_TOKEN",
    "COPILOT_OFFLINE",
    "OAUTH_TOKEN",
]


def default_model(state: dict) -> str | None:
    """Prefer Claude sonnet, then opus/haiku, then codex."""
    claude_models = state.get("claude_models") or {}
    for family in ("sonnet", "opus", "haiku"):
        if claude_models.get(family):
            return claude_models[family]
    codex_models = state.get("codex_models") or []
    if codex_models:
        return codex_models[0]
    return None


def render_env_overlay(workspace: str, model: str, token: str) -> dict[str, str]:
    return {
        "COPILOT_PROVIDER_TYPE": "openai",
        "COPILOT_PROVIDER_BASE_URL": build_copilot_base_url(workspace),
        "COPILOT_MODEL": model,
        "COPILOT_PROVIDER_BEARER_TOKEN": token,
        "COPILOT_OFFLINE": "true",
        "OAUTH_TOKEN": token,
    }


def build_runtime_env(workspace: str, model: str, token: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(render_env_overlay(workspace, model, token))
    return env


def build_mcp_server_entry(url: str) -> dict:
    return {
        "type": "http",
        "url": url,
        "headers": {
            "Authorization": "Bearer ${OAUTH_TOKEN}",
        },
        "tools": ["*"],
    }


def write_mcp_server_config(name: str, url: str) -> bool:
    backup_existing_file(COPILOT_MCP_CONFIG_PATH, COPILOT_MCP_BACKUP_PATH)
    existing = read_json_safe(COPILOT_MCP_CONFIG_PATH)
    mcp_servers = existing.get("mcpServers")
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
    removed = name in mcp_servers
    mcp_servers[name] = build_mcp_server_entry(url)
    existing["mcpServers"] = mcp_servers
    write_json_file(COPILOT_MCP_CONFIG_PATH, existing)
    return removed


def write_tool_config(
    state: dict,
    model: str,
    token: str | None = None,
) -> tuple[dict, str]:
    backup_existing_file(COPILOT_ENV_PATH, COPILOT_BACKUP_PATH)
    if token is None:
        token = get_databricks_token(state["workspace"])
    overlay = render_env_overlay(state["workspace"], model, token)
    existing = parse_dotenv(COPILOT_ENV_PATH)
    existing.update(overlay)
    write_dotenv(COPILOT_ENV_PATH, existing)
    state = mark_tool_managed(state, "copilot", MANAGED_KEYS)
    save_state(state)
    return state, token


def _refresh_token_once(state: dict) -> tuple[str, str]:
    model = default_model(state)
    if not model:
        raise RuntimeError("No Copilot model is available on this workspace.")
    _, token = write_tool_config(state, model)
    return model, token


def _refresh_forever(state: dict, stop_event: threading.Event) -> None:
    while not stop_event.wait(TOKEN_REFRESH_INTERVAL_SECONDS):
        try:
            _refresh_token_once(state)
        except RuntimeError:
            continue


def launch(state: dict, tool_args: list[str]) -> None:
    model, token = _refresh_token_once(state)
    env = build_runtime_env(state["workspace"], model, token)

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
    return [binary, "--prompt", "say hi in 5 words or less", "--allow-all-tools"]


def validate_env(state: dict) -> dict[str, str]:
    """Inject BYOK env vars for the validation subprocess (Copilot doesn't auto-load .env)."""
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured.")
    model = default_model(state)
    if not model:
        raise RuntimeError("No Copilot model is available on this workspace.")
    token = get_databricks_token(workspace)
    return build_runtime_env(workspace, model, token)
