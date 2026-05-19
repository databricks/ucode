"""Claude Code agent: writes ~/.claude/settings.json env block."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

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
    build_auth_shell_command,
    build_tool_base_url,
    get_databricks_token,
)
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version

CLAUDE_CONFIG_DIR = Path.home() / ".claude"
CLAUDE_SETTINGS_PATH = CLAUDE_CONFIG_DIR / "ucode-settings.json"
CLAUDE_BACKUP_PATH = APP_DIR / "claude-ucode-settings.backup.json"

SPEC: ToolSpec = {
    "binary": "claude",
    "package": "@anthropic-ai/claude-code",
    "display": "Claude Code",
    "config_path": CLAUDE_SETTINGS_PATH,
    "backup_path": CLAUDE_BACKUP_PATH,
}


def is_update_available() -> tuple[str, str] | None:
    return available_npm_package_update(SPEC["package"])


def _resolve_web_search_model(state: dict) -> str | None:
    """Pick the model the web_search MCP server should call. Prefers an
    explicit override in state, otherwise the first endpoint discovered as
    Responses-API-capable. Returns None if no GPT endpoint is available —
    callers should skip the MCP wiring in that case."""
    override = state.get("web_search_model")
    if isinstance(override, str) and override.strip():
        return override.strip()
    codex_models = state.get("codex_models") or []
    if isinstance(codex_models, list) and codex_models:
        first = codex_models[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return None


WEB_SEARCH_MCP_NAME = "web_search"


def _web_search_mcp_entry(workspace: str, search_model: str) -> dict:
    """Stdio MCP server entry pointing at `ucode mcp web-search`. Resolves
    the absolute path to the `ucode` binary so launchers without the right
    PATH (e.g. desktop GUI launchers) still find it."""
    ucode_binary = shutil.which("ucode") or "ucode"
    return {
        "type": "stdio",
        "command": ucode_binary,
        "args": ["mcp", "web-search"],
        "env": {
            "DATABRICKS_HOST": workspace,
            "UCODE_WEB_SEARCH_MODEL": search_model,
        },
    }


def render_overlay(
    workspace: str,
    model: str,
    claude_models: dict[str, str] | None = None,
    disable_web_search: bool = False,
) -> tuple[dict, list[list[str]]]:
    """Return (overlay, managed_key_paths) for Claude settings.json.

    NOTE: MCP servers are NOT written here. Claude Code reads `mcpServers`
    from `~/.claude.json`, not `~/.claude/settings.json` — registration goes
    through `claude mcp add-json` (see `_register_web_search_mcp`)."""
    base_url = build_tool_base_url("claude", workspace)
    # ANTHROPIC_CUSTOM_HEADERS is parsed as `key: value` pairs separated by
    # newlines (Anthropic SDK convention). Setting User-Agent here overrides
    # the SDK's default UA on outbound requests so the gateway can attribute
    # traffic to ucode.
    custom_headers = "\n".join(
        [
            "x-databricks-use-coding-agent-mode: true",
            f"User-Agent: ucode/{ucode_version()} claude/{agent_version('claude')}",
        ]
    )
    env: dict[str, str] = {
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_CUSTOM_HEADERS": custom_headers,
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_API_KEY_HELPER_TTL_MS": "900000",
    }
    if claude_models:
        if claude_models.get("opus"):
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = claude_models["opus"]
        if claude_models.get("sonnet"):
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = claude_models["sonnet"]
        if claude_models.get("haiku"):
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = claude_models["haiku"]
    overlay: dict = {"apiKeyHelper": build_auth_shell_command(workspace), "env": env}
    keys: list[list[str]] = [["apiKeyHelper"]] + [["env", k] for k in env]

    # Disable Claude Code's built-in WebSearch (it routes through Anthropic's
    # hosted infra and fails through the Databricks gateway). The replacement
    # `web_search` MCP server is registered separately via the claude CLI.
    if disable_web_search:
        overlay["disabledTools"] = ["WebSearch"]
        keys.append(["disabledTools"])

    return overlay, keys


def _register_web_search_mcp(workspace: str, search_model: str) -> None:
    """Register (or replace) the web_search MCP server in Claude Code's user
    scope via `claude mcp add-json`. Removes any prior entry first so re-runs
    pick up changes to the workspace, model, or ucode binary path."""
    # Imported lazily to avoid a circular import via ucode.mcp -> ucode.agents.
    from ucode.mcp import (
        MCP_CLEANUP_SCOPES,
        add_claude_mcp_server,
        remove_claude_mcp_server,
    )

    for scope in MCP_CLEANUP_SCOPES:
        try:
            remove_claude_mcp_server(WEB_SEARCH_MCP_NAME, scope)
        except RuntimeError:
            # Best-effort cleanup of stale entries — keep going.
            pass
    entry = _web_search_mcp_entry(workspace, search_model)
    add_claude_mcp_server(WEB_SEARCH_MCP_NAME, entry)


def _unregister_web_search_mcp() -> None:
    """Remove the web_search MCP server from all scopes. Used by revert."""
    from ucode.mcp import MCP_CLEANUP_SCOPES, remove_claude_mcp_server

    for scope in MCP_CLEANUP_SCOPES:
        try:
            remove_claude_mcp_server(WEB_SEARCH_MCP_NAME, scope)
        except RuntimeError:
            pass


def write_tool_config(state: dict, model: str) -> dict:
    backup_existing_file(CLAUDE_SETTINGS_PATH, CLAUDE_BACKUP_PATH)
    web_search_model = _resolve_web_search_model(state)
    overlay, managed_keys = render_overlay(
        state["workspace"],
        model,
        state.get("claude_models") or {},
        disable_web_search=web_search_model is not None,
    )
    existing = read_json_safe(CLAUDE_SETTINGS_PATH)
    merged = deep_merge_dict(existing, overlay)
    write_json_file(CLAUDE_SETTINGS_PATH, merged)

    if web_search_model:
        _register_web_search_mcp(state["workspace"], web_search_model)

    state = mark_tool_managed(state, "claude", managed_keys)
    save_state(state)
    return state


def default_model(state: dict) -> str | None:
    claude_models = state.get("claude_models") or {}
    return claude_models.get("opus") or claude_models.get("sonnet") or claude_models.get("haiku")


def launch(state: dict, tool_args: list[str]) -> None:
    binary = SPEC["binary"]
    workspace = state.get("workspace")
    if workspace:
        os.environ["OAUTH_TOKEN"] = get_databricks_token(workspace)
    os.execvp(binary, [binary, "--settings", str(CLAUDE_SETTINGS_PATH), *tool_args])


def validate_cmd(binary: str) -> list[str]:
    return [
        binary,
        "--settings",
        str(CLAUDE_SETTINGS_PATH),
        "-p",
        "say hi in 5 words or less",
        "--max-turns",
        "1",
    ]
