"""Claude Code agent: writes ~/.claude/settings.json env block."""

from __future__ import annotations

import os
from pathlib import Path

from coding_tool_gateway.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_json_safe,
    write_json_file,
)
from coding_tool_gateway.databricks import (
    build_auth_shell_command,
    build_tool_base_url,
    get_databricks_token,
)
from coding_tool_gateway.state import mark_tool_managed, save_state

CLAUDE_CONFIG_DIR = Path.home() / ".claude"
CLAUDE_SETTINGS_PATH = CLAUDE_CONFIG_DIR / "settings.json"
CLAUDE_BACKUP_PATH = APP_DIR / "claude-settings.backup.json"

SPEC: ToolSpec = {
    "binary": "claude",
    "package": "@anthropic-ai/claude-code",
    "display": "Claude Code",
    "config_path": CLAUDE_SETTINGS_PATH,
    "backup_path": CLAUDE_BACKUP_PATH,
}


def render_overlay(
    workspace: str,
    model: str,
    claude_models: dict[str, str] | None = None,
) -> tuple[dict, list[list[str]]]:
    """Return (overlay, managed_key_paths) for Claude settings.json."""
    base_url = build_tool_base_url("claude", workspace)
    env: dict[str, str] = {
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_CUSTOM_HEADERS": "x-databricks-use-coding-agent-mode: true",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_API_KEY_HELPER_TTL_MS": "1800000",
    }
    if claude_models:
        if claude_models.get("opus"):
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = claude_models["opus"]
        if claude_models.get("sonnet"):
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = claude_models["sonnet"]
        if claude_models.get("haiku"):
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = claude_models["haiku"]
    overlay = {"apiKeyHelper": build_auth_shell_command(workspace), "env": env}
    keys: list[list[str]] = [["apiKeyHelper"]] + [["env", k] for k in env]
    return overlay, keys


def write_tool_config(state: dict, model: str) -> dict:
    backup_existing_file(CLAUDE_SETTINGS_PATH, CLAUDE_BACKUP_PATH)
    overlay, managed_keys = render_overlay(
        state["workspace"],
        model,
        state.get("claude_models") or {},
    )
    existing = read_json_safe(CLAUDE_SETTINGS_PATH)
    merged = deep_merge_dict(existing, overlay)
    write_json_file(CLAUDE_SETTINGS_PATH, merged)
    state = mark_tool_managed(state, "claude", managed_keys)
    save_state(state)
    return state


def default_model(state: dict) -> str | None:
    claude_models = state.get("claude_models") or {}
    return claude_models.get("sonnet") or claude_models.get("opus") or claude_models.get("haiku")


def launch(state: dict, tool_args: list[str]) -> None:
    binary = SPEC["binary"]
    workspace = state.get("workspace")
    if workspace:
        os.environ["OAUTH_TOKEN"] = get_databricks_token(workspace)
    os.execvp(binary, [binary, *tool_args])


def validate_cmd(binary: str) -> list[str]:
    return [binary, "-p", "say hi in 5 words or less", "--max-turns", "1"]
