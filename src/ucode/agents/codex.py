"""Codex agent: writes ~/.codex/config.toml with a Databricks-backed model provider."""

from __future__ import annotations

import os
from pathlib import Path

from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_toml_safe,
    write_toml_file,
)
from ucode.databricks import (
    build_auth_shell_command,
    build_tool_base_url,
    get_databricks_token,
)
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version

CODEX_CONFIG_DIR = Path.home() / ".codex"
CODEX_CONFIG_PATH = CODEX_CONFIG_DIR / "config.toml"
CODEX_BACKUP_PATH = APP_DIR / "codex-config.backup.toml"

SPEC: ToolSpec = {
    "binary": "codex",
    "package": "@openai/codex",
    "display": "Codex",
    "config_path": CODEX_CONFIG_PATH,
    "backup_path": CODEX_BACKUP_PATH,
}

MANAGED_KEYS: list[list[str]] = [
    ["profile"],
    ["profiles", "default", "model_provider"],
    ["model_providers", "Databricks"],
    ["model_providers", "Databricks", "http_headers"],
]


def render_overlay(workspace: str) -> dict:
    auth_command = build_auth_shell_command(workspace)
    base_url = build_tool_base_url("codex", workspace)
    return {
        "profile": "default",
        "profiles": {"default": {"model_provider": "Databricks"}},
        "model_providers": {
            "Databricks": {
                "name": "Databricks AI Gateway",
                "base_url": base_url,
                "wire_api": "responses",
                "http_headers": {
                    "User-Agent": f"ucode/{ucode_version()} codex/{agent_version('codex')}",
                },
                "auth": {
                    "command": "sh",
                    "args": ["-c", auth_command],
                    "timeout_ms": 5000,
                    "refresh_interval_ms": 900000,
                },
            }
        },
    }


def write_tool_config(state: dict, model: str | None = None) -> dict:
    backup_existing_file(CODEX_CONFIG_PATH, CODEX_BACKUP_PATH)
    overlay = render_overlay(state["workspace"])
    doc = read_toml_safe(CODEX_CONFIG_PATH)
    deep_merge_dict(doc, overlay)
    write_toml_file(CODEX_CONFIG_PATH, doc)
    state = mark_tool_managed(state, "codex", MANAGED_KEYS)
    save_state(state)
    return state


def default_model(state: dict) -> str | None:
    return None


def launch(state: dict, tool_args: list[str]) -> None:
    binary = SPEC["binary"]
    workspace = state.get("workspace")
    if workspace:
        os.environ["OAUTH_TOKEN"] = get_databricks_token(workspace)
    os.execvp(binary, [binary, *tool_args])


def validate_cmd(binary: str) -> list[str]:
    return [binary, "exec", "--skip-git-repo-check", "say hi in 5 words or less"]
