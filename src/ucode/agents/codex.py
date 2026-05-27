"""Codex agent: writes ~/.codex/ucode.config.toml for Databricks-backed Codex."""

from __future__ import annotations

import os
import re
from pathlib import Path

from ucode.agent_updates import available_npm_package_update
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
CODEX_PROFILE_NAME = "ucode"
CODEX_CONFIG_PATH = CODEX_CONFIG_DIR / f"{CODEX_PROFILE_NAME}.config.toml"
CODEX_BACKUP_PATH = APP_DIR / "codex-ucode-config.backup.toml"
CODEX_MODEL_PROVIDER_NAME = "ucode-databricks"
MINIMUM_CODEX_VERSION = (0, 134, 0)
MINIMUM_CODEX_VERSION_TEXT = "0.134.0"


SPEC: ToolSpec = {
    "binary": "codex",
    "package": "@openai/codex",
    "display": "Codex",
    "config_path": CODEX_CONFIG_PATH,
    "backup_path": CODEX_BACKUP_PATH,
}

MANAGED_KEYS: list[list[str]] = [
    ["model_provider"],
    ["model"],
    ["model_providers", CODEX_MODEL_PROVIDER_NAME],
    ["model_providers", CODEX_MODEL_PROVIDER_NAME, "http_headers"],
]


def is_update_available() -> tuple[str, str] | None:
    return available_npm_package_update(SPEC["package"])


def _parse_version(value: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _installed_version_status() -> tuple[str, bool] | None:
    version = agent_version(SPEC["binary"])
    parsed = _parse_version(version)
    if parsed is None:
        return None
    return version, parsed < MINIMUM_CODEX_VERSION


def minimum_version_error() -> str | None:
    status = _installed_version_status()
    if status is None:
        return None
    version, is_too_old = status
    if not is_too_old:
        return None
    return (
        f"Codex CLI {version} is too old for ucode's Codex profile config. "
        f"Codex CLI must be updated to {MINIMUM_CODEX_VERSION_TEXT} or newer; "
        f"run `npm install -g {SPEC['package']}` or `ucode configure`."
    )


def required_update_message() -> str | None:
    status = _installed_version_status()
    if status is None:
        return None
    version, is_too_old = status
    if not is_too_old:
        return None
    return (
        f"Codex CLI {version} is older than required {MINIMUM_CODEX_VERSION_TEXT}; "
        "updating Codex is required for ucode's Codex profile config."
    )


def render_overlay(
    workspace: str, model: str | None = None, databricks_profile: str | None = None
) -> dict:
    auth_command = build_auth_shell_command(workspace, databricks_profile)
    base_url = build_tool_base_url("codex", workspace)
    overlay: dict = {"model_provider": CODEX_MODEL_PROVIDER_NAME}
    if model:
        overlay["model"] = model
    overlay["model_providers"] = {
        CODEX_MODEL_PROVIDER_NAME: {
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
    }
    return overlay


def _legacy_config_path() -> Path:
    return CODEX_CONFIG_PATH.parent / "config.toml"


def _legacy_backup_path() -> Path:
    return CODEX_BACKUP_PATH.with_name("codex-legacy-config.backup.toml")


def _remove_legacy_ucode_profile() -> None:
    """Remove ucode's old [profiles.ucode] entry from shared Codex config."""
    path = _legacy_config_path()
    if path == CODEX_CONFIG_PATH or not path.exists():
        return

    doc = read_toml_safe(path)
    changed = False

    profiles = doc.get("profiles")
    if isinstance(profiles, dict) and CODEX_PROFILE_NAME in profiles:
        backup_existing_file(path, _legacy_backup_path())
        profiles.pop(CODEX_PROFILE_NAME, None)
        if not profiles:
            doc.pop("profiles", None)
        changed = True

    if doc.get("profile") == CODEX_PROFILE_NAME:
        backup_existing_file(path, _legacy_backup_path())
        doc.pop("profile", None)
        changed = True

    if changed:
        write_toml_file(path, doc)


def write_tool_config(state: dict, model: str | None = None) -> dict:
    _remove_legacy_ucode_profile()
    backup_existing_file(CODEX_CONFIG_PATH, CODEX_BACKUP_PATH)
    overlay = render_overlay(
        state["workspace"], model or default_model(state), state.get("profile")
    )
    doc = read_toml_safe(CODEX_CONFIG_PATH)
    deep_merge_dict(doc, overlay)
    write_toml_file(CODEX_CONFIG_PATH, doc)
    state = mark_tool_managed(state, "codex", MANAGED_KEYS)
    save_state(state)
    return state


def default_model(state: dict) -> str | None:
    codex_models = state.get("codex_models") or []
    return codex_models[0] if codex_models else None


def launch(state: dict, tool_args: list[str]) -> None:
    binary = SPEC["binary"]
    workspace = state.get("workspace")
    if workspace:
        os.environ["OAUTH_TOKEN"] = get_databricks_token(workspace, state.get("profile"))
    os.execvp(binary, [binary, "--profile", CODEX_PROFILE_NAME, *tool_args])


def validate_cmd(binary: str) -> list[str]:
    return [
        binary,
        "--profile",
        CODEX_PROFILE_NAME,
        "exec",
        "--skip-git-repo-check",
        "say hi in 5 words or less",
    ]
