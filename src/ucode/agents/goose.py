"""Goose agent: merges Databricks settings into ~/.config/goose/config.yaml.

Goose has a built-in Databricks provider that reads DATABRICKS_HOST from the
config file and DATABRICKS_TOKEN from the environment (env var takes precedence
over keyring). We merge only the three keys we own into the existing config so
that user-defined extensions, preferences, and other settings are preserved.

The token is injected as DATABRICKS_TOKEN at launch and refreshed every 30
minutes so long-running sessions stay authenticated.

Install goose from https://github.com/aaif-goose/goose — it ships as a native
binary (not an npm package), typically installed to ~/.local/bin via:
  curl -fsSL https://github.com/aaif-goose/goose/releases/download/stable/download_cli.sh | bash
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from pathlib import Path

from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_yaml_safe,
    write_yaml_file,
)
from ucode.databricks import (
    TOKEN_REFRESH_INTERVAL_SECONDS,
    get_databricks_token,
)
from ucode.state import mark_tool_managed, save_state

GOOSE_CONFIG_DIR = Path.home() / ".config" / "goose"
GOOSE_CONFIG_PATH = GOOSE_CONFIG_DIR / "config.yaml"
GOOSE_BACKUP_PATH = APP_DIR / "goose-config.backup.yaml"

SPEC: ToolSpec = {
    "binary": "goose",
    "package": "",  # not an npm package; install from https://github.com/aaif-goose/goose
    "display": "Goose",
    "config_path": GOOSE_CONFIG_PATH,
    "backup_path": GOOSE_BACKUP_PATH,
}

MANAGED_KEYS: list[str] = [
    "DATABRICKS_HOST",
    "GOOSE_PROVIDER",
    "GOOSE_MODEL",
]


def is_update_available() -> tuple[str, str] | None:
    return None  # no npm update check for native binary


def default_model(state: dict) -> str | None:
    """Prefer Claude sonnet, then opus, then haiku."""
    claude_models = state.get("claude_models") or {}
    for family in ("sonnet", "opus", "haiku"):
        if claude_models.get(family):
            return claude_models[family]
    return None


def render_overlay(workspace: str, model: str) -> dict:
    """Return only the keys ucode manages — merged into the existing config."""
    return {
        "DATABRICKS_HOST": workspace,
        "GOOSE_PROVIDER": "databricks",
        "GOOSE_MODEL": model,
    }


def build_runtime_env(workspace: str, token: str) -> dict[str, str]:
    env = os.environ.copy()
    env["DATABRICKS_HOST"] = workspace
    env["DATABRICKS_TOKEN"] = token
    return env


def write_tool_config(
    state: dict,
    model: str,
    token: str | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[dict, str]:
    backup_existing_file(GOOSE_CONFIG_PATH, GOOSE_BACKUP_PATH)
    if token is None:
        token = get_databricks_token(state["workspace"], force_refresh=force_refresh)
    overlay = render_overlay(state["workspace"], model)
    existing = read_yaml_safe(GOOSE_CONFIG_PATH)
    deep_merge_dict(existing, overlay)
    write_yaml_file(GOOSE_CONFIG_PATH, existing)
    state = mark_tool_managed(state, "goose", MANAGED_KEYS)
    save_state(state)
    return state, token


def _refresh_token_once(state: dict, *, force_refresh: bool = False) -> tuple[str, str]:
    model = default_model(state)
    if not model:
        raise RuntimeError("No Goose model is available on this workspace.")
    _, token = write_tool_config(state, model, force_refresh=force_refresh)
    return model, token


def _refresh_forever(state: dict, stop_event: threading.Event) -> None:
    while not stop_event.wait(TOKEN_REFRESH_INTERVAL_SECONDS):
        try:
            _refresh_token_once(state, force_refresh=True)
        except RuntimeError:
            continue


def launch(state: dict, tool_args: list[str]) -> None:
    model, token = _refresh_token_once(state)
    env = build_runtime_env(state["workspace"], token)

    stop_event = threading.Event()
    refresher = threading.Thread(
        target=_refresh_forever,
        args=(state, stop_event),
        daemon=True,
    )
    refresher.start()

    proc = subprocess.Popen(["goose", "session", *tool_args], env=env)
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
    return [
        binary,
        "run",
        "--text",
        "say hi in 5 words or less",
        "--no-session",
        "--max-turns",
        "1",
    ]


def validate_env(state: dict) -> dict[str, str]:
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured.")
    if not default_model(state):
        raise RuntimeError("No Goose model is available on this workspace.")
    token = get_databricks_token(workspace)
    return build_runtime_env(workspace, token)
