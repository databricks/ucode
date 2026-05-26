"""Gemini CLI agent: writes ~/.gemini/.env and runs with periodic token refresh."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from pathlib import Path

from ucode.agent_updates import available_npm_package_update
from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    parse_dotenv,
    write_dotenv,
)
from ucode.databricks import (
    TOKEN_REFRESH_INTERVAL_SECONDS,
    build_tool_base_url,
    get_databricks_token,
)
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version

GEMINI_CONFIG_DIR = Path.home() / ".gemini"
GEMINI_ENV_PATH = GEMINI_CONFIG_DIR / "ucode.env"
GEMINI_BACKUP_PATH = APP_DIR / "gemini-ucode-env.backup"

SPEC: ToolSpec = {
    "binary": "gemini",
    "package": "@google/gemini-cli",
    "display": "Gemini CLI",
    "config_path": GEMINI_ENV_PATH,
    "backup_path": GEMINI_BACKUP_PATH,
}

MANAGED_KEYS: list[str] = [
    "GEMINI_MODEL",
    "GOOGLE_GEMINI_BASE_URL",
    "GEMINI_API_KEY_AUTH_MECHANISM",
    "GEMINI_API_KEY",
    "GEMINI_CLI_CUSTOM_HEADERS",
    "OAUTH_TOKEN",
]


def is_update_available() -> tuple[str, str] | None:
    return available_npm_package_update(SPEC["package"])


def render_env_overlay(workspace: str, model: str, token: str) -> dict[str, str]:
    # Gemini CLI parses GEMINI_CLI_CUSTOM_HEADERS as comma-separated
    # `Key:Value` pairs and spreads them after the SDK's default User-Agent,
    # so a key named `User-Agent` overrides the default. Resolved via
    # upstream issue google-gemini/gemini-cli#10088.
    custom_headers = f"User-Agent:ucode/{ucode_version()} gemini/{agent_version('gemini')}"
    return {
        "GEMINI_MODEL": model,
        "GOOGLE_GEMINI_BASE_URL": build_tool_base_url("gemini", workspace),
        "GEMINI_API_KEY_AUTH_MECHANISM": "bearer",
        "GEMINI_API_KEY": token,
        "GEMINI_CLI_CUSTOM_HEADERS": custom_headers,
        "OAUTH_TOKEN": token,
    }


def build_runtime_env(workspace: str, model: str, token: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(render_env_overlay(workspace, model, token))
    # Newer Gemini CLI releases refuse to run in untrusted directories;
    # opt every launch into trust so `ucode gemini` works in any folder.
    env["GEMINI_CLI_TRUST_WORKSPACE"] = "true"
    return env


def write_tool_config(
    state: dict,
    model: str,
    token: str | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[dict, str]:
    backup_existing_file(GEMINI_ENV_PATH, GEMINI_BACKUP_PATH)
    if token is None:
        token = get_databricks_token(
            state["workspace"], state.get("profile"), force_refresh=force_refresh
        )
    overlay = render_env_overlay(state["workspace"], model, token)
    existing = parse_dotenv(GEMINI_ENV_PATH)
    existing.update(overlay)
    write_dotenv(GEMINI_ENV_PATH, existing)
    state = mark_tool_managed(state, "gemini", MANAGED_KEYS)
    save_state(state)
    return state, token


def default_model(state: dict) -> str | None:
    gemini_models = state.get("gemini_models") or []
    return gemini_models[0] if gemini_models else None


def _refresh_token_once(state: dict, *, force_refresh: bool = False) -> str:
    model = default_model(state)
    if not model:
        raise RuntimeError("No Gemini model is configured.")
    _, token = write_tool_config(state, model, force_refresh=force_refresh)
    return token


def _refresh_forever(state: dict, stop_event: threading.Event) -> None:
    while not stop_event.wait(TOKEN_REFRESH_INTERVAL_SECONDS):
        try:
            _refresh_token_once(state, force_refresh=True)
        except RuntimeError:
            continue


def launch(state: dict, tool_args: list[str]) -> None:
    token = _refresh_token_once(state)
    model = default_model(state)
    if not model:
        raise RuntimeError("No Gemini model is configured.")
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
    return [binary, "-p", "say hi in 5 words or less"]


def validate_env(state: dict) -> dict[str, str]:
    """Inject env vars for the validation subprocess.

    The Gemini CLI's .env auto-loading skips ~/.gemini/.env when run from an
    untrusted folder, so we cannot rely on it during validation.
    """
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured.")
    model = default_model(state)
    if not model:
        raise RuntimeError("No Gemini model is configured.")
    token = get_databricks_token(workspace, state.get("profile"))
    return build_runtime_env(workspace, model, token)
