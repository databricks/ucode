"""Cursor IDE agent: writes ~/.ucode/cursor-databricks.json and opens Cursor.

Cursor stores API keys in secure storage, so ucode cannot patch Cursor settings
directly. Instead we persist a reference config (base URL, token, model list)
and print the steps to add each Databricks foundation-model endpoint in
Cursor Settings > Models.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    read_json_safe,
    write_json_file,
)
from ucode.databricks import (
    build_tool_base_url,
    get_databricks_token,
)
from ucode.state import mark_tool_managed, save_state
from ucode.ui import console, print_kv, print_note, print_section

CURSOR_CONFIG_PATH = APP_DIR / "cursor-databricks.json"
CURSOR_BACKUP_PATH = APP_DIR / "cursor-databricks.backup.json"
CURSOR_MAC_APP = Path("/Applications/Cursor.app")
CURSOR_MAC_CLI = CURSOR_MAC_APP / "Contents/Resources/app/bin/cursor"
# ~/.local/bin/cursor is often an agent-only shim that cannot open the IDE.
CURSOR_AGENT_SHIM = Path.home() / ".local/bin/cursor"

SPEC: ToolSpec = {
    "binary": "cursor",
    "package": "",
    "display": "Cursor",
    "config_path": CURSOR_CONFIG_PATH,
    "backup_path": CURSOR_BACKUP_PATH,
}

MANAGED_KEYS = [
    "openai_base_url",
    "openai_api_key",
    "models",
    "default_model",
    "workspace",
]

SETUP_STEPS = [
    "Open Cursor → Settings → Cursor Settings → Models → API Keys.",
    'Enable "Override OpenAI Base URL" and paste `openai_base_url` from the config file.',
    'Paste `openai_api_key` into the OpenAI API Key field (Databricks OAuth token).',
    'For each model in `models`, click "+ Add Custom Model", paste the name, and enable it.',
    "In Ask mode (Cmd+L / Ctrl+L), pick your custom model and send a test message.",
]


def is_update_available() -> tuple[str, str] | None:
    return None


def is_installed() -> bool:
    if _cursor_executable() is not None:
        return True
    return platform.system() == "Darwin" and CURSOR_MAC_APP.exists()


def _cursor_executable() -> Path | None:
    """Resolve a cursor CLI that can open the IDE.

    On macOS, prefer the binary inside Cursor.app over PATH. The common
    ``~/.local/bin/cursor`` shim only supports ``cursor agent`` and errors
    when asked to open a folder.
    """
    if platform.system() == "Darwin" and CURSOR_MAC_CLI.is_file():
        return CURSOR_MAC_CLI
    which = shutil.which(SPEC["binary"])
    if not which:
        return None
    path = Path(which)
    if (
        platform.system() == "Darwin"
        and path.resolve() == CURSOR_AGENT_SHIM.resolve()
        and CURSOR_MAC_APP.exists()
    ):
        return CURSOR_MAC_CLI if CURSOR_MAC_CLI.is_file() else None
    return path


def default_model(state: dict) -> str | None:
    models = state.get("cursor_models") or []
    return models[0] if models else None


def render_config_payload(workspace: str, token: str, models: list[str], model: str) -> dict:
    return {
        "workspace": workspace,
        "openai_base_url": build_tool_base_url("cursor", workspace),
        "openai_api_key": token,
        "models": models,
        "default_model": model,
        "cursor_settings_path": "Settings → Cursor Settings → Models → API Keys",
        "setup_steps": SETUP_STEPS,
    }


def write_tool_config(
    state: dict,
    model: str | None = None,
    token: str | None = None,
    *,
    force_refresh: bool = False,
) -> dict:
    models = list(state.get("cursor_models") or [])
    if not models:
        raise RuntimeError(
            "No Cursor-compatible models on this workspace. "
            "Enable foundation-model endpoints with cursor/v1/chat/completions support."
        )
    chosen = model or default_model(state)
    if not chosen:
        raise RuntimeError("No Cursor model is available on this workspace.")
    if chosen not in models:
        models = [chosen, *[m for m in models if m != chosen]]

    backup_existing_file(CURSOR_CONFIG_PATH, CURSOR_BACKUP_PATH)
    if token is None:
        token = get_databricks_token(state["workspace"], force_refresh=force_refresh)

    payload = render_config_payload(state["workspace"], token, models, chosen)
    existing = read_json_safe(CURSOR_CONFIG_PATH)
    existing.update(payload)
    write_json_file(CURSOR_CONFIG_PATH, existing)

    state = mark_tool_managed(state, "cursor", MANAGED_KEYS)
    state["cursor_selected_model"] = chosen
    save_state(state)
    return state


def print_setup_instructions(payload: dict) -> None:
    from rich.panel import Panel

    lines = [
        f"[bold]Config file:[/bold] [cyan]{CURSOR_CONFIG_PATH}[/cyan]",
        f"[bold]Base URL:[/bold] {payload['openai_base_url']}",
        f"[bold]Default model:[/bold] {payload['default_model']}",
        "",
        "[bold]Add these custom models in Cursor:[/bold]",
    ]
    for name in payload.get("models") or []:
        lines.append(f"  • {name}")
    lines.append("")
    lines.append("[dim]Cursor does not expose a config file for API keys — use the UI steps below.[/dim]")
    console.print(Panel("\n".join(lines), title="Cursor + Databricks", style="blue", expand=False))
    print_section("Manual steps in Cursor")
    for idx, step in enumerate(SETUP_STEPS, start=1):
        print_note(f"{idx}. {step}")


def _open_cursor(tool_args: list[str]) -> None:
    cwd = tool_args[0] if len(tool_args) == 1 and not tool_args[0].startswith("-") else "."
    args = tool_args or [cwd]
    executable = _cursor_executable()
    if executable is not None:
        subprocess.Popen([str(executable), *args])
        return
    if platform.system() == "Darwin" and CURSOR_MAC_APP.exists():
        subprocess.Popen(["open", "-a", "Cursor", *([] if args == ["."] else args)])
        return
    raise RuntimeError(
        "Cursor is not installed. Install from https://cursor.com/download "
        "or ensure the `cursor` CLI is on PATH."
    )


def launch(state: dict, tool_args: list[str]) -> None:
    model = default_model(state)
    if not model:
        raise RuntimeError("No Cursor model is configured.")
    write_tool_config(state, model, force_refresh=True)
    payload = read_json_safe(CURSOR_CONFIG_PATH)
    print_setup_instructions(payload)
    print_kv("Opening", SPEC["display"])
    _open_cursor(tool_args)


def validate_cmd(binary: str) -> list[str]:
    return []


def validate_gateway(
    workspace: str,
    token: str,
    models: list[str],
    *,
    model: str | None = None,
) -> tuple[bool, str]:
    """Validate Cursor + AI Gateway setup.

    The Cursor gateway does not implement OpenAI ``GET /v1/models``; dogfood returns
    BAD_REQUEST for that path. We already proved availability via
    ``fetch_cursor_models`` (foundation endpoints with ``cursor/v1/chat/completions``).
    Final setup is completed manually in the Cursor IDE UI.
    """
    del workspace, token, model  # reserved for a future live probe if the API stabilizes
    if not models:
        return False, "no Cursor-compatible foundation-model endpoints on this workspace"
    if not CURSOR_CONFIG_PATH.exists():
        return False, f"config file missing: {CURSOR_CONFIG_PATH}"
    return True, ""
