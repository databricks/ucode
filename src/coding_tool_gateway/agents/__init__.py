"""Per-agent modules + dispatch helpers.

Each `agents.<tool>` module owns its own config layout, overlay rendering,
config-file writer, default-model selection, launch logic, and validation
command. This `__init__` aggregates the registry and exposes uniform
dispatchers for the rest of the codebase.

Adding a new agent: create `agents/<name>.py` exposing `SPEC`, `write_tool_config`,
`default_model`, `launch`, `validate_cmd`. Then add an entry to `_MODULES`
below and to `TOOL_ALIASES` if needed.
"""

from __future__ import annotations

import json
import shutil
import subprocess

from coding_tool_gateway.config_io import ToolSpec
from coding_tool_gateway.databricks import (
    ensure_databricks_auth,
    install_databricks_cli,
)
from coding_tool_gateway.state import load_state, save_state
from coding_tool_gateway.ui import (
    console,
    print_err,
    print_section,
    print_success,
    print_warning,
    spinner,
)

from . import claude, codex, gemini, opencode

_MODULES = {
    "codex": codex,
    "claude": claude,
    "gemini": gemini,
    "opencode": opencode,
}

TOOL_SPECS: dict[str, ToolSpec] = {name: module.SPEC for name, module in _MODULES.items()}

TOOL_ALIASES = {
    "codex": "codex",
    "claude": "claude",
    "claude-code": "claude",
    "gemini": "gemini",
    "gemini-cli": "gemini",
    "opencode": "opencode",
}

DEFAULT_TOOL = "codex"
BUNDLE_VERSION = 1


def normalize_tool(tool: str) -> str:
    normalized = TOOL_ALIASES.get(tool.strip().lower())
    if not normalized:
        raise RuntimeError(
            f"Unsupported tool '{tool}'. Use one of: codex, claude, gemini, opencode."
        )
    return normalized


def install_tool_binary(tool: str, *, strict: bool = True) -> bool:
    spec = TOOL_SPECS[tool]
    binary = spec["binary"]
    package = spec["package"]

    if shutil.which(binary):
        return True

    if not shutil.which("npm"):
        message = f"`{binary}` is not installed and npm is not available to install it."
        if strict:
            raise RuntimeError(message)
        print_warning(message)
        return False

    print_section("Bootstrap")
    print_warning(f"`{binary}` was not found. Installing {spec['display']}...")
    try:
        subprocess.run(["npm", "install", "-g", package], check=True, timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        message = f"Failed to install {spec['display']} automatically."
        if strict:
            raise RuntimeError(message) from exc
        print_warning(f"{message} Continuing without it.")
        return False

    if not shutil.which(binary):
        message = f"{spec['display']} install completed, but `{binary}` is still not on PATH."
        if strict:
            raise RuntimeError(message)
        print_warning(f"{message} Continuing without it.")
        return False

    return True


def ensure_tool_binary_available(tool: str) -> None:
    spec = TOOL_SPECS[tool]
    binary = spec["binary"]
    if shutil.which(binary):
        return
    raise RuntimeError(
        f"{spec['display']} is not installed (`{binary}` was not found on PATH). "
        f"Install it with `npm install -g {spec['package']}` or run "
        f"`coding-gateway configure` to try automatic installation."
    )


def ensure_bootstrap_dependencies(tool: str) -> None:
    install_databricks_cli()
    install_tool_binary(tool, strict=True)


def default_model_for_tool(tool: str, state: dict) -> str | None:
    return _MODULES[tool].default_model(state)


def resolve_launch_model(
    tool: str,
    state: dict,
    explicit_model: str | None,
) -> tuple[dict, str | None]:
    if tool == "codex":
        return state, None
    model = explicit_model or default_model_for_tool(tool, state)
    if not model:
        raise RuntimeError(
            f"No models available for {tool}. "
            f"Run `coding-gateway configure` to set up your workspace."
        )
    return state, model


def configure_tool(tool: str, state: dict, model: str | None = None) -> dict:
    result: dict | tuple[dict, str]
    if tool == "codex":
        result = codex.write_tool_config(state)
    else:
        if not model:
            raise RuntimeError(f"A {tool} model must be selected before configuration.")
        if tool == "claude":
            result = claude.write_tool_config(state, model)
        elif tool == "gemini":
            result = gemini.write_tool_config(state, model)
        else:
            result = opencode.write_tool_config(state, model)
    # gemini/opencode return (state, token); codex/claude return state
    if isinstance(result, tuple):
        return result[0]
    return result


def launch(tool: str, state: dict, tool_args: list[str]) -> None:
    _MODULES[tool].launch(state, tool_args)


def check_gateway_endpoint(state: dict, tool: str) -> bool:
    """V2-only: a tool is available iff we discovered models for it."""
    if tool == "claude":
        return bool(state.get("claude_models"))
    if tool == "opencode":
        return bool(state.get("opencode_models"))
    if tool == "codex":
        return bool(state.get("codex_models"))
    if tool == "gemini":
        return bool(state.get("gemini_models"))
    return False


def configure_single_tool(tool: str, state: dict) -> dict:
    """Check availability, configure, and persist state for one tool only."""
    with spinner(f"Checking {TOOL_SPECS[tool]['display']} availability..."):
        ok = check_gateway_endpoint(state, tool)
    if not ok:
        raise RuntimeError(f"{TOOL_SPECS[tool]['display']} is not available on this workspace.")
    if tool == "codex":
        state = configure_tool("codex", state)
    else:
        state, model = resolve_launch_model(tool, state, None)
        state = configure_tool(tool, state, model)
    available_tools = list(set((state.get("available_tools") or []) + [tool]))
    state["available_tools"] = available_tools
    save_state(state)
    return state


def configure_all_tools(state: dict) -> dict:
    available_tools: list[str] = []
    unavailable_tools: list[str] = []

    for tool in TOOL_SPECS:
        with spinner(f"Checking {TOOL_SPECS[tool]['display']} availability..."):
            ok = check_gateway_endpoint(state, tool)
        if ok:
            available_tools.append(tool)
        else:
            unavailable_tools.append(tool)

    for tool in unavailable_tools:
        print_err(f"{TOOL_SPECS[tool]['display']} is not available on this workspace")

    for tool in available_tools:
        if tool == "codex":
            state = configure_tool("codex", state)
        else:
            state, model = resolve_launch_model(tool, state, None)
            state = configure_tool(tool, state, model)

    state["available_tools"] = available_tools
    save_state(state)
    return state


def ensure_provider_state(tool: str) -> dict:
    state = load_state()
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured. Run `coding-gateway configure` first.")
    available_tools = state.get("available_tools") or []
    if tool not in available_tools:
        raise RuntimeError(
            f"{TOOL_SPECS[tool]['display']} is not available on this workspace. "
            f"Run `coding-gateway configure` to set up your agents."
        )
    ensure_databricks_auth(workspace)
    return state


def validate_tool(tool: str) -> tuple[bool, str]:
    """Invoke a tool with a simple prompt to verify it works. Returns (ok, error_msg)."""
    spec = TOOL_SPECS[tool]
    binary = spec["binary"]
    cmd = _MODULES[tool].validate_cmd(binary)
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True, ""
        output = (result.stderr or result.stdout or "").strip()
        for line in output.splitlines():
            if "error" in line.lower() and ("message" in line.lower() or ":" in line):
                msg = line.strip()
                if "error_code" in msg:
                    try:
                        payload = json.loads(msg[msg.index("{") : msg.rindex("}") + 1])
                        return False, payload.get("message", msg)
                    except (json.JSONDecodeError, ValueError):
                        pass
                return False, msg
        last_line = output.splitlines()[-1] if output else "unknown error"
        return False, last_line
    except OSError as exc:
        return False, str(exc)
    except subprocess.TimeoutExpired:
        return False, "timed out"


def validate_all_tools(state: dict) -> None:
    from rich.panel import Panel  # local to avoid bumping module-level deps

    from coding_tool_gateway.config_io import restore_file

    console.print()
    console.print(
        Panel(
            "Testing each tool with a quick message...",
            title="Validating",
            style="bold blue",
            expand=False,
        )
    )
    results: list[tuple[str, bool]] = []
    available_tools = list(state.get("available_tools") or [])
    for tool, spec in TOOL_SPECS.items():
        if tool not in available_tools:
            continue
        with spinner(f"Validating {spec['display']}..."):
            ok, err = validate_tool(tool)
        results.append((tool, ok))
        if ok:
            print_success(f"{spec['display']} is working")
        else:
            print_err(f"{spec['display']}: {err}")
            managed = bool(state.get("managed_configs", {}).get(tool))
            restore_file(spec["config_path"], spec["backup_path"], managed)
            available_tools.remove(tool)
    state["available_tools"] = available_tools
    save_state(state)

    console.print()
    success_tools = [(t, s) for t, s in results if s]
    if success_tools:
        lines = []
        for tool, _ in success_tools:
            spec = TOOL_SPECS[tool]
            lines.append(
                f"[green]✓[/green] [bold]{spec['display']}[/bold] — "
                f"run with [cyan]coding-gateway --agent {tool}[/cyan]"
            )
        console.print(Panel("\n".join(lines), title="Ready", style="green", expand=False))
