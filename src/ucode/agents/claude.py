"""Claude Code agent: writes ~/.claude/settings.json env block."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
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
    build_auth_shell_command,
    build_tool_base_url,
    get_databricks_token,
)
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version
from ucode.tracing import tracing_env
from ucode.ui import print_note, print_success, print_warning

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
_CLAUDE_MODEL_RE = re.compile(r"^databricks-claude-(opus|sonnet)-(\d+)-(\d+)(.*)$")

# Env keys the MLflow Stop hook reads to route traces. Written into the
# settings `env` block alongside the hook itself.
CLAUDE_TRACING_ENV_KEYS = (
    "MLFLOW_CLAUDE_TRACING_ENABLED",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_ID",
    "MLFLOW_TRACING_SQL_WAREHOUSE_ID",
)
CLAUDE_TRACING_STOP_HOOK_SUFFIX = " autolog claude stop-hook"
# Tracing is driven by an `mlflow autolog claude stop-hook` Stop hook, run by
# the `mlflow` CLI on each session end. Pin to 3.11.x: 3.12 dropped the Unity
# Catalog trace-write path, so traces silently land in the classic store
# instead of the experiment's UC table. ucode installs this via `uv tool` at
# `configure tracing` time (where UV_INDEX_URL is set), then writes the hook
# with the resolved absolute path — so the hook needs no uv or index at run
# time, and can't be shadowed by a project venv's mlflow.
MLFLOW_CLI_SPEC = "mlflow[databricks]>=3.11,<3.12"
MINIMUM_MLFLOW_VERSION = (3, 11)
# Upper bound (exclusive) — an installed mlflow at or above this is too new and
# must be replaced, not just left alone.
MAXIMUM_MLFLOW_VERSION = (3, 12)


def _web_search_mcp_entry(workspace: str, search_model: str, profile: str | None = None) -> dict:
    """Stdio MCP server entry pointing at `ucode mcp web-search`. Resolves
    the absolute path to the `ucode` binary so launchers without the right
    PATH (e.g. desktop GUI launchers) still find it."""
    ucode_binary = shutil.which("ucode") or "ucode"
    env: dict[str, str] = {
        "DATABRICKS_HOST": workspace,
        "UCODE_WEB_SEARCH_MODEL": search_model,
    }
    if profile:
        env["DATABRICKS_CONFIG_PROFILE"] = profile
    return {
        "type": "stdio",
        "command": ucode_binary,
        "args": ["mcp", "web-search"],
        "env": env,
    }


def render_overlay(
    workspace: str,
    model: str,
    claude_models: dict[str, str] | None = None,
    disable_web_search: bool = False,
    profile: str | None = None,
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
        "ANTHROPIC_MODEL": _maybe_add_1m_suffix(model),
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_CUSTOM_HEADERS": custom_headers,
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_API_KEY_HELPER_TTL_MS": "900000",
    }
    if claude_models:
        if claude_models.get("opus"):
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = _maybe_add_1m_suffix(claude_models["opus"])
        if claude_models.get("sonnet"):
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = _maybe_add_1m_suffix(claude_models["sonnet"])
        if claude_models.get("haiku"):
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = claude_models["haiku"]
    overlay: dict = {"apiKeyHelper": build_auth_shell_command(workspace, profile), "env": env}
    keys: list[list[str]] = [["apiKeyHelper"]] + [["env", k] for k in env]

    # Disable Claude Code's built-in WebSearch (it routes through Anthropic's
    # hosted infra and fails through the Databricks gateway). The replacement
    # `web_search` MCP server is registered separately via the claude CLI.
    if disable_web_search:
        overlay["disabledTools"] = ["WebSearch"]
        keys.append(["disabledTools"])

    return overlay, keys


def _maybe_add_1m_suffix(model: str) -> str:
    if model.endswith("[1m]"):
        return model
    match = _CLAUDE_MODEL_RE.match(model)
    if not match:
        return model

    family, major_raw, minor_raw, _ = match.groups()
    major = int(major_raw)
    minor = int(minor_raw)
    should_suffix = (family == "opus" and (major, minor) >= (4, 6)) or (
        family == "sonnet" and (major, minor) >= (4, 6)
    )
    return f"{model}[1m]" if should_suffix else model


def _register_web_search_mcp(workspace: str, search_model: str, profile: str | None = None) -> bool:
    """Register (or replace) the web_search MCP server in Claude Code's user
    scope via `claude mcp add-json`. Removes any prior entry first so re-runs
    pick up changes to the workspace, model, or ucode binary path.

    Returns True if registration succeeded. Failures are non-blocking: we warn
    and return False so the rest of `ucode claude` setup can complete.
    """
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
    entry = _web_search_mcp_entry(workspace, search_model, profile)
    try:
        add_claude_mcp_server(WEB_SEARCH_MCP_NAME, entry)
    except RuntimeError as exc:
        print_warning(f"{exc} Web search will be unavailable; re-run `ucode claude` to retry.")
        return False
    return True


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
        profile=state.get("profile"),
    )
    tracing_env_vars = tracing_env(state, "claude")
    stop_hook_command = claude_tracing_stop_hook_command() if tracing_env_vars else None
    if tracing_env_vars:
        overlay["env"]["MLFLOW_CLAUDE_TRACING_ENABLED"] = "true"
        overlay["env"].update(tracing_env_vars)
        managed_keys = managed_keys + [["env", key] for key in CLAUDE_TRACING_ENV_KEYS]
        if stop_hook_command:
            managed_keys = managed_keys + [["hooks", "Stop"]]
        else:
            print_warning(
                "MLflow tracing env was written, but the `mlflow` CLI could not be located "
                "to install the Claude Stop hook — traces won't be emitted. Re-run "
                "`ucode configure tracing`."
            )

    existing = read_json_safe(CLAUDE_SETTINGS_PATH)
    merged = deep_merge_dict(existing, overlay)
    if tracing_env_vars and stop_hook_command:
        _upsert_tracing_stop_hook(merged, stop_hook_command)
    if not tracing_env_vars:
        env_block = merged.get("env")
        if isinstance(env_block, dict):
            for key in CLAUDE_TRACING_ENV_KEYS:
                env_block.pop(key, None)
        # Strip only ucode's tracing Stop hook so user hooks stay intact.
        _remove_tracing_stop_hook(merged)
    write_json_file(CLAUDE_SETTINGS_PATH, merged)

    if web_search_model:
        _register_web_search_mcp(state["workspace"], web_search_model, state.get("profile"))

    state = mark_tool_managed(state, "claude", managed_keys)
    save_state(state)
    return state


def _is_tracing_stop_hook(hook: object) -> bool:
    if not isinstance(hook, dict):
        return False
    hook = cast(dict, hook)
    if hook.get("type") != "command":
        return False
    command = hook.get("command")
    return isinstance(command, str) and command.endswith(CLAUDE_TRACING_STOP_HOOK_SUFFIX)


def _remove_tracing_stop_hook(settings: dict) -> None:
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return
    stop_entries = hooks.get("Stop")
    if not isinstance(stop_entries, list):
        return

    cleaned_entries = []
    for entry in stop_entries:
        if not isinstance(entry, dict):
            cleaned_entries.append(entry)
            continue
        hook_list = entry.get("hooks")
        if not isinstance(hook_list, list):
            cleaned_entries.append(entry)
            continue
        cleaned_hooks = [hook for hook in hook_list if not _is_tracing_stop_hook(hook)]
        if cleaned_hooks:
            cleaned_entry = dict(entry)
            cleaned_entry["hooks"] = cleaned_hooks
            cleaned_entries.append(cleaned_entry)

    if cleaned_entries:
        hooks["Stop"] = cleaned_entries
    else:
        hooks.pop("Stop", None)
    if not hooks:
        settings.pop("hooks", None)


def _upsert_tracing_stop_hook(settings: dict, command: str) -> None:
    _remove_tracing_stop_hook(settings)
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}
        settings["hooks"] = hooks
    stop_entries = hooks.get("Stop")
    if not isinstance(stop_entries, list):
        stop_entries = []
        hooks["Stop"] = stop_entries
    stop_entries.append({"hooks": [{"type": "command", "command": command}]})


def ensure_tracing_runtime() -> bool:
    """Ensure the MLflow tracing runtime is ready: a pinned `mlflow` CLI (3.11.x)
    installed via `uv tool`, whose absolute path the Stop hook will call.

    Best-effort — warns and returns False if it can't be set up, so
    `ucode configure tracing` can still finish for other agents."""
    return _ensure_mlflow_cli()


def _parse_mlflow_version(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\.(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _uv_tool_mlflow_path() -> str | None:
    """Absolute path to the `mlflow` installed by `uv tool`, or None.

    Resolved from `uv tool dir --bin` rather than ``shutil.which`` so a project
    venv's (possibly wrong-versioned) mlflow can't shadow the one ucode pins —
    the Stop hook must always run the uv-tool copy."""
    if not shutil.which("uv"):
        return None
    try:
        result = subprocess.run(
            ["uv", "tool", "dir", "--bin"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    bin_dir = (result.stdout or "").strip()
    if result.returncode != 0 or not bin_dir:
        return None
    candidate = Path(bin_dir) / "mlflow"
    return str(candidate) if candidate.exists() else None


def _installed_mlflow_version() -> tuple[int, int] | None:
    """The (major, minor) of the uv-tool `mlflow`, or None if absent."""
    path = _uv_tool_mlflow_path()
    if not path:
        return None
    try:
        result = subprocess.run(
            [path, "--version"], check=False, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _parse_mlflow_version(result.stdout or result.stderr or "")


def claude_tracing_stop_hook_command() -> str | None:
    """The Stop hook command string: the absolute uv-tool `mlflow` invoking its
    `autolog claude stop-hook` handler. None when mlflow isn't installed.

    Using the absolute path means the hook needs neither `uv` nor a package
    index at run time (the minimal env Claude runs hooks in lacks UV_INDEX_URL),
    and can't be shadowed by another mlflow on PATH."""
    path = _uv_tool_mlflow_path()
    if not path:
        return None
    return f"{path} autolog claude stop-hook"


def _ensure_mlflow_cli() -> bool:
    """Ensure the pinned `mlflow` CLI (3.11.x) is installed via `uv tool`,
    installing or replacing an out-of-range version when needed."""
    current = _installed_mlflow_version()
    if current and MINIMUM_MLFLOW_VERSION <= current < MAXIMUM_MLFLOW_VERSION:
        return True

    if not shutil.which("uv"):
        verb = "replace" if current else "install"
        print_warning(
            f"Claude tracing needs the `mlflow` CLI ({MLFLOW_CLI_SPEC}), but `uv` is not "
            f'available to {verb} it. Run `uv tool install "{MLFLOW_CLI_SPEC}"`, then '
            "re-run `ucode configure tracing`."
        )
        return False

    print_note(f"{'Replacing' if current else 'Installing'} the mlflow CLI ({MLFLOW_CLI_SPEC})...")
    # Always --force: it installs fresh when absent and replaces in place when
    # present. Keying it on `current` broke when an mlflow existed but its
    # version couldn't be parsed — uv still errors "Executable already exists".
    cmd = ["uv", "tool", "install", "--force", MLFLOW_CLI_SPEC]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print_warning(f"Could not install the mlflow CLI automatically: {exc}")
        return False

    if not _uv_tool_mlflow_path():
        print_warning(
            "Installed mlflow via `uv tool`, but its binary could not be located. "
            "Re-run `ucode configure tracing`."
        )
        return False
    print_success("mlflow CLI ready")
    return True


def default_model(state: dict) -> str | None:
    claude_models = state.get("claude_models") or {}
    return claude_models.get("opus") or claude_models.get("sonnet") or claude_models.get("haiku")


def launch(state: dict, tool_args: list[str]) -> None:
    binary = SPEC["binary"]
    workspace = state.get("workspace")
    if workspace:
        os.environ["OAUTH_TOKEN"] = get_databricks_token(workspace, state.get("profile"))
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
