"""Claude Code agent: writes ~/.claude/settings.json env block."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
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

# Env keys consumed by the MLflow Claude tracing plugin. Written into the
# settings `env` block; the plugin runtime (installed separately) reads them.
CLAUDE_TRACING_ENV_KEYS = (
    "MLFLOW_CLAUDE_TRACING_ENABLED",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_ID",
)
CLAUDE_TRACING_MARKETPLACE = "mlflow/mlflow"
CLAUDE_TRACING_PLUGIN = "mlflow-tracing@mlflow-plugins"
# The plugin runtime shells out to the `mlflow` CLI, so it must be on PATH at
# this minimum version. ucode installs/upgrades it via `uv tool`.
MLFLOW_CLI_SPEC = "mlflow[databricks]>=3.4"
MINIMUM_MLFLOW_VERSION = (3, 4)


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
    if tracing_env_vars:
        overlay["env"]["MLFLOW_CLAUDE_TRACING_ENABLED"] = "true"
        overlay["env"].update(tracing_env_vars)
        managed_keys = managed_keys + [["env", key] for key in CLAUDE_TRACING_ENV_KEYS]

    existing = read_json_safe(CLAUDE_SETTINGS_PATH)
    merged = deep_merge_dict(existing, overlay)
    if not tracing_env_vars:
        env_block = merged.get("env")
        if isinstance(env_block, dict):
            for key in CLAUDE_TRACING_ENV_KEYS:
                env_block.pop(key, None)
    write_json_file(CLAUDE_SETTINGS_PATH, merged)

    if web_search_model:
        _register_web_search_mcp(state["workspace"], web_search_model, state.get("profile"))

    state = mark_tool_managed(state, "claude", managed_keys)
    save_state(state)
    return state


def ensure_tracing_runtime() -> bool:
    """Ensure Claude's MLflow tracing runtime is ready: an `mlflow` CLI >= 3.4 on
    PATH (the plugin shells out to it) and the MLflow Claude plugin installed.

    Best-effort — warns and returns False if a piece can't be set up, so
    `ucode configure tracing` can still finish for other agents."""
    if not _ensure_mlflow_cli():
        return False
    return _install_claude_tracing_plugin()


def _parse_mlflow_version(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\.(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _installed_mlflow_version() -> tuple[int, int] | None:
    """The (major, minor) of the `mlflow` CLI on PATH, or None if absent."""
    if not shutil.which("mlflow"):
        return None
    try:
        result = subprocess.run(
            ["mlflow", "--version"], check=False, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _parse_mlflow_version(result.stdout or result.stderr or "")


def _ensure_mlflow_cli() -> bool:
    """Ensure an `mlflow` CLI >= 3.4 is on PATH, installing or upgrading it via
    `uv tool` when needed."""
    current = _installed_mlflow_version()
    if current and current >= MINIMUM_MLFLOW_VERSION:
        return True

    if not shutil.which("uv"):
        verb = "upgrade" if current else "install"
        print_warning(
            f"Claude tracing needs the `mlflow` CLI >= 3.4 on PATH, but `uv` is not "
            f'available to {verb} it. Run `uv tool install "{MLFLOW_CLI_SPEC}"` '
            f'(or `pip install "{MLFLOW_CLI_SPEC}"`), then re-run `ucode configure tracing`.'
        )
        return False

    print_note(f"{'Upgrading' if current else 'Installing'} the mlflow CLI ({MLFLOW_CLI_SPEC})...")
    # --force replaces an existing (older) uv-managed mlflow tool in place.
    cmd = ["uv", "tool", "install", MLFLOW_CLI_SPEC]
    if current:
        cmd.append("--force")
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print_warning(f"Could not install the mlflow CLI automatically: {exc}")
        return False

    if not shutil.which("mlflow"):
        print_warning(
            "Installed mlflow, but `mlflow` is still not on PATH. Ensure your uv tool "
            "bin directory (e.g. ~/.local/bin) is on PATH, then re-run `ucode configure tracing`."
        )
        return False
    print_success("mlflow CLI ready")
    return True


def _install_claude_tracing_plugin() -> bool:
    binary = SPEC["binary"]
    if not shutil.which(binary):
        print_warning("`claude` is not installed; skipping MLflow tracing plugin install.")
        return False
    commands = [
        [
            binary,
            "plugin",
            "marketplace",
            "add",
            CLAUDE_TRACING_MARKETPLACE,
            "--sparse",
            ".claude-plugin",
        ],
        [binary, "plugin", "install", CLAUDE_TRACING_PLUGIN],
    ]
    for cmd in commands:
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print_warning(f"Could not install the Claude MLflow plugin: {exc}")
            return False
        if result.returncode != 0:
            output = (result.stderr or result.stdout or "").strip()
            last = output.splitlines()[-1] if output else f"exit {result.returncode}"
            # `marketplace add` / `install` are idempotent; treat "already
            # added/installed" as success and keep going. Best-effort match
            # against stderr — an upstream wording change would degrade this
            # to a noisy warning on re-runs, but never corrupts state.
            if "already" in last.lower():
                continue
            print_warning(f"Claude MLflow plugin step failed: {last}")
            return False
    print_success("Claude MLflow tracing plugin installed")
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
