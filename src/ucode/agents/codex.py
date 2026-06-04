"""Codex agent: writes ~/.codex/ucode.config.toml for Databricks-backed Codex."""

from __future__ import annotations

import os
import re
import shlex
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from ucode.agent_updates import available_npm_package_update
from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_json_safe,
    read_toml_safe,
    write_json_file,
    write_toml_file,
)
from ucode.databricks import (
    build_auth_shell_command,
    build_tool_base_url,
    get_databricks_token,
)
from ucode.state import mark_tool_managed, resolve_policy_model, save_state
from ucode.telemetry import agent_version, ucode_version
from ucode.ui import print_warning

CODEX_CONFIG_DIR = Path.home() / ".codex"
CODEX_PROFILE_NAME = "ucode"
CODEX_CONFIG_PATH = CODEX_CONFIG_DIR / f"{CODEX_PROFILE_NAME}.config.toml"
CODEX_BACKUP_PATH = APP_DIR / "codex-ucode-config.backup.toml"
CODEX_HOOKS_PATH = CODEX_CONFIG_DIR / "hooks.json"
CODEX_HOOKS_BACKUP_PATH = APP_DIR / "codex-hooks.backup.json"
LEGACY_CODEX_CONFIG_PATH = CODEX_CONFIG_DIR / "config.toml"
LEGACY_CODEX_BACKUP_PATH = APP_DIR / "codex-config.backup.toml"
CODEX_MODEL_PROVIDER_NAME = "ucode-databricks"
MINIMUM_CODEX_VERSION = (0, 134, 0)
MINIMUM_CODEX_VERSION_TEXT = "0.134.0"
CODEX_USAGE_NOTIFY_PREFIX = ["ucode", "usage", "hook", "codex", "notify"]
CODEX_USAGE_BUDGET_HOOK_PREFIX = "usage hook codex prompt-submit"


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
    ["notify"],
    ["model_providers", CODEX_MODEL_PROVIDER_NAME],
    ["model_providers", CODEX_MODEL_PROVIDER_NAME, "http_headers"],
]

LEGACY_MANAGED_KEYS: list[list[str]] = [
    ["profile"],
    ["profiles", CODEX_PROFILE_NAME],
    ["notify"],
    ["model_providers", CODEX_MODEL_PROVIDER_NAME],
    ["model_providers", CODEX_MODEL_PROVIDER_NAME, "http_headers"],
]

_GPT_RE = re.compile(r"(?:databricks-)?gpt-(\d+)(?:[.-](\d+))?(?:[.-](\d+))?(-.+|[a-z].*)?")

# These models should use the Databricks ID, not the OpenAI ID, as the OpenAI
# ID is incompatible with Codex.
CODEX_OPENAI_ID_INCOMPATIBLE_MODELS = {
    "databricks-gpt-5-2-codex",
    "databricks-gpt-5-4-nano",
}


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


def _use_legacy_layout() -> bool:
    """Return True when the installed Codex CLI predates per-profile config files.

    Codex 0.134.0 introduced support for `--profile <name>` resolving to
    `~/.codex/<name>.config.toml`. Older releases only honor a single
    `~/.codex/config.toml` with `[profiles.<name>]` sections. When the version
    is unknown we keep the new layout (matches the prior "unknown does not
    block" semantic).
    """
    parsed = _parse_version(agent_version(SPEC["binary"]))
    if parsed is None:
        return False
    return parsed < MINIMUM_CODEX_VERSION


def _provider_block(workspace: str, databricks_profile: str | None) -> dict:
    auth_command = build_auth_shell_command(workspace, databricks_profile)
    base_url = build_tool_base_url("codex", workspace)
    return {
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


def render_overlay(
    workspace: str, model: str | None = None, databricks_profile: str | None = None
) -> dict:
    overlay: dict = {"model_provider": CODEX_MODEL_PROVIDER_NAME}
    if model:
        overlay["model"] = model
    overlay["model_providers"] = {
        CODEX_MODEL_PROVIDER_NAME: _provider_block(workspace, databricks_profile),
    }
    return overlay


def render_legacy_overlay(
    workspace: str, model: str | None = None, databricks_profile: str | None = None
) -> dict:
    """Overlay for Codex CLI < 0.134.0, which only reads `~/.codex/config.toml`.

    The shared file uses `profile = "ucode"` to select `[profiles.ucode]`, which
    points at the shared `[model_providers.ucode-databricks]` block.
    """
    profile_block: dict = {"model_provider": CODEX_MODEL_PROVIDER_NAME}
    if model:
        profile_block["model"] = model
    return {
        "profile": CODEX_PROFILE_NAME,
        "profiles": {CODEX_PROFILE_NAME: profile_block},
        "model_providers": {
            CODEX_MODEL_PROVIDER_NAME: _provider_block(workspace, databricks_profile),
        },
    }


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


def _openai_model_id(model: str | None) -> str | None:
    """Map Databricks GPT endpoint ids to OpenAI model ids for Codex metadata."""
    parsed = _parse_gpt(model)
    if parsed is None:
        return model
    major, minor, patch, suffix = parsed
    version = str(major)
    if minor is not None:
        version += f".{minor}"
    if patch is not None:
        version += f".{patch}"
    return f"gpt-{version}{suffix}"


def _codex_model_id(model: str | None) -> str | None:
    if model in CODEX_OPENAI_ID_INCOMPATIBLE_MODELS:
        return model
    return _openai_model_id(model)


def _parse_gpt(model: str | None) -> tuple[int, int | None, int | None, str] | None:
    if not model:
        return None
    match = _GPT_RE.fullmatch(model.split("/")[-1])
    if not match:
        return None
    major, minor, patch, suffix = match.groups()
    return (
        int(major),
        int(minor) if minor is not None else None,
        int(patch) if patch is not None else None,
        suffix or "",
    )


def write_tool_config(state: dict, model: str | None = None) -> dict:
    workspace = state["workspace"]
    base_model = model or default_model(state)
    if base_model:
        base_model = resolve_policy_model(state, "codex", base_model)
    chosen_model = _codex_model_id(base_model)
    databricks_profile = state.get("profile")

    if _use_legacy_layout():
        # Codex < 0.134.0 only reads ~/.codex/config.toml. Write the shared
        # config with [profiles.ucode] + shared [model_providers.ucode-databricks]
        # and skip the per-profile-file cleanup that would normally strip
        # ucode's entry from the shared file.
        backup_existing_file(LEGACY_CODEX_CONFIG_PATH, LEGACY_CODEX_BACKUP_PATH)
        overlay = render_legacy_overlay(workspace, chosen_model, databricks_profile)
        doc = read_toml_safe(LEGACY_CODEX_CONFIG_PATH)
        deep_merge_dict(doc, overlay)
        _apply_usage_notify(doc, state, chosen_model)
        _apply_usage_budget_hook(state["workspace"], chosen_model)
        write_toml_file(LEGACY_CODEX_CONFIG_PATH, doc)
        state = mark_tool_managed(state, "codex", LEGACY_MANAGED_KEYS)
        save_state(state)
        return state

    _remove_legacy_ucode_profile()
    backup_existing_file(CODEX_CONFIG_PATH, CODEX_BACKUP_PATH)
    overlay = render_overlay(workspace, chosen_model, databricks_profile)
    doc = read_toml_safe(CODEX_CONFIG_PATH)
    deep_merge_dict(doc, overlay)
    _apply_usage_notify(doc, state, chosen_model)
    _apply_usage_budget_hook(state["workspace"], chosen_model)
    write_toml_file(CODEX_CONFIG_PATH, doc)
    state = mark_tool_managed(state, "codex", MANAGED_KEYS)
    save_state(state)
    return state


def _usage_notify(workspace: str, model: str | None) -> list[str]:
    notify = [*CODEX_USAGE_NOTIFY_PREFIX]
    if model:
        notify.extend(["--model", model])
    notify.extend(["--workspace", workspace])
    return notify


def _is_ucode_usage_notify(value: object) -> bool:
    return (
        isinstance(value, list)
        and value[: len(CODEX_USAGE_NOTIFY_PREFIX)] == CODEX_USAGE_NOTIFY_PREFIX
    )


def _apply_usage_notify(doc: dict, state: dict, model: str | None) -> None:
    """Set Codex's notify hook for local usage unless another integration owns it."""
    if not model:
        return
    existing = doc.get("notify")
    notify = _usage_notify(state["workspace"], model)
    if existing is not None and list(existing) != notify and not _is_ucode_usage_notify(existing):
        print_warning(
            f"Codex `notify` is already set to {existing!r}; local usage tracking for "
            "Codex will be inactive until that hook is removed."
        )
        return
    doc["notify"] = notify


def _ucode_command_prefix() -> list[str]:
    """Use the editable checkout when running from source; fall back to PATH."""
    repo_root = Path(__file__).resolve().parents[3]
    if (repo_root / "pyproject.toml").exists() and (repo_root / "src" / "ucode").exists():
        return ["uv", "run", "--project", str(repo_root), "ucode"]
    return [shutil.which("ucode") or "ucode"]


def _usage_budget_hook_command(workspace: str, model: str | None) -> str:
    chosen_model = model or "codex"
    return " ".join(
        [
            *[shlex.quote(part) for part in _ucode_command_prefix()],
            "usage",
            "hook",
            "codex",
            "prompt-submit",
            "--model",
            shlex.quote(chosen_model),
            "--workspace",
            shlex.quote(workspace),
        ]
    )


def _is_ucode_usage_budget_hook(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    hook = cast(Mapping[str, object], value)
    command = hook.get("command")
    return isinstance(command, str) and (
        "usage hook codex prompt-submit" in command or "usage budget-check --agent codex" in command
    )


def _apply_usage_budget_hook(workspace: str, model: str | None) -> None:
    """Block new Codex prompts once the local Codex budget is exceeded."""
    backup_existing_file(CODEX_HOOKS_PATH, CODEX_HOOKS_BACKUP_PATH)
    doc = read_json_safe(CODEX_HOOKS_PATH)
    hooks = doc.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = {}
        doc["hooks"] = hooks
    prompt_blocks = hooks.setdefault("UserPromptSubmit", [])
    if not isinstance(prompt_blocks, list):
        prompt_blocks = []
        hooks["UserPromptSubmit"] = prompt_blocks

    filtered_blocks = []
    for block in prompt_blocks:
        if not isinstance(block, dict):
            filtered_blocks.append(block)
            continue
        block_hooks = block.get("hooks")
        if not isinstance(block_hooks, list):
            filtered_blocks.append(block)
            continue
        kept_hooks = [hook for hook in block_hooks if not _is_ucode_usage_budget_hook(hook)]
        if kept_hooks:
            updated = dict(block)
            updated["hooks"] = kept_hooks
            filtered_blocks.append(updated)
    filtered_blocks.append(
        {
            "hooks": [
                {
                    "type": "command",
                    "command": _usage_budget_hook_command(workspace, model),
                    "timeout": 10,
                }
            ]
        }
    )
    hooks["UserPromptSubmit"] = filtered_blocks
    write_json_file(CODEX_HOOKS_PATH, doc)


def default_model(state: dict) -> str | None:
    """Pick the newest GPT model when multiple are available.

    The discovery list is alphabetically sorted, which can put
    "databricks-gpt-5" ahead of "databricks-gpt-5-5". Prefer the
    highest semantic version instead. Falls back to the first
    discovered entry when parsing fails.
    """
    codex_models = state.get("codex_models") or []
    if not codex_models:
        return None

    def _gpt_version_key(mid: str) -> tuple[int, int, int, int]:
        parsed = _parse_gpt(mid)
        if parsed is None:
            return (0, 0, 0, 0)
        major, minor, patch, suffix = parsed
        base_bonus = 1 if not suffix else 0
        return (major, minor or 0, patch or 0, base_bonus)

    return max(codex_models, key=_gpt_version_key)


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
