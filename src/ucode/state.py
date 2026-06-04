"""Persistent state for ucode (per-workspace, versioned)."""

from __future__ import annotations

import json

from ucode.config_io import APP_DIR, is_dry_run
from ucode.databricks import build_auth_shell_command, build_shared_base_urls

STATE_PATH = APP_DIR / "state.json"
STATE_VERSION = 3
AUTH_COMMAND_TIMEOUT_MS = 5000
AUTH_REFRESH_INTERVAL_MS = 900_000


def load_full_state() -> dict:
    """Load the entire state file. Returns empty structure if missing or wrong version."""
    if not STATE_PATH.exists():
        return {"state_version": STATE_VERSION, "current_workspace": None, "workspaces": {}}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state_version": STATE_VERSION, "current_workspace": None, "workspaces": {}}
    if not isinstance(data, dict) or data.get("state_version") != STATE_VERSION:
        return {"state_version": STATE_VERSION, "current_workspace": None, "workspaces": {}}
    return data


def load_state() -> dict:
    """Load the current workspace's state as a flat dict."""
    full = load_full_state()
    workspace = full.get("current_workspace")
    if not workspace:
        return {}
    ws_state = full.get("workspaces", {}).get(workspace, {})
    ws_state["workspace"] = workspace
    return hydrate_state(ws_state)


def slice_state_for_export(full_state: dict, workspace: str) -> dict:
    """Return a single-workspace splice for publishing to UC.
        {
          "state_version": 3,
          "workspace": "<ws url>",
          "claude_models": {...},
          ...all other per-workspace fields...
        }

    The ``workspace`` URL is the only stable, machine-independent identity in the
    blob: every consumer resolves their own local Databricks profile from it via
    ``find_profile_name_for_host``. Machine-local fields derived from the
    exporter's profile (the top-level ``profile`` and the tracing
    ``tracking_uri``, which is ``databricks://<profile>``) are stripped so the
    admin's profile name never leaks into a user's config — the consumer
    recomputes both from the workspace on pull.
    """
    workspaces = full_state.get("workspaces")
    if not isinstance(workspaces, dict) or workspace not in workspaces:
        raise RuntimeError(
            f"No local state for workspace {workspace}. Run `ucode configure` for it first."
        )
    block = workspaces[workspace]
    if not isinstance(block, dict):
        raise RuntimeError(
            f"Local state for workspace {workspace} is malformed (expected an object)."
        )
    sliced = {
        **block,
        "state_version": full_state.get("state_version") or STATE_VERSION,
        "workspace": workspace,
    }
    sliced.pop("profile", None)
    sliced.pop("policies", None)
    tracing = sliced.get("tracing")
    if isinstance(tracing, dict):
        tracing = dict(tracing)
        tracing.pop("tracking_uri", None)
        sliced["tracing"] = tracing
    return sliced


def save_state(state: dict) -> None:
    """Save workspace state back into the per-workspace structure."""
    if is_dry_run():
        return
    full = load_full_state()
    workspace = state.get("workspace") or full.get("current_workspace")
    if workspace:
        full["current_workspace"] = workspace
        full["workspaces"][workspace] = hydrate_state(state)
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(full, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write state file: {STATE_PATH}") from exc


def set_current_workspace(workspace: str | None) -> None:
    """Set ``current_workspace`` without touching the per-workspace blocks.

    Used by flows like ``configure tracing`` that operate on a non-current
    workspace and must not silently change which workspace ``ucode launch``
    targets afterwards."""
    if is_dry_run():
        return
    full = load_full_state()
    if full.get("current_workspace") == workspace:
        return
    full["current_workspace"] = workspace
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(full, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write state file: {STATE_PATH}") from exc


def hydrate_state(state: dict) -> dict:
    """Normalize a workspace state entry and add derived harness config.

    :param state: Raw workspace state entry from ``state.json``.
    :returns: Hydrated workspace state with stable ``managed_configs``,
        ``base_urls`` and per-agent ``agents`` entries.
    """
    if not isinstance(state, dict):
        return {}

    hydrated = dict(state)
    managed_configs = hydrated.get("managed_configs")
    if not isinstance(managed_configs, dict):
        managed_configs = {}
    normalized: dict[str, dict] = {}
    for tool, entry in managed_configs.items():
        if isinstance(entry, dict):
            keys = entry.get("keys") if isinstance(entry.get("keys"), list) else []
            normalized[tool] = {"keys": keys}
        elif entry:
            normalized[tool] = {"keys": []}
    hydrated["managed_configs"] = normalized
    hydrated.pop("policies", None)

    workspace = hydrated.get("workspace")
    if workspace:
        hydrated["base_urls"] = build_shared_base_urls(workspace)
        hydrated["agents"] = build_agent_state(hydrated)
    else:
        hydrated["base_urls"] = {}
        hydrated["agents"] = {}

    return hydrated


def merge_managed_workspace(
    local_state: dict,
    remote_splice: dict,
    *,
    profile: str | None = None,
) -> dict:
    """Pull-and-replace: the admin's UC blob fully replaces the workspace block.

    The requested local workspace is authoritative. Older managed exports include
    a ``workspace`` field; newer callers may pass a workspace-less blob read from
    the current workspace's UC volume. If a remote workspace is present and
    disagrees with the requested workspace, ignore it.
    """
    workspace = local_state.get("workspace")
    if not isinstance(workspace, str) or not workspace:
        return local_state
    if not isinstance(remote_splice, dict):
        return local_state
    remote_workspace = remote_splice.get("workspace")
    if isinstance(remote_workspace, str) and remote_workspace != workspace:
        return local_state
    merged = {**remote_splice, "workspace": workspace}
    merged.pop("policies", None)
    if profile:
        merged["profile"] = profile
    _localize_tracking_uri(merged)
    return merged


def _localize_tracking_uri(state: dict) -> None:
    """Re-point a pulled tracing block's MLflow tracking URI at the local
    profile. The admin's export carries ``databricks://<admin-profile>``; that
    profile name has no matching ``~/.databrickscfg`` entry on the user's
    machine, so auth would fail. Derived in-place from the merged profile."""
    tracing = state.get("tracing")
    if not isinstance(tracing, dict):
        return
    profile = state.get("profile")
    tracing["tracking_uri"] = f"databricks://{profile}" if profile else "databricks"


def build_agent_state(state: dict) -> dict[str, dict]:
    """Build per-agent harness configuration for a workspace.

    The returned shape is intended for downstream tools that want to reuse
    ucode's configured gateway URLs and auth command without duplicating
    endpoint construction logic.

    :param state: Hydrated workspace state containing ``workspace``,
        ``base_urls``, and discovered model lists.
    :returns: Mapping from agent name to its reusable configuration.
    """
    workspace = state.get("workspace")
    if not isinstance(workspace, str) or not workspace:
        return {}

    profile = state.get("profile") if isinstance(state.get("profile"), str) else None
    base_urls_value = state.get("base_urls")
    base_urls = base_urls_value if isinstance(base_urls_value, dict) else {}
    auth_command = build_auth_shell_command(workspace, profile)
    claude_models_value = state.get("claude_models")
    claude_models: dict = claude_models_value if isinstance(claude_models_value, dict) else {}
    codex_models_value = state.get("codex_models")
    codex_models = codex_models_value if isinstance(codex_models_value, list) else []
    gemini_models_value = state.get("gemini_models")
    gemini_models = gemini_models_value if isinstance(gemini_models_value, list) else []

    claude_model = (
        claude_models.get("opus") or claude_models.get("sonnet") or claude_models.get("haiku")
    )
    codex_model = codex_models[0] if codex_models else None
    pi_model = claude_model or codex_model or (gemini_models[0] if gemini_models else None)

    agents: dict[str, dict] = {
        "claude": {
            "model": claude_model,
            "base_url": base_urls.get("claude"),
            "auth_command": auth_command,
            "auth_refresh_interval_ms": AUTH_REFRESH_INTERVAL_MS,
            "env": {
                "ANTHROPIC_BASE_URL": base_urls.get("claude"),
                "CLAUDE_CODE_API_KEY_HELPER_TTL_MS": str(AUTH_REFRESH_INTERVAL_MS),
                "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            },
        },
        "codex": {
            "model": codex_model,
            "base_url": base_urls.get("codex"),
            "auth_command": auth_command,
            "auth": {
                "command": "sh",
                "args": ["-c", auth_command],
                "timeout_ms": AUTH_COMMAND_TIMEOUT_MS,
                "refresh_interval_ms": AUTH_REFRESH_INTERVAL_MS,
            },
        },
        "pi": {
            "model": pi_model,
            "base_urls": base_urls.get("pi") if isinstance(base_urls.get("pi"), dict) else {},
            "auth_command": auth_command,
            "auth_refresh_interval_ms": AUTH_REFRESH_INTERVAL_MS,
        },
    }
    return {
        name: {key: value for key, value in config.items() if value is not None}
        for name, config in agents.items()
    }


def clear_state() -> None:
    """Remove the current workspace entry from state."""
    full = load_full_state()
    workspace = full.get("current_workspace")
    if workspace:
        full.get("workspaces", {}).pop(workspace, None)
        full["current_workspace"] = None
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(full, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to clear state file: {STATE_PATH}") from exc


def mark_tool_managed(state: dict, tool: str, managed_keys: list) -> dict:
    managed_configs = dict(state.get("managed_configs") or {})
    managed_configs[tool] = {"keys": list(managed_keys)}
    state["managed_configs"] = managed_configs
    state["last_tool"] = tool
    return state
