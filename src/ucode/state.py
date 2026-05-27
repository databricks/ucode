"""Persistent state for ucode (per-workspace, versioned)."""

from __future__ import annotations

import json

from ucode.config_io import APP_DIR, is_dry_run
from ucode.databricks import build_shared_base_urls

STATE_PATH = APP_DIR / "state.json"
STATE_VERSION = 3


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


def hydrate_state(state: dict) -> dict:
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

    workspace = hydrated.get("workspace")
    if workspace:
        hydrated["base_urls"] = build_shared_base_urls(workspace)
    else:
        hydrated["base_urls"] = {}

    return hydrated


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
