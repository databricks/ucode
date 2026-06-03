"""Persistent state for ucode (per-workspace, versioned)."""

from __future__ import annotations

import datetime as _dt
import json
from typing import cast

from ucode.config_io import APP_DIR, is_dry_run
from ucode.databricks import build_auth_shell_command, build_shared_base_urls

STATE_PATH = APP_DIR / "state.json"
STATE_VERSION = 3
AUTH_COMMAND_TIMEOUT_MS = 5000
AUTH_REFRESH_INTERVAL_MS = 900_000

_KNOWN_POLICY_AGENTS: frozenset[str] = frozenset(
    {"codex", "claude", "gemini", "opencode", "copilot", "pi"}
)

def _today_iso() -> str:
    """Return today's date as ``YYYY-MM-DD``. Factored for monkeypatching."""
    return _dt.date.today().isoformat()

def _is_iso_date(value: str) -> bool:
    try:
        _dt.date.fromisoformat(value)
    except ValueError:
        return False
    return True

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


def _normalize_spending_limit(raw: object) -> dict | None:
    """Validate one ``spending_limit`` block: ``{monthly_limit_usd, start_date}``."""
    if not isinstance(raw, dict):
        return None
    raw_dict = cast("dict[str, object]", raw)
    monthly = raw_dict.get("monthly_limit_usd")
    if not isinstance(monthly, (int, float)) or isinstance(monthly, bool) or monthly <= 0:
        return None
    start_date = raw_dict.get("start_date")
    if not (isinstance(start_date, str) and _is_iso_date(start_date)):
        start_date = _today_iso()
    return {
        "monthly_limit_usd": float(monthly),
        "start_date": start_date,
    }


def _normalize_tier(raw: object) -> dict | None:
    """Validate one tier block: ``{usage_percent, model}``."""
    if not isinstance(raw, dict):
        return None
    raw_dict = cast("dict[str, object]", raw)
    pct = raw_dict.get("usage_percent")
    model = raw_dict.get("model")
    if (
        not isinstance(pct, (int, float))
        or isinstance(pct, bool)
        or not (0 < pct <= 100)
        or not isinstance(model, str)
        or not model
    ):
        return None
    return {"usage_percent": float(pct), "model": model}


def _normalize_agent_policies(raw: object) -> dict | None:
    """Validate one agent's policy block and drop empty results."""
    if not isinstance(raw, dict):
        return None
    raw_dict = cast("dict[str, object]", raw)
    normalized: dict = {}
    spending = _normalize_spending_limit(raw_dict.get("spending_limit"))
    if spending is not None:
        normalized["spending_limit"] = spending
    default_model = raw_dict.get("default_model")
    if isinstance(default_model, str) and default_model:
        normalized["default_model"] = default_model
    tier_2 = _normalize_tier(raw_dict.get("tier_2"))
    if tier_2 is not None:
        normalized["tier_2"] = tier_2
    tier_3 = _normalize_tier(raw_dict.get("tier_3"))
    if tier_3 is not None:
        normalized["tier_3"] = tier_3
    return normalized or None


def _normalize_policies(raw: object) -> dict:
    """Return a well-formed per-agent ``policies`` block, dropping anything malformed.
    Schema (per agent — agent keys are codex/claude/gemini/opencode/copilot/pi)::

        {
          "<agent>": {
            "spending_limit": {                                          # optional
              "monthly_limit_usd": <number>,
              "start_date": "YYYY-MM-DD",        # auto-filled with today
            },
            "default_model": <model_id>,                                 # optional
            "tier_2": {"usage_percent": 50, "model": <model_id>},        # optional
            "tier_3": {"usage_percent": 90, "model": <model_id>},        # optional
          }
        }
    """
    if not isinstance(raw, dict):
        return {}
    raw_dict = cast("dict[str, object]", raw)
    out: dict = {}
    for agent, agent_raw in raw_dict.items():
        if agent not in _KNOWN_POLICY_AGENTS:
            continue
        normalized = _normalize_agent_policies(agent_raw)
        if normalized:
            out[agent] = normalized
    return out


def hydrate_state(state: dict) -> dict:
    """Normalize a workspace state entry and add derived harness config.

    :param state: Raw workspace state entry from ``state.json``.
    :returns: Hydrated workspace state with stable ``managed_configs``,
        ``policies``, ``base_urls``, and per-agent ``agents`` entries.
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
    hydrated["policies"] = _normalize_policies(hydrated.get("policies"))

    workspace = hydrated.get("workspace")
    if workspace:
        hydrated["base_urls"] = build_shared_base_urls(workspace)
        hydrated["agents"] = build_agent_state(hydrated)
    else:
        hydrated["base_urls"] = {}
        hydrated["agents"] = {}

    return hydrated


def select_model_for_policies(
    state: dict,
    agent: str,
    requested_model: str,
    current_spend_usd: float,
) -> str:
    """Apply any configured policies to a model selection decision.
    Today only the ``spending_limit`` policy is consulted.

    TODO: ``current_spend_usd`` is currently expected from the caller.
    Wire this to the Databricks Budgets API"""
    policies = state.get("policies") or {}
    if not isinstance(policies, dict):
        return requested_model
    agent_policy = policies.get(agent)
    if not isinstance(agent_policy, dict):
        return requested_model

    spending = agent_policy.get("spending_limit")
    if isinstance(spending, dict):
        monthly = spending.get("monthly_limit_usd")
        if isinstance(monthly, (int, float)) and not isinstance(monthly, bool) and monthly > 0:
            usage_pct = (current_spend_usd / monthly) * 100
            for tier_key in ("tier_3", "tier_2"):
                tier = agent_policy.get(tier_key)
                if not isinstance(tier, dict):
                    continue
                threshold = tier.get("usage_percent")
                model = tier.get("model")
                if (
                    isinstance(threshold, (int, float))
                    and not isinstance(threshold, bool)
                    and isinstance(model, str)
                    and model
                    and usage_pct >= threshold
                ):
                    return model

    default = agent_policy.get("default_model")
    if isinstance(default, str) and default:
        return default
    return requested_model


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
