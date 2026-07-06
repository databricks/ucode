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
import os
import shutil
import subprocess
from pathlib import Path
from typing import cast

from rich.panel import Panel
from rich.table import Table

from ucode.config_io import APP_DIR, ToolSpec
from ucode.databricks import (
    fetch_current_user_budget_spend_status,
    install_databricks_cli,
)
from ucode.state import load_state, save_state
from ucode.telemetry import agent_version
from ucode.ui import (
    console,
    is_low_verbosity,
    print_err,
    print_kv,
    print_note,
    print_section,
    print_success,
    print_warning,
    prompt_yes_no,
    spinner,
)

from . import claude, codex, copilot, gemini, opencode, pi

_MODULES = {
    "codex": codex,
    "claude": claude,
    "gemini": gemini,
    "opencode": opencode,
    "copilot": copilot,
    "pi": pi,
}

TOOL_SPECS: dict[str, ToolSpec] = {name: module.SPEC for name, module in _MODULES.items()}

TOOL_ALIASES = {
    "codex": "codex",
    "claude": "claude",
    "claude-code": "claude",
    "gemini": "gemini",
    "gemini-cli": "gemini",
    "opencode": "opencode",
    "copilot": "copilot",
    "pi": "pi",
}

DEFAULT_TOOL = "codex"
BUNDLE_VERSION = 1
BUDGET_STATUS_TOOLS = {"claude", "codex"}
BUDGET_POLICY_ENV_VAR = "UCODE_BUDGET_POLICY"
BUDGET_POLICY_PATH = APP_DIR / "budget-policy.json"
SMART_BUDGET_SWITCH_PERCENT = 80.0
SMART_BUDGET_LOW_TOOL = "claude"
SMART_BUDGET_HIGH_TOOL = "codex"
DEFAULT_BUDGET_POLICY_NAME = "Default smart budget policy"


def normalize_tool(tool: str) -> str:
    normalized = TOOL_ALIASES.get(tool.strip().lower())
    if not normalized:
        raise RuntimeError(
            f"Unsupported tool '{tool}'. Use one of: codex, claude, gemini, opencode, copilot, pi."
        )
    return normalized


def budget_policy_env_configured() -> bool:
    return bool(os.environ.get(BUDGET_POLICY_ENV_VAR, "").strip())


def budget_policy_configured() -> bool:
    """True when a budget policy is active via env override or the default file.

    The env var only selects an alternate policy path; an admin who drops a
    policy at the default location (~/.ucode/budget-policy.json) has configured
    one just as much, so enforcement must key off either source.
    """
    return budget_policy_env_configured() or _budget_policy_path().exists()


def _update_installed_tool_binary(tool: str, version: str | None = None) -> bool:
    spec = TOOL_SPECS[tool]
    binary = spec["binary"]
    package = spec["package"]
    target = f"{package}@{version}" if version else package

    if not shutil.which("npm"):
        print_warning(f"`npm` is not available to update {spec['display']}; continuing.")
        return False

    print_note(f"Updating {spec['display']}...")
    try:
        subprocess.run(["npm", "install", "-g", target], check=True, timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print_warning(f"Could not update {spec['display']}; continuing.")
        return False

    print_success(f"{spec['display']} is up to date")
    agent_version.cache_clear()
    return bool(shutil.which(binary))


def _minimum_version_error(tool: str) -> str | None:
    checker = getattr(_MODULES[tool], "minimum_version_error", None)
    if not callable(checker):
        return None
    return checker()


def _required_update_message(tool: str) -> str | None:
    checker = getattr(_MODULES[tool], "required_update_message", None)
    if not callable(checker):
        return None
    return checker()


def _confirm_update_installed_tool_binary(tool: str) -> bool:
    spec = TOOL_SPECS[tool]
    update = _MODULES[tool].is_update_available()

    if not update:
        return False
    current, latest = update
    return prompt_yes_no(f"(Optional) Update {spec['display']} from {current} to {latest}?")


def _too_new_downgrade(tool: str) -> tuple[str, str] | None:
    """Return (installed_version, downgrade_target) when the installed tool is
    too new to work, or None. Agents opt in by defining `too_new_downgrade`."""
    checker = getattr(_MODULES[tool], "too_new_downgrade", None)
    if not callable(checker):
        return None
    return checker()


def _maybe_downgrade_too_new_tool(tool: str, *, prompt: bool) -> bool:
    """Warn when the installed tool exceeds its supported version and offer to
    downgrade to the latest working release. Returns True when the tool was too
    new (regardless of whether the client accepted the downgrade).

    Unlike a required *upgrade*, a too-new build may still launch (it just
    misbehaves), so we never force the change — we warn and, when prompting is
    enabled, let the client press `y` to downgrade.
    """
    downgrade = _too_new_downgrade(tool)
    if not downgrade:
        return False
    spec = TOOL_SPECS[tool]
    installed, target = downgrade
    print_warning(
        f"{spec['display']} {installed} is newer than the latest version known to work "
        f"with the Databricks AI Gateway ({target})."
    )
    if prompt and prompt_yes_no(f"Downgrade {spec['display']} from {installed} to {target}?"):
        _update_installed_tool_binary(tool, version=target)
    return True


def install_tool_binary(
    tool: str,
    *,
    strict: bool = True,
    update_existing: bool = False,
    prompt_optional_updates: bool = True,
) -> bool:
    spec = TOOL_SPECS[tool]
    binary = spec["binary"]
    package = spec["package"]

    if shutil.which(binary):
        # A too-new build is a correctness blocker (the tool runs but misbehaves
        # against the gateway), so check it on every launch — not just when
        # auto-configuring — mirroring the minimum-version gate below.
        too_new = _maybe_downgrade_too_new_tool(tool, prompt=prompt_optional_updates)

        if update_existing and not too_new:
            required_update = _required_update_message(tool)
            if required_update:
                # Required updates are forced regardless of prompt preference;
                # the tool won't function on an unsupported version.
                print_warning(required_update)
                if not _update_installed_tool_binary(tool):
                    raise RuntimeError(_minimum_version_error(tool) or required_update)
            elif prompt_optional_updates and _confirm_update_installed_tool_binary(tool):
                _update_installed_tool_binary(tool)

        version_error = _minimum_version_error(tool)
        if version_error:
            raise RuntimeError(version_error)
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
        f"`ucode configure` to try automatic installation."
    )


def ensure_bootstrap_dependencies(
    tool: str,
    *,
    update_existing: bool = False,
    prompt_optional_updates: bool = True,
) -> None:
    install_databricks_cli()
    install_tool_binary(
        tool,
        strict=True,
        update_existing=update_existing,
        prompt_optional_updates=prompt_optional_updates,
    )


def default_model_for_tool(tool: str, state: dict) -> str | None:
    return _MODULES[tool].default_model(state)


def _resolve_policy_model_alias(tool: str, state: dict, model: str | None) -> str | None:
    if not model:
        return None
    raw = model.strip()
    if not raw:
        return None
    if tool == "claude":
        claude_models = state.get("claude_models") or {}
        if isinstance(claude_models, dict):
            value = claude_models.get(raw.lower())
            if isinstance(value, str) and value.strip():
                return value.strip()
            for candidate in claude_models.values():
                if isinstance(candidate, str) and (
                    candidate == raw or candidate.endswith(raw) or raw in candidate
                ):
                    return candidate
    if tool == "codex":
        codex_models = state.get("codex_models") or []
        if isinstance(codex_models, list):
            for candidate in codex_models:
                if not isinstance(candidate, str):
                    continue
                tail = candidate.rsplit("/", 1)[-1]
                tail = tail[len("system.ai.") :] if tail.startswith("system.ai.") else tail
                tail = tail[len("databricks-") :] if tail.startswith("databricks-") else tail
                if candidate == raw or tail == raw or candidate.endswith(raw):
                    return candidate
            # No exact match. On a model-services workspace every model routes by
            # its `system.ai.<name>` id, so a bare policy model (e.g.
            # `gpt-5-4-mini`) must keep that prefix rather than fall through to
            # the legacy OpenAI-id rewrite, which would mangle it into an
            # unroutable dotted id like `gpt-5.4-mini`.
            if not raw.startswith("system.ai.") and any(
                isinstance(c, str) and c.startswith("system.ai.") for c in codex_models
            ):
                return f"system.ai.{raw}"
    return raw


def resolve_launch_model(
    tool: str,
    state: dict,
    explicit_model: str | None,
) -> tuple[dict, str | None]:
    model = _resolve_policy_model_alias(tool, state, explicit_model) or default_model_for_tool(
        tool, state
    )
    if not model:
        raise RuntimeError(
            f"No models available for {tool}. Run `ucode configure` to set up your workspace."
        )
    return state, model


def configure_tool(tool: str, state: dict, model: str | None = None) -> dict:
    result: dict | tuple[dict, str]
    if tool == "codex":
        result = codex.write_tool_config(state, model)
    else:
        if not model:
            raise RuntimeError(f"A {tool} model must be selected before configuration.")
        if tool == "claude":
            result = claude.write_tool_config(state, model)
        elif tool == "gemini":
            result = gemini.write_tool_config(state, model)
        elif tool == "copilot":
            result = copilot.write_tool_config(state, model)
        elif tool == "pi":
            result = pi.write_tool_config(state, model)
        else:
            result = opencode.write_tool_config(state, model)
    # gemini/opencode/copilot/pi return (state, token); codex/claude return state
    if isinstance(result, tuple):
        return result[0]
    return result


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _format_usd(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${value:,.2f}"


def _short_budget_id(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "budget"
    return raw[:8]


def _format_budget_status(entry: dict) -> tuple[str, str]:
    budget_id = _short_budget_id(entry.get("budget_id"))
    spend_status = entry.get("spend_status")
    spend_status = spend_status if isinstance(spend_status, dict) else {}
    spend = _as_float(spend_status.get("spend"))
    threshold = _as_float(spend_status.get("effective_alert_threshold"))
    if threshold and threshold > 0 and spend is not None:
        percent = (spend / threshold) * 100
        return (
            f"Budget {budget_id}",
            f"{_format_usd(spend)} / {_format_usd(threshold)} ({percent:.0f}%)",
        )
    return f"Budget {budget_id}", _format_usd(spend)


def _budget_status_sort_key(entry: dict) -> float:
    percent = _budget_usage_percent(entry)
    return percent if percent is not None else -1.0


def _budget_usage_percent(entry: dict) -> float | None:
    spend_status = entry.get("spend_status")
    spend_status = spend_status if isinstance(spend_status, dict) else {}
    spend = _as_float(spend_status.get("spend"))
    threshold = _as_float(spend_status.get("effective_alert_threshold"))
    if spend is None or threshold is None or threshold <= 0:
        return None
    return (spend / threshold) * 100


def _max_budget_usage_percent(statuses: list[dict]) -> float | None:
    percents = [
        percent for entry in statuses if (percent := _budget_usage_percent(entry)) is not None
    ]
    return max(percents) if percents else None


def _primary_budget_status(statuses: list[dict]) -> dict:
    return max(statuses, key=_budget_status_sort_key)


def _budget_policy_path() -> Path:
    override = os.environ.get(BUDGET_POLICY_ENV_VAR, "").strip()
    return Path(override).expanduser() if override else BUDGET_POLICY_PATH


def _read_budget_policy_file() -> tuple[dict | None, Path, str | None]:
    path = _budget_policy_path()
    if not path.exists():
        return None, path, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, path, f"{path}: {exc}"
    if not isinstance(payload, dict):
        return None, path, f"{path}: top-level JSON value must be an object"
    return payload, path, None


def _policy_percent(value: object, default: float) -> float:
    percent = _as_float(value)
    if percent is None or percent <= 0 or percent > 100:
        return default
    return percent


def _policy_tool(value: object, default: str) -> str:
    raw: object
    if isinstance(value, dict):
        value_dict = cast(dict[str, object], value)
        raw = value_dict.get("agent") or value_dict.get("tool")
    else:
        raw = value
    if not isinstance(raw, str) or not raw.strip():
        return default
    tool = TOOL_ALIASES.get(raw.strip().lower().replace(" ", "-"))
    if tool in BUDGET_STATUS_TOOLS:
        return tool
    return default


def _policy_model(value: object) -> str | None:
    if isinstance(value, dict):
        value_dict = cast(dict[str, object], value)
        value = value_dict.get("model")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _policy_threshold(value: object) -> float | None:
    percent = _as_float(value)
    if percent is None or percent <= 0 or percent > 100:
        return None
    return percent


def _policy_tier_from_value(threshold: float, value: object) -> dict | None:
    default_tool = SMART_BUDGET_LOW_TOOL
    model: str | None = None
    if isinstance(value, list) and value:
        default_tool = _policy_tool(value[0], SMART_BUDGET_LOW_TOOL)
        if len(value) > 1:
            model = _policy_model(value[1])
    else:
        default_tool = _policy_tool(value, SMART_BUDGET_LOW_TOOL)
        model = _policy_model(value)
    return {"threshold": threshold, "tool": default_tool, "model": model}


def _policy_tier_from_dict(value: dict) -> dict | None:
    value_dict = cast(dict[str, object], value)
    threshold = (
        _policy_threshold(value_dict.get("until_percent"))
        or _policy_threshold(value_dict.get("threshold"))
        or _policy_threshold(value_dict.get("percent"))
    )
    if threshold is None:
        return None
    tool = _policy_tool(value_dict, SMART_BUDGET_LOW_TOOL)
    model = _policy_model(value_dict)
    return {"threshold": threshold, "tool": tool, "model": model}


def _policy_tiers(payload: dict) -> list[dict]:
    tiers_raw = payload.get("tiers")
    tiers: list[dict] = []
    if isinstance(tiers_raw, dict):
        tiers_dict = cast(dict[object, object], tiers_raw)
        for raw_threshold, raw_value in tiers_dict.items():
            threshold = _policy_threshold(raw_threshold)
            if threshold is None:
                continue
            tier = _policy_tier_from_value(threshold, raw_value)
            if tier:
                tiers.append(tier)
    elif isinstance(tiers_raw, list):
        for raw_value in tiers_raw:
            if isinstance(raw_value, dict):
                tier = _policy_tier_from_dict(raw_value)
                if tier:
                    tiers.append(tier)

    # Also support the compact shape the user sketched:
    # {"20": {"agent": "claude", "model": "opus"}, ...}
    for raw_threshold, raw_value in payload.items():
        threshold = _policy_threshold(raw_threshold)
        if threshold is None:
            continue
        tier = _policy_tier_from_value(threshold, raw_value)
        if tier:
            tiers.append(tier)

    return sorted(tiers, key=lambda tier: float(tier["threshold"]))


def _policy_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _policy_string_dict(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    value_dict = cast(dict[object, object], value)
    out: dict[str, str] = {}
    for key, val in value_dict.items():
        if isinstance(key, str) and key.strip() and isinstance(val, str) and val.strip():
            out[key.strip()] = val.strip()
    return out


def _policy_bool(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _load_budget_policy() -> dict:
    payload, path, reason = _read_budget_policy_file()
    if reason:
        print_warning(f"Budget policy file ignored: {reason}")
    payload = payload or {}
    return {
        "name": str(
            payload.get("policy_name") or payload.get("name") or DEFAULT_BUDGET_POLICY_NAME
        ),
        "version": int(payload.get("version") or 1)
        if isinstance(payload.get("version"), int)
        else 1,
        "switch_percent": _policy_percent(
            payload.get("switch_percent") or payload.get("threshold_percent"),
            SMART_BUDGET_SWITCH_PERCENT,
        ),
        "below_tool": _policy_tool(payload.get("below"), SMART_BUDGET_LOW_TOOL),
        "at_or_above_tool": _policy_tool(
            payload.get("at_or_above") or payload.get("above"),
            SMART_BUDGET_HIGH_TOOL,
        ),
        "budget_ids": _policy_string_list(
            payload.get("budget_ids")
            or payload.get("filter_budget_ids")
            or payload.get("budget")
            or payload.get("budget_id")
        ),
        "tiers": _policy_tiers(payload),
        "account_host": str(payload.get("account_host") or "").strip() or None,
        "account_id": str(payload.get("account_id") or "").strip() or None,
        "account_profile": str(
            payload.get("account_profile") or payload.get("budget_profile") or ""
        ).strip()
        or None,
        "budget_tags": _policy_string_dict(
            payload.get("budget_tags") or payload.get("filter_budget_tags")
        ),
        "filter_has_spending": _policy_bool(payload.get("filter_has_spending"), False),
        "source": str(path) if payload else "built-in",
    }


def _budget_policy_summary(policy: dict) -> str:
    tiers = policy.get("tiers")
    if isinstance(tiers, list) and tiers:
        tier_parts = [_tier_display(tier) for tier in tiers if isinstance(tier, dict)]
        return " → ".join(tier_parts)
    else:
        switch_percent = float(policy["switch_percent"])
        below_tool = str(policy["below_tool"])
        above_tool = str(policy["at_or_above_tool"])
        return (
            f"{TOOL_SPECS[below_tool]['display']} under {switch_percent:.0f}%, "
            f"{TOOL_SPECS[above_tool]['display']} at/after {switch_percent:.0f}%"
        )


def _budget_policy_filters(
    policy: dict,
) -> tuple[str | None, str | None, str | None, list[str] | None, dict[str, str] | None, bool]:
    account_host = policy.get("account_host")
    account_id = policy.get("account_id")
    account_profile = policy.get("account_profile")
    filter_has_spending = bool(policy.get("filter_has_spending"))
    filter_budget_ids: list[str] | None = None
    filter_budget_tags: dict[str, str] | None = None
    budget_ids = policy.get("budget_ids")
    if isinstance(budget_ids, list) and budget_ids:
        filter_budget_ids = [str(item) for item in budget_ids]
    budget_tags = policy.get("budget_tags")
    if isinstance(budget_tags, dict) and budget_tags:
        filter_budget_tags = cast(dict[str, str], budget_tags)
    return (
        account_host if isinstance(account_host, str) else None,
        account_id if isinstance(account_id, str) else None,
        account_profile if isinstance(account_profile, str) else None,
        filter_budget_ids,
        filter_budget_tags,
        filter_has_spending,
    )


def _tier_display(tier: dict) -> str:
    tool = str(tier["tool"])
    threshold = float(tier["threshold"])
    model = tier.get("model")
    model_suffix = f"/{model}" if isinstance(model, str) and model else ""
    return f"{threshold:.0f}% {TOOL_SPECS[tool]['display']}{model_suffix}"


def _active_budget_tier(policy: dict, percent: float) -> dict | None:
    tiers = policy.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        return None
    tier_dicts = [tier for tier in tiers if isinstance(tier, dict)]
    if not tier_dicts:
        return None
    for tier in tier_dicts:
        if percent < float(tier["threshold"]):
            return tier
    return tier_dicts[-1]


def _smart_budget_policy_message(policy: dict, percent: float) -> tuple[str, str]:
    tier = _active_budget_tier(policy, percent)
    if tier:
        tool = str(tier["tool"])
        threshold = float(tier["threshold"])
        comparator = "<" if percent < threshold else ">="
        model = tier.get("model")
        model_suffix = f" / {model}" if isinstance(model, str) and model else ""
        return (
            tool,
            f"{TOOL_SPECS[tool]['display']}{model_suffix} "
            f"(current {percent:.0f}% {comparator} {threshold:.0f}%)",
        )

    switch_percent = float(policy["switch_percent"])
    if percent < switch_percent:
        return (
            str(policy["below_tool"]),
            f"{TOOL_SPECS[str(policy['below_tool'])]['display']} "
            f"(current {percent:.0f}% < {switch_percent:.0f}%)",
        )
    return (
        str(policy["at_or_above_tool"]),
        f"{TOOL_SPECS[str(policy['at_or_above_tool'])]['display']} "
        f"(current {percent:.0f}% >= {switch_percent:.0f}%)",
    )


def _print_smart_budget_policy(tool: str, statuses: list[dict], policy: dict) -> None:
    percent = _max_budget_usage_percent(statuses)
    if percent is None:
        return
    recommended_tool, message = _smart_budget_policy_message(policy, percent)
    if tool == recommended_tool:
        print_kv("Recommended agent", message)
    else:
        print_warning(f"Budget policy recommends {message}")


def _budget_panel(policy: dict, statuses: list[dict], current_tool: str) -> Panel:
    percent = _max_budget_usage_percent(statuses)
    recommendation = "unknown"
    recommended_tool = None
    if percent is not None:
        recommended_tool, recommendation = _smart_budget_policy_message(policy, percent)

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(style="cyan", overflow="fold")
    table.add_row("Policy", _budget_policy_summary(policy))
    if percent is not None:
        table.add_row("Usage", f"{percent:.1f}%")
    table.add_row("Current", TOOL_SPECS[current_tool]["display"])
    table.add_row("Suggested", recommendation)
    if recommended_tool and recommended_tool != current_tool:
        table.add_row("Action", f"Switch to {TOOL_SPECS[recommended_tool]['display']}")
    for entry in statuses:
        key, value = _format_budget_status(entry)
        table.add_row(key, value)
    return Panel(table, title="Smart Budget Policy", style="bold blue", expand=False)


def _fetch_budget_statuses_for_policy(
    state: dict,
    policy: dict,
) -> tuple[list[dict], str | None]:
    workspace = state.get("workspace")
    if not isinstance(workspace, str) or not workspace.strip():
        return [], "no workspace configured"
    (
        account_host,
        account_id,
        account_profile,
        filter_budget_ids,
        filter_budget_tags,
        filter_has_spending,
    ) = _budget_policy_filters(policy)
    return fetch_current_user_budget_spend_status(
        workspace,
        state.get("profile"),
        account_host=account_host,
        account_id=account_id,
        account_profile=account_profile,
        filter_budget_ids=filter_budget_ids,
        filter_budget_tags=filter_budget_tags,
        filter_has_spending=filter_has_spending,
    )


def resolve_budget_policy_launch(state: dict) -> tuple[str, str | None] | None:
    policy = _load_budget_policy()
    statuses, reason = _fetch_budget_statuses_for_policy(state, policy)
    if reason or not statuses:
        return None
    percent = _max_budget_usage_percent(statuses)
    if percent is None:
        return None
    tier = _active_budget_tier(policy, percent)
    if tier:
        return str(tier["tool"]), _policy_model(tier)
    recommended_tool, _ = _smart_budget_policy_message(policy, percent)
    if recommended_tool == str(policy.get("below_tool")):
        return recommended_tool, _policy_model(policy.get("below"))
    return recommended_tool, _policy_model(policy.get("at_or_above") or policy.get("above"))


def print_budget_spend_status(tool: str, state: dict) -> None:
    """Best-effort launch-time budget status for tools backed by AI Gateway."""
    if tool not in BUDGET_STATUS_TOOLS:
        return
    workspace = state.get("workspace")
    if not isinstance(workspace, str) or not workspace.strip():
        return
    policy = _load_budget_policy()
    statuses, reason = _fetch_budget_statuses_for_policy(state, policy)
    if reason:
        print_kv("Budget policy", _budget_policy_summary(policy))
        print_warning(f"Budget spend unavailable: {reason}")
        return
    if not statuses:
        print_kv("Budget policy", _budget_policy_summary(policy))
        print_kv("Budget spend", "no active AI Gateway budget found")
        return
    console.print()
    console.print(_budget_panel(policy, statuses, tool))
    _print_smart_budget_policy(tool, statuses, policy)


def budget_hook_status_line(tool: str, state: dict) -> str:
    """Compact, non-blocking budget status for agent lifecycle hooks."""
    policy = _load_budget_policy()
    statuses, reason = _fetch_budget_statuses_for_policy(state, policy)
    if reason:
        return f"ucode budget: unavailable ({reason})"
    if not statuses:
        return "ucode budget: no active AI Gateway budget found"
    percent = _max_budget_usage_percent(statuses)
    recommendation = "unknown"
    if percent is not None:
        _, recommendation = _smart_budget_policy_message(policy, percent)
    _, budget_value = _format_budget_status(_primary_budget_status(statuses))
    return f"ucode budget: {budget_value}; policy suggests {recommendation}"


def launch(tool: str, state: dict, tool_args: list[str]) -> None:
    print_budget_spend_status(tool, state)
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
    if tool == "copilot":
        return bool(state.get("claude_models")) or bool(state.get("codex_models"))
    if tool == "pi":
        return (
            bool(state.get("claude_models"))
            or bool(state.get("codex_models"))
            or bool(state.get("gemini_models"))
        )
    return False


_TOOL_DISCOVERY_SOURCES: dict[str, tuple[str, ...]] = {
    "claude": ("claude",),
    "opencode": ("claude", "gemini"),
    "codex": ("codex",),
    "gemini": ("gemini",),
    "copilot": ("claude", "codex"),
    "pi": ("claude", "codex", "gemini"),
}


def _availability_failure_detail(tool: str, state: dict) -> str:
    reasons = state.get("_discovery_reasons") or {}
    if not reasons:
        return ""
    sources = _TOOL_DISCOVERY_SOURCES.get(tool, ())
    parts = [f"{source} discovery: {reasons[source]}" for source in sources if reasons.get(source)]
    if not parts:
        return ""
    return " (" + "; ".join(parts) + ")"


def configure_single_tool(tool: str, state: dict) -> dict:
    """Check availability, configure, and persist state for one tool only."""
    with spinner(f"Checking {TOOL_SPECS[tool]['display']} availability..."):
        ok = check_gateway_endpoint(state, tool)
    if not ok:
        detail = _availability_failure_detail(tool, state)
        raise RuntimeError(
            f"{TOOL_SPECS[tool]['display']} is not available on this workspace.{detail}"
        )
    if tool == "codex":
        state = configure_tool("codex", state)
    else:
        state, model = resolve_launch_model(tool, state, None)
        state = configure_tool(tool, state, model)
    available_tools = list(set((state.get("available_tools") or []) + [tool]))
    state["available_tools"] = available_tools
    save_state(state)
    return state


def configure_selected_tools(state: dict, tools: list[str]) -> dict:
    """Configure the given tools. Caller is responsible for ensuring each tool
    is available on the workspace.

    Merges newly-configured tools into state['available_tools'] rather than
    replacing it, so a previously-configured tool the user didn't pick this
    run is preserved.
    """
    for tool in tools:
        if tool == "codex":
            state = configure_tool("codex", state)
        else:
            state, model = resolve_launch_model(tool, state, None)
            state = configure_tool(tool, state, model)

    existing = state.get("available_tools") or []
    state["available_tools"] = sorted(set(existing) | set(tools))
    save_state(state)
    return state


def configure_all_tools(state: dict) -> dict:
    """Discover available tools on the workspace and configure all of them.

    Thin wrapper retained for callers that want the legacy "configure
    everything that works" behavior.
    """
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

    return configure_selected_tools(state, available_tools)


def ensure_provider_state(tool: str) -> dict:
    """Validate that workspace + tool are configured. Caller is expected to
    handle auth (typically via `configure_shared_state` immediately after)."""
    state = load_state()
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured. Run `ucode configure` first.")
    available_tools = state.get("available_tools") or []
    if tool not in available_tools:
        raise RuntimeError(
            f"{TOOL_SPECS[tool]['display']} is not available on this workspace. "
            f"Run `ucode configure` to set up your agents."
        )
    return state


def validate_tool(tool: str) -> tuple[bool, str]:
    """Invoke a tool with a simple prompt to verify it works. Returns (ok, error_msg)."""
    spec = TOOL_SPECS[tool]
    binary = spec["binary"]
    module = _MODULES[tool]
    cmd = module.validate_cmd(binary)
    env = None
    if hasattr(module, "validate_env"):
        try:
            env = module.validate_env(load_state())
        except RuntimeError:
            env = None
    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=60, env=env
        )
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

    from ucode.agents.pi import PI_SETTINGS_BACKUP_PATH, PI_SETTINGS_PATH
    from ucode.config_io import restore_file

    low_verbosity = is_low_verbosity()
    console.print()
    if low_verbosity:
        console.print("[bold blue]Validating...[/bold blue]")
    else:
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
            # Rollback settings.json for Pi
            if tool == "pi":
                restore_file(PI_SETTINGS_PATH, PI_SETTINGS_BACKUP_PATH, managed)
            available_tools.remove(tool)
    state["available_tools"] = available_tools
    save_state(state)

    success_tools = [(t, s) for t, s in results if s]
    if success_tools and not low_verbosity:
        console.print()
        lines = []
        for tool, _ in success_tools:
            spec = TOOL_SPECS[tool]
            lines.append(
                f"[green]✓[/green] [bold]{spec['display']}[/bold] — "
                f"run with [cyan]ucode {tool}[/cyan]"
            )
        console.print(Panel("\n".join(lines), title="Ready", style="green", expand=False))
