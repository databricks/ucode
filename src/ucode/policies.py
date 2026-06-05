"""Workspace policy YAML loading, validation, and local cache helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, cast

import yaml

import ucode.config_io as config_io

OnBudgetExhausted = Literal["block", "warn", "allow"]

POLICY_CACHE_DIRNAME = "policies"
DEFAULT_POLICY_NAME = "coding-agents-default"
DEFAULT_ON_BUDGET_EXHAUSTED: OnBudgetExhausted = "block"
VALID_ON_BUDGET_EXHAUSTED: frozenset[str] = frozenset({"block", "warn", "allow"})


def _policy_cache_dir() -> Path:
    return config_io.APP_DIR / POLICY_CACHE_DIRNAME


def policy_cache_path(workspace: str) -> Path:
    digest = hashlib.sha256(workspace.rstrip("/").encode("utf-8")).hexdigest()[:16]
    return _policy_cache_dir() / f"{digest}.yaml"


def normalize_policy(raw: object) -> dict | None:
    """Return a normalized policy dict, or ``None`` when the YAML is unusable."""
    if not isinstance(raw, dict):
        return None
    raw_dict = cast("dict[str, object]", raw)
    root = raw_dict.get("policy")
    if not isinstance(root, dict):
        return None
    policy_raw = cast("dict[str, object]", root)

    budget = policy_raw.get("daily_budget_usd")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0:
        return None

    name = policy_raw.get("name")
    if not isinstance(name, str) or not name.strip():
        name = DEFAULT_POLICY_NAME

    exhausted = policy_raw.get("on_budget_exhausted")
    if not isinstance(exhausted, str) or exhausted not in VALID_ON_BUDGET_EXHAUSTED:
        exhausted = DEFAULT_ON_BUDGET_EXHAUSTED

    tiers_raw = policy_raw.get("tiers")
    if not isinstance(tiers_raw, list) or not tiers_raw:
        return None

    tiers: list[dict[str, object]] = []
    for item in tiers_raw:
        if not isinstance(item, dict):
            return None
        item_dict = cast("dict[str, object]", item)
        tier_name = item_dict.get("name")
        pct = item_dict.get("activates_at_pct")
        harness = item_dict.get("harness")
        model = item_dict.get("model")
        if (
            not isinstance(tier_name, str)
            or not tier_name.strip()
            or not isinstance(pct, (int, float))
            or isinstance(pct, bool)
            or pct < 0
            or pct > 100
            or not isinstance(harness, str)
            or not harness.strip()
            or not isinstance(model, str)
            or not model.strip()
        ):
            return None
        tiers.append(
            {
                "name": tier_name.strip(),
                "activates_at_pct": float(pct),
                "harness": harness.strip(),
                "model": model.strip(),
            }
        )

    tiers.sort(key=lambda tier: float(tier["activates_at_pct"]))
    if tiers[0]["activates_at_pct"] != 0:
        return None

    return {
        "policy": {
            "name": name.strip(),
            "daily_budget_usd": float(budget),
            "tiers": tiers,
            "on_budget_exhausted": exhausted,
        }
    }


def validate_policy(raw: object) -> list[str]:
    """Return human-readable problems with ``raw``; an empty list means valid.

    Mirrors the constraints enforced by :func:`normalize_policy`, but reports a
    message per failure instead of collapsing everything to ``None``.
    """
    errors: list[str] = []
    if not isinstance(raw, dict):
        return ["Top-level YAML must be a mapping with a `policy:` key."]
    raw_dict = cast("dict[str, object]", raw)
    root = raw_dict.get("policy")
    if not isinstance(root, dict):
        return ["Missing `policy:` mapping at the top level."]
    root = cast("dict[str, object]", root)

    budget = root.get("daily_budget_usd")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool):
        errors.append("`daily_budget_usd` is required and must be a number.")
    elif budget <= 0:
        errors.append("`daily_budget_usd` must be greater than 0.")

    exhausted = root.get("on_budget_exhausted")
    if exhausted is not None and (
        not isinstance(exhausted, str) or exhausted not in VALID_ON_BUDGET_EXHAUSTED
    ):
        allowed = ", ".join(sorted(VALID_ON_BUDGET_EXHAUSTED))
        errors.append(f"`on_budget_exhausted` must be one of: {allowed}.")

    tiers_raw = root.get("tiers")
    if not isinstance(tiers_raw, list) or not tiers_raw:
        errors.append("`tiers` is required and must be a non-empty list.")
        return errors

    pcts: list[float] = []
    for index, item in enumerate(tiers_raw, start=1):
        if not isinstance(item, dict):
            errors.append(f"tier {index}: must be a mapping.")
            continue
        item = cast("dict[str, object]", item)
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"tier {index}: `name` is required.")
        pct = item.get("activates_at_pct")
        if not isinstance(pct, (int, float)) or isinstance(pct, bool):
            errors.append(f"tier {index}: `activates_at_pct` is required and must be a number.")
        elif pct < 0 or pct > 100:
            errors.append(f"tier {index}: `activates_at_pct` must be between 0 and 100.")
        else:
            pcts.append(float(pct))
        harness = item.get("harness")
        if not isinstance(harness, str) or not harness.strip():
            errors.append(f"tier {index}: `harness` is required.")
        model = item.get("model")
        if not isinstance(model, str) or not model.strip():
            errors.append(f"tier {index}: `model` is required.")

    if pcts and min(pcts) != 0:
        errors.append("The first tier must activate at 0% (one tier needs `activates_at_pct: 0`).")

    return errors


def parse_policy_yaml(text: str) -> dict | None:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return normalize_policy(raw)


def parse_and_validate_policy_yaml(text: str) -> tuple[dict | None, list[str]]:
    """Parse and validate ``text``, returning ``(policy, errors)``.

    On success ``errors`` is empty and ``policy`` is the normalized dict. On
    failure ``policy`` is ``None`` and ``errors`` lists what went wrong.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return None, [f"Invalid YAML: {exc}"]
    errors = validate_policy(raw)
    if errors:
        return None, errors
    return normalize_policy(raw), []


def policy_to_yaml(policy: dict) -> str:
    normalized = normalize_policy(policy)
    if normalized is None:
        raise RuntimeError("Policy is malformed and cannot be written.")
    return yaml.safe_dump(normalized, sort_keys=False)


def save_workspace_policy(workspace: str, policy: dict) -> None:
    if config_io.is_dry_run():
        return
    path = policy_cache_path(workspace)
    config_io.ensure_parent_dir(path)
    try:
        path.write_text(policy_to_yaml(policy), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write policy file: {path}") from exc


def load_workspace_policy(workspace: str | None) -> dict | None:
    if not workspace:
        return None
    path = policy_cache_path(workspace)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return parse_policy_yaml(text)


def delete_workspace_policy(workspace: str | None) -> None:
    if not workspace or config_io.is_dry_run():
        return
    try:
        policy_cache_path(workspace).unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to remove policy file for {workspace}") from exc


def daily_budget_usd(policy: dict | None) -> float | None:
    root = (policy or {}).get("policy") if isinstance(policy, dict) else None
    if not isinstance(root, dict):
        return None
    value = root.get("daily_budget_usd")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def on_budget_exhausted(policy: dict | None) -> OnBudgetExhausted:
    root = (policy or {}).get("policy") if isinstance(policy, dict) else None
    value = root.get("on_budget_exhausted") if isinstance(root, dict) else None
    if isinstance(value, str) and value in VALID_ON_BUDGET_EXHAUSTED:
        return cast("OnBudgetExhausted", value)
    return DEFAULT_ON_BUDGET_EXHAUSTED


def active_tier(policy: dict | None, spend_usd: float) -> dict | None:
    budget = daily_budget_usd(policy)
    root = (policy or {}).get("policy") if isinstance(policy, dict) else None
    tiers = root.get("tiers") if isinstance(root, dict) else None
    if budget is None or not isinstance(tiers, list) or not tiers:
        return None
    pct = (max(spend_usd, 0.0) / budget) * 100
    active: dict | None = None
    for tier in tiers:
        if not isinstance(tier, dict):
            continue
        threshold = tier.get("activates_at_pct")
        if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
            if pct >= float(threshold):
                active = tier
            else:
                break
    return active


def default_model_for_harness(
    policy: dict | None, harness: str, spend_usd: float = 0.0
) -> str | None:
    tier = active_tier(policy, spend_usd)
    if isinstance(tier, dict) and tier.get("harness") == harness:
        model = tier.get("model")
        return model if isinstance(model, str) and model else None
    return None


def resolve_policy_default_model(state: dict, harness: str, requested_model: str) -> str:
    """Apply the active YAML policy tier to an implicit default model.

    Policies only override default model selection for the harness named by the
    active tier. Explicit user models should bypass this helper.
    """
    workspace = state.get("workspace")
    policy = load_workspace_policy(workspace if isinstance(workspace, str) else None)
    if policy is None:
        return requested_model
    spend_usd = 0.0
    try:
        from ucode.usage import local_budget_status

        raw_spend = local_budget_status().get("spend_usd")
        if isinstance(raw_spend, (int, float)) and not isinstance(raw_spend, bool):
            spend_usd = float(raw_spend)
    except Exception:
        spend_usd = 0.0
    return default_model_for_harness(policy, harness, spend_usd) or requested_model
