from __future__ import annotations

import ucode.config_io as config_io
from ucode.policies import (
    _parse_on_budget_exhausted,
    active_tier,
    daily_budget_usd,
    default_model_for_harness,
    load_workspace_policy,
    normalize_policy,
    on_budget_exhausted,
    parse_and_validate_policy_yaml,
    parse_policy_yaml,
    policy_cache_path,
    save_workspace_policy,
    validate_policy,
)

POLICY_YAML = """
policy:
  name: coding-agents-default
  daily_budget_usd: 50
  tiers:
    - name: premium
      activates_at_pct: 0
      harness: claude
      model: claude-opus-4-8
    - name: standard
      activates_at_pct: 60
      harness: claude
      model: claude-sonnet-4-6
    - name: economy
      activates_at_pct: 80
      harness: opencode
      model: kimi-k2
  on_budget_exhausted: warn
"""


def test_parse_policy_yaml_normalizes_policy():
    policy = parse_policy_yaml(POLICY_YAML)

    assert policy is not None
    assert daily_budget_usd(policy) == 50.0
    assert on_budget_exhausted(policy) == "warn"
    assert policy["policy"]["tiers"][0]["activates_at_pct"] == 0.0


def test_rejects_policy_without_zero_tier():
    assert (
        normalize_policy(
            {
                "policy": {
                    "daily_budget_usd": 50,
                    "tiers": [
                        {
                            "name": "standard",
                            "activates_at_pct": 60,
                            "harness": "claude",
                            "model": "sonnet",
                        }
                    ],
                    "on_budget_exhausted": "block",
                }
            }
        )
        is None
    )


def test_active_tier_tracks_spend_thresholds():
    policy = parse_policy_yaml(POLICY_YAML)

    assert active_tier(policy, 0)["name"] == "premium"
    assert active_tier(policy, 30)["name"] == "standard"
    assert active_tier(policy, 40)["name"] == "economy"
    assert default_model_for_harness(policy, "opencode", 40) == "kimi-k2"
    assert default_model_for_harness(policy, "claude", 40) is None


def test_validate_policy_accepts_valid_yaml():
    import yaml

    assert validate_policy(yaml.safe_load(POLICY_YAML)) == []


def test_validate_policy_reports_zero_tier_rule():
    errors = validate_policy(
        {
            "policy": {
                "daily_budget_usd": 50,
                "tiers": [
                    {
                        "name": "standard",
                        "activates_at_pct": 60,
                        "harness": "claude",
                        "model": "sonnet",
                    }
                ],
                "on_budget_exhausted": "block",
            }
        }
    )
    assert any("0%" in message for message in errors)


def test_validate_policy_reports_budget_and_tier_fields():
    errors = validate_policy(
        {
            "policy": {
                "daily_budget_usd": 0,
                "tiers": [
                    {
                        "name": "premium",
                        "activates_at_pct": 0,
                        "harness": "claude",
                    }
                ],
            }
        }
    )
    assert any("daily_budget_usd" in message for message in errors)
    assert any("tier 1" in message and "model" in message for message in errors)


def test_parse_and_validate_policy_yaml_round_trip():
    policy, errors = parse_and_validate_policy_yaml(POLICY_YAML)
    assert errors == []
    assert daily_budget_usd(policy) == 50.0


def test_parse_and_validate_policy_yaml_reports_malformed_yaml():
    policy, errors = parse_and_validate_policy_yaml("policy: [unterminated")
    assert policy is None
    assert errors and "Invalid YAML" in errors[0]


SWITCH_POLICY_YAML = """
policy:
  name: coding-agents-default
  daily_budget_usd: 50
  tiers:
    - name: premium
      activates_at_pct: 0
      harness: claude
      model: claude-opus-4-8
  on_budget_exhausted:
    action: switch
    target:
      harness: opencode
      model: databricks-claude-haiku-4-5
"""


def test_parse_on_budget_exhausted_accepts_plain_strings():
    assert _parse_on_budget_exhausted("block") == "block"
    assert _parse_on_budget_exhausted("warn") == "warn"
    assert _parse_on_budget_exhausted("allow") == "allow"


def test_parse_on_budget_exhausted_rejects_unknown_string():
    assert _parse_on_budget_exhausted("stop") == "block"


def test_parse_on_budget_exhausted_accepts_switch_dict():
    result = _parse_on_budget_exhausted(
        {
            "action": "switch",
            "target": {"harness": "opencode", "model": "databricks-claude-haiku-4-5"},
        }
    )
    assert result == {
        "action": "switch",
        "target": {"harness": "opencode", "model": "databricks-claude-haiku-4-5"},
    }


def test_parse_on_budget_exhausted_rejects_switch_without_target():
    assert _parse_on_budget_exhausted({"action": "switch"}) == "block"
    assert _parse_on_budget_exhausted({"action": "switch", "target": {}}) == "block"
    assert (
        _parse_on_budget_exhausted({"action": "switch", "target": {"harness": "opencode"}})
        == "block"
    )


def test_parse_on_budget_exhausted_strips_whitespace():
    result = _parse_on_budget_exhausted(
        {"action": "switch", "target": {"harness": "  opencode  ", "model": "  haiku  "}}
    )
    assert result == {"action": "switch", "target": {"harness": "opencode", "model": "haiku"}}


def test_normalize_policy_accepts_switch_on_budget_exhausted():
    policy = parse_policy_yaml(SWITCH_POLICY_YAML)

    assert policy is not None
    assert on_budget_exhausted(policy) == {
        "action": "switch",
        "target": {"harness": "opencode", "model": "databricks-claude-haiku-4-5"},
    }


def test_on_budget_exhausted_returns_block_when_missing():
    policy = parse_policy_yaml(POLICY_YAML.replace("on_budget_exhausted: warn", ""))
    assert on_budget_exhausted(policy) == "block"


def test_validate_policy_accepts_switch_on_budget_exhausted():
    import yaml

    assert validate_policy(yaml.safe_load(SWITCH_POLICY_YAML)) == []


def test_validate_policy_rejects_switch_without_model():
    errors = validate_policy(
        {
            "policy": {
                "daily_budget_usd": 50,
                "tiers": [
                    {"name": "t1", "activates_at_pct": 0, "harness": "claude", "model": "opus"}
                ],
                "on_budget_exhausted": {"action": "switch", "target": {"harness": "opencode"}},
            }
        }
    )
    assert any("on_budget_exhausted" in e for e in errors)


def test_switch_policy_round_trips_through_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config_io, "APP_DIR", tmp_path)
    workspace = "https://example.databricks.com"
    policy = parse_policy_yaml(SWITCH_POLICY_YAML)

    save_workspace_policy(workspace, policy)
    loaded = load_workspace_policy(workspace)

    assert loaded == policy
    assert on_budget_exhausted(loaded) == {
        "action": "switch",
        "target": {"harness": "opencode", "model": "databricks-claude-haiku-4-5"},
    }


def test_policy_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config_io, "APP_DIR", tmp_path)
    workspace = "https://example.databricks.com"
    policy = parse_policy_yaml(POLICY_YAML)

    save_workspace_policy(workspace, policy)

    assert policy_cache_path(workspace).is_file()
    assert load_workspace_policy(workspace) == policy
