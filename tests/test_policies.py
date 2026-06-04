from __future__ import annotations

import ucode.config_io as config_io
from ucode.policies import (
    active_tier,
    daily_budget_usd,
    default_model_for_harness,
    load_workspace_policy,
    normalize_policy,
    on_budget_exhausted,
    parse_policy_yaml,
    policy_cache_path,
    save_workspace_policy,
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


def test_policy_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config_io, "APP_DIR", tmp_path)
    workspace = "https://example.databricks.com"
    policy = parse_policy_yaml(POLICY_YAML)

    save_workspace_policy(workspace, policy)

    assert policy_cache_path(workspace).is_file()
    assert load_workspace_policy(workspace) == policy
