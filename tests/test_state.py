"""Tests for state.py — load/save/hydrate/clear/mark_tool_managed."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import ucode.state as state_mod
from ucode.state import (
    STATE_VERSION,
    build_agent_state,
    clear_state,
    hydrate_state,
    load_full_state,
    load_state,
    mark_tool_managed,
    merge_managed_workspace,
    resolve_policy_model,
    save_state,
    select_model_for_policies,
    slice_state_for_export,
)

FAKE_WS = "https://example.databricks.com"
FAKE_URLS = {
    "codex": f"{FAKE_WS}/ai-gateway/codex/v1",
    "claude": f"{FAKE_WS}/ai-gateway/anthropic",
    "gemini": f"{FAKE_WS}/ai-gateway/gemini",
    "opencode": {
        "anthropic": f"{FAKE_WS}/ai-gateway/anthropic/v1",
        "gemini": f"{FAKE_WS}/ai-gateway/gemini/v1beta",
    },
    "copilot": f"{FAKE_WS}/ai-gateway/mlflow/v1",
    "pi": {
        "claude": f"{FAKE_WS}/ai-gateway/anthropic",
        "openai": f"{FAKE_WS}/ai-gateway/codex/v1",
        "gemini": f"{FAKE_WS}/ai-gateway/gemini/v1beta",
    },
}


@pytest.fixture(autouse=True)
def patch_state_path(tmp_path, monkeypatch):
    """Redirect STATE_PATH and APP_DIR to a temp directory for every test."""
    fake_state_path = tmp_path / "state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", fake_state_path)

    import ucode.config_io as config_io_mod

    monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)


@pytest.fixture(autouse=True)
def patch_build_urls():
    """Avoid real network calls from hydrate_state."""
    with patch("ucode.state.build_shared_base_urls", return_value=FAKE_URLS):
        yield


# ---------------------------------------------------------------------------
# load_full_state
# ---------------------------------------------------------------------------


class TestLoadFullState:
    def test_returns_empty_structure_when_missing(self):
        result = load_full_state()
        assert result["state_version"] == STATE_VERSION
        assert result["current_workspace"] is None
        assert result["workspaces"] == {}

    def test_returns_empty_when_wrong_version(self, tmp_path):
        state_mod.STATE_PATH.write_text(
            json.dumps({"state_version": 0, "current_workspace": None, "workspaces": {}}),
            encoding="utf-8",
        )
        result = load_full_state()
        assert result["workspaces"] == {}

    def test_returns_empty_on_corrupt_json(self, tmp_path):
        state_mod.STATE_PATH.write_text("not json", encoding="utf-8")
        result = load_full_state()
        assert result["current_workspace"] is None

    def test_loads_valid_state(self, tmp_path):
        data = {
            "state_version": STATE_VERSION,
            "current_workspace": FAKE_WS,
            "workspaces": {FAKE_WS: {"claude_models": {"sonnet": "s4"}}},
        }
        state_mod.STATE_PATH.write_text(json.dumps(data), encoding="utf-8")
        result = load_full_state()
        assert result["current_workspace"] == FAKE_WS


# ---------------------------------------------------------------------------
# save_state / load_state round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_round_trip(self):
        state = {
            "workspace": FAKE_WS,
            "claude_models": {"sonnet": "databricks-claude-sonnet-4"},
        }
        save_state(state)
        loaded = load_state()
        assert loaded["workspace"] == FAKE_WS
        assert loaded["claude_models"]["sonnet"] == "databricks-claude-sonnet-4"

    def test_save_respects_dry_run(self):
        import ucode.config_io as config_io_mod

        config_io_mod.set_dry_run(True)
        try:
            save_state({"workspace": FAKE_WS})
            assert not state_mod.STATE_PATH.exists()
        finally:
            config_io_mod.set_dry_run(False)

    def test_load_state_returns_empty_when_no_workspace(self):
        result = load_state()
        assert result == {}


# ---------------------------------------------------------------------------
# clear_state
# ---------------------------------------------------------------------------


class TestClearState:
    def test_clears_current_workspace(self):
        save_state({"workspace": FAKE_WS, "claude_models": {}})
        clear_state()
        full = load_full_state()
        assert full["current_workspace"] is None
        assert FAKE_WS not in full.get("workspaces", {})

    def test_clear_when_no_state_is_noop(self):
        clear_state()  # should not raise


# ---------------------------------------------------------------------------
# hydrate_state
# ---------------------------------------------------------------------------


class TestHydrateState:
    def test_empty_input_returns_empty(self):
        result = hydrate_state({})
        assert result == {
            "managed_configs": {},
            "policies": {},
            "base_urls": {},
            "agents": {},
        }

    def test_non_dict_returns_empty(self):
        assert hydrate_state(None) == {}  # type: ignore[arg-type]
        assert hydrate_state("string") == {}  # type: ignore[arg-type]

    def test_populates_base_urls_when_workspace_present(self):
        result = hydrate_state({"workspace": FAKE_WS})
        assert result["base_urls"] == FAKE_URLS

    def test_no_base_urls_when_no_workspace(self):
        result = hydrate_state({"claude_models": {}})
        assert result["base_urls"] == {}
        assert result["agents"] == {}

    def test_populates_agent_state_when_workspace_present(self):
        result = hydrate_state(
            {
                "workspace": FAKE_WS,
                "claude_models": {"opus": "claude-opus"},
                "codex_models": ["gpt-5"],
            }
        )

        assert result["agents"]["claude"]["model"] == "claude-opus"
        assert result["agents"]["claude"]["base_url"] == FAKE_URLS["claude"]
        assert result["agents"]["claude"]["auth_command"].startswith("if [ -n")
        assert result["agents"]["codex"]["model"] == "gpt-5"
        assert result["agents"]["codex"]["base_url"] == FAKE_URLS["codex"]
        assert (
            result["agents"]["codex"]["auth"]["args"][1]
            == result["agents"]["codex"]["auth_command"]
        )
        assert result["agents"]["pi"]["model"] == "claude-opus"
        assert result["agents"]["pi"]["base_urls"] == FAKE_URLS["pi"]

    def test_normalizes_managed_configs_dict_entry(self):
        state = {"managed_configs": {"claude": {"keys": [["env", "X"]]}}}
        result = hydrate_state(state)
        assert result["managed_configs"]["claude"] == {"keys": [["env", "X"]]}

    def test_normalizes_managed_configs_truthy_entry(self):
        state = {"managed_configs": {"codex": True}}
        result = hydrate_state(state)
        assert result["managed_configs"]["codex"] == {"keys": []}

    def test_drops_falsy_managed_configs(self):
        state = {"managed_configs": {"codex": False, "claude": None}}
        result = hydrate_state(state)
        assert "codex" not in result["managed_configs"]
        assert "claude" not in result["managed_configs"]


class TestBuildAgentState:
    def test_returns_empty_without_workspace(self):
        result = build_agent_state({"base_urls": FAKE_URLS})
        assert result == {}


# ---------------------------------------------------------------------------
# mark_tool_managed
# ---------------------------------------------------------------------------


class TestPolicies:
    """Schema preservation + pure policy evaluation for the per-agent layout."""

    _CLAUDE_POLICY = {
        "spending_limit": {
            "monthly_limit_usd": 300.0,
            "start_date": "2026-06-03",
        },
        "default_model": "databricks-claude-sonnet-4-5",
        "tier_2": {"usage_percent": 50.0, "model": "databricks-claude-sonnet-3-5"},
        "tier_3": {"usage_percent": 90.0, "model": "databricks-claude-haiku-3-5"},
    }

    def test_policies_round_trip_through_save_and_load(self):
        save_state({"workspace": FAKE_WS, "policies": {"claude": self._CLAUDE_POLICY}})
        loaded = load_state()
        assert loaded["policies"] == {"claude": self._CLAUDE_POLICY}

    def test_hydrate_defaults_policies_to_empty_dict(self):
        result = hydrate_state({"workspace": FAKE_WS})
        assert result["policies"] == {}

    def test_hydrate_drops_unknown_agent_keys(self):
        result = hydrate_state(
            {
                "workspace": FAKE_WS,
                "policies": {"not-an-agent": self._CLAUDE_POLICY},
            }
        )
        assert result["policies"] == {}

    def test_hydrate_drops_malformed_spending_limit(self):
        # Bad monthly_limit_usd → spending_limit dropped, but valid sibling
        # fields on the same agent (default_model) survive.
        result = hydrate_state(
            {
                "workspace": FAKE_WS,
                "policies": {
                    "claude": {
                        "spending_limit": {"monthly_limit_usd": "lots"},
                        "default_model": "databricks-claude-sonnet-4-5",
                    }
                },
            }
        )
        assert result["policies"] == {"claude": {"default_model": "databricks-claude-sonnet-4-5"}}

    def test_hydrate_autofills_missing_start_date_with_today(self, monkeypatch):
        monkeypatch.setattr("ucode.state._today_iso", lambda: "2026-06-03")
        result = hydrate_state(
            {
                "workspace": FAKE_WS,
                "policies": {"codex": {"spending_limit": {"monthly_limit_usd": 200}}},
            }
        )
        assert result["policies"]["codex"]["spending_limit"] == {
            "monthly_limit_usd": 200.0,
            "start_date": "2026-06-03",
        }

    def test_select_model_returns_requested_with_no_policies(self):
        assert (
            select_model_for_policies(
                {}, "claude", "databricks-claude-opus-4", current_spend_usd=9999.0
            )
            == "databricks-claude-opus-4"
        )

    def test_select_model_returns_default_under_tier_2_threshold(self):
        # spend = $50 of $300 → 16.7% < tier_2.usage_percent(50%), use default.
        assert (
            select_model_for_policies(
                {"policies": {"claude": self._CLAUDE_POLICY}},
                "claude",
                "databricks-claude-opus-4",
                current_spend_usd=50.0,
            )
            == "databricks-claude-sonnet-4-5"
        )

    def test_select_model_returns_tier_2_at_threshold(self):
        # spend = $150 of $300 = 50% → tier_2.
        assert (
            select_model_for_policies(
                {"policies": {"claude": self._CLAUDE_POLICY}},
                "claude",
                "databricks-claude-opus-4",
                current_spend_usd=150.0,
            )
            == "databricks-claude-sonnet-3-5"
        )

    def test_select_model_returns_tier_3_at_threshold(self):
        # spend = $270 of $300 = 90% → tier_3 wins over tier_2.
        assert (
            select_model_for_policies(
                {"policies": {"claude": self._CLAUDE_POLICY}},
                "claude",
                "databricks-claude-opus-4",
                current_spend_usd=270.0,
            )
            == "databricks-claude-haiku-3-5"
        )

    def test_select_model_is_per_agent(self):
        # codex has no policy entry → unaffected by claude's spending limit.
        state = {"policies": {"claude": self._CLAUDE_POLICY}}
        assert (
            select_model_for_policies(state, "codex", "gpt-5", current_spend_usd=10_000.0)
            == "gpt-5"
        )

    def test_select_model_skips_tiers_without_spending_limit(self):
        # tier_2 configured but no spending_limit → no denominator → fall back
        # to default_model regardless of spend.
        state = {
            "policies": {
                "claude": {
                    "default_model": "databricks-claude-sonnet-4-5",
                    "tier_2": {
                        "usage_percent": 50.0,
                        "model": "databricks-claude-haiku-3-5",
                    },
                }
            }
        }
        assert (
            select_model_for_policies(
                state, "claude", "databricks-claude-opus-4", current_spend_usd=10_000.0
            )
            == "databricks-claude-sonnet-4-5"
        )


class TestResolvePolicyModel:
    """`resolve_policy_model` — the wrapper that callers (CLI, agents) use.

    Verifies it threads the workspace-scoped spend stub into
    ``select_model_for_policies`` and surfaces the admin-pinned default.
    """

    def test_returns_admin_default_when_pinned(self):
        state = {
            "workspace": FAKE_WS,
            "policies": {"claude": {"default_model": "admin-pinned"}},
        }
        assert resolve_policy_model(state, "claude", "user-requested") == "admin-pinned"

    def test_returns_requested_when_no_policy(self):
        state = {"workspace": FAKE_WS}
        assert resolve_policy_model(state, "claude", "user-requested") == "user-requested"

    def test_returns_requested_when_no_workspace(self):
        # No workspace means no spend lookup, but absent policies still falls
        # straight through to the requested model.
        assert resolve_policy_model({}, "claude", "user-requested") == "user-requested"


class TestSliceStateForExport:
    """`slice_state_for_export` — produces the flat single-workspace blob that
    `ucode export` actually uploads to UC. The multi-workspace wrapper stays
    on the admin's local machine."""

    def test_returns_flat_block_with_workspace_field(self):
        full = {
            "state_version": 3,
            "current_workspace": FAKE_WS,
            "workspaces": {
                FAKE_WS: {
                    "workspace": FAKE_WS,
                    "claude_models": {"opus": "o4"},
                    "policies": {"claude": {"default_model": "haiku"}},
                },
            },
        }
        sliced = slice_state_for_export(full, FAKE_WS)
        assert sliced["workspace"] == FAKE_WS
        assert sliced["state_version"] == 3
        assert sliced["claude_models"] == {"opus": "o4"}
        assert sliced["policies"] == {"claude": {"default_model": "haiku"}}
        assert "workspaces" not in sliced
        assert "current_workspace" not in sliced

    def test_drops_other_workspaces_and_per_machine_fields(self):
        full = {
            "state_version": 3,
            "current_workspace": FAKE_WS,
            "workspaces": {
                FAKE_WS: {"workspace": FAKE_WS, "policies": {}},
                "https://other.databricks.com": {"claude_models": {"opus": "leaked"}},
            },
            "mcp_servers": ["should not be uploaded"],
        }
        sliced = slice_state_for_export(full, FAKE_WS)
        # Only the target workspace's fields are present (no merger of others).
        assert "mcp_servers" not in sliced
        assert all("leaked" not in str(v) for v in sliced.values())

    def test_workspace_field_authoritative_over_stale_block_value(self):
        full = {
            "state_version": 3,
            "workspaces": {
                FAKE_WS: {"workspace": "https://stale.databricks.com", "policies": {}},
            },
        }
        sliced = slice_state_for_export(full, FAKE_WS)
        assert sliced["workspace"] == FAKE_WS

    def test_raises_when_workspace_missing(self):
        full = {"state_version": 3, "workspaces": {FAKE_WS: {}}}
        with pytest.raises(RuntimeError, match="No local state for workspace"):
            slice_state_for_export(full, "https://never-configured.databricks.com")

    def test_publishes_block_minus_machine_local_fields(self):
        block = {
            "workspace": FAKE_WS,
            "profile": "admins-cli-profile",
            "managed_configs": {"claude": {"keys": [["env", "X"]]}},
            "claude_models": {"opus": "admin-pinned"},
            "available_tools": ["claude", "codex"],
            "mcp_servers": [{"name": "jira"}],
            "policies": {"claude": {}},
        }
        full = {"state_version": 3, "workspaces": {FAKE_WS: block}}
        sliced = slice_state_for_export(full, FAKE_WS)
        # The exporter's local profile name is machine-specific; consumers
        # resolve their own from the workspace URL, so it must not be published.
        assert "profile" not in sliced
        for key, value in block.items():
            if key == "profile":
                continue
            assert sliced[key] == value
        assert sliced["state_version"] == 3

    def test_strips_admin_tracking_uri_from_tracing_block(self):
        # The tracing tracking_uri is `databricks://<admin-profile>` — a local
        # credential pointer, not a shared address. It must not be published;
        # the rest of the tracing block (the shared experiment) is preserved.
        block = {
            "workspace": FAKE_WS,
            "tracing": {
                "enabled": True,
                "tracking_uri": "databricks://eng-ml-inference-team-us-east-1",
                "experiment_id": "111",
                "uc_destination": "main.default.ucode_traces",
                "sql_warehouse_id": "wh123",
            },
        }
        full = {"state_version": 3, "workspaces": {FAKE_WS: block}}
        sliced = slice_state_for_export(full, FAKE_WS)
        assert "tracking_uri" not in sliced["tracing"]
        assert sliced["tracing"]["enabled"] is True
        assert sliced["tracing"]["experiment_id"] == "111"
        assert sliced["tracing"]["uc_destination"] == "main.default.ucode_traces"
        assert sliced["tracing"]["sql_warehouse_id"] == "wh123"
        # The source block is not mutated.
        assert block["tracing"]["tracking_uri"] == "databricks://eng-ml-inference-team-us-east-1"


class TestMergeManagedWorkspace:
    def test_full_replace_from_remote(self):
        local = {
            "workspace": FAKE_WS,
            "profile": "user-local-profile",
            "claude_models": {"opus": "local-stale"},
            "managed_configs": {"claude": {"keys": [["env", "X"]]}},
            "agents": {"claude": {"auth_command": "local-cmd"}},
        }
        remote = {
            "workspace": FAKE_WS,
            "profile": "admin-profile",
            "claude_models": {"opus": "admin-pinned"},
            "base_urls": {"claude": "https://admin"},
            "managed_configs": {"claude": {"keys": [["admin-key"]]}},
            "agents": {"claude": {"auth_command": "admin-cmd"}},
            "policies": {"claude": {"default_model": "haiku"}},
        }
        merged = merge_managed_workspace(local, remote)
        # Every field in remote replaces local — even per-machine ones.
        assert merged == {**remote, "workspace": FAKE_WS}

    def test_accepts_workspace_less_remote_blob(self):
        local = {"workspace": FAKE_WS, "profile": "user-local-profile"}
        remote = {
            "claude_models": {"opus": "admin-pinned"},
            "available_tools": ["claude"],
        }

        merged = merge_managed_workspace(local, remote)

        assert merged == {
            **remote,
            "workspace": FAKE_WS,
        }

    def test_profile_override_uses_resolved_local_profile(self):
        local = {"workspace": FAKE_WS, "profile": "stale-local-profile"}
        remote = {
            "workspace": FAKE_WS,
            "profile": "admin-profile",
            "claude_models": {"opus": "admin-pinned"},
        }

        merged = merge_managed_workspace(local, remote, profile="user-local-profile")

        assert merged == {
            **remote,
            "workspace": FAKE_WS,
            "profile": "user-local-profile",
        }

    def test_localizes_tracing_uri_to_local_profile(self):
        # The admin's export bakes its own profile into the tracing URI; after a
        # pull it must point at the user's local profile or MLflow auth fails.
        local = {"workspace": FAKE_WS, "profile": "user-local-profile"}
        remote = {
            "workspace": FAKE_WS,
            "profile": "admin-profile",
            "tracing": {
                "enabled": True,
                "tracking_uri": "databricks://admin-profile",
                "experiment_id": "111",
            },
        }

        merged = merge_managed_workspace(local, remote, profile="user-local-profile")

        assert merged["tracing"]["tracking_uri"] == "databricks://user-local-profile"
        # Other tracing fields are preserved untouched.
        assert merged["tracing"]["experiment_id"] == "111"

    def test_localizes_tracing_uri_to_bare_databricks_without_profile(self):
        local = {"workspace": FAKE_WS}
        remote = {
            "workspace": FAKE_WS,
            "tracing": {"enabled": True, "tracking_uri": "databricks://admin-profile"},
        }

        merged = merge_managed_workspace(local, remote)

        assert merged["tracing"]["tracking_uri"] == "databricks"

    def test_sets_tracking_uri_when_export_omits_it(self):
        # New exports strip tracking_uri; the consumer must populate it from the
        # locally-resolved profile rather than leave it missing.
        local = {"workspace": FAKE_WS, "profile": "user-local-profile"}
        remote = {
            "workspace": FAKE_WS,
            "tracing": {"enabled": True, "experiment_id": "111"},
        }

        merged = merge_managed_workspace(local, remote, profile="user-local-profile")

        assert merged["tracing"]["tracking_uri"] == "databricks://user-local-profile"

    def test_no_op_when_workspaces_dont_match(self):
        local = {"workspace": FAKE_WS, "claude_models": {"opus": "local"}}
        remote = {
            "workspace": "https://other.databricks.com",
            "claude_models": {"opus": "wrong-workspace"},
        }
        assert merge_managed_workspace(local, remote) == local

    def test_no_op_when_local_has_no_workspace(self):
        local = {"claude_models": {}}
        remote = {"workspace": FAKE_WS, "policies": {"claude": {}}}
        assert merge_managed_workspace(local, remote) == local

    def test_no_op_when_remote_blob_malformed(self):
        local = {"workspace": FAKE_WS}
        assert merge_managed_workspace(local, "not-a-dict") == local  # type: ignore[arg-type]


class TestMarkToolManaged:
    def test_sets_managed_keys(self):
        state: dict = {}
        result = mark_tool_managed(state, "claude", [["env", "X"], ["apiKeyHelper"]])
        assert result["managed_configs"]["claude"] == {"keys": [["env", "X"], ["apiKeyHelper"]]}

    def test_sets_last_tool(self):
        state: dict = {}
        result = mark_tool_managed(state, "codex", [])
        assert result["last_tool"] == "codex"

    def test_preserves_existing_managed_configs(self):
        state = {"managed_configs": {"gemini": {"keys": [["GEMINI_MODEL"]]}}}
        result = mark_tool_managed(state, "codex", [["profile"]])
        assert "gemini" in result["managed_configs"]
        assert "codex" in result["managed_configs"]
