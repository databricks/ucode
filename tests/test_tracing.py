"""Tests for ucode.tracing and per-agent tracing wiring."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import ucode.databricks as databricks
from ucode import tracing
from ucode.agents import claude, codex, opencode

WS = "https://example.databricks.com"


_AGENT_EXPERIMENTS = {
    "claude": {
        "experiment_id": "111",
        "experiment_name": "/Users/me@example.com/ucode-claude-code-traces",
    },
    "codex": {
        "experiment_id": "222",
        "experiment_name": "/Users/me@example.com/ucode-codex-traces",
    },
    "opencode": {
        "experiment_id": "333",
        "experiment_name": "/Users/me@example.com/ucode-opencode-traces",
    },
}


def _enabled_state(profile: str | None = None) -> dict:
    return {
        "workspace": WS,
        "profile": profile,
        "tracing": {
            "enabled": True,
            "tracking_uri": f"databricks://{profile}" if profile else "databricks",
            "agents": {tool: dict(entry) for tool, entry in _AGENT_EXPERIMENTS.items()},
        },
    }


class TestTrackingUri:
    def test_with_profile(self):
        assert tracing.tracking_uri_for_state({"profile": "myprof"}) == "databricks://myprof"

    def test_without_profile(self):
        assert tracing.tracking_uri_for_state({}) == "databricks"


class TestTracingConfig:
    def test_none_when_absent(self):
        assert tracing.tracing_config({}) is None

    def test_none_when_disabled(self):
        state = {"tracing": {"enabled": False, "agents": {}}}
        assert tracing.tracing_config(state) is None

    def test_returns_cfg_when_enabled(self):
        state = _enabled_state()
        assert tracing.tracing_config(state) is state["tracing"]


class TestAgentTracing:
    def test_returns_agent_entry(self):
        entry = tracing.agent_tracing(_enabled_state(), "claude")
        assert entry["experiment_id"] == "111"

    def test_none_when_disabled(self):
        assert tracing.agent_tracing({}, "claude") is None

    def test_none_when_agent_absent(self):
        state = {"tracing": {"enabled": True, "tracking_uri": "databricks", "agents": {}}}
        assert tracing.agent_tracing(state, "claude") is None


class TestTracingEnv:
    def test_empty_when_disabled(self):
        assert tracing.tracing_env({}, "claude") == {}

    def test_per_agent_uri_and_experiment(self):
        env = tracing.tracing_env(_enabled_state("p"), "codex")
        assert env == {
            "MLFLOW_TRACKING_URI": "databricks://p",
            "MLFLOW_EXPERIMENT_ID": "222",
        }

    def test_distinct_experiment_per_agent(self):
        state = _enabled_state()
        assert tracing.tracing_env(state, "claude")["MLFLOW_EXPERIMENT_ID"] == "111"
        assert tracing.tracing_env(state, "opencode")["MLFLOW_EXPERIMENT_ID"] == "333"


class TestExperimentName:
    def test_per_user_claude_slug(self):
        assert (
            tracing.experiment_name("claude", "me@example.com")
            == "/Users/me@example.com/ucode-claude-code-traces"
        )

    def test_per_user_codex_slug(self):
        assert (
            tracing.experiment_name("codex", "me@example.com")
            == "/Users/me@example.com/ucode-codex-traces"
        )

    def test_shared_fallback_when_no_user(self):
        assert tracing.experiment_name("opencode", None) == "/Shared/ucode-opencode-traces"


class TestGetOrCreateExperiment:
    def test_existing_returns_id(self):
        with patch.object(
            databricks,
            "_http_get_json",
            return_value=({"experiment": {"experiment_id": "42"}}, None),
        ):
            exp, reason = databricks.get_or_create_mlflow_experiment(WS, "tok", "/Shared/x")
        assert exp == "42"
        assert reason is None

    def test_creates_when_missing(self):
        with (
            patch.object(
                databricks,
                "_http_get_json",
                return_value=(None, "HTTP 404 RESOURCE_DOES_NOT_EXIST"),
            ),
            patch.object(
                databricks, "_http_post_json", return_value=({"experiment_id": "99"}, None)
            ) as post,
        ):
            exp, reason = databricks.get_or_create_mlflow_experiment(WS, "tok", "/Shared/x")
        assert exp == "99"
        assert reason is None
        post.assert_called_once()

    def test_refetches_on_already_exists_race(self):
        get_results = iter(
            [
                (None, "HTTP 404 RESOURCE_DOES_NOT_EXIST"),
                ({"experiment": {"experiment_id": "7"}}, None),
            ]
        )
        with (
            patch.object(
                databricks, "_http_get_json", side_effect=lambda *a, **k: next(get_results)
            ),
            patch.object(
                databricks,
                "_http_post_json",
                return_value=(None, "HTTP 400: RESOURCE_ALREADY_EXISTS"),
            ),
        ):
            exp, reason = databricks.get_or_create_mlflow_experiment(WS, "tok", "/Shared/x")
        assert exp == "7"
        assert reason is None

    def test_returns_reason_on_failure(self):
        with (
            patch.object(databricks, "_http_get_json", return_value=(None, "HTTP 403 Forbidden")),
            patch.object(databricks, "_http_post_json", return_value=(None, "HTTP 403 Forbidden")),
        ):
            exp, reason = databricks.get_or_create_mlflow_experiment(WS, "tok", "/Shared/x")
        assert exp is None
        assert "403" in reason


class TestOpencodeTracingPlugin:
    def test_added_when_enabled(self):
        config: dict = {}
        opencode._apply_tracing_plugin(config, _enabled_state())
        assert config["plugin"] == ["@mlflow/opencode"]

    def test_not_duplicated(self):
        config = {"plugin": ["@mlflow/opencode"]}
        opencode._apply_tracing_plugin(config, _enabled_state())
        assert config["plugin"] == ["@mlflow/opencode"]

    def test_removed_when_disabled(self):
        config = {"plugin": ["@mlflow/opencode"]}
        opencode._apply_tracing_plugin(config, {})
        assert "plugin" not in config

    def test_preserves_user_plugins_when_enabled(self):
        config = {"plugin": ["user/thing"]}
        opencode._apply_tracing_plugin(config, _enabled_state())
        assert config["plugin"] == ["user/thing", "@mlflow/opencode"]

    def test_preserves_user_plugins_when_disabled(self):
        config = {"plugin": ["user/thing", "@mlflow/opencode"]}
        opencode._apply_tracing_plugin(config, {})
        assert config["plugin"] == ["user/thing"]


class TestCodexTracingNotify:
    def test_set_when_enabled(self):
        doc: dict = {}
        codex._apply_tracing_notify(doc, _enabled_state())
        assert doc["notify"] == ["mlflow-codex", "notify-hook"]

    def test_cleared_when_disabled(self):
        doc = {"notify": ["mlflow-codex", "notify-hook"]}
        codex._apply_tracing_notify(doc, {})
        assert "notify" not in doc

    def test_preserves_user_notify_when_disabled(self):
        doc = {"notify": ["user-notify"]}
        codex._apply_tracing_notify(doc, {})
        assert doc["notify"] == ["user-notify"]

    def test_warns_when_overwriting_user_notify(self):
        doc = {"notify": ["user-hook"]}
        with patch.object(codex, "print_warning") as warn:
            codex._apply_tracing_notify(doc, _enabled_state())
        assert doc["notify"] == ["mlflow-codex", "notify-hook"]
        warn.assert_called_once()
        msg = warn.call_args[0][0]
        assert "user-hook" in msg
        assert "backup" in msg.lower()

    def test_no_warn_when_already_ucode_notify(self):
        doc = {"notify": ["mlflow-codex", "notify-hook"]}
        with patch.object(codex, "print_warning") as warn:
            codex._apply_tracing_notify(doc, _enabled_state())
        warn.assert_not_called()


class TestApplyTracingEnv:
    def test_sets_keys_when_enabled(self):
        env: dict[str, str] = {}
        tracing.apply_tracing_env(env, _enabled_state(), "codex")
        assert env == {"MLFLOW_TRACKING_URI": "databricks", "MLFLOW_EXPERIMENT_ID": "222"}

    def test_clears_stale_keys_when_disabled(self):
        env = {
            "MLFLOW_TRACKING_URI": "databricks://stale",
            "MLFLOW_EXPERIMENT_ID": "9999",
            "UNRELATED": "keep-me",
        }
        tracing.apply_tracing_env(env, {}, "codex")
        assert "MLFLOW_TRACKING_URI" not in env
        assert "MLFLOW_EXPERIMENT_ID" not in env
        assert env["UNRELATED"] == "keep-me"

    def test_overwrites_stale_keys_when_enabled(self):
        env = {"MLFLOW_TRACKING_URI": "databricks://stale", "MLFLOW_EXPERIMENT_ID": "9999"}
        tracing.apply_tracing_env(env, _enabled_state("p"), "opencode")
        assert env["MLFLOW_TRACKING_URI"] == "databricks://p"
        assert env["MLFLOW_EXPERIMENT_ID"] == "333"


class TestClaudeTracingEnv:
    def _write(self, state: dict, tmp_path, monkeypatch) -> dict:
        settings = tmp_path / "ucode-settings.json"
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", settings)
        monkeypatch.setattr(claude, "CLAUDE_BACKUP_PATH", tmp_path / "backup.json")
        claude.write_tool_config(state, "databricks-claude-opus-4-7")
        return json.loads(settings.read_text())

    def test_injects_mlflow_env_when_enabled(self, tmp_path, monkeypatch):
        state = {**_enabled_state(), "claude_models": {}}
        env = self._write(state, tmp_path, monkeypatch)["env"]
        assert env["MLFLOW_CLAUDE_TRACING_ENABLED"] == "true"
        assert env["MLFLOW_TRACKING_URI"] == "databricks"
        assert env["MLFLOW_EXPERIMENT_ID"] == "111"

    def test_no_mlflow_env_when_disabled(self, tmp_path, monkeypatch):
        state = {"workspace": WS, "claude_models": {}}
        env = self._write(state, tmp_path, monkeypatch).get("env", {})
        assert "MLFLOW_TRACKING_URI" not in env

    def test_strips_stale_keys_when_disabled(self, tmp_path, monkeypatch):
        settings = tmp_path / "ucode-settings.json"
        settings.write_text(
            json.dumps(
                {
                    "env": {
                        "MLFLOW_CLAUDE_TRACING_ENABLED": "true",
                        "MLFLOW_TRACKING_URI": "databricks",
                        "MLFLOW_EXPERIMENT_ID": "1",
                    }
                }
            )
        )
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", settings)
        monkeypatch.setattr(claude, "CLAUDE_BACKUP_PATH", tmp_path / "backup.json")
        claude.write_tool_config(
            {"workspace": WS, "claude_models": {}}, "databricks-claude-opus-4-7"
        )
        env = json.loads(settings.read_text())["env"]
        assert "MLFLOW_TRACKING_URI" not in env
        assert "MLFLOW_EXPERIMENT_ID" not in env
        assert "MLFLOW_CLAUDE_TRACING_ENABLED" not in env


class TestSelectTracingWorkspace:
    def _full(self) -> dict:
        return {
            "current_workspace": "https://a.databricks.com",
            "workspaces": {
                "https://a.databricks.com": {"available_tools": ["claude"], "profile": "pa"},
                "https://b.databricks.com": {"available_tools": ["codex"], "profile": "pb"},
                # gemini isn't a tracing-capable agent → excluded from candidates
                "https://c.databricks.com": {"available_tools": ["gemini"]},
            },
        }

    def test_raises_when_none_configured(self):
        with patch.object(tracing, "load_full_state", return_value={"workspaces": {}}):
            with pytest.raises(RuntimeError, match="No tracing-capable"):
                tracing._select_tracing_workspace()

    def test_lists_current_first_and_excludes_non_tracing(self):
        captured: dict = {}

        def fake_prompt(desc, profiles):
            captured["profiles"] = profiles
            return ("https://a.databricks.com", "pa")

        with (
            patch.object(tracing, "load_full_state", return_value=self._full()),
            patch.object(tracing, "prompt_for_workspace", side_effect=fake_prompt),
        ):
            state = tracing._select_tracing_workspace()

        hosts = [host for host, _ in captured["profiles"]]
        assert hosts[0] == "https://a.databricks.com"
        assert "https://c.databricks.com" not in hosts
        assert state["workspace"] == "https://a.databricks.com"

    def test_returns_picked_workspace_state(self):
        with (
            patch.object(tracing, "load_full_state", return_value=self._full()),
            patch.object(
                tracing, "prompt_for_workspace", return_value=("https://b.databricks.com", "pb")
            ),
        ):
            state = tracing._select_tracing_workspace()
        assert state["workspace"] == "https://b.databricks.com"
        assert state["profile"] == "pb"
        assert "codex" in state["available_tools"]

    def test_raises_when_picked_workspace_unconfigured(self):
        with (
            patch.object(tracing, "load_full_state", return_value=self._full()),
            patch.object(
                tracing, "prompt_for_workspace", return_value=("https://x.databricks.com", None)
            ),
        ):
            with pytest.raises(RuntimeError, match="no tracing-capable agents"):
                tracing._select_tracing_workspace()

    def test_single_candidate_skips_prompt(self):
        full = {
            "current_workspace": "https://a.databricks.com",
            "workspaces": {
                "https://a.databricks.com": {"available_tools": ["claude"], "profile": "pa"},
            },
        }
        with (
            patch.object(tracing, "load_full_state", return_value=full),
            patch.object(tracing, "prompt_for_workspace") as prompt,
        ):
            state = tracing._select_tracing_workspace()
        prompt.assert_not_called()
        assert state["workspace"] == "https://a.databricks.com"


class TestSelectTracingWorkspaceOnlyEnabled:
    def test_empty_when_none_enabled(self):
        full = {
            "workspaces": {
                "https://a.databricks.com": {"available_tools": ["claude"]},
            },
        }
        with patch.object(tracing, "load_full_state", return_value=full):
            assert tracing._select_tracing_workspace(only_enabled=True) == {}

    def test_auto_selects_lone_enabled_workspace(self):
        full = {
            "current_workspace": "https://a.databricks.com",
            "workspaces": {
                "https://a.databricks.com": {"available_tools": ["claude"]},
                "https://b.databricks.com": {
                    "available_tools": ["claude"],
                    "tracing": {"enabled": True, "agents": {}},
                },
            },
        }
        with (
            patch.object(tracing, "load_full_state", return_value=full),
            patch.object(tracing, "prompt_for_workspace") as prompt,
        ):
            state = tracing._select_tracing_workspace(only_enabled=True)
        prompt.assert_not_called()
        assert state["workspace"] == "https://b.databricks.com"

    def test_prompts_when_multiple_enabled(self):
        full = {
            "current_workspace": "https://a.databricks.com",
            "workspaces": {
                "https://a.databricks.com": {
                    "available_tools": ["claude"],
                    "profile": "pa",
                    "tracing": {"enabled": True, "agents": {}},
                },
                "https://b.databricks.com": {
                    "available_tools": ["claude"],
                    "profile": "pb",
                    "tracing": {"enabled": True, "agents": {}},
                },
            },
        }
        with (
            patch.object(tracing, "load_full_state", return_value=full),
            patch.object(
                tracing, "prompt_for_workspace", return_value=("https://b.databricks.com", "pb")
            ) as prompt,
        ):
            state = tracing._select_tracing_workspace(only_enabled=True)
        prompt.assert_called_once()
        assert state["workspace"] == "https://b.databricks.com"


class TestConfigureTracingPreservesCurrentWorkspace:
    """``save_state`` flips ``current_workspace`` on every call. The tracing
    command must not change which workspace ``ucode launch`` targets, even
    when configuring tracing for a non-current workspace."""

    def _full(self) -> dict:
        return {
            "current_workspace": "https://a.databricks.com",
            "workspaces": {
                "https://a.databricks.com": {
                    "available_tools": ["claude"],
                    "profile": "pa",
                    "tracing": {"enabled": True, "agents": {}},
                },
                "https://b.databricks.com": {
                    "available_tools": ["claude"],
                    "profile": "pb",
                    "tracing": {"enabled": True, "agents": {}},
                },
            },
        }

    def test_disable_restores_original_current(self):
        captured: dict = {}
        with (
            patch.object(tracing, "load_full_state", return_value=self._full()),
            patch.object(
                tracing, "prompt_for_workspace", return_value=("https://b.databricks.com", "pb")
            ),
            patch.object(tracing, "ensure_databricks_auth"),
            patch.object(tracing, "_rewrite_agent_configs", side_effect=lambda s: s),
            patch.object(tracing, "save_state"),
            patch.object(
                tracing,
                "set_current_workspace",
                side_effect=lambda ws: captured.setdefault("restored_to", ws),
            ),
        ):
            tracing.configure_tracing_command(disable=True)
        assert captured["restored_to"] == "https://a.databricks.com"

    def test_disable_with_none_enabled_still_calls_restore(self):
        full = {
            "current_workspace": "https://a.databricks.com",
            "workspaces": {"https://a.databricks.com": {"available_tools": ["claude"]}},
        }
        captured: dict = {}
        with (
            patch.object(tracing, "load_full_state", return_value=full),
            patch.object(
                tracing,
                "set_current_workspace",
                side_effect=lambda ws: captured.setdefault("restored_to", ws),
            ),
        ):
            rc = tracing.configure_tracing_command(disable=True)
        assert rc == 0
        assert captured["restored_to"] == "https://a.databricks.com"


class TestDisableTracing:
    def test_sets_disabled_and_rewrites_configs(self):
        state = _enabled_state()
        with (
            patch.object(tracing, "_rewrite_agent_configs", side_effect=lambda s: s) as rewrite,
            patch.object(tracing, "save_state"),
        ):
            out = tracing.disable_tracing(state)
        assert out["tracing"]["enabled"] is False
        rewrite.assert_called_once()


class TestParseMlflowVersion:
    def test_parses_full_version(self):
        assert claude._parse_mlflow_version("mlflow, version 3.12.0") == (3, 12)

    def test_parses_major_minor(self):
        assert claude._parse_mlflow_version("mlflow version 3.4") == (3, 4)

    def test_returns_none_on_garbage(self):
        assert claude._parse_mlflow_version("not a version") is None


class TestEnsureMlflowCli:
    def test_noop_when_already_satisfied(self):
        with (
            patch.object(claude, "_installed_mlflow_version", return_value=(3, 5)),
            patch.object(claude.subprocess, "run") as run,
        ):
            assert claude._ensure_mlflow_cli() is True
        run.assert_not_called()

    def test_installs_when_missing(self, monkeypatch):
        monkeypatch.setattr(claude, "_installed_mlflow_version", lambda: None)
        monkeypatch.setattr(claude.shutil, "which", lambda binary: f"/bin/{binary}")
        with patch.object(claude.subprocess, "run") as run:
            assert claude._ensure_mlflow_cli() is True
        cmd = run.call_args[0][0]
        assert cmd[:3] == ["uv", "tool", "install"]
        assert "--force" not in cmd

    def test_force_upgrades_when_below_minimum(self, monkeypatch):
        monkeypatch.setattr(claude, "_installed_mlflow_version", lambda: (3, 1))
        monkeypatch.setattr(claude.shutil, "which", lambda binary: f"/bin/{binary}")
        with patch.object(claude.subprocess, "run") as run:
            assert claude._ensure_mlflow_cli() is True
        assert "--force" in run.call_args[0][0]

    def test_warns_when_uv_missing(self, monkeypatch):
        monkeypatch.setattr(claude, "_installed_mlflow_version", lambda: None)
        monkeypatch.setattr(claude.shutil, "which", lambda binary: None)
        with patch.object(claude.subprocess, "run") as run:
            assert claude._ensure_mlflow_cli() is False
        run.assert_not_called()
