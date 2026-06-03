"""Tests for ucode.tracing and per-agent tracing wiring."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import ucode.databricks as databricks
from ucode import tracing
from ucode.agents import claude, codex, opencode

WS = "https://example.databricks.com"


SHARED_EXPERIMENT_ID = "111"


def _enabled_state(profile: str | None = None) -> dict:
    return {
        "workspace": WS,
        "profile": profile,
        "tracing": {
            "enabled": True,
            "tracking_uri": f"databricks://{profile}" if profile else "databricks",
            "experiment_id": SHARED_EXPERIMENT_ID,
            "experiment_name": "/Shared/ucode-traces",
            "uc_destination": "main.default.ucode_traces",
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
    def test_returns_shared_entry(self):
        entry = tracing.agent_tracing(_enabled_state(), "claude")
        assert entry["experiment_id"] == "111"

    def test_same_entry_for_every_agent(self):
        state = _enabled_state()
        for tool in ("claude", "codex", "opencode"):
            assert tracing.agent_tracing(state, tool)["experiment_id"] == "111"

    def test_none_for_non_tracing_agent(self):
        assert tracing.agent_tracing(_enabled_state(), "gemini") is None

    def test_none_when_disabled(self):
        assert tracing.agent_tracing({}, "claude") is None

    def test_none_when_experiment_unresolved(self):
        state = {"tracing": {"enabled": True, "tracking_uri": "databricks"}}
        assert tracing.agent_tracing(state, "claude") is None


class TestTracingEnv:
    def test_empty_when_disabled(self):
        assert tracing.tracing_env({}, "claude") == {}

    def test_uri_and_experiment(self):
        env = tracing.tracing_env(_enabled_state("p"), "codex")
        assert env == {
            "MLFLOW_TRACKING_URI": "databricks://p",
            "MLFLOW_EXPERIMENT_ID": "111",
        }

    def test_shared_experiment_across_agents(self):
        state = _enabled_state()
        assert tracing.tracing_env(state, "claude")["MLFLOW_EXPERIMENT_ID"] == "111"
        assert tracing.tracing_env(state, "opencode")["MLFLOW_EXPERIMENT_ID"] == "111"


class TestExperimentName:
    def test_leaf_name(self):
        assert tracing.experiment_name() == "ucode-traces"


def _experiment(name: str, exp_id: str, uc_destination: str | None) -> dict:
    tags = [{"key": "mlflow.experiment.sourceName", "value": name}]
    if uc_destination is not None:
        tags.append({"key": databricks.UC_TRACE_DESTINATION_TAG, "value": uc_destination})
    return {"experiment_id": exp_id, "name": name, "tags": tags}


class TestFindUcBackedExperiment:
    def test_returns_uc_backed_match(self):
        payload = {
            "experiments": [
                _experiment("/Users/me@example.com/ucode-traces", "42", "main.default.ucode_traces")
            ]
        }
        with patch.object(databricks, "_http_post_json", return_value=(payload, None)):
            exp, reason = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert reason is None
        assert exp == {
            "experiment_id": "42",
            "experiment_name": "/Users/me@example.com/ucode-traces",
            "uc_destination": "main.default.ucode_traces",
        }

    def test_any_catalog_schema_table_qualifies(self):
        payload = {"experiments": [_experiment("/Shared/ucode-traces", "7", "cat.sch.tbl")]}
        with patch.object(databricks, "_http_post_json", return_value=(payload, None)):
            exp, _ = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert exp["uc_destination"] == "cat.sch.tbl"

    def test_none_when_no_experiment(self):
        with patch.object(databricks, "_http_post_json", return_value=({"experiments": []}, None)):
            exp, reason = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert exp is None
        assert "no experiment named 'ucode-traces'" in reason

    def test_none_when_match_not_uc_backed(self):
        payload = {"experiments": [_experiment("/Shared/ucode-traces", "9", None)]}
        with patch.object(databricks, "_http_post_json", return_value=(payload, None)):
            exp, reason = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert exp is None
        assert "not backed by Unity Catalog" in reason

    def test_rejects_non_three_part_destination(self):
        payload = {"experiments": [_experiment("/Shared/ucode-traces", "9", "main.default")]}
        with patch.object(databricks, "_http_post_json", return_value=(payload, None)):
            exp, reason = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert exp is None
        assert "not backed by Unity Catalog" in reason

    def test_leaf_match_excludes_substring_names(self):
        # "team-ucode-traces" ends with the leaf as a substring but is a
        # different experiment — only an exact final path segment counts.
        payload = {"experiments": [_experiment("/Shared/team-ucode-traces", "1", "c.s.t")]}
        with patch.object(databricks, "_http_post_json", return_value=(payload, None)):
            exp, reason = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert exp is None
        assert "no experiment named 'ucode-traces'" in reason

    def test_prefers_uc_backed_over_plain_duplicate(self):
        payload = {
            "experiments": [
                _experiment("/Users/a@x.com/ucode-traces", "1", None),
                _experiment("/Shared/ucode-traces", "2", "main.default.tbl"),
            ]
        }
        with patch.object(databricks, "_http_post_json", return_value=(payload, None)):
            exp, _ = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert exp["experiment_id"] == "2"

    def test_returns_reason_on_search_failure(self):
        with patch.object(databricks, "_http_post_json", return_value=(None, "HTTP 403 Forbidden")):
            exp, reason = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
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
        assert env == {"MLFLOW_TRACKING_URI": "databricks", "MLFLOW_EXPERIMENT_ID": "111"}

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
        assert env["MLFLOW_EXPERIMENT_ID"] == "111"


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


class TestEnableTracingForWorkspaces:
    """``configure --tracing`` enables tracing for explicit workspaces without
    prompting, and skips workspaces with no tracing-capable agent."""

    def _full(self) -> dict:
        return {
            "current_workspace": "https://a.databricks.com",
            "workspaces": {
                "https://a.databricks.com": {"available_tools": ["claude"], "profile": "pa"},
                # no tracing-capable agent → skipped, not an error
                "https://b.databricks.com": {"available_tools": ["gemini"], "profile": "pb"},
            },
        }

    def test_enables_without_prompting(self):
        enabled: list[str] = []
        with (
            patch.object(tracing, "load_full_state", return_value=self._full()),
            patch.object(tracing, "prompt_for_workspace") as prompt,
            patch.object(tracing, "set_current_workspace"),
            patch.object(
                tracing,
                "_enable_tracing_for_state",
                side_effect=lambda s: enabled.append(s["workspace"]) or s,
            ),
        ):
            rc = tracing.configure_tracing_command(workspaces=[("https://a.databricks.com", None)])
        prompt.assert_not_called()
        assert rc == 0
        assert enabled == ["https://a.databricks.com"]

    def test_skips_workspace_without_tracing_agent(self):
        enabled: list[str] = []
        with (
            patch.object(tracing, "load_full_state", return_value=self._full()),
            patch.object(tracing, "set_current_workspace"),
            patch.object(
                tracing,
                "_enable_tracing_for_state",
                side_effect=lambda s: enabled.append(s["workspace"]) or s,
            ),
        ):
            rc = tracing.configure_tracing_command(workspaces=[("https://b.databricks.com", None)])
        assert enabled == []
        assert rc == 1


class TestInstallAgentTracingDeps:
    """Deps install only for agents configured on the workspace, even though the
    shared experiment means ``agent_tracing`` would otherwise resolve for all."""

    def test_only_configured_agents_get_deps(self):
        # Codex is configured here; claude is not, so its runtime must be skipped.
        state = {**_enabled_state(), "available_tools": ["codex"]}
        with (
            patch("ucode.agents.claude.ensure_tracing_runtime") as claude_dep,
            patch("ucode.agents.codex.ensure_tracing_dependency") as codex_dep,
        ):
            tracing._install_agent_tracing_deps(state)
        claude_dep.assert_not_called()
        codex_dep.assert_called_once()


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
