"""Tests for agents/goose.py."""

from __future__ import annotations

from ucode.agents import goose

WS = "https://example.databricks.com"


class TestGooseSpec:
    def test_binary(self):
        assert goose.SPEC["binary"] == "goose"

    def test_package_is_empty(self):
        # Goose is a native binary, not an npm package.
        assert goose.SPEC["package"] == ""

    def test_display(self):
        assert goose.SPEC["display"] == "Goose"

    def test_config_path_is_yaml(self):
        assert goose.SPEC["config_path"].name == "config.yaml"

    def test_config_path_under_dot_config_goose(self):
        assert ".config/goose" in str(goose.SPEC["config_path"])


class TestDefaultModel:
    def test_prefers_claude_sonnet(self):
        state = {"claude_models": {"sonnet": "s4", "opus": "o4", "haiku": "h4"}}
        assert goose.default_model(state) == "s4"

    def test_falls_back_to_opus(self):
        state = {"claude_models": {"opus": "o4", "haiku": "h4"}}
        assert goose.default_model(state) == "o4"

    def test_falls_back_to_haiku(self):
        state = {"claude_models": {"haiku": "h4"}}
        assert goose.default_model(state) == "h4"

    def test_returns_none_when_no_claude_models(self):
        assert goose.default_model({}) is None

    def test_falls_back_to_gemini(self):
        state = {"gemini_models": ["gemini-pro"]}
        assert goose.default_model(state) == "gemini-pro"

    def test_ignores_codex_models(self):
        state = {"codex_models": ["gpt-5"]}
        assert goose.default_model(state) is None


class TestRenderOverlay:
    def test_sets_databricks_host(self):
        overlay = goose.render_overlay(WS, "databricks-claude-sonnet-4-6")
        assert overlay["DATABRICKS_HOST"] == WS

    def test_sets_goose_provider(self):
        overlay = goose.render_overlay(WS, "databricks-claude-sonnet-4-6")
        assert overlay["GOOSE_PROVIDER"] == "databricks"

    def test_sets_goose_model(self):
        overlay = goose.render_overlay(WS, "databricks-claude-sonnet-4-6")
        assert overlay["GOOSE_MODEL"] == "databricks-claude-sonnet-4-6"

    def test_contains_top_level_managed_keys(self):
        overlay = goose.render_overlay(WS, "m")
        assert {"DATABRICKS_HOST", "GOOSE_PROVIDER", "GOOSE_MODEL", "extensions"} <= set(overlay)

    def test_enables_skills_extension(self):
        overlay = goose.render_overlay(WS, "m")
        assert overlay["extensions"]["skills"]["enabled"] is True

    def test_skills_is_platform_type(self):
        overlay = goose.render_overlay(WS, "m")
        assert overlay["extensions"]["skills"]["type"] == "platform"


class TestBuildRuntimeEnv:
    def test_inherits_path(self):
        env = goose.build_runtime_env(WS, "tok")
        assert "PATH" in env

    def test_sets_databricks_host(self):
        env = goose.build_runtime_env(WS, "tok")
        assert env["DATABRICKS_HOST"] == WS

    def test_sets_databricks_token(self):
        env = goose.build_runtime_env(WS, "tok123")
        assert env["DATABRICKS_TOKEN"] == "tok123"

    def test_sets_oauth_token(self):
        env = goose.build_runtime_env(WS, "tok123")
        assert env["OAUTH_TOKEN"] == "tok123"


class TestIsUpdateAvailable:
    def test_returns_none(self):
        assert goose.is_update_available() is None


class TestManagedKeys:
    def test_includes_databricks_host(self):
        assert "DATABRICKS_HOST" in goose.MANAGED_KEYS

    def test_includes_goose_provider(self):
        assert "GOOSE_PROVIDER" in goose.MANAGED_KEYS

    def test_includes_goose_model(self):
        assert "GOOSE_MODEL" in goose.MANAGED_KEYS

    def test_includes_oauth_token(self):
        assert "OAUTH_TOKEN" in goose.MANAGED_KEYS


class TestValidateCmd:
    def test_starts_with_binary(self):
        cmd = goose.validate_cmd("goose")
        assert cmd[0] == "goose"

    def test_uses_run_subcommand(self):
        cmd = goose.validate_cmd("goose")
        assert cmd[1] == "run"

    def test_has_text_flag(self):
        cmd = goose.validate_cmd("goose")
        assert "--text" in cmd

    def test_text_prompt_is_non_empty(self):
        cmd = goose.validate_cmd("goose")
        idx = cmd.index("--text")
        assert cmd[idx + 1].strip()

    def test_has_no_session_flag(self):
        cmd = goose.validate_cmd("goose")
        assert "--no-session" in cmd

    def test_has_max_turns_1(self):
        cmd = goose.validate_cmd("goose")
        idx = cmd.index("--max-turns")
        assert cmd[idx + 1] == "1"


class TestValidateEnv:
    def test_raises_when_no_workspace(self):
        import pytest

        with pytest.raises(RuntimeError, match="No workspace"):
            goose.validate_env({})

    def test_raises_when_no_models(self):
        import pytest

        with pytest.raises(RuntimeError, match="No Goose model"):
            goose.validate_env({"workspace": WS})

    def test_returns_env_with_token(self, monkeypatch):
        import ucode.agents.goose as goose_mod

        monkeypatch.setattr(goose_mod, "get_databricks_token", lambda ws: "tok-from-cli")
        state = {"workspace": WS, "claude_models": {"sonnet": "databricks-claude-sonnet-4-6"}}
        env = goose.validate_env(state)
        assert env["DATABRICKS_TOKEN"] == "tok-from-cli"
        assert env["DATABRICKS_HOST"] == WS
        assert env["OAUTH_TOKEN"] == "tok-from-cli"


class TestBuildMcpServerEntry:
    def test_is_streamable_http(self):
        entry = goose.build_mcp_server_entry("databricks-sql", f"{WS}/api/2.0/mcp/sql")
        assert entry["type"] == "streamable_http"

    def test_uses_url_as_uri(self):
        url = f"{WS}/api/2.0/mcp/sql"
        entry = goose.build_mcp_server_entry("databricks-sql", url)
        assert entry["uri"] == url

    def test_is_enabled(self):
        entry = goose.build_mcp_server_entry("databricks-sql", f"{WS}/api/2.0/mcp/sql")
        assert entry["enabled"] is True

    def test_has_oauth_token_auth_header(self):
        entry = goose.build_mcp_server_entry("databricks-sql", f"{WS}/api/2.0/mcp/sql")
        assert "OAUTH_TOKEN" in entry["headers"]["Authorization"]

    def test_token_stored_in_envs(self):
        entry = goose.build_mcp_server_entry("databricks-sql", f"{WS}/api/2.0/mcp/sql", "tok123")
        assert entry["envs"]["OAUTH_TOKEN"] == "tok123"

    def test_env_keys_is_empty(self):
        entry = goose.build_mcp_server_entry("databricks-sql", f"{WS}/api/2.0/mcp/sql")
        assert entry["env_keys"] == []


class TestMcpSlug:
    def test_lowercases(self):
        assert goose._mcp_slug("GitHub-MCP") == "github_mcp"

    def test_replaces_dashes_with_underscores(self):
        assert goose._mcp_slug("databricks-sql") == "databricks_sql"


class TestWriteMcpServerConfig:
    def test_writes_extension_to_config(self, tmp_path, monkeypatch):
        import yaml

        import ucode.agents.goose as goose_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_path = tmp_path / "config.yaml"
        backup_path = tmp_path / "goose-backup.yaml"
        monkeypatch.setattr(goose_mod, "GOOSE_CONFIG_PATH", config_path)
        monkeypatch.setattr(goose_mod, "GOOSE_BACKUP_PATH", backup_path)

        goose_mod.write_mcp_server_config("databricks-sql", f"{WS}/api/2.0/mcp/sql")

        written = yaml.safe_load(config_path.read_text())
        assert "databricks_sql" in written["extensions"]
        assert written["extensions"]["databricks_sql"]["uri"] == f"{WS}/api/2.0/mcp/sql"
        assert written["extensions"]["databricks_sql"]["type"] == "streamable_http"

    def test_returns_false_when_new_entry(self, tmp_path, monkeypatch):
        import ucode.agents.goose as goose_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_path = tmp_path / "config.yaml"
        backup_path = tmp_path / "goose-backup.yaml"
        monkeypatch.setattr(goose_mod, "GOOSE_CONFIG_PATH", config_path)
        monkeypatch.setattr(goose_mod, "GOOSE_BACKUP_PATH", backup_path)

        removed = goose_mod.write_mcp_server_config("databricks-sql", f"{WS}/api/2.0/mcp/sql")
        assert removed is False

    def test_returns_true_when_replacing_existing(self, tmp_path, monkeypatch):
        import ucode.agents.goose as goose_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_path = tmp_path / "config.yaml"
        backup_path = tmp_path / "goose-backup.yaml"
        monkeypatch.setattr(goose_mod, "GOOSE_CONFIG_PATH", config_path)
        monkeypatch.setattr(goose_mod, "GOOSE_BACKUP_PATH", backup_path)

        goose_mod.write_mcp_server_config("databricks-sql", f"{WS}/api/2.0/mcp/sql")
        removed = goose_mod.write_mcp_server_config("databricks-sql", f"{WS}/api/2.0/mcp/sql")
        assert removed is True

    def test_preserves_existing_extensions(self, tmp_path, monkeypatch):
        import yaml

        import ucode.agents.goose as goose_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_path = tmp_path / "config.yaml"
        backup_path = tmp_path / "goose-backup.yaml"
        monkeypatch.setattr(goose_mod, "GOOSE_CONFIG_PATH", config_path)
        monkeypatch.setattr(goose_mod, "GOOSE_BACKUP_PATH", backup_path)

        config_path.write_text(
            yaml.dump({"extensions": {"developer": {"enabled": True, "type": "builtin"}}}),
            encoding="utf-8",
        )
        goose_mod.write_mcp_server_config("databricks-sql", f"{WS}/api/2.0/mcp/sql")

        written = yaml.safe_load(config_path.read_text())
        assert written["extensions"]["developer"]["enabled"] is True
        assert "databricks_sql" in written["extensions"]


class TestRemoveMcpServerConfig:
    def test_removes_extension(self, tmp_path, monkeypatch):
        import yaml

        import ucode.agents.goose as goose_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_path = tmp_path / "config.yaml"
        backup_path = tmp_path / "goose-backup.yaml"
        monkeypatch.setattr(goose_mod, "GOOSE_CONFIG_PATH", config_path)
        monkeypatch.setattr(goose_mod, "GOOSE_BACKUP_PATH", backup_path)

        goose_mod.write_mcp_server_config("databricks-sql", f"{WS}/api/2.0/mcp/sql")
        result = goose_mod.remove_mcp_server_config("databricks-sql")

        assert result is True
        written = yaml.safe_load(config_path.read_text())
        assert "databricks_sql" not in (written.get("extensions") or {})

    def test_returns_false_when_not_present(self, tmp_path, monkeypatch):
        import ucode.agents.goose as goose_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_path = tmp_path / "config.yaml"
        backup_path = tmp_path / "goose-backup.yaml"
        monkeypatch.setattr(goose_mod, "GOOSE_CONFIG_PATH", config_path)
        monkeypatch.setattr(goose_mod, "GOOSE_BACKUP_PATH", backup_path)

        result = goose_mod.remove_mcp_server_config("nonexistent")
        assert result is False

    def test_preserves_other_extensions(self, tmp_path, monkeypatch):
        import yaml

        import ucode.agents.goose as goose_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_path = tmp_path / "config.yaml"
        backup_path = tmp_path / "goose-backup.yaml"
        monkeypatch.setattr(goose_mod, "GOOSE_CONFIG_PATH", config_path)
        monkeypatch.setattr(goose_mod, "GOOSE_BACKUP_PATH", backup_path)

        config_path.write_text(
            yaml.dump(
                {
                    "extensions": {
                        "databricks_sql": {"type": "streamable_http", "enabled": True},
                        "developer": {"type": "builtin", "enabled": True},
                    }
                }
            ),
            encoding="utf-8",
        )
        goose_mod.remove_mcp_server_config("databricks-sql")

        written = yaml.safe_load(config_path.read_text())
        assert written["extensions"]["developer"]["enabled"] is True
        assert "databricks_sql" not in written["extensions"]


class TestWriteToolConfig:
    def test_writes_yaml_config(self, tmp_path, monkeypatch):
        import yaml

        import ucode.agents.goose as goose_mod
        import ucode.config_io as config_io_mod
        import ucode.state as state_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_path = tmp_path / "config.yaml"
        backup_path = tmp_path / "goose-backup.yaml"
        monkeypatch.setattr(goose_mod, "GOOSE_CONFIG_PATH", config_path)
        monkeypatch.setattr(goose_mod, "GOOSE_BACKUP_PATH", backup_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        monkeypatch.setattr(goose_mod, "get_databricks_token", lambda *a, **kw: "test-token")

        state = {"workspace": WS, "claude_models": {"sonnet": "databricks-claude-sonnet-4-6"}}
        returned_state, token = goose_mod.write_tool_config(state, "databricks-claude-sonnet-4-6")

        assert config_path.exists()
        written = yaml.safe_load(config_path.read_text())
        assert written["DATABRICKS_HOST"] == WS
        assert written["GOOSE_PROVIDER"] == "databricks"
        assert written["GOOSE_MODEL"] == "databricks-claude-sonnet-4-6"
        assert token == "test-token"
        assert "goose" in (returned_state.get("managed_configs") or {})
        assert written["extensions"]["skills"]["enabled"] is True

    def test_uses_explicit_token_when_provided(self, tmp_path, monkeypatch):
        import yaml

        import ucode.agents.goose as goose_mod
        import ucode.config_io as config_io_mod
        import ucode.state as state_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_path = tmp_path / "config.yaml"
        backup_path = tmp_path / "goose-backup.yaml"
        monkeypatch.setattr(goose_mod, "GOOSE_CONFIG_PATH", config_path)
        monkeypatch.setattr(goose_mod, "GOOSE_BACKUP_PATH", backup_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        # get_databricks_token should NOT be called when token is passed explicitly
        monkeypatch.setattr(
            goose_mod,
            "get_databricks_token",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("should not call get_databricks_token")
            ),
        )

        state = {"workspace": WS, "claude_models": {"sonnet": "databricks-claude-sonnet-4-6"}}
        _, token = goose_mod.write_tool_config(
            state, "databricks-claude-sonnet-4-6", token="explicit-tok"
        )

        assert token == "explicit-tok"
        written = yaml.safe_load(config_path.read_text())
        assert written["GOOSE_MODEL"] == "databricks-claude-sonnet-4-6"

    def test_updates_model_on_reconfigure(self, tmp_path, monkeypatch):
        import yaml

        import ucode.agents.goose as goose_mod
        import ucode.config_io as config_io_mod
        import ucode.state as state_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_path = tmp_path / "config.yaml"
        backup_path = tmp_path / "goose-backup.yaml"
        monkeypatch.setattr(goose_mod, "GOOSE_CONFIG_PATH", config_path)
        monkeypatch.setattr(goose_mod, "GOOSE_BACKUP_PATH", backup_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        monkeypatch.setattr(goose_mod, "get_databricks_token", lambda *a, **kw: "tok")

        state = {"workspace": WS, "claude_models": {"sonnet": "databricks-claude-sonnet-4-6"}}
        goose_mod.write_tool_config(state, "databricks-claude-sonnet-4-6")

        # Reconfigure with a different model (e.g., workspace now has a newer endpoint).
        goose_mod.write_tool_config(state, "databricks-claude-opus-4-7")

        written = yaml.safe_load(config_path.read_text())
        assert written["GOOSE_MODEL"] == "databricks-claude-opus-4-7"

    def test_preserves_existing_config_keys(self, tmp_path, monkeypatch):
        import yaml

        import ucode.agents.goose as goose_mod
        import ucode.config_io as config_io_mod
        import ucode.state as state_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_path = tmp_path / "config.yaml"
        backup_path = tmp_path / "goose-backup.yaml"
        monkeypatch.setattr(goose_mod, "GOOSE_CONFIG_PATH", config_path)
        monkeypatch.setattr(goose_mod, "GOOSE_BACKUP_PATH", backup_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        monkeypatch.setattr(goose_mod, "get_databricks_token", lambda *a, **kw: "tok")

        # Pre-populate with user settings that should be preserved.
        config_path.write_text(
            yaml.dump({"GOOSE_MAX_TOKENS": 16000, "extensions": {"developer": {"enabled": True}}}),
            encoding="utf-8",
        )

        state = {"workspace": WS, "claude_models": {"sonnet": "databricks-claude-sonnet-4-6"}}
        goose_mod.write_tool_config(state, "databricks-claude-sonnet-4-6")

        written = yaml.safe_load(config_path.read_text())
        assert written["GOOSE_MAX_TOKENS"] == 16000
        assert written["extensions"]["developer"]["enabled"] is True
        assert written["GOOSE_PROVIDER"] == "databricks"

    def test_refreshes_token_in_streamable_http_extension_envs(self, tmp_path, monkeypatch):
        import yaml

        import ucode.agents.goose as goose_mod
        import ucode.config_io as config_io_mod
        import ucode.state as state_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_path = tmp_path / "config.yaml"
        backup_path = tmp_path / "goose-backup.yaml"
        monkeypatch.setattr(goose_mod, "GOOSE_CONFIG_PATH", config_path)
        monkeypatch.setattr(goose_mod, "GOOSE_BACKUP_PATH", backup_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        monkeypatch.setattr(goose_mod, "get_databricks_token", lambda *a, **kw: "new-token")

        config_path.write_text(
            yaml.dump(
                {
                    "extensions": {
                        "my_mcp": {
                            "type": "streamable_http",
                            "envs": {"OAUTH_TOKEN": "old-token"},
                            "headers": {"Authorization": "Bearer ${OAUTH_TOKEN}"},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        state = {"workspace": WS, "claude_models": {"sonnet": "databricks-claude-sonnet-4-6"}}
        goose_mod.write_tool_config(state, "databricks-claude-sonnet-4-6")

        written = yaml.safe_load(config_path.read_text())
        assert written["extensions"]["my_mcp"]["envs"]["OAUTH_TOKEN"] == "new-token"
