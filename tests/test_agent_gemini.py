"""Tests for agents/gemini.py."""

from __future__ import annotations

import json

from coding_tool_gateway.agents import gemini

WS = "https://example.databricks.com"


class TestGeminiSpec:
    def test_binary(self):
        assert gemini.SPEC["binary"] == "gemini"

    def test_package(self):
        assert gemini.SPEC["package"] == "@google/gemini-cli"

    def test_display(self):
        assert gemini.SPEC["display"] == "Gemini CLI"


class TestRenderEnvOverlay:
    def test_sets_gemini_model(self):
        env = gemini.render_env_overlay(WS, "gemini-2.0-flash", "tok123")
        assert env["GEMINI_MODEL"] == "gemini-2.0-flash"

    def test_sets_base_url(self):
        env = gemini.render_env_overlay(WS, "gemini-2.0-flash", "tok123")
        assert env["GOOGLE_GEMINI_BASE_URL"] == f"{WS}/ai-gateway/gemini"

    def test_sets_api_key(self):
        env = gemini.render_env_overlay(WS, "gemini-2.0-flash", "tok123")
        assert env["GEMINI_API_KEY"] == "tok123"

    def test_sets_oauth_token_for_mcp(self):
        env = gemini.render_env_overlay(WS, "gemini-2.0-flash", "tok123")
        assert env["OAUTH_TOKEN"] == "tok123"

    def test_sets_bearer_auth_mechanism(self):
        env = gemini.render_env_overlay(WS, "gemini-2.0-flash", "tok123")
        assert env["GEMINI_API_KEY_AUTH_MECHANISM"] == "bearer"


class TestBuildRuntimeEnv:
    def test_merges_os_environment(self):
        env = gemini.build_runtime_env(WS, "gemini-2", "tok")
        assert "PATH" in env

    def test_overrides_gemini_vars(self):
        env = gemini.build_runtime_env(WS, "gemini-2.0-flash", "mytoken")
        assert env["GEMINI_MODEL"] == "gemini-2.0-flash"
        assert env["GEMINI_API_KEY"] == "mytoken"
        assert env["GEMINI_API_KEY_AUTH_MECHANISM"] == "bearer"

    def test_sets_base_url(self):
        env = gemini.build_runtime_env(WS, "gemini-2", "tok")
        assert env["GOOGLE_GEMINI_BASE_URL"] == f"{WS}/ai-gateway/gemini"

    def test_sets_oauth_token_for_mcp(self):
        env = gemini.build_runtime_env(WS, "gemini-2", "tok")
        assert env["OAUTH_TOKEN"] == "tok"


class TestGeminiDefaultModel:
    def test_returns_first_model(self):
        state = {"gemini_models": ["gemini-2", "gemini-1"]}
        assert gemini.default_model(state) == "gemini-2"

    def test_returns_none_when_empty_list(self):
        assert gemini.default_model({"gemini_models": []}) is None

    def test_returns_none_when_missing(self):
        assert gemini.default_model({}) is None


class TestGeminiValidateCmd:
    def test_starts_with_binary(self):
        cmd = gemini.validate_cmd("gemini")
        assert cmd[0] == "gemini"

    def test_has_p_flag(self):
        cmd = gemini.validate_cmd("gemini")
        assert "-p" in cmd

    def test_has_prompt_text(self):
        cmd = gemini.validate_cmd("gemini")
        assert len(cmd) > 2


class TestGeminiManagedKeys:
    def test_managed_keys_not_empty(self):
        assert len(gemini.MANAGED_KEYS) > 0

    def test_managed_keys_includes_model(self):
        assert "GEMINI_MODEL" in gemini.MANAGED_KEYS

    def test_managed_keys_includes_api_key(self):
        assert "GEMINI_API_KEY" in gemini.MANAGED_KEYS

    def test_managed_keys_includes_oauth_token(self):
        assert "OAUTH_TOKEN" in gemini.MANAGED_KEYS


class TestWriteToolConfigSetsAuthType:
    def test_writes_settings_json_with_gemini_api_key_auth(self, tmp_path, monkeypatch):
        import coding_tool_gateway.config_io as config_io_mod

        settings_path = tmp_path / "settings.json"
        monkeypatch.setattr(gemini, "GEMINI_ENV_PATH", tmp_path / ".env")
        monkeypatch.setattr(gemini, "GEMINI_SETTINGS_PATH", settings_path)
        monkeypatch.setattr(gemini, "GEMINI_BACKUP_PATH", tmp_path / "backup")
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr("coding_tool_gateway.state.save_state", lambda s: None)
        monkeypatch.setattr(
            "coding_tool_gateway.agents.gemini.get_databricks_token", lambda ws: "fake-token"
        )

        gemini.write_tool_config({"workspace": WS}, "some-model")

        settings = json.loads(settings_path.read_text())
        assert settings["security"]["auth"]["selectedType"] == "gemini-api-key"

    def test_preserves_existing_settings_json_keys(self, tmp_path, monkeypatch):
        import coding_tool_gateway.config_io as config_io_mod

        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps({"theme": "dark", "otherKey": 123}))
        monkeypatch.setattr(gemini, "GEMINI_ENV_PATH", tmp_path / ".env")
        monkeypatch.setattr(gemini, "GEMINI_SETTINGS_PATH", settings_path)
        monkeypatch.setattr(gemini, "GEMINI_BACKUP_PATH", tmp_path / "backup")
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr("coding_tool_gateway.state.save_state", lambda s: None)
        monkeypatch.setattr(
            "coding_tool_gateway.agents.gemini.get_databricks_token", lambda ws: "fake-token"
        )

        gemini.write_tool_config({"workspace": WS}, "some-model")

        settings = json.loads(settings_path.read_text())
        assert settings["theme"] == "dark"
        assert settings["otherKey"] == 123
        assert settings["security"]["auth"]["selectedType"] == "gemini-api-key"
