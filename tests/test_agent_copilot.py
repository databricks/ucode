"""Tests for agents/copilot.py."""

from __future__ import annotations

from coding_tool_gateway.agents import copilot

WS = "https://example.databricks.com"


class TestCopilotSpec:
    def test_binary(self):
        assert copilot.SPEC["binary"] == "copilot"

    def test_package(self):
        assert copilot.SPEC["package"] == "@github/copilot"

    def test_display(self):
        assert copilot.SPEC["display"] == "GitHub Copilot CLI"


class TestRenderEnvOverlay:
    def test_provider_type_is_openai(self):
        env = copilot.render_env_overlay(WS, "claude-sonnet-4-6", "tok")
        assert env["COPILOT_PROVIDER_TYPE"] == "openai"

    def test_base_url_points_at_mlflow_gateway(self):
        env = copilot.render_env_overlay(WS, "m", "t")
        assert env["COPILOT_PROVIDER_BASE_URL"] == f"{WS}/ai-gateway/mlflow/v1"

    def test_sets_model(self):
        env = copilot.render_env_overlay(WS, "claude-sonnet-4-6", "tok")
        assert env["COPILOT_MODEL"] == "claude-sonnet-4-6"

    def test_uses_bearer_token_env_var(self):
        env = copilot.render_env_overlay(WS, "m", "tok123")
        assert env["COPILOT_PROVIDER_BEARER_TOKEN"] == "tok123"
        assert "COPILOT_PROVIDER_API_KEY" not in env

    def test_sets_offline_true(self):
        env = copilot.render_env_overlay(WS, "m", "t")
        assert env["COPILOT_OFFLINE"] == "true"


class TestBuildRuntimeEnv:
    def test_inherits_path(self):
        env = copilot.build_runtime_env(WS, "m", "t")
        assert "PATH" in env

    def test_overrides_copilot_vars(self):
        env = copilot.build_runtime_env(WS, "m", "tok")
        assert env["COPILOT_PROVIDER_TYPE"] == "openai"
        assert env["COPILOT_PROVIDER_BEARER_TOKEN"] == "tok"


class TestDefaultModel:
    def test_prefers_claude_sonnet(self):
        state = {
            "claude_models": {"sonnet": "s4", "opus": "o4", "haiku": "h4"},
            "codex_models": ["gpt-5"],
        }
        assert copilot.default_model(state) == "s4"

    def test_falls_back_to_opus(self):
        state = {"claude_models": {"opus": "o4", "haiku": "h4"}}
        assert copilot.default_model(state) == "o4"

    def test_falls_back_to_haiku(self):
        state = {"claude_models": {"haiku": "h4"}}
        assert copilot.default_model(state) == "h4"

    def test_falls_back_to_codex_when_no_claude(self):
        state = {"codex_models": ["gpt-5", "gpt-5-mini"]}
        assert copilot.default_model(state) == "gpt-5"

    def test_returns_none_when_no_models(self):
        assert copilot.default_model({}) is None

    def test_ignores_gemini_models(self):
        # Gemini is excluded — Databricks' Gemini translator rejects copilot's request shape.
        state = {"gemini_models": ["gemini-2-5-pro"]}
        assert copilot.default_model(state) is None


class TestManagedKeys:
    def test_includes_required_vars(self):
        for key in (
            "COPILOT_PROVIDER_TYPE",
            "COPILOT_PROVIDER_BASE_URL",
            "COPILOT_MODEL",
            "COPILOT_PROVIDER_BEARER_TOKEN",
            "COPILOT_OFFLINE",
        ):
            assert key in copilot.MANAGED_KEYS


class TestValidateCmd:
    def test_starts_with_binary(self):
        cmd = copilot.validate_cmd("copilot")
        assert cmd[0] == "copilot"

    def test_has_prompt_flag(self):
        cmd = copilot.validate_cmd("copilot")
        assert "--prompt" in cmd
