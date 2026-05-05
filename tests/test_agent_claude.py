"""Tests for agents/claude.py."""

from __future__ import annotations

import os

from coding_tool_gateway.agents import claude

WS = "https://example.databricks.com"


class TestClaudeSpec:
    def test_binary(self):
        assert claude.SPEC["binary"] == "claude"

    def test_package(self):
        assert claude.SPEC["package"] == "@anthropic-ai/claude-code"

    def test_display(self):
        assert claude.SPEC["display"] == "Claude Code"


class TestRenderOverlay:
    def test_sets_anthropic_model(self):
        overlay, _ = claude.render_overlay(WS, "databricks-claude-sonnet-4")
        assert overlay["env"]["ANTHROPIC_MODEL"] == "databricks-claude-sonnet-4"

    def test_sets_anthropic_base_url(self):
        overlay, _ = claude.render_overlay(WS, "s4")
        assert overlay["env"]["ANTHROPIC_BASE_URL"] == f"{WS}/ai-gateway/anthropic"

    def test_sets_custom_headers(self):
        overlay, _ = claude.render_overlay(WS, "s4")
        assert "x-databricks-use-coding-agent-mode" in overlay["env"]["ANTHROPIC_CUSTOM_HEADERS"]

    def test_disables_experimental_betas(self):
        overlay, _ = claude.render_overlay(WS, "s4")
        assert overlay["env"]["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] == "1"

    def test_sets_api_key_helper(self):
        overlay, _ = claude.render_overlay(WS, "s4")
        assert "apiKeyHelper" in overlay
        assert WS in overlay["apiKeyHelper"]

    def test_model_overrides_when_all_provided(self):
        models = {"sonnet": "s4", "opus": "o4", "haiku": "h4"}
        overlay, _ = claude.render_overlay(WS, "s4", claude_models=models)
        env = overlay["env"]
        assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "s4"
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "o4"
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "h4"

    def test_model_overrides_partial(self):
        models = {"sonnet": "s4"}
        overlay, _ = claude.render_overlay(WS, "s4", claude_models=models)
        env = overlay["env"]
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" in env
        assert "ANTHROPIC_DEFAULT_OPUS_MODEL" not in env

    def test_model_overrides_not_set_when_no_models(self):
        overlay, _ = claude.render_overlay(WS, "s4")
        env = overlay["env"]
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" not in env

    def test_managed_keys_include_api_key_helper(self):
        _, keys = claude.render_overlay(WS, "s4")
        assert ["apiKeyHelper"] in keys

    def test_managed_keys_include_env_entries(self):
        _, keys = claude.render_overlay(WS, "s4")
        env_keys = [k for k in keys if len(k) == 2 and k[0] == "env"]
        assert len(env_keys) > 0


class TestClaudeDefaultModel:
    def test_prefers_sonnet(self):
        state = {"claude_models": {"sonnet": "s4", "opus": "o4", "haiku": "h4"}}
        assert claude.default_model(state) == "s4"

    def test_falls_back_to_opus(self):
        state = {"claude_models": {"opus": "o4", "haiku": "h4"}}
        assert claude.default_model(state) == "o4"

    def test_falls_back_to_haiku(self):
        state = {"claude_models": {"haiku": "h4"}}
        assert claude.default_model(state) == "h4"

    def test_returns_none_when_no_models(self):
        assert claude.default_model({}) is None
        assert claude.default_model({"claude_models": {}}) is None


class TestClaudeValidateCmd:
    def test_starts_with_binary(self):
        cmd = claude.validate_cmd("claude")
        assert cmd[0] == "claude"

    def test_has_p_flag(self):
        cmd = claude.validate_cmd("claude")
        assert "-p" in cmd

    def test_has_max_turns(self):
        cmd = claude.validate_cmd("claude")
        assert "--max-turns" in cmd
        idx = cmd.index("--max-turns")
        assert cmd[idx + 1] == "1"


class TestClaudeLaunch:
    def test_sets_oauth_token_before_exec(self, monkeypatch):
        exec_calls: list[tuple[str, list[str]]] = []

        def fake_execvp(binary: str, args: list[str]) -> None:
            exec_calls.append((binary, args))
            raise RuntimeError("stop")

        monkeypatch.delenv("OAUTH_TOKEN", raising=False)
        monkeypatch.setattr(claude, "get_databricks_token", lambda workspace: "fresh-token")
        monkeypatch.setattr(os, "execvp", fake_execvp)

        try:
            claude.launch({"workspace": WS}, ["--debug"])
        except RuntimeError as exc:
            assert str(exc) == "stop"

        assert os.environ["OAUTH_TOKEN"] == "fresh-token"
        assert exec_calls == [("claude", ["claude", "--debug"])]
