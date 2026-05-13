"""Tests for agents/codex.py."""

from __future__ import annotations

import os

from ucode.agents import codex

WS = "https://example.databricks.com"


class TestCodexSpec:
    def test_binary(self):
        assert codex.SPEC["binary"] == "codex"

    def test_package(self):
        assert codex.SPEC["package"] == "@openai/codex"

    def test_display(self):
        assert codex.SPEC["display"] == "Codex"


class TestRenderOverlay:
    def test_sets_default_profile(self):
        overlay = codex.render_overlay(WS)
        assert overlay["profile"] == "default"

    def test_sets_model_provider(self):
        overlay = codex.render_overlay(WS)
        assert overlay["profiles"]["default"]["model_provider"] == "Databricks"

    def test_provider_base_url(self):
        overlay = codex.render_overlay(WS)
        provider = overlay["model_providers"]["Databricks"]
        assert provider["base_url"] == f"{WS}/ai-gateway/codex/v1"

    def test_provider_wire_api(self):
        overlay = codex.render_overlay(WS)
        provider = overlay["model_providers"]["Databricks"]
        assert provider["wire_api"] == "responses"

    def test_auth_uses_sh(self):
        overlay = codex.render_overlay(WS)
        auth = overlay["model_providers"]["Databricks"]["auth"]
        assert auth["command"] == "sh"
        assert "-c" in auth["args"]

    def test_auth_contains_workspace(self):
        overlay = codex.render_overlay(WS)
        auth = overlay["model_providers"]["Databricks"]["auth"]
        assert any(WS in arg for arg in auth["args"])

    def test_auth_refresh_interval(self):
        overlay = codex.render_overlay(WS)
        auth = overlay["model_providers"]["Databricks"]["auth"]
        assert auth["refresh_interval_ms"] == 900_000


class TestCodexDefaultModel:
    def test_always_none(self):
        assert codex.default_model({}) is None
        assert codex.default_model({"codex_models": ["gpt-4o"]}) is None


class TestCodexValidateCmd:
    def test_starts_with_binary(self):
        cmd = codex.validate_cmd("codex")
        assert cmd[0] == "codex"

    def test_uses_exec_subcommand(self):
        cmd = codex.validate_cmd("codex")
        assert "exec" in cmd

    def test_has_prompt(self):
        cmd = codex.validate_cmd("codex")
        assert len(cmd) > 2

    def test_skips_git_repo_check(self):
        # Validation runs in arbitrary cwd (e.g., ~/Documents); without this
        # flag Codex refuses to run outside a trusted/git directory.
        cmd = codex.validate_cmd("codex")
        assert "--skip-git-repo-check" in cmd


class TestCodexLaunch:
    def test_sets_oauth_token_before_exec(self, monkeypatch):
        exec_calls: list[tuple[str, list[str]]] = []

        def fake_execvp(binary: str, args: list[str]) -> None:
            exec_calls.append((binary, args))
            raise RuntimeError("stop")

        monkeypatch.delenv("OAUTH_TOKEN", raising=False)
        monkeypatch.setattr(codex, "get_databricks_token", lambda workspace: "fresh-token")
        monkeypatch.setattr(os, "execvp", fake_execvp)

        try:
            codex.launch({"workspace": WS}, ["--search"])
        except RuntimeError as exc:
            assert str(exc) == "stop"

        assert os.environ["OAUTH_TOKEN"] == "fresh-token"
        assert exec_calls == [("codex", ["codex", "--search"])]
