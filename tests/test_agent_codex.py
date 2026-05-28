"""Tests for agents/codex.py."""

from __future__ import annotations

import os

from ucode.agents import codex
from ucode.config_io import read_toml_safe

WS = "https://example.databricks.com"


class TestCodexSpec:
    def test_binary(self):
        assert codex.SPEC["binary"] == "codex"

    def test_package(self):
        assert codex.SPEC["package"] == "@openai/codex"

    def test_display(self):
        assert codex.SPEC["display"] == "Codex"


class TestRenderOverlay:
    def test_uses_profile_file_shape_without_legacy_profiles(self):
        overlay = codex.render_overlay(WS)
        assert "profile" not in overlay
        assert "profiles" not in overlay

    def test_sets_model_provider(self):
        overlay = codex.render_overlay(WS)
        assert overlay["model_provider"] == "ucode-databricks"

    def test_sets_model_when_provided(self):
        overlay = codex.render_overlay(WS, "databricks-gpt-5")
        assert overlay["model"] == "databricks-gpt-5"

    def test_provider_base_url(self):
        overlay = codex.render_overlay(WS)
        provider = overlay["model_providers"]["ucode-databricks"]
        assert provider["base_url"] == f"{WS}/ai-gateway/codex/v1"

    def test_provider_wire_api(self):
        overlay = codex.render_overlay(WS)
        provider = overlay["model_providers"]["ucode-databricks"]
        assert provider["wire_api"] == "responses"

    def test_auth_uses_sh(self):
        overlay = codex.render_overlay(WS)
        auth = overlay["model_providers"]["ucode-databricks"]["auth"]
        assert auth["command"] == "sh"
        assert "-c" in auth["args"]

    def test_auth_contains_workspace(self):
        overlay = codex.render_overlay(WS)
        auth = overlay["model_providers"]["ucode-databricks"]["auth"]
        assert any(WS in arg for arg in auth["args"])

    def test_auth_refresh_interval(self):
        overlay = codex.render_overlay(WS)
        auth = overlay["model_providers"]["ucode-databricks"]["auth"]
        assert auth["refresh_interval_ms"] == 900_000


class TestRenderOverlayUserAgent:
    def test_user_agent_set_on_provider(self, monkeypatch):
        monkeypatch.setattr(codex, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.123.0")
        overlay = codex.render_overlay(WS)
        provider = overlay["model_providers"]["ucode-databricks"]
        assert provider["http_headers"]["User-Agent"] == "ucode/0.1.0 codex/0.123.0"

    def test_managed_keys_include_http_headers(self):
        # Revert must clean up the new key.
        assert ["model_providers", "ucode-databricks", "http_headers"] in codex.MANAGED_KEYS


class TestCodexWriteConfig:
    def test_writes_ucode_profile_config_file(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config({"workspace": WS, "codex_models": ["gpt-5"]})

        doc = read_toml_safe(config_path)
        assert doc["model_provider"] == "ucode-databricks"
        assert doc["model"] == "gpt-5"
        assert "profiles" not in doc

    def test_removes_legacy_ucode_profile_from_shared_config(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        profile_path = config_dir / "ucode.config.toml"
        legacy_path = config_dir / "config.toml"
        legacy_path.write_text(
            'profile = "ucode"\n\n'
            "[profiles.ucode]\n"
            'model_provider = "old"\n\n'
            "[profiles.other]\n"
            'model_provider = "keep"\n',
            encoding="utf-8",
        )
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        legacy_backup_path = tmp_path / "codex-legacy-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config({"workspace": WS, "codex_models": ["gpt-5"]})

        doc = read_toml_safe(legacy_path)
        assert "profile" not in doc
        assert "ucode" not in doc["profiles"]
        assert doc["profiles"]["other"]["model_provider"] == "keep"
        assert legacy_backup_path.exists()


class TestCodexMinimumVersion:
    def test_no_error_when_codex_is_new_enough(self, monkeypatch):
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")

        assert codex.minimum_version_error() is None
        assert codex.required_update_message() is None

    def test_errors_when_codex_is_too_old(self, monkeypatch):
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.133.0")

        assert "Codex CLI must be updated to 0.134.0 or newer" in codex.minimum_version_error()
        assert "updating Codex is required" in codex.required_update_message()

    def test_unknown_version_does_not_block(self, monkeypatch):
        monkeypatch.setattr(codex, "agent_version", lambda binary: "unknown")

        assert codex.minimum_version_error() is None
        assert codex.required_update_message() is None


class TestCodexDefaultModel:
    def test_returns_first_codex_model(self):
        assert codex.default_model({"codex_models": ["gpt-5", "gpt-4o"]}) == "gpt-5"

    def test_none_when_no_models(self):
        assert codex.default_model({}) is None


class TestCodexValidateCmd:
    def test_starts_with_binary(self):
        cmd = codex.validate_cmd("codex")
        assert cmd[0] == "codex"

    def test_uses_exec_subcommand(self):
        cmd = codex.validate_cmd("codex")
        assert "exec" in cmd

    def test_uses_ucode_profile(self):
        cmd = codex.validate_cmd("codex")
        assert cmd[:3] == ["codex", "--profile", "ucode"]

    def test_has_prompt(self):
        cmd = codex.validate_cmd("codex")
        assert len(cmd) > 2

    def test_skips_git_repo_check(self):
        # Validation runs in arbitrary cwd (e.g., ~/Documents); without this
        # flag Codex refuses to run outside a trusted/git directory.
        cmd = codex.validate_cmd("codex")
        assert "--skip-git-repo-check" in cmd


class TestCodexLaunch:
    def test_sets_oauth_token_and_ucode_profile_before_exec(self, monkeypatch):
        exec_calls: list[tuple[str, list[str]]] = []

        def fake_execvp(binary: str, args: list[str]) -> None:
            exec_calls.append((binary, args))
            raise RuntimeError("stop")

        monkeypatch.delenv("OAUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            codex, "get_databricks_token", lambda workspace, profile=None: "fresh-token"
        )
        monkeypatch.setattr(os, "execvp", fake_execvp)

        try:
            codex.launch({"workspace": WS}, ["--search"])
        except RuntimeError as exc:
            assert str(exc) == "stop"

        assert os.environ["OAUTH_TOKEN"] == "fresh-token"
        assert exec_calls == [("codex", ["codex", "--profile", "ucode", "--search"])]
