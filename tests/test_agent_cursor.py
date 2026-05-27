"""Tests for agents/cursor.py."""

from __future__ import annotations

from pathlib import Path

from ucode.agents import cursor

WS = "https://example.databricks.com"


class TestCursorSpec:
    def test_binary(self):
        assert cursor.SPEC["binary"] == "cursor"

    def test_display(self):
        assert cursor.SPEC["display"] == "Cursor"


class TestRenderConfigPayload:
    def test_base_url(self):
        payload = cursor.render_config_payload(WS, "tok", ["my-endpoint"], "my-endpoint")
        assert payload["openai_base_url"] == f"{WS}/ai-gateway/cursor/v1"
        assert payload["openai_api_key"] == "tok"
        assert payload["models"] == ["my-endpoint"]
        assert payload["default_model"] == "my-endpoint"


class TestCursorDefaultModel:
    def test_returns_first(self):
        assert cursor.default_model({"cursor_models": ["a", "b"]}) == "a"

    def test_none_when_empty(self):
        assert cursor.default_model({}) is None


class TestBuildToolBaseUrl:
    def test_cursor_path(self):
        from ucode.databricks import build_tool_base_url

        assert build_tool_base_url("cursor", WS) == f"{WS}/ai-gateway/cursor/v1"


class TestCursorExecutable:
    def test_prefers_mac_bundle_over_shim(self, monkeypatch, tmp_path):
        app = tmp_path / "Cursor.app"
        bundled_cli = app / "Contents/Resources/app/bin/cursor"
        bundled_cli.parent.mkdir(parents=True)
        bundled_cli.write_text("", encoding="utf-8")
        shim = tmp_path / "bin/cursor"
        shim.parent.mkdir()
        shim.write_text("#!/bin/sh\n", encoding="utf-8")
        shim.chmod(0o755)

        monkeypatch.setattr(cursor, "CURSOR_MAC_APP", app)
        monkeypatch.setattr(cursor, "CURSOR_MAC_CLI", bundled_cli)
        monkeypatch.setattr(cursor, "CURSOR_AGENT_SHIM", shim)
        monkeypatch.setattr(cursor.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cursor.shutil, "which", lambda _: str(shim))

        assert cursor._cursor_executable() == bundled_cli


class TestValidateGateway:
    def test_fails_without_models(self):
        ok, err = cursor.validate_gateway(WS, "tok", [])
        assert not ok
        assert "no Cursor-compatible" in err

    def test_succeeds_when_config_and_models_present(self, tmp_path, monkeypatch):
        import ucode.config_io as config_io_mod

        config_path = tmp_path / "cursor-databricks.json"
        config_path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(cursor, "CURSOR_CONFIG_PATH", config_path)

        ok, err = cursor.validate_gateway(WS, "tok", ["my-endpoint"])
        assert ok
        assert err == ""
