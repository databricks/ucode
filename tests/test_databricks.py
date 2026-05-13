"""Tests for databricks.py — pure helpers and URL builders that don't hit the network."""

from __future__ import annotations

import json
import os
import subprocess

import pytest

import ucode.databricks as db_mod
from ucode.databricks import (
    AI_GATEWAY_V2_DOCS_URL,
    build_auth_shell_command,
    build_databricks_cli_env,
    build_opencode_base_urls,
    build_shared_base_urls,
    build_tool_base_url,
    get_databricks_token,
    list_databricks_connections,
    workspace_hostname,
)

WS = "https://example.databricks.com"


class TestWorkspaceHostname:
    def test_extracts_hostname(self):
        assert workspace_hostname(WS) == "example.databricks.com"

    def test_handles_path(self):
        assert (
            workspace_hostname("https://foo.azuredatabricks.net/some/path")
            == "foo.azuredatabricks.net"
        )

    def test_invalid_url_raises(self):
        with pytest.raises((RuntimeError, ValueError)):
            workspace_hostname("")


class TestBuildDatabricksCliEnv:
    def test_sets_databricks_host(self):
        env = build_databricks_cli_env(WS)
        assert env["DATABRICKS_HOST"] == WS

    def test_scrubs_token_vars(self):
        import ucode.databricks as db_mod

        for var in db_mod.SCRUBBED_DATABRICKS_ENV_VARS:
            env = build_databricks_cli_env(WS)
            assert var not in env


class TestBuildToolBaseUrl:
    def test_codex(self):
        url = build_tool_base_url("codex", WS)
        assert url == f"{WS}/ai-gateway/codex/v1"

    def test_claude(self):
        url = build_tool_base_url("claude", WS)
        assert url == f"{WS}/ai-gateway/anthropic"

    def test_gemini(self):
        url = build_tool_base_url("gemini", WS)
        assert url == f"{WS}/ai-gateway/gemini"

    def test_opencode_raises(self):
        with pytest.raises(RuntimeError, match="multiple base URLs"):
            build_tool_base_url("opencode", WS)

    def test_unsupported_tool_raises(self):
        with pytest.raises(RuntimeError, match="Unsupported"):
            build_tool_base_url("unknown", WS)


class TestBuildOpencodeBaseUrls:
    def test_returns_anthropic_and_gemini(self):
        urls = build_opencode_base_urls(WS)
        assert urls["anthropic"] == f"{WS}/ai-gateway/anthropic/v1"
        assert urls["gemini"] == f"{WS}/ai-gateway/gemini/v1beta"


class TestBuildSharedBaseUrls:
    def test_contains_all_tools(self):
        urls = build_shared_base_urls(WS)
        assert "codex" in urls
        assert "claude" in urls
        assert "gemini" in urls
        assert "opencode" in urls

    def test_opencode_is_dict(self):
        urls = build_shared_base_urls(WS)
        assert isinstance(urls["opencode"], dict)

    def test_codex_url_format(self):
        urls = build_shared_base_urls(WS)
        assert urls["codex"] == f"{WS}/ai-gateway/codex/v1"


class TestBuildAuthShellCommand:
    def test_contains_workspace(self):
        cmd = build_auth_shell_command(WS)
        assert WS in cmd

    def test_parses_access_token(self):
        cmd = build_auth_shell_command(WS)
        assert "access_token" in cmd
        assert "grep" in cmd
        assert "sed" in cmd

    def test_unsets_scrubbed_vars(self):
        cmd = build_auth_shell_command(WS)
        assert "DATABRICKS_TOKEN" in cmd

    def test_returns_token_when_auth_succeeds(self, tmp_path):
        # Fake databricks binary that always returns a valid token JSON.
        fake = tmp_path / "databricks"
        fake.write_text(
            '#!/bin/sh\necho \'{"access_token": "good-token", "token_type": "Bearer"}\'\n'
        )
        fake.chmod(0o755)
        cmd = build_auth_shell_command(WS)
        result = subprocess.run(
            ["sh", "-c", cmd],
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        )
        assert result.stdout.strip() == "good-token"

    def test_reauths_and_retries_when_token_empty(self, tmp_path):
        # Fake databricks binary: returns empty token on `auth token`, succeeds on retry
        # after `auth login` is called. Uses a call-count file to track invocations.
        call_count = tmp_path / "calls"
        call_count.write_text("0")
        fake = tmp_path / "databricks"
        fake.write_text(
            "#!/bin/sh\n"
            f"count=$(cat {call_count})\n"
            f"echo $((count + 1)) > {call_count}\n"
            "# auth login is a no-op\n"
            'case "$*" in\n'
            '  *"auth login"*) exit 0 ;;\n'
            "esac\n"
            "# first auth token call returns empty; second returns a real token\n"
            'if [ "$count" -eq 0 ]; then\n'
            '  echo \'{"access_token": "", "token_type": "Bearer"}\'\n'
            "else\n"
            '  echo \'{"access_token": "refreshed-token", "token_type": "Bearer"}\'\n'
            "fi\n"
        )
        fake.chmod(0o755)
        cmd = build_auth_shell_command(WS)
        result = subprocess.run(
            ["sh", "-c", cmd],
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        )
        assert result.stdout.strip() == "refreshed-token"
        assert int(call_count.read_text()) >= 3  # auth token, auth login, auth token


class TestGetDatabricksToken:
    def _fake_databricks(self, tmp_path, script: str) -> dict:
        fake = tmp_path / "databricks"
        fake.write_text(f"#!/bin/sh\n{script}\n")
        fake.chmod(0o755)
        return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}

    def test_returns_token_on_success(self, tmp_path, monkeypatch):
        env = self._fake_databricks(
            tmp_path,
            'echo \'{"access_token": "good-token", "token_type": "Bearer"}\'',
        )
        monkeypatch.setattr("os.environ", env)
        token = get_databricks_token(WS)
        assert token == "good-token"

    def test_reauths_and_retries_when_token_empty(self, tmp_path, monkeypatch):
        call_count = tmp_path / "calls"
        call_count.write_text("0")
        env = self._fake_databricks(
            tmp_path,
            f"count=$(cat {call_count})\n"
            f"echo $((count + 1)) > {call_count}\n"
            'case "$*" in\n'
            '  *"auth login"*) exit 0 ;;\n'
            "esac\n"
            'if [ "$count" -eq 0 ]; then\n'
            '  echo \'{"access_token": "", "token_type": "Bearer"}\'\n'
            "else\n"
            '  echo \'{"access_token": "refreshed-token", "token_type": "Bearer"}\'\n'
            "fi",
        )
        monkeypatch.setattr("os.environ", env)
        token = get_databricks_token(WS)
        assert token == "refreshed-token"

    def test_raises_when_reauth_also_fails(self, tmp_path, monkeypatch):
        env = self._fake_databricks(
            tmp_path,
            'echo \'{"access_token": "", "token_type": "Bearer"}\'',
        )
        monkeypatch.setattr("os.environ", env)
        with pytest.raises(RuntimeError, match="no access token"):
            get_databricks_token(WS)


class TestListDatabricksConnections:
    def test_lists_paginated_connections_with_workspace_env(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            if "--page-token" in args:
                payload = {"connections": [{"name": "jira-mcp", "connection_type": "HTTP"}]}
            else:
                payload = {
                    "connections": [{"name": "confluence-mcp", "connection_type": "HTTP"}],
                    "next_page_token": "next-page",
                }
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload))

        monkeypatch.setattr(db_mod, "run", fake_run)

        assert list_databricks_connections(WS) == [
            {"name": "confluence-mcp", "connection_type": "HTTP"},
            {"name": "jira-mcp", "connection_type": "HTTP"},
        ]
        assert calls[0]["args"] == [
            "databricks",
            "connections",
            "list",
            "--max-results",
            "0",
            "--output",
            "json",
        ]
        assert calls[0]["kwargs"]["env"]["DATABRICKS_HOST"] == WS
        assert calls[1]["args"][-2:] == ["--page-token", "next-page"]

    def test_raises_on_invalid_json(self, monkeypatch):
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="not-json")

        monkeypatch.setattr(db_mod, "run", fake_run)

        with pytest.raises(RuntimeError, match="invalid JSON"):
            list_databricks_connections(WS)


class TestEnsureAiGatewayV2:
    """Test ensure_ai_gateway_v2 without real network calls."""

    def test_raises_on_404(self):
        from unittest.mock import MagicMock, patch
        from urllib.error import HTTPError

        exc = HTTPError(url="", code=404, msg="Not Found", hdrs=MagicMock(), fp=None)
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            from ucode.databricks import ensure_ai_gateway_v2

            with pytest.raises(RuntimeError, match=AI_GATEWAY_V2_DOCS_URL):
                ensure_ai_gateway_v2(WS, "fake-token")

    def test_raises_on_url_error(self):
        from unittest.mock import patch
        from urllib.error import URLError

        with patch(
            "ucode.databricks.urllib_request.urlopen",
            side_effect=URLError("connection refused"),
        ):
            from ucode.databricks import ensure_ai_gateway_v2

            with pytest.raises(RuntimeError, match=AI_GATEWAY_V2_DOCS_URL):
                ensure_ai_gateway_v2(WS, "fake-token")

    def test_succeeds_on_non_404_http_error(self):
        from unittest.mock import MagicMock, patch
        from urllib.error import HTTPError

        # 405 Method Not Allowed → still means v2 is there (method mismatch, not missing route)
        exc = HTTPError(url="", code=405, msg="Method Not Allowed", hdrs=MagicMock(), fp=None)
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            from ucode.databricks import ensure_ai_gateway_v2

            ensure_ai_gateway_v2(WS, "fake-token")  # should not raise

    def test_succeeds_when_urlopen_returns(self):
        from unittest.mock import MagicMock, patch

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("ucode.databricks.urllib_request.urlopen", return_value=mock_resp):
            from ucode.databricks import ensure_ai_gateway_v2

            ensure_ai_gateway_v2(WS, "fake-token")  # should not raise
