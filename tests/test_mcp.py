"""Tests for MCP server registration."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock

import pytest

from coding_tool_gateway import mcp

WS = "https://example.databricks.com"


class TestBuildMcpHttpEntry:
    def test_uses_http_url(self):
        entry = mcp.build_mcp_http_entry(f"{WS}/api/2.0/mcp/external/github")
        assert entry["type"] == "http"
        assert entry["url"] == f"{WS}/api/2.0/mcp/external/github"

    def test_uses_oauth_token_header_reference(self):
        entry = mcp.build_mcp_http_entry(f"{WS}/api/2.0/mcp/external/github")
        assert entry["headers"]["Authorization"] == "Bearer ${OAUTH_TOKEN}"
        assert "oauth" not in entry
        assert "headersHelper" not in entry


class TestAddClaudeMcpServer:
    def test_adds_user_scoped_json(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return MagicMock(returncode=0)

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        entry = mcp.build_mcp_http_entry(f"{WS}/api/2.0/mcp/external/github")
        mcp.add_claude_mcp_server("github", entry)

        assert calls
        args = calls[0]["args"]
        assert args[:4] == ["claude", "mcp", "add-json", "github"]
        assert json.loads(args[4]) == entry
        assert args[5:] == ["-s", "user"]
        assert "--client-secret" not in args
        assert "env" not in calls[0]["kwargs"]


class TestAddCodexMcpServer:
    def test_adds_http_server_with_bearer_token_env(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return MagicMock(returncode=0)

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        mcp.add_codex_mcp_server("github", f"{WS}/api/2.0/mcp/external/github")

        assert calls == [
            {
                "args": [
                    "codex",
                    "mcp",
                    "add",
                    "github",
                    "--url",
                    f"{WS}/api/2.0/mcp/external/github",
                    "--bearer-token-env-var",
                    "OAUTH_TOKEN",
                ],
                "kwargs": {
                    "check": True,
                    "capture_output": True,
                    "text": True,
                    "timeout": 30,
                },
            }
        ]


class TestAddGeminiMcpServer:
    def test_adds_user_scoped_http_server_with_auth_header(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return MagicMock(returncode=0)

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        mcp.add_gemini_mcp_server("github", f"{WS}/api/2.0/mcp/external/github")

        assert calls == [
            {
                "args": [
                    "gemini",
                    "mcp",
                    "add",
                    "github",
                    f"{WS}/api/2.0/mcp/external/github",
                    "--type",
                    "http",
                    "--scope",
                    "user",
                    "--header",
                    "Authorization: Bearer ${OAUTH_TOKEN}",
                ],
                "kwargs": {
                    "check": True,
                    "capture_output": True,
                    "text": True,
                    "timeout": 30,
                },
            }
        ]


class TestRemoveClaudeMcpServer:
    def test_returns_true_when_server_removed(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return MagicMock(returncode=0)

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        assert mcp.remove_claude_mcp_server("github", "user") is True
        assert calls == [["claude", "mcp", "remove", "github", "-s", "user"]]

    def test_returns_false_when_server_missing(self, monkeypatch):
        def fake_run(args, **kwargs):
            raise subprocess.CalledProcessError(1, args, stderr="No MCP server named github found")

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        assert mcp.remove_claude_mcp_server("github", "user") is False

    def test_returns_false_when_project_local_server_missing(self, monkeypatch):
        def fake_run(args, **kwargs):
            raise subprocess.CalledProcessError(
                1,
                args,
                stderr="No project-local MCP server found with name: github",
            )

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        assert mcp.remove_claude_mcp_server("github", "project") is False

    def test_returns_false_when_user_scoped_server_missing(self, monkeypatch):
        def fake_run(args, **kwargs):
            raise subprocess.CalledProcessError(
                1,
                args,
                stderr="No user-scoped MCP server found with name: github",
            )

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        assert mcp.remove_claude_mcp_server("github", "user") is False

    def test_unexpected_failure_raises(self, monkeypatch):
        def fake_run(args, **kwargs):
            raise subprocess.CalledProcessError(1, args, stderr="permission denied")

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        try:
            mcp.remove_claude_mcp_server("github", "user")
        except RuntimeError as exc:
            assert "Failed to remove MCP server 'github'" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")


class TestExternalMcpConnectionNames:
    def test_returns_sorted_http_connection_names(self):
        assert mcp.external_mcp_connection_names(
            [
                {"name": "jira-mcp", "connection_type": "HTTP"},
                {"name": "not-http", "connection_type": "POSTGRESQL"},
                {"name": "confluence-mcp", "connection_type": "http"},
                {"name": "jira-mcp", "connection_type": "HTTP"},
            ]
        ) == ["confluence-mcp", "jira-mcp"]

    def test_excludes_explicit_non_mcp_http_connections(self):
        assert mcp.external_mcp_connection_names(
            [
                {
                    "name": "analytics-api",
                    "connection_type": "HTTP",
                    "options": {"is_mcp": "false"},
                },
                {"name": "github-mcp", "connection_type": "HTTP", "options": {"is_mcp": "true"}},
            ]
        ) == ["github-mcp"]


class TestConfigureClientMcpServer:
    def test_configures_copilot_mcp_server(self, monkeypatch):
        calls: list[tuple[str, str]] = []

        monkeypatch.setattr(
            mcp.copilot,
            "write_mcp_server_config",
            lambda name, url: calls.append((name, url)) or False,
        )

        removed_scopes = mcp.configure_client_mcp_server(
            "copilot",
            "github",
            f"{WS}/api/2.0/mcp/external/github",
            mcp.build_mcp_http_entry(f"{WS}/api/2.0/mcp/external/github"),
        )

        assert removed_scopes == []
        assert calls == [("github", f"{WS}/api/2.0/mcp/external/github")]


class TestBuildMcpServerOptions:
    def test_includes_discovered_mcps_in_initial_options(self):
        assert mcp.build_mcp_server_options(["confluence-mcp", "github-mcp"]) == [
            ("external:confluence-mcp", "confluence-mcp"),
            ("external:github-mcp", "github-mcp"),
            ("managed:sql", "Databricks SQL"),
            ("vector-search", "Vector Search"),
            ("genie", "Genie Space"),
            ("uc-functions", "Unity Catalog functions"),
            ("custom", "Custom MCP URLs"),
        ]

    def test_keeps_manual_external_option_when_none_discovered(self):
        assert mcp.build_mcp_server_options([])[0] == (
            "external:manual",
            "External MCP connection",
        )

    def test_prompt_renders_managed_servers_after_sql_and_before_custom(self, monkeypatch, capsys):
        monkeypatch.setattr(mcp.console, "input", lambda *args, **kwargs: "")

        assert mcp.prompt_for_mcp_server_selection(["github-mcp"]) == "done"

        output = capsys.readouterr().out
        assert output.index("External MCP Servers") < output.index("Databricks SQL")
        assert output.index("Databricks SQL") < output.index("Databricks Managed MCP Servers")
        assert output.index("Vector Search") < output.index("Custom MCP URLs")
        assert output.index("Genie Space") < output.index("Custom MCP URLs")
        assert output.index("Unity Catalog functions") < output.index("Custom MCP URLs")
        assert "Built-in AI tools" not in output


class TestPromptForMcpSetupFields:
    def test_returns_none_when_user_types_back(self, monkeypatch, capsys):
        monkeypatch.setattr(mcp.console, "input", lambda *args, **kwargs: "back")

        assert mcp.prompt_for_mcp_setup_fields([("Catalog name", None)]) is None

        output = capsys.readouterr().out
        assert "Type `back` to return to MCP server selection." in output


class TestConfigureMcpCommand:
    def test_registers_external_server_without_oauth_state(self, monkeypatch):
        saved_states: list[dict] = []
        removed: list[tuple[str, str]] = []
        added: list[tuple[str, dict, str]] = []
        selections = iter(["external:manual", "done"])

        monkeypatch.setattr(mcp, "load_state", lambda: {"workspace": WS})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(mcp, "discover_external_mcp_connection_names", lambda workspace: [])
        monkeypatch.setattr(
            mcp, "prompt_for_mcp_server_selection", lambda *args, **kwargs: next(selections)
        )
        monkeypatch.setattr(mcp.console, "input", lambda *args, **kwargs: "github")
        monkeypatch.setattr(
            mcp,
            "remove_claude_mcp_server",
            lambda name, scope: removed.append((name, scope)) or False,
        )
        monkeypatch.setattr(
            mcp,
            "add_claude_mcp_server",
            lambda name, entry, scope: added.append((name, entry, scope)),
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert removed == [
            ("github", "local"),
            ("github", "project"),
            ("github", "user"),
        ]
        assert added == [
            (
                "github",
                {
                    "type": "http",
                    "url": f"{WS}/api/2.0/mcp/external/github",
                    "headers": {"Authorization": "Bearer ${OAUTH_TOKEN}"},
                },
                "user",
            )
        ]
        assert saved_states
        assert "mcp_oauth" not in saved_states[-1]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "github",
                "url": f"{WS}/api/2.0/mcp/external/github",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude"],
            }
        ]

    def test_updates_existing_server_state_by_name(self, monkeypatch):
        saved_states: list[dict] = []
        selections = iter(["external:manual", "done"])

        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {
                "workspace": WS,
                "mcp_servers": [
                    {
                        "name": "github",
                        "url": f"{WS}/old",
                        "client_id": "old-client-id",
                        "client_secret": "old-client-secret",
                    }
                ],
            },
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(mcp, "discover_external_mcp_connection_names", lambda workspace: [])
        monkeypatch.setattr(
            mcp, "prompt_for_mcp_server_selection", lambda *args, **kwargs: next(selections)
        )
        monkeypatch.setattr(mcp.console, "input", lambda *args, **kwargs: "github")
        monkeypatch.setattr(mcp, "remove_claude_mcp_server", lambda name, scope: False)
        monkeypatch.setattr(mcp, "add_claude_mcp_server", lambda name, entry, scope: None)
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "github",
                "url": f"{WS}/api/2.0/mcp/external/github",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude"],
            }
        ]

    def test_registers_discovered_external_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        selections = iter(["external:github-mcp", "done"])
        option_lists: list[list[tuple[str, str]]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {"workspace": WS})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace: None)
        monkeypatch.setattr(
            mcp,
            "available_mcp_clients",
            lambda: ["claude", "codex", "gemini", "opencode", "copilot"],
        )
        monkeypatch.setattr(
            mcp,
            "discover_external_mcp_connection_names",
            lambda workspace: ["confluence-mcp", "github-mcp"],
        )

        def fake_prompt_for_mcp_server_selection(*args, **kwargs):
            option_lists.append(mcp.build_mcp_server_options(["confluence-mcp", "github-mcp"]))
            return next(selections)

        monkeypatch.setattr(
            mcp,
            "prompt_for_mcp_server_selection",
            fake_prompt_for_mcp_server_selection,
        )

        def fake_configure_client_mcp_server(client, name, url, entry):
            configured.append((client, name, url, entry))
            return []

        monkeypatch.setattr(mcp, "configure_client_mcp_server", fake_configure_client_mcp_server)
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert option_lists[0][:2] == [
            ("external:confluence-mcp", "confluence-mcp"),
            ("external:github-mcp", "github-mcp"),
        ]
        expected_entry = {
            "type": "http",
            "url": f"{WS}/api/2.0/mcp/external/github-mcp",
            "headers": {"Authorization": "Bearer ${OAUTH_TOKEN}"},
        }
        assert configured == [
            (
                "claude",
                "github-mcp",
                f"{WS}/api/2.0/mcp/external/github-mcp",
                expected_entry,
            ),
            ("codex", "github-mcp", f"{WS}/api/2.0/mcp/external/github-mcp", expected_entry),
            ("gemini", "github-mcp", f"{WS}/api/2.0/mcp/external/github-mcp", expected_entry),
            ("opencode", "github-mcp", f"{WS}/api/2.0/mcp/external/github-mcp", expected_entry),
            ("copilot", "github-mcp", f"{WS}/api/2.0/mcp/external/github-mcp", expected_entry),
        ]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "github-mcp",
                "url": f"{WS}/api/2.0/mcp/external/github-mcp",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude", "codex", "gemini", "opencode", "copilot"],
            }
        ]

    def test_registers_databricks_sql_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        selections = iter(["managed:sql", "done"])

        monkeypatch.setattr(mcp, "load_state", lambda: {"workspace": WS})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(mcp, "discover_external_mcp_connection_names", lambda workspace: [])
        monkeypatch.setattr(
            mcp, "prompt_for_mcp_server_selection", lambda *args, **kwargs: next(selections)
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == [
            (
                "claude",
                "databricks-sql",
                f"{WS}/api/2.0/mcp/sql",
                {
                    "type": "http",
                    "url": f"{WS}/api/2.0/mcp/sql",
                    "headers": {"Authorization": "Bearer ${OAUTH_TOKEN}"},
                },
            )
        ]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "databricks-sql",
                "url": f"{WS}/api/2.0/mcp/sql",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude"],
            }
        ]

    def test_registers_vector_search_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        selections = iter(["vector-search", "done"])

        monkeypatch.setattr(mcp, "load_state", lambda: {"workspace": WS})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(mcp, "discover_external_mcp_connection_names", lambda workspace: [])
        monkeypatch.setattr(
            mcp, "prompt_for_mcp_server_selection", lambda *args, **kwargs: next(selections)
        )
        inputs = iter(["main", "search", "docs-index"])
        monkeypatch.setattr(mcp.console, "input", lambda *args, **kwargs: next(inputs))
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == [
            (
                "claude",
                "databricks-vector-search-main-search-docs-index",
                f"{WS}/api/2.0/mcp/vector-search/main/search/docs-index",
                {
                    "type": "http",
                    "url": f"{WS}/api/2.0/mcp/vector-search/main/search/docs-index",
                    "headers": {"Authorization": "Bearer ${OAUTH_TOKEN}"},
                },
            )
        ]

    @pytest.mark.parametrize("selection", ["vector-search", "genie", "uc-functions", "custom"])
    def test_returns_to_selection_when_backing_out_of_setup(self, monkeypatch, selection):
        configured: list[tuple[str, str, str, dict]] = []
        saved_states: list[dict] = []
        selections = iter([selection, "done"])

        monkeypatch.setattr(mcp, "load_state", lambda: {"workspace": WS})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(mcp, "discover_external_mcp_connection_names", lambda workspace: [])
        monkeypatch.setattr(
            mcp, "prompt_for_mcp_server_selection", lambda *args, **kwargs: next(selections)
        )
        monkeypatch.setattr(mcp.console, "input", lambda *args, **kwargs: "back")
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == []
        assert saved_states == []

    def test_registers_genie_space_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        selections = iter(["genie", "done"])

        monkeypatch.setattr(mcp, "load_state", lambda: {"workspace": WS})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(mcp, "discover_external_mcp_connection_names", lambda workspace: [])
        monkeypatch.setattr(
            mcp, "prompt_for_mcp_server_selection", lambda *args, **kwargs: next(selections)
        )
        monkeypatch.setattr(mcp.console, "input", lambda *args, **kwargs: "space-123")
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == [
            (
                "claude",
                "databricks-genie-space-123",
                f"{WS}/api/2.0/mcp/genie/space-123",
                {
                    "type": "http",
                    "url": f"{WS}/api/2.0/mcp/genie/space-123",
                    "headers": {"Authorization": "Bearer ${OAUTH_TOKEN}"},
                },
            )
        ]

    def test_registers_uc_functions_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        selections = iter(["uc-functions", "done"])
        inputs = iter(["main", "tools"])

        monkeypatch.setattr(mcp, "load_state", lambda: {"workspace": WS})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(mcp, "discover_external_mcp_connection_names", lambda workspace: [])
        monkeypatch.setattr(
            mcp, "prompt_for_mcp_server_selection", lambda *args, **kwargs: next(selections)
        )
        monkeypatch.setattr(mcp.console, "input", lambda *args, **kwargs: next(inputs))
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == [
            (
                "claude",
                "databricks-uc-main-tools",
                f"{WS}/api/2.0/mcp/functions/main/tools",
                {
                    "type": "http",
                    "url": f"{WS}/api/2.0/mcp/functions/main/tools",
                    "headers": {"Authorization": "Bearer ${OAUTH_TOKEN}"},
                },
            )
        ]
