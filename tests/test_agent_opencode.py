"""Tests for agents/opencode.py."""

from __future__ import annotations

import json
from unittest.mock import patch

from ucode.agents import opencode

WS = "https://example.databricks.com"


def _base_urls() -> dict[str, str]:
    return {
        "anthropic": f"{WS}/ai-gateway/anthropic/v1",
        "gemini": f"{WS}/ai-gateway/gemini/v1beta",
        "open-responses": f"{WS}/serving-endpoints",
    }


class TestOpencodeSpec:
    def test_binary(self):
        assert opencode.SPEC["binary"] == "opencode"

    def test_package(self):
        assert opencode.SPEC["package"] == "opencode-ai"

    def test_display(self):
        assert opencode.SPEC["display"] == "OpenCode"

    def test_config_path_is_under_ucode_xdg_home(self):
        assert opencode.SPEC["config_path"] == (
            opencode.OPENCODE_XDG_CONFIG_HOME / "opencode" / "opencode.json"
        )


class TestRenderOverlay:
    def test_sets_model(self):
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), {})
        assert overlay["model"] == "claude-sonnet"

    def test_anthropic_provider_added_when_models_present(self):
        models = {"anthropic": ["claude-sonnet"], "gemini": []}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        assert "databricks-anthropic" in overlay["provider"]

    def test_gemini_provider_added_when_models_present(self):
        models = {"anthropic": [], "gemini": ["gemini-2"]}
        overlay, _ = opencode.render_overlay("gemini-2", "tok", _base_urls(), models)
        assert "databricks-google" in overlay["provider"]

    def test_both_providers_when_both_present(self):
        models = {"anthropic": ["claude-sonnet"], "gemini": ["gemini-2"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        assert "databricks-anthropic" in overlay["provider"]
        assert "databricks-google" in overlay["provider"]

    def test_no_provider_key_when_no_models(self):
        overlay, _ = opencode.render_overlay("model", "tok", _base_urls(), {})
        assert "provider" not in overlay

    def test_anthropic_base_url(self):
        models = {"anthropic": ["claude-sonnet"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        options = overlay["provider"]["databricks-anthropic"]["options"]
        assert options["baseURL"] == f"{WS}/ai-gateway/anthropic/v1"

    def test_gemini_base_url(self):
        models = {"gemini": ["gemini-2"]}
        overlay, _ = opencode.render_overlay("gemini-2", "tok", _base_urls(), models)
        options = overlay["provider"]["databricks-google"]["options"]
        assert options["baseURL"] == f"{WS}/ai-gateway/gemini/v1beta"

    def test_token_in_api_key(self):
        models = {"anthropic": ["claude-sonnet"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "mytoken", _base_urls(), models)
        assert overlay["provider"]["databricks-anthropic"]["options"]["apiKey"] == "mytoken"

    def test_authorization_header(self):
        models = {"anthropic": ["claude-sonnet"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        headers = overlay["provider"]["databricks-anthropic"]["options"]["headers"]
        assert headers["Authorization"] == "Bearer tok"

    def test_anthropic_tool_streaming_disabled(self):
        # @ai-sdk/anthropic injects `eager_input_streaming: true` on tool defs,
        # which the Databricks gateway rejects. opencode's auto-disable skips
        # Claude models, so we opt out per-model. The setting must live in
        # `models.<m>.options` — per-call providerOptions — not provider options.
        models = {"anthropic": ["claude-sonnet"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        model_entry = overlay["provider"]["databricks-anthropic"]["models"]["claude-sonnet"]
        assert model_entry["options"]["toolStreaming"] is False

    def test_user_agent_header_anthropic(self, monkeypatch):
        # UA must live at the per-model level — OpenCode clobbers
        # provider-level `headers["User-Agent"]` in session/llm.ts.
        monkeypatch.setattr(opencode, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(opencode, "agent_version", lambda binary: "0.74.0")
        models = {"anthropic": ["claude-sonnet"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        model_headers = overlay["provider"]["databricks-anthropic"]["models"]["claude-sonnet"][
            "headers"
        ]
        assert model_headers["User-Agent"] == "ucode/0.1.0 opencode/0.74.0"

    def test_user_agent_header_gemini(self, monkeypatch):
        monkeypatch.setattr(opencode, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(opencode, "agent_version", lambda binary: "0.74.0")
        models = {"gemini": ["gemini-2"]}
        overlay, _ = opencode.render_overlay("gemini-2", "tok", _base_urls(), models)
        model_headers = overlay["provider"]["databricks-google"]["models"]["gemini-2"]["headers"]
        assert model_headers["User-Agent"] == "ucode/0.1.0 opencode/0.74.0"

    def test_provider_level_headers_only_authorization(self, monkeypatch):
        # Sanity: provider-level headers should NOT include User-Agent (since
        # it's clobbered there) — only Authorization.
        models = {"anthropic": ["claude-sonnet"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        provider_headers = overlay["provider"]["databricks-anthropic"]["options"]["headers"]
        assert "User-Agent" not in provider_headers
        assert provider_headers["Authorization"] == "Bearer tok"

    def test_managed_keys_include_model(self):
        _, keys = opencode.render_overlay("model", "tok", _base_urls(), {})
        assert ["model"] in keys

    def test_managed_keys_include_anthropic_provider(self):
        models = {"anthropic": ["claude-sonnet"]}
        _, keys = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        assert ["provider", "databricks-anthropic"] in keys

    def test_managed_keys_include_gemini_provider(self):
        models = {"gemini": ["gemini-2"]}
        _, keys = opencode.render_overlay("gemini-2", "tok", _base_urls(), models)
        assert ["provider", "databricks-google"] in keys

    def test_anthropic_models_listed(self):
        models = {"anthropic": ["claude-sonnet", "claude-haiku"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        provider_models = overlay["provider"]["databricks-anthropic"]["models"]
        assert "claude-sonnet" in provider_models
        assert "claude-haiku" in provider_models

    def test_prefixes_anthropic_model_with_provider_id(self):
        models = {"anthropic": ["claude-sonnet"], "gemini": []}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        assert overlay["model"] == "databricks-anthropic/claude-sonnet"

    def test_prefixes_gemini_model_with_provider_id(self):
        models = {"anthropic": [], "gemini": ["gemini-2"]}
        overlay, _ = opencode.render_overlay("gemini-2", "tok", _base_urls(), models)
        assert overlay["model"] == "databricks-google/gemini-2"

    def test_open_responses_provider_added_when_models_present(self):
        models = {"open-responses": ["databricks-kimi-k2-6-colo"]}
        overlay, _ = opencode.render_overlay(
            "databricks-kimi-k2-6-colo", "tok", _base_urls(), models
        )
        assert "databricks-open-responses" in overlay["provider"]

    def test_open_responses_base_url_is_serving_endpoints(self):
        models = {"open-responses": ["databricks-kimi-k2-6-colo"]}
        overlay, _ = opencode.render_overlay(
            "databricks-kimi-k2-6-colo", "tok", _base_urls(), models
        )
        options = overlay["provider"]["databricks-open-responses"]["options"]
        assert options["baseURL"] == f"{WS}/serving-endpoints"

    def test_open_responses_uses_openai_npm_package(self):
        models = {"open-responses": ["databricks-kimi-k2-6-colo"]}
        overlay, _ = opencode.render_overlay(
            "databricks-kimi-k2-6-colo", "tok", _base_urls(), models
        )
        assert overlay["provider"]["databricks-open-responses"]["npm"] == "@ai-sdk/openai"

    def test_open_responses_omits_api_key_so_loader_runs(self):
        # OpenCode only calls the plugin auth.loader (which installs the
        # /responses -> /open-responses fetch rewrite) when it has to resolve
        # credentials itself. An apiKey in options short-circuits that, so it
        # must be absent — the loader supplies the token from OAUTH_TOKEN.
        models = {"open-responses": ["databricks-kimi-k2-6-colo"]}
        overlay, _ = opencode.render_overlay(
            "databricks-kimi-k2-6-colo", "tok", _base_urls(), models
        )
        options = overlay["provider"]["databricks-open-responses"]["options"]
        assert "apiKey" not in options

    def test_open_responses_axon_headers(self):
        models = {"open-responses": ["databricks-kimi-k2-6-colo"]}
        overlay, _ = opencode.render_overlay(
            "databricks-kimi-k2-6-colo", "tok", _base_urls(), models
        )
        headers = overlay["provider"]["databricks-open-responses"]["options"]["headers"]
        assert headers["Authorization"] == "Bearer tok"
        # X-Axon-Mode: ROUTER 502s the -colo endpoint, so it is omitted.
        assert "X-Axon-Mode" not in headers
        assert headers["X-Axon-LB-Mode"] == "DICER"

    def test_open_responses_user_agent_header(self, monkeypatch):
        monkeypatch.setattr(opencode, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(opencode, "agent_version", lambda binary: "0.74.0")
        models = {"open-responses": ["databricks-kimi-k2-6-colo"]}
        overlay, _ = opencode.render_overlay(
            "databricks-kimi-k2-6-colo", "tok", _base_urls(), models
        )
        model_headers = overlay["provider"]["databricks-open-responses"]["models"][
            "databricks-kimi-k2-6-colo"
        ]["headers"]
        assert model_headers["User-Agent"] == "ucode/0.1.0 opencode/0.74.0"

    def test_managed_keys_include_open_responses_provider(self):
        models = {"open-responses": ["databricks-kimi-k2-6-colo"]}
        _, keys = opencode.render_overlay("databricks-kimi-k2-6-colo", "tok", _base_urls(), models)
        assert ["provider", "databricks-open-responses"] in keys

    def test_prefixes_open_responses_model_with_provider_id(self):
        models = {"open-responses": ["databricks-kimi-k2-6-colo"]}
        overlay, _ = opencode.render_overlay(
            "databricks-kimi-k2-6-colo", "tok", _base_urls(), models
        )
        assert overlay["model"] == "databricks-open-responses/databricks-kimi-k2-6-colo"


class TestMcpServerConfig:
    def test_builds_remote_server_entry_with_oauth_token_env_header(self):
        entry = opencode.build_mcp_server_entry(f"{WS}/api/2.0/mcp/external/github")

        assert entry == {
            "type": "remote",
            "url": f"{WS}/api/2.0/mcp/external/github",
            "enabled": True,
            "headers": {"Authorization": "Bearer {env:OAUTH_TOKEN}"},
        }

    def test_writes_mcp_server_without_clobbering_existing_config(self, tmp_path, monkeypatch):
        import ucode.agents.opencode as oc_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "opencode.json"
        backup_file = tmp_path / "opencode-backup.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        monkeypatch.setattr(oc_mod, "OPENCODE_BACKUP_PATH", backup_file)

        config_file.write_text(
            json.dumps(
                {
                    "model": "existing-model",
                    "mcp": {"old-server": {"type": "local", "command": ["old"]}},
                }
            ),
            encoding="utf-8",
        )

        removed = oc_mod.write_mcp_server_config(
            "github",
            f"{WS}/api/2.0/mcp/external/github",
        )

        written = json.loads(config_file.read_text())
        assert removed is False
        assert written["model"] == "existing-model"
        assert written["mcp"]["old-server"] == {"type": "local", "command": ["old"]}
        assert written["mcp"]["github"] == {
            "type": "remote",
            "url": f"{WS}/api/2.0/mcp/external/github",
            "enabled": True,
            "headers": {"Authorization": "Bearer {env:OAUTH_TOKEN}"},
        }

    def test_reports_replaced_mcp_server(self, tmp_path, monkeypatch):
        import ucode.agents.opencode as oc_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "opencode.json"
        backup_file = tmp_path / "opencode-backup.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        monkeypatch.setattr(oc_mod, "OPENCODE_BACKUP_PATH", backup_file)

        config_file.write_text(json.dumps({"mcp": {"github": {"old": True}}}), encoding="utf-8")

        removed = oc_mod.write_mcp_server_config(
            "github",
            f"{WS}/api/2.0/mcp/external/github",
        )

        assert removed is True
        written = json.loads(config_file.read_text())
        assert written["mcp"]["github"]["url"] == f"{WS}/api/2.0/mcp/external/github"

    def test_removes_mcp_server_without_clobbering_others(self, tmp_path, monkeypatch):
        import ucode.agents.opencode as oc_mod

        config_file = tmp_path / "opencode.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        config_file.write_text(
            json.dumps(
                {
                    "model": "existing-model",
                    "mcp": {
                        "github": {"url": "old"},
                        "jira": {"url": "keep"},
                    },
                }
            ),
            encoding="utf-8",
        )

        removed = oc_mod.remove_mcp_server_config("github")

        written = json.loads(config_file.read_text())
        assert removed is True
        assert "github" not in written["mcp"]
        assert written["mcp"]["jira"] == {"url": "keep"}
        assert written["model"] == "existing-model"


class TestBuildRuntimeEnv:
    def test_sets_oauth_token_for_mcp(self):
        env = opencode.build_runtime_env("tok")

        assert env["OAUTH_TOKEN"] == "tok"

    def test_sets_ucode_xdg_config_home(self):
        env = opencode.build_runtime_env("tok")

        assert env["XDG_CONFIG_HOME"] == str(opencode.OPENCODE_XDG_CONFIG_HOME)

    def test_sets_ucode_xdg_data_home(self):
        env = opencode.build_runtime_env("tok")

        assert env["XDG_DATA_HOME"] == str(opencode.OPENCODE_XDG_DATA_HOME)


class TestOpenResponsesAuth:
    def test_write_creates_api_entry(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "opencode" / "auth.json"
        monkeypatch.setattr(opencode, "OPENCODE_AUTH_PATH", auth_file)

        opencode._write_open_responses_auth("tok")

        auth = json.loads(auth_file.read_text())
        assert auth["databricks-open-responses"] == {"type": "api", "key": "tok"}

    def test_write_refreshes_token_and_preserves_others(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "opencode" / "auth.json"
        auth_file.parent.mkdir(parents=True, exist_ok=True)
        auth_file.write_text(
            json.dumps(
                {
                    "databricks-open-responses": {"type": "api", "key": "old"},
                    "other": {"type": "api", "key": "keep"},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(opencode, "OPENCODE_AUTH_PATH", auth_file)

        opencode._write_open_responses_auth("new")

        auth = json.loads(auth_file.read_text())
        assert auth["databricks-open-responses"]["key"] == "new"
        assert auth["other"] == {"type": "api", "key": "keep"}

    def test_remove_drops_only_open_responses_entry(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "opencode" / "auth.json"
        auth_file.parent.mkdir(parents=True, exist_ok=True)
        auth_file.write_text(
            json.dumps(
                {
                    "databricks-open-responses": {"type": "api", "key": "old"},
                    "other": {"type": "api", "key": "keep"},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(opencode, "OPENCODE_AUTH_PATH", auth_file)

        opencode._remove_open_responses_auth()

        auth = json.loads(auth_file.read_text())
        assert "databricks-open-responses" not in auth
        assert auth["other"] == {"type": "api", "key": "keep"}

    def test_remove_is_noop_when_absent(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "opencode" / "auth.json"
        monkeypatch.setattr(opencode, "OPENCODE_AUTH_PATH", auth_file)

        # No file on disk: removal must not raise or create one.
        opencode._remove_open_responses_auth()

        assert not auth_file.exists()


class TestOpencodeDefaultModel:
    def test_prefers_anthropic(self):
        state = {"opencode_models": {"anthropic": ["claude-sonnet"], "gemini": ["gemini-2"]}}
        assert opencode.default_model(state) == "claude-sonnet"

    def test_falls_back_to_gemini(self):
        state = {"opencode_models": {"anthropic": [], "gemini": ["gemini-2"]}}
        assert opencode.default_model(state) == "gemini-2"

    def test_falls_back_to_open_responses(self):
        state = {"opencode_models": {"anthropic": [], "gemini": [], "open-responses": ["kimi"]}}
        assert opencode.default_model(state) == "kimi"

    def test_returns_none_when_empty(self):
        assert opencode.default_model({}) is None
        assert opencode.default_model({"opencode_models": {}}) is None


class TestRefreshPreservesConfiguredModel:
    """The token-refresh loop must reuse the model already written to
    opencode.json (e.g. the cheaper model chosen at launch) instead of resetting
    it to the default — otherwise a refresh would clobber the selection."""

    def _setup(self, tmp_path, monkeypatch, configured_model):
        import ucode.agents.opencode as oc_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "opencode.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        monkeypatch.setattr(oc_mod, "OPENCODE_BACKUP_PATH", tmp_path / "opencode-backup.json")
        if configured_model is not None:
            config_file.write_text(json.dumps({"model": configured_model}), encoding="utf-8")
        return oc_mod, config_file

    def test_refresh_keeps_configured_model_over_default(self, tmp_path, monkeypatch):
        # Default would be opus (anthropic[0]); the config already selects haiku.
        oc_mod, config_file = self._setup(
            tmp_path, monkeypatch, "databricks-anthropic/databricks-claude-haiku-4-5"
        )
        state = {
            "workspace": WS,
            "opencode_models": {
                "anthropic": ["databricks-claude-opus-4-8", "databricks-claude-haiku-4-5"]
            },
        }
        monkeypatch.setattr(oc_mod, "get_databricks_token", lambda *a, **k: "tok")
        oc_mod._refresh_token_once(state)
        written = json.loads(config_file.read_text())
        assert written["model"] == "databricks-anthropic/databricks-claude-haiku-4-5"

    def test_refresh_uses_default_when_no_config(self, tmp_path, monkeypatch):
        oc_mod, config_file = self._setup(tmp_path, monkeypatch, None)
        state = {
            "workspace": WS,
            "opencode_models": {
                "anthropic": ["databricks-claude-opus-4-8", "databricks-claude-haiku-4-5"]
            },
        }
        monkeypatch.setattr(oc_mod, "get_databricks_token", lambda *a, **k: "tok")
        oc_mod._refresh_token_once(state)
        written = json.loads(config_file.read_text())
        assert written["model"] == "databricks-anthropic/databricks-claude-opus-4-8"


class TestOpencodeValidateCmd:
    def test_starts_with_binary(self):
        cmd = opencode.validate_cmd("opencode")
        assert cmd[0] == "opencode"

    def test_uses_run_subcommand(self):
        cmd = opencode.validate_cmd("opencode")
        assert "run" in cmd

    def test_has_prompt(self):
        cmd = opencode.validate_cmd("opencode")
        assert len(cmd) > 2


class TestWriteToolConfigStaleProviderCleanup:
    def test_stale_providers_removed_before_merge(self, tmp_path, monkeypatch):
        import ucode.agents.opencode as oc_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "opencode.json"
        backup_file = tmp_path / "opencode-backup.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        monkeypatch.setattr(oc_mod, "OPENCODE_BACKUP_PATH", backup_file)

        stale = {
            "provider": {
                "databricks-anthropic": {"old": True},
                "databricks-google": {"old": True},
                "other-provider": {"keep": True},
            }
        }
        config_file.write_text(json.dumps(stale), encoding="utf-8")

        state = {
            "workspace": WS,
            "base_urls": {"opencode": _base_urls()},
            "opencode_models": {"anthropic": ["claude-sonnet"]},
            "managed_configs": {},
        }

        with (
            patch("ucode.agents.opencode.get_databricks_token", return_value="tok"),
            patch("ucode.agents.opencode.save_state"),
        ):
            oc_mod.write_tool_config(state, "claude-sonnet", token="tok")

        written = json.loads(config_file.read_text())
        providers = written.get("provider", {})
        # stale entry is replaced with new data, not kept as-is
        assert providers.get("databricks-anthropic") != {"old": True}
        # unmanaged provider entry survives
        assert providers.get("other-provider") == {"keep": True}

    def test_config_written_with_correct_model(self, tmp_path, monkeypatch):
        import ucode.agents.opencode as oc_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "opencode.json"
        backup_file = tmp_path / "opencode-backup.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        monkeypatch.setattr(oc_mod, "OPENCODE_BACKUP_PATH", backup_file)

        state = {
            "workspace": WS,
            "base_urls": {"opencode": _base_urls()},
            "opencode_models": {"anthropic": ["claude-sonnet"]},
            "managed_configs": {},
        }

        with (
            patch("ucode.agents.opencode.get_databricks_token", return_value="tok"),
            patch("ucode.agents.opencode.save_state"),
        ):
            oc_mod.write_tool_config(state, "claude-sonnet", token="tok")

        written = json.loads(config_file.read_text())
        assert written["model"] == "databricks-anthropic/claude-sonnet"

    def test_config_prunes_stale_usage_plugin(self, tmp_path, monkeypatch):
        """The slow spawnSync usage plugin is removed from config and disk on setup."""
        import ucode.agents.opencode as oc_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "opencode.json"
        backup_file = tmp_path / "opencode-backup.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        monkeypatch.setattr(oc_mod, "OPENCODE_BACKUP_PATH", backup_file)

        plugin_path = str(config_file.parent / "plugins" / "ucode-usage.mjs")
        plugin_file = config_file.parent / "plugins" / "ucode-usage.mjs"
        plugin_file.parent.mkdir(parents=True, exist_ok=True)
        plugin_file.write_text("// stale", encoding="utf-8")
        config_file.write_text(
            json.dumps({"plugin": ["user-plugin", plugin_path]}), encoding="utf-8"
        )
        state = {
            "workspace": WS,
            "base_urls": {"opencode": _base_urls()},
            "opencode_models": {"anthropic": ["claude-sonnet"]},
            "managed_configs": {},
        }

        with (
            patch("ucode.agents.opencode.get_databricks_token", return_value="tok"),
            patch("ucode.agents.opencode.save_state"),
        ):
            oc_mod.write_tool_config(state, "claude-sonnet", token="tok")

        written = json.loads(config_file.read_text())
        assert written["plugin"] == ["user-plugin"]
        assert not plugin_file.exists()

    def test_open_responses_plugin_installed_when_kimi_present(self, tmp_path, monkeypatch):
        import ucode.agents.opencode as oc_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "opencode.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        monkeypatch.setattr(oc_mod, "OPENCODE_BACKUP_PATH", tmp_path / "opencode-backup.json")
        auth_file = tmp_path / "data" / "opencode" / "auth.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_AUTH_PATH", auth_file)

        state = {
            "workspace": WS,
            "base_urls": {"opencode": _base_urls()},
            "opencode_models": {"open-responses": ["databricks-kimi-k2-6-colo"]},
            "managed_configs": {},
        }

        with (
            patch("ucode.agents.opencode.get_databricks_token", return_value="tok"),
            patch("ucode.agents.opencode.save_state"),
        ):
            oc_mod.write_tool_config(state, "databricks-kimi-k2-6-colo", token="tok")

        written = json.loads(config_file.read_text())
        plugin_path = str(config_file.parent / "plugins" / "ucode-open-responses.mjs")
        assert plugin_path in written["plugin"]
        plugin = (config_file.parent / "plugins" / "ucode-open-responses.mjs").read_text()
        assert "ucode-managed-open-responses-plugin" in plugin
        assert "databricks-open-responses" in plugin
        assert "/open-responses" in plugin
        # The auth entry must exist so OpenCode runs the plugin's auth.loader.
        auth = json.loads(auth_file.read_text())
        assert auth["databricks-open-responses"] == {"type": "api", "key": "tok"}

    def test_open_responses_plugin_pruned_when_kimi_absent(self, tmp_path, monkeypatch):
        import ucode.agents.opencode as oc_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "opencode.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        monkeypatch.setattr(oc_mod, "OPENCODE_BACKUP_PATH", tmp_path / "opencode-backup.json")
        auth_file = tmp_path / "data" / "opencode" / "auth.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_AUTH_PATH", auth_file)
        # A previously-registered open-responses plugin path that should be dropped
        # now that no Kimi model is configured.
        stale_plugin = str(config_file.parent / "plugins" / "ucode-open-responses.mjs")
        config_file.write_text(
            json.dumps({"plugin": ["user-plugin", stale_plugin]}), encoding="utf-8"
        )
        # A previously-written auth entry that should also be pruned.
        auth_file.parent.mkdir(parents=True, exist_ok=True)
        auth_file.write_text(
            json.dumps(
                {
                    "databricks-open-responses": {"type": "api", "key": "stale"},
                    "other-provider": {"type": "api", "key": "keep"},
                }
            ),
            encoding="utf-8",
        )

        state = {
            "workspace": WS,
            "base_urls": {"opencode": _base_urls()},
            "opencode_models": {"anthropic": ["claude-sonnet"]},
            "managed_configs": {},
        }

        with (
            patch("ucode.agents.opencode.get_databricks_token", return_value="tok"),
            patch("ucode.agents.opencode.save_state"),
        ):
            oc_mod.write_tool_config(state, "claude-sonnet", token="tok")

        written = json.loads(config_file.read_text())
        assert stale_plugin not in written["plugin"]
        assert "user-plugin" in written["plugin"]
        # The open-responses auth entry is removed, unrelated entries preserved.
        auth = json.loads(auth_file.read_text())
        assert "databricks-open-responses" not in auth
        assert auth["other-provider"] == {"type": "api", "key": "keep"}
