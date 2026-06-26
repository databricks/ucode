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
        "openai": f"{WS}/ai-gateway/codex/v1",
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


class TestRenderOverlayCodex:
    """Regression coverage for #97: the GPT-5 / Codex (Responses) family must
    land in opencode.json alongside Anthropic and Gemini. Before the fix, only
    two providers were written and `databricks-gpt-5-5` could not be reached
    from OpenCode at all."""

    def test_openai_provider_added_when_codex_models_present(self):
        models = {"openai": ["databricks-gpt-5-5"]}
        overlay, _ = opencode.render_overlay("databricks-gpt-5-5", "tok", _base_urls(), models)
        assert "databricks-openai" in overlay["provider"]

    def test_openai_provider_uses_ai_sdk_openai_npm(self):
        models = {"openai": ["databricks-gpt-5-5"]}
        overlay, _ = opencode.render_overlay("databricks-gpt-5-5", "tok", _base_urls(), models)
        assert overlay["provider"]["databricks-openai"]["npm"] == "@ai-sdk/openai"

    def test_openai_base_url_points_at_codex_gateway(self):
        models = {"openai": ["databricks-gpt-5-5"]}
        overlay, _ = opencode.render_overlay("databricks-gpt-5-5", "tok", _base_urls(), models)
        options = overlay["provider"]["databricks-openai"]["options"]
        assert options["baseURL"] == f"{WS}/ai-gateway/codex/v1"

    def test_use_responses_api_flag_set_per_model(self):
        # Databricks GPT-5 / Codex models are Responses-only on
        # /ai-gateway/codex/v1. The per-model `useResponsesApi: true` lives in
        # `models.<m>.options` where opencode reads it.
        models = {"openai": ["databricks-gpt-5-5"]}
        overlay, _ = opencode.render_overlay("databricks-gpt-5-5", "tok", _base_urls(), models)
        model_entry = overlay["provider"]["databricks-openai"]["models"]["databricks-gpt-5-5"]
        assert model_entry["options"]["useResponsesApi"] is True

    def test_use_responses_api_set_on_every_codex_model(self):
        models = {"openai": ["databricks-gpt-5-5", "databricks-gpt-codex"]}
        overlay, _ = opencode.render_overlay("databricks-gpt-5-5", "tok", _base_urls(), models)
        provider_models = overlay["provider"]["databricks-openai"]["models"]
        for m in ("databricks-gpt-5-5", "databricks-gpt-codex"):
            assert provider_models[m]["options"]["useResponsesApi"] is True

    def test_openai_token_in_api_key(self):
        models = {"openai": ["databricks-gpt-5-5"]}
        overlay, _ = opencode.render_overlay("databricks-gpt-5-5", "mytoken", _base_urls(), models)
        assert overlay["provider"]["databricks-openai"]["options"]["apiKey"] == "mytoken"

    def test_openai_authorization_header(self):
        models = {"openai": ["databricks-gpt-5-5"]}
        overlay, _ = opencode.render_overlay("databricks-gpt-5-5", "tok", _base_urls(), models)
        headers = overlay["provider"]["databricks-openai"]["options"]["headers"]
        assert headers["Authorization"] == "Bearer tok"

    def test_user_agent_header_codex(self, monkeypatch):
        monkeypatch.setattr(opencode, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(opencode, "agent_version", lambda binary: "0.74.0")
        models = {"openai": ["databricks-gpt-5-5"]}
        overlay, _ = opencode.render_overlay("databricks-gpt-5-5", "tok", _base_urls(), models)
        model_headers = overlay["provider"]["databricks-openai"]["models"]["databricks-gpt-5-5"][
            "headers"
        ]
        assert model_headers["User-Agent"] == "ucode/0.1.0 opencode/0.74.0"

    def test_managed_keys_include_openai_provider(self):
        models = {"openai": ["databricks-gpt-5-5"]}
        _, keys = opencode.render_overlay("databricks-gpt-5-5", "tok", _base_urls(), models)
        assert ["provider", "databricks-openai"] in keys

    def test_prefixes_openai_model_with_provider_id(self):
        models = {"openai": ["databricks-gpt-5-5"]}
        overlay, _ = opencode.render_overlay("databricks-gpt-5-5", "tok", _base_urls(), models)
        assert overlay["model"] == "databricks-openai/databricks-gpt-5-5"

    def test_already_prefixed_codex_model_is_preserved(self):
        models = {"openai": ["databricks-gpt-5-5"]}
        overlay, _ = opencode.render_overlay(
            "databricks-openai/databricks-gpt-5-5", "tok", _base_urls(), models
        )
        assert overlay["model"] == "databricks-openai/databricks-gpt-5-5"

    def test_all_three_providers_when_all_present(self):
        models = {
            "anthropic": ["claude-sonnet"],
            "gemini": ["gemini-2"],
            "openai": ["databricks-gpt-5-5"],
        }
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        assert set(overlay["provider"].keys()) == {
            "databricks-anthropic",
            "databricks-google",
            "databricks-openai",
        }

    def test_provider_keys_listed_in_module(self):
        # `PROVIDER_KEYS` drives the stale-config cleanup. The codex provider
        # must be in this list or stale entries would leak across configures.
        assert ["provider", "databricks-openai"] in opencode.PROVIDER_KEYS


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


class TestOpencodeDefaultModel:
    def test_prefers_anthropic(self):
        state = {"opencode_models": {"anthropic": ["claude-sonnet"], "gemini": ["gemini-2"]}}
        assert opencode.default_model(state) == "claude-sonnet"

    def test_falls_back_to_openai_before_gemini(self):
        # Codex/GPT-5 ranks above Gemini in the fallback order — these are the
        # primary code models on Databricks once the codex provider is wired.
        state = {
            "opencode_models": {
                "anthropic": [],
                "openai": ["databricks-gpt-5-5"],
                "gemini": ["gemini-2"],
            }
        }
        assert opencode.default_model(state) == "databricks-gpt-5-5"

    def test_falls_back_to_gemini(self):
        state = {"opencode_models": {"anthropic": [], "gemini": ["gemini-2"]}}
        assert opencode.default_model(state) == "gemini-2"

    def test_returns_none_when_empty(self):
        assert opencode.default_model({}) is None
        assert opencode.default_model({"opencode_models": {}}) is None


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
