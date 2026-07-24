"""Tests for agents/opencode.py."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ucode.agents import opencode

WS = "https://example.databricks.com"


def _base_urls() -> dict[str, str]:
    return {
        "anthropic": f"{WS}/ai-gateway/anthropic/v1",
        "gemini": f"{WS}/ai-gateway/gemini/v1beta",
        "oss": f"{WS}/ai-gateway/mlflow/v1",
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

    def test_oss_provider_added_when_models_present(self):
        models = {"oss": ["system.ai.kimi-k2-7-code"]}
        overlay, _ = opencode.render_overlay(
            "system.ai.kimi-k2-7-code", "tok", _base_urls(), models
        )
        assert "databricks-oss" in overlay["provider"]

    def test_oss_provider_uses_ai_sdk_openai_package(self):
        models = {"oss": ["system.ai.kimi-k2-7-code"]}
        overlay, _ = opencode.render_overlay(
            "system.ai.kimi-k2-7-code", "tok", _base_urls(), models
        )
        assert overlay["provider"]["databricks-oss"]["npm"] == "@ai-sdk/openai"

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

    def test_oss_base_url(self):
        models = {"oss": ["system.ai.kimi-k2-7-code"]}
        overlay, _ = opencode.render_overlay(
            "system.ai.kimi-k2-7-code", "tok", _base_urls(), models
        )
        options = overlay["provider"]["databricks-oss"]["options"]
        assert options["baseURL"] == f"{WS}/ai-gateway/mlflow/v1"

    def test_glm_gets_token_limits(self):
        models = {"oss": ["system.ai.glm-5-2"]}
        overlay, _ = opencode.render_overlay("system.ai.glm-5-2", "tok", _base_urls(), models)
        glm = overlay["provider"]["databricks-oss"]["models"]["system.ai.glm-5-2"]
        # OpenCode's schema requires both context and output on `limit`.
        assert glm["limit"] == {"context": 200000, "output": 25000}

    def test_non_glm_oss_model_has_no_output_cap(self):
        models = {"oss": ["system.ai.kimi-k2-7-code"]}
        overlay, _ = opencode.render_overlay(
            "system.ai.kimi-k2-7-code", "tok", _base_urls(), models
        )
        kimi = overlay["provider"]["databricks-oss"]["models"]["system.ai.kimi-k2-7-code"]
        assert "limit" not in kimi

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

    def test_managed_keys_include_oss_provider(self):
        models = {"oss": ["system.ai.kimi-k2-7-code"]}
        _, keys = opencode.render_overlay("system.ai.kimi-k2-7-code", "tok", _base_urls(), models)
        assert ["provider", "databricks-oss"] in keys

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

    def test_prefixes_oss_model_with_provider_id(self):
        models = {"oss": ["system.ai.kimi-k2-7-code"]}
        overlay, _ = opencode.render_overlay(
            "system.ai.kimi-k2-7-code", "tok", _base_urls(), models
        )
        assert overlay["model"] == "databricks-oss/system.ai.kimi-k2-7-code"


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

    def test_falls_back_to_gemini(self):
        state = {"opencode_models": {"anthropic": [], "gemini": ["gemini-2"]}}
        assert opencode.default_model(state) == "gemini-2"

    def test_falls_back_to_oss(self):
        state = {
            "opencode_models": {
                "anthropic": [],
                "gemini": [],
                "oss": ["system.ai.kimi-k2-7-code"],
            }
        }
        assert opencode.default_model(state) == "system.ai.kimi-k2-7-code"

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


ALL_PROVIDER_IDS = ["databricks-anthropic", "databricks-google", "databricks-oss"]


class TestManagedProviderIds:
    def test_picks_out_provider_keys_only(self):
        managed_keys = [
            ["model"],
            ["provider", "databricks-anthropic"],
            ["provider", "databricks-oss"],
        ]
        assert opencode._managed_provider_ids(managed_keys) == [
            "databricks-anthropic",
            "databricks-oss",
        ]

    def test_empty_when_no_provider_keys(self):
        assert opencode._managed_provider_ids([["model"]]) == []


class TestRenderAuthPlugin:
    def test_contains_chat_headers_hook(self):
        js = opencode.render_auth_plugin(WS, None, ALL_PROVIDER_IDS)
        assert '"chat.headers"' in js

    def test_embeds_workspace_in_auth_token_argv(self):
        js = opencode.render_auth_plugin(WS, None, ALL_PROVIDER_IDS)
        assert "auth-token" in js
        assert "--host" in js
        assert WS in js

    def test_embeds_profile_flag_when_provided(self):
        js = opencode.render_auth_plugin(WS, "my-profile", ALL_PROVIDER_IDS)
        assert "--profile" in js
        assert "my-profile" in js

    def test_omits_profile_flag_when_not_provided(self):
        js = opencode.render_auth_plugin(WS, None, ALL_PROVIDER_IDS)
        assert "--profile" not in js

    def test_embeds_provider_id_guard_set_with_all_providers(self):
        js = opencode.render_auth_plugin(WS, None, ALL_PROVIDER_IDS)
        for provider_id in ALL_PROVIDER_IDS:
            assert provider_id in js
        assert "MANAGED_PROVIDER_IDS" in js

    def test_embeds_only_configured_subset_of_provider_ids(self):
        js = opencode.render_auth_plugin(WS, None, ["databricks-anthropic"])
        assert "databricks-anthropic" in js
        assert "databricks-google" not in js
        assert "databricks-oss" not in js

    def test_embeds_ttl_constant(self):
        js = opencode.render_auth_plugin(WS, None, ALL_PROVIDER_IDS)
        assert "TTL_MS" in js
        assert str(opencode.AUTH_PLUGIN_TOKEN_TTL_SECONDS * 1000) in js

    def test_exports_plugin_function(self):
        js = opencode.render_auth_plugin(WS, None, ALL_PROVIDER_IDS)
        assert "export const UcodeDatabricksAuth" in js

    def test_provider_scoping_guard_reads_provider_id_directly(self):
        # `chat.headers`'s `input.provider` is opencode's runtime Provider
        # record (id/name/env/options/source/models) -- NOT a nested
        # `{source, info, options}` wrapper. `id` lives directly on
        # `input.provider`, never `input.provider.info.id`. This was
        # confirmed empirically against a real opencode 1.17.10 hook
        # invocation (see #190 follow-up) after a version of this plugin
        # that read `input.provider?.info?.id` silently no-opped on every
        # request -- the guard's `!providerId` was always true, so the
        # hook returned immediately and never refreshed the Authorization
        # header. See TestRenderAuthPluginRuntimeBehavior below for a live
        # Node execution of the generated JS against the real hook shape,
        # which is what actually catches a regression here -- a
        # string-content assertion alone would not.
        js = opencode.render_auth_plugin(WS, None, ALL_PROVIDER_IDS)
        assert "input.provider?.id" in js
        assert "input.provider?.info?.id" not in js

    def test_fails_open_on_error(self):
        js = opencode.render_auth_plugin(WS, None, ALL_PROVIDER_IDS)
        assert "catch" in js
        assert "console.error" in js

    def test_uses_execfile_not_shell(self):
        js = opencode.render_auth_plugin(WS, None, ALL_PROVIDER_IDS)
        assert "execFile" in js
        assert "node:child_process" in js


class TestRenderAuthPluginRuntimeBehavior:
    """Executes the generated plugin under real Node against the actual
    opencode `chat.headers` hook input shape, rather than asserting on the
    generated JS's string content. A version of this plugin that read
    `input.provider?.info?.id` (instead of `input.provider?.id`) passed
    every prior test in this file -- because those tests only checked what
    string literals appear in the generated source -- while silently
    no-opping on every real request (the guard's `!providerId` was always
    true). This class exists specifically to catch that class of bug:
    a mismatch between our assumed hook-input shape and opencode's real
    runtime shape, confirmed via `opencode`'s own `@opencode-ai/plugin`
    type surface and an isolated end-to-end run against opencode 1.17.10
    (#190 follow-up). Skips if `node` isn't on PATH."""

    def _run_plugin(
        self,
        tmp_path,
        provider_id: str | None,
        fake_token: str | None = "fresh-token-xyz",
        headers_preset: bool = True,
    ):
        import shutil
        import subprocess
        import sys

        node = shutil.which("node")
        if not node:
            pytest.skip("`node` is not installed")

        fake_ucode = tmp_path / "fake_ucode.py"
        # `fake_token=None` simulates `ucode auth-token` exiting 0 but
        # printing nothing (or only whitespace) -- a successful subprocess
        # with unusable output.
        print_stmt = f"print({fake_token!r})" if fake_token is not None else 'print("   ")'
        # No shebang needed -- invoked directly via `sys.executable` below,
        # not executed as a standalone script.
        fake_ucode.write_text(f"{print_stmt}\n")

        with patch(
            "ucode.agents.opencode.build_auth_token_argv",
            # Use the same interpreter running this test rather than a bare
            # `python3`, which isn't guaranteed to be on PATH in every
            # environment (even one running Python tests).
            return_value=[sys.executable, str(fake_ucode)],
        ):
            js = opencode.render_auth_plugin(WS, None, ["databricks-anthropic"])

        plugin_path = tmp_path / "plugin.mjs"
        plugin_path.write_text(js)

        provider_literal = f"{{ id: {provider_id!r} }}" if provider_id else "undefined"
        output_literal = (
            '{ headers: { Authorization: "Bearer STALE-BOOTSTRAP-TOKEN" } }'
            if headers_preset
            else "{}"
        )
        harness = tmp_path / "harness.mjs"
        # A `file://` URL is a portable ESM import specifier on every OS;
        # `.as_posix()` produces a bare path, which isn't a valid ESM
        # specifier on Windows (`C:\\...`).
        harness.write_text(f"""
import {{ UcodeDatabricksAuth }} from {plugin_path.as_uri()!r};

const hooks = await UcodeDatabricksAuth({{}});
const output = {output_literal};
const input = {{
  sessionID: "ses_test",
  agent: "build",
  model: {{}},
  provider: {provider_literal},
  message: {{}},
}};
await hooks["chat.headers"](input, output);
console.log(JSON.stringify(output.headers ?? null));
""")

        result = subprocess.run(
            [node, str(harness)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result

    def test_managed_provider_id_directly_on_input_provider_refreshes_header(self, tmp_path):
        # This is the real opencode runtime shape: `input.provider` IS the
        # Provider record, `id` is a direct property.
        result = self._run_plugin(tmp_path, provider_id="databricks-anthropic")
        assert result.returncode == 0, result.stderr
        headers = json.loads(result.stdout.strip().splitlines()[-1])
        assert headers["Authorization"] == "Bearer fresh-token-xyz"

    def test_unmanaged_provider_id_leaves_static_header_untouched(self, tmp_path):
        result = self._run_plugin(tmp_path, provider_id="some-other-provider")
        assert result.returncode == 0, result.stderr
        headers = json.loads(result.stdout.strip().splitlines()[-1])
        assert headers["Authorization"] == "Bearer STALE-BOOTSTRAP-TOKEN"

    def test_missing_provider_leaves_static_header_untouched(self, tmp_path):
        result = self._run_plugin(tmp_path, provider_id=None)
        assert result.returncode == 0, result.stderr
        headers = json.loads(result.stdout.strip().splitlines()[-1])
        assert headers["Authorization"] == "Bearer STALE-BOOTSTRAP-TOKEN"

    def test_empty_token_output_fails_open_instead_of_setting_empty_bearer(self, tmp_path):
        # `ucode auth-token` exiting 0 with empty/whitespace-only stdout must
        # not be treated as a successful fetch -- otherwise the hook would
        # cache and send `Authorization: Bearer ` (empty), clobbering a
        # possibly-still-valid bootstrap token for the full TTL window.
        result = self._run_plugin(tmp_path, provider_id="databricks-anthropic", fake_token=None)
        assert result.returncode == 0, result.stderr
        headers = json.loads(result.stdout.strip().splitlines()[-1])
        assert headers["Authorization"] == "Bearer STALE-BOOTSTRAP-TOKEN"

    def test_initializes_missing_output_headers_instead_of_throwing(self, tmp_path):
        # If opencode ever invokes the hook with no pre-populated
        # `output.headers`, the hook must still set Authorization rather
        # than throwing (which the try/catch would otherwise silently
        # swallow, no-opping the refresh).
        result = self._run_plugin(
            tmp_path, provider_id="databricks-anthropic", headers_preset=False
        )
        assert result.returncode == 0, result.stderr
        headers = json.loads(result.stdout.strip().splitlines()[-1])
        assert headers["Authorization"] == "Bearer fresh-token-xyz"


class TestWriteAuthPlugin:
    def test_writes_plugin_file_containing_hook(self, tmp_path, monkeypatch):
        import ucode.agents.opencode as oc_mod

        plugin_path = tmp_path / "ucode-databricks-auth.js"
        monkeypatch.setattr(oc_mod, "OPENCODE_PLUGIN_PATH", plugin_path)

        state = {"workspace": WS, "profile": None}
        oc_mod.write_auth_plugin(state, ["databricks-anthropic"])

        assert plugin_path.exists()
        content = plugin_path.read_text()
        assert '"chat.headers"' in content
        assert "databricks-anthropic" in content


class TestWriteToolConfigAuthPlugin:
    def _write_and_return(self, tmp_path, monkeypatch, opencode_models):
        import ucode.agents.opencode as oc_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "opencode.json"
        backup_file = tmp_path / "opencode-backup.json"
        plugin_file = tmp_path / "ucode-databricks-auth.js"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        monkeypatch.setattr(oc_mod, "OPENCODE_BACKUP_PATH", backup_file)
        monkeypatch.setattr(oc_mod, "OPENCODE_PLUGIN_PATH", plugin_file)

        state = {
            "workspace": WS,
            "base_urls": {"opencode": _base_urls()},
            "opencode_models": opencode_models,
            "managed_configs": {},
        }

        with (
            patch("ucode.agents.opencode.get_databricks_token", return_value="tok"),
            patch("ucode.agents.opencode.save_state"),
        ):
            oc_mod.write_tool_config(state, "claude-sonnet", token="tok")

        return config_file, plugin_file

    def test_writes_plugin_file_to_configured_path(self, tmp_path, monkeypatch):
        _config_file, plugin_file = self._write_and_return(
            tmp_path, monkeypatch, {"anthropic": ["claude-sonnet"]}
        )

        assert plugin_file.exists()
        assert '"chat.headers"' in plugin_file.read_text()

    def test_plugin_scoped_to_configured_providers_only(self, tmp_path, monkeypatch):
        _config_file, plugin_file = self._write_and_return(
            tmp_path, monkeypatch, {"anthropic": ["claude-sonnet"]}
        )

        content = plugin_file.read_text()
        assert "databricks-anthropic" in content
        assert "databricks-google" not in content
        assert "databricks-oss" not in content

    def test_static_bootstrap_config_still_present_in_opencode_json(self, tmp_path, monkeypatch):
        config_file, _plugin_file = self._write_and_return(
            tmp_path, monkeypatch, {"anthropic": ["claude-sonnet"]}
        )

        written = json.loads(config_file.read_text())
        anthropic_options = written["provider"]["databricks-anthropic"]["options"]
        assert anthropic_options["apiKey"] == "tok"
        assert anthropic_options["headers"]["Authorization"] == "Bearer tok"
