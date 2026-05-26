"""Tests for agents/__init__.py — registry, dispatchers, normalize_tool."""

from __future__ import annotations

import subprocess

import pytest

import ucode.agents as agents_mod
from ucode.agents import (
    DEFAULT_TOOL,
    TOOL_SPECS,
    check_gateway_endpoint,
    configure_selected_tools,
    default_model_for_tool,
    ensure_tool_binary_available,
    install_tool_binary,
    normalize_tool,
    resolve_launch_model,
)


class TestToolSpecs:
    def test_all_tools_present(self):
        assert set(TOOL_SPECS) == {"codex", "claude", "gemini", "opencode", "copilot", "pi"}

    def test_each_spec_has_required_keys(self):
        required = {"binary", "package", "display", "config_path", "backup_path"}
        for tool, spec in TOOL_SPECS.items():
            missing = required - set(spec)
            assert not missing, f"{tool} spec missing: {missing}"

    def test_default_tool_is_codex(self):
        assert DEFAULT_TOOL == "codex"

    def test_each_agent_exposes_update_check(self):
        for tool, module in agents_mod._MODULES.items():
            assert callable(module.is_update_available), f"{tool} missing is_update_available"


class TestNormalizeTool:
    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("codex", "codex"),
            ("claude", "claude"),
            ("claude-code", "claude"),
            ("gemini", "gemini"),
            ("gemini-cli", "gemini"),
            ("opencode", "opencode"),
            ("copilot", "copilot"),
            ("pi", "pi"),
            ("CODEX", "codex"),
            ("  Claude  ", "claude"),
        ],
    )
    def test_known_aliases(self, alias, expected):
        assert normalize_tool(alias) == expected

    def test_unknown_raises(self):
        with pytest.raises(RuntimeError, match="Unsupported"):
            normalize_tool("unknown-agent")


class TestCheckGatewayEndpoint:
    def test_claude_available_when_models_present(self):
        assert check_gateway_endpoint({"claude_models": {"sonnet": "s4"}}, "claude") is True

    def test_claude_unavailable_when_no_models(self):
        assert check_gateway_endpoint({"claude_models": {}}, "claude") is False
        assert check_gateway_endpoint({}, "claude") is False

    def test_codex_available(self):
        assert check_gateway_endpoint({"codex_models": ["model-a"]}, "codex") is True

    def test_gemini_available(self):
        assert check_gateway_endpoint({"gemini_models": ["gemini-2"]}, "gemini") is True

    def test_opencode_available(self):
        state = {"opencode_models": {"anthropic": ["claude-sonnet"]}}
        assert check_gateway_endpoint(state, "opencode") is True

    def test_copilot_available_with_claude(self):
        assert check_gateway_endpoint({"claude_models": {"sonnet": "s4"}}, "copilot") is True

    def test_copilot_available_with_codex(self):
        assert check_gateway_endpoint({"codex_models": ["m"]}, "copilot") is True

    def test_copilot_unavailable_with_only_gemini(self):
        # Gemini is intentionally excluded from Copilot.
        assert check_gateway_endpoint({"gemini_models": ["g"]}, "copilot") is False

    def test_copilot_unavailable_when_no_models(self):
        assert check_gateway_endpoint({}, "copilot") is False

    def test_pi_available_with_claude(self):
        assert check_gateway_endpoint({"claude_models": {"sonnet": "s4"}}, "pi") is True

    def test_pi_available_with_codex(self):
        assert check_gateway_endpoint({"codex_models": ["m"]}, "pi") is True

    def test_pi_available_with_gemini(self):
        assert check_gateway_endpoint({"gemini_models": ["gemini-2"]}, "pi") is True

    def test_pi_unavailable_when_no_models(self):
        assert check_gateway_endpoint({}, "pi") is False


class TestDefaultModelForTool:
    def test_codex_returns_first_model(self):
        assert default_model_for_tool("codex", {"codex_models": ["c1", "c2"]}) == "c1"

    def test_codex_returns_none_when_no_models(self):
        assert default_model_for_tool("codex", {}) is None

    def test_claude_prefers_opus(self):
        state = {"claude_models": {"sonnet": "s4", "opus": "o4", "haiku": "h4"}}
        assert default_model_for_tool("claude", state) == "o4"

    def test_claude_falls_back_to_sonnet(self):
        state = {"claude_models": {"sonnet": "s4"}}
        assert default_model_for_tool("claude", state) == "s4"

    def test_claude_falls_back_to_haiku(self):
        state = {"claude_models": {"haiku": "h4"}}
        assert default_model_for_tool("claude", state) == "h4"

    def test_claude_returns_none_when_no_models(self):
        assert default_model_for_tool("claude", {}) is None

    def test_gemini_returns_first_model(self):
        state = {"gemini_models": ["gemini-2", "gemini-1"]}
        assert default_model_for_tool("gemini", state) == "gemini-2"

    def test_gemini_returns_none_when_no_models(self):
        assert default_model_for_tool("gemini", {}) is None

    def test_opencode_prefers_anthropic(self):
        state = {"opencode_models": {"anthropic": ["claude-sonnet"], "gemini": ["gemini-2"]}}
        assert default_model_for_tool("opencode", state) == "claude-sonnet"

    def test_opencode_falls_back_to_gemini(self):
        state = {"opencode_models": {"gemini": ["gemini-2"]}}
        assert default_model_for_tool("opencode", state) == "gemini-2"

    def test_pi_prefers_claude_opus(self):
        state = {"claude_models": {"opus": "o4", "sonnet": "s4"}, "codex_models": ["c"]}
        assert default_model_for_tool("pi", state) == "o4"

    def test_pi_falls_back_to_codex(self):
        state = {"claude_models": {}, "codex_models": ["c1"]}
        assert default_model_for_tool("pi", state) == "c1"

    def test_pi_falls_back_to_gemini(self):
        state = {"claude_models": {}, "codex_models": [], "gemini_models": ["gemini-2"]}
        assert default_model_for_tool("pi", state) == "gemini-2"

    def test_pi_returns_none_when_no_models(self):
        assert default_model_for_tool("pi", {}) is None


class TestResolveLaunchModel:
    def test_codex_default_model_used_when_no_explicit(self):
        state = {"codex_models": ["c1"]}
        _, model = resolve_launch_model("codex", state, None)
        assert model == "c1"

    def test_explicit_model_used_when_provided(self):
        _, model = resolve_launch_model("claude", {}, "my-model")
        assert model == "my-model"

    def test_default_model_used_when_no_explicit(self):
        state = {"claude_models": {"sonnet": "s4"}}
        _, model = resolve_launch_model("claude", state, None)
        assert model == "s4"

    def test_raises_when_no_models_available(self):
        with pytest.raises(RuntimeError, match="No models available"):
            resolve_launch_model("claude", {}, None)


class TestInstallToolBinary:
    def test_non_strict_returns_false_when_npm_missing(self, monkeypatch):
        monkeypatch.setattr("ucode.agents.shutil.which", lambda _: None)

        assert install_tool_binary("opencode", strict=False) is False

    def test_non_strict_returns_false_when_install_fails(self, monkeypatch):
        def fake_which(binary: str) -> str | None:
            if binary == "npm":
                return "/usr/bin/npm"
            return None

        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(1, args[0])

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)

        assert install_tool_binary("opencode", strict=False) is False

    def test_updates_existing_binary_when_requested(self, monkeypatch, capsys):
        calls: list[list[str]] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents._confirm_update_installed_tool_binary", lambda _: True)

        assert install_tool_binary("opencode", strict=False, update_existing=True) is True
        assert calls == [["npm", "install", "-g", "opencode-ai"]]
        output = capsys.readouterr().out
        assert "Updating OpenCode..." in output
        assert "OpenCode is up to date" in output

    def test_skips_existing_binary_update_when_latest_is_not_newer(self, monkeypatch, capsys):
        calls: list[list[str]] = []
        prompt_calls: list[str] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents.opencode.is_update_available", lambda: None)
        monkeypatch.setattr(
            "ucode.agents.prompt_yes_no", lambda prompt: prompt_calls.append(prompt) or True
        )

        assert install_tool_binary("opencode", strict=False, update_existing=True) is True
        assert calls == []
        assert prompt_calls == []
        assert "Updating OpenCode..." not in capsys.readouterr().out

    def test_prompts_and_updates_existing_binary_when_newer_version_exists(
        self, monkeypatch, capsys
    ):
        calls: list[list[str]] = []
        prompt_calls: list[str] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents.opencode.is_update_available", lambda: ("1.2.3", "1.2.4"))
        monkeypatch.setattr(
            "ucode.agents.prompt_yes_no", lambda prompt: prompt_calls.append(prompt) or True
        )

        assert install_tool_binary("opencode", strict=False, update_existing=True) is True
        assert prompt_calls == ["(Optional) Update OpenCode from 1.2.3 to 1.2.4?"]
        assert calls == [["npm", "install", "-g", "opencode-ai"]]
        assert "Updating OpenCode..." in capsys.readouterr().out

    def test_skips_existing_binary_update_when_user_declines(self, monkeypatch, capsys):
        calls: list[list[str]] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents._confirm_update_installed_tool_binary", lambda _: False)

        assert install_tool_binary("opencode", strict=False, update_existing=True) is True
        assert calls == []
        assert "Updating OpenCode..." not in capsys.readouterr().out

    def test_update_failure_keeps_existing_binary_available(self, monkeypatch):
        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(1, args[0])

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents._confirm_update_installed_tool_binary", lambda _: True)

        assert install_tool_binary("opencode", strict=True, update_existing=True) is True

    def test_existing_old_codex_raises_clear_blocker(self, monkeypatch):
        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        message = "Codex CLI must be updated to 0.134.0 or newer"
        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.codex.minimum_version_error", lambda: message)
        monkeypatch.setattr("ucode.agents.codex.required_update_message", lambda: None)

        with pytest.raises(RuntimeError, match="Codex CLI must be updated"):
            install_tool_binary("codex", strict=True, update_existing=False)

    def test_configure_updates_existing_old_codex_without_optional_prompt(
        self, monkeypatch, capsys
    ):
        calls: list[list[str]] = []
        prompt_calls: list[str] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr(
            "ucode.agents.codex.required_update_message",
            lambda: "Codex CLI 0.133.0 is older than required 0.134.0",
        )
        monkeypatch.setattr("ucode.agents.codex.minimum_version_error", lambda: None)
        monkeypatch.setattr(
            "ucode.agents.prompt_yes_no", lambda prompt: prompt_calls.append(prompt) or False
        )

        assert install_tool_binary("codex", strict=False, update_existing=True) is True
        assert calls == [["npm", "install", "-g", "@openai/codex"]]
        assert prompt_calls == []
        assert "older than required" in capsys.readouterr().out

    def test_ensure_tool_binary_available_raises_when_missing(self, monkeypatch):
        monkeypatch.setattr("ucode.agents.shutil.which", lambda _: None)

        with pytest.raises(RuntimeError, match="OpenCode is not installed"):
            ensure_tool_binary_available("opencode")


class TestConfigureSelectedTools:
    def test_merges_with_existing_available_tools(self, monkeypatch):
        """Configuring a new tool should not drop previously-configured tools
        from state['available_tools']."""
        monkeypatch.setattr("ucode.agents.configure_tool", lambda tool, state, model=None: state)
        monkeypatch.setattr("ucode.agents.save_state", lambda s: None)

        state = {
            "workspace": "https://x.databricks.com",
            "available_tools": ["codex", "claude"],
            "claude_models": {"sonnet": "s4"},
        }
        result = configure_selected_tools(state, ["claude"])
        assert set(result["available_tools"]) == {"codex", "claude"}

    def test_adds_new_tool_to_available_tools(self, monkeypatch):
        monkeypatch.setattr("ucode.agents.configure_tool", lambda tool, state, model=None: state)
        monkeypatch.setattr("ucode.agents.save_state", lambda s: None)

        state = {
            "workspace": "https://x.databricks.com",
            "available_tools": ["codex"],
            "claude_models": {"sonnet": "s4"},
        }
        result = configure_selected_tools(state, ["claude"])
        assert set(result["available_tools"]) == {"codex", "claude"}

    def test_empty_selection_preserves_existing(self, monkeypatch):
        monkeypatch.setattr("ucode.agents.configure_tool", lambda tool, state, model=None: state)
        monkeypatch.setattr("ucode.agents.save_state", lambda s: None)

        state = {"workspace": "https://x.databricks.com", "available_tools": ["codex"]}
        result = configure_selected_tools(state, [])
        assert result["available_tools"] == ["codex"]
