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
    def test_codex_returns_highest_gpt_model(self):
        models = ["databricks-gpt-5", "databricks-gpt-5-5"]
        assert default_model_for_tool("codex", {"codex_models": models}) == "databricks-gpt-5-5"

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
        state = {"codex_models": ["databricks-gpt-5"]}
        _, model = resolve_launch_model("codex", state, None)
        assert model == "databricks-gpt-5"

    def test_explicit_model_used_when_provided(self):
        _, model = resolve_launch_model("claude", {}, "my-model")
        assert model == "my-model"

    def test_claude_policy_model_alias_resolves_family(self):
        state = {"claude_models": {"opus": "system.ai.claude-opus-4-8"}}
        _, model = resolve_launch_model("claude", state, "opus")
        assert model == "system.ai.claude-opus-4-8"

    def test_codex_policy_model_alias_resolves_available_model(self):
        state = {"codex_models": ["system.ai.gpt-5-4-mini"]}
        _, model = resolve_launch_model("codex", state, "gpt-5-4-mini")
        assert model == "system.ai.gpt-5-4-mini"

    def test_codex_policy_model_keeps_system_ai_prefix_when_no_match(self):
        # The policy names a model the workspace did not discover, but the
        # workspace routes by `system.ai.<name>`. Keep that prefix rather than
        # falling through to the OpenAI-id rewrite (`gpt-5.4-mini`), which is
        # unroutable on a model-services gateway.
        state = {"codex_models": ["system.ai.gpt-5", "system.ai.gpt-5-mini"]}
        _, model = resolve_launch_model("codex", state, "gpt-5-4-mini")
        assert model == "system.ai.gpt-5-4-mini"

    def test_codex_policy_model_unchanged_on_foundation_models_workspace(self):
        # Legacy foundation-models workspaces use `databricks-*` endpoint names;
        # an unmatched bare id must not gain a spurious `system.ai.` prefix.
        state = {"codex_models": ["databricks-gpt-5-4"]}
        _, model = resolve_launch_model("codex", state, "gpt-5-4-mini")
        assert model == "gpt-5-4-mini"

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

    def test_optional_update_prompt_suppressed_when_disabled(self, monkeypatch):
        """prompt_optional_updates=False must skip the optional update check
        entirely — the confirm prompt should never be reached."""

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents._minimum_version_error", lambda _: None)
        monkeypatch.setattr("ucode.agents._required_update_message", lambda _: None)

        def boom(_tool: str) -> bool:
            raise AssertionError("optional update prompt should not be reached")

        monkeypatch.setattr("ucode.agents._confirm_update_installed_tool_binary", boom)

        assert (
            install_tool_binary(
                "opencode",
                strict=False,
                update_existing=True,
                prompt_optional_updates=False,
            )
            is True
        )

    def test_required_update_runs_even_when_optional_prompt_disabled(self, monkeypatch):
        """A required (minimum-version) update is forced regardless of the
        prompt_optional_updates preference."""
        calls: list[list[str]] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents._required_update_message", lambda _: "must upgrade")
        monkeypatch.setattr("ucode.agents._minimum_version_error", lambda _: None)

        assert (
            install_tool_binary(
                "opencode",
                strict=True,
                update_existing=True,
                prompt_optional_updates=False,
            )
            is True
        )
        assert calls and calls[0][:3] == ["npm", "install", "-g"]

    def test_too_new_tool_warns_and_downgrades_on_confirm(self, monkeypatch, capsys):
        """An installed build past its supported ceiling is offered as a
        downgrade (to a pinned working version), not an upgrade."""
        calls: list[list[str]] = []
        prompt_calls: list[str] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents.gemini.too_new_downgrade", lambda: ("0.45.0", "0.44.1"))
        # The optional-update path must never be reached for a too-new tool.
        monkeypatch.setattr(
            "ucode.agents._confirm_update_installed_tool_binary",
            lambda _: (_ for _ in ()).throw(AssertionError("should not reach optional update")),
        )
        monkeypatch.setattr(
            "ucode.agents.prompt_yes_no", lambda prompt: prompt_calls.append(prompt) or True
        )

        assert install_tool_binary("gemini", strict=False, update_existing=True) is True
        assert prompt_calls == ["Downgrade Gemini CLI from 0.45.0 to 0.44.1?"]
        assert calls == [["npm", "install", "-g", "@google/gemini-cli@0.44.1"]]
        out = capsys.readouterr().out
        assert "newer than the latest version known to work" in out

    def test_too_new_tool_warns_but_keeps_version_on_decline(self, monkeypatch, capsys):
        calls: list[list[str]] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents.gemini.too_new_downgrade", lambda: ("0.45.0", "0.44.1"))
        monkeypatch.setattr("ucode.agents.prompt_yes_no", lambda prompt: False)

        assert install_tool_binary("gemini", strict=False, update_existing=True) is True
        assert calls == []
        assert "newer than the latest version known to work" in capsys.readouterr().out

    def test_too_new_tool_warns_without_prompt_when_updates_disabled(self, monkeypatch, capsys):
        """With prompts suppressed we still warn, but never downgrade."""
        calls: list[list[str]] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents.gemini.too_new_downgrade", lambda: ("0.45.0", "0.44.1"))
        monkeypatch.setattr(
            "ucode.agents.prompt_yes_no",
            lambda prompt: (_ for _ in ()).throw(AssertionError("should not prompt")),
        )

        assert (
            install_tool_binary(
                "gemini",
                strict=False,
                update_existing=True,
                prompt_optional_updates=False,
            )
            is True
        )
        assert calls == []
        assert "newer than the latest version known to work" in capsys.readouterr().out

    def test_update_failure_keeps_existing_binary_available(self, monkeypatch):
        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(1, args[0])

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents._confirm_update_installed_tool_binary", lambda _: True)

        assert install_tool_binary("opencode", strict=True, update_existing=True) is True

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


class TestLaunchBudgetStatus:
    def test_prints_budget_status_for_codex(self, monkeypatch, capsys):
        launched: list[tuple[dict, list[str]]] = []
        monkeypatch.setenv("UCODE_BUDGET_POLICY", "/tmp/ucode-test-missing-policy.json")

        monkeypatch.setattr(
            agents_mod,
            "fetch_current_user_budget_spend_status",
            lambda workspace, profile=None, **kwargs: (
                [
                    {
                        "budget_id": "abcdef12-3456",
                        "spend_status": {
                            "spend": 3.5,
                            "effective_alert_threshold": 10,
                        },
                    }
                ],
                None,
            ),
        )
        monkeypatch.setattr(
            agents_mod._MODULES["codex"],
            "launch",
            lambda state, args: launched.append((state, args)),
        )

        state = {"workspace": "https://x.databricks.com", "profile": "acct-profile"}
        agents_mod.launch("codex", state, ["--search"])

        assert launched == [(state, ["--search"])]
        out = capsys.readouterr().out
        assert "Budget abcdef12" in out
        assert "$3.50 / $10.00 (35%)" in out
        assert "Claude Code under 80%" in out
        assert "Codex" in out and "at/after 80%" in out
        assert "Budget policy recommends Claude Code (current 35% < 80%)" in out

    def test_budget_policy_recommends_codex_at_or_after_80_percent(self, monkeypatch, capsys):
        launched: list[bool] = []
        monkeypatch.setenv("UCODE_BUDGET_POLICY", "/tmp/ucode-test-missing-policy.json")
        monkeypatch.setattr(
            agents_mod,
            "fetch_current_user_budget_spend_status",
            lambda workspace, profile=None, **kwargs: (
                [
                    {
                        "budget_id": "budget-1",
                        "spend_status": {
                            "spend": 8,
                            "effective_alert_threshold": 10,
                        },
                    }
                ],
                None,
            ),
        )
        monkeypatch.setattr(
            agents_mod._MODULES["claude"],
            "launch",
            lambda state, args: launched.append(True),
        )

        agents_mod.launch("claude", {"workspace": "https://x.databricks.com"}, [])

        assert launched == [True]
        out = capsys.readouterr().out
        assert "Budget policy recommends Codex (current 80% >= 80%)" in out

    def test_reads_custom_budget_policy_json(self, tmp_path, monkeypatch, capsys):
        policy_path = tmp_path / "budget-policy.json"
        calls: list[dict] = []
        policy_path.write_text(
            """
            {
              "name": "Team budget policy",
              "switch_percent": 60,
              "below": {"agent": "codex"},
              "at_or_above": {"agent": "claude"},
              "budget_ids": ["3f142f2c-c75d-495d-8b62-25f7ac37bf6d"],
              "budget_tags": {"demo": "true"},
              "filter_has_spending": true
            }
            """,
            encoding="utf-8",
        )
        monkeypatch.setenv("UCODE_BUDGET_POLICY", str(policy_path))
        monkeypatch.setattr(
            agents_mod,
            "fetch_current_user_budget_spend_status",
            lambda workspace, profile=None, **kwargs: (
                calls.append(kwargs)
                or (
                    [
                        {
                            "budget_id": "3f142f2c-c75d-495d-8b62-25f7ac37bf6d",
                            "spend_status": {
                                "spend": 5,
                                "effective_alert_threshold": 10,
                            },
                        }
                    ],
                    None,
                )
            ),
        )
        monkeypatch.setattr(
            agents_mod._MODULES["claude"],
            "launch",
            lambda state, args: None,
        )

        agents_mod.launch("claude", {"workspace": "https://x.databricks.com"}, [])

        out = capsys.readouterr().out
        assert "Codex under 60%" in out
        assert "Budget policy recommends Codex (current 50% < 60%)" in out
        assert calls == [
            {
                "account_host": None,
                "account_id": None,
                "account_profile": None,
                "filter_budget_ids": ["3f142f2c-c75d-495d-8b62-25f7ac37bf6d"],
                "filter_budget_tags": {"demo": "true"},
                "filter_has_spending": True,
            }
        ]

    def test_reads_tiered_budget_policy_json(self, tmp_path, monkeypatch, capsys):
        policy_path = tmp_path / "budget-policy.json"
        calls: list[dict] = []
        policy_path.write_text(
            """
            {
              "version": 1,
              "budget": "3f142f2c-c75d-495d-8b62-25f7ac37bf6d",
              "policy_name": "Demo tiered policy",
              "tiers": {
                "20": {"agent": "claude code", "model": "opus"},
                "60": {"agent": "claude code", "model": "sonnet"},
                "80": {"agent": "codex", "model": "gpt-5-4-mini"}
              }
            }
            """,
            encoding="utf-8",
        )
        monkeypatch.setenv("UCODE_BUDGET_POLICY", str(policy_path))
        monkeypatch.setattr(
            agents_mod,
            "fetch_current_user_budget_spend_status",
            lambda workspace, profile=None, **kwargs: (
                calls.append(kwargs)
                or (
                    [
                        {
                            "budget_id": "3f142f2c-c75d-495d-8b62-25f7ac37bf6d",
                            "spend_status": {
                                "spend": 250,
                                "effective_alert_threshold": 500,
                            },
                        }
                    ],
                    None,
                )
            ),
        )
        monkeypatch.setattr(
            agents_mod._MODULES["codex"],
            "launch",
            lambda state, args: None,
        )

        agents_mod.launch("codex", {"workspace": "https://x.databricks.com"}, [])

        out = capsys.readouterr().out
        assert "20% Claude Code/opus" in out
        assert "60% Claude" in out and "Code/sonnet" in out
        assert "80%" in out and "Codex/gpt-5-4-mini" in out
        assert "Budget policy recommends Claude Code / sonnet (current 50% < 60%)" in out
        assert calls == [
            {
                "account_host": None,
                "account_id": None,
                "account_profile": None,
                "filter_budget_ids": ["3f142f2c-c75d-495d-8b62-25f7ac37bf6d"],
                "filter_budget_tags": None,
                "filter_has_spending": False,
            }
        ]

    def test_prints_budget_warning_without_blocking_launch(self, monkeypatch, capsys):
        launched: list[bool] = []
        monkeypatch.setattr(
            agents_mod,
            "fetch_current_user_budget_spend_status",
            lambda workspace, profile=None, **kwargs: ([], "login required"),
        )
        monkeypatch.setattr(
            agents_mod._MODULES["claude"],
            "launch",
            lambda state, args: launched.append(True),
        )

        agents_mod.launch("claude", {"workspace": "https://x.databricks.com"}, [])

        assert launched == [True]
        assert "Budget spend unavailable: login required" in capsys.readouterr().out

    def test_skips_budget_status_for_other_tools(self, monkeypatch):
        fetched: list[bool] = []
        launched: list[bool] = []
        monkeypatch.setattr(
            agents_mod,
            "fetch_current_user_budget_spend_status",
            lambda workspace, profile=None, **kwargs: fetched.append(True) or ([], None),
        )
        monkeypatch.setattr(
            agents_mod._MODULES["gemini"],
            "launch",
            lambda state, args: launched.append(True),
        )

        agents_mod.launch("gemini", {"workspace": "https://x.databricks.com"}, [])

        assert fetched == []
        assert launched == [True]


class TestValidateAllToolsVerbosity:
    def _run(self, monkeypatch, capsys):
        from contextlib import nullcontext

        monkeypatch.setattr(agents_mod, "validate_tool", lambda tool: (True, ""))
        monkeypatch.setattr(agents_mod, "save_state", lambda s: None)
        monkeypatch.setattr(agents_mod, "spinner", lambda *_a, **_kw: nullcontext())
        agents_mod.validate_all_tools({"available_tools": ["codex"], "managed_configs": {}})
        return capsys.readouterr().out

    def test_normal_verbosity_renders_panels(self, monkeypatch, capsys):
        import ucode.ui as ui_mod

        monkeypatch.setattr(ui_mod, "_verbosity", "normal")
        out = self._run(monkeypatch, capsys)
        assert "Testing each tool with a quick message" in out
        assert "Ready" in out
        assert "Codex is working" in out

    def test_low_verbosity_omits_panels(self, monkeypatch, capsys):
        import ucode.ui as ui_mod

        monkeypatch.setattr(ui_mod, "_verbosity", "low")
        out = self._run(monkeypatch, capsys)
        assert "Validating..." in out
        assert "Testing each tool with a quick message" not in out
        assert "Ready" not in out
        # Per-tool success line is still printed.
        assert "Codex is working" in out
