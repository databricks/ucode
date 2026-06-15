"""Tests for CLI subcommand routing and passthrough args."""

from __future__ import annotations

import contextlib
import os
import re
import signal
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from ucode import cli
from ucode.cli import app
from ucode.databricks import MANAGED_CONFIG_VOLUME_PATH, MANAGED_POLICIES_VOLUME_PATH

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Drop SGR escape sequences so substring assertions match regardless of
    whether the runner forces color rendering (e.g. CI sets FORCE_COLOR=1,
    which makes rich split styled tokens like ``--agents`` with ANSI codes)."""
    return _ANSI_RE.sub("", text)


runner = CliRunner()

TOOLS = ["codex", "claude", "gemini", "opencode"]


@pytest.fixture(autouse=True)
def no_state_writes():
    """Prevent any test from writing to the real state file on disk."""
    with (
        patch("ucode.state.save_state"),
        patch("ucode.cli.save_state"),
        patch("ucode.agents.__init__.save_state"),
        patch("ucode.agents.codex.save_state"),
        patch("ucode.agents.claude.save_state"),
        patch("ucode.agents.gemini.save_state"),
        patch("ucode.agents.opencode.save_state"),
        patch("ucode.cli.local_budget_status", return_value={"state": "ok", "tool": "codex"}),
    ):
        yield


MINIMAL_STATE = {
    "workspace": "https://example.databricks.com",
    "base_urls": {
        "codex": "https://example.databricks.com/ai-gateway/codex",
        "claude": "https://example.databricks.com/ai-gateway/anthropic",
        "gemini": "https://example.databricks.com/ai-gateway/gemini",
        "opencode": "https://example.databricks.com/ai-gateway/opencode",
    },
    "claude_models": {
        "opus": "databricks-claude-opus-4-8",
        "sonnet": "databricks-claude-sonnet-4",
        "haiku": "databricks-claude-haiku-4-5",
    },
    "gemini_models": ["gemini-2.0-flash"],
    "codex_models": ["codex-mini"],
    "opencode_models": {"anthropic": ["databricks-claude-sonnet-4", "databricks-claude-haiku-4-5"]},
    "managed_configs": {},
    "available_tools": TOOLS,
}


class TestHelp:
    def test_help_lists_all_agent_subcommands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for tool in TOOLS:
            assert tool in result.output

    @pytest.mark.parametrize("tool", TOOLS)
    def test_subcommand_help(self, tool):
        result = runner.invoke(app, [tool, "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_configure_help_lists_agents_flag(self):
        result = runner.invoke(app, ["configure", "--help"])
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "--agents" in output
        assert "comma-separated list of agents" in output
        assert "--workspaces" in output

    def test_setup_help_lists_budget_command(self):
        result = runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0
        assert "budget" in _strip_ansi(result.output)


def _patch_launch(tool: str):
    """Return a context-manager stack that makes _launch_tool a no-op.

    load_state returns MINIMAL_STATE (workspace + tool already configured) so
    the auto-configure path is skipped entirely. configure_shared_state is
    also stubbed to avoid the launch-time refetch hitting the network.
    """
    return [
        patch("ucode.cli.ensure_bootstrap_dependencies"),
        patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
        patch(
            "ucode.cli.ensure_provider_state",
            return_value=MINIMAL_STATE,
        ),
        patch(
            "ucode.cli.configure_shared_state",
            return_value=MINIMAL_STATE,
        ),
        patch(
            "ucode.cli.resolve_launch_model",
            return_value=(MINIMAL_STATE, "databricks-claude-sonnet-4"),
        ),
        patch(
            "ucode.cli.configure_tool",
            return_value=MINIMAL_STATE,
        ),
        patch("ucode.cli.launch_agent"),
    ]


class TestSubcommandRouting:
    @pytest.mark.parametrize("tool", TOOLS)
    def test_subcommand_calls_correct_tool(self, tool):
        patches = _patch_launch(tool)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6] as mock_launch,
        ):
            result = runner.invoke(app, [tool])
        assert result.exit_code == 0, result.output
        mock_launch.assert_called_once()
        called_tool = mock_launch.call_args[0][0]
        assert called_tool == tool

    def test_codex_launch_shows_model_and_budget_panel(self):
        patches = _patch_launch("codex")
        budget_status = {
            "state": "ok",
            "tool": "codex",
            "spend_usd": 12.4,
            "limit_usd": 200,
            "remaining_usd": 187.6,
            "total_tokens": 456_000,
            "sessions": 6,
        }
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(MINIMAL_STATE, "databricks-claude-opus-4-8[1m]"),
            ),
            patches[5],
            patches[6],
            patch("ucode.cli.local_budget_status", return_value=budget_status),
        ):
            result = runner.invoke(app, ["codex"])

        assert result.exit_code == 0, result.output
        assert "ucode with Codex" in result.output
        assert "$12.40 / $200.00" in result.output
        assert "6% used" in result.output
        assert "$187.60" in result.output
        assert "Codex Tokens" in result.output
        assert "456.0K" in result.output
        assert "Codex Sessions" in result.output
        assert "6" in result.output

    def test_opencode_launch_shows_budget_panel(self):
        patches = _patch_launch("opencode")
        budget_status = {
            "state": "ok",
            "tool": "opencode",
            "spend_usd": 12.4,
            "limit_usd": 200,
            "remaining_usd": 187.6,
            "total_tokens": 456_000,
            "sessions": 6,
        }
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(MINIMAL_STATE, "databricks-claude-haiku-4-5"),
            ),
            patches[5],
            patches[6],
            patch("ucode.cli.local_budget_status", return_value=budget_status),
        ):
            result = runner.invoke(app, ["opencode"])

        assert result.exit_code == 0, result.output
        assert "ucode with OpenCode" in result.output
        assert "$12.40 / $200.00" in result.output
        assert "6% used" in result.output
        assert "Opencode Tokens" in result.output

    def test_no_agent_flag(self):
        """--agent flag must no longer exist."""
        result = runner.invoke(app, ["--agent", "claude"])
        assert result.exit_code != 0


class TestSetupBudgetCommand:
    @staticmethod
    def _admin_patches(stack):
        stack.enter_context(patch("ucode.cli.ensure_databricks_auth"))
        stack.enter_context(patch("ucode.cli.get_databricks_token", return_value="tok"))
        stack.enter_context(patch("ucode.cli.is_workspace_admin", return_value=True))

    def test_sets_daily_budget_for_current_workspace(self):
        state = {
            "workspace": "https://example.databricks.com",
            "available_tools": ["claude"],
            "claude_models": {"sonnet": "databricks-claude-sonnet-4"},
            "default_agent": "claude",
        }
        with contextlib.ExitStack() as stack:
            self._admin_patches(stack)
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            stack.enter_context(patch("ucode.cli.load_workspace_policy", return_value=None))
            choice = stack.enter_context(patch("ucode.cli.prompt_for_choice"))
            choice.side_effect = ["block", "claude", "databricks-claude-sonnet-4"]
            mock_save_policy = stack.enter_context(patch("ucode.cli.save_workspace_policy"))
            result = runner.invoke(app, ["setup", "budget"], input="250\n1\n")

        assert result.exit_code == 0, result.output
        workspace, policy = mock_save_policy.call_args[0]
        assert workspace == state["workspace"]
        assert policy["policy"]["name"] == "coding-agents-default"
        assert policy["policy"]["daily_budget_usd"] == 250.0
        assert policy["policy"]["on_budget_exhausted"] == "block"
        assert policy["policy"]["tiers"] == [
            {
                "name": "tier 1",
                "activates_at_pct": 0.0,
                "harness": "claude",
                "model": "databricks-claude-sonnet-4",
            }
        ]
        assert "Budget policy updated" in result.output

    def test_overrides_existing_daily_budget(self):
        state = {
            "workspace": "https://example.databricks.com",
            "available_tools": ["claude"],
            "claude_models": {"sonnet": "databricks-claude-sonnet-4"},
            "default_agent": "claude",
        }
        existing = {
            "policy": {
                "name": "existing",
                "daily_budget_usd": 125.0,
                "on_budget_exhausted": "warn",
                "tiers": [
                    {
                        "name": "premium",
                        "activates_at_pct": 0.0,
                        "harness": "claude",
                        "model": "old",
                    }
                ],
            }
        }
        with contextlib.ExitStack() as stack:
            self._admin_patches(stack)
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            stack.enter_context(patch("ucode.cli.load_workspace_policy", return_value=existing))
            choice = stack.enter_context(patch("ucode.cli.prompt_for_choice"))
            choice.side_effect = ["warn", "claude", "old"]
            mock_save_policy = stack.enter_context(patch("ucode.cli.save_workspace_policy"))
            result = runner.invoke(app, ["setup", "budget"], input="500\n1\n")

        assert result.exit_code == 0, result.output
        policy = mock_save_policy.call_args[0][1]
        assert policy["policy"]["name"] == "coding-agents-default"
        assert policy["policy"]["daily_budget_usd"] == 500.0
        assert policy["policy"]["on_budget_exhausted"] == "warn"
        assert policy["policy"]["tiers"][0]["name"] == "tier 1"
        assert policy["policy"]["tiers"][0]["model"] == "old"

    def test_selects_model_from_harness_inventory(self):
        state = {
            "workspace": "https://example.databricks.com",
            "available_tools": ["claude"],
            "claude_models": {
                "opus": "databricks-claude-opus-4-5",
                "sonnet": "databricks-claude-sonnet-4",
            },
            "default_agent": "claude",
        }
        captured: list[tuple[str, list[tuple[str, str]]]] = []

        def fake_choice(prompt, options):
            captured.append((prompt, options))
            if prompt == "At 100% of budget":
                return "block"
            if prompt == "Harness":
                return "claude"
            if prompt == "Model":
                return "databricks-claude-sonnet-4"
            raise AssertionError(prompt)

        with contextlib.ExitStack() as stack:
            self._admin_patches(stack)
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            stack.enter_context(patch("ucode.cli.load_workspace_policy", return_value=None))
            stack.enter_context(patch("ucode.cli.prompt_for_choice", side_effect=fake_choice))
            text_prompt = stack.enter_context(patch("ucode.cli._prompt_text"))
            mock_save_policy = stack.enter_context(patch("ucode.cli.save_workspace_policy"))
            result = runner.invoke(app, ["setup", "budget"], input="250\n1\n")

        assert result.exit_code == 0, result.output
        text_prompt.assert_not_called()
        model_prompt = [item for item in captured if item[0] == "Model"][0]
        assert model_prompt[1] == [
            ("databricks-claude-opus-4-5", "opus (databricks-claude-opus-4-5)"),
            ("databricks-claude-sonnet-4", "sonnet (databricks-claude-sonnet-4)"),
        ]
        policy = mock_save_policy.call_args[0][1]
        assert policy["policy"]["tiers"][0]["model"] == "databricks-claude-sonnet-4"

    def test_model_prompt_falls_back_to_text_when_no_inventory(self):
        state = {
            "workspace": "https://example.databricks.com",
            "available_tools": ["claude"],
            "claude_models": {},
            "default_agent": "claude",
        }

        def fake_choice(prompt, _options):
            if prompt == "At 100% of budget":
                return "block"
            if prompt == "Harness":
                return "claude"
            raise AssertionError(prompt)

        with contextlib.ExitStack() as stack:
            self._admin_patches(stack)
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            stack.enter_context(patch("ucode.cli.load_workspace_policy", return_value=None))
            stack.enter_context(patch("ucode.cli.prompt_for_choice", side_effect=fake_choice))
            stack.enter_context(patch("ucode.cli._prompt_text", return_value="manual-model"))
            mock_save_policy = stack.enter_context(patch("ucode.cli.save_workspace_policy"))
            result = runner.invoke(app, ["setup", "budget"], input="250\n1\n")

        assert result.exit_code == 0, result.output
        policy = mock_save_policy.call_args[0][1]
        assert policy["policy"]["tiers"][0]["model"] == "manual-model"

    def test_errors_when_workspace_missing(self):
        with patch("ucode.cli.load_state", return_value={}):
            result = runner.invoke(app, ["setup", "budget"])

        assert result.exit_code == 1
        assert "No workspace is configured" in result.output

    _VALID_POLICY_YAML = """
policy:
  name: coding-agents-default
  daily_budget_usd: 300
  tiers:
    - name: premium
      activates_at_pct: 0
      harness: claude
      model: databricks-claude-opus-4
  on_budget_exhausted: warn
"""

    def test_applies_policy_from_file(self, tmp_path):
        state = {
            "workspace": "https://example.databricks.com",
            "claude_models": {"opus": "databricks-claude-opus-4"},
        }
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(self._VALID_POLICY_YAML, encoding="utf-8")

        with contextlib.ExitStack() as stack:
            self._admin_patches(stack)
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            mock_save_policy = stack.enter_context(patch("ucode.cli.save_workspace_policy"))
            # No interactive input is supplied; the file path must bypass prompts.
            result = runner.invoke(app, ["setup", "budget", "-f", str(policy_file)])

        assert result.exit_code == 0, result.output
        workspace, policy = mock_save_policy.call_args[0]
        assert workspace == state["workspace"]
        assert policy["policy"]["daily_budget_usd"] == 300.0
        assert policy["policy"]["on_budget_exhausted"] == "warn"
        assert "Budget policy updated" in result.output

    def test_rejects_unsupported_harness_in_file(self, tmp_path):
        state = {"workspace": "https://example.databricks.com"}
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(
            "policy:\n  daily_budget_usd: 100\n  tiers:\n    - name: t1\n      "
            "activates_at_pct: 0\n      harness: cursor\n      model: whatever\n",
            encoding="utf-8",
        )

        with contextlib.ExitStack() as stack:
            self._admin_patches(stack)
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            mock_save_policy = stack.enter_context(patch("ucode.cli.save_workspace_policy"))
            result = runner.invoke(app, ["setup", "budget", "-f", str(policy_file)])

        assert result.exit_code == 1
        assert "Policy file is invalid" in result.output
        assert "cursor" in result.output
        mock_save_policy.assert_not_called()

    def test_rejects_unknown_model_when_inventory_known(self, tmp_path):
        state = {
            "workspace": "https://example.databricks.com",
            "claude_models": {"opus": "databricks-claude-opus-4"},
        }
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(
            "policy:\n  daily_budget_usd: 100\n  tiers:\n    - name: t1\n      "
            "activates_at_pct: 0\n      harness: claude\n      model: made-up-model\n",
            encoding="utf-8",
        )

        with contextlib.ExitStack() as stack:
            self._admin_patches(stack)
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            mock_save_policy = stack.enter_context(patch("ucode.cli.save_workspace_policy"))
            result = runner.invoke(app, ["setup", "budget", "-f", str(policy_file)])

        assert result.exit_code == 1
        assert "made-up-model" in result.output
        mock_save_policy.assert_not_called()

    def test_allows_free_text_model_when_inventory_empty(self, tmp_path):
        # No discovered inventory for the harness -> model is not constrained,
        # matching the interactive flow's free-text fallback.
        state = {"workspace": "https://example.databricks.com"}
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(self._VALID_POLICY_YAML, encoding="utf-8")

        with contextlib.ExitStack() as stack:
            self._admin_patches(stack)
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            mock_save_policy = stack.enter_context(patch("ucode.cli.save_workspace_policy"))
            result = runner.invoke(app, ["setup", "budget", "-f", str(policy_file)])

        assert result.exit_code == 0, result.output
        mock_save_policy.assert_called_once()

    def test_rejects_invalid_policy_file(self, tmp_path):
        state = {"workspace": "https://example.databricks.com"}
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(
            "policy:\n  daily_budget_usd: 0\n  tiers:\n    - name: a\n      "
            "activates_at_pct: 60\n      harness: claude\n      model: m\n",
            encoding="utf-8",
        )

        with contextlib.ExitStack() as stack:
            self._admin_patches(stack)
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            mock_save_policy = stack.enter_context(patch("ucode.cli.save_workspace_policy"))
            result = runner.invoke(app, ["setup", "budget", "-f", str(policy_file)])

        assert result.exit_code == 1
        assert "Policy file is invalid" in result.output
        mock_save_policy.assert_not_called()

    def test_errors_when_policy_file_missing(self, tmp_path):
        state = {"workspace": "https://example.databricks.com"}
        missing = tmp_path / "nope.yaml"

        with contextlib.ExitStack() as stack:
            self._admin_patches(stack)
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            result = runner.invoke(app, ["setup", "budget", "-f", str(missing)])

        assert result.exit_code == 1
        assert "Could not read policy file" in result.output


class TestSetupCommand:
    """`ucode setup` runs all four phases (agents, tracing, MCP, budget) and
    is admin-gated."""

    @staticmethod
    def _patch_preamble(stack, *, uc_tracing=False, uc_mcps=False, uc_policy=False):
        """Stub everything up to (and around) the four phase blocks, returning a
        dict of the phase-action mocks for the caller to assert against."""
        pre_pulled = {
            "workspace": "https://example.databricks.com",
            "profile": "my-profile",
            "available_tools": ["claude"],
            "tracing": {"enabled": uc_tracing},
            "mcp_servers": [{"name": "x"}] if uc_mcps else [],
        }
        stack.enter_context(patch("ucode.cli.install_databricks_cli"))
        stack.enter_context(
            patch(
                "ucode.cli._prompt_for_configuration",
                return_value=("https://example.databricks.com", "my-profile"),
            )
        )
        stack.enter_context(patch("ucode.cli.ensure_databricks_auth"))
        stack.enter_context(patch("ucode.cli.get_databricks_token", return_value="tok"))
        stack.enter_context(patch("ucode.cli.is_workspace_admin", return_value=True))
        stack.enter_context(patch("ucode.cli.configure_shared_state", return_value=pre_pulled))
        stack.enter_context(patch("ucode.cli._render_configuration_panel"))
        stack.enter_context(
            patch(
                "ucode.cli.load_workspace_policy",
                return_value={"policy": {}} if uc_policy else None,
            )
        )
        stack.enter_context(patch("ucode.cli.load_state", return_value=MINIMAL_STATE))
        # Agent phase internals.
        stack.enter_context(patch("ucode.cli.prompt_for_tools", return_value=["claude"]))
        mock_configure = stack.enter_context(
            patch("ucode.cli.configure_workspace_command", return_value=0)
        )
        stack.enter_context(patch("ucode.cli.save_state"))
        stack.enter_context(
            patch("ucode.cli._resolve_default_agent", return_value=("claude", "first tool"))
        )
        # Phase actions (tracing/mcp/budget).
        mock_tracing = stack.enter_context(patch("ucode.cli.configure_tracing_command"))
        mock_mcp = stack.enter_context(patch("ucode.cli.configure_mcp_command"))
        mock_budget = stack.enter_context(patch("ucode.cli._run_budget_setup"))
        return {
            "agents": mock_configure,
            "tracing": mock_tracing,
            "mcp": mock_mcp,
            "budget": mock_budget,
        }

    def test_runs_all_phases(self):
        with contextlib.ExitStack() as stack:
            mocks = self._patch_preamble(stack)
            # Three "yes" answers for the tracing / mcp / budget prompts.
            result = runner.invoke(app, ["setup"], input="y\ny\ny\n")

        assert result.exit_code == 0, result.output
        mocks["agents"].assert_called_once()
        mocks["tracing"].assert_called_once()
        mocks["mcp"].assert_called_once()
        mocks["budget"].assert_called_once()

    def test_blocks_non_admin(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("ucode.cli.install_databricks_cli"))
            stack.enter_context(
                patch(
                    "ucode.cli._prompt_for_configuration",
                    return_value=("https://example.databricks.com", "my-profile"),
                )
            )
            stack.enter_context(patch("ucode.cli.load_state", return_value=MINIMAL_STATE))
            stack.enter_context(patch("ucode.cli.ensure_databricks_auth"))
            stack.enter_context(patch("ucode.cli.get_databricks_token", return_value="tok"))
            stack.enter_context(patch("ucode.cli.is_workspace_admin", return_value=False))
            mock_budget = stack.enter_context(patch("ucode.cli._run_budget_setup"))
            result = runner.invoke(app, ["setup"])

        assert result.exit_code == 1, result.output
        assert "admin permissions" in result.output
        mock_budget.assert_not_called()


class TestDefaultLaunch:
    """Bare `ucode` launches the default agent, with budget-aware fallback."""

    def _launch_patches(self, stack):
        """Stub the launch pipeline so no network/disk work runs, returning the
        ``launch_agent`` mock for the caller to assert against."""
        stack.enter_context(patch("ucode.cli.install_databricks_cli"))
        stack.enter_context(patch("ucode.cli.ensure_bootstrap_dependencies"))
        stack.enter_context(patch("ucode.cli.ensure_provider_state", return_value=MINIMAL_STATE))
        stack.enter_context(patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE))
        stack.enter_context(
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(MINIMAL_STATE, "databricks-claude-sonnet-4"),
            )
        )
        stack.enter_context(patch("ucode.cli.configure_tool", return_value=MINIMAL_STATE))
        return stack.enter_context(patch("ucode.cli.launch_agent"))

    def test_launches_default_agent(self):
        from contextlib import ExitStack

        state = {**MINIMAL_STATE, "default_agent": "claude"}
        with ExitStack() as stack:
            mock_launch = self._launch_patches(stack)
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            stack.enter_context(
                patch(
                    "ucode.cli.local_budget_status", return_value={"state": "ok", "tool": "claude"}
                )
            )
            result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output
        mock_launch.assert_called_once()
        assert mock_launch.call_args[0][0] == "claude"

    def test_bare_launch_blocks_when_budget_exceeded(self):
        """The global budget is one pool, so bare `ucode` hard-stops when it's
        exhausted rather than falling back to another agent."""
        from contextlib import ExitStack

        state = {**MINIMAL_STATE, "default_agent": "codex", "available_tools": ["codex", "claude"]}

        with ExitStack() as stack:
            mock_launch = self._launch_patches(stack)
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            stack.enter_context(
                patch("ucode.cli.local_budget_status", return_value={"state": "exceeded"})
            )
            result = runner.invoke(app, [])
        assert result.exit_code == 1, result.output
        mock_launch.assert_not_called()
        assert "exhausted" in result.output

    def test_bare_launch_switches_agent_when_budget_exhausted_with_switch_policy(self):
        """When on_budget_exhausted is a switch dict, exhausted budget redirects to
        the target harness/model instead of blocking."""
        from contextlib import ExitStack

        state = {
            **MINIMAL_STATE,
            "default_agent": "claude",
            "available_tools": ["claude", "opencode"],
        }
        switch_policy = {
            "action": "switch",
            "target": {"harness": "opencode", "model": "databricks-claude-haiku-4-5"},
        }
        # First call returns exceeded+switch; all subsequent calls return ok so the
        # redirected opencode launch passes the gate without looping.
        call_count: list[int] = [0]

        def budget_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"state": "exceeded", "on_budget_exhausted": switch_policy}
            return {"state": "ok"}

        with ExitStack() as stack:
            mock_launch = self._launch_patches(stack)
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            stack.enter_context(
                patch("ucode.cli.local_budget_status", side_effect=budget_side_effect)
            )
            result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output
        assert "switching" in result.output.lower()
        mock_launch.assert_called()
        launched_tool = mock_launch.call_args[0][0]
        assert launched_tool == "opencode"

    def test_auto_configures_when_no_workspace(self):
        """Bare `ucode` with no workspace runs configure first, then launches."""
        from contextlib import ExitStack

        configured: list[bool] = []

        def fake_configure(**_kwargs):
            configured.append(True)
            return 0

        calls = {"n": 0}
        configured_state = {**MINIMAL_STATE, "default_agent": "codex"}

        def load_state_stub():
            calls["n"] += 1
            return {} if calls["n"] == 1 else configured_state

        with ExitStack() as stack:
            mock_launch = self._launch_patches(stack)
            stack.enter_context(
                patch("ucode.cli.configure_workspace_command", side_effect=fake_configure)
            )
            stack.enter_context(patch("ucode.cli.load_state", side_effect=load_state_stub))
            stack.enter_context(
                patch(
                    "ucode.cli.local_budget_status", return_value={"state": "ok", "tool": "codex"}
                )
            )
            result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output
        assert configured == [True]
        mock_launch.assert_called_once()
        assert mock_launch.call_args[0][0] == "codex"


class TestBudgetGate:
    """Budget gate at the launch boundary: exceeded hard-stops every agent,
    warn offers an interactive harness/model selector."""

    def _launch_patches(self, stack):
        """Stub the launch pipeline (same shape as TestDefaultLaunch), returning
        the (launch_agent, configure_tool) mocks for assertions."""
        stack.enter_context(patch("ucode.cli.install_databricks_cli"))
        stack.enter_context(patch("ucode.cli.ensure_bootstrap_dependencies"))
        stack.enter_context(patch("ucode.cli.ensure_provider_state", return_value=MINIMAL_STATE))
        stack.enter_context(patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE))
        stack.enter_context(patch("ucode.cli._auto_configure_tool"))

        def resolve(tool, state, explicit_model):
            return state, (explicit_model or "databricks-claude-sonnet-4")

        stack.enter_context(patch("ucode.cli.resolve_launch_model", side_effect=resolve))
        configure_tool = stack.enter_context(
            patch("ucode.cli.configure_tool", return_value=MINIMAL_STATE)
        )
        launch_agent = stack.enter_context(patch("ucode.cli.launch_agent"))
        return launch_agent, configure_tool

    def _run(self, stack, args, *, budget_state, choice="__unset__", state=None, status=None):
        state = state or {**MINIMAL_STATE, "default_agent": "claude"}
        launch_agent, configure_tool = self._launch_patches(stack)
        stack.enter_context(patch("ucode.cli.load_state", return_value=state))
        status = status or {"state": budget_state}
        stack.enter_context(patch("ucode.cli.local_budget_status", return_value=status))
        choice_mock = stack.enter_context(patch("ucode.cli.prompt_budget_warn_choice"))
        if choice != "__unset__":
            choice_mock.return_value = choice
        result = runner.invoke(app, args)
        return result, launch_agent, configure_tool, choice_mock

    def test_exceeded_blocks_tracked_agent(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            result, launch_agent, _, _ = self._run(stack, ["claude"], budget_state="exceeded")
        assert result.exit_code == 1, result.output
        launch_agent.assert_not_called()
        assert "exhausted" in result.output

    def test_exceeded_blocks_untracked_agent(self):
        """The exceeded hard stop applies to untracked agents too (closes the
        loophole where `ucode opencode` launched past the global cap)."""
        from contextlib import ExitStack

        with ExitStack() as stack:
            result, launch_agent, _, _ = self._run(stack, ["opencode"], budget_state="exceeded")
        assert result.exit_code == 1, result.output
        launch_agent.assert_not_called()

    def test_warn_selector_continue_default(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            result, launch_agent, _, _ = self._run(
                stack, ["claude"], budget_state="warn", choice="default"
            )
        assert result.exit_code == 0, result.output
        assert launch_agent.call_args[0][0] == "claude"

    def test_warn_selector_switches_to_policy_tier(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            result, launch_agent, configure_tool, _ = self._run(
                stack,
                ["claude"],
                budget_state="warn",
                choice="switch",
                status={
                    "state": "warn",
                    "active_tier": {
                        "name": "economy",
                        "harness": "opencode",
                        "model": "kimi-k2",
                    },
                },
            )
        assert result.exit_code == 0, result.output
        assert launch_agent.call_args[0][0] == "opencode"
        assert configure_tool.call_args[0][2] == "kimi-k2"

    def test_warn_selector_abort_none(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            result, launch_agent, _, _ = self._run(
                stack,
                ["claude"],
                budget_state="warn",
                choice=None,
                status={
                    "state": "warn",
                    "active_tier": {
                        "name": "economy",
                        "harness": "opencode",
                        "model": "kimi-k2",
                    },
                },
            )
        assert result.exit_code == 0, result.output
        launch_agent.assert_not_called()

    def test_warn_untracked_explicit_skips_selector(self):
        """`ucode opencode` in warn-state launches normally without prompting —
        the selector only makes sense for budget-tracked agents."""
        from contextlib import ExitStack

        with ExitStack() as stack:
            result, launch_agent, _, choice_mock = self._run(
                stack, ["opencode"], budget_state="warn"
            )
        assert result.exit_code == 0, result.output
        choice_mock.assert_not_called()
        assert launch_agent.call_args[0][0] == "opencode"

    def test_prompt_budget_warn_choice_options(self):
        """The ui helper offers the default-agent and policy-switch options,
        labeling the first with the passed display name."""
        from ucode.ui import prompt_budget_warn_choice

        captured = {}

        class FakeSelect:
            def ask(self):
                return "default"

        def fake_select(_message, *, choices, **_kwargs):
            captured["values"] = [c.value for c in choices]
            captured["titles"] = [c.title for c in choices]
            return FakeSelect()

        with patch("ucode.ui.questionary.select", side_effect=fake_select):
            prompt_budget_warn_choice(
                default_agent_display="Claude Code",
                switch_display="OpenCode / kimi-k2",
            )
        assert captured["values"] == ["switch", "default"]
        assert captured["titles"][0] == "Switch to OpenCode / kimi-k2 [Recommended]"
        assert captured["titles"][1] == "Continue with Claude Code"


class TestMcpSubcommands:
    def test_web_search_subcommand_help(self):
        result = runner.invoke(app, ["mcp", "web-search", "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_mcp_group_lists_web_search(self):
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "web-search" in result.output


class TestUsageCommands:
    def test_usage_default_uses_databricks_report(self):
        calls: list[str] = []
        with (
            patch("ucode.cli.install_databricks_cli", side_effect=lambda: calls.append("install")),
            patch("ucode.cli.usage_report", side_effect=lambda: calls.append("gateway")),
        ):
            result = runner.invoke(app, ["usage"])

        assert result.exit_code == 0, result.output
        assert calls == ["install", "gateway"]

    def test_usage_local_uses_local_report(self):
        calls: list[int] = []
        with patch("ucode.cli.local_usage_report", side_effect=lambda days: calls.append(days)):
            result = runner.invoke(app, ["usage", "--local", "--days", "3"])

        assert result.exit_code == 0, result.output
        assert calls == [3]

    def test_usage_record_overwrites_daily_spend(self):
        calls: list[float] = []
        with patch(
            "ucode.cli.set_local_daily_spend",
            side_effect=lambda cost_usd: calls.append(cost_usd) or {"cost_usd": cost_usd},
        ):
            result = runner.invoke(app, ["usage", "record", "50"])

        assert result.exit_code == 0, result.output
        assert calls == [50.0]
        assert "Today's local spend overwritten to $50.00" in result.output

    def test_usage_record_rejects_negative(self):
        # `--` ends typer/click's option parsing so the `-5` is forwarded as the
        # positional `cost` argument instead of being mistaken for a flag.
        with patch(
            "ucode.cli.set_local_daily_spend",
            side_effect=RuntimeError("cost_usd must be non-negative."),
        ):
            result = runner.invoke(app, ["usage", "record", "--", "-5"])

        assert result.exit_code == 2
        assert "non-negative" in result.output.lower()

    def test_budget_check_exits_nonzero_when_exceeded(self):
        message = "⛔ [UCODE USAGE BUDGET] Codex daily budget exceeded.\nBudget: $2.00 / $1.00"
        with (
            patch("ucode.cli.local_budget_status", return_value={"state": "exceeded"}),
            patch("ucode.cli.format_local_budget_status", return_value=message),
        ):
            result = runner.invoke(app, ["usage", "budget-check", "--agent", "codex"])

        assert result.exit_code == 1
        assert "╭" in result.output
        assert "Codex daily budget exceeded" in result.output
        assert "Budget: $2.00 / $1.00" in result.output
        assert "ERROR" not in result.output

    def test_usage_hook_claude_outputs_json(self):
        with patch("ucode.cli.claude_usage_hook", return_value={"decision": "block"}):
            result = runner.invoke(
                app,
                [
                    "usage",
                    "hook",
                    "claude",
                    "prompt-submit",
                    "--model",
                    "databricks-claude-sonnet-4",
                ],
                input="{}",
            )

        assert result.exit_code == 0, result.output
        assert result.output.strip() == '{"decision": "block"}'

    def test_usage_hook_codex_prompt_warning_outputs_valid_json(self):
        with patch(
            "ucode.cli.codex_usage_hook",
            return_value={
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": "warning text",
                }
            },
        ):
            result = runner.invoke(
                app,
                ["usage", "hook", "codex", "prompt-submit", "--model", "gpt-5.5"],
                input="{}",
            )

        assert result.exit_code == 0, result.output
        assert (
            result.output.strip()
            == '{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "warning text"}}'
        )

    def test_usage_hook_opencode_is_unsupported(self):
        result = runner.invoke(
            app,
            ["usage", "hook", "opencode", "chat-params", "--model", "databricks-claude"],
            input="{}",
        )

        assert result.exit_code == 0, result.output
        assert "Unsupported usage hook" in result.output


class TestStatus:
    def test_shows_mcp_list_commands(self):
        with patch("ucode.cli.load_state", return_value=MINIMAL_STATE):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "Managed by Databricks" not in result.output
        assert "MCP list command:" in result.output
        assert "claude mcp list" in result.output
        assert "codex mcp list" in result.output
        assert "gemini mcp list" in result.output
        assert "opencode mcp list" in result.output
        assert "copilot mcp list" not in result.output

    def test_shows_mcp_servers_configured_by_ucode(self):
        state = {
            **MINIMAL_STATE,
            "mcp_servers": [
                {
                    "name": "github-mcp",
                    "url": "https://example.databricks.com/api/2.0/mcp/external/github-mcp",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["claude", "codex"],
                },
                {
                    "name": "databricks-sql",
                    "url": "https://example.databricks.com/api/2.0/mcp/sql",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["gemini"],
                },
            ],
        }
        with patch("ucode.cli.load_state", return_value=state):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "github-mcp" in result.output
        assert "MCP servers: github-mcp" in result.output
        assert "databricks-sql" in result.output
        assert "MCP servers: databricks-sql" in result.output
        assert "MCP Servers" not in result.output
        assert "MCP Server:" not in result.output
        assert "Configured tools:" not in result.output

    def test_status_treats_available_tools_as_configured_agents(self):
        state = {
            **MINIMAL_STATE,
            "available_tools": ["copilot"],
            "base_urls": {
                **MINIMAL_STATE["base_urls"],
                "copilot": "https://example.databricks.com/ai-gateway/copilot",
            },
            "mcp_servers": [
                {
                    "name": "databricks-sql",
                    "url": "https://example.databricks.com/api/2.0/mcp/sql",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["copilot"],
                }
            ],
        }
        with patch("ucode.cli.load_state", return_value=state):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "copilot mcp list" in result.output
        assert "MCP servers: databricks-sql" in result.output
        assert "codex mcp list" not in result.output
        assert "claude mcp list" not in result.output
        assert "gemini mcp list" not in result.output
        assert "https://example.databricks.com/ai-gateway/anthropic" not in result.output
        assert "https://example.databricks.com/ai-gateway/gemini" not in result.output


class TestRevert:
    def test_reverts_mcp_configs_before_clearing_state(self):
        state = {
            **MINIMAL_STATE,
            "mcp_servers": [{"name": "github-mcp", "clients": ["claude"]}],
        }
        reverted_mcp: list[dict] = []
        cleared: list[bool] = []

        with (
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.restore_file", return_value=False),
            patch(
                "ucode.cli.revert_mcp_configs",
                side_effect=lambda loaded_state: (
                    reverted_mcp.append(loaded_state) or {"claude": True}
                ),
            ),
            patch("ucode.cli.clear_state", side_effect=lambda: cleared.append(True)),
        ):
            result = runner.invoke(app, ["revert"])

        assert result.exit_code == 0, result.output
        assert reverted_mcp == [state]
        assert cleared == [True]
        assert "Claude Code MCP config: restored" in result.output


class TestUnionDiscoveredModels:
    """The managed config must not be able to HIDE a deployed model: live
    discovery is unioned back in after the managed merge so an admin can still
    see (and add) newly-deployed endpoints."""

    def test_appends_discovered_open_responses_missing_from_managed(self):
        from ucode.cli import _union_discovered_models

        # Managed blob lists only -colo; discovery found both.
        state = {"opencode_models": {"open-responses": ["databricks-kimi-k2-6-colo"]}}
        discovered = {"open-responses": ["databricks-kimi-k2-6", "databricks-kimi-k2-6-colo"]}
        _union_discovered_models(state, None, None, None, discovered)
        # Managed order preserved, missing endpoint appended (no duplicates).
        assert state["opencode_models"]["open-responses"] == [
            "databricks-kimi-k2-6-colo",
            "databricks-kimi-k2-6",
        ]

    def test_preserves_managed_order_and_dedupes(self):
        from ucode.cli import _union_into

        assert _union_into(["b", "a"], ["a", "b", "c"]) == ["b", "a", "c"]

    def test_unions_gemini_and_codex_lists(self):
        from ucode.cli import _union_discovered_models

        state = {"gemini_models": ["g1"], "codex_models": ["c1"]}
        _union_discovered_models(state, None, ["g1", "g2"], ["c1", "c2"], None)
        assert state["gemini_models"] == ["g1", "g2"]
        assert state["codex_models"] == ["c1", "c2"]

    def test_unions_claude_models_by_id(self):
        from ucode.cli import _union_discovered_models

        # claude_models maps tier -> model id; union on ids, keep managed entries.
        state = {"claude_models": {"opus": "claude-opus"}}
        discovered = {"opus": "claude-opus", "haiku": "claude-haiku"}
        _union_discovered_models(state, discovered, None, None, None)
        assert state["claude_models"] == {"opus": "claude-opus", "haiku": "claude-haiku"}

    def test_seeds_lists_when_managed_blob_has_none(self):
        from ucode.cli import _union_discovered_models

        state: dict = {}
        _union_discovered_models(
            state, None, None, None, {"open-responses": ["databricks-kimi-k2-6"]}
        )
        assert state["opencode_models"]["open-responses"] == ["databricks-kimi-k2-6"]


class TestExport:
    """`ucode export` must upload a flat single-workspace splice to UC —
    no multi-workspace wrapper, no other workspaces, no per-machine fields."""

    @staticmethod
    def _full_state_with(workspace_block: dict) -> dict:
        ws = workspace_block.get("workspace", "https://example.databricks.com")
        return {
            "state_version": 3,
            "current_workspace": ws,
            "workspaces": {
                ws: workspace_block,
                "https://other.databricks.com": {
                    "workspace": "https://other.databricks.com",
                    "claude_models": {"opus": "should-not-be-uploaded"},
                },
            },
            "mcp_servers": [],
        }

    @staticmethod
    def _admin_patches(stack, *, admin: bool | None = True):
        stack.enter_context(patch("ucode.cli.get_databricks_token", return_value="tok"))
        stack.enter_context(patch("ucode.cli.is_workspace_admin", return_value=admin))

    def test_uploads_state_to_uc_volume(self):
        state = {**MINIMAL_STATE, "profile": "my-profile"}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("ucode.cli.install_databricks_cli"))
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            stack.enter_context(
                patch("ucode.cli.load_full_state", return_value=self._full_state_with(state))
            )
            mock_auth = stack.enter_context(patch("ucode.cli.ensure_databricks_auth"))
            self._admin_patches(stack)
            mock_upload = stack.enter_context(patch("ucode.cli.upload_managed_config"))
            mock_policy_path = stack.enter_context(patch("ucode.cli.policy_cache_path"))
            mock_policy_path.return_value.is_file.return_value = False
            result = runner.invoke(app, ["export"], input="y\n")

        assert result.exit_code == 0, result.output
        mock_auth.assert_called_once_with("https://example.databricks.com", "my-profile")
        mock_upload.assert_called_once()
        upload_args = mock_upload.call_args[0]
        assert upload_args[0] == "https://example.databricks.com"
        assert upload_args[1] == "my-profile"
        assert "state.json uploaded" in result.output
        assert MANAGED_CONFIG_VOLUME_PATH in result.output
        assert "No local policies.yaml found" in result.output

    def test_uploads_flat_single_workspace_blob(self):
        """The temp file handed to `upload_managed_config` must be the FLAT
        slice — no `workspaces` wrapper, no `current_workspace`, no other
        workspaces, no per-machine fields like `mcp_servers`."""
        import json as _json

        state = {**MINIMAL_STATE, "policies": {"claude": {"default_model": "admin"}}}
        captured: dict = {}

        def fake_upload(workspace, profile, path):
            captured.update(_json.loads(path.read_text(encoding="utf-8")))

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("ucode.cli.install_databricks_cli"))
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            stack.enter_context(
                patch("ucode.cli.load_full_state", return_value=self._full_state_with(state))
            )
            stack.enter_context(patch("ucode.cli.ensure_databricks_auth"))
            self._admin_patches(stack)
            stack.enter_context(patch("ucode.cli.upload_managed_config", side_effect=fake_upload))
            mock_policy_path = stack.enter_context(patch("ucode.cli.policy_cache_path"))
            mock_policy_path.return_value.is_file.return_value = False
            result = runner.invoke(app, ["export"], input="y\n")

        assert result.exit_code == 0, result.output
        assert "workspaces" not in captured
        assert "current_workspace" not in captured
        assert "mcp_servers" not in captured
        # Workspace is present, policy is uploaded separately.
        assert captured["workspace"] == "https://example.databricks.com"
        assert "policies" not in captured
        assert captured["state_version"] == 3
        assert "should-not-be-uploaded" not in _json.dumps(captured)

    def test_uploads_policy_yaml_when_present(self):
        state = {**MINIMAL_STATE, "profile": "my-profile"}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("ucode.cli.install_databricks_cli"))
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            stack.enter_context(
                patch("ucode.cli.load_full_state", return_value=self._full_state_with(state))
            )
            stack.enter_context(patch("ucode.cli.ensure_databricks_auth"))
            self._admin_patches(stack)
            mock_config_upload = stack.enter_context(patch("ucode.cli.upload_managed_config"))
            mock_policy_upload = stack.enter_context(patch("ucode.cli.upload_managed_policies"))
            mock_policy_path = stack.enter_context(patch("ucode.cli.policy_cache_path"))
            mock_policy_path.return_value.is_file.return_value = True
            result = runner.invoke(app, ["export"], input="y\n")

        assert result.exit_code == 0, result.output
        mock_config_upload.assert_called_once()
        mock_policy_upload.assert_called_once()
        assert mock_policy_upload.call_args[0][0] == "https://example.databricks.com"
        assert mock_policy_upload.call_args[0][1] == "my-profile"
        assert mock_policy_upload.call_args[0][2] == mock_policy_path.return_value
        assert MANAGED_CONFIG_VOLUME_PATH in result.output
        assert MANAGED_POLICIES_VOLUME_PATH in result.output
        assert "state.json uploaded" in result.output
        assert "policies.yaml uploaded" in result.output

    def test_uploads_without_profile(self):
        state = {k: v for k, v in MINIMAL_STATE.items() if k != "profile"}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("ucode.cli.install_databricks_cli"))
            stack.enter_context(patch("ucode.cli.load_state", return_value=state))
            stack.enter_context(
                patch("ucode.cli.load_full_state", return_value=self._full_state_with(state))
            )
            stack.enter_context(patch("ucode.cli.ensure_databricks_auth"))
            self._admin_patches(stack)
            mock_upload = stack.enter_context(patch("ucode.cli.upload_managed_config"))
            result = runner.invoke(app, ["export"], input="y\n")

        assert result.exit_code == 0, result.output
        assert mock_upload.call_args[0][1] is None

    def test_errors_when_no_workspace_configured(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.load_state", return_value={}),
            patch("ucode.cli.upload_managed_config") as mock_upload,
        ):
            result = runner.invoke(app, ["export"])

        assert result.exit_code == 1
        assert "No workspace is configured" in result.output
        mock_upload.assert_not_called()

    def test_blocks_non_admin(self):
        """`ucode export` must refuse to publish when SCIM Me reports non-admin
        — otherwise a developer could overwrite the whole org's managed config."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("ucode.cli.install_databricks_cli"))
            stack.enter_context(patch("ucode.cli.load_state", return_value=MINIMAL_STATE))
            stack.enter_context(
                patch(
                    "ucode.cli.load_full_state", return_value=self._full_state_with(MINIMAL_STATE)
                )
            )
            stack.enter_context(patch("ucode.cli.ensure_databricks_auth"))
            self._admin_patches(stack, admin=False)
            mock_upload = stack.enter_context(patch("ucode.cli.upload_managed_config"))
            result = runner.invoke(app, ["export"])

        assert result.exit_code == 1, result.output
        assert "admin permissions" in result.output
        mock_upload.assert_not_called()

    def test_surfaces_upload_failure_as_error(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("ucode.cli.install_databricks_cli"))
            stack.enter_context(patch("ucode.cli.load_state", return_value=MINIMAL_STATE))
            stack.enter_context(
                patch(
                    "ucode.cli.load_full_state",
                    return_value=self._full_state_with(MINIMAL_STATE),
                )
            )
            stack.enter_context(patch("ucode.cli.ensure_databricks_auth"))
            self._admin_patches(stack)
            stack.enter_context(
                patch(
                    "ucode.cli.upload_managed_config",
                    side_effect=RuntimeError("PERMISSION_DENIED: not an admin"),
                )
            )
            result = runner.invoke(app, ["export"], input="y\n")

        assert result.exit_code == 1
        assert "PERMISSION_DENIED" in result.output

    def test_cancel_at_confirmation_prompt(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("ucode.cli.install_databricks_cli"))
            stack.enter_context(patch("ucode.cli.load_state", return_value=MINIMAL_STATE))
            stack.enter_context(
                patch(
                    "ucode.cli.load_full_state",
                    return_value=self._full_state_with(MINIMAL_STATE),
                )
            )
            stack.enter_context(patch("ucode.cli.ensure_databricks_auth"))
            self._admin_patches(stack)
            mock_upload = stack.enter_context(patch("ucode.cli.upload_managed_config"))
            result = runner.invoke(app, ["export"], input="n\n")

        assert result.exit_code == 0, result.output
        assert "Export cancelled" in result.output
        mock_upload.assert_not_called()


class TestDeleteConfiguration:
    """`ucode delete-configuration` prompts for the target workspace, removes
    both UC files, is admin-gated, and requires a double confirmation before
    touching Unity Catalog."""

    _WORKSPACE = "https://example.databricks.com"

    @classmethod
    def _full_state(cls, *extra_workspaces: str) -> dict:
        workspaces = {cls._WORKSPACE: {"profile": "my-profile"}}
        for ws in extra_workspaces:
            workspaces[ws] = {"profile": None}
        return {
            "state_version": 3,
            "current_workspace": cls._WORKSPACE,
            "workspaces": workspaces,
        }

    @classmethod
    def _patch_resolution(cls, stack, *, admin: bool | None = True, picked: str | None = None):
        """Stub CLI install, full-state load, the workspace picker, auth, and the
        admin check. The picker returns ``picked`` (defaults to the example
        workspace) with a profile."""
        stack.enter_context(patch("ucode.cli.install_databricks_cli"))
        stack.enter_context(patch("ucode.cli.load_full_state", return_value=cls._full_state()))
        stack.enter_context(
            patch(
                "ucode.cli.prompt_for_workspace",
                return_value=(picked or cls._WORKSPACE, "my-profile"),
            )
        )
        stack.enter_context(patch("ucode.cli.ensure_databricks_auth"))
        stack.enter_context(patch("ucode.cli.get_databricks_token", return_value="tok"))
        stack.enter_context(patch("ucode.cli.is_workspace_admin", return_value=admin))

    def test_deletes_both_uc_files_after_double_confirmation(self):
        with contextlib.ExitStack() as stack:
            self._patch_resolution(stack)
            mock_config = stack.enter_context(
                patch("ucode.cli.delete_managed_config", return_value=True)
            )
            mock_policy = stack.enter_context(
                patch("ucode.cli.delete_managed_policies", return_value=True)
            )
            result = runner.invoke(app, ["delete-configuration"], input="y\ny\n")

        assert result.exit_code == 0, result.output
        mock_config.assert_called_once_with(self._WORKSPACE, "my-profile")
        mock_policy.assert_called_once_with(self._WORKSPACE, "my-profile")
        assert MANAGED_CONFIG_VOLUME_PATH in result.output
        assert MANAGED_POLICIES_VOLUME_PATH in result.output
        assert "Managed configuration deleted" in result.output

    def test_prompts_for_workspace_before_deleting(self):
        """The chosen workspace — not necessarily current_workspace — is the
        delete target."""
        other = "https://other.databricks.com"
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("ucode.cli.install_databricks_cli"))
            stack.enter_context(
                patch("ucode.cli.load_full_state", return_value=self._full_state(other))
            )
            mock_prompt = stack.enter_context(
                patch("ucode.cli.prompt_for_workspace", return_value=(other, None))
            )
            stack.enter_context(patch("ucode.cli.ensure_databricks_auth"))
            stack.enter_context(patch("ucode.cli.get_databricks_token", return_value="tok"))
            stack.enter_context(patch("ucode.cli.is_workspace_admin", return_value=True))
            mock_config = stack.enter_context(
                patch("ucode.cli.delete_managed_config", return_value=True)
            )
            stack.enter_context(patch("ucode.cli.delete_managed_policies", return_value=True))
            result = runner.invoke(app, ["delete-configuration"], input="y\ny\n")

        assert result.exit_code == 0, result.output
        mock_prompt.assert_called_once()
        # Profile coerced from "" -> None for the chosen workspace.
        mock_config.assert_called_once_with(other, None)

    def test_cancel_at_first_prompt_deletes_nothing(self):
        with contextlib.ExitStack() as stack:
            self._patch_resolution(stack)
            mock_config = stack.enter_context(patch("ucode.cli.delete_managed_config"))
            mock_policy = stack.enter_context(patch("ucode.cli.delete_managed_policies"))
            result = runner.invoke(app, ["delete-configuration"], input="n\n")

        assert result.exit_code == 0, result.output
        assert "Delete cancelled" in result.output
        mock_config.assert_not_called()
        mock_policy.assert_not_called()

    def test_cancel_at_second_prompt_deletes_nothing(self):
        with contextlib.ExitStack() as stack:
            self._patch_resolution(stack)
            mock_config = stack.enter_context(patch("ucode.cli.delete_managed_config"))
            mock_policy = stack.enter_context(patch("ucode.cli.delete_managed_policies"))
            result = runner.invoke(app, ["delete-configuration"], input="y\nn\n")

        assert result.exit_code == 0, result.output
        assert "Delete cancelled" in result.output
        mock_config.assert_not_called()
        mock_policy.assert_not_called()

    def test_blocks_non_admin(self):
        with contextlib.ExitStack() as stack:
            self._patch_resolution(stack, admin=False)
            mock_config = stack.enter_context(patch("ucode.cli.delete_managed_config"))
            result = runner.invoke(app, ["delete-configuration"], input="y\ny\n")

        assert result.exit_code == 1, result.output
        assert "admin permissions" in result.output
        mock_config.assert_not_called()

    def test_errors_when_no_workspace_configured(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("ucode.cli.install_databricks_cli"))
            stack.enter_context(
                patch(
                    "ucode.cli.load_full_state",
                    return_value={"state_version": 3, "current_workspace": None, "workspaces": {}},
                )
            )
            mock_prompt = stack.enter_context(patch("ucode.cli.prompt_for_workspace"))
            mock_config = stack.enter_context(patch("ucode.cli.delete_managed_config"))
            result = runner.invoke(app, ["delete-configuration"])

        assert result.exit_code == 1
        assert "No workspace is configured" in result.output
        mock_prompt.assert_not_called()
        mock_config.assert_not_called()

    def test_reports_when_nothing_to_delete(self):
        with contextlib.ExitStack() as stack:
            self._patch_resolution(stack)
            stack.enter_context(patch("ucode.cli.delete_managed_config", return_value=False))
            stack.enter_context(patch("ucode.cli.delete_managed_policies", return_value=False))
            result = runner.invoke(app, ["delete-configuration"], input="y\ny\n")

        assert result.exit_code == 0, result.output
        assert "Nothing to delete" in result.output


class TestAutoConfigureOnFirstRun:
    def test_triggers_when_no_workspace(self):
        """Auto-configure runs when state has no workspace."""
        empty_state = {}
        configured_state = {**MINIMAL_STATE}
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies") as mock_bootstrap,
            patch("ucode.cli.load_state", return_value=empty_state),
            patch("ucode.cli._auto_configure_tool") as mock_auto,
            patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE),
            patch(
                "ucode.cli.ensure_provider_state",
                return_value=configured_state,
            ),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(configured_state, "databricks-claude-sonnet-4"),
            ),
            patch("ucode.cli.configure_tool", return_value=configured_state),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, ["claude"])
        assert result.exit_code == 0, result.output
        mock_bootstrap.assert_called_once_with(
            "claude", update_existing=True, prompt_optional_updates=True
        )
        mock_auto.assert_called_once_with("claude")

    def test_triggers_when_tool_not_in_available_tools(self):
        """Auto-configure runs when workspace exists but the tool wasn't configured."""
        state_without_tool = {**MINIMAL_STATE, "available_tools": ["codex"]}
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies") as mock_bootstrap,
            patch("ucode.cli.load_state", return_value=state_without_tool),
            patch("ucode.cli._auto_configure_tool") as mock_auto,
            patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE),
            patch(
                "ucode.cli.ensure_provider_state",
                return_value=MINIMAL_STATE,
            ),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(MINIMAL_STATE, "databricks-claude-sonnet-4"),
            ),
            patch("ucode.cli.configure_tool", return_value=MINIMAL_STATE),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, ["claude"])
        assert result.exit_code == 0, result.output
        mock_bootstrap.assert_called_once_with(
            "claude", update_existing=True, prompt_optional_updates=True
        )
        mock_auto.assert_called_once_with("claude")

    def test_skipped_when_already_configured(self):
        """Auto-configure is skipped when workspace and tool are already set up."""
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies") as mock_bootstrap,
            patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
            patch("ucode.cli._auto_configure_tool") as mock_auto,
            patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE),
            patch(
                "ucode.cli.ensure_provider_state",
                return_value=MINIMAL_STATE,
            ),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(MINIMAL_STATE, "databricks-claude-sonnet-4"),
            ),
            patch("ucode.cli.configure_tool", return_value=MINIMAL_STATE),
            patch("ucode.cli.launch_agent"),
        ):
            runner.invoke(app, ["claude"])
        mock_bootstrap.assert_called_once_with(
            "claude", update_existing=False, prompt_optional_updates=True
        )
        mock_auto.assert_not_called()


class TestPassthroughArgs:
    @pytest.mark.parametrize(
        "tool,extra_args",
        [
            ("claude", ["-r"]),
            ("claude", ["--resume"]),
            ("codex", ["--full-auto"]),
            ("gemini", ["--debug"]),
            ("opencode", ["--model", "my-model"]),
            ("claude", ["-r", "--some-flag", "value"]),
        ],
    )
    def test_extra_args_forwarded(self, tool, extra_args):
        patches = _patch_launch(tool)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6] as mock_launch,
        ):
            result = runner.invoke(app, [tool, *extra_args])
        assert result.exit_code == 0, result.output
        forwarded = mock_launch.call_args[0][2]
        assert forwarded == extra_args

    def test_no_extra_args_passes_empty_list(self):
        patches = _patch_launch("claude")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6] as mock_launch,
        ):
            runner.invoke(app, ["claude"])
        forwarded = mock_launch.call_args[0][2]
        assert forwarded == []


class TestWatchFlag:
    """`ucode <agent> --watch` wraps the launch in a tmux split with a budget watcher."""

    @staticmethod
    def _tmux_argv(mock_run):
        assert mock_run.called, "tmux subprocess.run was not invoked"
        return mock_run.call_args[0][0]

    def test_watch_flag_invokes_tmux(self):
        with (
            patch("ucode.cli._tmux_available", return_value=True),
            patch("ucode.cli._in_tmux", return_value=False),
            patch("ucode.cli.shutil.get_terminal_size", return_value=os.terminal_size((160, 40))),
            patch("ucode.cli.subprocess.run") as mock_run,
            patch("ucode.cli.launch_agent") as mock_launch,
        ):
            mock_run.return_value.returncode = 0
            result = runner.invoke(app, ["claude", "--watch"])
        assert result.exit_code == 0, result.output
        argv = self._tmux_argv(mock_run)
        assert argv[0] == "tmux"
        assert "new-session" in argv
        assert "split-window" in argv
        assert "-h" in argv
        assert "-l" in argv
        assert str(cli.WATCH_PANE_PREFERRED_WIDTH) in argv
        assert any("budget-watch" in part for part in argv)
        # Mouse mode is enabled so the wheel scrolls pane history.
        assert "set-option" in argv
        assert "mouse" in argv
        # We return early before the in-place launch.
        mock_launch.assert_not_called()

    def test_watch_uses_preferred_right_pane_when_terminal_is_wide(self):
        assert cli._watch_tmux_layout_args(160) == (
            ["split-window", "-h", "-l", str(cli.WATCH_PANE_PREFERRED_WIDTH)],
            ["select-pane", "-L"],
        )

    def test_watch_uses_available_width_when_terminal_is_tight_but_still_wide_enough(self):
        # 90 cols - 40 agent floor = 50 available, capped at the preferred width.
        assert cli._watch_tmux_layout_args(90) == (
            ["split-window", "-h", "-l", "50"],
            ["select-pane", "-L"],
        )

    def test_watch_stays_on_right_when_terminal_is_narrow(self):
        # Zoomed-in / narrow terminal: still a right-side split (never bottom),
        # floored at the narrow width so the adaptive panel still fits.
        assert cli._watch_tmux_layout_args(60) == (
            ["split-window", "-h", "-l", str(cli.WATCH_PANE_NARROW_WIDTH)],
            ["select-pane", "-L"],
        )

    def test_watch_default_interval_is_5(self):
        with (
            patch("ucode.cli._tmux_available", return_value=True),
            patch("ucode.cli._in_tmux", return_value=False),
            patch("ucode.cli.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            runner.invoke(app, ["claude", "--watch"])
        argv = self._tmux_argv(mock_run)
        assert any("budget-watch --interval 5" in part for part in argv)

    def test_watch_interval_custom(self):
        with (
            patch("ucode.cli._tmux_available", return_value=True),
            patch("ucode.cli._in_tmux", return_value=False),
            patch("ucode.cli.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            runner.invoke(app, ["claude", "--watch", "--watch-interval", "30"])
        argv = self._tmux_argv(mock_run)
        assert any("budget-watch --interval 30" in part for part in argv)

    def test_watch_interval_equals_form(self):
        with (
            patch("ucode.cli._tmux_available", return_value=True),
            patch("ucode.cli._in_tmux", return_value=False),
            patch("ucode.cli.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            runner.invoke(app, ["claude", "--watch", "--watch-interval=45"])
        argv = self._tmux_argv(mock_run)
        assert any("budget-watch --interval 45" in part for part in argv)

    def test_budget_watch_refetches_on_timer_and_reflows_on_resize(self):
        captured: dict[str, object] = {}
        sleeps = {"n": 0}

        def fake_signal(signum, handler):
            captured["signum"] = signum
            captured["handler"] = handler

        def fake_sleep(_secs):
            # First poll slice: fire the resize handler the command registered.
            # Second slice: break out of the (otherwise infinite) loop.
            sleeps["n"] += 1
            if sleeps["n"] == 1:
                captured["handler"](signal.SIGWINCH, None)
            else:
                raise KeyboardInterrupt

        with (
            patch("ucode.cli.load_state", return_value={"workspace": "ws"}),
            patch("ucode.cli.sync_opencode_usage_from_messages"),
            patch("ucode.cli.sync_opencode_usage_from_state"),
            patch("ucode.cli.sync_codex_usage_recent"),
            patch("ucode.cli.local_budget_status", return_value={"state": "ok"}) as mock_status,
            patch("ucode.cli._build_budget_renderables", return_value=[]) as mock_build,
            patch("ucode.cli.signal.signal", side_effect=fake_signal),
            patch("ucode.cli.time.sleep", side_effect=fake_sleep),
        ):
            result = runner.invoke(app, ["usage", "budget-watch", "--interval", "1"])

        assert result.exit_code == 0, result.output
        # Registered a SIGWINCH handler so resizes drive redraws.
        assert captured["signum"] == signal.SIGWINCH
        # Spend data is fetched once on the timer; the resize redraw reuses the
        # cached status rather than re-fetching.
        assert mock_status.call_count == 1
        # Rendered twice: once on the initial timer tick, once on the resize.
        assert mock_build.call_count == 2

    def test_budget_watch_full_clears_each_frame(self):
        """Each redraw must fully clear the screen, wrapped in synchronized-output
        mode. An in-place overwrite would trust the cursor column to track Rich's
        width model, but glyphs like ⚠️ that the terminal renders narrower than
        Rich measures desync the two and leave stale characters (the stray digit
        seen beside the warning sign at 80%). A full clear can't leave residue."""
        import io as _io

        from rich.console import Console
        from rich.panel import Panel

        sleeps = {"n": 0}

        def fake_sleep(_secs):
            sleeps["n"] += 1
            raise KeyboardInterrupt

        buf = _io.StringIO()
        watch_console = Console(file=buf, force_terminal=True, width=80)

        with (
            patch("ucode.cli.console", watch_console),
            patch("ucode.cli.load_state", return_value={"workspace": "ws"}),
            patch("ucode.cli.sync_opencode_usage_from_messages"),
            patch("ucode.cli.sync_opencode_usage_from_state"),
            patch("ucode.cli.sync_codex_usage_recent"),
            patch("ucode.cli.local_budget_status", return_value={"state": "warn"}),
            patch(
                "ucode.cli._build_budget_renderables",
                return_value=[Panel("line one\nline two")],
            ),
            patch("ucode.cli.signal.signal"),
            patch("ucode.cli.time.sleep", side_effect=fake_sleep),
        ):
            runner.invoke(app, ["usage", "budget-watch", "--interval", "1"])

        frame = buf.getvalue()
        # Full erase-screen + home, wrapped in synchronized-output begin/end so
        # the clear and repaint present atomically (no flicker) on supporting
        # terminals, including the tmux pane the watcher runs in.
        assert "\033[2J\033[H" in frame
        begin = frame.index("\033[?2026h")
        clear = frame.index("\033[2J\033[H")
        end = frame.index("\033[?2026l")
        assert begin < clear < end

    def test_watch_strips_flags_from_agent_args(self):
        with (
            patch("ucode.cli._tmux_available", return_value=True),
            patch("ucode.cli._in_tmux", return_value=False),
            patch("ucode.cli.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            runner.invoke(app, ["claude", "--watch", "--watch-interval", "20", "--resume"])
        argv = self._tmux_argv(mock_run)
        # The left-pane agent command keeps real agent args but not the watch flags.
        agent_cmd = next(part for part in argv if part.startswith("ucode claude"))
        assert "--resume" in agent_cmd
        assert "--watch" not in agent_cmd
        assert "--watch-interval" not in agent_cmd

    def test_watch_falls_back_when_no_tmux(self):
        patches = _patch_launch("claude")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6] as mock_launch,
            patch("ucode.cli._tmux_available", return_value=False),
            patch("ucode.cli.subprocess.run") as mock_run,
        ):
            result = runner.invoke(app, ["claude", "--watch"])
        assert result.exit_code == 0, result.output
        mock_launch.assert_called_once()
        mock_run.assert_not_called()
        # The watch flag must not leak through to the agent.
        assert mock_launch.call_args[0][2] == []

    def test_watch_falls_back_when_already_in_tmux(self):
        patches = _patch_launch("claude")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6] as mock_launch,
            patch("ucode.cli._tmux_available", return_value=True),
            patch("ucode.cli._in_tmux", return_value=True),
            patch("ucode.cli.subprocess.run") as mock_run,
        ):
            result = runner.invoke(app, ["claude", "--watch"])
        assert result.exit_code == 0, result.output
        mock_launch.assert_called_once()
        mock_run.assert_not_called()
        assert mock_launch.call_args[0][2] == []


class TestConfigureAgentFlag:
    def test_no_flag_calls_configure_all(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(prompt_optional_updates=True)

    def test_agents_flag_calls_configure_with_tools(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", "claude,codex"])
        assert result.exit_code == 0, result.output
        mock_install.assert_not_called()
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"], prompt_optional_updates=True
        )

    def test_agents_flag_normalizes_aliases_and_dedupes(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", " claude-code, codex,claude "])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"], prompt_optional_updates=True
        )

    def test_workspaces_flag_calls_configure_with_workspaces(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                [
                    "configure",
                    "--workspaces",
                    "first.databricks.com,https://second.databricks.com/",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            workspaces=[
                ("https://first.databricks.com", None),
                ("https://second.databricks.com", None),
            ],
            prompt_optional_updates=True,
        )

    def test_agents_and_workspaces_flags_call_configure_with_both(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                ["configure", "--agents", "claude,codex", "--workspaces", "https://first.com"],
            )
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            workspaces=[("https://first.com", None)],
            prompt_optional_updates=True,
        )

    def test_agent_and_workspaces_flags_call_configure_with_both(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                ["configure", "--agent", "claude", "--workspaces", "https://first.com"],
            )
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once_with(
            "claude", strict=True, update_existing=True, prompt_optional_updates=True
        )
        mock_cfg.assert_called_once_with("claude", workspaces=[("https://first.com", None)])

    def test_agent_flag_calls_configure_with_tool(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agent", "claude"])
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once_with(
            "claude", strict=True, update_existing=True, prompt_optional_updates=True
        )
        mock_cfg.assert_called_once_with("claude")

    def test_skip_update_flag_disables_optional_update_prompt(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["--skip-update", "configure"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(prompt_optional_updates=False)

    def test_skip_update_flag_with_agent_skips_optional_update(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command"),
        ):
            result = runner.invoke(app, ["--skip-update", "configure", "--agent", "claude"])
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once_with(
            "claude", strict=True, update_existing=True, prompt_optional_updates=False
        )

    def test_skip_update_flag_with_agents_forwards_to_configure(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["--skip-update", "configure", "--agents", "claude,codex"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"], prompt_optional_updates=False
        )

    def test_agent_flag_normalizes_alias(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agent", "claude-code"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with("claude")

    def test_upgrade_runs_uv_tool_install(self):
        with patch("subprocess.run") as mock_run:
            result = runner.invoke(app, ["upgrade"])
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:3] == ["uv", "tool", "install"]
        assert "--reinstall" in cmd
        assert any("github.com/databricks/ucode" in s for s in cmd)

    def test_upgrade_handles_uv_missing(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = runner.invoke(app, ["upgrade"])
        assert result.exit_code != 0
        assert "uv" in result.output.lower()

    def test_agent_flag_rejects_unknown(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agent", "bogus"])
        assert result.exit_code != 0
        mock_cfg.assert_not_called()

    def test_agents_flag_rejects_unknown(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", "claude,bogus"])
        assert result.exit_code != 0
        assert "Unsupported tool 'bogus'" in result.output
        assert "codex, claude, gemini, opencode, copilot, pi" in " ".join(result.output.split())
        mock_cfg.assert_not_called()

    def test_agents_flag_rejects_empty_list(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", ","])
        assert result.exit_code != 0
        mock_cfg.assert_not_called()

    def test_agent_and_agents_flags_are_mutually_exclusive(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agent", "claude", "--agents", "codex"])
        assert result.exit_code != 0
        mock_cfg.assert_not_called()

    def test_workspaces_flag_rejects_empty_list(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--workspaces", ","])
        assert result.exit_code != 0
        mock_cfg.assert_not_called()


class TestConfigureAgentsSelection:
    def test_managed_available_tools_skip_picker(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {
            **MINIMAL_STATE,
            "available_tools": ["claude", "codex"],
            "_managed_config_pulled": True,
        }
        monkeypatch.setattr(
            cli_mod,
            "_prompt_for_configuration",
            lambda tool=None: ("https://example.com", None),
        )
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *args, **kwargs: state)
        checked: list[str] = []
        monkeypatch.setattr(
            cli_mod,
            "check_gateway_endpoint",
            lambda state, tool: checked.append(tool) or tool in {"claude", "codex"},
        )
        monkeypatch.setattr(
            cli_mod,
            "prompt_for_tools",
            lambda available: pytest.fail("prompt_for_tools should not be called"),
        )
        install_calls: list[str] = []
        monkeypatch.setattr(
            cli_mod,
            "install_tool_binary",
            lambda tool, strict=False, update_existing=False, prompt_optional_updates=True: (
                install_calls.append(tool) or True
            ),
        )
        configured: list[list[str]] = []
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: configured.append(tools) or {**state, "available_tools": tools},
        )
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda state: None)

        assert cli_mod.configure_workspace_command() == 0
        assert checked == ["claude", "codex"]
        assert install_calls == ["claude", "codex"]
        assert configured == [["claude", "codex"]]

    def test_selected_tools_override_managed_available_tools(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {
            **MINIMAL_STATE,
            "available_tools": ["claude"],
            "_managed_config_pulled": True,
        }
        monkeypatch.setattr(
            cli_mod,
            "_prompt_for_configuration",
            lambda tool=None: ("https://example.com", None),
        )
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *args, **kwargs: state)
        checked: list[str] = []
        monkeypatch.setattr(
            cli_mod,
            "check_gateway_endpoint",
            lambda state, tool: checked.append(tool) or tool == "codex",
        )
        monkeypatch.setattr(
            cli_mod,
            "prompt_for_tools",
            lambda available: pytest.fail("prompt_for_tools should not be called"),
        )
        configured: list[list[str]] = []
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: configured.append(tools) or {**state, "available_tools": tools},
        )
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda state: None)

        assert cli_mod.configure_workspace_command(selected_tools=["codex"]) == 0
        assert checked == ["codex"]
        assert configured == [["codex"]]

    def test_managed_available_tools_ignore_unknown_entries(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {
            **MINIMAL_STATE,
            "available_tools": ["codex", "bogus", None, "codex"],
            "_managed_config_pulled": True,
        }
        monkeypatch.setattr(
            cli_mod,
            "_prompt_for_configuration",
            lambda tool=None: ("https://example.com", None),
        )
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *args, **kwargs: state)
        checked: list[str] = []
        monkeypatch.setattr(
            cli_mod,
            "check_gateway_endpoint",
            lambda state, tool: checked.append(tool) or tool == "codex",
        )
        monkeypatch.setattr(
            cli_mod,
            "prompt_for_tools",
            lambda available: pytest.fail("prompt_for_tools should not be called"),
        )
        configured: list[list[str]] = []
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: configured.append(tools) or {**state, "available_tools": tools},
        )
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda state: None)

        assert cli_mod.configure_workspace_command() == 0
        assert checked == ["codex"]
        assert configured == [["codex"]]

    def test_selected_tools_skip_picker(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "available_tools": []}
        monkeypatch.setattr(
            cli_mod,
            "_prompt_for_configuration",
            lambda tool=None: ("https://example.com", None),
        )
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *args, **kwargs: state)
        monkeypatch.setattr(
            cli_mod, "check_gateway_endpoint", lambda state, tool: tool in {"claude", "codex"}
        )
        monkeypatch.setattr(
            cli_mod,
            "prompt_for_tools",
            lambda available: pytest.fail("prompt_for_tools should not be called"),
        )
        install_calls: list[str] = []
        monkeypatch.setattr(
            cli_mod,
            "install_tool_binary",
            lambda tool, strict=False, update_existing=False, prompt_optional_updates=True: (
                install_calls.append(tool) or True
            ),
        )
        configured: list[list[str]] = []
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: configured.append(tools) or {**state, "available_tools": tools},
        )
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda state: None)

        assert cli_mod.configure_workspace_command(selected_tools=["claude", "codex"]) == 0
        assert install_calls == ["claude", "codex"]
        assert configured == [["claude", "codex"]]

    def test_unavailable_selected_tool_errors_before_configure(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "available_tools": []}
        monkeypatch.setattr(
            cli_mod,
            "_prompt_for_configuration",
            lambda tool=None: ("https://example.com", None),
        )
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *args, **kwargs: state)
        monkeypatch.setattr(cli_mod, "check_gateway_endpoint", lambda state, tool: tool == "claude")
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: pytest.fail("configure_selected_tools should not be called"),
        )

        with pytest.raises(RuntimeError, match="Codex"):
            cli_mod.configure_workspace_command(selected_tools=["claude", "codex"])

    def test_multiple_workspaces_configure_all_and_use_first(self, monkeypatch):
        import ucode.cli as cli_mod

        states = {
            "https://first.com": {**MINIMAL_STATE, "workspace": "https://first.com"},
            "https://second.com": {**MINIMAL_STATE, "workspace": "https://second.com"},
        }
        configured_shared: list[tuple[str, str | None, tuple[str, ...] | None, bool]] = []

        def fake_configure_shared_state(workspace, profile=None, tools=None, force_login=False):
            configured_shared.append(
                (workspace, profile, tuple(tools) if tools is not None else None, force_login)
            )
            return states[workspace]

        saved: list[str] = []
        configured_tools: list[tuple[str, list[str]]] = []
        monkeypatch.setattr(cli_mod, "configure_shared_state", fake_configure_shared_state)
        monkeypatch.setattr(cli_mod, "save_state", lambda state: saved.append(state["workspace"]))
        monkeypatch.setattr(cli_mod, "check_gateway_endpoint", lambda state, tool: True)
        monkeypatch.setattr(cli_mod, "prompt_for_tools", lambda available: ["codex"])
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: (
                configured_tools.append((state["workspace"], tools))
                or {**state, "available_tools": tools}
            ),
        )
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda state: None)

        assert (
            cli_mod.configure_workspace_command(
                workspaces=[("https://first.com", None), ("https://second.com", None)]
            )
            == 0
        )
        assert configured_shared == [
            ("https://first.com", None, None, True),
            ("https://second.com", None, None, True),
        ]
        assert saved == ["https://first.com"]
        assert configured_tools == [("https://first.com", ["codex"])]


class TestConfigureSharedStateMcpCleanup:
    """A workspace switch should scrub the previous workspace's MCP entries from
    installed client configs. Switching to the same workspace must not."""

    @staticmethod
    def _stub_external_deps(monkeypatch):
        import ucode.cli as cli_mod

        monkeypatch.setattr(cli_mod, "normalize_workspace_url", lambda w: w)
        monkeypatch.setattr(cli_mod, "run_databricks_login", lambda w, p: None)
        monkeypatch.setattr(cli_mod, "ensure_databricks_auth", lambda w, p=None: None)
        monkeypatch.setattr(cli_mod, "find_profile_name_for_host", lambda w: None)
        monkeypatch.setattr(cli_mod, "get_databricks_token", lambda w, p: "token")
        monkeypatch.setattr(cli_mod, "ensure_ai_gateway_v2", lambda w, t: None)
        monkeypatch.setattr(cli_mod, "discover_claude_models", lambda w, t: ({}, None))
        monkeypatch.setattr(cli_mod, "discover_gemini_models", lambda w, t: ([], None))
        monkeypatch.setattr(cli_mod, "discover_codex_models", lambda w, t: ([], None))
        monkeypatch.setattr(cli_mod, "build_shared_base_urls", lambda w: {})
        monkeypatch.setattr(cli_mod, "download_managed_config", lambda w, p: None)
        monkeypatch.setattr(cli_mod, "download_managed_policies", lambda w, p: None)
        monkeypatch.setattr(cli_mod, "delete_workspace_policy", lambda w: None)

    def test_purges_residue_when_workspace_changes(self, monkeypatch):
        import ucode.cli as cli_mod

        self._stub_external_deps(monkeypatch)
        monkeypatch.setattr(
            cli_mod, "load_state", lambda: {"workspace": "https://old.databricks.com"}
        )
        purge_calls: list[tuple[dict, str]] = []
        monkeypatch.setattr(
            cli_mod,
            "purge_cross_workspace_mcp_residue",
            lambda state, workspace: purge_calls.append((state, workspace)),
        )

        cli_mod.configure_shared_state("https://new.databricks.com")

        assert len(purge_calls) == 1
        _, called_workspace = purge_calls[0]
        assert called_workspace == "https://new.databricks.com"

    def test_skips_purge_when_workspace_unchanged(self, monkeypatch):
        import ucode.cli as cli_mod

        self._stub_external_deps(monkeypatch)
        monkeypatch.setattr(
            cli_mod, "load_state", lambda: {"workspace": "https://same.databricks.com"}
        )
        purge_calls: list = []
        monkeypatch.setattr(
            cli_mod,
            "purge_cross_workspace_mcp_residue",
            lambda state, workspace: purge_calls.append((state, workspace)),
        )

        cli_mod.configure_shared_state("https://same.databricks.com")

        assert purge_calls == []


class TestConfigureSharedStatePullsManagedWorkspace:
    """`configure_shared_state` pulls the UC ``state.json`` and replaces the
    workspace block with it. Failures (no admin export yet, no
    permission, network down) must not block the user — ``download_managed_config``
    returns ``None`` and the local discovery path stands as-is."""

    @staticmethod
    def _stub_external_deps(monkeypatch):
        import ucode.cli as cli_mod

        monkeypatch.setattr(cli_mod, "normalize_workspace_url", lambda w: w)
        monkeypatch.setattr(cli_mod, "run_databricks_login", lambda w, p: None)
        monkeypatch.setattr(cli_mod, "ensure_databricks_auth", lambda w, p=None: None)
        monkeypatch.setattr(cli_mod, "find_profile_name_for_host", lambda w: None)
        monkeypatch.setattr(cli_mod, "get_databricks_token", lambda w, p: "token")
        monkeypatch.setattr(cli_mod, "ensure_ai_gateway_v2", lambda w, t: None)
        monkeypatch.setattr(cli_mod, "discover_claude_models", lambda w, t: ({}, None))
        monkeypatch.setattr(cli_mod, "discover_gemini_models", lambda w, t: ([], None))
        monkeypatch.setattr(cli_mod, "discover_codex_models", lambda w, t: ([], None))
        monkeypatch.setattr(cli_mod, "build_shared_base_urls", lambda w: {})
        monkeypatch.setattr(cli_mod, "load_state", lambda: {})
        monkeypatch.setattr(cli_mod, "purge_cross_workspace_mcp_residue", lambda *a, **k: None)

    def test_pulls_and_replaces_workspace_block_from_uc(self, monkeypatch):
        import ucode.cli as cli_mod

        ws = "https://example.databricks.com"
        self._stub_external_deps(monkeypatch)
        monkeypatch.setattr(
            cli_mod,
            "download_managed_config",
            lambda w, p: {
                "workspace": ws,
                "claude_models": {"opus": "admin-pinned"},
                "available_tools": ["claude"],
            },
        )
        saved_policy: list[dict] = []
        monkeypatch.setattr(cli_mod, "save_workspace_policy", lambda w, p: saved_policy.append(p))
        saved: list[dict] = []
        monkeypatch.setattr(cli_mod, "save_state", lambda state: saved.append(state))

        state = cli_mod.configure_shared_state(ws)

        assert len(saved) == 1
        assert saved[0]["claude_models"] == {"opus": "admin-pinned"}
        assert "policies" not in saved[0]
        assert saved_policy == []
        assert saved[0]["available_tools"] == ["claude"]
        assert "_managed_config_pulled" not in saved[0]
        assert state["_managed_config_pulled"] is True

    def test_pulls_and_caches_managed_policy_yaml(self, monkeypatch):
        import ucode.cli as cli_mod

        ws = "https://example.databricks.com"
        self._stub_external_deps(monkeypatch)
        monkeypatch.setattr(
            cli_mod,
            "download_managed_config",
            lambda w, p: {"workspace": ws, "available_tools": ["claude"]},
        )
        monkeypatch.setattr(
            cli_mod,
            "download_managed_policies",
            lambda w, p: (
                """
policy:
  name: coding-agents-default
  daily_budget_usd: 50
  tiers:
    - name: premium
      activates_at_pct: 0
      harness: claude
      model: opus
  on_budget_exhausted: block
"""
            ),
        )
        saved_policy: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            cli_mod,
            "save_workspace_policy",
            lambda workspace, policy: saved_policy.append((workspace, policy)),
        )
        monkeypatch.setattr(cli_mod, "save_state", lambda state: None)

        cli_mod.configure_shared_state(ws)

        assert saved_policy[0][0] == ws
        assert saved_policy[0][1]["policy"]["daily_budget_usd"] == 50.0

    def test_pulled_workspace_state_overwrites_local_stale_tools(self, monkeypatch):
        import ucode.cli as cli_mod

        ws = "https://example.databricks.com"
        self._stub_external_deps(monkeypatch)
        monkeypatch.setattr(
            cli_mod,
            "load_state",
            lambda: {
                "workspace": ws,
                "available_tools": ["codex", "pi", "gemini", "opencode", "copilot", "claude"],
                "gemini_models": ["local-stale"],
            },
        )
        monkeypatch.setattr(
            cli_mod,
            "download_managed_config",
            lambda w, p: {
                "available_tools": ["claude", "codex"],
                "claude_models": {"opus": "admin-pinned"},
            },
        )
        saved: list[dict] = []
        monkeypatch.setattr(cli_mod, "save_state", lambda state: saved.append(state))

        state = cli_mod.configure_shared_state(ws)

        assert len(saved) == 1
        assert saved[0]["workspace"] == ws
        assert saved[0]["available_tools"] == ["claude", "codex"]
        assert saved[0]["claude_models"] == {"opus": "admin-pinned"}
        assert "gemini_models" not in saved[0]
        assert "_managed_config_pulled" not in saved[0]
        assert state["_managed_config_pulled"] is True

    def test_silent_when_download_returns_none(self, monkeypatch):
        # First-time / no admin export yet — local state must save cleanly
        # without a policy block.
        import ucode.cli as cli_mod

        self._stub_external_deps(monkeypatch)
        monkeypatch.setattr(cli_mod, "download_managed_config", lambda w, p: None)
        saved: list[dict] = []
        monkeypatch.setattr(cli_mod, "save_state", lambda state: saved.append(state))

        cli_mod.configure_shared_state("https://example.databricks.com")

        assert len(saved) == 1
        assert "policies" not in saved[0]

    def test_installs_tracing_runtime_when_pulled_config_enables_it(self, monkeypatch):
        # A pulled config with tracing on must install the mlflow runtime so the
        # Claude Stop hook actually gets written by the later config writer.
        import ucode.cli as cli_mod

        ws = "https://example.databricks.com"
        self._stub_external_deps(monkeypatch)
        monkeypatch.setattr(
            cli_mod,
            "download_managed_config",
            lambda w, p: {
                "workspace": ws,
                "available_tools": ["claude"],
                "claude_models": {"opus": "admin-pinned"},
                "tracing": {
                    "enabled": True,
                    "tracking_uri": "databricks://admin-profile",
                    "experiment_id": "111",
                    "sql_warehouse_id": "wh123",
                },
            },
        )
        monkeypatch.setattr(cli_mod, "save_state", lambda state: None)
        installed: list[dict] = []
        monkeypatch.setattr(
            cli_mod, "install_tracing_runtime", lambda state: installed.append(state)
        )

        cli_mod.configure_shared_state(ws)

        assert len(installed) == 1
        assert installed[0]["tracing"]["enabled"] is True

    def test_install_tracing_runtime_noops_without_managed_pull(self, monkeypatch):
        import ucode.cli as cli_mod

        self._stub_external_deps(monkeypatch)
        monkeypatch.setattr(cli_mod, "download_managed_config", lambda w, p: None)
        monkeypatch.setattr(cli_mod, "save_state", lambda state: None)
        installed: list[dict] = []
        monkeypatch.setattr(
            cli_mod, "install_tracing_runtime", lambda state: installed.append(state)
        )

        cli_mod.configure_shared_state("https://example.databricks.com")

        assert installed == []


class TestPolicyModelOptionsOpencode:
    def test_surfaces_open_responses_kimi_model(self):
        import ucode.cli as cli_mod

        state = {
            "opencode_models": {
                "anthropic": ["databricks-claude-sonnet-4-6"],
                "open-responses": ["databricks-kimi-k2-6-colo"],
            }
        }
        options = cli_mod._policy_model_options("opencode", state)
        assert (
            "databricks-kimi-k2-6-colo",
            "open-responses (databricks-kimi-k2-6-colo)",
        ) in options
