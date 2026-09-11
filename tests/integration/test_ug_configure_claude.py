"""CUJs: configure Claude through ug, then use its real interactive session."""

import pytest
from utils.evidence import FileTask
from utils.terminal import AgentTerminal, ConfigureTerminal

pytestmark = [pytest.mark.live, pytest.mark.tui, pytest.mark.claude]


def test_ug_configure_claude_databricks(live_session, workspace):
    """Scenario: configure Claude with Databricks Hosted and use its TUI.

    Expected: configure succeeds, Claude returns a file value through the real
    gateway, exits normally, and can reopen the configuration ug created.
    Optional AI Tools are disabled; the selected agent version is kept pinned.
    """
    session = live_session
    task = FileTask(session)

    # Configure using the installed public CLI, including its normal validation.
    session.run(
        "configure",
        "--agents",
        "claude",
        "--workspaces",
        workspace,
        "--skip-upgrade",
        "--disable-databricks-ai-tools",
        timeout=240,
    )
    assert not session.workspace_state().get("provider_services", {}).get("claude")

    # Use the real TUI; a config file or startup banner alone is not success.
    with AgentTerminal(session, "claude", [str(session.binary), "claude"], "first-session") as tui:
        tui.boot()
        tui.submit(task.prompt)
        tui.wait_for_task(task)
        tui.exit_normally()
    task.assert_completed(session, "claude")
    session.assert_not_routed()

    # Reopen the same home, without configuring again or seeding onboarding state.
    with AgentTerminal(session, "claude", [str(session.binary), "claude"], "reopen") as tui:
        tui.boot()
        tui.check_input_and_exit()


def test_ug_configure_claude_anthropic_mps(live_session, workspace, claude_provider):
    """Scenario: choose the real Anthropic MPS in ug configure's provider picker.

    Expected: ug saves that provider, and launching Claude without --provider
    uses the saved choice to complete a file-reading task and exit normally.
    """
    session = live_session
    task = FileTask(session)

    # Provider selection is interactive; --agents would bypass this real picker.
    command = [
        str(session.binary),
        "configure",
        "--workspaces",
        workspace,
        "--skip-upgrade",
        "--disable-databricks-ai-tools",
    ]
    with ConfigureTerminal(session, "claude", command, "configure-provider") as configure:
        configure.select_agent("Claude Code")
        configure.choose("How should Claude Code get its models?", "External Models")
        configure.choose("Select a model provider service:", claude_provider)
        configure.finish(timeout=240)
    assert session.workspace_state()["provider_services"]["claude"] == claude_provider
    assert claude_provider in session.run("status").stdout

    with AgentTerminal(
        session, "claude", [str(session.binary), "claude"], "provider-session"
    ) as tui:
        tui.boot()
        tui.submit(task.prompt)
        tui.wait_for_task(task)
        tui.exit_normally()
    task.assert_completed(session, "claude")
    session.assert_not_routed()
