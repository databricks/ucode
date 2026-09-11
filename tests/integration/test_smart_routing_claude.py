"""Main CUJs: real first-prompt and child-task routing in Claude's TUI."""

import pytest
from evidence import FileTask, assert_subagent_routed
from terminal import AgentTerminal

pytestmark = [pytest.mark.live, pytest.mark.main, pytest.mark.tui, pytest.mark.claude]


def test_smart_routing_claude_first_prompt(live_session, workspace):
    """Scenario: enable smart routing and submit Claude's first interactive prompt.

    Expected: a real gateway decision selects a model, the prompt is replayed,
    and the assistant completes the task before a normal exit. Boot alone fails.
    """
    session = live_session
    task = FileTask(session)
    session.run(
        "configure",
        "--agents",
        "claude",
        "--workspaces",
        workspace,
        "--skip-validate",
        "--skip-upgrade",
        "--disable-databricks-ai-tools",
    )

    command = [str(session.binary), "claude", "--enable-smart-routing"]
    with AgentTerminal(session, "claude", command, "first-prompt") as tui:
        tui.boot()
        assert "[ROUTE]" not in session.routing_log("claude"), "Boot must not route a prompt"
        tui.submit(task.prompt)
        tui.wait_for_task(task)
        log = session.routing_log("claude")
        assert "[ROUTE] first prompt ->" in log, log
        assert "[REPLAY] first prompt submitted" in log, log
        tui.exit_normally()
    task.assert_completed(session, "claude")


def test_smart_routing_claude_subagent(live_session, workspace):
    """Scenario: ask Claude to delegate a file-reading task with routing enabled.

    Expected: a real child session has a correlated gateway routing decision,
    the child returns the file value, and the parent returns that result.
    Child model identity remains explicitly unknown if its event omits it.
    """
    session = live_session
    task = FileTask(session)
    session.run(
        "configure",
        "--agents",
        "claude",
        "--workspaces",
        workspace,
        "--skip-validate",
        "--skip-upgrade",
        "--disable-databricks-ai-tools",
    )

    command = [str(session.binary), "claude", "--enable-smart-routing"]
    with AgentTerminal(session, "claude", command, "subagent-task") as tui:
        tui.boot()
        tui.submit(task.delegate_prompt)
        tui.wait_for_task(task, timeout=240)
        task.assert_completed(session, "claude", child=True)
        assert_subagent_routed(session, "claude")
        tui.exit_normally()
    task.assert_completed(session, "claude")
