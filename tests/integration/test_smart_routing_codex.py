"""Main CUJs: real first-prompt and child-task routing in Codex's TUI."""

import pytest
from evidence import FileTask, assert_subagent_routed
from terminal import AgentTerminal

pytestmark = [pytest.mark.live, pytest.mark.main, pytest.mark.tui, pytest.mark.codex]


def test_smart_routing_codex_first_prompt(live_session, workspace):
    """Scenario: enable smart routing and submit Codex's first interactive prompt.

    Expected: a real gateway decision selects a model and Codex completes the
    file-reading task before a normal exit. A routing fallback does not pass.
    """
    session = live_session
    task = FileTask(session)
    session.run(
        "configure",
        "--agents",
        "codex",
        "--workspaces",
        workspace,
        "--skip-validate",
        "--skip-upgrade",
        "--disable-databricks-ai-tools",
    )

    command = [str(session.binary), "codex", "--enable-smart-routing"]
    with AgentTerminal(session, "codex", command, "first-prompt") as tui:
        tui.boot()
        assert "[ROUTE]" not in session.routing_log("codex"), "Boot must not route a prompt"
        tui.submit(task.prompt)
        tui.wait_for_task(task)
        log = session.routing_log("codex")
        assert "[ROUTE] selected" in log, log
        assert "selection failed" not in log and "settings/update timed out" not in log, log
        tui.exit_normally()
    task.assert_completed(session, "codex")


def test_smart_routing_codex_subagent(live_session, workspace):
    """Scenario: ask Codex to delegate a file-reading task with routing enabled.

    Expected: a real child session has a correlated gateway routing decision,
    the child returns the file value, and the parent returns that result.
    Child model identity remains explicitly unknown if its event omits it.
    """
    session = live_session
    task = FileTask(session)
    session.run(
        "configure",
        "--agents",
        "codex",
        "--workspaces",
        workspace,
        "--skip-validate",
        "--skip-upgrade",
        "--disable-databricks-ai-tools",
    )

    command = [str(session.binary), "codex", "--enable-smart-routing"]
    with AgentTerminal(session, "codex", command, "subagent-task") as tui:
        tui.boot()
        tui.submit(task.delegate_prompt)
        tui.wait_for_task(task, timeout=240)
        task.assert_completed(session, "codex", child=True)
        assert_subagent_routed(session, "codex")
        tui.exit_normally()
    task.assert_completed(session, "codex")
