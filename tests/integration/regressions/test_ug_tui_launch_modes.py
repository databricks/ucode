"""Boot the real interactive terminal, then reopen the state it actually wrote."""

import pytest
from terminal import AgentTerminal

pytestmark = [pytest.mark.live, pytest.mark.tui, pytest.mark.regression]


@pytest.mark.parametrize("launch", ["routing-off", "routing-on", "explicit-model"])
@pytest.mark.parametrize(
    "agent",
    [
        pytest.param("claude", marks=pytest.mark.claude),
        pytest.param("codex", marks=pytest.mark.codex),
    ],
)
def test_ug_tui_launch_mode_supports_boot_reopen_and_exit(live_session, workspace, agent, launch):
    """Scenario: boot and reopen a configured agent in each routing launch mode.

    Expected: onboarding, keyboard editing, and normal exit work; wrappers start
    only when requested. This regression makes no inference-completion claim.
    """
    session = live_session
    session.configure(agent, workspace)
    session.env["ENABLE_SMART_ROUTING_V2"] = "0" if launch == "routing-off" else "1"
    args = []
    if launch == "explicit-model":
        model = session.model_for_explicit_case(agent)
        args = ["--", "--model", model]

    for name in ("first-boot", "reopen"):
        command = [str(session.binary), agent, *args]
        with AgentTerminal(session, agent, command, name) as terminal:
            terminal.boot()
            terminal.check_input_and_exit()

    if launch == "routing-on":
        filename = "claude-v2-pty.log" if agent == "claude" else "codex-v2-interposer.log"
        path = session.home / ".ucode" / filename
        assert path.is_file(), "The real routing wrapper did not start"
        log = path.read_text()
        session.record("routing-boot.json", {"log": log})
        assert "[READY]" in log, log
        assert "[ROUTE]" not in log, "Booting without a model prompt should not route a turn"
    else:
        session.assert_not_routed()
