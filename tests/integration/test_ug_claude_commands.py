"""CUJs for inspecting claude commands through ug."""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.claude]


@pytest.mark.parametrize("routing", ["0", "1"], ids=["routing-off", "routing-on"])
def test_ug_claude_auth_help(live_session, workspace, routing):
    """Scenario: configure claude, then ask ug for auth subcommand help.

    Expected: the real claude help is returned, with no routing wrapper or
    model request. This verifies command dispatch, not an interactive session.
    """
    session = live_session
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
    session.env["ENABLE_SMART_ROUTING_V2"] = routing
    expected = session.run("auth", "--help", binary="claude").stdout.strip()
    actual = session.run("claude", "--", "auth", "--help").stdout
    assert expected and expected in actual, actual
    session.assert_not_routed()


@pytest.mark.parametrize("routing", ["0", "1"], ids=["routing-off", "routing-on"])
def test_ug_claude_mcp_help(live_session, workspace, routing):
    """Scenario: configure claude, then ask ug for mcp subcommand help.

    Expected: the real claude help is returned, with no routing wrapper or
    model request. This verifies command dispatch, not an interactive session.
    """
    session = live_session
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
    session.env["ENABLE_SMART_ROUTING_V2"] = routing
    expected = session.run("mcp", "--help", binary="claude").stdout.strip()
    actual = session.run("claude", "--", "mcp", "--help").stdout
    assert expected and expected in actual, actual
    session.assert_not_routed()
