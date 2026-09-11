"""CUJs for inspecting codex commands through ug."""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.codex]


@pytest.mark.parametrize("routing", ["0", "1"], ids=["routing-off", "routing-on"])
def test_ug_codex_app_help(live_session, workspace, routing):
    """Scenario: configure codex, then ask ug for app subcommand help.

    Expected: the real codex help is returned, with no routing wrapper or
    model request. This verifies command dispatch, not an interactive session.
    """
    session = live_session
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
    session.env["ENABLE_SMART_ROUTING_V2"] = routing
    expected = session.run("app", "--help", binary="codex").stdout.strip()
    actual = session.run("codex", "--", "app", "--help").stdout
    assert expected and expected in actual, actual
    session.assert_not_routed()


@pytest.mark.parametrize("routing", ["0", "1"], ids=["routing-off", "routing-on"])
def test_ug_codex_app_server_help(live_session, workspace, routing):
    """Scenario: configure codex, then ask ug for app-server subcommand help.

    Expected: the real codex help is returned, with no routing wrapper or
    model request. This verifies command dispatch, not an interactive session.
    """
    session = live_session
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
    session.env["ENABLE_SMART_ROUTING_V2"] = routing
    expected = session.run("app-server", "--help", binary="codex").stdout.strip()
    actual = session.run("codex", "--", "app-server", "--help").stdout
    assert expected and expected in actual, actual
    session.assert_not_routed()


@pytest.mark.parametrize("routing", ["0", "1"], ids=["routing-off", "routing-on"])
def test_ug_codex_exec_help(live_session, workspace, routing):
    """Scenario: configure codex, then ask ug for exec subcommand help.

    Expected: the real codex help is returned, with no routing wrapper or
    model request. This verifies command dispatch, not an interactive session.
    """
    session = live_session
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
    session.env["ENABLE_SMART_ROUTING_V2"] = routing
    expected = session.run("exec", "--help", binary="codex").stdout.strip()
    actual = session.run("codex", "--", "exec", "--help").stdout
    assert expected and expected in actual, actual
    session.assert_not_routed()


@pytest.mark.parametrize("routing", ["0", "1"], ids=["routing-off", "routing-on"])
def test_ug_codex_mcp_help(live_session, workspace, routing):
    """Scenario: configure codex, then ask ug for mcp subcommand help.

    Expected: the real codex help is returned, with no routing wrapper or
    model request. This verifies command dispatch, not an interactive session.
    """
    session = live_session
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
    session.env["ENABLE_SMART_ROUTING_V2"] = routing
    expected = session.run("mcp", "--help", binary="codex").stdout.strip()
    actual = session.run("codex", "--", "mcp", "--help").stdout
    assert expected and expected in actual, actual
    session.assert_not_routed()


@pytest.mark.parametrize("routing", ["0", "1"], ids=["routing-off", "routing-on"])
def test_ug_codex_app_reports_unknown_argument(live_session, workspace, routing):
    """Scenario: pass an unknown option directly to ug codex app.

    Expected: the actual Codex parser's error and exit status are preserved,
    without opening a desktop application or starting routing.
    """
    session = live_session
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
    session.env["ENABLE_SMART_ROUTING_V2"] = routing
    args = ["app", "--ug-integration-unknown-option"]
    expected = session.run(*args, binary="codex", ok=False)
    actual = session.run("codex", *args, ok=False)
    assert expected.returncode != 0 and "error:" in expected.stderr
    assert actual.returncode == expected.returncode
    parser_error = expected.stderr[expected.stderr.index("error:") :].strip()
    assert parser_error in actual.stderr, actual.stdout + actual.stderr
    session.assert_not_routed()
