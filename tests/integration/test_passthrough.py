"""Regression cases for #502 and real Codex config serialization (#496)."""

import pytest

pytestmark = pytest.mark.live

# Compare against each real agent's output. A wrapper's own help is not evidence
# that it dispatched the requested subcommand. No updater or desktop app is run.
HELP_CASES = [
    pytest.param("codex", ["app", "--help"], marks=pytest.mark.codex, id="codex-app"),
    pytest.param("codex", ["app-server", "--help"], marks=pytest.mark.codex, id="codex-app-server"),
    pytest.param("codex", ["exec", "--help"], marks=pytest.mark.codex, id="codex-exec"),
    pytest.param("codex", ["mcp", "--help"], marks=pytest.mark.codex, id="codex-mcp"),
    pytest.param("claude", ["mcp", "--help"], marks=pytest.mark.claude, id="claude-mcp"),
    pytest.param("claude", ["auth", "--help"], marks=pytest.mark.claude, id="claude-auth"),
]


@pytest.mark.parametrize("agent,args", HELP_CASES)
@pytest.mark.parametrize("routing", ["0", "1"], ids=["routing-off", "routing-on"])
def test_subcommand_help_reaches_real_agent(live_session, workspace, agent, args, routing):
    session = live_session
    session.configure(agent, workspace)
    session.env["ENABLE_SMART_ROUTING_V2"] = routing
    expected = session.run(*args, binary=agent).stdout.strip()
    assert expected
    actual = session.run(agent, "--", *args).stdout
    assert expected in actual, actual
    session.assert_not_routed()


@pytest.mark.codex
@pytest.mark.parametrize("routing", ["0", "1"], ids=["routing-off", "routing-on"])
def test_codex_app_argument_error_comes_from_real_agent(live_session, workspace, routing):
    session = live_session
    args = ["app", "--ug-integration-unknown-option"]
    expected = session.run(*args, binary="codex", ok=False)
    assert expected.returncode != 0 and expected.stderr.strip()
    session.configure("codex", workspace)
    session.env["ENABLE_SMART_ROUTING_V2"] = routing
    # Exercise the direct subcommand form without opening a desktop app or
    # allowing ug's own --help option to intercept the request.
    actual = session.run("codex", *args, ok=False)
    assert actual.returncode == expected.returncode
    assert expected.stderr.strip() in actual.stderr, actual.stdout + actual.stderr
    session.assert_not_routed()


@pytest.mark.codex
@pytest.mark.parametrize("separator", [False, True], ids=["direct", "launcher-separator"])
@pytest.mark.parametrize("routing", ["0", "1"], ids=["routing-off", "routing-on"])
def test_codex_app_server_protocol(live_session, workspace, separator, routing):
    session = live_session
    session.configure("codex", workspace)
    session.env["ENABLE_SMART_ROUTING_V2"] = routing
    args = ["app-server", "--listen", "stdio://"]
    if separator:
        args.insert(0, "--")
    session.app_server_handshake(args)
    session.assert_not_routed()
