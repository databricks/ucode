"""CUJ: a client connects to Codex's real app-server through ug."""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.codex]


@pytest.mark.parametrize("separator", [False, True], ids=["direct", "launcher-separator"])
@pytest.mark.parametrize("routing", ["0", "1"], ids=["routing-off", "routing-on"])
def test_ug_codex_app_server_client_initializes(live_session, workspace, separator, routing):
    """Scenario: configure Codex and connect a real stdio client to ug codex app-server.

    Expected: initialize returns a valid JSON-RPC result, diagnostics stay off
    the protocol stream, and this utility command never starts smart routing.
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

    args = ["app-server", "--listen", "stdio://"]
    if separator:
        args.insert(0, "--")
    response = session.app_server_handshake(args)
    assert response["result"]["userAgent"]
    session.assert_not_routed()
