"""Configure, repeat, and undo real setup; never manufacture ug state."""

import json
import tomllib

import pytest

pytestmark = pytest.mark.live


def test_configure_repeat_and_revert_preserves_user_settings(live_session, workspace, agent):
    session = live_session
    if agent == "claude":
        user_path = session.home / ".claude/settings.json"
        user_settings = '{"permissions":{"allow":["Read"]},"env":{"UG_USER_SETTING":"keep"}}\n'
        load = json.loads
    else:
        user_path = session.home / ".codex/config.toml"
        user_settings = "# user comment\n[notice]\nhide_rate_limit_model_nudge = true\n"
        load = tomllib.loads
    user_path.parent.mkdir(parents=True)
    user_path.write_text(user_settings)

    session.configure(agent, workspace)
    first = session.state()
    ws_state = first["workspaces"][workspace]
    assert first["current_workspace"] == workspace
    assert agent in ws_state["available_tools"]
    assert ws_state[f"{agent}_models"], "Discovery returned no models for the selected agent"
    assert workspace in session.run("status").stdout
    session.configure(agent, workspace)
    second = session.state()
    assert second["current_workspace"] == workspace
    assert second["workspaces"][workspace]["available_tools"] == ws_state["available_tools"]
    current = load(user_path.read_text())
    assert all(current.get(key) == value for key, value in load(user_settings).items())
    for path in (session.home / ".ucode").rglob("*"):
        if path.is_file():
            assert session.env["DATABRICKS_BEARER"] not in path.read_text(errors="replace"), (
                f"Bearer was persisted in {path.name}"
            )

    session.run("revert")
    assert load(user_path.read_text()) == load(user_settings)
    assert "Not Configured" in session.run("status").stdout
    private = ".claude/ucode-settings.json" if agent == "claude" else ".codex/ucode.config.toml"
    assert not (session.home / private).exists()
    session.run("revert")  # Reverting an already reverted setup is safe.


def test_rejected_credentials_do_not_report_success(live_session, workspace, agent):
    live_session.env["DATABRICKS_BEARER"] = "ug-integration-intentionally-invalid"
    result = live_session.configure(agent, workspace, ok=False)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "rejected the access token" in output or "401" in output, output
    assert "Configuration Complete" not in output
    assert not (live_session.home / ".ucode/state.json").exists()
