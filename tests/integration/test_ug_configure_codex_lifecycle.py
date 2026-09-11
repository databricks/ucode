"""CUJs for changing and undoing codex's ug setup."""

import tomllib

import pytest
from utils.evidence import FileTask
from utils.terminal import TerminalProcess

pytestmark = [pytest.mark.live, pytest.mark.codex]


def test_ug_configure_codex_repeat_and_revert(live_session, workspace):
    """Scenario: configure twice over user-owned settings, use the agent, then revert.

    Expected: repeat setup remains usable, unrelated settings survive, no bearer
    is saved in ug state, and repeated revert restores the original configuration.
    The generated ug file must be removed; a leftover file remains a failure.
    """
    session = live_session
    task = FileTask(session)
    # Ordinary pre-existing user settings, not manufactured gateway state.
    user_path = session.home / ".codex/config.toml"
    original = "# user comment\n[notice]\nhide_rate_limit_model_nudge = true\n"
    user_path.parent.mkdir(parents=True)
    user_path.write_text(original)

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
    first = session.workspace_state()
    assert "codex" in first["available_tools"]
    assert first["codex_models"]
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
    assert session.workspace_state()["available_tools"] == first["available_tools"]
    assert workspace in session.run("status").stdout
    current = tomllib.loads(user_path.read_text())
    assert all(current.get(key) == value for key, value in tomllib.loads(original).items())
    bearer_was_saved = any(
        session.env["DATABRICKS_BEARER"] in path.read_text(errors="replace")
        for path in (session.home / ".ucode").rglob("*")
        if path.is_file()
    )
    assert not bearer_was_saved, "The workspace bearer was saved in ug state"

    result = session.run(
        "codex", "--", "exec", "--skip-git-repo-check", "--json", task.prompt, timeout=180
    )
    task.assert_headless_answer("codex", result)
    with TerminalProcess(session, "ug", [str(session.binary), "revert"], "revert") as terminal:
        terminal.finish()
    assert tomllib.loads(user_path.read_text()) == tomllib.loads(original)
    assert "Not Configured" in session.run("status").stdout
    assert not (session.home / ".codex/ucode.config.toml").exists()
    with TerminalProcess(
        session, "ug", [str(session.binary), "revert"], "repeat-revert"
    ) as terminal:
        terminal.finish()


def test_ug_configure_codex_rejects_invalid_credentials(live_session, workspace):
    """Scenario: configure codex against the real workspace with a rejected bearer.

    Expected: authentication fails clearly and no successful setup is saved.
    """
    session = live_session
    session.env["DATABRICKS_BEARER"] = "ug-integration-intentionally-invalid"
    result = session.run(
        "configure",
        "--agents",
        "codex",
        "--workspaces",
        workspace,
        "--skip-validate",
        "--skip-upgrade",
        "--disable-databricks-ai-tools",
        ok=False,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "rejected the access token" in output or "401" in output, output
    assert "Configuration Complete" not in output
    assert not (session.home / ".ucode/state.json").exists()
