"""CUJs for using claude from scripts through installed ug."""

import json

import pytest
from utils.evidence import FileTask

pytestmark = [pytest.mark.live, pytest.mark.claude]


def test_ug_claude_headless_prompt_argument(live_session, workspace):
    """Scenario: configure claude and submit a headless prompt via argument.

    Expected: the real agent reads the fixture and returns its unknown value in
    its structured completed answer, with exit code zero.
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

    result = session.run(
        "claude",
        "--",
        "-p",
        task.prompt,
        "--output-format",
        "json",
        "--allowedTools",
        "Read",
        timeout=180,
    )
    task.assert_headless_answer("claude", result)
    session.assert_not_routed()


def test_ug_claude_headless_prompt_stdin(live_session, workspace):
    """Scenario: configure claude and submit a headless prompt via stdin.

    Expected: the real agent reads the fixture and returns its unknown value in
    its structured completed answer, with exit code zero.
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

    result = session.run(
        "claude",
        "--",
        "-p",
        "--output-format",
        "json",
        "--allowedTools",
        "Read",
        timeout=180,
        input_text=task.prompt + "\n",
    )
    task.assert_headless_answer("claude", result)
    session.assert_not_routed()


def test_ug_claude_headless_prompt_after_separator(live_session, workspace):
    """Scenario: configure claude and submit a headless prompt via after separator.

    Expected: the real agent reads the fixture and returns its unknown value in
    its structured completed answer, with exit code zero.
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

    result = session.run(
        "claude",
        "--",
        "-p",
        "--output-format",
        "json",
        "--allowedTools",
        "Read",
        "--",
        task.prompt,
        timeout=180,
    )
    task.assert_headless_answer("claude", result)
    session.assert_not_routed()


@pytest.mark.parametrize("model_form", ["separate", "equals"])
def test_ug_claude_headless_explicit_model_bypasses_routing(live_session, workspace, model_form):
    """Scenario: choose an explicit model while global smart routing is enabled.

    Expected: the model option is accepted, the real file task completes, and
    no routing wrapper overrides the caller's choice.
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
    model = session.model_for_explicit_case("claude")
    model_args = ["--model", model] if model_form == "separate" else [f"--model={model}"]
    session.env["ENABLE_SMART_ROUTING_V2"] = "1"
    result = session.run(
        "claude",
        "--",
        "-p",
        task.prompt,
        "--output-format",
        "json",
        "--allowedTools",
        "Read",
        *model_args,
        timeout=180,
    )
    task.assert_headless_answer("claude", result)
    session.assert_not_routed()


def test_ug_claude_preserves_caller_settings_and_hook(live_session, workspace):
    """Scenario: a launcher passes a settings path containing spaces to ug claude.

    Expected: the caller's real SessionStart hook executes, its input file stays
    unchanged, and gateway authentication still supports a completed file task.
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
    settings = session.cwd / "caller settings.json"
    # Ordinary user-owned input, not fabricated ug state or generated gateway config.
    content = json.dumps(
        {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {"type": "command", "command": "echo caller-hook-ran > caller-hook.txt"}
                        ]
                    }
                ]
            }
        }
    )
    settings.write_text(content)
    result = session.run(
        "claude",
        "--",
        "-p",
        task.prompt,
        "--output-format",
        "json",
        "--allowedTools",
        "Read",
        "--settings",
        str(settings),
        timeout=180,
    )
    task.assert_headless_answer("claude", result)
    assert (session.cwd / "caller-hook.txt").read_text().strip() == "caller-hook-ran"
    assert settings.read_text() == content


def test_ug_claude_reports_unsupported_short_model_option(live_session, workspace):
    """Scenario: pass -m to the selected Claude version, which does not support it.

    Expected: ug preserves the actual agent's unknown-option error and status.
    This is an error-reporting journey, not a successful inference claim.
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
    expected = session.run("-m", "sonnet", "-p", "hi", binary="claude", ok=False)
    actual = session.run("claude", "--", "-m", "sonnet", "-p", "hi", ok=False)
    assert expected.returncode != 0 and "unknown option '-m'" in expected.stderr
    assert actual.returncode == expected.returncode
    assert "unknown option '-m'" in actual.stderr
