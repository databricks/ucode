"""CUJs for using codex from scripts through installed ug."""

import pytest
from utils.evidence import FileTask

pytestmark = [pytest.mark.live, pytest.mark.codex]


def test_ug_codex_headless_prompt_argument(live_session, workspace):
    """Scenario: configure codex and submit a headless prompt via argument.

    Expected: the real agent reads the fixture and returns its unknown value in
    its structured completed answer, with exit code zero.
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

    result = session.run(
        "codex", "--", "exec", "--skip-git-repo-check", "--json", task.prompt, timeout=180
    )
    task.assert_headless_answer("codex", result)
    session.assert_not_routed()


def test_ug_codex_headless_prompt_stdin(live_session, workspace):
    """Scenario: configure codex and submit a headless prompt via stdin.

    Expected: the real agent reads the fixture and returns its unknown value in
    its structured completed answer, with exit code zero.
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

    result = session.run(
        "codex",
        "--",
        "exec",
        "--skip-git-repo-check",
        "--json",
        "-",
        timeout=180,
        input_text=task.prompt + "\n",
    )
    task.assert_headless_answer("codex", result)
    session.assert_not_routed()


def test_ug_codex_headless_prompt_after_separator(live_session, workspace):
    """Scenario: configure codex and submit a headless prompt via after separator.

    Expected: the real agent reads the fixture and returns its unknown value in
    its structured completed answer, with exit code zero.
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

    result = session.run(
        "codex", "--", "exec", "--skip-git-repo-check", "--json", "--", task.prompt, timeout=180
    )
    task.assert_headless_answer("codex", result)
    session.assert_not_routed()


@pytest.mark.parametrize("model_form", ["separate", "equals", "short"])
def test_ug_codex_headless_explicit_model_bypasses_routing(live_session, workspace, model_form):
    """Scenario: choose an explicit model while global smart routing is enabled.

    Expected: the model option is accepted, the real file task completes, and
    no routing wrapper overrides the caller's choice.
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
    model = session.model_for_explicit_case("codex")
    model_args = ["--model", model] if model_form == "separate" else [f"--model={model}"]
    if model_form == "short":
        model_args = ["-m", model]
    session.env["ENABLE_SMART_ROUTING_V2"] = "1"
    result = session.run(
        "codex",
        "--",
        "exec",
        "--skip-git-repo-check",
        "--json",
        task.prompt,
        *model_args,
        timeout=180,
    )
    task.assert_headless_answer("codex", result)
    session.assert_not_routed()
