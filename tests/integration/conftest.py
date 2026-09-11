"""Standalone fixtures: application state is created only by installed CLI commands."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest
from harness import UserSession


def pytest_generate_tests(metafunc):
    explicit = any(
        "agent"
        in (
            mark.args[0].replace(" ", "").split(",")
            if isinstance(mark.args[0], str)
            else mark.args[0]
        )
        for mark in metafunc.definition.iter_markers("parametrize")
    )
    if "agent" in metafunc.fixturenames and not explicit:
        agents = os.environ.get("UG_INTEGRATION_AGENTS", "claude,codex").split(",")
        metafunc.parametrize("agent", agents)


def pytest_collection_modifyitems(config, items):
    agents = os.environ.get("UG_INTEGRATION_AGENTS", "claude,codex").split(",")
    selected, deselected = [], []
    for item in items:
        # Fail if someone accidentally invokes this under the unit-test fixtures.
        if "monkeypatch" in item.fixturenames:
            raise pytest.UsageError("Use the integration runner; unit fixtures were inherited.")
        if any(item.get_closest_marker(a) and a not in agents for a in ("claude", "codex")):
            deselected.append(item)
        else:
            selected.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


@pytest.fixture(scope="session")
def installed_binary():
    raw = os.environ.get("UG_INTEGRATION_BIN")
    if not raw or not Path(raw).is_file():
        pytest.fail("Run scripts/run_integration.py to install the version under test.")
    return Path(raw)


@pytest.fixture(scope="session")
def workspace():
    value = os.environ.get("UCODE_TEST_WORKSPACE", "").strip().rstrip("/")
    if not value.startswith("https://") or not os.environ.get("DATABRICKS_BEARER", "").strip():
        pytest.fail("Live integration requires UCODE_TEST_WORKSPACE and DATABRICKS_BEARER.")
    return value


@pytest.fixture
def session(request, installed_binary):
    # Codex rejects helper installation beneath /tmp. Keep the disposable home
    # under the runner's own directory, never in the developer's agent folders.
    root = Path(os.environ["UG_INTEGRATION_RUN_DIR"])
    case = re.sub(r"[^a-zA-Z0-9_.-]", "_", request.node.name)
    with (
        tempfile.TemporaryDirectory(prefix="case-", dir=root) as temporary,
        tempfile.TemporaryDirectory(prefix="ug-integration-project-") as project,
    ):
        # Agents walk parent directories for project settings. Keeping cwd out
        # of the checkout prevents its .claude/AGENTS.md from influencing a run.
        yield UserSession(
            Path(temporary), Path(project), installed_binary, root / "artifacts" / case
        )


@pytest.fixture
def live_session(session, workspace):
    for binary in ["databricks", *os.environ["UG_INTEGRATION_AGENTS"].split(",")]:
        if not shutil.which(binary, path=session.env["PATH"]):
            pytest.fail(f"Required integration binary is missing: {binary}")
    session.env["DATABRICKS_BEARER"] = os.environ["DATABRICKS_BEARER"]
    return session


@pytest.fixture
def configured(live_session, workspace, agent):
    live_session.configure(agent, workspace)
    return live_session
