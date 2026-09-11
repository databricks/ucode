"""Smoke checks for the installed distribution, outside the source checkout."""

import pytest

pytestmark = pytest.mark.installation


def test_console_script(session):
    output = session.run("--help").stdout
    assert "configure" in output and "revert" in output
    assert session.run("--version").stdout.strip()


def test_fresh_home_is_unconfigured(session):
    assert "Not Configured" in session.run("status").stdout
    assert not (session.home / ".ucode/state.json").exists()


def test_auth_without_configuration_fails_with_guidance(session):
    result = session.run("auth-token", ok=False)
    assert result.returncode != 0
    assert "configure" in result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr
