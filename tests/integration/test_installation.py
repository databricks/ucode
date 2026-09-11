"""Smoke checks for the installed distribution, outside the source checkout."""

import pytest

pytestmark = pytest.mark.installation


def test_ug_installed_wheel_exposes_help_and_version(session):
    """Scenario: invoke the freshly installed ug console script.

    Expected: public commands are listed and the installed version is reported.
    """
    output = session.run("--help").stdout
    assert "configure" in output and "revert" in output
    assert session.run("--version").stdout.strip()


def test_ug_status_in_fresh_home_is_unconfigured(session):
    """Scenario: inspect ug before any setup in a fresh home.

    Expected: status reports Not Configured and does not create saved state.
    """
    assert "Not Configured" in session.run("status").stdout
    assert not (session.home / ".ucode/state.json").exists()


def test_ug_auth_without_configuration_explains_how_to_configure(session):
    """Scenario: request an auth token before configuring ug.

    Expected: nonzero exit and configure guidance, without a Python traceback.
    """
    result = session.run("auth-token", ok=False)
    assert result.returncode != 0
    assert "configure" in result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr
