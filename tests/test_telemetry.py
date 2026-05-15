"""Tests for ucode.telemetry."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from ucode import telemetry


class TestUcodeVersion:
    def test_returns_string(self):
        # Either a real version like "0.1.0" or "unknown" — both are strings.
        assert isinstance(telemetry.ucode_version(), str)
        assert telemetry.ucode_version() != ""


class TestAgentVersion:
    def setup_method(self):
        # The helper is @cache'd; clear between tests so each gets a clean run.
        telemetry.agent_version.cache_clear()

    def test_falls_back_when_binary_missing(self):
        assert telemetry.agent_version("definitely-not-a-real-binary-9f8a") == "unknown"

    def test_falls_back_on_timeout(self):
        with patch.object(
            subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="x", timeout=2),
        ):
            assert telemetry.agent_version("anything") == "unknown"

    def test_parses_claude_format(self):
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="2.1.136 (Claude Code)\n", stderr=""
        )
        with patch.object(subprocess, "run", return_value=result):
            assert telemetry.agent_version("claude") == "2.1.136"

    def test_parses_codex_format(self):
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="codex-cli 0.123.0-alpha.8\n", stderr=""
        )
        with patch.object(subprocess, "run", return_value=result):
            assert telemetry.agent_version("codex") == "0.123.0-alpha.8"

    def test_parses_gemini_format(self):
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout="0.40.0\n", stderr="")
        with patch.object(subprocess, "run", return_value=result):
            assert telemetry.agent_version("gemini") == "0.40.0"

    def test_parses_opencode_format(self):
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout="0.74.0\n", stderr="")
        with patch.object(subprocess, "run", return_value=result):
            assert telemetry.agent_version("opencode") == "0.74.0"

    def test_parses_copilot_format(self):
        # Copilot prints a banner with the version embedded mid-line.
        stdout = "GitHub Copilot CLI 1.0.42-0.\nRun 'copilot update' to check for updates.\n"
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
        with patch.object(subprocess, "run", return_value=result):
            assert telemetry.agent_version("copilot") == "1.0.42-0."

    def test_parses_pi_dev_build_format(self):
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="0.0.0-dev-202604280149\n", stderr=""
        )
        with patch.object(subprocess, "run", return_value=result):
            assert telemetry.agent_version("pi") == "0.0.0-dev-202604280149"

    def test_falls_back_to_stderr_when_stdout_empty(self):
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="my-tool 9.9.9\n"
        )
        with patch.object(subprocess, "run", return_value=result):
            assert telemetry.agent_version("foo") == "9.9.9"

    def test_unknown_when_no_semver_in_output(self):
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="hello world\n", stderr=""
        )
        with patch.object(subprocess, "run", return_value=result):
            assert telemetry.agent_version("foo") == "unknown"
