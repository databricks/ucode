"""Tests for npm-backed agent update checks."""

from __future__ import annotations

import subprocess

from ucode.agent_updates import available_npm_package_update


def test_returns_none_when_npm_missing(monkeypatch):
    monkeypatch.setattr("ucode.agent_updates.shutil.which", lambda _: None)

    assert available_npm_package_update("opencode-ai") is None


def test_returns_none_when_package_is_current(monkeypatch):
    monkeypatch.setattr("ucode.agent_updates.shutil.which", lambda _: "/usr/bin/npm")
    monkeypatch.setattr(
        "ucode.agent_updates.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="{}", stderr=""),
    )

    assert available_npm_package_update("opencode-ai") is None


def test_returns_current_and_latest_when_outdated(monkeypatch):
    monkeypatch.setattr("ucode.agent_updates.shutil.which", lambda _: "/usr/bin/npm")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0],
            1,
            stdout='{"opencode-ai":{"current":"1.2.3","wanted":"1.2.4","latest":"1.2.4"}}',
            stderr="",
        )

    monkeypatch.setattr("ucode.agent_updates.subprocess.run", fake_run)

    assert available_npm_package_update("opencode-ai") == ("1.2.3", "1.2.4")


def test_returns_none_for_malformed_output(monkeypatch):
    monkeypatch.setattr("ucode.agent_updates.shutil.which", lambda _: "/usr/bin/npm")
    monkeypatch.setattr(
        "ucode.agent_updates.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, stdout="not json", stderr=""
        ),
    )

    assert available_npm_package_update("opencode-ai") is None
