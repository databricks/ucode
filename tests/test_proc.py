"""Tests for proc.py — cross-platform npm/CLI command resolution."""

from __future__ import annotations

import os

import ucode.proc as proc
from ucode.proc import cli_command, npm_command


class TestCliCommand:
    def test_returns_none_when_not_on_path(self, monkeypatch):
        monkeypatch.setattr(proc.shutil, "which", lambda name: None)
        assert cli_command("claude", "mcp", "list") is None

    def test_posix_uses_resolved_path(self, monkeypatch):
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(proc.shutil, "which", lambda name: "/usr/local/bin/claude")
        assert cli_command("claude", "mcp", "list") == [
            "/usr/local/bin/claude",
            "mcp",
            "list",
        ]

    def test_windows_wraps_cmd_wrapper(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setattr(proc.shutil, "which", lambda name: r"C:\npm\claude.cmd")
        assert cli_command("claude", "mcp", "list") == [
            "cmd",
            "/c",
            r"C:\npm\claude.cmd",
            "mcp",
            "list",
        ]

    def test_windows_wraps_bat_wrapper(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setattr(proc.shutil, "which", lambda name: r"C:\npm\npm.bat")
        assert cli_command("npm", "install")[:3] == ["cmd", "/c", r"C:\npm\npm.bat"]

    def test_windows_real_exe_unwrapped(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setattr(proc.shutil, "which", lambda name: r"C:\tools\claude.exe")
        assert cli_command("claude") == [r"C:\tools\claude.exe"]


class TestNpmCommand:
    def test_delegates_to_cli_command(self, monkeypatch):
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(proc.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert npm_command("install", "-g", "pkg") == [
            "/usr/bin/npm",
            "install",
            "-g",
            "pkg",
        ]

    def test_none_when_npm_missing(self, monkeypatch):
        monkeypatch.setattr(proc.shutil, "which", lambda name: None)
        assert npm_command("install") is None
