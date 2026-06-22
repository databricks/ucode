"""Subprocess helpers for running npm-installed CLIs across platforms."""

from __future__ import annotations

import os
import shutil


def cli_command(name: str, *args: str) -> list[str] | None:
    """Resolve a ``<name> <args...>`` invocation for a CLI on PATH, or None when
    it isn't installed.

    On Windows, npm installs its CLIs (npm, claude, codex, gemini, ...) as
    ``.cmd`` wrappers. A bare ``"name"`` passed to subprocess fails with WinError
    2 because CreateProcess doesn't apply PATHEXT, and a ``.cmd`` can't be
    executed directly either — so the resolved wrapper is run through ``cmd /c``.
    On POSIX (and for a real executable) the resolved path is used unchanged."""
    exe = shutil.which(name)
    if not exe:
        return None
    if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", exe, *args]
    return [exe, *args]


def npm_command(*args: str) -> list[str] | None:
    """Convenience wrapper for ``cli_command("npm", ...)``."""
    return cli_command("npm", *args)
