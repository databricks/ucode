"""Helpers for building the outbound `User-Agent` we attach to agent traffic.

The gateway uses the UA to attribute requests to ucode and to a specific
wrapped agent + version. Format: `ucode/<ucode_ver> <agent>/<agent_ver>`.

Both helpers fall back to "unknown" rather than raising — telemetry must
never block a launch.
"""

from __future__ import annotations

import json
import re
import subprocess
from functools import cache
from importlib.metadata import Distribution, PackageNotFoundError, version

_SEMVER_RE = re.compile(r"\d+\.\d+\.\d+[-+0-9A-Za-z.]*")


@cache
def ucode_version() -> str:
    try:
        return version("ucode")
    except PackageNotFoundError:
        return "unknown"


@cache
def ucode_commit() -> str:
    """Return an identifier for the exact installed build of ucode.

    ucode is distributed straight from git (`uv tool install git+…`), so the
    PEP 610 `direct_url.json` metadata records how this copy was installed:
    a git install carries `vcs_info.commit_id` (the exact SHA), while a local
    editable checkout carries `dir_info.editable`. We surface the short SHA for
    git installs, "editable" for a working checkout, and "unknown" otherwise —
    never raising, so `--version` and telemetry stay resilient.
    """
    try:
        raw = Distribution.from_name("ucode").read_text("direct_url.json")
        if not raw:
            return "unknown"
        info = json.loads(raw)
    except (PackageNotFoundError, OSError, ValueError):
        return "unknown"
    commit = info.get("vcs_info", {}).get("commit_id")
    if commit:
        return commit[:7]
    if info.get("dir_info", {}).get("editable"):
        return "editable"
    return "unknown"


def ucode_version_string() -> str:
    """Human-facing version line, e.g. `ucode 0.1.0 (446a24a)`."""
    return f"ucode {ucode_version()} ({ucode_commit()})"


@cache
def agent_version(binary: str) -> str:
    """Return the agent CLI's reported version, or "unknown" on any failure.

    Spawned at most once per binary per session (cached). Each agent CLI
    formats `--version` differently — we extract the first semver-shaped
    token from stdout (then stderr) so the same parser handles all of them.
    """
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "unknown"
    for stream in (result.stdout, result.stderr):
        if not stream:
            continue
        match = _SEMVER_RE.search(stream)
        if match:
            return match.group(0)
    return "unknown"
