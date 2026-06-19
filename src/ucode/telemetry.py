"""Helpers for building the outbound `User-Agent` we attach to agent traffic.

The gateway uses the UA to attribute requests to voxcode and to a specific
wrapped agent + version. Format: `voxcode/<ver> <agent>/<agent_ver>`.

Both helpers fall back to "unknown" rather than raising — telemetry must
never block a launch.
"""

from __future__ import annotations

import re
import subprocess
from functools import cache
from importlib.metadata import PackageNotFoundError, version

_SEMVER_RE = re.compile(r"\d+\.\d+\.\d+[-+0-9A-Za-z.]*")


@cache
def voxcode_version() -> str:
    try:
        return version("voxcode")
    except PackageNotFoundError:
        return "unknown"


# Keep backward-compat alias
ucode_version = voxcode_version


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
