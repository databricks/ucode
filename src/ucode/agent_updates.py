"""Update checks for npm-installed coding agent CLIs."""

from __future__ import annotations

import json
import shutil
import subprocess


def available_npm_package_update(package: str) -> tuple[str, str] | None:
    if not shutil.which("npm"):
        return None
    try:
        result = subprocess.run(
            ["npm", "outdated", "-g", "--json", package],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    # npm outdated exits 1 when it finds outdated packages.
    if result.returncode not in (0, 1) or not result.stdout.strip():
        return None
    try:
        outdated = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    package_update = outdated.get(package)
    if not isinstance(package_update, dict):
        return None
    current = package_update.get("current")
    latest = package_update.get("latest")
    if not isinstance(current, str) or not isinstance(latest, str):
        return None
    return current, latest
