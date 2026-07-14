"""Best-effort nudge when the installed ucode build is behind `main`.

ucode is installed straight from git HEAD (`uv tool install git+…`), so "am I
up to date?" means "does my commit match `main`?". We fetch `main`'s HEAD SHA
from the GitHub API, compare it to the installed commit, and print a single
warning when they differ.

Everything here is best-effort: any failure (offline, rate-limited, editable
checkout, unknown build) is swallowed silently. An update check must never
block or break a launch — same principle as `telemetry`.
"""

from __future__ import annotations

import json
import time
import urllib.request

from ucode.config_io import APP_DIR, is_dry_run
from ucode.telemetry import ucode_commit
from ucode.ui import print_warning

CACHE_PATH = APP_DIR / "update-check.json"
# Only hit the network once a day — the nudge is a convenience, not a gate.
CHECK_INTERVAL_S = 24 * 60 * 60
_FETCH_TIMEOUT_S = 2
_MAIN_COMMIT_URL = "https://api.github.com/repos/databricks/ucode/commits/main"


def _fetch_remote_head() -> str | None:
    """Return the short SHA of `main`'s HEAD, or None on any failure."""
    req = urllib.request.Request(
        _MAIN_COMMIT_URL,
        headers={
            "Accept": "application/vnd.github.sha",
            "User-Agent": "ucode-update-check",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_S) as resp:
            sha = resp.read().decode("utf-8").strip()
    except Exception:
        return None
    return sha[:7] if sha else None


def _read_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_cache(payload: dict) -> None:
    if is_dry_run():
        return
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def maybe_warn_if_outdated() -> None:
    """Warn once (per interval) if the installed commit is behind `main`.

    No-ops for editable/unknown builds, when offline, or within the cache TTL.
    Never raises.
    """
    try:
        local = ucode_commit()
        # Nothing to compare against for a working checkout or an install whose
        # provenance we couldn't read.
        if local in ("editable", "unknown"):
            return

        now = time.time()
        cache = _read_cache()
        remote = cache.get("remote_head")
        if not remote or (now - cache.get("checked_at", 0)) >= CHECK_INTERVAL_S:
            remote = _fetch_remote_head()
            if remote:
                _write_cache({"remote_head": remote, "checked_at": now})
        if remote and remote != local:
            print_warning(
                f"A newer ucode is available (installed {local}, latest {remote}). "
                "Run `ucode upgrade` to update."
            )
    except Exception:
        # Telemetry-grade resilience: a broken update check must never surface.
        return
