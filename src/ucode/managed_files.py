"""Own and write agent config in OS-level *managed settings* files.

These files are root-owned and the highest-precedence config scope for their agent — a bare
``claude`` / ``codex`` (launched directly, without ucode) reads them, so writing here is what makes
the gateway config apply outside ``ucode <agent>``:

- Claude Code: ``/etc/claude-code/managed-settings.json`` (Linux),
  ``/Library/Application Support/ClaudeCode/managed-settings.json`` (macOS)
- Codex: ``/etc/codex/managed_config.toml`` (Linux + macOS)

ucode uses strict file ownership: it creates an adjacent ``.ucode-owner.json`` marker when it first
creates a managed file, and only updates or removes files carrying a valid marker whose content hash
still matches. A pre-existing unowned file is never modified. The marker persists the owning
workspace across sessions, allowing workspace A's file to become workspace B's and then be removed
when workspace C has no global managed config, without retaining a baseline copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path

from ucode.config_io import is_dry_run
from ucode.ui import console, print_err, print_warning

# Absolute path so a stripped PATH (desktop/GUI launchers) still finds it.
_SUDO = "/usr/bin/sudo"
_OWNER_VERSION = 1


class OS(Enum):
    """The host OS families this module distinguishes, off `sys.platform`."""

    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"
    OTHER = "other"


def current_os() -> OS:
    """Map `sys.platform` onto :class:`OS` (lowercased, so a mixed-case value can't slip through)."""
    platform = sys.platform.lower()
    if platform.startswith("linux"):
        return OS.LINUX
    if platform == "darwin":
        return OS.MACOS
    if platform.startswith("win"):
        return OS.WINDOWS
    return OS.OTHER


def managed_files_supported() -> bool:
    """True on the platforms whose managed-settings write path is implemented (Linux, macOS).

    The write path needs `sudo` (`sudo cp`, `chattr`/`chflags`), which is Unix-only — so Windows and
    any other platform are unsupported.
    """
    return current_os() in (OS.LINUX, OS.MACOS)


def _read_existing(path: Path) -> str:
    """Current file contents, or "" when absent. No sudo — the managed file is world-readable."""
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:
        return ""


def _path_exists(path: Path) -> bool | None:
    """Whether ``path`` exists, or None when permissions prevent determining it."""
    try:
        return path.exists()
    except OSError:
        return None


def ownership_path(path: Path) -> Path:
    """The durable ownership marker adjacent to ``path``."""
    return path.with_name(f"{path.name}.ucode-owner.json")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_owner(path: Path) -> dict | None:
    marker = ownership_path(path)
    marker_exists = _path_exists(marker)
    if marker_exists is False:
        return None
    if marker_exists is None:
        return {}
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    if (
        value.get("version") != _OWNER_VERSION
        or value.get("owner") != "ucode"
        or not isinstance(value.get("workspace"), str)
        or not isinstance(value.get("sha256"), str)
    ):
        return {}
    return value


def _owner_text(workspace: str, desired_text: str) -> str:
    return (
        json.dumps(
            {
                "version": _OWNER_VERSION,
                "owner": "ucode",
                "workspace": workspace,
                "sha256": _content_hash(desired_text),
            },
            indent=2,
        )
        + "\n"
    )


def write_managed_file(path: Path, desired_text: str, *, display: str, workspace: str) -> str:
    """Write ``desired_text`` only when ucode strictly owns ``path``.

    Returns ``"written"``, ``"unchanged"``, or ``"skipped"``. Never raises: a permission or immutable
    failure is surfaced as an actionable message and reported as ``"skipped"`` so the launch still
    proceeds (the private ucode config already lets ``ucode <agent>`` work).
    """
    if not managed_files_supported():
        print_warning(
            f"{display}: machine-wide managed settings aren't supported on this platform; "
            f"skipped {path}."
        )
        return "skipped"
    existing = _read_existing(path)
    owner = _load_owner(path)
    target_exists = _path_exists(path)
    if owner is None and target_exists is not False:
        print_warning(
            f"{display}: {path} already exists and is not owned by ucode; "
            "leaving it unchanged. Its machine-wide settings may override this ucode launch."
        )
        return "skipped"
    if owner == {}:
        print_warning(f"{display}: ownership metadata for {path} is invalid; leaving it unchanged.")
        return "skipped"
    if owner is not None and target_exists is None:
        print_warning(f"{display}: cannot verify ownership of {path}; leaving it unchanged.")
        return "skipped"
    if owner is not None and target_exists and owner["sha256"] != _content_hash(existing):
        print_warning(f"{display}: {path} changed after ucode wrote it; leaving it unchanged.")
        return "skipped"
    marker = ownership_path(path)
    marker_text = _owner_text(workspace, desired_text)
    if existing == desired_text and owner is not None and owner["workspace"] == workspace:
        return "unchanged"
    if is_dry_run():
        console.print(f"\n[bold]\\[dry run] {path} + {marker} (via sudo)[/bold]\n{desired_text}")
        return "written"
    existed = target_exists is True
    marker_existed = owner is not None
    existing_marker = _read_existing(marker) if marker_existed else ""
    try:
        _sudo_replace(path, desired_text)
        _sudo_replace(marker, marker_text)
    except PermissionError as exc:
        _rollback_owned_write(path, existing, existed, marker, existing_marker, marker_existed)
        print_err(
            f"{display}: cannot write {path} without root ({exc}). Re-run with `sudo ucode ...` to "
            "apply the config machine-wide."
        )
        return "skipped"
    except subprocess.CalledProcessError as exc:
        _rollback_owned_write(path, existing, existed, marker, existing_marker, marker_existed)
        _report_sudo_failure(path, display, exc)
        return "skipped"
    return "written"


def remove_managed_file(path: Path, *, display: str) -> str:
    """Remove ``path`` only when its ownership marker and content still match.

    Returns ``"removed"`, ``"unchanged"``, or ``"skipped"``. An unmarked file is not ours and is
    therefore unchanged; drift is skipped so ucode never deletes a file another actor took over.
    """
    marker = ownership_path(path)
    owner = _load_owner(path)
    if owner is None:
        return "unchanged"
    if not managed_files_supported():
        print_warning(
            f"{display}: machine-wide managed settings aren't supported on this platform; "
            f"could not remove {path}."
        )
        return "skipped"
    if owner == {}:
        print_warning(f"{display}: ownership metadata for {path} is invalid; leaving it unchanged.")
        return "skipped"
    target_exists = _path_exists(path)
    if target_exists is None:
        print_warning(f"{display}: cannot verify ownership of {path}; leaving it unchanged.")
        return "skipped"
    if target_exists and owner["sha256"] != _content_hash(_read_existing(path)):
        print_warning(f"{display}: {path} changed after ucode wrote it; leaving it unchanged.")
        return "skipped"
    if is_dry_run():
        console.print(f"\n[bold]\\[dry run] remove {path} + {marker} (via sudo)[/bold]")
        return "removed"
    try:
        if target_exists:
            _sudo_remove(path)
        _sudo_remove(marker)
    except PermissionError as exc:
        print_err(
            f"{display}: cannot remove {path} without root ({exc}). Re-run with `sudo ucode ...` "
            "to clear the stale workspace-managed settings."
        )
        return "skipped"
    except subprocess.CalledProcessError as exc:
        _report_sudo_failure(path, display, exc)
        return "skipped"
    return "removed"


def _sudo_replace(path: Path, desired_text: str) -> None:
    """Replace ``path`` with ``desired_text`` via sudo (temp file → ``sudo cp``), handling immutability.

    Writes the payload to a user-owned temp file first (no sudo), then copies it into place with
    ``sudo`` and makes it world-readable so the file it lays down is readable by the agent binary
    regardless of who launched it.
    """
    subprocess.run([_SUDO, "mkdir", "-p", str(path.parent)], check=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=path.suffix or ".tmp", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(desired_text)
        tmp_path = tmp.name
    try:
        restore_immutable = _clear_immutable(path)
        try:
            # capture_output so the CalledProcessError on failure (e.g. still-immutable dest) carries
            # cp's stderr for an actionable message.
            subprocess.run(
                [_SUDO, "cp", tmp_path, str(path)], capture_output=True, text=True, check=True
            )
            subprocess.run([_SUDO, "chmod", "a+rx", str(path.parent)], check=True)
            subprocess.run([_SUDO, "chmod", "a+r", str(path)], check=True)
        finally:
            if restore_immutable:
                _restore_immutable(path)
    finally:
        os.unlink(tmp_path)


def _sudo_remove(path: Path) -> None:
    """Remove a ucode-owned managed file or marker through sudo."""
    _clear_immutable(path)
    subprocess.run([_SUDO, "rm", "-f", str(path)], capture_output=True, text=True, check=True)


def _rollback_owned_write(
    path: Path,
    existing: str,
    existed: bool,
    marker: Path,
    existing_marker: str,
    marker_existed: bool,
) -> None:
    """Best-effort rollback when updating the managed file and marker fails partway through."""
    try:
        if existed:
            _sudo_replace(path, existing)
        else:
            _sudo_remove(path)
        if marker_existed:
            _sudo_replace(marker, existing_marker)
        else:
            _sudo_remove(marker)
    except (PermissionError, subprocess.CalledProcessError):
        print_warning(
            f"Could not fully roll back a failed managed-settings update at {path}; "
            "inspect the file and its ucode ownership marker before retrying."
        )


def _clear_immutable(path: Path) -> bool:
    """Clear an immutable flag a fleet golden image may have set. Returns whether to restore it.

    macOS: preserve JAMF's system-immutable ``schg`` across the update — inspect, unlock only when
    set, and report that it must be restored. Linux: best-effort ``chattr -i`` (not every filesystem
    supports it), never restored.
    """
    try:
        # `path.exists()` stats the file; under a root-locked parent dir (e.g. a 750 /etc/codex we
        # haven't opened yet) that raises PermissionError. There's nothing to unlock we can see, and
        # the subsequent `sudo cp` (as root) overwrites regardless, so treat it as "nothing to clear".
        if not path.exists():
            return False
    except OSError:
        return False
    if current_os() is OS.MACOS:
        result = subprocess.run(
            ["/usr/bin/stat", "-f", "%Sf", str(path)], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and "schg" in result.stdout.strip().split(","):
            subprocess.run(
                [_SUDO, "chflags", "noschg", str(path)], capture_output=True, text=True, check=True
            )
            return True
        return False
    subprocess.run([_SUDO, "chattr", "-i", str(path)], capture_output=True, text=True, check=False)
    return False


def _restore_immutable(path: Path) -> None:
    """Re-set macOS's ``schg`` flag after a write. Best-effort so it never masks the write result."""
    result = subprocess.run(
        [_SUDO, "chflags", "schg", str(path)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        print_warning(f"Could not restore the immutable flag on {path}.")


def _report_sudo_failure(path: Path, display: str, exc: subprocess.CalledProcessError) -> None:
    """Surface a sudo helper failure with a concrete fix. An immutable destination is the common
    cause — cp fails with EPERM even under root — so point at the OS-specific clear command."""
    stderr = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
    cmd = exc.cmd or []
    cp_failed = len(cmd) >= 2 and cmd[1] == "cp"
    if cp_failed and "Operation not permitted" in stderr:
        quoted = shlex.quote(str(path))
        clear_cmd = f"sudo {'chflags noschg' if current_os() is OS.MACOS else 'chattr -i'} {quoted}"
        print_err(
            f"{display}: {path} appears to be immutable. Clear the immutable attribute and re-run:\n"
            f"  {clear_cmd}\n  ucode ..."
        )
    else:
        print_err(f"{display}: failed to write managed settings at {path}: {stderr or exc}")
