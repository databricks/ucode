"""On-disk record of which skill directories were downloaded from Unity Catalog.

A single manifest, ``~/.ucode/skills.json``, links each downloaded skill install to
its UC source and its on-disk directories, so ucode can tell a downloaded skill from
a user-authored one and remove downloads by their UC schema. Kept separate from
``state.json`` so a state-version change and ``ug revert`` leave it untouched.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from ucode import config_io
from ucode.ui import print_warning

SKILLS_STATE_VERSION = 1


@dataclass(frozen=True)
class SkillInstall:
    """One skill written to one download base.

    ``base`` is the download root (a ``--path`` project dir or the user's home);
    ``dirs`` are the ``.claude`` and ``.agents`` skill directories written under it,
    which share a lifecycle. ``(metastore_id, fqn, base)`` is the logical key that
    identifies a prior install of the same skill at the same base.
    """

    fqn: str
    bundle_name: str
    workspace: str
    scope: str
    base: str
    dirs: tuple[str, ...]
    metastore_id: str | None = None
    skill_id: str | None = None
    uc_update_time: str | None = None


def _skills_state_path() -> Path:
    return config_io.APP_DIR / "skills.json"


def _quarantine_corrupt(path: Path) -> None:
    """Rename an unparseable manifest aside so the next write starts fresh without losing its bytes."""
    backup = path.with_name(f"{path.name}.corrupt-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    try:
        os.replace(path, backup)
        print_warning(f"Unreadable skills manifest {path}; moved it to {backup} and started fresh.")
    except OSError:
        pass


def _load() -> list[dict]:
    """Records in the manifest; ``[]`` if it is absent, unreadable, or an unrecognized version.

    A file that fails to parse is quarantined (see ``_quarantine_corrupt``) rather than read as
    empty, so a single bad byte doesn't let the next write silently erase every tracked skill.
    """
    path = _skills_state_path()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        _quarantine_corrupt(path)
        return []
    if not isinstance(data, dict) or data.get("version") != SKILLS_STATE_VERSION:
        return []
    downloads = data.get("skill_downloads")
    return [r for r in downloads if isinstance(r, dict)] if isinstance(downloads, list) else []


def _save(downloads: list[dict]) -> None:
    config_io.atomic_write_json(
        _skills_state_path(), {"version": SKILLS_STATE_VERSION, "skill_downloads": downloads}
    )


def _norm(path: str) -> str:
    return os.path.normpath(path)


def _record_key(record: dict) -> tuple[str | None, str | None, str]:
    return (record.get("metastore_id"), record.get("fqn"), _norm(record.get("base", "")))


def _same_install(record: dict, install: SkillInstall) -> bool:
    return _record_key(record) == (install.metastore_id, install.fqn, _norm(install.base))


def _claims_any(record: dict, dirs: set[str]) -> bool:
    return any(_norm(d) in dirs for d in record.get("dirs") or [])


def _delete_dirs(dirs: list[str]) -> None:
    for directory in dirs:
        shutil.rmtree(directory, ignore_errors=True)


def _to_record(install: SkillInstall) -> dict:
    record = {
        "fqn": install.fqn,
        "bundle_name": install.bundle_name,
        "metastore_id": install.metastore_id,
        "workspace": install.workspace,
        "scope": install.scope,
        "base": install.base,
        "dirs": list(install.dirs),
        "skill_id": install.skill_id,
        "uc_update_time": install.uc_update_time,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return {key: value for key, value in record.items() if value is not None}


def _reconcile(skills: list[dict], install: SkillInstall) -> list[dict]:
    """Drop records superseded by ``install`` before it is appended.

    The install's directories were just written, so a record still claiming one of
    them is stale attribution for content we overwrote: drop it, keep the files. A
    prior install of the same skill at the same base whose directories differ (the
    bundle name changed) leaves an orphaned copy on disk, so delete those.
    """
    target = {_norm(d) for d in install.dirs}
    kept: list[dict] = []
    for record in skills:
        if _claims_any(record, target):
            continue
        if _same_install(record, install):
            _delete_dirs(record.get("dirs") or [])
            continue
        kept.append(record)
    return kept


def record_downloads(installs: list[SkillInstall]) -> None:
    """Record each freshly downloaded install, reconciling superseded records."""
    if not installs:
        return
    skills = _load()
    for install in installs:
        skills = _reconcile(skills, install)
        skills.append(_to_record(install))
    _save(skills)


def list_downloaded() -> list[dict]:
    """Every recorded skill install, across all download bases."""
    return _load()


def attribution_for_dir(path: str | Path) -> dict | None:
    """The install whose directories include ``path``, or None if unattributed."""
    target = _norm(str(path))
    for record in list_downloaded():
        if target in {_norm(d) for d in record.get("dirs") or []}:
            return record
    return None


def records_for_schema(location: str, base: str | None = None) -> list[dict]:
    """Installs downloaded from ``<catalog>.<schema>``, optionally under one base."""
    prefix = f"{location}."
    base_norm = _norm(base) if base is not None else None
    return [
        record
        for record in list_downloaded()
        if record.get("fqn", "").startswith(prefix)
        and (base_norm is None or _norm(record.get("base", "")) == base_norm)
    ]


def forget(records: list[dict]) -> None:
    """Drop ``records`` from the manifest, leaving their on-disk directories alone."""
    if not records:
        return
    dropped = {_record_key(record) for record in records}
    _save([record for record in _load() if _record_key(record) not in dropped])
