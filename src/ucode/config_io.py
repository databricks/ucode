"""File I/O, dry-run flag, backup/restore, deep-merge, dotenv parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import tomlkit
import tomlkit.exceptions

from ucode.ui import console


class ToolSpec(TypedDict):
    binary: str
    package: str
    display: str
    config_path: Path
    backup_path: Path


APP_DIR = Path.home() / ".ucode"

_dry_run = False


def set_dry_run(value: bool) -> None:
    global _dry_run
    _dry_run = bool(value)


def is_dry_run() -> bool:
    return _dry_run


def ensure_parent_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to create directory for {path}") from exc


def backup_existing_file(config_path: Path, backup_path: Path) -> bool:
    if _dry_run:
        return False
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        if backup_path.exists():
            return True
        if not config_path.exists():
            return False
        backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        return True
    except OSError as exc:
        raise RuntimeError(f"Failed to back up config from {config_path}") from exc


def restore_file(config_path: Path, backup_path: Path, managed: bool) -> bool:
    try:
        if backup_path.exists():
            ensure_parent_dir(config_path)
            config_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
            backup_path.unlink()
            return True
        if managed and config_path.exists():
            config_path.unlink()
            return True
        return False
    except OSError as exc:
        raise RuntimeError(f"Failed to restore config at {config_path}") from exc


def write_text_file(path: Path, content: str) -> None:
    if _dry_run:
        console.print(f"\n[bold]\\[dry run] {path}[/bold]\n{content}")
        return
    ensure_parent_dir(path)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write config file: {path}") from exc


def write_json_file(path: Path, payload: dict) -> None:
    content = json.dumps(payload, indent=2) + "\n"
    if _dry_run:
        console.print(f"\n[bold]\\[dry run] {path}[/bold]\n{content}")
        return
    ensure_parent_dir(path)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write config file: {path}") from exc


def deep_merge_dict(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base; overlay wins for conflicting leaves.

    Mutates and returns base. Nested dicts are merged; everything else is replaced.
    """
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            deep_merge_dict(base[key], val)
        else:
            base[key] = val
    return base


def read_json_safe(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_toml_safe(path: Path) -> tomlkit.TOMLDocument:
    if not path.exists():
        return tomlkit.document()
    try:
        return tomlkit.parse(path.read_text(encoding="utf-8"))
    except (OSError, tomlkit.exceptions.TOMLKitError):
        return tomlkit.document()


def write_toml_file(path: Path, doc: tomlkit.TOMLDocument) -> None:
    content = tomlkit.dumps(doc)
    if _dry_run:
        console.print(f"\n[bold]\\[dry run] {path}[/bold]\n{content}")
        return
    ensure_parent_dir(path)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write config file: {path}") from exc


def parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE / KEY="VALUE" .env file, preserving insertion order.

    Comments and blank lines are dropped on round-trip. Lines that don't look
    like KEY=... are skipped.
    """
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        env[key] = val
    return env


def write_dotenv(path: Path, env: dict[str, str]) -> None:
    content = "".join(f'{key}="{val}"\n' for key, val in env.items())
    write_text_file(path, content)
