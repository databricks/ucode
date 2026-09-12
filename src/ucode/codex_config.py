"""Shared helpers for passing Codex configuration on the command line."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import tomlkit
from tomlkit.items import Item

from ucode.managed_files import OS, current_os

CODEX_PROFILE_NAME = "ucode"
DEFAULT_CODEX_CONFIG_PATH = Path.home() / ".codex" / f"{CODEX_PROFILE_NAME}.config.toml"


def codex_managed_config_path() -> Path | None:
    if current_os() in (OS.LINUX, OS.MACOS):
        return Path("/etc/codex/managed_config.toml")
    return None


def codex_config_precedence_paths(
    managed_path: Path | None,
    profile_path: Path = DEFAULT_CODEX_CONFIG_PATH,
) -> tuple[Path, ...]:
    """Return Codex config paths in managed, profile, then user precedence."""
    config_home = os.environ.get("CODEX_HOME")
    if config_home:
        profile_path = Path(config_home).expanduser() / f"{CODEX_PROFILE_NAME}.config.toml"
    return tuple(
        path
        for path in (managed_path, profile_path, profile_path.parent / "config.toml")
        if path is not None
    )


def _toml_item(value: object) -> Item:
    if isinstance(value, Mapping):
        inline = tomlkit.inline_table()
        for key, child in value.items():
            inline[str(key)] = _toml_item(child)
        return inline
    if isinstance(value, list):
        array = tomlkit.array()
        for child in value:
            array.append(_toml_item(child))
        return array
    if isinstance(value, Item):
        return value
    return tomlkit.item(value)


def _toml_value(value: object) -> str:
    return _toml_item(value).as_string()


def codex_config_args(config: dict) -> list[str]:
    """Render a Codex config layer as repeatable ``--config`` overrides."""
    args: list[str] = []
    for key, value in config.items():
        # These maps contain named entries. Override each entry individually so
        # the rest of the user's base map remains intact.
        if key in {"hooks", "model_providers"} and isinstance(value, dict):
            for entry_name, entry_config in value.items():
                args.extend(
                    [
                        "--config",
                        f"{key}.{entry_name}={_toml_value(entry_config)}",
                    ]
                )
        else:
            args.extend(["--config", f"{key}={_toml_value(value)}"])
    return args
