"""Shared Codex configuration helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import tomlkit
from tomlkit.items import Item

from ucode.config_io import read_json_safe, read_toml_safe
from ucode.managed_files import OS, current_os
from ucode.ui import print_warning

CODEX_PROFILE_NAME = "ucode"
DEFAULT_CODEX_CONFIG_PATH = Path.home() / ".codex" / f"{CODEX_PROFILE_NAME}.config.toml"


def codex_managed_config_path() -> Path | None:
    if current_os() in (OS.LINUX, OS.MACOS):
        return Path("/etc/codex/managed_config.toml")
    return None


def codex_config_precedence_paths(
    managed_path: Path | None,
    profile_path: Path,
) -> tuple[Path, ...]:
    """Return Codex config paths in managed, profile, then user precedence."""
    config_home = os.environ.get("CODEX_HOME")
    if config_home:
        profile_path = Path(config_home).expanduser() / f"{CODEX_PROFILE_NAME}.config.toml"

    # Highest precedence: machine-managed settings, normally /etc/codex/managed_config.toml.
    managed_config = managed_path
    # Middle precedence: Unity Gateway's ucode.config.toml layer passed to Codex via the CLI.
    cli_config = profile_path
    # Lowest precedence: $CODEX_HOME/config.toml, or ~/.codex/config.toml by default.
    default_config = profile_path.parent / "config.toml"
    return tuple(
        path
        for path in (managed_config, cli_config, default_config)
        if path is not None
    )


def custom_catalog_models() -> list[str] | None:
    """Read model slugs from the configured model_catalog_json, if present."""
    try:
        paths = codex_config_precedence_paths(
            codex_managed_config_path(),
            DEFAULT_CODEX_CONFIG_PATH,
        )
    except OSError:
        return None
    for path in paths:
        if not path.is_file():
            continue
        settings = read_toml_safe(path)
        catalog_ref = settings.get("model_catalog_json")
        if not isinstance(catalog_ref, str) or not catalog_ref.strip():
            continue
        slugs = _catalog_slugs(Path(catalog_ref).expanduser())
        if slugs:
            return slugs
        print_warning(
            f"Codex smart routing could not read models from the custom catalog {catalog_ref} "
            f"referenced by {path}; falling back to the cached model services."
        )
        return None
    return None


def _catalog_slugs(path: Path) -> list[str]:
    """Extract deduplicated model slugs from a Codex custom catalog JSON file."""
    catalog = read_json_safe(path)
    models = catalog.get("models")
    if not isinstance(models, list):
        return []
    slugs: list[str] = []
    seen: set[str] = set()
    for row in models:
        if not isinstance(row, dict) or not isinstance(row.get("slug"), str):
            continue
        slug = row["slug"].strip()
        if not slug or slug in seen:
            continue
        seen.add(slug)
        slugs.append(slug)
    return slugs


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
