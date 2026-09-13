"""Download Unity Catalog skills and write them to disk, one flat dir per skill."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import questionary

from ucode.databricks import get_databricks_token, workspace_org_id
from ucode.mcp import register_schemaless_skills_connection, setup_mcp_clients
from ucode.skills_api import (
    SkillRef,
    fetch_skill_bundle,
    get_skill,
    list_all_skills,
    list_schema_skills,
)
from ucode.skills_state import (
    SkillInstall,
    list_downloaded,
    record_downloads,
    records_for_fqns,
    records_for_schema,
    remove_downloads,
)
from ucode.state import load_state
from ucode.ui import (
    console,
    picker_style,
    print_note,
    print_success,
    print_warning,
    progress_bar,
    prompt_yes_no,
    scrolling_checkbox,
)

# `.claude/skills` (Claude) + `.agents/skills` (the alias other agents read).
SKILL_BASE_DIR_NAMES = (".claude/skills", ".agents/skills")

# Parallel skill fetches per schema; writes stay sequential (they prompt).
_MAX_FETCH_WORKERS = 8


# --- On-disk writer --------------------------------------------------------


def skill_dir_roots(project_dir: str | None) -> list[Path]:
    """The ``.claude/skills`` and ``.agents/skills`` roots to download into.

    ``project_dir`` must be an existing absolute directory when given; when
    omitted, roots default to the user's home directory (user scope).
    """
    if project_dir is None:
        base = Path.home()
    else:
        base = Path(project_dir)
        if not base.is_absolute():
            raise ValueError(f"--path must be an absolute path, got `{project_dir}`.")
        if not base.is_dir():
            raise ValueError(f"--path directory does not exist: `{project_dir}`.")
    return [base / name for name in SKILL_BASE_DIR_NAMES]


def _safe_relative_path(relative_path: str) -> Path | None:
    """A bundle file's path within its skill dir, or None if it escapes the dir.

    The Files API returns server-controlled paths, but ucode writes them to
    disk, so reject absolute paths and any ``..`` traversal.
    """
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _write_bundle(skill_dir: Path, bundle_name: str, files: dict[str, bytes]) -> None:
    for relative_path, content in files.items():
        safe_path = _safe_relative_path(relative_path)
        if safe_path is None:
            print_warning(f"Skipping unsafe path in `{bundle_name}`: {relative_path}")
            continue
        destination = skill_dir / safe_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def existing_skill_on_disk(roots: list[Path], bundle_name: str) -> bool:
    """Whether ``bundle_name`` already has a skill directory under any root."""
    return any((root / bundle_name).exists() for root in roots)


def should_download_skill(roots: list[Path], ref: SkillRef) -> bool:
    """Whether ``ref`` should be fetched and written into ``roots``.

    Applies the disk-only check that needs no bundle bytes: prompts before
    overwriting a skill already on disk (naming the source by ``ref.fqn``), so a
    declined skill is never fetched. Dedup keys on the bundle name, since that is
    the directory an agent would load. Name validity is the server's job --
    FinalizeSkill enforces the Agent Skills naming rules on ``bundle_name`` before
    we ever see it -- so ucode does not re-check it here.
    """
    if existing_skill_on_disk(roots, ref.bundle_name) and not prompt_yes_no(
        f"A skill named `{ref.bundle_name}` already exists. Overwrite it with `{ref.fqn}`?"
    ):
        print_note(f"Kept existing `{ref.bundle_name}`.")
        return False

    return True


def write_skill(roots: list[Path], ref: SkillRef, files: dict[str, bytes]) -> None:
    """Write ``ref``'s bundle (``{relpath: bytes}``) into every root.

    The directory is named for the bundle, so it matches the ``name:`` an agent
    reads from the written SKILL.md.
    """
    for root in roots:
        _write_bundle(root / ref.bundle_name, ref.bundle_name, files)


def _skill_installs(
    refs: list[SkillRef], roots: list[Path], path: str | None, workspace: str
) -> list[SkillInstall]:
    """Attribution records for ``refs`` written into ``roots`` (see ``skills_state``)."""
    base = path or str(Path.home())
    scope = "project" if path else "user"
    org_id = workspace_org_id(workspace)
    return [
        SkillInstall(
            fqn=ref.fqn,
            bundle_name=ref.bundle_name,
            workspace=workspace,
            scope=scope,
            base=base,
            dirs=tuple(str(root / ref.bundle_name) for root in roots),
            metastore_id=ref.metastore_id,
            workspace_id=org_id,
            skill_id=ref.skill_id,
            uc_update_time=ref.uc_update_time,
        )
        for ref in refs
    ]


# --- Orchestration ---------------------------------------------------------


def _fetch_bundles(
    workspace: str, token: str, refs: list[SkillRef], *, label: str
) -> dict[str, tuple[dict[str, bytes] | None, str | None]]:
    """Fetch every skill's bundle concurrently, keyed by FQN.

    Renders a ``k/n`` progress bar labeled ``label`` that advances as each fetch
    completes. Keying on the FQN keeps a cross-schema batch's securables apart,
    since a securable name is unique only within its own schema.
    """
    if not refs:
        return {}
    results: dict[str, tuple[dict[str, bytes] | None, str | None]] = {}
    with (
        progress_bar(label, len(refs)) as advance,
        ThreadPoolExecutor(max_workers=min(_MAX_FETCH_WORKERS, len(refs))) as pool,
    ):
        futures = {
            pool.submit(
                fetch_skill_bundle, workspace, token, ref.catalog, ref.schema, ref.securable_name
            ): ref.fqn
            for ref in refs
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            advance()
    return results


def _reject_bundle_name_collisions(refs: list[SkillRef]) -> list[SkillRef]:
    """``refs`` with any later skill that repeats an earlier one's bundle name dropped.

    Only the securable name is unique within a schema; ``bundle_name`` comes from
    each bundle's SKILL.md frontmatter and is never checked against its siblings,
    so two skills can claim the same directory. Writing both would land them on top
    of each other, leaving whichever finished last with no sign the other was lost,
    so keep the first and warn about the rest by FQN.
    """
    kept: list[SkillRef] = []
    claimed: dict[str, SkillRef] = {}
    for ref in refs:
        winner = claimed.get(ref.bundle_name)
        if winner is not None:
            print_warning(
                f"Skipping `{ref.fqn}`: its bundle name `{ref.bundle_name}` is already "
                f"claimed by `{winner.fqn}`. Rename one skill's SKILL.md `name:` to download both."
            )
            continue
        claimed[ref.bundle_name] = ref
        kept.append(ref)
    return kept


def _download_refs(
    workspace: str, token: str, refs: list[SkillRef], roots: list[Path], *, label: str
) -> tuple[list[SkillRef], int]:
    """Fetch and write ``refs`` into ``roots``, returning ``(written, total)``.

    The shared download core: drop siblings claiming one directory
    (``_reject_bundle_name_collisions``), prompt before overwriting a skill already
    on disk (``should_download_skill``, so a declined skill is never fetched), then
    fetch the survivors' bundles concurrently and write them. ``written`` are the
    refs that reached disk, so a caller can record their attribution; ``total`` is
    the count that could reach disk (dropped siblings excluded), so a caller's
    summary denominator is right. A per-skill fetch failure warns and skips it.
    """
    refs = _reject_bundle_name_collisions(refs)
    to_download = [ref for ref in refs if should_download_skill(roots, ref)]
    bundles = _fetch_bundles(workspace, token, to_download, label=label)
    written: list[SkillRef] = []
    for ref in to_download:
        files, reason = bundles[ref.fqn]
        if reason or files is None:
            print_warning(f"Skipping `{ref.fqn}`: {reason}.")
            continue
        write_skill(roots, ref, files)
        written.append(ref)
    console.print()
    return written, len(refs)


def download_skills_from_schema_locations(
    workspace: str,
    token: str,
    locations: list[str],
    path: str | None,
) -> None:
    """Download every skill in each ``<catalog>.<schema>`` location to disk.

    Locations are processed one at a time. Each lists the schema's finalized
    skills, then hands the refs to ``_download_refs`` and prints a per-location
    summary. Finishing one location before the next means a skill written for an
    earlier location is already on disk when a same-named skill in a later location
    reaches the overwrite prompt, so the prompt still fires. Downloading a named
    subset instead of whole schemas is a separate path (``download_selected_skills``
    over fully-qualified names).
    """
    roots = skill_dir_roots(path)
    roots_display = " and ".join(str(root) for root in roots)
    for location in locations:
        catalog, schema = location.split(".")
        refs, reason = list_schema_skills(workspace, token, catalog, schema)
        if reason:
            print_warning(f"Skipping `{location}`: {reason}.")
            continue
        if not refs:
            print_note(f"No skills found in `{location}`.")
            continue
        written, total = _download_refs(
            workspace, token, refs, roots, label=f"Fetching skills from {location}"
        )
        record_downloads(_skill_installs(written, roots, path, workspace))
        count = len(written)
        skipped = f"; {total - count} skipped" if count < total else ""
        print_success(
            f"Downloaded {count}/{total} skill(s){skipped} from `{location}` in {roots_display}."
        )


def download_selected_skills(workspace: str, token: str, fqns: list[str], path: str | None) -> None:
    """Download the skills named by ``fqns`` (``<catalog>.<schema>.<name>``) to disk.

    Resolves each FQN with ``GetSkill`` (a skill that cannot be downloaded warns and
    is skipped), then hands the flat, possibly cross-schema set to ``_download_refs``
    in one pass, so collisions are deduped across the whole selection under a single
    summary.
    """
    roots = skill_dir_roots(path)
    roots_display = " and ".join(str(root) for root in roots)
    refs: list[SkillRef] = []
    for fqn in fqns:
        ref = get_skill(workspace, token, fqn)
        if ref is None:
            print_warning(f"Skipping `{fqn}`: not a downloadable skill.")
            continue
        refs.append(ref)
    written, total = _download_refs(workspace, token, refs, roots, label="Fetching selected skills")
    record_downloads(_skill_installs(written, roots, path, workspace))
    count = len(written)
    skipped = f"; {total - count} skipped" if count < total else ""
    print_success(f"Downloaded {count}/{total} skill(s){skipped} in {roots_display}.")


def download_managed_skills_on_launch(
    workspace: str, token: str, locations: list[str], path: str | None = None
) -> list[str]:
    """Download admin-published skills to disk so the agent's ``/skills`` lists them.

    Runs on the managed launch path: the config only registers the skills MCP
    connection, so nothing else writes the bundles that ``/skills`` reads. Writes
    only skills not already on disk -- no overwrite prompt, so the launch never
    blocks on input and a developer's own same-named skill is never clobbered.
    Best-effort and never raises, so it can't block the launch. Returns the bundle
    names newly written.
    """
    roots = skill_dir_roots(path)
    written: list[str] = []
    for location in locations:
        if location.count(".") != 1:
            continue
        catalog, schema = location.split(".")
        refs, reason = list_schema_skills(workspace, token, catalog, schema)
        if reason:
            print_warning(f"Could not list workspace skills in `{location}`: {reason}.")
            continue
        refs = _reject_bundle_name_collisions(refs)
        missing = [ref for ref in refs if not existing_skill_on_disk(roots, ref.bundle_name)]
        if not missing:
            continue
        bundles = _fetch_bundles(
            workspace, token, missing, label=f"Fetching skills from {location}"
        )
        installed: list[SkillRef] = []
        for ref in missing:
            files, reason = bundles[ref.fqn]
            if reason or files is None:
                print_warning(f"Skipping `{ref.fqn}`: {reason}.")
                continue
            write_skill(roots, ref, files)
            installed.append(ref)
            written.append(ref.bundle_name)
        record_downloads(_skill_installs(installed, roots, path, workspace))
    return written


def configure_location_skills_download_command(locations: list[str], *, path: str | None) -> int:
    """Download every skill in each schema to disk and register the skills connection.

    Downloads to ``path`` (or the home dir when None), then registers/keeps the
    schema-less MCP connection. ``skill_locations`` is never touched, so a prior
    ``--mcp`` set survives a download run. Downloading a named subset instead of whole
    schemas is a separate command (``configure_selected_skills_download_command``)."""
    state = load_state()
    workspace, profile, clients = setup_mcp_clients(state, "Skills")
    token = get_databricks_token(workspace, profile)

    download_skills_from_schema_locations(workspace, token, locations, path)

    register_schemaless_skills_connection(state, workspace, profile, clients)
    return 0


def configure_selected_skills_download_command(fqns: list[str], path: str | None) -> int:
    """Download the fully-qualified, possibly cross-schema ``fqns`` and register the connection.

    The non-interactive counterpart to the picker: downloads the named skills with
    ``download_selected_skills`` (which alone does not register), then registers/keeps
    the schema-less MCP connection, exactly as the whole-schema download does."""
    state = load_state()
    workspace, profile, clients = setup_mcp_clients(state, "Skills")
    token = get_databricks_token(workspace, profile)

    download_selected_skills(workspace, token, fqns, path)

    register_schemaless_skills_connection(state, workspace, profile, clients)
    return 0


# --- Interactive picker (selective download) --------------------------------


def _skill_download_choice(ref: SkillRef, roots: list[Path]) -> questionary.Choice:
    """Picker row for one skill: value is its FQN, title flags an on-disk bundle.

    On-disk skills stay selectable, since re-downloading is a legitimate update and
    the existing overwrite prompt confirms it. The detail footer previews the
    description behind a bold bundle-name label (the row itself shows the FQN, so the
    bundle name is the one identifier not otherwise on screen).
    """
    on_disk = " (on disk)" if existing_skill_on_disk(roots, ref.bundle_name) else ""
    description = f"{ref.bundle_name}: {ref.description}" if ref.description else None
    return questionary.Choice(title=f"{ref.fqn}{on_disk}", value=ref.fqn, description=description)


def _skills_download_background_loader(
    workspace: str, token: str, roots: list[Path]
) -> Callable[[Callable[[list[questionary.Choice]], None]], None]:
    """A picker ``background_loader`` that streams the workspace-wide skill walk in as choices."""

    def loader(append: Callable[[list[questionary.Choice]], None]) -> None:
        def on_skills(refs: list[SkillRef]) -> None:
            append([_skill_download_choice(ref, roots) for ref in refs])

        list_all_skills(workspace, token, on_skills=on_skills)

    return loader


def prompt_for_skill_download_choices(
    roots: list[Path],
    background_loader: Callable[[Callable[[list[questionary.Choice]], None]], None],
) -> list[str] | None:
    """Show the skill-download picker, returning the selected FQNs or None on Ctrl-C."""
    selection = scrolling_checkbox(
        "Skills:",
        choices=[],
        instruction="(space to toggle, ctrl-a all, enter to save, type to filter)",
        style=picker_style(),
        background_loader=background_loader,
        loading_noun="skills",
        show_description=True,
    ).ask()
    if selection is None:
        return None
    return [str(value) for value in selection]


def configure_skills_download_picker_command(path: str | None = None) -> int:
    """Pick skills from an interactive workspace-wide list, download them, and register.

    Opens the picker immediately and streams skills in as discovery finds them.
    Ctrl-C downloads nothing and leaves the connection untouched.
    """
    state = load_state()
    workspace, profile, clients = setup_mcp_clients(state, "Skills")
    token = get_databricks_token(workspace, profile)
    roots = skill_dir_roots(path)

    loader = _skills_download_background_loader(workspace, token, roots)
    fqns = prompt_for_skill_download_choices(roots, loader)
    if fqns is None:
        return 0

    download_selected_skills(workspace, token, fqns, path)
    register_schemaless_skills_connection(state, workspace, profile, clients)
    return 0


# --- Removing and listing downloaded skills ---------------------------------


def _record_dirs_missing(record: dict) -> bool:
    """Whether any of a record's on-disk directories no longer exists."""
    return any(not Path(directory).exists() for directory in record.get("dirs") or [])


def _download_label(record: dict) -> str:
    label = f"{record.get('fqn')}  ({record.get('scope')}: {record.get('base')})"
    return f"{label}  (missing)" if _record_dirs_missing(record) else label


def _removal_choice(record: dict, index: int) -> questionary.Choice:
    """Picker row for one downloaded skill, labeled by its scope and base (and missing dirs)."""
    return questionary.Choice(title=_download_label(record), value=index)


def _prompt_for_downloaded_skill_removal(records: list[dict]) -> list[dict] | None:
    """Checklist of downloaded skills to remove, across every base.

    Returns the selected records, ``None`` if cancelled (Ctrl-C), or ``[]`` if nothing
    is checked. Only recorded downloads are offered, so a user-authored skill directory
    with no attribution can never be selected.
    """
    if not records:
        print_note("No downloaded skills to remove.")
        return []
    choices = [_removal_choice(record, index) for index, record in enumerate(records)]
    selection = scrolling_checkbox(
        "Remove downloaded skills:",
        choices=choices,
        style=picker_style(),
        instruction="(space to toggle, ctrl-a all, enter to remove, type to filter)",
    ).ask()
    if selection is None:
        return None
    return [records[int(index)] for index in selection]


def remove_downloaded_skills_command(
    locations: list[str], fqns: list[str] | None = None, *, path: str | None
) -> int:
    """`ug skill remove` (download side): delete downloaded skills and forget them.

    With ``fqns``, removes those fully-qualified skills; with ``locations``, every skill
    downloaded from those ``<catalog>.<schema>`` schemas; with neither, opens a picker over
    every downloaded skill. ``path`` limits any of these to one download base. Removal is
    driven entirely by attribution, so a same-named skill the user authored is never touched.
    """
    if fqns is not None:
        records = records_for_fqns(set(fqns), path)
        if not records:
            scope = f" under `{path}`" if path else ""
            joined = ", ".join(f"`{fqn}`" for fqn in fqns) or "those names"
            print_note(f"No downloaded skills matching {joined}{scope}.")
            return 0
    elif locations:
        records = [
            record for location in locations for record in records_for_schema(location, path)
        ]
        if not records:
            scope = f" under `{path}`" if path else ""
            joined = ", ".join(f"`{location}`" for location in locations)
            print_note(f"No downloaded skills from {joined}{scope}.")
            return 0
    else:
        selected = _prompt_for_downloaded_skill_removal(list_downloaded())
        if selected is None:
            return 0
        if not selected:
            print_note("No skills selected.")
            return 0
        records = selected

    remove_downloads(records)
    print_success(f"Removed {len(records)} downloaded skill(s).")
    return 0
