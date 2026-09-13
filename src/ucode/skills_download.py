"""Download Unity Catalog skills and write them to disk, one flat dir per skill."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import questionary

from ucode.databricks import (
    _http_get_bytes,
    _http_get_json,
    get_databricks_token,
    walk_catalog_schemas,
    workspace_hostname,
)
from ucode.mcp import register_schemaless_skills_connection, setup_mcp_clients
from ucode.skills_state import SkillInstall, record_downloads
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

SKILL_FILES_API_PREFIX = "Skills"

# Parallel skill fetches per schema; writes stay sequential (they prompt).
_MAX_FETCH_WORKERS = 8

# Wall-clock budget for the workspace-wide skill walk; a slow workspace degrades
# to partial results instead of hanging the picker.
_SKILLS_WALK_DEADLINE_SECONDS = 30.0


# --- Download client (UC skills API + Files API) ---------------------------


@dataclass(frozen=True)
class SkillRef:
    """A downloadable skill's UC location plus its two non-interchangeable names.

    ``catalog``/``schema``/``securable_name`` are the parts of ``skills/<cat>.<sch>.<leaf>``:
    ``securable_name`` is the leaf, the only name the Files API resolves, and the
    three together fully qualify the skill (``fqn``). ``bundle_name`` is the
    ``name:`` an agent reads from the bundle's SKILL.md frontmatter, so it names
    the on-disk directory. Finalize does not require the securable and bundle name
    to match, so a skill created under a securable that differs from its
    frontmatter carries both. ``description`` is the skill's UC description, used only
    to preview a skill in the interactive picker. ``metastore_id``, ``skill_id``, and
    ``uc_update_time`` are UC attribution metadata recorded when the skill is
    downloaded (see ``skills_state``); the download flow never reads them.
    """

    catalog: str
    schema: str
    securable_name: str
    bundle_name: str
    description: str | None = None
    metastore_id: str | None = None
    skill_id: str | None = None
    uc_update_time: str | None = None

    @property
    def fqn(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.securable_name}"


def _non_empty_str(value: object) -> str | None:
    """``value`` when it is a non-empty string, else None."""
    return value if isinstance(value, str) and value else None


def _skill_ref(skill: dict) -> SkillRef | None:
    """A finalized skill's ``SkillRef``, or None if it cannot be downloaded.

    A skill without a ``finalize_time`` has no bundle content yet and is skipped
    quietly, since that is a normal in-progress state.

    A finalized skill is expected to carry both names: ``name`` is immutable from
    creation, and finalize is the sole writer of ``bundle_name``. One missing is
    therefore an anomaly, so warn and skip rather than substituting the other
    name -- the two are not interchangeable, and guessing a directory name that
    doesn't match the bundle's SKILL.md ``name:`` would hide the skill from the
    agent meant to load it.
    """
    if not skill.get("finalize_time"):
        return None

    name = _non_empty_str(skill.get("name"))
    bundle_name = _non_empty_str(skill.get("bundle_name"))
    if name is None or bundle_name is None:
        missing = " or ".join(
            field
            for field, value in (("name", name), ("bundle_name", bundle_name))
            if value is None
        )
        print_warning(
            f"Skipping `{name or '<unnamed skill>'}`: the skills API returned no {missing}."
        )
        return None

    parts = name.split("/", 1)[-1].split(".")
    if len(parts) != 3:
        print_warning(f"Skipping `{name}`: expected a `catalog.schema.name` skill name.")
        return None
    catalog, schema, securable_name = parts
    return SkillRef(
        catalog=catalog,
        schema=schema,
        securable_name=securable_name,
        bundle_name=bundle_name,
        description=_non_empty_str(skill.get("description")),
        metastore_id=_non_empty_str(skill.get("metastore_id")),
        skill_id=_non_empty_str(skill.get("id")),
        uc_update_time=_non_empty_str(skill.get("update_time")),
    )


def list_schema_skills(
    workspace: str, token: str, catalog: str, schema: str
) -> tuple[list[SkillRef], str | None]:
    """List the finalized skills in ``<catalog>.<schema>``.

    A non-None reason indicates the listing call itself failed.
    """
    hostname = workspace_hostname(workspace)
    base_url = f"https://{hostname}/api/2.1/unity-catalog/skills"
    query = {"parent": f"schemas/{catalog}.{schema}"}

    refs: list[SkillRef] = []
    page_token: str | None = None
    while True:
        if page_token:
            query["page_token"] = page_token
        payload, reason = _http_get_json(f"{base_url}?{urlencode(query)}", token, timeout=30)
        if payload is None:
            return [], reason
        data = payload if isinstance(payload, dict) else {}
        for skill in data.get("skills") or []:
            ref = _skill_ref(skill) if isinstance(skill, dict) else None
            if ref:
                refs.append(ref)
        page_token = data.get("next_page_token")
        if not page_token:
            return refs, None


def list_skill_files(
    workspace: str, token: str, catalog: str, schema: str, securable: str
) -> tuple[list[str], str | None]:
    """List a skill bundle's files, as paths relative to the skill directory.

    Recursively walks the skill's Files API directory (including ``SKILL.md``).
    Takes the securable leaf, the only name the Files API resolves. A non-None
    reason indicates the listing call itself failed.
    """
    hostname = workspace_hostname(workspace)
    dirs_base = f"https://{hostname}/api/2.0/fs/directories"
    skill_prefix = f"/{SKILL_FILES_API_PREFIX}/{catalog}/{schema}/{securable}/"

    relative_paths: list[str] = []
    pending = [f"{SKILL_FILES_API_PREFIX}/{catalog}/{schema}/{securable}"]
    while pending:
        directory = pending.pop()
        page_token: str | None = None
        while True:
            url = f"{dirs_base}/{directory}"
            if page_token:
                url = f"{url}?{urlencode({'page_token': page_token})}"
            payload, reason = _http_get_json(url, token, timeout=30)
            if payload is None:
                return [], reason
            data = payload if isinstance(payload, dict) else {}
            for entry in data.get("contents") or []:
                path = entry.get("path") if isinstance(entry, dict) else None
                if not isinstance(path, str):
                    continue
                if entry.get("is_directory"):
                    pending.append(path.strip("/"))
                else:
                    relative_paths.append(path.removeprefix(skill_prefix))
            page_token = data.get("next_page_token")
            if not page_token:
                break
    return relative_paths, None


def fetch_skill_file(
    workspace: str, token: str, catalog: str, schema: str, securable: str, relative_path: str
) -> tuple[bytes | None, str | None]:
    """Fetch one skill bundle file's raw bytes from the Files API."""
    hostname = workspace_hostname(workspace)
    url = (
        f"https://{hostname}/api/2.0/fs/files/"
        f"{SKILL_FILES_API_PREFIX}/{catalog}/{schema}/{securable}/{relative_path}"
    )
    return _http_get_bytes(url, token, timeout=30)


def fetch_skill_bundle(
    workspace: str, token: str, catalog: str, schema: str, securable: str
) -> tuple[dict[str, bytes] | None, str | None]:
    """Fetch a whole skill bundle as ``{relative_path: bytes}``.

    Lists the skill's files then fetches each one. All-or-nothing: a non-None
    reason (and None bundle) means the listing or any file fetch failed, so a
    partially-downloaded skill is never written to disk.
    """
    relative_paths, reason = list_skill_files(workspace, token, catalog, schema, securable)
    if reason:
        return None, reason
    bundle: dict[str, bytes] = {}
    for relative_path in relative_paths:
        content, reason = fetch_skill_file(
            workspace, token, catalog, schema, securable, relative_path
        )
        if content is None:
            return None, reason
        bundle[relative_path] = content
    return bundle, None


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
    return [
        SkillInstall(
            fqn=ref.fqn,
            bundle_name=ref.bundle_name,
            workspace=workspace,
            scope=scope,
            base=base,
            dirs=tuple(str(root / ref.bundle_name) for root in roots),
            metastore_id=ref.metastore_id,
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


def get_skill(workspace: str, token: str, fqn: str) -> SkillRef | None:
    """The finalized skill named by ``fqn``, or None if it cannot be downloaded.

    ``GetSkill`` returns the same shape as a ``ListSkills`` entry, so the response
    runs through ``_skill_ref``; a missing, unfinalized, or malformed skill is None.
    """
    hostname = workspace_hostname(workspace)
    payload, _ = _http_get_json(
        f"https://{hostname}/api/2.1/unity-catalog/skills/{fqn}", token, timeout=30
    )
    return _skill_ref(payload) if isinstance(payload, dict) else None


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


# --- Interactive picker (workspace discovery + selective download) ----------


def list_all_skills(
    workspace: str,
    token: str,
    *,
    deadline_seconds: float = _SKILLS_WALK_DEADLINE_SECONDS,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_skills: Callable[[list[SkillRef]], None] | None = None,
) -> tuple[list[SkillRef], str | None]:
    """Return every finalized skill across all ``<catalog>.<schema>`` in the workspace, by FQN.

    The skills API is one-schema-per-call, so this walks catalogs -> schemas ->
    skills in parallel under a wall-clock budget, returning partial results once
    ``deadline_seconds`` is exceeded. ``on_progress`` is called as each schema
    completes with ``(schemas_done, schemas_total, skills_found)``, and
    ``on_skills`` with each schema's newly-found refs (deduped by FQN against
    everything emitted so far) so a picker can stream them in as the walk runs.
    The workspace-wide counterpart to ``list_schema_skills``.
    """
    deadline = time.monotonic() + deadline_seconds
    by_fqn: dict[str, SkillRef] = {}

    def probe(catalog: str, schema: str) -> tuple[list[SkillRef], str | None]:
        return list_schema_skills(workspace, token, catalog, schema)

    def collect(result: tuple[list[SkillRef], str | None], done: int, total: int) -> None:
        found, _ = result
        new = [ref for ref in found if ref.fqn not in by_fqn]
        for ref in new:
            by_fqn[ref.fqn] = ref
        if on_progress is not None:
            on_progress(done, total, len(by_fqn))
        if on_skills is not None and new:
            on_skills(sorted(new, key=lambda ref: ref.fqn))

    reason = walk_catalog_schemas(workspace, token, deadline=deadline, probe=probe, collect=collect)
    if reason is not None:
        return [], reason
    if not by_fqn:
        if time.monotonic() > deadline:
            return [], "deadline exceeded while listing skills"
        return [], "no skills found"
    return sorted(by_fqn.values(), key=lambda ref: ref.fqn), None


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
