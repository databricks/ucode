"""Read-only client for the Unity Catalog skills API: list and get skills, and fetch bundles.

The workspace-facing read layer shared by the download flow (``skills_download``) and the
skills MCP scope picker (``mcp``); it depends only on ``databricks`` so neither of those has
to import the other.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlencode

from ucode.databricks import (
    _http_get_bytes,
    _http_get_json,
    walk_catalog_schemas,
    workspace_hostname,
)
from ucode.ui import print_warning

SKILL_FILES_API_PREFIX = "Skills"

# Wall-clock budget for the workspace-wide skill walk; a slow workspace degrades
# to partial results instead of hanging the picker.
_SKILLS_WALK_DEADLINE_SECONDS = 30.0


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
