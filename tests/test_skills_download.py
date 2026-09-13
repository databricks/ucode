"""Tests for skills_download.py — the UC skill-download client, on-disk writer,
and download orchestration."""

from __future__ import annotations

import pytest

import ucode.skills_download as sd
from ucode import skills_state
from ucode.skills_download import (
    SkillRef,
    existing_skill_on_disk,
    should_download_skill,
    skill_dir_roots,
    write_skill,
)

WS = "https://example.databricks.com"


def ref(
    securable_name: str,
    bundle_name: str | None = None,
    *,
    catalog: str = "main",
    schema: str = "default",
    description: str | None = None,
) -> SkillRef:
    """A SkillRef whose two names match unless a differing bundle name is given."""
    return SkillRef(
        catalog=catalog,
        schema=schema,
        securable_name=securable_name,
        bundle_name=bundle_name or securable_name,
        description=description,
    )


class TestSkillDirRoots:
    def test_roots_under_project_dir(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))
        assert roots == [tmp_path / ".claude/skills", tmp_path / ".agents/skills"]

    def test_defaults_to_home_when_omitted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sd.Path, "home", classmethod(lambda cls: tmp_path))
        roots = skill_dir_roots(None)
        assert roots == [tmp_path / ".claude/skills", tmp_path / ".agents/skills"]

    def test_relative_path_rejected(self):
        with pytest.raises(ValueError, match="absolute"):
            skill_dir_roots("relative/dir")

    def test_missing_directory_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            skill_dir_roots(str(tmp_path / "nope"))


class TestShouldDownloadSkill:
    def test_new_skill_is_downloaded(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))

        assert should_download_skill(roots, ref("triage"))

    def test_existing_skill_prompt_keep(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        write_skill(roots, ref("triage"), {"SKILL.md": b"from-main"})

        monkeypatch.setattr(sd, "prompt_yes_no", lambda _: False)

        assert not should_download_skill(roots, ref("triage"))

    def test_existing_skill_prompt_overwrite(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        write_skill(roots, ref("triage"), {"SKILL.md": b"from-main"})

        monkeypatch.setattr(sd, "prompt_yes_no", lambda _: True)

        assert should_download_skill(roots, ref("triage"))

    def test_existing_skill_on_disk_checks_every_root(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))
        assert not existing_skill_on_disk(roots, "triage")

        (roots[1] / "triage").mkdir(parents=True)
        assert existing_skill_on_disk(roots, "triage")


class TestWriteSkill:
    def test_writes_bundle_into_every_root(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))
        files = {"SKILL.md": b"# skill", "scripts/run.py": b"print(1)"}

        write_skill(roots, ref("triage"), files)

        for root in roots:
            assert (root / "triage/SKILL.md").read_bytes() == b"# skill"
            assert (root / "triage/scripts/run.py").read_bytes() == b"print(1)"

    def test_path_traversal_is_rejected(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))

        write_skill(
            roots, ref("triage"), {"SKILL.md": b"ok", "../escape.md": b"nope", "/abs.md": b"nope"}
        )

        assert (roots[0] / "triage/SKILL.md").read_bytes() == b"ok"
        assert not (tmp_path / "escape.md").exists()


class TestFetchBundles:
    def test_empty_leaves_returns_empty_without_pool(self):
        # min(workers, 0) would raise ValueError in ThreadPoolExecutor; the
        # early return keeps _fetch_bundles safe regardless of caller.
        assert sd._fetch_bundles(WS, "token", [], label="main.default") == {}


class TestDownloadSkillsFromSchemaLocations:
    def test_fetches_and_writes_each_leaf(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda *a, **k: ([ref("pii-handling"), ref("triage")], None)
        )
        bundles = {
            "pii-handling": {"SKILL.md": b"pii"},
            "triage": {"SKILL.md": b"triage"},
        }
        monkeypatch.setattr(
            sd, "fetch_skill_bundle", lambda ws, tok, c, s, leaf: (bundles[leaf], None)
        )

        sd.download_skills_from_schema_locations(WS, "token", ["main.default"], str(tmp_path))

        assert (tmp_path / ".claude/skills/pii-handling/SKILL.md").read_bytes() == b"pii"
        assert (tmp_path / ".agents/skills/triage/SKILL.md").read_bytes() == b"triage"

    def test_sibling_bundle_name_collision_keeps_the_first(self, tmp_path, monkeypatch):
        # Only the securable name is unique in a schema, so two siblings can claim
        # one directory. Writing both would silently lose one.
        colliding = [ref("skill-a", "foo"), ref("skill-b", "foo")]
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: (colliding, None))
        bodies = {"skill-a": b"FROM A", "skill-b": b"FROM B"}
        fetched = []
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, securable_name: (
                fetched.append(securable_name) or ({"SKILL.md": bodies[securable_name]}, None)
            ),
        )
        warnings = []
        monkeypatch.setattr(sd, "print_warning", warnings.append)
        monkeypatch.setattr(
            sd, "prompt_yes_no", lambda msg: pytest.fail(f"unexpected prompt: {msg}")
        )

        sd.download_skills_from_schema_locations(WS, "token", ["main.default"], str(tmp_path))

        # The loser is dropped before the fetch, not after paying for it.
        assert fetched == ["skill-a"]
        assert (tmp_path / ".claude/skills/foo/SKILL.md").read_bytes() == b"FROM A"
        assert [d.name for d in (tmp_path / ".claude/skills").iterdir()] == ["foo"]
        assert len(warnings) == 1
        assert "skill-b" in warnings[0] and "already claimed by" in warnings[0]

    def test_same_bundle_name_across_locations_still_prompts(self, tmp_path, monkeypatch):
        # The collision guard is per location, so a later location's same-named
        # skill must still reach the overwrite prompt rather than being dropped.
        by_location = {
            "main.default": [ref("skill-a", "foo")],
            "ml.prod": [ref("skill-b", "foo", catalog="ml", schema="prod")],
        }
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda ws, tok, c, s: (by_location[f"{c}.{s}"], None)
        )
        monkeypatch.setattr(
            sd, "fetch_skill_bundle", lambda ws, tok, c, s, sn: ({"SKILL.md": sn.encode()}, None)
        )
        prompts = []
        monkeypatch.setattr(sd, "prompt_yes_no", lambda msg: bool(prompts.append(msg)) or True)

        sd.download_skills_from_schema_locations(
            WS, "token", ["main.default", "ml.prod"], str(tmp_path)
        )

        assert len(prompts) == 1
        assert (tmp_path / ".claude/skills/foo/SKILL.md").read_bytes() == b"skill-b"

    def test_fetches_by_securable_and_writes_under_bundle_name(self, tmp_path, monkeypatch):
        # The Files API resolves only the securable, while an agent loads the
        # directory matching the bundle's SKILL.md `name:`.
        diverging = ref("task-prioritizer", "task-triage")
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([diverging], None))
        fetched = []
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, securable_name: (
                fetched.append(securable_name) or ({"SKILL.md": b"name: task-triage"}, None)
            ),
        )

        sd.download_skills_from_schema_locations(WS, "token", ["main.default"], str(tmp_path))

        assert fetched == ["task-prioritizer"]
        for base in (".claude/skills", ".agents/skills"):
            assert (tmp_path / base / "task-triage/SKILL.md").read_bytes() == b"name: task-triage"
            assert not (tmp_path / base / "task-prioritizer").exists()

    def test_list_failure_skips_location(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([], "HTTP 404 Not Found"))
        called = []
        monkeypatch.setattr(
            sd, "fetch_skill_bundle", lambda *a, **k: called.append(1) or (None, None)
        )

        sd.download_skills_from_schema_locations(WS, "token", ["main.default"], str(tmp_path))

        assert called == []

    def test_declined_skill_is_not_fetched(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        write_skill(roots, ref("triage"), {"SKILL.md": b"kept"})
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([ref("triage")], None))
        monkeypatch.setattr(sd, "prompt_yes_no", lambda _: False)
        fetched = []
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, leaf: fetched.append(leaf) or ({"SKILL.md": b"new"}, None),
        )

        sd.download_skills_from_schema_locations(WS, "token", ["main.default"], str(tmp_path))

        assert fetched == []
        assert (roots[0] / "triage/SKILL.md").read_bytes() == b"kept"

    def test_bundle_failure_skips_that_skill_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda *a, **k: ([ref("good"), ref("bad")], None)
        )
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, leaf: (
                ({"SKILL.md": b"ok"}, None) if leaf == "good" else (None, "HTTP 500 Server Error")
            ),
        )

        sd.download_skills_from_schema_locations(WS, "token", ["main.default"], str(tmp_path))

        assert (tmp_path / ".claude/skills/good/SKILL.md").read_bytes() == b"ok"
        assert not (tmp_path / ".claude/skills/bad").exists()

    def test_prints_downloaded_count_and_roots_summary(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda *a, **k: ([ref("a"), ref("b"), ref("c")], None)
        )
        monkeypatch.setattr(sd, "fetch_skill_bundle", lambda *a, **k: ({"SKILL.md": b"x"}, None))

        sd.download_skills_from_schema_locations(WS, "token", ["main.default"], str(tmp_path))

        # Rich wraps long paths across lines; strip all whitespace from both sides to compare.
        roots = sd.skill_dir_roots(str(tmp_path))
        # No skips, so the summary carries no "; N skipped" suffix.
        expected = f"Downloaded 3/3 skill(s) from `main.default` in {roots[0]} and {roots[1]}."
        printed = "".join(capsys.readouterr().out.split())
        assert "".join(expected.split()) in printed
        assert "skipped" not in printed

    def test_summary_counts_only_written_skills(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda *a, **k: ([ref("good"), ref("bad")], None)
        )
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, leaf: (
                ({"SKILL.md": b"ok"}, None) if leaf == "good" else (None, "HTTP 500 Server Error")
            ),
        )

        sd.download_skills_from_schema_locations(WS, "token", ["main.default"], str(tmp_path))

        assert (
            "Downloaded 1/2 skill(s); 1 skipped from `main.default` in" in capsys.readouterr().out
        )

    def test_empty_schema_reports_no_skills_found(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([], None))

        sd.download_skills_from_schema_locations(WS, "token", ["main.default"], str(tmp_path))

        assert "No skills found in `main.default`." in capsys.readouterr().out


class TestDownloadRefs:
    def test_fetches_each_ref_from_its_own_schema(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        refs = [ref("triage"), ref("pii", catalog="ml", schema="prod")]
        fetched = []
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, leaf: (
                fetched.append((c, s, leaf)) or ({"SKILL.md": leaf.encode()}, None)
            ),
        )

        written, total = sd._download_refs(WS, "token", refs, roots, label="picked")

        assert (len(written), total) == (2, 2)
        assert sorted(fetched) == [("main", "default", "triage"), ("ml", "prod", "pii")]
        assert (tmp_path / ".claude/skills/triage/SKILL.md").read_bytes() == b"triage"
        assert (tmp_path / ".agents/skills/pii/SKILL.md").read_bytes() == b"pii"

    def test_bundle_name_collision_deduped_across_schemas(self, tmp_path, monkeypatch):
        # The per-location path dedups within a schema; a flat selection can pair
        # two schemas' skills claiming one directory, so the core dedups the set.
        roots = skill_dir_roots(str(tmp_path))
        refs = [
            ref("skill-a", "shared"),
            ref("skill-b", "shared", catalog="ml", schema="prod"),
        ]
        fetched = []
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, leaf: fetched.append(leaf) or ({"SKILL.md": leaf.encode()}, None),
        )
        warnings = []
        monkeypatch.setattr(sd, "print_warning", warnings.append)

        written, total = sd._download_refs(WS, "token", refs, roots, label="picked")

        assert (len(written), total) == (1, 1)
        assert fetched == ["skill-a"]
        assert (tmp_path / ".claude/skills/shared/SKILL.md").read_bytes() == b"skill-a"
        assert len(warnings) == 1
        assert "ml.prod.skill-b" in warnings[0] and "main.default.skill-a" in warnings[0]

    def test_failed_fetch_counts_toward_total_but_not_written(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        refs = [ref("good"), ref("bad", catalog="ml", schema="prod")]
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, leaf: (
                ({"SKILL.md": b"ok"}, None) if leaf == "good" else (None, "HTTP 500 Server Error")
            ),
        )

        written, total = sd._download_refs(WS, "token", refs, roots, label="picked")

        assert (len(written), total) == (1, 2)
        assert (tmp_path / ".claude/skills/good/SKILL.md").read_bytes() == b"ok"
        assert not (tmp_path / ".claude/skills/bad").exists()


class TestDownloadSelectedSkills:
    def test_downloads_each_resolved_fqn(self, tmp_path, monkeypatch):
        by_fqn = {
            "main.default.triage": ref("triage"),
            "ml.prod.pii": ref("pii", catalog="ml", schema="prod"),
        }
        monkeypatch.setattr(sd, "get_skill", lambda ws, tok, fqn: by_fqn[fqn])
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, leaf: ({"SKILL.md": leaf.encode()}, None),
        )

        sd.download_selected_skills(WS, "token", list(by_fqn), str(tmp_path))

        assert (tmp_path / ".claude/skills/triage/SKILL.md").read_bytes() == b"triage"
        assert (tmp_path / ".agents/skills/pii/SKILL.md").read_bytes() == b"pii"

    def test_unresolvable_fqn_warns_and_skips(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sd, "get_skill", lambda ws, tok, fqn: ref("triage") if fqn.endswith("triage") else None
        )
        monkeypatch.setattr(
            sd, "fetch_skill_bundle", lambda ws, tok, c, s, leaf: ({"SKILL.md": b"x"}, None)
        )
        warnings = []
        monkeypatch.setattr(sd, "print_warning", warnings.append)

        sd.download_selected_skills(
            WS, "token", ["main.default.gone", "main.default.triage"], str(tmp_path)
        )

        assert (tmp_path / ".claude/skills/triage/SKILL.md").exists()
        assert any("main.default.gone" in w for w in warnings)

    def test_prints_one_summary_for_the_selection(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sd, "get_skill", lambda ws, tok, fqn: ref(fqn.rsplit(".", 1)[-1]))
        monkeypatch.setattr(
            sd, "fetch_skill_bundle", lambda ws, tok, c, s, leaf: ({"SKILL.md": b"x"}, None)
        )

        sd.download_selected_skills(
            WS, "token", ["main.default.a", "main.default.b"], str(tmp_path)
        )

        out = capsys.readouterr().out
        assert "Downloaded 2/2 skill(s)" in out
        assert "skipped" not in out

    def test_records_downloaded_skills(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sd, "get_skill", lambda ws, tok, fqn: ref(fqn.rsplit(".", 1)[-1]))
        monkeypatch.setattr(
            sd, "fetch_skill_bundle", lambda ws, tok, c, s, leaf: ({"SKILL.md": b"x"}, None)
        )

        sd.download_selected_skills(WS, "token", ["main.default.triage"], str(tmp_path))

        record = skills_state.attribution_for_dir(tmp_path / ".claude/skills/triage")
        assert record is not None
        assert record["fqn"] == "main.default.triage"
        assert record["scope"] == "project"
        assert record["base"] == str(tmp_path)
        assert "workspace_id" not in record

    def test_records_workspace_id_when_known(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sd, "get_skill", lambda ws, tok, fqn: ref(fqn.rsplit(".", 1)[-1]))
        monkeypatch.setattr(
            sd, "fetch_skill_bundle", lambda ws, tok, c, s, leaf: ({"SKILL.md": b"x"}, None)
        )
        monkeypatch.setattr(sd, "workspace_org_id", lambda ws: "org-42")

        sd.download_selected_skills(WS, "token", ["main.default.triage"], str(tmp_path))

        record = skills_state.attribution_for_dir(tmp_path / ".claude/skills/triage")
        assert record["workspace_id"] == "org-42"


class TestDownloadManagedSkillsOnLaunch:
    def test_writes_missing_skills_and_returns_their_bundle_names(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda *a, **k: ([ref("triage"), ref("pii")], None)
        )
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, leaf: ({"SKILL.md": leaf.encode()}, None),
        )

        written = sd.download_managed_skills_on_launch(WS, "token", ["main.default"], str(tmp_path))

        assert sorted(written) == ["pii", "triage"]
        assert (tmp_path / ".claude/skills/triage/SKILL.md").read_bytes() == b"triage"
        assert (tmp_path / ".agents/skills/pii/SKILL.md").read_bytes() == b"pii"

    def test_skips_already_downloaded_skills_without_prompting(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        write_skill(roots, ref("triage"), {"SKILL.md": b"kept"})
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda *a, **k: ([ref("triage"), ref("pii")], None)
        )
        fetched = []
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, leaf: fetched.append(leaf) or ({"SKILL.md": b"new"}, None),
        )
        monkeypatch.setattr(sd, "prompt_yes_no", lambda msg: pytest.fail(f"prompted: {msg}"))

        written = sd.download_managed_skills_on_launch(WS, "token", ["main.default"], str(tmp_path))

        # Only the missing one is fetched; the existing skill is left untouched.
        assert fetched == ["pii"]
        assert written == ["pii"]
        assert (roots[0] / "triage/SKILL.md").read_bytes() == b"kept"

    def test_nothing_missing_fetches_nothing(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        write_skill(roots, ref("triage"), {"SKILL.md": b"kept"})
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([ref("triage")], None))
        monkeypatch.setattr(
            sd, "fetch_skill_bundle", lambda *a, **k: pytest.fail("should not fetch")
        )

        assert (
            sd.download_managed_skills_on_launch(WS, "token", ["main.default"], str(tmp_path)) == []
        )

    def test_list_failure_warns_and_skips_location(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([], "HTTP 404 Not Found"))
        monkeypatch.setattr(
            sd, "fetch_skill_bundle", lambda *a, **k: pytest.fail("should not fetch")
        )

        assert (
            sd.download_managed_skills_on_launch(WS, "token", ["main.default"], str(tmp_path)) == []
        )
        assert "Could not list workspace skills in `main.default`" in capsys.readouterr().out

    def test_bundle_failure_skips_that_skill_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda *a, **k: ([ref("good"), ref("bad")], None)
        )
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, leaf: (
                ({"SKILL.md": b"ok"}, None) if leaf == "good" else (None, "HTTP 500 Server Error")
            ),
        )

        written = sd.download_managed_skills_on_launch(WS, "token", ["main.default"], str(tmp_path))

        assert written == ["good"]
        assert (tmp_path / ".claude/skills/good/SKILL.md").read_bytes() == b"ok"
        assert not (tmp_path / ".claude/skills/bad").exists()

    def test_malformed_location_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda *a, **k: pytest.fail("should not list a bad location")
        )

        assert (
            sd.download_managed_skills_on_launch(WS, "token", ["not-a-schema"], str(tmp_path)) == []
        )


class TestConfigureLocationSkillsDownloadCommand:
    def _stub(self, monkeypatch):
        calls: dict[str, object] = {}
        monkeypatch.setattr(sd, "load_state", lambda: {"state": True})
        monkeypatch.setattr(
            sd, "setup_mcp_clients", lambda state, section: (WS, "profile", ["claude"])
        )
        monkeypatch.setattr(sd, "get_databricks_token", lambda ws, profile: "token")
        monkeypatch.setattr(
            sd,
            "download_skills_from_schema_locations",
            lambda ws, tok, locations, path: calls.update(download=(ws, tok, locations, path)),
        )
        monkeypatch.setattr(
            sd,
            "register_schemaless_skills_connection",
            lambda state, ws, profile, clients: calls.update(register=(ws, profile, clients)),
        )
        return calls

    def test_downloads_then_registers_connection(self, monkeypatch):
        calls = self._stub(monkeypatch)

        assert sd.configure_location_skills_download_command(["a.b"], path="/tmp/skills") == 0

        assert calls["download"] == (WS, "token", ["a.b"], "/tmp/skills")
        assert calls["register"] == (WS, "profile", ["claude"])

    def test_none_path_threads_through(self, monkeypatch):
        calls = self._stub(monkeypatch)

        assert sd.configure_location_skills_download_command(["a.b"], path=None) == 0

        assert calls["download"] == (WS, "token", ["a.b"], None)
        assert calls["register"] == (WS, "profile", ["claude"])


class TestConfigureSelectedSkillsDownloadCommand:
    def _stub(self, monkeypatch):
        calls: dict[str, object] = {}
        monkeypatch.setattr(sd, "load_state", lambda: {"state": True})
        monkeypatch.setattr(
            sd, "setup_mcp_clients", lambda state, section: (WS, "profile", ["claude"])
        )
        monkeypatch.setattr(sd, "get_databricks_token", lambda ws, profile: "token")
        monkeypatch.setattr(
            sd,
            "download_selected_skills",
            lambda ws, tok, fqns, path: calls.update(download=(ws, tok, fqns, path)),
        )
        monkeypatch.setattr(
            sd,
            "register_schemaless_skills_connection",
            lambda state, ws, profile, clients: calls.update(register=(ws, profile, clients)),
        )
        return calls

    def test_downloads_selected_then_registers(self, monkeypatch):
        calls = self._stub(monkeypatch)

        fqns = ["a.b.s1", "c.d.s2"]
        assert sd.configure_selected_skills_download_command(fqns, "/tmp/skills") == 0

        assert calls["download"] == (WS, "token", fqns, "/tmp/skills")
        assert calls["register"] == (WS, "profile", ["claude"])

    def test_none_path_threads_through(self, monkeypatch):
        calls = self._stub(monkeypatch)

        assert sd.configure_selected_skills_download_command(["a.b.s1"], None) == 0

        assert calls["download"] == (WS, "token", ["a.b.s1"], None)
        assert calls["register"] == (WS, "profile", ["claude"])


class _FakePrompt:
    def __init__(self, result):
        self._result = result

    def ask(self):
        return self._result


class TestSkillDownloadPicker:
    def test_choice_value_is_fqn_flags_on_disk_and_carries_description(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))

        fresh = sd._skill_download_choice(ref("triage", description="Routes tickets."), roots)
        assert fresh.value == "main.default.triage"
        assert "(on disk)" not in fresh.title
        assert fresh.description == "triage: Routes tickets."

        write_skill(roots, ref("triage"), {"SKILL.md": b"x"})
        existing = sd._skill_download_choice(ref("triage"), roots)
        assert existing.value == "main.default.triage"
        assert "(on disk)" in existing.title

    def test_choice_description_labels_by_bundle_name_not_securable(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))
        diverging = ref("task-prioritizer", "task-triage", description="Ranks work.")

        choice = sd._skill_download_choice(diverging, roots)

        assert choice.description == "task-triage: Ranks work."

    def test_choice_without_description_has_no_footer_text(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))

        assert sd._skill_download_choice(ref("triage"), roots).description is None

    def test_background_loader_streams_the_walk_in_as_choices(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        captured = {}

        def fake_list_all(ws, tok, *, on_skills=None, **kwargs):
            captured["token"] = tok
            on_skills([ref("triage"), ref("scoring", catalog="ml", schema="prod")])
            return [], None

        monkeypatch.setattr(sd, "list_all_skills", fake_list_all)
        appended = []

        sd._skills_download_background_loader(WS, "token", roots)(appended.extend)

        assert captured["token"] == "token"
        assert [c.value for c in appended] == ["main.default.triage", "ml.prod.scoring"]

    def test_prompt_returns_selected_fqns(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        loader = lambda append: None  # noqa: E731
        captured = {}

        def fake_checkbox(message, *, choices, instruction, style, background_loader, **kwargs):
            captured.update(background_loader=background_loader, **kwargs)
            return _FakePrompt(["main.default.triage", "ml.prod.scoring"])

        monkeypatch.setattr(sd, "scrolling_checkbox", fake_checkbox)

        assert sd.prompt_for_skill_download_choices(roots, loader) == [
            "main.default.triage",
            "ml.prod.scoring",
        ]
        assert captured["loading_noun"] == "skills"
        assert captured["show_description"] is True
        assert captured["background_loader"] is loader

    def test_prompt_returns_none_on_cancel(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        monkeypatch.setattr(sd, "scrolling_checkbox", lambda *a, **k: _FakePrompt(None))

        assert sd.prompt_for_skill_download_choices(roots, lambda append: None) is None


class TestConfigureSkillsDownloadPickerCommand:
    def _stub(self, monkeypatch, fqns):
        calls: dict[str, object] = {}
        monkeypatch.setattr(sd, "load_state", lambda: {"state": True})
        monkeypatch.setattr(
            sd, "setup_mcp_clients", lambda state, section: (WS, "profile", ["claude"])
        )
        monkeypatch.setattr(sd, "get_databricks_token", lambda ws, profile: "token")
        monkeypatch.setattr(
            sd, "_skills_download_background_loader", lambda ws, token, roots: "loader"
        )
        monkeypatch.setattr(sd, "prompt_for_skill_download_choices", lambda roots, loader: fqns)
        monkeypatch.setattr(
            sd,
            "download_selected_skills",
            lambda ws, tok, selected, path: calls.update(download=(ws, tok, selected, path)),
        )
        monkeypatch.setattr(
            sd,
            "register_schemaless_skills_connection",
            lambda state, ws, profile, clients: calls.update(register=(ws, profile, clients)),
        )
        return calls

    def test_downloads_selected_then_registers(self, tmp_path, monkeypatch):
        calls = self._stub(monkeypatch, ["main.default.triage"])

        assert sd.configure_skills_download_picker_command(path=str(tmp_path)) == 0

        assert calls["download"] == (WS, "token", ["main.default.triage"], str(tmp_path))
        assert calls["register"] == (WS, "profile", ["claude"])

    def test_cancel_downloads_nothing_and_skips_register(self, monkeypatch):
        calls = self._stub(monkeypatch, None)

        assert sd.configure_skills_download_picker_command() == 0

        assert "download" not in calls
        assert "register" not in calls


def _seed_downloads(monkeypatch, fqns: list[str], path: str) -> None:
    """Download ``fqns`` to ``path`` with a stubbed API, writing dirs and attribution."""

    def fake_get(ws, tok, fqn):
        catalog, schema, leaf = fqn.split(".")
        return ref(leaf, catalog=catalog, schema=schema)

    monkeypatch.setattr(sd, "get_skill", fake_get)
    monkeypatch.setattr(
        sd, "fetch_skill_bundle", lambda ws, tok, c, s, leaf: ({"SKILL.md": b"x"}, None)
    )
    sd.download_selected_skills(WS, "token", fqns, path)


class TestRemoveDownloadedSkillsCommand:
    def test_by_location_removes_across_all_bases(self, tmp_path, monkeypatch):
        home, proj = tmp_path / "home", tmp_path / "proj"
        home.mkdir()
        proj.mkdir()
        _seed_downloads(monkeypatch, ["main.default.triage"], str(home))
        _seed_downloads(monkeypatch, ["main.default.triage"], str(proj))
        _seed_downloads(monkeypatch, ["ml.prod.pii"], str(home))

        sd.remove_downloaded_skills_command(["main.default"], path=None)

        assert not (home / ".claude/skills/triage").exists()
        assert not (proj / ".claude/skills/triage").exists()
        assert (home / ".claude/skills/pii").exists()
        assert [r["fqn"] for r in skills_state.list_downloaded()] == ["ml.prod.pii"]

    def test_path_narrows_removal_to_one_base(self, tmp_path, monkeypatch):
        home, proj = tmp_path / "home", tmp_path / "proj"
        home.mkdir()
        proj.mkdir()
        _seed_downloads(monkeypatch, ["main.default.triage"], str(home))
        _seed_downloads(monkeypatch, ["main.default.triage"], str(proj))

        sd.remove_downloaded_skills_command(["main.default"], path=str(proj))

        assert (home / ".claude/skills/triage").exists()
        assert not (proj / ".claude/skills/triage").exists()
        assert {r["base"] for r in skills_state.list_downloaded()} == {str(home)}

    def test_user_authored_dir_without_record_is_untouched(self, tmp_path, monkeypatch):
        _seed_downloads(monkeypatch, ["main.default.triage"], str(tmp_path))
        mine = tmp_path / ".claude/skills/mine"
        mine.mkdir(parents=True)
        (mine / "SKILL.md").write_text("mine")

        sd.remove_downloaded_skills_command(["main.default"], path=None)

        assert mine.exists()
        assert not (tmp_path / ".claude/skills/triage").exists()

    def test_unknown_location_reports_and_keeps_records(self, tmp_path, monkeypatch, capsys):
        _seed_downloads(monkeypatch, ["main.default.triage"], str(tmp_path))

        sd.remove_downloaded_skills_command(["other.schema"], path=None)

        assert "No downloaded skills from `other.schema`" in capsys.readouterr().out
        assert (tmp_path / ".claude/skills/triage").exists()

    def test_by_fqns_removes_only_named_skills(self, tmp_path, monkeypatch):
        _seed_downloads(monkeypatch, ["main.default.triage", "ml.prod.pii"], str(tmp_path))

        sd.remove_downloaded_skills_command([], ["main.default.triage"], path=None)

        assert not (tmp_path / ".claude/skills/triage").exists()
        assert (tmp_path / ".claude/skills/pii").exists()
        assert [r["fqn"] for r in skills_state.list_downloaded()] == ["ml.prod.pii"]

    def test_unknown_fqn_reports_and_keeps_records(self, tmp_path, monkeypatch, capsys):
        _seed_downloads(monkeypatch, ["main.default.triage"], str(tmp_path))

        sd.remove_downloaded_skills_command([], ["main.default.gone"], path=None)

        assert "No downloaded skills matching `main.default.gone`" in capsys.readouterr().out
        assert (tmp_path / ".claude/skills/triage").exists()

    def test_picker_removes_selected(self, tmp_path, monkeypatch):
        _seed_downloads(monkeypatch, ["main.default.triage", "ml.prod.pii"], str(tmp_path))
        monkeypatch.setattr(
            sd,
            "_prompt_for_downloaded_skill_removal",
            lambda records: [r for r in records if r["fqn"] == "main.default.triage"],
        )

        sd.remove_downloaded_skills_command([], path=None)

        assert not (tmp_path / ".claude/skills/triage").exists()
        assert (tmp_path / ".claude/skills/pii").exists()

    def test_picker_labels_records_whose_dirs_are_missing(self, tmp_path, monkeypatch):
        import shutil

        _seed_downloads(monkeypatch, ["main.default.triage", "ml.prod.pii"], str(tmp_path))
        shutil.rmtree(tmp_path / ".claude/skills/triage")
        shutil.rmtree(tmp_path / ".agents/skills/triage")

        by_fqn = {
            r["fqn"]: sd._removal_choice(r, i) for i, r in enumerate(skills_state.list_downloaded())
        }

        assert "(missing)" in by_fqn["main.default.triage"].title
        assert "(missing)" not in by_fqn["ml.prod.pii"].title
