"""Tests for skills_state.py — the downloaded-skill attribution manifest."""

from __future__ import annotations

import json
from pathlib import Path

from ucode import config_io, skills_state
from ucode.skills_state import SkillInstall


def _install(
    base: Path,
    fqn: str,
    bundle_name: str,
    *,
    metastore_id: str = "metastore-1",
    skill_id: str = "skill-1",
) -> SkillInstall:
    dirs = tuple(
        str(base / family / bundle_name) for family in (".claude/skills", ".agents/skills")
    )
    return SkillInstall(
        fqn=fqn,
        bundle_name=bundle_name,
        workspace="https://example.databricks.com",
        scope="project",
        base=str(base),
        dirs=dirs,
        metastore_id=metastore_id,
        skill_id=skill_id,
        uc_update_time="2026-01-01T00:00:00Z",
    )


def _write_dirs(install: SkillInstall) -> None:
    for directory in install.dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "SKILL.md").write_text("bundle")


class TestReadWrite:
    def test_record_round_trips(self, tmp_path):
        base = tmp_path / "proj"
        skills_state.record_downloads([_install(base, "main.default.triage", "triage")])

        records = skills_state.list_downloaded()
        assert len(records) == 1
        assert records[0]["fqn"] == "main.default.triage"
        assert records[0]["metastore_id"] == "metastore-1"
        assert records[0]["dirs"] == [
            str(base / ".claude/skills/triage"),
            str(base / ".agents/skills/triage"),
        ]
        assert "downloaded_at" in records[0]

    def test_missing_file_is_empty(self):
        assert skills_state.list_downloaded() == []

    def test_corrupt_file_is_quarantined_then_reads_empty(self, capsys):
        (config_io.APP_DIR / "skills.json").write_text("{ not json")

        assert skills_state.list_downloaded() == []

        assert list(config_io.APP_DIR.glob("skills.json.corrupt-*"))
        assert not (config_io.APP_DIR / "skills.json").exists()
        assert "Unreadable skills manifest" in capsys.readouterr().out

    def test_write_after_corruption_starts_fresh(self, tmp_path):
        (config_io.APP_DIR / "skills.json").write_text("{ not json")
        skills_state.list_downloaded()

        skills_state.record_downloads([_install(tmp_path, "main.default.triage", "triage")])

        assert [r["fqn"] for r in skills_state.list_downloaded()] == ["main.default.triage"]

    def test_unrecognized_version_reads_empty(self):
        (config_io.APP_DIR / "skills.json").write_text(
            json.dumps({"version": 999, "skill_downloads": [{"fqn": "a.b.c"}]})
        )
        assert skills_state.list_downloaded() == []

    def test_save_leaves_no_temp_files(self, tmp_path):
        skills_state.record_downloads([_install(tmp_path, "main.default.triage", "triage")])

        assert (config_io.APP_DIR / "skills.json").exists()
        assert not list(config_io.APP_DIR.glob(".skills.json.*"))

    def test_absent_uc_fields_are_omitted(self, tmp_path):
        install = SkillInstall(
            fqn="main.default.triage",
            bundle_name="triage",
            workspace="https://example.databricks.com",
            scope="user",
            base=str(tmp_path),
            dirs=(str(tmp_path / ".claude/skills/triage"),),
        )
        skills_state.record_downloads([install])

        record = skills_state.list_downloaded()[0]
        assert "metastore_id" not in record
        assert "skill_id" not in record


class TestQueries:
    def test_attribution_for_dir(self, tmp_path):
        base = tmp_path / "proj"
        skills_state.record_downloads([_install(base, "main.default.triage", "triage")])

        assert skills_state.attribution_for_dir(base / ".claude/skills/triage")["fqn"] == (
            "main.default.triage"
        )
        assert skills_state.attribution_for_dir(base / ".claude/skills/other") is None

    def test_records_for_schema_filters_location_and_base(self, tmp_path):
        home, proj = tmp_path / "home", tmp_path / "proj"
        skills_state.record_downloads(
            [
                _install(home, "main.default.triage", "triage"),
                _install(proj, "main.default.triage", "triage"),
                _install(home, "ml.prod.pii", "pii"),
            ]
        )

        assert len(skills_state.records_for_schema("main.default")) == 2
        assert {r["base"] for r in skills_state.records_for_schema("main.default")} == {
            str(home),
            str(proj),
        }
        assert len(skills_state.records_for_schema("main.default", base=str(proj))) == 1
        assert skills_state.records_for_schema("ml.prod")[0]["fqn"] == "ml.prod.pii"

    def test_records_for_fqns_filters_by_exact_name_and_base(self, tmp_path):
        home, proj = tmp_path / "home", tmp_path / "proj"
        skills_state.record_downloads(
            [
                _install(home, "main.default.triage", "triage"),
                _install(proj, "main.default.triage", "triage"),
                _install(home, "ml.prod.pii", "pii"),
            ]
        )

        picked = skills_state.records_for_fqns({"main.default.triage", "ml.prod.pii"})
        assert {r["fqn"] for r in picked} == {"main.default.triage", "ml.prod.pii"}
        assert len(picked) == 3
        assert len(skills_state.records_for_fqns({"main.default.triage"}, base=str(proj))) == 1
        assert skills_state.records_for_fqns({"nope.nope.nope"}) == []

    def test_forget_drops_only_named_records(self, tmp_path):
        home = tmp_path / "home"
        keep = _install(home, "ml.prod.pii", "pii")
        drop = _install(home, "main.default.triage", "triage")
        skills_state.record_downloads([keep, drop])

        skills_state.forget([skills_state.records_for_schema("main.default")[0]])

        remaining = skills_state.list_downloaded()
        assert [r["fqn"] for r in remaining] == ["ml.prod.pii"]


class TestReconciliation:
    def test_re_download_same_name_keeps_single_record(self, tmp_path):
        base = tmp_path / "proj"
        first = _install(base, "main.default.triage", "triage")
        skills_state.record_downloads([first])
        skills_state.record_downloads([first])

        assert len(skills_state.list_downloaded()) == 1

    def test_re_download_renamed_bundle_deletes_orphan(self, tmp_path):
        base = tmp_path / "proj"
        old = _install(base, "main.default.triage", "triage")
        new = _install(base, "main.default.triage", "triage-renamed")
        _write_dirs(old)
        _write_dirs(new)

        skills_state.record_downloads([old])
        skills_state.record_downloads([new])

        records = skills_state.list_downloaded()
        assert len(records) == 1
        assert records[0]["bundle_name"] == "triage-renamed"
        assert not Path(old.dirs[0]).exists()
        assert Path(new.dirs[0]).exists()

    def test_target_dir_claimed_by_other_skill_drops_stale_record_keeps_files(self, tmp_path):
        base = tmp_path / "proj"
        existing = _install(base, "main.default.old", "shared", metastore_id="metastore-1")
        replacement = _install(base, "ml.prod.new", "shared", metastore_id="metastore-2")
        _write_dirs(existing)
        skills_state.record_downloads([existing])

        skills_state.record_downloads([replacement])

        records = skills_state.list_downloaded()
        assert [r["fqn"] for r in records] == ["ml.prod.new"]
        assert Path(existing.dirs[0]).exists()


class TestRemoveDownloads:
    def test_deletes_dirs_and_drops_records(self, tmp_path):
        base = tmp_path / "proj"
        keep = _install(base, "ml.prod.pii", "pii")
        drop = _install(base, "main.default.triage", "triage")
        _write_dirs(keep)
        _write_dirs(drop)
        skills_state.record_downloads([keep, drop])

        skills_state.remove_downloads(skills_state.records_for_schema("main.default"))

        assert [r["fqn"] for r in skills_state.list_downloaded()] == ["ml.prod.pii"]
        assert not Path(drop.dirs[0]).exists()
        assert Path(keep.dirs[0]).exists()

    def test_empty_is_a_noop(self):
        skills_state.remove_downloads([])
        assert skills_state.list_downloaded() == []
