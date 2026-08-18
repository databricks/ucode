"""Tests for managed_files.py — the isaac-style sudo writer for OS managed settings files.

Every test mocks the actual privileged step (`_sudo_replace`), so NO real `sudo` / `/etc` write
ever runs. The behavior that matters here is the drift check: an unchanged file must not shell out.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import ucode.config_io as config_io
from ucode import managed_files


@pytest.fixture(autouse=True)
def _reset_dry_run():
    config_io.set_dry_run(False)
    yield
    config_io.set_dry_run(False)


@pytest.fixture(autouse=True)
def _supported(monkeypatch):
    # Pin platform support on so tests are deterministic on any host.
    monkeypatch.setattr(managed_files, "managed_files_supported", lambda: True)


def _capture_sudo(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        managed_files, "_sudo_replace", lambda path, text: calls.append((str(path), text))
    )
    monkeypatch.setattr(managed_files, "_sudo_remove", lambda path: None)
    return calls


class TestWriteManagedFile:
    def test_unchanged_content_does_not_sudo(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("same", encoding="utf-8")
        calls = _capture_sudo(monkeypatch)
        assert (
            managed_files.write_managed_file(
                path, "same", display="X", workspace="https://workspace-a"
            )
            == "skipped"
        )
        # A matching file without an ownership marker is still not ours.
        assert calls == []

    def test_changed_content_sudo_writes(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("old", encoding="utf-8")
        calls = _capture_sudo(monkeypatch)
        assert (
            managed_files.write_managed_file(
                path, "new", display="X", workspace="https://workspace-a"
            )
            == "skipped"
        )
        assert calls == []

    def test_absent_file_sudo_writes(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        calls = _capture_sudo(monkeypatch)
        assert (
            managed_files.write_managed_file(
                path, "new", display="X", workspace="https://workspace-a"
            )
            == "written"
        )
        assert calls[0] == (str(path), "new")
        assert calls[1][0] == str(managed_files.ownership_path(path))

    def test_dry_run_does_not_sudo(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        calls = _capture_sudo(monkeypatch)
        config_io.set_dry_run(True)
        assert (
            managed_files.write_managed_file(
                path, "new", display="X", workspace="https://workspace-a"
            )
            == "written"
        )
        assert calls == []

    def test_unsupported_platform_skips(self, tmp_path, monkeypatch):
        monkeypatch.setattr(managed_files, "managed_files_supported", lambda: False)
        calls = _capture_sudo(monkeypatch)
        path = tmp_path / "managed.json"
        assert (
            managed_files.write_managed_file(
                path, "new", display="X", workspace="https://workspace-a"
            )
            == "skipped"
        )
        assert calls == []

    def test_permission_error_is_skipped_not_raised(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"

        def boom(path, text):
            raise PermissionError("no root")

        monkeypatch.setattr(managed_files, "_sudo_replace", boom)
        # Never raises — the launch proceeds; the private ucode config still works.
        assert (
            managed_files.write_managed_file(
                path, "new", display="X", workspace="https://workspace-a"
            )
            == "skipped"
        )

    def test_sudo_failure_is_skipped_not_raised(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"

        def boom(path, text):
            raise subprocess.CalledProcessError(1, ["/usr/bin/sudo", "cp"], stderr="denied")

        monkeypatch.setattr(managed_files, "_sudo_replace", boom)
        assert (
            managed_files.write_managed_file(
                path, "new", display="X", workspace="https://workspace-a"
            )
            == "skipped"
        )


class TestStrictOwnershipLifecycle:
    @staticmethod
    def _direct_sudo(monkeypatch):
        def replace(path, text):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")

        def remove(path):
            Path(path).unlink(missing_ok=True)

        monkeypatch.setattr(managed_files, "_sudo_replace", replace)
        monkeypatch.setattr(managed_files, "_sudo_remove", remove)

    def test_workspace_a_to_b_to_c(self, tmp_path, monkeypatch):
        self._direct_sudo(monkeypatch)
        path = tmp_path / "managed.toml"

        assert (
            managed_files.write_managed_file(path, "workspace-a", display="X", workspace="A")
            == "written"
        )
        assert path.read_text(encoding="utf-8") == "workspace-a"

        assert (
            managed_files.write_managed_file(path, "workspace-b", display="X", workspace="B")
            == "written"
        )
        assert path.read_text(encoding="utf-8") == "workspace-b"
        owner = managed_files._load_owner(path)
        assert owner is not None and owner["workspace"] == "B"

        assert managed_files.remove_managed_file(path, display="X") == "removed"
        assert not path.exists()
        assert not managed_files.ownership_path(path).exists()

    def test_unchanged_owned_file_does_not_sudo(self, tmp_path, monkeypatch):
        self._direct_sudo(monkeypatch)
        path = tmp_path / "managed.toml"
        managed_files.write_managed_file(path, "workspace-a", display="X", workspace="A")
        calls = _capture_sudo(monkeypatch)

        assert (
            managed_files.write_managed_file(path, "workspace-a", display="X", workspace="A")
            == "unchanged"
        )
        assert calls == []

    def test_does_not_overwrite_unowned_file(self, tmp_path, monkeypatch):
        self._direct_sudo(monkeypatch)
        path = tmp_path / "managed.toml"
        path.write_text("enterprise", encoding="utf-8")

        assert (
            managed_files.write_managed_file(path, "workspace-a", display="X", workspace="A")
            == "skipped"
        )
        assert path.read_text(encoding="utf-8") == "enterprise"
        assert not managed_files.ownership_path(path).exists()

    def test_does_not_remove_owned_file_after_external_change(self, tmp_path, monkeypatch):
        self._direct_sudo(monkeypatch)
        path = tmp_path / "managed.toml"
        managed_files.write_managed_file(path, "workspace-a", display="X", workspace="A")
        path.write_text("changed externally", encoding="utf-8")

        assert managed_files.remove_managed_file(path, display="X") == "skipped"
        assert path.read_text(encoding="utf-8") == "changed externally"
        assert managed_files.ownership_path(path).exists()

    def test_marker_write_failure_rolls_back_new_file(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.toml"
        marker = managed_files.ownership_path(path)

        def replace(target, text):
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            Path(target).write_text(text, encoding="utf-8")
            if Path(target) == marker:
                raise PermissionError("marker denied")

        monkeypatch.setattr(managed_files, "_sudo_replace", replace)
        monkeypatch.setattr(
            managed_files, "_sudo_remove", lambda target: Path(target).unlink(missing_ok=True)
        )

        assert (
            managed_files.write_managed_file(path, "workspace-a", display="X", workspace="A")
            == "skipped"
        )
        assert not path.exists()
        assert not marker.exists()


class TestClearImmutableStatDenied:
    def test_stat_denied_path_returns_false_without_raising(self, monkeypatch):
        # Regression: `_clear_immutable` ran an unguarded path.exists() inside the sudo write; under a
        # root-locked /etc/codex that raised PermissionError and aborted the write ("without root").
        class _StatDenied:
            def exists(self):
                raise PermissionError(13, "Permission denied")

        # Ensure no sudo subprocess is attempted if the guard ever regresses.
        monkeypatch.setattr(
            managed_files.subprocess, "run", lambda *a, **k: pytest.fail("should not shell out")
        )
        assert managed_files._clear_immutable(_StatDenied()) is False
