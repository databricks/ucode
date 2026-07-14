"""Tests for ucode.update_check."""

from __future__ import annotations

import json
from unittest.mock import patch

from ucode import update_check


def _configure(tmp_path, monkeypatch, *, local, remote, cache=None):
    """Point the module at a temp cache and stub the local/remote commits.

    Returns the cache path so tests can assert on what was persisted.
    """
    cache_path = tmp_path / "update-check.json"
    if cache is not None:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    monkeypatch.setattr(update_check, "CACHE_PATH", cache_path)
    monkeypatch.setattr(update_check, "ucode_commit", lambda: local)
    monkeypatch.setattr(update_check, "_fetch_remote_head", lambda: remote)
    return cache_path


class TestMaybeWarnIfOutdated:
    def test_warns_when_behind(self, tmp_path, monkeypatch):
        _configure(tmp_path, monkeypatch, local="aaaaaaa", remote="bbbbbbb")
        with patch.object(update_check, "print_warning") as warn:
            update_check.maybe_warn_if_outdated()
        warn.assert_called_once()
        assert "ucode upgrade" in warn.call_args.args[0]

    def test_silent_when_up_to_date(self, tmp_path, monkeypatch):
        _configure(tmp_path, monkeypatch, local="aaaaaaa", remote="aaaaaaa")
        with patch.object(update_check, "print_warning") as warn:
            update_check.maybe_warn_if_outdated()
        warn.assert_not_called()

    def test_silent_for_editable(self, tmp_path, monkeypatch):
        # Editable checkout has no meaningful remote comparison — never fetch/warn.
        _configure(tmp_path, monkeypatch, local="editable", remote="bbbbbbb")
        with (
            patch.object(update_check, "print_warning") as warn,
            patch.object(update_check, "_fetch_remote_head") as fetch,
        ):
            update_check.maybe_warn_if_outdated()
        warn.assert_not_called()
        fetch.assert_not_called()

    def test_silent_on_network_failure(self, tmp_path, monkeypatch):
        _configure(tmp_path, monkeypatch, local="aaaaaaa", remote=None)
        with patch.object(update_check, "print_warning") as warn:
            update_check.maybe_warn_if_outdated()
        warn.assert_not_called()

    def test_writes_cache_after_fetch(self, tmp_path, monkeypatch):
        cache_path = _configure(tmp_path, monkeypatch, local="aaaaaaa", remote="bbbbbbb")
        update_check.maybe_warn_if_outdated()
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        assert saved["remote_head"] == "bbbbbbb"
        assert "checked_at" in saved

    def test_uses_cache_within_ttl(self, tmp_path, monkeypatch):
        # A fresh cache entry means no network call this run.
        import time

        cache = {"remote_head": "bbbbbbb", "checked_at": time.time()}
        _configure(tmp_path, monkeypatch, local="aaaaaaa", remote="ccccccc", cache=cache)
        with (
            patch.object(update_check, "_fetch_remote_head") as fetch,
            patch.object(update_check, "print_warning") as warn,
        ):
            update_check.maybe_warn_if_outdated()
        fetch.assert_not_called()
        # Warns against the cached remote, not a fresh fetch.
        warn.assert_called_once()

    def test_refetches_after_ttl(self, tmp_path, monkeypatch):
        stale = {"remote_head": "bbbbbbb", "checked_at": 0}
        _configure(tmp_path, monkeypatch, local="aaaaaaa", remote="ccccccc", cache=stale)
        with patch.object(update_check, "_fetch_remote_head", return_value="ccccccc") as fetch:
            update_check.maybe_warn_if_outdated()
        fetch.assert_called_once()
