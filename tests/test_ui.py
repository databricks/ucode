"""Tests for ui.py — pure helpers that don't touch I/O or prompts."""

from __future__ import annotations

from datetime import timedelta

import pytest
import questionary

from ucode import ui as ui_mod
from ucode.ui import (
    format_duration,
    format_token_count,
    normalize_workspace_url,
    prompt_for_workspace,
    render_box_table,
    status_badge,
)


class TestNormalizeWorkspaceUrl:
    def test_adds_https_when_missing(self):
        assert normalize_workspace_url("example.databricks.com") == "https://example.databricks.com"

    def test_strips_trailing_slash(self):
        assert (
            normalize_workspace_url("https://example.databricks.com/")
            == "https://example.databricks.com"
        )

    def test_strips_multiple_trailing_slashes(self):
        assert (
            normalize_workspace_url("https://example.databricks.com///")
            == "https://example.databricks.com"
        )

    def test_preserves_https(self):
        assert (
            normalize_workspace_url("https://foo.azuredatabricks.net")
            == "https://foo.azuredatabricks.net"
        )

    def test_preserves_http(self):
        assert normalize_workspace_url("http://localhost:8080") == "http://localhost:8080"

    def test_strips_whitespace(self):
        assert (
            normalize_workspace_url("  https://example.databricks.com  ")
            == "https://example.databricks.com"
        )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_workspace_url("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_workspace_url("   ")


class TestFormatTokenCount:
    def test_small(self):
        assert format_token_count(0) == "0"
        assert format_token_count(999) == "999"

    def test_thousands(self):
        assert format_token_count(1000) == "1.0K"
        assert format_token_count(1500) == "1.5K"
        assert format_token_count(999_999) == "1000.0K"

    def test_millions(self):
        assert format_token_count(1_000_000) == "1.0M"
        assert format_token_count(2_500_000) == "2.5M"

    def test_billions(self):
        assert format_token_count(1_000_000_000) == "1.0B"
        assert format_token_count(2_200_000_000) == "2.2B"


class TestFormatDuration:
    def test_none_returns_dash(self):
        assert format_duration(None) == "-"

    def test_zero_returns_dash(self):
        assert format_duration(timedelta(seconds=0)) == "-"

    def test_negative_returns_dash(self):
        assert format_duration(timedelta(seconds=-5)) == "-"

    def test_minutes(self):
        assert format_duration(timedelta(minutes=5)) == "5m"
        assert format_duration(timedelta(minutes=59)) == "59m"

    def test_hours_fractional(self):
        result = format_duration(timedelta(hours=1, minutes=30))
        assert result == "1.5h"

    def test_hours_rounded(self):
        result = format_duration(timedelta(hours=10))
        assert result == "10h"

    def test_days(self):
        result = format_duration(timedelta(hours=48))
        assert result == "2.0d"


class TestStatusBadge:
    def test_ok_is_green(self):
        assert "green" in status_badge("OK", "ok")

    def test_warn_is_yellow(self):
        assert "yellow" in status_badge("Warning", "warn")

    def test_error_is_red(self):
        assert "red" in status_badge("Error", "error")

    def test_unknown_kind_uses_bold(self):
        result = status_badge("X", "unknown")
        assert "bold" in result
        assert "X" in result

    def test_text_is_included(self):
        assert "MyText" in status_badge("MyText", "ok")


class TestRenderBoxTable:
    def test_produces_box_chars(self):
        result = render_box_table(["A", "B"], [["x", "y"]])
        assert "┏" in result
        assert "┗" not in result  # bottom uses └
        assert "└" in result
        assert "A" in result
        assert "x" in result

    def test_empty_rows(self):
        result = render_box_table(["H1", "H2"], [])
        assert "H1" in result
        assert "H2" in result

    def test_cell_wraps_when_max_width_set(self):
        long_text = "a" * 30
        result = render_box_table(["Col"], [[long_text]], max_widths=[10])
        # wrapped lines mean the original 30-char string is broken up
        lines = result.splitlines()
        assert any(len(line.strip()) <= 14 for line in lines)

    def test_dash_for_empty_cell(self):
        result = render_box_table(["A"], [[""]])
        assert "-" in result


class _StubQuestion:
    def __init__(self, answer):
        self._answer = answer

    def ask(self):
        return self._answer


class TestPromptForWorkspace:
    """Capture the choices passed to ``questionary.select`` so we can assert on
    layout (header alignment + duplicate-host preservation) without driving
    real keyboard I/O."""

    def _capture_select(self, monkeypatch, answer):
        captured: dict = {}

        def fake_select(message, choices, **kwargs):
            captured["message"] = message
            captured["choices"] = choices
            captured["kwargs"] = kwargs
            return _StubQuestion(answer)

        monkeypatch.setattr(questionary, "select", fake_select)
        monkeypatch.setattr(ui_mod.questionary, "select", fake_select)
        return captured

    def test_shows_header_and_each_profile_row(self, monkeypatch):
        profiles = [
            ("https://a.cloud.databricks.com", "alpha"),
            ("https://b.cloud.databricks.com", "beta-profile-name"),
        ]
        captured = self._capture_select(monkeypatch, answer=profiles[0])
        url, profile = prompt_for_workspace("setup", profiles)

        assert (url, profile) == profiles[0]
        choices = captured["choices"]
        # Header (separator), 2 rows, "Enter a different URL" entry.
        assert len(choices) == 4
        assert isinstance(choices[0], questionary.Separator)
        header = choices[0].title
        assert "Profile Name" in header
        assert "Workspace URL" in header
        # Profile names ljust-padded to the longest name (17 chars).
        name_width = max(len(name) for _, name in profiles)
        assert "alpha".ljust(name_width) in choices[1].title
        assert profiles[0][0] in choices[1].title
        assert "beta-profile-name".ljust(name_width) in choices[2].title
        assert profiles[1][0] in choices[2].title
        # Final fallback entry still present.
        assert choices[3].title == "Enter a different URL"

    def test_keeps_duplicate_hosts_as_separate_rows(self, monkeypatch):
        profiles = [
            ("https://shared.cloud.databricks.com", "first"),
            ("https://shared.cloud.databricks.com", "second"),
        ]
        captured = self._capture_select(monkeypatch, answer=profiles[1])
        url, profile = prompt_for_workspace("setup", profiles)

        assert (url, profile) == profiles[1]
        # Both rows present — duplicates not collapsed.
        choices = captured["choices"]
        # Filter to choices whose value is a (host, profile) tuple — drops the
        # header separator and the trailing "Enter a different URL" entry.
        host_choices = [c for c in choices if isinstance(getattr(c, "value", None), tuple)]
        assert [c.value for c in host_choices] == profiles

    def test_returns_normalized_url_with_profile(self, monkeypatch):
        # Picker handed back a URL with a trailing slash — normalize_workspace_url
        # should strip it before returning.
        profiles = [("https://example.cloud.databricks.com/", "p")]
        self._capture_select(monkeypatch, answer=profiles[0])
        url, profile = prompt_for_workspace("setup", profiles)
        assert url == "https://example.cloud.databricks.com"
        assert profile == "p"
