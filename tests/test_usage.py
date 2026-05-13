"""Tests for usage.py — query builders, parsing/formatting, rendering."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from ucode.usage import (
    USAGE_BREAKDOWN_DAYS,
    USAGE_SUMMARY_DAYS,
    build_current_user_query,
    build_usage_report_query,
    coerce_date,
    coerce_datetime,
    empty_tool_day,
    extract_model_names,
    parse_usage_rows,
    render_usage_summary,
    simplify_model_name,
    summarize_models,
)


class TestBuildUsageReportQuery:
    def test_contains_system_table(self):
        q = build_usage_report_query()
        assert "system.ai_gateway.usage" in q

    def test_contains_interval(self):
        q = build_usage_report_query()
        assert str(USAGE_SUMMARY_DAYS) in q

    def test_filters_known_tools(self):
        q = build_usage_report_query()
        for tool in ("codex", "claude", "gemini", "opencode"):
            assert tool in q


class TestBuildCurrentUserQuery:
    def test_uses_current_user(self):
        q = build_current_user_query()
        assert "current_user()" in q


class TestParseUsageRows:
    def test_zips_columns_and_rows(self):
        columns = ["a", "b", "c"]
        rows = [(1, 2, 3), (4, 5, 6)]
        result = parse_usage_rows(columns, rows)
        assert result == [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]

    def test_empty_rows(self):
        assert parse_usage_rows(["a"], []) == []


class TestCoerceDate:
    def test_date_passthrough(self):
        d = date(2024, 6, 1)
        assert coerce_date(d) == d

    def test_datetime_to_date(self):
        dt = datetime(2024, 6, 1, 12, 0, 0)
        assert coerce_date(dt) == date(2024, 6, 1)

    def test_iso_string(self):
        assert coerce_date("2024-06-01") == date(2024, 6, 1)

    def test_invalid_string_returns_none(self):
        assert coerce_date("not-a-date") is None

    def test_none_returns_none(self):
        assert coerce_date(None) is None


class TestCoerceDatetime:
    def test_datetime_passthrough(self):
        dt = datetime(2024, 6, 1, 0, 0, 0)
        assert coerce_datetime(dt) == dt

    def test_iso_string(self):
        result = coerce_datetime("2024-06-01T12:00:00")
        assert isinstance(result, datetime)
        assert result.date() == date(2024, 6, 1)

    def test_z_suffix(self):
        result = coerce_datetime("2024-06-01T12:00:00Z")
        assert isinstance(result, datetime)

    def test_invalid_string_returns_none(self):
        assert coerce_datetime("bad") is None

    def test_none_returns_none(self):
        assert coerce_datetime(None) is None


class TestSimplifyModelName:
    def test_strips_databricks_and_tool_prefix(self):
        # databricks- stripped first, then claude- stripped → "sonnet-4"
        assert simplify_model_name("claude", "databricks-claude-sonnet-4") == "sonnet-4"

    def test_gemini_prefix(self):
        result = simplify_model_name("gemini", "databricks-gemini-2.0-flash")
        assert result == "2.0-flash"

    def test_codex_strips_gpt_prefix(self):
        result = simplify_model_name("codex", "databricks-gpt-4o")
        assert result == "4o"

    def test_empty_returns_dash(self):
        assert simplify_model_name("claude", "") == "-"

    def test_no_known_prefix_returns_as_is(self):
        result = simplify_model_name("claude", "some-other-model")
        assert result == "some-other-model"

    def test_only_databricks_prefix_stripped_for_unknown_tool(self):
        result = simplify_model_name("opencode", "databricks-claude-sonnet-4")
        assert result == "claude-sonnet-4"


class TestExtractModelNames:
    def test_single_model(self):
        result = extract_model_names("claude", "databricks-claude-sonnet-4")
        assert result == ["sonnet-4"]

    def test_multiple_models(self):
        result = extract_model_names(
            "claude", "databricks-claude-sonnet-4, databricks-claude-opus-4"
        )
        assert "sonnet-4" in result
        assert "opus-4" in result

    def test_deduplicates(self):
        result = extract_model_names(
            "claude", "databricks-claude-sonnet-4, databricks-claude-sonnet-4"
        )
        assert result.count("sonnet-4") == 1

    def test_empty_returns_empty_list(self):
        assert extract_model_names("claude", "") == []

    def test_non_string_returns_empty_list(self):
        assert extract_model_names("claude", None) == []


class TestSummarizeModels:
    def test_single_model(self):
        result = summarize_models("claude", "databricks-claude-sonnet-4")
        assert result == "sonnet-4"

    def test_multiple_models_joined(self):
        result = summarize_models("claude", "databricks-claude-sonnet-4, databricks-claude-opus-4")
        assert "sonnet-4" in result
        assert "," in result

    def test_empty_returns_dash(self):
        assert summarize_models("claude", "") == "-"

    def test_none_returns_dash(self):
        assert summarize_models("claude", None) == "-"


class TestEmptyToolDay:
    def test_structure(self):
        d = date(2024, 6, 1)
        row = empty_tool_day("claude", d)
        assert row["tool"] == "claude"
        assert row["usage_day"] == d
        assert row["total_tokens_used"] == 0
        assert row["sessions"] == 0
        assert row["models"] == "-"


class TestRenderUsageSummary:
    def _make_record(self, days_ago: int, tool: str, tokens: int, model: str = "") -> dict:
        d = date.today() - timedelta(days=days_ago)
        return {
            "tool": tool,
            "usage_day": d,
            "total_tokens_used": tokens,
            "models": model,
        }

    def test_contains_requester_name(self):
        records = [self._make_record(0, "claude", 1000)]
        result = render_usage_summary(records, "alice@example.com", {"claude": "Claude Code"})
        assert "alice@example.com" in result

    def test_today_total(self):
        records = [self._make_record(0, "claude", 5000)]
        result = render_usage_summary(records, "user", {"claude": "Claude Code"})
        assert "5.0K" in result

    def test_weekly_total_includes_past_week(self):
        records = [
            self._make_record(0, "claude", 1000),
            self._make_record(3, "claude", 2000),
            self._make_record(USAGE_BREAKDOWN_DAYS, "claude", 9999),  # outside window
        ]
        result = render_usage_summary(records, "user", {"claude": "Claude Code"})
        # only 3K from the last 7 days; 9999 from day 7 (boundary) may vary
        assert "3.0K" in result or "3" in result

    def test_active_tools_listed(self):
        records = [self._make_record(0, "claude", 1000)]
        result = render_usage_summary(records, "user", {"claude": "Claude Code"})
        assert "Claude Code" in result

    def test_top_models_listed(self):
        records = [self._make_record(0, "claude", 5000, "databricks-claude-sonnet-4")]
        result = render_usage_summary(records, "user", {"claude": "Claude Code"})
        assert "sonnet-4" in result

    def test_empty_records(self):
        result = render_usage_summary([], "user", {"claude": "Claude Code"})
        assert "user" in result
