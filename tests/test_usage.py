"""Tests for usage.py — query builders, parsing/formatting, rendering."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import ucode.usage as usage_mod
from ucode.usage import (
    USAGE_BREAKDOWN_DAYS,
    USAGE_SUMMARY_DAYS,
    build_current_user_query,
    build_tool_breakdown_rows,
    build_usage_report_query,
    coerce_date,
    coerce_datetime,
    configured_usage_tools,
    empty_tool_day,
    estimate_cost_usd,
    extract_model_names,
    extract_model_token_breakdown,
    filter_records_for_tools,
    format_local_budget_status,
    has_tool_usage_last_week,
    local_budget_status,
    local_daily_agent_budget_usd,
    local_price_multiplier,
    parse_usage_rows,
    query_local_budget_totals,
    query_local_usage_summary,
    query_local_usage_totals,
    record_local_usage_delta,
    record_local_usage_snapshot,
    render_local_budget_panel,
    render_local_usage_summary,
    render_usage_summary,
    simplify_model_name,
    summarize_model_tokens,
    summarize_models,
    usage,
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

    def test_includes_per_model_token_rollup(self):
        q = build_usage_report_query()
        assert "model_tokens" in q
        assert "SUM(total_tokens_used) AS model_tokens_used" in q
        assert "NAMED_STRUCT('model', destination_model, 'tokens', model_tokens_used)" in q


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


class TestConfiguredUsageTools:
    def test_uses_available_tools_in_display_order(self):
        tool_displays = {"claude": "Claude Code", "codex": "Codex", "gemini": "Gemini"}
        state = {"available_tools": ["codex", "claude"]}
        assert configured_usage_tools(state, tool_displays) == ["claude", "codex"]

    def test_falls_back_to_managed_configs(self):
        tool_displays = {"claude": "Claude Code", "codex": "Codex"}
        state = {"managed_configs": {"codex": {"keys": []}}}
        assert configured_usage_tools(state, tool_displays) == ["codex"]

    def test_ignores_unknown_tools(self):
        tool_displays = {"claude": "Claude Code"}
        state = {"available_tools": ["claude", "unknown"]}
        assert configured_usage_tools(state, tool_displays) == ["claude"]


class TestFilterRecordsForTools:
    def test_keeps_only_configured_tools(self):
        records = [
            {"tool": "claude", "total_tokens_used": 100},
            {"tool": "gemini", "total_tokens_used": 200},
            {"tool": "codex", "total_tokens_used": 300},
        ]
        assert filter_records_for_tools(records, ["claude", "codex"]) == [
            {"tool": "claude", "total_tokens_used": 100},
            {"tool": "codex", "total_tokens_used": 300},
        ]


class TestHasToolUsageLastWeek:
    def test_true_for_recent_tokens(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today(),
                "total_tokens_used": 100,
                "sessions": 1,
            }
        ]
        assert has_tool_usage_last_week(records, "claude") is True

    def test_true_for_recent_session_even_without_tokens(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today(),
                "total_tokens_used": 0,
                "sessions": 1,
            }
        ]
        assert has_tool_usage_last_week(records, "claude") is True

    def test_false_for_only_old_usage(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today() - timedelta(days=USAGE_BREAKDOWN_DAYS),
                "total_tokens_used": 100,
                "sessions": 1,
            }
        ]
        assert has_tool_usage_last_week(records, "claude") is False

    def test_false_for_other_tool_usage(self):
        records = [
            {
                "tool": "codex",
                "usage_day": date.today(),
                "total_tokens_used": 100,
                "sessions": 1,
            }
        ]
        assert has_tool_usage_last_week(records, "claude") is False


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


class TestModelTokenBreakdown:
    def test_extracts_json_model_tokens(self):
        raw = (
            '[{"model":"databricks-claude-opus-4", "tokens":236000}, '
            '{"model":"databricks-claude-haiku-4.5", "tokens":920}]'
        )
        result = extract_model_token_breakdown("claude", raw)
        assert result == [("opus-4", 236000), ("haiku-4.5", 920)]

    def test_merges_simplified_duplicate_model_names(self):
        raw = [
            {"model": "databricks-claude-opus-4", "tokens": 100},
            {"model": "claude-opus-4", "tokens": 50},
        ]
        result = extract_model_token_breakdown("claude", raw)
        assert result == [("opus-4", 150)]

    def test_single_model_legacy_fallback_uses_total_tokens(self):
        result = extract_model_token_breakdown(
            "codex",
            None,
            "databricks-gpt-5",
            13300,
        )
        assert result == [("5", 13300)]

    def test_multi_model_legacy_fallback_does_not_assign_total_to_each_model(self):
        result = extract_model_token_breakdown(
            "claude",
            None,
            "databricks-claude-haiku-4.5, databricks-claude-opus-4",
            237000,
        )
        assert result == [("haiku-4.5", 0), ("opus-4", 0)]

    def test_summarizes_tokens_next_to_each_model(self):
        raw = '[{"model":"databricks-claude-opus-4", "tokens":236000}]'
        result = summarize_model_tokens("claude", raw, "", 0)
        assert result == "opus-4 (236.0K)"


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

    def test_top_models_uses_per_model_token_totals(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today(),
                "total_tokens_used": 237000,
                "models": "databricks-claude-haiku-4.5, databricks-claude-opus-4",
                "model_tokens": (
                    '[{"model":"databricks-claude-haiku-4.5", "tokens":920}, '
                    '{"model":"databricks-claude-opus-4", "tokens":236080}]'
                ),
            },
            {
                "tool": "codex",
                "usage_day": date.today(),
                "total_tokens_used": 13300,
                "models": "databricks-gpt-5",
                "model_tokens": '[{"model":"databricks-gpt-5", "tokens":13300}]',
            },
        ]
        result = render_usage_summary(
            records,
            "user",
            {"claude": "Claude Code", "codex": "Codex"},
        )
        assert "opus-4 (236.1K)" in result
        assert "5 (13.3K)" in result
        assert "haiku-4.5 (920)" in result
        assert "haiku-4.5 (237.0K)" not in result

    def test_daily_table_shows_per_model_token_totals(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today(),
                "total_tokens_used": 237000,
                "sessions": 2,
                "models": "databricks-claude-haiku-4.5, databricks-claude-opus-4",
                "model_tokens": (
                    '[{"model":"databricks-claude-haiku-4.5", "tokens":920}, '
                    '{"model":"databricks-claude-opus-4", "tokens":236080}]'
                ),
            }
        ]
        rows = build_tool_breakdown_rows(records, "claude")
        assert rows[0][5] == "opus-4 (236.1K), haiku-4.5 (920)"

    def test_empty_records(self):
        result = render_usage_summary([], "user", {"claude": "Claude Code"})
        assert "user" in result


class TestLocalUsageLedger:
    def test_records_delta_events_and_aggregates_across_sessions(self, tmp_path):
        db_path = tmp_path / "usage.sqlite"

        record_local_usage_delta(
            session_id="s1",
            tool="claude",
            model="databricks-claude-sonnet-4",
            input_tokens=1_000,
            output_tokens=200,
            db_path=db_path,
        )
        record_local_usage_delta(
            session_id="s2",
            tool="codex",
            model="databricks-gpt-5",
            input_tokens=500,
            output_tokens=100,
            db_path=db_path,
        )

        totals = query_local_usage_totals(db_path=db_path)
        assert totals["sessions"] == 2
        assert totals["total_tokens"] == 1_800
        assert totals["cost_usd"] > 0

        rows = query_local_usage_summary(db_path=db_path)
        assert {row["tool"] for row in rows} == {"claude", "codex"}

    def test_snapshot_records_only_new_tokens(self, tmp_path):
        db_path = tmp_path / "usage.sqlite"

        first = record_local_usage_snapshot(
            session_id="s1",
            tool="claude",
            model="databricks-claude-sonnet-4",
            input_tokens=1_000,
            output_tokens=100,
            db_path=db_path,
        )
        second = record_local_usage_snapshot(
            session_id="s1",
            tool="claude",
            model="databricks-claude-sonnet-4",
            input_tokens=1_500,
            output_tokens=150,
            db_path=db_path,
        )
        third = record_local_usage_snapshot(
            session_id="s1",
            tool="claude",
            model="databricks-claude-sonnet-4",
            input_tokens=1_500,
            output_tokens=150,
            db_path=db_path,
        )

        assert first is not None
        assert first["total_tokens"] == 1_100
        assert second is not None
        assert second["input_tokens"] == 500
        assert second["output_tokens"] == 50
        assert second["total_tokens"] == 550
        assert third is None

        totals = query_local_usage_totals(db_path=db_path)
        assert totals["total_tokens"] == 1_650

    def test_snapshot_never_records_negative_delta(self, tmp_path):
        db_path = tmp_path / "usage.sqlite"
        record_local_usage_snapshot(
            session_id="s1",
            tool="claude",
            model="databricks-claude-sonnet-4",
            total_tokens=1_000,
            db_path=db_path,
        )

        event = record_local_usage_snapshot(
            session_id="s1",
            tool="claude",
            model="databricks-claude-sonnet-4",
            total_tokens=900,
            db_path=db_path,
        )

        assert event is None
        totals = query_local_usage_totals(db_path=db_path)
        assert totals["total_tokens"] == 1_000

    def test_unknown_model_tracks_tokens_with_zero_cost(self, tmp_path):
        db_path = tmp_path / "usage.sqlite"
        record_local_usage_delta(
            session_id="s1",
            tool="claude",
            model="unknown-model",
            total_tokens=42,
            db_path=db_path,
        )

        totals = query_local_usage_totals(db_path=db_path)
        assert totals["total_tokens"] == 42
        assert totals["cost_usd"] == 0

    def test_estimates_cost_from_input_and_output_rates(self):
        assert estimate_cost_usd("databricks-gpt-5", 1_000_000, 1_000_000) == 11.25

    def test_estimates_total_only_cost_with_input_rate(self):
        assert estimate_cost_usd("databricks-gpt-5", 0, 0, total_tokens=1_000_000) == 1.25

    def test_estimates_gpt_version_rates(self):
        assert (
            estimate_cost_usd(
                "gpt-5.5",
                1_000_000,
                1_000_000,
                cache_read_input_tokens=1_000_000,
            )
            == 35.5
        )
        assert (
            estimate_cost_usd(
                "databricks-gpt-5-4",
                1_000_000,
                1_000_000,
                cache_read_input_tokens=1_000_000,
            )
            == 17.75
        )
        assert (
            estimate_cost_usd(
                "gpt-5.4-mini",
                1_000_000,
                1_000_000,
                cache_read_input_tokens=1_000_000,
            )
            == 5.325
        )
        assert (
            estimate_cost_usd(
                "databricks-gpt-5-4-nano",
                1_000_000,
                1_000_000,
                cache_read_input_tokens=1_000_000,
            )
            == 1.47
        )
        assert (
            estimate_cost_usd(
                "gpt-5.2",
                1_000_000,
                1_000_000,
                cache_read_input_tokens=1_000_000,
            )
            == 15.925
        )

    def test_estimates_gpt_pro_without_cached_input_discount(self):
        assert (
            estimate_cost_usd(
                "databricks-gpt-5-5-pro",
                1_000_000,
                1_000_000,
                cache_read_input_tokens=1_000_000,
            )
            == 210.0
        )

    def test_estimates_new_claude_opus_variant(self):
        assert estimate_cost_usd("databricks-claude-opus-4-8", 1_000_000, 1_000_000) == 30.0

    def test_estimates_one_m_suffix(self):
        assert estimate_cost_usd("databricks-claude-opus-4-8[1m]", 1_000_000, 1_000_000) == 30.0

    def test_estimates_claude_cache_rates(self):
        assert (
            estimate_cost_usd(
                "claude-opus-4-8-latest",
                1_000_000,
                1_000_000,
                cache_read_input_tokens=1_000_000,
                cache_creation_input_tokens=1_000_000,
            )
            == 36.75
        )
        assert (
            estimate_cost_usd(
                "claude-sonnet-4-6",
                1_000_000,
                1_000_000,
                cache_read_input_tokens=1_000_000,
                cache_creation_input_tokens=1_000_000,
            )
            == 22.05
        )
        assert (
            estimate_cost_usd(
                "claude-haiku-4-5-20260101",
                1_000_000,
                1_000_000,
                cache_read_input_tokens=1_000_000,
                cache_creation_input_tokens=1_000_000,
            )
            == 7.35
        )

    def test_estimates_claude_opus_4_1_legacy_rate(self):
        assert (
            estimate_cost_usd(
                "databricks-claude-opus-4-1-20250805",
                1_000_000,
                1_000_000,
                cache_read_input_tokens=1_000_000,
                cache_creation_input_tokens=1_000_000,
            )
            == 110.25
        )

    def test_price_multiplier_env_inflates_cost(self, monkeypatch):
        monkeypatch.setenv("UCODE_USAGE_PRICE_MULTIPLIER", "10")

        assert estimate_cost_usd("databricks-gpt-5", 0, 0, total_tokens=1_000_000) == 12.5
        assert local_price_multiplier() == 10

    def test_invalid_price_multiplier_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("UCODE_USAGE_PRICE_MULTIPLIER", "not-a-number")

        assert local_price_multiplier() == 1.0

    def test_renders_local_summary(self, tmp_path):
        db_path = tmp_path / "usage.sqlite"
        record_local_usage_delta(
            session_id="s1",
            tool="codex",
            model="databricks-gpt-5",
            total_tokens=100,
            db_path=db_path,
        )
        result = render_local_usage_summary(
            query_local_usage_summary(db_path=db_path),
            query_local_usage_totals(db_path=db_path),
        )

        assert "Local Usage Summary" in result
        assert "100 tokens" in result
        assert "databricks-gpt-5" in result

    def test_reports_policy_budget_warning_status(self, tmp_path, monkeypatch):
        db_path = tmp_path / "usage.sqlite"
        monkeypatch.setattr(
            usage_mod,
            "_workspace_policy",
            lambda: {
                "policy": {
                    "daily_budget_usd": 20,
                    "tiers": [
                        {
                            "name": "premium",
                            "activates_at_pct": 0,
                            "harness": "codex",
                            "model": "gpt-5",
                        }
                    ],
                    "on_budget_exhausted": "block",
                }
            },
        )
        record_local_usage_delta(
            session_id="s1",
            tool="codex",
            model="databricks-gpt-5",
            total_tokens=13_000_000,
            db_path=db_path,
        )

        status = local_budget_status("codex", db_path)
        assert status["state"] == "warn"
        assert status["spend_usd"] == 16.25
        assert status["limit_usd"] == 20.0
        message = format_local_budget_status(status)
        assert "⚠️ [UCODE USAGE BUDGET] Codex is nearing" in message
        assert "Budget: $16.25 / $20.00 used today ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱ 81%." in message
        assert "Codex tokens used today" in message
        flattened = message.replace("\n", "")
        assert "daily budget. Budget" in flattened
        assert "Remaining: $3.75. Window" in flattened
        assert "day(s). Codex tokens" in flattened

    def test_budget_env_override(self, tmp_path, monkeypatch):
        db_path = tmp_path / "usage.sqlite"
        monkeypatch.setenv("UCODE_USAGE_DAILY_BUDGET_USD", "0.05")
        record_local_usage_delta(
            session_id="s1",
            tool="claude",
            model="databricks-claude-sonnet-4",
            total_tokens=20_000,
            db_path=db_path,
        )

        status = local_budget_status("claude", db_path)
        assert local_daily_agent_budget_usd("claude") == 0.05
        assert status["limit_usd"] == 0.05
        assert status["state"] == "exceeded"

    def test_budget_falls_back_to_default_without_policy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(usage_mod, "_workspace_policy", lambda: None)
        assert local_daily_agent_budget_usd("codex") == 500.0

    def test_budget_aggregates_spend_across_all_tools(self, tmp_path, monkeypatch):
        # The daily limit is a single pool: claude + codex spend combine and the
        # status is identical regardless of which tool label is requested.
        db_path = tmp_path / "usage.sqlite"
        monkeypatch.setattr(
            usage_mod,
            "_workspace_policy",
            lambda: {
                "policy": {
                    "daily_budget_usd": 20,
                    "tiers": [
                        {
                            "name": "premium",
                            "activates_at_pct": 0,
                            "harness": "codex",
                            "model": "gpt-5",
                        }
                    ],
                    "on_budget_exhausted": "block",
                }
            },
        )
        record_local_usage_delta(
            session_id="s1",
            tool="codex",
            model="databricks-gpt-5",
            total_tokens=10_000_000,
            db_path=db_path,
        )
        record_local_usage_delta(
            session_id="s2",
            tool="claude",
            model="databricks-gpt-5",
            total_tokens=8_000_000,
            db_path=db_path,
        )

        codex_status = local_budget_status("codex", db_path)
        claude_status = local_budget_status("claude", db_path)
        total_status = local_budget_status(db_path=db_path)
        # 18M tokens * $1.25/1M = $22.50 across both tools, over the $20 cap.
        assert (
            codex_status["spend_usd"]
            == claude_status["spend_usd"]
            == total_status["spend_usd"]
            == 22.5
        )
        assert (
            codex_status["state"] == claude_status["state"] == total_status["state"] == "exceeded"
        )
        assert codex_status["total_tokens"] == 10_000_000
        assert codex_status["sessions"] == 1
        assert claude_status["total_tokens"] == 8_000_000
        assert claude_status["sessions"] == 1
        assert total_status["total_tokens"] == 18_000_000
        assert total_status["sessions"] == 2

    def test_budget_status_exceeded(self, tmp_path, monkeypatch):
        db_path = tmp_path / "usage.sqlite"
        monkeypatch.setattr(
            usage_mod,
            "_workspace_policy",
            lambda: {
                "policy": {
                    "daily_budget_usd": 20,
                    "tiers": [
                        {
                            "name": "premium",
                            "activates_at_pct": 0,
                            "harness": "codex",
                            "model": "gpt-5",
                        }
                    ],
                    "on_budget_exhausted": "block",
                }
            },
        )
        record_local_usage_delta(
            session_id="s1",
            tool="codex",
            model="databricks-gpt-5",
            total_tokens=16_000_000,
            db_path=db_path,
        )

        status = local_budget_status("codex", db_path)
        assert status["state"] == "exceeded"
        message = format_local_budget_status(status)
        assert "Budget: $20.00 / $20.00 used today ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰ 100%." in message
        assert "Codex tokens used today: 16.0M.\nFurther tool use" in message

    def test_budget_totals_recompute_zero_cost_rows(self, tmp_path, monkeypatch):
        db_path = tmp_path / "usage.sqlite"
        monkeypatch.setenv("UCODE_USAGE_PRICE_MULTIPLIER", "1000")
        record_local_usage_delta(
            session_id="s1",
            tool="claude",
            model="unknown-model",
            input_tokens=1_000,
            output_tokens=100,
            db_path=db_path,
        )
        with usage_mod._connect_local_usage_db(db_path) as conn:
            conn.execute(
                "UPDATE usage_events SET model = ?, cost_usd = 0",
                ("databricks-claude-opus-4-8",),
            )

        totals = query_local_budget_totals(tool="claude", db_path=db_path)
        assert totals["cost_usd"] == 7.5


class TestRenderLocalBudgetPanel:
    def _render(self, panel) -> str:
        from rich.console import Console

        console = Console(width=80, no_color=True)
        with console.capture() as capture:
            console.print(panel)
        return capture.get()

    def _status(self, state: str, spend: float, limit: float) -> dict:
        return {
            "configured": True,
            "state": state,
            "tool": "codex",
            "limit_usd": limit,
            "spend_usd": spend,
            "remaining_usd": max(limit - spend, 0.0),
            "days": 1,
            "total_tokens": 1_200_000,
            "sessions": 8,
        }

    def test_ok_state_has_no_callout(self):
        text = self._render(render_local_budget_panel(self._status("ok", 120.0, 500.0)))
        assert "Codex · Daily Budget" in text
        assert "$120.00 / $500.00" in text
        assert "24% used" in text
        assert "Remaining" in text and "$380.00" in text
        assert "Codex Tokens" in text
        assert "Codex Sessions" in text
        assert "exceeded" not in text
        assert "nearing" not in text

    def test_global_panel_uses_total_labels(self):
        text = self._render(
            render_local_budget_panel(
                {
                    "configured": True,
                    "state": "ok",
                    "tool": None,
                    "limit_usd": 500.0,
                    "spend_usd": 120.0,
                    "remaining_usd": 380.0,
                    "days": 1,
                    "total_tokens": 1_200_000,
                    "sessions": 8,
                }
            )
        )
        assert "Total Tokens" in text
        assert "Total Sessions" in text

    def test_warn_state_shows_nearing_callout(self):
        text = self._render(render_local_budget_panel(self._status("warn", 45.0, 50.0)))
        assert "nearing its daily budget" in text
        assert "90% used" in text

    def test_exceeded_state_shows_blocked_callout(self):
        text = self._render(render_local_budget_panel(self._status("exceeded", 60.0, 50.0)))
        assert "budget exceeded" in text
        # Over-limit spend caps the percentage display at the bar, not the number.
        assert "120% used" in text

    def test_bar_fill_scales_with_percent(self):
        assert usage_mod._budget_bar_markup(0, "green").count("█") == 0
        assert usage_mod._budget_bar_markup(100, "red").count("█") == 28
        # Over 100% stays clamped to a full bar.
        assert usage_mod._budget_bar_markup(450, "red").count("█") == 28
        half = usage_mod._budget_bar_markup(50, "green")
        assert 0 < half.count("█") < 28


class TestUsageCommand:
    def test_filters_to_configured_agents_and_skips_inactive_tables(self, monkeypatch):
        today = date.today()
        old_day = today - timedelta(days=USAGE_BREAKDOWN_DAYS)
        columns = [
            "requester_name",
            "tool",
            "usage_day",
            "total_tokens_used",
            "sessions",
            "first_event_time",
            "last_event_time",
            "models",
            "model_tokens",
        ]
        rows = [
            (
                "user@example.com",
                "codex",
                today,
                100,
                1,
                None,
                None,
                "databricks-gpt-5",
                '[{"model":"databricks-gpt-5", "tokens":100}]',
            ),
            (
                "user@example.com",
                "claude",
                old_day,
                200,
                1,
                None,
                None,
                "databricks-claude-opus-4",
                '[{"model":"databricks-claude-opus-4", "tokens":200}]',
            ),
            (
                "user@example.com",
                "gemini",
                today,
                900,
                1,
                None,
                None,
                "databricks-gemini-2.0-flash",
                '[{"model":"databricks-gemini-2.0-flash", "tokens":900}]',
            ),
        ]

        printed: list[str] = []
        headings: list[str] = []
        notes: list[str] = []
        rendered_tables: list[list[list[str]]] = []

        class DummyConsole:
            def print(self, value):
                printed.append(str(value))

        def fake_render_box_table(headers, table_rows, max_widths=None):
            rendered_tables.append(table_rows)
            return "TABLE"

        monkeypatch.setattr(
            usage_mod,
            "load_state",
            lambda: {"workspace": "https://workspace", "available_tools": ["claude", "codex"]},
        )
        monkeypatch.setattr(usage_mod, "ensure_databricks_auth", lambda *args, **kwargs: None)
        monkeypatch.setattr(usage_mod, "get_databricks_token", lambda *args, **kwargs: "token")
        monkeypatch.setattr(
            usage_mod,
            "discover_sql_warehouse_http_path",
            lambda *args, **kwargs: "/sql/1.0/warehouses/abc",
        )
        monkeypatch.setattr(usage_mod, "run_usage_query", lambda *args, **kwargs: (columns, rows))
        monkeypatch.setattr(usage_mod, "console", DummyConsole())
        monkeypatch.setattr(usage_mod, "print_heading", headings.append)
        monkeypatch.setattr(usage_mod, "print_note", notes.append)
        monkeypatch.setattr(usage_mod, "render_box_table", fake_render_box_table)

        assert usage() == 0

        assert "Codex · Last 7 Days" in headings
        assert "Claude Code · Last 7 Days" in headings
        assert all("Gemini" not in heading for heading in headings)
        assert notes == [f"No usage for Claude Code in the last {USAGE_BREAKDOWN_DAYS} days."]
        assert len(rendered_tables) == 1
        assert rendered_tables[0][0][2] == "100"
        assert "gemini" not in "\n".join(printed).lower()
        assert "900" not in "\n".join(printed)
