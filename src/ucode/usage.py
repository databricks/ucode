"""Usage report querying & rendering.

Reads from `system.ai_gateway.usage` via a Databricks SQL warehouse.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import cast

from ucode.databricks import (
    discover_sql_warehouse_http_path,
    ensure_databricks_auth,
    get_databricks_token,
    run_usage_query,
)
from ucode.state import load_state
from ucode.ui import (
    console,
    format_duration,
    format_token_count,
    heading,
    label,
    print_heading,
    render_box_table,
    spinner,
    value,
)

USAGE_BREAKDOWN_DAYS = 7
USAGE_SUMMARY_DAYS = 30


def build_usage_report_query() -> str:
    return f"""
SELECT
  current_user() AS requester_name,
  CASE
    WHEN lower(user_agent) LIKE '%codex%' THEN 'codex'
    WHEN lower(user_agent) LIKE '%claude%' THEN 'claude'
    WHEN lower(user_agent) LIKE '%gemini%' THEN 'gemini'
    WHEN lower(user_agent) LIKE '%opencode%' THEN 'opencode'
    ELSE 'other'
  END AS tool,
  date(event_time) AS usage_day,
  SUM(COALESCE(total_tokens, 0)) AS total_tokens_used,
  COUNT(DISTINCT request_id) AS sessions,
  MIN(event_time) AS first_event_time,
  MAX(event_time) AS last_event_time,
  CONCAT_WS(', ', SORT_ARRAY(COLLECT_SET(destination_model))) AS models
FROM system.ai_gateway.usage
WHERE event_time >= current_timestamp() - interval {USAGE_SUMMARY_DAYS} days
  AND requester = current_user()
  AND (
    lower(user_agent) LIKE '%codex%'
    OR lower(user_agent) LIKE '%claude%'
    OR lower(user_agent) LIKE '%gemini%'
    OR lower(user_agent) LIKE '%opencode%'
  )
GROUP BY 1, 2, 3
ORDER BY usage_day DESC, tool ASC
""".strip()


def build_current_user_query() -> str:
    return "SELECT current_user() AS requester_name"


def parse_usage_rows(columns: list[str], rows: list[tuple]) -> list[dict[str, object]]:
    return [dict(zip(columns, row, strict=False)) for row in rows]


def coerce_date(value_obj: object) -> date | None:
    if isinstance(value_obj, date) and not isinstance(value_obj, datetime):
        return value_obj
    if isinstance(value_obj, datetime):
        return value_obj.date()
    if isinstance(value_obj, str):
        try:
            return datetime.fromisoformat(value_obj).date()
        except ValueError:
            return None
    return None


def coerce_datetime(value_obj: object) -> datetime | None:
    if isinstance(value_obj, datetime):
        return value_obj
    if isinstance(value_obj, str):
        candidate = value_obj.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            return None
    return None


def simplify_model_name(tool: str, model_name: str) -> str:
    normalized = (model_name or "").strip()
    if not normalized:
        return "-"

    prefix = "databricks-"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]

    tool_prefixes = {
        "claude": "claude-",
        "gemini": "gemini-",
        "codex": "gpt-",
    }
    tool_prefix = tool_prefixes.get(tool)
    if tool_prefix and normalized.startswith(tool_prefix):
        normalized = normalized[len(tool_prefix) :]
    return normalized


def extract_model_names(tool: str, raw_models: object) -> list[str]:
    if not isinstance(raw_models, str) or not raw_models.strip():
        return []

    unique_models: list[str] = []
    for item in raw_models.split(","):
        simplified = simplify_model_name(tool, item.strip())
        if simplified != "-" and simplified not in unique_models:
            unique_models.append(simplified)
    return unique_models


def summarize_models(tool: str, raw_models: object) -> str:
    if not isinstance(raw_models, str) or not raw_models.strip():
        return "-"
    parts = extract_model_names(tool, raw_models)
    return ", ".join(parts) if parts else "-"


def empty_tool_day(tool: str, usage_day: date) -> dict[str, object]:
    return {
        "tool": tool,
        "usage_day": usage_day,
        "total_tokens_used": 0,
        "sessions": 0,
        "first_event_time": None,
        "last_event_time": None,
        "models": "-",
    }


def build_tool_breakdown_rows(records: list[dict[str, object]], tool: str) -> list[list[str]]:
    today = date.today()
    rows_by_day: dict[date, dict[str, object]] = {}
    for record in records:
        if record.get("tool") != tool:
            continue
        usage_day = coerce_date(record.get("usage_day"))
        if usage_day:
            rows_by_day[usage_day] = record

    rendered_rows: list[list[str]] = []
    for day_offset in range(USAGE_BREAKDOWN_DAYS):
        usage_day = today - timedelta(days=day_offset)
        record = rows_by_day.get(usage_day) or empty_tool_day(tool, usage_day)
        first_event_time = coerce_datetime(record.get("first_event_time"))
        last_event_time = coerce_datetime(record.get("last_event_time"))
        duration = None
        if first_event_time and last_event_time:
            duration = last_event_time - first_event_time
        token_total = int(cast(int, record.get("total_tokens_used") or 0))
        session_total = int(cast(int, record.get("sessions") or 0))
        rendered_rows.append(
            [
                usage_day.strftime("%m-%d"),
                usage_day.strftime("%a"),
                format_token_count(token_total) if token_total else "-",
                str(session_total) if session_total else "-",
                format_duration(duration),
                summarize_models(tool, record.get("models")),
            ]
        )

    return rendered_rows


def find_requester_name(
    workspace: str,
    http_path: str,
    token: str,
    records: list[dict[str, object]],
) -> str:
    for record in records:
        requester_name = record.get("requester_name")
        if isinstance(requester_name, str) and requester_name.strip():
            return requester_name.strip()

    columns, rows = run_usage_query(workspace, http_path, token, build_current_user_query())
    parsed_rows = parse_usage_rows(columns, rows)
    if parsed_rows:
        requester_name = parsed_rows[0].get("requester_name")
        if isinstance(requester_name, str) and requester_name.strip():
            return requester_name.strip()
    return "current user"


def render_usage_summary(
    records: list[dict[str, object]],
    requester_name: str,
    tool_displays: dict[str, str],
) -> str:
    today = date.today()
    week_start = today - timedelta(days=USAGE_BREAKDOWN_DAYS - 1)
    month_start = today - timedelta(days=USAGE_SUMMARY_DAYS - 1)

    daily_total = 0
    weekly_total = 0
    monthly_total = 0
    active_tools_last_week: list[str] = []
    weekly_model_tokens: dict[str, int] = {}
    for record in records:
        usage_day = coerce_date(record.get("usage_day"))
        if not usage_day:
            continue
        token_total = int(cast(int, record.get("total_tokens_used") or 0))
        tool = record.get("tool")
        if usage_day >= month_start:
            monthly_total += token_total
        if usage_day >= week_start:
            weekly_total += token_total
            if (
                isinstance(tool, str)
                and tool in tool_displays
                and tool not in active_tools_last_week
            ):
                active_tools_last_week.append(tool)
            if isinstance(tool, str):
                for model_name in extract_model_names(tool, record.get("models")):
                    weekly_model_tokens[model_name] = (
                        weekly_model_tokens.get(model_name, 0) + token_total
                    )
        if usage_day == today:
            daily_total += token_total

    lines = [
        heading(f"Usage Summary for {requester_name}"),
        "",
        "[bold green]✓[/bold green] Databricks AI Gateway usage",
        f"{label('Today:')} {value(format_token_count(daily_total) + ' tokens')}",
        f"{label('Last 7 days:')} {value(format_token_count(weekly_total) + ' tokens')}",
        f"{label('Last 30 days:')} {value(format_token_count(monthly_total) + ' tokens')}",
    ]
    if active_tools_last_week:
        tool_text = ", ".join(tool_displays[tool] for tool in active_tools_last_week)
        lines.append(f"{label('Active tools:')} {value(tool_text)}")
    if weekly_model_tokens:
        top_models = sorted(
            weekly_model_tokens.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )[:3]
        models_text = ", ".join(
            f"{model_name} ({format_token_count(token_total)})"
            for model_name, token_total in top_models
        )
        lines.append(f"{label('Top models this week:')} {value(models_text)}")
    return "\n".join(lines)


def usage() -> int:
    # Late import to avoid circular import (agents → state, but usage uses TOOL_SPECS for displays).
    from ucode.agents import TOOL_SPECS

    state = load_state()
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("Workspace is not configured. Run `ucode configure` first.")

    ensure_databricks_auth(workspace)
    with spinner("Retrieving Databricks access token..."):
        token = get_databricks_token(workspace)

    with spinner("Discovering SQL warehouse..."):
        resolved_http_path = discover_sql_warehouse_http_path(workspace, token, quiet=False)

    with spinner("Querying system.ai_gateway.usage..."):
        columns, rows = run_usage_query(
            workspace,
            resolved_http_path,
            token,
            build_usage_report_query(),
        )
    records = parse_usage_rows(columns, rows)
    requester_name = find_requester_name(workspace, resolved_http_path, token, records)

    tool_displays = {tool: spec["display"] for tool, spec in TOOL_SPECS.items()}
    console.print(render_usage_summary(records, requester_name, tool_displays))

    table_headers = ["Date", "Day", "Tokens", "Sessions", "Duration", "Models"]
    table_widths = [8, 5, 10, 8, 8, 24]

    for tool, spec in TOOL_SPECS.items():
        print_heading(f"{spec['display']} · Last {USAGE_BREAKDOWN_DAYS} Days")
        console.print(
            render_box_table(
                table_headers,
                build_tool_breakdown_rows(records, tool),
                max_widths=table_widths,
            )
        )
    return 0
