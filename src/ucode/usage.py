"""Usage report querying, local accounting, and rendering.

Reads from `system.ai_gateway.usage` via a Databricks SQL warehouse.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

from rich.panel import Panel
from rich.text import Text

from ucode.config_io import APP_DIR
from ucode.databricks import (
    discover_sql_warehouse_http_path,
    ensure_databricks_auth,
    get_databricks_token,
    run_usage_query,
)
from ucode.policies import (
    active_tier,
    daily_budget_usd,
    load_workspace_policy,
    on_budget_exhausted,
)
from ucode.state import load_state
from ucode.ui import (
    console,
    format_duration,
    format_token_count,
    heading,
    label,
    print_heading,
    print_note,
    render_box_table,
    spinner,
    value,
)

USAGE_BREAKDOWN_DAYS = 7
USAGE_SUMMARY_DAYS = 30
LOCAL_USAGE_DB_PATH = APP_DIR / "usage.sqlite"
# Daily cap used when no workspace policy is configured.
DEFAULT_DAILY_BUDGET_USD = 500.0
LOCAL_BUDGET_WARN_AT = 0.8
ENV_DAILY_BUDGET_USD = "UCODE_USAGE_DAILY_BUDGET_USD"
ENV_PRICE_MULTIPLIER = "UCODE_USAGE_PRICE_MULTIPLIER"
_CLAUDE_FAMILY_RE = re.compile(r"^(?:databricks-)?claude-(opus|sonnet|haiku)-4(?:[-.].*)?$")
_CLAUDE_VERSION_RE = re.compile(
    r"^(?:databricks-)?claude-(opus|sonnet|haiku)-4(?:[-.](\d+))?(?:[-.].*)?$"
)


@dataclass(frozen=True)
class ModelPrice:
    """USD pricing per 1M tokens."""

    input: float
    output: float
    cache_read_input: float = 0.0
    cache_creation_input: float | None = None


# Keep this table intentionally small and explicit. Unknown models are still
# tracked for tokens, but spend is reported as $0 until a price is added here.
MODEL_PRICES_USD_PER_1M: dict[str, ModelPrice] = {
    "gpt-5": ModelPrice(input=1.25, output=10.0),
    "gpt-5.2": ModelPrice(input=1.75, output=14.0, cache_read_input=0.175),
    "gpt-5.4": ModelPrice(input=2.5, output=15.0, cache_read_input=0.25),
    "gpt-5.4-mini": ModelPrice(input=0.75, output=4.5, cache_read_input=0.075),
    "gpt-5.4-nano": ModelPrice(input=0.2, output=1.25, cache_read_input=0.02),
    "gpt-5.5": ModelPrice(input=50.0, output=300.0, cache_read_input=5.0),
    "gpt-5.5-mini": ModelPrice(input=0.25, output=2.0),
    "gpt-5.5-pro": ModelPrice(input=30.0, output=180.0),
    "databricks-gpt-5": ModelPrice(input=1.25, output=10.0),
    "databricks-gpt-5-2": ModelPrice(input=1.75, output=14.0, cache_read_input=0.175),
    "databricks-gpt-5-4": ModelPrice(input=2.5, output=15.0, cache_read_input=0.25),
    "databricks-gpt-5-4-mini": ModelPrice(input=0.75, output=4.5, cache_read_input=0.075),
    "databricks-gpt-5-4-nano": ModelPrice(input=0.2, output=1.25, cache_read_input=0.02),
    "databricks-gpt-5-5": ModelPrice(input=50.0, output=300.0, cache_read_input=5.0),
    "databricks-gpt-5-5-mini": ModelPrice(input=0.25, output=2.0),
    "databricks-gpt-5-5-pro": ModelPrice(input=30.0, output=180.0),
    "databricks-claude-opus-4": ModelPrice(
        input=5.0, output=25.0, cache_creation_input=6.25, cache_read_input=0.5
    ),
    "databricks-claude-sonnet-4": ModelPrice(
        input=3.0, output=15.0, cache_creation_input=3.75, cache_read_input=0.3
    ),
    "databricks-claude-haiku-4": ModelPrice(
        input=1.0, output=5.0, cache_creation_input=1.25, cache_read_input=0.1
    ),
    "databricks-claude-opus-4-1": ModelPrice(
        input=15.0, output=75.0, cache_creation_input=18.75, cache_read_input=1.5
    ),
    "databricks-claude-opus-4-5": ModelPrice(
        input=5.0, output=25.0, cache_creation_input=6.25, cache_read_input=0.5
    ),
    "databricks-claude-opus-4-6": ModelPrice(
        input=5.0, output=25.0, cache_creation_input=6.25, cache_read_input=0.5
    ),
    "databricks-claude-opus-4-8": ModelPrice(
        input=50.0, output=250.0, cache_creation_input=62.5, cache_read_input=5.0
    ),
    "databricks-claude-sonnet-4-5": ModelPrice(
        input=3.0, output=15.0, cache_creation_input=3.75, cache_read_input=0.3
    ),
    "databricks-claude-sonnet-4-6": ModelPrice(
        input=30.0, output=150.0, cache_creation_input=37.5, cache_read_input=3.0
    ),
    "databricks-claude-haiku-4-5": ModelPrice(
        input=1.0, output=5.0, cache_creation_input=1.25, cache_read_input=0.1
    ),
}


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_usage_report_query() -> str:
    return f"""
WITH usage_events AS (
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
  request_id,
  event_time,
  destination_model,
  COALESCE(total_tokens, 0) AS total_tokens_used
FROM system.ai_gateway.usage
WHERE event_time >= current_timestamp() - interval {USAGE_SUMMARY_DAYS} days
  AND requester = current_user()
  AND (
    lower(user_agent) LIKE '%codex%'
    OR lower(user_agent) LIKE '%claude%'
    OR lower(user_agent) LIKE '%gemini%'
    OR lower(user_agent) LIKE '%opencode%'
  )
),
daily_usage AS (
  SELECT
    requester_name,
    tool,
    usage_day,
    SUM(total_tokens_used) AS total_tokens_used,
    COUNT(DISTINCT request_id) AS sessions,
    MIN(event_time) AS first_event_time,
    MAX(event_time) AS last_event_time
  FROM usage_events
  GROUP BY 1, 2, 3
),
model_usage AS (
  SELECT
    requester_name,
    tool,
    usage_day,
    destination_model,
    SUM(total_tokens_used) AS model_tokens_used
  FROM usage_events
  WHERE destination_model IS NOT NULL AND destination_model != ''
  GROUP BY 1, 2, 3, 4
),
model_rollup AS (
  SELECT
    requester_name,
    tool,
    usage_day,
    CONCAT_WS(', ', SORT_ARRAY(COLLECT_SET(destination_model))) AS models,
    TO_JSON(
      SORT_ARRAY(
        COLLECT_LIST(
          NAMED_STRUCT('model', destination_model, 'tokens', model_tokens_used)
        )
      )
    ) AS model_tokens
  FROM model_usage
  GROUP BY 1, 2, 3
)
SELECT
  daily_usage.requester_name,
  daily_usage.tool,
  daily_usage.usage_day,
  daily_usage.total_tokens_used,
  daily_usage.sessions,
  daily_usage.first_event_time,
  daily_usage.last_event_time,
  COALESCE(model_rollup.models, '') AS models,
  COALESCE(model_rollup.model_tokens, '[]') AS model_tokens
FROM daily_usage
LEFT JOIN model_rollup
  ON daily_usage.requester_name = model_rollup.requester_name
  AND daily_usage.tool = model_rollup.tool
  AND daily_usage.usage_day = model_rollup.usage_day
ORDER BY daily_usage.usage_day DESC, daily_usage.tool ASC
""".strip()


def build_current_user_query() -> str:
    return "SELECT current_user() AS requester_name"


def _connect_local_usage_db(db_path: Path = LOCAL_USAGE_DB_PATH) -> sqlite3.Connection:
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to open local usage database: {db_path}") from exc
    return conn


def ensure_local_usage_schema(db_path: Path = LOCAL_USAGE_DB_PATH) -> None:
    with _connect_local_usage_db(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS usage_events (
              event_id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              session_id TEXT NOT NULL,
              tool TEXT NOT NULL,
              model TEXT NOT NULL,
              workspace TEXT,
              input_tokens INTEGER NOT NULL DEFAULT 0,
              output_tokens INTEGER NOT NULL DEFAULT 0,
              cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
              cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
              total_tokens INTEGER NOT NULL DEFAULT 0,
              cost_usd REAL NOT NULL DEFAULT 0,
              source TEXT NOT NULL DEFAULT 'manual'
            );
            CREATE INDEX IF NOT EXISTS usage_events_created_at_idx
              ON usage_events(created_at);
            CREATE INDEX IF NOT EXISTS usage_events_session_idx
              ON usage_events(session_id);
            CREATE INDEX IF NOT EXISTS usage_events_tool_model_idx
              ON usage_events(tool, model);

            CREATE TABLE IF NOT EXISTS usage_session_snapshots (
              session_id TEXT NOT NULL,
              tool TEXT NOT NULL,
              model TEXT NOT NULL,
              input_tokens INTEGER NOT NULL DEFAULT 0,
              output_tokens INTEGER NOT NULL DEFAULT 0,
              cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
              cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
              total_tokens INTEGER NOT NULL DEFAULT 0,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (session_id, tool, model)
            );

            CREATE TABLE IF NOT EXISTS usage_metadata (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            """
        )


def _coerce_token_count(value_obj: object) -> int:
    try:
        value = int(cast(int | float | str, value_obj or 0))
    except (TypeError, ValueError):
        value = 0
    return max(value, 0)


def _coerce_cost(value_obj: object) -> float:
    try:
        return float(cast(int | float | str, value_obj or 0))
    except (TypeError, ValueError):
        return 0.0


def _parse_utc_iso(value_obj: object) -> datetime | None:
    if not isinstance(value_obj, str) or not value_obj.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value_obj.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def ensure_local_usage_sync_started_at(
    key: str,
    db_path: Path = LOCAL_USAGE_DB_PATH,
) -> datetime:
    """Return a stable local sync watermark, creating it for fresh ledgers."""
    ensure_local_usage_schema(db_path)
    with _connect_local_usage_db(db_path) as conn:
        with conn:
            row = conn.execute("SELECT value FROM usage_metadata WHERE key = ?", (key,)).fetchone()
            if row:
                parsed = _parse_utc_iso(row["value"])
                if parsed is not None:
                    return parsed
            now = utc_now_iso()
            conn.execute(
                """
                INSERT INTO usage_metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, now),
            )
    parsed_now = _parse_utc_iso(now)
    if parsed_now is None:
        return datetime.now(UTC)
    return parsed_now


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    return value


def _workspace_policy() -> dict | None:
    try:
        state = load_state()
    except Exception:
        return None
    workspace = state.get("workspace")
    return load_workspace_policy(workspace if isinstance(workspace, str) else None)


def _policy_daily_limit_usd() -> float | None:
    """Read the daily budget from the current workspace policy YAML.

    This is a single budget shared across every coding tool. Returns ``None``
    when the workspace has no policy configured so callers can fall
    back to a default.
    """
    return daily_budget_usd(_workspace_policy())


def local_daily_agent_budget_usd(tool: str | None = None) -> float:
    """Daily spend cap shared across all coding tools.

    Resolution order: the ``UCODE_USAGE_DAILY_BUDGET_USD`` env override, then
    ``policy.daily_budget_usd`` from the workspace policy YAML, then a
    ``$500/day`` default when no policy is configured. ``tool`` is accepted for
    call-site compatibility but ignored — the limit is global.
    """
    del tool
    env_override = os.environ.get(ENV_DAILY_BUDGET_USD)
    if env_override is not None and env_override.strip():
        return _env_float(ENV_DAILY_BUDGET_USD, DEFAULT_DAILY_BUDGET_USD, minimum=0.01)
    daily_limit = _policy_daily_limit_usd()
    if daily_limit is not None:
        return daily_limit
    return DEFAULT_DAILY_BUDGET_USD


def local_price_multiplier() -> float:
    return _env_float(ENV_PRICE_MULTIPLIER, 1.0, minimum=0.0)


def _model_price(model: str) -> ModelPrice | None:
    normalized = model.split("/")[-1].removesuffix("[1m]")
    if normalized in MODEL_PRICES_USD_PER_1M:
        return MODEL_PRICES_USD_PER_1M[normalized]
    version_match = _CLAUDE_VERSION_RE.match(normalized)
    if version_match:
        family = version_match.group(1)
        minor = version_match.group(2)
        version_key = f"databricks-claude-{family}-4"
        if minor:
            version_key = f"{version_key}-{minor}"
        if version_key in MODEL_PRICES_USD_PER_1M:
            return MODEL_PRICES_USD_PER_1M[version_key]
    match = _CLAUDE_FAMILY_RE.match(normalized)
    if match:
        family = match.group(1)
        if family == "opus":
            return MODEL_PRICES_USD_PER_1M["databricks-claude-opus-4"]
        if family == "sonnet":
            return MODEL_PRICES_USD_PER_1M["databricks-claude-sonnet-4"]
        if family == "haiku":
            return MODEL_PRICES_USD_PER_1M["databricks-claude-haiku-4-5"]
    return MODEL_PRICES_USD_PER_1M.get(normalized)


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    total_tokens: int = 0,
) -> float:
    price = _model_price(model)
    if not price:
        return 0.0
    if (
        not any((input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens))
        and total_tokens
    ):
        return ((total_tokens * price.input) / 1_000_000) * local_price_multiplier()
    cache_creation_price = price.cache_creation_input
    if cache_creation_price is None:
        cache_creation_price = price.input
    return (
        (
            (input_tokens * price.input)
            + (output_tokens * price.output)
            + (cache_read_input_tokens * price.cache_read_input)
            + (cache_creation_input_tokens * cache_creation_price)
        )
        / 1_000_000
        * local_price_multiplier()
    )


def _sum_total_tokens(
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int,
    cache_creation_input_tokens: int,
    total_tokens: int,
) -> int:
    if total_tokens:
        return total_tokens
    return input_tokens + output_tokens + cache_read_input_tokens + cache_creation_input_tokens


def _event_from_row(row: sqlite3.Row) -> dict[str, object]:
    return {key: row[key] for key in row.keys()}


def record_local_usage_delta(
    *,
    session_id: str,
    tool: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    total_tokens: int = 0,
    workspace: str | None = None,
    source: str = "manual",
    db_path: Path = LOCAL_USAGE_DB_PATH,
    created_at: str | None = None,
) -> dict[str, object]:
    ensure_local_usage_schema(db_path)
    input_tokens = _coerce_token_count(input_tokens)
    output_tokens = _coerce_token_count(output_tokens)
    cache_read_input_tokens = _coerce_token_count(cache_read_input_tokens)
    cache_creation_input_tokens = _coerce_token_count(cache_creation_input_tokens)
    total_tokens = _sum_total_tokens(
        input_tokens,
        output_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
        _coerce_token_count(total_tokens),
    )
    cost_usd = estimate_cost_usd(
        model,
        input_tokens,
        output_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
        total_tokens,
    )
    event: dict[str, object] = {
        "event_id": str(uuid.uuid4()),
        "created_at": created_at or utc_now_iso(),
        "session_id": session_id,
        "tool": tool,
        "model": model,
        "workspace": workspace,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "source": source or "manual",
    }
    with _connect_local_usage_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO usage_events (
              event_id, created_at, session_id, tool, model, workspace,
              input_tokens, output_tokens, cache_read_input_tokens,
              cache_creation_input_tokens, total_tokens, cost_usd, source
            ) VALUES (
              :event_id, :created_at, :session_id, :tool, :model, :workspace,
              :input_tokens, :output_tokens, :cache_read_input_tokens,
              :cache_creation_input_tokens, :total_tokens, :cost_usd, :source
            )
            """,
            event,
        )
    return event


MANUAL_OVERRIDE_SOURCE = "manual-override"


def set_local_daily_spend(
    cost_usd: float,
    db_path: Path = LOCAL_USAGE_DB_PATH,
) -> dict[str, object]:
    """Overwrite today's local spend to exactly ``cost_usd`` dollars."""
    if cost_usd < 0:
        raise RuntimeError("cost_usd must be non-negative.")
    ensure_local_usage_schema(db_path)
    event: dict[str, object] = {
        "event_id": str(uuid.uuid4()),
        "created_at": utc_now_iso(),
        "session_id": "manual-override",
        "tool": "manual",
        "model": "manual",
        "workspace": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "total_tokens": 0,
        "cost_usd": float(cost_usd),
        "source": MANUAL_OVERRIDE_SOURCE,
    }
    with _connect_local_usage_db(db_path) as conn:
        conn.execute("DELETE FROM usage_events WHERE created_at >= datetime('now', '-1 days')")
        conn.execute(
            """
            INSERT INTO usage_events (
              event_id, created_at, session_id, tool, model, workspace,
              input_tokens, output_tokens, cache_read_input_tokens,
              cache_creation_input_tokens, total_tokens, cost_usd, source
            ) VALUES (
              :event_id, :created_at, :session_id, :tool, :model, :workspace,
              :input_tokens, :output_tokens, :cache_read_input_tokens,
              :cache_creation_input_tokens, :total_tokens, :cost_usd, :source
            )
            """,
            event,
        )
    return event


def record_local_usage_snapshot(
    *,
    session_id: str,
    tool: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    total_tokens: int = 0,
    workspace: str | None = None,
    source: str = "hook",
    db_path: Path = LOCAL_USAGE_DB_PATH,
    created_at: str | None = None,
) -> dict[str, object] | None:
    ensure_local_usage_schema(db_path)
    input_tokens = _coerce_token_count(input_tokens)
    output_tokens = _coerce_token_count(output_tokens)
    cache_read_input_tokens = _coerce_token_count(cache_read_input_tokens)
    cache_creation_input_tokens = _coerce_token_count(cache_creation_input_tokens)
    total_tokens = _sum_total_tokens(
        input_tokens,
        output_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
        _coerce_token_count(total_tokens),
    )
    with _connect_local_usage_db(db_path) as conn:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            previous = conn.execute(
                """
                SELECT input_tokens, output_tokens, cache_read_input_tokens,
                       cache_creation_input_tokens, total_tokens
                FROM usage_session_snapshots
                WHERE session_id = ? AND tool = ? AND model = ?
                """,
                (session_id, tool, model),
            ).fetchone()
            if previous:
                delta_input = max(input_tokens - int(previous["input_tokens"]), 0)
                delta_output = max(output_tokens - int(previous["output_tokens"]), 0)
                delta_cache_read = max(
                    cache_read_input_tokens - int(previous["cache_read_input_tokens"]), 0
                )
                delta_cache_creation = max(
                    cache_creation_input_tokens - int(previous["cache_creation_input_tokens"]), 0
                )
                delta_total = max(total_tokens - int(previous["total_tokens"]), 0)
            else:
                delta_input = input_tokens
                delta_output = output_tokens
                delta_cache_read = cache_read_input_tokens
                delta_cache_creation = cache_creation_input_tokens
                delta_total = total_tokens

            now = created_at or utc_now_iso()
            if previous and not any(
                (delta_input, delta_output, delta_cache_read, delta_cache_creation, delta_total)
            ):
                return None

            conn.execute(
                """
                INSERT INTO usage_session_snapshots (
                  session_id, tool, model, input_tokens, output_tokens,
                  cache_read_input_tokens, cache_creation_input_tokens,
                  total_tokens, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, tool, model) DO UPDATE SET
                  input_tokens = excluded.input_tokens,
                  output_tokens = excluded.output_tokens,
                  cache_read_input_tokens = excluded.cache_read_input_tokens,
                  cache_creation_input_tokens = excluded.cache_creation_input_tokens,
                  total_tokens = excluded.total_tokens,
                  updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    tool,
                    model,
                    input_tokens,
                    output_tokens,
                    cache_read_input_tokens,
                    cache_creation_input_tokens,
                    total_tokens,
                    now,
                ),
            )
            if not previous and not any(
                (delta_input, delta_output, delta_cache_read, delta_cache_creation, delta_total)
            ):
                return None

            cost_usd = estimate_cost_usd(
                model,
                delta_input,
                delta_output,
                delta_cache_read,
                delta_cache_creation,
                delta_total,
            )
            event: dict[str, object] = {
                "event_id": str(uuid.uuid4()),
                "created_at": now,
                "session_id": session_id,
                "tool": tool,
                "model": model,
                "workspace": workspace,
                "input_tokens": delta_input,
                "output_tokens": delta_output,
                "cache_read_input_tokens": delta_cache_read,
                "cache_creation_input_tokens": delta_cache_creation,
                "total_tokens": delta_total,
                "cost_usd": cost_usd,
                "source": source or "hook",
            }
            conn.execute(
                """
                INSERT INTO usage_events (
                  event_id, created_at, session_id, tool, model, workspace,
                  input_tokens, output_tokens, cache_read_input_tokens,
                  cache_creation_input_tokens, total_tokens, cost_usd, source
                ) VALUES (
                  :event_id, :created_at, :session_id, :tool, :model, :workspace,
                  :input_tokens, :output_tokens, :cache_read_input_tokens,
                  :cache_creation_input_tokens, :total_tokens, :cost_usd, :source
                )
                """,
                event,
            )
    return event


def query_local_usage_summary(
    *,
    days: int = USAGE_BREAKDOWN_DAYS,
    db_path: Path = LOCAL_USAGE_DB_PATH,
) -> list[dict[str, object]]:
    ensure_local_usage_schema(db_path)
    with _connect_local_usage_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
              date(created_at) AS usage_day,
              tool,
              model,
              COUNT(DISTINCT session_id) AS sessions,
              SUM(input_tokens) AS input_tokens,
              SUM(output_tokens) AS output_tokens,
              SUM(cache_read_input_tokens) AS cache_read_input_tokens,
              SUM(cache_creation_input_tokens) AS cache_creation_input_tokens,
              SUM(total_tokens) AS total_tokens,
              SUM(cost_usd) AS cost_usd,
              MIN(created_at) AS first_event_time,
              MAX(created_at) AS last_event_time
            FROM usage_events
            WHERE created_at >= datetime('now', ?)
            GROUP BY 1, 2, 3
            ORDER BY usage_day DESC, tool ASC, total_tokens DESC
            """,
            (f"-{days} days",),
        ).fetchall()
    return [_event_from_row(row) for row in rows]


def query_local_usage_totals(
    *,
    days: int = USAGE_BREAKDOWN_DAYS,
    tool: str | None = None,
    db_path: Path = LOCAL_USAGE_DB_PATH,
) -> dict[str, object]:
    ensure_local_usage_schema(db_path)
    tool_filter = "AND tool = ?" if tool else ""
    params: tuple[object, ...] = (f"-{days} days", tool) if tool else (f"-{days} days",)
    with _connect_local_usage_db(db_path) as conn:
        row = conn.execute(
            f"""
            SELECT
              COUNT(*) AS events,
              COUNT(DISTINCT session_id) AS sessions,
              COALESCE(SUM(input_tokens), 0) AS input_tokens,
              COALESCE(SUM(output_tokens), 0) AS output_tokens,
              COALESCE(SUM(cache_read_input_tokens), 0) AS cache_read_input_tokens,
              COALESCE(SUM(cache_creation_input_tokens), 0) AS cache_creation_input_tokens,
              COALESCE(SUM(total_tokens), 0) AS total_tokens,
              COALESCE(SUM(cost_usd), 0) AS cost_usd
            FROM usage_events
            WHERE created_at >= datetime('now', ?)
              {tool_filter}
            """,
            params,
        ).fetchone()
    return _event_from_row(cast(sqlite3.Row, row))


def query_local_budget_totals(
    *,
    days: int = 1,
    tool: str | None = None,
    db_path: Path = LOCAL_USAGE_DB_PATH,
) -> dict[str, object]:
    """Sum local usage over the trailing window.

    When ``tool`` is provided, only rows for that tool are included. Callers
    that need the shared daily-budget spend across all tools should omit it.
    """
    ensure_local_usage_schema(db_path)
    tool_filter = "AND tool = ?" if tool else ""
    params: tuple[object, ...] = (f"-{days} days", tool) if tool else (f"-{days} days",)
    with _connect_local_usage_db(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
              session_id,
              tool,
              model,
              input_tokens,
              output_tokens,
              cache_read_input_tokens,
              cache_creation_input_tokens,
              total_tokens,
              cost_usd
            FROM usage_events
            WHERE created_at >= datetime('now', ?)
              {tool_filter}
            """,
            params,
        ).fetchall()

    sessions: set[str] = set()
    input_tokens = 0
    output_tokens = 0
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0
    total_tokens = 0
    cost_usd = 0.0
    for row in rows:
        sessions.add(str(row["session_id"]))
        row_input_tokens = _coerce_token_count(row["input_tokens"])
        row_output_tokens = _coerce_token_count(row["output_tokens"])
        row_cache_read_input_tokens = _coerce_token_count(row["cache_read_input_tokens"])
        row_cache_creation_input_tokens = _coerce_token_count(row["cache_creation_input_tokens"])
        row_total_tokens = _coerce_token_count(row["total_tokens"])
        input_tokens += row_input_tokens
        output_tokens += row_output_tokens
        cache_read_input_tokens += row_cache_read_input_tokens
        cache_creation_input_tokens += row_cache_creation_input_tokens
        total_tokens += row_total_tokens
        stored_cost = _coerce_cost(row["cost_usd"])
        if stored_cost:
            cost_usd += stored_cost
        else:
            cost_usd += estimate_cost_usd(
                str(row["model"] or ""),
                row_input_tokens,
                row_output_tokens,
                row_cache_read_input_tokens,
                row_cache_creation_input_tokens,
                row_total_tokens,
            )

    return {
        "events": len(rows),
        "sessions": len(sessions),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


def local_budget_status(
    tool: str | None = None,
    db_path: Path = LOCAL_USAGE_DB_PATH,
    *,
    days: int = 1,
) -> dict[str, object]:
    """Budget status for the single daily limit shared across all tools.

    ``tool`` scopes the displayed session/token totals for launch panels, while
    spend and the limit remain aggregated globally across every coding tool."""
    limit_usd = local_daily_agent_budget_usd()
    policy = _workspace_policy()
    warn_at = LOCAL_BUDGET_WARN_AT
    totals = query_local_budget_totals(days=days, db_path=db_path)
    display_totals = (
        query_local_budget_totals(days=days, tool=tool, db_path=db_path) if tool else totals
    )
    spend_usd = _coerce_cost(totals.get("cost_usd"))
    warn_usd = limit_usd * warn_at
    if spend_usd >= limit_usd:
        state = "exceeded"
    elif spend_usd >= warn_usd:
        state = "warn"
    else:
        state = "ok"
    return {
        "configured": True,
        "state": state,
        "tool": tool,
        "limit_usd": limit_usd,
        "warn_at": warn_at,
        "warn_usd": warn_usd,
        "days": days,
        "spend_usd": spend_usd,
        "remaining_usd": max(limit_usd - spend_usd, 0.0),
        "total_tokens": _coerce_token_count(display_totals.get("total_tokens")),
        "sessions": _coerce_token_count(display_totals.get("sessions")),
        "on_budget_exhausted": on_budget_exhausted(policy),
        "active_tier": active_tier(policy, spend_usd),
    }


def _budget_usage_percent(spend_usd: float, limit_usd: float) -> int:
    if limit_usd <= 0:
        return 0
    return max(int(((spend_usd / limit_usd) * 100) + 0.5), 0)


def _budget_progress_bar(percent: int, *, width: int = 20) -> str:
    filled = min(max((percent * width + 50) // 100, 0), width)
    return ("▰" * filled) + ("▱" * (width - filled))


def _budget_line(spend_usd: float, limit_usd: float) -> str:
    percent = _budget_usage_percent(spend_usd, limit_usd)
    return (
        f"Budget: ${spend_usd:.2f} / ${limit_usd:.2f} used today "
        f"{_budget_progress_bar(percent)} {percent}%."
    )


def format_local_budget_status(status: dict[str, object]) -> str:
    state = str(status.get("state") or "ok")
    tool = str(status.get("tool") or "agent")
    display_tool = tool[:1].upper() + tool[1:] if tool else "Agent"
    spend_usd = _coerce_cost(status.get("spend_usd"))
    limit_usd = _coerce_cost(status.get("limit_usd"))
    days = _coerce_token_count(status.get("days"))
    remaining_usd = _coerce_cost(status.get("remaining_usd"))
    tokens = _coerce_token_count(status.get("total_tokens"))
    budget_line = _budget_line(spend_usd, limit_usd)
    token_subject = (
        f"{display_tool} tokens used today" if status.get("tool") else "Tokens used today"
    )
    tokens_line = f"{token_subject}: {format_token_count(tokens)}."
    if state == "exceeded":
        return (
            f"⛔ [UCODE USAGE BUDGET] {display_tool} daily budget exceeded. \n"
            f"{budget_line}\n"
            f"Window: {days} day(s). \n"
            f"{tokens_line}\n"
            "Further tool use is blocked for this agent today."
        )
    if state == "warn":
        return (
            f"⚠️ [UCODE USAGE BUDGET] {display_tool} is nearing its daily budget. \n"
            f"{budget_line}\n"
            f"Remaining: ${remaining_usd:.2f}. Window: {days} day(s). \n"
            f"{tokens_line}"
        )
    return (
        f"[UCODE USAGE BUDGET] {display_tool}. "
        f"{budget_line} Remaining: ${remaining_usd:.2f}. Window: {days} day(s). "
        f"{tokens_line}"
    )


_BUDGET_STATE_STYLE: dict[str, str] = {"ok": "green", "warn": "yellow", "exceeded": "red"}


def _budget_state_color(state: str) -> str:
    return _BUDGET_STATE_STYLE.get(state, "green")


def _budget_bar_markup(percent: int, color: str, *, width: int = 28) -> str:
    """A Rich-markup fill bar colored by budget state. The filled portion uses
    the state color; the remainder is dimmed."""
    capped = min(max(percent, 0), 100)
    filled = min(max((capped * width + 50) // 100, 0), width)
    # When spend is over the limit the bar is full and red regardless of cap.
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (width - filled)}[/dim]"


def _budget_header_line(state: str, display_tool: str | None) -> str | None:
    """The status callout shown above the bar for non-ok states.

    ``display_tool`` is ``None`` for the global, all-tools budget view."""
    if state == "exceeded":
        subject = f"{display_tool} daily budget" if display_tool else "Daily budget"
        return f"[bold red]⛔ {subject} exceeded[/bold red]"
    if state == "warn":
        nearing = (
            f"{display_tool} is nearing its daily budget"
            if display_tool
            else "Nearing daily budget"
        )
        return f"[bold yellow]⚠️  {nearing}[/bold yellow]"
    return None


def render_local_budget_panel(
    status: dict[str, object],
    *,
    title: str | None = None,
    extra_lines: list[str] | None = None,
) -> Panel:
    """Render a budget status as a bordered Rich panel with a color-coded fill
    bar. Used by `ucode usage budget-status` and the launch budget summary.

    Pass ``title`` to override the default ``"<Tool> · Daily Budget"`` panel
    title. Pass ``extra_lines`` to inject Rich-markup lines into the body
    between the progress bar and the Remaining/Tokens/Sessions footer — used
    by the warn-state switch prompt to surface a "currently X, recommend Y"
    sentence inside the same panel."""
    state = str(status.get("state") or "ok")
    raw_tool = status.get("tool")
    tool = str(raw_tool) if raw_tool else ""
    display_tool = tool[:1].upper() + tool[1:] if tool else None
    spend_usd = _coerce_cost(status.get("spend_usd"))
    limit_usd = _coerce_cost(status.get("limit_usd"))
    remaining_usd = _coerce_cost(status.get("remaining_usd"))
    tokens = _coerce_token_count(status.get("total_tokens"))
    sessions = _coerce_token_count(status.get("sessions"))
    percent = _budget_usage_percent(spend_usd, limit_usd)
    color = _budget_state_color(state)
    tokens_label = f"{display_tool} Tokens" if display_tool else "Total Tokens"
    sessions_label = f"{display_tool} Sessions" if display_tool else "Total Sessions"

    lines: list[str] = []
    header = _budget_header_line(state, display_tool)
    if header:
        lines.append(header)
        lines.append("")
    lines.append(
        f"[bold]${spend_usd:,.2f}[/bold] / ${limit_usd:,.2f}    [{color}]{percent}% used[/{color}]"
    )
    lines.append(_budget_bar_markup(percent, color))
    lines.append("")
    if extra_lines:
        for extra in extra_lines:
            lines.append(extra)
        lines.append("")
    lines.append(f"[bold]Remaining[/bold]   ${remaining_usd:,.2f}")
    lines.append(f"[bold]{tokens_label}[/bold]   {format_token_count(tokens)}")
    lines.append(f"[bold]{sessions_label}[/bold]   {sessions}")

    return Panel(
        Text.from_markup("\n".join(lines)),
        title=Text(title or f"{display_tool or 'Agent'} · Daily Budget", style=f"bold {color}"),
        border_style=color,
        expand=False,
        padding=(1, 2, 0, 2),
    )


def format_local_budget_hook_status(status: dict[str, object]) -> str:
    # The daily budget is a single global pool shared across all tools, so the
    # message is tool-agnostic — it never names the agent that triggered it.
    #
    # Agent hook UIs render this as plain text (Claude `systemMessage`, Codex
    # `additionalContext`) with no Rich/ANSI support, so the "rich" feel comes
    # from a Unicode fill bar plus a compact two-line layout: a headline, then a
    # spend/bar/remaining detail line.
    state = str(status.get("state") or "ok")
    spend_usd = _coerce_cost(status.get("spend_usd"))
    limit_usd = _coerce_cost(status.get("limit_usd"))
    remaining_usd = _coerce_cost(status.get("remaining_usd"))
    percent = _budget_usage_percent(spend_usd, limit_usd)
    bar = _budget_progress_bar(percent)
    if state == "exceeded":
        behavior = status.get("on_budget_exhausted") or "block"
        headline = "⛔ Daily budget — limit exceeded"
        action = "further tool use blocked today"
        if (
            isinstance(behavior, dict)
            and cast("dict[str, object]", behavior).get("action") == "switch"
        ):
            behavior_dict = cast("dict[str, object]", behavior)
            target = behavior_dict.get("target") or {}
            target_dict = cast("dict[str, object]", target) if isinstance(target, dict) else {}
            harness = str(target_dict.get("harness") or "")
            headline = "⚠️ Daily budget — limit exceeded"
            action = f"switching to {harness}"
        elif behavior == "warn":
            headline = "⚠️ Daily budget — limit exceeded"
            action = "continuing because policy is warn"
        elif behavior == "allow":
            action = "continuing because policy is allow"
        detail = f"${spend_usd:.2f} / ${limit_usd:.2f} ({percent}%)  {bar}  {action}"
    elif state == "warn":
        headline = "⚠️ Daily budget — nearing limit"
        detail = (
            f"${spend_usd:.2f} / ${limit_usd:.2f} ({percent}%)  {bar}  ${remaining_usd:.2f} left"
        )
    else:
        headline = "Daily budget"
        detail = (
            f"${spend_usd:.2f} / ${limit_usd:.2f} ({percent}%)  {bar}  ${remaining_usd:.2f} left"
        )
    return f"{headline}\n{detail}"


def render_local_usage_summary(
    summary_rows: list[dict[str, object]],
    totals: dict[str, object],
    *,
    days: int = USAGE_BREAKDOWN_DAYS,
) -> str:
    total_tokens = _coerce_token_count(totals.get("total_tokens"))
    cost_usd = _coerce_cost(totals.get("cost_usd"))
    sessions = _coerce_token_count(totals.get("sessions"))
    lines = [
        heading("Local Usage Summary"),
        "",
        "[bold green]✓[/bold green] SQLite local usage ledger",
        f"{label(f'Last {days} days:')} {value(format_token_count(total_tokens) + ' tokens')}",
        f"{label('Estimated spend:')} {value(f'${cost_usd:.4f}')}",
        f"{label('Sessions:')} {value(str(sessions))}",
    ]
    if not summary_rows:
        lines.append("")
        lines.append("No local usage events recorded yet.")
        return "\n".join(lines)

    top_models: dict[str, tuple[int, float]] = {}
    for row in summary_rows:
        model = str(row.get("model") or "-")
        token_total = _coerce_token_count(row.get("total_tokens"))
        cost_total = _coerce_cost(row.get("cost_usd"))
        existing_tokens, existing_cost = top_models.get(model, (0, 0.0))
        top_models[model] = (existing_tokens + token_total, existing_cost + cost_total)
    top_model_text = ", ".join(
        f"{model} ({format_token_count(tokens)}, ${cost:.4f})"
        for model, (tokens, cost) in sorted(
            top_models.items(),
            key=lambda item: (-item[1][0], item[0].lower()),
        )[:3]
    )
    if top_model_text:
        lines.append(f"{label('Top models:')} {value(top_model_text)}")
    return "\n".join(lines)


def local_usage(days: int = USAGE_BREAKDOWN_DAYS, db_path: Path = LOCAL_USAGE_DB_PATH) -> int:
    summary_rows = query_local_usage_summary(days=days, db_path=db_path)
    totals = query_local_usage_totals(days=days, db_path=db_path)
    console.print(render_local_usage_summary(summary_rows, totals, days=days))
    if summary_rows:
        table_rows = [
            [
                str(row.get("usage_day") or "-"),
                str(row.get("tool") or "-"),
                str(row.get("model") or "-"),
                str(_coerce_token_count(row.get("sessions"))),
                format_token_count(_coerce_token_count(row.get("total_tokens"))),
                f"${_coerce_cost(row.get('cost_usd')):.4f}",
            ]
            for row in summary_rows
        ]
        console.print(
            render_box_table(
                ["Date", "Tool", "Model", "Sessions", "Tokens", "Spend"],
                table_rows,
                max_widths=[10, 10, 28, 8, 10, 10],
            )
        )
    print_note(f"Local usage database: {db_path}")
    return 0


def parse_usage_rows(columns: list[str], rows: list[tuple]) -> list[dict[str, object]]:
    return [dict(zip(columns, row, strict=False)) for row in rows]


def configured_usage_tools(state: dict, tool_displays: dict[str, str]) -> list[str]:
    configured = state.get("available_tools") or state.get("managed_configs", {}).keys()
    if not isinstance(configured, list):
        configured = list(configured)
    return [tool for tool in tool_displays if tool in configured]


def filter_records_for_tools(
    records: list[dict[str, object]],
    tools: list[str],
) -> list[dict[str, object]]:
    configured = set(tools)
    return [record for record in records if record.get("tool") in configured]


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


def _coerce_model_token_item(tool: str, item: object) -> tuple[str, int] | None:
    if not isinstance(item, Mapping):
        return None
    item_mapping = cast(Mapping[str, object], item)

    raw_model = item_mapping.get("model")
    if not isinstance(raw_model, str) or not raw_model.strip():
        return None

    raw_tokens = item_mapping.get("tokens")
    try:
        token_total = int(cast(int | float | str, raw_tokens or 0))
    except (TypeError, ValueError):
        token_total = 0

    model_name = simplify_model_name(tool, raw_model)
    if model_name == "-":
        return None
    return model_name, token_total


def extract_model_token_breakdown(
    tool: str,
    raw_model_tokens: object,
    raw_models: object = None,
    total_tokens: int = 0,
) -> list[tuple[str, int]]:
    items: object
    if isinstance(raw_model_tokens, str) and raw_model_tokens.strip():
        try:
            items = json.loads(raw_model_tokens)
        except json.JSONDecodeError:
            items = []
    else:
        items = raw_model_tokens

    model_tokens: dict[str, int] = {}
    if isinstance(items, list):
        for item in items:
            coerced = _coerce_model_token_item(tool, item)
            if not coerced:
                continue
            model_name, token_total = coerced
            model_tokens[model_name] = model_tokens.get(model_name, 0) + token_total

    if model_tokens:
        return sorted(model_tokens.items(), key=lambda item: (-item[1], item[0].lower()))

    models = extract_model_names(tool, raw_models)
    if len(models) == 1 and total_tokens:
        return [(models[0], total_tokens)]
    return [(model_name, 0) for model_name in models]


def summarize_model_tokens(
    tool: str,
    raw_model_tokens: object,
    raw_models: object,
    total_tokens: int,
) -> str:
    model_tokens = extract_model_token_breakdown(
        tool,
        raw_model_tokens,
        raw_models,
        total_tokens,
    )
    if not model_tokens:
        return "-"
    return ", ".join(
        f"{model_name} ({format_token_count(token_total)})" if token_total else model_name
        for model_name, token_total in model_tokens
    )


def empty_tool_day(tool: str, usage_day: date) -> dict[str, object]:
    return {
        "tool": tool,
        "usage_day": usage_day,
        "total_tokens_used": 0,
        "sessions": 0,
        "first_event_time": None,
        "last_event_time": None,
        "models": "-",
        "model_tokens": "[]",
    }


def has_tool_usage_last_week(records: list[dict[str, object]], tool: str) -> bool:
    today = date.today()
    week_start = today - timedelta(days=USAGE_BREAKDOWN_DAYS - 1)
    for record in records:
        if record.get("tool") != tool:
            continue
        usage_day = coerce_date(record.get("usage_day"))
        if not usage_day or usage_day < week_start:
            continue
        token_total = int(cast(int, record.get("total_tokens_used") or 0))
        session_total = int(cast(int, record.get("sessions") or 0))
        if token_total or session_total:
            return True
    return False


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
                summarize_model_tokens(
                    tool,
                    record.get("model_tokens"),
                    record.get("models"),
                    token_total,
                ),
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
                for model_name, model_token_total in extract_model_token_breakdown(
                    tool,
                    record.get("model_tokens"),
                    record.get("models"),
                    token_total,
                ):
                    weekly_model_tokens[model_name] = (
                        weekly_model_tokens.get(model_name, 0) + model_token_total
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

    profile = state.get("profile")
    ensure_databricks_auth(workspace, profile)
    with spinner("Retrieving Databricks access token..."):
        token = get_databricks_token(workspace, profile)

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
    configured_tools = configured_usage_tools(state, tool_displays)
    configured_tool_displays = {tool: tool_displays[tool] for tool in configured_tools}
    records = filter_records_for_tools(records, configured_tools)

    console.print(render_usage_summary(records, requester_name, configured_tool_displays))

    table_headers = ["Date", "Day", "Tokens", "Sessions", "Duration", "Models"]
    table_widths = [8, 5, 10, 8, 8, 24]

    if not configured_tools:
        print_note("No coding agents configured. Run `ucode configure` to set up agents.")
        return 0

    for tool in configured_tools:
        display = tool_displays[tool]
        print_heading(f"{display} · Last {USAGE_BREAKDOWN_DAYS} Days")
        if not has_tool_usage_last_week(records, tool):
            print_note(f"No usage for {display} in the last {USAGE_BREAKDOWN_DAYS} days.")
            continue
        console.print(
            render_box_table(
                table_headers,
                build_tool_breakdown_rows(records, tool),
                max_widths=table_widths,
            )
        )
    return 0
