"""Usage hook adapters for coding-agent CLIs."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from ucode.usage import (
    LOCAL_USAGE_DB_PATH,
    ensure_local_usage_sync_started_at,
    format_local_budget_hook_status,
    local_budget_status,
    record_local_usage_snapshot,
)

CODEX_STATE_DB_PATH = Path.home() / ".codex" / "state_5.sqlite"
CODEX_SYNC_STARTED_AT_KEY = "codex_sync_started_at"
OPENCODE_STATE_DB_PATH = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
OPENCODE_SYNC_STARTED_AT_KEY = "opencode_sync_started_at"


def _coerce_int(value: object) -> int:
    try:
        return max(int(cast(int | float | str, value or 0)), 0)
    except (TypeError, ValueError):
        return 0


def _read_json_stdin() -> dict[str, object]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _usage_from_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    mapping = cast(Mapping[str, object], value)
    return {
        "input_tokens": _coerce_int(mapping.get("input_tokens")),
        "output_tokens": _coerce_int(mapping.get("output_tokens")),
        "cache_read_input_tokens": _coerce_int(mapping.get("cache_read_input_tokens")),
        "cache_creation_input_tokens": _coerce_int(mapping.get("cache_creation_input_tokens")),
        "total_tokens": _coerce_int(mapping.get("total_tokens")),
    }


def _merge_usage(total: dict[str, int], item: dict[str, int]) -> None:
    for key, value in item.items():
        total[key] = total.get(key, 0) + value


def claude_transcript_usage(transcript_path: Path) -> dict[str, int]:
    """Sum assistant-message usage from a Claude Code JSONL transcript."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "total_tokens": 0,
    }
    if not transcript_path.exists():
        return totals
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return totals

    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        usage = _usage_from_mapping(message.get("usage"))
        if not any(usage.values()):
            continue
        _merge_usage(totals, usage)

    if not totals["total_tokens"]:
        totals["total_tokens"] = (
            totals["input_tokens"]
            + totals["output_tokens"]
            + totals["cache_read_input_tokens"]
            + totals["cache_creation_input_tokens"]
        )
    return totals


def _codex_token_usage_from_mapping(value: object) -> dict[str, int]:
    usage = _usage_from_mapping(value)
    if not isinstance(value, Mapping):
        return usage
    mapping = cast(Mapping[str, object], value)
    cached_input_tokens = _coerce_int(mapping.get("cached_input_tokens"))
    if cached_input_tokens:
        usage["input_tokens"] = max(usage["input_tokens"] - cached_input_tokens, 0)
        usage["cache_read_input_tokens"] = cached_input_tokens
    if not usage["total_tokens"]:
        usage["total_tokens"] = (
            usage["input_tokens"] + usage["cache_read_input_tokens"] + usage["output_tokens"]
        )
    return usage


def _opencode_model_name(value: object) -> str:
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        value = parsed
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        model_id = mapping.get("id") or mapping.get("modelID") or mapping.get("model")
        provider_id = mapping.get("providerID") or mapping.get("provider")
        if isinstance(model_id, str) and model_id:
            if isinstance(provider_id, str) and provider_id:
                return f"{provider_id}/{model_id}"
            return model_id
    return "opencode"


def _opencode_usage_from_row(row: sqlite3.Row) -> dict[str, int]:
    output_tokens = _coerce_int(row["tokens_output"]) + _coerce_int(row["tokens_reasoning"])
    return {
        "input_tokens": _coerce_int(row["tokens_input"]),
        "output_tokens": output_tokens,
        "cache_read_input_tokens": _coerce_int(row["tokens_cache_read"]),
        "cache_creation_input_tokens": _coerce_int(row["tokens_cache_write"]),
        "total_tokens": (
            _coerce_int(row["tokens_input"])
            + output_tokens
            + _coerce_int(row["tokens_cache_read"])
            + _coerce_int(row["tokens_cache_write"])
        ),
    }


def _opencode_usage_from_step_tokens(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    mapping = cast(Mapping[str, object], value)
    cache = mapping.get("cache")
    cache_mapping = cast(Mapping[str, object], cache) if isinstance(cache, Mapping) else {}
    input_tokens = _coerce_int(mapping.get("input"))
    output_tokens = _coerce_int(mapping.get("output")) + _coerce_int(mapping.get("reasoning"))
    cache_read_input_tokens = _coerce_int(cache_mapping.get("read"))
    cache_creation_input_tokens = _coerce_int(cache_mapping.get("write"))
    total_tokens = (
        input_tokens + output_tokens + cache_read_input_tokens + cache_creation_input_tokens
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "total_tokens": total_tokens,
    }


def _opencode_message_usage(value: object) -> tuple[str, dict[str, int]] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Mapping):
        return None
    data = cast(Mapping[str, object], value)
    if data.get("role") != "assistant":
        return None
    usage = _opencode_usage_from_step_tokens(data.get("tokens"))
    if not any(usage.values()):
        return None
    return _opencode_model_name(data), usage


def codex_session_usage(session_path: Path) -> dict[str, int]:
    """Read the latest cumulative token snapshot from a Codex JSONL session."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "total_tokens": 0,
    }
    if not session_path.exists():
        return totals
    try:
        lines = session_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return totals

    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or entry.get("type") != "event_msg":
            continue
        payload = entry.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue
        info = payload.get("info")
        if not isinstance(info, dict):
            continue
        usage = _codex_token_usage_from_mapping(info.get("total_token_usage"))
        if any(usage.values()):
            totals = usage
    return totals


def _codex_session_path_from_state(session_id: str) -> Path | None:
    if not session_id or not CODEX_STATE_DB_PATH.exists():
        return None
    try:
        with sqlite3.connect(CODEX_STATE_DB_PATH, timeout=5) as conn:
            row = conn.execute(
                "SELECT rollout_path FROM threads WHERE id = ?",
                (session_id,),
            ).fetchone()
    except sqlite3.Error:
        return None
    if not row or not row[0]:
        return None
    return Path(str(row[0]))


def _codex_session_path_from_payload(payload: dict[str, object], session_id: str) -> Path | None:
    for key in ("transcript_path", "session_path", "rollout_path", "path"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return Path(value)
    return _codex_session_path_from_state(session_id)


def sync_codex_usage_from_state(
    *,
    workspace: str | None = None,
    session_id: str | None = None,
    state_db_path: Path = CODEX_STATE_DB_PATH,
    usage_db_path: Path = LOCAL_USAGE_DB_PATH,
    updated_since: datetime | None = None,
) -> int:
    """Import recent Codex thread token snapshots from Codex's local state DB.

    When ``session_id`` is given (the live prompt-submit path), sync just that
    thread unconditionally: skip the time watermark and the ``tokens_used > 0``
    gate, since Codex writes ``tokens_used`` lazily and a just-finished session
    often still reads 0 there even though its rollout file already has the real
    counts. The bulk path (no ``session_id``) keeps both gates to avoid dumping
    historical sessions into today's ledger. Either way ``record_local_usage_snapshot``
    records only per-session deltas, so re-syncing the same thread is idempotent.
    """
    if not state_db_path.exists():
        return 0
    provider_filter = (
        "AND (model_provider = 'ucode-databricks'"
        " OR model_provider = 'Databricks'"
        " OR model_provider LIKE '%databricks%')"
    )
    if session_id:
        where = f"WHERE id = ? {provider_filter}"
        params: tuple[object, ...] = (session_id,)
    else:
        since = updated_since or ensure_local_usage_sync_started_at(
            CODEX_SYNC_STARTED_AT_KEY,
            db_path=usage_db_path,
        )
        where = f"WHERE updated_at >= ? AND COALESCE(tokens_used, 0) > 0 {provider_filter}"
        params = (int(since.timestamp()),)
    try:
        with sqlite3.connect(state_db_path, timeout=5) as conn:
            rows = conn.execute(
                f"""
                SELECT id, rollout_path, tokens_used, model, model_provider
                FROM threads
                {where}
                """,
                params,
            ).fetchall()
    except sqlite3.Error:
        return 0

    imported = 0
    for session_id, rollout_path, tokens_used, model, _model_provider in rows:
        usage = codex_session_usage(Path(str(rollout_path))) if rollout_path else {}
        if not any(usage.values()):
            usage = _codex_token_usage_from_mapping({"total_tokens": tokens_used})
        if not any(usage.values()):
            continue
        event = record_local_usage_snapshot(
            session_id=str(session_id),
            tool="codex",
            model=str(model or "codex"),
            workspace=workspace,
            source="codex-state-sync",
            db_path=usage_db_path,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_read_input_tokens=usage["cache_read_input_tokens"],
            cache_creation_input_tokens=usage["cache_creation_input_tokens"],
            total_tokens=usage["total_tokens"],
        )
        if event is not None:
            imported += 1
    return imported


def sync_codex_usage_recent(
    *,
    workspace: str | None = None,
    within_seconds: int = 3600,
    state_db_path: Path = CODEX_STATE_DB_PATH,
    usage_db_path: Path = LOCAL_USAGE_DB_PATH,
    now: datetime | None = None,
) -> int:
    """Sync recently active Codex threads for live budget display.

    Codex 0.134's ``notify``/``Stop`` hooks don't fire reliably, and the
    ``UserPromptSubmit`` hook runs *before* a turn produces tokens, so spend
    only lands on the next prompt. Polling this from ``budget-status`` (which
    the ``--watch`` pane re-runs on an interval) keeps the displayed Codex
    spend current without depending on those hooks.

    Each recently-updated Databricks thread is synced session-scoped, which
    bypasses the ``tokens_used > 0`` gate so in-progress sessions count even
    before Codex lazily writes that column. The per-session snapshot ledger
    makes repeated polling idempotent. The recency window keeps stale history
    out of today's ledger.
    """
    if not state_db_path.exists():
        return 0
    current = now or datetime.now(UTC)
    cutoff = int(current.timestamp()) - max(within_seconds, 0)
    try:
        with sqlite3.connect(state_db_path, timeout=5) as conn:
            rows = conn.execute(
                """
                SELECT id
                FROM threads
                WHERE updated_at >= ?
                  AND (
                    model_provider = 'ucode-databricks'
                    OR model_provider = 'Databricks'
                    OR model_provider LIKE '%databricks%'
                  )
                """,
                (cutoff,),
            ).fetchall()
    except sqlite3.Error:
        return 0

    imported = 0
    for (session_id,) in rows:
        imported += sync_codex_usage_from_state(
            workspace=workspace,
            session_id=str(session_id),
            state_db_path=state_db_path,
            usage_db_path=usage_db_path,
        )
    return imported


def sync_opencode_usage_from_state(
    *,
    workspace: str | None = None,
    session_id: str | None = None,
    state_db_path: Path = OPENCODE_STATE_DB_PATH,
    usage_db_path: Path = LOCAL_USAGE_DB_PATH,
    updated_since: datetime | None = None,
) -> int:
    """Import recent OpenCode session token snapshots from its local state DB."""
    if not state_db_path.exists():
        return 0
    since = updated_since or ensure_local_usage_sync_started_at(
        OPENCODE_SYNC_STARTED_AT_KEY,
        db_path=usage_db_path,
    )
    since_ms = int(since.timestamp() * 1000)
    session_filter = "AND id = ?" if session_id else ""
    params: tuple[object, ...] = (since_ms, session_id) if session_id else (since_ms,)
    try:
        uri = f"file:{state_db_path}?mode=ro"
        with sqlite3.connect(uri, timeout=5, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT id, model, cost, tokens_input, tokens_output,
                       tokens_reasoning, tokens_cache_read, tokens_cache_write,
                       time_updated
                FROM session
                WHERE time_updated >= ?
                  {session_filter}
                  AND (
                    COALESCE(tokens_input, 0) > 0
                    OR COALESCE(tokens_output, 0) > 0
                    OR COALESCE(tokens_reasoning, 0) > 0
                    OR COALESCE(tokens_cache_read, 0) > 0
                    OR COALESCE(tokens_cache_write, 0) > 0
                  )
                """,
                params,
            ).fetchall()
    except sqlite3.Error:
        return 0

    imported = 0
    for row in rows:
        usage = _opencode_usage_from_row(row)
        if not any(usage.values()):
            continue
        event = record_local_usage_snapshot(
            session_id=str(row["id"]),
            tool="opencode",
            model=_opencode_model_name(row["model"]),
            workspace=workspace,
            source="opencode-state-sync",
            db_path=usage_db_path,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_read_input_tokens=usage["cache_read_input_tokens"],
            cache_creation_input_tokens=usage["cache_creation_input_tokens"],
            total_tokens=usage["total_tokens"],
        )
        if event is not None:
            imported += 1
    return imported


def sync_opencode_usage_from_messages(
    *,
    workspace: str | None = None,
    session_id: str | None = None,
    state_db_path: Path = OPENCODE_STATE_DB_PATH,
    usage_db_path: Path = LOCAL_USAGE_DB_PATH,
    updated_since: datetime | None = None,
) -> int:
    """Import OpenCode usage from assistant message token payloads.

    OpenCode updates the session aggregate lazily, but assistant messages carry
    token payloads as soon as they are persisted. Build cumulative per-session
    snapshots from those message rows and let the normal snapshot ledger record
    only new deltas.
    """
    if not state_db_path.exists():
        return 0
    since = updated_since or ensure_local_usage_sync_started_at(
        OPENCODE_SYNC_STARTED_AT_KEY,
        db_path=usage_db_path,
    )
    since_ms = int(since.timestamp() * 1000)
    try:
        uri = f"file:{state_db_path}?mode=ro"
        with sqlite3.connect(uri, timeout=5, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            if session_id:
                session_ids = [session_id]
            else:
                session_rows = conn.execute(
                    """
                    SELECT DISTINCT session_id
                    FROM message
                    WHERE time_updated >= ?
                    """,
                    (since_ms,),
                ).fetchall()
                session_ids = [str(row["session_id"]) for row in session_rows if row["session_id"]]
            if not session_ids:
                return 0

            placeholders = ",".join("?" for _ in session_ids)
            rows = conn.execute(
                f"""
                SELECT session_id, data
                FROM message
                WHERE session_id IN ({placeholders})
                """,
                tuple(session_ids),
            ).fetchall()
    except sqlite3.Error:
        return 0

    totals: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        parsed = _opencode_message_usage(row["data"])
        if parsed is None:
            continue
        model, usage = parsed
        key = (str(row["session_id"]), model)
        session_total = totals.setdefault(
            key,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "total_tokens": 0,
            },
        )
        _merge_usage(session_total, usage)

    imported = 0
    for (session_id, model), usage in totals.items():
        event = record_local_usage_snapshot(
            session_id=session_id,
            tool="opencode",
            model=model,
            workspace=workspace,
            source="opencode-message-sync",
            db_path=usage_db_path,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_read_input_tokens=usage["cache_read_input_tokens"],
            cache_creation_input_tokens=usage["cache_creation_input_tokens"],
            total_tokens=usage["total_tokens"],
        )
        if event is not None:
            imported += 1
    return imported


def _codex_budget_message(status: dict[str, object]) -> str:
    """Format the budget message for Codex's single-line, plain-text rendering.

    Codex renders the hook message on one line with no markdown: it collapses
    the headline/detail newline and the variation-selector ``⚠️`` glyph crowds
    the word after it (mirrors the two-space workaround at ``usage.py``'s rich
    headline). Widen the emoji gap and turn the line break into a visible gap.
    """
    message = format_local_budget_hook_status(status)
    return message.replace("⚠️ ", "⚠️  ").replace("⛔ ", "⛔  ").replace("\n", "  ")


def _hook_response_for_budget(
    status: dict[str, object],
    *,
    can_block: bool = False,
    quiet_warn: bool = False,
    message: str | None = None,
) -> dict[str, object]:
    state = str(status.get("state") or "ok")
    behavior = status.get("on_budget_exhausted") or "block"
    if message is None:
        message = format_local_budget_hook_status(status)
    if state in {"warn", "exceeded"} and (state == "warn" or behavior == "warn"):
        if quiet_warn:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": message,
                }
            }
        return {"systemMessage": message}
    if state == "exceeded":
        if behavior == "allow":
            return {}
        if can_block:
            return {"decision": "block", "reason": message}
        return {"continue": False, "stopReason": message}
    return {}


def claude_usage_hook(
    *,
    model: str,
    event: str,
    workspace: str | None = None,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = payload if payload is not None else _read_json_stdin()
    session_id = str(payload.get("session_id") or payload.get("sessionId") or "claude")
    transcript = payload.get("transcript_path")
    # Both the recording event (post-tool) and the enforcement event
    # (prompt-submit) carry a transcript path, so record spend whenever one is
    # present to keep the shared daily budget current.
    if isinstance(transcript, str) and transcript:
        usage = claude_transcript_usage(Path(transcript))
        record_local_usage_snapshot(
            session_id=session_id,
            tool="claude",
            model=model,
            workspace=workspace,
            source="claude-hook",
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_read_input_tokens=usage["cache_read_input_tokens"],
            cache_creation_input_tokens=usage["cache_creation_input_tokens"],
            total_tokens=usage["total_tokens"],
        )
    # Enforce on prompt submission (mirrors Codex): block the next prompt once
    # the global daily budget is exceeded, and warn otherwise. Claude renders a
    # `systemMessage` visibly to the user (unlike `additionalContext`, which is
    # injected silently into the model context), so leave quiet_warn off here to
    # surface the warning the same way Codex does.
    if event in {"prompt-submit", "user-prompt-submit"}:
        return _hook_response_for_budget(local_budget_status("claude"), can_block=True)
    return _hook_response_for_budget(local_budget_status("claude"))


def codex_usage_hook(
    *,
    model: str,
    event: str = "notify",
    workspace: str | None = None,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = payload if payload is not None else _read_json_stdin()
    session_id = str(payload.get("session_id") or payload.get("sessionId") or "codex")
    session_path = _codex_session_path_from_payload(payload, session_id)
    usage = codex_session_usage(session_path) if session_path is not None else {}
    if not any(usage.values()):
        usage = _codex_token_usage_from_mapping(payload.get("usage"))
    if not any(usage.values()):
        usage = _codex_token_usage_from_mapping(payload)
    if any(usage.values()):
        record_local_usage_snapshot(
            session_id=session_id,
            tool="codex",
            model=model,
            workspace=workspace,
            source="codex-hook",
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_read_input_tokens=usage["cache_read_input_tokens"],
            cache_creation_input_tokens=usage["cache_creation_input_tokens"],
            total_tokens=usage["total_tokens"],
        )
    if event in {"prompt-submit", "user-prompt-submit"}:
        sync_codex_usage_from_state(workspace=workspace, session_id=session_id)
        status = local_budget_status("codex")
        return _hook_response_for_budget(
            status,
            can_block=True,
            quiet_warn=True,
            message=_codex_budget_message(status),
        )
    if event == "stop":
        # Fires at turn end so spend from the turn that just finished surfaces
        # immediately. Warn only — blocking belongs to the prompt-submit hook,
        # and the turn is already over, so never return continue:False here.
        sync_codex_usage_from_state(workspace=workspace, session_id=session_id)
        status = local_budget_status("codex")
        if str(status.get("state") or "ok") in {"warn", "exceeded"}:
            return {"systemMessage": _codex_budget_message(status)}
        return {}
    # The notify callback only records spend — it must not surface the budget
    # warning, or the message would appear twice (once here, once from the
    # UserPromptSubmit hook). Enforcement/warning is the prompt-submit hook's job.
    return {}
