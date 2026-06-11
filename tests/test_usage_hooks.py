"""Tests for local usage hook adapters."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from ucode import usage_hooks


def test_claude_transcript_usage_sums_assistant_usage(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": "hi"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "usage": {
                                "input_tokens": 100,
                                "output_tokens": 20,
                                "cache_read_input_tokens": 5,
                                "cache_creation_input_tokens": 2,
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"usage": {"input_tokens": 50, "output_tokens": 10}},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    assert usage_hooks.claude_transcript_usage(transcript) == {
        "input_tokens": 150,
        "output_tokens": 30,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 2,
        "total_tokens": 187,
    }


def test_claude_post_tool_records_transcript_usage(tmp_path, monkeypatch):
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"message": {"usage": {"input_tokens": 100, "output_tokens": 20}}}),
        encoding="utf-8",
    )
    recorded: list[dict] = []
    monkeypatch.setattr(
        usage_hooks,
        "record_local_usage_snapshot",
        lambda **kwargs: recorded.append(kwargs) or {"total_tokens": 120},
    )
    monkeypatch.setattr(
        usage_hooks,
        "local_budget_status",
        lambda tool: {
            "configured": True,
            "state": "ok",
            "tool": tool,
            "spend_usd": 0.1,
            "limit_usd": 20.0,
            "days": 1,
            "total_tokens": 120,
        },
    )

    response = usage_hooks.claude_usage_hook(
        model="databricks-claude-sonnet-4",
        event="post-tool",
        payload={
            "session_id": "s1",
            "transcript_path": str(transcript),
        },
    )

    assert recorded[0]["session_id"] == "s1"
    assert recorded[0]["tool"] == "claude"
    assert recorded[0]["input_tokens"] == 100
    assert recorded[0]["output_tokens"] == 20
    assert response == {}


def test_claude_prompt_submit_blocks_when_budget_exceeded(monkeypatch):
    monkeypatch.setattr(
        usage_hooks,
        "local_budget_status",
        lambda tool: {
            "configured": True,
            "state": "exceeded",
            "tool": tool,
            "spend_usd": 21.0,
            "limit_usd": 20.0,
            "days": 1,
            "total_tokens": 120,
        },
    )

    response = usage_hooks.claude_usage_hook(
        model="databricks-claude-sonnet-4",
        event="prompt-submit",
        payload={"session_id": "s1"},
    )

    assert response["decision"] == "block"
    assert "⛔ Daily budget — limit exceeded" in response["reason"]
    assert "$21.00 / $20.00 (105%)" in response["reason"]
    assert "further tool use blocked today" in response["reason"]
    assert "Tokens used today" not in response["reason"]
    assert "Window:" not in response["reason"]


def test_claude_prompt_submit_warns_when_budget_exceeded_policy_warn(monkeypatch):
    monkeypatch.setattr(
        usage_hooks,
        "local_budget_status",
        lambda tool: {
            "configured": True,
            "state": "exceeded",
            "tool": tool,
            "spend_usd": 21.0,
            "limit_usd": 20.0,
            "days": 1,
            "remaining_usd": 0.0,
            "total_tokens": 120,
            "on_budget_exhausted": "warn",
        },
    )

    response = usage_hooks.claude_usage_hook(
        model="databricks-claude-sonnet-4",
        event="prompt-submit",
        payload={"session_id": "s1"},
    )

    assert "decision" not in response
    assert response["systemMessage"].startswith("⚠️ Daily budget — limit exceeded")
    assert "continuing because policy is warn" in response["systemMessage"]


def test_claude_prompt_submit_warns_visibly_when_nearing_budget(monkeypatch):
    # Claude renders `systemMessage` visibly to the user but injects
    # `additionalContext` silently, so the warn path must use systemMessage to
    # surface the budget warning the same way Codex does.
    monkeypatch.setattr(
        usage_hooks,
        "local_budget_status",
        lambda tool: {
            "configured": True,
            "state": "warn",
            "tool": tool,
            "spend_usd": 18.0,
            "limit_usd": 20.0,
            "days": 1,
            "remaining_usd": 2.0,
            "total_tokens": 120,
        },
    )

    response = usage_hooks.claude_usage_hook(
        model="databricks-claude-sonnet-4",
        event="prompt-submit",
        payload={"session_id": "s1"},
    )

    assert "additionalContext" not in response
    assert response["systemMessage"].startswith("⚠️ Daily budget — nearing limit")
    assert "$18.00 / $20.00 (90%)" in response["systemMessage"]
    assert "$2.00 left" in response["systemMessage"]


def test_codex_session_usage_reads_latest_token_count(tmp_path):
    session = tmp_path / "session.jsonl"
    session.write_text(
        "\n".join(
            [
                json.dumps({"type": "event_msg", "payload": {"type": "token_count", "info": None}}),
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 100,
                                    "cached_input_tokens": 30,
                                    "output_tokens": 20,
                                    "reasoning_output_tokens": 5,
                                    "total_tokens": 120,
                                }
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 250,
                                    "cached_input_tokens": 100,
                                    "output_tokens": 50,
                                    "reasoning_output_tokens": 10,
                                    "total_tokens": 300,
                                }
                            },
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    assert usage_hooks.codex_session_usage(session) == {
        "input_tokens": 150,
        "output_tokens": 50,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 0,
        "total_tokens": 300,
    }


def test_codex_notify_records_usage_payload_without_warning(monkeypatch):
    # The notify callback only records spend; it must return an empty response
    # so the budget warning is surfaced solely by the UserPromptSubmit hook
    # (otherwise the same message would appear twice).
    recorded: list[dict] = []
    monkeypatch.setattr(
        usage_hooks,
        "record_local_usage_snapshot",
        lambda **kwargs: recorded.append(kwargs) or {"total_tokens": 120},
    )
    monkeypatch.setattr(
        usage_hooks,
        "local_budget_status",
        lambda tool: {
            "configured": True,
            "state": "warn",
            "tool": tool,
            "spend_usd": 0.9,
            "limit_usd": 1.0,
            "days": 1,
            "remaining_usd": 0.1,
            "total_tokens": 120,
        },
    )

    response = usage_hooks.codex_usage_hook(
        model="databricks-gpt-5",
        payload={"session_id": "s2", "usage": {"input_tokens": 100, "output_tokens": 20}},
    )

    assert recorded[0]["session_id"] == "s2"
    assert recorded[0]["tool"] == "codex"
    assert recorded[0]["input_tokens"] == 100
    # Records spend but stays silent — no warning from the notify path.
    assert response == {}


def test_codex_prompt_submit_blocks_with_valid_prompt_json(monkeypatch):
    monkeypatch.setattr(usage_hooks, "sync_codex_usage_from_state", lambda **kwargs: 0)
    monkeypatch.setattr(
        usage_hooks,
        "local_budget_status",
        lambda tool: {
            "configured": True,
            "state": "exceeded",
            "tool": tool,
            "spend_usd": 2.0,
            "limit_usd": 1.0,
            "days": 1,
            "remaining_usd": 0.0,
            "total_tokens": 120,
        },
    )

    response = usage_hooks.codex_usage_hook(
        model="gpt-5.5",
        event="prompt-submit",
        payload={"session_id": "s2"},
    )

    assert response["decision"] == "block"
    assert "⛔ Daily budget — limit exceeded" in response["reason"]
    assert "further tool use blocked today" in response["reason"]
    assert "Tokens used today" not in response["reason"]
    assert "Window:" not in response["reason"]


def test_codex_prompt_submit_warns_when_budget_exceeded_policy_warn(monkeypatch):
    monkeypatch.setattr(usage_hooks, "sync_codex_usage_from_state", lambda **kwargs: 0)
    monkeypatch.setattr(
        usage_hooks,
        "local_budget_status",
        lambda tool: {
            "configured": True,
            "state": "exceeded",
            "tool": tool,
            "spend_usd": 2.0,
            "limit_usd": 1.0,
            "days": 1,
            "remaining_usd": 0.0,
            "total_tokens": 120,
            "on_budget_exhausted": "warn",
        },
    )

    response = usage_hooks.codex_usage_hook(
        model="gpt-5.5",
        event="prompt-submit",
        payload={"session_id": "s2"},
    )

    assert response == {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                "⚠️ Daily budget — limit exceeded\n"
                "$2.00 / $1.00 (200%)  ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰  "
                "continuing because policy is warn"
            ),
        }
    }


def test_codex_prompt_submit_allows_when_budget_exceeded_policy_allow(monkeypatch):
    monkeypatch.setattr(usage_hooks, "sync_codex_usage_from_state", lambda **kwargs: 0)
    monkeypatch.setattr(
        usage_hooks,
        "local_budget_status",
        lambda tool: {
            "configured": True,
            "state": "exceeded",
            "tool": tool,
            "spend_usd": 2.0,
            "limit_usd": 1.0,
            "days": 1,
            "remaining_usd": 0.0,
            "total_tokens": 120,
            "on_budget_exhausted": "allow",
        },
    )

    response = usage_hooks.codex_usage_hook(
        model="gpt-5.5",
        event="prompt-submit",
        payload={"session_id": "s2"},
    )

    assert response == {}


def test_codex_prompt_submit_returns_warning_additional_context(monkeypatch):
    monkeypatch.setattr(usage_hooks, "sync_codex_usage_from_state", lambda **kwargs: 0)
    monkeypatch.setattr(
        usage_hooks,
        "local_budget_status",
        lambda tool: {
            "configured": True,
            "state": "warn",
            "tool": tool,
            "spend_usd": 0.9,
            "limit_usd": 1.0,
            "days": 1,
            "remaining_usd": 0.1,
            "total_tokens": 120,
        },
    )

    response = usage_hooks.codex_usage_hook(
        model="gpt-5.5",
        event="prompt-submit",
        payload={"session_id": "s2"},
    )

    assert response == {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                "⚠️ Daily budget — nearing limit\n"
                "$0.90 / $1.00 (90%)  ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱  $0.10 left"
            ),
        }
    }


def test_codex_stop_warns_visibly_at_turn_end(monkeypatch):
    # The Stop hook fires at turn end and must surface the warning visibly
    # (systemMessage), not silently inject it like the prompt-submit hook does.
    synced = []
    monkeypatch.setattr(
        usage_hooks,
        "sync_codex_usage_from_state",
        lambda **kwargs: synced.append(kwargs) or 0,
    )
    monkeypatch.setattr(
        usage_hooks,
        "local_budget_status",
        lambda tool: {
            "configured": True,
            "state": "warn",
            "tool": tool,
            "spend_usd": 0.9,
            "limit_usd": 1.0,
            "days": 1,
            "remaining_usd": 0.1,
            "total_tokens": 120,
        },
    )

    response = usage_hooks.codex_usage_hook(
        model="gpt-5.5",
        event="stop",
        payload={"session_id": "s2"},
    )

    assert synced and synced[0]["session_id"] == "s2"
    assert response["systemMessage"].startswith("⚠️ Daily budget — nearing limit")


def test_codex_stop_does_not_block_when_budget_exceeded(monkeypatch):
    # Blocking belongs to prompt-submit; the turn is already over, so Stop only
    # warns even when the budget is exceeded under a block policy.
    monkeypatch.setattr(usage_hooks, "sync_codex_usage_from_state", lambda **kwargs: 0)
    monkeypatch.setattr(
        usage_hooks,
        "local_budget_status",
        lambda tool: {
            "configured": True,
            "state": "exceeded",
            "tool": tool,
            "spend_usd": 2.0,
            "limit_usd": 1.0,
            "days": 1,
            "remaining_usd": 0.0,
            "total_tokens": 120,
            "on_budget_exhausted": "block",
        },
    )

    response = usage_hooks.codex_usage_hook(
        model="gpt-5.5",
        event="stop",
        payload={"session_id": "s2"},
    )

    assert "decision" not in response
    assert response.get("continue") is not False
    assert "Daily budget — limit exceeded" in response["systemMessage"]


def test_codex_hook_records_session_snapshot(tmp_path, monkeypatch):
    session = tmp_path / "session.jsonl"
    session.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 40,
                            "output_tokens": 20,
                            "total_tokens": 120,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    recorded: list[dict] = []
    monkeypatch.setattr(
        usage_hooks,
        "record_local_usage_snapshot",
        lambda **kwargs: recorded.append(kwargs) or {"total_tokens": 120},
    )
    monkeypatch.setattr(
        usage_hooks,
        "local_budget_status",
        lambda tool: {
            "configured": True,
            "state": "ok",
            "tool": tool,
            "spend_usd": 0.1,
            "limit_usd": 20.0,
            "days": 1,
            "total_tokens": 120,
        },
    )

    response = usage_hooks.codex_usage_hook(
        model="gpt-5.5",
        payload={"session_id": "s3", "session_path": str(session)},
    )

    assert recorded[0]["session_id"] == "s3"
    assert recorded[0]["tool"] == "codex"
    assert recorded[0]["model"] == "gpt-5.5"
    assert recorded[0]["input_tokens"] == 60
    assert recorded[0]["cache_read_input_tokens"] == 40
    assert recorded[0]["output_tokens"] == 20
    assert recorded[0]["total_tokens"] == 120
    assert response == {}


def test_sync_codex_usage_from_state_imports_recent_threads(tmp_path, monkeypatch):
    state_db = tmp_path / "state_5.sqlite"
    session = tmp_path / "session.jsonl"
    session.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 25,
                            "output_tokens": 10,
                            "total_tokens": 110,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    with sqlite3.connect(state_db) as conn:
        conn.execute(
            """
            CREATE TABLE threads (
              id TEXT,
              rollout_path TEXT,
              tokens_used INTEGER,
              model TEXT,
              model_provider TEXT,
              updated_at INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?)",
            (
                "thread-1",
                str(session),
                110,
                "gpt-5.5",
                "ucode-databricks",
                int(datetime.now(UTC).timestamp()),
            ),
        )
    recorded: list[dict] = []
    monkeypatch.setattr(
        usage_hooks,
        "record_local_usage_snapshot",
        lambda **kwargs: recorded.append(kwargs) or {"total_tokens": 110},
    )

    imported = usage_hooks.sync_codex_usage_from_state(
        workspace="https://example.com",
        state_db_path=state_db,
        usage_db_path=tmp_path / "usage.sqlite",
        updated_since=datetime.fromtimestamp(0, UTC),
    )

    assert imported == 1
    assert recorded[0]["session_id"] == "thread-1"
    assert recorded[0]["tool"] == "codex"
    assert recorded[0]["model"] == "gpt-5.5"
    assert recorded[0]["input_tokens"] == 75
    assert recorded[0]["cache_read_input_tokens"] == 25
    assert recorded[0]["output_tokens"] == 10
    assert recorded[0]["source"] == "codex-state-sync"


def test_sync_codex_usage_from_state_filters_to_session_id(tmp_path, monkeypatch):
    state_db = tmp_path / "state_5.sqlite"
    current_session = tmp_path / "current.jsonl"
    unrelated_session = tmp_path / "unrelated.jsonl"
    for path, total_tokens in ((current_session, 110), (unrelated_session, 1_000_000)):
        path.write_text(
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": total_tokens,
                                "output_tokens": 0,
                                "total_tokens": total_tokens,
                            }
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
    with sqlite3.connect(state_db) as conn:
        conn.execute(
            """
            CREATE TABLE threads (
              id TEXT,
              rollout_path TEXT,
              tokens_used INTEGER,
              model TEXT,
              model_provider TEXT,
              updated_at INTEGER
            )
            """
        )
        now = int(datetime.now(UTC).timestamp())
        conn.executemany(
            "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("current-thread", str(current_session), 110, "gpt-5.5", "ucode-databricks", now),
                (
                    "unrelated-thread",
                    str(unrelated_session),
                    1_000_000,
                    "gpt-5.5",
                    "ucode-databricks",
                    now,
                ),
            ],
        )
    recorded: list[dict] = []
    monkeypatch.setattr(
        usage_hooks,
        "record_local_usage_snapshot",
        lambda **kwargs: recorded.append(kwargs) or {"total_tokens": kwargs["total_tokens"]},
    )

    imported = usage_hooks.sync_codex_usage_from_state(
        state_db_path=state_db,
        usage_db_path=tmp_path / "usage.sqlite",
        session_id="current-thread",
        updated_since=datetime.fromtimestamp(0, UTC),
    )

    assert imported == 1
    assert [event["session_id"] for event in recorded] == ["current-thread"]
    assert recorded[0]["total_tokens"] == 110


def test_sync_codex_usage_from_state_ignores_old_threads_for_fresh_ledger(tmp_path, monkeypatch):
    state_db = tmp_path / "state_5.sqlite"
    session = tmp_path / "session.jsonl"
    session.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "total_tokens": 110,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    with sqlite3.connect(state_db) as conn:
        conn.execute(
            """
            CREATE TABLE threads (
              id TEXT,
              rollout_path TEXT,
              tokens_used INTEGER,
              model TEXT,
              model_provider TEXT,
              updated_at INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?)",
            ("old-thread", str(session), 110, "gpt-5.5", "ucode-databricks", 1),
        )
    recorded: list[dict] = []
    monkeypatch.setattr(
        usage_hooks,
        "record_local_usage_snapshot",
        lambda **kwargs: recorded.append(kwargs) or {"total_tokens": 110},
    )

    imported = usage_hooks.sync_codex_usage_from_state(
        state_db_path=state_db,
        usage_db_path=tmp_path / "usage.sqlite",
    )

    assert imported == 0
    assert recorded == []


def test_sync_codex_usage_recent_syncs_in_progress_thread(tmp_path, monkeypatch):
    """A recent thread syncs from its rollout even when tokens_used is still 0.

    Codex writes ``tokens_used`` lazily, so live budget display must read the
    rollout file directly for in-progress sessions.
    """
    state_db = tmp_path / "state_5.sqlite"
    session = tmp_path / "session.jsonl"
    session.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "total_tokens": 110,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    now = datetime.now(UTC)
    with sqlite3.connect(state_db) as conn:
        conn.execute(
            """
            CREATE TABLE threads (
              id TEXT,
              rollout_path TEXT,
              tokens_used INTEGER,
              model TEXT,
              model_provider TEXT,
              updated_at INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?)",
            ("recent", str(session), 0, "gpt-5.5", "ucode-databricks", int(now.timestamp())),
        )
        conn.execute(
            "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?)",
            ("stale", str(session), 110, "gpt-5.5", "ucode-databricks", 1),
        )
    recorded: list[dict] = []
    monkeypatch.setattr(
        usage_hooks,
        "record_local_usage_snapshot",
        lambda **kwargs: recorded.append(kwargs) or {"total_tokens": 110},
    )

    imported = usage_hooks.sync_codex_usage_recent(
        workspace="https://example.com",
        within_seconds=3600,
        state_db_path=state_db,
        usage_db_path=tmp_path / "usage.sqlite",
        now=now,
    )

    assert imported == 1
    assert [r["session_id"] for r in recorded] == ["recent"]
    assert recorded[0]["total_tokens"] == 110


def test_sync_opencode_usage_from_state_imports_recent_sessions(tmp_path, monkeypatch):
    state_db = tmp_path / "opencode.db"
    with sqlite3.connect(state_db) as conn:
        conn.execute(
            """
            CREATE TABLE session (
              id TEXT,
              model TEXT,
              cost REAL,
              tokens_input INTEGER,
              tokens_output INTEGER,
              tokens_reasoning INTEGER,
              tokens_cache_read INTEGER,
              tokens_cache_write INTEGER,
              time_updated INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ses_1",
                json.dumps(
                    {"id": "databricks-claude-haiku-4-5", "providerID": "databricks-anthropic"}
                ),
                0.0,
                100,
                20,
                5,
                30,
                40,
                int(datetime.now(UTC).timestamp() * 1000),
            ),
        )
    recorded: list[dict] = []
    monkeypatch.setattr(
        usage_hooks,
        "record_local_usage_snapshot",
        lambda **kwargs: recorded.append(kwargs) or {"total_tokens": kwargs["total_tokens"]},
    )

    imported = usage_hooks.sync_opencode_usage_from_state(
        workspace="https://example.com",
        state_db_path=state_db,
        usage_db_path=tmp_path / "usage.sqlite",
        updated_since=datetime.fromtimestamp(0, UTC),
    )

    assert imported == 1
    assert recorded[0]["session_id"] == "ses_1"
    assert recorded[0]["tool"] == "opencode"
    assert recorded[0]["model"] == "databricks-anthropic/databricks-claude-haiku-4-5"
    assert recorded[0]["input_tokens"] == 100
    assert recorded[0]["output_tokens"] == 25
    assert recorded[0]["cache_read_input_tokens"] == 30
    assert recorded[0]["cache_creation_input_tokens"] == 40
    assert recorded[0]["total_tokens"] == 195
    assert recorded[0]["source"] == "opencode-state-sync"


def test_sync_opencode_usage_from_messages_imports_assistant_tokens(tmp_path, monkeypatch):
    state_db = tmp_path / "opencode.db"
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    with sqlite3.connect(state_db) as conn:
        conn.execute(
            """
            CREATE TABLE message (
              id TEXT,
              session_id TEXT,
              time_created INTEGER,
              time_updated INTEGER,
              data TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
            (
                "msg_user",
                "ses_1",
                now_ms,
                now_ms,
                json.dumps({"role": "user", "tokens": {"total": 999}}),
            ),
        )
        conn.execute(
            "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
            (
                "msg_assistant_1",
                "ses_1",
                now_ms,
                now_ms,
                json.dumps(
                    {
                        "role": "assistant",
                        "modelID": "databricks-claude-opus-4-8",
                        "providerID": "databricks-anthropic",
                        "tokens": {
                            "total": 100,
                            "input": 2,
                            "output": 8,
                            "reasoning": 1,
                            "cache": {"read": 80, "write": 9},
                        },
                    }
                ),
            ),
        )
        conn.execute(
            "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
            (
                "msg_assistant_2",
                "ses_1",
                now_ms,
                now_ms,
                json.dumps(
                    {
                        "role": "assistant",
                        "modelID": "databricks-claude-opus-4-8",
                        "providerID": "databricks-anthropic",
                        "tokens": {
                            "total": 50,
                            "input": 3,
                            "output": 7,
                            "reasoning": 0,
                            "cache": {"read": 30, "write": 10},
                        },
                    }
                ),
            ),
        )
    recorded: list[dict] = []
    monkeypatch.setattr(
        usage_hooks,
        "record_local_usage_snapshot",
        lambda **kwargs: recorded.append(kwargs) or {"total_tokens": kwargs["total_tokens"]},
    )

    imported = usage_hooks.sync_opencode_usage_from_messages(
        workspace="https://example.com",
        state_db_path=state_db,
        usage_db_path=tmp_path / "usage.sqlite",
        updated_since=datetime.fromtimestamp(0, UTC),
    )

    assert imported == 1
    assert recorded[0]["session_id"] == "ses_1"
    assert recorded[0]["tool"] == "opencode"
    assert recorded[0]["model"] == "databricks-anthropic/databricks-claude-opus-4-8"
    assert recorded[0]["source"] == "opencode-message-sync"
    assert recorded[0]["input_tokens"] == 5
    assert recorded[0]["output_tokens"] == 16
    assert recorded[0]["cache_read_input_tokens"] == 110
    assert recorded[0]["cache_creation_input_tokens"] == 19
    assert recorded[0]["total_tokens"] == 150
