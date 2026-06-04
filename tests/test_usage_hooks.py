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
