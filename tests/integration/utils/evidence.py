"""Read real agent transcripts and ug routing records without changing them."""

import json
import uuid
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            # A running agent may not have finished its last write yet.
            if index == len(lines) - 1 and not line.endswith("\n"):
                break
            raise
        if isinstance(value, dict):
            records.append(value)
    return records


def agent_sessions(session, agent: str) -> dict[str, list[dict]]:
    directory = session.home / (".claude/projects" if agent == "claude" else ".codex/sessions")
    return {
        str(path.relative_to(directory)): read_jsonl(path) for path in directory.rglob("*.jsonl")
    }


def assistant_answers(agent: str, records: list[dict]) -> list[str]:
    answers = []
    for record in records:
        if agent == "claude" and record.get("type") == "assistant":
            message = record.get("message", {})
            if message.get("role") == "assistant":
                answers.extend(
                    part["text"]
                    for part in message.get("content", [])
                    if part.get("type") == "text" and isinstance(part.get("text"), str)
                )
        if agent == "codex" and record.get("type") == "event_msg":
            payload = record.get("payload", {})
            if payload.get("type") == "task_complete" and payload.get("last_agent_message"):
                answers.append(payload["last_agent_message"])
    return answers


def is_child_session(agent: str, path: str, records: list[dict]) -> bool:
    if agent == "claude":
        return "/subagents/" in path
    return any(
        record.get("type") == "session_meta"
        and isinstance(record.get("payload", {}).get("source"), dict)
        and "subagent" in record["payload"]["source"]
        for record in records
    )


class FileTask:
    """Ordinary project input; the expected answer is never included in the prompt."""

    def __init__(self, session):
        self.value = uuid.uuid4().hex
        self.filename = "input-" + uuid.uuid4().hex[:8] + ".txt"
        (session.cwd / self.filename).write_text(self.value + "\n")
        self.prompt = f"Read {self.filename} using a tool. Reply with only its contents."
        self.delegate_prompt = (
            f"Delegate this task to one subagent: read {self.filename} using a tool and return "
            "its contents. Do not read the file yourself. Wait for the subagent and reply "
            "with only the value it returned."
        )

    def completed(self, session, agent: str, *, child: bool = False) -> bool:
        for path, records in agent_sessions(session, agent).items():
            if is_child_session(agent, path, records) != child:
                continue
            if any(self.value in text for text in assistant_answers(agent, records)):
                return True
        return False

    def assert_completed(self, session, agent: str, *, child: bool = False) -> None:
        sessions = agent_sessions(session, agent)
        session.record("agent-sessions.json", sessions)
        assert self.completed(session, agent, child=child), (
            f"No {'child' if child else 'parent'} assistant answer contained the file's value; "
            "echoed prompts and tool results do not count as completed answers."
        )

    def assert_headless_answer(self, agent: str, result) -> None:
        """Read the real CLI's structured final answer, never its echoed input."""
        payloads = []
        for line in result.stdout.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue  # ug may print human-readable launch status before agent JSON.
            if isinstance(value, dict):
                payloads.append(value)
        if agent == "claude":
            final = [row for row in payloads if row.get("type") == "result"]
            assert final and not final[-1].get("is_error"), result.stdout
            assert self.value in final[-1].get("result", ""), result.stdout
        else:
            assert any(row.get("type") == "turn.completed" for row in payloads), result.stdout
            answers = [
                row.get("item", {}).get("text", "")
                for row in payloads
                if row.get("type") == "item.completed"
                and row.get("item", {}).get("type") == "agent_message"
            ]
            assert any(self.value in answer for answer in answers), result.stdout


def assert_subagent_routed(session, agent: str, task: FileTask) -> None:
    """Require a real gateway decision correlated with an actual child start."""
    root = session.home / ".ucode"
    decisions = read_jsonl(root / f"{agent}-smart-routing-decisions.jsonl")
    audit = read_jsonl(root / f"{agent}-smart-routing-audit.jsonl")
    session.record("subagent-routing.json", {"decisions": decisions, "starts": audit})
    assert decisions, "No real subagent routing decision was recorded"
    for decision in decisions:
        assert decision.get("requested_model") and decision.get("router_model"), decision
    if agent == "codex":
        # Codex exposes parent linkage and the child's actual turn model in its
        # native rollouts. Its ug SubagentStart audit can be empty even when the
        # child ran. Match native evidence, including the completed file task.
        sessions = agent_sessions(session, agent)
        linked = []
        for path, records in sessions.items():
            metadata = next(
                (row["payload"] for row in records if row.get("type") == "session_meta"), {}
            )
            source = metadata.get("source")
            if not isinstance(source, dict):
                continue
            parent_id = source.get("subagent", {}).get("thread_spawn", {}).get("parent_thread_id")
            if not parent_id:
                continue
            # A child rollout starts with inherited parent history. Exclude
            # those turn IDs so a parent's answer/model cannot satisfy this check.
            parent_turn_ids = set()
            for other in sessions.values():
                first_meta = next(
                    (row["payload"] for row in other if row.get("type") == "session_meta"), {}
                )
                if first_meta.get("id") == parent_id:
                    parent_turn_ids.update(
                        row["payload"]["turn_id"]
                        for row in other
                        if row.get("type") == "turn_context" and row["payload"].get("turn_id")
                    )
            assert parent_turn_ids, f"No native parent turns found for {parent_id}"
            for decision in decisions:
                if decision.get("session_id") != parent_id:
                    continue
                routed_turn_ids = {
                    row["payload"]["turn_id"]
                    for row in records
                    if row.get("type") == "turn_context"
                    and row["payload"].get("model") == decision["requested_model"]
                    and row["payload"].get("turn_id")
                } - parent_turn_ids
                for row in records:
                    payload = row.get("payload", {})
                    if (
                        row.get("type") == "event_msg"
                        and payload.get("type") == "task_complete"
                        and payload.get("turn_id") in routed_turn_ids
                        and task.value in (payload.get("last_agent_message") or "")
                    ):
                        linked.append(
                            {
                                "decision_id": decision["decision_id"],
                                "parent_id": parent_id,
                                "child_id": metadata["id"],
                                "path": path,
                                "turn_id": payload["turn_id"],
                                "model": decision["requested_model"],
                            }
                        )
        session.record("subagent-routing.json", {"decisions": decisions, "native_children": linked})
        assert linked, "No routed native child turn completed the delegated file task"
        assert {row["decision_id"] for row in linked} == {
            row["decision_id"] for row in decisions
        }, "A routing decision had no matching completed child turn"
        return
    decision_ids = {decision["decision_id"] for decision in decisions}
    routed_starts = [row for row in audit if row.get("decision_id") in decision_ids]
    assert routed_starts and all(row.get("agent_id") for row in routed_starts), audit
    assert all(row.get("matches_router_decision") is not False for row in routed_starts), audit
    # Some agent versions omit the child's model from SubagentStart. The report
    # preserves that unknown value; this test claims decision + spawn + task,
    # not model-identity verification when the agent did not expose it.
