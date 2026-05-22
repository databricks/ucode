"""Stdio MCP server exposing a `web_search` tool backed by a Databricks-hosted
GPT model's native Responses API web search.

Claude Code on Databricks doesn't have working web search (the built-in
`WebSearch` tool talks to Anthropic's hosted infra, not the gateway). This
server bridges the gap: it advertises a single MCP tool, and on call it
forwards the query to the workspace's Responses API with
`tools: [{"type": "web_search"}]`, returning the model's text output.

Speaks MCP JSON-RPC 2.0 over stdio (newline-delimited JSON). Implemented by
hand to avoid pulling in the `mcp` SDK — keeps `ucode`'s dep footprint lean.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from ucode.databricks import get_databricks_token

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "ucode-web-search"
SERVER_VERSION = "0.1.0"

TOOL_NAME = "web_search"
TOOL_DESCRIPTION = (
    "Search the web for up-to-date public information, current events, "
    "real-time facts, and recent data. Use this when the user's question "
    "requires information beyond your training data."
)
TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query.",
        },
    },
    "required": ["query"],
}


def _tool_descriptor() -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "inputSchema": TOOL_INPUT_SCHEMA,
    }


def _result(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _tool_error(text: str) -> dict[str, Any]:
    """An MCP tool-call result with isError=true. Different from JSON-RPC errors:
    tool-level failures should be returned as results so the model can see and
    react to them, not as protocol errors that abort the call."""
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _extract_response_text(payload: dict[str, Any]) -> str:
    """Walk a Responses API payload and concatenate all `output_text` content
    from `message`-type output items. Skips reasoning, tool-call, and other
    item types — we only want the final user-facing answer."""
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts).strip()


def _call_responses_api(query: str) -> dict[str, Any]:
    """POST to the Databricks Codex (Responses API) gateway and return the
    parsed JSON payload. Raises RuntimeError on any failure with a message
    suitable for surfacing as a tool error."""
    workspace = os.environ.get("DATABRICKS_HOST", "").strip()
    model = os.environ.get("UCODE_WEB_SEARCH_MODEL", "").strip()
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", "").strip() or None
    if not workspace:
        raise RuntimeError("DATABRICKS_HOST env var is not set.")
    if not model:
        raise RuntimeError("UCODE_WEB_SEARCH_MODEL env var is not set.")

    try:
        token = get_databricks_token(workspace, profile)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to acquire Databricks token: {exc}") from exc

    body = json.dumps(
        {
            "model": model,
            "input": [{"role": "user", "content": query}],
            "tools": [{"type": "web_search"}],
            "store": False,
        }
    ).encode("utf-8")

    request = urllib_request.Request(
        f"{workspace.rstrip('/')}/ai-gateway/codex/v1/responses",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib_request.urlopen(request, timeout=180) as response:
            raw = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:500]
        except Exception:
            pass
        raise RuntimeError(f"Responses API returned HTTP {exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Responses API request failed: {exc.reason}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Responses API returned non-JSON payload: {exc}") from exc


def _handle_tools_call(arguments: dict[str, Any]) -> dict[str, Any]:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return _tool_error("`query` must be a non-empty string.")
    try:
        payload = _call_responses_api(query)
    except RuntimeError as exc:
        return _tool_error(str(exc))

    text = _extract_response_text(payload)
    if not text:
        return _tool_error("Web search returned no text output.")
    return {"content": [{"type": "text", "text": text}]}


def _handle_request(req: dict[str, Any]) -> dict[str, Any] | None:
    """Dispatch a single JSON-RPC request. Returns the response dict, or None
    for notifications (which must not produce a response per JSON-RPC spec)."""
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params") or {}
    is_notification = "id" not in req

    if method == "initialize":
        return _result(
            req_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return _result(req_id, {"tools": [_tool_descriptor()]})
    if method == "tools/call":
        if params.get("name") != TOOL_NAME:
            return _error(req_id, -32602, f"Unknown tool: {params.get('name')!r}")
        return _result(req_id, _handle_tools_call(params.get("arguments") or {}))

    if is_notification:
        return None
    return _error(req_id, -32601, f"Method not found: {method!r}")


def serve(stdin=None, stdout=None) -> None:
    """Read newline-delimited JSON-RPC requests from stdin, write responses to
    stdout. Loops until EOF. Injectable streams for testing."""
    in_stream = stdin if stdin is not None else sys.stdin
    out_stream = stdout if stdout is not None else sys.stdout

    for line in in_stream:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            response = _error(None, -32700, "Parse error")
            out_stream.write(json.dumps(response) + "\n")
            out_stream.flush()
            continue

        response = _handle_request(req)
        if response is None:
            continue
        out_stream.write(json.dumps(response) + "\n")
        out_stream.flush()
