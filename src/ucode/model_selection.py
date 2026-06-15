"""Shared model selection helpers for discovered AI Gateway models."""

from __future__ import annotations

import re

CLAUDE_FAMILY_ORDER = ("opus", "sonnet", "haiku")
_CODEX_GPT_RE = re.compile(r"(?:databricks-)?gpt-(\d+)(?:[.-](\d+))?(?:[.-](\d+))?(-.+|[a-z].*)?")


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def claude_model_options(state: dict) -> list[str]:
    """Return all selectable Claude models, falling back to legacy family defaults."""
    options = _string_list(state.get("claude_model_options"))
    if options:
        return _unique(options)

    claude_models_value = state.get("claude_models")
    claude_models = claude_models_value if isinstance(claude_models_value, dict) else {}
    return _unique(
        [
            str(claude_models[family]).strip()
            for family in CLAUDE_FAMILY_ORDER
            if isinstance(claude_models.get(family), str) and str(claude_models[family]).strip()
        ]
    )


def _is_codex_gpt_model(model: str) -> bool:
    tail = model.split("/")[-1]
    if tail.startswith("system.ai."):
        tail = tail[len("system.ai.") :]
    return _CODEX_GPT_RE.fullmatch(tail) is not None


def available_models_for_tool(tool: str, state: dict) -> list[str]:
    """Return the model ids this agent can use, in picker/default order."""
    if tool == "claude":
        return claude_model_options(state)
    if tool == "codex":
        return _unique(
            [
                model
                for model in _string_list(state.get("codex_models"))
                if _is_codex_gpt_model(model)
            ]
        )
    if tool == "gemini":
        return _unique(_string_list(state.get("gemini_models")))
    if tool == "opencode":
        opencode_models_value = state.get("opencode_models")
        opencode_models = opencode_models_value if isinstance(opencode_models_value, dict) else {}
        return _unique(
            _string_list(opencode_models.get("anthropic"))
            + _string_list(opencode_models.get("gemini"))
        )
    if tool == "copilot":
        return _unique(claude_model_options(state) + _string_list(state.get("codex_models")))
    if tool == "pi":
        return _unique(
            claude_model_options(state)
            + _string_list(state.get("codex_models"))
            + _string_list(state.get("gemini_models"))
        )
    return []


def selected_model_for_tool(tool: str, state: dict) -> str | None:
    """Return a persisted per-tool selection only if it is still selectable."""
    selected_models_value = state.get("selected_models")
    selected_models = selected_models_value if isinstance(selected_models_value, dict) else {}
    selected = selected_models.get(tool)
    if not isinstance(selected, str):
        return None
    selected = selected.strip()
    if selected and selected in available_models_for_tool(tool, state):
        return selected
    return None
