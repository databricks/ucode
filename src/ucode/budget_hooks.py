"""Shared budget hook config for agent launchers."""

from __future__ import annotations

import shlex
import shutil
from typing import cast

BUDGET_HOOK_TIMEOUT_SECONDS = 10
BUDGET_HOOK_STATUS_MESSAGE = "Loading ucode budget"


def budget_hook_command(tool: str) -> str:
    ucode_binary = shutil.which("ucode") or "ucode"
    return f"{shlex.quote(ucode_binary)} budget-hook --tool {shlex.quote(tool)}"


def budget_hook_handler(tool: str) -> dict:
    return {
        "type": "command",
        "command": budget_hook_command(tool),
        "timeout": BUDGET_HOOK_TIMEOUT_SECONDS,
        "statusMessage": BUDGET_HOOK_STATUS_MESSAGE,
    }


def is_budget_hook_handler(value: object, tool: str) -> bool:
    if not isinstance(value, dict):
        return False
    value_dict = cast(dict[object, object], value)
    command = value_dict.get("command")
    return isinstance(command, str) and f" budget-hook --tool {tool}" in f" {command}"
