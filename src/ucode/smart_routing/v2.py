"""Shared configuration for smart routing v2 — the runtime model-switching launch path.

Smart routing v2 launches an agent's real TUI against a ucode-run app-server with a
WebSocket interposer that switches the model mid-session (see e.g.
``smart_routing.codex_interposer``). The enable flag and cross-agent knobs live here so
every routing-capable agent (Codex today, Claude Code next) reads them from one place;
each agent keeps its own target model, paths, and launch wiring.
"""

from __future__ import annotations

import os

# Single env var that enables the v2 launch path for every routing-capable agent.
ENV_VAR = "ENABLE_SMART_ROUTING_V2"

# Turns to keep the session on its starting model before switching (0 = switch immediately).
SWITCH_AFTER_TURNS = 1


def enabled() -> bool:
    """Return whether the smart-routing-v2 launch path is enabled via the env var."""
    return os.environ.get(ENV_VAR) == "1"
