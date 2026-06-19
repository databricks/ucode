"""Platform-team-maintained allowlist of models available to users.

This is the single source of truth for which models are approved for use
through voxcode. Only models listed here will be offered during configuration
and allowed at launch time.

To update: edit the lists below and release a new version of voxcode.
The platform team owns this file.
"""

from __future__ import annotations

# ─── Anthropic models (served via AI Gateway anthropic-compatible endpoint) ───
# These appear in OpenCode as the "databricks-anthropic" provider.
ALLOWED_ANTHROPIC_MODELS: list[str] = [
    "claude-sonnet-4-20250514",
    "claude-haiku-4-20250514",
]

# ─── Google models (served via AI Gateway gemini-compatible endpoint) ─────────
# These appear in OpenCode as the "databricks-google" provider.
ALLOWED_GEMINI_MODELS: list[str] = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]


def filter_anthropic_models(discovered: dict[str, str]) -> dict[str, str]:
    """Filter a {tier: model_id} dict to only allowed models."""
    return {tier: model for tier, model in discovered.items() if model in ALLOWED_ANTHROPIC_MODELS}


def filter_gemini_models(discovered: list[str]) -> list[str]:
    """Filter a list of model IDs to only allowed models."""
    return [m for m in discovered if m in ALLOWED_GEMINI_MODELS]


def get_all_allowed_models() -> dict[str, list[str]]:
    """Return the full allowlist grouped by provider for display purposes."""
    return {
        "anthropic": ALLOWED_ANTHROPIC_MODELS,
        "gemini": ALLOWED_GEMINI_MODELS,
    }
