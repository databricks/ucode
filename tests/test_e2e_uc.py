"""End-to-end tests for the `--enable-uc` / `UCODE_ENABLE_UC` opt-in.

Verifies UC-securables discovery (model-services + MCP services) and the
flag-precedence ladder (CLI flag > env var > persisted state) against a
live Databricks workspace.

Run with:
    UCODE_TEST_WORKSPACE=https://your-workspace.databricks.com \
      uv run pytest tests/test_e2e_uc.py -v
"""

from __future__ import annotations

import pytest

from ucode.cli import configure_shared_state
from ucode.databricks import (
    discover_model_services,
    list_mcp_services,
    uc_enabled,
)
from ucode.state import load_state


def _has_uc_models(workspace: str, token: str) -> bool:
    claude, codex, gemini, _reason = discover_model_services(workspace, token)
    return bool(claude or codex or gemini)


def _all_resolved_model_ids(state: dict) -> list[str]:
    ids: list[str] = list((state.get("claude_models") or {}).values())
    ids += state.get("codex_models") or []
    ids += state.get("gemini_models") or []
    return ids


# ---------------------------------------------------------------------------
# UC discovery primitives — verify the endpoints return only `system.ai.*`
# entries (the per-family/connection filters drop everything else).
# ---------------------------------------------------------------------------


class TestDiscoverModelServicesE2E:
    def test_returns_only_system_ai_models(self, e2e_workspace, e2e_token):
        claude, codex, gemini, reason = discover_model_services(e2e_workspace, e2e_token)
        if not (claude or codex or gemini):
            pytest.skip(f"No system.ai.* model services on workspace: {reason}")
        non_system = sorted(
            {
                m
                for m in _all_resolved_model_ids(
                    {"claude_models": claude, "codex_models": codex, "gemini_models": gemini}
                )
                if not m.startswith("system.ai.")
            }
        )
        assert not non_system, f"Non-system.ai entries leaked through: {non_system[:5]}"


class TestListMcpServicesE2E:
    def test_returns_only_system_ai_mcp_services(self, e2e_workspace, e2e_token):
        names, reason = list_mcp_services(e2e_workspace, e2e_token)
        if not names:
            pytest.skip(f"No system.ai.* MCP services on workspace: {reason}")
        non_system = sorted({n for n in names if not n.startswith("system.ai.")})
        assert not non_system, f"Non-system.ai entries leaked through: {non_system[:5]}"


# ---------------------------------------------------------------------------
# `uc_enabled` precedence — env var alone (no `default` arg).
# ---------------------------------------------------------------------------


class TestUcEnabledEnvE2E:
    def test_env_on_overrides_default_off(self, monkeypatch):
        monkeypatch.setenv("UCODE_ENABLE_UC", "1")
        assert uc_enabled(default=False) is True

    def test_env_off_overrides_default_on(self, monkeypatch):
        monkeypatch.setenv("UCODE_ENABLE_UC", "0")
        assert uc_enabled(default=True) is False


# ---------------------------------------------------------------------------
# `configure_shared_state` end-to-end: flag resolution + persistence to
# `state["uc_enabled"]` + which discovery path runs (UC vs legacy).
# ---------------------------------------------------------------------------


class TestConfigureSharedStateEnableUcE2E:
    """Resolution ladder: CLI flag > env var > persisted state. Each test
    runs the full configure path against the live workspace and asserts on
    the resolved flag and the actual model namespaces written to state."""

    def test_explicit_true_persists_and_discovers_system_ai(
        self, monkeypatch, e2e_workspace, e2e_token
    ):
        if not _has_uc_models(e2e_workspace, e2e_token):
            pytest.skip("Workspace has no system.ai.* model services.")
        monkeypatch.delenv("UCODE_ENABLE_UC", raising=False)

        state = configure_shared_state(e2e_workspace, force_login=False, enable_uc=True)
        assert state["uc_enabled"] is True
        assert load_state()["uc_enabled"] is True
        ids = _all_resolved_model_ids(state)
        assert any(m.startswith("system.ai.") for m in ids), (
            f"Expected at least one system.ai.* model id, got: {ids[:5]}"
        )

    def test_env_off_overrides_persisted_true(self, monkeypatch, e2e_workspace):
        monkeypatch.delenv("UCODE_ENABLE_UC", raising=False)
        # Pre-seed the target workspace's state with uc_enabled=True.
        configure_shared_state(e2e_workspace, force_login=False, enable_uc=True)
        assert load_state()["uc_enabled"] is True

        # Now opt back out via env var, no CLI flag.
        monkeypatch.setenv("UCODE_ENABLE_UC", "0")
        state = configure_shared_state(e2e_workspace, force_login=False, enable_uc=None)
        assert state["uc_enabled"] is False
        assert load_state()["uc_enabled"] is False
        ids = _all_resolved_model_ids(state)
        assert all(not m.startswith("system.ai.") for m in ids), (
            f"Legacy discovery leaked system.ai entries: "
            f"{[m for m in ids if m.startswith('system.ai.')][:5]}"
        )

    def test_env_resolves_when_cli_flag_omitted(self, monkeypatch, e2e_workspace, e2e_token):
        if not _has_uc_models(e2e_workspace, e2e_token):
            pytest.skip("Workspace has no system.ai.* model services.")
        monkeypatch.setenv("UCODE_ENABLE_UC", "1")

        state = configure_shared_state(e2e_workspace, force_login=False, enable_uc=None)
        assert state["uc_enabled"] is True
        ids = _all_resolved_model_ids(state)
        assert any(m.startswith("system.ai.") for m in ids)

    def test_plain_configure_resets_persisted_uc_enabled(
        self, monkeypatch, e2e_workspace, e2e_token
    ):
        """`ucode configure` without `--enable-uc` and without
        UCODE_ENABLE_UC is a clean slate: a previously-persisted
        `uc_enabled=True` is flipped back to False, and discovery returns
        to legacy `databricks-*` ids."""
        if not _has_uc_models(e2e_workspace, e2e_token):
            pytest.skip("Workspace has no system.ai.* model services.")
        monkeypatch.delenv("UCODE_ENABLE_UC", raising=False)

        # First configure persists `uc_enabled=True`.
        configure_shared_state(e2e_workspace, force_login=False, enable_uc=True)
        assert load_state()["uc_enabled"] is True

        # Second configure with reset_uc=True (the explicit `ucode configure`
        # path) clears the flag.
        state = configure_shared_state(
            e2e_workspace, force_login=False, enable_uc=None, reset_uc=True
        )
        assert state["uc_enabled"] is False
        assert load_state()["uc_enabled"] is False
        ids = _all_resolved_model_ids(state)
        assert all(not m.startswith("system.ai.") for m in ids), (
            f"Reset run still pulled UC ids: {[m for m in ids if m.startswith('system.ai.')][:5]}"
        )

    def test_launch_path_preserves_persisted_uc_enabled(
        self, monkeypatch, e2e_workspace, e2e_token
    ):
        """Launch-time refetches (`ucode <agent>`) call configure_shared_state
        without `reset_uc`. They must keep an existing persisted True so a
        Claude/Codex/Gemini launch right after `--enable-uc` doesn't silently
        drop UC discovery."""
        if not _has_uc_models(e2e_workspace, e2e_token):
            pytest.skip("Workspace has no system.ai.* model services.")
        monkeypatch.delenv("UCODE_ENABLE_UC", raising=False)

        # User runs `ucode configure --enable-uc`.
        configure_shared_state(e2e_workspace, force_login=False, enable_uc=True)
        assert load_state()["uc_enabled"] is True

        # User then runs `ucode claude` — same call shape as
        # _launch_tool's refetch (no reset_uc, no enable_uc, no env).
        state = configure_shared_state(e2e_workspace, force_login=False, enable_uc=None)
        assert state["uc_enabled"] is True
        assert any(m.startswith("system.ai.") for m in _all_resolved_model_ids(state))

    def test_default_off_when_no_flag_no_env_no_state(self, monkeypatch, e2e_workspace):
        # Fresh state (autouse fixture redirects STATE_PATH per test) plus
        # no env var means the flag falls through to its default of False.
        monkeypatch.delenv("UCODE_ENABLE_UC", raising=False)

        state = configure_shared_state(e2e_workspace, force_login=False, enable_uc=None)
        assert state["uc_enabled"] is False
        ids = _all_resolved_model_ids(state)
        assert all(not m.startswith("system.ai.") for m in ids)

    def test_other_workspace_flag_does_not_leak_into_target(self, monkeypatch, e2e_workspace):
        """Regression: enabling UC on workspace A must not silently turn it
        on for a fresh `ucode configure` on workspace B. The default has to
        come from B's own persisted state, not A's (which is whatever
        happens to be `current_workspace`)."""
        from ucode.state import save_state

        monkeypatch.delenv("UCODE_ENABLE_UC", raising=False)

        other_ws = "https://other-workspace.cloud.databricks.com"
        save_state({"workspace": other_ws, "uc_enabled": True})

        state = configure_shared_state(e2e_workspace, force_login=False, enable_uc=None)
        assert state["uc_enabled"] is False, (
            "Cross-workspace leak: another workspace's uc_enabled bled into "
            "the target workspace's default."
        )
        ids = _all_resolved_model_ids(state)
        assert all(not m.startswith("system.ai.") for m in ids), (
            f"Discovery used UC despite per-workspace default being False: "
            f"{[m for m in ids if m.startswith('system.ai.')][:5]}"
        )
