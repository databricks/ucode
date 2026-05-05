"""End-to-end integration tests that require a live Databricks workspace.

Run with:
    CODING_GATEWAY_TEST_WORKSPACE=https://your-workspace.databricks.com uv run pytest tests/test_e2e.py -v

All tests in this file are skipped automatically when the env var is not set.
The agent-launch tests are also skipped per-agent/model when the binary is not
installed or no models are available.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from coding_tool_gateway.databricks import (
    build_shared_base_urls,
    build_tool_base_url,
    discover_sql_warehouse_http_path,
    ensure_ai_gateway_v2,
    fetch_ai_gateway_claude_models,
    fetch_codex_models,
    fetch_gemini_models,
    has_valid_databricks_auth,
    workspace_hostname,
)
from coding_tool_gateway.ui import normalize_workspace_url

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ws() -> str:
    raw = os.environ.get("CODING_GATEWAY_TEST_WORKSPACE", "").strip().rstrip("/")
    return normalize_workspace_url(raw) if raw else ""


def _skip_if_no_workspace():
    if not _ws():
        pytest.skip("Set CODING_GATEWAY_TEST_WORKSPACE=https://... to run E2E tests")


def _run_agent(
    cmd: list[str], env: dict | None = None, timeout: int = 60
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        stdin=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Databricks auth / token
# ---------------------------------------------------------------------------


class TestDatabricksAuth:
    def test_has_valid_auth(self, e2e_workspace):
        assert has_valid_databricks_auth(e2e_workspace), (
            "No valid Databricks auth found. Run `databricks auth login` first."
        )

    def test_get_token_returns_non_empty_string(self, e2e_token):
        assert isinstance(e2e_token, str) and len(e2e_token) > 10


# ---------------------------------------------------------------------------
# AI Gateway v2 probe
# ---------------------------------------------------------------------------


class TestAiGatewayV2:
    def test_ensure_ai_gateway_v2_does_not_raise(self, e2e_workspace, e2e_token):
        ensure_ai_gateway_v2(e2e_workspace, e2e_token)

    def test_workspace_hostname_resolves(self, e2e_workspace):
        hostname = workspace_hostname(e2e_workspace)
        assert "." in hostname


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


class TestModelDiscovery:
    def test_fetch_claude_models_returns_dict(self, e2e_workspace, e2e_token):
        models = fetch_ai_gateway_claude_models(e2e_workspace, e2e_token)
        assert isinstance(models, dict)
        assert models, "No Claude models found — is the Anthropic route enabled on this workspace?"

    def test_fetch_gemini_models_returns_list(self, e2e_workspace, e2e_token):
        models = fetch_gemini_models(e2e_workspace, e2e_token)
        assert isinstance(models, list)

    def test_fetch_codex_models_returns_list(self, e2e_workspace, e2e_token):
        models = fetch_codex_models(e2e_workspace, e2e_token)
        assert isinstance(models, list)


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------


class TestUrlBuilders:
    def test_codex_url_contains_workspace(self, e2e_workspace):
        assert e2e_workspace in build_tool_base_url("codex", e2e_workspace)

    def test_claude_url_contains_workspace(self, e2e_workspace):
        assert e2e_workspace in build_tool_base_url("claude", e2e_workspace)

    def test_shared_base_urls_all_tools(self, e2e_workspace):
        urls = build_shared_base_urls(e2e_workspace)
        for tool in ("codex", "claude", "gemini", "opencode"):
            assert tool in urls


# ---------------------------------------------------------------------------
# State round-trip
# ---------------------------------------------------------------------------


class TestStateRoundTrip:
    def test_configure_shared_state_and_reload(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace
    ):
        import coding_tool_gateway.config_io as config_io_mod
        import coding_tool_gateway.state as state_mod
        from coding_tool_gateway.state import load_state, save_state

        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)

        save_state(e2e_state)
        loaded = load_state()
        assert loaded["workspace"] == e2e_workspace
        assert loaded["claude_models"] == e2e_state["claude_models"]
        assert loaded["base_urls"]["codex"] == f"{e2e_workspace}/ai-gateway/codex/v1"


# ---------------------------------------------------------------------------
# SQL warehouse discovery
# ---------------------------------------------------------------------------


class TestSqlWarehouseDiscovery:
    def test_discovers_http_path(self, e2e_workspace, e2e_token):
        try:
            http_path = discover_sql_warehouse_http_path(e2e_workspace, e2e_token, quiet=True)
        except RuntimeError as exc:
            pytest.skip(f"No SQL warehouse available: {exc}")
        assert http_path.startswith("/sql/1.0/warehouses/")


# ---------------------------------------------------------------------------
# Agent launch tests — one test per (agent, model)
# ---------------------------------------------------------------------------
#
# Each test:
#   1. Writes the agent config for the specific model
#   2. Runs the binary with the validate_cmd prompt
#   3. Asserts exit code 0 and non-empty stdout
#
# Tests are skipped when the binary is not installed or no models are available.
# ---------------------------------------------------------------------------


def _require_binary(binary: str):
    if not shutil.which(binary):
        pytest.skip(f"`{binary}` is not installed")


class TestCodexLaunch:
    """Run codex against every available codex model."""

    def _codex_models(self, e2e_state: dict) -> list[str]:
        models = e2e_state.get("codex_models") or []
        if not models:
            pytest.skip("No Codex models available on this workspace")
        return models

    def test_launch_codex_per_model(self, tmp_path, monkeypatch, e2e_state, e2e_workspace):
        """Parametrized inline — iterates over all codex models and asserts each works."""
        import coding_tool_gateway.config_io as config_io_mod
        from coding_tool_gateway.agents import codex

        _require_binary("codex")
        models = self._codex_models(e2e_state)

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_dir = tmp_path / "codex_home" / ".codex"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.toml"
        backup_path = tmp_path / "codex-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)

        failures = []
        for model in models:
            state = {**e2e_state, "workspace": e2e_workspace}
            with pytest.MonkeyPatch().context() as mp:
                mp.setattr("coding_tool_gateway.state.save_state", lambda s: None)
                codex.write_tool_config(state)

            cmd = codex.validate_cmd("codex")
            result = _run_agent(cmd)
            if result.returncode != 0 or not (result.stdout or result.stderr).strip():
                failures.append(
                    f"model={model} rc={result.returncode} "
                    f"stdout={result.stdout[:200]!r} stderr={result.stderr[:200]!r}"
                )

        assert not failures, "Codex launch failures:\n" + "\n".join(failures)


class TestClaudeLaunch:
    """Run claude against every available claude model (sonnet, opus, haiku)."""

    def test_launch_claude_per_model(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace, e2e_token
    ):
        import coding_tool_gateway.config_io as config_io_mod
        from coding_tool_gateway.agents import claude

        _require_binary("claude")
        claude_models: dict = e2e_state.get("claude_models") or {}
        if not claude_models:
            pytest.skip("No Claude models available on this workspace")

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_path = tmp_path / "claude-settings.json"
        backup_path = tmp_path / "claude-settings.backup.json"
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", config_path)
        monkeypatch.setattr(claude, "CLAUDE_BACKUP_PATH", backup_path)

        base_url = build_tool_base_url("claude", e2e_workspace)

        failures = []
        for family, model_id in claude_models.items():
            with pytest.MonkeyPatch().context() as mp:
                mp.setattr("coding_tool_gateway.state.save_state", lambda s: None)
                claude.write_tool_config({**e2e_state, "workspace": e2e_workspace}, model_id)

            env = {
                **os.environ,
                "ANTHROPIC_MODEL": model_id,
                "ANTHROPIC_BASE_URL": base_url,
                "ANTHROPIC_API_KEY": e2e_token,
                "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            }
            cmd = claude.validate_cmd("claude")
            result = _run_agent(cmd, env=env, timeout=90)
            combined = (result.stdout + result.stderr).strip()
            if result.returncode != 0 or not combined:
                failures.append(
                    f"family={family} model={model_id} rc={result.returncode} "
                    f"stdout={result.stdout[:300]!r} stderr={result.stderr[:300]!r}"
                )

        assert not failures, "Claude launch failures:\n" + "\n".join(failures)


class TestGeminiLaunch:
    """Run gemini against every available gemini model."""

    def test_launch_gemini_per_model(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace, e2e_token
    ):
        import coding_tool_gateway.config_io as config_io_mod
        from coding_tool_gateway.agents import gemini

        _require_binary("gemini")
        gemini_models: list = e2e_state.get("gemini_models") or []
        if not gemini_models:
            pytest.skip("No Gemini models available on this workspace")

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        env_path = tmp_path / ".env"
        backup_path = tmp_path / "gemini-env.backup"
        monkeypatch.setattr(gemini, "GEMINI_ENV_PATH", env_path)
        monkeypatch.setattr(gemini, "GEMINI_BACKUP_PATH", backup_path)

        failures = []
        for model in gemini_models:
            with pytest.MonkeyPatch().context() as mp:
                mp.setattr("coding_tool_gateway.state.save_state", lambda s: None)
                mp.setattr(
                    "coding_tool_gateway.agents.gemini.get_databricks_token", lambda ws: e2e_token
                )
                gemini.write_tool_config(
                    {**e2e_state, "workspace": e2e_workspace}, model, token=e2e_token
                )

            env = gemini.build_runtime_env(e2e_workspace, model, e2e_token)
            cmd = gemini.validate_cmd("gemini")
            result = _run_agent(cmd, env=env, timeout=90)
            combined = (result.stdout + result.stderr).strip()
            if result.returncode != 0 or not combined:
                failures.append(
                    f"model={model} rc={result.returncode} "
                    f"stdout={result.stdout[:300]!r} stderr={result.stderr[:300]!r}"
                )

        assert not failures, "Gemini launch failures:\n" + "\n".join(failures)


class TestOpencodeLaunch:
    """Run opencode against every available opencode model (anthropic + gemini)."""

    def _all_models(self, e2e_state: dict) -> list[tuple[str, str]]:
        """Return [(provider, model_id), ...] for all opencode models."""
        opencode_models: dict = e2e_state.get("opencode_models") or {}
        out: list[tuple[str, str]] = []
        for provider, models in opencode_models.items():
            for model in models or []:
                out.append((provider, model))
        return out

    def test_launch_opencode_per_model(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace, e2e_token
    ):
        import coding_tool_gateway.config_io as config_io_mod
        from coding_tool_gateway.agents import opencode

        _require_binary("opencode")
        models = self._all_models(e2e_state)
        if not models:
            pytest.skip("No OpenCode models available on this workspace")

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_path = tmp_path / "opencode.json"
        backup_path = tmp_path / "opencode-config.backup.json"
        monkeypatch.setattr(opencode, "OPENCODE_CONFIG_PATH", config_path)
        monkeypatch.setattr(opencode, "OPENCODE_BACKUP_PATH", backup_path)

        failures = []
        for provider, model in models:
            # Reset config file before each model so configs don't bleed together
            if config_path.exists():
                config_path.unlink()

            with pytest.MonkeyPatch().context() as mp:
                mp.setattr("coding_tool_gateway.state.save_state", lambda s: None)
                mp.setattr(
                    "coding_tool_gateway.agents.opencode.get_databricks_token", lambda ws: e2e_token
                )
                opencode.write_tool_config(
                    {**e2e_state, "workspace": e2e_workspace},
                    model,
                    token=e2e_token,
                )

            cmd = opencode.validate_cmd("opencode")
            result = _run_agent(cmd, timeout=90)
            combined = (result.stdout + result.stderr).strip()
            if result.returncode != 0 or not combined:
                failures.append(
                    f"provider={provider} model={model} rc={result.returncode} "
                    f"stdout={result.stdout[:300]!r} stderr={result.stderr[:300]!r}"
                )

        assert not failures, "OpenCode launch failures:\n" + "\n".join(failures)
