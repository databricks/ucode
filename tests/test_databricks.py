"""Tests for databricks.py — pure helpers and URL builders that don't hit the network."""

from __future__ import annotations

import json
import os
import subprocess

import pytest

import ucode.databricks as db_mod
from ucode.databricks import (
    AI_GATEWAY_V2_DOCS_URL,
    _format_subprocess_result,
    _parse_databricks_cli_version,
    _run_databricks_cli_installer,
    _scrub_databrickscfg,
    _scrub_json,
    build_auth_shell_command,
    build_auth_token_argv,
    build_databricks_cli_env,
    build_opencode_base_urls,
    build_shared_base_urls,
    build_tool_base_url,
    ensure_databricks_cli_version,
    ensure_pat_bearer,
    get_databricks_token,
    list_databricks_apps,
    list_databricks_connections,
    list_genie_spaces,
    workspace_hostname,
)

WS = "https://example.databricks.com"


class TestWorkspaceHostname:
    def test_extracts_hostname(self):
        assert workspace_hostname(WS) == "example.databricks.com"

    def test_handles_path(self):
        assert (
            workspace_hostname("https://foo.azuredatabricks.net/some/path")
            == "foo.azuredatabricks.net"
        )

    def test_invalid_url_raises(self):
        with pytest.raises((RuntimeError, ValueError)):
            workspace_hostname("")


class TestBuildDatabricksCliEnv:
    def test_sets_databricks_host(self):
        env = build_databricks_cli_env(WS)
        assert env["DATABRICKS_HOST"] == WS

    def test_strips_ambient_profile_without_explicit_profile(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "other-workspace")

        env = build_databricks_cli_env(WS)

        assert env["DATABRICKS_HOST"] == WS
        assert "DATABRICKS_CONFIG_PROFILE" not in env

    def test_preserves_ambient_profile_with_explicit_profile(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "other-workspace")

        env = build_databricks_cli_env(WS, profile="stablebox")

        assert env["DATABRICKS_HOST"] == WS
        assert env["DATABRICKS_CONFIG_PROFILE"] == "other-workspace"


class TestBuildToolBaseUrl:
    def test_codex(self):
        url = build_tool_base_url("codex", WS)
        assert url == f"{WS}/ai-gateway/codex/v1"

    def test_claude(self):
        url = build_tool_base_url("claude", WS)
        assert url == f"{WS}/ai-gateway/anthropic"

    def test_gemini(self):
        url = build_tool_base_url("gemini", WS)
        assert url == f"{WS}/ai-gateway/gemini"

    def test_opencode_raises(self):
        with pytest.raises(RuntimeError, match="multiple base URLs"):
            build_tool_base_url("opencode", WS)

    def test_unsupported_tool_raises(self):
        with pytest.raises(RuntimeError, match="Unsupported"):
            build_tool_base_url("unknown", WS)


class TestBuildOpencodeBaseUrls:
    def test_returns_anthropic_gemini_and_oss(self):
        urls = build_opencode_base_urls(WS)
        assert urls["anthropic"] == f"{WS}/ai-gateway/anthropic/v1"
        assert urls["gemini"] == f"{WS}/ai-gateway/gemini/v1beta"
        assert urls["oss"] == f"{WS}/ai-gateway/mlflow/v1"

    def test_external_uses_classic_serving_endpoints_path(self):
        assert build_opencode_base_urls(WS)["external"] == f"{WS}/serving-endpoints"


class TestBuildExternalServingBaseUrl:
    def test_points_at_classic_serving_endpoints(self):
        # External models are reached off the unified gateway; the openai
        # chat-completions provider appends `/chat/completions`.
        assert db_mod.build_external_serving_base_url(WS) == f"{WS}/serving-endpoints"


class TestBuildSharedBaseUrls:
    def test_contains_all_tools(self):
        urls = build_shared_base_urls(WS)
        assert "codex" in urls
        assert "claude" in urls
        assert "gemini" in urls
        assert "opencode" in urls

    def test_opencode_is_dict(self):
        urls = build_shared_base_urls(WS)
        assert isinstance(urls["opencode"], dict)

    def test_codex_url_format(self):
        urls = build_shared_base_urls(WS)
        assert urls["codex"] == f"{WS}/ai-gateway/codex/v1"


class TestDiscoverClaudeModels:
    def test_selects_opus_4_8_when_advertised(self, monkeypatch):
        payload = {
            "data": [
                {"id": "databricks-claude-opus-4-7"},
                {"id": "databricks-claude-opus-4-8"},
                {"id": "databricks-claude-sonnet-4-6"},
            ]
        }
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: (payload, None))

        models, ids, reason = db_mod.discover_claude_models(WS, "token")

        assert reason is None
        assert models["opus"] == "databricks-claude-opus-4-8"
        assert set(ids) == {
            "databricks-claude-opus-4-7",
            "databricks-claude-opus-4-8",
            "databricks-claude-sonnet-4-6",
        }

    def test_two_digit_minor_version_beats_single_digit(self, monkeypatch):
        payload = {
            "data": [
                {"id": "databricks-claude-opus-4-8"},
                {"id": "databricks-claude-opus-4-10"},
            ]
        }
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: (payload, None))

        models, _ids, _reason = db_mod.discover_claude_models(WS, "token")

        assert models["opus"] == "databricks-claude-opus-4-10"

    def test_familyless_model_reported_in_ids_only(self, monkeypatch):
        # A Claude model outside opus/sonnet/haiku has no family env var, but
        # agents with their own picker (pi) must still see it.
        payload = {"data": [{"id": "databricks-claude-fable-5"}]}
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: (payload, None))

        models, ids, reason = db_mod.discover_claude_models(WS, "token")

        assert reason is None
        assert models == {}
        assert ids == ["databricks-claude-fable-5"]


ANTHROPIC = "anthropic/v1/messages"
RESPONSES = "openai/v1/responses"
GEMINI = "gemini/v1/generateContent"
MLFLOW_CHAT = "mlflow/v1/chat/completions"


def _model_service(model_id: str, api_types: list[str] | None = None) -> dict:
    """A model-services entry whose `name` strips to `model_id`.

    Omitting `api_types` models a workspace whose API predates
    `supported_api_types`, exercising the name-rule fallback.
    """
    entry: dict = {"name": f"model-services/{model_id}"}
    if api_types is not None:
        entry["supported_api_types"] = api_types
    return entry


class TestModelTokenLimits:
    def test_glm_is_capped(self):
        assert db_mod.model_token_limits("system.ai.glm-5-2") == {
            "context": 200_000,
            "output": 25_000,
        }

    def test_glm_matches_any_version(self):
        assert db_mod.model_token_limits("system.ai.glm-4-6-flash") == {
            "context": 200_000,
            "output": 25_000,
        }

    def test_uncapped_model_returns_none(self):
        assert db_mod.model_token_limits("system.ai.kimi-k2-7-code") is None


class TestDiscoverModelServicesByCapability:
    """Bucketing driven by each service's advertised `supported_api_types`."""

    def test_buckets_by_advertised_api_type(self, monkeypatch):
        payload = {
            "model_services": [
                _model_service("system.ai.claude-opus-4-8", [MLFLOW_CHAT, ANTHROPIC]),
                _model_service("system.ai.claude-sonnet-5", [MLFLOW_CHAT, ANTHROPIC]),
                _model_service("system.ai.gpt-5-6", [RESPONSES]),
                _model_service("system.ai.gemini-3-5-flash", [GEMINI]),
                _model_service("system.ai.glm-5-2", [MLFLOW_CHAT]),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        found = db_mod.discover_model_services(WS, "token")

        assert found.reason is None
        # Grouped by family prefix, newest-first within each family.
        assert found.claude_ids == ["system.ai.claude-opus-4-8", "system.ai.claude-sonnet-5"]
        assert found.codex_models == ["system.ai.gpt-5-6"]
        assert found.gemini_models == ["system.ai.gemini-3-5-flash"]
        assert found.oss_models == ["system.ai.glm-5-2"]

    def test_claude_ids_newest_first_within_a_family(self, monkeypatch):
        payload = {
            "model_services": [
                _model_service("system.ai.claude-opus-4-6", [ANTHROPIC]),
                _model_service("system.ai.claude-opus-4-8", [ANTHROPIC]),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        found = db_mod.discover_model_services(WS, "token")

        assert found.claude_ids == ["system.ai.claude-opus-4-8", "system.ai.claude-opus-4-6"]

    def test_gpt_oss_is_not_a_codex_model(self, monkeypatch):
        # gpt-oss-* only speaks mlflow chat-completions. Bucketing it as a
        # Responses model points pi's databricks-openai provider (and claude's
        # web_search MCP) at /ai-gateway/codex/v1, which rejects it.
        payload = {
            "model_services": [
                _model_service("system.ai.gpt-oss-120b", [MLFLOW_CHAT]),
                _model_service("system.ai.gpt-oss-20b", [MLFLOW_CHAT]),
                _model_service("system.ai.claude-opus-4-8", [MLFLOW_CHAT, ANTHROPIC]),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        found = db_mod.discover_model_services(WS, "token")

        assert found.codex_models == []
        # Not allowlisted for opencode either, so it lands nowhere.
        assert found.oss_models == []
        assert found.claude_ids == ["system.ai.claude-opus-4-8"]

    def test_claude_model_is_not_also_bucketed_as_oss(self, monkeypatch):
        # Every Claude model advertises mlflow chat-completions too; precedence
        # must keep it out of the oss bucket.
        payload = {
            "model_services": [
                _model_service("system.ai.claude-opus-4-8", [MLFLOW_CHAT, ANTHROPIC]),
                _model_service("system.ai.glm-5-2", [MLFLOW_CHAT]),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        found = db_mod.discover_model_services(WS, "token")

        assert found.oss_models == ["system.ai.glm-5-2"]
        assert found.claude_ids == ["system.ai.claude-opus-4-8"]

    def test_returns_every_claude_id_and_newest_per_family(self, monkeypatch):
        # claude-fable-5 belongs to no opus/sonnet/haiku family, so it is
        # reachable only through the full id list.
        payload = {
            "model_services": [
                _model_service(f"system.ai.{m}", [ANTHROPIC])
                for m in (
                    "claude-fable-5",
                    "claude-haiku-4-5",
                    "claude-opus-4-6",
                    "claude-opus-4-8",
                    "claude-sonnet-4-5",
                    "claude-sonnet-5",
                )
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        found = db_mod.discover_model_services(WS, "token")

        assert len(found.claude_ids) == 6
        assert "system.ai.claude-fable-5" in found.claude_ids
        assert found.claude_models == {
            "opus": "system.ai.claude-opus-4-8",
            "sonnet": "system.ai.claude-sonnet-5",
            "haiku": "system.ai.claude-haiku-4-5",
        }

    def test_responses_model_included_regardless_of_name(self, monkeypatch):
        # Capability wins over the name rule when the payload advertises one.
        payload = {"model_services": [_model_service("system.ai.o4-turbo", [RESPONSES])]}
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        assert db_mod.discover_model_services(WS, "token").codex_models == ["system.ai.o4-turbo"]


class TestDiscoverModelServices:
    """Name-rule fallback: workspaces whose payload omits supported_api_types."""

    def test_buckets_families_by_name(self, monkeypatch):
        payload = {
            "model_services": [
                _model_service("system.ai.claude-opus-4-7"),
                _model_service("system.ai.claude-opus-4-8"),
                _model_service("system.ai.claude-sonnet-4-6"),
                _model_service("system.ai.gpt-5"),
                _model_service("system.ai.gemini-2-5-flash"),
                _model_service("system.ai.gemini-3-5-flash"),
                _model_service("system.ai.kimi-k2-7-code"),
                _model_service("system.ai.glm-5-2"),
                _model_service("system.ai.llama-4-maverick"),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        found = db_mod.discover_model_services(WS, "token")

        assert found.reason is None
        # Newest opus wins; sonnet bucketed; haiku absent.
        assert found.claude_models == {
            "opus": "system.ai.claude-opus-4-8",
            "sonnet": "system.ai.claude-sonnet-4-6",
        }
        assert found.codex_models == ["system.ai.gpt-5"]
        # Gemini ordered newest-first via the shared sort key.
        assert found.gemini_models[0] == "system.ai.gemini-3-5-flash"
        # kimi and glm are the allowlisted OSS families; llama is not.
        assert found.oss_models == ["system.ai.glm-5-2", "system.ai.kimi-k2-7-code"]

    def test_gpt_oss_excluded_without_capability_data(self, monkeypatch):
        # The name rule requires a version digit after `gpt-`, so a workspace
        # with no supported_api_types still keeps gpt-oss out of codex.
        payload = {
            "model_services": [
                _model_service("system.ai.gpt-oss-120b"),
                _model_service("system.ai.gpt-5-6"),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        found = db_mod.discover_model_services(WS, "token")

        assert found.codex_models == ["system.ai.gpt-5-6"]

    def test_codex_models_sorted_newest_first(self, monkeypatch):
        payload = {
            "model_services": [
                _model_service("system.ai.gpt-4-1"),
                _model_service("system.ai.gpt-5-6"),
                _model_service("system.ai.gpt-5-2-codex"),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        found = db_mod.discover_model_services(WS, "token")

        # Alphabetical order would put gpt-4-1 first; pi and copilot take [0].
        assert found.codex_models[0] == "system.ai.gpt-5-6"

    def test_two_digit_minor_version_beats_single_digit(self, monkeypatch):
        # Lexicographic reverse-sort ranks `4-8` above `4-10`.
        payload = {
            "model_services": [
                _model_service("system.ai.claude-opus-4-8"),
                _model_service("system.ai.claude-opus-4-10"),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        found = db_mod.discover_model_services(WS, "token")

        assert found.claude_models["opus"] == "system.ai.claude-opus-4-10"

    def test_single_segment_version_buckets_into_family(self, monkeypatch):
        # `claude-sonnet-5` has no dotted minor; it must still be a sonnet.
        payload = {
            "model_services": [
                _model_service("system.ai.claude-sonnet-4-6"),
                _model_service("system.ai.claude-sonnet-5"),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        found = db_mod.discover_model_services(WS, "token")

        assert found.claude_models["sonnet"] == "system.ai.claude-sonnet-5"

    def test_oss_allowlist_drops_unsupported_families(self, monkeypatch):
        # Only kimi/glm are allowlisted; other families are dropped.
        payload = {
            "model_services": [
                _model_service("system.ai.glm-5-2"),
                _model_service("system.ai.kimi-k2-7-code"),
                _model_service("system.ai.qwen-3-coder"),
                _model_service("system.ai.deepseek-v3"),
                _model_service("system.ai.gte-large-embed"),
                _model_service("system.ai.bge-reranker-v2"),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        found = db_mod.discover_model_services(WS, "token")

        assert found.reason is None
        assert (found.claude_models, found.codex_models, found.gemini_models) == ({}, [], [])
        assert found.oss_models == ["system.ai.glm-5-2", "system.ai.kimi-k2-7-code"]

    def test_paginates_via_next_page_token(self, monkeypatch):
        pages = {
            None: {
                "model_services": [_model_service("system.ai.gpt-5")],
                "next_page_token": "tok2",
            },
            "tok2": {
                "model_services": [_model_service("system.ai.claude-opus-4-8")],
            },
        }

        def fake_get(url, token, timeout=10):
            token_param = None
            if "page_token=" in url:
                token_param = url.split("page_token=")[1].split("&")[0]
            return pages[token_param], None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        found = db_mod.discover_model_services(WS, "token")

        assert found.reason is None
        assert found.codex_models == ["system.ai.gpt-5"]
        assert found.claude_models == {"opus": "system.ai.claude-opus-4-8"}

    def test_http_failure_returns_reason(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (None, "HTTP 500 Server Error")
        )

        found = db_mod.discover_model_services(WS, "token")

        assert found.claude_models == {}
        assert (found.claude_ids, found.codex_models, found.gemini_models) == ([], [], [])
        assert found.oss_models == []
        assert found.reason == "HTTP 500 Server Error"

    def test_no_matching_families_reports_sample(self, monkeypatch):
        payload = {"model_services": [_model_service("system.ai.llama-4-maverick")]}
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        found = db_mod.discover_model_services(WS, "token")

        assert found.claude_models == {}
        assert (found.claude_ids, found.codex_models, found.gemini_models) == ([], [], [])
        assert found.oss_models == []
        assert found.reason is not None and "llama-4-maverick" in found.reason

    def test_ignores_non_system_ai_schemas(self, monkeypatch):
        # The metastore listing returns services from every schema; only
        # system.ai.* foundation models should be picked up.
        payload = {
            "model_services": [
                _model_service("system.ai.gpt-5"),
                _model_service("main.svenwb.gpt-5-5"),
                _model_service("temp.erni.kimi-k2-7-code"),
                _model_service("temp.erni.claude-opus-4-8"),
                _model_service("dnasi_agent_cuj.default.dnasi-gpt55-test"),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        found = db_mod.discover_model_services(WS, "token")

        assert found.reason is None
        assert found.codex_models == ["system.ai.gpt-5"]
        assert found.claude_models == {}  # temp.erni.claude-* must not be bucketed
        assert found.gemini_models == []
        assert found.oss_models == []

    def test_requests_bounded_page_size(self, monkeypatch):
        # The endpoint 499s without a bounded page_size, so every request must
        # carry one.
        urls: list[str] = []

        def fake_get(url, token, timeout=10):
            urls.append(url)
            return {"model_services": [_model_service("system.ai.gpt-5", [RESPONSES])]}, None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        services, reason = db_mod.list_model_services(WS, "token")

        assert services == {"system.ai.gpt-5": [RESPONSES]}
        assert reason is None
        assert all("page_size=" in u for u in urls)

    def test_missing_api_types_yields_empty_capability_list(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=10: (
                {"model_services": [_model_service("system.ai.x")]},
                None,
            ),
        )

        services, reason = db_mod.list_model_services(WS, "token")

        assert services == {"system.ai.x": []}
        assert reason is None

    def test_retries_page_before_giving_up(self, monkeypatch):
        payload = {"model_services": [_model_service("system.ai.gpt-5")]}
        calls = {"n": 0}

        def flaky_get(url, token, timeout=10):
            calls["n"] += 1
            if calls["n"] < 3:
                return None, "HTTP 499 Unknown"
            return payload, None

        monkeypatch.setattr(db_mod, "_http_get_json", flaky_get)

        services, reason = db_mod.list_model_services(WS, "token")

        assert reason is None
        assert list(services) == ["system.ai.gpt-5"]
        assert calls["n"] == 3  # two failures, third succeeds


def _serving_endpoint(
    name: str,
    *,
    endpoint_type: str = "EXTERNAL_MODEL",
    task: str = "llm/v1/chat",
    ready: str | None = "READY",
) -> dict:
    """A `/api/2.0/serving-endpoints` list entry matching the live API shape."""
    entry: dict = {"name": name, "endpoint_type": endpoint_type, "task": task}
    if ready is not None:
        entry["state"] = {"ready": ready}
    return entry


class TestDiscoverExternalServingModels:
    """Surfacing external-model chat serving endpoints reached via /serving-endpoints."""

    def test_surfaces_only_ready_external_chat_endpoints(self, monkeypatch):
        payload = {
            "endpoints": [
                _serving_endpoint("azure-foundry-gpt-5-6-tera"),
                _serving_endpoint("azure-foundry-gpt-5-5"),
                # Same task, but a Databricks-managed foundation model — excluded.
                _serving_endpoint("databricks-gpt-oss-120b", endpoint_type="FOUNDATION_MODEL_API"),
                # A ready chat endpoint of a non-external type (a user's own custom
                # model). Not reached by the OpenAI-external route — excluded. This
                # pins the filter to `== EXTERNAL_MODEL`, not merely `!= foundation`.
                _serving_endpoint("my-custom-chat", endpoint_type="CUSTOM_MODEL"),
                # External, but an embeddings endpoint — excluded.
                _serving_endpoint("azure-foundry-embed", task="llm/v1/embeddings"),
                # External chat, but not ready — excluded.
                _serving_endpoint("azure-foundry-warming", ready="NOT_READY"),
            ]
        }
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: (payload, None))

        models, reason = db_mod.discover_external_serving_models(WS, "token")

        assert reason is None
        assert models == ["azure-foundry-gpt-5-5", "azure-foundry-gpt-5-6-tera"]

    def test_foundation_endpoint_is_excluded_near_miss(self, monkeypatch):
        # Near-miss: an entry identical to a surfaced one except endpoint_type.
        payload = {"endpoints": [_serving_endpoint("x", endpoint_type="FOUNDATION_MODEL_API")]}
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: (payload, None))

        models, reason = db_mod.discover_external_serving_models(WS, "token")

        assert models == []
        assert reason == "no ready external-model chat serving endpoints found"

    def test_non_chat_task_is_excluded_near_miss(self, monkeypatch):
        # Near-miss: external endpoint that speaks a non-chat task.
        payload = {"endpoints": [_serving_endpoint("x", task="llm/v1/completions")]}
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: (payload, None))

        models, _ = db_mod.discover_external_serving_models(WS, "token")

        assert models == []

    def test_missing_state_is_treated_as_available(self, monkeypatch):
        payload = {"endpoints": [_serving_endpoint("x", ready=None)]}
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: (payload, None))

        models, reason = db_mod.discover_external_serving_models(WS, "token")

        assert models == ["x"]
        assert reason is None

    def test_http_failure_returns_reason(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token: (None, "HTTP 500 Server Error")
        )

        models, reason = db_mod.discover_external_serving_models(WS, "token")

        assert models == []
        assert reason == "HTTP 500 Server Error"


class TestListModelProviderServices:
    _PAYLOAD = {
        "model_provider_services": [
            {
                "name": "model-provider-services/main.aarushi.anthropic-svc",
                "config": {"provider_type": "EXTERNAL_MODEL_PROVIDER_TYPE_ANTHROPIC"},
            },
            {
                "name": "model-provider-services/main.aarushi.openai-svc",
                "config": {"provider_type": "EXTERNAL_MODEL_PROVIDER_TYPE_OPENAI"},
            },
            {
                "name": "model-provider-services/main.bob.bedrock-svc",
                "config": {
                    "provider_type": "EXTERNAL_MODEL_PROVIDER_TYPE_AMAZON_BEDROCK",
                    "allow_all_targets": False,
                    "targets": [
                        {
                            "model": "us.anthropic.claude-sonnet-4-6",
                            "native_api_types": ["anthropic/v1/messages"],
                        },
                        {"model": "global.anthropic.claude-opus-4-8"},
                    ],
                },
            },
            {
                "name": "model-provider-services/main.bob.bedrock-titan-svc",
                "config": {
                    "provider_type": "EXTERNAL_MODEL_PROVIDER_TYPE_AMAZON_BEDROCK",
                    "targets": [{"model": "amazon.titan-text-express-v1"}],
                },
            },
        ]
    }

    def test_strips_prefix_and_tags_provider_type(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (self._PAYLOAD, None)
        )
        services, reason = db_mod.list_model_provider_services(WS, "token")
        assert reason is None
        assert services[0] == {
            "name": "main.aarushi.anthropic-svc",
            "provider_type": "anthropic",
            "targets": [],
            "allow_all_targets": False,
        }
        assert {s["provider_type"] for s in services} == {
            "anthropic",
            "openai",
            "amazon_bedrock",
        }

    def test_extracts_targets(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (self._PAYLOAD, None)
        )
        services, _ = db_mod.list_model_provider_services(WS, "token")
        bedrock = next(s for s in services if s["name"] == "main.bob.bedrock-svc")
        assert bedrock["targets"] == [
            "us.anthropic.claude-sonnet-4-6",
            "global.anthropic.claude-opus-4-8",
        ]

    def test_returns_reason_on_failure(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (None, "HTTP 500 Server Error")
        )
        services, reason = db_mod.list_model_provider_services(WS, "token")
        assert services == []
        assert reason == "HTTP 500 Server Error"

    def test_claude_includes_anthropic_and_usable_bedrock(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (self._PAYLOAD, None)
        )
        names, reason = db_mod.list_tool_provider_services("claude", WS, "token")
        assert reason is None
        # Anthropic + the Bedrock service with Claude targets; the Bedrock service
        # exposing only Titan is hidden (no Claude models to pin).
        assert names == ["main.aarushi.anthropic-svc", "main.bob.bedrock-svc"]

    def test_codex_filters_to_openai(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (self._PAYLOAD, None)
        )
        names, _ = db_mod.list_tool_provider_services("codex", WS, "token")
        assert names == ["main.aarushi.openai-svc"]


class TestMapBedrockClaudeModels:
    def test_maps_families(self):
        models = db_mod.map_bedrock_claude_models(
            [
                "us.anthropic.claude-sonnet-4-6",
                "global.anthropic.claude-opus-4-8",
                "anthropic.claude-haiku-4-5",
                "amazon.titan-text-express-v1",
            ]
        )
        assert models == {
            "sonnet": "us.anthropic.claude-sonnet-4-6",
            "opus": "global.anthropic.claude-opus-4-8",
            "haiku": "anthropic.claude-haiku-4-5",
        }

    def test_prefers_highest_version(self):
        models = db_mod.map_bedrock_claude_models(
            ["us.anthropic.claude-sonnet-4-5", "us.anthropic.claude-sonnet-4-6"]
        )
        assert models["sonnet"] == "us.anthropic.claude-sonnet-4-6"

    def test_region_tie_break_prefers_global(self):
        models = db_mod.map_bedrock_claude_models(
            [
                "us.anthropic.claude-opus-4-8",
                "global.anthropic.claude-opus-4-8",
                "eu.anthropic.claude-opus-4-8",
            ]
        )
        assert models["opus"] == "global.anthropic.claude-opus-4-8"

    def test_empty_when_no_claude(self):
        assert db_mod.map_bedrock_claude_models(["amazon.titan-text-express-v1"]) == {}


class TestResolveProviderService:
    _PAYLOAD = TestListModelProviderServices._PAYLOAD

    def _patch(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (self._PAYLOAD, None)
        )

    def test_anthropic_ok(self, monkeypatch):
        self._patch(monkeypatch)
        service, error = db_mod.resolve_provider_service(
            "claude", "main.aarushi.anthropic-svc", WS, "token"
        )
        assert error is None
        assert service["provider_type"] == "anthropic"

    def test_bedrock_with_claude_ok(self, monkeypatch):
        self._patch(monkeypatch)
        service, error = db_mod.resolve_provider_service(
            "claude", "main.bob.bedrock-svc", WS, "token"
        )
        assert error is None
        assert service["provider_type"] == "amazon_bedrock"

    def test_wrong_type_rejected(self, monkeypatch):
        self._patch(monkeypatch)
        service, error = db_mod.resolve_provider_service(
            "claude", "main.aarushi.openai-svc", WS, "token"
        )
        assert service is None
        assert "can't route to" in error

    def test_bedrock_without_claude_rejected(self, monkeypatch):
        self._patch(monkeypatch)
        service, error = db_mod.resolve_provider_service(
            "claude", "main.bob.bedrock-titan-svc", WS, "token"
        )
        assert service is None
        assert "no Claude models" in error

    def test_not_found_lists_usable(self, monkeypatch):
        self._patch(monkeypatch)
        service, error = db_mod.resolve_provider_service("claude", "main.x.missing", WS, "token")
        assert service is None
        assert "was not found" in error
        assert "main.aarushi.anthropic-svc" in error

    def test_feature_unavailable(self, monkeypatch):
        reason = "HTTP 400 Bad Request: ModelProviderService feature is not available"
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token, timeout=30: (None, reason))
        service, error = db_mod.resolve_provider_service("claude", "main.x.y", WS, "token")
        assert service is None
        assert "not available" in error


class TestModelProviderFeatureUnavailable:
    def test_detects_feature_not_available(self):
        reason = (
            'HTTP 400 Bad Request: {"error_code":"BAD_REQUEST",'
            '"message":"ModelProviderService feature is not available"}'
        )
        assert db_mod.is_model_provider_feature_unavailable(reason) is True

    def test_false_for_other_errors(self):
        assert db_mod.is_model_provider_feature_unavailable("HTTP 500 Server Error") is False
        assert db_mod.is_model_provider_feature_unavailable(None) is False


class TestListMcpServices:
    def test_accepts_entries_without_connection_status(self, monkeypatch):
        payload = {
            "mcp_services": [
                {
                    "name": "mcp-services/system.ai.github",
                    "config": {"usage_tracking": {"enabled": True}, "tracing": {"enabled": True}},
                },
                {
                    "name": "mcp-services/system.ai.atlassian",
                    "config": {},
                },
                {
                    "name": "mcp-services/system.ai.slack",
                },
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

        names, reason = db_mod.list_mcp_services(WS, "token")

        assert reason is None
        assert names == ["system.ai.atlassian", "system.ai.github", "system.ai.slack"]

    def test_accepts_legacy_active_status(self, monkeypatch):
        payload = {
            "mcp_services": [
                {
                    "name": "mcp-services/system.ai.github",
                    "config": {"connection": {"status": "ACTIVE"}},
                },
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

        names, reason = db_mod.list_mcp_services(WS, "token")

        assert reason is None
        assert names == ["system.ai.github"]

    def test_rejects_explicit_non_active_status(self, monkeypatch):
        # If the field is present and non-ACTIVE, drop the entry — the
        # backing connection is broken and the proxy will fail.
        payload = {
            "mcp_services": [
                {
                    "name": "mcp-services/system.ai.github",
                    "config": {"connection": {"status": "ACTIVE"}},
                },
                {
                    "name": "mcp-services/system.ai.broken",
                    "config": {"connection": {"status": "FAILED"}},
                },
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

        names, _reason = db_mod.list_mcp_services(WS, "token")

        assert names == ["system.ai.github"]

    def test_ignores_non_system_ai_entries(self, monkeypatch):
        payload = {
            "mcp_services": [
                {"name": "mcp-services/system.ai.github"},
                {"name": "mcp-services/main.svenwb.github_mcp"},
                {"name": "mcp-services/temp.erni.github_mcp"},
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

        names, _reason = db_mod.list_mcp_services(WS, "token")

        assert names == ["system.ai.github"]

    def test_http_failure_propagates_reason(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=30: (None, "HTTP 500 Server Error"),
        )

        names, reason = db_mod.list_mcp_services(WS, "token")

        assert names == []
        assert reason == "HTTP 500 Server Error"

    def test_empty_payload_is_successful_with_no_reason(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: ({"mcp_services": []}, None)
        )

        names, reason = db_mod.list_mcp_services(WS, "token")

        assert names == []
        assert reason is None

    def test_custom_parent_passes_through_to_url(self, monkeypatch):
        captured: dict[str, str] = {}

        def fake_get(url, token, timeout=30):
            captured["url"] = url
            return {"mcp_services": []}, None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        db_mod.list_mcp_services(WS, "token", parent="main.svenwb")

        assert "parent=schemas%2Fmain.svenwb" in captured["url"]

    def test_custom_parent_filters_to_namespace(self, monkeypatch):
        payload = {
            "mcp_services": [
                {"name": "mcp-services/main.svenwb.github"},
                {"name": "mcp-services/main.svenwb.slack"},
                {"name": "mcp-services/system.ai.github"},
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

        names, reason = db_mod.list_mcp_services(WS, "token", parent="main.svenwb")

        assert reason is None
        assert names == ["main.svenwb.github", "main.svenwb.slack"]

    def test_http_404_reason_surfaces_for_invalid_parent(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=30: (None, "HTTP 404 Not Found: NOT_FOUND"),
        )

        names, reason = db_mod.list_mcp_services(WS, "token", parent="nope.nope")

        assert names == []
        assert reason and reason.startswith("HTTP 404")


def _foundation_models_payload(names):
    return {
        "endpoints": [
            {
                "name": name,
                "config": {
                    "served_entities": [
                        {
                            "foundation_model": {
                                "ai_gateway_v2_supported": True,
                                "api_types": ["gemini/v1/generateContent"],
                            }
                        }
                    ]
                },
            }
            for name in names
        ]
    }


class TestModelVersionSortKey:
    def test_orders_newest_version_first(self):
        names = [
            "databricks-gemini-2-5-flash",
            "databricks-gemini-2-5-pro",
            "databricks-gemini-3-1-flash-lite",
            "databricks-gemini-3-1-pro",
            "databricks-gemini-3-5-flash",
            "databricks-gemini-3-flash",
            "databricks-gemini-3-pro",
        ]
        ordered = sorted(names, key=db_mod.model_version_sort_key)
        assert ordered[0] == "databricks-gemini-3-5-flash"

    def test_treats_bare_major_as_dot_zero(self):
        # 3-flash is 3.0, so 3-5-flash (3.5) must sort ahead of it.
        names = ["databricks-gemini-3-flash", "databricks-gemini-3-5-flash"]
        ordered = sorted(names, key=db_mod.model_version_sort_key)
        assert ordered == [
            "databricks-gemini-3-5-flash",
            "databricks-gemini-3-flash",
        ]

    def test_unversioned_names_sort_last_alphabetically(self):
        names = ["databricks-gemini-2-5-flash", "custom-endpoint", "another-endpoint"]
        ordered = sorted(names, key=db_mod.model_version_sort_key)
        assert ordered[0] == "databricks-gemini-2-5-flash"
        assert ordered[1:] == ["another-endpoint", "custom-endpoint"]


class TestDiscoverGeminiModels:
    def test_returns_newest_flash_first(self, monkeypatch):
        payload = _foundation_models_payload(
            [
                "databricks-gemini-2-5-flash",
                "databricks-gemini-3-5-flash",
                "databricks-gemini-3-flash",
            ]
        )
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: (payload, None))

        models, reason = db_mod.discover_gemini_models(WS, "token")

        assert reason is None
        assert models[0] == "databricks-gemini-3-5-flash"

    def test_codex_discovery_keeps_alphabetical_order(self, monkeypatch):
        # Codex passes no sort_key, so ordering must stay the plain alphabetical
        # default — guarding against the gemini change leaking across tools.
        payload = {
            "endpoints": [
                {
                    "name": name,
                    "config": {
                        "served_entities": [
                            {
                                "foundation_model": {
                                    "ai_gateway_v2_supported": True,
                                    "api_types": ["openai/v1/responses"],
                                }
                            }
                        ]
                    },
                }
                for name in ["databricks-gpt-5-2-codex", "databricks-gpt-4-1"]
            ]
        }
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: (payload, None))

        models, reason = db_mod.discover_codex_models(WS, "token")

        assert reason is None
        assert models == ["databricks-gpt-4-1", "databricks-gpt-5-2-codex"]


class TestResolvePatToken:
    def test_reads_pat_profile_token_from_cfg(self, monkeypatch, tmp_path):
        cfg = tmp_path / "databrickscfg"
        cfg.write_text(f"[lakebox]\nhost = {WS}\ntoken = dapi-from-cfg\n")
        monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(cfg))
        monkeypatch.setattr(
            db_mod,
            "list_profile_entries",
            lambda: [{"name": "lakebox", "host": WS, "auth_type": "pat"}],
        )
        assert db_mod.resolve_pat_token("lakebox") == "dapi-from-cfg"

    def test_default_section_token_does_not_leak_into_named_profiles(self, monkeypatch, tmp_path):
        cfg = tmp_path / "databrickscfg"
        cfg.write_text(
            f"[DEFAULT]\nhost = {WS}\ntoken = dapi-default\n"
            "[other]\nhost = https://other.databricks.com\n"
        )
        monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(cfg))
        monkeypatch.setattr(
            db_mod,
            "list_profile_entries",
            lambda: [
                {"name": "DEFAULT", "host": WS, "auth_type": "pat"},
                {"name": "other", "host": "https://other.databricks.com", "auth_type": "pat"},
            ],
        )
        assert db_mod.resolve_pat_token("DEFAULT") == "dapi-default"
        assert db_mod.resolve_pat_token("other") is None

    def test_returns_none_for_oauth_profile(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "list_profile_entries",
            lambda: [{"name": "oauth", "host": WS, "auth_type": "databricks-cli"}],
        )
        assert db_mod.resolve_pat_token("oauth") is None

    def test_returns_none_without_profile(self):
        assert db_mod.resolve_pat_token(None) is None


class TestApplyPatEnvironment:
    @pytest.fixture(autouse=True)
    def _isolated_bearer(self):
        # apply_pat_environment writes os.environ directly; restore it even
        # though monkeypatch can't track writes made by code under test.
        original = os.environ.pop("DATABRICKS_BEARER", None)
        yield
        if original is None:
            os.environ.pop("DATABRICKS_BEARER", None)
        else:
            os.environ["DATABRICKS_BEARER"] = original

    def test_exports_bearer_for_use_pat_state(self, monkeypatch):
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")

        db_mod.apply_pat_environment({"use_pat": True, "profile": "DEFAULT"})

        assert os.environ["DATABRICKS_BEARER"] == "dapi-pat"

    def test_noop_without_use_pat(self, monkeypatch):
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")

        db_mod.apply_pat_environment({"profile": "DEFAULT"})

        assert "DATABRICKS_BEARER" not in os.environ

    def test_existing_bearer_wins(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_BEARER", "explicit-bearer")
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")

        db_mod.apply_pat_environment({"use_pat": True, "profile": "DEFAULT"})

        assert os.environ["DATABRICKS_BEARER"] == "explicit-bearer"


class TestBuildAuthTokenArgv:
    def test_basic_argv(self):
        argv = build_auth_token_argv(WS)
        # First element resolves to the ucode executable; the rest is the
        # cross-platform helper invocation — no `sh`, no `jq`, no shell syntax.
        assert argv[0].endswith("ucode") or argv[0] == "ucode"
        assert argv[1:] == ["auth-token", "--host", WS]

    def test_strips_trailing_slash_from_host(self):
        argv = build_auth_token_argv(WS + "/")
        assert "--host" in argv
        assert argv[argv.index("--host") + 1] == WS

    def test_embeds_profile_when_provided(self):
        argv = build_auth_token_argv(WS, profile="stablebox")
        assert argv[argv.index("--profile") + 1] == "stablebox"

    def test_profile_passed_as_separate_argv_element(self):
        # Metacharacters need no shell quoting — argv is never parsed by a shell.
        argv = build_auth_token_argv(WS, profile="weird name; rm -rf /")
        assert "weird name; rm -rf /" in argv

    def test_use_pat_flag(self):
        argv = build_auth_token_argv(WS, profile="DEFAULT", use_pat=True)
        assert "--use-pat" in argv
        assert argv[argv.index("--profile") + 1] == "DEFAULT"

    def test_no_use_pat_flag_by_default(self):
        assert "--use-pat" not in build_auth_token_argv(WS)


class TestBuildAuthShellCommand:
    def test_contains_workspace(self):
        cmd = build_auth_shell_command(WS)
        assert WS in cmd

    def test_is_ucode_auth_token_invocation(self):
        # The persisted helper now points at the `ucode auth-token` executable
        # on every platform — not a POSIX `databricks ... | jq` pipeline.
        cmd = build_auth_shell_command(WS)
        assert "auth-token" in cmd
        assert "--host" in cmd
        # POSIX-only constructs that broke Windows (#116) must be gone.
        assert "jq" not in cmd
        assert "if [ -n" not in cmd

    def test_embeds_profile_when_provided(self):
        cmd = build_auth_shell_command(WS, profile="stablebox")
        assert "--profile stablebox" in cmd

    def test_quotes_profile_shell_metacharacters(self):
        cmd = build_auth_shell_command(WS, profile="weird name; rm -rf /")
        # On POSIX shlex.join quotes the value so the string form cannot be
        # interpreted as a shell injection if a tool runs it via a shell.
        if os.name != "nt":
            assert "'weird name; rm -rf /'" in cmd

    def test_use_pat_emits_flag(self):
        cmd = build_auth_shell_command(WS, profile="DEFAULT", use_pat=True)
        assert "--use-pat" in cmd
        assert "--profile DEFAULT" in cmd


class TestEnsurePatBearer:
    """ensure_pat_bearer is the empty-aware DATABRICKS_BEARER export used by the
    --use-pat path on configure, launch, and the auth-token helper."""

    @pytest.fixture(autouse=True)
    def _isolated_bearer(self):
        # ensure_pat_bearer writes os.environ directly; restore it even though
        # monkeypatch can't track writes made by code under test.
        original = os.environ.pop("DATABRICKS_BEARER", None)
        yield
        if original is None:
            os.environ.pop("DATABRICKS_BEARER", None)
        else:
            os.environ["DATABRICKS_BEARER"] = original

    def test_exports_pat_when_env_absent(self, monkeypatch):
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")
        assert ensure_pat_bearer("p") is True
        assert os.environ["DATABRICKS_BEARER"] == "dapi-pat"

    def test_overwrites_empty_env(self, monkeypatch):
        # The regression: an empty DATABRICKS_BEARER must be treated as absent
        # so the PAT is still exported (old `if [ -n ... ]` parity).
        monkeypatch.setenv("DATABRICKS_BEARER", "")
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")
        assert ensure_pat_bearer("p") is True
        assert os.environ["DATABRICKS_BEARER"] == "dapi-pat"

    def test_non_empty_env_wins_without_resolving(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_BEARER", "ci-bearer")
        called = []
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: called.append(p) or "dapi-pat")
        assert ensure_pat_bearer("p") is True
        # Pre-set bearer is honored; we don't even read the PAT.
        assert os.environ["DATABRICKS_BEARER"] == "ci-bearer"
        assert called == []

    def test_returns_false_when_no_pat(self, monkeypatch):
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: None)
        assert ensure_pat_bearer("p") is False
        assert "DATABRICKS_BEARER" not in os.environ

    def test_whitespace_only_env_treated_as_empty(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_BEARER", "   ")
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")
        assert ensure_pat_bearer("p") is True
        assert os.environ["DATABRICKS_BEARER"] == "dapi-pat"

    def test_explicit_pat_arg_skips_cfg_read(self, monkeypatch):
        # Callers that already resolved the PAT (configure_shared_state) pass it
        # in; ensure_pat_bearer must use it without re-reading ~/.databrickscfg.
        called = []
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: called.append(p) or "from-cfg")
        assert ensure_pat_bearer("p", "explicit-pat") is True
        assert os.environ["DATABRICKS_BEARER"] == "explicit-pat"
        assert called == []


class TestFormatSubprocessResult:
    def test_suppresses_stdout_on_success(self):
        result = subprocess.CompletedProcess(
            args=["databricks", "auth", "token"],
            returncode=0,
            stdout='{"access_token": "dapi-secret-do-not-leak", "token_type": "Bearer"}',
            stderr="",
        )
        formatted = _format_subprocess_result(result)
        assert "dapi-secret-do-not-leak" not in formatted
        assert "rc=0" in formatted

    def test_includes_stdout_on_failure(self):
        result = subprocess.CompletedProcess(
            args=["databricks", "auth", "token"],
            returncode=1,
            stdout="useful diagnostic output",
            stderr="error: no matching profile",
        )
        formatted = _format_subprocess_result(result)
        assert "rc=1" in formatted
        assert "useful diagnostic output" in formatted
        assert "no matching profile" in formatted


class TestScrubDatabrickscfg:
    def test_redacts_token_value(self):
        text = "[DEFAULT]\nhost = https://example.databricks.com\ntoken = dapi-secret\n"
        scrubbed = _scrub_databrickscfg(text)
        assert "dapi-secret" not in scrubbed
        assert "token = <redacted>" in scrubbed
        assert "host = https://example.databricks.com" in scrubbed

    def test_redacts_various_secret_keys(self):
        text = (
            "[p]\n"
            "client_secret = secret-val-1\n"
            "bearer_token = secret-val-2\n"
            "api_key = secret-val-3\n"
            "password = secret-val-4\n"
            "auth_type = oauth-u2m\n"
        )
        scrubbed = _scrub_databrickscfg(text)
        for secret in ("secret-val-1", "secret-val-2", "secret-val-3", "secret-val-4"):
            assert secret not in scrubbed
        assert "auth_type = oauth-u2m" in scrubbed

    def test_preserves_comments_and_sections(self):
        text = "# comment\n[DEFAULT]\nhost = https://x\n; another comment with token = leak\n"
        scrubbed = _scrub_databrickscfg(text)
        assert "# comment" in scrubbed
        assert "[DEFAULT]" in scrubbed
        assert "; another comment with token = leak" in scrubbed

    def test_key_matching_is_case_insensitive(self):
        text = "[p]\nTOKEN = upper\nAccess_Token = mixed\n"
        scrubbed = _scrub_databrickscfg(text)
        assert "upper" not in scrubbed
        assert "mixed" not in scrubbed


class TestScrubJson:
    def test_redacts_secret_keys(self):
        payload = {
            "access_token": "dapi-secret",
            "host": "https://example.databricks.com",
        }
        scrubbed = _scrub_json(payload)
        assert isinstance(scrubbed, dict)
        assert scrubbed["access_token"] == "<redacted>"
        assert scrubbed["host"] == "https://example.databricks.com"

    def test_recurses_into_nested_structures(self):
        payload = {
            "profiles": [
                {"name": "DEFAULT", "client_secret": "abc"},
                {"name": "other", "password": "pw"},
            ]
        }
        scrubbed = _scrub_json(payload)
        assert scrubbed == {
            "profiles": [
                {"name": "DEFAULT", "client_secret": "<redacted>"},
                {"name": "other", "password": "<redacted>"},
            ]
        }

    def test_passes_through_scalars_and_non_secret_keys(self):
        assert _scrub_json("plain") == "plain"
        assert _scrub_json(42) == 42
        assert _scrub_json({"host": "x", "auth_type": "pat"}) == {
            "host": "x",
            "auth_type": "pat",
        }


class TestGetDatabricksToken:
    def _fake_databricks(self, tmp_path, script: str) -> dict:
        fake = tmp_path / "databricks"
        fake.write_text(f"#!/bin/sh\n{script}\n")
        fake.chmod(0o755)
        return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}

    def test_returns_token_on_success(self, tmp_path, monkeypatch):
        env = self._fake_databricks(
            tmp_path,
            'echo \'{"access_token": "good-token", "token_type": "Bearer"}\'',
        )
        monkeypatch.setattr("os.environ", env)
        token = get_databricks_token(WS)
        assert token == "good-token"

    def test_strips_ambient_profile_when_profile_not_provided(self, tmp_path, monkeypatch):
        profile_log = tmp_path / "profile"
        env = self._fake_databricks(
            tmp_path,
            f'printf "%s" "${{DATABRICKS_CONFIG_PROFILE:-}}" > {profile_log}\n'
            'echo \'{"access_token": "good-token", "token_type": "Bearer"}\'',
        )
        env["DATABRICKS_CONFIG_PROFILE"] = "other-workspace"
        monkeypatch.setattr("os.environ", env)

        token = get_databricks_token(WS)

        assert token == "good-token"
        assert profile_log.read_text() == ""

    def test_has_valid_auth_strips_ambient_profile_without_explicit_profile(
        self, tmp_path, monkeypatch
    ):
        profile_log = tmp_path / "profile"
        env = self._fake_databricks(
            tmp_path,
            f'printf "%s" "${{DATABRICKS_CONFIG_PROFILE:-}}" > {profile_log}\n'
            'echo \'{"access_token": "good-token", "token_type": "Bearer"}\'',
        )
        env["DATABRICKS_CONFIG_PROFILE"] = "other-workspace"
        monkeypatch.setattr("os.environ", env)

        assert db_mod.has_valid_databricks_auth(WS)
        assert profile_log.read_text() == ""

    def test_reauths_and_retries_when_token_empty(self, tmp_path, monkeypatch):
        call_count = tmp_path / "calls"
        call_count.write_text("0")
        env = self._fake_databricks(
            tmp_path,
            f"count=$(cat {call_count})\n"
            f"echo $((count + 1)) > {call_count}\n"
            'case "$*" in\n'
            '  *"auth login"*) exit 0 ;;\n'
            "esac\n"
            'if [ "$count" -eq 0 ]; then\n'
            '  echo \'{"access_token": "", "token_type": "Bearer"}\'\n'
            "else\n"
            '  echo \'{"access_token": "refreshed-token", "token_type": "Bearer"}\'\n'
            "fi",
        )
        monkeypatch.setattr("os.environ", env)
        token = get_databricks_token(WS)
        assert token == "refreshed-token"

    def test_raises_when_reauth_also_fails(self, tmp_path, monkeypatch):
        env = self._fake_databricks(
            tmp_path,
            'echo \'{"access_token": "", "token_type": "Bearer"}\'',
        )
        monkeypatch.setattr("os.environ", env)
        with pytest.raises(RuntimeError, match="no access token"):
            get_databricks_token(WS)

    def test_passes_profile_flag_when_provided(self, tmp_path, monkeypatch):
        # Fake CLI that records its argv to a file so we can assert the
        # --profile flag is forwarded to `databricks auth token`.
        argv_log = tmp_path / "argv"
        env = self._fake_databricks(
            tmp_path,
            f'printf "%s\\n" "$@" >> {argv_log}\n'
            'echo \'{"access_token": "good-token", "token_type": "Bearer"}\'',
        )
        monkeypatch.setattr("os.environ", env)
        token = get_databricks_token(WS, profile="stablebox")
        assert token == "good-token"
        argv = argv_log.read_text().splitlines()
        assert "--profile" in argv
        assert argv[argv.index("--profile") + 1] == "stablebox"

    def test_error_suggests_logout_when_matching_profile_exists(self, tmp_path, monkeypatch):
        env = self._fake_databricks(
            tmp_path,
            'case "$*" in\n'
            '  *"auth profiles"*) echo \'{"profiles": [{"host": "'
            + WS
            + '", "name": "example-profile", "auth_type": "databricks-cli"}]}\'; exit 0 ;;\n'
            '  *"auth login"*) exit 0 ;;\n'
            "esac\n"
            'echo \'{"access_token": "", "token_type": "Bearer"}\'',
        )
        monkeypatch.setattr("os.environ", env)

        with pytest.raises(RuntimeError) as exc_info:
            get_databricks_token(WS)

        message = str(exc_info.value)
        assert "stale or invalid" in message
        assert "databricks auth logout --profile example-profile" in message
        assert f"databricks auth login --host {WS} --profile example-profile" in message


class TestListDatabricksConnections:
    def test_lists_paginated_connections_with_workspace_env(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            if "--page-token" in args:
                payload = {"connections": [{"name": "jira-mcp", "connection_type": "HTTP"}]}
            else:
                payload = {
                    "connections": [{"name": "confluence-mcp", "connection_type": "HTTP"}],
                    "next_page_token": "next-page",
                }
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload))

        monkeypatch.setattr(db_mod, "run", fake_run)

        assert list_databricks_connections(WS) == [
            {"name": "confluence-mcp", "connection_type": "HTTP"},
            {"name": "jira-mcp", "connection_type": "HTTP"},
        ]
        assert calls[0]["args"] == [
            "databricks",
            "connections",
            "list",
            "--max-results",
            "0",
            "--output",
            "json",
        ]
        assert calls[0]["kwargs"]["env"]["DATABRICKS_HOST"] == WS
        assert calls[1]["args"][-2:] == ["--page-token", "next-page"]

    def test_passes_profile_when_provided(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"connections": []}))

        monkeypatch.setattr(db_mod, "run", fake_run)

        list_databricks_connections(WS, "my-profile")

        assert "--profile" in calls[0]
        assert calls[0][calls[0].index("--profile") + 1] == "my-profile"

    def test_raises_on_invalid_json(self, monkeypatch):
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="not-json")

        monkeypatch.setattr(db_mod, "run", fake_run)

        with pytest.raises(RuntimeError, match="invalid JSON"):
            list_databricks_connections(WS)


class TestListGenieSpaces:
    def test_lists_paginated_spaces_with_workspace_env(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            if "--page-token" in args:
                payload = {"spaces": [{"space_id": "space-2", "title": "Second"}]}
            else:
                payload = {
                    "spaces": [{"space_id": "space-1", "title": "First"}],
                    "next_page_token": "next-page",
                }
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload))

        monkeypatch.setattr(db_mod, "run", fake_run)

        assert list_genie_spaces(WS) == [
            {"space_id": "space-1", "title": "First"},
            {"space_id": "space-2", "title": "Second"},
        ]
        assert calls[0]["args"] == [
            "databricks",
            "genie",
            "list-spaces",
            "--page-size",
            "100",
            "--output",
            "json",
        ]
        assert calls[0]["kwargs"]["env"]["DATABRICKS_HOST"] == WS
        assert calls[1]["args"][-2:] == ["--page-token", "next-page"]

    def test_passes_profile_when_provided(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"spaces": []}))

        monkeypatch.setattr(db_mod, "run", fake_run)

        list_genie_spaces(WS, "my-profile")

        assert "--profile" in calls[0]
        assert calls[0][calls[0].index("--profile") + 1] == "my-profile"

    def test_raises_on_invalid_json(self, monkeypatch):
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="not-json")

        monkeypatch.setattr(db_mod, "run", fake_run)

        with pytest.raises(RuntimeError, match="invalid JSON"):
            list_genie_spaces(WS)


class TestListDatabricksApps:
    def test_lists_apps_with_workspace_env(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            payload = [
                {
                    "name": "my-app",
                    "url": "https://my-app.example.databricksapps.com",
                }
            ]
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload))

        monkeypatch.setattr(db_mod, "run", fake_run)

        assert list_databricks_apps(WS) == [
            {
                "name": "my-app",
                "url": "https://my-app.example.databricksapps.com",
            }
        ]
        assert calls[0]["args"] == [
            "databricks",
            "apps",
            "list",
            "--limit",
            "1000",
            "--output",
            "json",
        ]
        assert calls[0]["kwargs"]["env"]["DATABRICKS_HOST"] == WS

    def test_passes_profile_when_provided(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps([]))

        monkeypatch.setattr(db_mod, "run", fake_run)

        list_databricks_apps(WS, "my-profile")

        assert "--profile" in calls[0]
        assert calls[0][calls[0].index("--profile") + 1] == "my-profile"

    def test_accepts_object_wrapped_apps(self, monkeypatch):
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps({"apps": [{"name": "my-app", "url": "https://example.com"}]}),
            )

        monkeypatch.setattr(db_mod, "run", fake_run)

        assert list_databricks_apps(WS) == [{"name": "my-app", "url": "https://example.com"}]

    def test_raises_on_invalid_json(self, monkeypatch):
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="not-json")

        monkeypatch.setattr(db_mod, "run", fake_run)

        with pytest.raises(RuntimeError, match="invalid JSON"):
            list_databricks_apps(WS)


class TestEnsureAiGatewayV2:
    """Test ensure_ai_gateway_v2 without real network calls.

    The probe is `GET /api/ai-gateway/v2/endpoints`: a successful JSON
    response means v2 is wired up (even if `endpoints` is empty), while
    404/401/403/network errors all raise a RuntimeError with the docs URL.
    """

    @staticmethod
    def _mock_json_response(body: str):
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = body.encode("utf-8")
        return mock_resp

    @staticmethod
    def _http_error(code: int, msg: str, body: str = ""):
        import io
        from unittest.mock import MagicMock
        from urllib.error import HTTPError

        fp = io.BytesIO(body.encode("utf-8")) if body else None
        return HTTPError(url="", code=code, msg=msg, hdrs=MagicMock(), fp=fp)

    def test_raises_on_404(self):
        from unittest.mock import patch

        exc = self._http_error(404, "Not Found")
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            from ucode.databricks import ensure_ai_gateway_v2

            with pytest.raises(RuntimeError, match=AI_GATEWAY_V2_DOCS_URL) as excinfo:
                ensure_ai_gateway_v2(WS, "fake-token")
            assert "not enabled" in str(excinfo.value)

    def test_raises_on_401_with_auth_hint(self):
        from unittest.mock import patch

        exc = self._http_error(401, "Unauthorized")
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            from ucode.databricks import ensure_ai_gateway_v2

            with pytest.raises(RuntimeError, match="401") as excinfo:
                ensure_ai_gateway_v2(WS, "fake-token")
            message = str(excinfo.value)
            assert "rejected" in message.lower()
            assert "databricks auth login" in message

    def test_raises_on_400_invalid_token_with_auth_hint(self):
        """400 + body `Invalid Token` is the misleading-error case from issue #84."""
        from unittest.mock import patch

        exc = self._http_error(400, "Bad Request", body="Invalid Token")
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            from ucode.databricks import ensure_ai_gateway_v2

            with pytest.raises(RuntimeError) as excinfo:
                ensure_ai_gateway_v2(WS, "fake-token")
            message = str(excinfo.value)
            # The bug we are fixing: must NOT collapse to the generic
            # "v2 not available" message — must call out the auth failure
            # and point at re-login.
            assert "Invalid Token" in message
            assert "rejected" in message.lower()
            assert "databricks auth login" in message

    def test_400_without_invalid_token_falls_through_to_generic(self):
        """A 400 that is *not* an auth failure should still surface the body."""
        from unittest.mock import patch

        exc = self._http_error(400, "Bad Request", body="some other detail")
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            from ucode.databricks import ensure_ai_gateway_v2

            with pytest.raises(RuntimeError, match=AI_GATEWAY_V2_DOCS_URL) as excinfo:
                ensure_ai_gateway_v2(WS, "fake-token")
            assert "some other detail" in str(excinfo.value)

    def test_raises_on_url_error(self):
        from unittest.mock import patch
        from urllib.error import URLError

        with patch(
            "ucode.databricks.urllib_request.urlopen",
            side_effect=URLError("connection refused"),
        ):
            from ucode.databricks import ensure_ai_gateway_v2

            with pytest.raises(RuntimeError, match=AI_GATEWAY_V2_DOCS_URL):
                ensure_ai_gateway_v2(WS, "fake-token")

    def test_succeeds_with_endpoints_list(self):
        from unittest.mock import patch

        with patch(
            "ucode.databricks.urllib_request.urlopen",
            return_value=self._mock_json_response('{"endpoints": [{"name": "foo"}]}'),
        ):
            from ucode.databricks import ensure_ai_gateway_v2

            ensure_ai_gateway_v2(WS, "fake-token")  # should not raise

    def test_succeeds_with_empty_endpoints_list(self):
        from unittest.mock import patch

        # A 200 with no endpoints still means v2 is wired up on this workspace —
        # downstream discovery will surface "no models" with a clearer reason.
        with patch(
            "ucode.databricks.urllib_request.urlopen",
            return_value=self._mock_json_response('{"endpoints": []}'),
        ):
            from ucode.databricks import ensure_ai_gateway_v2

            ensure_ai_gateway_v2(WS, "fake-token")  # should not raise


class TestHttpGetJsonReason:
    """The `reason` string returned by `_http_get_json` must include the response body
    so callers (e.g. ensure_ai_gateway_v2) can route on it. Before issue #84's fix
    the body was logged only when UCODE_DEBUG=1 and dropped from the bubbled error."""

    @staticmethod
    def _http_error(code: int, msg: str, body: str = ""):
        import io
        from unittest.mock import MagicMock
        from urllib.error import HTTPError

        fp = io.BytesIO(body.encode("utf-8")) if body else None
        return HTTPError(url="", code=code, msg=msg, hdrs=MagicMock(), fp=fp)

    def test_reason_includes_body_on_http_error(self):
        from unittest.mock import patch

        from ucode.databricks import _http_get_json

        exc = self._http_error(400, "Bad Request", body="Invalid Token")
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            payload, reason = _http_get_json("https://x/y", "tok")
        assert payload is None
        assert "HTTP 400" in reason
        assert "Invalid Token" in reason

    def test_reason_without_body_is_status_only(self):
        from unittest.mock import patch

        from ucode.databricks import _http_get_json

        exc = self._http_error(404, "Not Found")
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            payload, reason = _http_get_json("https://x/y", "tok")
        assert payload is None
        assert reason == "HTTP 404 Not Found"


class TestParseDatabricksCliVersion:
    def test_parses_standard_format(self):
        assert _parse_databricks_cli_version("Databricks CLI v0.299.2") == (0, 299, 2)

    def test_parses_without_v_prefix(self):
        assert _parse_databricks_cli_version("Databricks CLI 0.298.0") == (0, 298, 0)

    def test_returns_none_on_garbage(self):
        assert _parse_databricks_cli_version("not a version") is None


class TestEnsureDatabricksCliVersion:
    def _fake_databricks(self, tmp_path, version_output: str) -> dict:
        fake = tmp_path / "databricks"
        fake.write_text(f"#!/bin/sh\necho '{version_output}'\n")
        fake.chmod(0o755)
        return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}

    def test_passes_when_version_meets_minimum(self, tmp_path, monkeypatch):
        env = self._fake_databricks(tmp_path, "Databricks CLI v0.298.0")
        monkeypatch.setattr("os.environ", env)
        ensure_databricks_cli_version()  # should not raise

    def test_passes_when_version_exceeds_minimum(self, tmp_path, monkeypatch):
        env = self._fake_databricks(tmp_path, "Databricks CLI v0.299.2")
        monkeypatch.setattr("os.environ", env)
        ensure_databricks_cli_version()

    def test_auto_upgrades_when_version_too_old(self, tmp_path, monkeypatch):
        import ucode.databricks as db_mod

        env = self._fake_databricks(tmp_path, "Databricks CLI v0.297.0")
        monkeypatch.setattr("os.environ", env)
        upgraded = []
        monkeypatch.setattr(
            db_mod,
            "_run_databricks_cli_installer",
            lambda brew_subcommand="install": upgraded.append(brew_subcommand),
        )
        # Stop the recursive re-check after upgrade
        call_count = [0]
        original = db_mod.ensure_databricks_cli_version

        def once(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                original()

        monkeypatch.setattr(db_mod, "ensure_databricks_cli_version", once)
        once()
        assert upgraded == ["upgrade"]

    def test_raises_when_version_unparseable(self, tmp_path, monkeypatch):
        env = self._fake_databricks(tmp_path, "completely broken output")
        monkeypatch.setattr("os.environ", env)
        with pytest.raises(RuntimeError, match="Could not parse"):
            ensure_databricks_cli_version()


class TestRunDatabricksCliInstaller:
    @pytest.mark.parametrize("brew_subcommand", ["install", "upgrade"])
    def test_macos_uses_fully_qualified_tap_formula(self, monkeypatch, brew_subcommand):
        calls = []
        monkeypatch.setattr(db_mod.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(db_mod.shutil, "which", lambda cmd: "/opt/homebrew/bin/brew")
        monkeypatch.setattr(db_mod, "run", lambda cmd, **kw: calls.append(cmd))

        _run_databricks_cli_installer(brew_subcommand=brew_subcommand)

        # The fully-qualified formula forces Homebrew to the Databricks CLI in
        # databricks/tap and fails if absent, rather than falling back to the
        # unrelated `databricks` cask.
        assert calls == [["brew", brew_subcommand, "databricks/tap/databricks"]]


class TestIsUsageTableAccessError:
    """Pin which `ServerOperationError` strings trigger the friendly
    `system.ai_gateway.usage` permissions hint vs. fall through to the
    generic `Usage query failed: ...` arm."""

    @staticmethod
    def _err(msg: str):
        from databricks.sql.exc import ServerOperationError

        return ServerOperationError(msg)

    def test_table_level_select_denial_matches(self):
        msg = (
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have SELECT on Table 'system.ai_gateway.usage'. "
            "SQLSTATE: 42501"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is True

    def test_schema_level_use_schema_denial_matches(self):
        msg = (
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have USE SCHEMA on Schema 'system.ai_gateway'. "
            "SQLSTATE: 42501"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is True

    def test_unrelated_catalog_denial_falls_through(self):
        msg = (
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have USE CATALOG on Catalog 'aarushi'. "
            "SQLSTATE: 42501"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is False

    def test_other_error_code_on_same_table_falls_through(self):
        """Different code on the right table must not trip the gate — the
        helper requires INSUFFICIENT_PERMISSIONS specifically so we don't
        mask e.g. missing-table failures with a permissions-shaped hint."""
        msg = (
            "[TABLE_OR_VIEW_NOT_FOUND] The table or view "
            "`system`.`ai_gateway`.`usage` cannot be found. SQLSTATE: 42P01"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is False

    @pytest.mark.parametrize(
        "quoted",
        [
            "`system`.`ai_gateway`.`usage`",
            "[system].[ai_gateway].[usage]",
        ],
    )
    def test_identifier_quoting_variants_all_match(self, quoted):
        msg = (
            f"[INSUFFICIENT_PERMISSIONS] User does not have SELECT on Table "
            f"{quoted}. SQLSTATE: 42501"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is True


class TestRunUsageQuery:
    """Cover the two control-flow arms `_is_usage_table_access_error` gates:
    friendly RuntimeError for matching errors, raw-text fallback for the rest.
    `from exc` chaining is also pinned so `--debug` still surfaces the
    underlying connector error."""

    @staticmethod
    def _patch_connect_to_raise(monkeypatch, exc):
        import databricks.sql as sql_mod

        def fake_connect(*args, **kwargs):
            raise exc

        monkeypatch.setattr(sql_mod, "connect", fake_connect)

    def test_raises_actionable_message_for_table_access_error(self, monkeypatch):
        from databricks.sql.exc import ServerOperationError

        original = ServerOperationError(
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have SELECT on Table 'system.ai_gateway.usage'. "
            "SQLSTATE: 42501"
        )
        self._patch_connect_to_raise(monkeypatch, original)

        with pytest.raises(RuntimeError, match="Ask your workspace admin") as exc_info:
            db_mod.run_usage_query(WS, "/sql/1.0/warehouses/abc", "tok", "SELECT 1")
        assert "system.ai_gateway.usage" in str(exc_info.value)
        # The original ServerOperationError must survive on __cause__ so
        # `--debug` / stack traces still show the underlying connector error.
        assert exc_info.value.__cause__ is original

    def test_falls_through_for_unrelated_permission_error(self, monkeypatch):
        from databricks.sql.exc import ServerOperationError

        original = ServerOperationError(
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have USE CATALOG on Catalog 'aarushi'. SQLSTATE: 42501"
        )
        self._patch_connect_to_raise(monkeypatch, original)

        with pytest.raises(RuntimeError, match="aarushi") as exc_info:
            db_mod.run_usage_query(WS, "/sql/1.0/warehouses/abc", "tok", "SELECT 1")
        assert "Ask your workspace admin" not in str(exc_info.value)
        assert str(exc_info.value).startswith("Usage query failed:")
