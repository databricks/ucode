"""Tests for agents/codex.py."""

from __future__ import annotations

import json
import os

from ucode.agents import codex
from ucode.config_io import read_toml_safe

WS = "https://example.databricks.com"


class TestCodexSpec:
    def test_binary(self):
        assert codex.SPEC["binary"] == "codex"

    def test_package(self):
        assert codex.SPEC["package"] == "@openai/codex"

    def test_display(self):
        assert codex.SPEC["display"] == "Codex"


class TestRenderOverlay:
    def test_uses_profile_file_shape_without_legacy_profiles(self):
        overlay = codex.render_overlay(WS)
        assert "profile" not in overlay
        assert "profiles" not in overlay

    def test_sets_model_provider(self):
        overlay = codex.render_overlay(WS)
        assert overlay["model_provider"] == "ucode-databricks"

    def test_sets_model_when_provided(self):
        overlay = codex.render_overlay(WS, "databricks-gpt-5")
        assert overlay["model"] == "databricks-gpt-5"

    def test_provider_base_url(self):
        overlay = codex.render_overlay(WS)
        provider = overlay["model_providers"]["ucode-databricks"]
        assert provider["base_url"] == f"{WS}/ai-gateway/codex/v1"

    def test_provider_wire_api(self):
        overlay = codex.render_overlay(WS)
        provider = overlay["model_providers"]["ucode-databricks"]
        assert provider["wire_api"] == "responses"

    def test_auth_uses_sh(self):
        overlay = codex.render_overlay(WS)
        auth = overlay["model_providers"]["ucode-databricks"]["auth"]
        assert auth["command"] == "sh"
        assert "-c" in auth["args"]

    def test_auth_contains_workspace(self):
        overlay = codex.render_overlay(WS)
        auth = overlay["model_providers"]["ucode-databricks"]["auth"]
        assert any(WS in arg for arg in auth["args"])

    def test_auth_refresh_interval(self):
        overlay = codex.render_overlay(WS)
        auth = overlay["model_providers"]["ucode-databricks"]["auth"]
        assert auth["refresh_interval_ms"] == 900_000


class TestRenderOverlayUserAgent:
    def test_user_agent_set_on_provider(self, monkeypatch):
        monkeypatch.setattr(codex, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.123.0")
        overlay = codex.render_overlay(WS)
        provider = overlay["model_providers"]["ucode-databricks"]
        assert provider["http_headers"]["User-Agent"] == "ucode/0.1.0 codex/0.123.0"

    def test_managed_keys_include_http_headers(self):
        # Revert must clean up the new key.
        assert ["model_providers", "ucode-databricks", "http_headers"] in codex.MANAGED_KEYS


class TestCodexWriteConfig:
    def test_writes_ucode_profile_config_file(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config({"workspace": WS, "codex_models": ["gpt-5"]})

        doc = read_toml_safe(config_path)
        assert doc["model_provider"] == "ucode-databricks"
        assert doc["model"] == "gpt-5"
        assert "profiles" not in doc

    def test_writes_openai_model_id_for_databricks_gpt_endpoint(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config(
            {"workspace": WS, "codex_models": ["databricks-gpt-5", "databricks-gpt-5-5"]}
        )

        doc = read_toml_safe(config_path)
        assert doc["model"] == "gpt-5.5"

    def test_preserves_databricks_model_id_when_openai_id_is_incompatible(
        self, tmp_path, monkeypatch
    ):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config(
            {"workspace": WS, "codex_models": ["databricks-gpt-5-2-codex"]},
            "databricks-gpt-5-2-codex",
        )

        doc = read_toml_safe(config_path)
        assert doc["model"] == "databricks-gpt-5-2-codex"

    def test_removes_legacy_ucode_profile_from_shared_config(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        profile_path = config_dir / "ucode.config.toml"
        legacy_path = config_dir / "config.toml"
        legacy_path.write_text(
            'profile = "ucode"\n\n'
            "[profiles.ucode]\n"
            'model_provider = "old"\n\n'
            "[profiles.other]\n"
            'model_provider = "keep"\n',
            encoding="utf-8",
        )
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        legacy_backup_path = tmp_path / "codex-legacy-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config({"workspace": WS, "codex_models": ["gpt-5"]})

        doc = read_toml_safe(legacy_path)
        assert "profile" not in doc
        assert "ucode" not in doc["profiles"]
        assert doc["profiles"]["other"]["model_provider"] == "keep"
        assert legacy_backup_path.exists()

    def test_writes_legacy_shared_config_when_codex_too_old(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".codex"
        legacy_path = config_dir / "config.toml"
        profile_path = config_dir / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        legacy_backup_path = tmp_path / "codex-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "LEGACY_CODEX_CONFIG_PATH", legacy_path)
        monkeypatch.setattr(codex, "LEGACY_CODEX_BACKUP_PATH", legacy_backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.133.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config({"workspace": WS, "codex_models": ["gpt-5"]})

        # Per-profile file must not be written for old Codex.
        assert not profile_path.exists()
        doc = read_toml_safe(legacy_path)
        assert doc["profile"] == "ucode"
        assert doc["profiles"]["ucode"]["model_provider"] == "ucode-databricks"
        assert doc["profiles"]["ucode"]["model"] == "gpt-5"
        provider = doc["model_providers"]["ucode-databricks"]
        assert provider["base_url"] == f"{WS}/ai-gateway/codex/v1"
        assert provider["wire_api"] == "responses"

    def test_legacy_write_preserves_other_profiles_in_shared_config(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        legacy_path = config_dir / "config.toml"
        legacy_path.write_text(
            '[profiles.other]\nmodel_provider = "keep"\n',
            encoding="utf-8",
        )
        profile_path = config_dir / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        legacy_backup_path = tmp_path / "codex-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "LEGACY_CODEX_CONFIG_PATH", legacy_path)
        monkeypatch.setattr(codex, "LEGACY_CODEX_BACKUP_PATH", legacy_backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.133.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config({"workspace": WS, "codex_models": ["gpt-5"]})

        doc = read_toml_safe(legacy_path)
        assert doc["profiles"]["other"]["model_provider"] == "keep"
        assert doc["profiles"]["ucode"]["model_provider"] == "ucode-databricks"


class TestCodexLegacyLayoutDetection:
    def test_new_codex_uses_modern_layout(self, monkeypatch):
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")

        assert codex._use_legacy_layout() is False

    def test_old_codex_uses_legacy_layout(self, monkeypatch):
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.133.0")

        assert codex._use_legacy_layout() is True

    def test_unknown_version_uses_modern_layout(self, monkeypatch):
        monkeypatch.setattr(codex, "agent_version", lambda binary: "unknown")

        assert codex._use_legacy_layout() is False


class TestCodexRemoveLegacyProfile:
    def test_drops_provider_block_on_modern_path(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        profile_path = config_dir / "ucode.config.toml"
        legacy_path = config_dir / "config.toml"
        legacy_path.write_text(
            'profile = "ucode"\n\n'
            "[profiles.ucode]\n"
            'model_provider = "ucode-databricks"\n\n'
            "[model_providers.ucode-databricks]\n"
            'name = "Databricks AI Gateway"\n\n'
            "[model_providers.other]\n"
            'name = "keep"\n',
            encoding="utf-8",
        )
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config({"workspace": WS, "codex_models": ["gpt-5"]})

        doc = read_toml_safe(legacy_path)
        assert "profile" not in doc
        assert "ucode" not in doc.get("profiles", {})
        assert "ucode-databricks" not in doc["model_providers"]
        assert doc["model_providers"]["other"]["name"] == "keep"


class TestCodexRevertLegacySharedConfig:
    def test_strips_all_ucode_entries(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        profile_path = config_dir / "ucode.config.toml"
        legacy_path = config_dir / "config.toml"
        legacy_path.write_text(
            'profile = "ucode"\n\n'
            "[profiles.ucode]\n"
            'model_provider = "ucode-databricks"\n\n'
            "[profiles.other]\n"
            'model_provider = "keep"\n\n'
            "[model_providers.ucode-databricks]\n"
            'name = "Databricks AI Gateway"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)

        assert codex.revert_legacy_shared_config() is True

        doc = read_toml_safe(legacy_path)
        assert "profile" not in doc
        assert "ucode" not in doc["profiles"]
        assert doc["profiles"]["other"]["model_provider"] == "keep"
        assert "model_providers" not in doc

    def test_returns_false_when_no_ucode_entries(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        profile_path = config_dir / "ucode.config.toml"
        legacy_path = config_dir / "config.toml"
        legacy_path.write_text('[profiles.other]\nmodel_provider = "keep"\n', encoding="utf-8")
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)

        assert codex.revert_legacy_shared_config() is False

        doc = read_toml_safe(legacy_path)
        assert doc["profiles"]["other"]["model_provider"] == "keep"

    def test_returns_false_when_no_shared_config(self, tmp_path, monkeypatch):
        profile_path = tmp_path / ".codex" / "ucode.config.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)

        assert codex.revert_legacy_shared_config() is False


class TestCodexDefaultModel:
    def test_picks_highest_semver_over_alpha(self):
        state = {"codex_models": ["databricks-gpt-5", "databricks-gpt-5-5"]}

        assert codex.default_model(state) == "databricks-gpt-5-5"

    def test_none_when_no_models(self):
        assert codex.default_model({}) is None

    def test_none_when_no_gpt_parseable_models(self):
        # A workspace whose responses-capable models aren't GPT (e.g. kimi)
        # must not pin an unroutable id as the Codex model.
        state = {"codex_models": ["moonshotai/kimi-k2.5", "claude-sonnet-4"]}

        assert codex.default_model(state) is None

    def test_ignores_non_gpt_candidates(self):
        state = {"codex_models": ["moonshotai/kimi-k2.5", "databricks-gpt-5-5"]}

        assert codex.default_model(state) == "databricks-gpt-5-5"

    def test_prefers_base_over_suffixed_same_version(self):
        models = ["gpt-5-5-mini", "gpt-5-5", "gpt-5"]

        assert codex.default_model({"codex_models": models}) == "gpt-5-5"

    def test_namespaced_models_use_same_version_parser(self):
        models = ["served-models/databricks-gpt-5", "served-models/databricks-gpt-5-5"]

        assert codex.default_model({"codex_models": models}) == "served-models/databricks-gpt-5-5"

    def test_openai_model_id_maps_databricks_naming(self):
        assert codex._openai_model_id("databricks-gpt-5-5") == "gpt-5.5"
        assert codex._openai_model_id("databricks-gpt-5-5-mini") == "gpt-5.5-mini"
        assert codex._openai_model_id("databricks-gpt-4o") == "gpt-4o"
        assert codex._openai_model_id("served-models/databricks-gpt-5-5") == "gpt-5.5"
        assert codex._openai_model_id("gpt-5.5") == "gpt-5.5"

    def test_codex_model_id_preserves_openai_incompatible_models(self):
        assert codex._codex_model_id("databricks-gpt-5-2-codex") == "databricks-gpt-5-2-codex"
        assert codex._codex_model_id("databricks-gpt-5-4-nano") == "databricks-gpt-5-4-nano"

    def test_codex_model_id_passes_model_services_id_verbatim(self):
        # UC model-services ids route by name, so they must not be rewritten
        # to the OpenAI id form.
        assert codex._codex_model_id("system.ai.gpt-5") == "system.ai.gpt-5"
        assert codex._codex_model_id("system.ai.gpt-5-2-codex") == "system.ai.gpt-5-2-codex"

    def test_default_model_selects_model_services_gpt(self):
        models = ["system.ai.gpt-5", "system.ai.gpt-5-5", "system.ai.claude-opus-4-8"]

        assert codex.default_model({"codex_models": models}) == "system.ai.gpt-5-5"
        assert codex._codex_model_id("databricks-gpt-5-5") == "gpt-5.5"


class TestCodexValidateCmd:
    def test_starts_with_binary(self):
        cmd = codex.validate_cmd("codex")
        assert cmd[0] == "codex"

    def test_uses_exec_subcommand(self):
        cmd = codex.validate_cmd("codex")
        assert "exec" in cmd

    def test_uses_ucode_profile(self):
        cmd = codex.validate_cmd("codex")
        assert cmd[:3] == ["codex", "--profile", "ucode"]

    def test_has_prompt(self):
        cmd = codex.validate_cmd("codex")
        assert len(cmd) > 2

    def test_skips_git_repo_check(self):
        # Validation runs in arbitrary cwd (e.g., ~/Documents); without this
        # flag Codex refuses to run outside a trusted/git directory.
        cmd = codex.validate_cmd("codex")
        assert "--skip-git-repo-check" in cmd


class TestCodexLaunch:
    def test_sets_oauth_token_and_ucode_profile_before_exec(self, monkeypatch):
        exec_calls: list[tuple[str, list[str]]] = []

        def fake_execvp(binary: str, args: list[str]) -> None:
            exec_calls.append((binary, args))
            raise RuntimeError("stop")

        monkeypatch.delenv("OAUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            codex, "get_databricks_token", lambda workspace, profile=None: "fresh-token"
        )
        monkeypatch.setattr(os, "execvp", fake_execvp)

        try:
            codex.launch({"workspace": WS}, ["--search"])
        except RuntimeError as exc:
            assert str(exc) == "stop"

        assert os.environ["OAUTH_TOKEN"] == "fresh-token"
        assert exec_calls == [("codex", ["codex", "--profile", "ucode", "--search"])]


class TestBuildModelCatalog:
    def test_returns_none_when_no_models(self):
        # Codex rejects empty catalogs at startup, so we must skip writing the
        # file entirely rather than emit `{"models": []}`.
        assert codex.build_model_catalog([]) is None
        assert codex.build_model_catalog(None) is None

    def test_emits_one_entry_per_model(self):
        catalog = codex.build_model_catalog(["system.ai.gpt-5", "system.ai.gpt-5-5"])

        slugs = [entry["slug"] for entry in catalog["models"]]
        assert slugs == ["system.ai.gpt-5", "system.ai.gpt-5-5"]

    def test_each_entry_has_required_keys(self):
        # The Codex protocol demands every non-`#[serde(default)]` field be
        # present; an omission causes Codex to refuse to start. Guard the
        # contract so a future field rename doesn't quietly break configure.
        catalog = codex.build_model_catalog(["system.ai.gpt-5"])
        entry = catalog["models"][0]

        for key in (
            "slug",
            "display_name",
            "description",
            "supported_reasoning_levels",
            "shell_type",
            "visibility",
            "supported_in_api",
            "priority",
            "availability_nux",
            "upgrade",
            "base_instructions",
            "supports_reasoning_summaries",
            "support_verbosity",
            "default_verbosity",
            "apply_patch_tool_type",
            "truncation_policy",
            "supports_parallel_tool_calls",
            "experimental_supported_tools",
        ):
            assert key in entry, f"missing required ModelInfo field: {key}"

    def test_truncation_policy_is_well_formed(self):
        # Codex requires both `mode` and `limit`; a malformed sub-struct fails
        # the whole catalog load with `unknown variant ...`.
        entry = codex.build_model_catalog(["system.ai.gpt-5"])["models"][0]

        assert entry["truncation_policy"] == {"mode": "bytes", "limit": 10000}

    def test_visibility_is_protocol_compliant(self):
        # `list` / `hide` / `none` are the only accepted strings — `custom` and
        # other values fail catalog deserialization.
        entry = codex.build_model_catalog(["system.ai.gpt-5"])["models"][0]

        assert entry["visibility"] in {"list", "hide", "none"}

    def test_apply_patch_uses_freeform_for_codex_models(self):
        # GPT-5 variants ship with the lark-grammar apply_patch tool; we keep
        # it on so users see the same agent capabilities regardless of which
        # discovery path their workspace uses.
        entry = codex.build_model_catalog(["system.ai.gpt-5"])["models"][0]

        assert entry["apply_patch_tool_type"] == "freeform"


class TestModelCatalogFile:
    def _patch_paths(self, tmp_path, monkeypatch):
        catalog_path = tmp_path / "codex-model-catalog.json"
        monkeypatch.setattr(codex, "CODEX_MODEL_CATALOG_PATH", catalog_path)
        return catalog_path

    def test_skips_file_when_use_model_services_false(self, tmp_path, monkeypatch):
        catalog_path = self._patch_paths(tmp_path, monkeypatch)

        result = codex._write_model_catalog_file(
            {"codex_models": ["databricks-gpt-5"], "use_model_services": False}
        )

        assert result is None
        assert not catalog_path.exists()

    def test_skips_file_when_no_codex_models(self, tmp_path, monkeypatch):
        # `use_model_services=True` without any GPT models would produce an
        # empty catalog, which Codex rejects. Don't write the file at all.
        catalog_path = self._patch_paths(tmp_path, monkeypatch)

        result = codex._write_model_catalog_file({"codex_models": [], "use_model_services": True})

        assert result is None
        assert not catalog_path.exists()

    def test_writes_catalog_when_use_model_services_true(self, tmp_path, monkeypatch):
        catalog_path = self._patch_paths(tmp_path, monkeypatch)

        result = codex._write_model_catalog_file(
            {"codex_models": ["system.ai.gpt-5", "system.ai.gpt-5-5"], "use_model_services": True}
        )

        assert result == catalog_path
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        slugs = [entry["slug"] for entry in catalog["models"]]
        assert slugs == ["system.ai.gpt-5", "system.ai.gpt-5-5"]

    def test_removes_stale_catalog_when_toggling_off_model_services(self, tmp_path, monkeypatch):
        # If a workspace previously had `UCODE_USE_MODEL_SERVICES=1` and now
        # doesn't, the old catalog must go away — otherwise `model_catalog_json`
        # in the toml would still resolve and Codex would silently pin the
        # stale model list.
        catalog_path = self._patch_paths(tmp_path, monkeypatch)
        catalog_path.write_text('{"models": []}', encoding="utf-8")

        result = codex._write_model_catalog_file(
            {"codex_models": ["databricks-gpt-5"], "use_model_services": False}
        )

        assert result is None
        assert not catalog_path.exists()


class TestRevertModelCatalogFile:
    def test_returns_false_when_no_catalog(self, tmp_path, monkeypatch):
        catalog_path = tmp_path / "codex-model-catalog.json"
        monkeypatch.setattr(codex, "CODEX_MODEL_CATALOG_PATH", catalog_path)

        assert codex.revert_model_catalog_file() is False

    def test_removes_existing_catalog(self, tmp_path, monkeypatch):
        catalog_path = tmp_path / "codex-model-catalog.json"
        catalog_path.write_text('{"models": []}', encoding="utf-8")
        monkeypatch.setattr(codex, "CODEX_MODEL_CATALOG_PATH", catalog_path)

        assert codex.revert_model_catalog_file() is True
        assert not catalog_path.exists()


class TestRenderOverlayWithModelCatalog:
    def test_omits_model_catalog_json_by_default(self):
        # AI-gateway path: Codex's `OpenAiModelsManager` should hit /v1/models
        # the way it always has.
        overlay = codex.render_overlay(WS, "databricks-gpt-5")

        assert "model_catalog_json" not in overlay

    def test_includes_model_catalog_json_when_path_provided(self, tmp_path):
        # model-services path: forcing `StaticModelsManager` is the whole
        # point of this knob, so the overlay must surface the path string.
        catalog_path = tmp_path / "codex-model-catalog.json"

        overlay = codex.render_overlay(WS, "system.ai.gpt-5", model_catalog_path=catalog_path)

        assert overlay["model_catalog_json"] == str(catalog_path)


class TestWriteToolConfigModelCatalog:
    def test_writes_model_catalog_when_use_model_services_true(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        catalog_path = tmp_path / "codex-model-catalog.json"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "CODEX_MODEL_CATALOG_PATH", catalog_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config(
            {
                "workspace": WS,
                "codex_models": ["system.ai.gpt-5", "system.ai.gpt-5-5"],
                "use_model_services": True,
            }
        )

        doc = read_toml_safe(config_path)
        assert doc["model_catalog_json"] == str(catalog_path)
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        assert {entry["slug"] for entry in catalog["models"]} == {
            "system.ai.gpt-5",
            "system.ai.gpt-5-5",
        }

    def test_omits_model_catalog_json_when_ai_gateway(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        catalog_path = tmp_path / "codex-model-catalog.json"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "CODEX_MODEL_CATALOG_PATH", catalog_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config({"workspace": WS, "codex_models": ["databricks-gpt-5"]})

        doc = read_toml_safe(config_path)
        assert "model_catalog_json" not in doc
        assert not catalog_path.exists()

    def test_clears_stale_model_catalog_json_when_toggling_off(self, tmp_path, monkeypatch):
        # Re-running `ucode configure` after dropping
        # `UCODE_USE_MODEL_SERVICES=1` must not leave the static-catalog
        # pointer behind — otherwise Codex still pins the cached list.
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        catalog_path = tmp_path / "codex-model-catalog.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(
            'model_catalog_json = "/tmp/old.json"\nmodel = "system.ai.gpt-5"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "CODEX_MODEL_CATALOG_PATH", catalog_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config({"workspace": WS, "codex_models": ["databricks-gpt-5"]})

        doc = read_toml_safe(config_path)
        assert "model_catalog_json" not in doc


class TestManagedKeysModelCatalog:
    def test_managed_keys_include_model_catalog_json(self):
        # Revert must strip `model_catalog_json` out of the toml; if it
        # vanishes from MANAGED_KEYS the cleanup silently regresses.
        assert ["model_catalog_json"] in codex.MANAGED_KEYS
