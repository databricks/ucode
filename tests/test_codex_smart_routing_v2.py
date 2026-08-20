"""Tests for the experimental ENABLE_SMART_ROUTING_V2 Codex launch path."""

from __future__ import annotations

import json

from ucode.agents import codex
from ucode.config_io import read_toml_safe
from ucode.smart_routing import codex_interposer

WS = "https://example.databricks.com"


class TestGenerateV2Home:
    def test_writes_provider_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(codex, "SMART_ROUTING_V2_HOME", tmp_path / "v2home")
        monkeypatch.setattr(codex, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.148.0")

        home = codex._generate_v2_app_server_home(
            {"workspace": WS, "profile": "myprof"}, "gpt-5.6-luna"
        )

        assert home == tmp_path / "v2home"
        doc = read_toml_safe(home / "config.toml")
        assert doc["model_provider"] == codex.CODEX_MODEL_PROVIDER_NAME
        assert doc["model"] == "gpt-5.6-luna"
        provider = doc["model_providers"][codex.CODEX_MODEL_PROVIDER_NAME]
        assert provider["base_url"].endswith("/ai-gateway/codex/v1")
        # Self-refreshing auth command is preserved (app-server rejects --profile).
        assert provider["auth"]["command"].endswith("ucode")
        assert "myprof" in provider["auth"]["args"]


class TestInterposerSession:
    """The interposer's hold-then-switch + settings-injection logic (the novel behavior)."""

    def _turn_start(self, model: str, thread_id: str = "t1") -> str:
        return json.dumps(
            {
                "method": "turn/start",
                "id": 1,
                "params": {"threadId": thread_id, "input": [], "model": model},
            }
        )

    def test_holds_first_turn_then_switches(self):
        sess = codex_interposer._Session("gpt-5.5", after=1, log=lambda _m: None)
        # Turn 1 passes through unchanged (still on the TUI's model).
        out1 = sess.on_tui_frame(self._turn_start("system.ai.gpt-5-6-luna"))
        assert json.loads(out1)["params"]["model"] == "system.ai.gpt-5-6-luna"
        # Turn 2 is rewritten to the target.
        out2 = sess.on_tui_frame(self._turn_start("system.ai.gpt-5-6-luna"))
        assert json.loads(out2)["params"]["model"] == "gpt-5.5"

    def test_after_zero_switches_immediately(self):
        sess = codex_interposer._Session("gpt-5.5", after=0, log=lambda _m: None)
        out1 = sess.on_tui_frame(self._turn_start("luna"))
        assert json.loads(out1)["params"]["model"] == "gpt-5.5"

    def test_non_turn_frames_pass_through(self):
        sess = codex_interposer._Session("gpt-5.5", after=1, log=lambda _m: None)
        frame = json.dumps({"method": "initialize", "id": 1, "params": {}})
        assert sess.on_tui_frame(frame) == frame

    def test_injects_settings_update_after_hold(self):
        sess = codex_interposer._Session("gpt-5.5", after=1, log=lambda _m: None)
        sess.on_tui_frame(self._turn_start("luna"))  # turn 1 (the hold)
        inj = sess.on_engine_frame(
            json.dumps(
                {
                    "method": "turn/completed",
                    "params": {"threadId": "t1", "turn": {"status": "completed"}},
                }
            )
        )
        assert inj is not None
        assert inj["method"] == codex_interposer.SETTINGS_UPDATED
        assert inj["params"]["threadId"] == "t1"
        assert inj["params"]["threadSettings"]["model"] == "gpt-5.5"

    def test_injects_only_once(self):
        sess = codex_interposer._Session("gpt-5.5", after=1, log=lambda _m: None)
        sess.on_tui_frame(self._turn_start("luna"))
        done = json.dumps({"method": "turn/completed", "params": {"threadId": "t1", "turn": {}}})
        assert sess.on_engine_frame(done) is not None
        assert sess.on_engine_frame(done) is None  # second completion: no re-inject
