"""Pi coding agent: writes ~/.pi/agent/models.json with Databricks-backed providers.

Pi (https://pi.dev) is a multi-provider coding agent. We register three
providers in its `models.json`, each speaking the API dialect best suited to
that family's gateway path:

- `databricks-claude`  (api: anthropic-messages)       → /ai-gateway/anthropic
- `databricks-openai`  (api: openai-responses)         → /ai-gateway/codex/v1
- `databricks-gemini`  (api: google-generative-ai)     → /ai-gateway/gemini/v1beta

Per-provider `compat` flags work around fields the gateway translators reject:

- claude: `supportsEagerToolInputStreaming: false` — the Anthropic translator
  rejects `tools[].eager_input_streaming` on the streaming + tools path that
  pi uses for every request. With this flag pi omits the per-tool field and
  sends the legacy `anthropic-beta: fine-grained-tool-streaming-...` header
  instead, which the gateway accepts.

OSS / Databricks-foundation models (Llama, Qwen, etc.) are not exposed via
pi today — they live behind /ai-gateway/mlflow/v1 with per-model
`max_tokens` caps that pi has no global way to honor without per-model
config we don't currently maintain.

The bearer token is baked into the file and refreshed by a background thread
while the session runs (same pattern as OpenCode/Copilot).
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading

from ucode.agent_updates import available_npm_package_update
from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_json_safe,
    write_json_file,
)
from ucode.databricks import (
    TOKEN_REFRESH_INTERVAL_SECONDS,
    build_pi_base_urls,
    get_databricks_token,
)
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version

PI_UCODE_HOME = APP_DIR / "pi-home"
PI_CONFIG_DIR = PI_UCODE_HOME / ".pi" / "agent"
PI_CONFIG_PATH = PI_CONFIG_DIR / "models.json"
PI_SETTINGS_PATH = PI_CONFIG_DIR / "settings.json"
PI_BACKUP_PATH = APP_DIR / "pi-models.backup.json"
PI_SETTINGS_BACKUP_PATH = APP_DIR / "pi-settings.backup.json"

SPEC: ToolSpec = {
    "binary": "pi",
    "package": "@earendil-works/pi-coding-agent",
    "display": "Pi",
    "config_path": PI_CONFIG_PATH,
    "backup_path": PI_BACKUP_PATH,
}

PROVIDER_NAMES = (
    "databricks-claude",
    "databricks-openai",
    "databricks-gemini",
    "databricks-external",
)

PROVIDER_KEYS: list[list[str]] = [["providers", name] for name in PROVIDER_NAMES]

# Old provider names earlier ucode versions wrote; cleaned up on each write so
# users don't end up with stale entries pointing at routes that 400.
LEGACY_PROVIDER_NAMES = ("databricks-anthropic", "databricks-codex", "databricks-oss")


def is_update_available() -> tuple[str, str] | None:
    return available_npm_package_update(SPEC["package"])


def _resolve_model_selector(
    model: str,
    claude_ids: list[str],
    codex_models: list[str],
    gemini_models: list[str],
    external_models: list[str],
) -> str:
    """Return a Pi model selector in `<provider>/<model>` form when possible."""
    for name in PROVIDER_NAMES:
        if model.startswith(f"{name}/"):
            return model
    if model in claude_ids:
        return f"databricks-claude/{model}"
    if model in codex_models:
        return f"databricks-openai/{model}"
    if model in gemini_models:
        return f"databricks-gemini/{model}"
    if model in external_models:
        return f"databricks-external/{model}"
    return model


def render_overlay(
    model: str,
    token: str,
    pi_base_urls: dict[str, str],
    claude_ids: list[str],
    codex_models: list[str],
    gemini_models: list[str],
    external_models: list[str],
) -> tuple[dict, list[list[str]]]:
    """Return (overlay, managed_key_paths) for ~/.pi/agent/models.json.

    ``claude_ids`` is every Anthropic-dialect model the workspace serves, not
    just the newest per family — Pi has its own model picker, so it should see
    the full list. ``external_models`` are external-model serving endpoints
    (e.g. Azure OpenAI), reached via the OpenAI chat-completions dialect on the
    classic `/serving-endpoints` path rather than the unified gateway.
    """
    providers: dict = {}
    keys: list[list[str]] = [["model"]]
    # Pi expands header values that match an env var name. Our UA contains
    # `/` and a space so it can never collide — safe to pass as a literal.
    ua_headers = {"User-Agent": f"ucode/{ucode_version()} pi/{agent_version('pi')}"}

    claude_ids = sorted(set(claude_ids))
    if claude_ids:
        providers["databricks-claude"] = {
            "baseUrl": pi_base_urls["claude"],
            "api": "anthropic-messages",
            "apiKey": token,
            "authHeader": True,
            # Gateway's Anthropic translator rejects per-tool
            # `eager_input_streaming` on the streaming + tools path. Pi sends
            # the legacy beta header instead when this is false.
            "compat": {"supportsEagerToolInputStreaming": False},
            "headers": ua_headers,
            "models": [{"id": m} for m in claude_ids],
        }
        keys.append(["providers", "databricks-claude"])
    if codex_models:
        providers["databricks-openai"] = {
            "baseUrl": pi_base_urls["openai"],
            "api": "openai-responses",
            "apiKey": token,
            "authHeader": True,
            "headers": ua_headers,
            "models": [{"id": m} for m in codex_models],
        }
        keys.append(["providers", "databricks-openai"])
    if gemini_models:
        providers["databricks-gemini"] = {
            "baseUrl": pi_base_urls["gemini"],
            "api": "google-generative-ai",
            "apiKey": token,
            "authHeader": True,
            "headers": ua_headers,
            "models": [{"id": m} for m in gemini_models],
        }
        keys.append(["providers", "databricks-gemini"])
    if external_models:
        # External-model serving endpoints speak OpenAI chat-completions on the
        # classic `/serving-endpoints/{name}/invocations` path; `openai-completions`
        # appends `/chat/completions` to the base URL and sends the endpoint name
        # as the model.
        providers["databricks-external"] = {
            "baseUrl": pi_base_urls["external"],
            "api": "openai-completions",
            "apiKey": token,
            "authHeader": True,
            "headers": ua_headers,
            "models": [{"id": m} for m in external_models],
        }
        keys.append(["providers", "databricks-external"])
    overlay: dict = {
        "model": _resolve_model_selector(
            model, claude_ids, codex_models, gemini_models, external_models
        ),
    }
    if providers:
        overlay["providers"] = providers
    return overlay, keys


def write_tool_config(
    state: dict,
    model: str,
    token: str | None = None,
    *,
    force_refresh: bool = False,
    update_settings: bool = True,
    pin_model: bool = False,
) -> tuple[dict, str]:
    """Write Pi's models.json, and seed settings.json unless told not to.

    ``update_settings=False`` is used by the background token refresh: it must
    rewrite models.json to rotate the baked-in bearer, but must not touch the
    model the user picked in Pi's own model selector.

    ``pin_model=True`` means the user named this model (`--model` or a pin), so
    it wins over whatever they last chose in Pi's selector.
    """
    backup_existing_file(PI_CONFIG_PATH, PI_BACKUP_PATH)
    if token is None:
        token = get_databricks_token(
            state["workspace"], state.get("profile"), force_refresh=force_refresh
        )
    pi_base_urls = state.get("base_urls", {}).get("pi") or build_pi_base_urls(state["workspace"])
    # State written before claude_model_ids existed only carries the family map;
    # fall back to its values so a stale state still renders a usable config.
    claude_ids = state.get("claude_model_ids") or list((state.get("claude_models") or {}).values())
    overlay, managed_keys = render_overlay(
        model,
        token,
        pi_base_urls,
        claude_ids,
        state.get("codex_models") or [],
        state.get("gemini_models") or [],
        state.get("external_models") or [],
    )
    existing = read_json_safe(PI_CONFIG_PATH)
    providers = existing.get("providers")
    if isinstance(providers, dict):
        for stale in (*PROVIDER_NAMES, *LEGACY_PROVIDER_NAMES):
            providers.pop(stale, None)
    merged = deep_merge_dict(existing, overlay)
    write_json_file(PI_CONFIG_PATH, merged)
    if update_settings:
        _write_settings(overlay["model"], overlay.get("providers") or {}, force=pin_model)
    state = mark_tool_managed(state, "pi", managed_keys)
    save_state(state)
    return state, token


def _settings_need_repair(existing: dict, providers: dict) -> bool:
    """True when ucode should (re)write Pi's defaultProvider/defaultModel.

    Pi persists the user's model-selector choice into settings.json via
    `setDefaultModelAndProvider`, so an existing pin is a user preference and
    must survive. We only write when there is nothing to preserve, or when what
    is there is ucode-authored residue that no longer resolves:

    - nothing pinned yet — seed it, so Pi's `findInitialModel` can't fall
      through to an env-key-backed provider (e.g. HF_TOKEN exposing huggingface)
    - pinned to a ucode provider we no longer register (e.g. `databricks-openai`
      on a workspace with no Responses models, or a `LEGACY_PROVIDER_NAMES` entry)
    - pinned to a ucode provider we do register, but to a model it no longer offers

    A pin naming a provider ucode does not manage is the user's own; leave it.
    """
    provider = existing.get("defaultProvider")
    model_id = existing.get("defaultModel")
    if not isinstance(provider, str) or not isinstance(model_id, str) or not model_id:
        return True
    if provider in providers:
        offered = {m["id"] for m in providers[provider].get("models") or []}
        return model_id not in offered
    return provider in (*PROVIDER_NAMES, *LEGACY_PROVIDER_NAMES)


def _write_settings(model_selector: str, providers: dict, *, force: bool = False) -> None:
    provider, _, model_id = model_selector.partition("/")
    if not model_id:
        return
    existing = read_json_safe(PI_SETTINGS_PATH)
    if not force and not _settings_need_repair(existing, providers):
        return
    backup_existing_file(PI_SETTINGS_PATH, PI_SETTINGS_BACKUP_PATH)
    merged = deep_merge_dict(existing, {"defaultProvider": provider, "defaultModel": model_id})
    write_json_file(PI_SETTINGS_PATH, merged)


def default_model(state: dict) -> str | None:
    """Prefer Claude opus → sonnet → haiku; fall back to codex, gemini, external."""
    claude_models = state.get("claude_models") or {}
    for family in ("opus", "sonnet", "haiku"):
        if claude_models.get(family):
            return claude_models[family]
    codex_models = state.get("codex_models") or []
    if codex_models:
        return codex_models[0]
    gemini_models = state.get("gemini_models") or []
    if gemini_models:
        return gemini_models[0]
    external_models = state.get("external_models") or []
    return external_models[0] if external_models else None


def _refresh_token_once(
    state: dict,
    model: str | None = None,
    *,
    force_refresh: bool = False,
    update_settings: bool = True,
    pin_model: bool = False,
) -> str:
    model = model or default_model(state)
    if not model:
        raise RuntimeError("No Pi model is available on this workspace.")
    _, token = write_tool_config(
        state,
        model,
        force_refresh=force_refresh,
        update_settings=update_settings,
        pin_model=pin_model,
    )
    return token


def _refresh_forever(state: dict, model: str | None, stop_event: threading.Event) -> None:
    while not stop_event.wait(TOKEN_REFRESH_INTERVAL_SECONDS):
        try:
            # Rotate the bearer baked into models.json, but leave settings.json
            # alone: mid-session the user may have switched models in Pi's
            # selector, and Pi persists that choice there.
            _refresh_token_once(state, model, force_refresh=True, update_settings=False)
        except RuntimeError:
            continue


def build_runtime_env(token: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OAUTH_TOKEN"] = token
    env["HOME"] = str(PI_UCODE_HOME)
    return env


def launch(state: dict, tool_args: list[str], model: str | None = None) -> None:
    # A non-None `model` was named by the user (`--model` or a pin), so it wins
    # over whatever they last selected inside Pi. None means "use the default,
    # but don't disturb their selector choice".
    token = _refresh_token_once(state, model, pin_model=model is not None)
    env = build_runtime_env(token)

    stop_event = threading.Event()
    refresher = threading.Thread(
        target=_refresh_forever,
        args=(state, model, stop_event),
        daemon=True,
    )
    refresher.start()

    proc = subprocess.Popen([SPEC["binary"], *tool_args], env=env)
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        returncode = proc.wait()
    finally:
        stop_event.set()
        refresher.join(timeout=1)

    raise SystemExit(returncode)


def validate_cmd(binary: str) -> list[str]:
    return [binary, "--print", "say hi in 5 words or less"]


def validate_env(state: dict) -> dict[str, str]:
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured.")
    return build_runtime_env(get_databricks_token(workspace, state.get("profile")))
