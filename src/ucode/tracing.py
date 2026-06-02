"""MLflow tracing: route coding-agent sessions to a Databricks experiment.

ucode points each agent's MLflow integration at a Databricks-hosted experiment
(tracking URI ``databricks``/``databricks://<profile>`` + a numeric experiment
id), reusing the workspace auth ucode already configures in ``~/.databrickscfg``.
Traces land in the experiment's default MLflow trace store (the MLflow Traces
UI), not Unity Catalog.

Scope: Claude Code, OpenCode, Codex. Gemini's exporter is OTLP-only and is not
wired here yet.

This module must not import ``mlflow`` (the heavy optional dependency) or
``ucode.agents`` at import time: agents import the small helpers here, and the
configure command imports agents lazily to avoid a cycle.
"""

from __future__ import annotations

from ucode.databricks import (
    ensure_databricks_auth,
    get_current_user_name,
    get_databricks_token,
    get_or_create_mlflow_experiment,
)
from ucode.state import hydrate_state, load_full_state, save_state
from ucode.ui import (
    print_kv,
    print_note,
    print_section,
    print_success,
    print_warning,
    prompt_for_workspace,
    spinner,
)

# Agents whose MLflow integration routes to a Databricks tracking URI. Gemini is
# excluded: its exporter is OTLP-only and needs a separate endpoint.
TRACING_AGENTS: tuple[str, ...] = ("claude", "opencode", "codex")

# Per-agent experiment-name slug, so each agent's sessions land in their own
# experiment (e.g. `/Users/<email>/ucode-claude-code-traces`) rather than a
# single shared one.
AGENT_EXPERIMENT_SLUG: dict[str, str] = {
    "claude": "claude-code",
    "codex": "codex",
    "opencode": "opencode",
}


def tracking_uri_for_state(state: dict) -> str:
    """The MLflow tracking URI for this workspace. ``databricks://<profile>``
    selects the matching ``~/.databrickscfg`` entry; bare ``databricks`` uses
    the default profile."""
    profile = state.get("profile")
    return f"databricks://{profile}" if profile else "databricks"


def experiment_name(tool: str, user_name: str | None) -> str:
    """Per-agent experiment path. Per-user (`/Users/<email>/...`) when the
    current user resolves, else a shared (`/Shared/...`) path."""
    slug = AGENT_EXPERIMENT_SLUG[tool]
    base = f"/Users/{user_name}" if user_name else "/Shared"
    return f"{base}/ucode-{slug}-traces"


def tracing_config(state: dict) -> dict | None:
    """Return the persisted tracing block iff tracing is enabled."""
    cfg = state.get("tracing")
    if isinstance(cfg, dict) and cfg.get("enabled"):
        return cfg
    return None


def agent_tracing(state: dict, tool: str) -> dict | None:
    """The resolved per-agent tracing entry ({experiment_id, experiment_name})
    for ``tool``, or None when tracing is off or that agent has no experiment."""
    cfg = tracing_config(state)
    if not cfg:
        return None
    entry = (cfg.get("agents") or {}).get(tool)
    if isinstance(entry, dict) and entry.get("experiment_id"):
        return entry
    return None


def tracing_env(state: dict, tool: str) -> dict[str, str]:
    """MLflow env vars for one agent. Empty when tracing is disabled for it. The
    tracking URI carries the profile, so auth resolves from ``~/.databrickscfg``
    without extra vars."""
    cfg = tracing_config(state)
    entry = agent_tracing(state, tool)
    if not cfg or not entry:
        return {}
    return {
        "MLFLOW_TRACKING_URI": str(cfg["tracking_uri"]),
        "MLFLOW_EXPERIMENT_ID": str(entry["experiment_id"]),
    }


def disable_tracing(state: dict) -> dict:
    """Mark tracing disabled and rewrite each configured agent's config so the
    injected tracing keys are stripped."""
    cfg = state.get("tracing")
    if isinstance(cfg, dict):
        cfg["enabled"] = False
        state["tracing"] = cfg
    save_state(state)
    return _rewrite_agent_configs(state)


def _configured_tracing_agents(state: dict) -> list[str]:
    available = set(state.get("available_tools") or [])
    return [tool for tool in TRACING_AGENTS if tool in available]


def _tracing_capable_workspaces(full: dict) -> list[str]:
    """Configured workspaces that have at least one tracing-capable agent."""
    workspaces = full.get("workspaces") or {}
    out: list[str] = []
    for ws, st in workspaces.items():
        if set((st or {}).get("available_tools") or []) & set(TRACING_AGENTS):
            out.append(ws)
    return out


def _select_tracing_workspace() -> dict:
    """Prompt for which configured workspace to trace, current first. Returns
    that workspace's hydrated flat state. Mirrors the top-level configure
    picker but offers only workspaces that already have tracing-capable agents."""
    full = load_full_state()
    workspaces = full.get("workspaces") or {}
    candidates = _tracing_capable_workspaces(full)
    if not candidates:
        raise RuntimeError(
            "No tracing-capable agents are configured. Run `ucode configure` for "
            "Claude Code, OpenCode, or Codex first."
        )

    current = full.get("current_workspace")
    candidates.sort(key=lambda ws: (ws != current, ws))
    # Coerce a missing profile to "" (falsy) so the type matches and downstream
    # resolves the default ~/.databrickscfg profile for that host.
    profiles = [(ws, (workspaces.get(ws) or {}).get("profile") or "") for ws in candidates]
    workspace, profile = prompt_for_workspace(
        "Select the workspace to configure MLflow tracing for", profiles
    )

    entry = dict(workspaces.get(workspace) or {})
    if not (set(entry.get("available_tools") or []) & set(TRACING_AGENTS)):
        raise RuntimeError(
            f"{workspace} has no tracing-capable agents configured. "
            "Run `ucode configure` for it first."
        )
    entry["workspace"] = workspace
    if profile:
        entry["profile"] = profile
    return hydrate_state(entry)


def _rewrite_agent_configs(state: dict) -> dict:
    """Re-run each configured agent's config writer so it folds the current
    tracing state into its config files (adds keys when enabled, strips them
    when disabled)."""
    from ucode.agents import configure_tool, default_model_for_tool

    for tool in _configured_tracing_agents(state):
        model = default_model_for_tool(tool, state)
        state = configure_tool(tool, state, model)
    return state


def _install_agent_tracing_deps(state: dict) -> None:
    """One-time, per-agent dependency installs (Claude plugin + mlflow CLI,
    Codex npm package), only for agents whose experiment resolved. OpenCode's
    plugin is auto-installed by OpenCode from the ``plugin`` list."""
    from ucode.agents import claude, codex

    if agent_tracing(state, "claude"):
        claude.ensure_tracing_runtime()
    if agent_tracing(state, "codex"):
        codex.ensure_tracing_dependency()


def configure_tracing_command(disable: bool = False) -> int:
    state = _select_tracing_workspace()
    workspace = state["workspace"]
    configured = _configured_tracing_agents(state)
    profile = state.get("profile")
    ensure_databricks_auth(workspace, profile)

    print_section("MLflow Tracing")
    print_kv("Workspace", workspace)

    # Running `ucode configure tracing` is itself the opt-in, so there's no
    # confirmation prompt; `--disable` is the explicit way back off.
    if disable:
        if (state.get("tracing") or {}).get("enabled"):
            disable_tracing(state)
            print_success("Tracing disabled")
        else:
            print_note("Tracing is not enabled — nothing to do.")
        return 0

    token = get_databricks_token(workspace, profile)
    user_name = get_current_user_name(workspace, token)

    agents_cfg: dict[str, dict] = {}
    for tool in configured:
        name = experiment_name(tool, user_name)
        with spinner(f"Resolving MLflow experiment for {tool}..."):
            exp_id, reason = get_or_create_mlflow_experiment(workspace, token, name)
        if not exp_id:
            print_warning(f"{tool}: could not resolve experiment {name}: {reason}")
            continue
        agents_cfg[tool] = {"experiment_id": exp_id, "experiment_name": name}

    if not agents_cfg:
        raise RuntimeError("Could not resolve an MLflow experiment for any configured agent.")

    state["tracing"] = {
        "enabled": True,
        "tracking_uri": tracking_uri_for_state(state),
        "agents": agents_cfg,
    }
    save_state(state)

    print_kv("Tracking URI", str(state["tracing"]["tracking_uri"]))
    for tool, entry in agents_cfg.items():
        print_kv(f"{tool} experiment", f"{entry['experiment_name']} (id {entry['experiment_id']})")

    _install_agent_tracing_deps(state)
    state = _rewrite_agent_configs(state)

    print_success(f"Tracing configured for: {', '.join(agents_cfg)}")
    return 0
