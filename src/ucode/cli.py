#!/usr/bin/env python3
"""CLI entry point for ucode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.panel import Panel

from ucode.agents import (
    TOOL_SPECS,
    check_gateway_endpoint,
    configure_selected_tools,
    configure_single_tool,
    configure_tool,
    default_model_for_tool,
    ensure_bootstrap_dependencies,
    ensure_provider_state,
    install_tool_binary,
    normalize_tool,
    resolve_launch_model,
    validate_all_tools,
    validate_tool,
)
from ucode.agents import (
    launch as launch_agent,
)
from ucode.agents.pi import PI_SETTINGS_BACKUP_PATH, PI_SETTINGS_PATH
from ucode.config_io import restore_file, set_dry_run
from ucode.databricks import (
    MANAGED_CONFIG_VOLUME_PATH,
    MANAGED_POLICIES_VOLUME_PATH,
    build_shared_base_urls,
    delete_managed_config,
    delete_managed_policies,
    discover_claude_models,
    discover_codex_models,
    discover_gemini_models,
    download_managed_config,
    download_managed_policies,
    ensure_ai_gateway_v2,
    ensure_databricks_auth,
    find_profile_name_for_host,
    get_databricks_profiles,
    get_databricks_token,
    install_databricks_cli,
    is_workspace_admin,
    normalize_workspace_url,
    run_databricks_login,
    upload_managed_config,
    upload_managed_policies,
)
from ucode.mcp import (
    MCP_CLIENTS,
    apply_mcp_server_changes,
    configure_mcp_command,
    purge_cross_workspace_mcp_residue,
    revert_mcp_configs,
)
from ucode.policies import (
    DEFAULT_POLICY_NAME,
    VALID_ON_BUDGET_EXHAUSTED,
    delete_workspace_policy,
    load_workspace_policy,
    normalize_policy,
    parse_and_validate_policy_yaml,
    parse_policy_yaml,
    policy_cache_path,
    save_workspace_policy,
)
from ucode.state import (
    STATE_PATH,
    clear_state,
    load_full_state,
    load_state,
    merge_managed_workspace,
    save_state,
    slice_state_for_export,
)
from ucode.tracing import configure_tracing_command, disable_tracing, install_tracing_runtime
from ucode.ui import (
    console,
    heading,
    print_err,
    print_err_panel,
    print_heading,
    print_kv,
    print_note,
    print_section,
    print_success,
    print_warning,
    prompt_budget_warn_choice,
    prompt_for_choice,
    prompt_for_tools,
    prompt_for_workspace,
    prompt_yes_no,
    spinner,
    status_badge,
)
from ucode.usage import (
    format_local_budget_status,
    local_budget_status,
    record_local_usage_delta,
    record_local_usage_snapshot,
    render_local_budget_panel,
)
from ucode.usage import (
    local_usage as local_usage_report,
)
from ucode.usage import (
    usage as usage_report,
)
from ucode.usage_hooks import (
    claude_usage_hook,
    codex_usage_hook,
    opencode_usage_hook,
    sync_opencode_usage_from_messages,
    sync_opencode_usage_from_state,
)

_DISCOVERY_CONSUMERS: dict[str, tuple[str, ...]] = {
    "claude": ("claude", "opencode", "copilot", "pi"),
    "codex": ("codex", "copilot", "pi"),
    "gemini": ("gemini", "opencode", "pi"),
}

# Agents that record local spend and therefore have a daily budget to report.
BUDGET_TRACKED_AGENTS: tuple[str, ...] = ("claude", "codex", "opencode")


def _tier_display(tier: dict) -> str:
    harness = str(tier.get("harness") or "agent")
    model = str(tier.get("model") or "model")
    display = TOOL_SPECS.get(harness, {}).get("display", harness)
    return f"{display} / {model}"


def _print_discovery_diagnostics(state: dict) -> None:
    """Surface per-source reasons after a failed discovery so the user knows
    which API call returned what — instead of the generic 'no agents' line."""
    reasons = state.get("_discovery_reasons") or {}
    if not reasons:
        return
    labels = {"claude": "Claude models", "codex": "Codex models", "gemini": "Gemini models"}
    for source, reason in reasons.items():
        consumers = ", ".join(_DISCOVERY_CONSUMERS.get(source, ()))
        label = labels.get(source, source)
        if reason:
            print_note(f"{label} (needed for: {consumers}): {reason}")
        else:
            print_note(f"{label} (needed for: {consumers}): no models returned")
    print_note("Re-run with `UCODE_DEBUG=1` to log raw discovery responses to ~/.ucode/debug.log.")


def _prompt_for_configuration(tool: str | None = None) -> tuple[str, str | None]:
    if tool is None:
        desc = "Configure your Databricks workspace"
    else:
        desc = f"Configure {TOOL_SPECS[tool]['display']} to use your Databricks endpoint."
    with spinner("Loading Databricks workspaces and profiles..."):
        profiles = get_databricks_profiles()
    return prompt_for_workspace(desc, profiles)


def _parse_agents_option(agents: str) -> list[str]:
    tools: list[str] = []
    for raw_tool in agents.split(","):
        raw_tool = raw_tool.strip()
        if not raw_tool:
            continue
        tool = normalize_tool(raw_tool)
        if tool not in tools:
            tools.append(tool)
    if not tools:
        raise RuntimeError(
            "No agents provided for --agents. Use a comma-separated list like `--agents claude,codex`."
        )
    return tools


def _managed_available_tools(state: dict) -> list[str] | None:
    if not state.get("_managed_config_pulled"):
        return None

    raw_tools = state.get("available_tools")
    if not isinstance(raw_tools, list):
        raw_tools = []

    tools: list[str] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, str):
            continue
        try:
            tool = normalize_tool(raw_tool)
        except RuntimeError:
            continue
        if tool not in tools:
            tools.append(tool)
    return tools


def _persistable_state(state: dict) -> dict:
    return {
        key: value
        for key, value in state.items()
        if key not in {"_managed_config_pulled", "_discovery_reasons"}
    }


def _parse_workspaces_option(workspaces: str) -> list[tuple[str, str | None]]:
    """Parse `--workspaces` into [(url, profile_name | None), ...].

    `--workspaces` supplies bare URLs; the matching profile (if any) is
    resolved later via `find_profile_name_for_host`.
    """
    workspace_entries: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for raw_workspace in workspaces.split(","):
        raw_workspace = raw_workspace.strip()
        if not raw_workspace:
            continue
        try:
            workspace = normalize_workspace_url(raw_workspace)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if workspace not in seen:
            seen.add(workspace)
            workspace_entries.append((workspace, None))
    if not workspace_entries:
        raise RuntimeError(
            "No workspaces provided for --workspaces. Use a comma-separated list like "
            "`--workspaces https://workspace.databricks.com`."
        )
    return workspace_entries


def configure_shared_state(
    workspace: str,
    profile: str | None = None,
    tools: list[str] | None = None,
    force_login: bool = False,
) -> dict:
    """Log into Databricks, enforce AI Gateway v2, fetch model lists, persist state.

    If tools is provided, only fetch models for those tools. Otherwise fetch all.
    If force_login is True, always run databricks auth login (used by explicit configure).
    ``profile`` is the Databricks CLI profile name to address — passed via
    ``--profile`` to every CLI invocation so ambiguous `~/.databrickscfg`
    entries (e.g. DEFAULT and a named profile both pointing at the same host)
    don't error out. If ``None``, we resolve it from the host after login.
    """
    workspace = normalize_workspace_url(workspace)
    previous_workspace = load_state().get("workspace")
    fetch_all = tools is None
    if force_login:
        run_databricks_login(workspace, profile)
    else:
        ensure_databricks_auth(workspace, profile)
    # After login the profile exists in ~/.databrickscfg, so a host->profile
    # lookup is reliable. Persist it so subsequent CLI calls disambiguate.
    if profile is None:
        profile = find_profile_name_for_host(workspace)
    with spinner("Verifying Unity AI Gateway..."):
        token = get_databricks_token(workspace, profile)
        ensure_ai_gateway_v2(workspace, token)
    print_success("Unity AI Gateway detected")

    want_claude = (
        fetch_all or "claude" in tools or "opencode" in tools or "copilot" in tools or "pi" in tools
    )
    want_gemini = fetch_all or "gemini" in tools or "opencode" in tools or "pi" in tools
    want_codex = fetch_all or "codex" in tools or "copilot" in tools or "pi" in tools

    claude_reason: str | None = None
    gemini_reason: str | None = None
    codex_reason: str | None = None
    with spinner("Fetching available models..."):
        if want_claude:
            claude_models, claude_reason = discover_claude_models(workspace, token)
        else:
            claude_models = {}
        if want_gemini:
            gemini_models, gemini_reason = discover_gemini_models(workspace, token)
        else:
            gemini_models = []
        if want_codex:
            codex_models, codex_reason = discover_codex_models(workspace, token)
        else:
            codex_models = []
    opencode_models: dict[str, list[str]] = {}
    if claude_models:
        opencode_models["anthropic"] = list(claude_models.values())
    if gemini_models:
        opencode_models["gemini"] = gemini_models

    # Merge into existing workspace state so prior tool configs are preserved.
    state = load_state()
    state["workspace"] = workspace
    if profile:
        state["profile"] = profile
    else:
        state.pop("profile", None)
    state["base_urls"] = build_shared_base_urls(workspace)
    if want_claude:
        state["claude_models"] = claude_models
    if want_gemini:
        state["gemini_models"] = gemini_models
    if want_codex:
        state["codex_models"] = codex_models
    if fetch_all or "opencode" in tools:
        state["opencode_models"] = opencode_models
    # Pull-and-replace: when the workspace publishes a UC state.json, that blob
    # is the authoritative workspace block. Local discovery above is only the
    # fallback for workspaces without a managed config.
    with spinner("Pulling managed config from your org..."):
        remote = download_managed_config(workspace, profile)
        remote_policy_yaml = download_managed_policies(workspace, profile)
    if remote is not None:
        state = merge_managed_workspace(state, remote, profile=profile)
        print_success("Managed config applied (set by your account admin)")
    else:
        print_note("No managed config published for this workspace — using local defaults.")

    if remote_policy_yaml is not None:
        remote_policy = parse_policy_yaml(remote_policy_yaml)
        if remote_policy is not None:
            save_workspace_policy(workspace, remote_policy)
            print_success("Managed policy applied (set by your account admin)")
        else:
            delete_workspace_policy(workspace)
            print_warning("Managed policies.yaml is malformed — ignoring budget policy.")
    elif remote is not None:
        delete_workspace_policy(workspace)

    save_state(state)
    if remote is not None:
        state = {**state, "_managed_config_pulled": True}
        # A pulled config may enable tracing. The agent config writer only
        # installs the Claude Stop hook when the pinned mlflow CLI is present,
        # so install the runtime now — otherwise the env vars get written but
        # the hook is silently skipped and no traces are emitted.
        install_tracing_runtime(state)
    # Scrub MCP entries that ucode wrote for the previous workspace so the new
    # workspace's agent configs aren't stale.
    if previous_workspace and previous_workspace != workspace:
        purge_cross_workspace_mcp_residue(state, workspace)
    # Diagnostic reasons are transient — attach after save_state so they don't
    # land on disk but are available to the caller for this run.
    state["_discovery_reasons"] = {
        "claude": claude_reason,
        "gemini": gemini_reason,
        "codex": codex_reason,
    }
    return state


def _configure_shared_workspace_states(
    workspaces: list[tuple[str, str | None]],
    tools: list[str] | None,
    *,
    force_login: bool,
) -> list[dict]:
    if not workspaces:
        raise RuntimeError("At least one workspace must be provided.")
    states: list[dict] = []
    for workspace, profile in workspaces:
        states.append(
            configure_shared_state(workspace, profile=profile, tools=tools, force_login=force_login)
        )
    return states


def configure_workspace_command(
    tool: str | None = None,
    selected_tools: list[str] | None = None,
    workspaces: list[tuple[str, str | None]] | None = None,
    *,
    update_existing: bool = True,
    validate: bool = True,
    panel_title: str | None = "Configuration Complete",
    pre_pulled_state: dict | None = None,
) -> int:
    """Configure one or more coding agents for the current/selected workspace."""
    if tool is not None and selected_tools is not None:
        raise RuntimeError("Use either --agent or --agents, not both.")

    workspace_entries = workspaces or [_prompt_for_configuration(tool)]

    if tool is not None:
        if pre_pulled_state is not None:
            states = [pre_pulled_state]
        else:
            states = _configure_shared_workspace_states(workspace_entries, [tool], force_login=True)
        state = states[0]
        state = configure_single_tool(tool, state)
        spec = TOOL_SPECS[tool]
        if panel_title is not None:
            _render_configuration_panel(state, configured_tools=[tool], title=panel_title)
        if not validate:
            return 0
        with spinner(f"Validating {spec['display']}..."):
            ok, err = validate_tool(tool)
        if ok:
            print_success(f"{spec['display']} is working")
        else:
            print_err(f"{spec['display']}: {err}")
            managed = bool(state.get("managed_configs", {}).get(tool))
            restore_file(spec["config_path"], spec["backup_path"], managed)
            available_tools = [t for t in (state.get("available_tools") or []) if t != tool]
            state["available_tools"] = available_tools
            save_state(state)
            raise RuntimeError(f"{spec['display']} validation failed — config reverted.")
        return 0

    if pre_pulled_state is not None:
        states = [pre_pulled_state]
    else:
        states = _configure_shared_workspace_states(
            workspace_entries, selected_tools, force_login=True
        )
    state = states[0]
    save_state(_persistable_state(state))

    managed_tools = _managed_available_tools(state) if selected_tools is None else None
    available_on_workspace: list[str] = []
    if selected_tools is not None:
        tools_to_check = selected_tools
    elif managed_tools is not None:
        tools_to_check = managed_tools
    else:
        tools_to_check = list(TOOL_SPECS)

    if managed_tools == []:
        print_note("No coding agents selected — nothing to configure.")
        return 0

    for tool_name in tools_to_check:
        with spinner(f"Checking {TOOL_SPECS[tool_name]['display']} availability..."):
            if check_gateway_endpoint(state, tool_name):
                available_on_workspace.append(tool_name)

    if not available_on_workspace:
        print_err("No coding agents are available on this workspace.")
        _print_discovery_diagnostics(state)
        return 1

    if managed_tools is not None:
        unavailable_tools = [
            tool_name for tool_name in managed_tools if tool_name not in available_on_workspace
        ]
        if unavailable_tools:
            _print_discovery_diagnostics(state)
            displays = ", ".join(
                TOOL_SPECS[tool_name]["display"] for tool_name in unavailable_tools
            )
            raise RuntimeError(f"Managed agent(s) not available on this workspace: {displays}.")
        picked = managed_tools
    elif selected_tools is None:
        picked = prompt_for_tools([(t, TOOL_SPECS[t]["display"]) for t in available_on_workspace])
    else:
        unavailable_tools = [
            tool_name for tool_name in selected_tools if tool_name not in available_on_workspace
        ]
        if unavailable_tools:
            _print_discovery_diagnostics(state)
            displays = ", ".join(
                TOOL_SPECS[tool_name]["display"] for tool_name in unavailable_tools
            )
            raise RuntimeError(f"Requested agent(s) not available on this workspace: {displays}.")
        picked = selected_tools

    if not picked:
        print_note("No coding agents selected — nothing to configure.")
        return 0

    for tool_name in picked:
        install_tool_binary(tool_name, strict=False, update_existing=update_existing)

    state = configure_selected_tools(_persistable_state(state), picked)

    if panel_title is not None:
        _render_configuration_panel(state, configured_tools=picked, title=panel_title)

    if validate:
        # Limit validation to just-configured tools so we don't re-validate
        # previously-configured tools the user didn't touch this run.
        validate_state = {**state, "available_tools": picked}
        validate_all_tools(validate_state)
    return 0


def _harness_display(harness: str) -> str:
    spec = TOOL_SPECS.get(harness)
    if isinstance(spec, dict):
        display = spec.get("display")
        if isinstance(display, str) and display:
            return display
    return harness


def _render_policy_summary_lines(
    workspace: str | None,
    *,
    policy_override: dict | None = None,
) -> list[str]:
    if policy_override is not None:
        policy = policy_override
    else:
        policy = load_workspace_policy(workspace) if isinstance(workspace, str) else None
    root = policy.get("policy") if isinstance(policy, dict) else None
    if not isinstance(root, dict):
        return []

    name = str(root.get("name") or DEFAULT_POLICY_NAME)
    daily = root.get("daily_budget_usd")
    daily_display = (
        f"${float(daily):.2f}/day"
        if isinstance(daily, (int, float)) and not isinstance(daily, bool)
        else "no daily cap"
    )
    on_exhausted_raw = root.get("on_budget_exhausted")
    if isinstance(on_exhausted_raw, dict) and on_exhausted_raw.get("action") == "switch":
        target = on_exhausted_raw.get("target") or {}
        target_harness = str(target.get("harness") or "") if isinstance(target, dict) else ""
        target_model = str(target.get("model") or "") if isinstance(target, dict) else ""
        on_exhausted_line = (
            f"  [dim]·[/dim] [bold]at 100%[/bold] "
            f"→ {_harness_display(target_harness)} · [magenta]{target_model}[/magenta]"
        )
    elif isinstance(on_exhausted_raw, str) and on_exhausted_raw in VALID_ON_BUDGET_EXHAUSTED:
        on_exhausted_line = (
            f"  [dim]·[/dim] [bold]at 100%[/bold] → [magenta]{on_exhausted_raw}[/magenta]"
        )
    else:
        on_exhausted_line = "  [dim]·[/dim] [bold]at 100%[/bold] → [magenta]block[/magenta]"

    lines = [f"[bold]Policy:[/bold] [cyan]{name}[/cyan]"]
    lines.append(f"  [dim]·[/dim] [bold]budget[/bold] → [magenta]{daily_display}[/magenta]")
    tiers = root.get("tiers") if isinstance(root, dict) else None
    if isinstance(tiers, list):
        for tier in tiers:
            if not isinstance(tier, dict):
                continue
            tier_name = str(tier.get("name") or "?")
            pct_raw = tier.get("activates_at_pct")
            pct = (
                f"{float(pct_raw):.0f}%"
                if isinstance(pct_raw, (int, float)) and not isinstance(pct_raw, bool)
                else "?"
            )
            harness = str(tier.get("harness") or "?")
            model = str(tier.get("model") or "?")
            lines.append(
                f"  [dim]·[/dim] [bold]{tier_name}[/bold] [dim]({pct})[/dim] "
                f"→ {_harness_display(harness)} · [magenta]{model}[/magenta]"
            )
    lines.append(on_exhausted_line)
    return lines


def _clear_managed_tracing(state: dict) -> bool:
    """Disable tracing locally and drop the ``tracing`` block from state so the
    next export publishes a state.json without any tracing section. Returns
    ``True`` when a previously-enabled tracing block was removed."""
    tracing = state.get("tracing")
    was_enabled = bool(tracing.get("enabled")) if isinstance(tracing, dict) else False
    if not was_enabled:
        return False
    disable_tracing(state)
    cleared = load_state()
    cleared.pop("tracing", None)
    save_state(cleared)
    return True


def _clear_managed_mcps(state: dict) -> bool:
    """Tear down every per-client MCP registration recorded in state and clear
    ``mcp_servers`` so the next export publishes a state.json with no MCPs.
    Returns ``True`` when entries were actually removed."""
    original = list(state.get("mcp_servers") or [])
    if not original:
        return False
    apply_mcp_server_changes(original, [], [])
    cleared = load_state()
    cleared["mcp_servers"] = []
    save_state(cleared)
    return True


def _clear_managed_policy(workspace: str, profile: str | None) -> tuple[bool, bool]:
    """Remove the workspace's local policies.yaml cache and, if present, the
    published copy in Unity Catalog. Returns ``(local_removed, uc_removed)``.
    UC failures are surfaced as warnings (not exceptions) so the rest of setup
    can complete; the local file is always cleared so a subsequent export
    won't republish a stale policy."""
    policy_path = policy_cache_path(workspace)
    local_existed = policy_path.is_file()
    if local_existed:
        delete_workspace_policy(workspace)
    uc_removed = False
    try:
        with spinner("Removing published policies.yaml from Unity Catalog..."):
            uc_removed = delete_managed_policies(workspace, profile)
    except RuntimeError as exc:
        print_warning(f"Could not remove {MANAGED_POLICIES_VOLUME_PATH} from Unity Catalog: {exc}")
    return local_existed, uc_removed


def _resolve_default_agent(workspace: str | None, configured_tools: list[str]) -> tuple[str, str]:
    """Pick the workspace's default coding agent as the harness named by policy
    tier 1, or if no policies, the first entry in ``available_tools``."""
    if not configured_tools:
        raise RuntimeError("Cannot pick a default agent — no tools configured.")
    policy = load_workspace_policy(workspace) if isinstance(workspace, str) else None
    root = policy.get("policy") if isinstance(policy, dict) else None
    tiers = root.get("tiers") if isinstance(root, dict) else None
    if isinstance(tiers, list) and tiers:
        first_tier = tiers[0] if isinstance(tiers[0], dict) else None
        harness = first_tier.get("harness") if isinstance(first_tier, dict) else None
        if isinstance(harness, str) and harness in configured_tools:
            return harness, "from policy tier 1"
    return configured_tools[0], "first configured agent"


def _render_configuration_panel(
    state: dict,
    configured_tools: list[str] | None = None,
    *,
    title: str = "Configuration Complete",
    policy_override: dict | None = None,
) -> None:
    """Render the configuration summary panel."""
    tools_to_show = configured_tools or list(state.get("available_tools") or [])

    summary_lines = [f"[bold]Workspace:[/bold] [cyan]{state.get('workspace', '?')}[/cyan]"]
    for tool_name in tools_to_show:
        spec = TOOL_SPECS.get(tool_name)
        if spec is None:
            continue
        summary_lines.append(f"[bold]{spec['display']}:[/bold] [green]configured[/green]")

    mcp_names = sorted(
        {
            str(server.get("name"))
            for server in (state.get("mcp_servers") or [])
            if server.get("name")
        }
    )
    summary_lines.append(
        f"[bold]MCPs:[/bold] {', '.join(mcp_names) if mcp_names else '[dim]none[/dim]'}"
    )

    tracing = state.get("tracing") or {}
    tracing_enabled = bool(tracing.get("enabled")) if isinstance(tracing, dict) else False
    summary_lines.append(
        f"[bold]Tracing:[/bold] "
        f"{'[green]enabled[/green]' if tracing_enabled else '[dim]disabled[/dim]'}"
    )

    # When a managed policies.yaml exists, surface it (name + daily budget +
    # tiers + on-exhausted). Falls back to the legacy state.policies daily
    # limit row when no YAML is published.
    policy_lines = _render_policy_summary_lines(
        state.get("workspace"), policy_override=policy_override
    )
    if policy_lines:
        summary_lines.extend(policy_lines)
    else:
        policies = state.get("policies") or {}
        daily_limit = policies.get("daily_limit_usd") if isinstance(policies, dict) else None
        if isinstance(daily_limit, (int, float)) and not isinstance(daily_limit, bool):
            summary_lines.append(f"[bold]Daily budget:[/bold] ${float(daily_limit):.2f}")
        else:
            summary_lines.append("[bold]Policy:[/bold] [dim]not set[/dim]")

    allowlist = state.get("allowlist")
    if isinstance(allowlist, list) and allowlist:
        allowlist_display = ", ".join(str(item) for item in allowlist)
        managed_suffix = "  [dim](managed)[/dim]" if state.get("_managed_config_pulled") else ""
        summary_lines.append(f"[bold]Allowlist:[/bold] {allowlist_display}{managed_suffix}")

    console.print(
        Panel(
            "\n".join(summary_lines),
            title=title,
            style="green",
            expand=False,
        )
    )


def status() -> int:
    state = load_state()
    workspace = state.get("workspace")
    managed_configs = state.get("managed_configs") or {}
    mcp_servers = state.get("mcp_servers") or []
    configured_tools = set(state.get("available_tools") or managed_configs.keys())

    console.print(heading("ucode status"))
    console.print(
        f"  {status_badge('Configured', 'ok') if workspace else status_badge('Not Configured', 'warn')}"
    )

    print_heading("Provider")
    print_kv("Workspace URL", workspace or "not configured")
    profile = state.get("profile")
    if profile:
        print_kv("CLI profile", profile)

    print_heading("Coding Agents")
    for tool, spec in TOOL_SPECS.items():
        configured = tool in configured_tools
        base_url = (
            state.get("base_urls", {}).get(tool, "not configured")
            if configured
            else "not configured"
        )
        config_path = spec["config_path"]
        print_kv("Coding Agent", spec["display"])
        print_kv("Configured", "yes" if configured else "no")
        print_kv("Base URL", base_url)
        if configured and tool in MCP_CLIENTS:
            tool_mcp_servers = [
                str(server.get("name"))
                for server in mcp_servers
                if tool in (server.get("clients") or []) and server.get("name")
            ]
            print_kv("MCP list command", str(MCP_CLIENTS[tool]["list_command"]))
            print_kv(
                "MCP servers",
                ", ".join(tool_mcp_servers) if tool_mcp_servers else "none saved by ucode",
            )
        print_kv("Config file", str(config_path) if config_path.exists() else "missing")
        console.print()

    print_heading("Tracing")
    tracing = state.get("tracing") or {}
    if tracing.get("enabled"):
        print_kv("Tracing", "enabled")
        print_kv("Tracking URI", str(tracing.get("tracking_uri") or "unknown"))
        print_kv(
            "Experiment",
            f"{tracing.get('experiment_name')} (id {tracing.get('experiment_id')})",
        )
        uc_destination = tracing.get("uc_destination")
        if uc_destination:
            print_kv("Unity Catalog", str(uc_destination))
        sql_warehouse_id = tracing.get("sql_warehouse_id")
        if sql_warehouse_id:
            print_kv("SQL warehouse", str(sql_warehouse_id))
    else:
        print_kv("Tracing", "disabled")

    print_heading("Policy")
    workspace_str = workspace if isinstance(workspace, str) else None
    policy = load_workspace_policy(workspace_str) if workspace_str else None
    policy_root = policy.get("policy") if isinstance(policy, dict) else None
    if isinstance(policy_root, dict) and workspace_str:
        print_kv("Name", str(policy_root.get("name") or DEFAULT_POLICY_NAME))
        daily = policy_root.get("daily_budget_usd")
        print_kv(
            "Daily budget",
            f"${float(daily):.2f}"
            if isinstance(daily, (int, float)) and not isinstance(daily, bool)
            else "not set",
        )
        on_exhausted = policy_root.get("on_budget_exhausted")
        if isinstance(on_exhausted, dict) and on_exhausted.get("action") == "switch":
            target = on_exhausted.get("target") or {}
            target_harness = target.get("harness", "") if isinstance(target, dict) else ""
            target_model = target.get("model", "") if isinstance(target, dict) else ""
            on_exhausted_display = f"switch → {target_harness} ({target_model})"
        elif isinstance(on_exhausted, str) and on_exhausted in VALID_ON_BUDGET_EXHAUSTED:
            on_exhausted_display = on_exhausted
        else:
            on_exhausted_display = "block"
        print_kv("On budget exhausted", on_exhausted_display)
        tiers = policy_root.get("tiers")
        if isinstance(tiers, list) and tiers:
            for tier in tiers:
                if not isinstance(tier, dict):
                    continue
                tier_name = str(tier.get("name") or "?")
                pct_raw = tier.get("activates_at_pct")
                pct = (
                    f"{float(pct_raw):.0f}%"
                    if isinstance(pct_raw, (int, float)) and not isinstance(pct_raw, bool)
                    else "?"
                )
                harness = str(tier.get("harness") or "?")
                model = str(tier.get("model") or "?")
                print_kv(
                    f"Tier · {tier_name} ({pct})",
                    f"{_harness_display(harness)} → {model}",
                )
        else:
            print_kv("Tiers", "none")
        print_kv("Policy file", str(policy_cache_path(workspace_str)))
    else:
        print_kv("Policy", "not set")
        if workspace_str:
            print_note(
                "Run `ucode setup budget` (admin) to author a policy, "
                "or `ucode configure` to pull the published one."
            )

    print_heading("State")
    print_kv("State file", str(STATE_PATH) if STATE_PATH.exists() else "missing")
    print_note("Use `ucode configure` to update workspace settings or configure new tools.")
    print_note(
        "Use `ucode configure mcp` to add Databricks MCP servers to configured coding tools."
    )
    print_note("Use `ucode configure tracing` to log coding sessions to an MLflow experiment.")
    print_note("Use `ucode revert` to clear managed configs and restore prior files.")
    return 0


def revert() -> int:
    state = load_state()
    managed_configs = state.get("managed_configs") or {}
    mcp_results = revert_mcp_configs(state)

    results: dict[str, bool] = {
        tool: restore_file(
            spec["config_path"], spec["backup_path"], bool(managed_configs.get(tool))
        )
        for tool, spec in TOOL_SPECS.items()
    }
    pi_settings_restored = restore_file(
        PI_SETTINGS_PATH, PI_SETTINGS_BACKUP_PATH, bool(managed_configs.get("pi"))
    )
    clear_state()

    print_heading("Revert")
    print_kv("Workspace", state.get("workspace") or "none")
    for tool, spec in TOOL_SPECS.items():
        print_kv(f"{spec['display']} config", "restored" if results[tool] else "unchanged")
    print_kv("Pi settings", "restored" if pi_settings_restored else "unchanged")
    for client, spec in MCP_CLIENTS.items():
        print_kv(
            f"{spec['display']} MCP config",
            "restored" if mcp_results.get(client) else "unchanged",
        )
    print_success("ucode state cleared")
    return 0


# ---------------------------------------------------------------------------
# typer app
# ---------------------------------------------------------------------------


app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
configure_app = typer.Typer(add_completion=False, no_args_is_help=False)
app.add_typer(configure_app, name="configure", help="Configure workspace and tool settings.")
setup_app = typer.Typer(add_completion=False)
app.add_typer(setup_app, name="setup", help="Configure workspace policy settings.")
mcp_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(mcp_app, name="mcp", help="MCP servers exposed by ucode.")
usage_app = typer.Typer(add_completion=False, no_args_is_help=False)
app.add_typer(usage_app, name="usage", help="Show or record coding-agent usage.")


@app.callback(invoke_without_command=True)
def default_launch(ctx: typer.Context) -> None:
    """Launch the configured default agent when ``ucode`` is run with no subcommand.

    Runs `ucode configure` first if the workspace isn't set up yet, then launches
    `default_agent` — falling back to another configured agent when the default's
    daily budget is exhausted."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        install_databricks_cli()
        if not load_state().get("workspace"):
            configure_workspace_command()
        state = load_state()
        if not state.get("workspace"):
            return
        tool = _resolve_default_launch_tool(state)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None
    _launch_tool(tool, ctx)


@mcp_app.command("web-search")
def mcp_web_search_cmd() -> None:
    """Run the web_search MCP server over stdio. Invoked as a subprocess by Claude Code."""
    from ucode.mcp_web_search import serve

    serve()


def _auto_configure_tool(tool: str) -> None:
    """First-time setup for a single tool — mirrors configure_workspace_command."""
    existing = load_state()
    workspace = existing.get("workspace")
    profile = existing.get("profile")
    if not workspace:
        workspace, profile = _prompt_for_configuration(tool)
    state = configure_shared_state(workspace, profile=profile, tools=[tool])

    state = configure_single_tool(tool, state)

    spec = TOOL_SPECS[tool]
    _render_configuration_panel(state, configured_tools=[tool])

    with spinner(f"Validating {spec['display']}..."):
        ok, err = validate_tool(tool)
    if ok:
        print_success(f"{spec['display']} is working")
    else:
        print_err(f"{spec['display']}: {err}")
        managed = bool(state.get("managed_configs", {}).get(tool))
        restore_file(spec["config_path"], spec["backup_path"], managed)
        available_tools = [t for t in (state.get("available_tools") or []) if t != tool]
        state["available_tools"] = available_tools
        save_state(state)
        raise RuntimeError(f"{spec['display']} validation failed — config reverted.")


def _resolve_default_launch_tool(state: dict) -> str:
    """Pick which agent bare ``ucode`` should launch.

    Returns ``default_agent`` (falling back to the first configured agent when
    it's unset or no longer available). The daily budget is a single global pool,
    so budget gating lives entirely in ``_apply_budget_gate`` at launch time
    rather than in per-agent selection here."""
    available = state.get("available_tools") or []
    if not available:
        raise RuntimeError("No coding agents are configured. Run `ucode configure` to set one up.")
    default_agent = state.get("default_agent")
    if default_agent not in available:
        default_agent = available[0]
    return default_agent


def _budget_switch_target(tool: str, status: dict[str, object]) -> tuple[str, str] | None:
    tier = status.get("active_tier")
    if not isinstance(tier, dict):
        return None
    tier = cast("dict[str, object]", tier)
    harness = tier.get("harness")
    model = tier.get("model")
    if (
        not isinstance(harness, str)
        or harness not in TOOL_SPECS
        or harness == tool
        or not isinstance(model, str)
        or not model
    ):
        return None
    return harness, model


def _apply_budget_gate(
    tool: str,
    *,
    offer_warn_selector: bool,
) -> tuple[str, str | None, str | None]:
    """Gate launch on the global daily budget, returning a decision string.

    The budget is a single global pool shared across every coding tool:
    - ``exceeded`` follows ``policy.on_budget_exhausted``: block, warn, or allow.
    - ``warn`` offers an interactive selector (when ``offer_warn_selector``) to
      continue with the agent being launched or switch to the active policy tier.
    - ``ok`` passes through.

    Returns ``("default", None, None)`` or ``("switch", harness, model)``.
    Raises ``typer.Exit`` on the exceeded hard stop and on quit/abort from the
    selector."""
    status = local_budget_status()
    budget_state = status.get("state")
    if budget_state == "exceeded":
        behavior = status.get("on_budget_exhausted") or "block"
        if (
            isinstance(behavior, dict)
            and cast("dict[str, object]", behavior).get("action") == "switch"
        ):
            behavior_dict = cast("dict[str, object]", behavior)
            target = behavior_dict.get("target") or {}
            target_dict = cast("dict[str, object]", target) if isinstance(target, dict) else {}
            harness = target_dict.get("harness")
            model = target_dict.get("model")
            if isinstance(harness, str) and harness and isinstance(model, str) and model:
                console.print(render_local_budget_panel(status, title="Daily budget exhausted"))
                print_warning(
                    f"Daily budget exhausted — switching to"
                    f" {TOOL_SPECS.get(harness, {}).get('display', harness)} ({model})."
                )
                return "switch", harness, model
        if behavior == "block":
            console.print(render_local_budget_panel(status, title="Daily budget exhausted"))
            print_err("Daily budget exhausted — no agents can be launched until it resets.")
            raise typer.Exit(1)
        if behavior == "warn":
            console.print(render_local_budget_panel(status, title="Daily budget exhausted"))
            print_warning("Daily budget exhausted — continuing because policy is set to warn.")
    if budget_state == "warn" and offer_warn_selector:
        target = _budget_switch_target(tool, status)
        if target is None:
            return "default", None, None
        switch_display = None
        tier = status.get("active_tier")
        switch_display = _tier_display(tier) if isinstance(tier, dict) else target[0]
        choice = prompt_budget_warn_choice(
            default_agent_display=TOOL_SPECS[tool]["display"],
            switch_display=switch_display,
        )
        if choice is None:
            print_note("Cancelled.")
            raise typer.Exit(0)
        if choice == "switch" and target is not None:
            target_tool, target_model = target
            return "switch", target_tool, target_model
    return "default", None, None


def _launch_tool(
    tool_name: str,
    ctx: typer.Context,
    *,
    explicit_model: str | None = None,
    _budget_redirected: bool = False,
) -> None:
    try:
        tool = normalize_tool(tool_name)
        existing = load_state()
        needs_auto_configure = not existing.get("workspace") or tool not in (
            existing.get("available_tools") or []
        )
        ensure_bootstrap_dependencies(tool, update_existing=needs_auto_configure)
        if needs_auto_configure:
            _auto_configure_tool(tool)
        state = ensure_provider_state(tool)
        # Re-fetch model lists on every launch so newly-added Databricks
        # endpoints show up without a manual `ucode configure` (and so that
        # tools like pi which read multiple model bundles never run on
        # stale state from before a tool added a new bundle).
        state = configure_shared_state(
            state["workspace"], profile=state.get("profile"), tools=[tool]
        )
        # Gate on the global daily budget before configuring/launching.
        # Skip the gate on budget-redirect calls — the switch already fired and
        # re-gating would loop forever since the budget is still exceeded.
        if not _budget_redirected:
            decision, target_tool, target_model = _apply_budget_gate(
                tool, offer_warn_selector=tool in BUDGET_TRACKED_AGENTS and explicit_model is None
            )
            if decision == "switch" and target_tool:
                return _launch_tool(
                    target_tool, ctx, explicit_model=target_model, _budget_redirected=True
                )
        state, resolved_model = resolve_launch_model(tool, state, explicit_model)
        state = configure_tool(tool, state, resolved_model)
        display = TOOL_SPECS[tool]["display"]
        _render_configuration_panel(state)
        if tool in BUDGET_TRACKED_AGENTS:
            # The daily budget is a single global pool shared across all tools,
            # so render it tool-agnostically (no per-tool label in the callout).
            # The panel title still names the tool being launched. The exceeded
            # state is already handled by `_apply_budget_gate` above, so this
            # panel only reports ok/warn spend here.
            console.print(
                render_local_budget_panel(
                    local_budget_status(),
                    title=f"ucode with {display}",
                )
            )
        else:
            print_section(f"ucode with {display}")
            if resolved_model:
                print_kv("Model", resolved_model)
        if tool in ("gemini", "opencode", "copilot", "pi"):
            print_note(
                f"{display} token refresh is managed automatically "
                f"every 30 minutes while the session is running."
            )
        print_success(f"Starting {display}")
        launch_agent(tool, state, ctx.args)
    except typer.Exit:
        # `_apply_budget_gate` raises typer.Exit for the budget hard stop and the
        # quit/abort selector path. typer.Exit subclasses RuntimeError, so let it
        # propagate with its intended exit code instead of the generic handler
        # below reporting it as an error.
        raise
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@app.command("codex", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def codex_cmd(ctx: typer.Context) -> None:
    """Launch Codex via Databricks."""
    _launch_tool("codex", ctx)


@app.command("claude", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def claude_cmd(ctx: typer.Context) -> None:
    """Launch Claude Code via Databricks."""
    _launch_tool("claude", ctx)


@app.command("gemini", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def gemini_cmd(ctx: typer.Context) -> None:
    """Launch Gemini CLI via Databricks."""
    _launch_tool("gemini", ctx)


@app.command(
    "opencode", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def opencode_cmd(ctx: typer.Context) -> None:
    """Launch OpenCode via Databricks."""
    _launch_tool("opencode", ctx)


@app.command("copilot", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def copilot_cmd(ctx: typer.Context) -> None:
    """Launch GitHub Copilot CLI via Databricks."""
    _launch_tool("copilot", ctx)


@app.command("pi", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def pi_cmd(ctx: typer.Context) -> None:
    """Launch Pi coding agent via Databricks."""
    _launch_tool("pi", ctx)


@configure_app.callback(invoke_without_command=True)
def configure(
    ctx: typer.Context,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print config files without writing them.")
    ] = False,
    agent: Annotated[
        str | None,
        typer.Option(
            "--agent",
            help="Configure only the named agent (e.g. claude, codex, gemini, opencode, copilot, pi).",
        ),
    ] = None,
    agents: Annotated[
        str | None,
        typer.Option(
            "--agents",
            help="Configure a comma-separated list of agents without prompting (e.g. claude,codex).",
        ),
    ] = None,
    workspaces: Annotated[
        str | None,
        typer.Option(
            "--workspaces",
            help="Configure a comma-separated list of workspaces without prompting.",
        ),
    ] = None,
    tracing: Annotated[
        bool,
        typer.Option(
            "--tracing",
            help="Also enable Tracing for the configured workspace(s).",
        ),
    ] = False,
) -> None:
    """Configure workspace URL and AI Gateway."""
    if ctx.invoked_subcommand is not None:
        return
    set_dry_run(dry_run)
    try:
        install_databricks_cli()
        if agent is not None and agents is not None:
            raise RuntimeError("Use either --agent or --agents, not both.")
        workspace_entries = _parse_workspaces_option(workspaces) if workspaces is not None else None
        if agent is not None:
            tool = normalize_tool(agent)
            install_tool_binary(tool, strict=True, update_existing=True)
            if workspace_entries is None:
                configure_workspace_command(tool)
            else:
                configure_workspace_command(tool, workspaces=workspace_entries)
        elif agents is not None:
            selected_tools = _parse_agents_option(agents)
            if workspace_entries is None:
                configure_workspace_command(selected_tools=selected_tools)
            else:
                configure_workspace_command(
                    selected_tools=selected_tools, workspaces=workspace_entries
                )
        else:
            # Tool binaries are installed after the user picks which agents
            # they want, in configure_workspace_command.
            if workspace_entries is None:
                configure_workspace_command()
            else:
                configure_workspace_command(workspaces=workspace_entries)
        if tracing:
            # The workspaces were just configured, so enable tracing for them
            # directly instead of re-prompting. Fall back to the workspace that
            # `configure_workspace_command` made current (the interactive pick).
            tracing_workspaces = workspace_entries
            if tracing_workspaces is None:
                current = load_full_state().get("current_workspace")
                tracing_workspaces = [(current, None)] if current else None
            if tracing_workspaces:
                configure_tracing_command(workspaces=tracing_workspaces)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@configure_app.command("mcp")
def configure_mcp() -> None:
    """Add Databricks MCP servers to installed coding tools."""
    try:
        configure_mcp_command()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@configure_app.command("tracing")
def configure_tracing(
    disable: Annotated[
        bool, typer.Option("--disable", help="Turn off Tracing for configured agents.")
    ] = False,
) -> None:
    """Send coding-session traces to an MLflow experiment in your workspace."""
    try:
        install_databricks_cli()
        configure_tracing_command(disable=disable)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


def _prompt_text(prompt: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        raw = console.input(f"{prompt}{suffix} › ").strip()
        if raw:
            return raw
        if default:
            return default
        print_err("Value cannot be empty.")


def _prompt_float(
    prompt: str,
    *,
    default: float | None = None,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    suffix = f" [{default:g}]" if default is not None else ""
    while True:
        raw = console.input(f"{prompt}{suffix} › ").strip()
        if not raw and default is not None:
            return default
        try:
            value = float(raw)
        except ValueError:
            print_err("Please enter a valid number.")
            continue
        if value < minimum or (maximum is not None and value > maximum):
            if maximum is None:
                print_err(f"Please enter at least {minimum:g}.")
            else:
                print_err(f"Please enter a value from {minimum:g} to {maximum:g}.")
            continue
        return value


def _prompt_int(prompt: str, *, default: int, minimum: int = 1) -> int:
    while True:
        value = _prompt_float(prompt, default=float(default), minimum=float(minimum))
        if value.is_integer():
            return int(value)
        print_err("Please enter a whole number.")


def _existing_policy_tiers(policy: dict | None) -> list[dict]:
    root = policy.get("policy") if isinstance(policy, dict) else None
    tiers = root.get("tiers") if isinstance(root, dict) else None
    return [tier for tier in tiers if isinstance(tier, dict)] if isinstance(tiers, list) else []


def _default_policy_tiers(state: dict) -> list[dict[str, object]]:
    primary = state.get("default_agent")
    if not isinstance(primary, str) or primary not in TOOL_SPECS:
        available = state.get("available_tools") or []
        primary = available[0] if available and available[0] in TOOL_SPECS else "claude"
    primary_model = default_model_for_tool(primary, state) or ""
    standard_model = primary_model
    if primary == "claude":
        claude_models = state.get("claude_models") or {}
        if isinstance(claude_models, dict):
            standard_model = str(claude_models.get("sonnet") or primary_model)
    economy_harness = "opencode" if "opencode" in TOOL_SPECS else primary
    economy_model = default_model_for_tool(economy_harness, state) or primary_model
    return [
        {
            "name": "premium",
            "activates_at_pct": 0.0,
            "harness": primary,
            "model": primary_model,
        },
        {
            "name": "standard",
            "activates_at_pct": 60.0,
            "harness": primary,
            "model": standard_model,
        },
        {
            "name": "economy",
            "activates_at_pct": 80.0,
            "harness": economy_harness,
            "model": economy_model,
        },
    ]


def _policy_model_options(harness: str, state: dict) -> list[tuple[str, str]]:
    if harness == "claude":
        claude_models = state.get("claude_models") or {}
        if not isinstance(claude_models, dict):
            return []
        family_order = {"opus": 0, "sonnet": 1, "haiku": 2}
        items = [
            (str(model), f"{family} ({model})")
            for family, model in claude_models.items()
            if isinstance(family, str) and isinstance(model, str) and model
        ]
        return sorted(items, key=lambda item: family_order.get(item[1].split(" ", 1)[0], 99))

    if harness == "opencode":
        opencode_models = state.get("opencode_models") or {}
        if not isinstance(opencode_models, dict):
            return []
        options: list[tuple[str, str]] = []
        for provider in ("anthropic", "gemini"):
            models = opencode_models.get(provider) or []
            if isinstance(models, list):
                options.extend(
                    (str(model), f"{provider} ({model})")
                    for model in models
                    if isinstance(model, str)
                )
        return options

    if harness == "pi":
        options = []
        for value in (state.get("claude_models") or {}).values():
            if isinstance(value, str):
                options.append((value, f"claude ({value})"))
        codex_models = state.get("codex_models") or []
        if isinstance(codex_models, list):
            options.extend(
                (str(model), f"codex ({model})") for model in codex_models if isinstance(model, str)
            )
        gemini_models = state.get("gemini_models") or []
        if isinstance(gemini_models, list):
            options.extend(
                (str(model), f"gemini ({model})")
                for model in gemini_models
                if isinstance(model, str)
            )
        return options

    state_key = f"{harness}_models"
    models = state.get(state_key) or []
    if not isinstance(models, list):
        return []
    return [(str(model), str(model)) for model in models if isinstance(model, str)]


def _prompt_policy_model(harness: str, state: dict, default: str | None = None) -> str:
    options = _policy_model_options(harness, state)
    if default and default not in {model for model, _label in options}:
        options = [(default, f"{default} (current)"), *options]
    if options:
        return prompt_for_choice("Model", options)
    return _prompt_text("Model", default=default)


def _run_budget_setup(workspace: str) -> None:
    """Interactively author the workspace's ``policies.yaml``."""
    state = load_state()
    existing_policy = load_workspace_policy(workspace)
    existing_root = existing_policy.get("policy") if isinstance(existing_policy, dict) else {}
    if not isinstance(existing_root, dict):
        existing_root = {}

    print_section("Budget")
    print_kv("Workspace", workspace)
    current_limit = existing_root.get("daily_budget_usd")
    if isinstance(current_limit, (int, float)) and not isinstance(current_limit, bool):
        print_note(f"Current daily limit: ${float(current_limit):.2f}")

    daily_default = (
        float(current_limit)
        if isinstance(current_limit, (int, float)) and not isinstance(current_limit, bool)
        else None
    )
    daily_limit = _prompt_float("Daily budget in USD", default=daily_default, minimum=0.01)
    current_exhausted = existing_root.get("on_budget_exhausted")
    exhausted_default = (
        current_exhausted
        if isinstance(current_exhausted, (str, dict)) and current_exhausted
        else "block"
    )
    exhausted_action = prompt_for_choice(
        "At 100% of budget",
        [(value, value) for value in sorted(VALID_ON_BUDGET_EXHAUSTED)]
        + [("switch", "switch to another harness/model")],
    )
    if exhausted_action == "switch":
        harness_options = [(tool, TOOL_SPECS[tool]["display"]) for tool in TOOL_SPECS]
        switch_harness = prompt_for_choice("Switch to harness", harness_options)
        switch_model = _prompt_policy_model(switch_harness, state, default=None)
        exhausted: str | dict = {
            "action": "switch",
            "target": {"harness": switch_harness, "model": switch_model},
        }
    elif exhausted_action in VALID_ON_BUDGET_EXHAUSTED:
        exhausted = exhausted_action
    else:
        exhausted = exhausted_default

    defaults = _existing_policy_tiers(existing_policy) or _default_policy_tiers(state)
    tier_count = _prompt_int("Number of tiers", default=len(defaults), minimum=1)
    tiers: list[dict[str, object]] = []
    harness_options = [(tool, TOOL_SPECS[tool]["display"]) for tool in TOOL_SPECS]
    for index in range(tier_count):
        default_tier = defaults[index] if index < len(defaults) else defaults[-1]
        print_section(f"Tier {index + 1}")
        if index == 0:
            activates_at = 0.0
            print_kv("Activates at", "0%")
        else:
            default_pct = default_tier.get("activates_at_pct")
            if not isinstance(default_pct, (int, float)) or isinstance(default_pct, bool):
                default_pct = 0.0
            activates_at = _prompt_float(
                "Activates at percent",
                default=float(default_pct),
                minimum=0,
                maximum=100,
            )
        default_harness = str(default_tier.get("harness") or "claude")
        print_note(
            f"Default harness: {TOOL_SPECS.get(default_harness, {}).get('display', default_harness)}"
        )
        harness = prompt_for_choice("Harness", harness_options)
        default_model = str(default_tier.get("model") or "")
        model = _prompt_policy_model(harness, state, default=default_model or None)
        tiers.append(
            {
                "name": f"tier {index + 1}",
                "activates_at_pct": activates_at,
                "harness": harness,
                "model": model,
            }
        )

    policy = normalize_policy(
        {
            "policy": {
                "name": DEFAULT_POLICY_NAME,
                "daily_budget_usd": daily_limit,
                "tiers": tiers,
                "on_budget_exhausted": exhausted,
            }
        }
    )
    if policy is None:
        raise RuntimeError("Policy is malformed. Check tier thresholds, harnesses, and models.")
    save_workspace_policy(workspace, policy)

    print_kv("Daily limit", f"${daily_limit:.2f}")
    print_kv("Policy file", str(policy_cache_path(workspace)))
    print_success("Budget policy updated")


def _validate_policy_harnesses_and_models(policy: dict, state: dict) -> list[str]:
    """Check each tier's harness/model against ucode's known agents and the
    workspace's discovered model inventory in ``state.json`` (no network call).

    Structural validity is assumed (caller runs ``parse_and_validate_policy_yaml``
    first). A model is only rejected when state has a non-empty inventory for that
    harness to check against — mirroring the interactive flow, which falls back to
    free-text model entry when discovery returned nothing.
    """
    errors: list[str] = []
    tiers = policy.get("policy", {}).get("tiers", [])
    for index, tier in enumerate(tiers, start=1):
        harness = tier.get("harness")
        if harness not in TOOL_SPECS:
            valid = ", ".join(sorted(TOOL_SPECS))
            errors.append(
                f"tier {index}: harness '{harness}' is not a supported ucode agent "
                f"(valid: {valid})."
            )
            continue
        model = tier.get("model")
        known_models = {value for value, _label in _policy_model_options(harness, state)}
        if known_models and model not in known_models:
            display = TOOL_SPECS[harness].get("display", harness)
            errors.append(
                f"tier {index}: model '{model}' is not available for {display} on this "
                f"workspace (known: {', '.join(sorted(known_models))})."
            )
    return errors


def _apply_budget_file(workspace: str, file: Path, state: dict) -> None:
    """Validate ``file`` and save it as the workspace policy."""
    print_section("Budget")
    print_kv("Workspace", workspace)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read policy file: {file}") from exc

    policy, errors = parse_and_validate_policy_yaml(text)
    if not errors:
        assert policy is not None  # structurally validated above
        errors = _validate_policy_harnesses_and_models(policy, state)
    if errors:
        print_err("Policy file is invalid:")
        for reason in errors:
            print_note(reason)
        raise typer.Exit(1)
    assert policy is not None

    save_workspace_policy(workspace, policy)
    daily_limit = float(policy["policy"]["daily_budget_usd"])
    print_kv("Daily limit", f"${daily_limit:.2f}")
    print_kv("Policy file", str(policy_cache_path(workspace)))
    print_success("Budget policy updated")


@setup_app.command("budget")
def setup_budget_cmd(
    file: Annotated[
        Path | None,
        typer.Option(
            "-f",
            "--file",
            help="Apply a policy YAML file instead of the interactive flow.",
        ),
    ] = None,
) -> None:
    """Set the shared daily spend budget for the current workspace."""
    try:
        state = load_state()
        workspace = state.get("workspace")
        if not isinstance(workspace, str) or not workspace:
            raise RuntimeError("No workspace is configured. Run `ucode configure` first.")
        profile = state.get("profile")

        ensure_databricks_auth(workspace, profile)
        with spinner("Checking workspace admin permissions..."):
            token = get_databricks_token(workspace, profile)
            admin = is_workspace_admin(workspace, token)
        if admin is False:
            raise RuntimeError(
                f"You do not have admin permissions to change the ucode budget "
                f"for {workspace}. `ucode setup budget` is restricted to workspace admins."
            )
        if admin is None:
            print_warning(
                "Could not verify workspace admin permissions (SCIM `Me` call failed). "
                "Proceeding optimistically — the final UC write will fail if you "
                "lack the required role."
            )
        else:
            print_success("Admin permissions verified")

        if file is not None:
            _apply_budget_file(workspace, file, state)
        else:
            _run_budget_setup(workspace)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@setup_app.command("mcp")
def setup_mcp_cmd() -> None:
    """Configure managed MCP servers for the current workspace."""
    try:
        state = load_state()
        workspace = state.get("workspace")
        if not isinstance(workspace, str) or not workspace:
            raise RuntimeError("No workspace is configured. Run `ucode configure` first.")
        profile = state.get("profile")

        # Pushing MCP config to UC is a workspace-wide write, so admin-only —
        # same gate as `ucode setup budget`.
        ensure_databricks_auth(workspace, profile)
        with spinner("Checking workspace admin permissions..."):
            token = get_databricks_token(workspace, profile)
            admin = is_workspace_admin(workspace, token)
        if admin is False:
            raise RuntimeError(
                f"You do not have admin permissions to change MCP settings "
                f"for {workspace}. `ucode setup mcp` is restricted to workspace admins."
            )
        if admin is None:
            print_warning(
                "Could not verify workspace admin permissions (SCIM `Me` call failed). "
                "Proceeding optimistically — the final UC write will fail if you "
                "lack the required role."
            )
        else:
            print_success("Admin permissions verified")

        configure_mcp_command()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@app.command("status")
def status_cmd() -> None:
    """Show current workspace, tool configs, and saved model selections."""
    try:
        status()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None


@app.command("revert")
def revert_cmd() -> None:
    """Clear ucode state and restore backed-up agent config files."""
    try:
        revert()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None


@usage_app.callback(invoke_without_command=True)
def usage_cmd(
    ctx: typer.Context,
    local: Annotated[
        bool,
        typer.Option("--local", help="Read the local SQLite usage ledger instead of AI Gateway."),
    ] = False,
    days: Annotated[
        int,
        typer.Option("--days", min=1, help="Number of days to include for local usage."),
    ] = 7,
) -> None:
    """Show Databricks AI Gateway usage summary."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        if local:
            local_usage_report(days=days)
        else:
            install_databricks_cli()
            usage_report()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None


@setup_app.callback(invoke_without_command=True)
def setup_cmd(
    ctx: typer.Context,
    agents: Annotated[
        bool,
        typer.Option("--agents", help="Run the agent/model setup phase."),
    ] = False,
    tracing: Annotated[
        bool,
        typer.Option("--tracing", help="Run the Tracing setup phase."),
    ] = False,
    mcp: Annotated[
        bool,
        typer.Option("--mcp", help="Run the managed MCP servers setup phase."),
    ] = False,
    budget: Annotated[
        bool,
        typer.Option("--budget", help="Run the budget policy setup phase."),
    ] = False,
) -> None:
    """Interactively build the managed ucode config for a workspace (admin-only).

    With no phase flags, runs all phases (agents, tracing, MCP, budget). Pass any
    of ``--agents``, ``--tracing``, ``--mcp``, ``--budget`` to run only those
    phases; phases you don't select are left untouched in Unity Catalog.
    """
    # Let subcommands (e.g. `ucode setup budget`) handle themselves.
    if ctx.invoked_subcommand is not None:
        return
    # No flags = run every phase (preserves the original all-in-one behavior).
    if not (agents or tracing or mcp or budget):
        agents = tracing = mcp = budget = True
    try:
        install_databricks_cli()
        console.print(
            Panel(
                "Author the managed coding config for your workspace.\n"
                "Developers in this workspace pull this automatically on `ucode configure`.",
                title="ucode setup",
                style="cyan",
                expand=False,
            )
        )

        state = load_state()
        workspace = state.get("workspace")
        profile = state.get("profile")
        if not workspace:
            workspace, profile = _prompt_for_configuration()
        ensure_databricks_auth(workspace, profile)

        with spinner("Checking workspace admin permissions..."):
            token = get_databricks_token(workspace, profile)
            admin = is_workspace_admin(workspace, token)
        if admin is False:
            raise RuntimeError(
                f"You do not have admin permissions to change the ucode config "
                f"for {workspace}. `ucode setup` is restricted to workspace admins."
            )
        if admin is None:
            print_warning(
                "Could not verify workspace admin permissions (SCIM `Me` call failed). "
                "Proceeding optimistically — the final UC write will fail if you "
                "lack the required role."
            )
        else:
            print_success("Admin permissions verified")

        pre_pulled = configure_shared_state(
            workspace, profile=profile, tools=None, force_login=True
        )
        workspace = pre_pulled.get("workspace") or workspace
        profile = pre_pulled.get("profile") or profile
        _render_configuration_panel(pre_pulled, title="Current Configuration")

        uc_available_tools = [
            t
            for t in (pre_pulled.get("available_tools") or [])
            if isinstance(t, str) and t in TOOL_SPECS
        ]
        uc_tracing_enabled = bool(
            (pre_pulled.get("tracing") or {}).get("enabled")
            if isinstance(pre_pulled.get("tracing"), dict)
            else False
        )
        uc_mcps_present = bool(pre_pulled.get("mcp_servers"))
        uc_has_policy = load_workspace_policy(workspace) is not None

        if agents:
            picked_tools = prompt_for_tools(
                [(tool, TOOL_SPECS[tool]["display"]) for tool in TOOL_SPECS],
                preselected=uc_available_tools or None,
            )
            if not picked_tools:
                print_note("No coding agents selected — skipping agent setup.")
            else:
                configure_rc = configure_workspace_command(
                    workspaces=[(workspace, profile)],
                    selected_tools=picked_tools,
                    update_existing=False,
                    validate=False,
                    panel_title=None,
                    pre_pulled_state=pre_pulled,
                )
                if configure_rc != 0:
                    raise typer.Exit(configure_rc)

                post_configure_state = load_state()
                configured_tools = [
                    tool
                    for tool in (post_configure_state.get("available_tools") or [])
                    if isinstance(tool, str) and tool in TOOL_SPECS
                ]
                if configured_tools:
                    # Default agent comes from the policy's tier-1 harness when one
                    # is both published and configured. Otherwise fall back to the
                    # first configured tool (mirrors `available_tools` ordering in UC).
                    default_agent, default_reason = _resolve_default_agent(
                        workspace, configured_tools
                    )
                    post_configure_state["default_agent"] = default_agent
                    save_state(post_configure_state)
                    print_success(
                        f"Default agent set to {TOOL_SPECS[default_agent]['display']} "
                        f"({default_reason})"
                    )

        if tracing:
            if prompt_yes_no("Set up Tracing for this workspace?", default=uc_tracing_enabled):
                configure_tracing_command()
            elif uc_tracing_enabled:
                print_note("Clearing Tracing for this workspace...")
                if _clear_managed_tracing(load_state()):
                    print_success("Tracing cleared (will be unset on next export)")

        if mcp:
            if prompt_yes_no(
                "Set up managed MCP servers for this workspace?", default=uc_mcps_present
            ):
                configure_mcp_command()
            elif uc_mcps_present:
                print_note("Clearing managed MCP servers for this workspace...")
                if _clear_managed_mcps(load_state()):
                    print_success("Managed MCP servers cleared (will be unset on next export)")

        if budget:
            if prompt_yes_no("Set up budget policies for this workspace?", default=uc_has_policy):
                _run_budget_setup(workspace)
            elif uc_has_policy:
                print_note("Clearing budget policy for this workspace...")
                local_removed, uc_removed = _clear_managed_policy(workspace, profile)
                if local_removed:
                    print_success("Local policies.yaml cleared")
                if uc_removed:
                    print_success(f"Removed {MANAGED_POLICIES_VOLUME_PATH}")
            else:
                print_note("Skipping budget setup — run `ucode setup budget` anytime to set one.")

        final_state = load_state()
        summary_default = final_state.get("default_agent")
        default_display = (
            TOOL_SPECS[summary_default]["display"]
            if isinstance(summary_default, str) and summary_default in TOOL_SPECS
            else "not set"
        )
        console.print(
            Panel(
                f"[bold]Workspace:[/bold] [cyan]{workspace}[/cyan]\n"
                f"[bold]Default agent:[/bold] [cyan]{default_display}[/cyan]\n"
                "\n"
                "[bold]Next step[/bold]\n"
                "  • Publish to Unity Catalog:  [bold cyan]ucode export[/bold cyan] [green][Recommended][/green]\n"
                "\n"
                "[dim]Optional[/dim]\n"
                "  [dim]• Update MCP servers:    ucode setup mcp[/dim]\n"
                "  [dim]• Update budget policy:  ucode setup budget[/dim]",
                title="Setup Complete",
                style="green",
                expand=False,
            )
        )
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@app.command("export")
def export_cmd() -> None:
    """Upload the local ucode config to Unity Catalog for workspace-wide distribution."""
    import json
    import tempfile
    from pathlib import Path

    try:
        install_databricks_cli()
        state = load_state()
        workspace = state.get("workspace")
        if not workspace:
            raise RuntimeError(
                "No workspace is configured. Run `ucode configure` before `ucode export`."
            )
        profile = state.get("profile")
        ensure_databricks_auth(workspace, profile)

        with spinner("Checking workspace admin permissions..."):
            token = get_databricks_token(workspace, profile)
            admin = is_workspace_admin(workspace, token)
        if admin is False:
            raise RuntimeError(
                f"You do not have admin permissions to publish a managed config "
                f"for {workspace}. `ucode export` is restricted to workspace admins."
            )
        if admin is None:
            print_warning(
                "Could not verify workspace admin permissions (SCIM `Me` call failed). "
                "Proceeding optimistically — the final UC write will fail if you "
                "lack the required role."
            )
        else:
            print_success("Admin permissions verified")

        # Preview what's about to be published before touching UC.
        print_section("Export")
        print_kv("Workspace", workspace)
        print_kv("Destination", MANAGED_CONFIG_VOLUME_PATH)
        _render_configuration_panel(state)

        if not prompt_yes_no("Proceed with upload to Unity Catalog?"):
            print_note("Export cancelled — no changes written.")
            return

        sliced = slice_state_for_export(load_full_state(), workspace)
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="ucode-export-",
            delete=False,
            encoding="utf-8",
        )
        tmp_path = Path(tmp.name)
        try:
            json.dump(sliced, tmp, indent=2)
            tmp.close()

            with spinner("Uploading state.json to Unity Catalog..."):
                upload_managed_config(workspace, profile, tmp_path)
            print_success(f"state.json uploaded to {MANAGED_CONFIG_VOLUME_PATH}")
            policy_path = policy_cache_path(workspace)
            if policy_path.is_file():
                print_kv("Policy destination", MANAGED_POLICIES_VOLUME_PATH)
                with spinner("Uploading policies.yaml to Unity Catalog..."):
                    upload_managed_policies(workspace, profile, policy_path)
                print_success(f"policies.yaml uploaded to {MANAGED_POLICIES_VOLUME_PATH}")
            else:
                print_note("No local policies.yaml found — run `ucode setup budget` to create one.")
        finally:
            tmp_path.unlink(missing_ok=True)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@app.command("delete-configuration")
def delete_configuration_cmd() -> None:
    """Delete the published managed config (state.json) and budget policy
    (policies.yaml) from Unity Catalog for a selected workspace (admin-only).

    Prompts for which configured workspace to delete — unlike `ucode export`,
    it never assumes the current workspace, since the delete is destructive."""
    try:
        install_databricks_cli()
        full = load_full_state()
        workspaces = full.get("workspaces") or {}
        configured = [ws for ws in workspaces if isinstance(ws, str) and ws]
        if not configured:
            raise RuntimeError(
                "No workspace is configured. Run `ucode configure` before "
                "`ucode delete-configuration`."
            )

        # Always make the admin choose which workspace's published config to
        # delete — this is destructive and org-wide, so we never assume the
        # current workspace the way `ucode export` does.
        current = full.get("current_workspace")
        configured.sort(key=lambda ws: (ws != current, ws))
        profiles = [(ws, (workspaces.get(ws) or {}).get("profile") or "") for ws in configured]
        workspace, profile = prompt_for_workspace(
            "Select the workspace whose published config to delete", profiles
        )
        profile = profile or None
        ensure_databricks_auth(workspace, profile)

        with spinner("Checking workspace admin permissions..."):
            token = get_databricks_token(workspace, profile)
            admin = is_workspace_admin(workspace, token)
        if admin is False:
            raise RuntimeError(
                f"You do not have admin permissions to delete the managed config "
                f"for {workspace}. `ucode delete-configuration` is restricted to "
                f"workspace admins."
            )
        if admin is None:
            print_warning(
                "Could not verify workspace admin permissions (SCIM `Me` call failed). "
                "Proceeding optimistically — the UC delete will fail if you "
                "lack the required role."
            )
        else:
            print_success("Admin permissions verified")

        print_section("Delete configuration")
        print_kv("Workspace", workspace)
        print_kv("Config file", MANAGED_CONFIG_VOLUME_PATH)
        print_kv("Policy file", MANAGED_POLICIES_VOLUME_PATH)
        print_warning(
            "This removes the published config from Unity Catalog. Developers in "
            "this workspace will no longer pull it on `ucode configure`."
        )

        if not prompt_yes_no(
            "Delete the published config and policy from Unity Catalog?", default=False
        ):
            print_note("Delete cancelled — no changes written.")
            return
        if not prompt_yes_no("Are you sure? This cannot be undone.", default=False):
            print_note("Delete cancelled — no changes written.")
            return

        config_removed = False
        policy_removed = False
        with spinner("Deleting published config from Unity Catalog..."):
            config_removed = delete_managed_config(workspace, profile)
            policy_removed = delete_managed_policies(workspace, profile)

        if config_removed:
            print_success(f"Removed {MANAGED_CONFIG_VOLUME_PATH}")
        else:
            print_note(f"No published config found at {MANAGED_CONFIG_VOLUME_PATH}")
        if policy_removed:
            print_success(f"Removed {MANAGED_POLICIES_VOLUME_PATH}")
        else:
            print_note(f"No published policy found at {MANAGED_POLICIES_VOLUME_PATH}")

        if not config_removed and not policy_removed:
            print_note(
                "Nothing to delete — Unity Catalog had no managed config for this workspace."
            )
        else:
            print_success("Managed configuration deleted")
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@usage_app.command("record")
def usage_record_cmd(
    tool: Annotated[str, typer.Option("--tool", help="Agent name, e.g. claude or codex.")],
    model: Annotated[str, typer.Option("--model", help="Model or endpoint name.")],
    session_id: Annotated[
        str,
        typer.Option("--session-id", help="Stable ID for the running agent session."),
    ],
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Use 'delta' for incremental events or 'snapshot' for cumulative totals.",
        ),
    ] = "snapshot",
    input_tokens: Annotated[int, typer.Option("--input-tokens", min=0)] = 0,
    output_tokens: Annotated[int, typer.Option("--output-tokens", min=0)] = 0,
    cache_read_input_tokens: Annotated[
        int,
        typer.Option("--cache-read-input-tokens", min=0),
    ] = 0,
    cache_creation_input_tokens: Annotated[
        int,
        typer.Option("--cache-creation-input-tokens", min=0),
    ] = 0,
    total_tokens: Annotated[int, typer.Option("--total-tokens", min=0)] = 0,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    source: Annotated[str, typer.Option("--source")] = "hook",
) -> None:
    """Record local token usage for budget checks and local aggregation."""
    try:
        if mode == "delta":
            event = record_local_usage_delta(
                session_id=session_id,
                tool=tool,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                total_tokens=total_tokens,
                workspace=workspace,
                source=source,
            )
        elif mode == "snapshot":
            event = record_local_usage_snapshot(
                session_id=session_id,
                tool=tool,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                total_tokens=total_tokens,
                workspace=workspace,
                source=source,
            )
        else:
            raise RuntimeError("Invalid --mode. Use 'delta' or 'snapshot'.")
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None

    if event is None:
        print_note("No new tokens recorded for this snapshot.")
        return
    cost_usd = event.get("cost_usd")
    if not isinstance(cost_usd, int | float | str):
        cost_usd = 0.0
    print_success(
        f"Recorded {event['total_tokens']} tokens for {tool} ({model}, ${float(cost_usd):.4f})."
    )


@usage_app.command("budget-status")
def usage_budget_status_cmd() -> None:
    """Show the current local spend budget state.

    The daily budget is a single pool shared across all coding tools, so this
    renders one combined view."""
    state = load_state()
    workspace = state.get("workspace")
    sync_workspace = workspace if isinstance(workspace, str) else None
    sync_opencode_usage_from_messages(workspace=sync_workspace)
    sync_opencode_usage_from_state(workspace=sync_workspace)
    console.print(
        render_local_budget_panel(local_budget_status(), title="Daily Budget · All Tools")
    )


@usage_app.command("budget-check")
def usage_budget_check_cmd(
    agent: Annotated[str, typer.Option("--agent", help="Agent name, e.g. claude or codex.")],
) -> None:
    """Exit nonzero when the local spend budget is exceeded."""
    status_obj = local_budget_status(agent)
    message = format_local_budget_status(status_obj)
    if status_obj.get("state") == "exceeded":
        print_err_panel(message)
        raise typer.Exit(1)
    if status_obj.get("state") == "warn":
        print_note(message)
    else:
        console.print(message)


@usage_app.command("hook")
def usage_hook_cmd(
    agent: Annotated[str, typer.Argument(help="Hook adapter: claude, codex, or opencode.")],
    event: Annotated[
        str,
        typer.Argument(help="Hook event: prompt-submit, post-tool, or notify."),
    ],
    model: Annotated[str, typer.Option("--model", help="Model or endpoint name.")],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Agent hook adapter. Reads hook JSON from stdin and writes hook JSON to stdout."""
    try:
        if agent == "claude":
            response = claude_usage_hook(model=model, event=event, workspace=workspace)
        elif agent == "codex":
            response = codex_usage_hook(model=model, event=event, workspace=workspace)
        elif agent == "opencode":
            response = opencode_usage_hook(model=model, event=event, workspace=workspace)
        else:
            raise RuntimeError("Unsupported usage hook. Use 'claude', 'codex', or 'opencode'.")
    except RuntimeError as exc:
        typer.echo(json.dumps({"systemMessage": str(exc)}))
        raise typer.Exit(0) from None
    typer.echo(json.dumps(response))


@app.command("upgrade")
def upgrade_cmd() -> None:
    """Upgrade ucode to the latest version from GitHub."""
    import subprocess

    git_url = "git+https://github.com/databricks/ucode"
    print_section("Upgrade")
    print_kv("Source", git_url)
    try:
        subprocess.run(
            ["uv", "tool", "install", "--reinstall", git_url],
            check=True,
        )
    except FileNotFoundError:
        print_err("`uv` was not found on PATH. Install uv to upgrade ucode.")
        raise typer.Exit(1) from None
    except subprocess.CalledProcessError as exc:
        print_err(f"Upgrade failed (exit code {exc.returncode}).")
        raise typer.Exit(1) from None
    print_success("ucode upgraded")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
