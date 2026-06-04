#!/usr/bin/env python3
"""CLI entry point for ucode."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.panel import Panel

from ucode.agents import (
    TOOL_SPECS,
    check_gateway_endpoint,
    configure_selected_tools,
    configure_single_tool,
    configure_tool,
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
    build_shared_base_urls,
    discover_claude_models,
    discover_codex_models,
    discover_gemini_models,
    download_managed_config,
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
)
from ucode.mcp import (
    MCP_CLIENTS,
    configure_mcp_command,
    purge_cross_workspace_mcp_residue,
    revert_mcp_configs,
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
from ucode.tracing import configure_tracing_command, install_tracing_runtime
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
    prompt_for_default_agent,
    prompt_for_tools,
    prompt_for_usd_amount,
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
from ucode.usage_hooks import claude_usage_hook, codex_usage_hook

_DISCOVERY_CONSUMERS: dict[str, tuple[str, ...]] = {
    "claude": ("claude", "opencode", "copilot", "pi"),
    "codex": ("codex", "copilot", "pi"),
    "gemini": ("gemini", "opencode", "pi"),
}

# Agents that record local spend and therefore have a daily budget to report.
BUDGET_TRACKED_AGENTS: tuple[str, ...] = ("claude", "codex")


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
    remote = download_managed_config(workspace, profile)
    if remote is not None:
        state = merge_managed_workspace(state, remote, profile=profile)

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
) -> int:
    if tool is not None and selected_tools is not None:
        raise RuntimeError("Use either --agent or --agents, not both.")

    workspace_entries = workspaces or [_prompt_for_configuration(tool)]

    if tool is not None:
        states = _configure_shared_workspace_states(workspace_entries, [tool], force_login=True)
        state = states[0]
        state = configure_single_tool(tool, state)
        spec = TOOL_SPECS[tool]
        console.print(
            Panel(
                f"[bold]Workspace:[/bold] [cyan]{state['workspace']}[/cyan]\n"
                f"[bold]{spec['display']}:[/bold] [green]configured[/green]",
                title="Configuration Complete",
                style="green",
                expand=False,
            )
        )
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

    states = _configure_shared_workspace_states(workspace_entries, selected_tools, force_login=True)
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
        install_tool_binary(tool_name, strict=False, update_existing=True)

    state = configure_selected_tools(_persistable_state(state), picked)

    summary_lines = [f"[bold]Workspace:[/bold] [cyan]{state['workspace']}[/cyan]"]
    for tool_name in picked:
        spec = TOOL_SPECS[tool_name]
        summary_lines.append(f"[bold]{spec['display']}:[/bold] [green]configured[/green]")
    console.print(
        Panel(
            "\n".join(summary_lines),
            title="Configuration Complete",
            style="green",
            expand=False,
        )
    )

    # Limit validation to just-configured tools so we don't re-validate
    # previously-configured tools the user didn't touch this run.
    validate_state = {**state, "available_tools": picked}
    validate_all_tools(validate_state)
    return 0


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
        print_kv("MLflow tracing", "enabled")
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
        print_kv("MLflow tracing", "disabled")

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
    console.print(
        Panel(
            f"[bold]Workspace:[/bold] [cyan]{state['workspace']}[/cyan]\n"
            f"[bold]{spec['display']}:[/bold] [green]configured[/green]",
            title="Configuration Complete",
            style="green",
            expand=False,
        )
    )

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


def _agent_budget_exceeded(tool: str) -> bool:
    """True when ``tool`` has a local daily budget and it's already exceeded.

    Only claude/codex record local spend, so other agents report ``ok`` and are
    always considered launchable."""
    try:
        return local_budget_status(tool).get("state") == "exceeded"
    except RuntimeError:
        return False


def _resolve_default_launch_tool(state: dict) -> str:
    """Pick which agent bare ``ucode`` should launch.

    Starts with ``default_agent`` (falling back to the first configured agent
    when it's unset or no longer available), then skips any agent whose daily
    budget is exhausted. If every agent is over budget we return the default so
    ``_launch_tool`` surfaces its budget panel and exits."""
    available = state.get("available_tools") or []
    if not available:
        raise RuntimeError("No coding agents are configured. Run `ucode configure` to set one up.")
    default_agent = state.get("default_agent")
    if default_agent not in available:
        default_agent = available[0]
    ordered = [default_agent] + [tool for tool in available if tool != default_agent]
    for tool in ordered:
        if not _agent_budget_exceeded(tool):
            if tool != default_agent:
                print_note(
                    f"{TOOL_SPECS[default_agent]['display']} has reached its daily budget; "
                    f"launching {TOOL_SPECS[tool]['display']} instead."
                )
            return tool
    return default_agent


def _launch_tool(tool_name: str, ctx: typer.Context) -> None:
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
        state, resolved_model = resolve_launch_model(tool, state, None)
        state = configure_tool(tool, state, resolved_model)
        display = TOOL_SPECS[tool]["display"]
        if tool in BUDGET_TRACKED_AGENTS:
            # The daily budget is a single global pool shared across all tools,
            # so render it tool-agnostically (no per-tool label in the callout).
            # The panel title still names the tool being launched.
            budget_status = local_budget_status()
            console.print(
                render_local_budget_panel(
                    budget_status,
                    title=f"ucode with {display}",
                )
            )
            # The panel above already shows the exceeded/warn state with the
            # full spend breakdown, so avoid re-printing the same numbers.
            if budget_status.get("state") == "exceeded":
                raise typer.Exit(1)
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
            help="Also enable MLflow tracing for the configured workspace(s).",
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
        bool, typer.Option("--disable", help="Turn off MLflow tracing for configured agents.")
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


def _run_budget_setup(workspace: str) -> None:
    """Interactive "set the workspace's daily spend budget" flow.
    used by both ucode setup and ucode setup budget subcommands."""
    state = load_state()
    policies = dict(state.get("policies") or {})
    current_limit = policies.get("daily_limit_usd")

    print_section("Budget")
    print_kv("Workspace", workspace)
    if isinstance(current_limit, (int, float)) and not isinstance(current_limit, bool):
        print_note(f"Current daily limit: ${float(current_limit):.2f}")

    daily_limit = prompt_for_usd_amount("Daily budget in USD")
    policies["daily_limit_usd"] = daily_limit
    state["policies"] = policies
    save_state(state)

    print_kv("Daily limit", f"${daily_limit:.2f}")
    print_success("Daily budget updated")


@setup_app.command("budget")
def setup_budget_cmd() -> None:
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
def setup_cmd(ctx: typer.Context) -> None:
    """Interactively build the managed ucode config for a workspace (admin-only)."""
    # Let subcommands (e.g. `ucode setup budget`) handle themselves.
    if ctx.invoked_subcommand is not None:
        return
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

        picked_tools = prompt_for_tools(
            [(tool, TOOL_SPECS[tool]["display"]) for tool in TOOL_SPECS]
        )
        if not picked_tools:
            print_note("No coding agents selected — nothing to configure.")
            return

        configure_rc = configure_workspace_command(
            workspaces=[(workspace, profile)],
            selected_tools=picked_tools,
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
            default_agent = prompt_for_default_agent(
                [(tool, TOOL_SPECS[tool]["display"]) for tool in configured_tools]
            )
            post_configure_state["default_agent"] = default_agent
            save_state(post_configure_state)
            print_success(
                f"Default agent set to {TOOL_SPECS[default_agent]['display']}"
            )

        if prompt_yes_no("Set up MLflow tracing for this workspace?"):
            configure_tracing_command()

        if prompt_yes_no("Set up managed MCP servers for this workspace?"):
            configure_mcp_command()

        if prompt_yes_no("Set up budget policies for this workspace?"):
            _run_budget_setup(workspace)
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
                "Run [bold]ucode export[/bold] to publish this config to Unity Catalog.",
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

            print_section("Export")
            print_kv("Workspace", workspace)
            print_kv("Destination", MANAGED_CONFIG_VOLUME_PATH)
            with spinner("Uploading state.json to Unity Catalog..."):
                upload_managed_config(workspace, profile, tmp_path)
            print_success(f"Config uploaded to {MANAGED_CONFIG_VOLUME_PATH}")
        finally:
            tmp_path.unlink(missing_ok=True)
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
    agent: Annotated[str, typer.Argument(help="Hook adapter: claude or codex.")],
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
        else:
            raise RuntimeError("Unsupported usage hook. Use 'claude' or 'codex'.")
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
