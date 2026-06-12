#!/usr/bin/env python3
"""CLI entry point for ucode."""

from __future__ import annotations

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
from ucode.agents.codex import revert_legacy_shared_config
from ucode.agents.pi import PI_SETTINGS_BACKUP_PATH, PI_SETTINGS_PATH
from ucode.config_io import restore_file, set_dry_run
from ucode.databricks import (
    build_shared_base_urls,
    discover_claude_models,
    discover_codex_models,
    discover_gemini_models,
    discover_model_services,
    ensure_ai_gateway_v2,
    ensure_databricks_auth,
    find_profile_name_for_host,
    get_databricks_profiles,
    get_databricks_token,
    install_databricks_cli,
    normalize_workspace_url,
    run_databricks_login,
    uc_enabled,
)
from ucode.mcp import (
    MCP_CLIENTS,
    configure_mcp_command,
    purge_cross_workspace_mcp_residue,
    revert_mcp_configs,
)
from ucode.state import STATE_PATH, clear_state, load_full_state, load_state, save_state
from ucode.tracing import configure_tracing_command
from ucode.ui import (
    console,
    heading,
    print_err,
    print_heading,
    print_kv,
    print_note,
    print_section,
    print_success,
    prompt_for_tools,
    prompt_for_workspace,
    set_verbosity,
    spinner,
    status_badge,
)
from ucode.usage import usage as usage_report

_DISCOVERY_CONSUMERS: dict[str, tuple[str, ...]] = {
    "claude": ("claude", "opencode", "copilot", "pi"),
    "codex": ("codex", "copilot", "pi"),
    "gemini": ("gemini", "opencode", "pi"),
}


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
    enable_uc: bool | None = None,
    reset_uc: bool = False,
) -> dict:
    """Log into Databricks, enforce AI Gateway v2, fetch model lists, persist state.

    If tools is provided, only fetch models for those tools. Otherwise fetch all.
    If force_login is True, always run databricks auth login (used by explicit configure).
    ``profile`` is the Databricks CLI profile name to address — passed via
    ``--profile`` to every CLI invocation so ambiguous `~/.databrickscfg`
    entries (e.g. DEFAULT and a named profile both pointing at the same host)
    don't error out. If ``None``, we resolve it from the host after login.
    ``enable_uc`` is the resolved CLI flag (`--enable-uc`): when not None
    it overrides both the env var and the persisted state.
    ``reset_uc`` is True only on the explicit ``ucode configure`` flow.
    """
    workspace = normalize_workspace_url(workspace)
    prior_state = load_state()
    previous_workspace = prior_state.get("workspace")
    # Precedence: explicit CLI flag > env var > (configure: reset to False;
    # launch: target workspace's persisted state). Use *target* state on the
    # launch path so the flag is sticky per-workspace and doesn't leak
    # across workspace switches.
    # TODO: when this flips uc_enabled True->False, prune any
    # `system.ai.*` MCP services from state["mcp_servers"] (and their
    # cross-tool registrations). Today they linger as orphans pointing at
    # /ai-gateway/mcp-services/* until the user re-runs `configure mcp`
    # or switches workspaces.
    if enable_uc is None:
        if reset_uc:
            enable_uc = uc_enabled(default=False)
        else:
            target_ws_state = load_full_state().get("workspaces", {}).get(workspace) or {}
            enable_uc = uc_enabled(default=bool(target_ws_state.get("uc_enabled")))
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
    claude_models = {}
    gemini_models = []
    codex_models = []
    if enable_uc:
        # Opt-in: one UC model-services call yields all families as
        # `system.ai.<model-name>` ids, bucketed by name. The single reason is
        # shared across the families that were requested.
        with spinner("Fetching available models (model services)..."):
            ms_claude, ms_codex, ms_gemini, ms_reason = discover_model_services(workspace, token)
        if want_claude:
            claude_models, claude_reason = ms_claude, ms_reason
        if want_gemini:
            gemini_models, gemini_reason = ms_gemini, ms_reason
        if want_codex:
            codex_models, codex_reason = ms_codex, ms_reason
    else:
        with spinner("Fetching available models..."):
            if want_claude:
                claude_models, claude_reason = discover_claude_models(workspace, token)
            if want_gemini:
                gemini_models, gemini_reason = discover_gemini_models(workspace, token)
            if want_codex:
                codex_models, codex_reason = discover_codex_models(workspace, token)
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
    # Persist the resolved flag so subsequent launches stay on the same
    # discovery path without the env var or CLI flag being re-passed.
    state["uc_enabled"] = enable_uc
    state["base_urls"] = build_shared_base_urls(workspace)
    if want_claude:
        state["claude_models"] = claude_models
    if want_gemini:
        state["gemini_models"] = gemini_models
    if want_codex:
        state["codex_models"] = codex_models
    if fetch_all or "opencode" in tools:
        state["opencode_models"] = opencode_models
    save_state(state)
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
    enable_uc: bool | None = None,
    reset_uc: bool = False,
) -> list[dict]:
    if not workspaces:
        raise RuntimeError("At least one workspace must be provided.")
    states: list[dict] = []
    for workspace, profile in workspaces:
        states.append(
            configure_shared_state(
                workspace,
                profile=profile,
                tools=tools,
                force_login=force_login,
                enable_uc=enable_uc,
                reset_uc=reset_uc,
            )
        )
    return states


def configure_workspace_command(
    tool: str | None = None,
    selected_tools: list[str] | None = None,
    workspaces: list[tuple[str, str | None]] | None = None,
    *,
    prompt_optional_updates: bool = True,
    enable_uc: bool | None = None,
    reset_uc: bool = False,
) -> int:
    if tool is not None and selected_tools is not None:
        raise RuntimeError("Use either --agent or --agents, not both.")

    workspace_entries = workspaces or [_prompt_for_configuration(tool)]

    if tool is not None:
        states = _configure_shared_workspace_states(
            workspace_entries, [tool], force_login=True, enable_uc=enable_uc, reset_uc=reset_uc
        )
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

    states = _configure_shared_workspace_states(
        workspace_entries,
        selected_tools,
        force_login=True,
        enable_uc=enable_uc,
        reset_uc=reset_uc,
    )
    state = states[0]
    save_state(state)

    available_on_workspace: list[str] = []
    tools_to_check = selected_tools or list(TOOL_SPECS)
    for tool_name in tools_to_check:
        with spinner(f"Checking {TOOL_SPECS[tool_name]['display']} availability..."):
            if check_gateway_endpoint(state, tool_name):
                available_on_workspace.append(tool_name)

    if not available_on_workspace:
        print_err("No coding agents are available on this workspace.")
        _print_discovery_diagnostics(state)
        return 1

    if selected_tools is None:
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
        install_tool_binary(
            tool_name,
            strict=False,
            update_existing=True,
            prompt_optional_updates=prompt_optional_updates,
        )

    state = configure_selected_tools(state, picked)

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
    # Older Codex (< 0.134.0) had ucode edit the shared ~/.codex/config.toml in
    # place; restoring the per-profile file above does not undo that.
    legacy_codex_stripped = revert_legacy_shared_config()
    clear_state()

    print_heading("Revert")
    print_kv("Workspace", state.get("workspace") or "none")
    for tool, spec in TOOL_SPECS.items():
        print_kv(f"{spec['display']} config", "restored" if results[tool] else "unchanged")
    if legacy_codex_stripped:
        print_kv("Codex shared config", "ucode entries removed")
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
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
configure_app = typer.Typer(add_completion=False, no_args_is_help=False)
app.add_typer(configure_app, name="configure", help="Configure workspace and tool settings.")
mcp_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(mcp_app, name="mcp", help="MCP servers exposed by ucode.")


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
        print_section(f"ucode with {TOOL_SPECS[tool]['display']}")
        if resolved_model:
            print_kv("Model", resolved_model)
        if tool in ("gemini", "opencode", "copilot", "pi"):
            print_note(
                f"{TOOL_SPECS[tool]['display']} token refresh is managed automatically "
                f"every 30 minutes while the session is running."
            )
        print_success(f"Starting {TOOL_SPECS[tool]['display']}")
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
    skip_upgrade: Annotated[
        bool,
        typer.Option(
            "--skip-upgrade",
            help="Don't prompt to upgrade already-installed agent CLIs to a newer version. "
            "Required updates (when an agent is below its minimum supported version) are "
            "still applied.",
        ),
    ] = False,
    verbose: Annotated[
        str,
        typer.Option(
            "--verbose",
            help="Output verbosity: 'normal' (default) renders decorative panels; "
            "'low' prints terse single-line status instead.",
        ),
    ] = "normal",
    enable_uc: Annotated[
        bool,
        typer.Option(
            "--enable-uc",
            help="Discover models via UC `model-services` (`system.ai.<model>`) and "
            "surface curated `system.ai.*` MCP services. Equivalent to setting "
            "UCODE_ENABLE_UC=1 for this configure run. The value is persisted so "
            "subsequent `ucode <agent>` launches stay on the same discovery path; "
            "re-run `ucode configure` without the flag (and without "
            "UCODE_ENABLE_UC=1 in the env) to turn UC discovery back off.",
        ),
    ] = False,
) -> None:
    """Configure workspace URL and AI Gateway."""
    if ctx.invoked_subcommand is not None:
        return
    if verbose not in ("normal", "low"):
        print_err("--verbose must be one of: normal, low.")
        raise typer.Exit(2)
    set_dry_run(dry_run)
    set_verbosity(verbose)
    prompt_optional_updates = not skip_upgrade
    flag_enable_uc: bool | None = True if enable_uc else None
    # Explicit `ucode configure` is a clean slate: when the user omits both
    # `--enable-uc` and `UCODE_ENABLE_UC`, persisted `uc_enabled=true` from
    # a prior run is reset to false.
    try:
        install_databricks_cli()
        if agent is not None and agents is not None:
            raise RuntimeError("Use either --agent or --agents, not both.")
        workspace_entries = _parse_workspaces_option(workspaces) if workspaces is not None else None
        if agent is not None:
            tool = normalize_tool(agent)
            install_tool_binary(
                tool,
                strict=True,
                update_existing=True,
                prompt_optional_updates=prompt_optional_updates,
            )
            if workspace_entries is None:
                configure_workspace_command(tool, enable_uc=flag_enable_uc, reset_uc=True)
            else:
                configure_workspace_command(
                    tool,
                    workspaces=workspace_entries,
                    enable_uc=flag_enable_uc,
                    reset_uc=True,
                )
        elif agents is not None:
            selected_tools = _parse_agents_option(agents)
            if workspace_entries is None:
                configure_workspace_command(
                    selected_tools=selected_tools,
                    prompt_optional_updates=prompt_optional_updates,
                    enable_uc=flag_enable_uc,
                    reset_uc=True,
                )
            else:
                configure_workspace_command(
                    selected_tools=selected_tools,
                    workspaces=workspace_entries,
                    prompt_optional_updates=prompt_optional_updates,
                    enable_uc=flag_enable_uc,
                    reset_uc=True,
                )
        else:
            # Tool binaries are installed after the user picks which agents
            # they want, in configure_workspace_command.
            if workspace_entries is None:
                configure_workspace_command(
                    prompt_optional_updates=prompt_optional_updates,
                    enable_uc=flag_enable_uc,
                    reset_uc=True,
                )
            else:
                configure_workspace_command(
                    workspaces=workspace_entries,
                    prompt_optional_updates=prompt_optional_updates,
                    enable_uc=flag_enable_uc,
                    reset_uc=True,
                )
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


@app.command("usage")
def usage_cmd() -> None:
    """Show Databricks AI Gateway usage summary (last 7 days)."""
    try:
        install_databricks_cli()
        usage_report()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None


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
