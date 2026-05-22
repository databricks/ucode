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
from ucode.config_io import restore_file, set_dry_run
from ucode.databricks import (
    build_shared_base_urls,
    discover_claude_models,
    discover_codex_models,
    discover_gemini_models,
    ensure_ai_gateway_v2,
    ensure_databricks_auth,
    get_databricks_profiles,
    get_databricks_token,
    install_databricks_cli,
    normalize_workspace_url,
    run_databricks_login,
)
from ucode.mcp import MCP_CLIENTS, configure_mcp_command, revert_mcp_configs
from ucode.state import STATE_PATH, clear_state, load_state, save_state
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


def _prompt_for_configuration(tool: str | None = None) -> str:
    if tool is None:
        desc = "Configure your Databricks workspace"
    else:
        desc = f"Configure {TOOL_SPECS[tool]['display']} to use your Databricks endpoint."
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


def _parse_workspaces_option(workspaces: str) -> list[str]:
    workspace_urls: list[str] = []
    for raw_workspace in workspaces.split(","):
        raw_workspace = raw_workspace.strip()
        if not raw_workspace:
            continue
        try:
            workspace = normalize_workspace_url(raw_workspace)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if workspace not in workspace_urls:
            workspace_urls.append(workspace)
    if not workspace_urls:
        raise RuntimeError(
            "No workspaces provided for --workspaces. Use a comma-separated list like "
            "`--workspaces https://workspace.databricks.com`."
        )
    return workspace_urls


def configure_shared_state(
    workspace: str, tools: list[str] | None = None, force_login: bool = False
) -> dict:
    """Log into Databricks, enforce AI Gateway v2, fetch model lists, persist state.

    If tools is provided, only fetch models for those tools. Otherwise fetch all.
    If force_login is True, always run databricks auth login (used by explicit configure).
    """
    workspace = normalize_workspace_url(workspace)
    fetch_all = tools is None
    if force_login:
        run_databricks_login(workspace)
    else:
        ensure_databricks_auth(workspace)
    with spinner("Verifying Unity AI Gateway..."):
        token = get_databricks_token(workspace)
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
    # Diagnostic reasons are transient — attach after save_state so they don't
    # land on disk but are available to the caller for this run.
    state["_discovery_reasons"] = {
        "claude": claude_reason,
        "gemini": gemini_reason,
        "codex": codex_reason,
    }
    return state


def _configure_shared_workspace_states(
    workspaces: list[str], tools: list[str] | None, *, force_login: bool
) -> list[dict]:
    if not workspaces:
        raise RuntimeError("At least one workspace must be provided.")
    states: list[dict] = []
    for workspace in workspaces:
        states.append(configure_shared_state(workspace, tools=tools, force_login=force_login))
    return states


def configure_workspace_command(
    tool: str | None = None,
    selected_tools: list[str] | None = None,
    workspaces: list[str] | None = None,
) -> int:
    if tool is not None and selected_tools is not None:
        raise RuntimeError("Use either --agent or --agents, not both.")

    workspace_urls = workspaces or [_prompt_for_configuration(tool)]

    if tool is not None:
        states = _configure_shared_workspace_states(workspace_urls, [tool], force_login=True)
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

    states = _configure_shared_workspace_states(workspace_urls, selected_tools, force_login=True)
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
        install_tool_binary(tool_name, strict=False, update_existing=True)

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

    print_heading("State")
    print_kv("State file", str(STATE_PATH) if STATE_PATH.exists() else "missing")
    print_note("Use `ucode configure` to update workspace settings or configure new tools.")
    print_note(
        "Use `ucode configure mcp` to add Databricks MCP servers to configured coding tools."
    )
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
    clear_state()

    print_heading("Revert")
    print_kv("Workspace", state.get("workspace") or "none")
    for tool, spec in TOOL_SPECS.items():
        print_kv(f"{spec['display']} config", "restored" if results[tool] else "unchanged")
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
    if not workspace:
        workspace = _prompt_for_configuration(tool)
    state = configure_shared_state(workspace, tools=[tool])

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
        state = configure_shared_state(state["workspace"], tools=[tool])
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
) -> None:
    """Configure workspace URL and AI Gateway."""
    if ctx.invoked_subcommand is not None:
        return
    set_dry_run(dry_run)
    try:
        install_databricks_cli()
        if agent is not None and agents is not None:
            raise RuntimeError("Use either --agent or --agents, not both.")
        workspace_urls = _parse_workspaces_option(workspaces) if workspaces is not None else None
        if agent is not None:
            tool = normalize_tool(agent)
            install_tool_binary(tool, strict=True, update_existing=True)
            if workspace_urls is None:
                configure_workspace_command(tool)
            else:
                configure_workspace_command(tool, workspaces=workspace_urls)
        elif agents is not None:
            selected_tools = _parse_agents_option(agents)
            if workspace_urls is None:
                configure_workspace_command(selected_tools=selected_tools)
            else:
                configure_workspace_command(
                    selected_tools=selected_tools, workspaces=workspace_urls
                )
        else:
            # Tool binaries are installed after the user picks which agents
            # they want, in configure_workspace_command.
            if workspace_urls is None:
                configure_workspace_command()
            else:
                configure_workspace_command(workspaces=workspace_urls)
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
