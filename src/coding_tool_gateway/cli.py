#!/usr/bin/env python3
"""CLI entry point for coding-gateway."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.panel import Panel

from coding_tool_gateway.agents import (
    DEFAULT_TOOL,
    TOOL_SPECS,
    configure_all_tools,
    configure_tool,
    default_model_for_tool,
    ensure_bootstrap_dependencies,
    ensure_provider_state,
    install_tool_binary,
    normalize_tool,
    resolve_launch_model,
    validate_all_tools,
)
from coding_tool_gateway.agents import (
    launch as launch_agent,
)
from coding_tool_gateway.config_io import restore_file, set_dry_run
from coding_tool_gateway.databricks import (
    build_shared_base_urls,
    ensure_ai_gateway_v2,
    fetch_ai_gateway_claude_models,
    fetch_codex_models,
    fetch_gemini_models,
    get_databricks_profiles,
    get_databricks_token,
    install_databricks_cli,
    normalize_workspace_url,
    run_databricks_login,
)
from coding_tool_gateway.mcp import configure_mcp_command
from coding_tool_gateway.state import STATE_PATH, clear_state, load_state, save_state
from coding_tool_gateway.ui import (
    console,
    heading,
    print_err,
    print_heading,
    print_kv,
    print_note,
    print_section,
    print_success,
    prompt_for_workspace,
    spinner,
    status_badge,
)
from coding_tool_gateway.usage import usage as usage_report


def _prompt_for_configuration(tool: str | None = None) -> str:
    if tool is None:
        desc = "Configure your Databricks workspace"
    else:
        desc = f"Configure {TOOL_SPECS[tool]['display']} to use your Databricks endpoint."
    profiles = get_databricks_profiles()
    return prompt_for_workspace(desc, profiles)


def configure_shared_state(workspace: str) -> dict:
    """Log into Databricks, enforce AI Gateway v2, fetch model lists, persist state."""
    workspace = normalize_workspace_url(workspace)
    run_databricks_login(workspace)
    with spinner("Verifying AI Gateway V2..."):
        token = get_databricks_token(workspace)
        ensure_ai_gateway_v2(workspace, token)
    print_success("AI Gateway V2 detected")

    with spinner("Fetching available models..."):
        claude_models = fetch_ai_gateway_claude_models(workspace, token)
        gemini_models = fetch_gemini_models(workspace, token)
        codex_models = fetch_codex_models(workspace, token)

    opencode_models: dict[str, list[str]] = {}
    if claude_models:
        opencode_models["anthropic"] = list(claude_models.values())
    if gemini_models:
        opencode_models["gemini"] = gemini_models

    state = {
        "workspace": workspace,
        "claude_models": claude_models,
        "gemini_models": gemini_models,
        "codex_models": codex_models,
        "opencode_models": opencode_models,
        "base_urls": build_shared_base_urls(workspace),
    }
    save_state(state)
    return state


def configure_workspace_command() -> int:
    workspace = _prompt_for_configuration()
    state = configure_shared_state(workspace)
    state = configure_all_tools(state)

    available_tools = state.get("available_tools") or []
    summary_lines = [
        f"[bold]Workspace:[/bold] [cyan]{state['workspace']}[/cyan]",
    ]
    for tool, spec in TOOL_SPECS.items():
        if tool in available_tools:
            summary_lines.append(f"[bold]{spec['display']}:[/bold] [green]configured[/green]")
        else:
            summary_lines.append(f"[bold]{spec['display']}:[/bold] [dim]not available[/dim]")
    console.print(
        Panel(
            "\n".join(summary_lines),
            title="Configuration Complete",
            style="green",
            expand=False,
        )
    )

    if available_tools:
        validate_all_tools(state)
    return 0


def status() -> int:
    state = load_state()
    workspace = state.get("workspace")
    managed_configs = state.get("managed_configs") or {}

    console.print(heading("coding-gateway status"))
    console.print(
        f"  {status_badge('Configured', 'ok') if workspace else status_badge('Not Configured', 'warn')}"
    )

    print_heading("Provider")
    print_kv("Workspace URL", workspace or "not configured")

    print_heading("Tools")
    for tool, spec in TOOL_SPECS.items():
        base_url = state.get("base_urls", {}).get(tool, "not configured")
        managed = bool(managed_configs.get(tool))
        config_path = spec["config_path"]
        print_kv("Tool", spec["display"])
        if tool != "codex":
            print_kv("Model", default_model_for_tool(tool, state) or "not available")
        print_kv("Base URL", base_url)
        print_kv("Managed by Databricks", "yes" if managed else "no")
        print_kv("Config file", str(config_path) if config_path.exists() else "missing")
        console.print()

    print_heading("MCP Servers (Claude Code)")
    print_note("Run `claude mcp list` to see configured MCP servers.")
    print_note("Run `coding-gateway configure mcp` to add Databricks MCP servers.")

    print_heading("State")
    print_kv("State file", str(STATE_PATH) if STATE_PATH.exists() else "missing")
    print_note("Use `coding-gateway configure` to update workspace settings or tool models.")
    print_note("Use `coding-gateway configure mcp` to add Databricks MCP servers to Claude Code.")
    print_note("Use `coding-gateway revert` to clear managed configs and restore prior files.")
    return 0


def revert() -> int:
    state = load_state()
    managed_configs = state.get("managed_configs") or {}

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
    print_success("coding-gateway state cleared")
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


@app.callback(invoke_without_command=True)
def launch(
    ctx: typer.Context,
    agent: Annotated[
        str,
        typer.Option("--agent", help="Agent to launch: codex, claude, gemini, or opencode."),
    ] = DEFAULT_TOOL,
) -> None:
    """Launch Codex, Claude Code, Gemini CLI, or OpenCode via Databricks."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        tool = normalize_tool(agent)
        ensure_bootstrap_dependencies(tool)
        state = ensure_provider_state(tool)
        state, resolved_model = resolve_launch_model(tool, state, None)
        state = configure_tool(tool, state, resolved_model)
        print_section("Launching")
        print_kv("Tool", TOOL_SPECS[tool]["display"])
        if resolved_model:
            print_kv("Model", resolved_model)
        print_kv("Base URL", str(state["base_urls"][tool]))
        if tool in ("gemini", "opencode"):
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


@configure_app.callback(invoke_without_command=True)
def configure(
    ctx: typer.Context,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print config files without writing them.")
    ] = False,
) -> None:
    """Configure workspace URL and AI Gateway."""
    if ctx.invoked_subcommand is not None:
        return
    set_dry_run(dry_run)
    try:
        install_databricks_cli()
        for t in TOOL_SPECS:
            install_tool_binary(t, strict=False)
        configure_workspace_command()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@configure_app.command("mcp")
def configure_mcp() -> None:
    """Add Databricks MCP servers to Claude Code."""
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
    """Clear coding-gateway state and restore backed-up agent config files."""
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
