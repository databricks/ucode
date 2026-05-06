"""MCP (Model Context Protocol) server registration for coding tools."""

from __future__ import annotations

import json
import shutil
import subprocess

from coding_tool_gateway.agents import copilot, opencode
from coding_tool_gateway.databricks import ensure_databricks_auth, list_databricks_connections
from coding_tool_gateway.state import load_state, save_state
from coding_tool_gateway.ui import (
    console,
    label,
    muted,
    print_err,
    print_heading,
    print_note,
    print_section,
    print_success,
    print_warning,
)

MCP_AUTH_TOKEN_ENV_VAR = "OAUTH_TOKEN"
MCP_USER_SCOPE = "user"
MCP_CLEANUP_SCOPES = ("local", "project", MCP_USER_SCOPE)
MCP_CLIENTS = {
    "claude": {"binary": "claude", "display": "Claude Code"},
    "codex": {"binary": "codex", "display": "Codex"},
    "gemini": {"binary": "gemini", "display": "Gemini CLI"},
    "opencode": {"binary": "opencode", "display": "OpenCode"},
    "copilot": {"binary": "copilot", "display": "GitHub Copilot CLI"},
}
MANUAL_EXTERNAL_MCP_VALUE = "external:manual"
EXTERNAL_MCP_SELECTION_PREFIX = "external:"
SQL_MCP_VALUE = "managed:sql"
VECTOR_SEARCH_VALUE = "vector-search"
GENIE_VALUE = "genie"
UC_FUNCTIONS_VALUE = "uc-functions"
CUSTOM_MCP_VALUE = "custom"
MCP_SETUP_BACK_VALUE = "back"
MCP_CONNECTION_MARKERS = (
    "is_mcp",
    "is_mcp_connection",
    "mcp",
    "mcp_enabled",
    "enable_mcp",
)


def build_mcp_http_entry(url: str) -> dict:
    return {
        "type": "http",
        "url": url,
        "headers": {
            "Authorization": f"Bearer ${{{MCP_AUTH_TOKEN_ENV_VAR}}}",
        },
    }


def add_claude_mcp_server(name: str, entry: dict, scope: str = MCP_USER_SCOPE) -> None:
    try:
        subprocess.run(
            ["claude", "mcp", "add-json", name, json.dumps(entry), "-s", scope],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to add MCP server '{name}' via claude CLI.") from exc


def _is_missing_mcp_server_output(output: str) -> bool:
    normalized = output.lower()
    return (
        "not found" in normalized
        or "no mcp server" in normalized
        or "no server named" in normalized
        or ("mcp server found with name" in normalized and "no " in normalized)
    )


def remove_claude_mcp_server(name: str, scope: str) -> bool:
    try:
        subprocess.run(
            ["claude", "mcp", "remove", name, "-s", scope],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return True
    except subprocess.CalledProcessError as exc:
        output = f"{exc.stderr or ''}\n{exc.stdout or ''}"
        if _is_missing_mcp_server_output(output):
            return False
        raise RuntimeError(f"Failed to remove MCP server '{name}' via claude CLI.") from exc


def add_codex_mcp_server(name: str, url: str) -> None:
    try:
        subprocess.run(
            [
                "codex",
                "mcp",
                "add",
                name,
                "--url",
                url,
                "--bearer-token-env-var",
                MCP_AUTH_TOKEN_ENV_VAR,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to add MCP server '{name}' via codex CLI.") from exc


def remove_codex_mcp_server(name: str) -> bool:
    try:
        result = subprocess.run(
            ["codex", "mcp", "remove", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out removing MCP server '{name}' via codex CLI.") from exc

    output = f"{result.stderr or ''}\n{result.stdout or ''}"
    if _is_missing_mcp_server_output(output):
        return False
    if result.returncode != 0:
        raise RuntimeError(f"Failed to remove MCP server '{name}' via codex CLI.")
    return True


def add_gemini_mcp_server(name: str, url: str) -> None:
    try:
        subprocess.run(
            [
                "gemini",
                "mcp",
                "add",
                name,
                url,
                "--type",
                "http",
                "--scope",
                MCP_USER_SCOPE,
                "--header",
                f"Authorization: Bearer ${{{MCP_AUTH_TOKEN_ENV_VAR}}}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to add MCP server '{name}' via gemini CLI.") from exc


def remove_gemini_mcp_server(name: str) -> bool:
    try:
        result = subprocess.run(
            ["gemini", "mcp", "remove", name, "--scope", MCP_USER_SCOPE],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out removing MCP server '{name}' via gemini CLI.") from exc

    output = f"{result.stderr or ''}\n{result.stdout or ''}"
    if _is_missing_mcp_server_output(output):
        return False
    if result.returncode != 0:
        raise RuntimeError(f"Failed to remove MCP server '{name}' via gemini CLI.")
    return True


def available_mcp_clients() -> list[str]:
    return [client for client, spec in MCP_CLIENTS.items() if shutil.which(str(spec["binary"]))]


def configure_client_mcp_server(client: str, name: str, url: str, entry: dict) -> list[str]:
    if client == "claude":
        removed_scopes = [
            scope for scope in MCP_CLEANUP_SCOPES if remove_claude_mcp_server(name, scope)
        ]
        add_claude_mcp_server(name, entry, MCP_USER_SCOPE)
        return removed_scopes
    if client == "codex":
        removed = remove_codex_mcp_server(name)
        add_codex_mcp_server(name, url)
        return [MCP_USER_SCOPE] if removed else []
    if client == "gemini":
        removed = remove_gemini_mcp_server(name)
        add_gemini_mcp_server(name, url)
        return [MCP_USER_SCOPE] if removed else []
    if client == "opencode":
        removed = opencode.write_mcp_server_config(name, url)
        return [MCP_USER_SCOPE] if removed else []
    if client == "copilot":
        removed = copilot.write_mcp_server_config(name, url)
        return [MCP_USER_SCOPE] if removed else []
    raise RuntimeError(f"Unsupported MCP client '{client}'.")


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return None


def _mcp_marker_value(connection: dict) -> bool | None:
    containers = [connection]
    options = connection.get("options")
    if isinstance(options, dict):
        containers.append(options)

    for container in containers:
        for marker in MCP_CONNECTION_MARKERS:
            if marker in container:
                value = _coerce_bool(container.get(marker))
                if value is not None:
                    return value
    return None


def is_external_mcp_connection(connection: dict) -> bool:
    connection_type = connection.get("connection_type")
    if not isinstance(connection_type, str) or connection_type.upper() != "HTTP":
        return False

    marker_value = _mcp_marker_value(connection)
    if marker_value is False:
        return False
    return True


def external_mcp_connection_names(connections: list[dict]) -> list[str]:
    names: set[str] = set()
    for connection in connections:
        if not is_external_mcp_connection(connection):
            continue
        name = connection.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return sorted(names)


def discover_external_mcp_connection_names(workspace: str) -> list[str]:
    return external_mcp_connection_names(list_databricks_connections(workspace))


def build_mcp_server_options(available_external_names: list[str]) -> list[tuple[str, str]]:
    options = [
        (f"{EXTERNAL_MCP_SELECTION_PREFIX}{name}", name) for name in available_external_names
    ]
    if not available_external_names:
        options.append((MANUAL_EXTERNAL_MCP_VALUE, "External MCP connection"))
    options.extend(
        [
            (SQL_MCP_VALUE, "Databricks SQL"),
            (VECTOR_SEARCH_VALUE, "Vector Search"),
            (GENIE_VALUE, "Genie Space"),
            (UC_FUNCTIONS_VALUE, "Unity Catalog functions"),
            (CUSTOM_MCP_VALUE, "Custom MCP URLs"),
        ]
    )
    return options


def prompt_for_manual_external_mcp_connection() -> str | None:
    values = prompt_for_mcp_setup_fields(
        [("MCP server name", "(e.g. confluence-mcp, jira-mcp)")]
    )
    if values is None:
        return None
    server_name = values[0]
    if not server_name:
        print_err("Server name cannot be empty.")
        return None
    return server_name


def prompt_for_mcp_setup_fields(fields: list[tuple[str, str | None]]) -> list[str] | None:
    print_note(f"Type `{MCP_SETUP_BACK_VALUE}` to return to MCP server selection.")
    values: list[str] = []
    for field_name, hint in fields:
        prompt = f"  {label(field_name)}"
        if hint:
            prompt += f" {muted(hint)}"
        prompt += f" {muted('›')} "
        raw_value = console.input(prompt).strip()
        if raw_value.lower() == MCP_SETUP_BACK_VALUE:
            return None
        values.append(raw_value)
    return values


def prompt_for_mcp_server_selection(available_external_names: list[str]) -> str:
    options = build_mcp_server_options(available_external_names)
    option_index = 1

    console.print()
    console.print("  [bold]External MCP Servers[/bold]")
    for value, option_label in options:
        if value == SQL_MCP_VALUE:
            break
        if value == MANUAL_EXTERNAL_MCP_VALUE or value.startswith(EXTERNAL_MCP_SELECTION_PREFIX):
            console.print(f"    [bold]{option_index}.[/bold] [cyan]{option_label}[/cyan]")
            option_index += 1

    console.print("  [bold]Databricks SQL[/bold]")
    console.print(f"    [bold]{option_index}.[/bold] [cyan]Databricks SQL[/cyan]")
    option_index += 1

    console.print("  [bold]Databricks Managed MCP Servers[/bold]")
    console.print(f"    [bold]{option_index}.[/bold] [cyan]Vector Search[/cyan]")
    option_index += 1
    console.print(f"    [bold]{option_index}.[/bold] [cyan]Genie Space[/cyan]")
    option_index += 1
    console.print(f"    [bold]{option_index}.[/bold] [cyan]Unity Catalog functions[/cyan]")
    option_index += 1

    console.print("  [bold]Custom MCP URLs[/bold]")
    console.print(f"    [bold]{option_index}.[/bold] [cyan]Custom MCP URLs[/cyan]")
    console.print("  [dim]Press Enter when finished.[/dim]")

    while True:
        raw_value = console.input(f"{label('Select MCP server')} {muted('›')} ").strip()
        if raw_value in {"", "0", "done"}:
            return "done"
        if raw_value.isdigit():
            selected_index = int(raw_value)
            if 1 <= selected_index <= len(options):
                return options[selected_index - 1][0]
        print_err("Please enter a valid option number.")


def configure_mcp_command() -> int:
    state = load_state()
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("Workspace is not configured. Run `coding-gateway configure` first.")

    clients = available_mcp_clients()
    if not clients:
        raise RuntimeError(
            "No supported MCP clients are installed. Install Claude, Codex, Gemini, OpenCode, "
            "or GitHub Copilot CLI."
        )
    missing_clients = [client for client in MCP_CLIENTS if client not in clients]

    ensure_databricks_auth(workspace)

    print_section("MCP Server Configuration")
    print_note("Configure installed coding tools to connect to Databricks MCP servers.")
    print_note(f"Workspace: {workspace}")
    print_note(
        "Databricks MCP servers are written to installed coding tool user MCP configs. "
        f"Auth uses `{MCP_AUTH_TOKEN_ENV_VAR}`, which coding-gateway sets before launching tools."
    )
    configured_client_names = ", ".join(str(MCP_CLIENTS[client]["display"]) for client in clients)
    print_note(f"Configuring MCP clients: {configured_client_names}")
    for client in missing_clients:
        print_warning(f"{MCP_CLIENTS[client]['display']} is not installed; skipping MCP config.")

    available_external_mcp_names: list[str] = []
    try:
        available_external_mcp_names = discover_external_mcp_connection_names(workspace)
    except RuntimeError as exc:
        print_warning(f"{exc} You can still enter an MCP connection name manually.")

    if available_external_mcp_names:
        print_success(
            f"Found {len(available_external_mcp_names)} external MCP connection"
            f"{'' if len(available_external_mcp_names) == 1 else 's'}."
        )
    else:
        print_warning("No external MCP connections were discovered.")

    mcp_servers: list[dict] = list(state.get("mcp_servers") or [])

    added: list[str] = []

    print_section("Add MCP Server")
    while True:
        selection = prompt_for_mcp_server_selection(available_external_mcp_names)

        if selection == "done":
            break

        if selection.startswith(EXTERNAL_MCP_SELECTION_PREFIX):
            if selection == MANUAL_EXTERNAL_MCP_VALUE:
                server_name = prompt_for_manual_external_mcp_connection()
            else:
                server_name = selection.removeprefix(EXTERNAL_MCP_SELECTION_PREFIX)
            if not server_name:
                continue
            url = f"{workspace}/api/2.0/mcp/external/{server_name}"
            entry_name = server_name

        elif selection == SQL_MCP_VALUE:
            url = f"{workspace}/api/2.0/mcp/sql"
            entry_name = "databricks-sql"

        elif selection == UC_FUNCTIONS_VALUE:
            values = prompt_for_mcp_setup_fields(
                [("Catalog name", None), ("Schema name", None)]
            )
            if values is None:
                continue
            catalog, schema = values
            if not catalog or not schema:
                print_err("Catalog and schema cannot be empty.")
                continue
            url = f"{workspace}/api/2.0/mcp/functions/{catalog}/{schema}"
            entry_name = f"databricks-uc-{catalog}-{schema}"

        elif selection == VECTOR_SEARCH_VALUE:
            values = prompt_for_mcp_setup_fields(
                [("Catalog name", None), ("Schema name", None), ("Index name", None)]
            )
            if values is None:
                continue
            catalog, schema, index_name = values
            if not catalog or not schema or not index_name:
                print_err("Catalog, schema, and index name cannot be empty.")
                continue
            url = f"{workspace}/api/2.0/mcp/vector-search/{catalog}/{schema}/{index_name}"
            entry_name = f"databricks-vector-search-{catalog}-{schema}-{index_name}"

        elif selection == GENIE_VALUE:
            values = prompt_for_mcp_setup_fields([("Genie space ID", None)])
            if values is None:
                continue
            space_id = values[0]
            if not space_id:
                print_err("Space ID cannot be empty.")
                continue
            url = f"{workspace}/api/2.0/mcp/genie/{space_id}"
            entry_name = f"databricks-genie-{space_id}"

        elif selection == CUSTOM_MCP_VALUE:
            values = prompt_for_mcp_setup_fields(
                [("Full MCP server URL", None), ("Server name", None)]
            )
            if values is None:
                continue
            url, entry_name = values
            if not url:
                print_err("URL cannot be empty.")
                continue
            if not entry_name:
                print_err("Server name cannot be empty.")
                continue

        else:
            continue

        entry = build_mcp_http_entry(url)
        for client in clients:
            removed_scopes = configure_client_mcp_server(client, entry_name, url, entry)
            if removed_scopes:
                scope_text = ", ".join(removed_scopes)
                print_warning(
                    f"Found existing {MCP_CLIENTS[client]['display']} MCP entry "
                    f"`{entry_name}` in: {scope_text}. Updating it."
                )
        added.append(entry_name)
        mcp_servers = [server for server in mcp_servers if server.get("name") != entry_name]
        mcp_servers.append(
            {
                "name": entry_name,
                "url": url,
                "auth": f"env:{MCP_AUTH_TOKEN_ENV_VAR}",
                "clients": clients,
            }
        )
        state["mcp_servers"] = mcp_servers
        save_state(state)
        print_success(f"Added {entry_name}")

    if not added:
        print_note("No MCP servers added.")
        return 0

    print_heading("MCP Configured")
    for name in added:
        console.print(f"  [bold green]●[/bold green] [cyan]{name}[/cyan]")
    print_success("MCP servers registered in installed coding tool user MCP configs")
    return 0
