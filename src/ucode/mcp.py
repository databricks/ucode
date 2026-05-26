"""MCP (Model Context Protocol) server registration for coding tools."""

from __future__ import annotations

import json
import shutil
import string
import subprocess
from collections.abc import Callable
from typing import Any

import questionary
from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition, IsDone
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.shortcuts import PromptSession
from questionary.prompts.common import InquirerControl
from questionary.question import Question
from questionary.styles import merge_styles_default

from ucode.agents import copilot, opencode
from ucode.config_io import restore_file
from ucode.databricks import (
    ensure_databricks_auth,
    list_databricks_apps,
    list_databricks_connections,
    list_genie_spaces,
)
from ucode.state import load_state, save_state
from ucode.ui import (
    print_note,
    print_section,
    print_success,
    print_warning,
)

MCP_AUTH_TOKEN_ENV_VAR = "OAUTH_TOKEN"
MCP_USER_SCOPE = "user"
MCP_CLEANUP_SCOPES = ("local", "project", MCP_USER_SCOPE)
MCP_PICKER_VISIBLE_ROWS = 10
MCP_CLIENTS = {
    "claude": {
        "binary": "claude",
        "display": "Claude Code",
        "list_command": "claude mcp list",
    },
    "codex": {
        "binary": "codex",
        "display": "Codex",
        "list_command": "codex mcp list",
    },
    "gemini": {
        "binary": "gemini",
        "display": "Gemini CLI",
        "list_command": "gemini mcp list",
    },
    "opencode": {
        "binary": "opencode",
        "display": "OpenCode",
        "list_command": "opencode mcp list",
    },
    "copilot": {
        "binary": "copilot",
        "display": "GitHub Copilot CLI",
        "list_command": "copilot mcp list",
    },
}
EXTERNAL_MCP_SELECTION_PREFIX = "external:"
SQL_MCP_VALUE = "managed:sql"
GENIE_SPACE_SELECTION_PREFIX = "genie-space:"
APP_MCP_SELECTION_PREFIX = "app:"
MCP_ADD_PREFIX = "add:"
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


def configured_mcp_clients(state: dict, installed_clients: list[str]) -> list[str]:
    configured_tools = state.get("available_tools") or []
    if not isinstance(configured_tools, list):
        configured_tools = []
    configured = set(configured_tools)
    return [
        client for client in MCP_CLIENTS if client in configured and client in installed_clients
    ]


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


def remove_client_mcp_server(client: str, name: str) -> list[str]:
    if client == "claude":
        return [scope for scope in MCP_CLEANUP_SCOPES if remove_claude_mcp_server(name, scope)]
    if client == "codex":
        return [MCP_USER_SCOPE] if remove_codex_mcp_server(name) else []
    if client == "gemini":
        return [MCP_USER_SCOPE] if remove_gemini_mcp_server(name) else []
    if client == "opencode":
        return [MCP_USER_SCOPE] if opencode.remove_mcp_server_config(name) else []
    if client == "copilot":
        return [MCP_USER_SCOPE] if copilot.remove_mcp_server_config(name) else []
    raise RuntimeError(f"Unsupported MCP client '{client}'.")


def revert_mcp_configs(state: dict) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for server in state.get("mcp_servers") or []:
        name = server.get("name")
        if not isinstance(name, str) or not name:
            continue
        for client in server.get("clients") or []:
            if client not in MCP_CLIENTS:
                continue
            removed_scopes = remove_client_mcp_server(client, name)
            results[client] = bool(removed_scopes) or results.get(client, False)

    # OpenCode MCP entries live in the normal OpenCode config and are restored
    # by the main agent config revert. Copilot stores MCP servers separately,
    # so restore its original MCP file after removing per-server entries above.
    results["copilot"] = restore_file(
        copilot.COPILOT_MCP_CONFIG_PATH,
        copilot.COPILOT_MCP_BACKUP_PATH,
        any(
            "copilot" in (server.get("clients") or []) for server in state.get("mcp_servers") or []
        ),
    ) or results.get("copilot", False)
    return results


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


def discover_external_mcp_connection_names(workspace: str, profile: str | None = None) -> list[str]:
    return external_mcp_connection_names(list_databricks_connections(workspace, profile))


def genie_mcp_servers(spaces: list[dict], workspace: str) -> list[dict]:
    servers: list[dict] = []
    seen_names: set[str] = set()
    for space in spaces:
        space_id = space.get("space_id")
        if not isinstance(space_id, str) or not space_id.strip():
            continue
        title = space.get("title")
        server_name = f"databricks-genie-{space_id.strip()}"
        if server_name in seen_names:
            continue
        seen_names.add(server_name)
        servers.append(
            {
                "name": server_name,
                "title": title.strip() if isinstance(title, str) and title.strip() else space_id,
                "url": f"{workspace}/api/2.0/mcp/genie/{space_id.strip()}",
            }
        )
    return sorted(servers, key=lambda server: str(server["title"]).lower())


def discover_genie_mcp_servers(workspace: str, profile: str | None = None) -> list[dict]:
    return genie_mcp_servers(list_genie_spaces(workspace, profile), workspace)


def app_mcp_servers(apps: list[dict]) -> list[dict]:
    servers: list[dict] = []
    seen_names: set[str] = set()
    for app in apps:
        app_name = app.get("name")
        app_url = app.get("url")
        if not isinstance(app_name, str) or not app_name.strip():
            continue
        if not app_name.strip().startswith("mcp-"):
            continue
        if not isinstance(app_url, str) or not app_url.strip():
            continue
        name = app_name.strip()
        server_name = f"databricks-app-{name}"
        if server_name in seen_names:
            continue
        seen_names.add(server_name)
        servers.append(
            {
                "name": server_name,
                "title": name,
                "url": f"{app_url.strip().rstrip('/')}/mcp",
            }
        )
    return sorted(servers, key=lambda server: str(server["title"]).lower())


def discover_app_mcp_servers(workspace: str, profile: str | None = None) -> list[dict]:
    return app_mcp_servers(list_databricks_apps(workspace, profile))


def _picker_style() -> questionary.Style:
    return questionary.Style(
        [
            ("pointer", "fg:cyan bold"),
            ("highlighted", "fg:white noinherit"),
            ("selected", "fg:white noinherit"),
            ("answer", "fg:cyan"),
        ]
    )


def _server_name(server: dict) -> str | None:
    name = server.get("name")
    return name if isinstance(name, str) and name else None


def _servers_by_name(mcp_servers: list[dict]) -> dict[str, dict]:
    servers: dict[str, dict] = {}
    for server in mcp_servers:
        name = _server_name(server)
        if name:
            servers[name] = server
    return servers


def _server_choice(name: str, checked: bool, title: str | None = None) -> questionary.Choice:
    return questionary.Choice(
        title=title or name,
        value=name,
        checked=checked,
    )


def _add_choice(selection: str, title: str) -> questionary.Choice:
    return questionary.Choice(title=title, value=f"{MCP_ADD_PREFIX}{selection}")


def _scrolling_checkbox(
    message: str,
    choices: list[questionary.Choice | questionary.Separator],
    instruction: str,
    style: questionary.Style,
) -> Question:
    merged_style = merge_styles_default(
        [
            questionary.Style([("bottom-toolbar", "noreverse")]),
            style,
        ]
    )
    control = InquirerControl(
        choices,
        pointer="›",
        show_description=False,
    )

    def get_prompt_tokens() -> list[tuple[str, str]]:
        tokens = [("class:qmark", ""), ("class:question", f" {message} ")]
        if control.is_answered:
            selected_count = len(control.selected_options)
            answer = "done" if selected_count == 0 else f"done ({selected_count} selections)"
            tokens.append(("class:answer", answer))
        else:
            tokens.append(("class:instruction", instruction))
        return tokens

    def get_selected_values() -> list[Any]:
        return [choice.value for choice in control.get_selected_values()]

    def perform_validation() -> bool:
        control.error_message = None
        return True

    prompt_session: PromptSession = PromptSession(get_prompt_tokens, reserve_space_for_menu=0)
    visible_rows = min(MCP_PICKER_VISIBLE_ROWS, max(1, len(choices)))
    has_more_choices = len(choices) > MCP_PICKER_VISIBLE_ROWS

    @Condition
    def has_search_string() -> bool:
        return control.get_search_string_tokens() is not None

    validation_prompt: PromptSession = PromptSession(bottom_toolbar=lambda: control.error_message)
    layout = Layout(
        HSplit(
            [
                prompt_session.layout.container,
                ConditionalContainer(
                    Window(control, height=Dimension(preferred=visible_rows, max=visible_rows)),
                    filter=~IsDone(),
                ),
                ConditionalContainer(
                    Window(
                        height=Dimension.exact(1),
                        content=FormattedTextControl(
                            lambda: [("class:instruction", "  ↑/↓ scroll for more")]
                        ),
                    ),
                    filter=Condition(lambda: has_more_choices) & ~IsDone(),
                ),
                ConditionalContainer(
                    Window(
                        height=Dimension.exact(2),
                        content=FormattedTextControl(control.get_search_string_tokens),
                    ),
                    filter=has_search_string & ~IsDone(),
                ),
                ConditionalContainer(
                    validation_prompt.layout.container,
                    filter=Condition(lambda: control.error_message is not None),
                ),
            ]
        )
    )

    bindings = KeyBindings()

    @bindings.add(Keys.ControlQ, eager=True)
    @bindings.add(Keys.ControlC, eager=True)
    def _(event: Any) -> None:
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    @bindings.add(" ", eager=True)
    def _(_event: Any) -> None:
        pointed_choice = control.get_pointed_at().value
        if pointed_choice in control.selected_options:
            control.selected_options.remove(pointed_choice)
        else:
            control.selected_options.append(pointed_choice)
        perform_validation()

    def move_cursor_down(event: Any) -> None:
        control.select_next()
        while not control.is_selection_valid():
            control.select_next()

    def move_cursor_up(event: Any) -> None:
        control.select_previous()
        while not control.is_selection_valid():
            control.select_previous()

    def search_filter(event: Any) -> None:
        control.add_search_character(event.key_sequence[0].key)

    for character in string.printable:
        if character in string.whitespace:
            continue
        bindings.add(character, eager=True)(search_filter)
    bindings.add(Keys.Backspace, eager=True)(search_filter)

    bindings.add(Keys.Down, eager=True)(move_cursor_down)
    bindings.add(Keys.Up, eager=True)(move_cursor_up)
    bindings.add(Keys.ControlN, eager=True)(move_cursor_down)
    bindings.add(Keys.ControlP, eager=True)(move_cursor_up)

    @bindings.add(Keys.ControlM, eager=True)
    def _(event: Any) -> None:
        control.submission_attempted = True
        if perform_validation():
            control.is_answered = True
            event.app.exit(result=get_selected_values())

    @bindings.add(Keys.Any)
    def _(_event: Any) -> None:
        """Ignore other text input."""

    return Question(
        Application(
            layout=layout,
            key_bindings=bindings,
            style=merged_style,
        )
    )


def build_mcp_picker_choices(
    available_external_names: list[str],
    available_genie_servers: list[dict],
    available_app_servers: list[dict],
    original_servers: list[dict],
) -> list[questionary.Choice | questionary.Separator]:
    original_by_name = _servers_by_name(original_servers)
    known_names = set(original_by_name)

    choices: list[questionary.Choice | questionary.Separator] = []
    displayed_names: set[str] = set()

    if "databricks-sql" in known_names:
        choices.append(_server_choice("databricks-sql", True, "Databricks SQL"))
    else:
        choices.append(_add_choice(SQL_MCP_VALUE, "Databricks SQL"))
    displayed_names.add("databricks-sql")

    for name in available_external_names:
        if name in known_names:
            choices.append(_server_choice(name, True, name))
        else:
            choices.append(_add_choice(f"{EXTERNAL_MCP_SELECTION_PREFIX}{name}", name))
        displayed_names.add(name)

    for server in available_genie_servers:
        name = _server_name(server)
        title = server.get("title")
        if not name:
            continue
        display_title = f"Genie: {title}" if isinstance(title, str) and title else name
        if name in known_names:
            choices.append(_server_choice(name, True, display_title))
        else:
            choices.append(
                _add_choice(
                    f"{GENIE_SPACE_SELECTION_PREFIX}{name.removeprefix('databricks-genie-')}",
                    display_title,
                )
            )
        displayed_names.add(name)

    for server in available_app_servers:
        name = _server_name(server)
        title = server.get("title")
        if not name:
            continue
        display_title = f"App: {title}" if isinstance(title, str) and title else name
        if name in known_names:
            choices.append(_server_choice(name, True, display_title))
        else:
            choices.append(
                _add_choice(
                    f"{APP_MCP_SELECTION_PREFIX}{name.removeprefix('databricks-app-')}",
                    display_title,
                )
            )
        displayed_names.add(name)

    for name in sorted(known_names - displayed_names):
        choices.append(_server_choice(name, True))
    return choices


def prompt_for_mcp_server_choices(
    available_external_names: list[str],
    available_genie_servers: list[dict],
    available_app_servers: list[dict],
    original_servers: list[dict],
) -> list[str] | None:
    selection = _scrolling_checkbox(
        "MCP:",
        choices=build_mcp_picker_choices(
            available_external_names,
            available_genie_servers,
            available_app_servers,
            original_servers,
        ),
        style=_picker_style(),
        instruction="(space to toggle, enter to save, type to filter)",
    ).ask()
    if selection is None:
        return None
    return [str(value) for value in selection]


def _mcp_server_clients(server: dict) -> list[str]:
    return [client for client in (server.get("clients") or []) if client in MCP_CLIENTS]


def _resolve_mcp_selection(
    selection: str,
    workspace: str,
    available_app_servers: list[dict] | None = None,
) -> tuple[str, str]:
    if selection.startswith(APP_MCP_SELECTION_PREFIX):
        app_name = selection.removeprefix(APP_MCP_SELECTION_PREFIX)
        if not app_name:
            raise RuntimeError("missing Databricks app name")
        server = _servers_by_name(available_app_servers or []).get(f"databricks-app-{app_name}")
        if not server:
            raise RuntimeError(f"Databricks app `{app_name}` was not in the discovered app list")
        url = server.get("url")
        if not isinstance(url, str) or not url:
            raise RuntimeError(f"Databricks app `{app_name}` has no MCP URL")
        return f"databricks-app-{app_name}", url

    if selection.startswith(GENIE_SPACE_SELECTION_PREFIX):
        space_id = selection.removeprefix(GENIE_SPACE_SELECTION_PREFIX)
        if not space_id:
            raise RuntimeError("missing Genie space id")
        return f"databricks-genie-{space_id}", f"{workspace}/api/2.0/mcp/genie/{space_id}"

    if selection.startswith(EXTERNAL_MCP_SELECTION_PREFIX):
        server_name = selection.removeprefix(EXTERNAL_MCP_SELECTION_PREFIX)
        if not server_name:
            raise RuntimeError("missing external connection name")
        return server_name, f"{workspace}/api/2.0/mcp/external/{server_name}"

    if selection == SQL_MCP_VALUE:
        return "databricks-sql", f"{workspace}/api/2.0/mcp/sql"

    raise RuntimeError(f"unrecognized selection prefix in `{selection}`")


def _discover_mcp_source(label: str, discover: Callable[[], list[Any]]) -> list[Any]:
    try:
        return discover()
    except RuntimeError:
        print_warning(f"Skipped {label}.")
        return []


def apply_mcp_server_changes(
    original_servers: list[dict],
    working_servers: list[dict],
    clients: list[str],
) -> bool:
    original_by_name = _servers_by_name(original_servers)
    working_by_name = _servers_by_name(working_servers)
    changed = False

    for name, server in original_by_name.items():
        if name not in working_by_name:
            for client in _mcp_server_clients(server):
                remove_client_mcp_server(client, name)
            changed = True

    for name, server in working_by_name.items():
        original = original_by_name.get(name)
        if original == server:
            continue
        url = server.get("url")
        if not isinstance(url, str) or not url:
            continue
        entry = build_mcp_http_entry(url)
        for client in clients:
            configure_client_mcp_server(client, name, url, entry)
        changed = True

    return changed


def configure_mcp_command() -> int:
    state = load_state()
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("Workspace is not configured. Run `ucode configure` first.")

    installed_clients = available_mcp_clients()
    if not installed_clients:
        raise RuntimeError(
            "No supported MCP clients are installed. Install Claude, Codex, Gemini, OpenCode, "
            "or GitHub Copilot CLI."
        )
    clients = configured_mcp_clients(state, installed_clients)
    if not clients:
        raise RuntimeError(
            "No configured MCP-capable coding agents are installed. Run `ucode configure` "
            "for Codex, Claude, Gemini, OpenCode, or GitHub Copilot CLI first."
        )
    configured_tools = set(state.get("available_tools") or [])
    missing_clients = [
        client for client in MCP_CLIENTS if client in configured_tools and client not in clients
    ]

    profile = state.get("profile")
    ensure_databricks_auth(workspace, profile)

    print_section("MCP Servers")
    client_names = ", ".join(str(MCP_CLIENTS[client]["display"]) for client in clients)
    print_note(f"Configuring for: {client_names}")
    for client in missing_clients:
        print_warning(
            f"{MCP_CLIENTS[client]['display']} is configured in ucode but not installed; "
            "skipping MCP config."
        )

    available_external_mcp_names = _discover_mcp_source(
        "external connections",
        lambda: discover_external_mcp_connection_names(workspace, profile),
    )
    available_genie_mcp_servers = _discover_mcp_source(
        "Genie spaces",
        lambda: discover_genie_mcp_servers(workspace, profile),
    )
    available_app_mcp_servers = _discover_mcp_source(
        "Databricks apps",
        lambda: discover_app_mcp_servers(workspace, profile),
    )

    original_mcp_servers: list[dict] = list(state.get("mcp_servers") or [])
    original_by_name = _servers_by_name(original_mcp_servers)
    selections = prompt_for_mcp_server_choices(
        available_external_mcp_names,
        available_genie_mcp_servers,
        available_app_mcp_servers,
        original_mcp_servers,
    )
    if selections is None:
        return 0

    working_mcp_servers: list[dict] = []
    working_names: set[str] = set()
    add_selections: list[str] = []
    for selection in selections:
        if selection.startswith(MCP_ADD_PREFIX):
            add_selections.append(selection.removeprefix(MCP_ADD_PREFIX))
            continue
        original = original_by_name.get(selection)
        if original and selection not in working_names:
            working_mcp_servers.append(original.copy())
            working_names.add(selection)

    for selection in add_selections:
        try:
            entry_name, url = _resolve_mcp_selection(
                selection,
                workspace,
                available_app_mcp_servers,
            )
        except RuntimeError as exc:
            print_warning(f"Skipped MCP selection `{selection}`: {exc}.")
            continue
        if entry_name in working_names:
            continue
        working_mcp_servers.append(
            {
                "name": entry_name,
                "url": url,
                "auth": f"env:{MCP_AUTH_TOKEN_ENV_VAR}",
                "clients": clients,
            }
        )
        working_names.add(entry_name)

    changed = apply_mcp_server_changes(original_mcp_servers, working_mcp_servers, clients)
    if changed or original_mcp_servers != working_mcp_servers:
        state["mcp_servers"] = working_mcp_servers
        save_state(state)
        print_success("Saved")
    elif not selections and not original_mcp_servers:
        # User submitted the picker without toggling anything --> make it clear nothing was selected
        print_note("No MCP servers selected. Press space to toggle an item, then enter to save.")
    return 0
