"""Rich/questionary presentation primitives. No project knowledge."""

from __future__ import annotations

import itertools
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from datetime import timedelta

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)

# Output verbosity. "normal" (default) renders decorative panels; "low" trades
# them for terse single-line output. Set once at CLI entry via set_verbosity.
_verbosity = "normal"


def set_verbosity(value: str) -> None:
    global _verbosity
    _verbosity = value or "normal"


def get_verbosity() -> str:
    return _verbosity


def is_low_verbosity() -> bool:
    return _verbosity == "low"


_skip_update = False


def set_skip_update(value: bool) -> None:
    global _skip_update
    _skip_update = bool(value)


def is_skip_update() -> bool:
    return _skip_update


def print_section(title: str) -> None:
    console.print()
    console.print(Panel(title, style="bold blue", expand=False))


def print_heading(text: str) -> None:
    console.print()
    console.print(f"[bold]{text}[/bold]")


def print_kv(key: str, val: str) -> None:
    console.print(f"  [bold]{key}:[/bold] [cyan]{val}[/cyan]")


def print_note(text: str) -> None:
    console.print(f"[dim]•[/dim] {text}")


def print_success(message: str) -> None:
    console.print(f"[bold green]✔[/bold green] {message}")


def print_warning(message: str) -> None:
    console.print(f"[bold yellow]![/bold yellow] {message}")


def print_err(message: str) -> None:
    err_console.print(f"[bold red]ERROR[/bold red] {message}")


def render_error_panel(message: str) -> Panel:
    lines = message.splitlines()
    title = lines[0] if lines else "ERROR"
    body = "\n".join(lines[1:]) if len(lines) > 1 else ""
    return Panel(
        Text(body or message),
        title=Text(title),
        border_style="red",
        style="red",
        expand=False,
        padding=(1, 2),
    )


def print_err_panel(message: str) -> None:
    err_console.print(render_error_panel(message))


def heading(text: str) -> str:
    return f"[bold blue]{text}[/bold blue]"


def label(text: str) -> str:
    return f"[bold]{text}[/bold]"


def value(text: str) -> str:
    return f"[cyan]{text}[/cyan]"


def muted(text: str) -> str:
    return f"[dim]{text}[/dim]"


def status_badge(text: str, kind: str) -> str:
    color = {"ok": "green", "warn": "yellow", "error": "red", "info": "blue"}.get(kind, "bold")
    return f"[bold {color}]{text}[/bold {color}]"


@contextmanager
def spinner(message: str):
    if not sys.stdout.isatty():
        yield
        return

    stop_event = threading.Event()

    def spin() -> None:
        for frame in itertools.cycle("|/-\\"):
            if stop_event.is_set():
                break
            sys.stdout.write(f"\r\033[2m{frame}\033[0m {message}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(message) + 4) + "\r")
        sys.stdout.flush()

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1)


def render_box_table(
    headers: list[str],
    rows: list[list[str]],
    max_widths: list[int] | None = None,
) -> str:
    wrapped_rows: list[list[list[str]]] = []
    widths = [len(header) for header in headers]

    for row in rows:
        wrapped_row: list[list[str]] = []
        for index, cell in enumerate(row):
            raw_cell = cell if cell else "-"
            width_limit = max_widths[index] if max_widths and index < len(max_widths) else None
            if width_limit:
                cell_lines = textwrap.wrap(raw_cell, width=width_limit) or ["-"]
            else:
                cell_lines = raw_cell.splitlines() or ["-"]
            wrapped_row.append(cell_lines)
            widths[index] = max(widths[index], max(len(line) for line in cell_lines))
        wrapped_rows.append(wrapped_row)

    top = "┏" + "┳".join("━" * (w + 2) for w in widths) + "┓"
    header = "┃ " + " ┃ ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " ┃"
    middle = "┡" + "╇".join("━" * (w + 2) for w in widths) + "┩"
    bottom = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    body_lines: list[str] = []
    for wrapped_row in wrapped_rows:
        row_height = max(len(cell_lines) for cell_lines in wrapped_row)
        for line_index in range(row_height):
            body_lines.append(
                "│ "
                + " │ ".join(
                    (
                        wrapped_row[col][line_index] if line_index < len(wrapped_row[col]) else ""
                    ).ljust(widths[col])
                    for col in range(len(headers))
                )
                + " │"
            )

    return "\n".join([top, header, middle, *body_lines, bottom])


def format_token_count(token_count: int) -> str:
    value_float = float(token_count)
    if token_count >= 1_000_000_000:
        return f"{value_float / 1_000_000_000:.1f}B"
    if token_count >= 1_000_000:
        return f"{value_float / 1_000_000:.1f}M"
    if token_count >= 1_000:
        return f"{value_float / 1_000:.1f}K"
    return str(token_count)


def format_duration(duration_value: timedelta | None) -> str:
    if not duration_value or duration_value.total_seconds() <= 0:
        return "-"
    total_minutes = duration_value.total_seconds() / 60
    if total_minutes < 60:
        return f"{int(round(total_minutes))}m"
    total_hours = total_minutes / 60
    if total_hours < 10:
        return f"{total_hours:.1f}h"
    if total_hours < 24:
        return f"{round(total_hours):.0f}h"
    return f"{total_hours / 24:.1f}d"


def normalize_workspace_url(workspace: str) -> str:
    workspace = workspace.strip()
    if not workspace:
        raise ValueError("Workspace URL cannot be empty.")
    if not workspace.startswith(("http://", "https://")):
        workspace = f"https://{workspace}"
    return workspace.rstrip("/")


def prompt_for_workspace(
    description: str,
    profiles: list[tuple[str, str]] | None = None,
) -> tuple[str, str | None]:
    """Ask the user for a workspace URL, offering profiles as quick-select.

    `profiles` is a list of (host_url, profile_name) tuples. Caller fetches
    them — `ui.py` stays Databricks-agnostic. Returns ``(url, profile_name)``;
    profile_name is ``None`` when the user typed a URL manually.
    """
    console.print()
    console.print(Panel(description, title="ucode setup", style="bold blue", expand=False))

    if profiles:
        choices = [
            questionary.Choice(title=host, value=(host, profile_name))
            for host, profile_name in profiles
        ]
        choices.append(questionary.Choice(title="Enter a different URL", value=None))
        style = questionary.Style(
            [
                ("highlighted", "fg:#2a6885 bold"),
                ("pointer", "fg:#2a6885 bold"),
                ("answer", "fg:#2a6885"),
            ]
        )
        choice = questionary.select(
            "Select workspace:", choices=choices, style=style, pointer="›", qmark=""
        ).ask()
        if isinstance(choice, tuple):
            host, profile_name = choice
            return normalize_workspace_url(host), profile_name

    while True:
        raw_value = console.input(f"  [bold]Workspace URL[/bold] {muted('›')} ").strip()
        try:
            return normalize_workspace_url(raw_value), None
        except ValueError as exc:
            print_err(str(exc))


def prompt_for_tools(
    available: list[tuple[str, str]],
    preselected: list[str] | set[str] | None = None,
) -> list[str]:
    """Multi-select picker for coding agents.

    `available` is [(tool_id, display_name), ...]. Returns the chosen tool_ids.
    When ``preselected`` is ``None`` every option is checked by default (so
    hitting Enter selects everything). When ``preselected`` is provided only
    those tool ids are pre-checked. Returns [] if the user submits an empty
    selection.
    """
    style = questionary.Style(
        [
            # questionary applies `selected` to *checked* rows and
            # `highlighted` to the cursor row — overriding both to plain
            # white means only the indicator and the `›` pointer carry
            # signal, instead of the entire row inverting.
            ("pointer", "fg:#2a6885 bold"),
            ("highlighted", "fg:white noinherit"),
            ("selected", "fg:white noinherit"),
            ("answer", "fg:#2a6885"),
        ]
    )
    preselected_set: set[str] | None = (
        {str(item) for item in preselected} if preselected is not None else None
    )
    choices = [
        questionary.Choice(
            title=display,
            value=tool_id,
            checked=(preselected_set is None or tool_id in preselected_set),
        )
        for tool_id, display in available
    ]
    answer = questionary.checkbox(
        "Select coding agents to configure:",
        choices=choices,
        style=style,
        pointer="›",
        qmark="",
        instruction="(space to toggle, enter to confirm)",
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return list(answer)


def prompt_yes_no(prompt: str, default: bool | None = None) -> bool:
    """Prompt for a yes/no answer.

    When ``default`` is ``True`` or ``False``, hitting Enter on empty input
    returns the default. The hint reflects the default by capitalising the
    chosen letter (``(Y/n)`` for ``True``, ``(y/N)`` for ``False``).
    """
    if default is True:
        hint = "(Y/n)"
    elif default is False:
        hint = "(y/N)"
    else:
        hint = "(y/n)"
    while True:
        response = console.input(f"{label(prompt)} {muted(hint)} {muted('›')} ").strip().lower()
        if not response and default is not None:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print_err("Please answer yes or no.")


def prompt_for_default_agent(available: list[tuple[str, str]]) -> str:
    """Single-select picker for the workspace's default coding agent."""
    if not available:
        raise RuntimeError("Cannot pick a default agent from an empty list.")
    if len(available) == 1:
        return available[0][0]

    style = questionary.Style(
        [
            ("pointer", "fg:#2a6885 bold"),
            ("highlighted", "fg:white noinherit"),
            ("selected", "fg:white noinherit"),
            ("answer", "fg:#2a6885"),
        ]
    )
    while True:
        choices = [
            questionary.Choice(title=display, value=tool_id, checked=False)
            for tool_id, display in available
        ]
        answer = questionary.checkbox(
            "Select the default coding agent (exactly one):",
            choices=choices,
            style=style,
            pointer="›",
            qmark="",
            instruction="(space to toggle, enter to confirm)",
        ).ask()
        if answer is None:
            raise KeyboardInterrupt
        picks = list(answer)
        if len(picks) == 1:
            return picks[0]
        if not picks:
            print_err("You must select exactly one default agent. Try again.")
        else:
            print_err(
                f"You selected {len(picks)} agents; the default must be exactly one. Try again."
            )


def prompt_budget_warn_choice(
    *,
    default_agent_display: str,
    switch_display: str | None = None,
) -> str | None:
    """Selector shown when the global daily budget is nearing its limit.

    Returns ``"default"`` (continue with the agent being launched) or
    ``"switch"`` (switch to the policy-recommended tier), or ``None`` if the
    user aborts (Ctrl-C / ESC). Stays state-agnostic: the caller passes display
    labels for the available choices."""
    style = questionary.Style(
        [
            ("highlighted", "fg:#2a6885 bold"),
            ("pointer", "fg:#2a6885 bold"),
            ("answer", "fg:#2a6885"),
        ]
    )
    choices = []
    if switch_display:
        choices.append(
            questionary.Choice(title=f"Switch to {switch_display} [Recommended]", value="switch")
        )
    choices.append(
        questionary.Choice(title=f"Continue with {default_agent_display}", value="default")
    )
    return questionary.select(
        "Daily budget is nearing its limit — how would you like to continue?",
        choices=choices,
        style=style,
        pointer="›",
        qmark="",
    ).ask()


def prompt_for_choice(prompt: str, options: list[tuple[str, str]]) -> str:
    style = questionary.Style(
        [
            ("highlighted", "fg:#2a6885 bold"),
            ("pointer", "fg:#2a6885 bold"),
            ("answer", "fg:#2a6885"),
        ]
    )
    choices = [
        questionary.Choice(title=option_label, value=option_id)
        for option_id, option_label in options
    ]
    answer = questionary.select(
        prompt,
        choices=choices,
        style=style,
        pointer="›",
        qmark="",
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return str(answer)


def prompt_for_client_id() -> str:
    while True:
        client_id = console.input(f"{label('OAuth client ID')} {muted('›')} ").strip()
        if client_id:
            return client_id
        print_err("Client ID cannot be empty.")


def prompt_for_client_secret() -> str:
    while True:
        client_secret = console.input(f"{label('OAuth client secret')} {muted('›')} ").strip()
        if client_secret:
            return client_secret
        print_err("Client secret cannot be empty.")


def prompt_for_usd_amount(prompt: str, *, minimum: float = 0.01) -> float:
    """Prompt for a positive USD amount and return it as a float."""
    while True:
        raw_value = console.input(f"{label(prompt)} {muted('›')} ").strip()
        cleaned = raw_value.replace("$", "").replace(",", "")
        try:
            amount = float(cleaned)
        except ValueError:
            print_err("Please enter a valid number.")
            continue
        if amount >= minimum:
            return amount
        print_err(f"Please enter at least ${minimum:.2f}.")
