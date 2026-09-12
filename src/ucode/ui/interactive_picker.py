"""Streaming multi-select picker built on questionary / prompt_toolkit.

Presentation-only primitives shared by the product pickers (MCP services, skills). They operate on
a generic `questionary.Choice` list and an optional `background_loader`, with no MCP- or
Databricks-specific logic, so any caller can supply its own choices and discovery walk.
"""

from __future__ import annotations

import string
import threading
import time
from collections.abc import Callable
from contextlib import suppress
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

PICKER_VISIBLE_ROWS = 10

# Cap the highlighted-row description preview so a long one (skill descriptions run
# to ~1024 chars) stays within the footer instead of dominating the screen.
_DESCRIPTION_PREVIEW_CHARS = 240
# Left margin for the description footer, applied to wrapped lines too (see get_line_prefix).
_DESCRIPTION_INDENT = "  "


def _description_preview(description: str) -> str:
    """``description`` truncated to the footer budget, with an ellipsis when clipped."""
    if len(description) <= _DESCRIPTION_PREVIEW_CHARS:
        return description
    return description[: _DESCRIPTION_PREVIEW_CHARS - 1].rstrip() + "…"


def _description_footer_tokens(description: str) -> list[tuple[str, str]]:
    """Footer tokens for the highlighted row's description.

    A caller emphasizes a leading label by formatting the description as ``"label: text"``:
    the ``label:`` renders bold and the rest as the truncated preview. A description with no
    ``": "`` renders entirely as the preview."""
    label, sep, body = description.partition(": ")
    if not sep:
        return [("class:instruction", _description_preview(description))]
    return [("bold", f"{label}{sep}"), ("class:instruction", _description_preview(body))]


class _Back:
    """Sentinel type: a wizard step returns the `_BACK` instance when the user
    presses Left (←) to go back. Distinct from None (cancel) and [] (empty)."""


# Singleton instance used everywhere; compare with `is _BACK`.
_BACK = _Back()


def picker_style() -> questionary.Style:
    return questionary.Style(
        [
            ("pointer", "fg:cyan bold"),
            ("highlighted", "noinherit"),
            ("selected", "noinherit"),
            ("answer", "fg:cyan"),
        ]
    )


class StreamingInquirerControl(InquirerControl):
    """`InquirerControl` that tolerates an empty or all-disabled choice list.

    Stock `InquirerControl.__init__` ends with ``if not self.is_selection_valid(): raise`` and
    `is_selection_valid` dereferences `pointed_at`, which `_init_choices` leaves unset when no row
    is selectable — so constructing it with an empty (or every-row-disabled) list raises, and
    navigation later hits the same unset cursor. Our picker intentionally opens on an empty list
    and fills it in via the background loader, and `ug mcp add` can legitimately show only
    already-configured (disabled) rows. Default the cursor and treat "nothing selectable" as valid
    so construction, rendering, and navigation don't crash."""

    def is_selection_valid(self) -> bool:
        if getattr(self, "pointed_at", None) is None:
            self.pointed_at = 0
        selectable = any(
            not isinstance(c, questionary.Separator) and not c.disabled for c in self.choices
        )
        if not selectable:
            # Empty, or every row a separator/disabled: nothing to validate, and nothing for the
            # navigation skip-loop to land on — report valid so we neither raise nor spin.
            return True
        if self.pointed_at >= len(self.choices):
            return False
        return super().is_selection_valid()

    def _get_choice_tokens(self):
        # Stock rendering unconditionally reads `filtered_choices[pointed_at]`, which raises on an
        # empty list. Render nothing when there are no rows (the picker is still streaming them in).
        if not self.filtered_choices:
            return []
        return super()._get_choice_tokens()


def merge_new_choices(
    existing: list[questionary.Choice | questionary.Separator],
    new_choices: list[questionary.Choice],
) -> list[questionary.Choice]:
    """Return the choices from ``new_choices`` not already present in ``existing`` (compared by
    Choice value). Used to dedupe background-streamed picker rows against what's already shown."""
    shown = {c.value for c in existing if isinstance(c, questionary.Choice)}
    return [c for c in new_choices if isinstance(c, questionary.Choice) and c.value not in shown]


def scrolling_checkbox(
    message: str,
    choices: list[questionary.Choice | questionary.Separator],
    instruction: str,
    style: questionary.Style,
    allow_back: bool = False,
    background_loader: Callable[[Callable[[list[questionary.Choice]], None]], None] | None = None,
    loading_noun: str = "MCP services",
    show_description: bool = False,
) -> Question:
    """Multi-select checkbox picker.

    ``background_loader``, if given, streams more choices in after the picker is already
    on screen: it's run on a daemon thread and handed an ``append(choices)`` callback that
    adds rows (deduped by value) and repaints, so the picker opens instantly on whatever
    ``choices`` are ready and fills in the rest without blocking. A footer shows a live
    "loading more {loading_noun}…" count while it runs.

    ``show_description`` adds a footer previewing the highlighted row's ``Choice.description``.
    It's a separate window rather than questionary's inline ``show_description`` because the
    choices window is sized to the row count, so an inline line would be clipped."""
    merged_style = merge_styles_default(
        [
            questionary.Style([("bottom-toolbar", "noreverse")]),
            style,
        ]
    )
    # Empty-tolerant control: the picker can open with zero selectable rows (streaming in via the
    # background loader, or an `mcp add` where everything is already configured) — see the subclass.
    control = StreamingInquirerControl(
        choices,
        pointer="›",
        show_description=False,
    )
    # Live loading state for the background-loader footer (see below).
    loading = {"active": background_loader is not None, "found": 0}

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

    @Condition
    def has_more_choices() -> bool:
        # Live so the scroll hint appears as background-loaded rows stream in.
        return len(control.choices) > PICKER_VISIBLE_ROWS

    @Condition
    def is_loading() -> bool:
        return bool(loading["active"])

    def loading_tokens() -> list[tuple[str, str]]:
        return [
            ("class:instruction", f"  ⏳ loading more {loading_noun}… ({loading['found']} found)")
        ]

    @Condition
    def has_search_string() -> bool:
        return control.get_search_string_tokens() is not None

    def pointed_description() -> str | None:
        if not (show_description and control.filtered_choices):
            return None
        try:
            pointed = control.get_pointed_at()
        except IndexError:
            return None
        description = getattr(pointed, "description", None)
        return description if isinstance(description, str) and description else None

    def description_tokens() -> list[tuple[str, str]]:
        description = pointed_description()
        if description is None:
            return []
        return _description_footer_tokens(description)

    @Condition
    def has_description() -> bool:
        return pointed_description() is not None

    validation_prompt: PromptSession = PromptSession(bottom_toolbar=lambda: control.error_message)
    # Render the prompt as a fixed 1-row window rather than a PromptSession
    # container: the latter expands to fill the terminal height, which in a tall
    # window pushes the choices list to the very bottom (a large blank gap).
    layout = Layout(
        HSplit(
            [
                Window(
                    height=Dimension.exact(1),
                    content=FormattedTextControl(get_prompt_tokens),
                ),
                ConditionalContainer(
                    # Height tracks the live choice count (capped at the visible max) so the
                    # window grows as background-loaded rows stream in, with no blank gap when
                    # only a few choices are present.
                    Window(
                        control,
                        height=lambda: Dimension.exact(
                            min(PICKER_VISIBLE_ROWS, max(1, len(control.choices)))
                        ),
                    ),
                    filter=~IsDone(),
                ),
                ConditionalContainer(
                    Window(
                        height=Dimension.exact(1),
                        content=FormattedTextControl(
                            lambda: [("class:instruction", "  ↑/↓ scroll for more")]
                        ),
                    ),
                    filter=has_more_choices & ~IsDone(),
                ),
                ConditionalContainer(
                    Window(
                        height=Dimension.exact(1),
                        content=FormattedTextControl(loading_tokens),
                    ),
                    filter=is_loading & ~IsDone(),
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
                # Pinned at the bottom, one blank line below the other footers, so the
                # highlighted row's description reads as a separate detail pane.
                ConditionalContainer(
                    HSplit(
                        [
                            Window(height=Dimension.exact(1)),
                            Window(
                                height=Dimension.exact(2),
                                content=FormattedTextControl(description_tokens),
                                wrap_lines=True,
                                get_line_prefix=lambda line, wrap: _DESCRIPTION_INDENT,
                            ),
                        ]
                    ),
                    filter=has_description & ~IsDone(),
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
        if control.choice_count == 0:
            return  # nothing to toggle (e.g. picker still streaming, or all rows filtered out)
        pointed = control.get_pointed_at()
        if isinstance(pointed, questionary.Separator) or pointed.disabled:
            return  # separators and already-configured (disabled) rows aren't toggleable
        pointed_choice = pointed.value
        if pointed_choice in control.selected_options:
            control.selected_options.remove(pointed_choice)
        else:
            control.selected_options.append(pointed_choice)
        perform_validation()

    @bindings.add(Keys.ControlA, eager=True)
    def _(_event: Any) -> None:
        # Toggle-all: select every selectable choice, or clear the selection if
        # everything is already selected. `a` alone is reserved for type-to-filter.
        selectable = [
            choice.value
            for choice in control.choices
            if not isinstance(choice, questionary.Separator) and not choice.disabled
        ]
        if all(value in control.selected_options for value in selectable):
            control.selected_options = []
        else:
            control.selected_options = list(selectable)
        perform_validation()

    def move_cursor_down(event: Any) -> None:
        if control.choice_count == 0:
            return
        control.select_next()
        # Bound the skip-past-disabled scan so an all-disabled list can't spin forever.
        tries = 0
        while not control.is_selection_valid() and tries < control.choice_count:
            control.select_next()
            tries += 1

    def move_cursor_up(event: Any) -> None:
        if control.choice_count == 0:
            return
        control.select_previous()
        tries = 0
        while not control.is_selection_valid() and tries < control.choice_count:
            control.select_previous()
            tries += 1

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

    if allow_back:

        @bindings.add(Keys.Left, eager=True)
        def _(event: Any) -> None:
            # Wizard back-navigation: exit this step with the _BACK sentinel so
            # the caller re-shows the previous step. Left arrow is otherwise
            # unused in this multi-select (cursor moves with up/down).
            event.app.exit(result=_BACK)

    @bindings.add(Keys.Any)
    def _(_event: Any) -> None:
        """Ignore other text input."""

    app: Application = Application(
        layout=layout,
        key_bindings=bindings,
        style=merged_style,
    )

    if background_loader is not None:
        started = False

        def run_on_loop(fn: Callable[[], None]) -> None:
            # Background updates MUST run on the picker's event-loop thread: mutating the control
            # off-thread drops rows (prompt_toolkit's invalidate() no-ops until the app is running)
            # and disturbs live input (toggling/removal). Wait briefly for the app to start, then
            # hand `fn` to the loop; bail once the picker has closed or if it never starts in time.
            nonlocal started
            deadline = time.monotonic() + 15.0
            while not (app.is_running and app.loop is not None):
                if started or time.monotonic() > deadline:
                    return
                time.sleep(0.02)
            started = True
            with suppress(Exception):
                app.loop.call_soon_threadsafe(fn)

        def append(new_choices: list[questionary.Choice]) -> None:
            def apply() -> None:
                # On the UI thread: rebind choices (atomic; render reads the list live) and repaint.
                # Selections track by value, so appended rows never disturb checkboxes/scroll/filter.
                additions = merge_new_choices(control.choices, new_choices)
                if additions:
                    control.choices = [*control.choices, *additions]
                    loading["found"] += len(additions)
                    app.invalidate()

            run_on_loop(apply)

        def worker() -> None:
            try:
                background_loader(append)
            except Exception:
                # Discovery is best-effort; a failed background walk just stops streaming.
                pass
            finally:

                def finish() -> None:
                    loading["active"] = False
                    app.invalidate()

                run_on_loop(finish)

        threading.Thread(target=worker, name="picker-loader", daemon=True).start()

    return Question(app)
