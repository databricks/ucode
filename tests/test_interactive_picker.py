"""Tests for the streaming multi-select picker primitives."""

from __future__ import annotations

import questionary

from ucode.ui.interactive_picker import StreamingInquirerControl, merge_new_choices


def test_merge_new_choices_dedupes_by_value():
    # Streamed rows are deduped against what's already shown, by Choice value, so a row already
    # listed (e.g. from a fast up-front pass) isn't added twice.
    a = questionary.Choice(title="A", value="cat.sch.a")
    b = questionary.Choice(title="B", value="cat.sch.b")
    merged = merge_new_choices([a], [a, b])
    assert [c.value for c in merged] == ["cat.sch.b"]


class TestStreamingInquirerControl:
    """The picker must tolerate an empty or all-disabled choice list — it opens empty and streams
    rows in via the background loader, and additive pickers can show only already-configured rows.
    Stock InquirerControl raises in __init__/render for these; the subclass must not."""

    def test_tolerates_empty_choice_list(self):
        control = StreamingInquirerControl([], pointer="›", show_description=False)
        assert control.is_selection_valid() is True
        assert control._get_choice_tokens() == []

    def test_tolerates_all_disabled_choices(self):
        disabled = questionary.Choice(title="svc", value="svc", disabled="already configured")
        assert disabled.disabled
        control = StreamingInquirerControl([disabled], pointer="›", show_description=False)
        assert control.is_selection_valid() is True
        control._get_choice_tokens()  # must not raise
