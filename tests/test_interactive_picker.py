"""Tests for the streaming multi-select picker primitives."""

from __future__ import annotations

import questionary

from ucode.ui.interactive_picker import (
    _DESCRIPTION_PREVIEW_CHARS,
    StreamingInquirerControl,
    _description_footer_tokens,
    _description_preview,
    merge_new_choices,
)


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


class TestDescriptionPreview:
    def test_short_description_is_unchanged(self):
        assert _description_preview("Routes tickets.") == "Routes tickets."

    def test_at_the_limit_is_unchanged(self):
        text = "x" * _DESCRIPTION_PREVIEW_CHARS
        assert _description_preview(text) == text

    def test_long_description_is_clipped_with_an_ellipsis(self):
        preview = _description_preview("y" * (_DESCRIPTION_PREVIEW_CHARS + 50))
        assert len(preview) == _DESCRIPTION_PREVIEW_CHARS
        assert preview.endswith("…")


class TestDescriptionFooterTokens:
    def test_bold_label_precedes_the_preview_body(self):
        tokens = _description_footer_tokens("Routes tickets.")
        assert tokens == [("bold", "Skill description: "), ("class:instruction", "Routes tickets.")]

    def test_body_is_the_truncated_preview(self):
        long = "z" * (_DESCRIPTION_PREVIEW_CHARS + 10)
        _label_style, label_text = _description_footer_tokens(long)[0]
        body_style, body_text = _description_footer_tokens(long)[1]
        assert label_text == "Skill description: "
        assert body_style == "class:instruction"
        assert body_text == _description_preview(long)
