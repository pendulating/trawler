"""Tests for the CONFAIDE prompt builder, focused on the opt-in Tier 2
``force_answer_format`` nudge for verbose instruct models (e.g. phi-4).

The paper's reference prompt must be unchanged when the flag is off; the
directive must appear only for Tier 2 (rating) prompts when the flag is on.
"""

from __future__ import annotations

import pytest

from dagspaces.confaide.prompts import build_prompt_for_row, _TIER2_FORMAT_DIRECTIVE


class TestTier2FormatNudge:
    def test_paper_prompt_unchanged_by_default(self):
        row = {"tier": "2a", "text": "Rate this scenario."}
        assert build_prompt_for_row(row) == "Rate this scenario.\nAnswer: "
        assert _TIER2_FORMAT_DIRECTIVE not in build_prompt_for_row(row)

    def test_directive_appended_when_forced(self):
        row = {"tier": "2a", "text": "Rate this scenario."}
        out = build_prompt_for_row(row, force_answer_format=True)
        assert _TIER2_FORMAT_DIRECTIVE in out
        # Original text and the trailing 'Answer: ' anchor are preserved.
        assert out.startswith("Rate this scenario.")
        assert out.rstrip().endswith("Answer:")

    def test_tier2b_also_nudged(self):
        row = {"tier": "2b", "text": "Rate this."}
        assert _TIER2_FORMAT_DIRECTIVE in build_prompt_for_row(row, force_answer_format=True)

    def test_tier3_not_affected_by_nudge(self):
        # The directive is rating-specific; it must not leak into Tier 3 prompts.
        row = {"tier": "3_control", "story": "S", "control_question": "Q?"}
        assert _TIER2_FORMAT_DIRECTIVE not in build_prompt_for_row(row, force_answer_format=True)

    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError):
            build_prompt_for_row({"tier": "99"}, force_answer_format=True)
