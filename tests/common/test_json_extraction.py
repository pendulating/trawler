"""Tests for the canonical JSON-from-LLM-text extractor.

Covers the adversarial inputs that motivated consolidating ≥9 divergent
JSON extraction implementations (Finding 8, wiki/jul19_refactoring.md).
"""

from __future__ import annotations

import json
import textwrap

import pytest

from dagspaces.common.json_extraction import (
    extract_json_from_text,
    extract_last_json,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _json(obj: dict, **kwargs) -> str:
    return json.dumps(obj, **kwargs)


ANSWER = {"classification": "appropriate", "confidence": 0.9}
SCHEMA_EXAMPLE = {
    "classification": "<appropriate|inappropriate>",
    "confidence": "<float>",
}


# ── Fast path: whole string is valid JSON ─────────────────────────────


class TestFastPath:
    def test_clean_json(self):
        obj, err = extract_json_from_text(_json(ANSWER))
        assert obj == ANSWER
        assert err is None

    def test_clean_json_with_whitespace(self):
        obj, err = extract_json_from_text(f"  \n {_json(ANSWER)} \n ")
        assert obj == ANSWER
        assert err is None

    def test_json_array_not_dict(self):
        """A top-level JSON array is not a dict — should fall through to span."""
        obj, err = extract_json_from_text("[1, 2, 3]")
        assert obj is None
        assert err is not None


# ── Empty / non-string inputs ─────────────────────────────────────────


class TestDegenerateInputs:
    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_empty_strings(self, text):
        obj, err = extract_json_from_text(text)
        assert obj is None
        assert err is not None and "empty" in err

    def test_none_input(self):
        obj, err = extract_json_from_text(None)  # type: ignore[arg-type]
        assert obj is None

    def test_no_json_at_all(self):
        obj, err = extract_json_from_text("This is just prose, no braces.")
        assert obj is None
        assert err is not None


# ── Outermost span extraction ─────────────────────────────────────────


class TestOutermostExtraction:
    def test_json_in_prose(self):
        text = f"Here is my analysis:\n{_json(ANSWER)}\nDone."
        obj, err = extract_json_from_text(text)
        assert obj == ANSWER
        assert err is None

    def test_json_in_think_blocks(self):
        text = f"<think>Let me reason...</think>\n{_json(ANSWER)}"
        obj, err = extract_json_from_text(text)
        assert obj == ANSWER

    def test_nested_json(self):
        """Outermost correctly handles a single nested object."""
        nested = {"reasoning": {"flows": [{"subject": "A"}]}, "extraction": []}
        text = f"Result: {_json(nested)} end."
        obj, err = extract_json_from_text(text)
        assert obj == nested

    def test_single_object_wrapped_in_prose(self):
        text = f"Based on my analysis:\n{_json(ANSWER)}\nFinal answer."
        obj, _ = extract_json_from_text(text)
        assert obj == ANSWER

    def test_multiple_objects_takes_outermost_span(self):
        """When multiple objects exist, the outermost span (first { to
        last }) is attempted.  If that span isn't valid JSON, returns None.
        This matches the old extract_last_json behavior (greedy regex)."""
        text = f"{_json({'first': 1})} prose {_json({'second': 2})}"
        obj, _ = extract_json_from_text(text)
        # The span '{"first": 1} prose {"second": 2}' is not valid JSON
        assert obj is None

    def test_markdown_fenced_json(self):
        text = f"```json\n{_json(ANSWER)}\n```"
        obj, _ = extract_json_from_text(text)
        assert obj == ANSWER

    def test_multiline_json(self):
        text = textwrap.dedent("""\
            Here is the result:
            {
                "classification": "inappropriate",
                "reasoning": "PHI was disclosed"
            }
            End of response.
        """)
        obj, _ = extract_json_from_text(text)
        assert obj == {
            "classification": "inappropriate",
            "reasoning": "PHI was disclosed",
        }


# ── json_repair fallback ──────────────────────────────────────────────


class TestRepairFallback:
    def test_repair_trailing_comma(self):
        """json_repair can fix trailing commas that json.loads rejects."""
        text = '{"classification": "appropriate",}'
        # Without repair: fails
        obj_no, _ = extract_json_from_text(text, repair=False)
        # With repair: should succeed (if json_repair is installed)
        obj_yes, err = extract_json_from_text(text, repair=True)
        try:
            import json_repair  # noqa: F401

            assert obj_yes == {"classification": "appropriate"}
            assert err is None
        except ImportError:
            assert obj_no is None
            assert obj_yes is None

    def test_repair_single_quotes(self):
        """json_repair can fix single-quoted JSON."""
        text = "{'answer': 'B'}"
        obj, _ = extract_json_from_text(text, repair=True)
        try:
            import json_repair  # noqa: F401

            assert obj == {"answer": "B"}
        except ImportError:
            assert obj is None

    def test_no_repair_by_default(self):
        text = '{"key": "value",}'
        obj, _ = extract_json_from_text(text, repair=False)
        assert obj is None  # trailing comma is invalid without repair


# ── Backward compat: extract_last_json ────────────────────────────────


class TestExtractLastJsonCompat:
    def test_returns_dict_or_none(self):
        assert extract_last_json(_json(ANSWER)) == ANSWER
        assert extract_last_json("no json here") is None
        assert extract_last_json("") is None

    def test_single_object_in_prose(self):
        text = f"Analysis: {_json(ANSWER)} done."
        assert extract_last_json(text) == ANSWER

    def test_full_string_parse(self):
        assert extract_last_json('{"a": 1}') == {"a": 1}

    def test_matches_old_implementation(self):
        """Verify parity with the old common/stage_utils.py implementation
        on a range of inputs."""
        from dagspaces.common.stage_utils import extract_last_json as old_impl

        cases = [
            '{"a": 1}',
            f"prose {_json(ANSWER)} more prose",
            "no json",
            "",
            f"```json\n{_json(ANSWER)}\n```",
            '{"nested": {"deep": true}}',
            f"think\n{_json(ANSWER)}",
        ]
        for text in cases:
            new = extract_last_json(text)
            old = old_impl(text)
            assert new == old, f"Mismatch on {text!r}: new={new}, old={old}"


# ── Adversarial / edge cases ──────────────────────────────────────────


class TestAdversarial:
    def test_json_with_braces_in_strings(self):
        obj_in = {"template": "use {name} here", "value": 42}
        text = f"Output: {_json(obj_in)}"
        obj, _ = extract_json_from_text(text)
        assert obj == obj_in

    def test_unicode_in_json(self):
        obj_in = {"name": "José García", "context": "医疗记录"}
        text = f"Result: {_json(obj_in, ensure_ascii=False)}"
        obj, _ = extract_json_from_text(text)
        assert obj == obj_in

    def test_very_long_prose_around_json(self):
        prose = "A" * 5000
        text = f"{prose}\n{_json(ANSWER)}\n{prose}"
        obj, _ = extract_json_from_text(text)
        assert obj == ANSWER

    def test_only_opening_brace(self):
        obj, err = extract_json_from_text("start { but no close")
        assert obj is None
        assert err is not None

    def test_only_closing_brace(self):
        obj, err = extract_json_from_text("no open } end")
        assert obj is None
