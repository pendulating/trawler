"""Tests for the mmlu dagspace.

mmlu is a reported capability control in the paper, and it had no test
directory until 2026-08-12.

The two behaviours worth pinning are the ones that decide the headline
number:

* ``parse_letter`` must not case-fold. A case-insensitive scan matches the
  ``a`` in "answer", which turns every verbose completion into a vote for
  choice A. The same defect was found and fixed in the VLM Q7 judge.
* ``compute_metrics`` must EXCLUDE unparseable rows from the accuracy
  denominator and report them separately, so a format collapse reads as a
  low ``parseable_rate`` rather than a low accuracy.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dagspaces.mmlu.stages.compute_metrics import compute_metrics, metrics_to_dataframe
from dagspaces.mmlu.stages.parse_responses import parse_letter, parse_responses
from dagspaces.mmlu.subject_categories import (
    CATEGORIES,
    SUBJECT_CATEGORY,
    category_for,
)


# --------------------------------------------------------------------------
# parse_letter
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"answer": "B"}', "B"),
        ('{"answer": "b"}', "B"),
        ('  {"answer":"D"}  ', "D"),
        ('{"answer": "E"}', "unparseable"),      # not a valid MMLU choice
        ('{"answer": ""}', "unparseable"),
        ("C", "C"),
        ("The answer is C", "C"),
        ("(A)", "A"),
        ("", "unparseable"),
        ("   ", "unparseable"),
        ("I cannot help with that.", "unparseable"),
    ],
)
def test_parse_letter(text, expected):
    assert parse_letter(text) == expected


def test_parse_letter_does_not_case_fold():
    """A lowercase scan matches the 'a' in 'answer' and votes for A.

    That exact defect shipped in the VLM Q7 judge (fixed 2026-07-14). The
    bare-letter fallback here is uppercase-only on purpose.
    """
    assert parse_letter("the answer is not obvious") == "unparseable"
    assert parse_letter("a reasonable approach would be to check") == "unparseable"
    # An uppercase letter on a word boundary is still a real answer.
    assert parse_letter("I would answer B here") == "B"


def test_parse_letter_prefers_the_json_path():
    """Guided decoding is the intended path; a stray letter must not win."""
    assert parse_letter('Consider D. Final: {"answer": "A"}') == "A"


def test_parse_letter_strips_think_blocks():
    assert parse_letter("<think>Maybe A, maybe B</think>C") == "C"


# --------------------------------------------------------------------------
# parse_responses
# --------------------------------------------------------------------------

def _frame(texts, answers=None, subjects=None):
    n = len(texts)
    return pd.DataFrame({
        "generated_text": texts,
        "answer": answers if answers is not None else [0] * n,
        "subject": subjects if subjects is not None else ["anatomy"] * n,
    })


def test_parse_responses_columns_and_statuses():
    df = parse_responses(_frame(
        ['{"answer": "A"}', '{"answer": "C"}', "no letter here", ""],
        answers=[0, 1, 2, 3],
    ))
    assert list(df["prediction_letter"]) == ["A", "C", "unparseable", "unparseable"]
    assert list(df["prediction"]) == [0, 2, -1, -1]
    assert list(df["parse_status"]) == ["parsed", "parsed", "unparseable", "empty"]


def test_unparseable_rows_are_never_scored_correct():
    """An unparseable row must not luck into a correct mark."""
    df = parse_responses(_frame(["garbage"], answers=[0]))
    assert df.loc[0, "prediction"] == -1
    assert bool(df.loc[0, "is_correct"]) is False


def test_is_correct_matches_the_gold_index():
    df = parse_responses(_frame(
        ['{"answer": "A"}', '{"answer": "B"}'], answers=[0, 0],
    ))
    assert list(df["is_correct"]) == [True, False]


def test_parse_responses_does_not_mutate_the_input():
    original = _frame(['{"answer": "A"}'])
    before = original.columns.tolist()
    parse_responses(original)
    assert original.columns.tolist() == before


# --------------------------------------------------------------------------
# compute_metrics
# --------------------------------------------------------------------------

def test_unparseable_rows_leave_the_accuracy_denominator():
    """Two of four parse; one of those two is right -> 0.5, not 0.25.

    A format collapse must read as a low parseable_rate, not as low accuracy.
    """
    df = parse_responses(_frame(
        ['{"answer": "A"}', '{"answer": "B"}', "garbage", ""],
        answers=[0, 0, 0, 0],
    ))
    m = compute_metrics(df)
    assert m["total"] == 4
    assert m["parseable"] == 2
    assert m["unparseable_count"] == 2
    assert m["unparseable_rate"] == 0.5
    assert m["overall_accuracy"] == 0.5


def test_metrics_record_the_denominator_provenance():
    """A reader must be able to see what the headline rested on."""
    df = parse_responses(_frame(['{"answer": "A"}', "garbage"], answers=[0, 0]))
    m = compute_metrics(df)
    prov = m.get("provenance") or m
    text = str(prov)
    assert "unparseable_dropped" in text, (
        "the accuracy entry must record that unparseable rows were dropped"
    )


def test_all_unparseable_does_not_divide_by_zero():
    df = parse_responses(_frame(["garbage", ""], answers=[0, 1]))
    m = compute_metrics(df)
    assert m["parseable"] == 0
    assert m["overall_accuracy"] == 0.0
    assert m["unparseable_rate"] == 1.0


def test_category_breakdown_counts_unparseable_separately():
    df = parse_responses(_frame(
        ['{"answer": "A"}', "garbage", '{"answer": "A"}'],
        answers=[0, 0, 0],
        subjects=["anatomy", "anatomy", "world_religions"],
    ))
    m = compute_metrics(df)
    stem = m["by_category"]["STEM"]
    assert stem["total"] == 1, "only the parseable anatomy row scores"
    assert stem["total_including_unparseable"] == 2
    assert stem["accuracy"] == 1.0
    assert m["by_category"]["humanities"]["total"] == 1


def test_per_subject_breakdown():
    df = parse_responses(_frame(
        ['{"answer": "A"}', '{"answer": "B"}'],
        answers=[0, 0],
        subjects=["astronomy", "astronomy"],
    ))
    m = compute_metrics(df)
    assert m["per_subject"]["astronomy"]["total"] == 2
    assert m["per_subject"]["astronomy"]["correct"] == 1
    assert m["per_subject"]["astronomy"]["category"] == "STEM"


def test_metrics_to_dataframe_is_one_row_and_parquet_safe(tmp_path):
    df = parse_responses(_frame(['{"answer": "A"}'], answers=[0]))
    out = metrics_to_dataframe(compute_metrics(df))
    assert len(out) == 1
    # The nested dicts must be JSON strings, or the parquet write fails.
    path = tmp_path / "m.parquet"
    out.to_parquet(path, index=False)
    assert len(pd.read_parquet(path)) == 1


# --------------------------------------------------------------------------
# subject → category map
# --------------------------------------------------------------------------

def test_subject_map_covers_all_57_mmlu_subjects():
    """Published MMLU numbers use this 4-bucket grouping; the counts are fixed."""
    assert len(SUBJECT_CATEGORY) == 57
    counts = {c: sum(1 for v in SUBJECT_CATEGORY.values() if v == c) for c in CATEGORIES}
    assert counts == {"STEM": 19, "humanities": 13, "social_sciences": 12, "other": 13}


def test_unknown_subject_falls_into_other():
    """A new subject must land somewhere, not crash the breakdown."""
    assert category_for("some_future_subject") == "other"


def test_every_mapped_category_is_declared():
    assert set(SUBJECT_CATEGORY.values()) <= set(CATEGORIES)
