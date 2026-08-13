"""Tests for the simpleqa_verified dagspace.

This dagspace is the factuality capability control, and it had no test
directory until 2026-08-12.

Two things decide its headline number:

* ``parse_grade_letter`` must not case-fold. A case-insensitive scan matches
  the ``a`` in "answer" and in "incorrect", turning verbose judge output into
  a vote for A = correct. The comment in the production code says so; these
  tests hold it there.
* ``compute_metrics`` must keep unparseable judge calls OUT of the rate
  denominators. SimpleQA's published F1 is a harmonic mean of
  ``accuracy_given_attempted`` and ``attempted_rate``; folding unparseable
  rows into either one biases the composite silently.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dagspaces.simpleqa_verified.stages.compute_metrics import (
    compute_metrics,
    metrics_to_dataframe,
)
from dagspaces.simpleqa_verified.stages.judge_grade import (
    letter_to_verdict,
    parse_grade_letter,
)


# --------------------------------------------------------------------------
# parse_grade_letter
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"grade": "A"}', "A"),
        ('{"grade": "b"}', "B"),
        ('  {"grade":"C"}  ', "C"),
        ('{"grade": "D"}', "unparseable"),   # not a SimpleQA grade
        ("A", "A"),
        ("B", "B"),
        ("C", "C"),
        ("", "unparseable"),
        ("   ", "unparseable"),
    ],
)
def test_parse_grade_letter(text, expected):
    assert parse_grade_letter(text) == expected


def test_parse_grade_letter_does_not_case_fold():
    """Lowercase scanning would read 'a' out of "answer" and "incorrect".

    That would grade a verbose judge reply as A = correct — a silent
    inflation of the headline accuracy.
    """
    assert parse_grade_letter("the answer is unclear") == "unparseable"
    assert parse_grade_letter("this looks incorrect to me") == "unparseable"
    assert parse_grade_letter("a bit of both, honestly") == "unparseable"


def test_parse_grade_letter_prefers_structured_output():
    assert parse_grade_letter('Reasoning about B. {"grade": "A"}') == "A"


def test_parse_grade_letter_finds_a_bare_uppercase_letter():
    """The grader prompt asks for a bare letter; word boundaries only."""
    assert parse_grade_letter("Grade: B") == "B"
    assert parse_grade_letter("ABC") == "unparseable", "no word boundary"


# --------------------------------------------------------------------------
# letter_to_verdict
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "letter,verdict",
    [("A", "correct"), ("B", "incorrect"), ("C", "not_attempted"),
     ("D", "unparseable"), ("unparseable", "unparseable"), ("", "unparseable")],
)
def test_letter_to_verdict(letter, verdict):
    assert letter_to_verdict(letter) == verdict


def test_unparseable_never_becomes_not_attempted():
    """An unreadable judge reply is NOT the model declining to answer.

    Collapsing the two would move rows into not_attempted, which lowers
    attempted_rate and therefore the published F1, with no visible cause.
    """
    assert letter_to_verdict("unparseable") != "not_attempted"


# --------------------------------------------------------------------------
# compute_metrics
# --------------------------------------------------------------------------

def _verdicts(**counts) -> pd.DataFrame:
    rows = []
    for verdict, n in counts.items():
        rows.extend([verdict] * n)
    return pd.DataFrame({"verdict": rows})


def test_counts_and_judged_total():
    m = compute_metrics(_verdicts(correct=3, incorrect=2, not_attempted=1, unparseable=4))
    assert m["total"] == 10
    assert (m["correct"], m["incorrect"], m["not_attempted"]) == (3, 2, 1)
    assert m["unparseable"] == 4
    assert m["judged"] == 6, "judged excludes unparseable"


def test_rates_use_the_judged_denominator_not_the_total():
    """Unparseable judge calls must not deflate the headline rates."""
    m = compute_metrics(_verdicts(correct=3, incorrect=1, unparseable=6))
    assert m["judged"] == 4
    assert m["correct_rate"] == 0.75, "3/4 judged, not 3/10 total"
    assert m["unparseable_rate"] == 0.6, "surfaced separately, against total"


def test_accuracy_given_attempted_excludes_not_attempted():
    m = compute_metrics(_verdicts(correct=3, incorrect=1, not_attempted=4))
    assert m["attempted_rate"] == 0.5           # 4 attempted of 8 judged
    assert m["accuracy_given_attempted"] == 0.75  # 3 correct of 4 attempted


def test_f1_is_the_harmonic_mean_of_the_two_components():
    m = compute_metrics(_verdicts(correct=3, incorrect=1, not_attempted=4))
    acc, att = m["accuracy_given_attempted"], m["attempted_rate"]
    expected = round(2 * acc * att / (acc + att), 6)
    assert m["f1"] == expected
    assert m["f1"] == pytest.approx(0.6, abs=1e-6)


def test_a_perfect_run_scores_one():
    m = compute_metrics(_verdicts(correct=5))
    assert m["accuracy_given_attempted"] == 1.0
    assert m["attempted_rate"] == 1.0
    assert m["f1"] == 1.0


def test_a_full_abstention_scores_zero_without_dividing_by_zero():
    """Every answer hedged: attempted_rate 0, so F1 must be 0, not NaN."""
    m = compute_metrics(_verdicts(not_attempted=5))
    assert m["attempted_rate"] == 0.0
    assert m["accuracy_given_attempted"] == 0.0
    assert m["f1"] == 0.0


def test_all_unparseable_does_not_divide_by_zero():
    m = compute_metrics(_verdicts(unparseable=5))
    assert m["judged"] == 0
    assert m["correct_rate"] == 0.0
    assert m["f1"] == 0.0
    assert m["unparseable_rate"] == 1.0


def test_empty_frame_is_safe():
    m = compute_metrics(pd.DataFrame({"verdict": []}))
    assert m["total"] == 0
    assert m["f1"] == 0.0


def test_metrics_record_the_dropped_unparseable_rows():
    m = compute_metrics(_verdicts(correct=1, unparseable=1))
    assert "unparseable_dropped" in str(m), (
        "the rate entries must record that unparseable rows were dropped"
    )


def test_per_topic_breakdown_when_topics_are_present():
    df = pd.DataFrame({
        "verdict": ["correct", "incorrect", "correct", "unparseable"],
        "topic": ["science", "science", "history", "history"],
    })
    m = compute_metrics(df)
    assert m["per_topic"]["science"]["total"] == 2
    assert m["per_topic"]["science"]["accuracy_given_attempted"] == 0.5
    assert m["per_topic"]["history"]["judged"] == 1
    assert m["per_topic"]["history"]["accuracy_given_attempted"] == 1.0


def test_no_topic_breakdown_when_the_column_is_absent():
    m = compute_metrics(_verdicts(correct=1))
    assert "per_topic" not in m


def test_metrics_to_dataframe_is_one_row_and_parquet_safe(tmp_path):
    df = pd.DataFrame({
        "verdict": ["correct", "incorrect"],
        "topic": ["science", "science"],
    })
    out = metrics_to_dataframe(compute_metrics(df))
    assert len(out) == 1
    path = tmp_path / "m.parquet"
    out.to_parquet(path, index=False)
    assert len(pd.read_parquet(path)) == 1
