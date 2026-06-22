"""Universe-level exact dedup of norms (2026-06-09 review, F5)."""

import pandas as pd

from dagspaces.grpo_training.stages.norm_universe import dedup_universe_norms


def _norm_row(gid, articulation, force="obligatory"):
    return {
        "gutenberg_id": gid,
        "raz_norm_articulation": articulation,
        "raz_norm_subject": "a guest",
        "raz_prescriptive_element": "ought to",
        "raz_norm_act": "keep confidences",
        "raz_condition_of_application": "",
        "raz_context": "",
        "raz_normative_force": force,
    }


class TestDedupUniverseNorms:
    def test_same_book_exact_duplicate_dropped(self):
        df = pd.DataFrame([
            _norm_row("11", "A guest ought to keep confidences."),
            _norm_row("11", "A guest ought to keep confidences."),
        ])
        out = dedup_universe_norms(df, "gutenberg_id")
        assert len(out) == 1

    def test_case_and_whitespace_insensitive(self):
        df = pd.DataFrame([
            _norm_row("11", "A guest OUGHT to keep confidences. "),
            _norm_row("11", "a guest ought to keep confidences."),
        ])
        out = dedup_universe_norms(df, "gutenberg_id")
        assert len(out) == 1

    def test_same_text_different_books_both_kept(self):
        df = pd.DataFrame([
            _norm_row("11", "A guest ought to keep confidences."),
            _norm_row("1342", "A guest ought to keep confidences."),
        ])
        out = dedup_universe_norms(df, "gutenberg_id")
        assert len(out) == 2

    def test_different_force_is_a_different_norm(self):
        # Identity is the full embedding text, not just the articulation.
        df = pd.DataFrame([
            _norm_row("11", "A guest ought to keep confidences.", "obligatory"),
            _norm_row("11", "A guest ought to keep confidences.", "recommended"),
        ])
        out = dedup_universe_norms(df, "gutenberg_id")
        assert len(out) == 2

    def test_keeps_first_occurrence_and_order(self):
        df = pd.DataFrame([
            _norm_row("11", "Norm A."),
            _norm_row("11", "Norm B."),
            _norm_row("11", "Norm A."),
            _norm_row("11", "Norm C."),
        ])
        out = dedup_universe_norms(df, "gutenberg_id")
        assert list(out["raz_norm_articulation"]) == ["Norm A.", "Norm B.", "Norm C."]
