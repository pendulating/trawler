"""Tests for paired shift metrics vs. the baseline control.

Hand-computed fixture: 4 images, baseline + smart_glasses. Image 4's
variant prediction is unparseable, so it must drop out of every paired
metric (n_paired=3) rather than default to a label.

Pairs (image: baseline -> variant):
    1: A -> A   (no flip, delta 0)
    2: B -> A   (flip, delta -1, toward abstention)
    3: C -> B   (flip, delta -1, toward abstention)
    4: B -> N/A (dropped)
"""

from __future__ import annotations

import pandas as pd
import pytest

from dagspaces.vlm_geoprivacy_aug.stages.hypothetical_metrics import (
    compute_hypothetical_metrics,
    hypothetical_metrics_to_dataframe,
)


def _fixture_df() -> pd.DataFrame:
    ids = ["1", "2", "3", "4"]
    base = pd.DataFrame(
        {
            "numeric_id": ids,
            "hyp_id": "baseline",
            "hyp_dimension": "control",
            "Q7_pred": ["A", "B", "C", "B"],
            "Q7_true": ["A", "B", "B", "B"],
            "Q7_gen": ["x", "x", "x", "x"],
        }
    )
    glasses = pd.DataFrame(
        {
            "numeric_id": ids,
            "hyp_id": "smart_glasses",
            "hyp_dimension": "capture_device",
            "Q7_pred": ["A", "A", "B", "N/A"],
            "Q7_true": ["A", "B", "B", "B"],
            "Q7_gen": ["x", "x", "x", "x"],
        }
    )
    return pd.concat([base, glasses], ignore_index=True)


class TestComputeHypotheticalMetrics:
    def test_paired_shift_metrics(self):
        m = compute_hypothetical_metrics(_fixture_df())
        shifts = m["shifts"]["smart_glasses"]["Q7"]

        assert shifts["n_paired"] == 3
        assert shifts["flip_rate"] == pytest.approx(2 / 3, abs=1e-6)
        assert shifts["mean_ordinal_shift"] == pytest.approx(-2 / 3, abs=1e-6)
        assert shifts["toward_abstention_rate"] == pytest.approx(2 / 3, abs=1e-6)
        assert shifts["toward_disclosure_rate"] == 0.0
        # Abstention among the 3 pairs: baseline 1/3 -> variant 2/3.
        assert shifts["delta_abstention_rate"] == pytest.approx(1 / 3, abs=1e-6)

    def test_provenance_records_dropped_pair(self):
        m = compute_hypothetical_metrics(_fixture_df())
        prov = m["metric_provenance"]["shifts.smart_glasses.Q7.flip_rate"]
        assert prov["n_total"] == 4
        assert prov["n_real"] == 3
        assert prov["n_defaulted"] == 1

    def test_per_variant_distributions(self):
        m = compute_hypothetical_metrics(_fixture_df())
        base_q7 = m["per_variant"]["baseline"]["Q7"]
        glasses_q7 = m["per_variant"]["smart_glasses"]["Q7"]

        assert base_q7["parseable_rate"] == 1.0
        assert base_q7["abstention_rate"] == pytest.approx(0.25, abs=1e-6)
        assert glasses_q7["parseable_rate"] == pytest.approx(0.75, abs=1e-6)
        assert glasses_q7["abstention_rate"] == pytest.approx(2 / 3, abs=1e-6)

    def test_baseline_scored_against_original_labels(self):
        m = compute_hypothetical_metrics(_fixture_df())
        # Only the control is scored on human labels (annotated un-framed).
        original = m["baseline"]["original_label_metrics"]
        # Baseline preds A,B,C,B vs true A,B,B,B -> accuracy 3/4.
        assert original["per_question"]["Q7"]["accuracy"] == pytest.approx(0.75, abs=1e-6)

    def test_missing_baseline_raises(self):
        df = _fixture_df()
        df = df[df["hyp_id"] != "baseline"]
        with pytest.raises(ValueError, match="Baseline"):
            compute_hypothetical_metrics(df)

    def test_missing_hyp_id_raises(self):
        df = _fixture_df().drop(columns=["hyp_id"])
        with pytest.raises(ValueError, match="hyp_id"):
            compute_hypothetical_metrics(df)

    def test_duplicate_baseline_ids_raise(self):
        df = pd.concat([_fixture_df()] * 2, ignore_index=True)
        with pytest.raises(ValueError, match="not unique"):
            compute_hypothetical_metrics(df)

    def test_no_pred_columns_raises(self):
        df = _fixture_df().drop(columns=["Q7_pred"])
        with pytest.raises(ValueError, match="Q\\*_pred"):
            compute_hypothetical_metrics(df)


class TestFlatten:
    def test_one_row_per_variant_with_shift_columns(self):
        m = compute_hypothetical_metrics(_fixture_df())
        flat = hypothetical_metrics_to_dataframe(m)

        assert set(flat["hyp_id"]) == {"baseline", "smart_glasses"}
        glasses = flat[flat["hyp_id"] == "smart_glasses"].iloc[0]
        assert glasses["Q7_flip_rate"] == pytest.approx(2 / 3, abs=1e-6)
        assert bool(flat[flat["hyp_id"] == "baseline"].iloc[0]["is_baseline"]) is True
