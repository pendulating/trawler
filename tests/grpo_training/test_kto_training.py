"""K2 wiring invariants (wiki/2026-07-31_kto_plan.md §6–§8 + K1-review
carry-forwards): arm row selection is exact and uncontaminated; a dataset
that isn't the one its metadata describes refuses to train; SFT-CTRL sees
desirables only; the probe subset is deterministic and covers every
held-out gold-NO chunk; each §8 gate can individually block promotion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dagspaces.grpo_training.stages.kto_probe import (
    evaluate_promotion_gates,
    probe_row,
    select_probe_chunks,
    summarize_checkpoint,
)
from dagspaces.grpo_training.stages.kto_training import (
    ARM_DEPTH,
    assert_arm_composition,
    assert_dataset_identity,
    compute_fingerprint,
    compute_save_steps,
    select_arm_rows,
)


# ---------------------------------------------------------------------------
# Fixture dataset: 2 shared undesirables + 1 mined + 2 abstains + 2 edit
# pairs per depth, over 2 chunks.
# ---------------------------------------------------------------------------
def _rows():
    rows = []

    def add(completion, label, recipe, depth, ck):
        rows.append({"prompt": f"<p:{ck}>", "completion": completion,
                     "label": label, "recipe": recipe, "depth": depth,
                     "book": ck.split("|")[0], "chunk_key": ck,
                     "split": "train"})

    for ck in ("1|a", "1|b"):
        add(f"und-{ck}", False, "und", "shared", ck)
        for depth in ("verdict", "citation", "scrutinize"):
            add(f"edit-{depth}-{ck}", True, "edit", depth, ck)
    add("mine-1", True, "mine", "shared", "1|a")
    add("abstain-synth", True, "abstain", "shared", "2|c")
    add("abstain-halluc", False, "abstain", "shared", "2|c")
    return pd.DataFrame(rows)


def _metadata(rows):
    n_shared_d = 2  # mine-1 + abstain-synth
    n_shared_u = 3  # 2 und + abstain-halluc
    return {
        "fingerprint": compute_fingerprint(rows),
        "recipe_stats": {
            "mine_desirable": 1, "undesirable": 2,
            "edit_verdict": 2, "edit_citation": 2, "edit_scrutinize": 2,
            "abstain_undesirable": 1, "abstain_desirable_sampled": 0,
            "abstain_desirable_synth": 1,
        },
        "arm_class_weights": {
            d: {"n_desirable": n_shared_d + 2, "n_undesirable": n_shared_u,
                "desirable_weight": 0.863, "undesirable_weight": 1.0}
            for d in ("verdict", "citation", "scrutinize")
        },
    }


class TestArmSelection:
    def test_kto_arm_rows_and_no_cross_depth_contamination(self):
        rows = _rows()
        for arm in ("verdict", "citation", "scrutinize"):
            sel = select_arm_rows(rows, arm)
            assert set(sel["depth"]) == {"shared", ARM_DEPTH[arm]}
            assert len(sel) == 5 + 2  # shared streams + this depth's edits
            other = {"verdict", "citation", "scrutinize"} - {arm}
            assert not sel["completion"].str.contains(
                "|".join(f"edit-{d}" for d in other)).any()

    def test_sft_ctrl_is_citation_desirables_only(self):
        sel = select_arm_rows(_rows(), "sft_ctrl")
        assert sel["label"].all()
        assert set(sel["depth"]) == {"shared", "citation"}
        assert len(sel) == 4  # mine + abstain-synth + 2 citation edits

    def test_unknown_arm_raises(self):
        with pytest.raises(ValueError, match="unknown arm"):
            select_arm_rows(_rows(), "dpo")


class TestDatasetIdentity:
    def test_consistent_dataset_passes(self):
        rows = _rows()
        assert_dataset_identity(rows, _metadata(rows))

    def test_recomposed_dataset_raises(self):
        rows = _rows()
        meta = _metadata(rows)
        tampered = rows.copy()
        tampered.loc[0, "label"] = True
        with pytest.raises(ValueError, match="fingerprint"):
            assert_dataset_identity(tampered, meta)

    def test_truncated_dataset_raises(self):
        rows = _rows()
        meta = _metadata(rows)
        meta["fingerprint"] = compute_fingerprint(rows.iloc[:-1])
        with pytest.raises(ValueError, match="rows"):
            assert_dataset_identity(rows.iloc[:-1], meta)

    def test_arm_composition_mismatch_raises(self):
        rows = _rows()
        meta = _metadata(rows)
        meta["arm_class_weights"]["citation"]["n_desirable"] += 1
        with pytest.raises(ValueError, match="realized"):
            assert_arm_composition(
                select_arm_rows(rows, "citation"), "citation", meta)

    def test_arm_composition_returns_metadata_weights(self):
        rows = _rows()
        w = assert_arm_composition(
            select_arm_rows(rows, "verdict"), "verdict", _metadata(rows))
        assert w["desirable_weight"] == 0.863


class TestSchedule:
    def test_save_steps_is_ten_percent(self):
        total, save = compute_save_steps(20059, 4, 8, 0.10)
        assert total == 627 and save == 63

    def test_tiny_dataset_never_zero(self):
        total, save = compute_save_steps(3, 4, 8, 0.10)
        assert total == 1 and save == 1


class TestProbeSubset:
    _HELD = [f"1|{i}" for i in range(40)] + [f"2|{i}" for i in range(40)] \
        + [f"no|{i}" for i in range(10)]
    _YES = {k: not k.startswith("no|") for k in _HELD}

    def test_full_tier_is_everything(self):
        assert select_probe_chunks(self._HELD, self._YES, "full") \
            == sorted(self._HELD)

    def test_screen_keeps_all_gold_no_and_hits_target(self):
        sel = select_probe_chunks(self._HELD, self._YES, "screen",
                                  n_screen=30, seed=1)
        assert len(sel) == 30
        assert all(k in sel for k in self._HELD if k.startswith("no|"))
        # round-robin: both books represented
        assert {k.split("|")[0] for k in sel} == {"1", "2", "no"}

    def test_screen_deterministic_and_seed_sensitive(self):
        a = select_probe_chunks(self._HELD, self._YES, "screen", 30, seed=1)
        b = select_probe_chunks(self._HELD, self._YES, "screen", 30, seed=1)
        c = select_probe_chunks(self._HELD, self._YES, "screen", 30, seed=2)
        assert a == b and a != c


class TestGates:
    _BASE = {"minority_acc": 0.10, "majority_acc": 0.90,
             "gate_fail_rate": 0.03, "abstain_rate_gold_no": 0.09,
             "miss_rate": 0.20}

    def _point(self, name, **over):
        return {"checkpoint": name, "minority_acc": 0.15,
                "majority_acc": 0.90, "gate_fail_rate": 0.03,
                "abstain_rate_gold_no": 0.09, "miss_rate": 0.20, **over}

    def test_sustained_gain_promotes(self):
        v = evaluate_promotion_gates(
            [self._point("c1"), self._point("c2")], self._BASE)
        assert not v[0]["promoted"]      # first save: not yet sustained
        assert v[1]["promoted"]

    def test_single_spike_does_not_promote(self):
        v = evaluate_promotion_gates(
            [self._point("c1", minority_acc=0.10), self._point("c2"),
             self._point("c3", minority_acc=0.10)], self._BASE)
        assert not any(p["promoted"] for p in v)

    def test_each_gate_blocks_individually(self):
        blockers = [
            {"majority_acc": 0.80},          # g2 seesaw
            {"gate_fail_rate": 0.10},        # g3 format
            {"abstain_rate_gold_no": 0.02},  # g3 abstention
            {"miss_rate": 0.30},             # g4 recall dodge
        ]
        for over in blockers:
            v = evaluate_promotion_gates(
                [self._point("c1"), self._point("c2", **over)], self._BASE)
            assert not v[1]["promoted"], over

    def test_gain_at_or_below_bar_never_promotes(self):
        # gain 0.02 <= promotion bar 0.022 — sustained but insufficient
        v = evaluate_promotion_gates(
            [self._point("c1", minority_acc=0.12),
             self._point("c2", minority_acc=0.12)], self._BASE)
        assert not any(p["promoted"] for p in v)


class TestProbeRows:
    def test_probe_row_reduces_per_flow(self):
        r = probe_row("1|a", True, {
            "status": "scored",
            "per_flow": [("inappropriate", True), ("inappropriate", False),
                         ("appropriate", True)],
        }, no_flow=False, n_teacher_flows=5)
        assert (r["n_viol"], r["n_viol_correct"]) == (2, 1)
        assert (r["n_appr"], r["n_appr_correct"]) == (1, 1)
        assert r["n_matched"] == 3

    def test_summarize_pools_flows_not_completions(self):
        df = pd.DataFrame([
            probe_row("1|a", True, {"status": "scored",
                                    "per_flow": [("inappropriate", True)] * 3},
                      False, 3),
            probe_row("1|b", True, {"status": "scored",
                                    "per_flow": [("inappropriate", False)]},
                      False, 2),
            probe_row("no|c", False, {"status": "scored"}, True, 0),
            probe_row("no|d", False, {"status": "gate_fail"}, False, 0),
        ])
        m = summarize_checkpoint(df)
        assert m["minority_acc"] == pytest.approx(3 / 4)   # pooled flows
        assert m["gate_fail_rate"] == pytest.approx(1 / 4)
        assert m["abstain_rate_gold_no"] == pytest.approx(1 / 2)
        assert m["miss_rate"] == pytest.approx(1 - 4 / 5)
