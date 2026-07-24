"""m-series stratified prescreen + m1 cache signature (redesign item 6).

Covers the frozen contract in
``wiki/grpo_redesign/prescreen-and-gates.md`` / ``migration.md``:

  * the v10 fix — variance ranks WITHIN strata, never across, so a
    high-variance majority stratum cannot crowd out a minority one (a
    force-blind top-N screen doubles the skew; m1 preserves the configured
    mix within rounding);
  * hard floor >= 1 per non-empty eligible stratum;
  * determinism (same inputs -> same selection + report);
  * report accounting (per-stratum pool/selected counts sum correctly);
  * ``task_mix`` vignette->0 excludes vignettes (the -vignette cell);
  * m1 cache signature: each ingredient flips it; irrelevant key-ordering does
    not; module-list order is canonicalized; ``formula_version == "m1"``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dagspaces.grpo_training.stages.prescreen_m1 import (
    M1_FORMULA_VERSION,
    _m1_signature_payload,
    m1_cache_signature,
    stratified_prescreen,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def _extract_pool(n_prohibited: int, n_obligatory: int,
                  hi_var: float = 0.90, lo_var: float = 0.05):
    """Extraction pool with a majority prohibited (gold-no) stratum carrying
    HIGH variance and a minority obligatory (gold-yes) stratum carrying LOW
    variance — the v10 failure geometry a force-blind screen mishandles.

    ``reward_std`` is monotone-decreasing within each stratum so ranking is
    unambiguous and the row id tie-break never has to fire.
    """
    rows = []
    for i in range(n_prohibited):
        rows.append({
            "prompt_id": f"proh-{i:04d}",
            "task_type": "extract",
            "gold_class": "no",
            "force_class": "prohibited",
            # majority strictly above every minority score
            "reward_std": hi_var - i * 1e-4,
        })
    for i in range(n_obligatory):
        rows.append({
            "prompt_id": f"obl-{i:04d}",
            "task_type": "extract",
            "gold_class": "yes",
            "force_class": "obligatory",
            "reward_std": lo_var - i * 1e-4,
        })
    return pd.DataFrame(rows)


def _forceblind_topn(rows: pd.DataFrame, target_n: int) -> pd.DataFrame:
    """The legacy variance-only screen: global top-N by reward_std."""
    return rows.sort_values("reward_std", ascending=False,
                            kind="mergesort").head(target_n)


# ---------------------------------------------------------------------------
# Stratification guarantee — the v10 fix
# ---------------------------------------------------------------------------

class TestStratificationGuarantee:
    def test_minority_not_crowded_out_and_mix_preserved(self):
        # Pool 3:1 prohibited:obligatory; majority has all the variance.
        rows = _extract_pool(n_prohibited=300, n_obligatory=100)
        target_n = 100
        selected, report = stratified_prescreen(
            rows, target_n=target_n, seed=0, task_mix={"extract": 1.0},
        )

        # m1 allocates by pool proportion within the (single) task:
        # 75 prohibited : 25 obligatory.
        sel_force = selected["force_class"].value_counts().to_dict()
        assert sel_force.get("prohibited") == 75
        assert sel_force.get("obligatory") == 25

        # Realized force mix matches the pool mix within rounding (1 row / N).
        pool_share = 300 / 400
        realized_share = report["realized_force_mix"]["prohibited"]
        assert abs(realized_share - pool_share) <= 1.0 / target_n

        # The force-blind screen doubles (here: explodes) the skew: it takes
        # only high-variance prohibited rows, dropping the minority entirely.
        fb = _forceblind_topn(rows, target_n)
        assert (fb["force_class"] == "obligatory").sum() == 0
        # m1 keeps the minority present; force-blind does not.
        assert (selected["force_class"] == "obligatory").sum() > 0

    def test_within_stratum_ranking_takes_top_variance(self):
        # Minority stratum still ranks by variance *within itself*.
        rows = _extract_pool(n_prohibited=40, n_obligatory=40)
        selected, _ = stratified_prescreen(
            rows, target_n=40, seed=0, task_mix={"extract": 1.0},
        )
        # 20 from each stratum; the obligatory picks are the highest-variance
        # obligatory rows (ids obl-0000..obl-0019), never the low-variance tail.
        obl = selected[selected["force_class"] == "obligatory"]["prompt_id"]
        assert set(obl) == {f"obl-{i:04d}" for i in range(20)}


# ---------------------------------------------------------------------------
# Floor >= 1
# ---------------------------------------------------------------------------

class TestFloor:
    def test_tiny_minority_stratum_gets_at_least_one(self):
        # 1000 majority, a single minority row whose proportional target rounds
        # to well under 1 — the floor must still keep it.
        rows = _extract_pool(n_prohibited=1000, n_obligatory=1)
        selected, report = stratified_prescreen(
            rows, target_n=50, seed=0, task_mix={"extract": 1.0},
        )
        assert (selected["force_class"] == "obligatory").sum() == 1
        assert report["strata"]["extract|yes|obligatory"]["selected"] == 1

    def test_floor_only_for_configured_tasks(self):
        # A task mixed to 0 must NOT receive a floored selection.
        rows = _mixed_task_pool()
        selected, report = stratified_prescreen(
            rows, target_n=20, seed=0,
            task_mix={"extract": 1.0, "vignette": 0.0},
        )
        assert (selected["task_type"] == "vignette").sum() == 0
        assert report["strata"]["vignette|none|mixed"]["selected"] == 0


def _mixed_task_pool():
    rows = []
    for i in range(60):
        rows.append({
            "prompt_id": f"ex-{i:04d}", "task_type": "extract",
            "gold_class": "no" if i % 2 else "yes",
            "force_class": "prohibited" if i % 2 else "obligatory",
            "reward_std": 0.5 + i * 1e-4,
        })
    for i in range(40):
        rows.append({
            "prompt_id": f"vg-{i:04d}", "task_type": "vignette",
            "gold_class": "none", "force_class": "mixed",
            "reward_std": 0.3 + i * 1e-4,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# task_mix honoured
# ---------------------------------------------------------------------------

class TestTaskMix:
    def test_configured_task_mix_realized_within_rounding(self):
        rows = _mixed_task_pool()  # 60 extract, 40 vignette
        target_n = 50
        selected, report = stratified_prescreen(
            rows, target_n=target_n, seed=0,
            task_mix={"extract": 0.7, "vignette": 0.3},
        )
        n_vig = int((selected["task_type"] == "vignette").sum())
        n_ext = int((selected["task_type"] == "extract").sum())
        assert n_ext + n_vig == len(selected)
        # 0.7/0.3 of 50 = 35 / 15, within one row of rounding.
        assert abs(n_ext - 35) <= 1
        assert abs(n_vig - 15) <= 1
        assert abs(report["realized_task_mix"]["extract"] - 0.7) <= 1.0 / target_n

    def test_default_task_mix_follows_pool(self):
        rows = _mixed_task_pool()  # 60:40 pool
        selected, _ = stratified_prescreen(rows, target_n=50, seed=0)
        n_ext = int((selected["task_type"] == "extract").sum())
        # Pool proportion 0.6 -> ~30 of 50.
        assert abs(n_ext - 30) <= 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_selection_and_report(self):
        rows = _mixed_task_pool()
        sel1, rep1 = stratified_prescreen(
            rows, target_n=37, seed=7, task_mix={"extract": 0.7, "vignette": 0.3})
        sel2, rep2 = stratified_prescreen(
            rows, target_n=37, seed=7, task_mix={"extract": 0.7, "vignette": 0.3})
        assert list(sel1["prompt_id"]) == list(sel2["prompt_id"])
        assert rep1 == rep2

    def test_target_ge_pool_selects_all_eligible(self):
        rows = _extract_pool(n_prohibited=10, n_obligatory=5)
        selected, report = stratified_prescreen(
            rows, target_n=999, seed=0, task_mix={"extract": 1.0})
        assert len(selected) == len(rows)
        assert report["n_selected"] == len(rows)


# ---------------------------------------------------------------------------
# Report accounting
# ---------------------------------------------------------------------------

class TestReportAccounting:
    def test_pool_and_selected_counts_sum(self):
        rows = _mixed_task_pool()
        selected, report = stratified_prescreen(
            rows, target_n=50, seed=0,
            task_mix={"extract": 0.7, "vignette": 0.3})
        strata = report["strata"]
        assert sum(s["pool"] for s in strata.values()) == report["n_pool"] == len(rows)
        assert sum(s["selected"] for s in strata.values()) == report["n_selected"]
        assert report["n_selected"] == len(selected)
        assert report["n_pool"] - report["n_selected"] == report["n_dropped"]

    def test_report_carries_formula_version_and_seed(self):
        rows = _mixed_task_pool()
        _, report = stratified_prescreen(rows, target_n=10, seed=3)
        assert report["formula_version"] == "m1"
        assert report["seed"] == 3
        assert report["variance_col"] == "reward_std"

    def test_custom_variance_column(self):
        rows = _extract_pool(n_prohibited=20, n_obligatory=20)
        rows = rows.rename(columns={"reward_std": "group_spread"})
        selected, report = stratified_prescreen(
            rows, target_n=20, seed=0, task_mix={"extract": 1.0},
            variance_col="group_spread")
        assert report["variance_col"] == "group_spread"
        assert len(selected) == 20

    def test_missing_variance_column_raises(self):
        rows = _extract_pool(n_prohibited=5, n_obligatory=5)
        rows = rows.drop(columns=["reward_std"])
        with pytest.raises(KeyError):
            stratified_prescreen(rows, target_n=5, seed=0)


# ---------------------------------------------------------------------------
# m1 cache signature
# ---------------------------------------------------------------------------

def _sig(**over):
    base = dict(
        module_list=["ground", "contrast"],
        task_mix={"extract": 0.7, "vignette": 0.3},
        seed=0,
        data_fingerprint="fp-abc",
    )
    base.update(over)
    return m1_cache_signature(**base)


class TestM1CacheSignature:
    def test_formula_version_embedded(self):
        payload = _m1_signature_payload(
            module_list=["ground"], task_mix={"extract": 1.0}, seed=0,
            data_fingerprint="fp")
        assert payload["formula_version"] == M1_FORMULA_VERSION == "m1"

    def test_identical_inputs_hit_cache(self):
        assert _sig() == _sig()

    def test_module_list_content_flips(self):
        assert _sig(module_list=["ground", "contrast"]) != _sig(module_list=["ground"])
        assert _sig(module_list=["ground"]) != _sig(module_list=["contrast"])

    def test_module_list_order_canonicalized(self):
        # order is irrelevant — it is the reward-auxiliary *set* that keys.
        assert _sig(module_list=["ground", "contrast"]) == _sig(module_list=["contrast", "ground"])

    def test_task_mix_flips(self):
        assert _sig(task_mix={"extract": 0.7, "vignette": 0.3}) != \
            _sig(task_mix={"extract": 1.0, "vignette": 0.0})

    def test_task_mix_key_order_irrelevant(self):
        # irrelevant-key insensitivity: insertion order must not change the hash.
        a = _sig(task_mix={"extract": 0.7, "vignette": 0.3})
        b = _sig(task_mix={"vignette": 0.3, "extract": 0.7})
        assert a == b

    def test_seed_flips(self):
        assert _sig(seed=0) != _sig(seed=1)

    def test_data_fingerprint_flips(self):
        assert _sig(data_fingerprint="fp-abc") != _sig(data_fingerprint="fp-xyz")

    def test_extra_flips(self):
        assert _sig(extra={"answerer": "gemma-4-31b"}) != \
            _sig(extra={"answerer": "qwen3.6-27b"})

    def test_extra_none_equals_empty(self):
        assert _sig(extra=None) == _sig(extra={})

    def test_extra_key_order_irrelevant(self):
        a = _sig(extra={"answerer": "g", "judge": "j"})
        b = _sig(extra={"judge": "j", "answerer": "g"})
        assert a == b

    def test_returns_sha1_hexdigest(self):
        s = _sig()
        assert len(s) == 40
        assert all(c in "0123456789abcdef" for c in s)
