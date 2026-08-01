"""k-series dataset build (wiki/2026-07-31_kto_plan.md §4–§5 + K0 §12).

Load-bearing invariants: D1′ labels false alarms undesirable; held-out
chunks contribute ZERO rows (leakage raises); every edited desirable
round-trips the production gate; every required stream is non-empty or the
build raises; per-arm class weights hit the configured TRL ratio; the
split is deterministic and stratum-covering.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dagspaces.grpo_training.stages.kto_data_prep import (
    build_kto_dataset,
    d1_prime_label,
    label_completion,
    stratified_split,
    synth_abstain_completion,
)
from dagspaces.grpo_training.stages.modular_reward import valid_gate


class TestD1Prime:
    def test_missed_violation_is_undesirable(self):
        assert d1_prime_label([("inappropriate", False),
                               ("appropriate", True)]) == "undesirable"

    def test_caught_violation_is_desirable(self):
        assert d1_prime_label([("inappropriate", True),
                               ("appropriate", True)]) == "desirable"

    def test_false_alarm_is_undesirable_not_unlabeled(self):
        # K0 §12 amendment: 16.1% of completions cried wolf; D1 left them out.
        assert d1_prime_label([("appropriate", False),
                               ("appropriate", True)]) == "undesirable"

    def test_all_appropriate_all_correct_is_desirable(self):
        assert d1_prime_label([("appropriate", True)] * 3) == "desirable"

    def test_empty_is_excluded(self):
        assert d1_prime_label([]) == "excluded"

    def test_caught_violation_but_sloppy_majority_is_neither(self):
        # violation right but <half of matched flows correct overall
        assert d1_prime_label([("inappropriate", True),
                               ("appropriate", False),
                               ("appropriate", False),
                               ("appropriate", False)]) == "neither"


# ---------------------------------------------------------------------------
# Fixture world: 2 gold-YES chunks (one mixed, one all-appropriate) + 1
# gold-NO chunk, orthogonal embeddings steered by information_type.
# ---------------------------------------------------------------------------
_E = {"good": np.array([1.0, 0.0], dtype=np.float32),
      "bad": np.array([0.0, 1.0], dtype=np.float32)}


class _FakeGold:
    def __init__(self):
        norm = {"articulation": "Servants must never disclose household "
                                "affairs to outsiders without leave.",
                "normative_force": "prohibited", "act_polarity": "performing",
                "norm_subject": "a servant", "norm_act": "disclose affairs",
                "condition_of_application": "outside the household",
                "context": "domestic service"}
        anorm = {**norm, "normative_force": "recommended",
                 "articulation": "Duty reports to the master are proper."}
        self.index = {
            ("1", "mixed"): {"golds": ["appropriate", "inappropriate"],
                             "emb": np.stack([_E["good"], _E["bad"]]),
                             "norms": [anorm, norm]},
            ("1", "allappr"): {"golds": ["appropriate"],
                               "emb": _E["good"][None, :],
                               "norms": [anorm]},
        }

    def get(self, a, b):
        return self.index.get((str(a), str(b)))

    def embed_flows(self, flows):
        return np.stack([_E.get(str(f.get("information_type")), _E["good"] * 0)
                         for f in flows])


def _comp(flows):
    return json.dumps({"reasoning": "r", "has_information_exchange": True,
                       "flows": [
                           {"sender": "s", "recipient": "r", "subject": "x",
                            "information_type": it,
                            "transmission_principle": "t", "context": "c",
                            "appropriateness": lab, "norms_invoked": [],
                            "norm_source": "implicit", "is_new_flow": False,
                            "confidence": 5}
                           for it, lab in flows]})


_BLANKET = _comp([("good", "appropriate"), ("bad", "appropriate")])   # miss
_CORRECT = _comp([("good", "appropriate"), ("bad", "inappropriate")])  # catch
_CRYWOLF = _comp([("good", "inappropriate")])                          # alarm
_NOFLOW = json.dumps({"reasoning": "r", "has_information_exchange": False,
                      "flows": []})


def _samples():
    rows = []
    for i, t in enumerate([_BLANKET, _CORRECT, _BLANKET.replace('"x"', '"x2"')]):
        rows.append({"k0": "1", "k1": "mixed", "sample": i, "text": t})
    for i, t in enumerate([_CRYWOLF, _comp([("good", "appropriate")])]):
        rows.append({"k0": "1", "k1": "allappr", "sample": i, "text": t})
    for i, t in enumerate([_comp([("good", "appropriate")]), _NOFLOW]):
        rows.append({"k0": "1", "k1": "goldno", "sample": i, "text": t})
    return pd.DataFrame(rows)


_INFO = {("1", "mixed"): {"book": "1", "mixed": True, "gold_yes": True},
         ("1", "allappr"): {"book": "1", "mixed": False, "gold_yes": True},
         ("1", "goldno"): {"book": "1", "mixed": False, "gold_yes": False}}
_PROMPTS = {k: f"<user>chunk {k[1]}</user>" for k in _INFO}


def _build(**kw):
    kw.setdefault("heldout_frac", 0.0)  # tiny fixture: keep all in train
    return build_kto_dataset(_samples(), _FakeGold(), _INFO, _PROMPTS, **kw)


class TestBuild:
    def test_streams_and_labels(self):
        rows, meta = _build()
        s = meta["recipe_stats"]
        assert s["mine_desirable"] >= 1          # _CORRECT mined
        assert s["undesirable"] >= 2             # blanket + crywolf (D1')
        assert s["edit_verdict"] == s["edit_citation"] == s["edit_scrutinize"]
        assert s["abstain_desirable_synth"] == 2
        assert s["abstain_undesirable"] == 1     # hallucinated extraction

    def test_edited_rows_roundtrip_and_fix_the_label(self):
        rows, _ = _build()
        edits = rows[rows["recipe"] == "edit"]
        assert len(edits)
        for _, r in edits.iterrows():
            g = valid_gate(r["completion"])
            assert g.passed
            labs = {f["information_type"]: f["appropriateness"]
                    for f in g.flows}
            if r["chunk_key"] == "1|mixed":
                # blanket miss corrected to the violation gold
                assert labs.get("bad") == "inappropriate"
            else:
                # cry-wolf false alarm corrected back to appropriate
                assert labs.get("good") == "appropriate"

    def test_scrutinize_rationale_present_and_citation_written(self):
        rows, _ = _build()
        rows = rows[rows["chunk_key"] == "1|mixed"]
        scr = rows[rows["depth"] == "scrutinize"].iloc[0]["completion"]
        parsed = json.loads(scr)
        assert "Servants must never disclose" in parsed["reasoning"]
        assert parsed["reasoning"].rstrip(".").endswith("inappropriate")
        cit = rows[rows["depth"] == "citation"].iloc[0]["completion"]
        flows = json.loads(cit)["flows"]
        bad = [f for f in flows if f["information_type"] == "bad"][0]
        assert bad["norms_invoked"] == [
            "Servants must never disclose household affairs to outsiders "
            "without leave."]

    def test_teacher_rationale_used_with_template_fallback(self):
        calls = []
        MARK = "As the household's own rule puts it"

        def fn(c, parsed):
            calls.append(c)
            if len(calls) == 1:  # valid teacher rationale for ANY correction
                art = (c.norm or {}).get("articulation")
                return (f'{MARK}: "{art}" — this flow therefore is '
                        f"{c.gold}.")
            return "unrelated babble"

        rows, meta = _build(rationale_fn=fn)
        texts = " ".join(rows[rows["depth"] == "scrutinize"]["completion"])
        assert MARK in texts                       # first: teacher accepted
        assert meta["rationale_fallbacks"] >= 1    # later: fell back

    def test_arm_weights_hit_the_ratio(self):
        rows, meta = _build(target_weight_ratio=1.15)
        for depth, w in meta["arm_class_weights"].items():
            realized = (w["desirable_weight"] * w["n_desirable"]
                        / (w["undesirable_weight"] * w["n_undesirable"]))
            assert realized == pytest.approx(1.15, abs=0.02)

    def test_heldout_chunks_contribute_zero_rows(self):
        # Force every stratum to hold out its chunk (frac 1.0 with >=2 keys
        # per stratum would be needed; instead craft 2 chunks per stratum).
        info = dict(_INFO)
        info[("1", "mixed2")] = {"book": "1", "mixed": True, "gold_yes": True}
        gold = _FakeGold()
        gold.index[("1", "mixed2")] = gold.index[("1", "mixed")]
        prompts = {**_PROMPTS, ("1", "mixed2"): "<user>chunk mixed2</user>"}
        samples = pd.concat([
            _samples(),
            pd.DataFrame([{"k0": "1", "k1": "mixed2", "sample": 0,
                           "text": _BLANKET}])], ignore_index=True)
        rows, meta = build_kto_dataset(
            samples, gold, info, prompts, heldout_frac=0.5)
        held = set(meta["heldout_keys"])
        assert held  # at least one held-out chunk in the 2-chunk stratum
        assert not any(ck in held for ck in rows["chunk_key"])

    def test_split_deterministic(self):
        info = {("1", f"c{i}"): {"book": "1", "mixed": True, "gold_yes": True}
                for i in range(10)}
        a = stratified_split(info, 0.2, seed=7)
        b = stratified_split(info, 0.2, seed=7)
        c = stratified_split(info, 0.2, seed=8)
        assert a == b
        assert a != c
        assert sum(1 for v in a.values() if v == "heldout") == 2

    def test_empty_required_stream_raises(self):
        # no gold-NO chunks -> abstain stream empty -> loud failure
        info = {k: v for k, v in _INFO.items() if v["gold_yes"]}
        samples = _samples()
        samples = samples[samples["k1"] != "goldno"]
        with pytest.raises(ValueError, match="required stream"):
            build_kto_dataset(samples, _FakeGold(), info,
                              _PROMPTS, heldout_frac=0.0)

    def test_synth_abstain_passes_gate_as_no_flow(self):
        g = valid_gate(synth_abstain_completion())
        assert g.passed and g.no_flow


class TestLabelCompletion:
    def test_gate_fail_status(self):
        r = label_completion("not json", _FakeGold().get("1", "mixed"),
                             _FakeGold().embed_flows, 0.55, 0.55)
        assert r["status"] == "gate_fail"

    def test_scored_with_corrections(self):
        gold = _FakeGold()
        r = label_completion(_BLANKET, gold.get("1", "mixed"),
                             gold.embed_flows, 0.55, 0.55)
        assert r["status"] == "scored"
        assert r["label"] == "undesirable"
        assert len(r["corrections"]) == 1
        assert r["corrections"][0].gold == "inappropriate"
