"""Tests for the m-series battery builder (`T-VIGNETTE` construction).

Covers the task-vignettes.md / data.md contract:
  * cluster_contexts: greedy single-linkage, threshold gating, determinism;
  * build_batteries: eligibility (governs ∧ decisive ∧ non-empty context),
    min_k skip, composition floors (minority_floor/target), both-polarity mixing,
    single-polarity clusters, per-cluster grouping, multiple non-overlapping
    batteries, leak exclusion + count, determinism;
  * anti-leak: force option words appear only in the instruction, never in a
    scenario_text (PrivacyLens canary pattern).
"""

import math

import numpy as np
import pytest

from dagspaces.grpo_training.stages import batteries as B
from dagspaces.grpo_training.stages import probes


# --------------------------------------------------------------------------- #
# cluster_contexts                                                            #
# --------------------------------------------------------------------------- #

def _embed_from(mapping):
    """embed_fn stub: look each context up in `mapping` → fixed vector."""
    def embed(contexts):
        return np.array([mapping[c] for c in contexts], dtype=float)
    return embed


# unit vectors at 0°, 30°, 60° — A·B = cos30 ≈ .866, B·C = cos30, A·C = cos60 = .5
_A = [1.0, 0.0]
_B = [math.cos(math.radians(30)), math.sin(math.radians(30))]
_C = [math.cos(math.radians(60)), math.sin(math.radians(60))]


class TestClusterContexts:
    def test_empty(self):
        assert B.cluster_contexts([], _embed_from({})) == []

    def test_single(self):
        assert B.cluster_contexts(["x"], _embed_from({"x": _A})) == [0]

    def test_identical_pool(self):
        m = {"a": _A, "b": _A}
        assert B.cluster_contexts(["a", "b"], _embed_from(m), threshold=0.8) == [0, 0]

    def test_orthogonal_split(self):
        m = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        assert B.cluster_contexts(["a", "b"], _embed_from(m), threshold=0.8) == [0, 1]

    def test_single_linkage_chain(self):
        # A~B (.866) and B~C (.866) both clear .8, but A~C (.5) does not; single
        # linkage pulls C into A's cluster *through* B.
        m = {"A": _A, "B": _B, "C": _C}
        assert B.cluster_contexts(["A", "B", "C"], _embed_from(m), threshold=0.8) == [0, 0, 0]

    def test_no_link_without_bridge(self):
        # Same A and C without the bridging B: .5 < .8 → separate clusters.
        m = {"A": _A, "C": _C}
        assert B.cluster_contexts(["A", "C"], _embed_from(m), threshold=0.8) == [0, 1]

    def test_threshold_gates(self):
        m = {"A": _A, "B": _B}  # sim .866
        assert B.cluster_contexts(["A", "B"], _embed_from(m), threshold=0.9) == [0, 1]
        assert B.cluster_contexts(["A", "B"], _embed_from(m), threshold=0.8) == [0, 0]

    def test_deterministic(self):
        m = {"A": _A, "B": _B, "C": _C}
        r1 = B.cluster_contexts(["A", "B", "C"], _embed_from(m), threshold=0.8)
        r2 = B.cluster_contexts(["A", "B", "C"], _embed_from(m), threshold=0.8)
        assert r1 == r2

    def test_zero_norm_vector_no_crash(self):
        m = {"a": [0.0, 0.0], "b": [1.0, 0.0]}
        # zero-norm row → 0 similarity to everything → its own cluster
        assert B.cluster_contexts(["a", "b"], _embed_from(m), threshold=0.8) == [0, 1]


# --------------------------------------------------------------------------- #
# build_batteries fixtures                                                     #
# --------------------------------------------------------------------------- #

def _bn(subject, force, context="a shared narrative context", governs=True,
        articulation="the withheld source rule text", condition="", act="share information"):
    return {
        "norm_subject": subject,
        "norm_act": act,
        "condition_of_application": condition,
        "context": context,
        "normative_force": force,
        "governs_info_flow": governs,
        "norm_articulation": articulation,
    }


def _yes(subject="a doctor", **kw):
    return _bn(subject, "obligatory", **kw)


def _no(subject="a patriarch", **kw):
    return _bn(subject, "prohibited", **kw)


# --------------------------------------------------------------------------- #
# build_batteries: composition                                                 #
# --------------------------------------------------------------------------- #

class TestBuildComposition:
    def test_basic_both_polarity_mix(self):
        book = [_yes(f"y{i}") for i in range(6)] + [_no(f"n{i}") for i in range(2)]
        cluster_ids = [0] * 8
        bats = B.build_batteries(book, "1342", cluster_ids, k=8, min_k=4)
        assert len(bats) == 1
        b = bats[0]
        assert b["composition"]["n"] == 8
        assert b["composition"]["n_gold_no"] == 2
        assert b["composition"]["n_gold_yes"] == 6
        assert b["cluster_id"] == 0
        assert b["gutenberg_id"] == "1342"
        assert b["n_leak_skipped"] == 0
        # items carry the frozen item schema
        for it in b["items"]:
            assert set(it) == {"norm_index", "gold_force", "scenario_text", "articulation"}
            assert it["gold_force"] in ("obligatory", "prohibited")

    def test_minority_floor_when_scarce(self):
        # 7 yes + 1 no: minority (no) below target 2 but floor 1 must hold.
        book = [_yes(f"y{i}") for i in range(7)] + [_no("only-no")]
        bats = B.build_batteries(book, "1342", [0] * 8, k=8, min_k=4,
                                 minority_floor=1, minority_target=2)
        assert len(bats) == 1
        assert bats[0]["composition"]["n_gold_no"] == 1
        assert bats[0]["composition"]["n_gold_yes"] == 7

    def test_minority_target_reached_when_available(self):
        book = [_yes(f"y{i}") for i in range(6)] + [_no(f"n{i}") for i in range(6)]
        # k=8, target 2 → 2 minority (no) + 6 majority (yes)
        bats = B.build_batteries(book, "1342", [0] * 12, k=8, min_k=4,
                                 minority_target=2)
        # 12 norms, k=8 → first battery 8, remainder 4 → second battery 4
        first = bats[0]
        assert first["composition"]["n"] == 8
        assert first["composition"]["n_gold_no"] == 2

    def test_both_polarities_present_when_cluster_has_them(self):
        book = [_yes(f"y{i}") for i in range(3)] + [_no(f"n{i}") for i in range(3)]
        bats = B.build_batteries(book, "1342", [0] * 6, k=8, min_k=4)
        assert len(bats) == 1
        c = bats[0]["composition"]
        assert c["n_gold_no"] >= 1 and c["n_gold_yes"] >= 1

    def test_single_polarity_cluster_allowed(self):
        book = [_yes(f"y{i}") for i in range(5)]
        bats = B.build_batteries(book, "1342", [0] * 5, k=8, min_k=4)
        assert len(bats) == 1
        assert bats[0]["composition"]["n_gold_no"] == 0
        assert bats[0]["composition"]["n_gold_yes"] == 5


# --------------------------------------------------------------------------- #
# build_batteries: skips, grouping, multiple batteries                         #
# --------------------------------------------------------------------------- #

class TestBuildStructure:
    def test_min_k_skip(self):
        book = [_yes(f"y{i}") for i in range(3)]  # < min_k
        bats = B.build_batteries(book, "1342", [0] * 3, k=8, min_k=4)
        assert bats == []

    def test_per_cluster_grouping(self):
        # 4 yes in cluster 0, 4 no in cluster 1 → two single-polarity batteries.
        book = [_yes(f"y{i}") for i in range(4)] + [_no(f"n{i}") for i in range(4)]
        cluster_ids = [0, 0, 0, 0, 1, 1, 1, 1]
        bats = B.build_batteries(book, "1342", cluster_ids, k=8, min_k=4)
        assert len(bats) == 2
        by_cluster = {b["cluster_id"]: b for b in bats}
        assert by_cluster[0]["composition"]["n_gold_yes"] == 4
        assert by_cluster[0]["composition"]["n_gold_no"] == 0
        assert by_cluster[1]["composition"]["n_gold_no"] == 4
        assert by_cluster[1]["composition"]["n_gold_yes"] == 0

    def test_multiple_batteries_no_norm_reuse(self):
        # 12 yes + 4 no in one cluster, k=8 → two batteries, disjoint norms.
        book = [_yes(f"y{i}") for i in range(12)] + [_no(f"n{i}") for i in range(4)]
        bats = B.build_batteries(book, "1342", [0] * 16, k=8, min_k=4)
        assert len(bats) == 2
        idx0 = {it["norm_index"] for it in bats[0]["items"]}
        idx1 = {it["norm_index"] for it in bats[1]["items"]}
        assert idx0.isdisjoint(idx1)
        # leak count attributed to the first battery only
        assert bats[1]["n_leak_skipped"] == 0

    def test_eligibility_filters(self):
        book = [
            _yes("keep-1"),
            _bn("permit", "permitted"),            # non-decisive → excluded
            _bn("nongov", "prohibited", governs=False),  # not governing → excluded
            _bn("nocontext", "prohibited", context=""),  # empty context → excluded
            _no("keep-2"),
            _yes("keep-3"),
            _yes("keep-4"),
        ]
        bats = B.build_batteries(book, "1342", [0] * len(book), k=8, min_k=4)
        assert len(bats) == 1
        subjects = {book[it["norm_index"]]["norm_subject"] for it in bats[0]["items"]}
        assert subjects == {"keep-1", "keep-2", "keep-3", "keep-4"}


# --------------------------------------------------------------------------- #
# build_batteries: leak exclusion + anti-leak canary                           #
# --------------------------------------------------------------------------- #

class TestLeakExclusion:
    def test_leaky_norm_skipped_and_counted(self):
        # 3 clean yes + 1 clean no + 1 leaky (force word in context) → battery of
        # 4 clean, leaky excluded and counted.
        book = [
            _yes("y0"), _yes("y1"), _yes("y2"),
            _no("n0"),
            _no("leaky", context="a prohibited exchange between families"),
        ]
        bats = B.build_batteries(book, "1342", [0] * 5, k=8, min_k=4)
        assert len(bats) == 1
        b = bats[0]
        assert b["n_leak_skipped"] == 1
        assert b["composition"]["n"] == 4
        subjects = {book[it["norm_index"]]["norm_subject"] for it in b["items"]}
        assert "leaky" not in subjects

    def test_leak_can_push_cluster_below_min_k(self):
        # 4 eligible but 1 leaks → 3 clean < min_k → no battery.
        book = [
            _yes("y0"), _yes("y1"), _yes("y2"),
            _no("leaky", context="a prohibited disclosure"),
        ]
        bats = B.build_batteries(book, "1342", [0] * 4, k=8, min_k=4)
        assert bats == []

    def test_no_force_word_in_any_scenario_text(self):
        book = [_yes(f"y{i}") for i in range(5)] + [_no(f"n{i}") for i in range(3)]
        bats = B.build_batteries(book, "1342", [0] * 8, k=8, min_k=4)
        assert bats
        for b in bats:
            for it in b["items"]:
                norm = book[it["norm_index"]]
                # scenario text must survive the same leak check as probes
                assert probes.probe_leaks(it["scenario_text"], norm) is False

    def test_force_options_allowed_in_instruction(self):
        book = [_yes(f"y{i}") for i in range(4)] + [_no(f"n{i}") for i in range(4)]
        bats = B.build_batteries(book, "1342", [0] * 8, k=8, min_k=4)
        prompt = bats[0]["prompt_text"]
        # the five force option words live in the instruction (allowed there)
        for word in ("obligatory", "recommended", "permitted", "discouraged", "prohibited"):
            assert word in prompt
        # every scenario is numbered
        for i in range(1, bats[0]["composition"]["n"] + 1):
            assert f"Scenario {i}:" in prompt


# --------------------------------------------------------------------------- #
# build_batteries: determinism                                                 #
# --------------------------------------------------------------------------- #

class TestDeterminism:
    def test_same_inputs_identical_batteries(self):
        book = [_yes(f"y{i}") for i in range(6)] + [_no(f"n{i}") for i in range(6)]
        cluster_ids = [0] * 12
        b1 = B.build_batteries(book, "1342", cluster_ids, k=8, min_k=4)
        b2 = B.build_batteries(book, "1342", cluster_ids, k=8, min_k=4)
        assert [x["battery_id"] for x in b1] == [x["battery_id"] for x in b2]
        assert [[it["norm_index"] for it in x["items"]] for x in b1] == \
               [[it["norm_index"] for it in x["items"]] for x in b2]

    def test_seed_varies_with_gutenberg_id(self):
        # Different book id → different RNG stream → generally different selection
        # of which minority/majority norms land in the battery (12 norms, k=8).
        book = [_yes(f"y{i}") for i in range(8)] + [_no(f"n{i}") for i in range(4)]
        b_a = B.build_batteries(book, "1342", [0] * 12, k=8, min_k=4)
        b_b = B.build_batteries(book, "9999", [0] * 12, k=8, min_k=4)
        sel_a = tuple(it["norm_index"] for it in b_a[0]["items"])
        sel_b = tuple(it["norm_index"] for it in b_b[0]["items"])
        # not asserting inequality strictly (could coincide), but ids must differ
        assert b_a[0]["battery_id"] != b_b[0]["battery_id"]
        assert b_a[0]["gutenberg_id"] == "1342"
        assert b_b[0]["gutenberg_id"] == "9999"
