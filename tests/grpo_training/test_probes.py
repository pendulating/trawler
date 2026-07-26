"""Tests for the m-series outcome-probe builder (`R-OUTCOME` core).

Covers the reward-outcome.md probe-generation contract:
  * flow_to_query parity with the legacy online_rground._flow_to_query;
  * probe determinism (same chunk_id ⇒ identical sample);
  * force stratification (mixed pool ⇒ both classes, >=1 gold-no);
  * K = min(4, |pool|);
  * union-dedupe across flows retrieving the same norm;
  * decisive-force + governs_info_flow filtering (permitted/unknown skipped);
  * leak canaries (no force word, no >=6-token articulation run);
  * apply_null_filter drop/keep semantics (missing key ⇒ keep);
  * gold matches FORCE_TO_GOLD for all four decisive forces.
"""

import pytest

from dagspaces.grpo_training.stages import probes as p
from dagspaces.grpo_training.stages.deontic import FORCE_TO_GOLD


# --------------------------------------------------------------------------- #
# fixtures / helpers                                                          #
# --------------------------------------------------------------------------- #

def _norm(
    subject="a family's patriarch",
    act="disclose his daughter's private circumstances to a suitor",
    condition="",
    context="marriage negotiations between families",
    force="prohibited",
    governs=True,
    articulation="",
    gutenberg_id="1342",
    **extra,
):
    n = {
        "norm_subject": subject,
        "norm_act": act,
        "condition_of_application": condition,
        "context": context,
        "normative_force": force,
        "governs_info_flow": governs,
        "norm_articulation": articulation,
        "gutenberg_id": gutenberg_id,
    }
    n.update(extra)
    return n


def _retriever(book_norms):
    """A retrieve_top_k that returns indices of norms whose context substring
    appears in the query, else the first k — deterministic, no embedding."""
    def retrieve(query, k):
        q = query.lower()
        hits = [i for i, n in enumerate(book_norms) if n.get("context", "").lower() and n["context"].lower() in q]
        if not hits:
            hits = list(range(len(book_norms)))
        return hits[:k]
    return retrieve


def _fixed_retriever(index_map):
    """retrieve_top_k that returns a fixed index list per query string."""
    def retrieve(query, k):
        return list(index_map.get(query, []))[:k]
    return retrieve


# --------------------------------------------------------------------------- #
# flow_to_query parity                                                         #
# --------------------------------------------------------------------------- #

class TestFlowToQueryParity:
    _FIXTURES = [
        {"sender": "Alice", "recipient": "Bob", "information_type": "health record",
         "context": "clinic", "transmission_principle": "consent", "subject": "patient",
         "norms_invoked": ["confidentiality", "HIPAA"]},
        {"sender": "", "recipient": "Bob"},
        {"norms_invoked": ["x", "y"]},
        {},  # falls back to "information flow"
        {"sender": "S", "norms_invoked": "not-a-list"},  # non-list ignored
        {"information_type": "diagnosis", "subject": "the daughter"},
    ]

    @pytest.mark.parametrize("flow", _FIXTURES)
    def test_matches_legacy(self, flow):
        from dagspaces.grpo_training.stages.online_rground import _flow_to_query
        assert p.flow_to_query(flow) == _flow_to_query(flow)

    def test_empty_flow_fallback(self):
        assert p.flow_to_query({}) == "information flow"

    def test_join_order(self):
        q = p.flow_to_query({"sender": "A", "recipient": "B", "information_type": "C",
                             "context": "D", "transmission_principle": "E", "subject": "F",
                             "norms_invoked": ["G"]})
        assert q == "A B C D E F G"


# --------------------------------------------------------------------------- #
# dedupe key / probe_id / decisive forces                                      #
# --------------------------------------------------------------------------- #

class TestIdentity:
    def test_decisive_forces_equal_force_to_gold_keys(self):
        assert p.DECISIVE_FORCES == set(FORCE_TO_GOLD)

    def test_dedupe_key_case_and_whitespace_insensitive(self):
        a = _norm(subject="  A Family's Patriarch ", force="Prohibited")
        b = _norm(subject="a family's patriarch", force="prohibited")
        assert p.norm_dedupe_key(a) == p.norm_dedupe_key(b)

    def test_dedupe_key_distinguishes_force(self):
        a = _norm(force="prohibited")
        b = _norm(force="obligatory")
        assert p.norm_dedupe_key(a) != p.norm_dedupe_key(b)

    def test_dedupe_key_reads_raz_force(self):
        n = _norm(force=None)
        n.pop("normative_force")
        n["raz_normative_force"] = "Discouraged"
        assert p.norm_dedupe_key(n)[-1] == "discouraged"

    def test_probe_id_deterministic_and_book_scoped(self):
        n = _norm()
        assert p.probe_id("1342", n) == p.probe_id("1342", n)
        assert p.probe_id("1342", n) != p.probe_id("11", n)


# --------------------------------------------------------------------------- #
# templating + leak canaries                                                   #
# --------------------------------------------------------------------------- #

class TestTemplateAndLeak:
    def test_template_has_question_suffix(self):
        text = p.template_probe(_norm())
        assert text.strip().endswith("Answer yes or no.")
        assert "should this information be shared" in text.lower()

    def test_template_omits_articulation(self):
        n = _norm(articulation="A patriarch must never reveal a daughter's secrets to suitors under any circumstance whatsoever.")
        text = p.template_probe(n)
        assert "must never reveal" not in text.lower()

    def test_clean_norm_does_not_leak(self):
        assert p.probe_leaks(p.template_probe(_norm()), _norm()) is False

    def test_force_word_in_field_is_a_leak(self):
        # A field carrying an explicit force word taints the prompt.
        n = _norm(context="a prohibited disclosure between families")
        assert p.probe_leaks(p.template_probe(n), n) is True

    @pytest.mark.parametrize("word", [
        "obligatory", "obligated", "obliged", "required", "mandatory",
        "prohibited", "prohibition", "forbidden", "banned", "impermissible",
        "permitted", "permissible", "allowed",
        "discouraged", "recommended", "advised", "encouraged",
    ])
    def test_force_words_detected(self, word):
        """Adjectival/participial forms state what force GOVERNS the act."""
        assert p.probe_leaks(f"Something {word} happened. Answer yes or no.", _norm()) is True

    @pytest.mark.parametrize("word", [
        "obligation", "obligations",     # "social obligations" — duties, 15 FPs
        "permission",                    # "ask permission" — an act, 33 FPs
        "recommendation", "recommendations",  # "letter of recommendation", 4 FPs
        "discourage", "recommend", "permit",  # verbs/nouns = acts, not forces
    ])
    def test_content_words_are_not_force_leaks(self, word):
        """Noun/verb forms are ordinary content. The old stem-prefix match
        dropped 59 legitimate scenarios on these (measured 2026-07-26)."""
        assert p.probe_leaks(f"Something about a {word} here. Answer yes or no.", _norm()) is False

    def test_articulation_six_token_run_is_a_leak(self):
        art = "the patriarch shall not disclose the daughter secrets to any suitor"
        # A prompt that contains a >=6-token verbatim run from the articulation.
        prompt = "In this scene the patriarch shall not disclose the daughter secrets openly."
        n = _norm(articulation=art)
        assert p.probe_leaks(prompt, n) is True

    def test_articulation_short_overlap_not_a_leak(self):
        art = "the patriarch shall stay silent"  # < 6 tokens
        prompt = "the patriarch shall stay silent about it. Answer yes or no."
        n = _norm(articulation=art, context="", subject="x", act="y")
        # articulation < 6 tokens ⇒ no n-gram leak; and no force word present.
        assert p.probe_leaks(prompt, n) is False

    def test_leaky_norm_excluded_from_pool(self):
        # A governing, decisive norm whose FIELDS contain a force word must be
        # dropped by build_probe_pool, not merely produce a clean prompt.
        leaky = _norm(context="a prohibited exchange", gutenberg_id="1342")
        clean = _norm(subject="a doctor", act="share records", context="a clinic visit",
                      force="obligatory", gutenberg_id="1342")
        book = [leaky, clean]
        pool, stats = p.build_probe_pool_with_stats(
            [{"context": "a prohibited exchange"}, {"context": "a clinic visit"}],
            book, _retriever(book), k=3,
        )
        assert all("prohibited" not in pr["prompt_text"].lower() for pr in pool)
        assert stats["n_leak_skipped"] == 1
        # The clean obligatory norm survives.
        assert any(pr["gold"] == "yes" for pr in pool)


# --------------------------------------------------------------------------- #
# build_probe_pool: dedupe, filtering, gold                                    #
# --------------------------------------------------------------------------- #

class TestBuildProbePool:
    def test_gold_matches_force_to_gold_all_four(self):
        book = [
            _norm(subject="s1", force="obligatory", context="c1"),
            _norm(subject="s2", force="recommended", context="c2"),
            _norm(subject="s3", force="prohibited", context="c3"),
            _norm(subject="s4", force="discouraged", context="c4"),
        ]
        # one flow per norm, each retrieving its own index
        flows = [{"context": f"c{i}"} for i in range(1, 5)]
        pool = p.build_probe_pool(flows, book, _retriever(book), k=1)
        by_force = {pr["norm"]["normative_force"]: pr["gold"] for pr in pool}
        for force in ("obligatory", "recommended", "prohibited", "discouraged"):
            assert by_force[force] == FORCE_TO_GOLD[force]

    def test_permitted_and_nongoverning_skipped(self):
        book = [
            _norm(subject="permit-norm", force="permitted", context="cp"),
            _norm(subject="unknown-norm", force="whatever", context="cu"),
            _norm(subject="nongov", force="prohibited", governs=False, context="cn"),
            _norm(subject="keep", force="prohibited", governs=True, context="ck"),
        ]
        flows = [{"context": c} for c in ("cp", "cu", "cn", "ck")]
        pool = p.build_probe_pool(flows, book, _retriever(book), k=1)
        subjects = {pr["norm"]["norm_subject"] for pr in pool}
        assert subjects == {"keep"}

    def test_dedupe_across_flows_retrieving_same_norm(self):
        book = [_norm(subject="dup", force="prohibited", context="cx")]
        # two different flows both retrieve index 0
        flows = [{"context": "cx one"}, {"context": "cx two"}]
        retr = _fixed_retriever({
            p.flow_to_query(flows[0]): [0],
            p.flow_to_query(flows[1]): [0],
        })
        pool = p.build_probe_pool(flows, book, retr, k=3)
        assert len(pool) == 1

    def test_norm_index_points_into_book_norms(self):
        book = [
            _norm(subject="s0", force="permitted", context="c0"),
            _norm(subject="s1", force="prohibited", context="c1"),
        ]
        flows = [{"context": "c1"}]
        pool = p.build_probe_pool(flows, book, _retriever(book), k=2)
        assert len(pool) == 1
        assert pool[0]["norm_index"] == 1
        assert book[pool[0]["norm_index"]]["norm_subject"] == "s1"

    def test_out_of_range_indices_ignored(self):
        book = [_norm(force="prohibited", context="c")]
        retr = _fixed_retriever({p.flow_to_query({"context": "c"}): [5, -1, 0]})
        pool = p.build_probe_pool([{"context": "c"}], book, retr, k=3)
        assert len(pool) == 1

    def test_deterministic_first_appearance_order(self):
        book = [
            _norm(subject=f"s{i}", force="prohibited", context=f"c{i}") for i in range(3)
        ]
        flows = [{"context": "q"}]
        retr = _fixed_retriever({p.flow_to_query({"context": "q"}): [2, 0, 1]})
        pool = p.build_probe_pool(flows, book, retr, k=3)
        assert [pr["norm_index"] for pr in pool] == [2, 0, 1]


# --------------------------------------------------------------------------- #
# sample_probes: determinism, stratification, K                                #
# --------------------------------------------------------------------------- #

def _pool(n_yes, n_no):
    pool = []
    for i in range(n_yes):
        pool.append({"probe_id": f"y{i}", "gold": "yes", "norm_index": i,
                     "norm": {}, "prompt_text": f"yes {i}"})
    for j in range(n_no):
        pool.append({"probe_id": f"n{j}", "gold": "no", "norm_index": 100 + j,
                     "norm": {}, "prompt_text": f"no {j}"})
    return pool


class TestSampleProbes:
    def test_k_is_min_kmax_pool(self):
        assert len(p.sample_probes(_pool(2, 1), "chunk-a", k_max=4)) == 3
        assert len(p.sample_probes(_pool(5, 5), "chunk-a", k_max=4)) == 4
        assert len(p.sample_probes([], "chunk-a")) == 0

    def test_same_chunk_id_identical_sample(self):
        pool = _pool(5, 5)
        s1 = p.sample_probes(pool, "chunk-42", k_max=4)
        s2 = p.sample_probes(pool, "chunk-42", k_max=4)
        assert [x["probe_id"] for x in s1] == [x["probe_id"] for x in s2]

    def test_different_chunk_ids_generally_differ(self):
        pool = _pool(6, 6)
        samples = {
            tuple(sorted(x["probe_id"] for x in p.sample_probes(pool, f"chunk-{i}", k_max=4)))
            for i in range(12)
        }
        # With a 12-item pool and K=4, distinct chunk_ids should not all collapse
        # to one sample.
        assert len(samples) > 1

    def test_mixed_pool_contains_both_classes(self):
        pool = _pool(5, 5)
        for i in range(20):
            golds = {x["gold"] for x in p.sample_probes(pool, f"c{i}", k_max=4)}
            assert golds == {"yes", "no"}, f"chunk c{i} missed a class: {golds}"

    def test_at_least_one_gold_no_when_present(self):
        # Skewed pool: only one gold-no among many gold-yes.
        pool = _pool(10, 1)
        for i in range(30):
            sample = p.sample_probes(pool, f"c{i}", k_max=4)
            assert any(x["gold"] == "no" for x in sample), f"chunk c{i} dropped the gold-no"

    def test_single_class_pool_ok(self):
        pool = _pool(6, 0)
        sample = p.sample_probes(pool, "c", k_max=4)
        assert len(sample) == 4
        assert all(x["gold"] == "yes" for x in sample)

    def test_output_in_pool_order(self):
        pool = _pool(5, 5)
        order = {pr["probe_id"]: i for i, pr in enumerate(pool)}
        sample = p.sample_probes(pool, "chunk-xyz", k_max=4)
        idxs = [order[x["probe_id"]] for x in sample]
        assert idxs == sorted(idxs)

    def test_no_duplicates_in_sample(self):
        pool = _pool(4, 4)
        sample = p.sample_probes(pool, "chunk-dup", k_max=4)
        ids = [x["probe_id"] for x in sample]
        assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# apply_null_filter                                                            #
# --------------------------------------------------------------------------- #

class TestNullFilter:
    def test_drops_at_or_above_threshold(self):
        pool = _pool(3, 0)
        frac = {"y0": 0.9, "y1": 0.8, "y2": 0.5}
        kept = p.apply_null_filter(pool, frac, p_null=0.8)
        ids = {x["probe_id"] for x in kept}
        assert ids == {"y2"}  # y0, y1 dropped (>= 0.8)

    def test_missing_key_is_kept(self):
        pool = _pool(2, 0)
        kept = p.apply_null_filter(pool, {}, p_null=0.8)
        assert {x["probe_id"] for x in kept} == {"y0", "y1"}

    def test_below_threshold_kept(self):
        pool = _pool(1, 1)
        kept = p.apply_null_filter(pool, {"y0": 0.79, "n0": 0.0}, p_null=0.8)
        assert len(kept) == 2

    def test_threshold_boundary_is_drop(self):
        pool = _pool(1, 0)
        assert p.apply_null_filter(pool, {"y0": 0.8}, p_null=0.8) == []


# ===========================================================================
# v2 — flow-anchored probes (2026-07-25)
# ===========================================================================
from dagspaces.grpo_training.stages.probes import (  # noqa: E402
    build_flow_probe_pool,
    flow_probe_leaks,
    sample_flow_probes,
    template_flow_probe,
)


def _flow(**over):
    f = {"sender": "Dr. Bob", "recipient": "the insurer", "subject": "Alice",
         "information_type": "HIV status", "transmission_principle": "without consent",
         "context": "medical treatment"}
    f.update(over)
    return f


def _n(force="prohibited", **over):
    n = {"norm_subject": "a physician", "norm_act": "disclose a diagnosis to an insurer",
         "context": "medical treatment", "normative_force": force,
         "governs_info_flow": True, "gutenberg_id": "1342",
         "norm_articulation": "A physician must not disclose."}
    n.update(over)
    return n


class TestTemplateFlowProbe:
    def test_contains_context_and_referring_expression(self):
        t = template_flow_probe(_flow())
        assert "medical treatment" in t and "HIV status" in t
        assert t.endswith("Is this flow appropriate? Answer yes or no.")

    def test_withholds_judgment_relevant_fields(self):
        # sender / recipient / transmission_principle must come from the
        # EXTRACTION, not the probe — else the probe is self-answering.
        t = template_flow_probe(_flow())
        for withheld in ("Dr. Bob", "the insurer", "without consent"):
            assert withheld not in t

    def test_context_is_mandatory(self):
        # Without a context the appropriateness question is ill-posed, not hard.
        assert template_flow_probe(_flow(context="")) == ""

    def test_falls_back_to_subject_when_no_information_type(self):
        assert "Alice" in template_flow_probe(_flow(information_type=""))

    def test_flow_probe_leaks_detects_withheld_field(self):
        assert flow_probe_leaks("… the insurer … appropriate?", _flow())
        assert not flow_probe_leaks(template_flow_probe(_flow()), _flow())


class TestBuildFlowProbePool:
    def _retriever(self, idx):
        return lambda q, k: [idx]

    def test_gold_uses_polarity_corrected_appropriateness(self):
        norms = [_n(force="obligatory", norm_act="refrain from disclosing")]
        # performing => obligatory means "do it" => appropriate => yes
        pool, _ = build_flow_probe_pool([_flow()], norms, self._retriever(0),
                                        polarity_lookup={"0": "performing"})
        assert pool[0]["gold"] == "yes"
        # refraining => obligation to refrain => the FLOW is inappropriate => no
        pool, _ = build_flow_probe_pool([_flow()], norms, self._retriever(0),
                                        polarity_lookup={"0": "refraining"})
        assert pool[0]["gold"] == "no"
        assert pool[0]["appropriateness"] == "inappropriate"

    def test_one_probe_per_flow_anchored_to_top1(self):
        norms = [_n(), _n(force="obligatory")]
        pool, _ = build_flow_probe_pool(
            [_flow(), _flow(information_type="billing records")],
            norms, lambda q, k: [0], polarity_lookup={})
        assert len(pool) == 2
        assert all(p["norm_index"] == 0 for p in pool)

    def test_permitted_is_scorable_as_appropriate(self):
        # Decision 2026-07-25: a permission is not a violation, so a permitted
        # flow is APPROPRIATE and probe-eligible (it used to be excluded).
        pool, _ = build_flow_probe_pool([_flow()], [_n(force="permitted")],
                                        self._retriever(0))
        assert len(pool) == 1
        assert pool[0]["gold"] == "yes"
        assert pool[0]["appropriateness"] == "appropriate"

    def test_unknown_force_excluded(self):
        pool, stats = build_flow_probe_pool([_flow()], [_n(force="")],
                                            self._retriever(0))
        assert pool == [] and stats["n_non_directional"] == 1

    def test_ill_posed_flow_excluded(self):
        pool, stats = build_flow_probe_pool([_flow(context="")], [_n()],
                                            self._retriever(0))
        assert pool == [] and stats["n_ill_posed"] == 1

    def test_non_governing_norm_excluded(self):
        pool, stats = build_flow_probe_pool(
            [_flow()], [_n(governs_info_flow=False)], self._retriever(0))
        assert pool == [] and stats["n_no_norm"] == 1

    def test_duplicate_probe_text_deduped(self):
        pool, _ = build_flow_probe_pool([_flow(), _flow()], [_n()],
                                        self._retriever(0))
        assert len(pool) == 1


class TestSampleFlowProbes:
    def _pool(self, n_yes, n_no):
        p = [{"gold": "yes", "prompt_text": f"y{i}"} for i in range(n_yes)]
        p += [{"gold": "no", "prompt_text": f"n{i}"} for i in range(n_no)]
        return p

    def test_minority_class_always_represented(self):
        # 1 minority among 20 — a uniform sample would usually miss it.
        s = sample_flow_probes(self._pool(20, 1), "c1", k_max=4)
        assert any(p["gold"] == "no" for p in s)
        assert any(p["gold"] == "yes" for p in s)

    def test_minority_is_adaptive_not_hardcoded_to_no(self):
        # When 'yes' is the rarer class it is the one reserved — nothing
        # assumes a global skew.
        s = sample_flow_probes(self._pool(1, 20), "c1", k_max=4)
        assert any(p["gold"] == "yes" for p in s)

    def test_k_is_min_of_kmax_and_pool(self):
        assert len(sample_flow_probes(self._pool(1, 1), "c1", k_max=4)) == 2
        assert len(sample_flow_probes(self._pool(10, 10), "c1", k_max=4)) == 4

    def test_deterministic_by_chunk_id(self):
        pool = self._pool(6, 6)
        a = sample_flow_probes(pool, "chunk-A", k_max=4)
        b = sample_flow_probes(pool, "chunk-A", k_max=4)
        assert [p["prompt_text"] for p in a] == [p["prompt_text"] for p in b]

    def test_empty_pool(self):
        assert sample_flow_probes([], "c1") == []
