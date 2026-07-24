"""Tests for `ModularReward` — the m-series modular reward stack (redesign item 4).

Mocks the frozen answerer (no live server) and the listwise-judge auxiliaries
(injected callables), so every test exercises the reward *core* — gate, routing
table, outcome EM, the 2:1 weight rule, group-neutral failure fallback,
per-module W&B metrics — with no GPU and no network.

Covers wiki/grpo_redesign/{reward-valid,reward-abstain,reward-outcome,
reward-ground,reward-contrast,ablation-protocol,migration}.md and the frozen
wave-1 contracts (probes / deontic_distance / answerer_client) it composes.
"""

from __future__ import annotations

import json

import pytest

from dagspaces.grpo_training.stages.modular_reward import (
    ModularReward,
    attach_probes,
    compute_module_weights,
    is_modular_composition,
    valid_gate,
)


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------
class FakeAnswerer:
    """Frozen-answerer stub: returns a constant answer per probe slot, or fails.

    ``answer`` is the token returned for every probe (so a caller can drive
    EM = 1 by making the probe golds match it, or EM = 0 by mismatching).
    ``fail`` forces the retry-exhausted failure path. Exposes the same
    ``answer_probes`` shape as :class:`AnswererClient`; ``em`` is the real
    static method (imported from the frozen client) via ModularReward.
    """

    def __init__(self, answer: str = "yes", fail: bool = False):
        self.answer = answer
        self.fail = fail
        self.calls = 0

    def answer_probes(self, extraction_flows, probes):
        self.calls += 1
        if self.fail:
            return {
                "answers": ["cannot_determine"] * len(probes),
                "failed": True,
                "raw": "",
            }
        return {"answers": [self.answer] * len(probes), "failed": False, "raw": ""}


class FlakyAnswerer(FakeAnswerer):
    """Succeeds on every call except the ``fail_on`` (1-indexed) one."""

    def __init__(self, fail_on: int = 2, answer: str = "yes"):
        super().__init__(answer=answer)
        self.fail_on = fail_on

    def answer_probes(self, extraction_flows, probes):
        self.calls += 1
        if self.calls == self.fail_on:
            return {
                "answers": ["cannot_determine"] * len(probes),
                "failed": True,
                "raw": "",
            }
        return {"answers": [self.answer] * len(probes), "failed": False, "raw": ""}


def const_aux(value: float):
    def _scorer(*, completions, prompts, metadata_list):
        return [value] * len(completions)

    return _scorer


# ---------------------------------------------------------------------------
# Completion / metadata fixtures
# ---------------------------------------------------------------------------
def extraction(sender="alice", **over) -> str:
    flow = {
        "sender": sender,
        "recipient": "bob",
        "subject": "carol",
        "information_type": "medical history",
        "transmission_principle": "confidentiality",
    }
    flow.update(over)
    return json.dumps(
        {"reasoning": "a flow is present", "has_information_exchange": True, "flows": [flow]}
    )


def no_flow() -> str:
    return json.dumps(
        {"reasoning": "nothing here", "has_information_exchange": False, "flows": []}
    )


def probe(gold: str, pid: str) -> dict:
    return {"probe_id": pid, "gold": gold, "prompt_text": f"Q {pid}? Answer yes or no."}


def extract_meta(gold, probes=None) -> dict:
    return {
        "task_type": "extract",
        "gold_has_exchange": gold,
        "probes": probes if probes is not None else [probe("yes", "p1")],
        "chunk_id": "c1",
        "source_id": "1342",
    }


# ---------------------------------------------------------------------------
# Weights (incl. reward_core=false)
# ---------------------------------------------------------------------------
def test_weight_table_all_rows():
    assert compute_module_weights(["ground", "contrast"], True) == pytest.approx(
        {"outcome": 0.5, "ground": 0.25, "contrast": 0.25}
    )
    assert compute_module_weights(["contrast"], True) == pytest.approx(
        {"outcome": 2 / 3, "contrast": 1 / 3}
    )
    assert compute_module_weights(["ground"], True) == pytest.approx(
        {"outcome": 2 / 3, "ground": 1 / 3}
    )
    assert compute_module_weights([], True) == pytest.approx({"outcome": 1.0})
    # -outcome cell: reward_core=False splits weight equally among auxiliaries.
    assert compute_module_weights(["ground", "contrast"], False) == pytest.approx(
        {"ground": 0.5, "contrast": 0.5}
    )
    # order of auxiliaries is irrelevant (set semantics)
    assert compute_module_weights(["contrast", "ground"], True) == pytest.approx(
        {"outcome": 0.5, "ground": 0.25, "contrast": 0.25}
    )


# ---------------------------------------------------------------------------
# R-VALID gate: each criterion fails independently
# ---------------------------------------------------------------------------
def test_gate_passes_valid_extraction():
    g = valid_gate(extraction())
    assert g.passed and g.reason is None and g.no_flow is False and len(g.flows) == 1


def test_gate_passes_valid_no_flow():
    g = valid_gate(no_flow())
    assert g.passed and g.no_flow is True and g.reason is None


def test_gate_fail_parse():
    g = valid_gate("this is not json at all")
    assert not g.passed and g.reason == "parse" and g.parsed is None


def test_gate_fail_schema_missing_key():
    # missing has_information_exchange
    txt = json.dumps({"reasoning": "x", "flows": []})
    g = valid_gate(txt)
    assert not g.passed and g.reason == "schema"


def test_gate_fail_schema_wrong_type():
    # flows is not a list
    txt = json.dumps({"reasoning": "x", "has_information_exchange": False, "flows": "none"})
    g = valid_gate(txt)
    assert not g.passed and g.reason == "schema"


def test_gate_fail_consistency_flag_true_no_flows():
    txt = json.dumps({"reasoning": "x", "has_information_exchange": True, "flows": []})
    g = valid_gate(txt)
    assert not g.passed and g.reason == "consistency"


def test_gate_fail_consistency_flag_false_with_flows():
    flow = json.loads(extraction())["flows"][0]
    txt = json.dumps({"reasoning": "x", "has_information_exchange": False, "flows": [flow]})
    g = valid_gate(txt)
    assert not g.passed and g.reason == "consistency"


def test_gate_fail_core_fields_missing():
    txt = json.dumps(
        {"reasoning": "x", "has_information_exchange": True, "flows": [{"sender": "a"}]}
    )
    g = valid_gate(txt)
    assert not g.passed and g.reason == "core_fields"


def test_gate_fail_core_fields_empty_string():
    flow = json.loads(extraction())["flows"][0]
    flow["recipient"] = "   "
    txt = json.dumps({"reasoning": "x", "has_information_exchange": True, "flows": [flow]})
    g = valid_gate(txt)
    assert not g.passed and g.reason == "core_fields"


def test_gate_fail_field_cap_including_context():
    # context is now an answerer-visible field (2026-07-24 whitelist) so the cap
    # must cover it: a 65-token context blows the field cap.
    flow = json.loads(extraction())["flows"][0]
    flow["context"] = " ".join(["word"] * 65)
    txt = json.dumps({"reasoning": "x", "has_information_exchange": True, "flows": [flow]})
    g = valid_gate(txt, field_cap_tokens=64)
    assert not g.passed and g.reason == "field_cap"


def test_gate_strips_think_block():
    g = valid_gate("<think>scratch {not json}</think>" + extraction())
    assert g.passed


# ---------------------------------------------------------------------------
# A-ABSTAIN routing table (all rows, incl. unknown gold)
# ---------------------------------------------------------------------------
def _reward(auxiliaries=(), reward_core=True, answerer=None, **kw):
    return ModularReward(
        auxiliaries=auxiliaries,
        reward_core=reward_core,
        answerer=answerer or FakeAnswerer(answer="yes"),
        abstain={"wrong": 0.1, "correct": 0.6, "unknown": 0.4},
        **kw,
    )


def test_routing_table_all_rows():
    prompts = ["yn", "yy", "nn", "ne", "un", "ue"]
    completions = [
        no_flow(),      # gold YES + no-flow  → 0.1 (wrong abstention)
        extraction(),   # gold YES + extract  → normal path (>0)
        no_flow(),      # gold NO  + no-flow  → 0.6 (correct abstention)
        extraction(),   # gold NO  + extract  → 0.4 (unverifiable, no calls)
        no_flow(),      # unknown  + no-flow  → 0.4
        extraction(),   # unknown  + extract  → 0.4 (no calls)
    ]
    meta = {
        "yn": extract_meta(True),
        "yy": extract_meta(True),
        "nn": extract_meta(False),
        "ne": extract_meta(False),
        "un": extract_meta(None),
        "ue": extract_meta(None),
    }
    r = _reward(reward_core=True)  # core cell: normal path score == EM
    r.prompt_metadata = meta
    scores = r(prompts=prompts, completions=completions)
    assert scores[0] == pytest.approx(0.1)   # wrong abstention
    assert scores[1] == pytest.approx(1.0)   # normal path, EM = 1
    assert scores[2] == pytest.approx(0.6)   # correct abstention
    assert scores[3] == pytest.approx(0.4)   # gold-NO extraction (neutral)
    assert scores[4] == pytest.approx(0.4)   # unknown + no-flow
    assert scores[5] == pytest.approx(0.4)   # unknown + extraction


def test_gate_fail_scores_zero_beneath_table():
    r = _reward(reward_core=True)
    r.prompt_metadata = {"p": extract_meta(True)}
    scores = r(prompts=["p"], completions=["not json"])
    assert scores[0] == 0.0  # invalid < wrong-but-valid (0.1)


def test_goldno_extraction_makes_no_server_calls():
    ans = FakeAnswerer(answer="yes")
    r = _reward(reward_core=True, answerer=ans)
    r.prompt_metadata = {"a": extract_meta(False), "b": extract_meta(None)}
    r(prompts=["a", "b"], completions=[extraction(), extraction()])
    assert ans.calls == 0  # gold-NO / unknown extractions never reach the answerer


# ---------------------------------------------------------------------------
# Vignette vs extract routing
# ---------------------------------------------------------------------------
def test_vignette_routing_scored_by_battery():
    gold_items = [
        {"gold_force": "prohibited", "articulation": "must not disclose the secret"},
        {"gold_force": "obligatory", "articulation": "must report the hazard"},
    ]
    completion = json.dumps(
        {
            "items": [
                {"id": 1, "force": "prohibited", "reasoning": "r", "governing_norm": "must not disclose the secret"},
                {"id": 2, "force": "obligatory", "reasoning": "r", "governing_norm": "must report the hazard"},
            ]
        }
    )
    r = _reward()
    r.prompt_metadata = {"v": {"task_type": "vignette", "gold_items": gold_items}}
    scores = r(prompts=["v"], completions=[completion])
    # Both forces exact → battery 1.0; cite has real overlap → r_vig in (0.7, 1].
    assert scores[0] > 0.7
    assert "vignette/antithesis_frac" in r.last_metrics
    assert r.last_metrics["vignette/antithesis_frac"] == pytest.approx(0.0)


def test_vignette_and_extract_in_one_batch_route_separately():
    gold_items = [{"gold_force": "prohibited", "articulation": "keep it private"}]
    vig = json.dumps({"items": [{"id": 1, "force": "prohibited", "governing_norm": "keep it private"}]})
    r = _reward(reward_core=True)
    r.prompt_metadata = {
        "v": {"task_type": "vignette", "gold_items": gold_items},
        "e": extract_meta(True),
    }
    scores = r(prompts=["v", "e"], completions=[vig, extraction()])
    assert scores[0] > 0.0  # vignette battery
    assert scores[1] == pytest.approx(1.0)  # extraction normal path (EM=1)


# ---------------------------------------------------------------------------
# R-OUTCOME + weights on the normal path
# ---------------------------------------------------------------------------
def test_full_cell_combines_outcome_and_auxiliaries():
    r = ModularReward(
        auxiliaries=["ground", "contrast"],
        reward_core=True,
        answerer=FakeAnswerer(answer="yes"),
        ground_scorer=const_aux(0.8),
        contrast_scorer=const_aux(0.4),
        abstain={"wrong": 0.1, "correct": 0.6, "unknown": 0.4},
    )
    r.prompt_metadata = {"p": extract_meta(True, probes=[probe("yes", "p1"), probe("no", "p2")])}
    # answerer says "yes" to both; gold = [yes, no] → EM = 0.5.
    scores = r(prompts=["p"], completions=[extraction()])
    expected = 0.5 * 0.5 + 0.25 * 0.8 + 0.25 * 0.4
    assert scores[0] == pytest.approx(expected)


def test_minus_outcome_cell_skips_answerer():
    ans = FakeAnswerer(answer="yes")
    r = ModularReward(
        auxiliaries=["ground", "contrast"],
        reward_core=False,  # -outcome
        answerer=ans,
        ground_scorer=const_aux(0.8),
        contrast_scorer=const_aux(0.4),
        abstain={"wrong": 0.1, "correct": 0.6, "unknown": 0.4},
    )
    r.prompt_metadata = {"p": extract_meta(True)}
    scores = r(prompts=["p"], completions=[extraction()])
    assert ans.calls == 0  # no outcome term ⇒ answerer never called
    assert scores[0] == pytest.approx(0.5 * 0.8 + 0.5 * 0.4)


# ---------------------------------------------------------------------------
# Group-neutral answerer-failure fallback
# ---------------------------------------------------------------------------
def test_group_neutral_answerer_failure():
    # Two gold-YES extraction completions share one prompt (one group). The
    # answerer fails on the 2nd call → the WHOLE group gets outcome 0.5.
    r = _reward(reward_core=True, answerer=FlakyAnswerer(fail_on=2))
    r.prompt_metadata = {"g": extract_meta(True)}
    scores = r(prompts=["g", "g"], completions=[extraction("a"), extraction("b")])
    assert scores[0] == pytest.approx(0.5)  # core cell → 1.0 * 0.5
    assert scores[1] == pytest.approx(0.5)
    assert r.last_metrics["reward/outcome/answerer_failed_frac"] == pytest.approx(1.0)


def test_separate_groups_isolate_failure():
    # Two gold-YES completions on DIFFERENT prompts are separate groups; a
    # failure in one does not poison the other.
    r = _reward(reward_core=True, answerer=FlakyAnswerer(fail_on=2))
    r.prompt_metadata = {"g1": extract_meta(True), "g2": extract_meta(True)}
    scores = r(prompts=["g1", "g2"], completions=[extraction("a"), extraction("b")])
    assert scores[0] == pytest.approx(1.0)  # g1 group succeeded (EM=1)
    assert scores[1] == pytest.approx(0.5)  # g2 group failed → neutral


# ---------------------------------------------------------------------------
# Probe attachment determinism (dataset-build hook)
# ---------------------------------------------------------------------------
def test_attach_probes_determinism():
    pool = [probe("yes", f"p{i}") for i in range(6)] + [probe("no", "pn")]
    a = attach_probes(pool, "chunk-42", k_max=4)
    b = attach_probes(pool, "chunk-42", k_max=4)
    assert [p["probe_id"] for p in a] == [p["probe_id"] for p in b]
    assert len(a) == 4
    # A gold-no probe is always reserved (Forbid-recall carrier).
    assert any(p["gold"] == "no" for p in a)


def test_attach_probes_null_filter_applied():
    pool = [probe("yes", "keep"), probe("yes", "drop")]
    # drop is answerable from an empty extraction (frac >= p_null) → filtered out.
    out = attach_probes(
        pool, "c1", k_max=4, null_correct_frac={"drop": 0.9, "keep": 0.1}, p_null=0.8
    )
    ids = {p["probe_id"] for p in out}
    assert "keep" in ids and "drop" not in ids


def test_attach_probes_different_chunk_can_differ():
    pool = [probe("yes", f"p{i}") for i in range(8)]
    a = [p["probe_id"] for p in attach_probes(pool, "chunkA", k_max=4)]
    b = [p["probe_id"] for p in attach_probes(pool, "chunkB", k_max=4)]
    # Determinism per chunk; the seed differs so at least the selection may
    # differ — but both are valid length-4 subsets of the pool.
    assert len(a) == 4 and len(b) == 4
    assert set(a) <= {f"p{i}" for i in range(8)}


# ---------------------------------------------------------------------------
# W&B metric dict shape
# ---------------------------------------------------------------------------
def test_wandb_metric_dict_shape():
    gold_items = [{"gold_force": "prohibited", "articulation": "keep private"}]
    vig = json.dumps({"items": [{"id": 1, "force": "permitted", "governing_norm": "x"}]})
    r = _reward(reward_core=True, answerer=FakeAnswerer(answer="yes"))
    r.prompt_metadata = {
        "yy": extract_meta(True, probes=[probe("yes", "a"), probe("no", "b")]),
        "yn": extract_meta(True),  # will be answered too
        "nn": extract_meta(False),
        "bad": extract_meta(True),
        "v": {"task_type": "vignette", "gold_items": gold_items},
    }
    completions = [extraction(), no_flow(), no_flow(), "garbage", vig]
    r(prompts=["yy", "yn", "nn", "bad", "v"], completions=completions)
    m = r.last_metrics
    # outcome namespace
    assert "reward/outcome/em_mean" in m
    assert "reward/outcome/em_mean_by_force/yes" in m
    assert "reward/outcome/em_mean_by_force/no" in m
    assert "reward/outcome/cannot_determine_frac" in m
    assert "reward/outcome/answerer_failed_frac" in m
    assert "reward/outcome/group_spread" in m
    # vignette namespace
    assert "vignette/antithesis_frac" in m
    assert "vignette/hedge_frac" in m
    # abstain + valid + diag namespaces
    assert "reward/valid/gate_frac" in m
    assert any(k.startswith("abstain/") for k in m)
    assert "diag/direction_consistency" in m
    # gate_frac reflects the one garbage completion out of four extract rows
    assert m["reward/valid/gate_frac"] == pytest.approx(0.75)


def test_metrics_by_force_values():
    # answerer says "yes"; probe golds one yes one no → yes-force EM 1, no 0.
    r = _reward(reward_core=True, answerer=FakeAnswerer(answer="yes"))
    r.prompt_metadata = {"p": extract_meta(True, probes=[probe("yes", "a"), probe("no", "b")])}
    r(prompts=["p"], completions=[extraction()])
    m = r.last_metrics
    assert m["reward/outcome/em_mean_by_force/yes"] == pytest.approx(1.0)
    assert m["reward/outcome/em_mean_by_force/no"] == pytest.approx(0.0)
    assert m["reward/outcome/em_mean"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Config dispatch: directional (keeper) never selects the modular path
# ---------------------------------------------------------------------------
def test_is_modular_composition_dispatch():
    assert is_modular_composition("modular") is True
    assert is_modular_composition("MODULAR") is True
    assert is_modular_composition("directional") is False  # the keeper path
    assert is_modular_composition("additive") is False
    assert is_modular_composition("gated") is False
    assert is_modular_composition(None) is False


def test_grpo_training_uses_dispatch_helper():
    # The keeper guard: grpo_training must route through is_modular_composition,
    # so a non-modular composition (directional/additive) keeps selecting
    # CompositeRewardFunction.
    import inspect

    from dagspaces.grpo_training.stages import grpo_training

    src = inspect.getsource(grpo_training)
    assert "is_modular_composition" in src
    assert "make_modular_reward_from_cfg" in src
    # The legacy branch is still present and unconditional-per-else.
    assert "CompositeRewardFunction(" in src


def test_active_aux_without_scorer_raises():
    # A live-launch safety: an active auxiliary with no injected scorer must
    # fail loudly on the normal path rather than silently score neutral.
    r = ModularReward(
        auxiliaries=["ground"],
        reward_core=True,
        answerer=FakeAnswerer(answer="yes"),
        abstain={"wrong": 0.1, "correct": 0.6, "unknown": 0.4},
    )
    r.prompt_metadata = {"p": extract_meta(True)}
    with pytest.raises(RuntimeError):
        r(prompts=["p"], completions=[extraction()])


# ---------------------------------------------------------------------------
# build_modular_dataset: gold-NO chunks are A-ABSTAIN rows, never excluded
# (regression: 2026-07-24 wave-1 built 0 gold-no rows — the empty-pool
# exclusion ate all 126 gold-NO chunks and collapsed the two-sided
# abstention signal)
# ---------------------------------------------------------------------------
class _StubTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False):
        return f"<user>{messages[0]['content']}</user>"


def _chunks_df():
    import pandas as pd
    return pd.DataFrame([
        {"chunk_text": "Alice tells Bob a secret.", "source_id": "135",
         "chunk_id": "c-yes-pool", "has_information_exchange": True},
        {"chunk_text": "A flow the retriever found no norms for.",
         "source_id": "135", "chunk_id": "c-yes-nopool",
         "has_information_exchange": True},
        {"chunk_text": "A quiet essayistic passage.", "source_id": "135",
         "chunk_id": "c-goldno", "has_information_exchange": False},
    ])


def _pools_parquet(tmp_path):
    import pandas as pd
    pq = tmp_path / "pools.parquet"
    pd.DataFrame([
        {"gutenberg_id": "135", "chunk_id": "c-yes-pool", "probe_id": "p1",
         "gold": "yes", "prompt_text": "Should this be shared?",
         "norm_index": 0},
    ]).to_parquet(pq)
    return str(pq)


def test_goldno_chunks_kept_as_probeless_abstain_rows(tmp_path):
    from dagspaces.grpo_training.stages.modular_reward import build_modular_dataset

    grpo_cfg = {
        "probes": {"pools_path": _pools_parquet(tmp_path), "k_max": 4},
        "task_mix": {"extract": 1.0, "vignette": 0.0},
        "prescreen": {"target_n": 10},
        "battery": {},
    }
    rf = _reward()
    dataset, metadata = build_modular_dataset(
        cfg=None, grpo_cfg=grpo_cfg, chunks_df=_chunks_df(),
        norm_universes={}, reward_fn=rf, tokenizer=_StubTokenizer(),
        ci_prompt_template="Extract flows: {{chunk_text}}",
        output_dir=str(tmp_path), seed=0, embed_fn=None,
    )

    rows = list(dataset)
    gold_classes = sorted(str(r["gold_has_exchange"]) for r in rows)
    # gold-NO kept; flow-bearing empty-pool excluded; with-pool kept.
    assert len(rows) == 2, rows
    assert gold_classes == ["False", "True"], rows

    # The gold-NO row carries NO probes (A-ABSTAIN needs no server calls).
    goldno_meta = [m for m in rf.prompt_metadata.values()
                   if m.get("gold_has_exchange") is False]
    assert len(goldno_meta) == 1
    assert goldno_meta[0].get("probes") in ([], None)

    # Exclusion accounting is split, not conflated.
    assert metadata["n_extract_excluded_empty_pool"] == 1
    assert metadata["n_src_goldno_chunks"] == 1
    assert metadata["n_goldno_rows_prescreen_pool"] == 1


# ---------------------------------------------------------------------------
# Ordering-invariant floor (2026-07-24 review A): a valid gold-YES extraction
# never scores at/below the invalid gate (0) or loses to a wrong abstention
# (0.1). VALID_PATH_FLOOR sits in (0.1, 0.4).
# ---------------------------------------------------------------------------
from dagspaces.grpo_training.stages.modular_reward import (  # noqa: E402
    VALID_PATH_FLOOR,
    _is_embedding_abort,
)


def test_valid_extraction_all_wrong_probes_floored_above_abstention():
    # core cell, answerer says "yes" but gold is "no" ⇒ EM = 0 for the extraction.
    prompts = ["ext", "abs"]
    completions = [extraction(), no_flow()]
    meta = {
        "ext": extract_meta(True, probes=[probe("no", "p1")]),  # gold-YES, EM=0
        "abs": extract_meta(True),                              # gold-YES no-flow → 0.1
    }
    r = _reward(reward_core=True, answerer=FakeAnswerer(answer="yes"))
    r.prompt_metadata = meta
    scores = r(prompts=prompts, completions=completions)
    assert scores[0] == pytest.approx(VALID_PATH_FLOOR)  # not 0.0
    assert 0.1 < VALID_PATH_FLOOR < 0.4
    assert scores[0] > scores[1]  # engaging beats wrong-abstaining
    # gate failure still scores strictly below the floor.
    assert r(prompts=["bad"], completions=["not json"],
             )[0] == pytest.approx(0.0)


def test_valid_floor_does_not_depress_good_extractions():
    # EM = 1 must stay 1.0 (floor only binds when composite < floor).
    r = _reward(reward_core=True, answerer=FakeAnswerer(answer="yes"))
    r.prompt_metadata = {"e": extract_meta(True, probes=[probe("yes", "p1")])}
    assert r(prompts=["e"], completions=[extraction()])[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Embedding fail-loud abort propagates; ordinary aux error neutralizes
# (2026-07-24 review C).
# ---------------------------------------------------------------------------
def _raising_aux(exc):
    def _scorer(*, completions, prompts, metadata_list):
        raise exc
    return _scorer


def test_embedding_abort_propagates_not_neutralized():
    abort = RuntimeError(
        "[EmbeddingClient] 3 consecutive ... aborting instead of training on zeroed R_ground"
    )
    r = _reward(auxiliaries=("ground",), answerer=FakeAnswerer(answer="yes"),
                ground_scorer=_raising_aux(abort))
    r.prompt_metadata = {"e": extract_meta(True, probes=[probe("yes", "p1")])}
    with pytest.raises(RuntimeError, match="aborting instead of training"):
        r(prompts=["e"], completions=[extraction()])


def test_ordinary_aux_error_still_group_neutral():
    r = _reward(auxiliaries=("ground",), answerer=FakeAnswerer(answer="yes"),
                ground_scorer=_raising_aux(ValueError("transient judge hiccup")))
    r.prompt_metadata = {"e": extract_meta(True, probes=[probe("yes", "p1")])}
    scores = r(prompts=["e"], completions=[extraction()])  # must not raise
    assert len(scores) == 1


def test_is_embedding_abort_predicate():
    assert _is_embedding_abort(RuntimeError("... aborting instead of training ..."))
    assert not _is_embedding_abort(RuntimeError("some other runtime error"))
    assert not _is_embedding_abort(ValueError("aborting instead of training"))


# ---------------------------------------------------------------------------
# Symmetric probe-survival guard + chunk_text threading (2026-07-24 review)
# ---------------------------------------------------------------------------
def test_probe_join_miss_raises_symmetric_guard(tmp_path):
    # Pool whose chunk_id does NOT match any source chunk → every gold-YES
    # chunk falls to empty-pool → zero probe-bearing rows → must raise.
    import pandas as pd
    from dagspaces.grpo_training.stages.modular_reward import build_modular_dataset
    pq = tmp_path / "pools.parquet"
    pd.DataFrame([
        {"gutenberg_id": "135", "chunk_id": "NO-SUCH-CHUNK", "probe_id": "p1",
         "gold": "yes", "prompt_text": "Q?", "norm_index": 0},
    ]).to_parquet(pq)
    grpo_cfg = {
        "probes": {"pools_path": str(pq), "k_max": 4},
        "task_mix": {"extract": 1.0, "vignette": 0.0},
        "prescreen": {"target_n": 10}, "battery": {},
    }
    with pytest.raises(ValueError, match="ZERO got probes attached"):
        build_modular_dataset(
            cfg=None, grpo_cfg=grpo_cfg, chunks_df=_chunks_df(),
            norm_universes={}, reward_fn=_reward(), tokenizer=_StubTokenizer(),
            ci_prompt_template="Extract: {{chunk_text}}",
            output_dir=str(tmp_path), seed=0, embed_fn=None,
        )


def test_extract_meta_carries_chunk_text(tmp_path):
    from dagspaces.grpo_training.stages.modular_reward import build_modular_dataset
    grpo_cfg = {
        "probes": {"pools_path": _pools_parquet(tmp_path), "k_max": 4},
        "task_mix": {"extract": 1.0, "vignette": 0.0},
        "prescreen": {"target_n": 10}, "battery": {},
    }
    rf = _reward()
    build_modular_dataset(
        cfg=None, grpo_cfg=grpo_cfg, chunks_df=_chunks_df(),
        norm_universes={}, reward_fn=rf, tokenizer=_StubTokenizer(),
        ci_prompt_template="Extract: {{chunk_text}}",
        output_dir=str(tmp_path), seed=0, embed_fn=None,
    )
    ex = [m for m in rf.prompt_metadata.values() if m.get("task_type") == "extract"]
    assert ex and all(m.get("chunk_text") for m in ex)  # non-empty passage carried
