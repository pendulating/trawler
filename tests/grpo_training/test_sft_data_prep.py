"""Tests for ``dagspaces/grpo_training/stages/sft_data_prep.py``.

Covers the three flow-toggle ablations that ship with the COLM pipeline:

* **baseline** — all four ``flow_*`` toggles default-on; the per-flow
  schema carries the full 5+5 field set, and the user-facing instruction
  prose contains the full metadata clause.
* **minimal_tuple** — all four ``flow_*`` toggles off; per-flow schema
  is the base 5-tuple, instruction prose drops the metadata clause.
* **no_norms_meta** — ``flow_norms_meta=False`` bundles three fields
  (``norms_invoked``, ``norm_source``, ``is_new_flow``) out together;
  the other three toggles stay on.

The fixtures depend on the real fiction10 parquets at
``/share/pierson/matt/n2s4cir/data/fiction10/``. When those files are
absent (e.g. on a fresh clone or non-cluster checkout) the entire
module is ``pytest.skip``ped at collection time so the suite stays
green for everyone else.

Migrated from ``scripts/test_sft_data_prep.py`` (custom runner) to
pytest on 2026-05-12.
"""

from __future__ import annotations

import json
import os
import tempfile

import pandas as pd
import pytest
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Data-availability gate (module-level skip)
# ---------------------------------------------------------------------------

_DATA_DIR = "/share/pierson/matt/n2s4cir/data/fiction10"
_CI_REASONING_PATH = os.path.join(_DATA_DIR, "ci_reasoning.parquet")
_CI_FLOWS_PATH = os.path.join(_DATA_DIR, "ci_flows.parquet")

if not (os.path.exists(_CI_REASONING_PATH) and os.path.exists(_CI_FLOWS_PATH)):
    pytest.skip(
        f"fiction10 parquets missing at {_DATA_DIR}; sft_data_prep tests "
        "require the real corpus and are cluster-only",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Constants — must match the production schema
# ---------------------------------------------------------------------------

BASE_FIELDS = [
    "sender", "recipient", "subject",
    "information_type", "transmission_principle",
]
ALL_META_FIELDS = [
    "context", "appropriateness",
    "norms_invoked", "norm_source", "is_new_flow",
    "confidence",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ci_reasoning() -> pd.DataFrame:
    return pd.read_parquet(_CI_REASONING_PATH)


@pytest.fixture(scope="module")
def ci_flows() -> pd.DataFrame:
    return pd.read_parquet(_CI_FLOWS_PATH)


def _make_cfg(**flow_overrides):
    """Build a minimal cfg with the given ``training.sft.flow_*`` overrides."""
    if flow_overrides:
        return OmegaConf.create({"training": {"sft": dict(flow_overrides)}})
    return OmegaConf.create({})


def _iter_positive_pair(result_df):
    """Yield ``(i, user_text, completion)`` for the first row whose
    completion carries a non-empty ``flows`` list.

    Negative rows (no information exchange detected) have ``flows: []``
    and don't exercise the per-flow schema, so we always probe a
    positive sample.
    """
    for i, row in result_df.iterrows():
        msgs = json.loads(row["messages"])
        completion = json.loads(msgs[1]["content"])
        if completion.get("flows"):
            return i, msgs[0]["content"], completion
    raise AssertionError("No positive pair (with non-empty flows) found in result")


# ---------------------------------------------------------------------------
# Baseline — all toggles default-on, full schema
# ---------------------------------------------------------------------------

class TestBaseline:
    @pytest.fixture(scope="class")
    def baseline_result(self, ci_reasoning, ci_flows):
        from dagspaces.grpo_training.stages.sft_data_prep import (
            run_sft_data_prep_stage,
        )

        cfg = _make_cfg()
        return run_sft_data_prep_stage(ci_reasoning.copy(), ci_flows.copy(), cfg)

    def test_produces_at_least_one_pair(self, baseline_result):
        assert len(baseline_result) > 0

    def test_output_schema(self, baseline_result):
        assert set(baseline_result.columns) == {"messages", "source_id", "task_type"}
        assert (baseline_result["task_type"] == "ci_extraction").all()

    def test_messages_have_user_then_assistant_roles(self, baseline_result):
        # Probe the first 50 rows — full sweep is unnecessary; schema is uniform.
        for i, row in baseline_result.head(50).iterrows():
            msgs = json.loads(row["messages"])
            assert len(msgs) == 2, f"row {i}: expected 2 messages, got {len(msgs)}"
            assert msgs[0]["role"] == "user", f"row {i}: first role != user"
            assert msgs[1]["role"] == "assistant", f"row {i}: second role != assistant"

    def test_completions_are_valid_json_with_full_schema(self, baseline_result):
        for i, row in baseline_result.head(50).iterrows():
            msgs = json.loads(row["messages"])
            completion = json.loads(msgs[1]["content"])  # raises if invalid JSON

            assert "reasoning" in completion
            assert isinstance(completion["reasoning"], str)
            assert "has_information_exchange" in completion
            assert "flows" in completion

            for j, flow in enumerate(completion["flows"]):
                for field in BASE_FIELDS + ["context", "appropriateness", "confidence"]:
                    assert field in flow, (
                        f"row {i}, flow {j}: missing required field '{field}'"
                    )

    def test_instruction_prose_matches_historical_constant(self, baseline_result):
        """The baseline instruction text must be byte-identical to
        ``_CI_INSTRUCTION``. Drift here means the SFT pair distribution
        has changed, which invalidates any baseline-mode finetune
        previously trained on the historical prose."""
        from dagspaces.grpo_training.stages.sft_data_prep import _CI_INSTRUCTION

        _, user_text, _ = _iter_positive_pair(baseline_result)
        assert user_text.startswith(_CI_INSTRUCTION + "\n\n")

    def test_parquet_roundtrip_preserves_rows_and_messages(self, baseline_result):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp_path = f.name
        try:
            baseline_result.to_parquet(tmp_path, index=False)
            reloaded = pd.read_parquet(tmp_path)
            assert len(reloaded) == len(baseline_result)
            sample = json.loads(reloaded.iloc[0]["messages"])
            assert len(sample) == 2
        finally:
            os.unlink(tmp_path)

    def test_compatible_with_trl_dataset_loader(self, baseline_result):
        """``trl`` reads messages as a list of dicts; we ship them as a JSON
        string in parquet. Confirm the conventional ``ds.map(json.loads)``
        unwrap step works end-to-end."""
        from datasets import Dataset

        dataset = Dataset.from_pandas(baseline_result)
        dataset = dataset.map(
            lambda row: {
                "messages": (
                    json.loads(row["messages"])
                    if isinstance(row["messages"], str)
                    else row["messages"]
                )
            }
        )
        sample = dataset[0]
        assert isinstance(sample["messages"], list)
        assert sample["messages"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# Minimal tuple — all four flow_* off, base 5-tuple only
# ---------------------------------------------------------------------------

class TestMinimalTuple:
    @pytest.fixture(scope="class")
    def minimal_result(self, ci_reasoning, ci_flows):
        from dagspaces.grpo_training.stages.sft_data_prep import (
            run_sft_data_prep_stage,
        )

        cfg = _make_cfg(
            flow_context=False,
            flow_appropriateness=False,
            flow_norms_meta=False,
            flow_confidence=False,
        )
        return run_sft_data_prep_stage(ci_reasoning.copy(), ci_flows.copy(), cfg)

    def test_produces_at_least_one_pair(self, minimal_result):
        assert len(minimal_result) > 0

    def test_flows_carry_only_base_5_tuple(self, minimal_result):
        _, _, completion = _iter_positive_pair(minimal_result)
        for j, flow in enumerate(completion["flows"]):
            for field in BASE_FIELDS:
                assert field in flow, f"flow {j}: missing base field '{field}'"
            for field in ALL_META_FIELDS:
                assert field not in flow, (
                    f"flow {j}: forbidden meta field '{field}' present "
                    f"under minimal_tuple mode"
                )

    def test_instruction_prose_drops_metadata_clause(self, minimal_result):
        """Scan ONLY the instruction prose, not the article body — the
        novel text itself often contains words like "context" or "along
        with" verbatim (Orwell's "along with him" is a known false positive).
        Split on the ``\\n\\n`` that separates the instruction from the
        article."""
        _, user_text, _ = _iter_positive_pair(minimal_result)
        instruction_text = user_text.split("\n\n", 1)[0]
        forbidden_phrases = [
            "context", "appropriateness", "confidence",
            "norms", "metadata", "along with",
        ]
        for phrase in forbidden_phrases:
            assert phrase not in instruction_text, (
                f"minimal_tuple instruction still mentions {phrase!r}: "
                f"{instruction_text!r}"
            )

    def test_instruction_sentence_ends_at_base_tuple(self, minimal_result):
        _, user_text, _ = _iter_positive_pair(minimal_result)
        instruction_text = user_text.split("\n\n", 1)[0]
        assert instruction_text.rstrip().endswith("transmission_principle)."), (
            f"minimal_tuple instruction does not end with "
            f"'transmission_principle).': {instruction_text!r}"
        )


# ---------------------------------------------------------------------------
# No-norms-meta — bundles norms_invoked + norm_source + is_new_flow out
# ---------------------------------------------------------------------------

class TestNoNormsMeta:
    @pytest.fixture(scope="class")
    def norms_off_result(self, ci_reasoning, ci_flows):
        from dagspaces.grpo_training.stages.sft_data_prep import (
            run_sft_data_prep_stage,
        )

        cfg = _make_cfg(flow_norms_meta=False)
        return run_sft_data_prep_stage(ci_reasoning.copy(), ci_flows.copy(), cfg)

    def test_produces_at_least_one_pair(self, norms_off_result):
        assert len(norms_off_result) > 0

    def test_flows_keep_base_plus_three_remaining_fields(self, norms_off_result):
        _, _, completion = _iter_positive_pair(norms_off_result)
        for j, flow in enumerate(completion["flows"]):
            for field in BASE_FIELDS + ["context", "appropriateness", "confidence"]:
                assert field in flow, f"flow {j}: missing '{field}'"

    def test_norms_trio_drops_together(self, norms_off_result):
        """``flow_norms_meta=False`` is the bundled toggle: all three of
        ``norms_invoked``, ``norm_source``, ``is_new_flow`` must drop
        together. If any one survives, the toggle is no longer atomic
        and the ablation analysis breaks."""
        _, _, completion = _iter_positive_pair(norms_off_result)
        for j, flow in enumerate(completion["flows"]):
            for field in ("norms_invoked", "norm_source", "is_new_flow"):
                assert field not in flow, (
                    f"flow {j}: forbidden field {field!r} "
                    f"(must drop with flow_norms_meta=False)"
                )

    def test_instruction_prose_drops_norms_keeps_other_three(self, norms_off_result):
        _, user_text, _ = _iter_positive_pair(norms_off_result)
        instruction_text = user_text.split("\n\n", 1)[0]
        assert "norms" not in instruction_text, (
            f"instruction still mentions 'norms': {instruction_text!r}"
        )
        # The other three meta-fields stay in the instruction prose.
        assert "context" in instruction_text
        assert "appropriateness" in instruction_text
        assert "confidence" in instruction_text


# ---------------------------------------------------------------------------
# Negative selection — curate gold=False negatives to truly-contentless chunks
# ---------------------------------------------------------------------------

def _iter_negative_reasonings(result_df):
    """Yield the reasoning text of every negative (no-flow) SFT pair."""
    for _, row in result_df.iterrows():
        completion = json.loads(json.loads(row["messages"])[1]["content"])
        if not completion.get("has_information_exchange") and not completion.get("flows"):
            yield completion.get("reasoning", "")


class TestContentlessClassifier:
    """`_is_contentless_chunk` separates genuinely flow-free chunks from
    has-exchange-but-no-norm chunks (the v6 negative-curation fix)."""

    def test_flags_real_exchanges_as_not_contentless(self):
        from dagspaces.grpo_training.stages.sft_data_prep import _is_contentless_chunk
        has_exchange = [
            "The text describes a conversation between Fred and Mary, where Fred "
            "discloses personal financial issues and asks for her help.",
            "Lydgate is disclosing financial difficulties to his wife Rosamond.",
            "The old man warns Montparnasse about the dangers of idleness.",
            "The provided text is a dialogue between Danglars and Monte Cristo.",
            "While it involves the exchange of information between the characters, "
            "it does not prescribe or regulate it.",
            "There are several instances of information being exchanged.",
        ]
        for rt in has_exchange:
            assert not _is_contentless_chunk(rt), f"should NOT be contentless: {rt!r}"

    def test_flags_genuinely_empty_as_contentless(self):
        from dagspaces.grpo_training.stages.sft_data_prep import _is_contentless_chunk
        contentless = [
            "The preface is a critical and appreciative essay about Jane Austen's "
            "work. It does not contain any prescriptive content that regulates the "
            "exchange of information between agents.",
            "The provided text is a descriptive passage about a hunting scene.",
            "A historical narrative about the political changes following 1815.",
            "",  # empty reasoning → nothing to extract
        ]
        for rt in contentless:
            assert _is_contentless_chunk(rt), f"should be contentless: {rt!r}"

    def test_boilerplate_alone_does_not_trip_the_detector(self):
        """The gold=False boilerplate ('regulates the exchange of information
        between agents') must NOT, by itself, mark a chunk has-exchange —
        otherwise every negative would be dropped."""
        from dagspaces.grpo_training.stages.sft_data_prep import _is_contentless_chunk
        assert _is_contentless_chunk(
            "The text does not contain any prescriptive content that commands, "
            "prohibits, or regulates the exchange of information between agents."
        )


class TestNegativeSelection:
    def test_contentless_keeps_fewer_and_only_contentless_negatives(
        self, ci_reasoning, ci_flows
    ):
        from dagspaces.grpo_training.stages.sft_data_prep import (
            run_sft_data_prep_stage,
            _is_contentless_chunk,
        )

        all_neg = run_sft_data_prep_stage(
            ci_reasoning.copy(), ci_flows.copy(), _make_cfg(negative_selection="all"))
        contentless = run_sft_data_prep_stage(
            ci_reasoning.copy(), ci_flows.copy(), _make_cfg(negative_selection="contentless"))

        n_all = sum(1 for _ in _iter_negative_reasonings(all_neg))
        n_cl = sum(1 for _ in _iter_negative_reasonings(contentless))
        assert n_cl > 0, "contentless mode produced no negatives"
        assert n_cl <= n_all, (
            f"contentless negatives ({n_cl}) should not exceed all-negatives ({n_all})"
        )
        # Every kept negative must pass the classifier (that IS the filter).
        for rt in _iter_negative_reasonings(contentless):
            assert _is_contentless_chunk(rt), (
                f"contentless mode kept a has-exchange negative: {rt[:160]!r}"
            )

    def test_default_is_all(self, ci_reasoning, ci_flows):
        """Omitting the knob preserves legacy behavior (all negatives)."""
        from dagspaces.grpo_training.stages.sft_data_prep import run_sft_data_prep_stage
        default = run_sft_data_prep_stage(ci_reasoning.copy(), ci_flows.copy(), _make_cfg())
        explicit_all = run_sft_data_prep_stage(
            ci_reasoning.copy(), ci_flows.copy(), _make_cfg(negative_selection="all"))
        assert sum(1 for _ in _iter_negative_reasonings(default)) == \
            sum(1 for _ in _iter_negative_reasonings(explicit_all))
