"""SFT/GRPO extract-prompt parity (R1 fix, 2026-07-28).

The m1 wave sampled GRPO rollouts on `conf/prompt/ci_extraction.yaml` — an
instruction the SFT adapter never trained on — and paid 34.4% vs 2.7% R-VALID
gate failure for it (A/B probe, outputs/2026-07-28_ab_prompt_probe). The fix
routes m-series rollout prompts through `sft_aligned_extract_template`, which
imports the SAME instruction builder SFT data prep uses. This test pins the
byte-parity end-to-end: the template with the chunk substituted must equal the
user message `run_sft_data_prep_stage` actually emits — under default AND
toggled field configs, so the two paths cannot drift apart silently.
"""
from __future__ import annotations

import json

import pandas as pd
from omegaconf import OmegaConf

from dagspaces.grpo_training.stages.sft_data_prep import (
    run_sft_data_prep_stage,
    sft_aligned_extract_template,
)

_ARTICLE = "Elizabeth told Jane about Mr. Darcy's letter, in confidence."


def _tiny_frames():
    reasoning = pd.DataFrame([{
        "gutenberg_id": "1342",
        "chunk_id": 7,
        "article_text": _ARTICLE,
        "has_information_exchange": True,
        "ci_reasoning_text": "",
    }])
    extraction = pd.DataFrame([{
        "gutenberg_id": "1342",
        "chunk_id": 7,
        "ci_flow_index": 0,
        "ci_flow_snippet": "told Jane about the letter",
        "ci_reasoning_trace": "A confidential disclosure between sisters.",
        "ci_sender": "Elizabeth",
        "ci_recipient": "Jane",
        "ci_subject": "Mr. Darcy",
        "ci_information_type": "contents of a private letter",
        "ci_transmission_principle": "in confidence",
        "ci_context": "family",
        "ci_appropriateness": "appropriate",
        "ci_norms_invoked": "[]",
        "ci_norm_source": "implicit",
        "ci_is_new_flow": False,
        "ci_confidence_qual": 8,
    }])
    return reasoning, extraction


def _sft_user_message(cfg) -> str:
    reasoning, extraction = _tiny_frames()
    out = run_sft_data_prep_stage(reasoning, extraction, cfg)
    positives = [
        json.loads(m)
        for m in out["messages"]
        if json.loads(m)[1]["content"] != ""  # all rows; just parse
    ]
    # One positive pair from the single flow-bearing chunk.
    user_msgs = [msgs[0]["content"] for msgs in positives
                 if _ARTICLE in msgs[0]["content"]]
    assert len(user_msgs) == 1
    return user_msgs[0]


class TestPromptAlignment:
    def test_template_has_single_chunk_placeholder(self):
        t = sft_aligned_extract_template(OmegaConf.create({}))
        assert t.count("{{chunk_text}}") == 1

    def test_byte_parity_with_sft_pairs_default_toggles(self):
        cfg = OmegaConf.create({})
        rendered = sft_aligned_extract_template(cfg).replace(
            "{{chunk_text}}", _ARTICLE)
        assert rendered == _sft_user_message(cfg)

    def test_byte_parity_with_sft_pairs_toggled(self):
        cfg = OmegaConf.create(
            {"training": {"sft": {"flow_confidence": False,
                                  "flow_norms_meta": False}}})
        rendered = sft_aligned_extract_template(cfg).replace(
            "{{chunk_text}}", _ARTICLE)
        assert rendered == _sft_user_message(cfg)

    def test_aligned_differs_from_config_prompt(self):
        # Regression guard on the incident itself: the aligned template must
        # not silently be the ci_extraction.yaml text again.
        import pathlib

        yaml_path = (pathlib.Path(__file__).parents[2] / "dagspaces" /
                     "grpo_training" / "conf" / "prompt" / "ci_extraction.yaml")
        cfg_prompt = OmegaConf.load(yaml_path)
        aligned = sft_aligned_extract_template(OmegaConf.create({}))
        assert str(cfg_prompt.instruction).strip() not in aligned
