"""Tests for the agent-action token budget in PrivacyLens inference.

Regression coverage for the 2026-06-04 sweep finding: reasoning models burn
the action budget on hidden CoT before emitting the ReAct ``Action:`` block.
At the old flat ``max_tokens=4096`` they truncated mid-reasoning
(context-reasoner/ppo: 227/493 actions hit finish_reason=length), tanking
``agent_action_format_rate``. ``run_action_inference`` now bumps the budget to
8192 for reasoning models (and leaves non-reasoning models at 4096, since they
do not truncate — gpt-oss completion maxed at 2222 — so a bigger budget would
only mask format issues).

The budget is set inside ``run_action_inference`` and threaded to vLLM via the
per-row ``sampling_params``. We capture it by stubbing ``run_vllm_inference``
(it would otherwise spin up vLLM) and the prompt builder, then running the
stage's own ``preprocess`` closure on one row.
"""

from __future__ import annotations

import pandas as pd
import pytest
from omegaconf import OmegaConf

import dagspaces.privacylens.stages.llm_inference as li


def _run_and_capture_action_sp(model_cfg: dict) -> dict:
    """Run ``run_action_inference`` with the heavy bits stubbed, returning the
    sampling_params the stage assigned to a row."""
    captured: dict = {}

    def fake_run_vllm_inference(df, cfg, preprocess, postprocess, stage_name):
        row = preprocess(df.iloc[0].to_dict())
        captured.update(row["sampling_params"])
        return df

    cfg = OmegaConf.create({
        "sampling_params": {"temperature": 0.0, "max_tokens": 512},
        "model": model_cfg,
    })
    df = pd.DataFrame([{"trajectory": {}}])

    orig_rvi = li.run_vllm_inference
    orig_bap = li.build_action_prompt
    li.run_vllm_inference = fake_run_vllm_inference
    li.build_action_prompt = lambda row: {"system": "s", "user": "u"}
    try:
        li.run_action_inference(df, cfg)
    finally:
        li.run_vllm_inference = orig_rvi
        li.build_action_prompt = orig_bap
    return captured


class TestActionBudget:
    def test_reasoning_model_gets_8192(self):
        # enable_thinking=false → reasoning model → larger budget.
        sp = _run_and_capture_action_sp(
            {"model_source": "/zoo/CIRL", "chat_template_kwargs": {"enable_thinking": False}}
        )
        assert sp["max_tokens"] == 8192

    def test_path_opaque_reasoning_model_gets_8192_via_family(self):
        # OpenThinker3-7B: qwen3 reasoning model whose path hides the family.
        sp = _run_and_capture_action_sp(
            {
                "model_source": "/zoo/OpenThinker3-7B",
                "model_family": "qwen3",
                "chat_template_kwargs": {"enable_thinking": True},
            }
        )
        assert sp["max_tokens"] == 8192

    def test_non_reasoning_model_stays_4096(self):
        # Phi-4: no reasoning parser, no enable_thinking flag → flat 4096.
        sp = _run_and_capture_action_sp(
            {"model_source": "/zoo/Phi-4", "model_family": "phi-4", "chat_template_kwargs": {}}
        )
        assert sp["max_tokens"] == 4096

    def test_base_sampling_params_preserved(self):
        # The budget override must not clobber other sampling params.
        sp = _run_and_capture_action_sp(
            {"model_source": "/zoo/Phi-4", "chat_template_kwargs": {}}
        )
        assert sp["temperature"] == 0.0
