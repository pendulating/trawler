"""Tests for ``model_needs_reasoning_budget``.

Regression coverage for the eval_all audit (2026-06): short-answer eval
stages (confaide, cirl_vignettes, mmlu) default to a tiny max_tokens and
bump it for reasoning models. The original detector keyed only on
``chat_template_kwargs.enable_thinking == false``, so gpt-oss — whose
config is a bare ``chat_template_kwargs: {}`` but which *always* emits a
harmony ``analysis`` channel before ``final`` — was missed and truncated
its answer inside the 16-token budget (parseable_rate=0).
"""

from __future__ import annotations

from omegaconf import OmegaConf

from dagspaces.common.vllm_inference import model_needs_reasoning_budget


def _cfg(d):
    return OmegaConf.create(d)


class TestReasoningBudgetDetection:
    def test_enable_thinking_false_triggers(self):
        cfg = _cfg({"model_source": "/zoo/CIRL", "chat_template_kwargs": {"enable_thinking": False}})
        assert model_needs_reasoning_budget(cfg) is True

    def test_enable_thinking_true_does_not_trigger_for_plain_family(self):
        cfg = _cfg({"model_source": "/zoo/Qwen2.5", "chat_template_kwargs": {"enable_thinking": True}})
        assert model_needs_reasoning_budget(cfg) is False

    def test_gpt_oss_harmony_triggers_without_enable_thinking_key(self):
        # The exact gpt-oss config shape: bare chat_template_kwargs, no flag.
        cfg = _cfg({"model_source": "/zoo/GPT-OSS-20B", "chat_template_kwargs": {}})
        assert model_needs_reasoning_budget(cfg) is True

    def test_qwen3_family_triggers_via_parser(self):
        cfg = _cfg({"model_source": "/zoo/Qwen3.5-9B", "chat_template_kwargs": {}})
        assert model_needs_reasoning_budget(cfg) is True

    def test_path_opaque_reasoning_model_triggers_via_model_family(self):
        # OpenThinker3-7B is a qwen3 reasoning model, but "qwen3" appears only
        # in model_family — never in the checkpoint path. Detection must consult
        # model_family, else the model truncates its CoT inside a small budget
        # (2026-06-04 sweep: 66/493 PrivacyLens actions hit finish_reason=length).
        cfg = _cfg({
            "model_source": "/zoo/OpenThinker3-7B",
            "model_family": "qwen3",
            "chat_template_kwargs": {"enable_thinking": True},
        })
        assert model_needs_reasoning_budget(cfg) is True

    def test_plain_family_field_does_not_trigger(self):
        # A declared non-reasoning model_family must not flip the result — guards
        # against the model_family check over-firing (e.g. "qwen2.5" must not
        # match the "qwen3" substring rule).
        cfg = _cfg({
            "model_source": "/zoo/Qwen2.5-7B",
            "model_family": "qwen2.5",
            "chat_template_kwargs": {"enable_thinking": True},
        })
        assert model_needs_reasoning_budget(cfg) is False

    def test_plain_instruct_family_does_not_trigger(self):
        # Phi-4 / Llama / Gemma-3: no reasoning parser, no enable_thinking flag.
        for src in ("/zoo/Phi-4", "/zoo/Llama-3.1-8B", "/zoo/gemma-3-12b"):
            cfg = _cfg({"model_source": src, "chat_template_kwargs": {}})
            assert model_needs_reasoning_budget(cfg) is False, src

    def test_plain_dict_input(self):
        # Helper must accept a plain dict, not just DictConfig.
        assert model_needs_reasoning_budget(
            {"model_source": "/zoo/GPT-OSS-20B", "chat_template_kwargs": {}}
        ) is True
        assert model_needs_reasoning_budget(
            {"model_source": "/zoo/Llama-3.1-8B", "chat_template_kwargs": {}}
        ) is False
