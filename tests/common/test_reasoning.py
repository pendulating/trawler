"""Tests for the reasoning/thinking-block extraction module.

Covers the correctness-critical harmony + <think> splitting logic
extracted from vllm_inference.py (Finding 7, wiki/jul19_refactoring.md).
"""

from __future__ import annotations

from dagspaces.common.reasoning import (
    _fallback_strip_reasoning,
    _is_harmony_model,
    _split_harmony,
    _detect_reasoning_parser,
    _strip_think_blocks,
    model_needs_reasoning_budget,
)


# ── _fallback_strip_reasoning / _strip_think_blocks ───────────────────


class TestFallbackStripReasoning:
    def test_think_blocks_removed(self):
        text = "<think>Let me reason about this.</think>The answer is 42."
        assert _fallback_strip_reasoning(text) == "The answer is 42."

    def test_multiple_think_blocks(self):
        text = "<think>first</think>answer1<think>second</think>answer2"
        assert _fallback_strip_reasoning(text) == "answer1answer2"

    def test_unterminated_think_block(self):
        text = "<think>reasoning that never ends..."
        assert _fallback_strip_reasoning(text) == ""

    def test_begin_of_thought(self):
        text = "<|begin_of_thought|>reasoning<|end_of_thought|>The answer."
        assert _fallback_strip_reasoning(text) == "The answer."

    def test_unterminated_begin_of_thought(self):
        text = "<|begin_of_thought|>reasoning that never ends"
        assert _fallback_strip_reasoning(text) == ""

    def test_no_think_blocks(self):
        text = "Just a plain answer."
        assert _fallback_strip_reasoning(text) == "Just a plain answer."

    def test_empty_string(self):
        assert _fallback_strip_reasoning("") == ""

    def test_alias_identity(self):
        assert _strip_think_blocks is _fallback_strip_reasoning


# ── _is_harmony_model ─────────────────────────────────────────────────


class TestIsHarmonyModel:
    def test_gpt_oss(self):
        assert _is_harmony_model("gpt-oss-20b") is True
        assert _is_harmony_model("/zoo/models/gpt-oss-20b") is True
        assert _is_harmony_model("GPT-OSS") is True

    def test_non_harmony(self):
        assert _is_harmony_model("qwen3-8b") is False
        assert _is_harmony_model("llama3.1-8b") is False
        assert _is_harmony_model("") is False


# ── _split_harmony ────────────────────────────────────────────────────


class TestSplitHarmony:
    def test_standard_channels(self):
        text = (
            "<|channel|>analysis<|message|>Let me think about this. "
            "The patient's data should not be shared.<|end|>"
            "<|channel|>final<|message|>The answer is: do not share.<|end|>"
        )
        result = _split_harmony(text)
        assert result is not None
        reasoning, content = result
        assert "think about this" in reasoning
        assert "do not share" in content

    def test_no_harmony_structure(self):
        text = "Just a plain answer with no channels."
        assert _split_harmony(text) is None

    def test_truncated_before_final(self):
        """A generation that never reaches the final channel yields empty content."""
        text = "<|channel|>analysis<|message|>I'm still thinking..."
        result = _split_harmony(text)
        assert result is not None
        reasoning, content = result
        assert "still thinking" in reasoning
        assert content == ""

    def test_stripped_delimiters_salvage(self):
        """When skip_special_tokens strips delimiters, salvage on 'assistantfinal'."""
        text = "analysisLet me reason.assistantfinalThe answer is 42."
        result = _split_harmony(text)
        assert result is not None
        reasoning, content = result
        assert "42" in content


# ── _detect_reasoning_parser ──────────────────────────────────────────


class TestDetectReasoningParser:
    def test_qwen3(self):
        assert _detect_reasoning_parser("qwen3-8b") == "qwen3"
        assert _detect_reasoning_parser("qwen3.5-4b") == "qwen3"

    def test_deepseek(self):
        assert _detect_reasoning_parser("deepseek-r1-7b") == "deepseek_r1"

    def test_gemma4(self):
        assert _detect_reasoning_parser("gemma-4-12b") == "gemma4"

    def test_non_thinking(self):
        assert _detect_reasoning_parser("llama3.1-8b") is None
        assert _detect_reasoning_parser("phi-4") is None

    def test_gpt_oss_returns_none(self):
        """gpt-oss deliberately returns None — harmony is handled separately."""
        assert _detect_reasoning_parser("gpt-oss-20b") is None


# ── model_needs_reasoning_budget ──────────────────────────────────────


class TestModelNeedsReasoningBudget:
    def test_enable_thinking_false(self):
        cfg = {"chat_template_kwargs": {"enable_thinking": False}}
        assert model_needs_reasoning_budget(cfg) is True

    def test_enable_thinking_true(self):
        cfg = {"chat_template_kwargs": {"enable_thinking": True}}
        # enable_thinking=True alone doesn't trigger — need structural reasoning
        assert model_needs_reasoning_budget(cfg) is False

    def test_qwen3_structural(self):
        cfg = {"model_source": "qwen3-8b"}
        assert model_needs_reasoning_budget(cfg) is True

    def test_gpt_oss_structural(self):
        cfg = {"model_source": "gpt-oss-20b"}
        assert model_needs_reasoning_budget(cfg) is True

    def test_non_reasoning_model(self):
        cfg = {"model_source": "llama3.1-8b"}
        assert model_needs_reasoning_budget(cfg) is False

    def test_model_family_fallback(self):
        """OpenThinker3-7B is a qwen3 model but 'qwen3' isn't in its path."""
        cfg = {"model_source": "OpenThinker3-7B", "model_family": "qwen3"}
        assert model_needs_reasoning_budget(cfg) is True

    def test_empty_config(self):
        assert model_needs_reasoning_budget({}) is False


# ── Backward compat: imports from vllm_inference still work ──────────


class TestBackwardCompat:
    def test_vllm_inference_reexports(self):
        from dagspaces.common.vllm_inference import (
            _detect_reasoning_parser as drp,
            _is_harmony_model as ihm,
            _strip_think_blocks as stb,
            model_needs_reasoning_budget as mnrb,
        )

        assert stb is _strip_think_blocks
        assert ihm is _is_harmony_model
        assert drp is _detect_reasoning_parser
        assert mnrb is model_needs_reasoning_budget
