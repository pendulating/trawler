"""gpt-oss speaks the OpenAI *harmony* response format, not `<think>`.

Harmony (https://github.com/openai/harmony) is a channel protocol: the assistant
emits `<|channel|>NAME<|message|>...` segments and only the **final** channel is
the answer. `analysis` is hidden chain-of-thought.

Two bugs this suite locks down (both live until 2026-07-13):

1. `_detect_reasoning_parser` returned `"gptoss"`, but vLLM registers the parser
   as `"openai_gptoss"` -> KeyError -> swallowed by `except Exception: pass` ->
   fell back to the `<think>` regex, which harmony never emits. Result: the
   entire analysis channel was handed downstream **as the answer**, and
   `reasoning` came back empty. Fixing the name alone does not help: vLLM's
   parser raises NotImplementedError for non-streaming input, and this repo only
   calls `LLM.generate()` offline.

2. vLLM's `skip_special_tokens=True` default *deletes* the delimiters, collapsing
   the output to `"analysisFOOassistantfinalBAR"` — unsplittable by anyone.

The invariant these tests defend: **hidden CoT must never be returned as
content.** An empty answer is a visible failure (the format-adherence gate catches
it); a plausible-looking analysis blob graded as the answer is an invisible one.
"""

import pytest

from dagspaces.common.vllm_inference import (
    _is_harmony_model,
    _split_harmony,
    _split_reasoning,
)

GPT_OSS = "/share/pierson/matt/zoo/models/GPT-OSS-20B"


class TestHarmonyDetection:
    def test_gpt_oss_is_harmony(self):
        assert _is_harmony_model(GPT_OSS)
        assert _is_harmony_model("/zoo/GPT-OSS-20B-SFT-merged")

    def test_other_families_are_not(self):
        for src in (
            "/zoo/Qwen3.5-9B",
            "/zoo/Gemma-4-31B-it",
            "/zoo/Llama-3.1-8B-Instruct",
            "/zoo/Phi-4",
        ):
            assert not _is_harmony_model(src), src

    def test_gpt_oss_has_no_vllm_reasoning_parser(self):
        """Harmony is handled by _split_harmony, NOT a vLLM parser.

        Returning a parser name here is what caused the original bug — vLLM has
        no non-streaming gpt-oss parser to return.
        """
        from dagspaces.common.vllm_inference import _detect_reasoning_parser

        assert _detect_reasoning_parser(GPT_OSS) is None


class TestHarmonySplit:
    def test_analysis_never_leaks_into_content(self):
        raw = (
            "<|channel|>analysis<|message|>The nurse would be violating "
            "confidentiality.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Answer: 3<|return|>"
        )
        reasoning, content = _split_harmony(raw)
        assert content == "Answer: 3"
        assert "confidentiality" in reasoning
        assert "<|" not in content

    def test_commentary_channel_is_not_the_answer(self):
        """Tool traffic is not content either — only `final` is."""
        raw = (
            "<|channel|>analysis<|message|>Think.<|end|>"
            "<|start|>assistant<|channel|>commentary to=browser.search"
            "<|message|>{'q': 'hipaa'}<|call|>"
            "<|start|>assistant<|channel|>final<|message|>Answer: 2<|return|>"
        )
        reasoning, content = _split_harmony(raw)
        assert content == "Answer: 2"
        assert "hipaa" in reasoning

    def test_truncated_generation_yields_empty_content(self):
        """No `final` channel -> no answer. Must NOT fall back to the CoT.

        This is the whole point: a run that blows its token budget inside the
        analysis channel produced no answer, and must report that honestly.
        """
        raw = "<|channel|>analysis<|message|>Still reasoning when the budget ran"
        reasoning, content = _split_harmony(raw)
        assert content == ""
        assert "Still reasoning" in reasoning

    def test_non_harmony_text_returns_none(self):
        assert _split_harmony("Answer: 3") is None

    def test_stripped_delimiters_are_salvaged(self):
        """skip_special_tokens=True ate the markers; salvage on 'assistantfinal'."""
        reasoning, content = _split_harmony(
            "analysisWeigh the norms here.assistantfinalAnswer: 3"
        )
        assert content == "Answer: 3"
        assert reasoning == "Weigh the norms here."


class TestRunVllmInferenceScope:
    """The harmony hook in run_vllm_inference must use the in-scope name.

    Regression lock for 2026-07-13: the hook was written as
    `_is_harmony_model(model_source)`, but in that scope the bound name is
    `_model_source` — the bare `model_source` exists only inside the nested
    `if lora_path:` branch. The result was `UnboundLocalError` for **every
    model**, not just gpt-oss, and it took down both top100 production runs.

    Nothing caught it beforehand:
      * the unit suite never executes `run_vllm_inference` (it needs a GPU), and
      * pyflakes/ruff do NOT flag it — `model_source` *is* assigned in the
        function, just conditionally, so it is a legal local.

    This is a narrow textual lock, not a proof of correctness. The real
    verification is running an actual stage. It exists because the failure mode
    (a crash on every model, from a gpt-oss-only feature) is worth one cheap
    assertion.
    """

    def test_harmony_hook_uses_the_bound_name(self):
        import ast
        import inspect

        from dagspaces.common import vllm_inference

        src = inspect.getsource(vllm_inference)
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_vllm_inference"
        )

        calls = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_is_harmony_model"
        ]
        assert calls, "the harmony hook vanished from run_vllm_inference"

        for call in calls:
            arg = call.args[0]
            assert isinstance(arg, ast.Name), "harmony hook arg should be a plain name"
            assert arg.id == "_model_source", (
                f"_is_harmony_model() called with {arg.id!r}; the name bound in "
                f"run_vllm_inference's scope is '_model_source'. Bare "
                f"'model_source' is assigned only inside the `if lora_path:` "
                f"branch and raises UnboundLocalError for every model."
            )


class TestSplitReasoningIntegration:
    """The public entry point must route gpt-oss to the harmony splitter."""

    def test_end_to_end_no_leak(self):
        raw = (
            "<|channel|>analysis<|message|>Hidden CoT.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Answer: 4<|return|>"
        )
        reasoning, content = _split_reasoning(
            raw, GPT_OSS, thinking_enabled=True, tokenizer=None,
        )
        assert content == "Answer: 4"
        assert reasoning == "Hidden CoT."

    def test_bare_prose_passes_through(self):
        reasoning, content = _split_reasoning(
            "Answer: 3", GPT_OSS, thinking_enabled=True, tokenizer=None,
        )
        assert content == "Answer: 3"
        assert reasoning == ""

    @pytest.mark.parametrize("raw", ["", None])
    def test_empty_input(self, raw):
        assert _split_reasoning(raw, GPT_OSS, True, None) == ("", "")
