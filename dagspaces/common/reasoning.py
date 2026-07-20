"""Reasoning / thinking-block extraction for LLM outputs.

Extracted from ``vllm_inference.py`` (Finding 7, wiki/jul19_refactoring.md).
These are pure, unit-testable functions that split model output into
``(reasoning, content)`` pairs — correctness-critical for the gpt-oss
harmony protocol and ``<think>`` block handling.

Public API
----------
split_reasoning(text, model_source, thinking_enabled, tokenizer)
    Primary entry point — vLLM parser with regex fallback.
fallback_strip_reasoning(text)
    Regex-only stripping (no parser, no tokenizer needed).
strip_think_blocks(text)
    Backward-compat alias for ``fallback_strip_reasoning``.
is_harmony_model(model_source)
    True for gpt-oss (harmony channel protocol).
split_harmony(text)
    Split harmony channels into ``(reasoning, final_content)``.
detect_reasoning_parser(model_source)
    Map a model path to its vLLM reasoning-parser name.
model_needs_reasoning_budget(model_cfg)
    True if the model needs a larger ``max_tokens`` budget.
"""

from __future__ import annotations

import re
from typing import Any


def _fallback_strip_reasoning(text: str) -> str:
    """Fallback regex-based stripping of reasoning/thinking blocks.

    Used only when no family-specific vLLM reasoning parser is available or
    the parser fails. See ``_split_reasoning`` for the primary path.

    Handles multiple formats used by different model families:
    - ``<think>...</think>`` (Qwen3+, DeepSeek-R1, open-source reasoning models)
    - ``<|begin_of_thought|>...<|end_of_thought|>`` (context-reasoner-ppo, some PPO models)

    Also handles unterminated blocks (model ran out of tokens mid-reasoning).
    Returns the remaining text, stripped.
    """
    # <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"<think>[\s\S]*$", "", text)
    # <|begin_of_thought|...end_of_thought|> (with optional trailing ] or >)
    text = re.sub(r"<\|begin_of_thought\|[\s\S]*?<\|end_of_thought\|[>\]\s]*", "", text)
    text = re.sub(r"<\|begin_of_thought\|[\s\S]*$", "", text)
    return text.strip()


# Backwards-compat alias — imported by dagspaces/grpo_training/stages/rewards.py.
# Prefer `_split_reasoning` for new code.
_strip_think_blocks = _fallback_strip_reasoning


def _is_harmony_model(model_source: str) -> bool:
    """True for gpt-oss, which speaks the OpenAI *harmony* response format.

    Harmony is not a `<think>`-style wrapper — it is a channel protocol
    (https://github.com/openai/harmony). The assistant emits one or more
    `<|channel|>NAME<|message|>...` segments and only the **final** channel is
    the answer; `analysis` is hidden CoT and `commentary` is tool traffic.
    It therefore needs its own splitter, not a vLLM reasoning parser — see
    `_split_harmony`.
    """
    return "gpt-oss" in (model_source or "").lower()


# One harmony segment: an optional <|start|>role, the channel name, an optional
# `to=...` tool route (commentary channel), then the payload up to any terminator.
_HARMONY_SEGMENT = re.compile(
    r"<\|channel\|>(?P<channel>[a-zA-Z_]+)"
    r"(?:\s+to=[^<]*)?"
    r"<\|message\|>(?P<content>.*?)"
    r"(?=<\|end\|>|<\|return\|>|<\|call\|>|<\|start\|>|\Z)",
    re.DOTALL,
)

# What the harmony delimiters detokenize to once skip_special_tokens=True has
# eaten them: "<|start|>assistant<|channel|>final<|message|>" -> "assistantfinal".
_HARMONY_STRIPPED_FINAL = re.compile(r"assistantfinal", re.IGNORECASE)

_HARMONY_WARNED = False


def _split_harmony(text: str) -> tuple[str, str] | None:
    """Split a gpt-oss harmony completion into ``(reasoning, final_content)``.

    Returns ``None`` if the text carries no harmony structure at all, so the
    caller can fall through.

    A truncated generation that never reaches the `final` channel yields
    ``content == ""`` — deliberately. The alternative (hand back the `analysis`
    text) would let hidden CoT be graded as the answer, which is exactly the bug
    this function exists to prevent. An empty answer is visible to the
    format-adherence gate; a plausible-looking wrong one is not.
    """
    global _HARMONY_WARNED

    segments = [
        (m.group("channel").lower(), m.group("content"))
        for m in _HARMONY_SEGMENT.finditer(text)
    ]
    if segments:
        final = "\n".join(c for ch, c in segments if ch == "final").strip()
        reasoning = "\n".join(c for ch, c in segments if ch != "final").strip()
        return reasoning, final

    # No delimiters. Almost always means skip_special_tokens=True stripped them
    # (see run_vllm_inference). Salvage what we can, but say so loudly — silently
    # returning the smashed text is how the analysis channel got graded as the
    # answer in the first place.
    if _HARMONY_STRIPPED_FINAL.search(text):
        if not _HARMONY_WARNED:
            _HARMONY_WARNED = True
            print(
                "[vllm_inference] WARNING: harmony output arrived WITHOUT channel "
                "delimiters — skip_special_tokens was True somewhere. Salvaging on "
                "the 'assistantfinal' marker, but fix the sampling params: the "
                "reasoning/answer split is unreliable in this mode."
            )
        head, _, tail = text.rpartition("assistantfinal")
        reasoning = head.strip()
        if reasoning.lower().startswith("analysis"):
            reasoning = reasoning[len("analysis") :].strip()
        return reasoning, tail.strip()

    return None


def _detect_reasoning_parser(model_source: str) -> str | None:
    """Map a model path to the vLLM reasoning-parser name for that family.

    Returns a parser name registered in ``vllm.reasoning.ReasoningParserManager``,
    or ``None`` for non-thinking families (Phi-4, Llama, Gemma-3, etc.) where
    no reasoning extraction is needed.

    gpt-oss is deliberately absent. It used to return ``"gptoss"``, which is not
    a registered name (vLLM calls it ``openai_gptoss``), so the lookup raised
    KeyError, the caller swallowed it, and every gpt-oss run silently fell back
    to the `<think>` regex — which harmony never emits. Fixing the *name* is not
    enough either: vLLM's `openai_gptoss` parser raises NotImplementedError for
    non-streaming input ("gpt-oss has a special branch for parsing reasoning in
    non-streaming mode"), and this module only ever calls `LLM.generate()`
    offline. Harmony is handled by `_split_harmony` instead.
    """
    s = (model_source or "").lower()
    # Order matters — check more specific names first.
    if "gemma-4" in s or "gemma4" in s:
        return "gemma4"
    if "deepseek-r1" in s or "deepseek_r1" in s or "deepseek-v3" in s:
        return "deepseek_r1"
    if "qwen3" in s:  # covers qwen3, qwen3.5, qwen3-vl, etc.
        return "qwen3"
    # Non-thinking families: Phi-4, Llama-3.x, Gemma-3, Qwen2.5, OpenThinker (custom tags → regex).
    return None


def model_needs_reasoning_budget(model_cfg: Any) -> bool:
    """True if the model reasons before its final answer and therefore needs a
    generous ``max_tokens`` budget on short-answer benchmarks.

    Short-answer eval stages (confaide ratings, cirl_vignettes A/B, mmlu
    letters) default to a tiny ``max_tokens`` (16-64). A reasoning model spends
    that budget on hidden chain-of-thought and never emits the parseable answer,
    yielding ``parseable_rate=0`` and a sanity failure. Two independent triggers
    flag such models so callers can bump the budget:

      1. ``chat_template_kwargs.enable_thinking`` is explicitly ``False`` — vLLM
         strips ``<think>`` blocks from the output, but the model still *spends*
         tokens reasoning first.
      2. The model reasons structurally — either it ships a vLLM reasoning
         parser (qwen3, deepseek-r1) or it speaks harmony (gpt-oss). These
         reason regardless of ``enable_thinking`` — gpt-oss always emits an
         ``analysis`` channel before ``final``, which is why a bare
         ``chat_template_kwargs: {}`` config (no ``enable_thinking`` key) still
         needs the larger budget.

    Harmony must be checked *separately* from ``_detect_reasoning_parser``.
    That function deliberately returns None for gpt-oss (vLLM has no usable
    non-streaming harmony parser — see ``_split_harmony``), so keying Trigger 2
    on it alone would silently stop flagging the one family that motivated the
    trigger. And the consequence is now worse than it used to be: since
    ``_split_harmony`` honestly reports an unfinished generation as
    ``content == ""``, an under-budgeted gpt-oss returns **empty** answers on
    every short-answer benchmark rather than merely garbage ones.
    Guarded by tests/common/test_reasoning_budget.py.
    """
    # Trigger 1: enable_thinking explicitly false.
    try:
        ctk = getattr(model_cfg, "chat_template_kwargs", None)
        if ctk is None and isinstance(model_cfg, dict):
            ctk = model_cfg.get("chat_template_kwargs")
        ctk = ctk or {}
        if hasattr(ctk, "enable_thinking"):
            if not bool(ctk.enable_thinking):
                return True
        elif isinstance(ctk, dict) and "enable_thinking" in ctk:
            if not bool(ctk.get("enable_thinking")):
                return True
    except Exception:
        pass

    # Trigger 2: the model reasons structurally → always reasons.
    # Check both the model_source path AND the declared model_family: some
    # reasoning models carry a custom checkpoint name whose path hides the
    # base family (e.g. OpenThinker3-7B is a qwen3 model, but "qwen3" never
    # appears in its path — only in model_family). Sniffing the path alone
    # missed it, so it truncated its CoT inside a too-small budget.
    try:
        src = getattr(model_cfg, "model_source", None)
        if src is None and isinstance(model_cfg, dict):
            src = model_cfg.get("model_source")
        fam = getattr(model_cfg, "model_family", None)
        if fam is None and isinstance(model_cfg, dict):
            fam = model_cfg.get("model_family")
        for ident in (src, fam):
            if not ident:
                continue
            # Harmony (gpt-oss) has no vLLM parser but always reasons.
            if _is_harmony_model(str(ident)):
                return True
            if _detect_reasoning_parser(str(ident)) is not None:
                return True
    except Exception:
        pass

    return False


def _split_reasoning(
    text: str,
    model_source: str,
    thinking_enabled: bool,
    tokenizer,
) -> tuple[str, str]:
    """Split model output into ``(reasoning, content)``.

    Primary path: vLLM's family-specific ``ReasoningParser``. These parsers
    understand the exact reasoning format for each architecture (Qwen3
    ``<think>...</think>``, Gemma-4 ``thought\\n...\\n``, etc.) and are
    maintained upstream alongside each model's chat template.

    Fallback path: regex (``_fallback_strip_reasoning``) when no parser
    matches the model family, the parser fails, or the parser returns
    content that still contains raw reasoning tags.

    Args:
        text: raw decoded model output.
        model_source: path or identifier used to pick a parser.
        thinking_enabled: whether the chat template was configured with
            thinking on — passed to the parser so it classifies truncated
            output correctly (unterminated ``<think>`` is reasoning when
            enabled, content when disabled).
        tokenizer: the tokenizer used for generation (parsers need it).

    Returns:
        ``(reasoning, content)`` — either may be the empty string.
    """
    if not text:
        return "", ""

    # Harmony (gpt-oss) first: it is a channel protocol, not a <think> wrapper,
    # and vLLM's parser refuses to run outside streaming mode.
    if _is_harmony_model(model_source):
        parsed = _split_harmony(text)
        if parsed is not None:
            return parsed
        # Genuinely unstructured output (e.g. the model emitted bare prose).
        # Treat it as content — there is no reasoning channel to separate.
        return "", text.strip()

    parser_name = _detect_reasoning_parser(model_source)
    if parser_name is not None:
        try:
            from vllm.reasoning import ReasoningParserManager

            parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
            parser = parser_cls(
                tokenizer,
                chat_template_kwargs={"enable_thinking": thinking_enabled},
            )
            reasoning, content = parser.extract_reasoning(text, None)
            reasoning = (reasoning or "").strip()
            content = (content or "").strip()
            # Safety: if parser handed back content that still contains raw
            # reasoning tags, something went wrong — fall through to regex.
            if "<think>" not in content and "</think>" not in content:
                return reasoning, content
        except Exception:
            pass  # fall through

    # Fallback path.
    content = _fallback_strip_reasoning(text)
    m = re.search(r"<think>([\s\S]*?)</think>", text)
    if m:
        reasoning = m.group(1).strip()
    else:
        m2 = re.search(r"<\|begin_of_thought\|([\s\S]*?)<\|end_of_thought\|", text)
        reasoning = m2.group(1).strip() if m2 else ""
    return reasoning, content


# ── Public aliases (underscore-free names for new code) ────────────────
split_reasoning = _split_reasoning
fallback_strip_reasoning = _fallback_strip_reasoning
strip_think_blocks = _strip_think_blocks
is_harmony_model = _is_harmony_model
split_harmony = _split_harmony
detect_reasoning_parser = _detect_reasoning_parser
