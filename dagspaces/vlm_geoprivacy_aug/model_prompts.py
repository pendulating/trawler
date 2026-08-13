"""Per-family VLM prompt builders — shared with ``vlm_geoprivacy_bench``.

This dagspace adds hypothetical capture-context frames to the benchmark. It
does NOT change how a prompt renders for a model family, so it uses the
builders of ``vlm_geoprivacy_bench`` without a copy.

Before 2026-08-12 this file was a copy. The copy fell behind: it had no
``build_gemma4_prompt`` and no ``"gemma-4"`` key in ``PROMPT_BUILDERS``. The
gemma-4 configs all declare ``model_family: gemma-4``, and
``get_prompt_builder`` RAISES ``ValueError`` on an unknown family — it does
not substitute another builder. So gemma-4 could not run on this dagspace at
all; the stage stopped with "Unknown model_family 'gemma-4'".

The failure was loud, not silent. It produced no wrong output, because it
produced no output. ``vlm_geoprivacy_bench`` added the builder on 2026-07-18
and this dagspace never got it, because nobody knew that the file was a copy.

Read ``build_gemma4_prompt`` in the benchmark module for the separate reason
that gemma-4 needs its OWN builder: the gemma-3 builder passes a list-valued
system message, which Gemma4Processor renders as a Python repr rather than
text. That is a reason not to REUSE the gemma-3 builder. It is not what
happened here.
"""

from __future__ import annotations

from dagspaces.vlm_geoprivacy_bench.model_prompts import (
    PROMPT_BUILDERS,
    build_deepseek_vl2_prompt,
    build_gemma3_prompt,
    build_gemma4_prompt,
    build_internvl2_5_prompt,
    build_llama_vision_prompt,
    build_phi4mm_prompt,
    build_qwen2_5_vl_prompt,
    build_qwen3_5_prompt,
    get_prompt_builder,
)

__all__ = [
    "PROMPT_BUILDERS",
    "build_deepseek_vl2_prompt",
    "build_gemma3_prompt",
    "build_gemma4_prompt",
    "build_internvl2_5_prompt",
    "build_llama_vision_prompt",
    "build_phi4mm_prompt",
    "build_qwen2_5_vl_prompt",
    "build_qwen3_5_prompt",
    "get_prompt_builder",
]
