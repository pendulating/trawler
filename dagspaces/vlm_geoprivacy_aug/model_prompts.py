"""Per-family VLM prompt builders — shared with ``vlm_geoprivacy_bench``.

This dagspace adds hypothetical capture-context frames to the benchmark. It
does NOT change how a prompt renders for a model family, so it uses the
builders of ``vlm_geoprivacy_bench`` without a copy.

Before 2026-08-12 this file was a copy. The copy fell behind: it had no
``build_gemma4_prompt``, so a gemma-4 run went to the gemma-3 builder. That
builder gives ``Gemma4Processor`` a list-valued system message, and the
processor renders the list as its Python repr. The system prompt is then
corrupt, and nothing raises an error. ``vlm_geoprivacy_bench`` corrected this
on 2026-07-18. This dagspace never got the correction, because nobody knew
that the file was a copy.
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
