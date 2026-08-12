"""Question data and prompt construction — extends ``vlm_geoprivacy_bench``.

The question set, the system messages, the judge prompt, and the answer parser
are the benchmark's. This module re-exports them, so there is one copy.

This dagspace adds one thing: it can "inpaint" a hypothetical capture-context
frame into the prompt. :func:`prepare_question_prompt` therefore takes an
extra ``hypothetical`` argument. With ``hypothetical=None``, or with the
baseline variant, it returns exactly what the benchmark returns.

Before 2026-08-12 this file held a copy of all the question data. The copy
also lost the parity-review notes that the benchmark carries in its own
docstring. Read ``dagspaces/vlm_geoprivacy_bench/prompts.py`` for those notes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dagspaces.vlm_geoprivacy_bench.prompts import (
    GRANULARITY_JUDGE,
    INST_FREE_FORM,
    INST_LABEL_STRICT,
    NUM_QUESTIONS,
    QUESTION_DATA,
    REFUSAL_PHRASES,
    SYS_MSG,
    parse_answers,
)
from dagspaces.vlm_geoprivacy_bench.prompts import (
    prepare_question_prompt as _prepare_base_question_prompt,
)

if TYPE_CHECKING:
    from .hypotheticals import HypotheticalVariant

__all__ = [
    "GRANULARITY_JUDGE",
    "INST_FREE_FORM",
    "INST_LABEL_STRICT",
    "NUM_QUESTIONS",
    "QUESTION_DATA",
    "REFUSAL_PHRASES",
    "SYS_MSG",
    "parse_answers",
    "prepare_question_prompt",
]


def prepare_question_prompt(
    mode: str,
    is_free_form: bool,
    include_heuristics: bool,
    enforce_format: bool = True,
    hypothetical: HypotheticalVariant | None = None,
) -> tuple[str, list[str]]:
    """Build the system prompt and the user prompt parts, with an optional frame.

    Args:
        mode: The prompt mode, e.g. ``"zs"``.
        is_free_form: True for the free-form question, False for the MCQ set.
        include_heuristics: True to add the per-question heuristics.
        enforce_format: True to add the strict JSON answer instruction.
        hypothetical: An optional capture-context variant. See
            ``hypotheticals.py``. ``None`` and the baseline variant both give
            a prompt that is byte-identical to the benchmark's.

    Returns:
        A tuple ``(sys_prompt, usr_prompts)``. The caller joins ``usr_prompts``
        into one user message.
    """
    from .hypotheticals import render_user_frame

    sys_prompt, usr_prompts = _prepare_base_question_prompt(
        mode=mode,
        is_free_form=is_free_form,
        include_heuristics=include_heuristics,
        enforce_format=enforce_format,
    )

    if hypothetical is not None and not hypothetical.is_baseline:
        if hypothetical.position == "system_suffix":
            sys_prompt = f"{sys_prompt} {hypothetical.frame}"
        else:
            # The frame goes first, before the question blocks. This is the
            # position that the copy of this function used before 2026-08-12.
            usr_prompts.insert(0, render_user_frame(hypothetical))

    return sys_prompt, usr_prompts
