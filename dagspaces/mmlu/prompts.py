"""Prompt construction for MMLU.

Implements the canonical MMLU prompt format from Hendrycks et al. (2021):

    The following are multiple choice questions (with answers) about {subject_phrase}.

    {question}
    A. {choice_a}
    B. {choice_b}
    C. {choice_c}
    D. {choice_d}
    Answer:

With optional 5-shot few-shot examples from the ``dev`` split prepended.
Few-shot is off by default — most modern instruct-tuned evaluations run
MMLU zero-shot, and few-shot inflates context cost ~6×.

The model's reply is constrained via guided JSON decoding to a single
letter ``A``/``B``/``C``/``D`` (see :data:`MMLU_LETTER_SCHEMA`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field
from typing import Literal


# ---------------------------------------------------------------------------
# MCQ schema (guided-decoded into the model's response)
# ---------------------------------------------------------------------------

class MMLUAnswer(BaseModel):
    """The model's selected option for one MMLU question."""

    answer: Literal["A", "B", "C", "D"] = Field(
        ..., description="The letter of the chosen option.",
    )


MMLU_LETTER_SCHEMA: Dict[str, Any] = MMLUAnswer.model_json_schema()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

LETTERS = ("A", "B", "C", "D")


def _subject_phrase(subject: str) -> str:
    """Render a subject name into the MMLU prompt's stem phrase.

    e.g. ``high_school_us_history`` → ``high school us history``. The
    paper uses spaces; we lowercase + replace underscores to match.
    """
    return str(subject or "").replace("_", " ").strip().lower()


def _format_one_example(
    question: str,
    choices: Sequence[str],
    *,
    answer_letter: Optional[str] = None,
) -> str:
    """Render one MMLU example block.

    Args:
        question: The question text.
        choices: 4 choices in order [A, B, C, D].
        answer_letter: If set, appends ``<letter>`` after ``Answer:`` —
            used for few-shot priming. None for the question under test.
    """
    lines = [str(question).strip()]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}. {str(choice).strip()}")
    lines.append("Answer:" + (f" {answer_letter}" if answer_letter else ""))
    return "\n".join(lines)


def build_mmlu_prompt(
    question: str,
    choices: Sequence[str],
    subject: str,
    *,
    few_shot_examples: Optional[Sequence[Dict[str, Any]]] = None,
    instruction_response_json: bool = True,
) -> str:
    """Assemble the user-turn prompt for one MMLU question.

    Args:
        question: The MCQ stem.
        choices: 4 answer choices in [A, B, C, D] order.
        subject: The MMLU subject slug (used in the header phrase).
        few_shot_examples: Optional list of dicts with keys
            ``question``, ``choices``, ``answer`` (int 0-3). When
            provided, prepended in MMLU's canonical 5-shot format.
        instruction_response_json: If True, append a one-line nudge
            telling the model to reply with the JSON schema. The
            guided_decoding path enforces this regardless, but priming
            it in text helps non-guided fallbacks and weaker models.
    """
    header = (
        f"The following are multiple choice questions (with answers) "
        f"about {_subject_phrase(subject)}."
    )

    blocks: List[str] = [header, ""]
    # Callers may pass None or a list; we avoid `or []` because a numpy
    # array (common after a parquet round-trip) raises on bool().
    examples = few_shot_examples if few_shot_examples is not None else []
    for ex in examples:
        ex_answer_idx = int(ex["answer"])
        blocks.append(_format_one_example(
            question=ex["question"],
            choices=list(ex["choices"]),
            answer_letter=LETTERS[ex_answer_idx],
        ))
        blocks.append("")

    blocks.append(_format_one_example(question=question, choices=list(choices)))

    prompt = "\n".join(blocks)
    if instruction_response_json:
        prompt += (
            '\n\nRespond with a JSON object: '
            '{"answer": "A" | "B" | "C" | "D"}.'
        )
    return prompt


__all__ = [
    "LETTERS",
    "MMLU_LETTER_SCHEMA",
    "MMLUAnswer",
    "build_mmlu_prompt",
]
