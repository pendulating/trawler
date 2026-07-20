"""Prompt construction for CI heuristic traversals (ladder levels L0-L4).

Deliberative L5 prompts (steps 5/7/8 ensembles, step 9 moderator) live in
`deliberation.py` — this module covers the solo-model ladder.

All prompts return (sys_prompt, usr_prompt) string pairs; chat-template
rendering is the inference stage's job (text-only models here, so the
common chat pathways apply, not the VLM prompt builders).
"""

from __future__ import annotations

import json
from typing import Any

from .heuristic_text import FIREWALL_GUARD, FULL_HEURISTIC_TEXT, HEURISTIC_STEPS
from .schemas import DESCRIPTIVE_STEPS, STEP_ORDER

SYS_ANALYST = (
    "You are a careful analyst applying Helen Nissenbaum's Contextual Integrity "
    "(CI) decision heuristic to a novel sociotechnical practice. You follow the "
    "heuristic's steps exactly as specified, grounding every claim in the "
    "practice description and prior findings. You do not import conclusions "
    "from steps that have not happened yet."
)

SYS_PLAIN = "You are a helpful assistant."


def build_l0_prompt(practice_input: str) -> tuple[str, str]:
    """Zero-shot verdict, no heuristic in context."""
    usr = (
        f"Consider the following practice:\n\n{practice_input}\n\n"
        "Does this practice violate privacy? Should it continue as is, be "
        "modified, or be rejected? Answer as JSON with fields "
        "violates_privacy (yes/no), decision (continue/modify/reject), and reasoning."
    )
    return SYS_PLAIN, usr


def build_l1_prompt(practice_input: str) -> tuple[str, str]:
    """Single completion with the full heuristic in context."""
    usr = (
        "Apply the Contextual Integrity decision heuristic to the practice "
        "below. The heuristic:\n\n"
        f"{FULL_HEURISTIC_TEXT}\n\n"
        f"Practice under analysis:\n\n{practice_input}\n\n"
        "Work through all nine steps in order and return the complete analysis "
        "as a single JSON object with fields s1_flows, s2_context, s3_actors, "
        "s4_transmission_principles, s5_norms, s6_prima_facie, s7_factors, "
        "s8_contextual_meaning, s9_recommendation."
    )
    return SYS_ANALYST, usr


def build_step_prompt(
    practice_input: str,
    step: str,
    state: dict[str, Any],
    include_guiding_questions: bool = False,
    exemplar: str | None = None,
) -> tuple[str, str]:
    """Step-wise chain prompt (L2 without guiding questions, L3 with, L4 with exemplar).

    Args:
        practice_input: The practice description (constant across steps).
        step: One of STEP_ORDER ("s1".."s9").
        state: Accumulated prior step artifacts {step_key: parsed_json}.
        include_guiding_questions: L3+ — append Kumar et al.'s guiding questions.
        exemplar: L4 — a rendered worked example (contaminated Tier A case).
    """
    if step not in STEP_ORDER:
        raise ValueError(f"Unknown step {step!r}; expected one of {STEP_ORDER}")
    meta = HEURISTIC_STEPS[step]
    idx = STEP_ORDER.index(step) + 1

    parts = []
    if exemplar:
        parts.append(f"Worked example of a complete traversal (different practice):\n{exemplar}\n---\n")

    parts.append(f"Practice under analysis:\n\n{practice_input}\n")

    prior = {k: state[k] for k in STEP_ORDER if k in state}
    if prior:
        parts.append(
            "Findings from prior steps (treat as given):\n"
            + json.dumps(prior, indent=1)
            + "\n"
        )

    parts.append(
        f"You are performing STEP {idx} of 9 — {meta['title']}.\n"
        f"Step specification: {meta['text']}\n"
        f"Aim: {meta['goal']}"
    )

    if include_guiding_questions:
        qs = "\n".join(f"- {q}" for q in meta["guiding_questions"])
        parts.append(f"Guiding questions to consider:\n{qs}")

    if step in DESCRIPTIVE_STEPS:
        parts.append(FIREWALL_GUARD)

    parts.append(
        "Produce ONLY this step's artifact as JSON matching the requested schema."
    )

    return SYS_ANALYST, "\n\n".join(parts)


def build_tp_elicitation_prompt(flow_description: str, persona: str | None = None) -> tuple[str, str]:
    """The 'this flow is fine IF ___' probe (Kumar et al.'s TP identification method)."""
    voice = persona or "a reasonable member of the context in which this flow occurs"
    usr = (
        f"Consider the information flow: {flow_description}\n\n"
        f"From the perspective of {voice}, complete the sentence:\n"
        '"This information flow is fine IF ___."\n\n'
        "List every distinct condition that could complete the sentence. Each "
        "condition expresses one constraint on the flow. Return JSON with a "
        "'conditions' list."
    )
    return SYS_PLAIN, usr


def render_exemplar(gold: dict[str, Any]) -> str:
    """Render a Tier A gold file as an L4 few-shot exemplar (contaminated cases only)."""
    if not gold.get("meta", {}).get("contaminated", False):
        raise ValueError(
            "Refusing to render a non-contaminated gold case as an exemplar — "
            "held-out cases must never appear in prompts"
        )
    # Step artifacts are s<digit>_... keys ("steps_present" is metadata, not a step)
    body = {k: v for k, v in gold.items() if len(k) > 1 and k[0] == "s" and k[1].isdigit()}
    return (
        f"Practice: {gold['meta']['practice']}\n"
        f"Traversal:\n{json.dumps(body, indent=1)}"
    )
