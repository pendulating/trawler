"""Deliberative structures for the evaluative heuristic steps (L5).

Where the theory is collective, a single model speaking from nowhere is
theoretically inappropriate (P4; McDonald & Forte). This module owns:

- step 5: norm-elicitation ensemble — N varied personas give expectation
  judgments; convergence = entrenched, split = contested, widespread
  unfamiliarity = incomplete; a synthesizer folds the population's
  expectations into the S5Norms artifact.
- step 7: stakeholder ensemble (Ensemble / Chain / Debate structures) with
  second-wave (perspective-valuing) combination instructions adapted from
  PLURALS; factors merged with persona provenance and alias-dedup.
- step 8: contextual-values analyst (single agent over s7 output).
- step 9: juror-style moderator over the full state.

Native-first: structures run as batched rounds inside the existing traversal
chain (cases x personas in one vLLM batch). PLURALS remains the optional
path for ANES-sampled personas (see prompt_dev/plurals_spike.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from .prompts import SYS_ANALYST
from .scoring.matchers import alias_match

# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Persona:
    id: str
    description: str
    marginalized: bool = False


# Step 5: a small varied population for norm elicitation (hand-built source;
# ANES-sampled personas are the PLURALS path, persona_source config).
NORM_POPULATION: list[Persona] = [
    Persona("retired_teacher", "a 68-year-old retired schoolteacher in a small town, cautious with technology"),
    Persona("gig_courier", "a 26-year-old gig-economy courier in a large city, on their phone all day"),
    Persona("nurse_parent", "a 41-year-old hospital nurse and parent of two teenagers"),
    Persona("small_business_owner", "a 55-year-old owner of a family hardware store"),
    Persona("cs_student", "a 20-year-old computer science undergraduate active on social platforms"),
    Persona("rural_farmer", "a 60-year-old farmer in a rural county with spotty internet"),
    Persona("recent_immigrant", "a 35-year-old recent immigrant working two service jobs", marginalized=True),
    Persona("wheelchair_user", "a 30-year-old wheelchair user who relies on delivery services", marginalized=True),
    Persona("union_electrician", "a 47-year-old unionized electrician"),
    Persona("suburban_realtor", "a 52-year-old suburban real-estate agent"),
]

# Step 7: stakeholder roles. {practice} is interpolated per case.
STAKEHOLDERS: list[Persona] = [
    Persona("subject", "a person whose information is captured by the practice — you appear in the data without having sought to"),
    Persona("operator", "the person or organization operating the capturing system — you benefit from and are responsible for it"),
    Persona("recipient_company", "an executive at the company that receives and monetizes the data"),
    Persona("regulator", "a public-sector regulator responsible for privacy and consumer protection"),
    Persona("civil_liberties", "a civil-liberties advocate focused on surveillance, speech, and assembly"),
    Persona("marginalized_resident", "a resident of an over-policed neighborhood; data-driven systems have historically been used against people like you", marginalized=True),
    Persona("undocumented_worker", "an undocumented worker for whom being identifiable in captured data carries severe consequences", marginalized=True),
]


def stakeholder_set(include_marginalized: bool = True) -> list[Persona]:
    """The step-7 panel; the McDonald-Forte test toggles the marginalized members."""
    return [p for p in STAKEHOLDERS if include_marginalized or not p.marginalized]


# ---------------------------------------------------------------------------
# Combination instructions (adapted from PLURALS templates)
# ---------------------------------------------------------------------------

SECOND_WAVE = (
    "Others affected by this practice have said:\n{prior}\n\n"
    "Respect and weigh their perspectives — do not dismiss or merely summarize "
    "them — but speak from your own experience and interests. Add what they "
    "cannot see from where they stand."
)

CRITIQUE_REVISE = (
    "A prior analysis of this practice said:\n{prior}\n\n"
    "Critique it from your perspective: what does it miss, overweight, or get "
    "wrong for people like you? Then give your own account."
)

DEBATE_INSTRUCTIONS = {
    "defender": "You believe the practice is, on balance, defensible. Make the strongest honest case for it, engaging the critic's latest points:\n{prior}",
    "critic": "You believe the practice is, on balance, harmful. Make the strongest honest case against it, engaging the defender's latest points:\n{prior}",
}


# ---------------------------------------------------------------------------
# Step 5: norm elicitation
# ---------------------------------------------------------------------------


class NormExpectation(BaseModel):
    """One persona's expectation judgment about the flow."""
    familiar: bool = Field(..., description="Do situations like this, with flows like this, already happen in life as you know it?")
    appropriate: Literal["yes", "no", "unsure"] = Field(..., description="Would people like you consider this flow appropriate/expected in this context?")
    expectation: str = Field(..., description="One sentence: how information like this normally flows in this context, as you understand it")


NORM_ELICIT_TEMPLATE = """You are {persona}.

Consider this context and information flow:
{practice}

Flows so far identified by an analyst: {flows}

Answer as yourself, about the settled expectations of people like you — not
about what would be ideal. Return JSON with fields familiar (true/false),
appropriate (yes/no/unsure), expectation (one sentence)."""


def build_norm_elicitation_prompts(
    practice_input: str, state: dict[str, Any], personas: list[Persona]
) -> list[tuple[str, str]]:
    flows = json.dumps((state.get("s1") or {}).get("flows", []))
    return [
        ("You answer only as the person described, from their lived experience.",
         NORM_ELICIT_TEMPLATE.format(persona=p.description, practice=practice_input, flows=flows))
        for p in personas
    ]


def aggregate_expectations(
    expectations: list[dict[str, Any]],
    entrenched_threshold: float = 0.8,
    incomplete_threshold: float = 0.5,
) -> dict[str, Any]:
    """Population statistics -> completeness verdict for the located norm.

    - incomplete: a majority of the population finds the flow unfamiliar
      (no settled practice to have expectations about).
    - entrenched: among familiar respondents, appropriateness agreement
      reaches the threshold in EITHER direction (a settled positive norm or
      a settled proscription both count as entrenched norms).
    - contested: familiar, but the population splits.
    """
    valid = [e for e in expectations if isinstance(e, dict) and "appropriate" in e]
    n = len(valid)
    if n == 0:
        return {"completeness": "incomplete", "n_valid": 0, "agreement": None,
                "unfamiliar_rate": None, "expectations": []}

    unfamiliar_rate = sum(not e.get("familiar", True) for e in valid) / n
    if unfamiliar_rate >= incomplete_threshold:
        completeness, agreement = "incomplete", None
    else:
        votes = [e["appropriate"] for e in valid if e["appropriate"] in ("yes", "no")]
        if not votes:
            completeness, agreement = "incomplete", None
        else:
            yes_rate = votes.count("yes") / len(votes)
            agreement = max(yes_rate, 1 - yes_rate)
            completeness = "entrenched" if agreement >= entrenched_threshold else "contested"

    return {
        "completeness": completeness,
        "n_valid": n,
        "agreement": round(agreement, 6) if agreement is not None else None,
        "unfamiliar_rate": round(unfamiliar_rate, 6),
        "expectations": [str(e.get("expectation", "")) for e in valid if e.get("expectation")],
    }


NORM_SYNTH_TEMPLATE = """Practice under analysis:
{practice}

Prior findings: {state}

A varied population was asked about settled expectations for flows like
these. Population statistics: completeness={completeness},
appropriateness-agreement={agreement}, unfamiliar-rate={unfamiliar_rate}.
Their stated expectations:
{expectations}

Perform STEP 5 of the CI decision heuristic: locate the applicable
entrenched informational norms and points of departure. Ground the norms in
the population's expectations (norms are collective, not your own view), set
each norm's completeness consistent with the population statistics, and
identify how the practice departs from them. Return the S5 JSON artifact."""


def build_norm_synthesis_prompt(
    practice_input: str, state: dict[str, Any], stats: dict[str, Any]
) -> tuple[str, str]:
    prior = {k: state[k] for k in ("s1", "s2", "s3", "s4") if k in state}
    usr = NORM_SYNTH_TEMPLATE.format(
        practice=practice_input,
        state=json.dumps(prior, indent=1),
        completeness=stats["completeness"],
        agreement=stats["agreement"],
        unfamiliar_rate=stats["unfamiliar_rate"],
        expectations="\n".join(f"- {e}" for e in stats["expectations"]) or "(none parseable)",
    )
    return SYS_ANALYST, usr


# ---------------------------------------------------------------------------
# Step 7: stakeholder ensemble
# ---------------------------------------------------------------------------

STAKEHOLDER_TEMPLATE = """You are {persona}.

The practice under discussion:
{practice}

The prima facie assessment so far: {s6}
{combination}
What does this practice threaten or promise FOR PEOPLE LIKE YOU — your
autonomy, your standing, your power relative to the operators and recipients?
Be concrete; name who is affected and how. Return the S7 JSON artifact
(factors list) containing ONLY factors that matter from your position."""


def build_stakeholder_prompt(
    practice_input: str,
    state: dict[str, Any],
    persona: Persona,
    prior_responses: list[str] | None = None,
    combination_template: str = SECOND_WAVE,
) -> tuple[str, str]:
    combination = ""
    if prior_responses:
        combination = "\n" + combination_template.format(prior="\n---\n".join(prior_responses)) + "\n"
    usr = STAKEHOLDER_TEMPLATE.format(
        persona=persona.description,
        practice=practice_input,
        s6=json.dumps(state.get("s6") or {}),
        combination=combination,
    )
    return ("You answer only as the person described, from their lived experience.", usr)


def merge_factor_artifacts(member_artifacts: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Union stakeholders' factors with alias-dedup; keep persona provenance.

    member_artifacts: [(persona_id, s7_artifact_dict), ...]. Dedup keeps the
    first occurrence and appends later personas to its affected_parties
    provenance tag rather than dropping their voice silently.
    """
    merged: list[dict[str, Any]] = []
    raised_by: list[list[str]] = []
    for pid, artifact in member_artifacts:
        for f in (artifact or {}).get("factors") or []:
            if not isinstance(f, dict):
                continue
            key = f"{f.get('kind', '')}: {f.get('factor', '')}"
            dup = next(
                (i for i, m in enumerate(merged)
                 if alias_match(key, f"{m.get('kind', '')}: {m.get('factor', '')}", threshold=0.6)),
                None,
            )
            if dup is None:
                merged.append(dict(f))
                raised_by.append([pid])
            else:
                raised_by[dup].append(pid)
    for f, pids in zip(merged, raised_by):
        f["raised_by"] = pids
    return {"factors": merged}


# ---------------------------------------------------------------------------
# Steps 8 and 9
# ---------------------------------------------------------------------------

ANALYST_S8_TEMPLATE = """Practice under analysis:
{practice}

The context and its ends (step 2): {s2}
Moral and political factors raised by stakeholders (step 7): {s7}

Perform STEP 8: for each factor worth carrying forward, state what it MEANS
in relation to THIS context's specific values, ends, and purposes — not
generic ethical significance. An argument that would read the same for any
context is not doing step 8. Return the S8 JSON artifact (meanings list,
factor_ref indexing the step-7 list)."""


def build_s8_analyst_prompt(practice_input: str, state: dict[str, Any]) -> tuple[str, str]:
    usr = ANALYST_S8_TEMPLATE.format(
        practice=practice_input,
        s2=json.dumps(state.get("s2") or {}),
        s7=json.dumps(state.get("s7") or {}),
    )
    return SYS_ANALYST, usr


MODERATOR_S9_TEMPLATE = """You are moderating a contextual-integrity determination.

Practice: {practice}

Findings:
- prima facie assessment (step 6): {s6}
- stakeholder factors (step 7): {s7}
- contextual significance (step 8): {s8}

Weigh them the way the framework requires: presumption favors entrenched
practice; a prima facie violation stands unless the factors and contextual
ends, on balance, redeem the new practice; a practice that undermines the
context's own ends cannot be redeemed by generic efficiency gains; where the
normative structure is incomplete, recommend the conditions under which the
practice could proceed while norms form. Deliver the S9 JSON artifact:
decision (continue/modify/reject), binding conditions if any, and the
specific findings that carry the decision."""


def build_moderator_prompt(practice_input: str, state: dict[str, Any]) -> tuple[str, str]:
    usr = MODERATOR_S9_TEMPLATE.format(
        practice=practice_input,
        s6=json.dumps(state.get("s6") or {}),
        s7=json.dumps(state.get("s7") or {}),
        s8=json.dumps(state.get("s8") or {}),
    )
    return SYS_ANALYST, usr
