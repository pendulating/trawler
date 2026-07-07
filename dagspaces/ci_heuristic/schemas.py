"""Pydantic schemas for the CI decision-heuristic traversal state object.

One schema per heuristic step (S1..S9) for guided decoding, plus the L0/L1
monolithic outputs and the TP-elicitation probe. Mirrors the state object in
`planning/ci-heuristic-llm-experiments.md` §4 and the gold schema in
`corpus/gold_schema.json` — keep the three in sync.

Call `.model_json_schema()` for vLLM guided decoding, as in
dagspaces.common.eval_schemas.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared fragments
# ---------------------------------------------------------------------------

DepartedParameter = Literal[
    "sender", "recipient", "subject", "information_type", "transmission_principle"
]


class Flow(BaseModel):
    """One information flow (S1 unit of analysis)."""
    sender: str = Field(..., description="Entity transmitting the information")
    recipient: str = Field(..., description="Entity receiving the information")
    subject: str = Field(..., description="Entity the information is about")
    information_type: str = Field(..., description="Type of information flowing")
    medium: str = Field("", description="How the information moves (conversation, sync, post, ...)")
    novelty: str = Field("", description="What about this flow is new vs. an extension of existing practice")


# ---------------------------------------------------------------------------
# Step artifacts S1..S9
# ---------------------------------------------------------------------------

class S1Flows(BaseModel):
    """Step 1: describe the new practice in terms of information flows."""
    flows: List[Flow] = Field(..., description="Each distinct (sender, recipient, subject, info-type) flow")


class S2Context(BaseModel):
    """Step 2: identify information types, activities, purposes; link to a prevailing context."""
    domain: str = Field(..., description="The prevailing social domain (NOT a place or platform)")
    nested_contexts: List[str] = Field(default_factory=list, description="Overlapping or nested contexts with potential impact")
    activities: List[str] = Field(default_factory=list, description="Descriptive actions occurring in the practice")
    purposes: List[str] = Field(default_factory=list, description="Explanatory purposes the information is used for")
    values_ends: List[str] = Field(default_factory=list, description="The context's constitutive values, ends, and goals")


class NonhumanRole(BaseModel):
    entity: str = Field(..., description="The non-human entity (device, platform, algorithm)")
    treated_as: Literal["sender", "recipient", "instrument", "ambiguous"] = Field(
        ..., description="Role assigned to the entity in the flow analysis")


class S3Actors(BaseModel):
    """Step 3: identify information subjects, senders, and recipients."""
    senders: List[str] = Field(..., description="Entities that transmit information")
    recipients: List[str] = Field(..., description="Entities information is transmitted to")
    subjects: List[str] = Field(..., description="Entities the information pertains to")
    nonhuman_roles: List[NonhumanRole] = Field(default_factory=list)


class TransmissionPrinciple(BaseModel):
    principle: str = Field(..., description="The constraint on the flow (e.g., consent, need, secrecy, aggregation)")
    explicit: bool = Field(..., description="Codified in law/policy/statement (true) or implicit/inferred (false)")
    evidence: str = Field("", description="What in the practice indicates this principle")


class S4TransmissionPrinciples(BaseModel):
    """Step 4: identify transmission principles — conditions governing the flow."""
    transmission_principles: List[TransmissionPrinciple] = Field(
        ..., description="Every condition under which the flow ought (not) to occur")


class Norm(BaseModel):
    norm_flow: str = Field(..., description="The entrenched flow, stated with all five parameters")
    entrenchment_evidence: str = Field("", description="Why this norm counts as entrenched (law, culture, scholarship, practice)")
    departures: List[str] = Field(default_factory=list, description="Specific ways the new practice departs from this norm")
    completeness: Literal["entrenched", "contested", "incomplete"] = Field(
        ..., description="incomplete = no settled norm governs this kind of flow yet")


class S5Norms(BaseModel):
    """Step 5: locate applicable entrenched informational norms and points of departure."""
    norms: List[Norm] = Field(..., description="Applicable norms; use completeness='incomplete' when none govern")


class S6PrimaFacie(BaseModel):
    """Step 6: prima facie assessment of contextual integrity."""
    violation: Literal["yes", "no", "incomplete_norms"] = Field(
        ..., description="Prima facie violation? incomplete_norms when no entrenched norm applies")
    departed_parameters: List[DepartedParameter] = Field(
        default_factory=list, description="Which parameters depart from entrenched norms (empty if none)")
    justification: str = Field(..., description="Why, grounded in the step-5 norms; presumption favors entrenched practice")


class Factor(BaseModel):
    factor: str = Field(..., description="The specific moral/political factor affected by the practice")
    kind: str = Field(..., description="autonomy | freedom | power | justice | equality | fairness | democracy | discrimination | information_asymmetry | coercion | trust | other(<label>)")
    affected_parties: List[str] = Field(default_factory=list)
    direction: Literal["harm", "benefit", "mixed"] = Field(...)


class S7Factors(BaseModel):
    """Step 7 (Evaluation I): moral and political factors affected by the practice."""
    factors: List[Factor] = Field(..., description="Harms, threats to autonomy/freedom, power effects, justice/equality implications")


class ContextualMeaning(BaseModel):
    factor_ref: int = Field(..., description="Index into step 7's factors list")
    contextual_end: str = Field(..., description="The SPECIFIC end/value/purpose of THIS context the factor bears on")
    advances_or_undermines: Literal["advances", "undermines", "mixed"] = Field(...)
    argument: str = Field(..., description="Why the factor means what it does relative to this context's ends — not a restatement")


class S8ContextualMeaning(BaseModel):
    """Step 8 (Evaluation II): significance of the factors in light of contextual values, ends, purposes."""
    meanings: List[ContextualMeaning] = Field(...)


class S9Recommendation(BaseModel):
    """Step 9: recommendation for or against the system or practice."""
    decision: Literal["continue", "modify", "reject"] = Field(...)
    conditions: List[str] = Field(default_factory=list, description="Binding modifications/conditions if decision=modify")
    carrying_findings: List[str] = Field(..., description="The specific step 6-8 findings that carry the decision")


STEP_SCHEMAS = {
    "s1": S1Flows,
    "s2": S2Context,
    "s3": S3Actors,
    "s4": S4TransmissionPrinciples,
    "s5": S5Norms,
    "s6": S6PrimaFacie,
    "s7": S7Factors,
    "s8": S8ContextualMeaning,
    "s9": S9Recommendation,
}

STEP_ORDER = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]

# Steps where the descriptive/prescriptive firewall applies ("do not evaluate yet")
DESCRIPTIVE_STEPS = ["s1", "s2", "s3", "s4", "s5"]


# ---------------------------------------------------------------------------
# Monolithic outputs (ladder levels L0/L1)
# ---------------------------------------------------------------------------

class L0Verdict(BaseModel):
    """Zero-shot verdict, no heuristic in context."""
    violates_privacy: Literal["yes", "no"] = Field(...)
    decision: Literal["continue", "modify", "reject"] = Field(...)
    reasoning: str = Field(...)


class L1Traversal(BaseModel):
    """Single-completion traversal with the heuristic text in context."""
    s1_flows: S1Flows
    s2_context: S2Context
    s3_actors: S3Actors
    s4_transmission_principles: S4TransmissionPrinciples
    s5_norms: S5Norms
    s6_prima_facie: S6PrimaFacie
    s7_factors: S7Factors
    s8_contextual_meaning: S8ContextualMeaning
    s9_recommendation: S9Recommendation


# ---------------------------------------------------------------------------
# TP-elicitation probe (E2)
# ---------------------------------------------------------------------------

class TPElicitation(BaseModel):
    """'This information flow is fine IF ___' — each condition is a transmission principle."""
    conditions: List[str] = Field(..., description="Every distinct completion of 'this flow is fine IF ...'")
