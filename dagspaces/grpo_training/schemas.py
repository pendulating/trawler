"""Pydantic schemas for GRPO reward validation.

These schemas mirror the CI extraction schemas from historical_norms/ci_schema.py
and are used by the reward functions to validate GRPO completions.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# CI information flow schemas (matches historical_norms/ci_schema.py)
# ---------------------------------------------------------------------------

class CIReasoningEntry(BaseModel):
    original_text_snippet: str
    reasoning: str
    context_identified: str
    flow_direction: str
    potential_appropriateness: Literal["appropriate", "inappropriate", "ambiguous"]
    is_new_flow: bool


class CIReasoningList(BaseModel):
    flows: List[CIReasoningEntry] = Field(default_factory=list, max_length=10)
    has_information_exchange: bool


class InformationFlowTuple(BaseModel):
    subject: Optional[str] = None
    sender: str
    recipient: str
    information_type: str
    transmission_principle: str


class ContextualIntegrityFlow(BaseModel):
    flow: InformationFlowTuple
    context: str
    appropriateness: Literal["appropriate", "inappropriate", "ambiguous"]
    norms_invoked: List[str] = Field(default_factory=list)
    norm_source: Literal["explicit", "implicit", "both"]
    is_new_flow: bool = False
    confidence_qual: Literal[
        "very_uncertain", "uncertain", "somewhat_certain", "certain", "very_certain"
    ]
    confidence_quant: int = Field(ge=0, le=10)


# ---------------------------------------------------------------------------
# Combined completion schema (SFT/GRPO single-pass output)
# ---------------------------------------------------------------------------

class CICompletionResult(BaseModel):
    """Schema for a combined reasoning-then-extraction completion."""
    reasoning: CIReasoningList
    extraction: List[ContextualIntegrityFlow] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Reward judge output schema
# ---------------------------------------------------------------------------

class FlowGovernanceJudgment(BaseModel):
    """Per-flow R_ground judge output with two decomposable signals.

    norm_match_score: Does the model's norms_invoked match retrieved norms?
    governance_score: Is this flow governed by the retrieved norms?
    """
    norm_match: bool = Field(
        description="Whether any of the flow's norms_invoked semantically match a retrieved norm",
    )
    norm_match_score: float = Field(
        ge=0.0, le=1.0,
        description="How well the invoked norms match retrieved norms (0=no match, 1=strong match)",
    )
    matched_norm: Optional[str] = Field(
        None,
        description="The retrieved norm that best matches the invoked norms, or null",
    )
    flow_governed: bool = Field(
        description="Whether this information flow is governed by at least one retrieved norm",
    )
    governance_score: float = Field(
        ge=0.0, le=1.0,
        description="How well the flow is governed by the retrieved norms (0=unrelated, 1=directly governed)",
    )
    governing_norm: Optional[str] = Field(
        None,
        description="The norm that most directly governs this flow, or null",
    )
    appropriateness_consistent: bool = Field(
        description="Whether the appropriateness judgment is consistent with the governing norm",
    )
    explanation: str = Field(
        description="Brief explanation of norm matching and governance assessment",
    )


class NoFlowCoverageJudgment(BaseModel):
    """Judge output for assessing whether a text passage contains
    information flows governed by norms from the normative universe.

    Used by OnlineRGround to score no-flow completions: if the passage
    clearly contains governed flows but the model declared none, R_ground
    should be low.
    """
    passage_contains_governed_flows: bool = Field(
        description="Whether the passage describes information flows governed by the provided norms",
    )
    coverage_score: float = Field(
        ge=0.0, le=1.0,
        description="How strongly the passage's information flows are governed by the norms "
                    "(0=no governed flows, 1=clear governed flows present)",
    )
    explanation: str = Field(
        description="Brief explanation of the coverage assessment",
    )


class RankedCompletion(BaseModel):
    """One candidate's position in a within-group grounding ranking."""
    model_config = ConfigDict(extra="forbid")

    candidate_index: int = Field(
        ge=0,
        description="0-based index of the candidate, as numbered in the prompt",
    )
    rank: int = Field(
        ge=1,
        description="1 = best grounded in the provided norms; no ties allowed",
    )
    grounding_score: float = Field(
        ge=0.0, le=1.0,
        description="Absolute normative grounding of this candidate against "
                    "the provided norms (0=ungrounded, 1=fully grounded)",
    )


class CompletionRankingJudgment(BaseModel):
    """Listwise R_ground judge output: rank all candidate completions of the
    same passage by how well their extractions are grounded in the retrieved
    norms. Forces tie-breaking that per-candidate absolute scoring cannot —
    the May 2026 sweep showed 60% of G-completion groups received identical
    absolute R_ground (zero GRPO advantage).
    """
    model_config = ConfigDict(extra="forbid")

    # No explanation/justification field: it is never consumed, and under
    # guided decoding a required free-text field both dominates generation
    # latency and truncates the response at max_tokens (observed 2026-06-10:
    # ~100s/call and mid-array cutoffs at 1024 tokens). The ranking criteria
    # live in the prompt; emission order (rankings-first) means a trailing
    # explanation could not have served as reasoning anyway.
    rankings: List[RankedCompletion] = Field(
        description="Exactly one entry per candidate, each with a distinct rank",
    )


class NormJudgmentResult(BaseModel):
    """Expected output schema for norm judgment vignettes.

    The model is asked whether an action is appropriate in a given
    social context, grounding its judgment in privacy/information norms.
    """
    judgment: Literal["yes", "no"] = Field(
        description="Whether the described action is appropriate in the given context",
    )
    reasoning: str = Field(
        description="Explanation of why the action is or is not appropriate",
    )
    norms_considered: List[str] = Field(
        default_factory=list,
        description="Norms of information sharing considered in the judgment",
    )


# Keep backward-compatible alias
NormGroundingJudgment = FlowGovernanceJudgment
