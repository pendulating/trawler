"""Pydantic schemas for GRPO reward validation.

These schemas mirror the CI extraction schemas from historical_norms/ci_schema.py
and are used by the reward functions to validate GRPO completions.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


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

class NormGroundingJudgment(BaseModel):
    """Output schema for the R_ground LLM judge."""
    norms_match_universe: bool
    matched_universe_norms: List[str] = Field(default_factory=list)
    appropriateness_grounded: bool
    grounding_explanation: str
    score: float = Field(ge=0.0, le=1.0)
