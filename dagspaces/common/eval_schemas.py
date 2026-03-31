"""Pydantic schemas for structured decoding in evaluation dagspaces.

Each schema defines the exact JSON structure that vLLM's guided decoding
enforces during generation. Call `.model_json_schema()` to get the JSON
Schema dict for passing to `sampling_params["guided_decoding"]["json"]`.

Usage in a preprocess() function:
    from dagspaces.common.eval_schemas import ComplianceResult
    sp_dict["guided_decoding"] = {"json": ComplianceResult.model_json_schema()}
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


# ---------------------------------------------------------------------------
# GoldCoin-HIPAA
# ---------------------------------------------------------------------------

class ApplicabilityResult(BaseModel):
    """Whether the HIPAA Privacy Rule is applicable to a given case."""
    classification: Literal["Applicable", "Not Applicable"] = Field(
        ..., description="Whether the HIPAA Privacy Rule is applicable to the case",
    )
    reasoning: str = Field(
        ..., description="Brief explanation of the classification",
    )


class ComplianceResult(BaseModel):
    """Whether the HIPAA Privacy Rule permits or forbids the described action."""
    classification: Literal["Permit", "Forbid"] = Field(
        ..., description="Whether the HIPAA Privacy Rule permits or forbids the action",
    )
    reasoning: str = Field(
        ..., description="Brief explanation of the classification",
    )


# ---------------------------------------------------------------------------
# PrivacyLens
# ---------------------------------------------------------------------------

class YesNoResult(BaseModel):
    """A Yes/No answer with reasoning. Used for QA probing and leakage judging."""
    answer: Literal["Yes", "No"] = Field(
        ..., description="Yes or No",
    )
    reasoning: str = Field(
        ..., description="Brief rationale for the answer",
    )


class HelpfulnessResult(BaseModel):
    """Helpfulness score for a generated agent action (PrivacyLens 4-point scale)."""
    score: Literal[0, 1, 2, 3] = Field(
        ..., description="Helpfulness: 0=Poor, 1=Unsatisfactory, 2=Good, 3=Excellent",
    )
    reasoning: str = Field(
        ..., description="Step-by-step reasoning for the score",
    )


# ---------------------------------------------------------------------------
# VLM-GeoPrivacy
# ---------------------------------------------------------------------------

class MCQResult(BaseModel):
    """Answers for the 7 multiple-choice geolocation privacy questions."""
    Q1: str = Field(..., description="Answer label for Q1")
    Q2: str = Field(..., description="Answer label for Q2")
    Q3: str = Field(..., description="Answer label for Q3")
    Q4: str = Field(..., description="Answer label for Q4")
    Q5: str = Field(..., description="Answer label for Q5")
    Q6: str = Field(..., description="Answer label for Q6")
    Q7: str = Field(..., description="Answer label for Q7")


class GranularityJudgeResult(BaseModel):
    """Granularity level classification for a geolocation response."""
    label: Literal["A", "B", "C", "D"] = Field(
        ..., description="Granularity level from A (most specific) to D (least specific)",
    )
