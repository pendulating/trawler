"""Contextual Integrity (CI) information flow schemas.

Pydantic models for extracting Nissenbaum's CI information flow tuples
from text. Separate from the IG 2.0 schemas in schema.py.
"""

from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Reasoning stage schemas
# ---------------------------------------------------------------------------

class CIReasoningEntry(BaseModel):
    """Single reasoning entry for CI flow identification."""
    original_text_snippet: str = Field(
        ...,
        description="The exact quote from the text containing the information exchange",
    )
    reasoning: str = Field(
        ...,
        description="Detailed explanation of the context, actors, and information exchange pattern",
    )
    context_identified: str = Field(
        ...,
        description="The societal context or sphere of life (e.g., courtship, family, legal, commerce, social etiquette)",
    )
    flow_direction: str = Field(
        ...,
        description="Brief description of who transmits what information to whom",
    )
    potential_appropriateness: Literal["appropriate", "inappropriate", "ambiguous"] = Field(
        ...,
        description="Whether the information flow conforms to or violates the norms of its context",
    )
    is_new_flow: bool = Field(
        default=False,
        description=(
            "True if this represents a socially-novel form of information "
            "transmission for which no established norms yet exist in the "
            "depicted society or tradition. Typically paired with an "
            "'inappropriate' or 'ambiguous' appropriateness judgment."
        ),
    )


class CIReasoningList(BaseModel):
    """Top-level reasoning output for CI flow identification.

    Analogous to NormReasoningList in schema.py but focused on
    information exchanges rather than institutional statements.
    """
    flows: List[CIReasoningEntry] = Field(
        ...,
        description=(
            "Information flows identified in the text. "
            "Output an empty array [] if no information exchanges are found. "
            "Limit to 1-5 most significant flows per chunk."
        ),
        max_length=10,
    )
    has_information_exchange: bool = Field(
        ...,
        description="Whether the text contains any exchange, disclosure, or withholding of information between agents",
    )


# ---------------------------------------------------------------------------
# Extraction stage schemas
# ---------------------------------------------------------------------------

class InformationFlowTuple(BaseModel):
    """Nissenbaum's 5-component CI information flow tuple."""
    subject: Optional[str] = Field(
        None,
        description="The individual about whom information pertains (may be the same as sender)",
    )
    sender: str = Field(
        ...,
        description="The agent transmitting or disclosing the information",
    )
    recipient: str = Field(
        ...,
        description="The agent receiving the information",
    )
    information_type: str = Field(
        ...,
        description="The category or nature of the information exchanged (e.g., marital intentions, financial standing, personal reputation)",
    )
    transmission_principle: str = Field(
        ...,
        description="The norm or principle governing the flow (e.g., confidentiality, reciprocity, consent, entitlement, notice, need-to-know)",
    )


class ContextualIntegrityFlow(BaseModel):
    """Full CI extraction with contextual metadata."""
    flow: InformationFlowTuple
    context: str = Field(
        ...,
        description="The societal sphere or domain in which the exchange occurs (e.g., healthcare, courtship, legal, family, commerce)",
    )
    appropriateness: Literal["appropriate", "inappropriate", "ambiguous"] = Field(
        ...,
        description="Whether the flow is treated as appropriate or inappropriate within its context",
    )
    norms_invoked: List[str] = Field(
        default_factory=list,
        description="Explicit or implicit norms that govern the appropriateness of this flow",
    )
    norm_source: Literal["explicit", "implicit", "both"] = Field(
        ...,
        description="Whether the norms are stated explicitly in the text, implied, or both",
    )
    is_new_flow: bool = Field(
        default=False,
        description=(
            "True if this represents a socially-novel form of information "
            "transmission for which no established norms yet exist in the "
            "depicted society or tradition"
        ),
    )
    confidence_qual: Literal[
        "very_uncertain", "uncertain", "somewhat_certain", "certain", "very_certain"
    ] = Field(
        ...,
        description="Qualitative extraction confidence (5-point Likert)",
    )
    confidence_quant: int = Field(
        ...,
        ge=0,
        le=10,
        description="Numeric extraction confidence (0-10)",
    )


class CIExtractionResult(BaseModel):
    """Top-level extraction output for CI information flows."""
    information_flows: List[ContextualIntegrityFlow] = Field(
        ...,
        description="All extracted contextual integrity information flows",
    )


# ---------------------------------------------------------------------------
# Raz norm extraction schemas (prescriptive texts)
# ---------------------------------------------------------------------------

class RazNormTuple(BaseModel):
    """Raz's 4-component anatomy of a norm."""
    prescriptive_element: str = Field(
        ...,
        description=(
            "The deontic 'ought': the sense in which the action is prescribed, "
            "prohibited, or permitted (e.g., 'must', 'must not', 'shall', "
            "'is forbidden to', 'ought to')"
        ),
    )
    norm_subject: str = Field(
        ...,
        description=(
            "The role or class of persons upon whom the obligation falls "
            "(e.g., 'the believer', 'a parent', 'every witness')"
        ),
    )
    norm_act: str = Field(
        ...,
        description=(
            "The action prescribed or proscribed, stated as a verb phrase "
            "(e.g., 'bear false testimony', 'teach scripture to one's children')"
        ),
    )
    condition_of_application: Optional[str] = Field(
        None,
        description=(
            "The circumstances under which the norm applies. "
            "Null if the norm is unconditional."
        ),
    )


class PrescriptiveNormExtraction(BaseModel):
    """A single norm extracted from prescriptive text, with metadata."""
    norm: RazNormTuple
    context: str = Field(
        ...,
        description=(
            "The societal sphere or domain the norm governs "
            "(e.g., worship/ritual, legal/judicial, family, communal life)"
        ),
    )
    normative_force: Literal[
        "obligatory", "prohibited", "permitted", "recommended", "discouraged"
    ] = Field(
        ...,
        description="Deontic classification of the norm",
    )
    norm_articulation: str = Field(
        ...,
        description=(
            "The norm stated as a complete sentence, as the tradition itself "
            "would articulate it"
        ),
    )
    norm_source: Literal["explicit", "implicit", "both"] = Field(
        ...,
        description="Whether the norm is directly stated, implied, or both",
    )
    governs_information_flow: bool = Field(
        ...,
        description=(
            "True if the prescribed act regulates transmission, disclosure, "
            "concealment, or withholding of information between agents"
        ),
    )
    information_flow_note: Optional[str] = Field(
        None,
        description=(
            "If governs_information_flow is true, a brief CI-vocabulary "
            "description of the flow the norm constrains (sender, recipient, "
            "information type). Null otherwise."
        ),
    )
    confidence_qual: Literal[
        "very_low", "low", "moderate", "high", "very_high"
    ] = Field(
        ...,
        description="Qualitative extraction confidence (5-point Likert)",
    )
    confidence_quant: int = Field(
        ...,
        ge=0,
        le=10,
        description="Numeric extraction confidence (0-10)",
    )
    source_snippet: str = Field(
        ...,
        description="The exact quote from the source text containing the norm",
    )
    reasoning_trace: str = Field(
        ...,
        description="The reasoning chain produced by the model for this norm",
    )


class PrescriptiveNormExtractionResult(BaseModel):
    """Top-level extraction output for Raz norms from prescriptive texts."""
    norms: List[PrescriptiveNormExtraction] = Field(
        ...,
        description="All extracted norms (Raz anatomy) from the text",
    )


# ---------------------------------------------------------------------------
# Raz norm reasoning schemas (prescriptive texts)
# ---------------------------------------------------------------------------

class RazNormReasoning(BaseModel):
    """Single reasoning entry for Raz norm identification in prescriptive texts.

    Field names `original_text_snippet` and `reasoning` are preserved
    because the extraction stage depends on them via column mapping in
    norm_reasoning.py (reasoning_trace ← reasoning, norm_snippet ←
    original_text_snippet).
    """
    original_text_snippet: str = Field(
        ...,
        description="The exact quote from the text containing the norm",
    )
    reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning through Raz's lens: norm subject, "
            "prescriptive element, norm act, condition of application, "
            "normative force, and societal context"
        ),
    )
    preliminary_normative_force: Literal[
        "obligatory", "prohibited", "permitted", "recommended", "discouraged"
    ] = Field(
        ...,
        description="Initial deontic classification of the norm",
    )
    governs_information_flow: bool = Field(
        ...,
        description=(
            "True if the norm's prescribed act regulates transmission, "
            "disclosure, concealment, or withholding of information "
            "between agents; false for non-informational conduct norms"
        ),
    )


class RazNormReasoningList(BaseModel):
    """Top-level reasoning output for Raz norm identification.

    Replaces CIReasoningList / NormReasoningList for prescriptive-text
    pipelines that use Raz's anatomy of norms.
    """
    norms: List[RazNormReasoning] = Field(
        ...,
        description=(
            "Norms identified in the text. "
            "Output an empty array [] if no prescriptive norms are found. "
            "Limit to 1-5 most significant norms per chunk."
        ),
        max_length=10,
    )
    has_prescriptive_content: bool = Field(
        ...,
        description=(
            "Whether the text contains any prescriptive statements — "
            "norms about what agents ought or ought not to do"
        ),
    )
