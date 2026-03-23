"""Contextual Integrity (CI) information flow schemas.

Pydantic models for extracting Nissenbaum's CI information flow tuples
from text. Separate from the IG 2.0 schemas in schema.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Literal
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

    The ``reasoning`` field is placed first so that guided decoding
    produces the overall assessment *before* the model commits to the
    flows list (chain-of-thought ordering).
    """
    reasoning: str = Field(
        ...,
        description=(
            "Overall reasoning about the passage. Explain what the text "
            "describes and whether it contains information exchanges between "
            "agents. If no information flows are found, explain why — e.g., "
            "the passage is purely descriptive, scene-setting, internal "
            "monologue, or action without information transfer."
        ),
    )
    flows: List[CIReasoningEntry] = Field(
        ...,
        description=(
            "Information flows identified in the text. "
            "Output an empty array [] if no information exchanges are found. "
            "Provide up to 10 flows per chunk."
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
    """Top-level extraction output for a single CI information flow.

    Each extraction call targets exactly one flow identified during the
    reasoning stage (rows are expanded 1:1 per flow before extraction),
    so the schema constrains output to a single flow.
    """
    flow: ContextualIntegrityFlow = Field(
        ...,
        description="The extracted contextual integrity information flow",
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
            "The social role or class of persons upon whom the obligation falls. "
            "MUST be a social role, NEVER a named character. "
            "Examples: 'a gentleman', 'an unmarried woman of marriageable age', "
            "'a parent', 'a host', 'a servant in a gentleman's household'. "
            "NEVER use character names like 'Elizabeth', 'Mr. Darcy', 'Mrs. Bennet'."
        ),
    )
    norm_act: str = Field(
        ...,
        description=(
            "The generalizable action prescribed or proscribed, stated as a verb "
            "phrase. Must describe a recurring type of action, not a one-time plot "
            "event. Examples: 'receive a proposal with courtesy', 'call on new "
            "neighbors', 'supervise the social conduct of one's unmarried daughters'. "
            "NEVER reference specific characters or scene-specific plot events."
        ),
    )
    condition_of_application: Optional[str] = Field(
        None,
        description=(
            "The recurring social circumstances under which the norm applies. "
            "Must describe a situation that could arise across multiple scenes, "
            "families, or social occasions — not a scene-specific plot event. "
            "Examples: 'at a formal social gathering', 'when receiving a proposal "
            "of marriage', 'while a guest is convalescing in one's household'. "
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


class PrescriptiveNormExtractionResult(BaseModel):
    """Top-level extraction output for Raz norms from prescriptive texts."""
    norms: List[PrescriptiveNormExtraction] = Field(
        ...,
        description="All extracted norms (Raz anatomy) from the text",
    )


# ---------------------------------------------------------------------------
# Norm consolidation schemas
# ---------------------------------------------------------------------------

class ConsolidatedNormTuple(BaseModel):
    """Canonical merged Raz norm tuple produced by consolidation."""
    prescriptive_element: str = Field(
        ...,
        description="The deontic 'ought' that best captures the shared force of the cluster",
    )
    norm_subject: str = Field(
        ...,
        description="The role or class of persons, abstracted to cover all cluster members",
    )
    norm_act: str = Field(
        ...,
        description="The action prescribed or proscribed, merged into a single verb phrase",
    )
    condition_of_application: Optional[str] = Field(
        None,
        description="The broadest condition satisfied by all cluster members, or null",
    )


class AbstractionMapEntry(BaseModel):
    """Shows how a specific Raz component was abstracted during merge."""
    canonical: str = Field(
        ...,
        description="The canonical (merged) term chosen for this component",
    )
    originals: List[str] = Field(
        ...,
        description="The original terms from cluster members that were abstracted",
    )


class ConsolidatedNormResult(BaseModel):
    """LLM output for merging a cluster of similar norms."""
    canonical_norm: ConsolidatedNormTuple = Field(
        ...,
        description="The merged 4-component Raz norm tuple",
    )
    canonical_articulation: str = Field(
        ...,
        description="The merged norm stated as a complete sentence",
    )
    normative_force: Literal[
        "obligatory", "prohibited", "permitted", "recommended", "discouraged"
    ] = Field(
        ...,
        description="Deontic classification of the canonical norm (majority of cluster)",
    )
    context: str = Field(
        ...,
        description="Societal sphere of the canonical norm",
    )
    governs_information_flow: bool = Field(
        ...,
        description="Whether the canonical norm regulates information exchange",
    )
    information_flow_note: Optional[str] = Field(
        None,
        description="CI-vocabulary description if governs_information_flow is true",
    )
    abstraction_map: Optional[Dict[str, AbstractionMapEntry]] = Field(
        None,
        description=(
            "Map of Raz components where abstraction occurred. "
            "Keys are component names (norm_subject, norm_act, etc.), "
            "values show canonical vs original terms. "
            "Null if no abstraction was needed."
        ),
    )
    consolidation_rationale: str = Field(
        ...,
        description="1-3 sentences explaining why these norms were merged and what nuance was lost",
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
            "Provide up to 10 norms per chunk."
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


# ---------------------------------------------------------------------------
# Role abstraction schemas
# ---------------------------------------------------------------------------

class RoleAbstractedNorm(BaseModel):
    """A norm rewritten with functional social roles instead of character names.

    Only includes fields the LLM actually rewrites.  Metadata fields
    (normative_force, context, confidence, etc.) are preserved from the
    extraction stage by the orchestration code — the LLM does not output them.
    """
    norm_subject: str = Field(
        ...,
        description=(
            "The functional social role rewritten from any character name. "
            "Must capture: (1) social position, (2) relational context, "
            "(3) functional capacity that makes the norm binding. "
            "Example: 'a mother of unmarried daughters whose social duty "
            "includes securing advantageous matches'. "
            "MUST NOT contain any character names or place names."
        ),
    )
    norm_act: str = Field(
        ...,
        description=(
            "The prescribed action, rewritten only if it contained character "
            "names or plot-specific references. Must be a generalizable verb phrase."
        ),
    )
    condition_of_application: Optional[str] = Field(
        None,
        description=(
            "The condition rewritten to remove character names, place names, "
            "or scene-specific references. Must describe a recurring social "
            "situation. Null if unconditional."
        ),
    )
    norm_articulation: str = Field(
        ...,
        description=(
            "The norm restated as a complete sentence using the abstracted "
            "role, act, and condition. Must be name-free."
        ),
    )
    role_rationale: str = Field(
        ...,
        description=(
            "1-3 sentences explaining: (a) what social position the character "
            "occupies, (b) what relational context activates the norm, "
            "(c) what functional capacity or duty makes the norm binding. "
            "If the input was already fully abstracted, state that no rewrite was needed."
        ),
    )


class RoleAbstractionResult(BaseModel):
    """Top-level output for the role abstraction stage."""
    norm: RoleAbstractedNorm = Field(
        ...,
        description="The norm rewritten with functional social roles",
    )
