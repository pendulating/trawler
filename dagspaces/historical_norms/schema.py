from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class RegulativeNorm(BaseModel):
    """
    IG 2.0 Regulative Statement (ADICO framework).
    Governs behavior by describing actions linked to specific actors.
    """
    statement_type: Literal["regulative"] = "regulative"
    
    # A - Attributes
    attributes: str | None = Field(None, description="Core identifiers of the actor (e.g., age, role, status)")
    
    # D - Deontic
    deontic: str = Field(..., description="Prescriptive operator (e.g., must, shall, may, is forbidden, is expected to)")
    
    # I - Aim
    aim: str = Field(..., description="The activity, goal, or outcome being regulated")
    
    # B/C - Object
    direct_object: str | None = Field(None, description="The inanimate or animate target of the action")
    indirect_object: str | None = Field(None, description="The recipient of the action or affected party")
    
    # C - Context
    activation_conditions: list[str] = Field(default_factory=list, description="Settings where the focal action applies (When/Where)")
    execution_constraints: list[str] = Field(default_factory=list, description="Qualifications on how the action is performed")
    
    # O - Or else (Vertical Nesting)
    or_else: Union[str, InstitutionalStatement] | None = Field(None, description="Sanctions or consequences for violation. Can be a nested statement.")
    
    reasoning_trace: str = Field(..., description="The full reasoning chain produced by the model for this specific norm")
    source_snippet: str = Field(..., description="The original text passage containing the norm")

class ConstitutiveNorm(BaseModel):
    """
    IG 2.0 Constitutive Statement (E-MFP-C framework).
    Defines features or properties of a system or entity.
    """
    statement_type: Literal["constitutive"] = "constitutive"
    
    # E - Constituted Entity
    constituted_entity: str = Field(..., description="What is being constituted or defined")
    
    # M - Modal
    modal: str | None = Field(None, description="Operator signaling necessity or possibility (e.g., is, is not, can, cannot)")
    
    # F - Constitutive Function
    constitutive_function: str = Field(..., description="Expression linking the entity to the institutional setting (e.g., 'counts as', 'represents')")
    
    # P - Constituting Properties
    constituting_properties: list[str] = Field(default_factory=list, description="Properties linked to the entity")
    
    # C - Context (IG 2.0 split)
    activation_conditions: list[str] = Field(default_factory=list, description="Settings where the definition applies (Triggers)")
    execution_constraints: list[str] = Field(default_factory=list, description="Qualifications on the definition")
    
    # O - Or else (Vertical Nesting)
    or_else: Union[str, InstitutionalStatement] | None = Field(None, description="Consequences of violation/invalidity. Can be a nested statement.")
    
    reasoning_trace: str = Field(..., description="The full reasoning chain produced by the model for this specific norm")
    source_snippet: str = Field(..., description="The original text passage containing the norm")

class InstitutionalStatement(BaseModel):
    """Recursive container for any IG 2.0 statement extracted from text."""
    statement: Union[RegulativeNorm, ConstitutiveNorm] = Field(..., discriminator="statement_type")
    
    # Horizontal Nesting support
    combination_operator: Literal["AND", "OR", "XOR"] | None = Field(None, description="Logical operator if combined with other statements")
    combined_with: list[InstitutionalStatement] = Field(default_factory=list, description="Other statements linked via the combination_operator")
    
    confidence: float = Field(..., ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

class ExtractionResult(BaseModel):
    """Top-level result for the extraction stage."""
    statements: list[InstitutionalStatement] = Field(..., description="All extracted institutional statements")

class NormReasoning(BaseModel):
    """Reasoning trace for a single institutional statement."""
    original_text_snippet: str = Field(..., description="The specific part of the text containing the norm")
    reasoning: str = Field(..., description="Detailed explanation of the social/historical context and normative force")
    potential_type: Literal["regulative", "constitutive"] = Field(..., description="Initial classification of the statement type")

class NormReasoningList(BaseModel):
    """List of reasoning traces found in a single text chunk."""
    norms: list[NormReasoning] = Field(
        ...,  # Required - model MUST output this array (can be empty [])
        description="Institutional statements identified in the text. Output an empty array [] if no norms found. Limit to 1-5 most significant norms per chunk.",
        max_length=10  # Hard cap to prevent excessive extraction
    )
    is_historical_context_present: bool = Field(
        ...,  # Required
        description="Whether the text contains relevant historical/social normative information"
    )

# Handle recursive type references
RegulativeNorm.model_rebuild()
ConstitutiveNorm.model_rebuild()
InstitutionalStatement.model_rebuild()
ExtractionResult.model_rebuild()
