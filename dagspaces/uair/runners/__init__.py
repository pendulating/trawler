"""Stage runner registry and exports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import StageRunner

__all__ = [
    "StageRunner",
    "ClassificationEUActRunner",
    "ClassificationRelevanceRunner",
    "ClassificationRisksBenefitsRunner",
    "DecomposeRunner",
    "DecomposeNBLRunner",
    "GroundedSummaryRunner",
    "SynthesisRunner",
    "TaxonomyRunner",
    "TopicRunner",
    "VerificationRunner",
    "VerificationNBLRunner",
    "get_stage_registry",
]

# Lazy import to avoid circular dependency with orchestrator
# The registry is built lazily when get_stage_registry() is first called
_STAGE_REGISTRY: dict[str, "StageRunner"] | None = None


def get_stage_registry() -> dict[str, "StageRunner"]:
    """Get the stage registry mapping stage names to runner instances."""
    global _STAGE_REGISTRY
    if _STAGE_REGISTRY is None:
        # Import runners lazily to break circular dependency
        from .base import StageRunner
        from .classification_eu_act import ClassificationEUActRunner
        from .classification_relevance import ClassificationRelevanceRunner
        from .classification_risks_benefits import ClassificationRisksBenefitsRunner
        from .decompose import DecomposeRunner
        from .decompose_nbl import DecomposeNBLRunner
        from .grounded_summary import GroundedSummaryRunner
        from .synthesis import SynthesisRunner
        from .taxonomy import TaxonomyRunner
        from .topic import TopicRunner
        from .verification import VerificationRunner
        from .verification_nbl import VerificationNBLRunner
        
        _STAGE_REGISTRY = {
            "classify_relevance": ClassificationRelevanceRunner(),
            "classify_eu_act": ClassificationEUActRunner(),
            "classify_risk_and_benefits": ClassificationRisksBenefitsRunner(),
            "decompose": DecomposeRunner(),
            "decompose_nbl": DecomposeNBLRunner(),
            "taxonomy": TaxonomyRunner(),
            "topic": TopicRunner(),
            "verification": VerificationRunner(),
            "verify_nbl": VerificationNBLRunner(),
            "synthesis": SynthesisRunner(),
            "grounded_summary": GroundedSummaryRunner(),
        }
    return _STAGE_REGISTRY.copy()

