"""Stage runner registry and exports for historical_norms."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import StageRunner

__all__ = [
    "StageRunner",
    "get_stage_registry",
]

_STAGE_REGISTRY: dict[str, "StageRunner"] | None = None


def get_stage_registry() -> dict[str, "StageRunner"]:
    """Get the stage registry mapping stage names to runner instances."""
    global _STAGE_REGISTRY
    if _STAGE_REGISTRY is None:
        from .base import StageRunner
        from .fetch_gutenberg import FetchGutenbergRunner
        from .norm_reasoning import NormReasoningRunner
        from .norm_extraction import NormExtractionRunner
        from .ci_reasoning import CIReasoningRunner
        from .ci_extraction import CIExtractionRunner
        from .norm_consolidation import NormConsolidationRunner
        from .norm_consolidation_from_clusters import NormConsolidationFromClustersRunner
        from .norm_role_abstraction import NormRoleAbstractionRunner

        _STAGE_REGISTRY = {
            "fetch_gutenberg": FetchGutenbergRunner(),
            "norm_reasoning": NormReasoningRunner(),
            "norm_extraction": NormExtractionRunner(),
            "norm_role_abstraction": NormRoleAbstractionRunner(),
            "norm_consolidation": NormConsolidationRunner(),
            "norm_consolidation_from_clusters": NormConsolidationFromClustersRunner(),
            "ci_reasoning": CIReasoningRunner(),
            "ci_extraction": CIExtractionRunner(),
        }
    return _STAGE_REGISTRY.copy()
