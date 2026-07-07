"""Stage runner registry for the ci_heuristic dagspace."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dagspaces.common.runners.base import StageRunner

__all__ = ["get_stage_registry"]

_STAGE_REGISTRY: dict[str, "StageRunner"] | None = None


def get_stage_registry() -> dict[str, "StageRunner"]:
    """Get the stage registry mapping stage names to runner instances."""
    global _STAGE_REGISTRY
    if _STAGE_REGISTRY is None:
        from .eval_stages import (
            LoadCasesRunner,
            ScoreTraversalRunner,
            TPProbeRunner,
            TraverseRunner,
        )

        _STAGE_REGISTRY = {
            "load_cases": LoadCasesRunner(),
            "traverse": TraverseRunner(),
            "tp_probe": TPProbeRunner(),
            "score_traversal": ScoreTraversalRunner(),
        }
    return _STAGE_REGISTRY.copy()
