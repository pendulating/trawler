"""Stage runner registry and exports for grpo_training."""

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
        from .grpo_training import GRPOTrainingRunner
        from .kto_training import KTOTrainingRunner
        from .norm_universe import NormUniverseRunner
        from .reward_prep import RewardPrepRunner
        from .sft_data_prep import SFTDataPrepRunner
        from .sft_training import SFTTrainingRunner

        _STAGE_REGISTRY = {
            "norm_universe": NormUniverseRunner(),
            "sft_data_prep": SFTDataPrepRunner(),
            "sft_training": SFTTrainingRunner(),
            "reward_prep": RewardPrepRunner(),
            "grpo_training": GRPOTrainingRunner(),
            "kto_training": KTOTrainingRunner(),
        }
    return _STAGE_REGISTRY.copy()
