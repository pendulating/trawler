"""Base classes for Trawler stage runners.

Stage runners are the interface between the orchestrator and stage implementations.
Each dagspace defines concrete runners that inherit from StageRunner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import Protocol
    
    class StageExecutionContext(Protocol):
        """Protocol for stage execution context (defined by each dagspace's orchestrator)."""
        cfg: Any
        node: Any
        inputs: Dict[str, str]
        output_paths: Dict[str, str]
        output_dir: str
        output_root: str
    
    class StageResult(Protocol):
        """Protocol for stage results (defined by each dagspace's orchestrator)."""
        outputs: Dict[str, str]
        metadata: Dict[str, Any]


class StageRunner:
    """Base class for all stage runners.
    
    Subclasses must:
    1. Set the `stage_name` class attribute
    2. Implement the `run` method
    
    Example:
        class MyStageRunner(StageRunner):
            stage_name = "mystage"
            
            def run(self, context):
                # Process data and return results
                ...
    """
    
    stage_name: str

    def run(self, context: "StageExecutionContext") -> "StageResult":
        """Execute the stage with the given context.
        
        Args:
            context: Execution context with config, inputs, outputs, etc.
            
        Returns:
            StageResult with output paths and metadata
        """
        raise NotImplementedError(
            f"StageRunner subclass {self.__class__.__name__} must implement run()"
        )
