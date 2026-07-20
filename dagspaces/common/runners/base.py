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


class DataFrameStageRunner(StageRunner):
    """Generic runner for DataFrame-in → DataFrame-out stages.

    Implements the standard boilerplate that was copy-pasted across
    ~30 runners (Finding 3, wiki/jul19_refactoring.md):

    1. Read input dataset via ``prepare_stage_input``
    2. Call ``self.transform(df, cfg)``
    3. Save outputs via ``_save_stage_outputs``
    4. Collect output paths via ``_collect_outputs``
    5. Return ``StageResult`` with row-count metadata

    Subclasses only need to set ``stage_name`` and implement
    ``transform()``.  Stages with bespoke I/O (multiple outputs,
    metrics JSON, non-DataFrame results) should keep the explicit
    ``StageRunner`` form — do not force-fit them here.

    Example::

        class NormReasoningRunner(DataFrameStageRunner):
            stage_name = "norm_reasoning"

            def transform(self, df, cfg):
                from ..stages.norm_reasoning import run_norm_reasoning_stage
                return run_norm_reasoning_stage(df, cfg)
    """

    stage_name: str = ""
    input_key: str = "dataset"

    def transform(self, df: Any, cfg: Any) -> Any:
        """The actual stage logic.  Override in subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement transform()"
        )

    def run(self, context: "StageExecutionContext") -> "StageResult":
        from dagspaces.common.orchestrator import (
            StageResult,
            _collect_outputs,
            _save_stage_outputs,
            prepare_stage_input,
        )

        dataset_path = context.inputs.get(self.input_key)
        if not dataset_path:
            raise ValueError(
                f"Node '{context.node.key}' requires '{self.input_key}' input"
            )

        cfg = context.cfg
        df, _, _ = prepare_stage_input(cfg, dataset_path, self.stage_name)
        input_rows = len(df) if hasattr(df, "__len__") else 0
        print(f"[{self.stage_name}] Input: {input_rows} rows")

        out = self.transform(df, cfg)

        output_rows = len(out) if hasattr(out, "__len__") else 0
        print(f"[{self.stage_name}] Output: {output_rows} rows")
        _save_stage_outputs(out, context.output_paths)

        metadata: Dict[str, Any] = {
            "rows": output_rows,
            "input_rows": input_rows,
            "streaming": False,
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)
