"""Base runners for the benchmark evaluation dagspaces.

Every benchmark dagspace runs the same four stage shapes:

1. **load** — build a DataFrame from a source, write it.
2. **inference** — read a DataFrame, add model completions, write it.
3. **parse** — read a DataFrame, add predictions, write it, then report parse
   health.
4. **metrics** — read a DataFrame, compute a metric dict, write both a
   ``metrics.json`` and a one-row-per-metric DataFrame.

Before 2026-08-12 each dagspace wrote all four by hand. The 8 files held 1,429
lines, and 10 of the classes were identical once you removed the string
constants. The classes here hold that shared body one time.

Why not ``DataFrameStageRunner``
--------------------------------
``dagspaces/common/runners/base.py`` already has a DataFrame runner, but a
benchmark dagspace CANNOT use it. It loads through
``orchestrator.prepare_stage_input``, which:

* raises ``RuntimeError`` unless the parquet has an ``article_text`` or a
  ``chunk_text`` column. That contract comes from the pre-COLM AI-news
  project. No benchmark dataset has either column.
* re-applies ``runtime.sample_n`` at EVERY stage, with
  ``df.sample(random_state=777)``. A benchmark samples one time, in its load
  stage, so a second application would reshuffle the rows mid-pipeline.

``historical_norms`` is the only current user because novel chunks carry
``chunk_text``. The two runner families therefore stay separate: this module
reads with a plain ``pd.read_parquet`` and never re-samples.

How to use
----------
Set ``stage_name`` and implement the one hook of the class you pick::

    class LLMInferenceRunner(EvalStageRunner):
        stage_name = "llm_inference"

        def transform(self, df, context):
            from ..stages.llm_inference import run_llm_inference
            return run_llm_inference(df, context.cfg)

A stage with I/O that does not fit — several inputs, a non-DataFrame result,
an async export — should stay a plain ``StageRunner``. Do not force it here.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from dagspaces.common.orchestrator import StageResult
from dagspaces.common.runners.base import StageRunner

__all__ = [
    "EvalLoadRunner",
    "EvalMetricsRunner",
    "EvalParseRunner",
    "EvalStageRunner",
    "runtime_sample_n",
    "write_dataset",
]


def write_dataset(df: pd.DataFrame, out_path: str) -> str:
    """Write ``df`` to ``out_path`` as parquet and make the parent directory.

    Returns:
        ``out_path``, so a caller can write ``outputs={"dataset": write_dataset(...)}``.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def runtime_sample_n(cfg: Any) -> int | None:
    """Read ``runtime.sample_n`` from a config, as an int or ``None``.

    Every benchmark load stage needs this value, and each one used to read it
    with the same six lines.
    """
    runtime = getattr(cfg, "runtime", None)
    if not runtime:
        return None
    sample_n = getattr(runtime, "sample_n", None)
    return None if sample_n is None else int(sample_n)


class EvalStageRunner(StageRunner):
    """Read one parquet, transform it, write one parquet.

    Subclasses set ``stage_name`` and implement :meth:`transform`.
    """

    input_key: str = "dataset"
    output_key: str = "dataset"

    def transform(self, df: pd.DataFrame, context: Any) -> pd.DataFrame:
        """Return the output DataFrame. Override this."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement transform()"
        )

    def stage_metadata(self, context: Any, df: pd.DataFrame) -> dict[str, Any]:
        """Extra metadata keys for this stage, e.g. ``{"tier": "2a"}``.

        Args:
            context: The stage execution context.
            df: The DataFrame this stage is about to write. A stage that
                summarises its own output (a per-step parse rate, a class
                count) reads it here.

        The base ``rows`` key is always present and does not come from here.
        """
        return {}

    def run(self, context: Any) -> StageResult:
        df = pd.read_parquet(context.inputs[self.input_key])
        input_n = len(df)

        result_df = self.transform(df, context)
        out_path = write_dataset(result_df, context.output_paths[self.output_key])

        metadata: dict[str, Any] = {"rows": len(result_df)}
        metadata.update(self.stage_metadata(context, result_df))

        outputs = {self.output_key: out_path}
        self.after_write(context, result_df, input_n, metadata, outputs)
        return StageResult(outputs=outputs, metadata=metadata)

    def after_write(
        self,
        context: Any,
        result_df: pd.DataFrame,
        input_n: int,
        metadata: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        """Hook that runs after the write, before the result returns.

        ``metadata`` and ``outputs`` are the live dicts, so a subclass can add
        to either. :class:`EvalParseRunner` uses this for the health report.
        """
        return None


class EvalLoadRunner(StageRunner):
    """Build a DataFrame from a source and write it. Takes no dataset input.

    Subclasses set ``stage_name`` and implement :meth:`load`.
    """

    output_key: str = "dataset"

    def load(self, context: Any) -> pd.DataFrame:
        """Return the loaded DataFrame. Override this."""
        raise NotImplementedError(f"{type(self).__name__} must implement load()")

    def stage_metadata(self, context: Any, df: pd.DataFrame) -> dict[str, Any]:
        """Extra metadata keys, e.g. ``{"k_shot": 5}`` or a per-class count."""
        return {}

    def run(self, context: Any) -> StageResult:
        df = self.load(context)
        out_path = write_dataset(df, context.output_paths[self.output_key])

        metadata: dict[str, Any] = {"rows": len(df)}
        metadata.update(self.stage_metadata(context, df))
        return StageResult(outputs={self.output_key: out_path}, metadata=metadata)


class EvalParseRunner(EvalStageRunner):
    """A transform stage that also reports parse health.

    Subclasses set ``stage_name``, ``health_dagspace``, and implement
    :meth:`transform`. Set ``label_col`` when the stage writes ONE prediction
    column and the health report should read it.
    """

    health_dagspace: str = ""
    status_col: str = "parse_status"
    completion_col: str = "generated_text"
    # These four defaults MATCH ``compute_parse_health``'s own defaults. Keep
    # them matched: a subclass that sets nothing must get the library
    # behavior, not a value invented here. ``label_col=None`` is what the
    # vlm_geoprivacy parse stages rely on.
    label_col: str | None = None
    finish_reason_col: str | None = "finish_reason"

    def health_stage(self, context: Any) -> str:
        """Name this stage carries in the health report.

        Override when the report must distinguish a task or a tier, e.g.
        ``f"{self.stage_name}_{tier}"``.
        """
        return self.stage_name

    def after_write(
        self,
        context: Any,
        result_df: pd.DataFrame,
        input_n: int,
        metadata: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        from dagspaces.common.eval_sanity import compute_parse_health
        from dagspaces.common.runners.sanity import (
            log_sanity_to_context,
            sanity_overrides,
            task_model_name,
        )

        if not self.health_dagspace:
            raise ValueError(
                f"{type(self).__name__} must set health_dagspace — it names "
                f"this dagspace in the parse-health report."
            )

        thresholds, patterns = sanity_overrides(context.cfg)
        report = compute_parse_health(
            result_df,
            dagspace=self.health_dagspace,
            stage=self.health_stage(context),
            model=task_model_name(context.cfg),
            status_col=self.status_col,
            completion_col=self.completion_col,
            label_col=self.label_col,
            finish_reason_col=self.finish_reason_col,
            expected_input_n=input_n,
            refusal_patterns=patterns,
            thresholds=thresholds,
        )
        # log_sanity_to_context adds its own keys to `metadata`, so it must run
        # after stage_metadata() has filled it. EvalStageRunner.run guarantees
        # that order.
        log_sanity_to_context(context, report, metadata=metadata)


class EvalMetricsRunner(StageRunner):
    """Compute metrics, then write both ``metrics.json`` and a DataFrame.

    Subclasses set ``stage_name`` and implement :meth:`compute` and
    :meth:`to_dataframe`.

    This runner does NOT log the metrics itself. It puts them under the
    ``metrics`` key of the result metadata, and the orchestrator logs them
    from there, in ``_log_eval_metrics``.
    """

    input_key: str = "dataset"
    output_key: str = "dataset"
    metrics_json_name: str = "metrics.json"

    def compute(self, df: pd.DataFrame, context: Any) -> dict[str, Any]:
        """Return the metric dict. Override this."""
        raise NotImplementedError(f"{type(self).__name__} must implement compute()")

    def to_dataframe(self, metrics: dict[str, Any]) -> pd.DataFrame:
        """Turn the metric dict into a DataFrame. Override this."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement to_dataframe()"
        )

    def stage_metadata(self, context: Any, df: pd.DataFrame) -> dict[str, Any]:
        """Extra metadata keys. ``rows`` and ``metrics`` are always present."""
        return {}

    def run(self, context: Any) -> StageResult:
        df = pd.read_parquet(context.inputs[self.input_key])
        metrics = self.compute(df, context)

        metrics_json_path = os.path.join(context.output_dir, self.metrics_json_name)
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        metrics_df = self.to_dataframe(metrics)
        out_path = write_dataset(metrics_df, context.output_paths[self.output_key])

        metadata: dict[str, Any] = {"rows": len(metrics_df), "metrics": metrics}
        metadata.update(self.stage_metadata(context, metrics_df))
        return StageResult(
            outputs={self.output_key: out_path, "metrics_json": metrics_json_path},
            metadata=metadata,
        )
