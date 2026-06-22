"""Stage-runner classes for the simpleqa_verified dagspace.

One runner per stage in the pipeline yamls. Each subclasses
:class:`dagspaces.common.runners.base.StageRunner` and consumes a
:class:`StageExecutionContext`.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import pandas as pd

from dagspaces.common.runners.base import StageRunner
from dagspaces.common.orchestrator import StageResult
from dagspaces.common.eval_sanity import compute_judge_health
from dagspaces.common.runners.sanity import (
    log_sanity_to_context,
    sanity_overrides,
    task_model_name,
)


# ---------------------------------------------------------------------------
# Load + inference
# ---------------------------------------------------------------------------

class LoadDatasetRunner(StageRunner):
    stage_name = "load_dataset"

    def run(self, context: Any) -> StageResult:
        from ..stages.load_dataset import load_dataset

        cfg = context.cfg
        data_cfg = cfg.data

        sample_n = None
        runtime = getattr(cfg, "runtime", None)
        if runtime:
            sample_n = getattr(runtime, "sample_n", None)
            if sample_n is not None:
                sample_n = int(sample_n)

        df = load_dataset(
            hf_dataset=str(getattr(data_cfg, "hf_dataset", "google/simpleqa-verified")),
            hf_config=getattr(data_cfg, "hf_config", None),
            split=str(getattr(data_cfg, "split", "eval")),
            hf_token=getattr(data_cfg, "hf_token", None),
            sample_n=sample_n,
        )

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(df)},
        )


class LLMInferenceRunner(StageRunner):
    stage_name = "llm_inference"

    def run(self, context: Any) -> StageResult:
        from ..stages.llm_inference import run_llm_inference

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        result_df = run_llm_inference(df, context.cfg)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(result_df)},
        )


# ---------------------------------------------------------------------------
# Judging (live + async export)
# ---------------------------------------------------------------------------

class JudgeGradeLiveRunner(StageRunner):
    """Inline judge call — appends judge_response + verdict + parse_status."""

    stage_name = "judge_grade_live"

    def run(self, context: Any) -> StageResult:
        from ..stages.judge_grade import judge_grade_live

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        result_df = judge_grade_live(df, context.cfg)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        # Judge-stage sanity: valid_labels are {correct, incorrect,
        # not_attempted}; "unparseable" rows count toward judge_unparseable_rate.
        thresholds, _patterns = sanity_overrides(context.cfg)
        report = compute_judge_health(
            result_df,
            dagspace="simpleqa_verified",
            stage=self.stage_name,
            judge_model=task_model_name(context.cfg),
            label_col="verdict",
            valid_labels=["correct", "incorrect", "not_attempted"],
            raw_response_col="judge_response",
            id_col="question_id",
            thresholds=thresholds,
        )
        metadata: Dict[str, Any] = {"rows": len(result_df)}
        log_sanity_to_context(context, report, metadata=metadata)

        return StageResult(outputs={"dataset": out_path}, metadata=metadata)


class JudgeGradeBatchExportRunner(StageRunner):
    """Write the sidecar-consumable judge JSONL bundle and exit.

    Outputs ``pending.parquet`` (as the node's ``dataset`` output) plus
    ``requests.jsonl`` + ``items.parquet`` + ``manifest.json`` in the
    same directory. The eval_all judge_sidecar will fan out the requests
    and drop ``output.jsonl`` + ``done.flag`` next to the manifest;
    ``finalize_async`` consumes that.
    """

    stage_name = "judge_grade_batch_export"

    def run(self, context: Any) -> StageResult:
        from ..stages.judge_grade import judge_grade_batch_export

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        out_path = context.output_paths["dataset"]
        output_dir = os.path.dirname(out_path)
        os.makedirs(output_dir, exist_ok=True)

        result_df = judge_grade_batch_export(df, context.cfg, output_dir)
        # The pending.parquet that judge_grade_batch_export already wrote
        # IS our node output. Re-emit it under the declared path to make
        # the ArtifactRegistry happy when the dagspace declares
        # outputs/judge_grade/pending.parquet.
        if os.path.abspath(out_path) != os.path.abspath(os.path.join(output_dir, "pending.parquet")):
            result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={
                "dataset": out_path,
                "requests_jsonl": os.path.join(output_dir, "requests.jsonl"),
                "items_parquet": os.path.join(output_dir, "items.parquet"),
                "manifest": os.path.join(output_dir, "manifest.json"),
            },
            metadata={"rows": len(result_df)},
        )


# ---------------------------------------------------------------------------
# Async finalize (drain → parse → metrics)
# ---------------------------------------------------------------------------

class FinalizeAsyncRunner(StageRunner):
    """Read sidecar output.jsonl → verdict parquet → metrics.{json,parquet}."""

    stage_name = "finalize_async"

    def run(self, context: Any) -> StageResult:
        from ..stages.finalize_async import finalize_async

        result = finalize_async(output_root=context.output_root)

        # Verdict parquet for sanity + downstream consumers. Lives under
        # the standard compute_metrics directory the metrics outputs go to.
        metrics_dir = os.path.dirname(result["metrics_json"])
        verdicts_path = os.path.join(metrics_dir, "verdicts.parquet")
        result["verdicts_df"].to_parquet(verdicts_path, index=False)

        # Echo the metrics parquet to the node's declared dataset path
        # so the ArtifactRegistry can resolve downstream references.
        out_path = context.output_paths.get("dataset")
        if out_path and os.path.abspath(out_path) != os.path.abspath(result["metrics_parquet"]):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            from shutil import copyfile
            copyfile(result["metrics_parquet"], out_path)

        # Sanity report on the verdict stamp.
        thresholds, _patterns = sanity_overrides(context.cfg)
        report = compute_judge_health(
            result["verdicts_df"],
            dagspace="simpleqa_verified",
            stage=self.stage_name,
            judge_model=task_model_name(context.cfg),
            label_col="verdict",
            valid_labels=["correct", "incorrect", "not_attempted"],
            raw_response_col="judge_response",
            id_col="question_id",
            thresholds=thresholds,
        )
        metadata: Dict[str, Any] = {
            "rows": len(result["verdicts_df"]),
            "metrics": result["metrics"],
        }
        log_sanity_to_context(context, report, metadata=metadata)

        return StageResult(
            outputs={
                "dataset": out_path or result["metrics_parquet"],
                "metrics_json": result["metrics_json"],
                "verdicts_parquet": verdicts_path,
            },
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Standalone metrics (used by the live pipeline after judge_grade_live)
# ---------------------------------------------------------------------------

class ComputeMetricsRunner(StageRunner):
    stage_name = "compute_metrics"

    def run(self, context: Any) -> StageResult:
        from ..stages.compute_metrics import compute_metrics, metrics_to_dataframe

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)
        metrics = compute_metrics(df)

        metrics_json_path = os.path.join(context.output_dir, "metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        metrics_df = metrics_to_dataframe(metrics)
        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        metrics_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path, "metrics_json": metrics_json_path},
            metadata={"rows": len(metrics_df), "metrics": metrics},
        )
