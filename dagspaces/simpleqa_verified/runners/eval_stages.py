"""Stage-runner classes for the simpleqa_verified dagspace.

One runner per stage in the pipeline yamls.

The load, inference, and metrics stages use the shared bases in
``dagspaces/common/runners/eval_base.py``. The three judge stages keep the
plain ``StageRunner`` form: they write several outputs, they report JUDGE
health rather than parse health, and ``finalize_async`` reads a sidecar
directory instead of its node input.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from dagspaces.common.eval_sanity import compute_judge_health
from dagspaces.common.orchestrator import StageResult
from dagspaces.common.runners.base import StageRunner
from dagspaces.common.runners.eval_base import (
    EvalLoadRunner,
    EvalMetricsRunner,
    EvalStageRunner,
    runtime_sample_n,
)
from dagspaces.common.runners.sanity import (
    log_sanity_to_context,
    sanity_overrides,
    task_model_name,
)

# ---------------------------------------------------------------------------
# Load + inference
# ---------------------------------------------------------------------------

class LoadDatasetRunner(EvalLoadRunner):
    stage_name = "load_dataset"

    def load(self, context: Any) -> pd.DataFrame:
        from ..stages.load_dataset import load_dataset

        cfg = context.cfg
        data_cfg = cfg.data
        return load_dataset(
            hf_dataset=str(getattr(data_cfg, "hf_dataset", "google/simpleqa-verified")),
            hf_config=getattr(data_cfg, "hf_config", None),
            split=str(getattr(data_cfg, "split", "eval")),
            hf_token=getattr(data_cfg, "hf_token", None),
            sample_n=runtime_sample_n(cfg),
        )


class LLMInferenceRunner(EvalStageRunner):
    stage_name = "llm_inference"

    def transform(self, df: pd.DataFrame, context: Any) -> pd.DataFrame:
        from ..stages.llm_inference import run_llm_inference

        return run_llm_inference(df, context.cfg)


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
        metadata: dict[str, Any] = {"rows": len(result_df)}
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
        metadata: dict[str, Any] = {
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

class ComputeMetricsRunner(EvalMetricsRunner):
    stage_name = "compute_metrics"

    def compute(self, df: pd.DataFrame, context: Any) -> dict[str, Any]:
        from ..stages.compute_metrics import compute_metrics

        return compute_metrics(df)

    def to_dataframe(self, metrics: dict[str, Any]) -> pd.DataFrame:
        from ..stages.compute_metrics import metrics_to_dataframe

        return metrics_to_dataframe(metrics)
