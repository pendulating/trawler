"""Runner classes for PrivacyLens evaluation stages."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import OmegaConf

from dagspaces.common.runners.base import StageRunner
from dagspaces.common.orchestrator import StageResult
from dagspaces.common.eval_sanity import (
    SanityReport,
    compute_parse_health,
)


def _sanity_overrides(cfg: Any) -> tuple[Optional[Dict[str, float]], Optional[List[str]]]:
    """Pull per-benchmark sanity overrides from cfg, if any.

    Looks at ``cfg.sanity.thresholds`` (dict like ``{parseable_rate:lt: 0.9}``)
    and ``cfg.sanity.refusal_patterns`` (list[str]). Both are optional;
    omitted keys fall back to ``DEFAULT_THRESHOLDS`` /
    ``DEFAULT_REFUSAL_PATTERNS``.
    """
    thresholds = None
    patterns = None
    try:
        sanity_cfg = OmegaConf.select(cfg, "sanity")
        if sanity_cfg is not None:
            t = OmegaConf.select(sanity_cfg, "thresholds")
            if t is not None:
                thresholds = {str(k): float(v) for k, v in OmegaConf.to_container(t, resolve=True).items()}
            p = OmegaConf.select(sanity_cfg, "refusal_patterns")
            if p is not None:
                patterns = [str(x) for x in OmegaConf.to_container(p, resolve=True)]
    except Exception:
        pass
    return thresholds, patterns


def _model_name(cfg: Any) -> str:
    """Best-effort task-LLM identifier for failure-row attribution."""
    try:
        for key in ("model.model_source", "model.served_model_name", "model.model_family"):
            v = OmegaConf.select(cfg, key)
            if v:
                return str(v)
    except Exception:
        pass
    return ""


def _log_sanity(
    context: Any,
    report: SanityReport,
    *,
    metadata: Dict[str, Any],
) -> None:
    """Log a SanityReport via context.logger and fold a small summary
    into the stage's StageResult.metadata so the orchestrator's
    pipeline_manifest captures the headline numbers.
    """
    try:
        if context.logger is not None:
            context.logger.log_sanity_report(report)
    except Exception as exc:
        # Sanity logging never fails the pipeline.
        print(f"[sanity] log failure for {report.stage}: {exc}", flush=True)

    # Compact summary into stage metadata. Full failure rows live in W&B.
    metadata.setdefault("sanity", {})
    metadata["sanity"][report.stage] = {
        "metrics": dict(report.metrics),
        "n_warnings": len(report.warnings),
        "n_failures": int(len(report.failure_rows)),
        "failures_dropped": int(report.failures_dropped),
        "warnings": [w.message() for w in report.warnings],
    }


class LoadDatasetRunner(StageRunner):
    stage_name = "load_dataset"

    def run(self, context: Any) -> StageResult:
        from ..stages.load_dataset import load_dataset

        cfg = context.cfg
        data_cfg = getattr(cfg, "data", {})

        sample_n = None
        runtime = getattr(cfg, "runtime", None)
        if runtime:
            sample_n = getattr(runtime, "sample_n", None)
            if sample_n is not None:
                sample_n = int(sample_n)

        df = load_dataset(
            hf_dataset=str(getattr(data_cfg, "hf_dataset", "SALT-NLP/PrivacyLens")),
            hf_config=getattr(data_cfg, "hf_config", None),
            split=str(getattr(data_cfg, "split", "train")),
            max_examples=int(getattr(data_cfg, "max_examples", 0) or 0),
            hf_token=str(getattr(data_cfg, "hf_token", "") or "") or None,
            sample_n=sample_n,
        )

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(df)},
        )


class QAProbeInferenceRunner(StageRunner):
    stage_name = "qa_probe_inference"

    def run(self, context: Any) -> StageResult:
        from ..stages.llm_inference import run_qa_probe_inference
        from ..stages.parse_responses import parse_qa_responses

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)
        input_n = len(df)

        result_df = run_qa_probe_inference(df, context.cfg)
        result_df = parse_qa_responses(result_df, expected_answer="no")

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        thresholds, patterns = _sanity_overrides(context.cfg)
        report = compute_parse_health(
            result_df,
            dagspace="privacylens",
            stage=self.stage_name,
            model=_model_name(context.cfg),
            status_col="parse_status",
            completion_col="generated_text",
            label_col="predicted_label",
            # QA fans out 3× per input row (S/V/T axes), so the expected
            # output count is 3 × input_n. Skip row_count_drop unless we
            # can be exact.
            expected_input_n=None,
            refusal_patterns=patterns,
            thresholds=thresholds,
        )
        metadata: Dict[str, Any] = {"rows": len(result_df)}
        _log_sanity(context, report, metadata=metadata)
        return StageResult(outputs={"dataset": out_path}, metadata=metadata)


class AgentActionInferenceRunner(StageRunner):
    stage_name = "agent_action_inference"

    def run(self, context: Any) -> StageResult:
        from ..stages.llm_inference import run_action_inference

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        result_df = run_action_inference(df, context.cfg)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(result_df)},
        )


class LeakageJudgeInferenceRunner(StageRunner):
    stage_name = "leakage_judge_inference"

    def run(self, context: Any) -> StageResult:
        from ..stages.llm_inference import run_leakage_judge_inference
        from ..stages.parse_responses import parse_leakage_responses

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)
        input_n = len(df)

        result_df = run_leakage_judge_inference(df, context.cfg)
        result_df = parse_leakage_responses(result_df)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        thresholds, patterns = _sanity_overrides(context.cfg)
        report = compute_parse_health(
            result_df,
            dagspace="privacylens",
            stage=self.stage_name,
            model=_model_name(context.cfg),
            status_col="parse_status",
            completion_col="leak_judge_text",
            label_col="leak_flag",
            expected_input_n=input_n,
            refusal_patterns=patterns,
            thresholds=thresholds,
        )
        metadata: Dict[str, Any] = {"rows": len(result_df)}
        _log_sanity(context, report, metadata=metadata)
        return StageResult(outputs={"dataset": out_path}, metadata=metadata)


class HelpfulnessJudgeInferenceRunner(StageRunner):
    stage_name = "helpfulness_judge_inference"

    def run(self, context: Any) -> StageResult:
        from ..stages.llm_inference import run_helpfulness_judge_inference
        from ..stages.parse_responses import parse_helpfulness_responses

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)
        input_n = len(df)

        result_df = run_helpfulness_judge_inference(df, context.cfg)
        result_df = parse_helpfulness_responses(result_df)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        thresholds, patterns = _sanity_overrides(context.cfg)
        report = compute_parse_health(
            result_df,
            dagspace="privacylens",
            stage=self.stage_name,
            model=_model_name(context.cfg),
            status_col="parse_status",
            completion_col="helpfulness_judge_text",
            label_col="helpfulness_score",
            expected_input_n=input_n,
            refusal_patterns=patterns,
            thresholds=thresholds,
        )
        metadata: Dict[str, Any] = {"rows": len(result_df)}
        _log_sanity(context, report, metadata=metadata)
        return StageResult(outputs={"dataset": out_path}, metadata=metadata)


class LeakageJudgeBatchExportRunner(StageRunner):
    """Write leakage-judge requests as an OpenAI Batch API JSONL file.

    Produces, in the node's output directory:
        - pending.parquet  (dataset + judge_custom_id column)
        - requests.jsonl   (Batch API input)
        - manifest.json    (count, model, provider, schema name)

    The node's declared ``dataset`` output should point at pending.parquet
    so downstream ingest stages can pick it up via the ArtifactRegistry.
    """

    stage_name = "leakage_judge_batch_export"

    def run(self, context: Any) -> StageResult:
        from ..stages.llm_inference import export_leakage_judge_batch

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        out_path = context.output_paths["dataset"]
        output_dir = os.path.dirname(out_path)
        os.makedirs(output_dir, exist_ok=True)

        result_df = export_leakage_judge_batch(df, context.cfg, output_dir)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={
                "dataset": out_path,
                "requests_jsonl": os.path.join(output_dir, "requests.jsonl"),
                "manifest": os.path.join(output_dir, "manifest.json"),
            },
            metadata={"rows": len(result_df)},
        )


class HelpfulnessJudgeBatchExportRunner(StageRunner):
    """Write helpfulness-judge requests as an OpenAI Batch API JSONL file."""

    stage_name = "helpfulness_judge_batch_export"

    def run(self, context: Any) -> StageResult:
        from ..stages.llm_inference import export_helpfulness_judge_batch

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        out_path = context.output_paths["dataset"]
        output_dir = os.path.dirname(out_path)
        os.makedirs(output_dir, exist_ok=True)

        result_df = export_helpfulness_judge_batch(df, context.cfg, output_dir)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={
                "dataset": out_path,
                "requests_jsonl": os.path.join(output_dir, "requests.jsonl"),
                "manifest": os.path.join(output_dir, "manifest.json"),
            },
            metadata={"rows": len(result_df)},
        )


class ComputeMetricsRunner(StageRunner):
    stage_name = "compute_metrics"

    def run(self, context: Any) -> StageResult:
        from ..stages.compute_metrics import compute_metrics, metrics_to_dataframe

        qa_path = context.inputs["qa_dataset"]
        leakage_path = context.inputs["leakage_dataset"]

        qa_df = pd.read_parquet(qa_path)
        leakage_df = pd.read_parquet(leakage_path)

        # Helpfulness is optional for backward compatibility
        helpfulness_df = None
        helpfulness_path = context.inputs.get("helpfulness_dataset")
        if helpfulness_path:
            helpfulness_df = pd.read_parquet(helpfulness_path)

        metrics = compute_metrics(qa_df, leakage_df, helpfulness_df)

        # Save metrics as JSON
        output_dir = os.path.dirname(context.output_paths["dataset"])
        os.makedirs(output_dir, exist_ok=True)
        metrics_json_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Save as parquet for pipeline compatibility
        metrics_df = metrics_to_dataframe(metrics)
        out_path = context.output_paths["dataset"]
        metrics_df.to_parquet(out_path, index=False)

        # Print summary
        qa = metrics.get("qa_probing", {})
        leak = metrics.get("leakage", {})
        help_m = metrics.get("helpfulness", {})
        adj = metrics.get("adjusted_leakage", {})
        print(flush=True)
        print("=" * 60, flush=True)
        print("  PRIVACYLENS EVALUATION RESULTS", flush=True)
        print("=" * 60, flush=True)
        print(f"  QA Probing:", flush=True)
        print(f"    Overall accuracy:  {qa.get('accuracy', 0):.4f}", flush=True)
        for axis, am in qa.get("per_axis", {}).items():
            print(f"    {axis} accuracy:       {am.get('accuracy', 0):.4f} ({am.get('correct', 0)}/{am.get('total', 0)})", flush=True)
        print(f"    Unparseable:       {qa.get('unparseable_count', 0)}/{qa.get('total', 0)}", flush=True)
        print(f"  Leakage:", flush=True)
        print(f"    Leakage rate:      {leak.get('leakage_rate', 0):.4f} ({leak.get('leaking_count', 0)}/{leak.get('total', 0)})", flush=True)
        print(f"    Mean leak prob:    {leak.get('mean_leak_probability', 0):.4f}", flush=True)
        if help_m:
            print(f"  Helpfulness:", flush=True)
            print(f"    Mean score:        {help_m.get('mean_score', 0):.4f}", flush=True)
            print(f"    Helpful rate:      {help_m.get('helpful_rate', 0):.4f} ({help_m.get('helpful_count', 0)}/{help_m.get('total', 0)})", flush=True)
        if adj:
            print(f"  Adjusted Leakage (helpful only):", flush=True)
            print(f"    Adjusted rate:     {adj.get('adjusted_leakage_rate', 0):.4f} ({adj.get('leaking_among_helpful', 0)}/{adj.get('total_helpful', 0)})", flush=True)
        print("=" * 60, flush=True)

        return StageResult(
            outputs={"dataset": out_path, "metrics_json": metrics_json_path},
            metadata={"rows": len(metrics_df), "metrics": metrics},
        )
