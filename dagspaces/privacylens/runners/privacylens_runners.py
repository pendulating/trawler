"""Runner classes for PrivacyLens evaluation stages."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from dagspaces.common.runners.base import StageRunner
from dagspaces.common.orchestrator import StageResult


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

        result_df = run_qa_probe_inference(df, context.cfg)
        result_df = parse_qa_responses(result_df, expected_answer="no")

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(result_df)},
        )


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

        result_df = run_leakage_judge_inference(df, context.cfg)
        result_df = parse_leakage_responses(result_df)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
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

        metrics = compute_metrics(qa_df, leakage_df)

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
        print("=" * 60, flush=True)

        return StageResult(
            outputs={"dataset": out_path, "metrics_json": metrics_json_path},
            metadata={"rows": len(metrics_df), "metrics": metrics},
        )
