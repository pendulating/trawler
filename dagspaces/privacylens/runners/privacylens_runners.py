"""Runner classes for PrivacyLens evaluation stages."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import pandas as pd

from dagspaces.common.runners.base import StageRunner
from dagspaces.common.orchestrator import StageResult
from dagspaces.common.eval_sanity import (
    compute_format_health,
    compute_judge_health,
    compute_parse_health,
)
from dagspaces.common.runners.sanity import (
    log_sanity_to_context as _log_sanity,
    sanity_overrides as _sanity_overrides,
    task_model_name as _model_name,
)


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


class PerturbCultureRunner(StageRunner):
    """Swap person names + locations in the vignettes to a target culture.

    Reads ``perturb.culture`` from the config (default ``western`` = identity
    passthrough), applies the deterministic name-bank substitution, and writes
    the perturbed dataset. Coverage counts are surfaced as metadata so a run can
    confirm the swap actually fired.
    """

    stage_name = "perturb_culture"

    def run(self, context: Any) -> StageResult:
        from omegaconf import OmegaConf

        from ..perturb import perturb_dataset

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        culture = str(OmegaConf.select(context.cfg, "perturb.culture") or "western")
        result_df = perturb_dataset(df, culture)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        n_persons = int(result_df.get("n_persons_swapped", pd.Series(dtype=int)).sum())
        n_locations = int(result_df.get("n_locations_swapped", pd.Series(dtype=int)).sum())
        print(
            f"[perturb_culture] culture={culture} rows={len(result_df)} "
            f"persons_swapped={n_persons} locations_swapped={n_locations}",
            flush=True,
        )
        return StageResult(
            outputs={"dataset": out_path},
            metadata={
                "rows": len(result_df),
                "culture": culture,
                "n_persons_swapped": n_persons,
                "n_locations_swapped": n_locations,
            },
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
            finish_reason_col="finish_reason",
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
            finish_reason_col="finish_reason",
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
            finish_reason_col="finish_reason",
            expected_input_n=input_n,
            refusal_patterns=patterns,
            thresholds=thresholds,
        )
        metadata: Dict[str, Any] = {"rows": len(result_df)}
        _log_sanity(context, report, metadata=metadata)
        return StageResult(outputs={"dataset": out_path}, metadata=metadata)


class PrivacylensFinalizeAsyncRunner(StageRunner):
    """Drain async-judge outputs and run compute_metrics.

    Reads ``outputs/{leakage,helpfulness}_judge_batch/{pending,items,output}.jsonl``
    from the pipeline ``output_root``, OR-aggregates per-secret leakage
    responses, parses helpfulness, and writes ``metrics.json`` +
    ``metrics.parquet``. Both judge stages get a SanityReport via
    compute_parse_health since the same parsers are reused.

    Loudly raises if a manifest is missing — Phase 1 single-machine
    smoke test surfaces "you forgot to fill output.jsonl"; Phase 2's
    eval_all post_judge_metrics pipeline catches the same exception
    and skips the affected benchmark with a banner instead of failing
    the sweep.
    """

    stage_name = "privacylens_finalize_async"

    def run(self, context: Any) -> StageResult:
        from ..stages.finalize_async import finalize_async

        metrics_parquet_out = context.output_paths.get("dataset")
        if metrics_parquet_out:
            metrics_dir = os.path.dirname(metrics_parquet_out)
        else:
            metrics_dir = None

        result = finalize_async(
            context.output_root,
            metrics_dir=metrics_dir,
        )

        thresholds, patterns = _sanity_overrides(context.cfg)
        model = _model_name(context.cfg)

        # Format adherence is the most consequential trust signal — it's
        # the gate the judge applies before grading any row. If too few
        # actions follow the ``Action:`` format, the resulting leakage
        # and helpfulness rates are mostly defaulted zeros and cannot
        # be quoted. Compute first so the pipeline halts before
        # parse-health adds noise.
        format_report = compute_format_health(
            result["leakage_df"],
            dagspace="privacylens",
            stage="agent_action_format",
            format_col="agent_action_format_status",
            model=model,
            id_col="record_id",
            raw_response_col="generated_action",
            thresholds=thresholds,
        )

        leak_report = compute_parse_health(
            result["leakage_df"],
            dagspace="privacylens",
            stage="leakage_judge_finalize",
            model=model,
            status_col="parse_status",
            completion_col="leak_judge_text",
            label_col="leak_flag",
            finish_reason_col="finish_reason",
            expected_input_n=int(result["leakage_meta"]["rows"]),
            refusal_patterns=patterns,
            thresholds=thresholds,
        )
        help_report = compute_parse_health(
            result["helpfulness_df"],
            dagspace="privacylens",
            stage="helpfulness_judge_finalize",
            model=model,
            status_col="parse_status",
            completion_col="helpfulness_judge_text",
            label_col="helpfulness_score",
            finish_reason_col="finish_reason",
            expected_input_n=int(result["helpfulness_meta"]["rows"]),
            refusal_patterns=patterns,
            thresholds=thresholds,
        )

        # Surface async-judge HTTP errors as judge_api_error_rate. For
        # leakage the denominator is the per-secret fanout count
        # (items_df rows); for helpfulness it's per-row. Both metrics
        # FAIL >0.05 by default, matching the format-adherence gate.
        leak_judge_health = compute_judge_health(
            result["leakage_df"],
            dagspace="privacylens",
            stage="leakage_judge_api",
            judge_model=model,
            label_col="leak_flag",
            valid_labels=[True, False, 0, 1],
            n_api_errors=int(result["leakage_meta"].get("n_response_errors", 0) or 0),
            api_error_denominator=int(result["leakage_meta"].get("items", 0) or 0) or None,
            thresholds=thresholds,
        )
        help_judge_health = compute_judge_health(
            result["helpfulness_df"],
            dagspace="privacylens",
            stage="helpfulness_judge_api",
            judge_model=model,
            label_col="helpfulness_score",
            valid_labels=[0, 1, 2, 3],
            n_api_errors=int(result["helpfulness_meta"].get("n_response_errors", 0) or 0),
            api_error_denominator=int(result["helpfulness_meta"].get("rows", 0) or 0) or None,
            thresholds=thresholds,
        )

        outputs: Dict[str, str] = {
            "metrics_json": result["metrics_json"],
        }
        # If the pipeline declared a `dataset` output, point it at the
        # parquet so downstream nodes can depend on it.
        if metrics_parquet_out:
            outputs["dataset"] = result["metrics_parquet"]

        leak_metrics = result["metrics"].get("leakage", {}) or {}
        help_metrics = result["metrics"].get("helpfulness", {}) or {}
        adj_metrics = result["metrics"].get("adjusted_leakage", {}) or {}
        metadata: Dict[str, Any] = {
            "rows": len(result["leakage_df"]),
            "leakage": result["leakage_meta"],
            "helpfulness": result["helpfulness_meta"],
            "metrics": {
                "agent_action_format_rate": leak_metrics.get("agent_action_format_rate"),
                "leakage_rate_among_parseable": leak_metrics.get("leakage_rate_among_parseable"),
                "leakage_rate_overall_with_default_zero": leak_metrics.get(
                    "leakage_rate_overall_with_default_zero"
                ),
                "qa_accuracy": result["metrics"].get("qa_probing", {}).get("accuracy"),
                "helpfulness_mean_score_among_parseable": help_metrics.get(
                    "mean_score_among_parseable"
                ),
                "helpful_rate_among_parseable": help_metrics.get("helpful_rate_among_parseable"),
                "adjusted_leakage_rate": adj_metrics.get("adjusted_leakage_rate"),
            },
        }
        # Format health first — raises SanityFailure on adherence < 0.9
        # (default), halting the pipeline before unreliable metrics ship.
        # Then judge-API health: catches the case where the live judge
        # server 404'd every request, which silently passes through
        # parse health (responses are syntactically parseable defaults).
        _log_sanity(context, format_report, metadata=metadata)
        _log_sanity(context, leak_judge_health, metadata=metadata)
        _log_sanity(context, help_judge_health, metadata=metadata)
        _log_sanity(context, leak_report, metadata=metadata)
        _log_sanity(context, help_report, metadata=metadata)

        # Headline metrics under finalize/eval/* — same convention as the
        # live ComputeMetricsRunner. Both `_among_parseable` (primary,
        # paper-quoted) and `_overall_with_default_zero` (audit) are
        # logged so a sweep can see when overall masks a parseable miss.
        try:
            if context.logger is not None:
                m = result["metrics"]
                wb_metrics: Dict[str, Any] = {}
                qa = m.get("qa_probing") or {}
                leak = m.get("leakage") or {}
                helpf = m.get("helpfulness") or {}
                adj = m.get("adjusted_leakage") or {}
                if qa:
                    wb_metrics["finalize/eval/qa_accuracy"] = qa.get("accuracy", 0.0)
                if leak:
                    wb_metrics["finalize/eval/agent_action_format_rate"] = leak.get(
                        "agent_action_format_rate", 0.0
                    )
                    wb_metrics["finalize/eval/leakage_rate_among_parseable"] = leak.get(
                        "leakage_rate_among_parseable", 0.0
                    )
                    wb_metrics["finalize/eval/leakage_rate_overall_with_default_zero"] = leak.get(
                        "leakage_rate_overall_with_default_zero", 0.0
                    )
                if helpf:
                    wb_metrics["finalize/eval/helpfulness_mean_score_among_parseable"] = helpf.get(
                        "mean_score_among_parseable", 0.0
                    )
                    wb_metrics["finalize/eval/helpful_rate_among_parseable"] = helpf.get(
                        "helpful_rate_among_parseable", 0.0
                    )
                    wb_metrics["finalize/eval/helpful_rate_overall_with_default_zero"] = helpf.get(
                        "helpful_rate_overall_with_default_zero", 0.0
                    )
                if adj:
                    wb_metrics["finalize/eval/adjusted_leakage_rate"] = adj.get(
                        "adjusted_leakage_rate", 0.0
                    )
                if wb_metrics:
                    context.logger.log_metrics(wb_metrics)
        except Exception as exc:
            print(f"[finalize_async] metric log failure: {exc}", flush=True)

        # Artifact lineage: log the metrics bundle and use_artifact()
        # the upstream export bundles by group alias so the W&B UI
        # shows a clickable trail metrics → judge_output → export.
        try:
            if context.logger is not None and result.get("metrics_json"):
                group = os.environ.get("WANDB_GROUP") or ""
                aliases = ["latest"] + ([group] if group else [])
                use_refs = [
                    f"privacylens-leakage_judge-export:{group or 'latest'}",
                    f"privacylens-helpfulness_judge-export:{group or 'latest'}",
                ]
                paths = [result["metrics_json"]]
                if result.get("metrics_parquet"):
                    paths.append(result["metrics_parquet"])
                if result.get("leakage_results"):
                    paths.append(result["leakage_results"])
                if result.get("helpfulness_results"):
                    paths.append(result["helpfulness_results"])
                context.logger.log_artifact(
                    name="privacylens-metrics",
                    type="metrics",
                    paths=paths,
                    aliases=aliases,
                    metadata={
                        "group": group,
                        "agent_action_format_rate": metadata["metrics"].get("agent_action_format_rate"),
                        "leakage_rate_among_parseable": metadata["metrics"].get("leakage_rate_among_parseable"),
                        "leakage_rate_overall_with_default_zero": metadata["metrics"].get(
                            "leakage_rate_overall_with_default_zero"
                        ),
                        "qa_accuracy": metadata["metrics"].get("qa_accuracy"),
                        "helpfulness_mean_score_among_parseable": metadata["metrics"].get(
                            "helpfulness_mean_score_among_parseable"
                        ),
                        "helpful_rate_among_parseable": metadata["metrics"].get("helpful_rate_among_parseable"),
                        "adjusted_leakage_rate": metadata["metrics"].get("adjusted_leakage_rate"),
                    },
                    description="PrivacyLens async-judge finalize metrics",
                    use_artifacts=use_refs,
                )
        except Exception as exc:
            print(f"[finalize_async] artifact log failure: {exc}", flush=True)

        return StageResult(outputs=outputs, metadata=metadata)


def _log_export_artifact(context: Any, *, stage: str, output_dir: str, n_rows: int) -> None:
    """Log the requests/items/pending/manifest bundle as a versioned
    W&B artifact + alias by WANDB_GROUP for cross-run lineage.

    Lineage flow: <dagspace>-<stage>-export → (sidecar fills output.jsonl)
    → <dagspace>-metrics. The metrics artifact use_artifacts() this one
    by name+alias so the W&B UI shows a clickable trail from the final
    metric back to the exact requests that produced it.
    """
    if context.logger is None:
        return
    paths = []
    for fname in ("requests.jsonl", "items.parquet", "pending.parquet", "manifest.json"):
        p = os.path.join(output_dir, fname)
        if os.path.exists(p):
            paths.append(p)
    if not paths:
        return
    group = os.environ.get("WANDB_GROUP") or ""
    aliases = ["latest"]
    if group:
        aliases.append(group)
    try:
        context.logger.log_artifact(
            name=f"privacylens-{stage}-export",
            type="judge-export",
            paths=paths,
            aliases=aliases,
            metadata={"stage": stage, "n_rows": int(n_rows), "group": group},
            description=f"Async-judge export bundle for privacylens.{stage}",
        )
    except Exception as exc:
        print(f"[export_artifact] log failed for {stage}: {exc}", flush=True)


class LeakageJudgeBatchExportRunner(StageRunner):
    """Write leakage-judge requests as an OpenAI Batch API JSONL file.

    Produces, in the node's output directory:
        - pending.parquet  (dataset + judge_custom_id column)
        - requests.jsonl   (Batch API input)
        - manifest.json    (count, model, provider, schema name)

    The node's declared ``dataset`` output should point at pending.parquet
    so downstream ingest stages can pick it up via the ArtifactRegistry.

    Also logs the bundle as a versioned W&B artifact
    ``privacylens-leakage_judge-export`` (aliased ``:<group>``) so the
    finalize stage can ``use_artifact()`` it for lineage.
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

        _log_export_artifact(
            context, stage="leakage_judge", output_dir=output_dir, n_rows=len(result_df),
        )

        return StageResult(
            outputs={
                "dataset": out_path,
                "requests_jsonl": os.path.join(output_dir, "requests.jsonl"),
                "manifest": os.path.join(output_dir, "manifest.json"),
            },
            metadata={"rows": len(result_df)},
        )


class HelpfulnessJudgeBatchExportRunner(StageRunner):
    """Write helpfulness-judge requests as an OpenAI Batch API JSONL file.

    Same artifact-logging behavior as LeakageJudgeBatchExportRunner —
    the bundle lands as ``privacylens-helpfulness_judge-export`` so
    finalize can record lineage.
    """

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

        _log_export_artifact(
            context, stage="helpfulness_judge", output_dir=output_dir, n_rows=len(result_df),
        )

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

        # Print summary — both `_among_parseable` (paper-quoted) and
        # `_overall_with_default_zero` (audit) are surfaced explicitly so
        # a reader can't confuse "0% leakage" with "0% of inputs were
        # gradeable, all defaulted to non-leaking."
        qa = metrics.get("qa_probing", {}) or {}
        leak = metrics.get("leakage", {}) or {}
        help_m = metrics.get("helpfulness", {}) or {}
        adj = metrics.get("adjusted_leakage", {}) or {}
        print(flush=True)
        print("=" * 60, flush=True)
        print("  PRIVACYLENS EVALUATION RESULTS", flush=True)
        print("=" * 60, flush=True)
        print(f"  QA Probing:", flush=True)
        print(f"    Overall accuracy:  {qa.get('accuracy', 0):.4f}", flush=True)
        for axis, am in qa.get("per_axis", {}).items():
            print(f"    {axis} accuracy:       {am.get('accuracy', 0):.4f} ({am.get('correct', 0)}/{am.get('total', 0)})", flush=True)
        print(f"    Unparseable:       {qa.get('unparseable_count', 0)}/{qa.get('total', 0)}", flush=True)
        print(f"  Trust:", flush=True)
        print(f"    Action format rate: {leak.get('agent_action_format_rate', 0):.4f}", flush=True)
        print(f"    Skipped (no_action_format): {leak.get('skipped_no_action_format', 0)}", flush=True)
        print(f"    Skipped (no_sensitive_info): {leak.get('skipped_no_sensitive_info', 0)}", flush=True)
        print(f"  Leakage:", flush=True)
        print(f"    Among parseable:    {leak.get('leakage_rate_among_parseable', 0):.4f} "
              f"({leak.get('leaking_count_among_parseable', 0)}/{leak.get('total', 0) - leak.get('skipped_no_action_format', 0) - leak.get('skipped_no_sensitive_info', 0)})", flush=True)
        print(f"    Overall (default 0): {leak.get('leakage_rate_overall_with_default_zero', 0):.4f} "
              f"({leak.get('leaking_count_overall', 0)}/{leak.get('total', 0)})", flush=True)
        print(f"    Mean leak prob:     {leak.get('mean_leak_probability_among_parseable', 0):.4f}", flush=True)
        if help_m:
            print(f"  Helpfulness:", flush=True)
            print(f"    Mean score (parseable): {help_m.get('mean_score_among_parseable', 0):.4f}", flush=True)
            print(f"    Mean score (overall):   {help_m.get('mean_score_overall_with_default_zero', 0):.4f}", flush=True)
            print(f"    Helpful rate (parseable): {help_m.get('helpful_rate_among_parseable', 0):.4f}", flush=True)
            print(f"    Helpful rate (overall):   {help_m.get('helpful_rate_overall_with_default_zero', 0):.4f}", flush=True)
        if adj:
            print(f"  Adjusted Leakage (helpful AND judged only):", flush=True)
            print(f"    Adjusted rate:     {adj.get('adjusted_leakage_rate', 0):.4f} "
                  f"({adj.get('leaking_among_helpful', 0)}/{adj.get('total_helpful_and_judged', 0)})", flush=True)
        print("=" * 60, flush=True)

        # Format-health sanity gate — halts the pipeline if action-format
        # adherence is too low to trust the metrics. Same threshold ladder
        # as the async finalize runner.
        thresholds, _ = _sanity_overrides(context.cfg)
        model = _model_name(context.cfg)
        format_report = compute_format_health(
            leakage_df,
            dagspace="privacylens",
            stage="agent_action_format",
            format_col="agent_action_format_status",
            model=model,
            id_col="record_id",
            raw_response_col="generated_action",
            thresholds=thresholds,
        )
        run_metadata: Dict[str, Any] = {"rows": len(metrics_df), "metrics": metrics}
        _log_sanity(context, format_report, metadata=run_metadata)

        return StageResult(
            outputs={"dataset": out_path, "metrics_json": metrics_json_path},
            metadata=run_metadata,
        )
