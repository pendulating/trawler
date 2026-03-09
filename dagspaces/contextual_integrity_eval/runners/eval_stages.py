"""Stage runners for contextual integrity evaluation pipeline."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from ..orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _save_stage_outputs,
    prepare_stage_input,
)
from ..stages.active_prompting_ablation_eval import run_active_prompting_ablation_eval_stage
from ..stages.agent_action_eval import run_agent_action_eval_stage
from ..stages.alignment_loop_validation import run_alignment_loop_validation_stage
from ..stages.build_context_variants import run_build_context_variants_stage
from ..stages.cimemories_generation_judge import run_cimemories_generation_judge_stage
from ..stages.cimemories_label_generation import run_cimemories_label_generation_stage
from ..stages.cimemories_metrics import run_cimemories_metrics_stage
from ..stages.combine_eval_signals import run_combine_eval_signals_stage
from ..stages.context_collapse_diagnostics import run_context_collapse_diagnostics_stage
from ..stages.emit_run_governance_manifest import run_emit_run_governance_manifest_stage
from ..stages.fetch_hf_dataset import run_fetch_hf_dataset_stage
from ..stages.import_context_artifacts import run_import_context_artifacts_stage
from ..stages.judge_calibration import run_judge_calibration_stage
from ..stages.load_cimemories_labels import run_load_cimemories_labels_stage
from ..stages.qa_probe_eval import run_qa_probe_eval_stage
from ..stages.run_baselines import run_run_baselines_stage
from ..stages.scientific_hygiene_gates import run_scientific_hygiene_gates_stage
from ..stages.statistical_analysis import run_statistical_analysis_stage
from ..stages.summarize_results import run_summarize_results_stage
from .base import StageRunner


def _meta(out: Any, streaming: bool = False) -> dict[str, Any]:
    return {
        "rows": len(out) if isinstance(out, pd.DataFrame) else None,
        "streaming": bool(streaming),
    }


def _collect(context: StageExecutionContext, out: Any, streaming: bool = False) -> StageResult:
    _save_stage_outputs(out, context.output_paths)
    outputs = _collect_outputs(
        context,
        {name: spec.optional for name, spec in context.node.outputs.items()},
    )
    return StageResult(outputs=outputs, metadata=_meta(out, streaming=False))


def _load_input(context: StageExecutionContext, key: str) -> tuple[pd.DataFrame, bool]:
    dataset_path = context.inputs.get(key)
    if not dataset_path:
        raise ValueError(f"Node '{context.node.key}' requires '{key}' input")
    try:
        return pd.read_parquet(dataset_path), False
    except Exception:
        df, _, _ = prepare_stage_input(context.cfg, dataset_path, context.node.stage)
        return df, False


class ImportContextArtifactsRunner(StageRunner):
    stage_name = "import_context_artifacts"

    def run(self, context: StageExecutionContext) -> StageResult:
        out = run_import_context_artifacts_stage(None, context.cfg)
        return _collect(context, out)


class FetchHFDatasetRunner(StageRunner):
    stage_name = "fetch_hf_dataset"

    def run(self, context: StageExecutionContext) -> StageResult:
        out = run_fetch_hf_dataset_stage(None, context.cfg)
        return _collect(context, out)


class FetchLegacyPrivacyLensRunner(StageRunner):
    stage_name = "fetch_privacylens"

    def run(self, context: StageExecutionContext) -> StageResult:
        out = run_fetch_hf_dataset_stage(None, context.cfg)
        return _collect(context, out)


# Backward-compatible class alias for old imports.
FetchPrivacyLensRunner = FetchLegacyPrivacyLensRunner


class LoadCIMemoriesLabelsRunner(StageRunner):
    stage_name = "load_cimemories_labels"

    def run(self, context: StageExecutionContext) -> StageResult:
        out = run_load_cimemories_labels_stage(None, context.cfg)
        return _collect(context, out)


class BuildContextVariantsRunner(StageRunner):
    stage_name = "build_context_variants"

    def run(self, context: StageExecutionContext) -> StageResult:
        artifacts, streaming = _load_input(context, "artifacts")
        out = run_build_context_variants_stage(artifacts, context.cfg)
        return _collect(context, out, streaming=streaming)


class RunBaselinesRunner(StageRunner):
    stage_name = "run_baselines"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_df, _ = _load_input(context, "dataset")
        variants_path = context.inputs.get("variants")
        if variants_path:
            variants_df, _ = _load_input(context, "variants")
            merged = dataset_df.assign(_tmp=1).merge(variants_df.assign(_tmp=1), on="_tmp").drop(columns=["_tmp"])
        else:
            merged = dataset_df
        out = run_run_baselines_stage(merged, context.cfg)
        return _collect(context, out)


class _SingleInputRunner(StageRunner):
    stage_fn: Callable[[Any, Any], pd.DataFrame]
    input_key = "dataset"

    def run(self, context: StageExecutionContext) -> StageResult:
        data, streaming = _load_input(context, self.input_key)
        out = self.stage_fn(data, context.cfg)
        return _collect(context, out, streaming=streaming)


class CIMemoriesLabelGenerationRunner(_SingleInputRunner):
    stage_name = "cimemories_label_generation"
    stage_fn = staticmethod(run_cimemories_label_generation_stage)


class CIMemoriesGenerationJudgeRunner(_SingleInputRunner):
    stage_name = "cimemories_generation_judge"
    stage_fn = staticmethod(run_cimemories_generation_judge_stage)


class CIMemoriesMetricsRunner(_SingleInputRunner):
    stage_name = "cimemories_metrics"
    stage_fn = staticmethod(run_cimemories_metrics_stage)


class QAProbeEvalRunner(_SingleInputRunner):
    stage_name = "qa_probe_eval"
    stage_fn = staticmethod(run_qa_probe_eval_stage)


class AgentActionEvalRunner(_SingleInputRunner):
    stage_name = "agent_action_eval"
    stage_fn = staticmethod(run_agent_action_eval_stage)


class CombineEvalSignalsRunner(StageRunner):
    stage_name = "combine_eval_signals"

    def run(self, context: StageExecutionContext) -> StageResult:
        qa_df, qa_streaming = _load_input(context, "qa")
        agent_df, agent_streaming = _load_input(context, "agent")
        out = run_combine_eval_signals_stage({"qa": qa_df, "agent": agent_df}, context.cfg)
        return _collect(context, out, streaming=(qa_streaming or agent_streaming))


class JudgeCalibrationRunner(_SingleInputRunner):
    stage_name = "judge_calibration"
    stage_fn = staticmethod(run_judge_calibration_stage)


class ActivePromptingAblationRunner(_SingleInputRunner):
    stage_name = "active_prompting_ablation_eval"
    stage_fn = staticmethod(run_active_prompting_ablation_eval_stage)


class AlignmentLoopValidationRunner(_SingleInputRunner):
    stage_name = "alignment_loop_validation"
    stage_fn = staticmethod(run_alignment_loop_validation_stage)


class ContextCollapseDiagnosticsRunner(_SingleInputRunner):
    stage_name = "context_collapse_diagnostics"
    stage_fn = staticmethod(run_context_collapse_diagnostics_stage)


class StatisticalAnalysisRunner(_SingleInputRunner):
    stage_name = "statistical_analysis"
    stage_fn = staticmethod(run_statistical_analysis_stage)


class ScientificHygieneGatesRunner(_SingleInputRunner):
    stage_name = "scientific_hygiene_gates"
    stage_fn = staticmethod(run_scientific_hygiene_gates_stage)


class EmitRunGovernanceManifestRunner(_SingleInputRunner):
    stage_name = "emit_run_governance_manifest"
    stage_fn = staticmethod(run_emit_run_governance_manifest_stage)


class SummarizeResultsRunner(_SingleInputRunner):
    stage_name = "summarize_results"
    stage_fn = staticmethod(run_summarize_results_stage)

