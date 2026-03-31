"""Orchestrator for CIRL-Vignettes dagspace.

Copied from goldcoin_hipaa/orchestrator.py with project-specific
name changes. Straightforward sequential pipeline execution with optional
SLURM submission and robust NFS result handling.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from hydra.core.hydra_config import HydraConfig

from dagspaces.common.config_schema import (
    PipelineGraphSpec,
    PipelineNodeSpec,
    load_pipeline_graph,
    resolve_output_root,
)
from dagspaces.common.orchestrator import (
    ArtifactRegistry,
    StageExecutionContext,
    StageResult,
    _NoOpLogger,
    _node_inputs,
    _node_output_paths,
    _print_status,
    _safe_log_table,
    build_run_config,
    common_parent,
    prepare_node_config,
)

from .runners import get_stage_registry
from .wandb_logger import WandbConfig, WandbLogger

try:
    import submitit
    _SUBMITIT_AVAILABLE = True
except Exception:
    submitit = None
    _SUBMITIT_AVAILABLE = False

_CONF_DIR = os.path.join(os.path.dirname(__file__), "conf")

_STAGE_REGISTRY: Dict[str, Any] = get_stage_registry()


def _get_wandb_logger(cfg: DictConfig, stage: str, run_id: Optional[str] = None, run_config: Optional[Dict[str, Any]] = None):
    wb_config = WandbConfig.from_hydra_config(cfg)
    if wb_config.enabled:
        return WandbLogger(cfg, stage=stage, run_id=run_id, run_config=run_config)
    return _NoOpLogger(cfg, stage=stage, run_id=run_id, run_config=run_config)


def _log_eval_metrics(logger, metrics: Dict[str, Any], stage: str) -> None:
    """Log evaluation metrics to W&B and print a structured summary."""
    task = metrics.get("task", "unknown")
    if task == "cirl_trajectory":
        _log_trajectory_metrics(logger, metrics, stage)
    else:
        _log_probing_metrics(logger, metrics, stage)


def _log_probing_metrics(logger, metrics: Dict[str, Any], stage: str) -> None:
    """Log probing evaluation metrics."""
    prefix = f"{stage}/eval"

    wb_metrics: Dict[str, Any] = {}

    for key in ("accuracy", "unparseable_rate", "unparseable_count",
                "total", "parseable", "reject_count", "accept_count"):
        if key in metrics:
            wb_metrics[f"{prefix}/{key}"] = metrics[key]

    for level, level_m in metrics.get("per_level", {}).items():
        for k, v in level_m.items():
            wb_metrics[f"{prefix}/{level}/{k}"] = v

    if wb_metrics:
        logger.log_metrics(wb_metrics)

    task = metrics.get("task", "unknown")
    print(flush=True)
    print("=" * 60, flush=True)
    print(f"  PROBING RESULTS — {task.upper()}", flush=True)
    print("=" * 60, flush=True)
    print(f"  Samples:      {metrics.get('total', '?')} total, "
          f"{metrics.get('parseable', '?')} parseable, "
          f"{metrics.get('unparseable_count', '?')} unparseable "
          f"({metrics.get('unparseable_rate', 0) * 100:.1f}%)", flush=True)
    print(f"  Rejection acc (overall): {metrics.get('accuracy', '?')}", flush=True)
    print(f"  Reject (B): {metrics.get('reject_count', '?')}  "
          f"Accept (A): {metrics.get('accept_count', '?')}", flush=True)
    print("-" * 60, flush=True)
    for level, level_m in metrics.get("per_level", {}).items():
        print(f"  {level:>12s}:  "
              f"{level_m.get('reject_count', '?')}/{level_m.get('total', '?')} reject  "
              f"(acc={level_m.get('accuracy', '?')})", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)


def _log_trajectory_metrics(logger, metrics: Dict[str, Any], stage: str) -> None:
    """Log trajectory I/U/C evaluation metrics."""
    prefix = f"{stage}/eval"

    wb_metrics: Dict[str, Any] = {}
    for key in ("integrity", "utility", "complete", "leakage_rate",
                "adjusted_leakage_rate", "avg_helpfulness_score",
                "helpful_rate", "total"):
        if key in metrics:
            wb_metrics[f"{prefix}/{key}"] = metrics[key]

    if wb_metrics:
        logger.log_metrics(wb_metrics)

    print(flush=True)
    print("=" * 60, flush=True)
    print("  TRAJECTORY RESULTS — CIRL I/U/C", flush=True)
    print("=" * 60, flush=True)
    print(f"  Cases:          {metrics.get('total', '?')}", flush=True)
    print(f"  Integrity (I):  {metrics.get('integrity', '?')}", flush=True)
    print(f"  Utility (U):    {metrics.get('utility', '?')}", flush=True)
    print(f"  Complete (C):   {metrics.get('complete', '?')}", flush=True)
    print("-" * 60, flush=True)
    print(f"  Leakage rate:          {metrics.get('leakage_rate', '?')}", flush=True)
    print(f"  Adj. leakage rate:     {metrics.get('adjusted_leakage_rate', '?')}", flush=True)
    print(f"  Avg helpfulness:       {metrics.get('avg_helpfulness_score', '?')}", flush=True)
    print(f"  Helpful rate (>=2):    {metrics.get('helpful_rate', '?')}", flush=True)
    hdist = metrics.get("helpfulness_distribution", {})
    if hdist:
        print(f"  Helpfulness dist:      {hdist}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)


def _serialize_context_data(
    node_cfg: DictConfig,
    node: PipelineNodeSpec,
    inputs: Dict[str, str],
    output_paths: Dict[str, str],
    output_dir: str,
    output_root: str,
) -> Dict[str, Any]:
    return {
        "cfg": OmegaConf.to_container(node_cfg, resolve=True),
        "node": {
            "key": node.key,
            "stage": node.stage,
            "depends_on": node.depends_on,
            "inputs": node.inputs,
            "outputs": {k: {"path": v.path, "type": v.type, "optional": v.optional} for k, v in node.outputs.items()},
            "overrides": node.overrides,
            "launcher": node.launcher,
            "parallel_group": node.parallel_group,
            "max_attempts": node.max_attempts,
            "retry_backoff_s": node.retry_backoff_s,
            "wandb_suffix": node.wandb_suffix,
        },
        "inputs": inputs,
        "output_paths": output_paths,
        "output_dir": output_dir,
        "output_root": output_root,
    }


def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single stage -- designed to be submitted as a SLURM job."""
    from dagspaces.common.stage_utils import ensure_dotenv
    ensure_dotenv()
    from omegaconf import OmegaConf
    from dagspaces.common.config_schema import PipelineNodeSpec, OutputSpec

    cfg = OmegaConf.create(context_data["cfg"])
    node_dict = context_data["node"]

    outputs = {}
    for out_key, out_val in node_dict.get("outputs", {}).items():
        outputs[out_key] = OutputSpec.from_config(out_key, out_val)

    node = PipelineNodeSpec(
        key=node_dict["key"],
        stage=node_dict["stage"],
        depends_on=node_dict.get("depends_on", []),
        inputs=node_dict.get("inputs", {}),
        outputs=outputs,
        overrides=node_dict.get("overrides", {}),
        launcher=node_dict.get("launcher"),
        parallel_group=node_dict.get("parallel_group"),
        max_attempts=node_dict.get("max_attempts", 1),
        retry_backoff_s=node_dict.get("retry_backoff_s", 0.0),
        wandb_suffix=node_dict.get("wandb_suffix"),
    )

    context = StageExecutionContext(
        cfg=cfg,
        node=node,
        inputs=context_data["inputs"],
        output_paths=context_data["output_paths"],
        output_dir=context_data["output_dir"],
        output_root=context_data["output_root"],
    )

    stage_registry = dict(_STAGE_REGISTRY)
    runner = stage_registry.get(node.stage)
    if runner is None:
        raise ValueError(f"No runner registered for stage '{node.stage}' (node '{node.key}')")

    wandb_run_id = node.wandb_suffix or node.key
    run_config = build_run_config(cfg, node, context.inputs, context.output_paths, dagspace_name="cirl_vignettes")

    with _get_wandb_logger(cfg, stage=node.stage, run_id=wandb_run_id, run_config=run_config) as logger:
        context.logger = logger
        _print_status({"node": node.key, "stage": node.stage, "status": "running", "inputs": context.inputs})
        stage_start = time.time()

        result = runner.run(context)

        if result.outputs and "dataset" in result.outputs:
            try:
                df_out = pd.read_parquet(result.outputs["dataset"])
                _safe_log_table(logger, df_out, f"{node.stage}/results")
            except Exception as e:
                print(f"Warning: Failed to log output table for {node.key}: {e}", flush=True)

        duration_s = time.time() - stage_start
        logger.log_metrics({
            f"{node.stage}/duration_s": duration_s,
            f"{node.stage}/rows_processed": result.metadata.get("rows", 0),
        })

        eval_metrics = result.metadata.get("metrics")
        if eval_metrics and isinstance(eval_metrics, dict):
            _log_eval_metrics(logger, eval_metrics, node.stage)

        return {"outputs": result.outputs, "metadata": result.metadata}


def _resolve_hydra_output_dir() -> Optional[str]:
    """Get the Hydra runtime output directory (correct for both run and multirun)."""
    try:
        hc = HydraConfig.get()
        if hc and hc.runtime and hc.runtime.output_dir:
            return str(hc.runtime.output_dir)
    except Exception:
        pass
    return None


def run_experiment(cfg: DictConfig) -> None:
    """Execute the CIRL-Vignettes evaluation pipeline."""
    with _get_wandb_logger(cfg, stage="orchestrator", run_id="monitor") as logger:
        try:
            graph_spec: PipelineGraphSpec = load_pipeline_graph(cfg)

            hydra_output_dir = _resolve_hydra_output_dir()
            if hydra_output_dir:
                output_root = os.path.join(hydra_output_dir, "cirl_vignettes")
            else:
                output_root = resolve_output_root(graph_spec, cfg)
            os.makedirs(output_root, exist_ok=True)
            print(f"[orchestrator] output_root={output_root}", flush=True)

            registry = ArtifactRegistry()
            for source_key, source in graph_spec.sources.items():
                path = source.path
                if not os.path.isabs(path):
                    path = os.path.abspath(os.path.expanduser(path))
                registry.register_source(source_key, path)

            stage_registry = dict(_STAGE_REGISTRY)
            ordered_nodes = graph_spec.topological_order()
            pipeline_start = time.time()

            manifest: Dict[str, Any] = {
                "output_root": output_root,
                "nodes": {},
            }

            for node_key in ordered_nodes:
                node = graph_spec.nodes[node_key]
                runner = stage_registry.get(node.stage)
                if runner is None:
                    raise ValueError(f"No runner registered for stage '{node.stage}' (node '{node.key}')")

                inputs = _node_inputs(node, registry)
                output_paths = _node_output_paths(node, registry, output_root)
                output_dir = common_parent(output_paths.values())
                if not output_dir:
                    output_dir = os.path.join(output_root, node.key)
                os.makedirs(output_dir, exist_ok=True)

                node_cfg = prepare_node_config(cfg, node, output_dir)
                context = StageExecutionContext(
                    cfg=node_cfg,
                    node=node,
                    inputs=inputs,
                    output_paths=output_paths,
                    output_dir=output_dir,
                    output_root=output_root,
                )

                node_start = time.time()

                if node.launcher and _SUBMITIT_AVAILABLE:
                    from dagspaces.common.orchestrator import (
                        _create_submitit_executor,
                        _load_launcher_config,
                        _submit_slurm_job,
                    )

                    _print_status({"node": node.key, "stage": node.stage, "status": "submitting", "launcher": node.launcher})

                    launcher_cfg = _load_launcher_config(cfg, node.launcher, config_dir=_CONF_DIR)

                    log_base = hydra_output_dir if hydra_output_dir else output_root
                    log_folder = os.path.join(log_base, ".slurm_jobs", node.key)
                    os.makedirs(log_folder, exist_ok=True)

                    executor = _create_submitit_executor(launcher_cfg, f"CIRLVignettes-{node.key}", log_folder, use_srun=False)
                    context_data = _serialize_context_data(node_cfg, node, inputs, output_paths, output_dir, output_root)

                    job = _submit_slurm_job(executor, execute_stage_job, context_data, node.key, node.launcher)

                    job_result: Optional[Dict[str, Any]] = None
                    try:
                        job_result = job.result()
                    except Exception as exc:
                        exc_text = str(exc).lower()
                        result_path = job.paths.result_pickle
                        missing_result = (
                            "has not produced any output" in exc_text
                            or "result_pickle" in exc_text
                            or ("result" in exc_text and "pickle" in exc_text)
                        )
                        if missing_result:
                            wait_s = int(OmegaConf.select(cfg, "runtime.submitit_result_wait_s", default=120))
                            _print_status({
                                "debug": "waiting_for_result_pickle",
                                "job_id": job.job_id,
                                "path": str(result_path),
                                "max_wait_s": wait_s,
                            })
                            deadline = time.time() + wait_s
                            while time.time() < deadline and not os.path.exists(result_path):
                                time.sleep(2)
                            if os.path.exists(result_path):
                                with open(result_path, "rb") as f:
                                    _outcome, _result = pickle.load(f)
                                job_result = _result
                                _print_status({"debug": "recovered_result_after_wait", "job_id": job.job_id})

                        if job_result is None:
                            try:
                                check = subprocess.run(
                                    ["squeue", "-j", str(job.job_id), "-h", "-o", "%t"],
                                    capture_output=True, text=True, check=False,
                                )
                                state = check.stdout.strip()
                                if state in ("R", "PD", "CG"):
                                    _print_status({
                                        "debug": "job_still_running_in_squeue",
                                        "job_id": job.job_id,
                                        "state": state,
                                    })
                                    while True:
                                        time.sleep(30)
                                        check = subprocess.run(
                                            ["squeue", "-j", str(job.job_id), "-h", "-o", "%t"],
                                            capture_output=True, text=True, check=False,
                                        )
                                        if not check.stdout.strip() or check.stdout.strip() not in ("R", "PD", "CG"):
                                            break
                                    for _ in range(30):
                                        if os.path.exists(result_path):
                                            break
                                        time.sleep(2)
                                    if os.path.exists(result_path):
                                        with open(result_path, "rb") as f:
                                            _outcome, _result = pickle.load(f)
                                        job_result = _result
                                        _print_status({"debug": "recovered_result_after_squeue_wait", "job_id": job.job_id})
                            except Exception as inner_exc:
                                _print_status({
                                    "debug": "squeue_fallback_failed",
                                    "job_id": job.job_id,
                                    "error": str(inner_exc),
                                })

                        if job_result is None:
                            _print_status({"node": node.key, "status": "failed", "job_id": job.job_id, "error": str(exc)})
                            raise

                    if isinstance(job_result, tuple) and len(job_result) == 2:
                        outcome, payload = job_result
                        if outcome == "error":
                            raise RuntimeError(f"SLURM job {job.job_id} failed:\n{payload}")
                        job_result = payload
                    result = StageResult(outputs=job_result["outputs"], metadata=job_result["metadata"])
                else:
                    _print_status({"node": node.key, "stage": node.stage, "status": "running", "inputs": inputs})
                    try:
                        result = runner.run(context)
                    except Exception as exc:
                        _print_status({"node": node.key, "stage": node.stage, "status": "failed", "error": str(exc)})
                        raise

                eval_metrics = result.metadata.get("metrics")
                if eval_metrics and isinstance(eval_metrics, dict):
                    _log_eval_metrics(logger, eval_metrics, node.stage)

                registry.register_outputs(node.key, result.outputs)
                duration = time.time() - node_start
                manifest["nodes"][node.key] = {
                    "stage": node.stage,
                    "inputs": inputs,
                    "outputs": result.outputs,
                    "metadata": result.metadata,
                    "duration_s": round(duration, 3),
                }
                _print_status({
                    "node": node.key,
                    "stage": node.stage,
                    "status": "completed",
                    "duration_s": round(duration, 3),
                    "outputs": result.outputs,
                })

            manifest_path = os.path.join(output_root, "pipeline_manifest.json")
            try:
                with open(manifest_path, "w", encoding="utf-8") as fh:
                    json.dump(manifest, fh, indent=2)
            except Exception:
                pass

            total_duration = time.time() - pipeline_start
            _print_status({
                "pipeline": {
                    "output_root": output_root,
                    "nodes": ordered_nodes,
                    "duration_s": round(total_duration, 3),
                    "manifest": manifest_path,
                }
            })
        except Exception as e:
            print(f"[orchestrator] PIPELINE FAILED: {e}", file=sys.stderr, flush=True)
            try:
                logger.set_summary("orchestrator/status", "failed")
                logger.set_summary("orchestrator/error", str(e))
            except Exception:
                pass
            raise
