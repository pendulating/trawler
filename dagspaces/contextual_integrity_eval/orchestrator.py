from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from dagspaces.common.config_schema import (
    PipelineGraphSpec,
    PipelineNodeSpec,
    load_pipeline_graph,
    resolve_output_root,
)
from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
    ArtifactRegistry,
    _NoOpLogger,
    clone_config,
    merge_overrides,
    ensure_section,
    common_parent,
    prepare_node_config,
    _load_parquet_dataset,
    prepare_stage_input,
    _collect_outputs,
    _ensure_output_dirs,
    _node_optional_outputs,
    _node_output_paths,
    _node_inputs,
    _save_stage_outputs,
    _safe_log_table,
    _print_status,
    _probe_single_gpu,
    _update_slurm_gpu_envs,
    _adjust_tensor_parallel_env,
    _log_gpu_environment,
    _sanitize_cuda_visible_devices,
    _clean_slurm_env,
    _create_submitit_executor,
)
from .runners import get_stage_registry
from .wandb_logger import WandbLogger, WandbConfig

try:
    import submitit  # type: ignore
    _SUBMITIT_AVAILABLE = True
except Exception:
    submitit = None  # type: ignore
    _SUBMITIT_AVAILABLE = False

_CONF_DIR = os.path.join(os.path.dirname(__file__), "conf")

_GPU_BUNDLE_STAGE_KEYS = ("qa_probe_eval", "agent_action_eval")
_GPU_BUNDLE_LAUNCHERS = {"g2_slurm_pierson_clean"}


def _inject_prompt_from_file(cfg: DictConfig, prompt_filename: str) -> None:
    """Inject prompt from YAML file into cfg.prompt."""
    from dagspaces.common.orchestrator import _inject_prompt_from_file as _common_inject
    _common_inject(cfg, prompt_filename, config_dir=_CONF_DIR)


def _load_launcher_config(cfg: DictConfig, launcher_name: str) -> Optional[DictConfig]:
    """Load a launcher configuration using this dagspace's conf/ directory."""
    from dagspaces.common.orchestrator import _load_launcher_config as _common_load
    return _common_load(cfg, launcher_name, config_dir=_CONF_DIR)


def _get_wandb_logger(cfg: DictConfig, stage: str, run_id: Optional[str] = None, run_config: Optional[Dict[str, Any]] = None):
    """Get WandbLogger if enabled, otherwise return a no-op logger.

    This ensures wandb initialization is completely skipped when wandb.enabled is False.
    """
    wb_config = WandbConfig.from_hydra_config(cfg)
    if wb_config.enabled:
        return WandbLogger(cfg, stage=stage, run_id=run_id, run_config=run_config)
    else:
        return _NoOpLogger(cfg, stage=stage, run_id=run_id, run_config=run_config)


# _compute_doc_level_verification moved to .runners.base

# StageRunner base class moved to .runners.base
# Import it from runners module when needed
from .runners.base import StageRunner


# All runner classes have been moved to the .runners module
# Import the stage registry from runners
_STAGE_REGISTRY: Dict[str, StageRunner] = get_stage_registry()


def _runtime_ray_mode(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "runtime.ray.mode", default="per_stage")).strip().lower()


def _runtime_gpu_execution_mode(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "runtime.gpu.execution_mode", default="mixed")).strip().lower()


def _runtime_gpu_bundle_stage_keys(cfg: DictConfig) -> List[str]:
    keys = OmegaConf.select(cfg, "runtime.gpu.bundle_stage_keys", default=None)
    if isinstance(keys, list) and keys:
        return [str(k).strip() for k in keys if str(k).strip()]
    return list(_GPU_BUNDLE_STAGE_KEYS)


def _runtime_gpu_bundle_launchers(cfg: DictConfig) -> set[str]:
    launchers = OmegaConf.select(cfg, "runtime.gpu.bundle_launchers", default=None)
    if isinstance(launchers, list) and launchers:
        return {str(v).strip() for v in launchers if str(v).strip()}
    return set(_GPU_BUNDLE_LAUNCHERS)


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


def _list_descendant_pids(root_pid: int) -> List[int]:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=", "ppid="],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = proc.stdout.splitlines()
    except Exception:
        return []

    children: Dict[int, List[int]] = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except Exception:
            continue
        children.setdefault(ppid, []).append(pid)

    descendants: List[int] = []
    stack = list(children.get(root_pid, []))
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        descendants.append(pid)
        stack.extend(children.get(pid, []))
    return descendants


def _reap_descendant_processes(root_pid: int, grace_s: float = 5.0) -> Dict[str, Any]:
    descendants = [pid for pid in _list_descendant_pids(root_pid) if pid != root_pid]
    if not descendants:
        return {"terminated": 0, "killed": 0, "remaining": 0}

    terminated = 0
    for pid in descendants:
        try:
            os.kill(pid, signal.SIGTERM)
            terminated += 1
        except Exception:
            pass
    time.sleep(max(0.0, grace_s))

    remaining = [pid for pid in descendants if os.path.exists(f"/proc/{pid}")]
    killed = 0
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except Exception:
            pass
    time.sleep(0.5)
    still_alive = [pid for pid in descendants if os.path.exists(f"/proc/{pid}")]
    return {"terminated": terminated, "killed": killed, "remaining": len(still_alive)}


def _emit_privacylens_metrics(logger: Any, stage: str, df_out: pd.DataFrame) -> None:
    """Emit PrivacyLens benchmark metrics from stage outputs."""
    if df_out.empty:
        return

    metrics: Dict[str, float] = {}
    summary: Dict[str, float] = {}

    if stage == "qa_probe_eval":
        axis_map = {
            "qa_S_correct": "accuracy_s",
            "qa_V_correct": "accuracy_v",
            "qa_T_correct": "accuracy_t",
            "qa_overall_correct": "accuracy_overall",
        }
        for col, suffix in axis_map.items():
            if col in df_out.columns:
                val = float(df_out[col].astype(float).mean())
                metrics[f"{stage}/{suffix}"] = val
                summary[f"{stage}/{suffix}"] = val
    elif stage == "agent_action_eval":
        if "leak_flag" in df_out.columns:
            lr = float(df_out["leak_flag"].astype(float).mean())
            metrics[f"{stage}/leakage_rate"] = lr
            summary[f"{stage}/leakage_rate"] = lr
    elif stage == "statistical_analysis":
        required = {"metric", "mean", "ci_low", "ci_high"}
        if required.issubset(set(df_out.columns)):
            for _, row in df_out.iterrows():
                metric_name = str(row.get("metric", "")).strip()
                if not metric_name:
                    continue
                mean = float(row.get("mean"))
                ci_low = float(row.get("ci_low"))
                ci_high = float(row.get("ci_high"))
                metrics[f"{stage}/{metric_name}_mean"] = mean
                metrics[f"{stage}/{metric_name}_ci_low"] = ci_low
                metrics[f"{stage}/{metric_name}_ci_high"] = ci_high
                summary[f"{stage}/{metric_name}_mean"] = mean

    if metrics:
        logger.log_metrics(metrics)
    for key, val in summary.items():
        try:
            logger.set_summary(key, val)
        except Exception:
            pass


def execute_gpu_bundle_job(bundle_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a bundle of GPU stages sequentially in a single SLURM job.

    vLLM runs as an isolated HTTP server subprocess (``vllm serve``),
    managing its own CUDA initialization and multi-GPU tensor parallelism.
    No Ray cluster is needed.  The server is started on first use by
    ``_build_vllm()`` and shut down in the finally block via
    ``cleanup_vllm_runtime()``.
    """
    stages = bundle_data.get("stages", [])
    if not isinstance(stages, list) or not stages:
        raise ValueError("GPU bundle job requires a non-empty 'stages' list")

    started_at = time.time()
    node_results: Dict[str, Dict[str, Any]] = {}
    try:
        for stage_data in stages:
            node_key = str(stage_data.get("node", {}).get("key", "unknown"))
            stage_started = time.time()
            stage_result = execute_stage_job(stage_data)
            node_results[node_key] = {
                "outputs": stage_result["outputs"],
                "metadata": stage_result["metadata"],
                "duration_s": round(time.time() - stage_started, 3),
            }
        total_s = round(time.time() - started_at, 3)
        return {"node_results": node_results, "duration_s": total_s}
    finally:
        try:
            from .stages.runtime_shared import cleanup_vllm_runtime
            cleanup_info = cleanup_vllm_runtime()
            _print_status({"gpu_bundle": {"status": "vllm_cleanup", **cleanup_info}})
        except Exception as cleanup_exc:
            _print_status({"gpu_bundle": {"status": "vllm_cleanup_failed", "error": str(cleanup_exc)}})

        reap = _reap_descendant_processes(os.getpid(), grace_s=5.0)
        _print_status({"gpu_bundle": {"status": "process_reap", **reap}})


def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single stage - designed to be submitted as a SLURM job."""
    # Reconstruct context from serialized data
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(context_data["cfg"])
    node_dict = context_data["node"]

    # Reconstruct PipelineNodeSpec
    from dagspaces.common.config_schema import PipelineNodeSpec, OutputSpec
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

    # Get the stage runner
    stage_registry = dict(_STAGE_REGISTRY)
    runner = stage_registry.get(node.stage)
    if runner is None:
        raise ValueError(f"No runner registered for stage '{node.stage}' (node '{node.key}')")

    # Execute stage with wandb logging context
    wandb_run_id = node.wandb_suffix or node.key
    run_config = {
        "node": node.key,
        "stage": node.stage,
        "inputs": list(context.inputs.keys()),
        "outputs": list(context.output_paths.keys()),
    }

    with _get_wandb_logger(cfg, stage=node.stage, run_id=wandb_run_id, run_config=run_config) as logger:
        try:
            # Update context with logger
            context.logger = logger

            # Execute the stage
            _print_status({"node": node.key, "stage": node.stage, "status": "running", "inputs": context.inputs})
            stage_start = time.time()

            result = runner.run(context)

            # Log output table if available
            if result.outputs and "dataset" in result.outputs:
                try:
                    df_out = pd.read_parquet(result.outputs["dataset"])
                    # Use stage-specific logic for preferred columns
                    prefer_cols = None
                    if node.stage == "norm_reasoning":
                        prefer_cols = ["norm_snippet", "reasoning_trace", "potential_type", "article_text"]
                    elif node.stage == "norm_extraction":
                        # Log flattened IG components
                        prefer_cols = [c for c in df_out.columns if c.startswith("ig20_")]
                        if "norm_snippet" in df_out.columns:
                            prefer_cols.append("norm_snippet")
                    elif node.stage == "ci_reasoning":
                        prefer_cols = [
                            "has_information_exchange", "ci_flow_count",
                            "ci_reasoning_json", "ci_reasoning_parse_error",
                            "article_text", "gutenberg_id", "chunk_id",
                        ]
                    elif node.stage == "ci_extraction":
                        # Log flattened CI tuple components first, then metadata
                        prefer_cols = [c for c in df_out.columns if c.startswith("ci_")]
                        for extra in ["article_text", "gutenberg_id", "chunk_id"]:
                            if extra in df_out.columns and extra not in prefer_cols:
                                prefer_cols.append(extra)
                    elif node.stage == "qa_probe_eval":
                        prefer_cols = [
                            "record_id",
                            "baseline_arm",
                            "qa_S_correct",
                            "qa_V_correct",
                            "qa_T_correct",
                            "qa_overall_correct",
                            "qa_S_predicted_label",
                            "qa_V_predicted_label",
                            "qa_T_predicted_label",
                        ]
                    elif node.stage == "agent_action_eval":
                        prefer_cols = [
                            "record_id",
                            "baseline_arm",
                            "trajectory_prompt",
                            "generated_action",
                            "leak_judge_text",
                            "leak_probability",
                            "leak_flag",
                        ]
                    elif node.stage == "statistical_analysis":
                        prefer_cols = ["metric", "mean", "ci_low", "ci_high"]
                    elif node.stage == "summarize_results":
                        prefer_cols = [
                            "baseline_arm",
                            "qa_accuracy",
                            "leakage_rate",
                            "utility_score",
                            "delta_vs_zero_shot",
                        ]

                    _safe_log_table(logger, df_out, f"{node.stage}/results", prefer_cols=prefer_cols)
                except Exception as e:
                    print(f"Warning: Failed to log output table for {node.key}: {e}", flush=True)
                try:
                    _emit_privacylens_metrics(logger, node.stage, df_out)
                except Exception as e:
                    print(f"Warning: Failed to log benchmark metrics for {node.key}: {e}", flush=True)

            # Log completion metrics
            duration_s = time.time() - stage_start
            try:
                logger.set_summary(f"{node.stage}/status", "completed")
            except Exception:
                pass
            logger.log_metrics({
                f"{node.stage}/duration_s": duration_s,
                f"{node.stage}/rows_processed": result.metadata.get("rows", 0),
            })

            return {
                "outputs": result.outputs,
                "metadata": result.metadata,
            }
        except Exception as e:
            # Log failure
            try:
                logger.set_summary(f"{node.stage}/status", "failed")
                logger.set_summary(f"{node.stage}/error", str(e))
            except Exception:
                pass
            raise


def run_experiment(cfg: DictConfig) -> None:
    # Execute entire pipeline with wandb logging context
    with _get_wandb_logger(cfg, stage="orchestrator", run_id="monitor", run_config={"type": "pipeline"}) as logger:
        try:
            # Get the parent/monitor group ID to pass to child jobs
            # This ensures all stages in one pipeline run are grouped together
            parent_group = logger.wb_config.group if logger.wb_config else None
            if parent_group:
                # Set in environment so child jobs can inherit it
                os.environ["WANDB_GROUP"] = parent_group
                print(f"[orchestrator] Setting WANDB_GROUP={parent_group} for child stages", flush=True)

            graph_spec: PipelineGraphSpec = load_pipeline_graph(cfg)
            output_root = resolve_output_root(graph_spec, cfg)
            os.makedirs(output_root, exist_ok=True)
            registry = ArtifactRegistry()
            for source_key, source in graph_spec.sources.items():
                path = source.path
                if not os.path.isabs(path):
                    path = os.path.abspath(os.path.expanduser(path))
                registry.register_source(source_key, path)
            manifest: Dict[str, Any] = {
                "output_root": output_root,
                "nodes": {},
                "runtime": {
                    "ray_mode": _runtime_ray_mode(cfg),
                    "gpu_execution_mode": _runtime_gpu_execution_mode(cfg),
                    "gpu_bundle_stage_keys": _runtime_gpu_bundle_stage_keys(cfg),
                },
            }
            stage_registry = dict(_STAGE_REGISTRY)
            ordered_nodes = graph_spec.topological_order()
            pipeline_start = time.time()
            ray_mode = _runtime_ray_mode(cfg)
            gpu_exec_mode = _runtime_gpu_execution_mode(cfg)
            bundle_stage_keys = _runtime_gpu_bundle_stage_keys(cfg)
            bundle_launchers = _runtime_gpu_bundle_launchers(cfg)
            ordered_gpu_bundle_keys = [k for k in ordered_nodes if k in bundle_stage_keys]
            bundle_enabled = (
                ray_mode == "per_run"
                and gpu_exec_mode in {"mixed", "sequential_reuse"}
                and len(ordered_gpu_bundle_keys) == len(bundle_stage_keys)
                and all(graph_spec.nodes[k].launcher in bundle_launchers for k in ordered_gpu_bundle_keys)
            )
            bundle_cache: Dict[str, Dict[str, Any]] = {}
            bundle_launched = False
            if bundle_enabled:
                _print_status(
                    {
                        "gpu_bundle": {
                            "enabled": True,
                            "ray_mode": ray_mode,
                            "execution_mode": gpu_exec_mode,
                            "stages": ordered_gpu_bundle_keys,
                        }
                    }
                )

            # Log pipeline structure to wandb: numeric to charts; structure to config
            logger.log_metrics({
                "orchestrator/total_nodes": len(ordered_nodes),
                "orchestrator/ray_per_run_enabled": 1 if ray_mode == "per_run" else 0,
                "orchestrator/gpu_bundle_enabled": 1 if bundle_enabled else 0,
            })
            try:
                logger.set_config({
                    "orchestrator": {
                        "node_order": ordered_nodes,
                        "total_nodes": len(ordered_nodes),
                    }
                })
            except Exception:
                pass

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
                cached_bundle_entry = bundle_cache.get(node_key)
                if cached_bundle_entry is not None:
                    result = StageResult(
                        outputs=dict(cached_bundle_entry["outputs"]),
                        metadata=dict(cached_bundle_entry["metadata"]),
                    )
                    duration = float(cached_bundle_entry.get("duration_s", time.time() - node_start))
                    registry.register_outputs(node.key, result.outputs)
                    manifest["nodes"][node.key] = {
                        "stage": node.stage,
                        "inputs": inputs,
                        "outputs": result.outputs,
                        "metadata": result.metadata,
                        "duration_s": duration,
                    }
                    _print_status({
                        "node": node.key,
                        "stage": node.stage,
                        "status": "completed",
                        "duration_s": round(duration, 3),
                        "outputs": result.outputs,
                        "source": "gpu_bundle_cache",
                    })
                    continue

                # Check if this node should be launched as a separate SLURM job
                if node.launcher:
                    if bundle_enabled and not bundle_launched and node_key == ordered_gpu_bundle_keys[0]:
                        _print_status({
                            "gpu_bundle": {
                                "status": "submitting",
                                "trigger_node": node_key,
                                "stages": ordered_gpu_bundle_keys,
                            }
                        })
                        launcher_cfg = _load_launcher_config(cfg, node.launcher)
                        bundle_contexts: List[Dict[str, Any]] = []
                        for bundle_node_key in ordered_gpu_bundle_keys:
                            bundle_node = graph_spec.nodes[bundle_node_key]
                            bundle_inputs = _node_inputs(bundle_node, registry)
                            bundle_output_paths = _node_output_paths(bundle_node, registry, output_root)
                            bundle_output_dir = common_parent(bundle_output_paths.values()) or os.path.join(output_root, bundle_node.key)
                            os.makedirs(bundle_output_dir, exist_ok=True)
                            bundle_node_cfg = prepare_node_config(cfg, bundle_node, bundle_output_dir)
                            bundle_contexts.append(
                                _serialize_context_data(
                                    bundle_node_cfg,
                                    bundle_node,
                                    bundle_inputs,
                                    bundle_output_paths,
                                    bundle_output_dir,
                                    output_root,
                                )
                            )

                        bundle_log_folder = None
                        try:
                            hydra_cfg = HydraConfig.get()
                            if hydra_cfg and hydra_cfg.runtime and hydra_cfg.runtime.output_dir:
                                bundle_log_folder = os.path.join(hydra_cfg.runtime.output_dir, ".slurm_jobs", "gpu_bundle")
                        except Exception:
                            bundle_log_folder = None
                        if not bundle_log_folder:
                            bundle_log_folder = os.path.join(output_root, ".slurm_jobs", "gpu_bundle")
                        bundle_log_folder = os.path.abspath(bundle_log_folder)
                        os.makedirs(bundle_log_folder, exist_ok=True)

                        bundle_executor = _create_submitit_executor(launcher_cfg, "HNORMS-gpu_bundle", bundle_log_folder, use_srun=False)
                        if parent_group:
                            try:
                                current_setup = list(launcher_cfg.get("setup", []))
                                insert_idx = 0
                                for i, cmd in enumerate(current_setup):
                                    if "source" in cmd or "export HYDRA_FULL_ERROR" in cmd:
                                        insert_idx = i + 1
                                wandb_group_export = f"export WANDB_GROUP={parent_group}"
                                if wandb_group_export not in current_setup:
                                    current_setup.insert(insert_idx, wandb_group_export)
                                    bundle_executor.update_parameters(slurm_setup=current_setup)
                            except Exception:
                                pass

                        bundle_payload = {
                            "ray_mode": ray_mode,
                            "execution_mode": gpu_exec_mode,
                            "stages": bundle_contexts,
                        }
                        with _clean_slurm_env():
                            bundle_job = bundle_executor.submit(execute_gpu_bundle_job, bundle_payload)
                        _print_status({"gpu_bundle": {"status": "submitted", "job_id": bundle_job.job_id}})
                        bundle_result = bundle_job.result()
                        node_results = dict(bundle_result.get("node_results", {}))
                        if not node_results:
                            raise RuntimeError("GPU bundle job completed without node results")
                        for bundle_node_key, bundle_node_result in node_results.items():
                            bundle_cache[bundle_node_key] = {
                                "outputs": dict(bundle_node_result.get("outputs", {})),
                                "metadata": dict(bundle_node_result.get("metadata", {})),
                                "duration_s": float(bundle_node_result.get("duration_s", 0.0)),
                            }
                        bundle_launched = True
                        cached_now = bundle_cache.get(node_key)
                        if cached_now is None:
                            raise RuntimeError(f"GPU bundle did not return results for node '{node_key}'")
                        result = StageResult(outputs=dict(cached_now["outputs"]), metadata=dict(cached_now["metadata"]))
                        duration = float(cached_now.get("duration_s", time.time() - node_start))
                        registry.register_outputs(node.key, result.outputs)
                        manifest["nodes"][node.key] = {
                            "stage": node.stage,
                            "inputs": inputs,
                            "outputs": result.outputs,
                            "metadata": result.metadata,
                            "duration_s": duration,
                        }
                        _print_status({
                            "node": node.key,
                            "stage": node.stage,
                            "status": "completed",
                            "duration_s": round(duration, 3),
                            "outputs": result.outputs,
                            "source": "gpu_bundle",
                        })
                        continue

                    _print_status({"node": node.key, "stage": node.stage, "status": "submitting", "launcher": node.launcher, "inputs": inputs})
                    try:
                        launcher_cfg = _load_launcher_config(cfg, node.launcher)
                    except ValueError as e:
                        raise ValueError(f"Could not load launcher config '{node.launcher}' for node '{node.key}': {e}") from e

                    # Create submitit executor - store logs in the Hydra multirun directory
                    # Structure: multirun/YYYY-MM-DD/HH-MM-SS/0/.slurm_jobs/STAGE_NAME/
                    log_folder = None
                    try:
                        # Priority 1: Use HydraConfig to get runtime output directory
                        hydra_cfg = HydraConfig.get()
                        if hydra_cfg and hydra_cfg.runtime and hydra_cfg.runtime.output_dir:
                            hydra_output_dir = hydra_cfg.runtime.output_dir
                            log_folder = os.path.join(hydra_output_dir, ".slurm_jobs", node.key)
                            _print_status({"debug": "using_hydra_output_dir", "log_folder": log_folder})
                    except Exception as e:
                        _print_status({"debug": "hydra_config_error", "error": str(e)})

                    # Priority 2: Fall back to output_root
                    if not log_folder:
                        log_folder = os.path.join(output_root, ".slurm_jobs", node.key)
                        _print_status({"debug": "using_output_root_fallback", "log_folder": log_folder, "output_root": output_root})

                    log_folder = os.path.abspath(log_folder)
                    os.makedirs(log_folder, exist_ok=True)
                    job_name = f"HNORMS-{node.key}"
                    executor = _create_submitit_executor(launcher_cfg, job_name, log_folder, use_srun=False)

                    # Ensure child job uses parent's W&B group for proper grouping
                    # Submitit doesn't auto-inherit env vars, so we need to explicitly set them
                    if parent_group:
                        try:
                            # Get current setup commands and prepend WANDB_GROUP export
                            current_setup = list(launcher_cfg.get("setup", []))
                            # Insert explicit WANDB_GROUP export at the beginning (after shebang/source commands)
                            # Find insertion point (after source commands)
                            insert_idx = 0
                            for i, cmd in enumerate(current_setup):
                                if "source" in cmd or "export HYDRA_FULL_ERROR" in cmd:
                                    insert_idx = i + 1
                            # Insert WANDB_GROUP export
                            wandb_group_export = f"export WANDB_GROUP={parent_group}"
                            if wandb_group_export not in current_setup:
                                current_setup.insert(insert_idx, wandb_group_export)
                                executor.update_parameters(slurm_setup=current_setup)
                                _print_status({"debug": "injected_wandb_group", "group": parent_group, "node": node.key})
                        except Exception as e:
                            _print_status({"debug": "failed_to_inject_wandb_group", "error": str(e)})

                    # Prepare serializable context data
                    context_data = _serialize_context_data(
                        node_cfg,
                        node,
                        inputs,
                        output_paths,
                        output_dir,
                        output_root,
                    )

                    # Submit the job
                    with _clean_slurm_env():
                        job = executor.submit(execute_stage_job, context_data)
                    _print_status({"node": node.key, "stage": node.stage, "status": "submitted", "job_id": job.job_id})

                    # Wait for the job to complete
                    job_result: Optional[Dict[str, Any]] = None
                    try:
                        # Blocking call to get result
                        job_result = job.result()
                    except Exception as exc:
                        # Submitit can race on networked filesystems: state may be
                        # TERMINATING before result pickle metadata is visible. In
                        # that case, explicitly wait for the result pickle and retry.
                        exc_text = str(exc)
                        exc_text_lower = exc_text.lower()
                        result_path = job.paths.result_pickle
                        missing_result_artifact = (
                            "has not produced any output" in exc_text_lower
                            or "result_pickle" in exc_text_lower
                            or ("result" in exc_text_lower and "pickle" in exc_text_lower)
                        )
                        if missing_result_artifact:
                            try:
                                wait_s = int(OmegaConf.select(cfg, "runtime.submitit_result_wait_s", default=120))
                                deadline = time.time() + wait_s
                                _print_status({
                                    "debug": "waiting_for_result_pickle",
                                    "job_id": job.job_id,
                                    "path": str(result_path),
                                    "max_wait_s": wait_s,
                                })
                                while time.time() < deadline and not os.path.exists(result_path):
                                    time.sleep(2)
                                if os.path.exists(result_path):
                                    with open(result_path, "rb") as f:
                                        import pickle
                                        _outcome, _result = pickle.load(f)
                                        job_result = _result
                                    _print_status({
                                        "debug": "recovered_result_after_wait",
                                        "job_id": job.job_id,
                                    })
                                    missing_result_artifact = False
                            except Exception as wait_exc:
                                _print_status({
                                    "debug": "result_pickle_wait_failed",
                                    "job_id": job.job_id,
                                    "error": str(wait_exc),
                                })
                        elif os.path.exists(result_path):
                            try:
                                with open(result_path, "rb") as f:
                                    import pickle
                                    _outcome, _result = pickle.load(f)
                                    job_result = _result
                                _print_status({
                                    "debug": "recovered_result_after_exception",
                                    "job_id": job.job_id,
                                })
                            except Exception:
                                pass

                        if missing_result_artifact and job_result is None:
                            # Check if the job was actually cancelled or just misreported
                            try:
                                # Use squeue to check if it's still alive
                                import subprocess
                                check = subprocess.run(["squeue", "-j", str(job.job_id), "-h", "-o", "%t"], capture_output=True, text=True)
                                if check.stdout.strip() in ("R", "PD", "CG"):
                                    _print_status({"debug": "job_misreported_as_failed", "job_id": job.job_id, "state": check.stdout.strip()})
                                    # Fallback to manual wait
                                    while True:
                                        time.sleep(30)
                                        check = subprocess.run(["squeue", "-j", str(job.job_id), "-h", "-o", "%t"], capture_output=True, text=True)
                                        if not check.stdout.strip():
                                            break
                                        if check.stdout.strip() not in ("R", "PD", "CG"):
                                            break
                                    # Try one last time to get the result from file
                                    if os.path.exists(job.paths.result_pickle):
                                        with open(job.paths.result_pickle, "rb") as f:
                                            import pickle
                                            _outcome, _result = pickle.load(f)
                                            job_result = _result
                                    else:
                                        _print_status({"node": node.key, "stage": node.stage, "status": "failed", "job_id": job.job_id, "error": str(exc)})
                                        raise
                                else:
                                    _print_status({"node": node.key, "stage": node.stage, "status": "failed", "job_id": job.job_id, "error": str(exc)})
                                    raise
                            except Exception as inner_exc:
                                _print_status({"node": node.key, "stage": node.stage, "status": "failed", "job_id": job.job_id, "error": f"{exc} (inner: {inner_exc})"})
                                raise
                        if job_result is None:
                            # Any non-recovered failure should propagate as before.
                            raise

                    result = StageResult(
                        outputs=job_result["outputs"],
                        metadata=job_result["metadata"],
                    )
                else:
                    # Run locally in the current process
                    _print_status({"node": node.key, "stage": node.stage, "status": "running", "inputs": inputs})
                    try:
                        result = runner.run(context)
                    except Exception as exc:
                        _print_status({"node": node.key, "stage": node.stage, "status": "failed", "error": str(exc)})
                        raise

                registry.register_outputs(node.key, result.outputs)
                duration = time.time() - node_start
                manifest["nodes"][node.key] = {
                    "stage": node.stage,
                    "inputs": inputs,
                    "outputs": result.outputs,
                    "metadata": result.metadata,
                    "duration_s": duration,
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

            # Log final pipeline metrics to wandb
            try:
                logger.set_summary("orchestrator/status", "completed")
            except Exception:
                pass
            logger.log_metrics({
                "orchestrator/total_duration_s": round(total_duration, 3),
                "orchestrator/nodes_completed": len(manifest["nodes"]),
            })

            _print_status({
                "pipeline": {
                    "output_root": output_root,
                    "nodes": ordered_nodes,
                    "duration_s": round(total_duration, 3),
                    "manifest": manifest_path,
                }
            })
        except Exception as e:
            try:
                logger.set_summary("orchestrator/status", "failed")
                logger.set_summary("orchestrator/error", str(e))
            except Exception:
                pass
            raise
