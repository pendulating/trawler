from __future__ import annotations

import json
import os
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
    build_run_config,
    _probe_single_gpu,
    _update_slurm_gpu_envs,
    _adjust_tensor_parallel_env,
    _log_gpu_environment,
    _sanitize_cuda_visible_devices,
    _clean_slurm_env,
    _create_submitit_executor,
    _submit_slurm_job,
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
# Historical norms uses its own env-var prefix for GPU sanitisation book-keeping
_GPU_SANITIZE_PREFIX = "HISTORICAL_NORMS"


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


# StageRunner base class moved to .runners.base
# Import it from runners module when needed
from .runners.base import StageRunner


# All runner classes have been moved to the .runners module
# Import the stage registry from runners
_STAGE_REGISTRY: Dict[str, StageRunner] = get_stage_registry()


def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single stage - designed to be submitted as a SLURM job."""
    from dagspaces.common.stage_utils import ensure_dotenv
    ensure_dotenv()
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

    _sanitize_cuda_visible_devices(reason=f"job:{node.key}", env_prefix=_GPU_SANITIZE_PREFIX, cfg=cfg)
    
    # Execute stage with wandb logging context
    wandb_run_id = node.wandb_suffix or node.key
    run_config = build_run_config(cfg, node, context.inputs, context.output_paths, dagspace_name="historical_norms")

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
                    dataset_path = result.outputs["dataset"]
                    if dataset_path.endswith(".csv"):
                        df_out = pd.read_csv(dataset_path)
                    else:
                        df_out = pd.read_parquet(dataset_path)
                    # Use stage-specific logic for preferred columns
                    prefer_cols = None
                    if node.stage == "norm_reasoning":
                        prefer_cols = [
                            "norm_snippet", "reasoning_trace",
                            "preliminary_normative_force", "governs_information_flow",
                            "has_prescriptive_content", "norm_count", "article_text",
                        ]
                    elif node.stage == "norm_extraction":
                        prefer_cols = [c for c in df_out.columns if c.startswith("raz_")]
                        for extra in ["norm_snippet", "article_text", "gutenberg_id", "chunk_id"]:
                            if extra in df_out.columns and extra not in prefer_cols:
                                prefer_cols.append(extra)
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
                    
                    _safe_log_table(logger, df_out, f"{node.stage}/results", prefer_cols=prefer_cols)
                except Exception as e:
                    print(f"Warning: Failed to log output table for {node.key}: {e}", flush=True)

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
            }
            stage_registry = dict(_STAGE_REGISTRY)
            ordered_nodes = graph_spec.topological_order()
            pipeline_start = time.time()
            
            # Log pipeline structure to wandb: numeric to charts; structure to config
            logger.log_metrics({
                "orchestrator/total_nodes": len(ordered_nodes),
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
                
                # Check if this node should be launched as a separate SLURM job
                if node.launcher:
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
                    executor = _create_submitit_executor(launcher_cfg, job_name, log_folder)
                    
                    # Ensure child job uses parent's W&B group for proper grouping
                    # Submitit doesn't auto-inherit env vars, so we need to explicitly set them
                    if parent_group:
                        # Method 1: Set environment variable on executor
                        # This ensures it's available in the SLURM job's environment
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
                    context_data = {
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
                    
                    # Submit the job
                    job = _submit_slurm_job(executor, execute_stage_job, context_data, node.key, node.launcher)
                    
                    # Wait for the job to complete
                    try:
                        # Blocking call to get result
                        job_result = job.result()
                    except Exception as exc:
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

                    # Submitit may return (outcome, payload) tuple or the payload directly
                    if isinstance(job_result, tuple) and len(job_result) == 2:
                        outcome, payload = job_result
                        if outcome == "error":
                            raise RuntimeError(f"SLURM job {job.job_id} failed:\n{payload}")
                        job_result = payload

                    result = StageResult(
                        outputs=job_result["outputs"],
                        metadata=job_result["metadata"],
                    )
                else:
                    # Run locally in the current process
                    _print_status({"node": node.key, "stage": node.stage, "status": "running", "inputs": inputs})
                    try:
                        _sanitize_cuda_visible_devices(reason=f"node:{node.key}", env_prefix=_GPU_SANITIZE_PREFIX, cfg=node_cfg)
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
            print(f"[orchestrator] PIPELINE FAILED: {e}", file=sys.stderr, flush=True)
            try:
                logger.set_summary("orchestrator/status", "failed")
                logger.set_summary("orchestrator/error", str(e))
            except Exception:
                pass
            raise
