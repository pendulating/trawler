from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from dagspaces.common.config_schema import (
    PipelineGraphSpec,
    PipelineNodeSpec,
    OutputSpec,
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
_GPU_SANITIZE_PREFIX = "GRPO_TRAINING"


def _inject_prompt_from_file(cfg: DictConfig, prompt_filename: str) -> None:
    """Inject prompt from YAML file into cfg.prompt."""
    from dagspaces.common.orchestrator import _inject_prompt_from_file as _common_inject
    _common_inject(cfg, prompt_filename, config_dir=_CONF_DIR)


def _load_launcher_config(cfg: DictConfig, launcher_name: str) -> Optional[DictConfig]:
    """Load a launcher configuration using this dagspace's conf/ directory."""
    from dagspaces.common.orchestrator import _load_launcher_config as _common_load
    return _common_load(cfg, launcher_name, config_dir=_CONF_DIR)


def _get_wandb_logger(cfg: DictConfig, stage: str, run_id: Optional[str] = None, run_config: Optional[Dict[str, Any]] = None):
    """Get WandbLogger if enabled, otherwise return a no-op logger."""
    wb_config = WandbConfig.from_hydra_config(cfg)
    if wb_config.enabled:
        return WandbLogger(cfg, stage=stage, run_id=run_id, run_config=run_config)
    else:
        return _NoOpLogger(cfg, stage=stage, run_id=run_id, run_config=run_config)


from .runners.base import StageRunner

_STAGE_REGISTRY: Dict[str, StageRunner] = get_stage_registry()


def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single stage - designed to be submitted as a SLURM job."""
    from dagspaces.common.stage_utils import ensure_dotenv
    ensure_dotenv()
    from omegaconf import OmegaConf
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

    # Only sanitize GPUs for stages that need them
    _GPU_STAGES = {"sft_training", "grpo_training", "reward_prep", "norm_universe"}
    if node.stage in _GPU_STAGES:
        _sanitize_cuda_visible_devices(reason=f"job:{node.key}", env_prefix=_GPU_SANITIZE_PREFIX, cfg=cfg)

    wandb_run_id = node.wandb_suffix or node.key
    run_config = build_run_config(cfg, node, context.inputs, context.output_paths, dagspace_name="grpo_training")

    with _get_wandb_logger(cfg, stage=node.stage, run_id=wandb_run_id, run_config=run_config) as logger:
        try:
            context.logger = logger
            _print_status({"node": node.key, "stage": node.stage, "status": "running", "inputs": context.inputs})
            stage_start = time.time()

            result = runner.run(context)

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
            try:
                logger.set_summary(f"{node.stage}/status", "failed")
                logger.set_summary(f"{node.stage}/error", str(e))
            except Exception:
                pass
            raise


def run_experiment(cfg: DictConfig) -> None:
    """Execute the GRPO training pipeline."""
    with _get_wandb_logger(cfg, stage="orchestrator", run_id="monitor", run_config={"type": "pipeline"}) as logger:
        try:
            parent_group = logger.wb_config.group if logger.wb_config else None
            if parent_group:
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

                if node.launcher:
                    _print_status({"node": node.key, "stage": node.stage, "status": "submitting", "launcher": node.launcher, "inputs": inputs})
                    try:
                        launcher_cfg = _load_launcher_config(cfg, node.launcher)
                    except ValueError as e:
                        raise ValueError(f"Could not load launcher config '{node.launcher}' for node '{node.key}': {e}") from e

                    log_folder = None
                    try:
                        hydra_cfg = HydraConfig.get()
                        if hydra_cfg and hydra_cfg.runtime and hydra_cfg.runtime.output_dir:
                            hydra_output_dir = hydra_cfg.runtime.output_dir
                            log_folder = os.path.join(hydra_output_dir, ".slurm_jobs", node.key)
                    except Exception:
                        pass

                    if not log_folder:
                        log_folder = os.path.join(output_root, ".slurm_jobs", node.key)

                    log_folder = os.path.abspath(log_folder)
                    os.makedirs(log_folder, exist_ok=True)
                    job_name = f"GRPO-{node.key}"
                    executor = _create_submitit_executor(launcher_cfg, job_name, log_folder)

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
                                executor.update_parameters(slurm_setup=current_setup)
                        except Exception:
                            pass

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

                    with _clean_slurm_env():
                        job = executor.submit(execute_stage_job, context_data)
                    _print_status({"node": node.key, "stage": node.stage, "status": "submitted", "job_id": job.job_id})

                    try:
                        job_result = job.result()
                    except Exception as exc:
                        import subprocess as _sp
                        import pickle as _pkl
                        # NFS latency: result pickle may not be visible yet.
                        # Wait for SLURM job to finish, then retry reading result.
                        _resolved = False
                        for _wait in range(40):
                            check = _sp.run(["squeue", "-j", str(job.job_id), "-h", "-o", "%t"], capture_output=True, text=True)
                            state = check.stdout.strip()
                            if state and state in ("R", "PD", "CG"):
                                time.sleep(30)
                                continue
                            # Job finished — give NFS time to sync then try pickle
                            for _nfs_wait in range(6):
                                time.sleep(10)
                                if os.path.exists(job.paths.result_pickle):
                                    try:
                                        with open(job.paths.result_pickle, "rb") as f:
                                            _outcome, _result = _pkl.load(f)
                                            job_result = _result
                                            _resolved = True
                                    except Exception:
                                        continue
                                    break
                            break
                        if not _resolved:
                            _print_status({"node": node.key, "stage": node.stage, "status": "failed", "job_id": job.job_id, "error": str(exc)})
                            raise

                    # Submitit may return (outcome, payload) tuple or the payload directly
                    if isinstance(job_result, tuple) and len(job_result) == 2:
                        outcome, payload = job_result
                        if outcome == "error":
                            raise RuntimeError(f"SLURM job {job.job_id} failed:\n{payload}")
                        job_result = payload

                    if isinstance(job_result, StageResult):
                        result = job_result
                    elif isinstance(job_result, dict) and "outputs" in job_result:
                        result = StageResult(
                            outputs=job_result["outputs"],
                            metadata=job_result.get("metadata", {}),
                        )
                    else:
                        raise RuntimeError(
                            f"SLURM job {job.job_id} for node '{node.key}' returned unexpected "
                            f"result type {type(job_result).__name__}: {str(job_result)[:500]}"
                        )
                else:
                    _print_status({"node": node.key, "stage": node.stage, "status": "running", "inputs": inputs})
                    try:
                        _GPU_STAGES = {"sft_training", "grpo_training", "reward_prep", "norm_universe"}
                        if node.stage in _GPU_STAGES:
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
