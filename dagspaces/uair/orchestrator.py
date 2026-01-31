from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
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
from .runners import get_stage_registry
from .wandb_logger import WandbLogger, WandbConfig

try:
    import ray  # type: ignore

    _RAY_AVAILABLE = True
except Exception:  # pragma: no cover - Ray optional dependency
    ray = None  # type: ignore
    _RAY_AVAILABLE = False

try:
    import submitit  # type: ignore
    _SUBMITIT_AVAILABLE = True
except Exception:
    submitit = None  # type: ignore
    _SUBMITIT_AVAILABLE = False


_STREAMING_COMPATIBLE_STAGES = {"classify_relevance", "taxonomy", "verification"}


class _NoOpLogger:
    """No-op logger that matches WandbLogger interface when wandb is disabled."""
    def __init__(self, cfg, stage: str, run_id: Optional[str] = None, run_config: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        self.stage = stage
        self.run_id = run_id
        self.run_config = run_config or {}
        self.wb_config = WandbConfig.from_hydra_config(cfg)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
        pass
    
    def log_table(self, df, key: str, prefer_cols: Optional[List[str]] = None, max_rows: Optional[int] = None, panel_group: Optional[str] = None) -> None:
        pass
    
    def set_summary(self, key: str, value: Any) -> None:
        pass
    
    def set_config(self, data: Dict[str, Any], allow_val_change: bool = True) -> None:
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _get_wandb_logger(cfg: DictConfig, stage: str, run_id: Optional[str] = None, run_config: Optional[Dict[str, Any]] = None):
    """Get WandbLogger if enabled, otherwise return a no-op logger.
    
    This ensures wandb initialization is completely skipped when wandb.enabled is False.
    """
    wb_config = WandbConfig.from_hydra_config(cfg)
    if wb_config.enabled:
        return WandbLogger(cfg, stage=stage, run_id=run_id, run_config=run_config)
    else:
        return _NoOpLogger(cfg, stage=stage, run_id=run_id, run_config=run_config)


def _probe_single_gpu(device: str) -> bool:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    code = (
        "import sys\n"
        "try:\n"
        "    import torch\n"
        "except Exception:\n"
        "    sys.exit(1)\n"
        "available = torch.cuda.is_available()\n"
        "count = torch.cuda.device_count() if available else 0\n"
        "if not (available and count >= 1):\n"
        "    sys.exit(1)\n"
        "try:\n"
        "    torch.cuda.set_device(0)\n"
        "    x = torch.randn((8, 8), device='cuda')\n"
        "    y = torch.randn((8, 8), device='cuda')\n"
        "    _ = torch.mm(x, y)\n"
        "    torch.cuda.synchronize()\n"
        "except Exception:\n"
        "    sys.exit(2)\n"
        "sys.exit(0)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable or "python", "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def _update_slurm_gpu_envs(valid_devices: List[str]) -> None:
    count = len(valid_devices)
    if count <= 0:
        return
    gpu_list = ",".join(valid_devices)
    for var in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE"):
        val = os.environ.get(var)
        if not val:
            continue
        if "," in val:
            os.environ[var] = gpu_list
        elif ":" in val:
            prefix = val.split(":", 1)[0]
            os.environ[var] = f"{prefix}:{count}"
        else:
            try:
                int(val)
                os.environ[var] = str(count)
            except Exception:
                os.environ[var] = gpu_list
    for var in ("SLURM_GPUS_PER_NODE", "SLURM_GPUS_PER_TASK"):
        val = os.environ.get(var)
        if not val:
            continue
        if ":" in val:
            prefix = val.split(":", 1)[0]
            os.environ[var] = f"{prefix}:{count}"
        else:
            try:
                current = int(val)
                os.environ[var] = str(min(count, current))
            except Exception:
                os.environ[var] = str(count)


def _adjust_tensor_parallel_env(valid_count: int) -> None:
    tp_env = os.environ.get("UAIR_TENSOR_PARALLEL_SIZE")
    if not tp_env:
        return
    try:
        tp_val = max(1, int(tp_env))
        if valid_count > 0 and tp_val > valid_count:
            os.environ["UAIR_TENSOR_PARALLEL_SIZE"] = str(valid_count)
    except Exception:
        pass


def _log_gpu_environment(reason: str) -> None:
    try:
        cuda_visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
        dropped = [d.strip() for d in os.environ.get("UAIR_SANITIZED_DROPPED_GPUS", "").split(",") if d.strip()]
        original = [d.strip() for d in os.environ.get("UAIR_GPU_SANITIZE_ORIGINAL", "").split(",") if d.strip()]
        payload: Dict[str, Any] = {
            "reason": reason,
            "cuda_visible_devices": cuda_visible,
        }
        if original:
            payload["sanitized_original"] = original
        if dropped:
            payload["sanitized_dropped"] = dropped
        tp_env = os.environ.get("UAIR_TENSOR_PARALLEL_SIZE")
        if tp_env:
            try:
                payload["tensor_parallel_size"] = int(tp_env)
            except Exception:
                payload["tensor_parallel_size"] = tp_env
        _print_status({"gpu_env": payload})
    except Exception:
        pass


def _sanitize_cuda_visible_devices(reason: str = "") -> None:
    if os.environ.get("UAIR_SKIP_GPU_SANITIZE"):
        return
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not current:
        return
    devices = [d.strip() for d in current.split(",") if d.strip()]
    if len(devices) <= 1:
        return
    normalized = ",".join(devices)
    valid: List[str] = []
    invalid: List[str] = []
    for dev in devices:
        if _probe_single_gpu(dev):
            valid.append(dev)
        else:
            invalid.append(dev)
    if not invalid:
        return
    if not valid:
        # If everything failed, do not modify CUDA_VISIBLE_DEVICES but log once
        os.environ["UAIR_GPU_SANITIZE_REASON"] = reason or "stage_start"
        os.environ["UAIR_GPU_SANITIZE_TS"] = str(int(time.time()))
        os.environ.pop("UAIR_GPU_SANITIZE_ORIGINAL", None)
        os.environ.pop("UAIR_SANITIZED_DROPPED_GPUS", None)
        _print_status({
            "gpu_sanitize": {
                "reason": reason or "stage_start",
                "original": normalized,
                "error": "all_devices_failed",
            }
        })
        return
    new_devices = ",".join(valid)
    os.environ["CUDA_VISIBLE_DEVICES"] = new_devices
    os.environ["UAIR_SANITIZED_DROPPED_GPUS"] = ",".join(invalid)
    os.environ["UAIR_GPU_SANITIZE_REASON"] = reason or "stage_start"
    os.environ["UAIR_GPU_SANITIZE_TS"] = str(int(time.time()))
    os.environ["UAIR_GPU_SANITIZE_ORIGINAL"] = normalized
    _update_slurm_gpu_envs(valid)
    _adjust_tensor_parallel_env(len(valid))
    _print_status({
        "gpu_sanitize": {
            "reason": reason or "stage_start",
            "original": normalized,
            "sanitized": new_devices,
            "dropped": ",".join(invalid),
        }
    })
    _log_gpu_environment(reason or "stage_start")


@dataclass
class StageExecutionContext:
    cfg: DictConfig
    node: PipelineNodeSpec
    inputs: Dict[str, str]
    output_paths: Dict[str, str]
    output_dir: str
    output_root: str
    logger: Optional['WandbLogger'] = None


@dataclass
class StageResult:
    outputs: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _convert_to_pandas_if_needed(out: Any) -> pd.DataFrame:
    """Convert Ray Dataset to pandas DataFrame if needed."""
    if hasattr(out, "to_pandas"):
        return out.to_pandas()
    return out


def _save_stage_outputs(out: pd.DataFrame, output_paths: Dict[str, str]) -> None:
    """Save DataFrame outputs to disk."""
    if isinstance(out, pd.DataFrame):
        for output_name, output_path in output_paths.items():
            out.to_parquet(output_path, index=False)


def _safe_log_table(
    logger: Optional['WandbLogger'],
    df: pd.DataFrame,
    key: str,
    prefer_cols: Optional[List[str]] = None,
    panel_group: str = "inspect_results"
) -> None:
    """Safely log DataFrame to wandb."""
    if logger and isinstance(df, pd.DataFrame):
        try:
            logger.log_table(df, key, prefer_cols=prefer_cols, panel_group=panel_group)
        except Exception as e:
            print(f"Warning: Failed to log {key} to wandb: {e}", flush=True)


# _compute_doc_level_verification moved to .runners.base


def _inject_prompt_from_file(cfg: DictConfig, prompt_filename: str) -> None:
    """Inject prompt from YAML file into cfg.prompt.
    
    Supports subdirectory paths like 'general_ai/classify.yaml' or just 'classify.yaml'.
    """
    try:
        base_dir = os.path.dirname(__file__)
        # Support subdirectory paths (e.g., 'general_ai/classify.yaml')
        prompt_path = os.path.join(base_dir, "conf", "prompt", prompt_filename)
        if os.path.exists(prompt_path):
            prompt_cfg = OmegaConf.load(prompt_path)
            ensure_section(cfg, "prompt")
            sys_p = prompt_cfg.get("system_prompt")
            usr_p = prompt_cfg.get("prompt_template")
            if sys_p:
                OmegaConf.update(cfg, "prompt.system_prompt", sys_p, merge=True)
            if usr_p:
                OmegaConf.update(cfg, "prompt.prompt_template", usr_p, merge=True)
    except Exception:
        pass  # Non-critical, stage may have defaults


# StageRunner base class moved to .runners.base
# Import it from runners module when needed
from .runners.base import StageRunner


class ArtifactRegistry:
    def __init__(self) -> None:
        self._artifacts: Dict[str, str] = {}

    def register_source(self, name: str, path: str) -> None:
        self._artifacts[name] = path

    def register_outputs(self, node_key: str, outputs: Mapping[str, str]) -> None:
        for out_name, out_path in outputs.items():
            self._artifacts[f"{node_key}.{out_name}"] = out_path

    def resolve(self, ref: str) -> str:
        if ref in self._artifacts:
            return self._artifacts[ref]
        candidate = os.path.abspath(os.path.expanduser(ref))
        if os.path.exists(candidate) or os.path.isabs(ref):
            return candidate
        raise KeyError(f"Unknown artifact reference '{ref}'")

    def resolve_output_path(self, path: str, output_root: str, node_key: str) -> str:
        if not path:
            raise ValueError(f"Node '{node_key}' output path is empty")
        resolved = path
        if not os.path.isabs(resolved):
            resolved = os.path.join(output_root, resolved)
        return os.path.abspath(resolved)


def clone_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))  # type: ignore[return-value]


def merge_overrides(base_cfg: DictConfig, overrides: Optional[Mapping[str, Any]]) -> DictConfig:
    if not overrides:
        return base_cfg
    # Apply each override using OmegaConf.update to properly handle dot notation
    for key, value in overrides.items():
        OmegaConf.update(base_cfg, key, value, merge=True)
    return base_cfg


def ensure_section(cfg: DictConfig, section: str) -> None:
    if OmegaConf.select(cfg, section) is None:
        OmegaConf.update(cfg, section, {}, merge=True)


def common_parent(paths: Iterable[str]) -> Optional[str]:
    try:
        parents = [os.path.dirname(p) for p in paths]
        if not parents:
            return None
        return os.path.commonpath(parents)
    except Exception:
        return None


def prepare_node_config(base_cfg: DictConfig, node: PipelineNodeSpec, output_dir: str) -> DictConfig:
    cfg_copy = clone_config(base_cfg)
    cfg_copy = merge_overrides(cfg_copy, node.overrides)
    ensure_section(cfg_copy, "runtime")
    OmegaConf.update(cfg_copy, "runtime.stage", node.stage, merge=True)
    OmegaConf.update(cfg_copy, "runtime.output_dir", output_dir, merge=True)
    OmegaConf.update(cfg_copy, "runtime.output_csv", None, merge=True)
    return cfg_copy


def _load_parquet_dataset(parquet_path: str, columns: Mapping[str, str], debug: bool, sample_n: Optional[int]) -> pd.DataFrame:
    if not isinstance(parquet_path, str) or parquet_path.strip() == "":
        raise ValueError("data.parquet_path is required")
    if not os.path.isabs(parquet_path):
        parquet_path = os.path.abspath(parquet_path)
    df = pd.read_parquet(parquet_path)
    col_map = {
        columns.get("article_text", "article_text"): "article_text",
        columns.get("article_path", "article_path"): "article_path",
        columns.get("country", "country"): "country",
        columns.get("year", "year"): "year",
        columns.get("article_id", "article_id"): "article_id",
    }
    present = {src: dst for src, dst in col_map.items() if src in df.columns}
    if present:
        df = df.rename(columns=present)
    if "article_text" not in df.columns and "chunk_text" not in df.columns:
        raise RuntimeError("Parquet missing required text column (article_text) or chunk_text")

    def _safe_str(x: Any) -> str:
        if x is None:
            return ""
        try:
            return "" if (isinstance(x, float) and pd.isna(x)) else str(x).strip()
        except Exception:
            return str(x) if x is not None else ""

    for column in ("article_path", "country", "year", "article_id"):
        if column not in df.columns:
            df[column] = None
        else:
            try:
                df[column] = df[column].apply(_safe_str)
            except Exception:
                pass

    # Apply sample_n regardless of debug flag - it's a runtime limit, not just for debugging
    if isinstance(sample_n, int) and sample_n > 0:
        try:
            n = min(int(sample_n), int(len(df)))
        except Exception:
            n = int(sample_n)
        try:
            seed_env = os.environ.get("UAIR_SAMPLE_SEED", "777")
            seed = int(seed_env) if seed_env is not None else 777
        except Exception:
            seed = 777
        try:
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)
            print(f"[_load_parquet_dataset] Applied sample_n={n} (seed={seed}), processing {len(df)} rows", flush=True)
        except Exception:
            df = df.head(n)
            print(f"[_load_parquet_dataset] Applied sample_n={n} (head), processing {len(df)} rows", flush=True)
    return df


def _parse_cpus_on_node(val: str) -> int:
    """Parse SLURM_CPUS_ON_NODE value which can be in various formats."""
    try:
        v = val.strip()
        if "(x" in v and v.endswith(")"):
            import re as _re
            m = _re.match(r"^(\d+)\(x(\d+)\)$", v)
            if m:
                return max(1, int(m.group(1)) * int(m.group(2)))
        if "," in v:
            acc = 0
            for p in v.split(","):
                acc += int(p)
            return max(1, acc)
        return max(1, int(v))
    except Exception:
        return -1


def _ensure_ray_init_with_cpu_limits(cfg: DictConfig) -> None:
    """Initialize Ray with SLURM-aware CPU limits for orchestrator use."""
    if not _RAY_AVAILABLE or ray.is_initialized():
        return
    
    # Detect SLURM CPU allocation
    cpus_alloc = None
    try:
        cpt = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpt is not None and str(cpt).strip() != "":
            cpus_alloc = int(cpt)
        else:
            con = os.environ.get("SLURM_CPUS_ON_NODE")
            if con is not None and str(con).strip() != "":
                cpus_alloc = _parse_cpus_on_node(con)
    except Exception:
        cpus_alloc = None
    
    # Get memory configuration
    try:
        job_mem_gb = int(getattr(cfg.runtime, "job_memory_gb", 64) or 64)
    except Exception:
        job_mem_gb = 64
    
    try:
        obj_store_bytes = int(max(1, job_mem_gb) * (1024 ** 3) * 0.90)
    except Exception:
        obj_store_bytes = int(64 * (1024 ** 3) * 0.90)
    
    namespace = os.environ.get("RAY_NAMESPACE") or os.environ.get("WANDB_GROUP") or "uair"
    
    # Detect SLURM GPU allocation
    gpus_alloc = None
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible.strip():
            gpus_alloc = len([x for x in cuda_visible.split(",") if x.strip()])
        else:
            slurm_gpus = os.environ.get("SLURM_GPUS_ON_NODE") or os.environ.get("SLURM_GPUS_PER_NODE")
            if slurm_gpus:
                if ":" in slurm_gpus:
                    gpus_alloc = int(slurm_gpus.split(":")[-1])
                else:
                    try:
                        gpus_alloc = int(slurm_gpus)
                    except ValueError:
                        gpus_alloc = None
    except Exception:
        gpus_alloc = None
    
    # Try full initialization, fallback to basic if that fails
    try:
        init_kwargs = {"log_to_driver": True, "object_store_memory": obj_store_bytes, "namespace": str(namespace)}
        if cpus_alloc is not None and int(cpus_alloc) > 0:
            init_kwargs["num_cpus"] = int(cpus_alloc)
        ray.init(**init_kwargs)
    except Exception:
        # Fallback: basic initialization
        init_kwargs = {"log_to_driver": True}
        if cpus_alloc is not None and int(cpus_alloc) > 0:
            init_kwargs["num_cpus"] = int(cpus_alloc)
        ray.init(**init_kwargs)
    
    # Note: Ray Data CPU limits will be set in _prepare_streaming_dataset with proper parallelism
    # Don't override here to allow for higher concurrency than raw CPU count


def _log_parquet_metadata(dataset_path: str) -> tuple[Optional[int], Optional[int], Optional[float]]:
    """Log parquet metadata for debugging. Returns metadata tuple for compatibility."""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(dataset_path)
        metadata = pf.metadata
        num_row_groups = metadata.num_row_groups
        
        row_group_sizes = []
        for i in range(num_row_groups):
            rg = metadata.row_group(i)
            total_bytes = rg.total_byte_size
            row_group_sizes.append(total_bytes)
        
        if not row_group_sizes:
            return None, None, None
        
        max_size = max(row_group_sizes)
        avg_size = sum(row_group_sizes) / len(row_group_sizes)
        max_rg_size_mb = max_size / (1024 ** 2)
        avg_rg_size_mb = avg_size / (1024 ** 2)
        
        print(f"[_prepare_streaming_dataset] Parquet metadata: {num_row_groups} row groups, "
              f"max {max_rg_size_mb:.1f} MB, avg {avg_rg_size_mb:.1f} MB per row group", flush=True)
        
        return num_row_groups, max_size, avg_size
    except Exception:
        return None, None, None


def _calculate_target_blocks(
    num_row_groups: Optional[int],
    max_rg_size_bytes: Optional[int],
    size_bytes: Optional[int],
    cfg: DictConfig
) -> int:
    """Calculate target number of blocks for repartitioning based on file size and row group structure."""
    target_block_size_bytes = 64 * 1024 * 1024  # 64MB decompressed target (reduced from 128MB)
    
    # If we have row group metadata, use it for better estimation
    if num_row_groups is not None and max_rg_size_bytes is not None:
        max_rg_size_mb = max_rg_size_bytes / (1024 ** 2)
        
        # If row groups are too large (>128MB compressed), split aggressively
        # Reduced threshold from 256MB to 128MB for more aggressive splitting
        if max_rg_size_mb > 128:
            # Estimate decompressed size (conservative: assume 3x expansion)
            estimated_decompressed_mb = max_rg_size_mb * 3
            # Target: split each large row group into multiple blocks (64MB target)
            blocks_per_row_group = max(12, int(estimated_decompressed_mb / 64))  # More blocks per row group
            target_blocks = max(150, num_row_groups * blocks_per_row_group)  # Increased minimum
            target_blocks = min(target_blocks, 600)  # Increased cap to 600 blocks
            print(f"[_prepare_streaming_dataset] Large row groups detected. Splitting into {target_blocks} blocks "
                  f"({blocks_per_row_group} blocks per row group)", flush=True)
            return target_blocks
    
    # Standard calculation based on file size
    # Use smaller target block size (64MB instead of 128MB) for more aggressive splitting
    target_block_size_bytes = 64 * 1024 * 1024  # 64MB decompressed target
    if size_bytes and size_bytes > 0:
        size_gb = size_bytes / float(1024 ** 3)
        estimated_decompressed_bytes = size_bytes * 3  # Assume 3x expansion
        target_blocks = max(100, int(estimated_decompressed_bytes / target_block_size_bytes))
        target_blocks = min(max(100, target_blocks), 300)  # Increased max to 300 for more blocks
        estimated_block_size_mb = estimated_decompressed_bytes / target_blocks / (1024**2)
        print(f"[_prepare_streaming_dataset] Dataset: {size_gb:.2f} GB compressed, estimated {estimated_decompressed_bytes/(1024**3):.2f} GB decompressed, targeting {target_blocks} blocks (~{estimated_block_size_mb:.1f} MB per block)", flush=True)
        return target_blocks
    
    # Fallback: use CPU-based defaults
    try:
        cpus_alloc = None
        cpt = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpt is not None and str(cpt).strip() != "":
            cpus_alloc = int(cpt)
        else:
            con = os.environ.get("SLURM_CPUS_ON_NODE")
            if con is not None and str(con).strip() != "":
                cpus_alloc = _parse_cpus_on_node(con)
        target_blocks = max(100, (cpus_alloc if cpus_alloc and cpus_alloc > 0 else 8) * 15)
        return min(target_blocks, 200)
    except Exception:
        return 100  # Safe default


def _prepare_streaming_dataset(dataset_path: str, columns: Mapping[str, str], cfg: DictConfig, stage: str) -> tuple[Optional[Any], bool]:
    if not _RAY_AVAILABLE:
        return None, False
    if stage not in _STREAMING_COMPATIBLE_STAGES:
        return None, False
    try:
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.abspath(dataset_path)
        # Ensure Ray is initialized with CPU limits BEFORE reading parquet
        _ensure_ray_init_with_cpu_limits(cfg)
        if not ray.is_initialized():
            # Fallback initialization if the above didn't work
            namespace = os.environ.get("RAY_NAMESPACE") or os.environ.get("WANDB_GROUP") or "uair"
            try:
                ray.init(log_to_driver=True, namespace=str(namespace))
            except Exception:
                ray.init(log_to_driver=True)
        
        # Configure Ray Data to use smaller block sizes and explicit memory limits
        # This prevents creating huge 13+GB blocks that cause OOM issues
        try:
            ctx = ray.data.DataContext.get_current()
            # Set VERY conservative block sizes to keep memory bounded (1MB min, 64MB max)
            # Reduced from 128MB to 64MB to be even more aggressive about splitting
            ctx.target_min_block_size = 1 * 1024 * 1024  # 1MB
            ctx.target_max_block_size = 64 * 1024 * 1024  # 64MB (reduced from 128MB)
            ctx.execution_options.verbose_progress = False
            
            # Set explicit memory limits for Ray Data tasks
            # This is critical - Ray needs to know how much memory each task needs
            try:
                # Limit concurrent tasks to reduce peak memory usage
                # Use 1/4 of available CPUs or 4, whichever is smaller
                cpus_alloc = None
                try:
                    cpt = os.environ.get("SLURM_CPUS_PER_TASK")
                    if cpt is not None and str(cpt).strip() != "":
                        cpus_alloc = int(cpt)
                    else:
                        con = os.environ.get("SLURM_CPUS_ON_NODE")
                        if con is not None and str(con).strip() != "":
                            cpus_alloc = _parse_cpus_on_node(con)
                except Exception:
                    cpus_alloc = None
                
                # Limit parallelism to reduce concurrent memory usage
                # Each task needs ~5GB (one per row group), so limit concurrent tasks
                # Use up to 16 concurrent tasks for optimal throughput
                # Allow more tasks than CPUs since they can be I/O bound
                if cpus_alloc is not None and cpus_alloc > 0:
                    max_concurrent_tasks = min(cpus_alloc * 2, 16)  # Allow 2x CPUs up to 16
                else:
                    max_concurrent_tasks = 16  # Default to 16 if CPU count unknown
                
                # Set memory per task to 5GB (one task per row group)
                # This tells Ray to allocate 5GB memory per task
                ctx.execution_options.resource_limits = ctx.execution_options.resource_limits.copy(
                    cpu=max_concurrent_tasks,
                    memory=5 * 1024 * 1024 * 1024  # 5GB per task
                )
                
                print(f"[_prepare_streaming_dataset] Configured Ray Data: max_concurrent_tasks={max_concurrent_tasks}, memory_per_task=5GB", flush=True)
            except Exception as e:
                print(f"[_prepare_streaming_dataset] Warning: Failed to set Ray Data memory limits: {e}", flush=True)
            
            # Enable dynamic block splitting - Ray will automatically split blocks that exceed max size
            try:
                import ray.data.context as ray_data_ctx
                # Set factor for dynamic block splitting (default is 1.5)
                # Lower values = more aggressive splitting
                ray_data_ctx.MAX_SAFE_BLOCK_SIZE_FACTOR = 1.1  # Split blocks that exceed 1.1x max size (more aggressive)
            except Exception:
                pass
        except Exception:
            pass
        
        # Log parquet metadata for debugging
        num_row_groups, max_rg_size_bytes, avg_rg_size_bytes = _log_parquet_metadata(dataset_path)
        
        # Read parquet file(s) - Ray creates one block per row group
        # With proper row grouping (small row groups from us_agg.py), this creates many manageable blocks
        # No repartitioning needed - rely on the natural partitioning from row groups
        print(f"[_prepare_streaming_dataset] Reading parquet with {num_row_groups or 'unknown'} row groups (no repartitioning)", flush=True)
        ds = ray.data.read_parquet(dataset_path)
        
        col_map = {
            columns.get("article_text", "article_text"): "article_text",
            columns.get("article_path", "article_path"): "article_path",
            columns.get("country", "country"): "country",
            columns.get("year", "year"): "year",
            columns.get("article_id", "article_id"): "article_id",
        }

        def _ensure_canon(row: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(row)
            for src, dst in col_map.items():
                if dst not in out and src in row:
                    out[dst] = row.get(src)
            return out

        try:
            # Note: Memory limits are set via DataContext.execution_options.resource_limits above
            ds = ds.map(_ensure_canon)
        except Exception:
            pass
        debug = bool(getattr(cfg.runtime, "debug", False))
        sample_n = getattr(cfg.runtime, "sample_n", None)
        # Apply sample_n regardless of debug flag - it's a runtime limit, not just for debugging
        if isinstance(sample_n, int) and sample_n > 0:
            try:
                ds = ds.limit(max(1, int(sample_n)))
                print(f"[_prepare_streaming_dataset] Applied sample_n={sample_n}, limiting dataset to {sample_n} rows", flush=True)
            except Exception:
                pass
        return ds, True
    except Exception:
        return None, False


def _estimate_dataset_size_bytes(path: str) -> Optional[int]:
    if not path:
        return None
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        if os.path.isdir(path):
            total = 0
            for root, _, files in os.walk(path):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        continue
            return total
    except OSError:
        return None
    return None


def prepare_stage_input(cfg: DictConfig, dataset_path: str, stage: str) -> tuple[Optional[pd.DataFrame], Optional[Any], bool]:
    debug = bool(getattr(cfg.runtime, "debug", False))
    sample_n = getattr(cfg.runtime, "sample_n", None)
    columns = dict(getattr(cfg.data, "columns", {})) if getattr(cfg, "data", None) else {}
    runtime_cfg = getattr(cfg, "runtime", None)

    if dataset_path and not os.path.isabs(dataset_path):
        dataset_path = os.path.abspath(dataset_path)

    streaming_enabled = bool(getattr(runtime_cfg, "streaming_io", False)) if runtime_cfg is not None else False
    auto_stream_attempted = False

    if stage in _STREAMING_COMPATIBLE_STAGES and not streaming_enabled:
        auto_streaming_enabled = True if runtime_cfg is None else bool(getattr(runtime_cfg, "auto_streaming_io", True))
        raw_threshold = 1.0 if runtime_cfg is None else getattr(runtime_cfg, "auto_streaming_min_file_gb", 1.0)
        threshold_gb: Optional[float]
        try:
            threshold_candidate = float(raw_threshold)
            threshold_gb = threshold_candidate if threshold_candidate > 0 else None
        except Exception:
            threshold_gb = None

        if auto_streaming_enabled and threshold_gb is not None:
            size_bytes = _estimate_dataset_size_bytes(dataset_path)
            if size_bytes is not None:
                size_gb = size_bytes / float(1024 ** 3)
                if size_gb >= threshold_gb:
                    streaming_enabled = True
                    auto_stream_attempted = True
                    print(
                        f"[prepare_stage_input] Auto-enabled streaming IO for stage '{stage}' on dataset '{dataset_path}' (size {size_gb:.2f} GB >= threshold {threshold_gb:.2f} GB)",
                        flush=True,
                    )
            else:
                print(
                    f"[prepare_stage_input] Unable to determine dataset size for '{dataset_path}'; continuing without auto streaming",
                    flush=True,
                )

    ds = None
    use_streaming = False
    if streaming_enabled:
        ds, use_streaming = _prepare_streaming_dataset(dataset_path, columns, cfg, stage)
        if auto_stream_attempted and not use_streaming:
            print(
                f"[prepare_stage_input] Streaming IO requested for stage '{stage}' but Ray streaming is unavailable; falling back to pandas load",
                flush=True,
            )

    df = None
    if not use_streaming:
        df = _load_parquet_dataset(dataset_path, columns, debug=debug, sample_n=sample_n)
    return df, ds, use_streaming


def _collect_outputs(context: StageExecutionContext, optional: Mapping[str, bool]) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for key, path in context.output_paths.items():
        if os.path.exists(path):
            resolved[key] = path
        else:
            if optional.get(key, False):
                continue
            raise FileNotFoundError(
                f"Expected output '{key}' for node '{context.node.key}' at '{path}' not found"
            )
    return resolved


# All runner classes have been moved to the .runners module
# Import the stage registry from runners
_STAGE_REGISTRY: Dict[str, StageRunner] = get_stage_registry()


def _ensure_output_dirs(paths: Iterable[str]) -> None:
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _node_optional_outputs(node: PipelineNodeSpec) -> Dict[str, bool]:
    return {name: spec.optional for name, spec in node.outputs.items()}


def _node_output_paths(node: PipelineNodeSpec, registry: ArtifactRegistry, output_root: str) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for out_name, spec in node.outputs.items():
        resolved[out_name] = registry.resolve_output_path(spec.path, output_root, node.key)
    _ensure_output_dirs(resolved.values())
    return resolved


def _node_inputs(node: PipelineNodeSpec, registry: ArtifactRegistry) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for alias, ref in node.inputs.items():
        resolved[alias] = registry.resolve(ref)
    return resolved


def _print_status(payload: Dict[str, Any]) -> None:
    try:
        print(json.dumps(payload, indent=2))
    except Exception:
        pass


def _load_launcher_config(cfg: DictConfig, launcher_name: str) -> Optional[DictConfig]:
    """Load a launcher configuration from Hydra config."""
    try:
        # Find the config path - use the location of this file as reference
        config_path = os.path.join(os.path.dirname(__file__), "conf")
        
        if not os.path.exists(config_path):
            # Try to get from hydra runtime
            hydra_cfg = getattr(cfg, "hydra", None)
            if hydra_cfg:
                runtime_cfg = getattr(hydra_cfg, "runtime", None)
                if runtime_cfg:
                    sources = getattr(runtime_cfg, "config_sources", [])
                    for source in sources:
                        if hasattr(source, "provider") and source.provider == "main":
                            config_path = source.path
                            break
        
        if not config_path or not os.path.exists(config_path):
            raise ValueError(f"Could not find config directory")
            
        launcher_file = os.path.join(config_path, "hydra", "launcher", f"{launcher_name}.yaml")
        if not os.path.exists(launcher_file):
            raise ValueError(f"Launcher config file not found: {launcher_file}")
        
        # Load the launcher config
        launcher_cfg = OmegaConf.load(launcher_file)
        # Resolve interpolations with the main config as context
        launcher_cfg = OmegaConf.merge({"runtime": cfg.get("runtime", {})}, launcher_cfg)
        return launcher_cfg
    except Exception as e:
        raise ValueError(f"Failed to load launcher config '{launcher_name}': {e}") from e


def _create_submitit_executor(launcher_cfg: DictConfig, job_name: str, log_folder: str) -> Any:
    """Create a submitit executor from launcher configuration."""
    if not _SUBMITIT_AVAILABLE or submitit is None:
        raise RuntimeError("submitit is not available but is required for SLURM job submission")
    
    executor = submitit.AutoExecutor(folder=log_folder)
    
    # Map launcher config to submitit parameters
    executor.update_parameters(
        timeout_min=int(launcher_cfg.get("timeout_min", 120)),
        slurm_partition=str(launcher_cfg.get("partition", "pierson")),
        slurm_mem=f"{int(launcher_cfg.get('mem_gb', 8))}GB",
        slurm_cpus_per_task=int(launcher_cfg.get("cpus_per_task", 2)),
        slurm_gpus_per_node=int(launcher_cfg.get("gpus_per_node", 0)),
        slurm_nodes=int(launcher_cfg.get("nodes", 1)),
        slurm_tasks_per_node=int(launcher_cfg.get("tasks_per_node", 1)),
        slurm_array_parallelism=int(launcher_cfg.get("array_parallelism", 1)),
        name=job_name,
        slurm_additional_parameters=launcher_cfg.get("additional_parameters", {}),
        slurm_setup=launcher_cfg.get("setup", []),
    )
    
    return executor


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
    
    _sanitize_cuda_visible_devices(reason=f"job:{node.key}")
    _log_gpu_environment(reason=f"job:{node.key}")

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
                    job_name = f"UAIR-{node.key}"
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
                    job = executor.submit(execute_stage_job, context_data)
                    _print_status({"node": node.key, "stage": node.stage, "status": "submitted", "job_id": job.job_id})
                    
                    # Wait for the job to complete
                    try:
                        job_result = job.result()  # This blocks until the job completes
                        result = StageResult(
                            outputs=job_result["outputs"],
                            metadata=job_result["metadata"],
                        )
                    except Exception as exc:
                        _print_status({"node": node.key, "stage": node.stage, "status": "failed", "job_id": job.job_id, "error": str(exc)})
                        raise
                else:
                    # Run locally in the current process
                    _print_status({"node": node.key, "stage": node.stage, "status": "running", "inputs": inputs})
                    try:
                        _sanitize_cuda_visible_devices(reason=f"node:{node.key}")
                        _log_gpu_environment(reason=f"node:{node.key}")
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
