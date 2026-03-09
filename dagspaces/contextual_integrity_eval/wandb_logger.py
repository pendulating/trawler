"""Centralized W&B logging for contextual_integrity_eval dagspace.

This module provides a unified interface for W&B logging across all pipeline stages,
handling distributed execution (Ray, SLURM) and run lifecycle management.

Key features:
- Single source of truth for W&B configuration
- Proper context management for runs
- Online mode by default for real-time syncing (configurable via WANDB_MODE)
- Service daemon completely disabled using wandb.Settings() API
- Ray-aware: automatically skips initialization in Ray workers to avoid socket conflicts
- Thread-safe logging
- Graceful degradation when W&B is unavailable

Configuration (wandb 0.22.0+ best practices):
- Service daemon disabled via WANDB_DISABLE_SERVICE environment variable
- In-process mode is used instead of service daemon for SLURM/Ray compatibility
- Parameters like 'mode' and 'dir' passed directly to wandb.init()
- Settings() object used only for specific options (disable_git, disable_job_creation, etc.)
- WANDB_DIR (optional): specify writable directory for wandb files (defaults to SLURM_SUBMIT_DIR or CWD)
- WANDB_MODE: set to "offline" for deferred syncing, "online" (default) for real-time

Best Practices:
- Use WandbLogger context manager for run lifecycle
- All logging goes through log_metrics/log_table/log_artifact methods
- Never call wandb.init() or wandb.finish() directly outside this module
- Settings object properly configures wandb for distributed environments

References:
- wandb.Settings docs: https://docs.wandb.ai/ref/python/settings
- Distributed training guide: https://docs.wandb.ai/guides/track/advanced/distributed-training
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import platform
import socket
import traceback
import getpass
import shutil

# Configure local storage for temporary files (Ray, vLLM, W&B)
# MUST be set before importing wandb or initializing Ray
# Use /scratch if available (klara), otherwise /tmp
def _ensure_local_tmpdir():
    current_tmp = os.environ.get("TMPDIR", "")
    if not current_tmp or current_tmp.startswith("/share"):
        local_base = "/scratch" if os.path.exists("/scratch") else "/tmp"
        new_tmp = os.path.join(local_base, getpass.getuser(), "contextual_integrity_eval")
        try:
            os.makedirs(new_tmp, exist_ok=True)
            os.environ["TMPDIR"] = new_tmp
            # Also set WANDB_DIR to a safe local path if not set, 
            # but keep it on network if the user wants artifacts preserved.
            # WANDB_DIR usually defaults to CWD which is /share/...
        except Exception:
            pass

_ensure_local_tmpdir()

# Apply defaults from repo-level wandb/settings before importing wandb
def _apply_wandb_settings_defaults() -> None:
    try:
        # Allow explicit override via environment
        settings_path = os.environ.get("WANDB_SETTINGS_PATH")
        if not settings_path:
            # Default to repo root: <repo>/wandb/settings
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            settings_path = os.path.join(base_dir, "wandb", "settings")
        if not (isinstance(settings_path, str) and os.path.exists(settings_path)):
            return
        import configparser  # Local import to avoid global dependency
        cp = configparser.ConfigParser()
        try:
            cp.read(settings_path)
        except Exception:
            return
        sect = "default" if cp.has_section("default") else (cp.sections()[0] if cp.sections() else None)
        if not sect:
            return
        sec = cp[sect]
        # Extract values; only set env vars if not already present to honor explicit overrides
        try:
            entity = sec.get("entity", fallback=None)
        except Exception:
            entity = None
        try:
            project = sec.get("project", fallback=None)
        except Exception:
            project = None
        try:
            base_url = sec.get("base_url", fallback=None)
        except Exception:
            base_url = None
        if entity and not os.environ.get("WANDB_ENTITY"):
            os.environ["WANDB_ENTITY"] = str(entity)
        if project and not os.environ.get("WANDB_PROJECT"):
            os.environ["WANDB_PROJECT"] = str(project)
        if base_url and not os.environ.get("WANDB_BASE_URL"):
            os.environ["WANDB_BASE_URL"] = str(base_url)
    except Exception:
        # Silent best-effort; never fail pipeline due to settings parsing
        pass

_apply_wandb_settings_defaults()

# Import wandb after environment is configured
import wandb as wandb_module

# In-process mode: do not require legacy service; service is disabled via env

# Import base classes after wandb is configured
from omegaconf import DictConfig, OmegaConf

@dataclass
class WandbConfig:
    """W&B configuration extracted from Hydra config."""
    
    enabled: bool = False
    project: str = "nov10-workshop"
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    table_sample_rows: int = 1000
    table_sample_seed: int = 777
    
    @classmethod
    def from_hydra_config(cls, cfg) -> "WandbConfig":
        """Extract W&B config from Hydra config."""
        try:
            wandb_cfg = getattr(cfg, "wandb", None)
            # Pick up environment defaults (possibly injected from wandb/settings)
            env_entity = os.environ.get("WANDB_ENTITY")
            env_project = os.environ.get("WANDB_PROJECT")
            if wandb_cfg is None:
                return cls(
                    enabled=False,
                    project=(env_project or "nov10-workshop"),
                    entity=(env_entity if env_entity and env_entity.strip() else None),
                    group=_get_group_from_config(cfg),
                    tags=[],
                    table_sample_rows=1000,
                    table_sample_seed=777,
                )
            
            # Resolve project with fallback to environment, then default
            proj_attr = getattr(wandb_cfg, "project", None)
            if proj_attr is None or str(proj_attr).strip() == "":
                project = env_project or "nov10-workshop"
            else:
                project = str(proj_attr or "nov10-workshop")
            # Resolve entity with fallback to environment
            entity_cfg = _get_optional_str(wandb_cfg, "entity")
            entity = entity_cfg or (env_entity if env_entity and env_entity.strip() else None)
            return cls(
                enabled=bool(getattr(wandb_cfg, "enabled", False)),
                project=project,
                entity=entity,
                group=_get_group_from_config(cfg),
                tags=_get_list(wandb_cfg, "tags"),
                table_sample_rows=int(getattr(wandb_cfg, "table_sample_rows", 1000)),
                table_sample_seed=int(getattr(wandb_cfg, "table_sample_seed", 777)),
            )
        except Exception as e:
            print(f"[wandb] Warning: Failed to parse config: {e}", file=sys.stderr)
            return cls()


def _get_optional_str(obj, attr: str) -> Optional[str]:
    """Get optional string attribute, returning None if empty."""
    try:
        val = getattr(obj, attr, None)
        if val is not None and str(val).strip():
            return str(val)
    except Exception:
        pass
    return None


def _get_list(obj, attr: str) -> List[str]:
    """Get list attribute safely."""
    try:
        val = getattr(obj, attr, None)
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            return [str(x) for x in val]
        return [str(val)]
    except Exception:
        return []


def _get_group_from_config(cfg) -> Optional[str]:
    """Extract group from config with fallback to environment variables.
    
    Priority order:
    1. cfg.wandb.group
    2. WANDB_GROUP env var
    3. SUBMITIT_JOB_ID (parent SLURM job)
    4. SLURM_JOB_ID (current SLURM job)
    """
    # Priority 1: Explicit config
    try:
        grp = getattr(cfg.wandb, "group", None)
        if grp and str(grp).strip():
            return str(grp)
    except Exception:
        pass
    
    # Priority 2: Environment variable
    env_group = os.environ.get("WANDB_GROUP")
    if env_group and env_group.strip():
        return env_group
    
    # Priority 3: Submitit parent job ID
    submitit_job_id = os.environ.get("SUBMITIT_JOB_ID")
    if submitit_job_id and submitit_job_id.strip():
        return f"slurm-{submitit_job_id}"
    
    # Priority 4: Current SLURM job ID
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id and slurm_job_id.strip():
        return f"slurm-{slurm_job_id}"
    
    return None


def _nvidia_smi_query(*fields: str) -> Optional[str]:
    """Run nvidia-smi --query-gpu and return stdout, or None on failure.

    IMPORTANT: This avoids importing torch / initializing CUDA in the
    orchestrator process, which would poison forked subprocess GPU visibility
    (PyTorch caches cudaGetDeviceCount in a C++ static, and the NVML-based
    device_count() can disagree with it in child processes).
    """
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip()
    except Exception:
        return None


def _detect_num_gpus() -> int:
    """Detect the number of GPUs allocated to this job.

    Uses only environment variables and nvidia-smi — never imports torch
    so that CUDA is not initialized in the orchestrator process.
    """
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible.strip():
            gpu_indices = [x.strip() for x in cuda_visible.split(",") if x.strip()]
            if gpu_indices:
                return len(gpu_indices)
    except Exception:
        pass

    try:
        slurm_gpus = os.environ.get("SLURM_GPUS_PER_NODE") or os.environ.get("SLURM_GPUS_ON_NODE")
        if slurm_gpus:
            try:
                if ":" in slurm_gpus:
                    return int(slurm_gpus.split(":")[-1])
                return int(slurm_gpus)
            except Exception:
                pass
    except Exception:
        pass

    smi = _nvidia_smi_query("index")
    if smi:
        return len(smi.splitlines())

    return 0


def _detect_gpu_type() -> Optional[str]:
    """Detect the GPU type/model name via nvidia-smi (no torch import)."""
    smi = _nvidia_smi_query("name")
    if smi:
        return smi.splitlines()[0].strip()
    return None


def _parse_cpus_on_node(val: str) -> int:
    """Parse SLURM_CPUS_ON_NODE format (e.g., '32', '16(x2)', '2,2')."""
    if not isinstance(val, str):
        return -1
    try:
        v = val.strip()
        if "(x" in v and v.endswith(")"):
            import re as _re
            m = _re.match(r"^(\d+)\(x(\d+)\)$", v)
            if m:
                a = int(m.group(1))
                b = int(m.group(2))
                return max(1, a * b)
        if "," in v:
            parts = [p for p in v.split(",") if p.strip()]
            acc = 0
            for p in parts:
                try:
                    acc += int(p)
                except Exception:
                    return -1
            return max(1, acc)
        return max(1, int(v))
    except Exception:
        return -1


def _detect_num_cpus() -> Optional[int]:
    """Detect the number of CPUs allocated to this job.
    
    Priority order:
    1. SLURM_CPUS_PER_TASK or SLURM_CPUS_ON_NODE (job allocation)
    2. os.cpu_count() (system total)
    """
    try:
        cpt = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpt is not None and str(cpt).strip() != "":
            return int(cpt)
        
        con = os.environ.get("SLURM_CPUS_ON_NODE")
        if con is not None and str(con).strip() != "":
            cpus = _parse_cpus_on_node(con)
            if cpus > 0:
                return cpus
    except Exception:
        pass
    
    # Fallback to system CPU count
    try:
        return os.cpu_count()
    except Exception:
        pass
    
    return None


def _read_int_file(path: str) -> int:
    """Read an integer from a file, handling 'max' keyword."""
    try:
        with open(path, "r") as f:
            s = f.read().strip()
        if s.lower() == "max":
            return -1
        return int(s)
    except Exception:
        return -1


def _detect_cgroup_mem_limit_bytes() -> int:
    """Return cgroup memory limit in bytes when available; otherwise -1.
    
    Supports cgroup v2 (memory.max) and v1 (memory.limit_in_bytes).
    """
    # cgroup v2
    v2 = "/sys/fs/cgroup/memory.max"
    lim = _read_int_file(v2)
    if lim > 0:
        # Filter out unrealistically large values (no limit)
        try:
            if lim > (1 << 56):
                return -1
        except Exception:
            pass
        return lim
    
    # cgroup v1
    v1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    lim = _read_int_file(v1)
    if lim > 0:
        try:
            if lim > (1 << 56):
                return -1
        except Exception:
            pass
        return lim
    
    return -1


def _detect_slurm_job_mem_bytes() -> int:
    """Infer SLURM job memory allocation in bytes from env vars.
    
    Prefers SLURM_MEM_PER_NODE; otherwise uses SLURM_MEM_PER_CPU * SLURM_CPUS_ON_NODE.
    Values are MB according to SLURM docs.
    """
    try:
        mem_per_node_mb = os.environ.get("SLURM_MEM_PER_NODE")
        if mem_per_node_mb:
            mb = int(mem_per_node_mb)
            if mb > 0:
                return mb * 1024 * 1024
    except Exception:
        pass
    
    try:
        mem_per_cpu_mb = os.environ.get("SLURM_MEM_PER_CPU")
        cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE")
        if mem_per_cpu_mb and cpus_on_node:
            mb = int(mem_per_cpu_mb)
            cpus = _parse_cpus_on_node(cpus_on_node)
            if mb > 0 and cpus > 0:
                return mb * cpus * 1024 * 1024
    except Exception:
        pass
    
    return -1


def _detect_memory_gb() -> Optional[float]:
    """Detect available memory in GB.
    
    Priority order:
    1. cgroup memory limit (container/SLURM cgroup)
    2. SLURM env-based memory inference
    3. System total memory (psutil/os.sysconf)
    """
    # cgroup limit first
    cg = _detect_cgroup_mem_limit_bytes()
    if cg > 0:
        return cg / (1024 ** 3)
    
    # SLURM-derived
    sj = _detect_slurm_job_mem_bytes()
    if sj > 0:
        return sj / (1024 ** 3)
    
    # Fallback to system total
    try:
        import psutil  # type: ignore
        tot = int(getattr(psutil.virtual_memory(), "total", 0))
        if tot > 0:
            return tot / (1024 ** 3)
    except Exception:
        pass
    
    try:
        tot = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        if tot > 0:
            return tot / (1024 ** 3)
    except Exception:
        pass
    
    return None


def _collect_gpu_details() -> List[Dict[str, Any]]:
    """Collect detailed information for each GPU via nvidia-smi (no torch).

    Avoids CUDA initialization in the orchestrator process, which would
    cache a stale device count and poison subprocess GPU visibility.
    """
    gpus: List[Dict[str, Any]] = []
    smi = _nvidia_smi_query("index", "name", "memory.total")
    if not smi:
        return gpus
    for line in smi.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_memory_gb": round(float(parts[2]) / 1024.0, 2),
                })
            except (ValueError, IndexError):
                pass
    return gpus


def collect_compute_metadata(cfg=None) -> Dict[str, Any]:
    """Collect comprehensive compute metadata for wandb logging.
    
    Captures system configuration including:
    - CPU count and architecture
    - GPU count, type, and memory
    - RAM allocation
    - SLURM job parameters
    - Python environment
    - Model configuration (if available in cfg)
    
    Args:
        cfg: Optional Hydra config to extract model/runtime parameters
    
    Returns:
        Dictionary with compute metadata suitable for wandb.config
    """
    metadata: Dict[str, Any] = {}
    
    # System info
    try:
        metadata["system"] = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
        }
    except Exception:
        pass
    
    # CPU info
    num_cpus = _detect_num_cpus()
    if num_cpus is not None:
        metadata["compute.cpu_count"] = num_cpus
    
    try:
        metadata["compute.cpu_architecture"] = platform.machine()
    except Exception:
        pass
    
    # GPU info
    num_gpus = _detect_num_gpus()
    metadata["compute.gpu_count"] = num_gpus
    
    if num_gpus > 0:
        gpu_type = _detect_gpu_type()
        if gpu_type:
            metadata["compute.gpu_type"] = gpu_type
        
        # Detailed GPU info
        gpu_details = _collect_gpu_details()
        if gpu_details:
            metadata["compute.gpus"] = gpu_details

    # Capture current CUDA device mapping
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        metadata["compute.cuda_visible_devices"] = [d.strip() for d in cuda_visible.split(",") if d.strip()]

    # Memory info
    mem_gb = _detect_memory_gb()
    if mem_gb is not None:
        metadata["compute.memory_gb"] = round(mem_gb, 2)
    
    # SLURM job info
    slurm_info = {}
    for key in [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_JOB_NUM_NODES",
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_GPUS_PER_NODE",
        "SLURM_TASKS_PER_NODE",
        "SLURM_PARTITION",
        "SLURM_SUBMIT_DIR",
        "SUBMITIT_JOB_ID",
    ]:
        val = os.environ.get(key)
        if val:
            slurm_info[key.lower()] = val
    
    if slurm_info:
        metadata["slurm"] = slurm_info
    
    # Extract model and runtime config if provided
    if cfg is not None:
        try:
            # Model config
            model_cfg = getattr(cfg, "model", None)
            if model_cfg:
                model_info = {}
                
                # Model source
                model_source = getattr(model_cfg, "model_source", None)
                if model_source:
                    model_info["model_source"] = str(model_source)
                
                # Engine kwargs (vLLM parameters)
                engine_kwargs = getattr(model_cfg, "engine_kwargs", None)
                if engine_kwargs:
                    ek = {}
                    for key in [
                        "max_model_len",
                        "max_num_seqs",
                        "max_num_batched_tokens",
                        "gpu_memory_utilization",
                        "tensor_parallel_size",
                        "enable_chunked_prefill",
                        "enable_prefix_caching",
                        "dtype",
                        "kv_cache_dtype",
                    ]:
                        try:
                            val = getattr(engine_kwargs, key, None)
                            if val is not None:
                                ek[key] = val
                        except Exception:
                            pass
                    if ek:
                        model_info["engine_kwargs"] = ek
                
                # Batch size and concurrency
                batch_size = getattr(model_cfg, "batch_size", None)
                if batch_size is not None:
                    model_info["batch_size"] = int(batch_size)
                
                concurrency = getattr(model_cfg, "concurrency", None)
                if concurrency is not None:
                    model_info["concurrency"] = int(concurrency)
                
                if model_info:
                    metadata["model"] = model_info
        except Exception:
            pass
        
        try:
            # Runtime config
            runtime_cfg = getattr(cfg, "runtime", None)
            if runtime_cfg:
                runtime_info = {}
                for key in [
                    "debug",
                    "sample_n",
                    "job_memory_gb",
                    "rows_per_block",
                    "use_llm_classify",
                    "use_llm_decompose",
                    "prefilter_mode",
                    "keyword_buffering",
                ]:
                    try:
                        val = getattr(runtime_cfg, key, None)
                        if val is not None:
                            runtime_info[key] = val
                    except Exception:
                        pass
                if runtime_info:
                    metadata["runtime"] = runtime_info
        except Exception:
            pass
    
    return metadata


class WandbLogger:
    """Thread-safe centralized W&B logger.
    
    Usage:
        # As context manager (recommended)
        with WandbLogger(cfg, stage="classify", run_id="classify-001") as logger:
            logger.log_metrics({"accuracy": 0.95})
            logger.log_table(df, "results")
        
        # Manual lifecycle
        logger = WandbLogger(cfg, stage="classify", run_id="classify-001")
        logger.start()
        try:
            logger.log_metrics({"accuracy": 0.95})
        finally:
            logger.finish()
    """
    
    _lock = threading.Lock()
    _wandb = None
    _wandb_available = None
    
    def __init__(
        self,
        cfg,
        stage: str,
        run_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize wandb logger.
        
        Args:
            cfg: Hydra config object
            stage: Stage name (e.g., "classify", "topic", "orchestrator")
            run_id: Optional run identifier/suffix
            run_config: Optional run configuration dict to log
        """
        self.cfg = cfg
        self.stage = stage
        self.run_id = run_id
        self.run_config = run_config or {}
        self.wb_config = WandbConfig.from_hydra_config(cfg)
        self._run = None
        
        # Use the globally imported wandb module configured with legacy service
        if WandbLogger._wandb is None:
            with WandbLogger._lock:
                if WandbLogger._wandb is None:
                    WandbLogger._wandb = wandb_module
                    WandbLogger._wandb_available = True
    
    @property
    def enabled(self) -> bool:
        """Check if wandb logging is enabled."""
        return self.wb_config.enabled and WandbLogger._wandb_available
    
    @property
    def wandb(self):
        """Get wandb module (or None if not available)."""
        return WandbLogger._wandb
    
    def _get_run_name(self) -> str:
        """Generate run name."""
        try:
            exp_name = str(getattr(self.cfg.experiment, "name", "UAIR") or "UAIR")
        except Exception:
            exp_name = "UAIR"
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Derive pipeline name from Hydra overrides (e.g., pipeline=topic_with_synthesis)
        pipeline_name = None
        try:
            from hydra.core.hydra_config import HydraConfig  # type: ignore
            hydra_cfg = HydraConfig.get()
            if hydra_cfg and getattr(hydra_cfg, "job", None):
                override_dir = getattr(hydra_cfg.job, "override_dirname", None)
                if override_dir:
                    for part in str(override_dir).split(","):
                        p = part.strip()
                        if p.startswith("pipeline="):
                            pipeline_name = p.split("=", 1)[1]
                            break
        except Exception:
            # Fallback: allow explicit pipeline name via env if provided
            pipeline_name = os.environ.get("UAIR_PIPELINE_NAME") or None

        # Variant for classify stage (e.g., classification_profile: eu_ai_act)
        variant = None
        try:
            if self.stage == "classify":
                var = getattr(getattr(self.cfg, "runtime", object()), "classification_profile", None)
                if var is not None and str(var).strip() != "":
                    variant = str(var).strip()
        except Exception:
            variant = None

        # Build name: [pipeline-]experiment-stage-variant?-timestamp
        parts = []
        if pipeline_name:
            parts.append(pipeline_name)
        parts.append(exp_name)
        parts.append(self.stage)
        if variant:
            parts.append(variant)
        parts.append(timestamp)

        return "-".join(parts)
    
    def _get_mode(self) -> str:
        """Determine wandb mode.
        
        Returns:
            Mode string: "online", "offline", or "disabled"
            
        Priority:
            1. WANDB_MODE environment variable
            2. Default to "online" for real-time syncing
            
        Note: We use online mode by default even for workers because:
        - WANDB__REQUIRE_LEGACY_SERVICE=TRUE prevents daemon/socket issues
        - Real-time syncing provides better visibility
        - Legacy service mode is more stable in distributed contexts (SLURM, Ray)
        """
        # Check for explicit mode override
        mode_env = os.environ.get("WANDB_MODE")
        if mode_env:
            mode_lower = mode_env.lower().strip()
            if mode_lower in ("online", "offline", "disabled"):
                return mode_lower
        
        # Default to online mode for real-time syncing
        return "online"
    
    def _debug_env_snapshot(self, wandb_dir: str) -> Dict[str, Any]:
        """Collect a safe environment snapshot for debugging wandb init issues."""
        snapshot: Dict[str, Any] = {}
        try:
            snapshot.update({
                "user": getpass.getuser(),
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "python": platform.python_version(),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                },
                "cwd": os.getcwd(),
                "wandb": {
                    "version": getattr(wandb_module, "__version__", None),
                    "module_file": getattr(wandb_module, "__file__", None),
                },
                "env": {
                    "WANDB_MODE": os.environ.get("WANDB_MODE"),
                    "WANDB_DISABLE_SERVICE": os.environ.get("WANDB_DISABLE_SERVICE"),
                    "WANDB_SERVICE_PRESENT": bool(os.environ.get("WANDB_SERVICE")),
                    "WANDB_BASE_URL": os.environ.get("WANDB_BASE_URL"),
                    "WANDB_DIR": os.environ.get("WANDB_DIR"),
                    "TMPDIR": os.environ.get("TMPDIR"),
                    "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
                    "SLURM_SUBMIT_DIR": os.environ.get("SLURM_SUBMIT_DIR"),
                    "SUBMITIT_JOB_ID": os.environ.get("SUBMITIT_JOB_ID"),
                    # Do NOT log API keys/secrets
                    "WANDB_API_KEY_SET": bool(os.environ.get("WANDB_API_KEY")),
                },
            })
        except Exception:
            pass
        # Check directories
        def _dir_info(path: Optional[str]) -> Dict[str, Any]:
            info: Dict[str, Any] = {"path": path}
            try:
                if not path:
                    return info
                info["exists"] = os.path.exists(path)
                info["isdir"] = os.path.isdir(path)
                info["writable"] = os.access(path if os.path.isdir(path) else os.path.dirname(path), os.W_OK)
                try:
                    usage = shutil.disk_usage(path if os.path.isdir(path) else os.path.dirname(path))
                    info["disk_free_gb"] = round(usage.free / (1024 ** 3), 2)
                except Exception:
                    pass
            except Exception:
                pass
            return info
        try:
            snapshot["paths"] = {
                "wandb_dir": _dir_info(wandb_dir),
                "tmpdir": _dir_info(os.environ.get("TMPDIR")),
            }
        except Exception:
            pass
        return snapshot
    
    def start(self) -> None:
        """Start wandb run."""
        if not self.enabled:
            return

        if self._run is not None:
            print(f"[wandb] Warning: Run already started for {self.stage}", file=sys.stderr)
            return
        
        # Ensure we do not inherit/target a parent service socket across nodes
        try:
            for k in ("WANDB_SERVICE", "WANDB__SERVICE", "WANDB_SERVICE_SOCKET", "WANDB_SERVICE_TRANSPORT"):
                if k in os.environ:
                    os.environ.pop(k, None)
            # Reinforce in-process mode
            os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
        except Exception:
            pass

        mode = self._get_mode()
        run_name = self._get_run_name()
        
        # Determine wandb directory (must be writable in SLURM environments)
        wandb_dir = os.environ.get("WANDB_DIR")
        if not wandb_dir:
            # Default to a writable location
            # Use SLURM_SUBMIT_DIR if available (job submission directory), otherwise CWD
            wandb_dir = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())
        
        # Emit detailed debug snapshot once per run start
        try:
            dbg = self._debug_env_snapshot(wandb_dir)
            print(f"[wandb] Debug init context: {json.dumps(dbg, ensure_ascii=False)}", flush=True)
        except Exception:
            pass
        print(f"[wandb] Starting run: {run_name} (mode={mode}, dir={wandb_dir})", flush=True)
        
        try:
            # Initialize run with legacy service (configured at module import)
            self._run = self.wandb.init(
                project=self.wb_config.project,
                entity=self.wb_config.entity,
                group=self.wb_config.group,
                job_type=self.stage,
                name=run_name,
                config=self.run_config,
                mode=mode,
                dir=wandb_dir,
                tags=self.wb_config.tags,
            )
            
            # Collect and log compute metadata
            try:
                compute_metadata = collect_compute_metadata(self.cfg)
                if compute_metadata:
                    self.set_config(compute_metadata, allow_val_change=True)
                    print(f"[wandb] ✓ Logged compute metadata: {compute_metadata.get('compute.cpu_count', 'N/A')} CPUs, {compute_metadata.get('compute.gpu_count', 0)} GPUs", flush=True)
            except Exception as e:
                print(f"[wandb] Warning: Failed to collect compute metadata: {e}", file=sys.stderr, flush=True)
            
            if mode == "offline":
                print(f"[wandb] ✓ Run started: {run_name} (OFFLINE - will sync on finish)", flush=True)
            elif mode == "online":
                print(f"[wandb] ✓ Run started: {run_name} (ONLINE - real-time syncing)", flush=True)
            else:
                print(f"[wandb] ✓ Run started: {run_name} (mode={mode})", flush=True)
            
        except Exception as e:
            # Print rich error context including traceback for easier diagnosis
            tb = traceback.format_exc()
            print(f"[wandb] ✗ Failed to start run: {e}", file=sys.stderr, flush=True)
            try:
                print(f"[wandb] Traceback:\n{tb}", file=sys.stderr, flush=True)
            except Exception:
                pass
            print(f"[wandb] Logging will be disabled for {self.stage}", file=sys.stderr, flush=True)
            self._run = None
    
    def finish(self) -> None:
        """Finish wandb run and sync if offline."""
        if not self.enabled or self._run is None:
            return
        
        try:
            run_name = getattr(self._run, "name", "unknown")
            run_mode = getattr(getattr(self._run, "settings", None), "mode", "unknown")
            
            print(f"[wandb] Finishing run: {run_name} (mode={run_mode})", flush=True)
            
            self.wandb.finish()
            
            if run_mode == "offline":
                print(f"[wandb] ✓ Offline run '{run_name}' synced to cloud", flush=True)
            else:
                print(f"[wandb] ✓ Online run '{run_name}' completed", flush=True)
            
            self._run = None
            
        except Exception as e:
            print(f"[wandb] ✗ Failed to finish run: {e}", file=sys.stderr, flush=True)
            self._run = None
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
        """Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number
            commit: Whether to increment step counter
        """
        if not self.enabled or self._run is None:
            return
        
        try:
            if metrics:
                self.wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            print(f"[wandb] Warning: Failed to log metrics: {e}", file=sys.stderr)
    
    def log_table(
        self,
        df,
        key: str,
        prefer_cols: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
        panel_group: Optional[str] = None,
    ) -> None:
        """Log pandas DataFrame as wandb table with random sampling when needed.
        
        Args:
            df: Pandas DataFrame
            key: Table name/key (e.g., "classify/results")
            prefer_cols: Optional list of preferred columns to include
            max_rows: Max rows to sample (default from config)
            panel_group: Optional panel group name (e.g., "inspect_results")
                        Will prefix the key as "panel_group/key"
        """
        if not self.enabled or self._run is None or df is None:
            return
        
        try:
            import pandas as pd
            
            # Apply panel group prefix if specified
            table_key = f"{panel_group}/{key}" if panel_group else key
            
            # Deduplicate columns by name (keep first occurrence)
            try:
                df_local = df.loc[:, ~df.columns.duplicated()]
            except Exception:
                df_local = df

            # Universally drop internal LLM/mechanics columns before any selection (applies to all stages)
            def _drop_internal_llm_columns(df_in):
                try:
                    internal_cols = {
                        # vLLM/Ray internals
                        "generated_text",
                        "llm_output",
                        "messages",
                        "params",
                        "prompt",
                        "sampling_params",
                        "json",
                        "guided_decoding",
                        "response_format",
                        "structured_output",
                        # usage/token counts often nested
                        "usage",
                        "token_counts",
                        # Nested dicts/arrays that cause wandb schema issues
                        "reasoning_data",  # Contains nested 'norms' array with varying lengths
                        "ci_flows_raw",  # Contains nested CI flow objects with varying lengths
                    }
                    # Pattern-based exclusions
                    pattern_prefixes = [
                        "eu_ai_raw_json",
                    ]
                    pattern_names = {
                        "llm_json",
                    }
                    cols_present = [c for c in internal_cols if c in df_in.columns]
                    # Add pattern matches
                    for c in list(df_in.columns):
                        try:
                            if c in pattern_names:
                                cols_present.append(c)
                                continue
                            for pref in pattern_prefixes:
                                if c.startswith(pref):
                                    cols_present.append(c)
                                    break
                        except Exception:
                            continue
                    if cols_present:
                        return df_in.drop(columns=cols_present)
                    return df_in
                except Exception:
                    return df_in
            df_local = _drop_internal_llm_columns(df_local)

            # Select columns
            # For decompose stage, log ALL available columns in inspect_results table,
            # but drop heavy ndarray/list-like columns that cause schema issues.
            def _filter_heavy_columns(df_local, candidate_cols):
                keep = []
                for c in candidate_cols:
                    try:
                        if c in {"generated_tokens", "prompt_token_ids", "embeddings"}:
                            continue
                        # Find a representative non-null value
                        sample = None
                        for v in df_local[c].values:
                            if v is not None:
                                sample = v
                                break
                        if sample is not None:
                            name = getattr(type(sample), "__name__", "") or ""
                            # Exclude numpy arrays (ndarray), but DO NOT exclude numpy scalar types
                            if name == "ndarray":
                                continue
                        keep.append(c)
                    except Exception:
                        # Best-effort: if we can't inspect, keep the column
                        keep.append(c)
                return keep

            if (
                (
                    self.stage in ("decompose", "decompose_nbl", "verify_nbl")
                    and (
                        panel_group == "inspect_results"
                        or key.startswith("decompose/")
                        or key.startswith("decompose_nbl/")
                        or key.startswith("verify_nbl/")
                    )
                )
            or (
                self.stage in {
                    "classify",
                    "classify_eu_act",
                    "classify_risk_and_benefits",
                    "reasoning",
                    "extraction",
                    "norm_reasoning",
                    "norm_extraction",
                    "ci_reasoning",
                    "ci_extraction",
                }
                and (
                    panel_group == "inspect_results" 
                    or key.startswith("classify/")
                    or key.startswith("reasoning/")
                    or key.startswith("extraction/")
                    or key.startswith("norm_reasoning/")
                    or key.startswith("norm_extraction/")
                    or key.startswith("ci_reasoning/")
                    or key.startswith("ci_extraction/")
                )
            )
            ):
                cols = _filter_heavy_columns(df_local, list(df_local.columns))
            else:
                cols = [c for c in (prefer_cols or []) if c in df_local.columns]
                if not cols:
                    cols = list(df_local.columns)[:12]
                cols = _filter_heavy_columns(df_local, cols)

            try:
                print(
                    f"[wandb_logger] stage={self.stage} key={key} logging {len(cols)} columns: {cols}",
                    flush=True,
                )
            except Exception:
                pass

            # Final safeguard: ensure internal columns are not selected even if present in prefer_cols
            universal_exclude = {
                "generated_text",
                "llm_output",
                "messages",
                "params",
                "prompt",
                "sampling_params",
                "json",
                "guided_decoding",
                "response_format",
                "structured_output",
                "usage",
                "token_counts",
            }
            def _is_universal_excluded(name: str) -> bool:
                if name in universal_exclude:
                    return True
                if name == "llm_json":
                    return True
                if name.startswith("eu_ai_raw_json"):
                    return True
                return False
            cols = [c for c in cols if not _is_universal_excluded(c)]
            
            # Sample if needed (always use random sampling, not head())
            max_rows = max_rows or self.wb_config.table_sample_rows
            total_rows = len(df_local)
            
            if total_rows > max_rows:
                df_sample = df_local.sample(
                    n=max_rows,
                    random_state=self.wb_config.table_sample_seed
                ).reset_index(drop=True)
                sampled = True
            else:
                df_sample = df_local.reset_index(drop=True)
                sampled = False
            
            # Ensure no duplicate columns in the sampled DataFrame
            try:
                df_sample = df_sample.loc[:, ~df_sample.columns.duplicated()]
            except Exception:
                pass
            
            # Create table
            table = self.wandb.Table(dataframe=df_sample[cols])
            
            # Log table with metadata
            # Table goes to panel_group (e.g., "inspect_results")
            # But row counts go to stage-specific panel (use original key, not table_key)
            log_data = {
                table_key: table,
                f"{key}/rows": len(df_sample),
                f"{key}/total_rows": total_rows,
            }
            
            # Add sampling metadata if applicable
            if sampled:
                log_data[f"{key}/sampled"] = True
                log_data[f"{key}/sample_seed"] = self.wb_config.table_sample_seed
            
            self.wandb.log(log_data)
            
            # Print sampling info
            if sampled:
                print(f"[wandb] ✓ Logged table '{table_key}': {len(df_sample):,} rows (randomly sampled from {total_rows:,})", flush=True)
            else:
                print(f"[wandb] ✓ Logged table '{table_key}': {total_rows:,} rows", flush=True)
            
        except Exception as e:
            print(f"[wandb] Warning: Failed to log table '{key}': {e}", file=sys.stderr)
    
    def log_artifact(self, artifact_path: str, name: str, type: str = "dataset") -> None:
        """Log artifact to wandb.
        
        Args:
            artifact_path: Path to artifact file/directory
            name: Artifact name
            type: Artifact type (e.g., "dataset", "model")
        """
        if not self.enabled or self._run is None:
            return
        
        try:
            artifact = self.wandb.Artifact(name=name, type=type)
            artifact.add_file(artifact_path)
            self.wandb.log_artifact(artifact)
        except Exception as e:
            print(f"[wandb] Warning: Failed to log artifact '{name}': {e}", file=sys.stderr)
    
    def log_plot(self, key: str, figure) -> None:
        """Log matplotlib/plotly figure or wandb plot object.
        
        Args:
            key: Plot name/key
            figure: Matplotlib figure, Plotly figure, or wandb plot object (Image, Plotly, Html, etc.)
        """
        if not self.enabled or self._run is None:
            return
        
        try:
            # Check if it's already a wandb data type (Plotly, Image, Html, etc.)
            if hasattr(figure, '__class__') and hasattr(figure.__class__, '__module__'):
                module = figure.__class__.__module__
                if module and 'wandb' in module:
                    # Already a wandb data type, log directly
                    self.wandb.log({key: figure})
                    return
            
            # Raw plotly figure (has to_html method)
            if hasattr(figure, 'to_html'):
                self.wandb.log({key: figure})
            # Matplotlib figure (has savefig method)
            elif hasattr(figure, 'savefig'):
                self.wandb.log({key: self.wandb.Image(figure)})
            else:
                # Log directly and let wandb handle it
                self.wandb.log({key: figure})
        except Exception as e:
            print(f"[wandb] Warning: Failed to log plot '{key}': {e}", file=sys.stderr)
    
    def set_summary(self, key: str, value: Any) -> None:
        """Set a run-level summary field (useful for categorical/string values)."""
        if not self.enabled or self._run is None:
            return
        try:
            self._run.summary[key] = value
        except Exception as e:
            print(f"[wandb] Warning: Failed to set summary '{key}': {e}", file=sys.stderr)
    
    def set_config(self, data: Dict[str, Any], allow_val_change: bool = True) -> None:
        """Update run config for stable, non-time-series metadata (e.g., backend name)."""
        if not self.enabled or self._run is None or not data:
            return
        try:
            self._run.config.update(dict(data), allow_val_change=allow_val_change)
        except Exception as e:
            print(f"[wandb] Warning: Failed to update config: {e}", file=sys.stderr)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
        return False

