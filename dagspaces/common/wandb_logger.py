"""Centralized W&B logging — canonical common implementation.

This module provides a unified interface for W&B logging across all pipeline
dagspaces, handling SLURM distributed execution and run lifecycle management.

Key features:
- Single source of truth for W&B configuration
- Proper context management for runs
- Online mode by default for real-time syncing (configurable via WANDB_MODE)
- Service daemon completely disabled using wandb.Settings() API
- Thread-safe logging
- Graceful degradation when W&B is unavailable

Configuration (wandb 0.22.0+ best practices):
- Service daemon disabled via WANDB_DISABLE_SERVICE environment variable
- In-process mode is used instead of service daemon for SLURM compatibility
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

import getpass
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Set


# ---------------------------------------------------------------------------
# Tmpdir helpers
# ---------------------------------------------------------------------------

def ensure_local_tmpdir(dagspace_name: str) -> None:
    """Configure TMPDIR to a local path suitable for Ray/vLLM/W&B sockets.

    Uses /scratch if available (e.g. klara), otherwise /tmp.  Resets the
    variable if it currently points at a /share network path, which causes
    socket-file issues when jobs span multiple SLURM nodes.

    Args:
        dagspace_name: Sub-directory name placed under the local base
                       (e.g. "historical_norms", "contextual_integrity_eval").
    """
    current_tmp = os.environ.get("TMPDIR", "")
    if not current_tmp or current_tmp.startswith("/share"):
        local_base = "/scratch" if os.path.exists("/scratch") else "/tmp"
        new_tmp = os.path.join(local_base, getpass.getuser(), dagspace_name)
        try:
            os.makedirs(new_tmp, exist_ok=True)
            os.environ["TMPDIR"] = new_tmp
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Wandb settings bootstrap
# ---------------------------------------------------------------------------

def _apply_wandb_settings_defaults() -> None:
    """Apply defaults from repo-level wandb/settings before importing wandb.

    Only sets env vars if they are not already present, so explicit
    overrides in the calling environment are always honoured.
    """
    try:
        settings_path = os.environ.get("WANDB_SETTINGS_PATH")
        if not settings_path:
            # Default to repo root: <repo>/wandb/settings
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            settings_path = os.path.join(base_dir, "wandb", "settings")
        if not (isinstance(settings_path, str) and os.path.exists(settings_path)):
            return
        import configparser  # local import to avoid global dependency

        cp = configparser.ConfigParser()
        try:
            cp.read(settings_path)
        except Exception:
            return
        sect = (
            "default"
            if cp.has_section("default")
            else (cp.sections()[0] if cp.sections() else None)
        )
        if not sect:
            return
        sec = cp[sect]
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
import wandb as wandb_module  # noqa: E402



# ---------------------------------------------------------------------------
# WandbConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class WandbConfig:
    """W&B configuration extracted from Hydra config.

    Fields:
        enabled: Whether W&B logging is active.
        project: W&B project name.  Each dagspace passes its own default.
        entity: W&B entity (team/user).  Resolved from config then env.
        group: Run group (e.g. SLURM job ID).
        tags: List of tags to attach to runs.
        table_sample_rows: Max rows when logging DataFrame tables.
        table_sample_seed: Random seed for table row sampling.
        default_experiment_name: Fallback for cfg.experiment.name in run names.
        env_var_prefix: Prefix for dagspace-specific sanitizer env vars
                        (e.g. "UAIR", "HISTORICAL_NORMS").  Empty string means
                        no sanitizer vars are read.
        full_column_stages: Stage names for which all DataFrame columns are
                            logged (not just the first 12).
        full_column_key_prefixes: Key prefixes that also trigger full-column
                                  logging regardless of stage name.
        extra_internal_columns: Additional column names to drop before logging
                                tables (beyond the universal set).
        extra_pattern_prefixes: Column name prefixes triggering exclusion.
        extra_pattern_names: Exact column names (beyond universal set) to
                             exclude from table logging.
        extra_runtime_keys: Extra keys to extract from cfg.runtime when
                            building compute metadata.
        classify_variant_field: If set, extract this field from cfg.runtime
                                and append it to run names when stage=="classify".
    """

    enabled: bool = False
    project: str = "wandb-project"
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    table_sample_rows: int = 1000
    table_sample_seed: int = 777

    # Dagspace-specific knobs
    default_experiment_name: str = "experiment"
    env_var_prefix: str = ""
    full_column_stages: FrozenSet[str] = field(
        default_factory=lambda: frozenset()
    )
    full_column_key_prefixes: FrozenSet[str] = field(
        default_factory=lambda: frozenset()
    )
    extra_internal_columns: FrozenSet[str] = field(
        default_factory=lambda: frozenset()
    )
    extra_pattern_prefixes: List[str] = field(default_factory=list)
    extra_pattern_names: FrozenSet[str] = field(
        default_factory=lambda: frozenset()
    )
    extra_runtime_keys: List[str] = field(default_factory=list)
    classify_variant_field: Optional[str] = None

    @classmethod
    def from_hydra_config(
        cls,
        cfg,
        *,
        default_project: str = "wandb-project",
        default_experiment_name: str = "experiment",
        env_var_prefix: str = "",
        full_column_stages: Optional[FrozenSet[str]] = None,
        full_column_key_prefixes: Optional[FrozenSet[str]] = None,
        extra_internal_columns: Optional[FrozenSet[str]] = None,
        extra_pattern_prefixes: Optional[List[str]] = None,
        extra_pattern_names: Optional[FrozenSet[str]] = None,
        extra_runtime_keys: Optional[List[str]] = None,
        classify_variant_field: Optional[str] = None,
    ) -> "WandbConfig":
        """Extract W&B config from Hydra config.

        All dagspace-specific keyword arguments have sensible defaults so
        callers can override only what differs.

        Args:
            cfg: Hydra config object.
            default_project: Fallback W&B project name.
            default_experiment_name: Fallback for cfg.experiment.name.
            env_var_prefix: Prefix for GPU sanitizer env vars (e.g. "UAIR").
            full_column_stages: Stage names that log all DataFrame columns.
            full_column_key_prefixes: Key prefixes that also trigger full-column
                logging.
            extra_internal_columns: Additional column names to always drop.
            extra_pattern_prefixes: Column-name prefixes to drop.
            extra_pattern_names: Exact column names to drop (beyond universal).
            extra_runtime_keys: Extra runtime config keys to log as metadata.
            classify_variant_field: cfg.runtime field to append to run names
                when stage == "classify".
        """
        try:
            wandb_cfg = getattr(cfg, "wandb", None)
            env_entity = os.environ.get("WANDB_ENTITY")
            env_project = os.environ.get("WANDB_PROJECT")
            if wandb_cfg is None:
                return cls(
                    enabled=False,
                    project=(env_project or default_project),
                    entity=(
                        env_entity if env_entity and env_entity.strip() else None
                    ),
                    group=_get_group_from_config(cfg),
                    tags=[],
                    table_sample_rows=1000,
                    table_sample_seed=777,
                    default_experiment_name=default_experiment_name,
                    env_var_prefix=env_var_prefix,
                    full_column_stages=full_column_stages or frozenset(),
                    full_column_key_prefixes=full_column_key_prefixes
                    or frozenset(),
                    extra_internal_columns=extra_internal_columns or frozenset(),
                    extra_pattern_prefixes=extra_pattern_prefixes or [],
                    extra_pattern_names=extra_pattern_names or frozenset(),
                    extra_runtime_keys=extra_runtime_keys or [],
                    classify_variant_field=classify_variant_field,
                )

            # Resolve project with fallback to environment, then default
            proj_attr = getattr(wandb_cfg, "project", None)
            if proj_attr is None or str(proj_attr).strip() == "":
                project = env_project or default_project
            else:
                project = str(proj_attr or default_project)

            # Resolve entity with fallback to environment
            entity_cfg = _get_optional_str(wandb_cfg, "entity")
            entity = entity_cfg or (
                env_entity if env_entity and env_entity.strip() else None
            )
            return cls(
                enabled=bool(getattr(wandb_cfg, "enabled", False)),
                project=project,
                entity=entity,
                group=_get_group_from_config(cfg),
                tags=_get_list(wandb_cfg, "tags"),
                table_sample_rows=int(
                    getattr(wandb_cfg, "table_sample_rows", 1000)
                ),
                table_sample_seed=int(
                    getattr(wandb_cfg, "table_sample_seed", 777)
                ),
                default_experiment_name=default_experiment_name,
                env_var_prefix=env_var_prefix,
                full_column_stages=full_column_stages or frozenset(),
                full_column_key_prefixes=full_column_key_prefixes or frozenset(),
                extra_internal_columns=extra_internal_columns or frozenset(),
                extra_pattern_prefixes=extra_pattern_prefixes or [],
                extra_pattern_names=extra_pattern_names or frozenset(),
                extra_runtime_keys=extra_runtime_keys or [],
                classify_variant_field=classify_variant_field,
            )
        except Exception as e:
            print(
                f"[wandb] Warning: Failed to parse config: {e}",
                file=sys.stderr,
            )
            return cls(
                default_experiment_name=default_experiment_name,
                env_var_prefix=env_var_prefix,
            )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_optional_str(obj, attr: str) -> Optional[str]:
    """Return attribute as non-empty string or None."""
    try:
        val = getattr(obj, attr, None)
        if val is not None and str(val).strip():
            return str(val)
    except Exception:
        pass
    return None


def _get_list(obj, attr: str) -> List[str]:
    """Return list attribute safely."""
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
    try:
        grp = getattr(cfg.wandb, "group", None)
        if grp and str(grp).strip():
            return str(grp)
    except Exception:
        pass

    env_group = os.environ.get("WANDB_GROUP")
    if env_group and env_group.strip():
        return env_group

    submitit_job_id = os.environ.get("SUBMITIT_JOB_ID")
    if submitit_job_id and submitit_job_id.strip():
        return f"slurm-{submitit_job_id}"

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id and slurm_job_id.strip():
        return f"slurm-{slurm_job_id}"

    return None


# ---------------------------------------------------------------------------
# GPU / compute detection (nvidia-smi only — never imports torch)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _query_nvidia_smi_inventory() -> List[Dict[str, Any]]:
    """Query GPU inventory via nvidia-smi without touching torch.cuda.

    Best practice for pre-fork multiprocessing is to avoid initialising CUDA
    in the parent process.  Scheduler env vars and nvidia-smi provide the
    same inventory data without poisoning a later fork-based worker launch.

    Tries two nvidia-smi field sets: one with compute_cap (not available on
    all driver versions) and one without, falling back gracefully.
    """
    field_sets = [
        ["index", "uuid", "name", "memory.total", "compute_cap"],
        ["index", "uuid", "name", "memory.total"],
    ]
    try:
        proc = None
        fields: List[str] = []
        for candidate_fields in field_sets:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={','.join(candidate_fields)}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.returncode == 0:
                fields = candidate_fields
                break
        if proc is None or proc.returncode != 0:
            return []

        gpus: List[Dict[str, Any]] = []
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(fields):
                continue
            gpu_info: Dict[str, Any] = {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
            }
            try:
                gpu_info["total_memory_gb"] = round(float(parts[3]) / 1024.0, 2)
            except Exception:
                pass
            if len(parts) > 4 and parts[4]:
                gpu_info["compute_capability"] = parts[4]
                try:
                    major_s, minor_s = parts[4].split(".", 1)
                    gpu_info["major"] = int(major_s)
                    gpu_info["minor"] = int(minor_s)
                except Exception:
                    pass
            gpus.append(gpu_info)
        return gpus
    except Exception:
        return []


def _collect_gpu_details() -> List[Dict[str, Any]]:
    """Return a copy of the cached GPU inventory."""
    return _query_nvidia_smi_inventory().copy()


def _detect_num_gpus() -> int:
    """Detect the number of GPUs allocated to this job.

    Priority order:
    1. CUDA_VISIBLE_DEVICES environment variable (set by launcher)
    2. SLURM_GPUS_PER_NODE or SLURM_GPUS_ON_NODE
    3. nvidia-smi inventory (does not initialize CUDA in-process)
    4. Return 0 if no GPUs detected
    """
    # Priority 1: CUDA_VISIBLE_DEVICES (most reliable for actual allocation)
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible.strip():
            # Parse comma-separated GPU indices (e.g., "0,1,2,3" -> 4 GPUs)
            gpu_indices = [x.strip() for x in cuda_visible.split(",") if x.strip()]
            if gpu_indices:
                return len(gpu_indices)
    except Exception:
        pass

    # Priority 2: SLURM environment variables
    try:
        slurm_gpus = os.environ.get("SLURM_GPUS_PER_NODE") or os.environ.get(
            "SLURM_GPUS_ON_NODE"
        )
        if slurm_gpus:
            # Can be a number like "4" or format like "gpu:4"
            try:
                if ":" in slurm_gpus:
                    return int(slurm_gpus.split(":")[-1])
                return int(slurm_gpus)
            except Exception:
                pass
    except Exception:
        pass

    # Priority 3: GPU inventory from nvidia-smi
    gpu_details = _collect_gpu_details()
    if gpu_details:
        return len(gpu_details)

    return 0


def _detect_gpu_type() -> Optional[str]:
    """Detect the GPU type/model name via nvidia-smi (no torch import).

    Returns a normalised GPU type string (e.g. 'NVIDIA RTX A6000').
    """
    gpu_details = _collect_gpu_details()
    if gpu_details:
        gpu_name = gpu_details[0].get("name")
        if gpu_name:
            return str(gpu_name)
    return None


# ---------------------------------------------------------------------------
# CPU / memory detection
# ---------------------------------------------------------------------------

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
    v2 = "/sys/fs/cgroup/memory.max"
    lim = _read_int_file(v2)
    if lim > 0:
        try:
            if lim > (1 << 56):
                return -1
        except Exception:
            pass
        return lim

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

    Prefers SLURM_MEM_PER_NODE; otherwise uses SLURM_MEM_PER_CPU *
    SLURM_CPUS_ON_NODE.  Values are MB according to SLURM docs.
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
    cg = _detect_cgroup_mem_limit_bytes()
    if cg > 0:
        return cg / (1024 ** 3)

    sj = _detect_slurm_job_mem_bytes()
    if sj > 0:
        return sj / (1024 ** 3)

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


# ---------------------------------------------------------------------------
# Compute metadata collection
# ---------------------------------------------------------------------------

#: Runtime config keys that all dagspaces always extract.
_BASE_RUNTIME_KEYS: List[str] = [
    "debug",
    "sample_n",
    "job_memory_gb",
    "rows_per_block",
]

#: GPU sanitizer env-var suffixes (appended to the dagspace prefix).
_SANITIZER_SUFFIXES = (
    "SANITIZED_DROPPED_GPUS",
    "GPU_SANITIZE_ORIGINAL",
    "GPU_SANITIZE_REASON",
    "GPU_SANITIZE_TS",
    "TENSOR_PARALLEL_SIZE",
)


def collect_compute_metadata(
    cfg=None,
    *,
    env_var_prefix: str = "",
    extra_runtime_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Collect comprehensive compute metadata for wandb logging.

    Captures system configuration including CPU count and architecture, GPU
    count/type/memory, RAM allocation, SLURM job parameters, Python
    environment, and model configuration.

    Args:
        cfg: Optional Hydra config to extract model/runtime parameters.
        env_var_prefix: Dagspace prefix for GPU sanitizer env vars (e.g.
            "UAIR" reads UAIR_GPU_SANITIZE_ORIGINAL etc.).  Empty string
            skips sanitizer var collection.
        extra_runtime_keys: Additional runtime config keys to log beyond the
            base set.

    Returns:
        Dictionary with compute metadata suitable for wandb.config.
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

        gpu_details = _collect_gpu_details()
        if gpu_details:
            metadata["compute.gpus"] = gpu_details

    # CUDA device mapping
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        metadata["compute.cuda_visible_devices"] = [
            d.strip() for d in cuda_visible.split(",") if d.strip()
        ]

    # GPU sanitizer metadata (dagspace-specific env var prefix)
    if env_var_prefix:
        pfx = env_var_prefix.rstrip("_") + "_"
        dropped = os.environ.get(f"{pfx}SANITIZED_DROPPED_GPUS")
        original = os.environ.get(f"{pfx}GPU_SANITIZE_ORIGINAL")
        reason = os.environ.get(f"{pfx}GPU_SANITIZE_REASON")
        ts = os.environ.get(f"{pfx}GPU_SANITIZE_TS")
        tp_raw = os.environ.get(f"{pfx}TENSOR_PARALLEL_SIZE")

        sanitize_meta: Dict[str, Any] = {}
        if original:
            sanitize_meta["original"] = [
                d.strip() for d in original.split(",") if d.strip()
            ]
        if dropped:
            sanitize_meta["dropped"] = [
                d.strip() for d in dropped.split(",") if d.strip()
            ]
        if reason:
            sanitize_meta["reason"] = reason
        if ts:
            try:
                sanitize_meta["timestamp"] = int(ts)
            except Exception:
                sanitize_meta["timestamp"] = ts
        if tp_raw:
            try:
                sanitize_meta["tensor_parallel_size"] = int(tp_raw)
            except Exception:
                sanitize_meta["tensor_parallel_size"] = tp_raw
        if sanitize_meta:
            metadata["compute.gpu_sanitize"] = sanitize_meta

    # Memory info
    mem_gb = _detect_memory_gb()
    if mem_gb is not None:
        metadata["compute.memory_gb"] = round(mem_gb, 2)

    # SLURM job info
    slurm_info: Dict[str, Any] = {}
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

    # Model and runtime config
    if cfg is not None:
        try:
            model_cfg = getattr(cfg, "model", None)
            if model_cfg:
                model_info: Dict[str, Any] = {}

                model_source = getattr(model_cfg, "model_source", None)
                if model_source:
                    model_info["model_source"] = str(model_source)

                engine_kwargs = getattr(model_cfg, "engine_kwargs", None)
                if engine_kwargs:
                    ek: Dict[str, Any] = {}
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
            runtime_cfg = getattr(cfg, "runtime", None)
            if runtime_cfg:
                runtime_info: Dict[str, Any] = {}
                all_runtime_keys = _BASE_RUNTIME_KEYS + (
                    extra_runtime_keys or []
                )
                for key in all_runtime_keys:
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


# ---------------------------------------------------------------------------
# WandbLogger
# ---------------------------------------------------------------------------

#: Columns that are always dropped before table logging across all dagspaces.
_UNIVERSAL_EXCLUDE: FrozenSet[str] = frozenset(
    {
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
)


class WandbLogger:
    """Thread-safe centralised W&B logger.

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
        """Initialise wandb logger.

        Args:
            cfg: Hydra config object.
            stage: Stage name (e.g. "classify", "topic", "orchestrator").
            run_id: Optional run identifier/suffix.
            run_config: Optional run configuration dict to log.
        """
        self.cfg = cfg
        self.stage = stage
        self.run_id = run_id
        self.run_config = run_config or {}
        self.wb_config = WandbConfig.from_hydra_config(cfg)
        self._run = None

        if WandbLogger._wandb is None:
            with WandbLogger._lock:
                if WandbLogger._wandb is None:
                    WandbLogger._wandb = wandb_module
                    WandbLogger._wandb_available = True

    @classmethod
    def with_config(
        cls,
        cfg,
        stage: str,
        wb_config: WandbConfig,
        run_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> "WandbLogger":
        """Create a WandbLogger with a pre-built WandbConfig.

        Preferred factory when callers supply a fully-parameterised
        WandbConfig (e.g. from their dagspace shim).

        Args:
            cfg: Hydra config object (used for experiment name, runtime, etc.)
            stage: Stage name.
            wb_config: Pre-built WandbConfig instance.
            run_id: Optional run identifier/suffix.
            run_config: Optional run configuration dict to log.
        """
        instance = cls.__new__(cls)
        instance.cfg = cfg
        instance.stage = stage
        instance.run_id = run_id
        instance.run_config = run_config or {}
        instance.wb_config = wb_config
        instance._run = None

        if WandbLogger._wandb is None:
            with WandbLogger._lock:
                if WandbLogger._wandb is None:
                    WandbLogger._wandb = wandb_module
                    WandbLogger._wandb_available = True

        return instance

    @property
    def enabled(self) -> bool:
        """Check if wandb logging is enabled."""
        return self.wb_config.enabled and WandbLogger._wandb_available

    @property
    def wandb(self):
        """Return the wandb module (or None if not available)."""
        return WandbLogger._wandb

    def _get_run_name(self) -> str:
        """Generate a human-readable run name."""
        exp_name_default = self.wb_config.default_experiment_name
        try:
            exp_name = str(
                getattr(self.cfg.experiment, "name", exp_name_default)
                or exp_name_default
            )
        except Exception:
            exp_name = exp_name_default

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Derive pipeline name from Hydra overrides (e.g. pipeline=topic_with_synthesis)
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
            pipeline_name = os.environ.get("WANDB_PIPELINE_NAME") or None

        # Optional stage variant (e.g. classification_profile for classify stage)
        variant = None
        variant_field = self.wb_config.classify_variant_field
        if variant_field:
            try:
                if self.stage == "classify":
                    var = getattr(
                        getattr(self.cfg, "runtime", object()),
                        variant_field,
                        None,
                    )
                    if var is not None and str(var).strip() != "":
                        variant = str(var).strip()
            except Exception:
                variant = None

        # Build name: [pipeline-]experiment-stage[-variant]-timestamp
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
            Mode string: "online", "offline", or "disabled".

        Priority:
            1. WANDB_MODE environment variable
            2. Default to "online" for real-time syncing
        """
        mode_env = os.environ.get("WANDB_MODE")
        if mode_env:
            mode_lower = mode_env.lower().strip()
            if mode_lower in ("online", "offline", "disabled"):
                return mode_lower
        return "online"

    def _debug_env_snapshot(self, wandb_dir: str) -> Dict[str, Any]:
        """Collect a safe environment snapshot for debugging wandb init issues."""
        snapshot: Dict[str, Any] = {}
        try:
            snapshot.update(
                {
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
                        "WANDB_DISABLE_SERVICE": os.environ.get(
                            "WANDB_DISABLE_SERVICE"
                        ),
                        "WANDB_SERVICE_PRESENT": bool(
                            os.environ.get("WANDB_SERVICE")
                        ),
                        "WANDB_BASE_URL": os.environ.get("WANDB_BASE_URL"),
                        "WANDB_DIR": os.environ.get("WANDB_DIR"),
                        "TMPDIR": os.environ.get("TMPDIR"),
                        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
                        "SLURM_SUBMIT_DIR": os.environ.get("SLURM_SUBMIT_DIR"),
                        "SUBMITIT_JOB_ID": os.environ.get("SUBMITIT_JOB_ID"),
                        # Do NOT log API keys/secrets
                        "WANDB_API_KEY_SET": bool(
                            os.environ.get("WANDB_API_KEY")
                        ),
                    },
                }
            )
        except Exception:
            pass

        def _dir_info(path: Optional[str]) -> Dict[str, Any]:
            info: Dict[str, Any] = {"path": path}
            try:
                if not path:
                    return info
                info["exists"] = os.path.exists(path)
                info["isdir"] = os.path.isdir(path)
                info["writable"] = os.access(
                    path if os.path.isdir(path) else os.path.dirname(path),
                    os.W_OK,
                )
                try:
                    usage = shutil.disk_usage(
                        path if os.path.isdir(path) else os.path.dirname(path)
                    )
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
            print(
                f"[wandb] Warning: Run already started for {self.stage}",
                file=sys.stderr,
            )
            return

        # Ensure we do not inherit/target a parent service socket across nodes
        try:
            for k in (
                "WANDB_SERVICE",
                "WANDB__SERVICE",
                "WANDB_SERVICE_SOCKET",
                "WANDB_SERVICE_TRANSPORT",
            ):
                if k in os.environ:
                    os.environ.pop(k, None)
            os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
        except Exception:
            pass

        mode = self._get_mode()
        run_name = self._get_run_name()

        # Determine wandb directory (must be writable in SLURM environments)
        wandb_dir = os.environ.get("WANDB_DIR")
        if not wandb_dir:
            wandb_dir = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())

        try:
            dbg = self._debug_env_snapshot(wandb_dir)
            print(
                f"[wandb] Debug init context: {json.dumps(dbg, ensure_ascii=False)}",
                flush=True,
            )
        except Exception:
            pass
        print(
            f"[wandb] Starting run: {run_name} (mode={mode}, dir={wandb_dir})",
            flush=True,
        )

        try:
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

            try:
                compute_metadata = collect_compute_metadata(
                    self.cfg,
                    env_var_prefix=self.wb_config.env_var_prefix,
                    extra_runtime_keys=self.wb_config.extra_runtime_keys,
                )
                if compute_metadata:
                    self.set_config(compute_metadata, allow_val_change=True)
                    print(
                        f"[wandb] Logged compute metadata: "
                        f"{compute_metadata.get('compute.cpu_count', 'N/A')} CPUs, "
                        f"{compute_metadata.get('compute.gpu_count', 0)} GPUs",
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"[wandb] Warning: Failed to collect compute metadata: {e}",
                    file=sys.stderr,
                    flush=True,
                )

            if mode == "offline":
                print(
                    f"[wandb] Run started: {run_name} (OFFLINE - will sync on finish)",
                    flush=True,
                )
            elif mode == "online":
                print(
                    f"[wandb] Run started: {run_name} (ONLINE - real-time syncing)",
                    flush=True,
                )
            else:
                print(
                    f"[wandb] Run started: {run_name} (mode={mode})", flush=True
                )

        except Exception as e:
            tb = traceback.format_exc()
            print(
                f"[wandb] Failed to start run: {e}", file=sys.stderr, flush=True
            )
            try:
                print(
                    f"[wandb] Traceback:\n{tb}", file=sys.stderr, flush=True
                )
            except Exception:
                pass
            print(
                f"[wandb] Logging will be disabled for {self.stage}",
                file=sys.stderr,
                flush=True,
            )
            self._run = None

    def finish(self) -> None:
        """Finish wandb run and sync if offline."""
        if not self.enabled or self._run is None:
            return

        try:
            run_name = getattr(self._run, "name", "unknown")
            run_mode = getattr(
                getattr(self._run, "settings", None), "mode", "unknown"
            )

            print(
                f"[wandb] Finishing run: {run_name} (mode={run_mode})",
                flush=True,
            )

            self.wandb.finish()

            if run_mode == "offline":
                print(
                    f"[wandb] Offline run '{run_name}' synced to cloud",
                    flush=True,
                )
            else:
                print(
                    f"[wandb] Online run '{run_name}' completed", flush=True
                )

            self._run = None

        except Exception as e:
            print(
                f"[wandb] Failed to finish run: {e}", file=sys.stderr, flush=True
            )
            self._run = None

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """Log metrics to wandb.

        Args:
            metrics: Dictionary of metric name -> value.
            step: Optional step number.
            commit: Whether to increment step counter.
        """
        if not self.enabled or self._run is None:
            return

        try:
            if metrics:
                self.wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            print(
                f"[wandb] Warning: Failed to log metrics: {e}", file=sys.stderr
            )

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
            df: Pandas DataFrame.
            key: Table name/key (e.g. "classify/results").
            prefer_cols: Optional list of preferred columns to include.
            max_rows: Max rows to sample (default from config).
            panel_group: Optional panel group name (e.g. "inspect_results").
                Will prefix the key as "panel_group/key".
        """
        if not self.enabled or self._run is None or df is None:
            return

        try:
            import pandas as pd  # noqa: F401 (guard import)

            table_key = f"{panel_group}/{key}" if panel_group else key

            # Deduplicate columns by name (keep first occurrence)
            try:
                df_local = df.loc[:, ~df.columns.duplicated()]
            except Exception:
                df_local = df

            # --- Drop internal LLM/mechanics columns ---
            def _drop_internal_llm_columns(df_in):
                try:
                    internal_cols: Set[str] = set(_UNIVERSAL_EXCLUDE)
                    internal_cols.update(self.wb_config.extra_internal_columns)

                    pattern_prefixes = list(self.wb_config.extra_pattern_prefixes)
                    pattern_names: Set[str] = set(self.wb_config.extra_pattern_names)

                    cols_present = [c for c in internal_cols if c in df_in.columns]
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

            # --- Filter heavy ndarray columns ---
            def _filter_heavy_columns(df_in, candidate_cols: List[str]) -> List[str]:
                keep = []
                for c in candidate_cols:
                    try:
                        if c in {"generated_tokens", "prompt_token_ids", "embeddings"}:
                            continue
                        sample = None
                        for v in df_in[c].values:
                            if v is not None:
                                sample = v
                                break
                        if sample is not None:
                            name = getattr(type(sample), "__name__", "") or ""
                            if name == "ndarray":
                                continue
                        keep.append(c)
                    except Exception:
                        keep.append(c)
                return keep

            # --- Decide whether to log all columns or only preferred ones ---
            full_column_stages = self.wb_config.full_column_stages
            full_column_key_prefixes = self.wb_config.full_column_key_prefixes

            key_triggers_full = panel_group == "inspect_results" or any(
                key.startswith(pfx) for pfx in full_column_key_prefixes
            )

            if self.stage in full_column_stages and key_triggers_full:
                cols = _filter_heavy_columns(df_local, list(df_local.columns))
            else:
                cols = [c for c in (prefer_cols or []) if c in df_local.columns]
                if not cols:
                    cols = list(df_local.columns)[:12]
                cols = _filter_heavy_columns(df_local, cols)

            try:
                print(
                    f"[wandb_logger] stage={self.stage} key={key} "
                    f"logging {len(cols)} columns: {cols}",
                    flush=True,
                )
            except Exception:
                pass

            # --- Final safeguard: strip any remaining excluded columns ---
            extra_excluded_names: Set[str] = set(self.wb_config.extra_pattern_names)
            extra_excluded_prefixes: List[str] = list(
                self.wb_config.extra_pattern_prefixes
            )

            def _is_excluded(name: str) -> bool:
                if name in _UNIVERSAL_EXCLUDE:
                    return True
                if name in extra_excluded_names:
                    return True
                for pref in extra_excluded_prefixes:
                    if name.startswith(pref):
                        return True
                return False

            cols = [c for c in cols if not _is_excluded(c)]

            # --- Sample ---
            max_rows = max_rows or self.wb_config.table_sample_rows
            total_rows = len(df_local)

            if total_rows > max_rows:
                df_sample = df_local.sample(
                    n=max_rows,
                    random_state=self.wb_config.table_sample_seed,
                ).reset_index(drop=True)
                sampled = True
            else:
                df_sample = df_local.reset_index(drop=True)
                sampled = False

            try:
                df_sample = df_sample.loc[:, ~df_sample.columns.duplicated()]
            except Exception:
                pass

            table = self.wandb.Table(dataframe=df_sample[cols])

            log_data: Dict[str, Any] = {
                table_key: table,
                f"{key}/rows": len(df_sample),
                f"{key}/total_rows": total_rows,
            }

            if sampled:
                log_data[f"{key}/sampled"] = True
                log_data[f"{key}/sample_seed"] = self.wb_config.table_sample_seed

            self.wandb.log(log_data)

            if sampled:
                print(
                    f"[wandb] Logged table '{table_key}': {len(df_sample):,} rows "
                    f"(randomly sampled from {total_rows:,})",
                    flush=True,
                )
            else:
                print(
                    f"[wandb] Logged table '{table_key}': {total_rows:,} rows",
                    flush=True,
                )

        except Exception as e:
            print(
                f"[wandb] Warning: Failed to log table '{key}': {e}",
                file=sys.stderr,
            )

    def log_artifact(
        self, artifact_path: str, name: str, type: str = "dataset"
    ) -> None:
        """Log artifact to wandb.

        Args:
            artifact_path: Path to artifact file/directory.
            name: Artifact name.
            type: Artifact type (e.g. "dataset", "model").
        """
        if not self.enabled or self._run is None:
            return

        try:
            artifact = self.wandb.Artifact(name=name, type=type)
            artifact.add_file(artifact_path)
            self.wandb.log_artifact(artifact)
        except Exception as e:
            print(
                f"[wandb] Warning: Failed to log artifact '{name}': {e}",
                file=sys.stderr,
            )

    def log_plot(self, key: str, figure) -> None:
        """Log matplotlib/plotly figure or wandb plot object.

        Args:
            key: Plot name/key.
            figure: Matplotlib figure, Plotly figure, or wandb plot object
                    (Image, Plotly, Html, etc.)
        """
        if not self.enabled or self._run is None:
            return

        try:
            if hasattr(figure, "__class__") and hasattr(
                figure.__class__, "__module__"
            ):
                module = figure.__class__.__module__
                if module and "wandb" in module:
                    self.wandb.log({key: figure})
                    return

            if hasattr(figure, "to_html"):
                self.wandb.log({key: figure})
            elif hasattr(figure, "savefig"):
                self.wandb.log({key: self.wandb.Image(figure)})
            else:
                self.wandb.log({key: figure})
        except Exception as e:
            print(
                f"[wandb] Warning: Failed to log plot '{key}': {e}",
                file=sys.stderr,
            )

    def set_summary(self, key: str, value: Any) -> None:
        """Set a run-level summary field (useful for categorical/string values)."""
        if not self.enabled or self._run is None:
            return
        try:
            self._run.summary[key] = value
        except Exception as e:
            print(
                f"[wandb] Warning: Failed to set summary '{key}': {e}",
                file=sys.stderr,
            )

    def set_config(
        self, data: Dict[str, Any], allow_val_change: bool = True
    ) -> None:
        """Update run config for stable non-time-series metadata."""
        if not self.enabled or self._run is None or not data:
            return
        try:
            self._run.config.update(dict(data), allow_val_change=allow_val_change)
        except Exception as e:
            print(
                f"[wandb] Warning: Failed to update config: {e}", file=sys.stderr
            )

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
        return False


__all__ = [
    "WandbConfig",
    "WandbLogger",
    "ensure_local_tmpdir",
    "collect_compute_metadata",
]
