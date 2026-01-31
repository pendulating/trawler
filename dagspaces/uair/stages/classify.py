"""
Classification stage module - DEPRECATED.

This module contains the legacy implementation of the classification stage.
It is maintained for internal reference only.

DEPRECATION NOTICE:
    This module is deprecated. Use profile-specific modules instead:
    - classify_relevance: Relevance classification
    - classify_eu_act: EU AI Act classification  
    - classify_risks_benefits: Risks and Benefits classification
    
    The legacy run_classification_stage() function will be removed in a future version.
    Profile-specific modules call run_classification_core() directly.
    
    NOTE: _run_classification_stage_impl is kept as an internal reference only.
    It is not part of the public API and should not be used by external code.
"""

import warnings
from typing import Any, Dict, List, Set

# Import shared utilities for backward compatibility
from .classify_shared import (
    BASE_INPUT_COLUMNS,
    EU_INPUT_KEYS,
    EU_OUTPUT_COLUMNS,
    EU_PROFILE_FLAG_COLUMNS,
    RELEVANCE_OUTPUT_COLUMNS,
    RESULT_BASE_COLUMNS,
    RISKS_BENEFITS_OUTPUT_COLUMNS,
    get_allowed_result_columns,
    get_required_input_columns,
    prune_result_columns,
)

# Deprecation warning flag
_DEPRECATION_WARNING_SHOWN = False

import pandas as pd

# Public API exports
__all__ = [
    "run_classification_stage",
    "run_classification_core",
    "get_required_input_columns",
    "get_allowed_result_columns",
    "BASE_INPUT_COLUMNS",
    "EU_INPUT_KEYS",
    "EU_OUTPUT_COLUMNS",
    "EU_PROFILE_FLAG_COLUMNS",
    "RELEVANCE_OUTPUT_COLUMNS",
    "RESULT_BASE_COLUMNS",
    "RISKS_BENEFITS_OUTPUT_COLUMNS",
]


def run_classification_stage(df: pd.DataFrame, cfg) -> Any:
    """
    DEPRECATED: Classification stage function.
    
    This function is deprecated. Use profile-specific runners instead:
    - ClassificationRelevanceRunner for relevance classification
    - ClassificationEUActRunner for EU AI Act classification
    - ClassificationRisksBenefitsRunner for risks and benefits classification
    """
    global _DEPRECATION_WARNING_SHOWN
    if not _DEPRECATION_WARNING_SHOWN:
        warnings.warn(
            "classify.run_classification_stage is deprecated. "
            "Use profile-specific runners (ClassificationRelevanceRunner, "
            "ClassificationEUActRunner, ClassificationRisksBenefitsRunner) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _DEPRECATION_WARNING_SHOWN = True
    
    # Determine profile and call appropriate function directly
    try:
        classification_profile = str(
            getattr(cfg.runtime, "classification_profile", "relevance") or "relevance"
        ).strip().lower()
    except Exception:
        classification_profile = "relevance"
    
    if classification_profile == "eu_ai_act":
        from .classify_eu_act import run_classification_eu_act
        return run_classification_eu_act(df, cfg)
    elif classification_profile == "risks_and_benefits":
        from .classify_risks_benefits import run_classification_risks_benefits
        return run_classification_risks_benefits(df, cfg)
    else:
        from .classify_relevance import run_classification_relevance
        return run_classification_relevance(df, cfg)

# Column definitions and utility functions have been moved to classify_shared.py
# These are re-exported above for convenience

# Import legacy implementation dependencies
try:
    import ray  # noqa: F401
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig  # type: ignore
    _RAY_OK = True
except Exception:
    _RAY_OK = False

from vllm.sampling_params import SamplingParams, GuidedDecodingParams  # type: ignore
import re
from bisect import bisect_right
import numpy as np
import os
import logging
import json
from omegaconf import OmegaConf

# Import shared utilities
from .classify_shared import (
    _maybe_silence_vllm_logs as maybe_silence_vllm_logs,
    _read_int_file as read_int_file,
    _detect_cgroup_mem_limit_bytes as detect_cgroup_mem_limit_bytes,
    _parse_int as parse_int,
    _parse_cpus_on_node as parse_cpus_on_node,
    _detect_slurm_job_mem_bytes as detect_slurm_job_mem_bytes,
    _effective_total_memory_bytes as effective_total_memory_bytes,
    _ensure_ray_init as ensure_ray_init,
    _to_json_str as to_json_str,
    _serialize_arrow_unfriendly_in_row as serialize_arrow_unfriendly_in_row,
    _sanitize_for_json as sanitize_for_json,
    _coerce_bool_like as coerce_bool_like,
    _coerce_boolish_row as coerce_boolish_row,
    _coerce_boolish_df as coerce_boolish_df,
    _detect_num_gpus as detect_num_gpus,
    _detect_gpu_type as detect_gpu_type,
    _apply_gpu_aware_batch_settings as apply_gpu_aware_batch_settings,
    _filter_vllm_engine_kwargs as filter_vllm_engine_kwargs,
    _build_relevant_regex as build_relevant_regex,
    _generate_relevant_blocks as generate_relevant_blocks,
    _normalize_profile_columns as normalize_profile_columns,
    _merge_result_parts as merge_result_parts,
    _prune_result_columns as prune_result_columns,
)

# Legacy aliases for backward compatibility
_VLLM_LOGS_SILENCED = False
_DEBUG_GUIDED_LOG = {"pre": 0, "fail": 0}

# Import shared utilities - use actual function names from classify_shared
from .classify_shared import (
    maybe_silence_vllm_logs,
    read_int_file,
    detect_cgroup_mem_limit_bytes,
    parse_int,
    parse_cpus_on_node,
    detect_slurm_job_mem_bytes,
    effective_total_memory_bytes,
    ensure_ray_init,
    to_json_str,
    serialize_arrow_unfriendly_in_row,
    sanitize_for_json,
    coerce_bool_like,
    coerce_boolish_row,
    coerce_boolish_df,
    detect_num_gpus,
    detect_gpu_type,
    apply_gpu_aware_batch_settings,
    filter_vllm_engine_kwargs,
    build_relevant_regex,
    generate_relevant_blocks,
    normalize_profile_columns,
    merge_result_parts,
    prune_result_columns,
)

# Create aliases with _ prefix for legacy code compatibility
_maybe_silence_vllm_logs = maybe_silence_vllm_logs
_read_int_file = read_int_file
_detect_cgroup_mem_limit_bytes = detect_cgroup_mem_limit_bytes
_parse_int = parse_int
_parse_cpus_on_node = parse_cpus_on_node
_detect_slurm_job_mem_bytes = detect_slurm_job_mem_bytes
_effective_total_memory_bytes = effective_total_memory_bytes
_ensure_ray_init = ensure_ray_init
_to_json_str = to_json_str
_serialize_arrow_unfriendly_in_row = serialize_arrow_unfriendly_in_row
_sanitize_for_json = sanitize_for_json
_coerce_bool_like = coerce_bool_like
_coerce_boolish_row = coerce_boolish_row
_coerce_boolish_df = coerce_boolish_df
_detect_num_gpus = detect_num_gpus
_detect_gpu_type = detect_gpu_type
_apply_gpu_aware_batch_settings = apply_gpu_aware_batch_settings
_filter_vllm_engine_kwargs = filter_vllm_engine_kwargs
_build_relevant_regex = build_relevant_regex
_generate_relevant_blocks = generate_relevant_blocks
_normalize_profile_columns = normalize_profile_columns
_merge_result_parts = merge_result_parts
_prune_result_columns = prune_result_columns


def _read_int_file(path: str) -> int:
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


def _parse_int(val: str) -> int:
    try:
        return int(val)
    except Exception:
        return -1


def _parse_cpus_on_node(val: str) -> int:
    if not isinstance(val, str):
        return -1
    try:
        # Common forms: "32", "16(x2)", "2,2", "2,1"
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


def _detect_slurm_job_mem_bytes() -> int:
    """Infer SLURM job memory allocation in bytes from env vars.

    Prefers SLURM_MEM_PER_NODE; otherwise uses SLURM_MEM_PER_CPU * SLURM_CPUS_ON_NODE.
    Values are MB according to SLURM docs.
    """
    try:
        mem_per_node_mb = os.environ.get("SLURM_MEM_PER_NODE")
        if mem_per_node_mb:
            mb = _parse_int(mem_per_node_mb)
            if mb > 0:
                return mb * 1024 * 1024
    except Exception:
        pass
    try:
        mem_per_cpu_mb = os.environ.get("SLURM_MEM_PER_CPU")
        cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE")
        if mem_per_cpu_mb and cpus_on_node:
            mb = _parse_int(mem_per_cpu_mb)
            cpus = _parse_cpus_on_node(cpus_on_node)
            if mb > 0 and cpus > 0:
                return mb * cpus * 1024 * 1024
    except Exception:
        pass
    return -1


def _effective_total_memory_bytes() -> int:
    """Best-effort job-aware total memory for sizing Ray object store.

    Order of preference:
    1) cgroup memory limit (container/SLURM cgroup)
    2) SLURM env-based memory inference
    3) System total memory (psutil/sysconf)
    """
    # cgroup limit first
    cg = _detect_cgroup_mem_limit_bytes()
    if cg > 0:
        return cg
    # SLURM-derived
    sj = _detect_slurm_job_mem_bytes()
    if sj > 0:
        return sj
    # Fallback to system total
    try:
        import psutil  # type: ignore
        tot = int(getattr(psutil.virtual_memory(), "total", 0))
        if tot > 0:
            return tot
    except Exception:
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        pass
    return -1


def _ensure_ray_init(cfg) -> None:
    try:
        import ray  # type: ignore
        if not ray.is_initialized():
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
            # Prefer proportion of system memory when available; fallback to job_memory_gb.
            obj_store_bytes = None
            try:
                # Allow explicit override via cfg.runtime.object_store_proportion (0.0-1.0)
                prop = getattr(cfg.runtime, "object_store_proportion", None)
                prop = float(prop) if prop is not None else None
            except Exception:
                prop = None
            # Honor env proportion if set and no explicit override provided
            try:
                if prop is None:
                    env_prop = os.environ.get("RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION")
                    if env_prop is not None and str(env_prop).strip() != "":
                        prop = float(env_prop)
            except Exception:
                pass
            if prop is not None and 0.0 < prop <= 0.95:
                try:
                    total_bytes = _effective_total_memory_bytes()
                    if total_bytes:
                        obj_store_bytes = int(total_bytes * float(prop))
                except Exception:
                    obj_store_bytes = None
            if obj_store_bytes is None:
                try:
                    # Prefer SLURM job mem if available to avoid using full node memory
                    slurm_bytes = _detect_slurm_job_mem_bytes()
                    if slurm_bytes > 0:
                        job_mem_gb = max(1, int(slurm_bytes / (1024 ** 3)))
                    else:
                        job_mem_gb = int(getattr(cfg.runtime, "job_memory_gb", 64) or 64)
                except Exception:
                    job_mem_gb = 64
                try:
                    job_mem_gb = int(getattr(cfg.runtime, "job_memory_gb", 64) or 64)
                except Exception:
                    job_mem_gb = 64
                try:
                    obj_store_bytes = int(max(1, job_mem_gb) * (1024 ** 3) * 0.90)
                except Exception:
                    obj_store_bytes = int(64 * (1024 ** 3) * 0.90)
            try:
                if cpus_alloc is not None and int(cpus_alloc) > 0:
                    ray.init(log_to_driver=True, object_store_memory=int(obj_store_bytes), num_cpus=int(cpus_alloc))
                else:
                    ray.init(log_to_driver=True, object_store_memory=int(obj_store_bytes))
            except Exception:
                # Best-effort fallback: let Ray auto-init
                try:
                    ray.init(log_to_driver=True)
                except Exception:
                    pass
            # Constrain Ray Data CPU limits to SLURM allocation when available
            try:
                if cpus_alloc is not None and int(cpus_alloc) > 0:
                    ctx = ray.data.DataContext.get_current()
                    ctx.execution_options.resource_limits = ctx.execution_options.resource_limits.copy(cpu=int(cpus_alloc))
            except Exception:
                pass
    except Exception:
        pass

def _to_json_str(value: Any):
    """Serialize Python objects to JSON string for Arrow/Parquet friendliness.

    Returns None for None input; falls back to str(value) on failure.
    """
    try:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        try:
            return str(value)
        except Exception:
            return None


def _serialize_arrow_unfriendly_in_row(row: Dict[str, Any], columns: List[str]) -> None:
    """In-place convert nested/dict/list columns to JSON strings in a row dict."""
    for col in columns:
        if col in row:
            val = row.get(col)
            if isinstance(val, (dict, list, tuple)):
                row[col] = _to_json_str(val)
            elif isinstance(val, GuidedDecodingParams):
                row[col] = str(val)
            elif isinstance(val, SamplingParams):
                row[col] = str(val)


def _sanitize_for_json(value: Any):
    """Recursively convert value to JSON-serializable builtins.

    - dict -> dict with str keys and sanitized values
    - list/tuple/set -> list of sanitized values
    - objects with .tolist() -> use tolist() (e.g., numpy arrays)
    - other non-serializables -> str(value)
    """
    try:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                try:
                    key = str(k)
                except Exception:
                    key = repr(k)
                out[key] = _sanitize_for_json(v)
            return out
        if isinstance(value, (list, tuple, set)):
            return [_sanitize_for_json(v) for v in value]
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                return _sanitize_for_json(tolist())
            except Exception:
                pass
        return str(value)
    except Exception:
        return None


_BOOLISH_SUFFIXES = ("_verified",)
_BOOLISH_EXACT = {
    "core_tuple_verified",
    "doc_any_component_verified",
    "ver_tuples_any_verified",
    "ver_tuple_overall_pass",
    "relevant_keyword",
    "too_vague_to_process",
    "is_relevant",
}

def _coerce_bool_like(value: Any) -> bool:
    try:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except Exception:
            pass
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                return _coerce_bool_like(value.item())
            except Exception:
                pass
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(int(value))
        if isinstance(value, str):
            lv = value.strip().lower()
            if lv in {"", "none", "null", "nan"}:
                return False
            if lv in {"true", "1", "yes", "y", "t"}:
                return True
            if lv in {"false", "0", "no", "n", "f"}:
                return False
        return bool(value)
    except Exception:
        return False


def _coerce_boolish_row(r: Dict[str, Any]) -> Dict[str, Any]:
    for key in list(r.keys()):
        if key in _BOOLISH_EXACT or key.endswith(_BOOLISH_SUFFIXES):
            try:
                r[key] = _coerce_bool_like(r.get(key))
            except Exception:
                r[key] = False
        else:
            val = r.get(key)
            if isinstance(val, np.ndarray):
                try:
                    val = val.tolist()
                except Exception:
                    val = list(val)
                r[key] = val
            elif hasattr(val, "tolist") and not isinstance(val, (str, bytes)):
                try:
                    candidate = val.tolist()
                    r[key] = candidate
                except Exception:
                    pass
    return r


def _coerce_boolish_df(pdf: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(pdf, pd.DataFrame):
        return pdf
    cols = list(pdf.columns)
    for col in cols:
        if col in _BOOLISH_EXACT or col.endswith(_BOOLISH_SUFFIXES):
            try:
                pdf[col] = pdf[col].apply(_coerce_bool_like)
            except Exception:
                try:
                    pdf[col] = pdf[col].fillna(False).astype(bool)
                except Exception:
                    pass
        else:
            try:
                sample = next(
                    (v for v in pdf[col].values if v is not None and not (isinstance(v, float) and pd.isna(v))),
                    None,
                )
            except Exception:
                sample = None
            if isinstance(sample, np.ndarray) or (
                hasattr(sample, "tolist") and not isinstance(sample, (str, bytes))
            ):
                try:
                    pdf[col] = pdf[col].apply(
                        lambda v: v.tolist() if isinstance(v, np.ndarray) else (
                            v.tolist() if (hasattr(v, "tolist") and not isinstance(v, (str, bytes))) else v
                        )
                    )
                except Exception:
                    pass
    return pdf


def _detect_num_gpus() -> int:
    """Detect the number of GPUs allocated to this job.
    
    Priority order:
    1. CUDA_VISIBLE_DEVICES environment variable (set by launcher)
    2. SLURM_GPUS_PER_NODE or SLURM_GPUS_ON_NODE
    3. torch.cuda.device_count() if CUDA is available
    4. Return 1 as safe fallback
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
        slurm_gpus = os.environ.get("SLURM_GPUS_PER_NODE") or os.environ.get("SLURM_GPUS_ON_NODE")
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
    
    # Priority 3: Torch CUDA device count
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                return count
    except Exception:
        pass
    
    # Fallback: 1 GPU
    return 1


def _detect_gpu_type() -> str:
    """Detect the GPU type/model name.
    
    Returns a normalized GPU type string (e.g., 'rtx_a6000', 'rtx_a5000', 'unknown').
    """
    try:
        import torch  # type: ignore
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # Get the name of the first GPU (assuming homogeneous GPUs)
            gpu_name = torch.cuda.get_device_name(0).lower()
            
            # Normalize common GPU names
            if "a6000" in gpu_name:
                return "rtx_a6000"
            elif "a5000" in gpu_name:
                return "rtx_a5000"
            elif "a100" in gpu_name:
                return "a100"
            elif "v100" in gpu_name:
                return "v100"
            elif "a40" in gpu_name:
                return "a40"
            elif "rtx" in gpu_name:
                # Generic RTX
                return "rtx_generic"
            
            return "unknown"
    except Exception:
        pass
    
    return "unknown"


def _apply_gpu_aware_batch_settings(engine_kwargs: Dict[str, Any], cfg) -> None:
    """Apply GPU-type-aware batch size and max_num_seqs if not explicitly set.
    
    GPU-specific defaults (can be overridden in config):
    - RTX A6000: batch_size=4, max_num_seqs=4
    - RTX A5000: batch_size=2, max_num_seqs=2
    - Others: Use config defaults or fallback values
    """
    # GPU-aware defaults mapping
    GPU_BATCH_SETTINGS = {
        "rtx_a6000": {"batch_size": 4, "max_num_seqs": 4},
        "rtx_a5000": {"batch_size": 2, "max_num_seqs": 2},
        "a100": {"batch_size": 8, "max_num_seqs": 8},
        "v100": {"batch_size": 4, "max_num_seqs": 4},
        "a40": {"batch_size": 4, "max_num_seqs": 4},
    }
    
    gpu_type = _detect_gpu_type()
    gpu_settings = GPU_BATCH_SETTINGS.get(gpu_type, {})
    
    # Apply max_num_seqs from GPU settings if not in engine_kwargs and GPU type is recognized
    if "max_num_seqs" not in engine_kwargs and gpu_settings:
        try:
            engine_kwargs["max_num_seqs"] = gpu_settings["max_num_seqs"]
            if not os.environ.get("RULE_TUPLES_SILENT"):
                print(f"Auto-set max_num_seqs={gpu_settings['max_num_seqs']} for {gpu_type}")
        except Exception:
            pass
    
    # Note: batch_size is handled separately in the calling code since it's a model config param, not engine_kwargs
    # We'll return the gpu_settings for use there
    return gpu_settings


def _filter_vllm_engine_kwargs(ek: Dict[str, Any]) -> Dict[str, Any]:
    """Drop engine kwargs unsupported by the installed vLLM version.

    We try to introspect vllm.AsyncEngineArgs for accepted fields. If that
    fails, conservatively drop known newer flags.
    """
    try:
        import vllm as _v
        accepted = None
        # Prefer dataclass fields (older vLLM uses dataclasses)
        try:
            fields = getattr(getattr(_v, "AsyncEngineArgs", None), "__dataclass_fields__", None)
            if isinstance(fields, dict) and fields:
                accepted = set(fields.keys())
        except Exception:
            accepted = None
        # Fallback to signature introspection
        if accepted is None:
            try:
                import inspect as _inspect
                sig = _inspect.signature(_v.AsyncEngineArgs.__init__)
                accepted = set(k for k in sig.parameters.keys() if k != "self")
            except Exception:
                accepted = None
        if accepted:
            filtered = {k: v for k, v in ek.items() if k in accepted}
            if "guided_decoding" in ek and "guided_decoding" not in filtered and not os.environ.get("RULE_TUPLES_SILENT"):
                try:
                    print("[classify] Warning: guided_decoding not accepted by AsyncEngineArgs; keeping original value", flush=True)
                except Exception:
                    pass
                filtered["guided_decoding"] = ek.get("guided_decoding")
            if len(filtered) != len(ek):
                try:
                    if not os.environ.get("RULE_TUPLES_SILENT"):
                        dropped = [k for k in ek.keys() if k not in filtered]
                        print(f"Filtering unsupported vLLM engine kwargs: {dropped}")
                except Exception:
                    pass
            return filtered
    except Exception:
        pass
    # Conservative fallback for unknown versions: drop newer flags
    ek = dict(ek)
    for k in ("use_v2_block_manager",):
        ek.pop(k, None)
    return ek


def _build_relevant_regex() -> re.Pattern:
    
    phrases = [
        # Core AI Technologies & Acronyms
        r"\bai\b",
        r"\bml\b",
        r"\bnlp\b",
        r"\bllm\b",
        r"\bagi\b",
        r"\bxai\b",
        r"\biot\b",
        r"artificial\s+intelligence",
        r"machine\s+learning",
        r"neural\s+network",
        r"large\s+language\s+model",
        r"transformer",
        r"chatgpt|gpt-\d+|gpt-",
        r"openai|anthropic|claude|gemini|qwen",
        r"fine-?tuning|inference|prompt(ing)?|agent(s)?",
        
        # Journalist-Friendly AI Terms
        r"computer",
        r"computers",
        r"software",
        r"program",
        r"programs",
        r"programming",
        r"coded",
        r"coding",
        r"app",
        r"apps",
        r"application",
        r"applications",
        r"tool",
        r"tools",
        r"technology",
        r"tech",
        r"innovation",
        r"innovations",
        r"breakthrough",
        r"breakthroughs",
        r"advancement",
        r"advancements",
        
        # Anthropomorphic/Accessible Descriptions
        r"robot",
        r"robots",
        r"robotic",
        r"bot",
        r"bots",
        r"chatbot",
        r"chatbots",
        r"virtual\s+assistant",
        r"digital\s+assistant",
        r"smart\s+assistant",
        r"machine",
        r"machines",
        r"device",
        r"devices",
        r"smart\s+device",
        r"intelligent\s+system",
        r"thinking\s+machine",
        r"electronic\s+brain",
        r"digital\s+brain",
        r"computer\s+brain",
        
        # Process-Oriented Terms
        r"automated",
        r"automation",
        r"automatic",
        r"automatically",
        r"self-?learning",
        r"self-?teaching",
        r"self-?improving",
        r"adaptive",
        r"smart",
        r"intelligent",
        r"cognitive",
        r"thinking",
        r"reasoning",
        r"decision-?making",
        r"processing",
        r"analyzing",
        r"analysis",
        r"pattern\s+recognition",
        r"image\s+recognition",
        r"voice\s+recognition",
        r"language\s+processing",
        
        # Capability Descriptions
        r"learns",
        r"learning",
        r"teaches\s+itself",
        r"trains",
        r"training",
        r"trained",
        r"understands",
        r"understanding",
        r"recognizes",
        r"recognition",
        r"identifies",
        r"identification",
        r"predicts",
        r"prediction",
        r"predictions",
        r"forecasts",
        r"forecasting",
        r"generates",
        r"generation",
        r"creates",
        r"creation",
        r"produces",
        r"mimics",
        r"simulates",
        r"replicates",
        
        # Business/Industry Terms
        r"silicon\s+valley",
        r"tech\s+company",
        r"tech\s+companies",
        r"tech\s+giant",
        r"tech\s+giants",
        r"startup",
        r"startups",
        r"big\s+tech",
        r"platform",
        r"platforms",
        r"service",
        r"services",
        r"product",
        r"products",
        r"solution",
        r"solutions",
        r"ecosystem",
        r"infrastructure",
        
        # Buzzword/Hype Terms
        r"revolutionary",
        r"game-?changing",
        r"cutting-?edge",
        r"state-?of-?the-?art",
        r"next-?generation",
        r"futuristic",
        r"advanced",
        r"sophisticated",
        r"powerful",
        r"disruptive",
        r"transformative",
        r"groundbreaking",
        r"innovative",
        r"emerging",
        r"novel",
        r"pioneering",
        
        # Comparison/Analogy Terms
        r"human-?like",
        r"human-?level",
        r"superhuman",
        r"brain-?like",
        r"mimicking\s+humans?",
        r"replacing\s+humans?",
        r"outsmarting\s+humans?",
        r"beating\s+humans?",
        r"surpassing\s+humans?",
        r"artificial\s+brain",
        r"electronic\s+mind",
        r"digital\s+worker",
        r"virtual\s+employee",
        
        # AI Safety & Governance
        r"safety",
        r"alignment",
        r"governance",
        r"responsible",
        r"trustworthy",
        r"ethics",
        r"ethical",
        r"bias",
        r"biased",
        r"fairness",
        r"explainable",
        r"transparency",
        r"transparent",
        r"accountability",
        r"accountable",
        r"regulation",
        r"oversight",
        r"compliance",
        
        # Risk & Harm Terms
        r"risk",
        r"risks",
        r"harm",
        r"harms",
        r"harmful",
        r"danger",
        r"dangerous",
        r"threat",
        r"threats",
        r"vulnerability",
        r"vulnerabilities",
        r"attack",
        r"attacks",
        r"exploitation",
        r"manipulation",
        r"weaponization",
        
        # Societal Impact Terms
        r"deepfake",
        r"misinformation",
        r"disinformation",
        r"fake\s+news",
        r"discrimination",
        r"discriminatory",
        r"surveillance",
        r"capitalism",
        r"privacy",
        r"invasion",
        r"facial\s+recognition",
        r"predictive\s+policing",
        r"social\s+credit",
        r"filter\s+bubble",
        r"echo\s+chamber",
        r"polarization",
        r"radicalization",
        
        # Economic & Labor
        r"displacement",
        r"unemployment",
        r"automation",
        r"automated",
        r"replacement",
        r"disruption",
        r"workforce",
        r"labor",
        r"labour",
        r"jobs",
        r"employment",
        r"gig\s+economy",
        r"platform\s+workers?",
        r"algorithmic\s+management",
        
        # AI Systems & Applications
        r"algorithm",
        r"algorithms",
        r"algorithmic",
        r"autonomous",
        r"self-?driving",
        r"recommendation",
        r"moderation",
        r"computer\s+vision",
        r"speech\s+recognition",
        r"sentiment\s+analysis",
        r"predictive\s+analytics",
        r"decision\s+support",
        r"synthetic\s+media",
        r"voice\s+cloning",
        
        # Major Tech Companies
        r"microsoft",
        r"google",
        r"amazon",
        r"meta",
        r"facebook",
        r"nvidia",
        r"intel",
        r"apple",
        r"tesla",
        r"deepmind",
        r"hugging\s?face",
        r"stability\s+ai",
        r"midjourney",
        r"dall-?e",
        r"baidu",
        r"alibaba",
        r"tencent",
        r"bytedance",
        
        # Regulatory & Legal
        r"gdpr",
        r"ccpa",
        r"regulation",
        r"regulatory",
        r"policy",
        r"policies",
        r"governance",
        r"antitrust",
        r"monopoly",
        r"section\s+230",
        r"digital\s+rights",
        r"data\s+protection",
        r"impact\s+assessment",
        
        # Social & Cultural Harms
        r"manipulation",
        r"identity\s+theft",
        r"cyberbullying",
        r"harassment",
        r"hate\s+speech",
        r"democratic",
        r"democracy",
        r"election",
        r"elections",
        r"voting",
        r"interference",
        
        # Technical Risks
        r"adversarial",
        r"poisoning",
        r"backdoor",
        r"inference",
        r"inversion",
        r"leakage",
        r"robustness",
        r"brittleness",
        r"hallucination",
        r"confabulation",
        r"distribution\s+shift",
        r"out-?of-?distribution",
        
        # Long-term & Existential
        r"superintelligence",
        r"existential",
        r"x-?risk",
        r"control\s+problem",
        r"value\s+alignment",
        r"mesa-?optimization",
        r"instrumental\s+convergence",
        r"orthogonality",
        
        # Domain Applications
        r"city",
        r"cities",
        r"urban",
        r"climate",
        r"earth",
        r"environment",
        r"environmental",
        r"transport",
        r"transportation",
        r"smart\s+grid",
        r"infrastructure",
        r"connected\s+device",
        r"monitoring",
        r"carbon",
        r"energy",
        r"sustainable",
        r"sustainability",
        r"green",
        
        # Healthcare
        r"medical",
        r"healthcare",
        r"health",
        r"diagnostic",
        r"clinical",
        r"telemedicine",
        r"therapeutics",
        r"equity",
        
        # Financial
        r"trading",
        r"advisor",
        r"credit",
        r"scoring",
        r"financial",
        r"inclusion",
        r"redlining",
        r"lending",
        r"predatory",
        
        # Education
        r"education",
        r"educational",
        r"edtech",
        r"learning",
        r"student",
        r"academic",
        r"integrity",
        r"cheating",
        r"proctoring",
        
        # Criminal Justice
        r"criminal",
        r"justice",
        r"recidivism",
        r"bail",
        r"sentencing",
        r"policing",
        r"enforcement",
        
        # Emerging Tech
        r"quantum",
        r"neuromorphic",
        r"edge\s+computing",
        r"distributed",
        r"swarm",
        r"multi-?agent",
        
        # General Tech Terms
        r"technology",
        r"technological",
        r"digital",
        r"cyber",
        r"online",
        r"internet",
        r"platform",
        r"platforms",
        r"data",
        r"dataset",
        r"datasets",
        r"model",
        r"models",
        r"system",
        r"systems",
        
    ]
    pattern = r"(" + r"|".join(phrases) + r")"
    return re.compile(pattern, flags=re.IGNORECASE)


def _generate_relevant_blocks(text: str, compiled_regex: re.Pattern, window_words: int = 100) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    token_matches = list(re.finditer(r"\S+", text))
    if not token_matches:
        return []
    token_starts = [m.start() for m in token_matches]
    intervals: List[List[int]] = []
    for m in compiled_regex.finditer(text):
        idx = max(0, min(len(token_starts) - 1, bisect_right(token_starts, m.start()) - 1))
        start_token = max(0, idx - window_words)
        end_token = min(len(token_matches) - 1, idx + window_words)
        start_char = token_matches[start_token].start()
        end_char = token_matches[end_token].end()
        intervals.append([start_char, end_char])
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged: List[List[int]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [text[s:e] for s, e in merged]




def run_classification_core(df: pd.DataFrame, cfg):
    """
    Core classification implementation.
    
    This is the actual implementation that performs classification.
    Profile-specific modules call this function after setting up their profile-specific configuration.
    
    NOTE: This is the legacy implementation preserved for backward compatibility.
    The full implementation (2965 lines) is kept here to maintain functionality
    while we transition to profile-specific modules.
    
    Article relevance classification stage:
    - Heuristic: detect AI-related keywords in text
    - LLM: borrow prompt shape from experiments/prompts/base.yaml (relevance_v1) when available,
      else fall back to UAIR classify prompt. Produces both `relevance_answer` and `is_relevant`.
    Uses `article_text` as the canonical input text column.
    """
    # Ensure Ray is initialized with desired object store cap based on job_memory_gb
    _ensure_ray_init(cfg)
    # Streaming path: if a Ray Dataset is passed, use it end-to-end
    is_ray_ds = hasattr(df, "map_batches") and hasattr(df, "count") and _RAY_OK
    if not is_ray_ds:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["article_id","article_text","is_relevant"])  # minimal
        out = df.copy()
        # Ensure article_id exists for all rows (deterministic hash of path or text)
        try:
            import hashlib as _hash
            if "article_id" not in out.columns:
                out["article_id"] = None
            def _gen_id_row(r: pd.Series) -> str:
                try:
                    src = r.get("article_path")
                    if not isinstance(src, str) or src.strip() == "":
                        src = r.get("article_text") or r.get("chunk_text") or ""
                    return _hash.sha1(str(src).encode("utf-8")).hexdigest()
                except Exception:
                    return _hash.sha1(str(r.get("article_text") or r.get("chunk_text") or "").encode("utf-8")).hexdigest()
            def _need_id(val: Any) -> bool:
                # Treat non-strings, empty strings, and NaNs as missing IDs
                return not (isinstance(val, str) and val.strip() != "")
            try:
                mask_missing = out["article_id"].apply(_need_id)
            except Exception:
                mask_missing = True
            if isinstance(mask_missing, bool):
                out["article_id"] = out.apply(_gen_id_row, axis=1)
            else:
                out.loc[mask_missing, "article_id"] = out.loc[mask_missing].apply(_gen_id_row, axis=1)
        except Exception:
            pass
        out = _coerce_boolish_df(out)
    def _heuristic(text: Any) -> bool:
        s = str(text or "").lower()
        if not s:
            return False
        # Simple AI/news relevance heuristic aligned to experiments.relevance_stage intent
        keywords = [
            "artificial intelligence"," ai ","machine learning","ml ","neural network","deep learning",
            "large language model","llm","chatgpt","gpt-","openai","anthropic","claude","gemini","qwen",
            "transformer model","fine-tuning","inference","prompting","agents","autonomous agent","model weights",
        ]
        return any(k in s for k in keywords)
    use_llm = bool(getattr(cfg.runtime, "use_llm_classify", False))
    # Select classification profile (Hydra overrideable): 'relevance' (default), 'eu_ai_act', or 'risks_and_benefits'
    try:
        classification_profile = str(getattr(cfg.runtime, "classification_profile", "relevance") or "relevance").strip().lower()
    except Exception:
        classification_profile = "relevance"
    is_eu_profile = (classification_profile == "eu_ai_act")
    is_risks_benefits_profile = (classification_profile == "risks_and_benefits")
    try:
        print(
            f"[classify] classification_profile={classification_profile} eu={is_eu_profile} rb={is_risks_benefits_profile}",
            flush=True,
        )
    except Exception:
        pass
    # Prefilter gating: pre_gating (filter before LLM), post_gating (compute flag only), off
    try:
        prefilter_mode = str(getattr(cfg.runtime, "prefilter_mode", "pre_gating")).strip().lower()
    except Exception:
        prefilter_mode = "pre_gating"
    if not use_llm:
        if is_ray_ds:
            def _heuristic_batch(pdf: pd.DataFrame) -> pd.DataFrame:
                pdf = pdf.copy()
                pdf["is_relevant"] = pdf["article_text"].apply(_heuristic)
                pdf["classification_mode"] = "heuristic"
                return pdf
            return df.map_batches(_heuristic_batch, batch_format="pandas")
        out["is_relevant"] = out["article_text"].apply(_heuristic)
        out["classification_mode"] = "heuristic"
        out_df = out
        return out_df

    if not _RAY_OK:
        out["is_relevant"] = out["article_text"].apply(_heuristic)
        out["classification_mode"] = "heuristic_fallback"
        try:
            print("Warning: Ray LLM not available; falling back to heuristic classification.")
        except Exception:
            pass
        return out

    # LLM classification via Ray vLLM
    # Optional keyword-based buffering (±window words around matches)
    try:
        enable_kw_buf = bool(getattr(cfg.runtime, "keyword_buffering", True))
    except Exception:
        enable_kw_buf = True

    # EU and risks_benefits profiles: disable keyword buffering/gating entirely
    if is_eu_profile or is_risks_benefits_profile:
        prefilter_mode = "off"
        enable_kw_buf = False
    try:
        window_words = int(getattr(cfg.runtime, "keyword_window_words", 100) or 100)
    except Exception:
        window_words = 100
    kw_regex = _build_relevant_regex() if enable_kw_buf else None
    # Compute keyword prefilter flag for gating
    def _kw_flag(text: Any) -> bool:
        if kw_regex is None:
            return True
        try:
            return bool(kw_regex.search(str(text or "")))
        except Exception:
            return True
    
    # Extract specific keywords that matched in the text
    def _extract_matched_keywords(text: Any) -> tuple[bool, list[str], int]:
        """Extract which keywords matched in the text.
        
        Returns:
            tuple: (has_keyword, matched_keywords_list, match_count)
        """
        if kw_regex is None:
            return True, [], 0
        try:
            text_str = str(text or "").lower()
            if not text_str:
                return False, [], 0
            
            # Use finditer() and .group() to get full matches (not capture groups)
            # This avoids the issue where findall() with capturing groups returns tuples
            matches = [m.group() for m in kw_regex.finditer(text_str)]
            if not matches:
                return False, [], 0
            
            # Deduplicate and sort for consistency
            unique_keywords = sorted(set(matches))
            match_count = len(matches)  # Total matches (including duplicates)
            
            return True, unique_keywords, match_count
        except Exception:
            return True, [], 0
    # Prefer experiments-style prompts when available; otherwise fall back to UAIR prompt
    if is_eu_profile:
        # EU AI Act classification: try nested prompts.eu_ai_act first; else cfg.prompt.* (Hydra-composed)
        try:
            system_prompt = (
                getattr(getattr(cfg, "prompts", {}), "eu_ai_act", {}).get("system")  # type: ignore
                if hasattr(cfg, "prompts") else None
            )
        except Exception:
            system_prompt = None
        if not system_prompt:
            system_prompt = str(getattr(cfg.prompt, "system_prompt", ""))

        try:
            user_template_eu = (
                getattr(getattr(cfg, "prompts", {}), "eu_ai_act", {}).get("user_template")  # type: ignore
                if hasattr(cfg, "prompts") else None
            )
        except Exception:
            user_template_eu = None
        if not user_template_eu:
            user_template_eu = str(getattr(cfg.prompt, "prompt_template", ""))
    elif is_risks_benefits_profile:
        # Risks and Benefits classification: try nested prompts.risks_and_benefits first; else cfg.prompt.* (Hydra-composed)
        try:
            system_prompt = (
                getattr(getattr(cfg, "prompts", {}), "risks_and_benefits", {}).get("system")  # type: ignore
                if hasattr(cfg, "prompts") else None
            )
        except Exception:
            system_prompt = None
        if not system_prompt:
            system_prompt = str(getattr(cfg.prompt, "system_prompt", ""))

        try:
            user_template_risks_benefits = (
                getattr(getattr(cfg, "prompts", {}), "risks_and_benefits", {}).get("user_template")  # type: ignore
                if hasattr(cfg, "prompts") else None
            )
        except Exception:
            user_template_risks_benefits = None
        if not user_template_risks_benefits:
            user_template_risks_benefits = str(getattr(cfg.prompt, "prompt_template", ""))
    else:
        try:
            system_prompt = (
                getattr(getattr(cfg, "prompts", {}), "relevance_v1", {}).get("system")  # type: ignore
                if hasattr(cfg, "prompts") else None
            )
        except Exception:
            system_prompt = None
        if not system_prompt:
            system_prompt = str(getattr(cfg.prompt, "system_prompt", ""))

        try:
            user_template = (
                getattr(getattr(cfg, "prompts", {}), "relevance_v1", {}).get("user_template")  # type: ignore
                if hasattr(cfg, "prompts") else None
            )
        except Exception:
            user_template = None
        if not user_template:
            user_template = str(getattr(cfg.prompt, "prompt_template", ""))

    def _format_user(article_text: str, row: Dict[str, Any]) -> str:
        text_val = str((row.get("chunk_text") if row.get("chunk_text") else article_text) or "")
        # Provide basic ids to satisfy experiments-style templates when present
        try:
            art_id = row.get("article_id") or row.get("name") or None
        except Exception:
            art_id = None
        if not art_id:
            try:
                import hashlib as _hash
                art_id = _hash.sha1(text_val.encode("utf-8")).hexdigest()[:12]
            except Exception:
                art_id = "unknown"
        # If template uses '{chunk_text}' style, format with .format; else support UAIR '{{rule_text}}'
        if "{chunk_text}" in user_template or "{article_id}" in user_template:
            try:
                return user_template.format(article_id=art_id, chunk_id=0, num_chunks=1, chunk_text=text_val)
            except Exception:
                pass
        # Maintain UAIR template compatibility
        return user_template.replace("{{rule_text}}", text_val).replace("{{article_text}}", text_val)

    # Utilities for EU AI Act prompt formatting
    try:
        _EU_MISSING_PH = str(getattr(cfg.runtime, "eu_missing_placeholder", "Not known/specified"))
    except Exception:
        _EU_MISSING_PH = "Not known/specified"

    def _is_nan_like(v: Any) -> bool:
        try:
            # pandas/NumPy NaN/NA checks
            if v is None:
                return True
            if isinstance(v, float):
                return v != v
            # pandas NA scalar string repr
            return str(v).strip().lower() in {"nan", "na", "none"}
        except Exception:
            return False

    def _norm_str(val: Any) -> str:
        try:
            if _is_nan_like(val):
                return _EU_MISSING_PH
            # Flatten common containers
            if isinstance(val, (list, tuple, set)):
                vals = [str(x).strip() for x in val if x is not None and str(x).strip() != ""]
                return ", ".join(vals) if vals else _EU_MISSING_PH
            s = str(val).strip()
            return s if s else _EU_MISSING_PH
        except Exception:
            return _EU_MISSING_PH

    def _first_present(row: Dict[str, Any], keys: List[str]) -> Any:
        for k in keys:
            try:
                v = row.get(k)
                if isinstance(v, str) and v.strip() != "":
                    return v
                if v not in (None, "") and not isinstance(v, str):
                    return v
            except Exception:
                pass
        return None

    def _format_user_eu(row: Dict[str, Any]) -> str:
        # Map decompose outputs to EU fields with graceful fallbacks
        domain = _norm_str(_first_present(row, [
            "deployment_domain", "domain", "use_domain"
        ]))
        purpose = _norm_str(_first_present(row, [
            "deployment_purpose", "purpose", "goal", "objective"
        ]))
        capability = _norm_str(_first_present(row, [
            "deployment_capability", "capability", "capabilities", "function", "ability"
        ]))
        ai_developer = _norm_str(_first_present(row, [
            "identity_of_ai_developer", "ai_developer", "developer", "vendor", "builder", "provider", "manufacturer"
        ]))
        ai_deployer = _norm_str(_first_present(row, [
            "identity_of_ai_deployer", "deployer", "operator", "implementer", "user", "agency", "organization_deployer"
        ]))
        ai_deployer_location = _norm_str(_first_present(row, [
            "location_of_ai_deployer", "deployer_location", "operator_location", "location_deployer"
        ]))
        ai_subject = _norm_str(_first_present(row, [
            "identity_of_ai_subject", "ai_subject", "subject", "data_subject", "affected_party", "individual", "group"
        ]))
        ai_subject_location = _norm_str(_first_present(row, [
            "location_of_ai_subject", "subject_location", "location_subject", "where"
        ]))
        date_time = _norm_str(_first_present(row, [
            "date_and_time_of_event", "date___time_of_event", "datetime", "date_time", "date", "time", "event_time", "when"
        ]))
        # Attempt to fill extended 9-field template first; fallback to 5-field template; else plain text

        return user_template_eu.format(
                domain,
                purpose,
                capability,
                ai_developer,
                ai_deployer,
                ai_deployer_location,
                ai_subject,
                ai_subject_location,
                date_time,
            )

    def _format_user_risks_benefits(row: Dict[str, Any]) -> str:
        # Map decompose outputs to risks_benefits fields (same 9-field structure as EU)
        domain = _norm_str(_first_present(row, [
            "deployment_domain", "domain", "use_domain"
        ]))
        purpose = _norm_str(_first_present(row, [
            "deployment_purpose", "purpose", "goal", "objective"
        ]))
        capability = _norm_str(_first_present(row, [
            "deployment_capability", "capability", "capabilities", "function", "ability"
        ]))
        ai_developer = _norm_str(_first_present(row, [
            "identity_of_ai_developer", "ai_developer", "developer", "vendor", "builder", "provider", "manufacturer"
        ]))
        ai_deployer = _norm_str(_first_present(row, [
            "identity_of_ai_deployer", "deployer", "operator", "implementer", "user", "agency", "organization_deployer"
        ]))
        ai_deployer_location = _norm_str(_first_present(row, [
            "location_of_ai_deployer", "deployer_location", "operator_location", "location_deployer"
        ]))
        ai_subject = _norm_str(_first_present(row, [
            "identity_of_ai_subject", "ai_subject", "subject", "data_subject", "affected_party", "individual", "group"
        ]))
        ai_subject_location = _norm_str(_first_present(row, [
            "location_of_ai_subject", "subject_location", "location_subject", "where"
        ]))
        date_time = _norm_str(_first_present(row, [
            "date_and_time_of_event", "date___time_of_event", "datetime", "date_time", "date", "time", "event_time", "when"
        ]))

        return user_template_risks_benefits.format(
                domain,
                purpose,
                capability,
                ai_developer,
                ai_deployer,
                ai_deployer_location,
                ai_subject,
                ai_subject_location,
                date_time,
            )


    # Tokenizer-based trimming helpers
    def _get_tokenizer(model_source: str):
        try:
            from transformers import AutoTokenizer  # type: ignore
            return AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)
        except Exception:
            return None

    _TOK_CACHED = None  # lazy per-process tokenizer cache
    def _get_tokenizer_cached(model_source: str):
        nonlocal _TOK_CACHED
        try:
            if _TOK_CACHED is None:
                _TOK_CACHED = _get_tokenizer(model_source)
            return _TOK_CACHED
        except Exception:
            return None

    def _get_max_user_input_tokens(tokenizer, system_text: str) -> int:
        try:
            system_tokens = len(tokenizer.encode(system_text, add_special_tokens=False)) if tokenizer else 0
            max_model_len = int(getattr(cfg.model, "engine_kwargs", {}).get("max_model_len", 4096))
            try:
                sp = getattr(cfg, "sampling_params_classify", getattr(cfg, "sampling_params", {}))
                if hasattr(sp, "max_tokens"):
                    max_output = int(getattr(sp, "max_tokens"))
                else:
                    max_output = int((sp or {}).get("max_tokens", 8))
            except Exception:
                max_output = 8
            safety = 512
            return max(512, max_model_len - max_output - system_tokens - safety)
        except Exception:
            return 2048

    def _trim_text_for_prompt(text: str, tokenizer, system_text: str) -> str:
        # Tokenizer-aware trimming when available; otherwise conservative char-based fallback
        if tokenizer:
            try:
                ids = tokenizer.encode(text or "", add_special_tokens=False)
                max_user = _get_max_user_input_tokens(tokenizer, system_text)
                if len(ids) <= max_user:
                    return text
                ids = ids[:max_user]
                return tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                pass
        # Fallback: approximate 4 chars per token budget
        try:
            max_user = _get_max_user_input_tokens(tokenizer, system_text)
        except Exception:
            max_user = 2048
        approx_chars_per_token = 4
        max_chars = int(max_user) * approx_chars_per_token
        try:
            return text if len(text or "") <= max_chars else str(text or "")[:max_chars]
        except Exception:
            return text

    def _extract_last_json(text: str):
        """Extract the last JSON object from text, if any.

        Tries direct parse first; otherwise finds brace-delimited snippets and
        returns the last one that parses as a JSON object.
        """
        try:
            if not isinstance(text, str) or not text.strip():
                return None
            s = text
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            try:
                import re as _re
                snippets = _re.findall(r"\{[\s\S]*\}", s)
                for snip in reversed(snippets or []):
                    try:
                        obj = json.loads(snip)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        continue
            except Exception:
                pass
            return None
        except Exception:
            return None

    # Constrain GPU mem via vLLM engine args: prefer provided config; otherwise set conservative defaults
    ek = dict(getattr(cfg.model, "engine_kwargs", {}))
    ek.setdefault("max_model_len", 4096)
    ek.setdefault("max_num_seqs", 16)
    ek.setdefault("gpu_memory_utilization", 0.85)
    tp_env = os.environ.get("UAIR_TENSOR_PARALLEL_SIZE")
    if "tensor_parallel_size" not in ek and tp_env:
        try:
            tp_val = max(1, int(tp_env))
            ek["tensor_parallel_size"] = tp_val
            if not os.environ.get("RULE_TUPLES_SILENT"):
                print(f"Using tensor_parallel_size={tp_val} from UAIR_TENSOR_PARALLEL_SIZE")
        except Exception:
            pass
    # Auto-detect tensor_parallel_size from allocated GPUs if not explicitly set
    if "tensor_parallel_size" not in ek:
        try:
            num_gpus = _detect_num_gpus()
            ek["tensor_parallel_size"] = num_gpus
            if not os.environ.get("RULE_TUPLES_SILENT"):
                print(f"Auto-detected {num_gpus} GPU(s) for tensor parallelism")
        except Exception:
            ek.setdefault("tensor_parallel_size", 1)
    # Apply GPU-type-aware batch settings (max_num_seqs and batch_size)
    gpu_settings = _apply_gpu_aware_batch_settings(ek, cfg)
    # vLLM best-practice safe defaults (overridable via config)
    ek.setdefault("enable_prefix_caching", True)
    ek.setdefault("use_v2_block_manager", True)
    ek.setdefault("tokenizer_mode", "auto")
    ek.setdefault("trust_remote_code", True)
    ek.setdefault("dtype", "auto")
    ek.setdefault("kv_cache_dtype", "auto")
    ek = _filter_vllm_engine_kwargs(ek)
    # Prefer xgrammar backend for stricter JSON Schema enforcement (align with decompose_nbl)
    ek.setdefault("guided_decoding_backend", "xgrammar")
    # Determine batch_size: explicit config > GPU-aware default > fallback
    try:
        batch_size_cfg = getattr(cfg.model, "batch_size", None)
        if batch_size_cfg is not None:
            batch_size = int(batch_size_cfg)
        elif gpu_settings and "batch_size" in gpu_settings:
            batch_size = gpu_settings["batch_size"]
            if not os.environ.get("RULE_TUPLES_SILENT"):
                print(f"Auto-set batch_size={batch_size} for {_detect_gpu_type()}")
        else:
            batch_size = 16
    except Exception:
        batch_size = 16
    engine_config = vLLMEngineProcessorConfig(
        model_source=str(getattr(cfg.model, "model_source")),
        runtime_env={
            "env_vars": {
                # Doc-supported knob to reduce verbosity safely
                "VLLM_LOGGING_LEVEL": str(os.environ.get("VLLM_LOGGING_LEVEL", "WARNING")),
                # Propagate wandb config to Ray workers (uses in-process mode)
                "WANDB_DISABLE_SERVICE": str(os.environ.get("WANDB_DISABLE_SERVICE", "true")),
                "WANDB_SILENT": str(os.environ.get("WANDB_SILENT", "true")),
            }
        },
        engine_kwargs=ek,
        concurrency=int(getattr(cfg.model, "concurrency", 1) or 1),
        batch_size=int(batch_size),
    )

    # Prefer stage/profile-specific sampling params when present; convert nested DictConfig -> dict
    try:
        if is_eu_profile:
            sp_src = getattr(cfg, "sampling_params_eu_ai_act", getattr(cfg, "sampling_params_classify", getattr(cfg, "sampling_params", {})))
        elif is_risks_benefits_profile:
            sp_src = getattr(cfg, "sampling_params_risks_and_benefits", getattr(cfg, "sampling_params_classify", getattr(cfg, "sampling_params", {})))
        else:
            sp_src = getattr(cfg, "sampling_params_classify", getattr(cfg, "sampling_params", {}))
        sampling_params = OmegaConf.to_container(sp_src, resolve=True) if isinstance(sp_src, (dict,)) or hasattr(sp_src, "_get_node") else dict(sp_src)
    except Exception:
        sampling_params = dict(getattr(cfg, "sampling_params", {}))
    # Defaults: EU and risks_benefits need larger output; relevance keeps short YES/NO (or rationales)
    try:
        if is_eu_profile:
            try:
                eu_default = int(getattr(cfg.runtime, "eu_max_tokens_default", 512) or 512)
            except Exception:
                eu_default = 512
            sampling_params.setdefault("max_tokens", eu_default)
        elif is_risks_benefits_profile:
            try:
                rb_default = int(getattr(cfg.runtime, "risks_benefits_max_tokens_default", 2048) or 2048)
            except Exception:
                rb_default = 2048
            sampling_params.setdefault("max_tokens", rb_default)
        else:
            if bool(getattr(cfg.runtime, "log_rationales", False)):
                sampling_params.setdefault("max_tokens", 32)
            else:
                sampling_params.setdefault("max_tokens", 8)
    except Exception:
        sampling_params.setdefault("max_tokens", 8)

    # JSON schema for EU AI Act structured decoding (ensure valid JSON and constrained label)
    EU_GUIDED_JSON_SCHEMA = None
    if is_eu_profile:
        try:
            EU_GUIDED_JSON_SCHEMA = {
                "type": "object",
                "properties": {
                    "Description": {"type": "string"},
                    "Classification": {"enum": ["Prohibited", "High Risk", "Limited or Low Risk"]},
                    "Relevant Text from the EU AI Act": {"type": "string"},
                    "Reasoning": {"type": "string"},
                },
                "required": ["Description", "Classification", "Relevant Text from the EU AI Act", "Reasoning"],
                "additional_properties": False,
            }
        except Exception:
            EU_GUIDED_JSON_SCHEMA = None

    # JSON schema for Risks and Benefits structured decoding (complex nested structure)
    RISKS_BENEFITS_GUIDED_JSON_SCHEMA = None
    if is_risks_benefits_profile:
        try:
            RISKS_BENEFITS_GUIDED_JSON_SCHEMA = {
                "type": "object",
                "properties": {
                    "Description": {"type": "string"},
                    "Assessment of impact on Human Rights": {
                        "type": "object",
                        "properties": {
                            "Positive Impacts": {"type": "array", "items": {"type": "object"}},
                            "Negative Impacts": {"type": "array", "items": {"type": "object"}},
                        },
                    },
                    "Assessment of impact on Sustainable Development Goals": {
                        "type": "object",
                        "properties": {
                            "Positive Impacts": {"type": "array", "items": {"type": "object"}},
                            "Negative Impacts": {"type": "array", "items": {"type": "object"}},
                        },
                    },
                    "Assessment of additional impacts": {
                        "type": "object",
                        "properties": {
                            "Positive Impacts": {"type": "array", "items": {"type": "object"}},
                            "Negative Impacts": {"type": "array", "items": {"type": "object"}},
                        },
                    },
                },
                "required": ["Description"],
                "additional_properties": False,
            }
        except Exception:
            RISKS_BENEFITS_GUIDED_JSON_SCHEMA = None

    # Driver-computed conservative char budget for user content (captured into Ray workers)
    try:
        mm_len = int(getattr(cfg.model, "engine_kwargs", {}).get("max_model_len", 4096))
    except Exception:
        mm_len = 4096
    try:
        max_out = int(sampling_params.get("max_tokens", 8) or 8)
    except Exception:
        max_out = 8
    _approx_user_char_budget = max(2048, (mm_len - max_out - 512) * 4)

    # Columns to drop from final stage outputs (internal Ray/vLLM mechanics)
    _INTERNAL_DROP_COLS = [
        "messages",
        "sampling_params",
        "usage",
        "token_counts",
        "generated_text",
        "json",
        "guided_decoding",
        "response_format",
        "structured_output",
        "prompt",
        "prompt_token_ids",
        "request_id",
        "params",
        "llm_json",
        "generated_tokens",
        "num_generated_tokens",
        "num_input_tokens",
        "time_taken_llm",
        "embeddings",
    ]
    
    # Additional columns to drop for risks_and_benefits profile (keep only structured fields)
    # Note: rb_raw_json is no longer created, so no need to drop it
    _RISKS_BENEFITS_DROP_COLS = []

    def _attach_chunk_text(row: Dict[str, Any]) -> Dict[str, Any]:
        if not enable_kw_buf or kw_regex is None:
            return row
        text_val = row.get("article_text")
        try:
            blocks = _generate_relevant_blocks(text_val, kw_regex, window_words)
        except Exception:
            blocks = []
        if blocks:
            row["chunk_text"] = "\n\n".join(blocks)
        else:
            row.setdefault("chunk_text", str(text_val or ""))
        return row

    # Normalize NA/blank inputs for EU and risks_benefits fields so ChatTemplate sees consistent strings
    def _normalize_na_blanks(r: Dict[str, Any]) -> Dict[str, Any]:
        if not is_eu_profile and not is_risks_benefits_profile:
            # Ensure article_text/chunk_text are strings at minimum
            try:
                r["article_text"] = str(r.get("article_text") or "")
            except Exception:
                pass
            try:
                if r.get("chunk_text") is not None:
                    r["chunk_text"] = str(r.get("chunk_text") or "")
            except Exception:
                pass
            return _coerce_boolish_row(r)

        eu_keys = [
            "deployment_domain",
            "deployment_purpose",
            "deployment_capability",
            "identity_of_ai_developer",
            "identity_of_ai_deployer",
            "location_of_ai_deployer",
            "identity_of_ai_subject",
            "location_of_ai_subject",
            "date_and_time_of_event",
            "date___time_of_event",
        ]
        for k in eu_keys:
            try:
                v = r.get(k)
                s = _norm_str(v)
                r[k] = s
            except Exception:
                try:
                    r[k] = _EU_MISSING_PH
                except Exception:
                    pass
        # Ensure article_text/chunk_text are strings
        try:
            r["article_text"] = str(r.get("article_text") or "")
        except Exception:
            pass
        try:
            if r.get("chunk_text") is not None:
                r["chunk_text"] = str(r.get("chunk_text") or "")
        except Exception:
            pass
        return _coerce_boolish_row(r)

    # EU input completeness flags
    # Authoritative count of expected EU input fields (date field may appear under either of two names)
    _EU_TOTAL_INPUTS = 9

    def _is_valid_eu_value(v: Any) -> bool:
        try:
            s = _norm_str(v)
            return bool(s) and s != _EU_MISSING_PH
        except Exception:
            return False

    def _add_eu_vague_flags(r: Dict[str, Any]) -> Dict[str, Any]:
        if not is_eu_profile and not is_risks_benefits_profile:
            return r
        cnt = 0
        for k in EU_INPUT_KEYS:
            try:
                if _is_valid_eu_value(r.get(k)):
                    cnt += 1
            except Exception:
                continue
        # If decompose provided a 'missing' column (possibly as a stringified list), use it as a stronger signal
        # Valid count from 'missing' is TOTAL_INPUTS - len(missing)
        try:
            miss_candidates = ("missing", "missing_fields", "eu_missing_fields")
            missing_list = None
            for mk in miss_candidates:
                if mk in r and r.get(mk) is not None:
                    mv = r.get(mk)
                    # Already a list/tuple
                    if isinstance(mv, (list, tuple, set)):
                        missing_list = list(mv)
                        break
                    # Try JSON then AST literal_eval
                    if isinstance(mv, str):
                        sv = mv.strip()
                        if sv:
                            try:
                                import json as _json
                                if sv.startswith("[") or sv.startswith("{"):
                                    parsed = _json.loads(sv)
                                else:
                                    import ast as _ast
                                    parsed = _ast.literal_eval(sv)
                            except Exception:
                                # Fallback: naive comma-split
                                body = sv.strip().strip("[](){}")
                                parsed = [x.strip() for x in body.split(",") if x.strip() != ""]
                            if isinstance(parsed, (list, tuple, set)):
                                missing_list = list(parsed)
                                break
            if missing_list is not None:
                try:
                    miss_cnt = int(len(missing_list))
                except Exception:
                    miss_cnt = 0
                # Clamp to [0, _EU_TOTAL_INPUTS]
                if miss_cnt < 0:
                    miss_cnt = 0
                if miss_cnt > _EU_TOTAL_INPUTS:
                    miss_cnt = _EU_TOTAL_INPUTS
                cnt_from_missing = max(0, _EU_TOTAL_INPUTS - miss_cnt)
                # Use the stricter interpretation: take the minimum valid count
                cnt = min(int(cnt), int(cnt_from_missing))
        except Exception:
            pass
        r["eu_valid_input_count"] = int(cnt)
        r["too_vague_to_process"] = bool(int(cnt) < 5)
        return r

    # Stable baseline for sampling params to avoid Arrow struct field drift across batches
    _SAMPLING_DEFAULTS: Dict[str, Any] = {
        # Core generation controls
        "max_tokens": 8,               # int
        "temperature": 1.0,            # float
        "top_p": 1.0,                  # float
        "top_k": -1,                   # int (-1 disables)
        "min_p": 0.0,                  # float
        # Penalties
        "presence_penalty": 0.0,       # float
        "frequency_penalty": 0.0,      # float
        "repetition_penalty": 1.0,     # float
        "length_penalty": 1.0,         # float (beam search)
        # Decoding strategy
        "use_beam_search": False,      # bool
        "num_beams": 1,                # int
        "early_stopping": False,       # bool
        # Output formatting/logprobs
        "detokenize": True,            # bool
        "logprobs": 0,                 # int
        # Multi-sample
        "n": 1,                        # int
        "best_of": 1,                  # int
        # Stopping conditions
        "stop": [],                    # list[str]
        "stop_token_ids": [],          # list[int]
        # EOS handling
        "ignore_eos_token": False,     # bool
    }

    def _filter_vllm_sampling_params(sp: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sampling params to only those supported by installed vLLM.

        Uses introspection of vllm.SamplingParams signature; if that fails,
        conservatively drops known problematic/optional keys.
        """
        try:
            import inspect as _inspect
            import vllm as _v
            try:
                sig = _inspect.signature(_v.SamplingParams.__init__)
                accepted = {k for k in sig.parameters.keys() if k != "self"}
            except Exception:
                accepted = None
            if accepted:
                filtered = {k: v for k, v in sp.items() if k in accepted}
                if "guided_decoding" in sp and "guided_decoding" not in filtered:
                    filtered["guided_decoding"] = sp["guided_decoding"]
                    if not os.environ.get("RULE_TUPLES_SILENT"):
                        try:
                            print("[classify] Warning: guided_decoding not in SamplingParams signature; preserving manually", flush=True)
                        except Exception:
                            pass
                # Debug print of dropped keys (best-effort)
                try:
                    dropped = [k for k in sp.keys() if k not in filtered]
                    if dropped and not os.environ.get("RULE_TUPLES_SILENT"):
                        print(f"Filtering unsupported vLLM sampling params: {dropped}")
                except Exception:
                    pass
                return filtered
        except Exception:
            pass
        # Conservative fallback: drop keys commonly missing in older vLLM
        sp2 = dict(sp)
        for k in (
            "early_stopping",
            "length_penalty",
            "response_format",
            "structured_output",
        ):
            sp2.pop(k, None)
        # Ensure guided_decoding survives fallback path
        if "guided_decoding" in sp:
            sp2["guided_decoding"] = sp["guided_decoding"]
        return sp2

    # Ensure sampling_params have a stable schema across rows to avoid Arrow concat issues
    def _stabilize_sampling_params(sp: Dict[str, Any]) -> Dict[str, Any]:
        try:
            out = dict(_SAMPLING_DEFAULTS)
            out.update(sp or {})
            stop_val = out.get("stop")
            if stop_val is None:
                out["stop"] = []
            elif not isinstance(stop_val, list):
                out["stop"] = [str(stop_val)]
            for key, value in out.items():
                if isinstance(_SAMPLING_DEFAULTS.get(key), bool) and value is None:
                    out[key] = bool(_SAMPLING_DEFAULTS.get(key))
            # Finally, drop any keys not supported by the installed vLLM
            return _filter_vllm_sampling_params(out)
        except Exception:
            return dict(sp or {})

    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        _maybe_silence_vllm_logs()
        global _DEBUG_GUIDED_LOG
        # Ensure article_id is present on every row
        try:
            import hashlib as _hash
            r = dict(row)
            aid = r.get("article_id")
            if not (isinstance(aid, str) and aid.strip() != ""):
                src = r.get("article_path")
                if not isinstance(src, str) or src.strip() == "":
                    src = r.get("article_text") or r.get("chunk_text") or ""
                r["article_id"] = _hash.sha1(str(src).encode("utf-8")).hexdigest()
            row = r
        except Exception:
            pass
        if enable_kw_buf and kw_regex is not None:
            row = _attach_chunk_text(dict(row))
        # Normalize any NA/blank values before building messages
        row = _normalize_na_blanks(dict(row))
        # Construct user content and perform token-aware trimming to model context
        if is_eu_profile:
            user_raw = _format_user_eu(row)
        elif is_risks_benefits_profile:
            user_raw = _format_user_risks_benefits(row)
        else:
            user_raw = _format_user(row.get("article_text"), row)
        tok_local = _get_tokenizer_cached(str(getattr(cfg.model, "model_source", "")))
        # First, ensure chunk_text itself is trimmed defensively
        try:
            txt0 = row.get("chunk_text") or row.get("article_text")
            row["chunk_text"] = _trim_text_for_prompt(str(txt0 or ""), tok_local, system_prompt)
        except Exception:
            pass
        user = _trim_text_for_prompt(user_raw, tok_local, system_prompt)
        # Sanitize sampling params; then stabilize to a fixed-key schema
        sp_local = _sanitize_for_json(dict(sampling_params or {}))
        if is_eu_profile and EU_GUIDED_JSON_SCHEMA is not None:
            # Always include guided_decoding payload to keep Arrow schema stable across blocks
            try:
                guided_params = GuidedDecodingParams(
                    json=EU_GUIDED_JSON_SCHEMA,
                    disable_fallback=True,
                    disable_additional_properties=True,
                )
                sp_local["guided_decoding"] = guided_params
                if is_eu_profile and _DEBUG_GUIDED_LOG["pre"] < 5:
                    try:
                        print(
                            "[classify][eu_ai_act] Injected GuidedDecodingParams into sampling params "
                            f"(type={type(guided_params)} keys={list(sp_local.keys())})",
                            flush=True,
                        )
                    except Exception:
                        pass
                    _DEBUG_GUIDED_LOG["pre"] += 1
            except Exception as exc:
                try:
                    print(
                        f"[classify][eu_ai_act] Failed to build GuidedDecodingParams: {exc}",
                        flush=True,
                    )
                except Exception:
                    pass
                try:
                    schema_json_str = json.dumps(EU_GUIDED_JSON_SCHEMA, ensure_ascii=False)
                except Exception:
                    schema_json_str = ""
                sp_local["guided_decoding"] = {"json": schema_json_str or ""}
        elif is_risks_benefits_profile and RISKS_BENEFITS_GUIDED_JSON_SCHEMA is not None:
            # Always include guided_decoding payload to keep Arrow schema stable across blocks
            try:
                guided_params = GuidedDecodingParams(
                    json=RISKS_BENEFITS_GUIDED_JSON_SCHEMA,
                    disable_fallback=True,
                    disable_additional_properties=True,
                )
                sp_local["guided_decoding"] = guided_params
                if is_risks_benefits_profile and _DEBUG_GUIDED_LOG["pre"] < 5:
                    try:
                        print(
                            "[classify][risks_benefits] Injected GuidedDecodingParams into sampling params "
                            f"(type={type(guided_params)} keys={list(sp_local.keys())})",
                            flush=True,
                        )
                    except Exception:
                        pass
                    _DEBUG_GUIDED_LOG["pre"] += 1
            except Exception as exc:
                try:
                    print(
                        f"[classify][risks_benefits] Failed to build GuidedDecodingParams: {exc}",
                        flush=True,
                    )
                except Exception:
                    pass
                try:
                    schema_json_str = json.dumps(RISKS_BENEFITS_GUIDED_JSON_SCHEMA, ensure_ascii=False)
                except Exception:
                    schema_json_str = ""
                sp_local["guided_decoding"] = {"json": schema_json_str or ""}
        # Stabilize schema to avoid Arrow concat errors across batches
        sp_local = _stabilize_sampling_params(sp_local)
        from datetime import datetime as _dt
        # Drop any conflicting keys from input row to avoid clobbering our constructed fields
        base = {k: v for k, v in row.items() if k not in {"messages", "sampling_params", "generated_text", "llm_output", "json", "guided_decoding", "response_format", "structured_output"}}
        return {
            **base,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            "sampling_params": sp_local,
            "ts_start": _dt.now().timestamp(),
        }

    def _post(row: Dict[str, Any]) -> Dict[str, Any]:
        from datetime import datetime as _dt
        global _DEBUG_GUIDED_LOG
        ts_end = _dt.now().timestamp()
        usage = row.get("usage") or row.get("token_counts") or None
        if is_risks_benefits_profile:
            raw = str(row.get("generated_text") or "").strip()
            # Robust JSON extraction similar to decompose stage
            parsed = _extract_last_json(raw)
            # Extract main fields and immediately serialize to JSON strings to avoid Arrow schema conflicts
            rb_desc = None
            rb_human_rights = None
            rb_sdgs = None
            rb_additional = None
            if isinstance(parsed, dict):
                rb_desc = parsed.get("Description")
                # Serialize nested structures immediately to JSON strings for Arrow compatibility
                # Variable-length arrays cause schema conflicts, so we convert to strings
                try:
                    hr = parsed.get("Assessment of impact on Human Rights")
                    rb_human_rights = _to_json_str(hr) if hr is not None else None
                except Exception:
                    rb_human_rights = None
                try:
                    sdgs = parsed.get("Assessment of impact on Sustainable Development Goals")
                    rb_sdgs = _to_json_str(sdgs) if sdgs is not None else None
                except Exception:
                    rb_sdgs = None
                try:
                    additional = parsed.get("Assessment of additional impacts")
                    rb_additional = _to_json_str(additional) if additional is not None else None
                except Exception:
                    rb_additional = None
            # Serialize other nested columns for Arrow/Parquet compatibility
            try:
                serialize_nested = bool(getattr(cfg.runtime, "serialize_nested_json", True))
            except Exception:
                serialize_nested = True
            if serialize_nested:
                _serialize_arrow_unfriendly_in_row(row, [
                    "messages",
                    "sampling_params",
                    "usage",
                    "token_counts",
                    "matched_keywords",
                    # Include potential engine outputs that may appear as structs
                    "json",
                    "structured_output",
                    "guided_decoding",
                    "response_format",
                ])
            return {
                **row,
                "rb_desc": rb_desc,
                "rb_human_rights": rb_human_rights,
                "rb_sdgs": rb_sdgs,
                "rb_additional": rb_additional,
                "classification_mode": "risks_and_benefits",
                "too_vague_to_process": False,  # LLM-processed rows are not too vague
                "latency_s": (float(ts_end) - float(row.get("ts_start", ts_end))),
                "token_usage_prompt": ((usage or {}).get("prompt_tokens") or (usage or {}).get("input_tokens")),
                "token_usage_output": ((usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens")),
                "token_usage_total": ((usage or {}).get("total_tokens")),
                "_progress_row": 1,
            }
        elif is_eu_profile:
            raw = str(row.get("generated_text") or "").strip()
            # Robust JSON extraction similar to decompose stage
            parsed = _extract_last_json(raw)
            # Extract fields with robustness to variations
            eu_desc = None
            eu_label = None
            eu_reltext = None
            eu_reason = None
            if isinstance(parsed, dict):
                eu_desc = parsed.get("Description")
                lab = parsed.get("Classification")
                if isinstance(lab, list) and lab:
                    eu_label = lab[0]
                elif isinstance(lab, str):
                    eu_label = lab
                eu_reltext = parsed.get("Relevant Text from the EU AI Act")
                eu_reason = parsed.get("Reasoning")
            else:
                snippet = raw.replace("\n", " ").strip()
                if snippet:
                    snippet = snippet[:400]
                try:
                    print(
                        "[classify][eu_ai_act] Failed to parse LLM JSON output; raw snippet: "
                        f"{snippet}",
                        flush=True,
                    )
                    if _DEBUG_GUIDED_LOG["fail"] < 5:
                        print(
                            "[classify][eu_ai_act][debug] Full generated_text dump due to parse failure:\n"
                            f"{raw}",
                            flush=True,
                    )
                        _DEBUG_GUIDED_LOG["fail"] += 1
                except Exception:
                    pass
            # Optionally serialize nested columns for Arrow/Parquet compatibility
            try:
                serialize_nested = bool(getattr(cfg.runtime, "serialize_nested_json", True))
            except Exception:
                serialize_nested = True
            if serialize_nested:
                _serialize_arrow_unfriendly_in_row(row, [
                    "messages",
                    "sampling_params",
                    "usage",
                    "token_counts",
                    "matched_keywords",
                    # Include potential engine outputs that may appear as structs
                    "json",
                    "structured_output",
                    "guided_decoding",
                    "response_format",
                ])
            # Serialize raw parsed JSON explicitly to string to avoid struct schema drift
            try:
                eu_ai_raw_json_str = _to_json_str(parsed)
            except Exception:
                eu_ai_raw_json_str = None
            return {
                **row,
                "eu_ai_desc": eu_desc,
                "eu_ai_label": eu_label,
                "eu_ai_relevant_text": eu_reltext,
                "eu_ai_reason": eu_reason,
                "eu_ai_raw_json": eu_ai_raw_json_str,
                "llm_output": row.get("generated_text"),
                "classification_mode": "eu_ai_act",
                "too_vague_to_process": False,  # LLM-processed rows are not too vague
                "latency_s": (float(ts_end) - float(row.get("ts_start", ts_end))),
                "token_usage_prompt": ((usage or {}).get("prompt_tokens") or (usage or {}).get("input_tokens")),
                "token_usage_output": ((usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens")),
                "token_usage_total": ((usage or {}).get("total_tokens")),
                "_progress_row": 1,
            }
        else:
            text = str(row.get("generated_text") or "").strip().upper()
            is_rel = text.startswith("YES") or ("YES" in text and "NO" not in text)
            # Optionally serialize nested columns for Arrow/Parquet compatibility
            try:
                serialize_nested = bool(getattr(cfg.runtime, "serialize_nested_json", True))
            except Exception:
                serialize_nested = True
            if serialize_nested:
                _serialize_arrow_unfriendly_in_row(row, [
                    "messages",
                    "sampling_params",
                    "usage",
                    "token_counts",
                    "matched_keywords",  # List of matched keyword strings
                    # Include potential engine outputs that may appear as structs
                    "json",
                    "structured_output",
                    "guided_decoding",
                    "response_format",
                ])
            return {
                **row,
                "relevance_answer": row.get("generated_text"),
                "is_relevant": bool(is_rel),
                "llm_output": row.get("generated_text"),
                "classification_mode": "llm_relevance",
                "latency_s": (float(ts_end) - float(row.get("ts_start", ts_end))),
                "token_usage_prompt": ((usage or {}).get("prompt_tokens") or (usage or {}).get("input_tokens")),
                "token_usage_output": ((usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens")),
                "token_usage_total": ((usage or {}).get("total_tokens")),
                # Signal a single-row progress marker for downstream logging
                "_progress_row": 1,
            }

    # Pre-attach chunk_text in a separate map when streaming; apply prefilter gating and trimming
    if is_ray_ds:
        ds_in = df
        ds_all = None  # Track ALL articles (including filtered ones)
        
        # Debug: inspect incoming schema before any pruning to track problematic columns
        try:
            schema = ds_in.schema()
            cols = list(schema.names)
            print(f"[classify] Incoming Ray dataset columns ({len(cols)}): {cols}", flush=True)
            try:
                import pyarrow as pa  # type: ignore
                schema_desc = [
                    f"{field.name}:{field.type}" if isinstance(field, pa.Field) else str(field)
                    for field in schema
                ]
                print(
                    "[classify] Incoming schema field types: " + ", ".join(schema_desc),
                    flush=True,
                )
            except Exception:
                pass
        except Exception:
            pass

        # Limit columns to those required for prompting/metadata to avoid schema drift
        try:
            current_cols = set(ds_in.schema().names)
        except Exception:
            current_cols = set()
        base_required = {
            "article_id",
            "article_text",
            "article_path",
            "chunk_text",
            "country",
            "year",
            "relevant_keyword",
            "matched_keywords",
            "keyword_match_count",
            "core_tuple_verified",
            "doc_any_component_verified",
            "missing",
            "missing_fields",
            "eu_missing_fields",
        }
        profile_specific = set()
        if is_eu_profile or is_risks_benefits_profile:
            profile_specific.update(EU_INPUT_KEYS)
        allowed_columns = get_required_input_columns(is_eu_profile, is_risks_benefits_profile)
        allowed_columns = allowed_columns & current_cols
        # Ensure essential columns are retained even if missing from current columns
        allowed_columns.update({col for col in ("article_id", "article_text") if col in current_cols})
        try:
            if allowed_columns and allowed_columns != current_cols:
                ds_in = ds_in.select_columns(sorted(allowed_columns))
                print(
                    f"[classify] Pruned dataset columns to {sorted(allowed_columns)}",
                    flush=True,
                )
        except Exception:
            pass

        # After pruning, log the updated schema for confirmation
        try:
            new_schema = ds_in.schema()
            new_cols = list(new_schema.names)
            print(f"[classify] Columns after pruning ({len(new_cols)}): {new_cols}", flush=True)
            try:
                import pyarrow as pa  # type: ignore
                new_schema_desc = [
                    f"{field.name}:{field.type}" if isinstance(field, pa.Field) else str(field)
                    for field in new_schema
                ]
                print(
                    "[classify] Post-prune schema field types: "
                    + ", ".join(new_schema_desc),
                    flush=True,
                )
            except Exception:
                pass
        except Exception:
            pass

        # Ensure every row has an article_id BEFORE any materialization/merging
        def _ensure_article_id_map(r: Dict[str, Any]) -> Dict[str, Any]:
            try:
                aid = r.get("article_id")
                if not (isinstance(aid, str) and aid.strip() != ""):
                    src = r.get("article_path")
                    if not isinstance(src, str) or src.strip() == "":
                        src = r.get("article_text") or r.get("chunk_text") or ""
                    import hashlib as _hash
                    r["article_id"] = _hash.sha1(str(src).encode("utf-8")).hexdigest()
            except Exception:
                pass
            return r
        try:
            ds_in = ds_in.map(_ensure_article_id_map)
        except Exception:
            pass

        def _coerce_boolish_map(r: Dict[str, Any]) -> Dict[str, Any]:
            return _coerce_boolish_row(dict(r))

        try:
            ds_in = ds_in.map(_coerce_boolish_map)
        except Exception:
            pass

        # Keyword flag + gating
        if prefilter_mode in ("pre_gating", "post_gating"):
            # First mark keyword and tally BEFORE any filter so W&B sees true totals
            usage_actor = None  # type: ignore
            try:
                if _RAY_OK:
                    usage_actor = ray.get_actor("uair_usage_agg")  # type: ignore
            except Exception:
                usage_actor = None  # type: ignore
            # Add keyword information: binary flag, matched keywords list, and match count
            def _add_keyword_info(r: Dict[str, Any]) -> Dict[str, Any]:
                has_kw, matched_kws, match_count = _extract_matched_keywords(r.get("article_text"))
                return {
                    **r,
                    "relevant_keyword": has_kw,
                    "matched_keywords": matched_kws,
                    "keyword_match_count": match_count,
                }
            ds_in = ds_in.map(_add_keyword_info)
            if usage_actor is not None:
                def _acc_kw(pdf: pd.DataFrame) -> pd.DataFrame:
                    try:
                        total_checked = int(len(pdf))
                    except Exception:
                        total_checked = 0
                    try:
                        marked = int(pdf["relevant_keyword"].astype(bool).sum()) if "relevant_keyword" in pdf.columns else 0
                    except Exception:
                        marked = 0
                    try:
                        usage_actor.record.remote("classify", kw_marked=int(marked), kw_checked=int(total_checked))  # type: ignore
                    except Exception:
                        pass
                    return pdf
                try:
                    ds_in = ds_in.map_batches(_acc_kw, batch_format="pandas", batch_size=512)
                except Exception:
                    pass
            if prefilter_mode == "pre_gating":
                # Keep a copy of ALL articles before filtering
                # Must materialize to cache the unfiltered data, otherwise ds_all will reference
                # the same lazy execution plan as ds_in and both will be filtered
                ds_all = ds_in.materialize()
                # Filter to only articles WITH keywords for LLM processing
                ds_in = ds_all.filter(lambda r: bool(r.get("relevant_keyword", True)))
        # EU and risks_benefits: compute input completeness flags and split off too-vague rows before LLM
        ds_vague = None
        if is_eu_profile or is_risks_benefits_profile:
            try:
                ds_in = ds_in.map(_add_eu_vague_flags)
                # Capture too-vague rows and filter them out from LLM-bound dataset
                ds_vague = ds_in.filter(lambda r: bool(r.get("too_vague_to_process", False)))
                ds_in = ds_in.filter(lambda r: not bool(r.get("too_vague_to_process", False)))
                # Debug: count vague rows
                try:
                    vague_count = ds_vague.count()
                    print(f"[classify] Filtered {vague_count} too-vague rows before LLM processing", flush=True)
                except Exception:
                    pass
            except Exception:
                ds_vague = None
        # Attach chunk_text
        if enable_kw_buf and kw_regex is not None:
            ds_in = ds_in.map(_attach_chunk_text)
        # Trim chunk_text to fit model context budget
        sys_text = system_prompt
        def _trim_row(r: Dict[str, Any]) -> Dict[str, Any]:
            txt = r.get("chunk_text") or r.get("article_text")
            # Use char-based trimming to avoid heavy tokenizer in Ray workers.
            r["chunk_text"] = _trim_text_for_prompt(str(txt or ""), None, sys_text)
            return r
        ds_in = ds_in.map(_trim_row)
        processor = build_llm_processor(engine_config, preprocess=_pre, postprocess=_post)
        ds_llm_results = processor(ds_in)
        # Sanitize internal columns at the dataset level
        # Only drop columns that actually exist in the dataset
        try:
            # Get schema to check which columns exist
            schema = ds_llm_results.schema()
            if hasattr(schema, 'names'):
                existing_cols = schema.names
            else:
                # Fallback: try to get from first batch
                try:
                    first_batch = next(iter(ds_llm_results.iter_batches(batch_size=1)))
                    existing_cols = first_batch.column_names if hasattr(first_batch, 'column_names') else []
                except Exception:
                    existing_cols = []
            # Filter to only columns that exist
            cols_to_drop = [col for col in _INTERNAL_DROP_COLS if col in existing_cols]
            if cols_to_drop:
                ds_llm_results = ds_llm_results.drop_columns(cols_to_drop)
        except Exception as e:
            print(f"[classify] Warning: Failed to drop internal columns: {e}", flush=True)
        # Drop risks_benefits specific columns if in that profile
        if is_risks_benefits_profile:
            try:
                # Get schema to check which columns exist
                schema = ds_llm_results.schema()
                if hasattr(schema, 'names'):
                    existing_cols = schema.names
                else:
                    try:
                        first_batch = next(iter(ds_llm_results.iter_batches(batch_size=1)))
                        existing_cols = first_batch.column_names if hasattr(first_batch, 'column_names') else []
                    except Exception:
                        existing_cols = []
                # Filter to only columns that exist
                cols_to_drop = [col for col in _RISKS_BENEFITS_DROP_COLS if col in existing_cols]
                if cols_to_drop:
                    ds_llm_results = ds_llm_results.drop_columns(cols_to_drop)
            except Exception as e:
                print(f"[classify] Warning: Failed to drop risks_benefits columns: {e}", flush=True)
        
        skip_merge_filtered = bool(getattr(cfg.runtime, "skip_merge_filtered", False))
        print(
            f"[classify] Checking merge conditions: ds_all={ds_all is not None}, ds_vague={ds_vague is not None}, skip_merge_filtered={skip_merge_filtered}",
            flush=True,
        )
        if skip_merge_filtered:
            df_llm = ds_llm_results.to_pandas().drop(columns=_INTERNAL_DROP_COLS, errors="ignore")
            if is_risks_benefits_profile:
                df_llm = df_llm.drop(columns=_RISKS_BENEFITS_DROP_COLS, errors="ignore")
            if is_eu_profile or is_risks_benefits_profile:
                if "too_vague_to_process" not in df_llm.columns:
                    df_llm["too_vague_to_process"] = False
                else:
                    df_llm["too_vague_to_process"] = df_llm["too_vague_to_process"].fillna(False)
            print(f"[classify] skip_merge_filtered enabled; returning {len(df_llm)} rows without merging filtered/vague", flush=True)
            return df_llm
        if ds_all is not None or ds_vague is not None:
            print(f"[classify] Starting merge process...", flush=True)
            try:
                # Convert to pandas for easier merging
                df_llm = ds_llm_results.to_pandas()
                df_all = ds_all.to_pandas() if ds_all is not None else None
                
                # Mark filtered articles and ensure consistent columns across profiles
                def _mark_filtered(pdf: pd.DataFrame) -> pd.DataFrame:
                    pdf = pdf.copy()
                    pdf["classification_mode"] = "filtered_by_keyword"
                    if is_eu_profile:
                        # For EU profile, ensure eu_* columns exist with None defaults
                        for col in ("eu_ai_desc", "eu_ai_label", "eu_ai_relevant_text", "eu_ai_reason", "eu_ai_raw_json"):
                            if col not in pdf.columns:
                                pdf[col] = None
                    elif is_risks_benefits_profile:
                        # For risks_benefits profile, ensure rb_* columns exist with None defaults
                        for col in ("rb_desc", "rb_human_rights", "rb_sdgs", "rb_additional"):
                            if col not in pdf.columns:
                                pdf[col] = None
                    else:
                        pdf["is_relevant"] = False
                        if "relevance_answer" not in pdf.columns:
                            pdf["relevance_answer"] = None
                    return pdf
                
                result_parts = [
                    _normalize_profile_columns(df_llm, is_eu_profile, is_risks_benefits_profile)
                ]
                if df_all is not None:
                    processed_ids = set(df_llm["article_id"].unique())
                    # Exclude too-vague IDs from keyword-filtered to avoid duplication when merging back
                    vague_ids: set[str] = set()
                    try:
                        if 'df_vague' in locals() and isinstance(df_vague, pd.DataFrame) and "article_id" in df_vague.columns:
                            vague_ids = set(df_vague["article_id"].unique())
                    except Exception:
                        vague_ids = set()
                    excluded_ids = processed_ids.union(vague_ids)
                    df_filtered = df_all[~df_all["article_id"].isin(list(excluded_ids))].copy()
                    df_filtered["classification_mode"] = "filtered_by_keyword"
                    df_filtered = _normalize_profile_columns(df_filtered, is_eu_profile, is_risks_benefits_profile)
                    if "matched_keywords" in df_filtered.columns:
                        df_filtered["matched_keywords"] = df_filtered["matched_keywords"].apply(_to_json_str)
                    result_parts.append(df_filtered)
                if ds_vague is not None and (is_eu_profile or is_risks_benefits_profile):
                    df_vague = ds_vague.to_pandas().copy()
                    print(f"[classify] Merging back {len(df_vague)} too-vague rows to output", flush=True)
                    df_vague["classification_mode"] = "too_vague_to_process"
                    df_vague["too_vague_to_process"] = True
                    df_vague = _normalize_profile_columns(df_vague, is_eu_profile, is_risks_benefits_profile)
                    result_parts.append(df_vague)
                else:
                    print(f"[classify] No too-vague rows to merge (ds_vague is None: {ds_vague is None})", flush=True)
                result_df = _merge_result_parts(result_parts)
                # Only ensure too_vague column for EU/RB profiles
                if is_eu_profile or is_risks_benefits_profile:
                    if "too_vague_to_process" not in result_df.columns:
                        result_df["too_vague_to_process"] = False
                    else:
                        result_df["too_vague_to_process"] = result_df["too_vague_to_process"].fillna(False)
                # Drop internal columns before returning
                try:
                    result_df = result_df.drop(columns=_INTERNAL_DROP_COLS, errors="ignore")
                except Exception:
                    pass
                # Drop risks_benefits specific columns if in that profile
                if is_risks_benefits_profile:
                    try:
                        result_df = result_df.drop(columns=_RISKS_BENEFITS_DROP_COLS, errors="ignore")
                    except Exception:
                        pass
                
                try:
                    num_kw = len(df_filtered) if 'df_filtered' in locals() else 0
                except Exception:
                    num_kw = 0
                try:
                    num_vague = len(df_vague) if 'df_vague' in locals() else 0
                except Exception:
                    num_vague = 0
                print(f"[classify] Merged results: {len(df_llm)} LLM-processed + {num_kw} keyword-filtered + {num_vague} too-vague = {len(result_df)} total", flush=True)
                result_df = _prune_result_columns(result_df, is_eu_profile, is_risks_benefits_profile)
                return result_df
            except Exception as e:
                print(f"[classify] ERROR: Failed to merge filtered articles back into results: {e}", flush=True)
                import traceback
                traceback.print_exc()
                # Fall back to just LLM results
                return ds_llm_results
        
        print(f"[classify] No merge needed, returning LLM results directly", flush=True)
        return ds_llm_results

    # Pandas path: build Ray Dataset directly and push prefilter/trim into maps
    # Build a Ray Dataset with multiple blocks to enable true streaming.
    # A single huge block (default from_pandas behavior) forces upstream UDFs
    # like ChatTemplateUDF/TokenizeUDF to process the entire dataset before
    # the vLLM stage sees any rows, causing large startup latency.
    try:
        total_rows = int(len(out))
    except Exception:
        total_rows = 0
    try:
        desired_rows_per_block = int(getattr(cfg.runtime, "rows_per_block", 2000) or 2000)
    except Exception:
        desired_rows_per_block = 2000
    step = max(1, desired_rows_per_block)
    if total_rows > step:
        try:
            dfs_list = [out.iloc[i:i + step] for i in range(0, total_rows, step)]
            ds = ray.data.from_pandas(dfs_list)
        except Exception:
            ds = ray.data.from_pandas(out)
    else:
        ds = ray.data.from_pandas(out)

    # Apply keyword flag + pre-gating and trimming in Ray maps (same as streaming path)
    ds_in = ds

    # Debug instrumentation for pandas path: capture schema prior to pruning
    try:
        schema = ds_in.schema()
        cols = list(schema.names)
        print(f"[classify] (pandas path) incoming dataset columns ({len(cols)}): {cols}", flush=True)
        try:
            import pyarrow as pa  # type: ignore
            schema_desc = [
                f"{field.name}:{field.type}" if isinstance(field, pa.Field) else str(field)
                for field in schema
            ]
            print(
                "[classify] (pandas path) incoming schema field types: "
                + ", ".join(schema_desc),
                flush=True,
            )
        except Exception:
            pass
    except Exception:
        cols = []
        schema = None

    try:
        current_cols = set(schema.names) if schema else set()
    except Exception:
        current_cols = set()
    allowed_columns = get_required_input_columns(is_eu_profile, is_risks_benefits_profile)
    allowed_columns = allowed_columns & current_cols
    allowed_columns.update({col for col in ("article_id", "article_text") if col in current_cols})
    try:
        if allowed_columns and allowed_columns != current_cols:
            ds_in = ds_in.select_columns(sorted(allowed_columns))
            print(
                f"[classify] (pandas path) pruned dataset columns to {sorted(allowed_columns)}",
                flush=True,
            )
            try:
                new_schema = ds_in.schema()
                new_cols = list(new_schema.names)
                print(
                    f"[classify] (pandas path) columns after pruning ({len(new_cols)}): {new_cols}",
                    flush=True,
                )
                try:
                    import pyarrow as pa  # type: ignore
                    new_schema_desc = [
                        f"{field.name}:{field.type}" if isinstance(field, pa.Field) else str(field)
                        for field in new_schema
                    ]
                    print(
                        "[classify] (pandas path) post-prune schema field types: "
                        + ", ".join(new_schema_desc),
                        flush=True,
                    )
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass

    ds_all = None  # Track ALL articles (including filtered ones)
    ds_vague = None  # Track too-vague EU rows
    
    if prefilter_mode in ("pre_gating", "post_gating"):
        # Add keyword information: binary flag, matched keywords list, and match count
        def _add_keyword_info_pandas(r: Dict[str, Any]) -> Dict[str, Any]:
            has_kw, matched_kws, match_count = _extract_matched_keywords(r.get("article_text"))
            return {
                **r,
                "relevant_keyword": has_kw,
                "matched_keywords": matched_kws,
                "keyword_match_count": match_count,
            }
        ds_in = ds_in.map(_add_keyword_info_pandas)
        if prefilter_mode == "pre_gating":
            # Keep a copy of ALL articles before filtering
            # Must materialize to cache the unfiltered data, otherwise ds_all will reference
            # the same lazy execution plan as ds_in and both will be filtered
            ds_all = ds_in.materialize()
            # Filter to only articles WITH keywords for LLM processing
            ds_in = ds_all.filter(lambda r: bool(r.get("relevant_keyword", True)))
    # EU and risks_benefits: compute input completeness flags and split off too-vague rows before LLM
    if is_eu_profile or is_risks_benefits_profile:
        try:
            ds_in = ds_in.map(_add_eu_vague_flags)
            ds_vague = ds_in.filter(lambda r: bool(r.get("too_vague_to_process", False)))
            ds_in = ds_in.filter(lambda r: not bool(r.get("too_vague_to_process", False)))
            # Debug: count vague rows
            try:
                vague_count = ds_vague.count()
                print(f"[classify] Filtered {vague_count} too-vague rows before LLM processing", flush=True)
            except Exception:
                pass
        except Exception:
            ds_vague = None
    if enable_kw_buf and kw_regex is not None:
        ds_in = ds_in.map(_attach_chunk_text)
    sys_text = system_prompt
    def _trim_row(r: Dict[str, Any]) -> Dict[str, Any]:
        txt = r.get("chunk_text") or r.get("article_text")
        # Use char-based trimming to avoid heavy tokenizer in Ray workers.
        r["chunk_text"] = _trim_text_for_prompt(str(txt or ""), None, sys_text)
        return r
    ds_in = ds_in.map(_trim_row)

    processor = build_llm_processor(engine_config, preprocess=_pre, postprocess=_post)
    # Avoid an extra materialize; converting to pandas will trigger execution.
    out_df = processor(ds_in).to_pandas()
    # Drop internal columns prior to any merging
    try:
        out_df = out_df.drop(columns=_INTERNAL_DROP_COLS, errors="ignore")
    except Exception:
        pass
    # Drop risks_benefits specific columns if in that profile
    if is_risks_benefits_profile:
        try:
            out_df = out_df.drop(columns=_RISKS_BENEFITS_DROP_COLS, errors="ignore")
        except Exception:
            pass
    
    # If we filtered articles, merge LLM results back with filtered articles
    print(f"[classify] Checking merge conditions: ds_all={ds_all is not None}, ds_vague={ds_vague is not None}", flush=True)
    if ds_all is not None or (ds_vague is not None and (is_eu_profile or is_risks_benefits_profile)):
        print(f"[classify] Starting merge process...", flush=True)
        try:
            df_llm = out_df
            df_all = ds_all.to_pandas() if ds_all is not None else None
            
            result_parts = [
                _normalize_profile_columns(df_llm, is_eu_profile, is_risks_benefits_profile)
            ]
            if df_all is not None:
                processed_ids = set(df_llm["article_id"].unique())
                # Exclude too-vague IDs from keyword-filtered to avoid duplication when merging back
                vague_ids: set[str] = set()
                try:
                    if 'df_vague' in locals() and isinstance(df_vague, pd.DataFrame) and "article_id" in df_vague.columns:
                        vague_ids = set(df_vague["article_id"].unique())
                except Exception:
                    vague_ids = set()
                excluded_ids = processed_ids.union(vague_ids)
                df_filtered = df_all[~df_all["article_id"].isin(list(excluded_ids))].copy()
                df_filtered["classification_mode"] = "filtered_by_keyword"
                df_filtered = _normalize_profile_columns(df_filtered, is_eu_profile, is_risks_benefits_profile)
                if "matched_keywords" in df_filtered.columns:
                    df_filtered["matched_keywords"] = df_filtered["matched_keywords"].apply(_to_json_str)
                result_parts.append(df_filtered)
            if ds_vague is not None and (is_eu_profile or is_risks_benefits_profile):
                df_vague = ds_vague.to_pandas().copy()
                print(f"[classify] Merging back {len(df_vague)} too-vague rows to output", flush=True)
                df_vague["classification_mode"] = "too_vague_to_process"
                df_vague["too_vague_to_process"] = True
                df_vague = _normalize_profile_columns(df_vague, is_eu_profile, is_risks_benefits_profile)
                result_parts.append(df_vague)
            else:
                print(f"[classify] No too-vague rows to merge (ds_vague is None: {ds_vague is None})", flush=True)
            out_df = _merge_result_parts(result_parts)
            # Only ensure too_vague column for EU/RB profiles
            if is_eu_profile or is_risks_benefits_profile:
                if "too_vague_to_process" not in out_df.columns:
                    out_df["too_vague_to_process"] = False
                else:
                    out_df["too_vague_to_process"] = out_df["too_vague_to_process"].fillna(False)
            try:
                out_df = out_df.drop(columns=_INTERNAL_DROP_COLS, errors="ignore")
            except Exception:
                pass
            # Drop risks_benefits specific columns if in that profile
            if is_risks_benefits_profile:
                try:
                    out_df = out_df.drop(columns=_RISKS_BENEFITS_DROP_COLS, errors="ignore")
                except Exception:
                    pass
            
            try:
                num_kw = len(df_filtered) if 'df_filtered' in locals() else 0
            except Exception:
                num_kw = 0
            try:
                num_vague = len(df_vague) if 'df_vague' in locals() else 0
            except Exception:
                num_vague = 0
            print(f"[classify] Merged results: {len(df_llm)} LLM-processed + {num_kw} keyword-filtered + {num_vague} too-vague = {len(out_df)} total", flush=True)
        except Exception as e:
            print(f"[classify] ERROR: Failed to merge filtered articles back into results: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Fall back to just LLM results
    else:
        print(f"[classify] No merge needed, returning LLM results directly", flush=True)
    
    # Stage-scoped logging is handled by the orchestrator
    out_df = _prune_result_columns(out_df, is_eu_profile, is_risks_benefits_profile)
    return out_df

def _normalize_profile_columns(df: pd.DataFrame, is_eu_profile: bool, is_risks_benefits_profile: bool) -> pd.DataFrame:
    df = df.copy()
    if is_eu_profile:
        defaults = {
            "eu_ai_desc": None,
            "eu_ai_label": None,
            "eu_ai_relevant_text": None,
            "eu_ai_reason": None,
            "eu_ai_raw_json": None,
            "too_vague_to_process": False,
            "classification_mode": "filtered_by_keyword",
        }
    elif is_risks_benefits_profile:
        defaults = {
            "rb_desc": None,
            "rb_human_rights": None,
            "rb_sdgs": None,
            "rb_additional": None,
            "too_vague_to_process": False,
            "classification_mode": "filtered_by_keyword",
        }
    else:
        defaults = {
            "is_relevant": False,
            "relevance_answer": None,
            "classification_mode": "filtered_by_keyword",
        }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        elif isinstance(default, bool):
            df[col] = df[col].fillna(default).astype(bool)
        else:
            # Only fill when default is not None; pandas fillna(None) raises
            if default is not None:
                df[col] = df[col].fillna(default)
    if is_risks_benefits_profile:
        for col in ("rb_human_rights", "rb_sdgs", "rb_additional"):
            if col in df.columns:
                df[col] = df[col].apply(lambda v: _to_json_str(v) if not isinstance(v, str) else v)
    return df


def _merge_result_parts(parts: List[pd.DataFrame]) -> pd.DataFrame:
    """Safely merge heterogeneous DataFrame parts without relying on pd.concat.

    Ray/pandas interoperability can yield mixes of numpy-backed and Arrow-backed
    blocks. In those cases, pd.concat may raise dimensionality errors when the
    underlying array managers disagree on ndim. Converting each part to records
    ensures uniform row-wise structures before rebuilding the final DataFrame.
    """

    records: List[Dict[str, Any]] = []
    column_order: List[str] = []

    for part in parts:
        if part is None:
            continue

        if isinstance(part, pd.Series):
            part_df = part.to_frame().T
        elif not isinstance(part, pd.DataFrame):
            try:
                part_df = pd.DataFrame(part)
            except Exception:
                part_df = pd.DataFrame([part])
        else:
            part_df = part

        for col in part_df.columns:
            if col not in column_order:
                column_order.append(col)

        records.extend(part_df.to_dict(orient="records"))

    if not records:
        return pd.DataFrame(columns=column_order)

    merged = pd.DataFrame.from_records(records)
    if column_order:
        merged = merged.reindex(columns=column_order)
    return merged


# Internal reference implementation - kept for reference only, not for external use
# Profile modules should use run_classification_core() instead
_run_classification_stage_impl = run_classification_core
