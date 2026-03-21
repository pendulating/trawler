"""Shared orchestrator utilities for all dagspace pipelines.

Functions and classes extracted from uair, historical_norms, and
privacylens orchestrators. Each orchestrator imports from
this module and re-exports the symbols for backward-compatibility with
runner files that do ``from ..orchestrator import X``.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.config_schema import PipelineNodeSpec

try:
    import submitit  # type: ignore
    _SUBMITIT_AVAILABLE = True
except Exception:
    submitit = None  # type: ignore
    _SUBMITIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StageExecutionContext:
    cfg: DictConfig
    node: PipelineNodeSpec
    inputs: Dict[str, str]
    output_paths: Dict[str, str]
    output_dir: str
    output_root: str
    logger: Optional[Any] = None


@dataclass
class StageResult:
    outputs: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# No-op logger
# ---------------------------------------------------------------------------

class _NoOpLogger:
    """No-op logger that matches WandbLogger interface when wandb is disabled."""

    def __init__(
        self,
        cfg: Any,
        stage: str,
        run_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Import lazily so that dagspaces without a local wandb_logger can still
        # use this class (e.g. if imported from common directly).
        self.cfg = cfg
        self.stage = stage
        self.run_id = run_id
        self.run_config = run_config or {}
        # Attempt to populate wb_config; tolerate missing attribute.
        try:
            from dagspaces.common.wandb_config import WandbConfig  # type: ignore
            self.wb_config = WandbConfig.from_hydra_config(cfg)
        except Exception:
            self.wb_config = None

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        pass

    def log_table(
        self,
        df: Any,
        key: str,
        prefer_cols: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
        panel_group: Optional[str] = None,
    ) -> None:
        pass

    def set_summary(self, key: str, value: Any) -> None:
        pass

    def set_config(self, data: Dict[str, Any], allow_val_change: bool = True) -> None:
        pass

    def __enter__(self) -> "_NoOpLogger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False


# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------

def clone_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))  # type: ignore[return-value]


def merge_overrides(
    base_cfg: DictConfig, overrides: Optional[Mapping[str, Any]]
) -> DictConfig:
    if not overrides:
        return base_cfg
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


def prepare_node_config(
    base_cfg: DictConfig, node: PipelineNodeSpec, output_dir: str
) -> DictConfig:
    cfg_copy = clone_config(base_cfg)
    cfg_copy = merge_overrides(cfg_copy, node.overrides)
    ensure_section(cfg_copy, "runtime")
    OmegaConf.update(cfg_copy, "runtime.stage", node.stage, merge=True)
    OmegaConf.update(cfg_copy, "runtime.output_dir", output_dir, merge=True)
    OmegaConf.update(cfg_copy, "runtime.output_csv", None, merge=True)
    return cfg_copy


# ---------------------------------------------------------------------------
# Standardized run_config for W&B logging
# ---------------------------------------------------------------------------


def build_run_config(
    cfg: DictConfig,
    node: PipelineNodeSpec,
    inputs: Dict[str, str],
    output_paths: Dict[str, str],
    *,
    dagspace_name: str = "",
) -> Dict[str, Any]:
    """Build a standardized run_config dict for W&B logging.

    Centralises the run_config construction that was previously duplicated
    across all dagspace orchestrators.  Adds pipeline_name, eval_task,
    dagspace, and checkpoint_name metadata for cross-model comparison.
    """
    run_config: Dict[str, Any] = {
        "node": node.key,
        "stage": node.stage,
        "inputs": list(inputs.keys()),
        "outputs": list(output_paths.keys()),
    }

    if dagspace_name:
        run_config["dagspace"] = dagspace_name

    pipeline_name = _resolve_pipeline_name()
    if pipeline_name:
        run_config["pipeline_name"] = pipeline_name

    eval_task = _resolve_eval_task(cfg, node)
    if eval_task:
        run_config["eval_task"] = eval_task

    checkpoint_name = _resolve_checkpoint_name(cfg)
    if checkpoint_name:
        run_config["checkpoint_name"] = checkpoint_name

    return run_config


def _resolve_pipeline_name() -> Optional[str]:
    """Extract pipeline name from Hydra overrides."""
    try:
        from hydra.core.hydra_config import HydraConfig
        hydra_cfg = HydraConfig.get()
        if hydra_cfg and getattr(hydra_cfg, "job", None):
            override_dir = getattr(hydra_cfg.job, "override_dirname", None)
            if override_dir:
                for part in str(override_dir).split(","):
                    p = part.strip()
                    if p.startswith("pipeline="):
                        return p.split("=", 1)[1]
    except Exception:
        pass
    return os.environ.get("WANDB_PIPELINE_NAME") or None


def _resolve_eval_task(cfg: DictConfig, node: PipelineNodeSpec) -> Optional[str]:
    """Extract eval task identifier from node overrides or prompt config."""
    try:
        task = (node.overrides or {}).get("prompt", {}).get("task")
        if task:
            return str(task)
    except Exception:
        pass
    try:
        task = OmegaConf.select(cfg, "prompt.task")
        if task:
            return str(task)
    except Exception:
        pass
    return None


def _resolve_checkpoint_name(cfg: DictConfig) -> Optional[str]:
    """Derive checkpoint name from model config."""
    try:
        model_cfg = getattr(cfg, "model", None)
        if not model_cfg:
            return None
        cn = getattr(model_cfg, "checkpoint_name", None)
        if cn:
            return str(cn)
        model_source = getattr(model_cfg, "model_source", None)
        lora_path = getattr(model_cfg, "lora_path", None)
        if lora_path:
            from dagspaces.common.wandb_logger import _derive_checkpoint_name
            return _derive_checkpoint_name(
                str(lora_path), str(model_source) if model_source else ""
            )
        elif model_source:
            return os.path.basename(str(model_source))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Prompt / launcher config injection (requires caller-supplied config_dir)
# ---------------------------------------------------------------------------

def _inject_prompt_from_file(
    cfg: DictConfig, prompt_filename: str, config_dir: str
) -> None:
    """Inject prompt from YAML file into ``cfg.prompt``.

    Supports subdirectory paths like ``'general_ai/classify.yaml'`` or plain
    ``'classify.yaml'``.  ``config_dir`` must be the absolute path to the
    dagspace's ``conf/`` directory so that this function remains location-agnostic.
    """
    try:
        prompt_path = os.path.join(config_dir, "prompt", prompt_filename)
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


def _load_launcher_config(
    cfg: DictConfig, launcher_name: str, config_dir: str
) -> Optional[DictConfig]:
    """Load a launcher configuration from Hydra config.

    ``config_dir`` must be the absolute path to the dagspace's ``conf/``
    directory (e.g. ``os.path.join(os.path.dirname(__file__), "conf")``).
    """
    try:
        if not os.path.exists(config_dir):
            # Try to get from hydra runtime
            hydra_cfg = getattr(cfg, "hydra", None)
            if hydra_cfg:
                runtime_cfg = getattr(hydra_cfg, "runtime", None)
                if runtime_cfg:
                    sources = getattr(runtime_cfg, "config_sources", [])
                    for source in sources:
                        if hasattr(source, "provider") and source.provider == "main":
                            config_dir = source.path
                            break

        if not config_dir or not os.path.exists(config_dir):
            raise ValueError("Could not find config directory")

        launcher_file = os.path.join(
            config_dir, "hydra", "launcher", f"{launcher_name}.yaml"
        )
        # Fall back to common/conf/hydra/launcher/ if not found locally
        if not os.path.exists(launcher_file):
            common_conf = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "conf"
            )
            launcher_file = os.path.join(
                common_conf, "hydra", "launcher", f"{launcher_name}.yaml"
            )
        if not os.path.exists(launcher_file):
            raise ValueError(f"Launcher config file not found: {launcher_name}.yaml")

        launcher_cfg = OmegaConf.load(launcher_file)
        launcher_cfg = OmegaConf.merge(
            {"runtime": cfg.get("runtime", {})}, launcher_cfg
        )
        return launcher_cfg
    except Exception as e:
        raise ValueError(
            f"Failed to load launcher config '{launcher_name}': {e}"
        ) from e


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_parquet_dataset(
    parquet_path: str,
    columns: Mapping[str, str],
    debug: bool,
    sample_n: Optional[int],
) -> pd.DataFrame:
    if not isinstance(parquet_path, str) or parquet_path.strip() == "":
        raise ValueError("data.parquet_path is required")
    if not os.path.isabs(parquet_path):
        parquet_path = os.path.abspath(parquet_path)
    df = pd.read_parquet(parquet_path)

    # Apply caller-specified column renames (e.g. data.columns in Hydra config).
    # Only rename columns that are explicitly mapped and present in the data.
    if columns:
        col_map = {src: dst for src, dst in columns.items() if src in df.columns}
        if col_map:
            df = df.rename(columns=col_map)

    if "article_text" not in df.columns and "chunk_text" not in df.columns:
        raise RuntimeError(
            "Parquet missing required text column (article_text) or chunk_text"
        )

    # Apply sample_n regardless of debug flag - it's a runtime limit, not just for
    # debugging.
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
            print(
                f"[_load_parquet_dataset] Applied sample_n={n} (seed={seed}), "
                f"processing {len(df)} rows",
                flush=True,
            )
        except Exception:
            df = df.head(n)
            print(
                f"[_load_parquet_dataset] Applied sample_n={n} (head), "
                f"processing {len(df)} rows",
                flush=True,
            )
    return df


def prepare_stage_input(
    cfg: DictConfig, dataset_path: str, stage: str
) -> "tuple[Optional[pd.DataFrame], Optional[Any], bool]":
    """Load stage input as a pandas DataFrame.

    Returns ``(df, None, False)`` – streaming via Ray is no longer supported.
    The tuple shape is kept for backward compatibility with existing runner code.
    """
    debug = bool(getattr(cfg.runtime, "debug", False))
    sample_n = getattr(cfg.runtime, "sample_n", None)
    columns = dict(getattr(cfg.data, "columns", {})) if getattr(cfg, "data", None) else {}

    if dataset_path and not os.path.isabs(dataset_path):
        dataset_path = os.path.abspath(dataset_path)

    df = _load_parquet_dataset(dataset_path, columns, debug=debug, sample_n=sample_n)
    return df, None, False


# ---------------------------------------------------------------------------
# Stage I/O
# ---------------------------------------------------------------------------

def _clean_df_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame to avoid PyArrow serialisation issues.

    Converts complex nested columns to JSON strings and drops columns that are
    known to cause Arrow errors with empty structs.
    """
    import json as _json

    # Columns that commonly cause Arrow issues with empty structs
    problematic_cols = [
        "metadata",
        "reasoning_data",
        "ig20_statements_raw",
        "raz_norms_raw",
        "__inference_error__",
        "embeddings",
        "params",
        "metrics",
        "prompt_token_ids",
        "logprobs",
        "prompt_logprobs",
    ]

    def _json_fallback(value: Any) -> Any:
        """Convert non-JSON-native values (e.g. numpy) into serialisable objects."""
        try:
            import numpy as np  # type: ignore

            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass
        if isinstance(value, (set, tuple)):
            return list(value)
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="replace")
            except Exception:
                return str(value)
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        return str(value)

    for col in list(df.columns):
        if col in problematic_cols:
            df = df.drop(columns=[col], errors="ignore")
            continue

        # Check for any column with dict/list values and convert to JSON strings.
        # Never drop a whole semantic column due to nested types.
        try:
            sample = df[col].dropna().head(5)
            if len(sample) > 0:
                has_complex = any(isinstance(v, (dict, list)) for v in sample)
                if has_complex:
                    df[col] = df[col].apply(
                        lambda x: _json.dumps(x, default=_json_fallback)
                        if isinstance(x, (dict, list))
                        else x
                    )
        except Exception:
            try:
                # Last-resort value-level coercion; keep the column.
                df[col] = df[col].apply(
                    lambda x: _json.dumps(x, default=_json_fallback)
                    if isinstance(x, (dict, list))
                    else x
                )
            except Exception:
                # Only coerce to plain strings if serialisation still fails.
                try:
                    df[col] = df[col].astype(str)
                except Exception:
                    pass

    return df


def _save_stage_outputs(out: pd.DataFrame, output_paths: Dict[str, str]) -> None:
    """Save DataFrame outputs to disk with parquet primary, CSV/pickle fallback."""
    if isinstance(out, pd.DataFrame):
        for _output_name, output_path in output_paths.items():
            out_clean = _clean_df_for_parquet(out.copy())

            try:
                out_clean.to_parquet(output_path, index=False)
                print(
                    f"[orchestrator] Saved {len(out_clean)} rows to {output_path}"
                )
            except Exception as parquet_err:
                print(f"[orchestrator] Parquet save failed: {parquet_err}")

                csv_path = output_path.replace(".parquet", ".csv")
                try:
                    out_clean.to_csv(csv_path, index=False)
                    print(f"[orchestrator] CSV fallback saved to {csv_path}")
                except Exception as csv_err:
                    print(f"[orchestrator] CSV fallback also failed: {csv_err}")

                    pickle_path = output_path.replace(".parquet", ".pkl")
                    try:
                        out.to_pickle(pickle_path)
                        print(
                            f"[orchestrator] Pickle fallback saved to {pickle_path}"
                        )
                    except Exception as pkl_err:
                        print(
                            f"[orchestrator] All save methods failed: {pkl_err}"
                        )
                        raise parquet_err  # Re-raise original error


def _safe_log_table(
    logger: Optional[Any],
    df: pd.DataFrame,
    key: str,
    prefer_cols: Optional[List[str]] = None,
    panel_group: str = "inspect_results",
) -> None:
    """Safely log DataFrame to wandb."""
    if logger and isinstance(df, pd.DataFrame):
        try:
            logger.log_table(
                df, key, prefer_cols=prefer_cols, panel_group=panel_group
            )
        except Exception as e:
            print(f"Warning: Failed to log {key} to wandb: {e}", flush=True)


def _collect_outputs(
    context: StageExecutionContext, optional: Mapping[str, bool]
) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for key, path in context.output_paths.items():
        if os.path.exists(path):
            resolved[key] = path
        else:
            # Check fallback extensions written by _save_stage_outputs
            found = False
            for alt_ext in (".csv", ".pkl"):
                alt_path = (
                    path.replace(".parquet", alt_ext)
                    if path.endswith(".parquet")
                    else None
                )
                if alt_path and os.path.exists(alt_path):
                    resolved[key] = alt_path
                    print(
                        f"[orchestrator] Using fallback output for '{key}': {alt_path}"
                    )
                    found = True
                    break
            if not found:
                if optional.get(key, False):
                    continue
                raise FileNotFoundError(
                    f"Expected output '{key}' for node '{context.node.key}' "
                    f"at '{path}' not found"
                )
    return resolved


def _ensure_output_dirs(paths: Iterable[str]) -> None:
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _node_optional_outputs(node: PipelineNodeSpec) -> Dict[str, bool]:
    return {name: spec.optional for name, spec in node.outputs.items()}


def _node_output_paths(
    node: PipelineNodeSpec, registry: "ArtifactRegistry", output_root: str
) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for out_name, spec in node.outputs.items():
        resolved[out_name] = registry.resolve_output_path(
            spec.path, output_root, node.key
        )
    _ensure_output_dirs(resolved.values())
    return resolved


def _node_inputs(
    node: PipelineNodeSpec, registry: "ArtifactRegistry"
) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for alias, ref in node.inputs.items():
        resolved[alias] = registry.resolve(ref)
    return resolved


# ---------------------------------------------------------------------------
# Artifact registry
# ---------------------------------------------------------------------------

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

    def resolve_output_path(
        self, path: str, output_root: str, node_key: str
    ) -> str:
        if not path:
            raise ValueError(f"Node '{node_key}' output path is empty")
        resolved = path
        if not os.path.isabs(resolved):
            resolved = os.path.join(output_root, resolved)
        return os.path.abspath(resolved)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_status(payload: Dict[str, Any]) -> None:
    try:
        print(json.dumps(payload, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------

def _probe_single_gpu(device: str) -> Dict[str, Any]:
    """Probe a single logical GPU in a subprocess.

    Using a subprocess keeps CUDA initialisation out of the orchestrator/stage
    parent process, which is important before vLLM chooses its multiprocessing
    strategy.
    """
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
            text=True,
            timeout=15,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": (result.stdout or "").strip(),
            "stderr": (result.stderr or "").strip(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }


def _update_slurm_gpu_envs(valid_devices: List[str]) -> None:
    """Update SLURM GPU environment variables to reflect the sanitised device list."""
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


def _adjust_tensor_parallel_env(
    valid_count: int, env_prefix: str = "UAIR"
) -> None:
    """Clamp the tensor-parallel env var to the sanitised GPU count."""
    tp_env_name = f"{env_prefix}_TENSOR_PARALLEL_SIZE"
    tp_env = os.environ.get(tp_env_name)
    if not tp_env:
        return
    try:
        tp_val = max(1, int(tp_env))
        if valid_count > 0 and tp_val > valid_count:
            os.environ[tp_env_name] = str(valid_count)
    except Exception:
        pass


def _log_gpu_environment(reason: str, env_prefix: str = "UAIR") -> None:
    """Print a structured JSON status block describing the current GPU environment."""
    try:
        cuda_visible = [
            d.strip()
            for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if d.strip()
        ]
        dropped = [
            d.strip()
            for d in os.environ.get(
                f"{env_prefix}_SANITIZED_DROPPED_GPUS", ""
            ).split(",")
            if d.strip()
        ]
        original = [
            d.strip()
            for d in os.environ.get(
                f"{env_prefix}_GPU_SANITIZE_ORIGINAL", ""
            ).split(",")
            if d.strip()
        ]
        payload: Dict[str, Any] = {
            "reason": reason,
            "cuda_visible_devices": cuda_visible,
        }
        if original:
            payload["sanitized_original"] = original
        if dropped:
            payload["sanitized_dropped"] = dropped
        tp_env = os.environ.get(f"{env_prefix}_TENSOR_PARALLEL_SIZE")
        if tp_env:
            try:
                payload["tensor_parallel_size"] = int(tp_env)
            except Exception:
                payload["tensor_parallel_size"] = tp_env
        _print_status({"gpu_env": payload})
    except Exception:
        pass


def _sanitize_cuda_visible_devices(
    reason: str = "",
    env_prefix: str = "UAIR",
    cfg: Optional[DictConfig] = None,
) -> None:
    """Remove bad logical GPUs from ``CUDA_VISIBLE_DEVICES`` before stage startup.

    Parameters
    ----------
    reason:
        Short label included in the structured status output, e.g.
        ``"stage_start"`` or ``"node:my_stage"``.
    env_prefix:
        Prefix for the sanitisation book-keeping env vars
        (e.g. ``"UAIR"`` or ``"HISTORICAL_NORMS"``).
    cfg:
        Optional Hydra config.  When provided, ``model.engine_kwargs.tensor_parallel_size``
        is clamped to the number of valid GPUs via OmegaConf (historical_norms behaviour).
        When ``None``, only the env-var-based adjustment is applied.
    """
    skip_env = f"{env_prefix}_SKIP_GPU_SANITIZE"
    if os.environ.get(skip_env):
        return
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not current:
        return
    devices = [d.strip() for d in current.split(",") if d.strip()]
    if len(devices) <= 1:
        _log_gpu_environment(reason or "stage_start", env_prefix=env_prefix)
        return

    normalized = ",".join(devices)
    valid: List[str] = []
    invalid: List[str] = []
    probe_failures: Dict[str, Dict[str, Any]] = {}
    for dev in devices:
        probe_result = _probe_single_gpu(dev)
        if probe_result.get("ok"):
            valid.append(dev)
        else:
            invalid.append(dev)
            probe_failures[dev] = probe_result

    if not invalid:
        _log_gpu_environment(reason or "stage_start", env_prefix=env_prefix)
        return

    if not valid:
        # If everything failed, surface the probe failures immediately.
        os.environ[f"{env_prefix}_GPU_SANITIZE_REASON"] = reason or "stage_start"
        os.environ[f"{env_prefix}_GPU_SANITIZE_TS"] = str(int(time.time()))
        os.environ.pop(f"{env_prefix}_GPU_SANITIZE_ORIGINAL", None)
        os.environ.pop(f"{env_prefix}_SANITIZED_DROPPED_GPUS", None)
        _print_status(
            {
                "gpu_sanitize": {
                    "reason": reason or "stage_start",
                    "original": normalized,
                    "error": "all_devices_failed",
                }
            }
        )
        failure_summary: Dict[str, Dict[str, Any]] = {}
        for dev, details in probe_failures.items():
            failure_summary[dev] = {
                "returncode": details.get("returncode"),
                "stdout": details.get("stdout"),
                "stderr": details.get("stderr"),
            }
        _print_status({"gpu_sanitize_failures": failure_summary})
        raise RuntimeError(
            "GPU sanitize failed for every visible device before stage startup. "
            f"reason={reason or 'stage_start'} visible={normalized}. "
            f"Set {skip_env}=1 to bypass the preflight check if you explicitly want to continue."
        )

    new_devices = ",".join(valid)
    os.environ["CUDA_VISIBLE_DEVICES"] = new_devices
    os.environ[f"{env_prefix}_SANITIZED_DROPPED_GPUS"] = ",".join(invalid)
    os.environ[f"{env_prefix}_GPU_SANITIZE_REASON"] = reason or "stage_start"
    os.environ[f"{env_prefix}_GPU_SANITIZE_TS"] = str(int(time.time()))
    os.environ[f"{env_prefix}_GPU_SANITIZE_ORIGINAL"] = normalized
    _update_slurm_gpu_envs(valid)

    # Clamp tensor-parallel env var
    _adjust_tensor_parallel_env(len(valid), env_prefix=env_prefix)

    # Clamp tensor-parallel in cfg when provided (historical_norms style)
    if cfg is not None:
        try:
            current_tp = OmegaConf.select(
                cfg, "model.engine_kwargs.tensor_parallel_size"
            )
            if current_tp is not None:
                current_tp_int = max(1, int(current_tp))
                if current_tp_int > len(valid):
                    OmegaConf.update(
                        cfg,
                        "model.engine_kwargs.tensor_parallel_size",
                        len(valid),
                        merge=True,
                    )
        except Exception:
            pass

    _print_status(
        {
            "gpu_sanitize": {
                "reason": reason or "stage_start",
                "original": normalized,
                "sanitized": new_devices,
                "dropped": ",".join(invalid),
            }
        }
    )
    _log_gpu_environment(reason or "stage_start", env_prefix=env_prefix)


# ---------------------------------------------------------------------------
# SLURM / launcher
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _clean_slurm_env():  # type: ignore[return]
    """Temporarily remove Slurm environment variables to prevent incorrect
    inheritance when nesting submitit jobs inside a SLURM allocation.

    All ``SLURM*`` and ``SBATCH*`` env vars are saved, removed for the duration
    of the ``with`` block, then restored in the ``finally`` clause.
    """
    slurm_vars = {
        k: v
        for k, v in os.environ.items()
        if k.startswith("SLURM") or k.startswith("SBATCH")
    }
    for k in slurm_vars:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        os.environ.update(slurm_vars)


def _create_submitit_executor(
    launcher_cfg: DictConfig,
    job_name: str,
    log_folder: str,
    *,
    use_srun: bool = False,
) -> Any:
    """Create a submitit ``AutoExecutor`` from launcher configuration.

    Parameters
    ----------
    launcher_cfg:
        Hydra DictConfig loaded from a ``conf/hydra/launcher/*.yaml`` file.
    job_name:
        SLURM job name.
    log_folder:
        Directory where submitit writes stdout/stderr and result pickles.
    use_srun:
        Pass ``slurm_use_srun=True`` to submitit (default ``False``).
        privacylens explicitly disables srun; other dagspaces
        rely on the submitit default.
    """
    if not _SUBMITIT_AVAILABLE or submitit is None:
        raise RuntimeError(
            "submitit is not available but is required for SLURM job submission"
        )

    with _clean_slurm_env():
        executor = submitit.AutoExecutor(folder=log_folder)

    params: Dict[str, Any] = dict(
        timeout_min=int(launcher_cfg.get("timeout_min", 120)),
        slurm_partition=str(launcher_cfg.get("partition", "pierson")),
        slurm_mem=f"{int(launcher_cfg.get('mem_gb', 8))}GB",
        slurm_cpus_per_task=int(launcher_cfg.get("cpus_per_task", 2)),
        slurm_gpus_per_node=int(launcher_cfg.get("gpus_per_node", 0)),
        slurm_nodes=int(launcher_cfg.get("nodes", 1)),
        slurm_tasks_per_node=int(launcher_cfg.get("tasks_per_node", 1)),
        slurm_array_parallelism=int(launcher_cfg.get("array_parallelism", 1)),
        name=f"matt-{job_name}",
        slurm_additional_parameters=launcher_cfg.get("additional_parameters", {}),
        slurm_setup=launcher_cfg.get("setup", []),
    )
    if not use_srun:
        params["slurm_use_srun"] = False

    executor.update_parameters(**params)
    return executor
