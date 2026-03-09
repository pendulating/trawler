"""Shared vLLM direct inference utility.

Replaces the Ray-based build_llm_processor + vLLMEngineProcessorConfig pattern
with direct vLLM LLM.generate() calls. Designed for single-machine multi-GPU setups.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Callable, Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# GPU / environment helpers
# ---------------------------------------------------------------------------

def get_pcie_nccl_env_vars() -> Dict[str, str]:
    """Return NCCL environment variables required for PCIe-only GPUs (no NVLink)."""
    return {
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
        "NCCL_CUMEM_HOST_ENABLE": "0",
        "NCCL_DEBUG": "WARN",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }


def detect_num_gpus() -> int:
    """Detect the number of GPUs available.

    Resolution order:
    1. UAIR_TENSOR_PARALLEL_SIZE env override
    2. CUDA_VISIBLE_DEVICES
    3. SLURM GPU env vars
    4. torch.cuda.device_count()
    5. Fallback to 1
    """
    # Env override
    tp_env = os.environ.get("UAIR_TENSOR_PARALLEL_SIZE")
    if tp_env:
        try:
            val = int(tp_env)
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass

    # CUDA_VISIBLE_DEVICES
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible.strip():
            ids = [x.strip() for x in cuda_visible.split(",") if x.strip()]
            if ids:
                return len(ids)
    except Exception:
        pass

    # SLURM
    try:
        slurm_gpus = (
            os.environ.get("SLURM_GPUS_PER_NODE")
            or os.environ.get("SLURM_GPUS_ON_NODE")
        )
        if slurm_gpus:
            if ":" in slurm_gpus:
                return int(slurm_gpus.split(":")[-1])
            return int(slurm_gpus)
    except Exception:
        pass

    # torch
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                return count
    except Exception:
        pass

    return 1


def detect_gpu_type() -> str:
    """Detect GPU type, returning a normalised string like 'rtx_a6000'."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            name = torch.cuda.get_device_name(0).lower()
            if "a6000" in name:
                return "rtx_a6000"
            if "a5000" in name:
                return "rtx_a5000"
            if "a100" in name:
                return "a100"
            if "h100" in name:
                return "h100"
            if "v100" in name:
                return "v100"
            if "a40" in name:
                return "a40"
            if "rtx" in name:
                return "rtx_generic"
            return "unknown"
    except Exception:
        pass
    return "unknown"


def apply_gpu_aware_settings(engine_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Set max_num_seqs based on GPU type if not already specified.

    Returns the GPU-type defaults dict (may be empty).
    """
    GPU_DEFAULTS = {
        "rtx_a6000": {"batch_size": 4, "max_num_seqs": 4},
        "rtx_a5000": {"batch_size": 2, "max_num_seqs": 2},
        "a100": {"batch_size": 8, "max_num_seqs": 8},
        "h100": {"batch_size": 16, "max_num_seqs": 16},
        "v100": {"batch_size": 4, "max_num_seqs": 4},
        "a40": {"batch_size": 4, "max_num_seqs": 4},
    }
    gpu_type = detect_gpu_type()
    defaults = GPU_DEFAULTS.get(gpu_type, {})
    if defaults and "max_num_seqs" not in engine_kwargs:
        engine_kwargs["max_num_seqs"] = defaults["max_num_seqs"]
        print(f"[vllm_inference] Auto-set max_num_seqs={defaults['max_num_seqs']} for {gpu_type}")
    return defaults


def filter_vllm_engine_kwargs(ek: Dict[str, Any]) -> Dict[str, Any]:
    """Drop engine kwargs not accepted by the installed vLLM LLM class."""
    try:
        import inspect
        from vllm import LLM as _LLM
        sig = inspect.signature(_LLM.__init__)
        accepted = {k for k in sig.parameters if k != "self"}
        filtered = {k: v for k, v in ek.items() if k in accepted}
        dropped = [k for k in ek if k not in accepted]
        if dropped:
            print(f"[vllm_inference] Dropped unsupported vLLM kwargs: {dropped}")
        return filtered
    except Exception:
        pass
    # Conservative fallback
    ek = dict(ek)
    for k in ("use_v2_block_manager", "concurrency", "batch_size"):
        ek.pop(k, None)
    return ek


# ---------------------------------------------------------------------------
# Engine kwargs builder
# ---------------------------------------------------------------------------

def _build_engine_kwargs(cfg) -> Dict[str, Any]:
    """Build vLLM LLM constructor kwargs from Hydra config."""
    model_source = str(getattr(cfg.model, "model_source"))
    ek = dict(getattr(cfg.model, "engine_kwargs", {}))

    # Model
    ek["model"] = model_source

    # Tensor parallelism
    if "tensor_parallel_size" not in ek:
        ek["tensor_parallel_size"] = detect_num_gpus()
        print(f"[vllm_inference] Auto-detected {ek['tensor_parallel_size']} GPU(s) for tensor parallelism")

    # GPU-aware tuning
    apply_gpu_aware_settings(ek)

    # Safe defaults
    ek.setdefault("trust_remote_code", True)
    ek.setdefault("distributed_executor_backend", "mp")

    # AWQ auto-detection
    if "awq" in model_source.lower() and "quantization" not in ek:
        ek["quantization"] = "awq"

    # Filter to accepted kwargs
    ek = filter_vllm_engine_kwargs(ek)
    return ek


# ---------------------------------------------------------------------------
# SamplingParams builder
# ---------------------------------------------------------------------------

def _build_sampling_params(sp_dict: Dict[str, Any]):
    """Convert a plain dict to vLLM SamplingParams, handling guided_decoding."""
    from vllm import SamplingParams

    sp = dict(sp_dict or {})

    # Extract guided_decoding if present
    guided = sp.pop("guided_decoding", None)
    # Remove keys that aren't valid SamplingParams fields
    for k in ("early_stopping", "length_penalty", "response_format",
              "structured_output", "detokenize"):
        sp.pop(k, None)

    if guided and isinstance(guided, dict):
        try:
            from vllm.sampling_params import GuidedDecodingParams
            sp["guided_decoding"] = GuidedDecodingParams(**guided)
        except Exception:
            # Older vLLM may not support this; drop it
            pass

    return SamplingParams(**sp)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_vllm_inference(
    df: pd.DataFrame,
    cfg,
    preprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    postprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    stage_name: str = "vllm_inference",
) -> pd.DataFrame:
    """Run vLLM batch inference on a DataFrame.

    1. Calls preprocess(row_dict) for each row. The preprocessor must set
       ``row["messages"]`` (list of chat dicts) and ``row["sampling_params"]``
       (plain dict).
    2. Builds prompts via ``tokenizer.apply_chat_template``.
    3. Calls ``llm.generate(prompts, sampling_params)`` in one batch.
    4. Sets ``row["generated_text"]`` and usage info, then calls postprocess.
    5. Returns a DataFrame of all postprocessed rows.

    Args:
        df: Input DataFrame (or anything with .to_pandas()).
        cfg: Hydra config with model.model_source, model.engine_kwargs, etc.
        preprocess: Row dict -> row dict with "messages" and "sampling_params".
        postprocess: Row dict (with "generated_text") -> final row dict.
        stage_name: Label for log messages.

    Returns:
        pd.DataFrame of postprocessed results.
    """
    # Handle Ray Dataset or other non-pandas inputs
    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        print(f"[{stage_name}] Converting input to pandas DataFrame...")
        df = df.to_pandas()

    if df is None or len(df) == 0:
        print(f"[{stage_name}] Empty input, returning empty DataFrame")
        return pd.DataFrame()

    # Set NCCL env vars for PCIe GPUs before importing vLLM
    for k, v in get_pcie_nccl_env_vars().items():
        os.environ.setdefault(k, v)

    # Import vLLM after env vars are set
    from vllm import LLM, SamplingParams

    # Build engine kwargs and initialize LLM
    engine_kwargs = _build_engine_kwargs(cfg)
    print(f"[{stage_name}] Initializing vLLM with: { {k: v for k, v in engine_kwargs.items() if k != 'model'} }")
    print(f"[{stage_name}] Model: {engine_kwargs.get('model')}")

    llm = LLM(**engine_kwargs)
    tokenizer = llm.get_tokenizer()

    # Preprocess all rows
    print(f"[{stage_name}] Preprocessing {len(df)} rows...")
    preprocessed_rows: List[Dict[str, Any]] = []
    for row in df.to_dict("records"):
        try:
            preprocessed_rows.append(preprocess(row))
        except Exception as e:
            row["__preprocess_error__"] = str(e)
            preprocessed_rows.append(row)

    # Build prompts from chat messages
    prompts: List[str] = []
    sampling_params_obj = None  # Will be built from first row's params
    first_sp_dict = None

    for row in preprocessed_rows:
        messages = row.get("messages")
        if messages:
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback: simple concatenation
                parts = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    parts.append(f"{role}: {content}")
                prompt = "\n\n".join(parts) + "\n\nAssistant:"
        else:
            prompt = str(row.get("article_text", ""))
        prompts.append(prompt)

        # Capture sampling params from first row (all rows typically share the same)
        if first_sp_dict is None:
            first_sp_dict = row.get("sampling_params", {})

    # Build SamplingParams
    if first_sp_dict:
        sampling_params_obj = _build_sampling_params(first_sp_dict)
    else:
        sampling_params_obj = SamplingParams()

    # Run inference
    print(f"[{stage_name}] Running inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params_obj)

    # Postprocess
    print(f"[{stage_name}] Postprocessing {len(outputs)} outputs...")
    results: List[Dict[str, Any]] = []
    for idx, (row, output) in enumerate(zip(preprocessed_rows, outputs)):
        # Attach generated text
        if output.outputs:
            row["generated_text"] = output.outputs[0].text
        else:
            row["generated_text"] = ""

        # Attach usage info
        try:
            prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0
            completion_tokens = (
                len(output.outputs[0].token_ids) if output.outputs and output.outputs[0].token_ids else 0
            )
            row["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        except Exception:
            row["usage"] = None

        # Run postprocess
        try:
            result = postprocess(row)
        except Exception as e:
            row["__postprocess_error__"] = str(e)
            result = row
        results.append(result)

    print(f"[{stage_name}] Completed inference, {len(results)} results")
    return pd.DataFrame(results)
