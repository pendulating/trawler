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


def get_vllm_runtime_env_vars() -> Dict[str, str]:
    """Return the shared runtime environment for in-process vLLM launches.

    Note: VLLM_USE_V1 and VLLM_ENABLE_V1_MULTIPROCESSING were removed in
    vLLM 0.10+ (V1 is now the only engine). We no longer set them.
    """
    return {
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True",
        "CUDA_LAUNCH_BLOCKING": "0",
    }


def _run_nvidia_smi(args: List[str]) -> List[str]:
    """Run nvidia-smi without importing torch in the parent process."""
    try:
        result = subprocess.run(
            ["nvidia-smi", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def detect_num_gpus() -> int:
    """Detect the number of GPUs available.

    Resolution order:
    1. UAIR_TENSOR_PARALLEL_SIZE env override
    2. CUDA_VISIBLE_DEVICES
    3. SLURM GPU env vars
    4. nvidia-smi -L
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

    # nvidia-smi
    gpu_lines = _run_nvidia_smi(["-L"])
    if gpu_lines:
        return len(gpu_lines)

    return 1


def detect_gpu_type() -> str:
    """Detect GPU type, returning a normalised string like 'rtx_a6000'."""
    names = _run_nvidia_smi(["--query-gpu=name", "--format=csv,noheader,nounits"])
    if names:
        name = names[0].lower()
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
    """Drop engine kwargs not accepted by the installed vLLM LLM class.

    vLLM's LLM.__init__ accepts **kwargs and forwards them to EngineArgs,
    so we check both signatures to build the accepted set.
    """
    try:
        import inspect
        from vllm import LLM as _LLM

        sig = inspect.signature(_LLM.__init__)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

        if has_var_keyword:
            # LLM forwards **kwargs to EngineArgs — check EngineArgs too
            accepted = {k for k in sig.parameters if k != "self"}
            try:
                from vllm.config import EngineArgs
                ea_sig = inspect.signature(EngineArgs.__init__)
                accepted |= {k for k in ea_sig.parameters if k != "self"}
            except ImportError:
                try:
                    from vllm.engine.arg_utils import EngineArgs
                    ea_sig = inspect.signature(EngineArgs.__init__)
                    accepted |= {k for k in ea_sig.parameters if k != "self"}
                except ImportError:
                    # Can't resolve EngineArgs — pass everything through
                    ek = dict(ek)
                    for k in ("concurrency", "batch_size"):
                        ek.pop(k, None)
                    return ek
        else:
            accepted = {k for k in sig.parameters if k != "self"}

        filtered = {k: v for k, v in ek.items() if k in accepted}
        dropped = [k for k in ek if k not in accepted]
        if dropped:
            print(f"[vllm_inference] Dropped unsupported vLLM kwargs: {dropped}")
        return filtered
    except Exception:
        pass
    # Conservative fallback — only drop known non-vLLM keys
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
    if int(ek.get("tensor_parallel_size", 1) or 1) > 1:
        ek.setdefault("disable_custom_all_reduce", True)

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
    """Convert a plain dict to vLLM SamplingParams, handling structured output.

    Supports both vLLM <=0.11 (GuidedDecodingParams) and >=0.12
    (StructuredOutputsParams) APIs transparently.
    """
    from vllm import SamplingParams

    sp = dict(sp_dict or {})

    # Extract guided_decoding / structured_output if present
    guided = sp.pop("guided_decoding", None) or sp.pop("structured_output", None)
    # Remove keys that aren't valid SamplingParams fields
    for k in ("early_stopping", "length_penalty", "response_format",
              "detokenize"):
        sp.pop(k, None)

    if guided and isinstance(guided, dict):
        # vLLM >=0.12: StructuredOutputsParams replaces GuidedDecodingParams
        try:
            from vllm.sampling_params import StructuredOutputsParams
            sp["structured_outputs"] = StructuredOutputsParams(**guided)
        except ImportError:
            # vLLM <=0.11: fall back to GuidedDecodingParams
            try:
                from vllm.sampling_params import GuidedDecodingParams
                sp["guided_decoding"] = GuidedDecodingParams(**guided)
            except ImportError:
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
    3. Calls ``llm.generate(prompts, sampling_params)`` in configurable batches.
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

    # Set runtime env vars before importing vLLM.
    for k, v in {**get_pcie_nccl_env_vars(), **get_vllm_runtime_env_vars()}.items():
        os.environ.setdefault(k, v)

    env_snapshot = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS", "<unset>"),
        "SLURM_GPUS_ON_NODE": os.environ.get("SLURM_GPUS_ON_NODE", "<unset>"),
        "VLLM_WORKER_MULTIPROC_METHOD": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "<unset>"),
    }
    print(f"[{stage_name}] Runtime env: {env_snapshot}")

    # Import vLLM after env vars are set
    from vllm import LLM, SamplingParams

    # Build engine kwargs and initialize LLM
    engine_kwargs = _build_engine_kwargs(cfg)
    print(
        f"[{stage_name}] Initializing vLLM with: "
        f"{ {k: v for k, v in engine_kwargs.items() if k != 'model'} }"
    )
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

    # Build prompts and per-row sampling params.
    # Optimization: when all rows share the same sampling_params dict object
    # (common in stage code), build SamplingParams once and pass it as a
    # single instance — vLLM broadcasts it to all prompts internally.
    prompts: List[str] = []
    sp_objects: List[Any] = []
    _sp_cache: Dict[int, Any] = {}  # id(dict) -> SamplingParams

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

        sp_dict = row.get("sampling_params", {})
        sp_id = id(sp_dict)
        if sp_id not in _sp_cache:
            _sp_cache[sp_id] = _build_sampling_params(sp_dict)
        sp_objects.append(_sp_cache[sp_id])

    if len(_sp_cache) == 1:
        sampling_params_list = sp_objects[0]  # single object, vLLM broadcasts
        print(f"[{stage_name}] Using shared SamplingParams for all {len(prompts)} prompts")
    else:
        sampling_params_list = sp_objects

    try:
        batch_size = int(getattr(cfg.model, "batch_size", 0) or 0)
    except Exception:
        batch_size = 0
    if batch_size <= 0:
        batch_size = len(prompts)

    # Run inference in batches to keep prompt and output memory bounded.
    print(
        f"[{stage_name}] Running inference on {len(prompts)} prompts "
        f"(batch_size={batch_size})..."
    )
    outputs = []
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        prompt_batch = prompts[start:end]
        sampling_batch = sampling_params_list[start:end]
        print(
            f"[{stage_name}] Generating batch {start // batch_size + 1}: "
            f"rows {start}-{end - 1}",
        )
        outputs.extend(llm.generate(prompt_batch, sampling_batch))

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
