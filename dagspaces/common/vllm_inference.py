"""Shared vLLM direct inference utility.

Replaces the Ray-based build_llm_processor + vLLMEngineProcessorConfig pattern
with direct vLLM LLM.generate() calls. Designed for single-machine multi-GPU setups.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from omegaconf import OmegaConf


def _strip_think_blocks(text: str) -> str:
    """Strip ``<think>...</think>`` reasoning blocks from model output.

    Handles both complete blocks and unterminated ``<think>`` (when the model
    ran out of tokens mid-reasoning).  Returns the remaining text, stripped.
    """
    # Complete blocks
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    # Unterminated block (model ran out of tokens while thinking)
    text = re.sub(r"<think>[\s\S]*$", "", text)
    return text.strip()


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

    # Preserve data_parallel_size (our key, not a vLLM LLM kwarg) before filtering
    dp_size = ek.pop("data_parallel_size", None)

    # Filter to accepted kwargs
    ek = filter_vllm_engine_kwargs(ek)

    # Re-attach data_parallel_size for run_vllm_inference to consume
    if dp_size is not None:
        ek["data_parallel_size"] = dp_size

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
# Data-parallel worker
# ---------------------------------------------------------------------------

def _dp_worker(
    rank: int,
    dp_size: int,
    master_ip: str,
    master_port: int,
    engine_kwargs: Dict[str, Any],
    prompts: List[str],
    sp_dict: Dict[str, Any],
    result_queue: multiprocessing.Queue,
    stage_name: str,
):
    """Run vLLM inference in a data-parallel worker process.

    Each worker sets DP env vars, creates its own LLM instance on a subset
    of GPUs, runs generate() on its shard of prompts, and puts serialisable
    results into `result_queue`.
    """
    try:
        os.environ["VLLM_DP_RANK"] = str(rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = master_ip
        os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

        # Set runtime env vars
        for k, v in {**get_pcie_nccl_env_vars(), **get_vllm_runtime_env_vars()}.items():
            os.environ.setdefault(k, v)

        from vllm import LLM

        print(f"[{stage_name}] DP rank {rank}/{dp_size}: creating LLM with "
              f"{len(prompts)} prompts", flush=True)
        llm = LLM(**engine_kwargs)

        sampling_params = _build_sampling_params(sp_dict)
        outputs = llm.generate(prompts, sampling_params)

        # Serialise outputs: extract text and token counts
        serialised = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            prompt_tokens = len(out.prompt_token_ids) if out.prompt_token_ids else 0
            completion_tokens = (
                len(out.outputs[0].token_ids)
                if out.outputs and out.outputs[0].token_ids else 0
            )
            serialised.append({
                "generated_text": text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })

        print(f"[{stage_name}] DP rank {rank}/{dp_size}: done, "
              f"{len(serialised)} outputs", flush=True)
        result_queue.put((rank, serialised, None))

        # Give engine time to pause processing loops before exit
        # (matches official vLLM DP example)
        import time
        time.sleep(1)
    except Exception as e:
        import traceback
        result_queue.put((rank, None, traceback.format_exc()))


def _run_data_parallel(
    engine_kwargs: Dict[str, Any],
    dp_size: int,
    prompts: List[str],
    sp_dict: Dict[str, Any],
    stage_name: str,
    timeout: int = 86400,
) -> List[Dict[str, Any]]:
    """Spawn dp_size worker processes, each running vLLM on a prompt shard.

    Returns a list of output dicts in the same order as `prompts`.
    """
    if len(prompts) < dp_size:
        raise RuntimeError(
            f"[{stage_name}] Too few prompts ({len(prompts)}) for "
            f"data_parallel_size={dp_size}. Each DP rank needs at least 1 "
            f"prompt. Reduce dp_size or increase input data."
        )

    from vllm.utils.network_utils import get_open_port

    master_ip = "127.0.0.1"
    master_port = get_open_port()

    # Split prompts across DP ranks
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def shard_start(rank):
        return rank * floor + min(rank, remainder)

    shards = []
    for r in range(dp_size):
        s = prompts[shard_start(r):shard_start(r + 1)]
        shards.append(s)
        print(f"[{stage_name}] DP rank {r}: {len(s)} prompts")

    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    procs = []
    for rank in range(dp_size):
        proc = multiprocessing.Process(
            target=_dp_worker,
            args=(rank, dp_size, master_ip, master_port,
                  engine_kwargs, shards[rank], sp_dict, result_queue, stage_name),
        )
        proc.start()
        procs.append(proc)

    # Collect results
    rank_results: Dict[int, List[Dict[str, Any]]] = {}
    errors = []
    for _ in range(dp_size):
        rank, outputs, error = result_queue.get(timeout=timeout)
        if error:
            errors.append(f"DP rank {rank} failed:\n{error}")
        else:
            rank_results[rank] = outputs

    for proc in procs:
        proc.join(timeout=30)
        if proc.is_alive():
            print(f"[{stage_name}] Killing hung DP process {proc.pid}")
            proc.kill()

    if errors:
        raise RuntimeError(
            f"[{stage_name}] Data-parallel inference failed:\n"
            + "\n".join(errors)
        )

    # Reassemble in original order (rank 0 first, then rank 1, etc.)
    all_outputs = []
    for rank in range(dp_size):
        results = rank_results[rank]
        if len(results) != len(shards[rank]):
            raise RuntimeError(
                f"[{stage_name}] DP rank {rank} output count mismatch: "
                f"expected {len(shards[rank])}, got {len(results)}"
            )
        all_outputs.extend(results)

    return all_outputs


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

    # Build engine kwargs
    engine_kwargs = _build_engine_kwargs(cfg)

    # Check for LoRA adapter path in model config
    lora_path = None
    lora_request = None
    try:
        lora_path = str(OmegaConf.select(cfg, "model.lora_path") or "")
    except Exception as e:
        print(f"[{stage_name}] LoRA path lookup via OmegaConf.select failed: {e}")
    # Fallback: direct attribute access (handles plain dicts and DictConfig)
    if not lora_path:
        try:
            lora_path = str(getattr(cfg.model, "lora_path", "") or "")
        except Exception as e:
            print(f"[{stage_name}] LoRA path fallback failed: {e}")
    print(f"[{stage_name}] LoRA path resolved: {repr(lora_path)}")
    if lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("sft", 1, lora_path)
        print(f"[{stage_name}] LoRA adapter: {lora_path}")

    # Determine whether to strip <think> blocks from outputs.
    # Strip when enable_thinking is explicitly False — the model may still
    # produce think blocks if its chat template doesn't support the flag.
    _strip_thinking = False
    try:
        ctk = getattr(cfg.model, "chat_template_kwargs", None) or {}
        if hasattr(ctk, "enable_thinking"):
            _strip_thinking = not bool(ctk.enable_thinking)
        elif isinstance(ctk, dict):
            _strip_thinking = not bool(ctk.get("enable_thinking", True))
    except Exception:
        pass
    if _strip_thinking:
        print(f"[{stage_name}] Thinking block stripping enabled (enable_thinking=false)")

    # Check for data parallelism
    dp_size = int(engine_kwargs.pop("data_parallel_size", 1) or 1)

    print(
        f"[{stage_name}] Initializing vLLM with: "
        f"{ {k: v for k, v in engine_kwargs.items() if k != 'model'} }"
    )
    print(f"[{stage_name}] Model: {engine_kwargs.get('model')}")
    if dp_size > 1:
        print(f"[{stage_name}] Data parallelism enabled: {dp_size} replicas "
              f"x TP={engine_kwargs.get('tensor_parallel_size', 1)}")

    # Load tokenizer without full LLM for prompt templating.
    # For DP mode we can't create the LLM in the parent process; for
    # single-process mode we create LLM first and reuse its tokenizer.
    if dp_size > 1:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            engine_kwargs["model"], trust_remote_code=True
        )
        llm = None  # LLM created in worker processes
    else:
        llm = LLM(**engine_kwargs)
        tokenizer = llm.get_tokenizer()

    # Preprocess all rows
    print(f"[{stage_name}] Preprocessing {len(df)} rows...")
    preprocessed_rows: List[Dict[str, Any]] = []
    failed_indices: List[int] = []  # indices of preprocess-failed rows
    for idx, row in enumerate(df.to_dict("records")):
        try:
            preprocessed_rows.append(preprocess(row))
        except Exception as e:
            row["__preprocess_error__"] = str(e)
            print(f"[{stage_name}] Preprocess error on row {idx}: {e}")
            preprocessed_rows.append(row)
            failed_indices.append(idx)

    if failed_indices:
        print(f"[{stage_name}] WARNING: {len(failed_indices)} rows failed "
              f"preprocessing and will be skipped for inference")

    # Separate valid rows from failed ones for inference
    failed_set = set(failed_indices)
    valid_indices = [i for i in range(len(preprocessed_rows)) if i not in failed_set]

    # Build prompts and sampling params dicts for valid rows only.
    prompts: List[str] = []
    sp_dicts: List[Dict[str, Any]] = []

    for i in valid_indices:
        row = preprocessed_rows[i]
        messages = row.get("messages")
        if messages:
            try:
                chat_template_kwargs = dict(
                    getattr(cfg.model, "chat_template_kwargs", {}) or {}
                )
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    **chat_template_kwargs,
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
        sp_dicts.append(row.get("sampling_params", {}))

    # -----------------------------------------------------------------------
    # Inference: data-parallel or single-process
    # -----------------------------------------------------------------------
    if dp_size > 1:
        # All rows share the same sampling_params in our pipeline stages
        sp_dict = sp_dicts[0] if sp_dicts else {}
        print(f"[{stage_name}] Running data-parallel inference: "
              f"{len(prompts)} prompts across {dp_size} replicas...")
        dp_outputs = _run_data_parallel(
            engine_kwargs=engine_kwargs,
            dp_size=dp_size,
            prompts=prompts,
            sp_dict=sp_dict,
            stage_name=stage_name,
        )

        # Verify output count
        if len(dp_outputs) != len(prompts):
            raise RuntimeError(
                f"[{stage_name}] DP output count mismatch: "
                f"expected {len(prompts)}, got {len(dp_outputs)}"
            )

        # Postprocess — merge DP outputs back with failed rows
        print(f"[{stage_name}] Postprocessing {len(dp_outputs)} outputs...")
        results: List[Dict[str, Any]] = []
        output_iter = iter(dp_outputs)
        for idx, row in enumerate(preprocessed_rows):
            if idx in failed_set:
                row["generated_text"] = ""
                try:
                    result = postprocess(row)
                except Exception as e:
                    row["__postprocess_error__"] = str(e)
                    result = row
                results.append(result)
                continue

            out = next(output_iter)
            gen_text = out.get("generated_text", "")
            if _strip_thinking and gen_text:
                gen_text = _strip_think_blocks(gen_text)
            row["generated_text"] = gen_text
            row["usage"] = {
                "prompt_tokens": out.get("prompt_tokens", 0),
                "completion_tokens": out.get("completion_tokens", 0),
                "total_tokens": out.get("prompt_tokens", 0) + out.get("completion_tokens", 0),
            }
            try:
                result = postprocess(row)
            except Exception as e:
                row["__postprocess_error__"] = str(e)
                result = row
            results.append(result)

    else:
        # Single-process path (original behaviour)
        # Build SamplingParams objects with dedup optimization
        sp_objects: List[Any] = []
        _sp_cache: Dict[int, Any] = {}  # id(dict) -> SamplingParams
        for sp_dict in sp_dicts:
            sp_id = id(sp_dict)
            if sp_id not in _sp_cache:
                _sp_cache[sp_id] = _build_sampling_params(sp_dict)
            sp_objects.append(_sp_cache[sp_id])

        if len(_sp_cache) == 1 and sp_objects:
            sampling_params_list = sp_objects[0]  # single object, vLLM broadcasts
            print(f"[{stage_name}] Using shared SamplingParams for all {len(prompts)} prompts")
        else:
            sampling_params_list = sp_objects

        try:
            batch_size = int(getattr(cfg.model, "batch_size", 0) or 0)
        except Exception:
            batch_size = 0
        if batch_size <= 0:
            batch_size = max(len(prompts), 1)

        # Run inference in batches
        print(
            f"[{stage_name}] Running inference on {len(prompts)} prompts "
            f"(batch_size={batch_size})..."
        )
        outputs = []
        shared_sp = sampling_params_list if not isinstance(sampling_params_list, list) else None
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            prompt_batch = prompts[start:end]
            sampling_batch = shared_sp if shared_sp else sampling_params_list[start:end]
            print(
                f"[{stage_name}] Generating batch {start // batch_size + 1}: "
                f"rows {start}-{end - 1}",
            )
            outputs.extend(llm.generate(prompt_batch, sampling_batch, lora_request=lora_request))

        # Verify output count matches input count
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"[{stage_name}] vLLM output count mismatch: "
                f"expected {len(prompts)} outputs for {len(prompts)} prompts, "
                f"got {len(outputs)}. This indicates silent data loss."
            )

        # Postprocess — merge inference outputs back with failed rows
        print(f"[{stage_name}] Postprocessing {len(outputs)} outputs...")
        results: List[Dict[str, Any]] = []
        output_iter = iter(outputs)
        for idx, row in enumerate(preprocessed_rows):
            if idx in failed_set:
                row["generated_text"] = ""
                try:
                    result = postprocess(row)
                except Exception as e:
                    row["__postprocess_error__"] = str(e)
                    result = row
                results.append(result)
                continue

            output = next(output_iter)
            if output.outputs:
                gen_text = output.outputs[0].text
                if _strip_thinking and gen_text:
                    gen_text = _strip_think_blocks(gen_text)
                row["generated_text"] = gen_text
            else:
                row["generated_text"] = ""

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

            try:
                result = postprocess(row)
            except Exception as e:
                row["__postprocess_error__"] = str(e)
                result = row
            results.append(result)

    print(f"[{stage_name}] Completed inference, {len(results)} results")
    return pd.DataFrame(results)
