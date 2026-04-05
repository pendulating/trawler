"""Shared vLLM direct inference utility.

Replaces the Ray-based build_llm_processor + vLLMEngineProcessorConfig pattern
with direct vLLM LLM.generate() calls. Designed for single-machine multi-GPU setups.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import pandas as pd
from omegaconf import OmegaConf


def _remap_lora_keys_for_vlm(lora_path: str, model_source: str, stage_name: str) -> str:
    """Remap LoRA adapter keys from CausalLM to VLM prefix if needed.

    Adapters trained on AutoModelForCausalLM have keys like
    ``base_model.model.model.layers.X...`` which vLLM parses to
    ``model.layers.X...``.  But VLM architectures (e.g.
    Qwen3_5ForConditionalGeneration) expect ``model.language_model.layers.X...``.

    If the base model uses a ``language_model`` prefix and the adapter does not,
    creates a remapped copy of the adapter in a ``_vlm_remapped/`` subdirectory.
    Returns the (possibly new) lora_path.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    import glob

    # Quick check: does the base model use language_model prefix?
    sf_files = sorted(glob.glob(os.path.join(model_source, "*.safetensors")))
    if not sf_files:
        return lora_path
    with safe_open(sf_files[0], framework="pt") as f:
        base_keys = f.keys()
        has_lm_prefix = any("language_model.layers." in k for k in base_keys)
    if not has_lm_prefix:
        return lora_path

    # Check adapter keys
    adapter_sf = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_sf):
        return lora_path
    with safe_open(adapter_sf, framework="pt") as f:
        adapter_keys = list(f.keys())
    # After vLLM strips "base_model.model.", keys become "model.layers.X...".
    # We need them to be "model.language_model.layers.X..." instead.
    needs_remap = any(
        k.startswith("base_model.model.model.layers.") for k in adapter_keys
    ) and not any(
        "language_model.layers." in k for k in adapter_keys
    )
    if not needs_remap:
        return lora_path

    # Create remapped adapter
    remapped_dir = os.path.join(lora_path, "_vlm_remapped")
    remapped_sf = os.path.join(remapped_dir, "adapter_model.safetensors")
    if os.path.exists(remapped_sf):
        print(f"[{stage_name}] Using cached VLM-remapped LoRA: {remapped_dir}")
        return remapped_dir

    print(f"[{stage_name}] Remapping LoRA keys: model.layers → model.language_model.layers")
    os.makedirs(remapped_dir, exist_ok=True)

    # Remap and save weights
    import torch
    tensors = {}
    with safe_open(adapter_sf, framework="pt") as f:
        for key in f.keys():
            new_key = key.replace(
                "base_model.model.model.layers.",
                "base_model.model.model.language_model.layers.",
            )
            tensors[new_key] = f.get_tensor(key)
    save_file(tensors, remapped_sf)

    # Copy adapter_config.json and other metadata
    import shutil
    for fname in ("adapter_config.json", "tokenizer_config.json", "tokenizer.json",
                  "chat_template.jinja", "README.md"):
        src = os.path.join(lora_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, remapped_dir)

    print(f"[{stage_name}] VLM-remapped LoRA saved to {remapped_dir} "
          f"({len(tensors)} tensors)")
    return remapped_dir


def _fallback_strip_reasoning(text: str) -> str:
    """Fallback regex-based stripping of reasoning/thinking blocks.

    Used only when no family-specific vLLM reasoning parser is available or
    the parser fails. See ``_split_reasoning`` for the primary path.

    Handles multiple formats used by different model families:
    - ``<think>...</think>`` (Qwen3+, DeepSeek-R1, open-source reasoning models)
    - ``<|begin_of_thought|>...<|end_of_thought|>`` (context-reasoner-ppo, some PPO models)

    Also handles unterminated blocks (model ran out of tokens mid-reasoning).
    Returns the remaining text, stripped.
    """
    # <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"<think>[\s\S]*$", "", text)
    # <|begin_of_thought|...end_of_thought|> (with optional trailing ] or >)
    text = re.sub(r"<\|begin_of_thought\|[\s\S]*?<\|end_of_thought\|[>\]\s]*", "", text)
    text = re.sub(r"<\|begin_of_thought\|[\s\S]*$", "", text)
    return text.strip()


# Backwards-compat alias — imported by dagspaces/grpo_training/stages/rewards.py.
# Prefer `_split_reasoning` for new code.
_strip_think_blocks = _fallback_strip_reasoning


def _detect_reasoning_parser(model_source: str) -> Optional[str]:
    """Map a model path to the vLLM reasoning-parser name for that family.

    Returns a parser name registered in ``vllm.reasoning.ReasoningParserManager``,
    or ``None`` for non-thinking families (Phi-4, Llama, Gemma-3, etc.) where
    no reasoning extraction is needed.
    """
    s = (model_source or "").lower()
    # Order matters — check more specific names first.
    if "gemma-4" in s or "gemma4" in s:
        return "gemma4"
    if "gpt-oss" in s:
        return "gptoss"
    if "deepseek-r1" in s or "deepseek_r1" in s or "deepseek-v3" in s:
        return "deepseek_r1"
    if "qwen3" in s:  # covers qwen3, qwen3.5, qwen3-vl, etc.
        return "qwen3"
    # Non-thinking families: Phi-4, Llama-3.x, Gemma-3, Qwen2.5, OpenThinker (custom tags → regex).
    return None


def _split_reasoning(
    text: str,
    model_source: str,
    thinking_enabled: bool,
    tokenizer,
) -> Tuple[str, str]:
    """Split model output into ``(reasoning, content)``.

    Primary path: vLLM's family-specific ``ReasoningParser``. These parsers
    understand the exact reasoning format for each architecture (Qwen3
    ``<think>...</think>``, Gemma-4 ``thought\\n...\\n``, etc.) and are
    maintained upstream alongside each model's chat template.

    Fallback path: regex (``_fallback_strip_reasoning``) when no parser
    matches the model family, the parser fails, or the parser returns
    content that still contains raw reasoning tags.

    Args:
        text: raw decoded model output.
        model_source: path or identifier used to pick a parser.
        thinking_enabled: whether the chat template was configured with
            thinking on — passed to the parser so it classifies truncated
            output correctly (unterminated ``<think>`` is reasoning when
            enabled, content when disabled).
        tokenizer: the tokenizer used for generation (parsers need it).

    Returns:
        ``(reasoning, content)`` — either may be the empty string.
    """
    if not text:
        return "", ""

    parser_name = _detect_reasoning_parser(model_source)
    if parser_name is not None:
        try:
            from vllm.reasoning import ReasoningParserManager
            parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
            parser = parser_cls(
                tokenizer,
                chat_template_kwargs={"enable_thinking": thinking_enabled},
            )
            reasoning, content = parser.extract_reasoning(text, None)
            reasoning = (reasoning or "").strip()
            content = (content or "").strip()
            # Safety: if parser handed back content that still contains raw
            # reasoning tags, something went wrong — fall through to regex.
            if "<think>" not in content and "</think>" not in content:
                return reasoning, content
        except Exception:
            pass  # fall through

    # Fallback path.
    content = _fallback_strip_reasoning(text)
    m = re.search(r"<think>([\s\S]*?)</think>", text)
    if m:
        reasoning = m.group(1).strip()
    else:
        m2 = re.search(r"<\|begin_of_thought\|([\s\S]*?)<\|end_of_thought\|", text)
        reasoning = m2.group(1).strip() if m2 else ""
    return reasoning, content


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
    from omegaconf import OmegaConf as _OC
    _raw_ek = getattr(cfg.model, "engine_kwargs", {})
    ek = _OC.to_container(_raw_ek, resolve=True) if _OC.is_config(_raw_ek) else dict(_raw_ek)

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

    # Convert nested hf_overrides dicts to a callable that does deep updates.
    # vLLM's config.update() with a dict like {"text_config": {"vocab_size": X}}
    # replaces text_config entirely instead of merging.  A callable gets the
    # PretrainedConfig object and can update nested attributes properly.
    _hf_ov = ek.get("hf_overrides")
    if isinstance(_hf_ov, dict) and any(isinstance(v, dict) for v in _hf_ov.values()):
        def _make_hf_override_fn(overrides):
            def _fn(config):
                for key, val in overrides.items():
                    if isinstance(val, dict) and hasattr(config, key):
                        sub = getattr(config, key)
                        for sk, sv in val.items():
                            setattr(sub, sk, sv)
                    else:
                        setattr(config, key, val)
                return config
            return _fn
        ek["hf_overrides"] = _make_hf_override_fn(_hf_ov)

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

_DP_WORKER_SCRIPT = r'''
"""Standalone DP worker script — executed as a fresh subprocess.

Reads task from a pickle file, runs vLLM inference, writes results to a
pickle file.  Completely isolated from the parent process (no shared CUDA
context, no inherited NCCL state).
"""
import json, os, pickle, sys, time, traceback

def main():
    task_path = sys.argv[1]
    result_path = sys.argv[2]

    with open(task_path, "rb") as f:
        task = pickle.load(f)

    rank        = task["rank"]
    dp_size     = task["dp_size"]
    engine_kwargs = task["engine_kwargs"]
    prompts     = task["prompts"]
    sp_dict     = task["sp_dict"]
    stage_name  = task["stage_name"]
    pcie_env    = task["pcie_env"]
    runtime_env = task["runtime_env"]

    # Apply env vars (set before any CUDA/torch import)
    for k, v in {**pcie_env, **runtime_env}.items():
        os.environ.setdefault(k, v)

    # Clear any inherited vLLM DP coordination vars
    for var in ("VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE",
                "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT"):
        os.environ.pop(var, None)

    print(f"[{stage_name}] DP rank {rank}/{dp_size}: starting "
          f"(pid={os.getpid()}, CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
          f"prompts={len(prompts)})", flush=True)

    try:
        t0 = time.time()
        from vllm import LLM, SamplingParams

        print(f"[{stage_name}] DP rank {rank}/{dp_size}: vLLM imported in "
              f"{time.time() - t0:.1f}s, creating LLM engine...", flush=True)

        t1 = time.time()
        llm = LLM(**engine_kwargs)
        print(f"[{stage_name}] DP rank {rank}/{dp_size}: LLM created in "
              f"{time.time() - t1:.1f}s, starting generation...", flush=True)

        # Build SamplingParams (inline to avoid import from parent package)
        sp = dict(sp_dict or {})
        guided = sp.pop("guided_decoding", None) or sp.pop("structured_output", None)
        for k in ("early_stopping", "length_penalty", "response_format", "detokenize"):
            sp.pop(k, None)
        if guided and isinstance(guided, dict):
            try:
                from vllm.sampling_params import StructuredOutputsParams
                sp["structured_outputs"] = StructuredOutputsParams(**guided)
            except ImportError:
                try:
                    from vllm.sampling_params import GuidedDecodingParams
                    sp["guided_decoding"] = GuidedDecodingParams(**guided)
                except ImportError:
                    pass
        sampling_params = SamplingParams(**sp)

        t2 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        print(f"[{stage_name}] DP rank {rank}/{dp_size}: generation done in "
              f"{time.time() - t2:.1f}s ({len(outputs)} outputs)", flush=True)

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

        with open(result_path, "wb") as f:
            pickle.dump({"rank": rank, "outputs": serialised, "error": None}, f)

        print(f"[{stage_name}] DP rank {rank}/{dp_size}: wrote {len(serialised)} "
              f"results, total elapsed {time.time() - t0:.1f}s", flush=True)

    except Exception:
        tb = traceback.format_exc()
        print(f"[{stage_name}] DP rank {rank}/{dp_size}: FAILED\n{tb}",
              flush=True, file=sys.stderr)
        with open(result_path, "wb") as f:
            pickle.dump({"rank": rank, "outputs": None, "error": tb}, f)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''


def _run_data_parallel(
    engine_kwargs: Dict[str, Any],
    dp_size: int,
    prompts: List[str],
    sp_dict: Dict[str, Any],
    stage_name: str,
    timeout: int = 86400,
) -> List[Dict[str, Any]]:
    """Spawn dp_size fully-isolated subprocess workers for vLLM inference.

    Each worker is a fresh Python interpreter (subprocess.Popen) with its own
    CUDA_VISIBLE_DEVICES slice.  This avoids the NCCL deadlocks that occur
    when multiprocessing.Process is used with vLLM's ``mp`` executor backend
    (inherited CUDA context + competing NCCL process groups).

    Returns a list of output dicts in the same order as ``prompts``.
    """
    if len(prompts) < dp_size:
        raise RuntimeError(
            f"[{stage_name}] Too few prompts ({len(prompts)}) for "
            f"data_parallel_size={dp_size}."
        )

    tp_size = engine_kwargs.get("tensor_parallel_size", 1)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    all_devices = [d.strip() for d in visible.split(",") if d.strip()] if visible else []
    needed = dp_size * tp_size
    if all_devices and len(all_devices) < needed:
        raise RuntimeError(
            f"[{stage_name}] data_parallel_size={dp_size} x "
            f"tensor_parallel_size={tp_size} = {needed} GPUs required, "
            f"but only {len(all_devices)} visible: {all_devices}"
        )

    # Split prompts across DP ranks
    floor_n = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def shard_range(rank: int) -> tuple:
        start = rank * floor_n + min(rank, remainder)
        end = (rank + 1) * floor_n + min(rank + 1, remainder)
        return start, end

    shards = []
    for r in range(dp_size):
        s, e = shard_range(r)
        shards.append(prompts[s:e])

    # Prepare per-rank env and task files
    pcie_env = get_pcie_nccl_env_vars()
    runtime_env = get_vllm_runtime_env_vars()

    tmpdir = os.environ.get("TMPDIR", "/tmp")
    task_files = []
    result_files = []
    procs: List[subprocess.Popen] = []

    # Write worker script to a temp file
    worker_script = tempfile.NamedTemporaryFile(
        mode="w", suffix="_dp_worker.py", dir=tmpdir, delete=False,
    )
    worker_script.write(_DP_WORKER_SCRIPT)
    worker_script.close()

    print(f"[{stage_name}] Launching {dp_size} DP workers "
          f"(TP={tp_size}, {len(prompts)} total prompts)...", flush=True)

    for rank in range(dp_size):
        # GPU slice for this rank
        if all_devices:
            rank_devices = all_devices[rank * tp_size:(rank + 1) * tp_size]
        else:
            rank_devices = []

        task = {
            "rank": rank,
            "dp_size": dp_size,
            "engine_kwargs": engine_kwargs,
            "prompts": shards[rank],
            "sp_dict": sp_dict,
            "stage_name": stage_name,
            "pcie_env": pcie_env,
            "runtime_env": runtime_env,
        }

        task_path = os.path.join(tmpdir, f"{stage_name}_dp{rank}_task.pkl")
        result_path = os.path.join(tmpdir, f"{stage_name}_dp{rank}_result.pkl")
        with open(task_path, "wb") as f:
            pickle.dump(task, f)
        task_files.append(task_path)
        result_files.append(result_path)

        # Build a clean env: inherit parent env, override GPU assignment,
        # and strip any vLLM DP coordination vars.
        child_env = dict(os.environ)
        if rank_devices:
            child_env["CUDA_VISIBLE_DEVICES"] = ",".join(rank_devices)
        for var in ("VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE",
                     "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT"):
            child_env.pop(var, None)

        devices_str = child_env.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(f"[{stage_name}] DP rank {rank}: {len(shards[rank])} prompts, "
              f"CUDA_VISIBLE_DEVICES={devices_str}", flush=True)

        proc = subprocess.Popen(
            [sys.executable, worker_script.name, task_path, result_path],
            env=child_env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        procs.append(proc)

    # Wait for all workers
    print(f"[{stage_name}] Waiting for {dp_size} DP workers "
          f"(timeout={timeout}s)...", flush=True)
    errors = []
    for rank, proc in enumerate(procs):
        try:
            retcode = proc.wait(timeout=timeout)
            if retcode != 0:
                errors.append(
                    f"DP rank {rank} (pid={proc.pid}) exited with code {retcode}"
                )
        except subprocess.TimeoutExpired:
            proc.kill()
            errors.append(
                f"DP rank {rank} (pid={proc.pid}) timed out after {timeout}s, killed"
            )

    # Collect results
    rank_results: Dict[int, List[Dict[str, Any]]] = {}
    for rank in range(dp_size):
        result_path = result_files[rank]
        if not os.path.exists(result_path):
            errors.append(f"DP rank {rank}: no result file at {result_path}")
            continue
        with open(result_path, "rb") as f:
            result = pickle.load(f)
        if result.get("error"):
            errors.append(f"DP rank {rank} failed:\n{result['error']}")
        else:
            rank_results[rank] = result["outputs"]

    # Cleanup temp files
    for p in [worker_script.name, *task_files, *result_files]:
        try:
            os.unlink(p)
        except OSError:
            pass

    if errors:
        raise RuntimeError(
            f"[{stage_name}] Data-parallel inference failed:\n"
            + "\n".join(errors)
        )

    # Reassemble in original order
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
# Transformers fallback for models vLLM no longer supports (e.g. Mllama)
# ---------------------------------------------------------------------------

# Model families that bypass vLLM entirely and use transformers generate().
_TRANSFORMERS_FALLBACK_FAMILIES = {"llama-vision"}


def _run_transformers_text_inference(
    df: pd.DataFrame,
    cfg,
    preprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    postprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    stage_name: str = "transformers_inference",
) -> pd.DataFrame:
    """Text-only inference via native transformers for unsupported vLLM models.

    Mirrors the run_vllm_inference interface (preprocess/postprocess callables)
    but uses AutoModelForCausalLM.generate() instead of vLLM.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_source = str(cfg.model.model_source)
    print(f"[{stage_name}] Using native transformers fallback for {model_source}")

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation

    # Use flash_attention_2 if available for ~2x speedup on long sequences
    try:
        import flash_attn  # noqa: F401
        _attn_impl = "flash_attention_2"
    except ImportError:
        _attn_impl = "sdpa"
    print(f"[{stage_name}] Attention: {_attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,  # Mllama has intentional embed/lm_head size diffs
        attn_implementation=_attn_impl,
    )

    # Load and merge LoRA adapter if specified
    _lora_path = str(getattr(cfg.model, "lora_path", "") or "")
    if _lora_path:
        from peft import PeftModel
        print(f"[{stage_name}] Loading LoRA adapter: {_lora_path}")
        model = PeftModel.from_pretrained(model, _lora_path)
        model = model.merge_and_unload()
        print(f"[{stage_name}] LoRA merged into base model")

    model.eval()

    # Determine think-block stripping
    _strip_thinking = False
    try:
        ctk = getattr(cfg.model, "chat_template_kwargs", None) or {}
        if hasattr(ctk, "enable_thinking"):
            _strip_thinking = not bool(ctk.enable_thinking)
        elif isinstance(ctk, dict):
            _strip_thinking = not bool(ctk.get("enable_thinking", True))
    except Exception:
        pass

    # Preprocess rows
    print(f"[{stage_name}] Preprocessing {len(df)} rows...")
    preprocessed_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(df.to_dict("records")):
        try:
            preprocessed_rows.append(preprocess(row))
        except Exception as e:
            row["__preprocess_error__"] = str(e)
            preprocessed_rows.append(row)

    # Separate valid rows from failed ones
    valid_indices = []
    failed_rows = []
    prompts = []
    sp_first = {}
    for i, row in enumerate(preprocessed_rows):
        if "__preprocess_error__" in row:
            row["generated_text"] = ""
            row["generated_reasoning"] = ""
            failed_rows.append((i, postprocess(row)))
        else:
            messages = row.get("messages", [])
            sp = row.get("sampling_params", {})
            if not sp_first:
                sp_first = sp  # sampling params are same for all rows
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
            valid_indices.append(i)

    max_tokens = int(sp_first.get("max_tokens", 1024))
    temperature = float(sp_first.get("temperature", 0.0))
    do_sample = temperature > 0

    # Batch inference — Mllama 11B in bf16 ≈ 22GB, leaving ~26GB on A6000 for
    # KV cache.  batch_size=16 with max_tokens=1024 fits comfortably.
    batch_size = 16
    generated_texts: List[str] = []
    generated_reasonings: List[str] = []
    _thinking_enabled = not _strip_thinking
    print(f"[{stage_name}] Generating {len(prompts)} prompts in batches of {batch_size}...")

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=8192,
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
        for j in range(len(batch_prompts)):
            gen_ids = output_ids[j, prompt_len:]
            raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            reasoning, content = _split_reasoning(
                raw_text, model_source, _thinking_enabled, tokenizer,
            )
            generated_texts.append(content)
            generated_reasonings.append(reasoning)

        print(f"[{stage_name}] Batch {start // batch_size + 1}: "
              f"{min(start + batch_size, len(prompts))}/{len(prompts)} done")

    # Reassemble results in original order
    results = [None] * len(preprocessed_rows)
    for i, row_data in failed_rows:
        results[i] = row_data
    for vi, gen_text, gen_reasoning in zip(valid_indices, generated_texts, generated_reasonings):
        row = preprocessed_rows[vi]
        row["generated_text"] = gen_text
        row["generated_reasoning"] = gen_reasoning
        results[vi] = postprocess(row)

    print(f"[{stage_name}] Completed transformers inference, {len(results)} results")
    return pd.DataFrame(results)


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

    # Models whose architectures vLLM no longer supports get routed to a
    # native transformers generate() fallback.
    _model_family = str(getattr(cfg.model, "model_family", ""))
    if _model_family in _TRANSFORMERS_FALLBACK_FAMILIES:
        return _run_transformers_text_inference(
            df=df, cfg=cfg, preprocess=preprocess, postprocess=postprocess,
            stage_name=stage_name,
        )

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
        # Remap adapter keys if needed (CausalLM → VLM prefix mismatch)
        model_source = engine_kwargs.get("model", "")
        lora_path = _remap_lora_keys_for_vlm(lora_path, model_source, stage_name)
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("sft", 1, lora_path)
        print(f"[{stage_name}] LoRA adapter: {lora_path}")

    # Determine thinking mode for reasoning extraction.
    # The chat template flag controls *what prompt* the model sees; reasoning
    # is always extracted from output via `_split_reasoning` regardless.
    # We still read `enable_thinking` to tell the parser whether truncated
    # output (no closing tag) should be classified as reasoning or content.
    _strip_thinking = False
    try:
        ctk = getattr(cfg.model, "chat_template_kwargs", None) or {}
        if hasattr(ctk, "enable_thinking"):
            _strip_thinking = not bool(ctk.enable_thinking)
        elif isinstance(ctk, dict):
            _strip_thinking = not bool(ctk.get("enable_thinking", True))
    except Exception:
        pass
    _thinking_enabled = not _strip_thinking
    _model_source = str(engine_kwargs.get("model", "") or "")
    _parser_name = _detect_reasoning_parser(_model_source)
    print(f"[{stage_name}] Reasoning extraction: parser={_parser_name or 'regex-fallback'}, "
          f"thinking_enabled={_thinking_enabled}")

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
    preliminary_valid = [i for i in range(len(preprocessed_rows)) if i not in failed_set]

    # Determine model's max context length for prompt validation.
    _max_model_len = None
    if llm is not None:
        try:
            _max_model_len = llm.llm_engine.model_config.max_model_len
        except Exception:
            pass
    if _max_model_len is None:
        try:
            _max_model_len = int(getattr(tokenizer, "model_max_length", 0) or 0)
            if _max_model_len <= 0 or _max_model_len > 1_000_000:
                _max_model_len = None
        except Exception:
            pass

    # Build prompts and sampling params dicts for valid rows only.
    prompts: List[str] = []
    sp_dicts: List[Dict[str, Any]] = []
    valid_indices: List[int] = []
    _oversized_count = 0

    for i in preliminary_valid:
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

        # Validate prompt length against model context window
        if _max_model_len is not None:
            sp = row.get("sampling_params", {})
            max_new = int(sp.get("max_tokens", 0) or 0)
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if prompt_tokens + max(max_new, 1) > _max_model_len:
                row["__preprocess_error__"] = (
                    f"Prompt too long: {prompt_tokens} tokens + "
                    f"{max_new} max_tokens > {_max_model_len} context limit"
                )
                failed_set.add(i)
                _oversized_count += 1
                continue

        prompts.append(prompt)
        sp_dicts.append(row.get("sampling_params", {}))
        valid_indices.append(i)

    if _oversized_count:
        print(
            f"[{stage_name}] WARNING: {_oversized_count} prompts exceed model "
            f"context length ({_max_model_len}) and will be skipped"
        )

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
                row["generated_reasoning"] = ""
                try:
                    result = postprocess(row)
                except Exception as e:
                    row["__postprocess_error__"] = str(e)
                    result = row
                results.append(result)
                continue

            out = next(output_iter)
            raw_text = out.get("generated_text", "")
            reasoning, content = _split_reasoning(
                raw_text, _model_source, _thinking_enabled, tokenizer,
            )
            row["generated_text"] = content
            row["generated_reasoning"] = reasoning
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
                row["generated_reasoning"] = ""
                try:
                    result = postprocess(row)
                except Exception as e:
                    row["__postprocess_error__"] = str(e)
                    result = row
                results.append(result)
                continue

            output = next(output_iter)
            if output.outputs:
                raw_text = output.outputs[0].text
                reasoning, content = _split_reasoning(
                    raw_text, _model_source, _thinking_enabled, tokenizer,
                )
                row["generated_text"] = content
                row["generated_reasoning"] = reasoning
            else:
                row["generated_text"] = ""
                row["generated_reasoning"] = ""

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
