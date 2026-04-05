"""SFT training stage: LoRA fine-tuning via TRL's SFTTrainer.

Takes SFT pairs (in chat messages format) and fine-tunes a base model
using parameter-efficient LoRA adaptation.

Multi-GPU support:
  - Uses `accelerate launch` for DDP data-parallel training across GPUs.
    The runner spawns this as a subprocess so each GPU gets its own process.
  - When LOCAL_RANK is set (i.e. inside accelerate), the Trainer handles
    device placement automatically.
"""

import json
import os
import time
from typing import Any, Dict, List

import pandas as pd
from omegaconf import OmegaConf



# TRL-compatible chat templates with {% generation %} blocks for loss masking.
# Each template marks assistant content so SFTTrainer only computes loss on
# completion tokens. The model family is detected from the tokenizer's native
# template or config architecture.

def _qwen_sft_template(thinking_enabled: bool) -> str:
    """Build the Qwen SFT chat template.

    When ``thinking_enabled=False`` (no-think training), the assistant
    generation block is prefixed with the empty-think sentinel
    ``<think>\\n\\n</think>\\n`` — the Qwen3 official recipe that teaches
    the model "emit empty-think, then answer" at inference time.

    When ``thinking_enabled=True``, no prefix is injected; the model is
    free to emit its own ``<think>...</think>`` block before the content.
    Note: unless the SFT dataset contains reasoning traces, this won't
    actively *train* reasoning — it merely leaves the pretrained
    reasoning distribution untouched.
    """
    prefix = "<think>\n\n</think>\n" if not thinking_enabled else ""
    return (
        "{%- for message in messages %}"
        "{%- if message.role == 'system' %}"
        "<|im_start|>system\n{{ message.content | trim }}<|im_end|>\n"
        "{%- elif message.role == 'user' %}"
        "<|im_start|>user\n{{ message.content | trim }}<|im_end|>\n"
        "{%- elif message.role == 'assistant' %}"
        "<|im_start|>assistant\n"
        "{% generation %}" + prefix + "{{ message.content | trim }}{% endgeneration %}"
        "<|im_end|>\n"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{%- endif %}"
    )


_SFT_TEMPLATES = {
    # Qwen family: <|im_start|>role\ncontent<|im_end|>
    # Preserved as a no-think template for backwards compat; use
    # ``_qwen_sft_template(thinking_enabled=...)`` to build either variant.
    "qwen": _qwen_sft_template(thinking_enabled=False),
    # Phi-4: <|im_start|>role<|im_sep|>content<|im_end|>
    "phi-4": (
        "{%- for message in messages %}"
        "{%- if message.role == 'system' %}"
        "<|im_start|>system<|im_sep|>{{ message.content | trim }}<|im_end|>"
        "{%- elif message.role == 'user' %}"
        "<|im_start|>user<|im_sep|>{{ message.content | trim }}<|im_end|>"
        "{%- elif message.role == 'assistant' %}"
        "<|im_start|>assistant<|im_sep|>"
        "{% generation %}{{ message.content | trim }}{% endgeneration %}"
        "<|im_end|>"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "<|im_start|>assistant<|im_sep|>"
        "{%- endif %}"
    ),
    # Phi-4-multimodal: <|role|>content<|end|>
    "phi-4-mm": (
        "{%- for message in messages %}"
        "{%- if message.role == 'system' %}"
        "<|system|>{{ message.content | trim }}<|end|>"
        "{%- elif message.role == 'user' %}"
        "<|user|>{{ message.content | trim }}<|end|>"
        "{%- elif message.role == 'assistant' %}"
        "<|assistant|>"
        "{% generation %}{{ message.content | trim }}{% endgeneration %}"
        "<|end|>"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "<|assistant|>"
        "{%- endif %}"
    ),
    # Gemma-3: <start_of_turn>role\ncontent<end_of_turn>
    # System message is prepended to first user message.
    "gemma": (
        "{{ bos_token }}"
        "{%- set system_message = '' %}"
        "{%- for message in messages %}"
        "{%- if message.role == 'system' %}"
        "{%- set system_message = message.content | trim + '\n\n' %}"
        "{%- elif message.role == 'user' %}"
        "<start_of_turn>user\n"
        "{%- if loop.first and system_message %}{{ system_message }}{%- endif %}"
        "{{ message.content | trim }}<end_of_turn>\n"
        "{%- elif message.role == 'assistant' %}"
        "<start_of_turn>model\n"
        "{% generation %}{{ message.content | trim }}{% endgeneration %}"
        "<end_of_turn>\n"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "<start_of_turn>model\n"
        "{%- endif %}"
    ),
    # Llama-3: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
    "llama": (
        "{{ bos_token }}"
        "{%- for message in messages %}"
        "{%- if message.role == 'system' %}"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "{{ message.content | trim }}<|eot_id|>"
        "{%- elif message.role == 'user' %}"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "{{ message.content | trim }}<|eot_id|>"
        "{%- elif message.role == 'assistant' %}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{% generation %}{{ message.content | trim }}{% endgeneration %}"
        "<|eot_id|>"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{%- endif %}"
    ),
    # GPT-OSS-20B: <|start|>role<|message|>content<|end|> / <|return|>
    "gpt-oss": (
        "{%- for message in messages %}"
        "{%- if message.role == 'system' %}"
        "<|start|>system<|message|>{{ message.content | trim }}<|end|>"
        "{%- elif message.role == 'user' %}"
        "<|start|>user<|message|>{{ message.content | trim }}<|end|>"
        "{%- elif message.role == 'assistant' %}"
        "<|start|>assistant<|channel|>final<|message|>"
        "{% generation %}{{ message.content | trim }}{% endgeneration %}"
        "<|end|>"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "<|start|>assistant<|channel|>final<|message|>"
        "{%- endif %}"
    ),
}


def _detect_template_family(tokenizer, model_path: str) -> str:
    """Detect which SFT chat template to use based on model family."""
    path_lower = model_path.lower()
    if "gpt-oss" in path_lower:
        return "gpt-oss"
    if "phi-4-multimodal" in path_lower:
        return "phi-4-mm"
    if "phi-4" in path_lower or "phi-3" in path_lower:
        return "phi-4"
    if "gemma" in path_lower:
        return "gemma"
    if "llama" in path_lower:
        return "llama"
    # Check tokenizer for Qwen-style markers
    native = tokenizer.chat_template or ""
    if "<|im_start|>" in native:
        return "qwen"
    if "<start_of_turn>" in native:
        return "gemma"
    if "<|start_header_id|>" in native:
        return "llama"
    # Default to Qwen (ChatML) as fallback
    return "qwen"


def _append_traces(path: str, entries: List[Dict[str, Any]]) -> None:
    """Append trace entries to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _log_init_trace(
    trace_path: str,
    model,
    peft_config,
    base_model: str,
    family: str,
    dataset_size: int,
    sft_cfg: dict,
) -> None:
    """Log model/LoRA architecture summary at training start."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # Collect which modules PEFT actually targeted
    targeted = set()
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A"):
            # Extract the short module name (e.g. "q_proj" from "model.layers.0.self_attn.q_proj")
            targeted.add(name.rsplit(".", 1)[-1] if "." in name else name)

    _append_traces(trace_path, [{
        "type": "init",
        "base_model": base_model,
        "template_family": family,
        "dataset_size": dataset_size,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 2) if total else 0,
        "lora_rank": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "lora_target_modules_config": (
            peft_config.target_modules
            if isinstance(peft_config.target_modules, str)
            else list(peft_config.target_modules)
        ),
        "lora_target_modules_resolved": sorted(targeted),
        "num_epochs": sft_cfg.get("num_epochs", 3),
        "per_device_batch_size": sft_cfg.get("per_device_batch_size", 4),
        "gradient_accumulation_steps": sft_cfg.get("gradient_accumulation_steps", 4),
        "learning_rate": sft_cfg.get("learning_rate", 2e-5),
        "max_seq_length": sft_cfg.get("max_seq_length", 8192),
    }])


def _log_tokenization_samples(
    trace_path: str,
    trainer,
    tokenizer,
    n_samples: int = 3,
) -> None:
    """Log a few tokenized examples showing label masking."""
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    for i in range(min(n_samples, batch["input_ids"].size(0))):
        input_ids = batch["input_ids"][i]
        labels = batch["labels"][i]

        # Find boundaries: prompt (masked) vs completion (trained)
        train_mask = labels != -100
        n_train = train_mask.sum().item()
        n_masked = (~train_mask).sum().item()
        total = input_ids.size(0)

        # Decode prompt and completion portions separately
        prompt_ids = input_ids[~train_mask].tolist()
        completion_ids = input_ids[train_mask].tolist()

        _append_traces(trace_path, [{
            "type": "tokenization_sample",
            "idx": i,
            "total_tokens": total,
            "prompt_tokens": n_masked,
            "completion_tokens": n_train,
            "prompt_text": tokenizer.decode(prompt_ids, skip_special_tokens=False),
            "completion_text": tokenizer.decode(completion_ids, skip_special_tokens=False),
        }])


class _SFTTraceCallback:
    """Transformers TrainerCallback that logs periodic step traces to JSONL."""

    def __init__(self, trace_path: str, trace_every: int, tokenizer):
        from transformers import TrainerCallback
        self._base = TrainerCallback
        self.trace_path = trace_path
        self.trace_every = trace_every
        self.tokenizer = tokenizer
        self._start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step % self.trace_every != 0:
            return
        entry = {
            "type": "step",
            "global_step": state.global_step,
            "epoch": round(state.epoch, 3) if state.epoch else 0,
            "wall_seconds": round(time.time() - self._start_time, 1),
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
        }
        _append_traces(self.trace_path, [entry])

    def on_train_end(self, args, state, control, **kwargs):
        _append_traces(self.trace_path, [{
            "type": "final",
            "global_step": state.global_step,
            "total_wall_seconds": round(time.time() - self._start_time, 1),
            "best_metric": state.best_metric,
            "total_flos": state.total_flos,
        }])


def _make_trace_callback(trace_path: str, trace_every: int, tokenizer):
    """Build a TrainerCallback for SFT trace logging."""
    from transformers import TrainerCallback

    class SFTTraceCallback(TrainerCallback):
        def __init__(self):
            self._inner = _SFTTraceCallback(trace_path, trace_every, tokenizer)

        def on_log(self, args, state, control, logs=None, **kwargs):
            self._inner.on_log(args, state, control, logs=logs, **kwargs)

        def on_train_end(self, args, state, control, **kwargs):
            self._inner.on_train_end(args, state, control, **kwargs)

    return SFTTraceCallback()


def run_sft_training_stage(
    dataset_path: str,
    base_model: str,
    output_dir: str,
    cfg: Any,
) -> None:
    """Run SFT training with TRL.

    Args:
        dataset_path: Path to sft_pairs.parquet with 'messages' column.
        base_model: HuggingFace model ID or local path for the base model.
        output_dir: Directory to save the LoRA checkpoint.
        cfg: Hydra config with training.sft section.
    """
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
    import torch

    sft_cfg = OmegaConf.to_container(
        OmegaConf.select(cfg, "training.sft"), resolve=True
    )

    # Load training data
    df = pd.read_parquet(dataset_path)
    print(f"[sft_training] Loaded {len(df)} training pairs from {dataset_path}")

    # Book-level filter (safety net for reused sft_pairs.parquet)
    book_id = OmegaConf.select(cfg, "runtime.book_id", default=None)
    if book_id is not None and "source_id" in df.columns:
        book_id = str(book_id)
        pre = len(df)
        df = df[df["source_id"].astype(str) == book_id].reset_index(drop=True)
        print(f"[sft_training] Filtered to book_id={book_id}: {len(df)}/{pre} pairs")

    # Parse messages from JSON strings back to lists
    def parse_messages(row):
        msgs = row["messages"]
        if isinstance(msgs, str):
            return {"messages": json.loads(msgs)}
        return {"messages": msgs}

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(parse_messages)

    # Optional: subsample training data for quick iteration
    sample_fraction = sft_cfg.get("sample_fraction")
    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        n_keep = max(1, int(len(dataset) * sample_fraction))
        dataset = dataset.shuffle(seed=42).select(range(n_keep))
        print(f"[sft_training] Sampled {n_keep}/{len(df)} examples ({sample_fraction:.0%})")

    print(f"[sft_training] Base model: {base_model}")
    print(f"[sft_training] Output dir: {output_dir}")

    # Configure LoRA
    lora_cfg = sft_cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("rank", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        task_type=TaskType.CAUSAL_LM,
    )

    # Device placement: accelerate sets LOCAL_RANK for DDP
    is_distributed = os.environ.get("LOCAL_RANK") is not None
    n_gpus = torch.cuda.device_count()

    # Prefer flash_attention_2 for full attention layers; fall back to sdpa
    try:
        import flash_attn  # noqa: F401
        _attn_impl = "flash_attention_2"
    except ImportError:
        _attn_impl = "sdpa"
    print(f"[sft_training] Using attention implementation: {_attn_impl}")

    # QLoRA + flash_attention_2 can cause illegal memory access on some
    # architectures (e.g. Qwen3.5) due to position ID handling.  Fall back
    # to eager attention when 4-bit quantization is enabled.
    quant_cfg_check = sft_cfg.get("quantization")
    if quant_cfg_check and quant_cfg_check.get("load_in_4bit"):
        _attn_impl = "eager"
        print(f"[sft_training] QLoRA detected — forcing attn_implementation=eager")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": _attn_impl,
        "low_cpu_mem_usage": True,
    }

    # Handle pre-quantized models (e.g. Mxfp4) that don't support training:
    # dequantize on load to bf16, then LoRA attaches normally.  Mxfp4 can't
    # be stacked with BnB QLoRA, so we skip BnB for these models.
    # GPT-OSS-20B dequantized to bf16 + LoRA fits on a 48GB A6000 (~43GB).
    _needs_dequantize = False
    try:
        from transformers import AutoConfig as _AC
        _model_cfg = _AC.from_pretrained(base_model, trust_remote_code=True)
        _qcfg = getattr(_model_cfg, "quantization_config", None)
        if _qcfg is not None:
            _qtype = _qcfg.get("quant_method", "") if isinstance(_qcfg, dict) else getattr(_qcfg, "quant_method", "")
            if _qtype and _qtype.lower() not in ("bitsandbytes", "gptq", "awq"):
                _needs_dequantize = True
    except Exception:
        pass

    if _needs_dequantize:
        from transformers import Mxfp4Config
        model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        # GPT-OSS requires eager attention for training
        model_kwargs["attn_implementation"] = "eager"
        print(f"[sft_training] Model ships pre-quantized — dequantizing to bf16 for training")

    # Optional 4-bit quantization (QLoRA) for large models
    quant_cfg = sft_cfg.get("quantization")
    print(f"[sft_training] DEBUG quantization config: {quant_cfg}, needs_dequantize={_needs_dequantize}")
    if quant_cfg and quant_cfg.get("load_in_4bit") and not _needs_dequantize:
        from transformers import BitsAndBytesConfig
        compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )
        # DDP: pin to local GPU. Single-GPU: use "auto" with max_memory so
        # transformers offloads to CPU during loading — the bf16→4bit
        # conversion temporarily needs more VRAM than the final model.
        if is_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            model_kwargs["device_map"] = {"": local_rank}
        else:
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {0: "40GiB", "cpu": "120GiB"}
        print(f"[sft_training] QLoRA: 4-bit quantization enabled (nf4, compute={compute_dtype}, device_map={model_kwargs['device_map']})")

    if is_distributed:
        print(f"[sft_training] DDP mode (LOCAL_RANK={os.environ['LOCAL_RANK']}, {n_gpus} GPUs)")
    else:
        print(f"[sft_training] Single-GPU mode ({n_gpus} GPUs visible)")

    # Load model
    # Phi-4-multimodal-instruct workaround: the model's __init__ internally
    # calls get_peft_model() for vision LoRA, which requires
    # prepare_inputs_for_generation on the inner model.  Since transformers
    # 4.50 removed GenerationMixin from PreTrainedModel, Phi4MMModel no
    # longer has this method.  Patch it in before loading.
    try:
        from transformers import AutoConfig as _ACfg
        _pre_cfg = _ACfg.from_pretrained(base_model, trust_remote_code=True)
        if getattr(_pre_cfg, "model_type", "") == "phi4mm":
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            _cls_ref = _pre_cfg.auto_map.get("AutoModelForCausalLM", "")
            if _cls_ref:
                # Load the CausalLM class (triggers dynamic module caching),
                # then grab Phi4MMModel from the same module.
                _causal_cls = get_class_from_dynamic_module(
                    _cls_ref, base_model, trust_remote_code=True)
                import sys
                _mod = sys.modules[_causal_cls.__module__]
                _inner_cls = getattr(_mod, "Phi4MMModel", None)
                if _inner_cls is not None and not hasattr(_inner_cls, "prepare_inputs_for_generation"):
                    _inner_cls.prepare_inputs_for_generation = lambda self, *a, **kw: {}
                    print("[sft_training] Patched Phi4MMModel.prepare_inputs_for_generation for PEFT compat")
    except Exception as _e:
        print(f"[sft_training] Phi4MM patch skipped: {_e}")

    print(f"[sft_training] Loading model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    # Fix _no_split_modules for FSDP: PEFT's fsdp_auto_wrap_policy uses this
    # to find the transformer layer class. For newer architectures (e.g. Qwen3.5)
    # the inherited value from the parent class may reference a non-existent class.
    # Patch it to match the config if specified.
    fsdp_layer_cls = sft_cfg.get("fsdp", {}).get("transformer_layer_cls_to_wrap")
    if fsdp_layer_cls:
        model._no_split_modules = [fsdp_layer_cls]
        print(f"[sft_training] Patched _no_split_modules -> {model._no_split_modules}")

    # Phi-4-multimodal text-only SFT: the model ships with built-in LoRA
    # adapters (speech, vision) wrapped around every linear layer via PEFT's
    # base_layer mechanism.  PEFT's get_peft_model() cannot add new LoRA on
    # top of existing PEFT-wrapped modules — gradients don't flow (grad_norm=0).
    #
    # Fix: merge the built-in LoRAs into the base weights and unwrap the
    # base_layer structure, giving PEFT clean nn.Linear modules to work with.
    # Also remove unused modality encoders for text-only SFT.
    _is_phi4mm = "phi-4-multimodal" in base_model.lower()
    if _is_phi4mm:
        import gc as _gc
        from peft import PeftModel as _PeftModel

        _ete = model.model.embed_tokens_extend
        # Remove both image and audio encoders (text-only)
        if hasattr(_ete, "audio_embed"):
            del _ete.audio_embed
            print("[sft_training] Phi-4-MM: deleted audio_embed")
        if hasattr(_ete, "image_embed"):
            del _ete.image_embed
            print("[sft_training] Phi-4-MM: deleted image_embed")

        # Merge built-in LoRA adapters into base weights and unwrap.
        # The model's linear layers are PEFT LoraLayer instances with
        # base_layer + lora_A/lora_B.  We need to:
        # 1. Compute merged weight: W = base_layer.weight + sum(B @ A * scaling)
        # 2. Replace the LoraLayer with a plain nn.Linear using the merged weight.
        _n_unwrapped = 0
        for _layer in model.model.layers:
            for _parent_name in ("self_attn", "mlp"):
                _parent = getattr(_layer, _parent_name, None)
                if _parent is None:
                    continue
                for _proj_name in list(vars(_parent).keys()):
                    _proj = getattr(_parent, _proj_name, None)
                    if _proj is None or not hasattr(_proj, "base_layer"):
                        continue
                    # Merge: start with base weight
                    _base = _proj.base_layer
                    _merged_w = _base.weight.data.clone()
                    _has_bias = _base.bias is not None
                    _merged_b = _base.bias.data.clone() if _has_bias else None
                    # Add each adapter's contribution: B @ A * scaling
                    if hasattr(_proj, "lora_A") and hasattr(_proj, "lora_B"):
                        _scaling = getattr(_proj, "scaling", {})
                        for _adapter_name in list(getattr(_proj.lora_A, "_modules", {}).keys()):
                            _a = _proj.lora_A[_adapter_name].weight.data
                            _b = _proj.lora_B[_adapter_name].weight.data
                            _s = _scaling.get(_adapter_name, 1.0)
                            _merged_w += (_b @ _a) * _s
                    # Create clean linear layer
                    _new = torch.nn.Linear(
                        _merged_w.shape[1], _merged_w.shape[0],
                        bias=_has_bias, dtype=_merged_w.dtype, device=_merged_w.device,
                    )
                    _new.weight.data.copy_(_merged_w)
                    if _has_bias:
                        _new.bias.data.copy_(_merged_b)
                    setattr(_parent, _proj_name, _new)
                    _n_unwrapped += 1

        print(f"[sft_training] Phi-4-MM: merged + unwrapped {_n_unwrapped} LoRA layers to plain nn.Linear")

        # Patch embed_tokens_extend to skip the training assert for text-only
        _orig_ete_forward = _ete.forward
        def _patched_ete_forward(*args, **kwargs):
            was_training = _ete.training
            _ete.training = False
            try:
                return _orig_ete_forward(*args, **kwargs)
            finally:
                _ete.training = was_training
        _ete.forward = _patched_ete_forward
        _gc.collect()
        torch.cuda.empty_cache()
        print("[sft_training] Phi-4-MM: text-only cleanup complete")

    # Configure SFT training
    gc_kwargs = sft_cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False})
    # FSDP for models that need sharding across GPUs (dequantized large models).
    # DDP replicates the full model on each GPU; FSDP shards it.
    fsdp_cfg = sft_cfg.get("fsdp")
    fsdp_kwargs = {}
    if fsdp_cfg:
        fsdp_kwargs["fsdp"] = fsdp_cfg.get("strategy", "full_shard auto_wrap")
        fsdp_config = {
            "auto_wrap_policy": fsdp_cfg.get("auto_wrap_policy", "TRANSFORMER_BASED_WRAP"),
            "backward_prefetch": fsdp_cfg.get("backward_prefetch", "BACKWARD_PRE"),
            "forward_prefetch": fsdp_cfg.get("forward_prefetch", False),
            "cpu_ram_efficient_loading": fsdp_cfg.get("cpu_ram_efficient_loading", True),
        }
        if fsdp_cfg.get("transformer_layer_cls_to_wrap"):
            fsdp_config["transformer_layer_cls_to_wrap"] = fsdp_cfg["transformer_layer_cls_to_wrap"]
            # PEFT's fsdp_auto_wrap_policy reads this env var to find the layer
            # class, overriding whatever SFTConfig sets. Set it explicitly so
            # models not in PEFT's auto-detection list (e.g. Qwen3.5) work.
            os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = fsdp_cfg["transformer_layer_cls_to_wrap"]
        fsdp_kwargs["fsdp_config"] = fsdp_config
        print(f"[sft_training] FSDP enabled: {fsdp_kwargs['fsdp']}")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_cfg.get("num_epochs", 3),
        per_device_train_batch_size=sft_cfg.get("per_device_batch_size", 4),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=sft_cfg.get("learning_rate", 2e-5),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.1),
        weight_decay=sft_cfg.get("weight_decay", 0.01),
        max_length=sft_cfg.get("max_seq_length", 8192),
        gradient_checkpointing=sft_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=gc_kwargs,
        max_grad_norm=sft_cfg.get("max_grad_norm", 1.0),
        bf16=sft_cfg.get("bf16", True),
        logging_steps=sft_cfg.get("logging_steps", 10),
        save_strategy=sft_cfg.get("save_strategy", "epoch"),
        report_to="wandb" if OmegaConf.select(cfg, "wandb.enabled") else "none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=sft_cfg.get("dataloader_num_workers", 4),
        ddp_find_unused_parameters=False,
        # Enable loss masking via {% generation %} blocks in the chat template.
        # Without this, SFTTrainer trains on all tokens (prompt + completion).
        assistant_only_loss=True,
        **fsdp_kwargs,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # SFTTrainer needs {% generation %} blocks in the chat template so it
    # knows which tokens are assistant completions (and should contribute to
    # loss). Without this, all labels become -100 and training produces zero
    # loss / NaN gradients.
    #
    # Prefer the model's native template if it already includes generation
    # blocks (e.g. Qwen3+, SmolLM3). Fall back to our manual templates for
    # architectures that don't ship them yet.
    native_template = tokenizer.chat_template or ""
    has_native_generation = "{% generation %}" in native_template

    # Resolve thinking mode from cfg.model (single source of truth). For SFT
    # this controls whether the Qwen manual template injects the empty-think
    # sentinel prefix. For the native-template path, we simply log the mode
    # so downstream GRPO/eval can verify alignment.
    from dagspaces.common.stage_utils import resolve_thinking_mode
    model_cfg = getattr(cfg, "model", None) or {}
    _thinking_enabled_sft = resolve_thinking_mode(model_cfg, default=False)
    print(f"[sft_training] Thinking mode: "
          f"{'on' if _thinking_enabled_sft else 'off'} "
          f"(from cfg.model.thinking_mode or chat_template_kwargs.enable_thinking)")

    if has_native_generation:
        family = _detect_template_family(tokenizer, base_model)
        print(f"[sft_training] Chat template: native (family={family}, has {{% generation %}} blocks)")
    else:
        family = _detect_template_family(tokenizer, base_model)
        if family == "qwen":
            sft_template = _qwen_sft_template(thinking_enabled=_thinking_enabled_sft)
        else:
            sft_template = _SFT_TEMPLATES[family]
        tokenizer.chat_template = sft_template
        print(f"[sft_training] Chat template: manual override "
              f"(family={family}, thinking={'on' if _thinking_enabled_sft else 'off'})")

    # Trace logging: log diagnostics ~10 times during training.
    trace_log_path = os.path.join(output_dir, "sft_traces.jsonl")
    total_steps = int(
        (len(dataset) * sft_cfg.get("num_epochs", 3))
        / max(sft_cfg.get("per_device_batch_size", 4) * sft_cfg.get("gradient_accumulation_steps", 4), 1)
    )
    trace_every = max(total_steps // 10, 1)

    print(f"[sft_training] Starting SFT with LoRA (r={peft_config.r}, alpha={peft_config.lora_alpha})")

    # Gemma 3 requires token_type_ids during training (used for text vs image
    # token masking in the bidirectional attention). For text-only SFT, these
    # are all zeros.  Wrap the default collator to inject them.
    data_collator = None
    if family == "gemma":
        import torch

        class _GemmaCollator:
            """Injects zero-valued token_type_ids for Gemma 3 text-only training."""
            def __init__(self, inner):
                self.inner = inner
            def __call__(self, features):
                batch = self.inner(features)
                if "token_type_ids" not in batch and "input_ids" in batch:
                    batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
                return batch

        print("[sft_training] Gemma 3: injecting token_type_ids via custom data collator")

    if family == "phi-4-mm":
        import torch

        class _Phi4MMCollator:
            """Injects input_mode=0 (LANGUAGE) for Phi-4-multimodal text-only SFT."""
            def __init__(self, inner):
                self.inner = inner
            def __call__(self, features):
                batch = self.inner(features)
                if "input_mode" not in batch:
                    bs = batch["input_ids"].shape[0]
                    batch["input_mode"] = torch.zeros(bs, dtype=torch.long, device=batch["input_ids"].device)
                return batch

        print("[sft_training] Phi-4-MM: text-only SFT setup")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[_make_trace_callback(trace_log_path, trace_every, tokenizer)],
    )

    # Wrap trainer's collator after init (SFTTrainer builds its own internally)
    if family == "gemma":
        trainer.data_collator = _GemmaCollator(trainer.data_collator)
    if family == "phi-4-mm":
        trainer.data_collator = _Phi4MMCollator(trainer.data_collator)

    # Log init trace (LoRA resolved modules, param counts, config)
    _log_init_trace(trace_log_path, model, peft_config, base_model, family, len(dataset), sft_cfg)
    print(f"[sft_training] Traces -> {trace_log_path} (every {trace_every} steps)")

    # Validate tokenization and label masking on first batch + log samples
    sample = next(iter(trainer.get_train_dataloader()))
    labels = sample["labels"][0]
    n_train = (labels != -100).sum().item()
    n_masked = (labels == -100).sum().item()
    print(f"[sft_training] Label check: {n_train} train tokens, {n_masked} masked tokens")
    if n_train == 0:
        raise ValueError(
            "All labels are -100 — chat template masking is broken. "
            "Ensure the tokenizer chat_template includes {% generation %} blocks."
        )
    if n_masked == 0:
        raise ValueError(
            "No labels are masked — loss is computed on the entire sequence "
            "(prompt + completion). Ensure assistant_only_loss=True is set in "
            "SFTConfig and the chat template includes {% generation %} blocks."
        )
    _log_tokenization_samples(trace_log_path, trainer, tokenizer, n_samples=3)

    trainer.train()

    # Save final checkpoint.
    # Restore the original chat template before saving — the {% generation %}
    # blocks are only needed during training and must NOT persist to checkpoints.
    original_template = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True
    ).chat_template
    if original_template:
        tokenizer.chat_template = original_template

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[sft_training] Saved LoRA checkpoint to {output_dir}")
