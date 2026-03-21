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
from typing import Any

import pandas as pd
from omegaconf import OmegaConf


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
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
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

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": _attn_impl,
    }

    # Optional 4-bit quantization (QLoRA) for large models
    quant_cfg = sft_cfg.get("quantization")
    if quant_cfg and quant_cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )
        # Each DDP process must load onto its own GPU
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model_kwargs["device_map"] = {"": local_rank}
        print(f"[sft_training] QLoRA: 4-bit quantization enabled (nf4, compute={compute_dtype}, device={local_rank})")

    if is_distributed:
        print(f"[sft_training] DDP mode (LOCAL_RANK={os.environ['LOCAL_RANK']}, {n_gpus} GPUs)")
    else:
        print(f"[sft_training] Single-GPU mode ({n_gpus} GPUs visible)")

    # Load model
    print(f"[sft_training] Loading model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    # Configure SFT training
    gc_kwargs = sft_cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False})
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
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Override chat template with TRL-compatible version that includes
    # {% generation %} blocks so SFTTrainer knows which tokens are assistant
    # completions (and should contribute to loss). Without this, all labels
    # become -100 and training produces zero loss / NaN gradients.
    _SFT_CHAT_TEMPLATE = (
        "{%- for message in messages %}"
        "{%- if message.role == 'system' %}"
        "<|im_start|>system\n{{ message.content | trim }}<|im_end|>\n"
        "{%- elif message.role == 'user' %}"
        "<|im_start|>user\n{{ message.content | trim }}<|im_end|>\n"
        "{%- elif message.role == 'assistant' %}"
        "<|im_start|>assistant\n"
        "{% generation %}{{ message.content | trim }}{% endgeneration %}"
        "<|im_end|>\n"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{%- endif %}"
    )
    tokenizer.chat_template = _SFT_CHAT_TEMPLATE

    print(f"[sft_training] Starting SFT with LoRA (r={peft_config.r}, alpha={peft_config.lora_alpha})")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Validate tokenization and label masking on first batch
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

    trainer.train()

    # Save final checkpoint.
    # Restore the original Qwen3 chat template (with enable_thinking support)
    # before saving — the _SFT_CHAT_TEMPLATE with {% generation %} blocks is
    # only needed during training and must NOT persist to inference checkpoints.
    original_template = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True
    ).chat_template
    if original_template:
        tokenizer.chat_template = original_template

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[sft_training] Saved LoRA checkpoint to {output_dir}")
