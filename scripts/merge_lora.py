#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model, producing a standalone model directory.

Usage:
    python scripts/merge_lora.py \
        --base /share/pierson/matt/zoo/models/Qwen3.5-9B \
        --lora /path/to/sft/checkpoint \
        --output /share/pierson/matt/zoo/models/Qwen3.5-9B-SFT-CI

    # With a custom suffix (auto-names output as {base}-{suffix}):
    python scripts/merge_lora.py \
        --base /share/pierson/matt/zoo/models/Qwen3.5-9B \
        --lora /path/to/sft/checkpoint \
        --suffix SFT-CI
"""

import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(base_path: str, lora_path: str, output_path: str) -> None:
    if os.path.exists(os.path.join(output_path, "config.json")):
        print(f"Merged model already exists at {output_path}, skipping.")
        return

    print(f"Loading base model: {base_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging weights...")
    merged = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving merged model to {output_path}")
    merged.save_pretrained(output_path)

    # Save tokenizer from the LoRA checkpoint (has chat template overrides),
    # falling back to the base model tokenizer.
    tok_path = lora_path if os.path.exists(os.path.join(lora_path, "tokenizer_config.json")) else base_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print(f"Done! Merged model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base", required=True, help="Path to base model")
    parser.add_argument("--lora", required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output", default=None, help="Output path for merged model")
    parser.add_argument("--suffix", default=None, help="Auto-name output as {base}-{suffix}")
    args = parser.parse_args()

    if args.output is None and args.suffix is None:
        parser.error("Provide either --output or --suffix")

    output_path = args.output
    if output_path is None:
        base_name = os.path.basename(args.base.rstrip("/"))
        output_path = os.path.join(os.path.dirname(args.base.rstrip("/")), f"{base_name}-{args.suffix}")

    merge_lora(args.base, args.lora, output_path)


if __name__ == "__main__":
    main()
