#!/usr/bin/env python3
"""Merge LoRA adapter weights directly into base model safetensors.

Works at the safetensors level — no need to load the full model through
transformers. Only modifies the unquantized attention projection weights
that the LoRA targets.

Usage:
    python scripts/merge_lora_safetensors.py \
        --base /share/pierson/matt/zoo/models/GPT-OSS-20B \
        --adapter /share/pierson/matt/UAIR/multirun/2026-03-24_grpo_training/22-42-18/sft_only/outputs/sft/checkpoint \
        --output /share/pierson/matt/zoo/models/GPT-OSS-20B-SFT-merged
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into safetensors")
    parser.add_argument("--base", required=True, help="Base model directory")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Output merged model directory")
    args = parser.parse_args()

    base_dir = Path(args.base)
    adapter_dir = Path(args.adapter)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load adapter config
    with open(adapter_dir / "adapter_config.json") as f:
        adapter_cfg = json.load(f)
    lora_r = adapter_cfg["r"]
    lora_alpha = adapter_cfg["lora_alpha"]
    scaling = lora_alpha / lora_r
    target_modules = adapter_cfg["target_modules"]
    print(f"LoRA r={lora_r}, alpha={lora_alpha}, scaling={scaling}")
    print(f"Target modules: {target_modules}")

    # Load all LoRA weights
    lora_path = adapter_dir / "adapter_model.safetensors"
    lora_weights = {}
    with safe_open(str(lora_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_weights[key] = f.get_tensor(key)
    print(f"Loaded {len(lora_weights)} LoRA tensors")

    # Build a map: base_weight_key -> (lora_A, lora_B)
    # LoRA keys: base_model.model.model.layers.N.self_attn.X_proj.lora_A.weight
    # Base keys: model.layers.N.self_attn.X_proj.weight
    merge_map = {}
    for lora_key in lora_weights:
        if ".lora_A.weight" in lora_key:
            # Extract the base key
            base_key = (
                lora_key.replace("base_model.model.", "")
                .replace(".lora_A.weight", ".weight")
            )
            b_key = lora_key.replace(".lora_A.weight", ".lora_B.weight")
            merge_map[base_key] = (lora_weights[lora_key], lora_weights[b_key])
    print(f"Will merge {len(merge_map)} weight matrices")

    # Load weight index
    index_path = base_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_index = json.load(f)
    weight_map = weight_index["weight_map"]

    # Group merge targets by shard file
    shard_targets = {}
    for base_key in merge_map:
        shard_file = weight_map[base_key]
        shard_targets.setdefault(shard_file, []).append(base_key)

    # Process each shard
    all_shard_files = set(weight_map.values())
    for shard_file in sorted(all_shard_files):
        src = base_dir / shard_file
        dst = out_dir / shard_file

        if shard_file not in shard_targets:
            # No merges needed — just copy
            print(f"Copying {shard_file} (no merges needed)")
            shutil.copy2(src, dst)
            continue

        targets = shard_targets[shard_file]
        print(f"Processing {shard_file}: merging {len(targets)} weights...")

        # Load all tensors from this shard
        tensors = {}
        with safe_open(str(src), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        # Apply LoRA merges: W' = W + scaling * B @ A
        for base_key in targets:
            lora_A, lora_B = merge_map[base_key]
            W = tensors[base_key]
            orig_dtype = W.dtype
            # Compute in float32 for precision
            delta = scaling * (lora_B.float() @ lora_A.float())
            tensors[base_key] = (W.float() + delta).to(orig_dtype)
            print(f"  Merged {base_key}: {W.shape} ({orig_dtype})")

        # Save merged shard
        save_file(tensors, str(dst))

    # Copy the weight index (unchanged — same shard layout)
    shutil.copy2(index_path, out_dir / "model.safetensors.index.json")

    # Copy all non-safetensors files (config, tokenizer, etc.)
    skip_suffixes = {".safetensors"}
    skip_names = {"model.safetensors.index.json"}
    for item in base_dir.iterdir():
        if item.is_dir():
            continue
        if item.suffix in skip_suffixes or item.name in skip_names:
            continue
        dst = out_dir / item.name
        if not dst.exists():
            print(f"Copying {item.name}")
            shutil.copy2(item, dst)

    # Copy chat_template from adapter if present (may reflect SFT prompt format)
    adapter_chat_tpl = adapter_dir / "chat_template.jinja"
    if adapter_chat_tpl.exists():
        print(f"Copying chat_template.jinja from adapter")
        shutil.copy2(adapter_chat_tpl, out_dir / "chat_template.jinja")

    print(f"\nMerged model saved to {out_dir}")


if __name__ == "__main__":
    main()
