#!/usr/bin/env python3
"""Convert Meta Llama 3.2 Vision checkpoint to HuggingFace format.

Patches the bundled transformers conversion script to remap state_dict
keys for transformers 5.x before load_state_dict.

Usage:
    python scripts/convert_mllama_weights_to_hf.py \
        --input_dir /path/to/meta/checkpoint \
        --output_dir /path/to/hf/output \
        --num_shards 1
"""

import importlib.util
import torch


def remap_keys_for_v5(state_dict: dict) -> dict:
    """Remap 4.x-style keys to transformers 5.x structure."""
    new_sd = {}
    for key, value in state_dict.items():
        if key.startswith("language_model.lm_head."):
            new_key = key.replace("language_model.lm_head.", "lm_head.")
        elif key.startswith("language_model.model."):
            new_key = "model.language_model." + key[len("language_model.model."):]
        elif key.startswith("vision_model."):
            new_key = "model." + key
        elif key.startswith("multi_modal_projector."):
            new_key = "model." + key
        else:
            new_key = key
        new_sd[new_key] = value
    return new_sd


# Monkey-patch torch.nn.Module.load_state_dict to remap keys
_original_load_state_dict = torch.nn.Module.load_state_dict


def _patched_load_state_dict(self, state_dict, strict=True, assign=False):
    # Check if this looks like a 4.x state_dict being loaded into a 5.x model
    has_old_keys = any(k.startswith("language_model.model.") for k in state_dict)
    has_new_keys = any(k.startswith("model.language_model.") for k in state_dict)
    if has_old_keys and not has_new_keys:
        print("Remapping state_dict keys from transformers 4.x to 5.x format...")
        state_dict = remap_keys_for_v5(state_dict)
    return _original_load_state_dict(self, state_dict, strict=strict, assign=assign)


torch.nn.Module.load_state_dict = _patched_load_state_dict

# Now import and run the bundled conversion script
spec = importlib.util.spec_from_file_location(
    "__main__",  # pretend it's the main module so its if __name__ == "__main__" block runs
    "/home/mwf62/.local/lib/python3.12/site-packages/transformers/models/mllama/convert_mllama_weights_to_hf.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
