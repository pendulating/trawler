#!/usr/bin/env python3
"""Apply compatibility patches to the local virtualenv for Qwen3.5 + TRL 0.29.1 + vLLM 0.18.

Adapted from n2s4cir/scripts/apply_env_patches.py (training-scripts branch)
for the Trawler stack: vLLM 0.18.0, TRL 0.29.1, transformers 5.3.0.0

These patches are idempotent and can be run before every job launch.

Patches applied:
  1. vLLM qwen3_vl.py  — text-only mrope fallback (no vision_config)
  2. vLLM qwen3_5.py   — guard self.visual init + ProcessingInfo mm limits
  3. TRL vllm_generation.py — Qwen3.5 weight key remapping (model.* → language_model.model.*)
  4. TRL vllm_serve.py  — truncate_prompt_tokens TypeError guard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _replace_text(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}:\n  {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# 1. vLLM qwen3_vl.py — text-only mrope fallback
# ---------------------------------------------------------------------------

def patch_vllm_qwen3_vl(site_packages: Path) -> list[str]:
    changed: list[str] = []
    qwen3vl = site_packages / "vllm/model_executor/models/qwen3_vl.py"
    try:
        if _replace_text(
            qwen3vl,
            "    def get_mrope_input_positions(\n"
            "        self,\n"
            "        input_tokens: list[int],\n"
            "        mm_features: list[MultiModalFeatureSpec],\n"
            "    ) -> tuple[torch.Tensor, int]:\n"
            "        return self._get_mrope_input_positions(\n"
            "            input_tokens=input_tokens,\n"
            "            mm_features=mm_features,\n"
            "            config=self.config,\n"
            "        )\n",
            "    def get_mrope_input_positions(\n"
            "        self,\n"
            "        input_tokens: list[int],\n"
            "        mm_features: list[MultiModalFeatureSpec],\n"
            "    ) -> tuple[torch.Tensor, int]:\n"
            "        # Handle text-only config (no vision_config)\n"
            "        if not hasattr(self.config, 'vision_config'):\n"
            "            import torch\n"
            "            llm_pos_ids = torch.arange(\n"
            "                len(input_tokens), dtype=torch.int,\n"
            "                device=self.device if hasattr(self, 'device') else 'cpu',\n"
            "            )\n"
            "            return torch.stack([llm_pos_ids] * 3), 0\n"
            "\n"
            "        return self._get_mrope_input_positions(\n"
            "            input_tokens=input_tokens,\n"
            "            mm_features=mm_features,\n"
            "            config=self.config,\n"
            "        )\n",
        ):
            changed.append(str(qwen3vl))
    except RuntimeError:
        pass
    return changed


# ---------------------------------------------------------------------------
# 2. vLLM qwen3_5.py — guard vision init + ProcessingInfo mm limits
# ---------------------------------------------------------------------------

def patch_vllm_qwen3_5_model(site_packages: Path) -> list[str]:
    changed: list[str] = []
    qwen35 = site_packages / "vllm/model_executor/models/qwen3_5.py"

    # 2a. Guard self.visual init for Qwen3_5ForConditionalGeneration
    try:
        if _replace_text(
            qwen35,
            "        with self._mark_tower_model(vllm_config, {\"image\", \"video\"}):\n"
            "            self.visual = Qwen3_VisionTransformer(\n"
            "                config.vision_config,\n"
            "                norm_eps=getattr(config, \"rms_norm_eps\", 1e-6),\n"
            "                quant_config=quant_config,\n"
            "                prefix=maybe_prefix(prefix, \"visual\"),\n"
            "            )\n"
            "\n"
            "        with self._mark_language_model(vllm_config):\n"
            "            self.language_model = Qwen3_5ForCausalLM(\n",
            "        with self._mark_tower_model(vllm_config, {\"image\", \"video\"}):\n"
            "            self.visual = None\n"
            "            if hasattr(config, 'vision_config') and config.vision_config is not None:\n"
            "                self.visual = Qwen3_VisionTransformer(\n"
            "                    config.vision_config,\n"
            "                    norm_eps=getattr(config, \"rms_norm_eps\", 1e-6),\n"
            "                    quant_config=quant_config,\n"
            "                    prefix=maybe_prefix(prefix, \"visual\"),\n"
            "                )\n"
            "\n"
            "        with self._mark_language_model(vllm_config):\n"
            "            self.language_model = Qwen3_5ForCausalLM(\n",
        ):
            changed.append(f"{qwen35} (dense visual guard)")
    except RuntimeError:
        pass

    # 2b. Guard self.visual init for Qwen3_5MoeForConditionalGeneration
    try:
        if _replace_text(
            qwen35,
            "        with self._mark_tower_model(vllm_config, {\"image\", \"video\"}):\n"
            "            self.visual = Qwen3_VisionTransformer(\n"
            "                config.vision_config,\n"
            "                norm_eps=getattr(config, \"rms_norm_eps\", 1e-6),\n"
            "                quant_config=quant_config,\n"
            "                prefix=maybe_prefix(prefix, \"visual\"),\n"
            "            )\n"
            "\n"
            "        with self._mark_language_model(vllm_config):\n"
            "            self.language_model = Qwen3_5MoeForCausalLM(\n",
            "        with self._mark_tower_model(vllm_config, {\"image\", \"video\"}):\n"
            "            self.visual = None\n"
            "            if hasattr(config, 'vision_config') and config.vision_config is not None:\n"
            "                self.visual = Qwen3_VisionTransformer(\n"
            "                    config.vision_config,\n"
            "                    norm_eps=getattr(config, \"rms_norm_eps\", 1e-6),\n"
            "                    quant_config=quant_config,\n"
            "                    prefix=maybe_prefix(prefix, \"visual\"),\n"
            "                )\n"
            "\n"
            "        with self._mark_language_model(vllm_config):\n"
            "            self.language_model = Qwen3_5MoeForCausalLM(\n",
        ):
            changed.append(f"{qwen35} (MoE visual guard)")
    except RuntimeError:
        pass

    # 2c. get_hf_config: accept Qwen3_5TextConfig (from transformers 5.x) in
    # addition to vLLM's composite Qwen3_5Config.  transformers 5.x returns
    # Qwen3_5TextConfig for text-only models; vLLM's type check rejects it.
    try:
        if _replace_text(
            qwen35,
            "class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):\n"
            "    def get_supported_mm_limits(self):\n"
            "        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
            "            return {}\n"
            "        return super().get_supported_mm_limits()\n"
            "\n"
            "    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):\n"
            "        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
            "            return 0\n"
            "        return super().get_mm_max_tokens_per_item(seq_len, mm_counts)\n"
            "\n"
            "    def get_hf_processor(self, **kwargs):\n"
            "        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
            "            return None\n"
            "        return super().get_hf_processor(**kwargs)\n"
            "\n"
            "    def get_hf_config(self):\n"
            "        return self.ctx.get_hf_config(Qwen3_5Config)\n",
            "class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):\n"
            "    def get_supported_mm_limits(self):\n"
            "        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
            "            return {}\n"
            "        return super().get_supported_mm_limits()\n"
            "\n"
            "    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):\n"
            "        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
            "            return 0\n"
            "        return super().get_mm_max_tokens_per_item(seq_len, mm_counts)\n"
            "\n"
            "    def get_hf_processor(self, **kwargs):\n"
            "        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
            "            return None\n"
            "        return super().get_hf_processor(**kwargs)\n"
            "\n"
            "    def get_hf_config(self):\n"
            "        # Accept both vLLM Qwen3_5Config and transformers Qwen3_5TextConfig\n"
            "        try:\n"
            "            return self.ctx.get_hf_config(Qwen3_5Config)\n"
            "        except TypeError:\n"
            "            return self.ctx.get_hf_config()\n",
        ):
            changed.append(f"{qwen35} (Qwen3_5ProcessingInfo get_hf_config TypeError guard)")
    except RuntimeError:
        pass

    # Same fix for Qwen3_5MoeProcessingInfo
    try:
        if _replace_text(
            qwen35,
            "class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):\n"
            "    def get_supported_mm_limits(self):\n"
            "        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
            "            return {}\n"
            "        return super().get_supported_mm_limits()\n"
            "\n"
            "    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):\n"
            "        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
            "            return 0\n"
            "        return super().get_mm_max_tokens_per_item(seq_len, mm_counts)\n"
            "\n"
            "    def get_hf_processor(self, **kwargs):\n"
            "        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
            "            return None\n"
            "        return super().get_hf_processor(**kwargs)\n"
            "\n"
            "    def get_hf_config(self):\n"
            "        return self.ctx.get_hf_config(Qwen3_5MoeConfig)\n",
            "class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):\n"
            "    def get_supported_mm_limits(self):\n"
            "        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
            "            return {}\n"
            "        return super().get_supported_mm_limits()\n"
            "\n"
            "    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):\n"
            "        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
            "            return 0\n"
            "        return super().get_mm_max_tokens_per_item(seq_len, mm_counts)\n"
            "\n"
            "    def get_hf_processor(self, **kwargs):\n"
            "        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
            "            return None\n"
            "        return super().get_hf_processor(**kwargs)\n"
            "\n"
            "    def get_hf_config(self):\n"
            "        # Accept both vLLM Qwen3_5MoeConfig and transformers config\n"
            "        try:\n"
            "            return self.ctx.get_hf_config(Qwen3_5MoeConfig)\n"
            "        except TypeError:\n"
            "            return self.ctx.get_hf_config()\n",
        ):
            changed.append(f"{qwen35} (Qwen3_5MoeProcessingInfo get_hf_config TypeError guard)")
    except RuntimeError:
        pass

    # 2d. ProcessingInfo: guard get_supported_mm_limits + get_mm_max_tokens_per_item
    # (only applies if patches above were already applied on a previous run)
    text = qwen35.read_text(encoding="utf-8")
    if "get_supported_mm_limits" not in text:
        # Add vision-config guards to both ProcessingInfo classes
        for cls in ("Qwen3_5ProcessingInfo", "Qwen3_5MoeProcessingInfo"):
            try:
                old_block = (
                    f"class {cls}(Qwen3VLProcessingInfo):\n"
                    f"    def get_hf_config(self):\n"
                )
                new_block = (
                    f"class {cls}(Qwen3VLProcessingInfo):\n"
                    f"    def get_supported_mm_limits(self):\n"
                    f"        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
                    f"            return {{}}\n"
                    f"        return super().get_supported_mm_limits()\n"
                    f"\n"
                    f"    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):\n"
                    f"        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
                    f"            return 0\n"
                    f"        return super().get_mm_max_tokens_per_item(seq_len, mm_counts)\n"
                    f"\n"
                    f"    def get_hf_processor(self, **kwargs):\n"
                    f"        if not hasattr(self.get_hf_config(), 'vision_config'):\n"
                    f"            return None\n"
                    f"        return super().get_hf_processor(**kwargs)\n"
                    f"\n"
                    f"    def get_hf_config(self):\n"
                )
                if _replace_text(qwen35, old_block, new_block):
                    changed.append(f"{qwen35} ({cls} mm guards)")
            except RuntimeError:
                pass

    return changed


# ---------------------------------------------------------------------------
# 3. TRL vllm_generation.py — Qwen3.5 weight key remapping
# ---------------------------------------------------------------------------

def patch_trl_vllm_generation(site_packages: Path) -> list[str]:
    changed: list[str] = []
    trl_file = site_packages / "trl/generation/vllm_generation.py"
    text = trl_file.read_text(encoding="utf-8")

    # 3a. Add _is_qwen35_model detection before _init_vllm()
    if "_is_qwen35_model" not in text:
        try:
            if _replace_text(
                trl_file,
                "        self.generation_kwargs = generation_kwargs or {}\n"
                "\n"
                "        self._init_vllm()\n",
                "        self.generation_kwargs = generation_kwargs or {}\n"
                "\n"
                "        model_type = getattr(getattr(model, 'config', None), 'model_type', None)\n"
                "        self._is_qwen35_model = isinstance(model_type, str) and model_type.startswith('qwen3_5')\n"
                "\n"
                "        self._init_vllm()\n",
            ):
                changed.append(f"{trl_file} (_is_qwen35_model detection)")
                text = trl_file.read_text(encoding="utf-8")
        except RuntimeError:
            pass

    # 3b. Add language_model.* prefix remapping to _fix_param_name_to_vllm
    if "language_model." not in text or "# Qwen3.5 vLLM" not in text:
        try:
            if _replace_text(
                trl_file,
                "        for prefix in prefixes:\n"
                "            name = name.replace(prefix, \"\")\n"
                "        return name\n",
                "        for prefix in prefixes:\n"
                "            name = name.replace(prefix, \"\")\n"
                "\n"
                "        # Qwen3.5 vLLM models are multimodal and expose language weights under\n"
                "        # `language_model.*`, while the HF CausalLM training model uses `model.*`.\n"
                "        if self._is_qwen35_model:\n"
                "            if name.startswith('model.'):\n"
                "                name = f'language_model.{name}'\n"
                "            elif name.startswith('lm_head.'):\n"
                "                name = f'language_model.{name}'\n"
                "        return name\n",
            ):
                changed.append(f"{trl_file} (Qwen3.5 weight key remapping)")
        except RuntimeError:
            pass

    return changed


# ---------------------------------------------------------------------------
# 4a. vLLM renderer base.py — skip multimodal init when language_model_only
# ---------------------------------------------------------------------------

def patch_vllm_renderer_language_model_only(site_packages: Path) -> list[str]:
    changed: list[str] = []
    base_renderer = site_packages / "vllm/renderers/base.py"
    try:
        if _replace_text(
            base_renderer,
            "        if config.model_config.is_multimodal_model:\n",
            "        _lm_only = getattr(getattr(config.model_config, 'multimodal_config', None), 'language_model_only', False)\n"
            "        if config.model_config.is_multimodal_model and not _lm_only:\n",
        ):
            changed.append(f"{base_renderer} (skip mm processor when language_model_only)")
    except RuntimeError:
        pass
    return changed


# ---------------------------------------------------------------------------
# 4b. TRL vllm_serve.py — truncate_prompt_tokens TypeError guard
# ---------------------------------------------------------------------------

def patch_trl_vllm_serve(site_packages: Path) -> list[str]:
    changed: list[str] = []
    serve_file = site_packages / "trl/scripts/vllm_serve.py"

    # There are two occurrences (generate endpoint + chat endpoint)
    for context_after in [
        "        # Evenly distribute prompts across DP ranks\n"
        "        chunked_prompts",
        "        # Evenly distribute prompts across DP ranks\n"
        "        chunked_messages",
    ]:
        try:
            if _replace_text(
                serve_file,
                "        generation_kwargs[structured_outputs_key] = structured_outputs\n"
                "        sampling_params = SamplingParams(**generation_kwargs)\n"
                "\n" + context_after,
                "        generation_kwargs[structured_outputs_key] = structured_outputs\n"
                "        try:\n"
                "            sampling_params = SamplingParams(**generation_kwargs)\n"
                "        except TypeError as exc:\n"
                "            if 'truncate_prompt_tokens' in str(exc):\n"
                "                generation_kwargs.pop('truncate_prompt_tokens', None)\n"
                "                sampling_params = SamplingParams(**generation_kwargs)\n"
                "            else:\n"
                "                raise\n"
                "\n" + context_after,
            ):
                changed.append(f"{serve_file} (truncate_prompt_tokens guard)")
        except RuntimeError:
            pass

    return changed


# ---------------------------------------------------------------------------
# 5. transformers core_model_loading.py — bnb 4-bit quantization OOM (#43032)
# ---------------------------------------------------------------------------

def patch_transformers_bnb_loading(site_packages: Path) -> list[str]:
    """Workaround for transformers v5 regression where bnb 4-bit quantization
    OOMs during model loading.  The new loading pipeline materializes bf16
    tensors on GPU before quantizing; this patch forces params that need
    quantization to materialize on CPU first.
    See: https://github.com/huggingface/transformers/issues/43032
    """
    changed: list[str] = []
    target = site_packages / "transformers/core_model_loading.py"
    try:
        if _replace_text(
            target,
            "            if future_or_tensor is None:\n"
            "                param_device = get_device(device_map, renamed_key, valid_torch_device=True)\n"
            "                future_or_tensor = spawn_materialize(thread_pool, tensor, param_device, _dtype)\n",
            "            if future_or_tensor is None:\n"
            "                param_device = get_device(device_map, renamed_key, valid_torch_device=True)\n"
            "                # Workaround for transformers v5 regression (#43032): materialize\n"
            "                # quantizable params on CPU to avoid bf16 OOM before quantization.\n"
            "                if getattr(mapping, 'quantization_operation', None) is not None:\n"
            "                    param_device = 'cpu'\n"
            "                future_or_tensor = spawn_materialize(thread_pool, tensor, param_device, _dtype)\n",
        ):
            changed.append(str(target))
    except RuntimeError:
        pass
    return changed


# ---------------------------------------------------------------------------
# 6. vLLM column_parallel_linear.py — heterogeneous 2-slice packed LoRA
# ---------------------------------------------------------------------------

def patch_vllm_lora_column_parallel(site_packages: Path) -> list[str]:
    """Fix LoRA dimension mismatch for 2-slice packed modules with unequal sizes.

    Qwen3.5 has packed modules like in_proj_qkvz = [in_proj_qkv, in_proj_z]
    where the two slices have different output sizes (e.g. 8192 vs 2048).
    MergedColumnParallelLinearWithLoRA assumes equal sizes and crashes.
    Route these to MergedColumnParallelLinearVariableSliceWithLoRA instead.
    """
    changed: list[str] = []
    target = site_packages / "vllm/lora/layers/column_parallel_linear.py"

    # 6a. MergedColumnParallelLinearWithLoRA.can_replace_layer: reject
    # 2-slice modules where output sizes differ.
    try:
        if _replace_text(
            target,
            "        return (\n"
            "            type(source_layer) is MergedColumnParallelLinear\n"
            "            and len(packed_modules_list) == 2\n"
            "        )\n"
            "\n"
            "\n"
            "class QKVParallelLinearWithLoRA",
            "        if type(source_layer) is not MergedColumnParallelLinear:\n"
            "            return False\n"
            "        if len(packed_modules_list) != 2:\n"
            "            return False\n"
            "        # Reject when output_sizes has more entries than packed_modules_list\n"
            "        # (e.g. in_proj_qkvz packs 2 logical modules across 4 physical slices).\n"
            "        # VariableSlice handles the splitting.\n"
            "        if hasattr(source_layer, 'output_sizes'):\n"
            "            if len(source_layer.output_sizes) != len(packed_modules_list):\n"
            "                return False\n"
            "        return True\n"
            "\n"
            "\n"
            "class QKVParallelLinearWithLoRA",
        ):
            changed.append(f"{target} (MergedColumnParallelLinear reject heterogeneous)")
    except RuntimeError:
        pass

    # 6b. MergedColumnParallelLinearVariableSliceWithLoRA.can_replace_layer:
    # accept 2-slice modules where output sizes differ.
    try:
        if _replace_text(
            target,
            "        # If packed_modules_list has exactly 2 items, let\n"
            "        # MergedColumnParallelLinearWithLoRA handle it\n"
            "        if len(packed_modules_list) == 2:\n"
            "            return False\n",
            "        # If packed_modules_list has exactly 2 items, check if output_sizes\n"
            "        # has more entries (physical slices > logical modules) — if so, we\n"
            "        # handle the splitting (not the equal-size class)\n"
            "        if len(packed_modules_list) == 2:\n"
            "            if hasattr(source_layer, 'output_sizes'):\n"
            "                if len(source_layer.output_sizes) != len(packed_modules_list):\n"
            "                    return True\n"
            "            return False\n",
        ):
            changed.append(f"{target} (VariableSlice accept 2-slice heterogeneous)")
    except RuntimeError:
        pass

    # 6c. VariableSlice.set_lora: handle list shorter than n_slices.
    # When packed_modules_list has 2 items but output_sizes has 4 (e.g.
    # in_proj_qkv covering Q+K+V = 3 slices, in_proj_z covering 1 slice),
    # lora_b is a list of 2 tensors but n_slices=4. Each tensor must be
    # split across the physical slices it covers.
    try:
        if _replace_text(
            target,
            "        # lora_b shape: (total_output_size, rank) -\n"
            "        # split along dim 0 based on output_sizes\n"
            "        if isinstance(lora_b, torch.Tensor):\n"
            "            output_sizes = self.base_layer.output_sizes\n"
            "            lora_b_list = []\n"
            "            start_idx = 0\n"
            "            for output_size in output_sizes:\n"
            "                end_idx = start_idx + output_size\n"
            "                lora_b_list.append(lora_b[start_idx:end_idx, :])\n"
            "                start_idx = end_idx\n"
            "            lora_b = lora_b_list\n"
            "\n"
            "        # Now call parent's set_lora which expects lists\n"
            "        super().set_lora(index, lora_a, lora_b)\n",
            "        # lora_b: split into n_slices based on output_sizes\n"
            "        if isinstance(lora_b, torch.Tensor):\n"
            "            lora_b_list = []\n"
            "            start_idx = 0\n"
            "            for output_size in output_sizes:\n"
            "                end_idx = start_idx + output_size\n"
            "                lora_b_list.append(lora_b[start_idx:end_idx, :])\n"
            "                start_idx = end_idx\n"
            "            lora_b = lora_b_list\n"
            "        elif isinstance(lora_b, list) and len(lora_b) < self.n_slices:\n"
            "            # Each lora_b tensor may cover multiple physical slices.\n"
            "            # Greedily match: consume output_sizes entries until cumulative\n"
            "            # size matches the tensor's dim 0.\n"
            "            expanded_b = []\n"
            "            expanded_a = []\n"
            "            slice_idx = 0\n"
            "            for j, b_tensor in enumerate(lora_b):\n"
            "                if b_tensor is None:\n"
            "                    expanded_b.append(None)\n"
            "                    expanded_a.append(lora_a[j] if j < len(lora_a) else None)\n"
            "                    slice_idx += 1\n"
            "                    continue\n"
            "                tensor_dim = b_tensor.shape[0]\n"
            "                cumulative = 0\n"
            "                start_slice = slice_idx\n"
            "                while slice_idx < len(output_sizes) and cumulative < tensor_dim:\n"
            "                    cumulative += output_sizes[slice_idx]\n"
            "                    slice_idx += 1\n"
            "                split_start = 0\n"
            "                a_for_slice = lora_a[j] if j < len(lora_a) else None\n"
            "                for s in range(start_slice, slice_idx):\n"
            "                    sz = output_sizes[s]\n"
            "                    expanded_b.append(b_tensor[split_start:split_start + sz, :])\n"
            "                    expanded_a.append(a_for_slice)\n"
            "                    split_start += sz\n"
            "            lora_b = expanded_b\n"
            "            lora_a = expanded_a\n"
            "\n"
            "        # Now call parent's set_lora which expects lists\n"
            "        super().set_lora(index, lora_a, lora_b)\n",
        ):
            changed.append(f"{target} (VariableSlice split list shorter than n_slices)")
    except RuntimeError:
        pass

    return changed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def resolve_site_packages(venv_path: Path) -> Path:
    lib_dir = venv_path / "lib"
    if not lib_dir.exists():
        raise RuntimeError(f"Missing venv lib directory: {lib_dir}")
    candidates = sorted(lib_dir.glob("python*/site-packages"))
    if not candidates:
        raise RuntimeError(f"Could not find site-packages under {lib_dir}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply Qwen3.5 compatibility patches to .venv packages."
    )
    parser.add_argument(
        "--venv",
        default=".venv",
        help="Path to virtualenv root (default: .venv)",
    )
    args = parser.parse_args()

    venv_path = Path(args.venv).resolve()
    site_packages = resolve_site_packages(venv_path)

    changed: list[str] = []
    changed.extend(patch_vllm_qwen3_vl(site_packages))
    changed.extend(patch_vllm_qwen3_5_model(site_packages))
    changed.extend(patch_vllm_renderer_language_model_only(site_packages))
    changed.extend(patch_trl_vllm_generation(site_packages))
    changed.extend(patch_trl_vllm_serve(site_packages))
    changed.extend(patch_transformers_bnb_loading(site_packages))
    changed.extend(patch_vllm_lora_column_parallel(site_packages))

    if changed:
        print(f"Applied {len(changed)} patches:")
        for path in changed:
            print(f"  - {path}")
    else:
        print("All compatibility patches already applied.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
