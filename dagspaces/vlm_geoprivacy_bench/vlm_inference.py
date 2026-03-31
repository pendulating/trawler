"""Multimodal VLM inference utility for VLM-GeoPrivacyBench.

Reuses GPU/env helpers from dagspaces.common.vllm_inference but handles
image inputs via vLLM's multi-modal API.  Falls back to native
transformers for model architectures that vLLM no longer supports
(e.g. Mllama / Llama 3.2 Vision, dropped after vLLM 0.10.2).
"""

from __future__ import annotations

import base64
import logging
import os
import re
from typing import Any, Dict, List

import pandas as pd

from dagspaces.common.vllm_inference import (
    _build_engine_kwargs,
    _build_sampling_params,
    detect_num_gpus,
    filter_vllm_engine_kwargs,
    get_pcie_nccl_env_vars,
    get_vllm_runtime_env_vars,
)

logger = logging.getLogger(__name__)

# Model families that require the native transformers fallback because
# vLLM dropped support for their multimodal architecture.
_TRANSFORMERS_FALLBACK_FAMILIES = {"llama-vision"}


def _run_transformers_vlm_inference(
    df: pd.DataFrame,
    cfg: Any,
    prompt_text: str,
    image_col: str,
    stage_name: str,
    sys_msg: str = "",
    usr_text: str = "",
) -> pd.DataFrame:
    """Native transformers inference for VLMs that vLLM can't serve.

    Uses MllamaForConditionalGeneration + AutoProcessor for Llama 3.2 Vision.
    The processor must receive structured chat messages (not a pre-rendered
    string) so it can generate the correct cross_attention_mask linking
    image tokens to pixel data.

    Args:
        sys_msg: Raw system message text (before chat template).
        usr_text: Raw user message text (before chat template).
            If both are empty, falls back to prompt_text (legacy).
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    model_source = str(cfg.model.model_source)
    print(f"[{stage_name}] Using native transformers fallback for {model_source}")

    processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,  # Mllama has intentional embed/lm_head size diffs
    )

    # Load and merge LoRA adapter if specified
    _lora_path = str(getattr(cfg.model, "lora_path", "") or "")
    if _lora_path:
        from peft import PeftModel
        print(f"[{stage_name}] Loading LoRA adapter: {_lora_path}")
        model = PeftModel.from_pretrained(model, _lora_path)
        model = model.merge_and_unload()
        print(f"[{stage_name}] LoRA merged into base model")

    # Sampling config
    sp_dict = dict(getattr(cfg, "sampling_params", {}) or {})
    max_tokens = int(sp_dict.get("max_tokens", 1024))
    temperature = float(sp_dict.get("temperature", 0.0))
    do_sample = temperature > 0

    # Build Mllama-compatible prompt with <|image|> placeholder.
    # The processor needs the literal <|image|> token in the text so it can
    # create the cross_attention_mask linking image tokens to pixel data.
    # MllamaProcessor does NOT have apply_chat_template, so we build the
    # prompt string manually in Llama 3.x chat format.
    if sys_msg or usr_text:
        combined_text = (sys_msg + "\n\n" + usr_text).strip()
        mllama_prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"<|image|>{combined_text}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        mllama_prompt = None

    results = []
    valid_indices = []

    print(f"[{stage_name}] Processing {len(df)} images...")
    for i, (idx, row) in enumerate(df.iterrows()):
        img_path = str(row[image_col])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            results.append("")
            valid_indices.append(idx)
            continue

        text = mllama_prompt if mllama_prompt is not None else prompt_text

        inputs = processor(
            images=image,
            text=text,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
        # Decode only the generated tokens (skip the prompt)
        gen_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        gen_text = processor.decode(gen_ids[0], skip_special_tokens=True).strip()
        results.append(gen_text)
        valid_indices.append(idx)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"[{stage_name}] Processed {i + 1}/{len(df)} images")

    result_df = df.copy()
    result_df["generated_text"] = ""
    for idx, text in zip(valid_indices, results):
        result_df.at[idx, "generated_text"] = text

    print(f"[{stage_name}] Completed transformers VLM inference, {len(results)} results")
    return result_df


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_vlm_inference(
    df: pd.DataFrame,
    cfg: Any,
    prompt_text: str,
    image_col: str = "image_path",
    stage_name: str = "vlm_inference",
    sys_msg: str = "",
    usr_text: str = "",
    extra_sampling_params: dict | None = None,
) -> pd.DataFrame:
    """Run multimodal vLLM batch inference on a DataFrame of images.

    Args:
        df: DataFrame with at least an image path column.
        cfg: Hydra config with model.model_source, model.engine_kwargs,
             model.vlm_kwargs, and sampling_params.
        prompt_text: Pre-built prompt string (same for all rows).
        image_col: Column name containing image file paths.
        stage_name: Label for log messages.
        sys_msg: Raw system message (for transformers fallback).
        usr_text: Raw user message text (for transformers fallback).
        extra_sampling_params: Additional sampling params to merge (e.g.
            guided_decoding for structured output).

    Returns:
        Copy of df with 'generated_text' column added.
    """
    if df is None or len(df) == 0:
        print(f"[{stage_name}] Empty input, returning empty DataFrame")
        return pd.DataFrame()

    # Route to native transformers for unsupported vLLM architectures
    model_family = str(getattr(cfg.model, "model_family", ""))
    if model_family in _TRANSFORMERS_FALLBACK_FAMILIES:
        return _run_transformers_vlm_inference(
            df=df, cfg=cfg, prompt_text=prompt_text,
            image_col=image_col, stage_name=stage_name,
            sys_msg=sys_msg, usr_text=usr_text,
        )

    # Set runtime env vars before importing vLLM
    for k, v in {**get_pcie_nccl_env_vars(), **get_vllm_runtime_env_vars()}.items():
        os.environ.setdefault(k, v)

    env_snapshot = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "SLURM_GPUS_ON_NODE": os.environ.get("SLURM_GPUS_ON_NODE", "<unset>"),
        "VLLM_WORKER_MULTIPROC_METHOD": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "<unset>"),
    }
    print(f"[{stage_name}] Runtime env: {env_snapshot}")

    from vllm import LLM
    from vllm.multimodal.utils import fetch_image

    # Check for LoRA adapter
    lora_path = None
    lora_request = None
    try:
        from omegaconf import OmegaConf
        lora_path = str(OmegaConf.select(cfg, "model.lora_path") or "")
    except Exception:
        pass
    if lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("sft", 1, lora_path)
        print(f"[{stage_name}] LoRA adapter: {lora_path}")

    # Build engine kwargs from common builder, then overlay VLM-specific kwargs
    engine_kwargs = _build_engine_kwargs(cfg)

    vlm_kwargs = dict(getattr(cfg.model, "vlm_kwargs", {}) or {})
    for key in ("mm_processor_kwargs", "limit_mm_per_prompt"):
        if key in vlm_kwargs:
            val = vlm_kwargs[key]
            if hasattr(val, "items"):
                engine_kwargs[key] = dict(val)
            else:
                engine_kwargs[key] = val

    engine_kwargs = filter_vllm_engine_kwargs(engine_kwargs)

    print(
        f"[{stage_name}] Initializing vLLM VLM with: "
        f"{ {k: v for k, v in engine_kwargs.items() if k != 'model'} }"
    )
    print(f"[{stage_name}] Model: {engine_kwargs.get('model')}")

    llm = LLM(**engine_kwargs)

    # Determine whether to strip <think> blocks
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

    # Build sampling params
    sp_dict = dict(getattr(cfg, "sampling_params", {}) or {})
    if extra_sampling_params:
        sp_dict.update(extra_sampling_params)
    sampling_params = _build_sampling_params(sp_dict)

    # Build batch inputs with images
    print(f"[{stage_name}] Loading {len(df)} images and building batch inputs...")
    batch_inputs: List[Dict[str, Any]] = []
    valid_indices: List[int] = []

    for idx, row in df.iterrows():
        img_path = str(row[image_col])
        try:
            b64 = _image_to_base64(img_path)
            image_data = fetch_image(f"data:image/jpg;base64,{b64}")
            batch_inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": {"image": image_data},
            })
            valid_indices.append(idx)
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")

    if not batch_inputs:
        print(f"[{stage_name}] No valid images loaded, returning empty DataFrame")
        return pd.DataFrame()

    # Run inference in batches
    try:
        batch_size = int(getattr(cfg.model, "batch_size", 0) or 0)
    except Exception:
        batch_size = 0
    if batch_size <= 0:
        batch_size = len(batch_inputs)

    print(f"[{stage_name}] Running inference on {len(batch_inputs)} images (batch_size={batch_size})...")
    outputs = []
    for start in range(0, len(batch_inputs), batch_size):
        end = min(start + batch_size, len(batch_inputs))
        print(f"[{stage_name}] Generating batch {start // batch_size + 1}: rows {start}-{end - 1}")
        outputs.extend(llm.generate(batch_inputs[start:end], sampling_params, lora_request=lora_request))

    # Attach results to dataframe
    result_df = df.copy()
    result_df["generated_text"] = ""
    from dagspaces.common.vllm_inference import _strip_think_blocks
    for idx, output in zip(valid_indices, outputs):
        if output.outputs:
            gen_text = output.outputs[0].text.strip()
            if _strip_thinking and gen_text:
                gen_text = _strip_think_blocks(gen_text)
            result_df.at[idx, "generated_text"] = gen_text

    print(f"[{stage_name}] Completed VLM inference, {len(outputs)} results")
    return result_df
