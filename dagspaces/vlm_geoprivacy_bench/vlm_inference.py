"""Multimodal vLLM inference utility for VLM-GeoPrivacyBench.

Reuses GPU/env helpers from dagspaces.common.vllm_inference but handles
image inputs via vLLM's multi-modal API.
"""

from __future__ import annotations

import base64
import logging
import os
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


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_vlm_inference(
    df: pd.DataFrame,
    cfg: Any,
    prompt_text: str,
    image_col: str = "image_path",
    stage_name: str = "vlm_inference",
) -> pd.DataFrame:
    """Run multimodal vLLM batch inference on a DataFrame of images.

    Args:
        df: DataFrame with at least an image path column.
        cfg: Hydra config with model.model_source, model.engine_kwargs,
             model.vlm_kwargs, and sampling_params.
        prompt_text: Pre-built prompt string (same for all rows).
        image_col: Column name containing image file paths.
        stage_name: Label for log messages.

    Returns:
        Copy of df with 'generated_text' column added.
    """
    if df is None or len(df) == 0:
        print(f"[{stage_name}] Empty input, returning empty DataFrame")
        return pd.DataFrame()

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

    # Build sampling params
    sp_dict = dict(getattr(cfg, "sampling_params", {}) or {})
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
        outputs.extend(llm.generate(batch_inputs[start:end], sampling_params))

    # Attach results to dataframe
    result_df = df.copy()
    result_df["generated_text"] = ""
    for idx, output in zip(valid_indices, outputs):
        if output.outputs:
            result_df.at[idx, "generated_text"] = output.outputs[0].text.strip()

    print(f"[{stage_name}] Completed VLM inference, {len(outputs)} results")
    return result_df
