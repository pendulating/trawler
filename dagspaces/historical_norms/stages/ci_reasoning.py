# CI reasoning stage using the shared direct vLLM helper.
#
# Identifies information flows in text through a Contextual Integrity lens.
# Produces reasoning traces about who exchanges what information with whom,
# in what societal context, and whether the flow is appropriate.

import pandas as pd
import json
from omegaconf import OmegaConf
from typing import Any, Dict

from dagspaces.common.vllm_inference import run_vllm_inference
from ..ci_schema import CIReasoningList
from ._utils import extract_json, clean_for_parquet


def run_ci_reasoning_stage(df, cfg: Any) -> pd.DataFrame:
    """
    CI Reasoning: Identify information flows in text via Contextual Integrity.

    Uses Qwen 2.5 72B to identify information exchanges between agents and
    reason about their context, direction, and appropriateness.

    Args:
        df: Input pandas DataFrame
        cfg: Configuration object
    """
    if df is None or len(df) == 0:
        print("[ci_reasoning] Empty input, returning empty")
        return pd.DataFrame()

    prompt_cfg = OmegaConf.select(cfg, "prompt_ci_reasoning") or OmegaConf.select(cfg, "prompt")
    if prompt_cfg is None:
        raise RuntimeError(
            "[ci_reasoning] No prompt config found at 'prompt_ci_reasoning' or 'prompt'. "
            "Check config.yaml defaults and pipeline overrides."
        )

    system_prompt = OmegaConf.select(prompt_cfg, "system_prompt")
    prompt_template = OmegaConf.select(prompt_cfg, "prompt_template")
    if not system_prompt or not prompt_template:
        raise RuntimeError(
            f"[ci_reasoning] Prompt config is missing required fields. "
            f"system_prompt={'present' if system_prompt else 'MISSING'}, "
            f"prompt_template={'present' if prompt_template else 'MISSING'}"
        )
    system_prompt = str(system_prompt)
    prompt_template = str(prompt_template)
    print(f"[ci_reasoning] Loaded prompt from config "
          f"(system_prompt: {len(system_prompt)} chars, prompt_template: {len(prompt_template)} chars)",
          flush=True)

    def _format_prompt(row_or_text) -> str:
        if isinstance(row_or_text, dict):
            article_text = str(row_or_text.get("article_text") or "")
            book_summary = str(row_or_text.get("book_summary") or "")
        else:
            article_text = str(row_or_text or "")
            book_summary = ""
        return (prompt_template
                .replace("{{article_text}}", article_text)
                .replace("{{book_summary}}", book_summary))

    sampling_params = dict(
        OmegaConf.to_container(
            OmegaConf.select(cfg, "sampling_params"),
            resolve=True,
        ) or {}
    )
    sampling_params.setdefault("temperature", 0.0)
    sampling_params.setdefault("max_tokens", 4096)
    json_schema = CIReasoningList.model_json_schema()
    # Use vLLM guided decoding for structured output — constrains token
    # generation to valid JSON matching the schema. Faster and more reliable
    # than appending the schema as text and hoping for valid JSON.
    sampling_params["guided_decoding"] = {"json": json_schema}

    def _preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        result_row = dict(row)
        user_prompt = _format_prompt(result_row)
        result_row["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result_row["sampling_params"] = sampling_params
        return result_row

    def _postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        result_row = dict(row)
        result_row.pop("messages", None)
        result_row.pop("sampling_params", None)
        result_row.pop("usage", None)
        gen_text = result_row.get("generated_text", "{}")
        obj, parse_error = extract_json(gen_text)
        result_row["ci_reasoning_json"] = json.dumps(obj) if obj else None
        result_row["ci_reasoning_parse_error"] = parse_error

        if obj is not None:
            flows = obj.get("flows", [])
            result_row["has_information_exchange"] = bool(obj.get("has_information_exchange", len(flows) > 0))
            result_row["ci_flow_count"] = len(flows)
            result_row["ci_reasoning_text"] = obj.get("reasoning", "")
        else:
            result_row["has_information_exchange"] = False
            result_row["ci_flow_count"] = 0
            result_row["ci_reasoning_text"] = ""
        return result_row

    result_df = run_vllm_inference(
        df=df,
        cfg=cfg,
        preprocess=_preprocess,
        postprocess=_postprocess,
        stage_name="ci_reasoning",
    )

    records = result_df.to_dict("records")
    n_with_flows = sum(1 for r in records if r.get("has_information_exchange"))
    n_without_flows = len(records) - n_with_flows
    n_parse_errors = sum(1 for r in records if r.get("ci_reasoning_parse_error"))
    print(f"[ci_reasoning] Completed inference, {len(records)} results: "
          f"{n_with_flows} with flows, {n_without_flows} without flows, "
          f"{n_parse_errors} parse errors")

    result_df = clean_for_parquet(result_df, extra_cols=["ci_reasoning_data"], stage_name="ci_reasoning")
    return result_df
