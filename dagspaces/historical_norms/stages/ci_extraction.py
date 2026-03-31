# CI extraction stage using the shared direct vLLM helper.
#
# Converts CI reasoning traces into structured 5-component information flow
# tuples following Nissenbaum's Contextual Integrity framework.

import pandas as pd
import json
from omegaconf import OmegaConf
from typing import Any, Dict

from dagspaces.common.vllm_inference import run_vllm_inference
from ..ci_schema import CIExtractionResult
from ._utils import extract_json, clean_for_parquet


def _parse_reasoning_json(raw: str) -> Dict[str, Any]:
    """Parse the ci_reasoning_json column, handling string or dict input."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def run_ci_extraction_stage(df, cfg: Any) -> pd.DataFrame:
    """
    CI Extraction: Convert reasoning traces to structured CI information flow tuples.

    Uses constrained decoding to map CI reasoning traces to structured
    5-component information flow tuples (Subject, Sender, Recipient,
    Information Type, Transmission Principle) with contextual metadata.

    Args:
        df: Input pandas DataFrame
        cfg: Configuration object
    """
    # Filter to rows that have CI reasoning JSON with actual flows
    total_rows = len(df)
    df = df[df["ci_reasoning_json"].notna() & (df["ci_reasoning_json"] != "")]

    # Further filter: only keep rows where the reasoning stage found flows
    if "has_information_exchange" in df.columns:
        df = df[df["has_information_exchange"] == True]  # noqa: E712
    if "ci_flow_count" in df.columns:
        df = df[df["ci_flow_count"] > 0]

    skipped = total_rows - len(df)
    print(f"[ci_extraction] Filtered to {len(df)} rows with flows "
          f"(skipped {skipped} rows with no information exchange)")

    if len(df) == 0:
        print("[ci_extraction] No rows with CI reasoning flows, returning empty")
        return df

    # Expand rows: one row per identified flow from the reasoning stage
    expanded_rows = []
    for _, row in df.iterrows():
        reasoning_obj = _parse_reasoning_json(row.get("ci_reasoning_json", "{}"))
        flows = reasoning_obj.get("flows", [])

        if not flows:
            # Safety fallback -- should not reach here after filtering above
            continue

        for i, flow_entry in enumerate(flows):
            new_row = row.to_dict()
            new_row["ci_flow_index"] = i
            new_row["ci_reasoning_trace"] = flow_entry.get("reasoning", "")
            new_row["ci_flow_snippet"] = flow_entry.get("original_text_snippet", "")
            new_row["ci_flow_context"] = flow_entry.get("context_identified", "")
            new_row["ci_flow_direction"] = flow_entry.get("flow_direction", "")
            new_row["ci_is_new_flow_reasoning"] = flow_entry.get("is_new_flow", False)
            expanded_rows.append(new_row)

    if not expanded_rows:
        print("[ci_extraction] No flows to expand after parsing, returning empty")
        return pd.DataFrame()

    df = pd.DataFrame(expanded_rows)
    print(f"[ci_extraction] Expanded to {len(df)} rows from reasoning flows")

    prompt_cfg = OmegaConf.select(cfg, "prompt_ci_extraction") or OmegaConf.select(cfg, "prompt")
    if prompt_cfg is None:
        raise RuntimeError(
            "[ci_extraction] No prompt config found at 'prompt_ci_extraction' or 'prompt'. "
            "Check config.yaml defaults and pipeline overrides."
        )

    system_prompt = OmegaConf.select(prompt_cfg, "system_prompt")
    prompt_template = OmegaConf.select(prompt_cfg, "prompt_template")
    if not system_prompt or not prompt_template:
        raise RuntimeError(
            f"[ci_extraction] Prompt config is missing required fields. "
            f"system_prompt={'present' if system_prompt else 'MISSING'}, "
            f"prompt_template={'present' if prompt_template else 'MISSING'}"
        )
    system_prompt = str(system_prompt)
    prompt_template = str(prompt_template)
    print(f"[ci_extraction] Loaded prompt from config "
          f"(system_prompt: {len(system_prompt)} chars, prompt_template: {len(prompt_template)} chars)",
          flush=True)

    def _format_prompt(row: Dict[str, Any]) -> str:
        text = str(row.get("ci_flow_snippet") or row.get("article_text") or "")
        reasoning = str(row.get("ci_reasoning_trace", ""))
        book_summary = str(row.get("book_summary") or "")
        return (prompt_template
                .replace("{{article_text}}", text)
                .replace("{{reasoning_trace}}", reasoning)
                .replace("{{book_summary}}", book_summary))

    sampling_params = dict(
        OmegaConf.to_container(
            OmegaConf.select(cfg, "sampling_params"),
            resolve=True,
        ) or {}
    )
    sampling_params.setdefault("temperature", 0.0)
    sampling_params.setdefault("max_tokens", 2048)
    json_schema = CIExtractionResult.model_json_schema()
    # Use vLLM guided decoding for structured output — constrains token
    # generation to valid JSON matching the schema.
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
        if obj is not None:
            # Schema constrains output to a single flow per call.
            ci_flow = obj.get("flow", {})
            flow_tuple = ci_flow.get("flow", {})
            # Coerce all tuple fields to str — the LLM occasionally
            # returns a list (e.g. multiple subjects) instead of a
            # single string, which causes Arrow mixed-type failures.
            def _to_str(v):
                if v is None:
                    return None
                if isinstance(v, list):
                    return "; ".join(str(x) for x in v)
                return str(v)
            result_row["ci_subject"] = _to_str(flow_tuple.get("subject"))
            result_row["ci_sender"] = _to_str(flow_tuple.get("sender"))
            result_row["ci_recipient"] = _to_str(flow_tuple.get("recipient"))
            result_row["ci_information_type"] = _to_str(flow_tuple.get("information_type"))
            result_row["ci_transmission_principle"] = _to_str(flow_tuple.get("transmission_principle"))
            result_row["ci_context"] = _to_str(ci_flow.get("context"))
            result_row["ci_appropriateness"] = _to_str(ci_flow.get("appropriateness"))
            result_row["ci_norms_invoked"] = ci_flow.get("norms_invoked", [])
            result_row["ci_norm_source"] = _to_str(ci_flow.get("norm_source"))
            result_row["ci_is_new_flow"] = bool(ci_flow.get("is_new_flow", False))
            result_row["ci_confidence_qual"] = _to_str(ci_flow.get("confidence_qual"))
            result_row["ci_confidence_quant"] = ci_flow.get("confidence_quant")
        else:
            print(f"Error parsing generated JSON: {parse_error}")
            result_row["extraction_error"] = str(parse_error)
        return result_row

    result_df = run_vllm_inference(
        df=df,
        cfg=cfg,
        preprocess=_preprocess,
        postprocess=_postprocess,
        stage_name="ci_extraction",
    )

    print(f"[ci_extraction] Completed inference, {len(result_df)} results")
    result_df = clean_for_parquet(result_df, extra_cols=["ci_reasoning_data", "ci_norms_invoked"], stage_name="ci_extraction")
    return result_df
