# Norm extraction stage using the shared direct vLLM helper.

import pandas as pd
import json
from omegaconf import OmegaConf
from typing import Any, Dict

from dagspaces.common.vllm_inference import run_vllm_inference
from ..ci_schema import PrescriptiveNormExtractionResult


def _clean_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame to avoid PyArrow serialization issues.
    
    Removes or converts columns that cause parquet write errors:
    - Empty struct columns (e.g., 'metadata' with {})
    - Complex nested types that Arrow can't handle
    """
    # Columns that commonly cause issues - drop them
    problematic_cols = [
        "metadata", "reasoning_data", "raz_norms_raw",
        "__inference_error__", "embeddings"
    ]
    
    for col in problematic_cols:
        if col in df.columns:
            # Check if column contains empty dicts/structs
            try:
                sample = df[col].dropna().head(1)
                if len(sample) > 0:
                    val = sample.iloc[0]
                    # Drop if it's an empty dict or list of empty dicts
                    if val == {} or val == [] or (isinstance(val, list) and all(v == {} for v in val)):
                        df = df.drop(columns=[col])
                        print(f"[norm_extraction] Dropped empty struct column: {col}")
                        continue
            except Exception:
                pass
            
            # Convert complex objects to JSON strings for safe parquet storage
            try:
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
            except Exception:
                df = df.drop(columns=[col])
                print(f"[norm_extraction] Dropped problematic column: {col}")
    
    return df

try:
    from json_repair import repair_json
    _JSON_REPAIR_OK = True
except ImportError:
    _JSON_REPAIR_OK = False


def run_norm_extraction_stage(df, cfg: Any) -> pd.DataFrame:
    """
    Stage 2 of Norm Extraction: Raz Norm Tuple Extraction.
    Uses constrained decoding to map reasoning traces to structured Raz norm tuples.

    Args:
        df: Input pandas DataFrame
        cfg: Configuration object
    """
    # Filter to rows that have a reasoning trace
    df = df[df["reasoning_trace"].notna() & (df["reasoning_trace"] != "")]
    if len(df) == 0:
        print("[norm_extraction] No rows with reasoning traces, returning empty")
        return df

    prompt_cfg = OmegaConf.select(cfg, "prompt_extraction") or OmegaConf.select(cfg, "prompt")
    if prompt_cfg is None:
        raise RuntimeError(
            "[norm_extraction] No prompt config found at 'prompt_extraction' or 'prompt'. "
            "Check config.yaml defaults and pipeline overrides."
        )

    system_prompt = OmegaConf.select(prompt_cfg, "system_prompt")
    prompt_template = OmegaConf.select(prompt_cfg, "prompt_template")
    if not system_prompt or not prompt_template:
        raise RuntimeError(
            f"[norm_extraction] Prompt config is missing required fields. "
            f"system_prompt={'present' if system_prompt else 'MISSING'}, "
            f"prompt_template={'present' if prompt_template else 'MISSING'}"
        )
    system_prompt = str(system_prompt)
    prompt_template = str(prompt_template)
    print(f"[norm_extraction] Loaded prompt from config "
          f"(system_prompt: {len(system_prompt)} chars, prompt_template: {len(prompt_template)} chars)",
          flush=True)

    def _format_prompt(row: Dict[str, Any]) -> str:
        text = str(row.get("norm_snippet") or row.get("article_text") or "")
        reasoning = str(row.get("reasoning_trace", ""))
        return (prompt_template
                .replace("{{article_text}}", text)
                .replace("{{reasoning_trace}}", reasoning))

    sampling_params = dict(
        OmegaConf.to_container(
            OmegaConf.select(cfg, "sampling_params"),
            resolve=True,
        ) or {}
    )
    sampling_params.setdefault("temperature", 0.0)
    sampling_params.setdefault("max_tokens", 2048)
    json_schema = PrescriptiveNormExtractionResult.model_json_schema()
    sampling_params["guided_decoding"] = {"json": json_schema}

    def _extract_json(gen_text: str) -> tuple[dict | None, str | None]:
        obj = None
        parse_error = None
        json_text = gen_text

        if "{" in gen_text:
            start = gen_text.find("{")
            end = gen_text.rfind("}") + 1
            if start < end:
                json_text = gen_text[start:end]

        try:
            obj = json.loads(json_text)
        except json.JSONDecodeError as e:
            parse_error = e
            if _JSON_REPAIR_OK:
                try:
                    repaired = repair_json(json_text, return_objects=True)
                    if isinstance(repaired, dict):
                        obj = repaired
                except Exception as repair_err:
                    parse_error = f"JSON repair failed: {repair_err}"
        return obj, parse_error

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
        obj, parse_error = _extract_json(gen_text)
        if obj is not None:
            norms = obj.get("norms", [])
            result_row["raz_norms_raw"] = norms
            result_row["raz_norm_count"] = len(norms)

            if norms:
                first = norms[0]
                norm_tuple = first.get("norm", {})
                def _to_str(v):
                    if v is None:
                        return None
                    if isinstance(v, list):
                        return "; ".join(str(x) for x in v)
                    return str(v)
                for k, v in norm_tuple.items():
                    result_row[f"raz_{k}"] = _to_str(v)
                result_row["raz_normative_force"] = _to_str(first.get("normative_force"))
                result_row["raz_norm_articulation"] = _to_str(first.get("norm_articulation"))
                result_row["raz_norm_source"] = _to_str(first.get("norm_source"))
                result_row["raz_governs_info_flow"] = first.get("governs_information_flow")
                result_row["raz_info_flow_note"] = _to_str(first.get("information_flow_note"))
                result_row["raz_confidence_qual"] = _to_str(first.get("confidence_qual"))
                result_row["raz_confidence_quant"] = first.get("confidence_quant")
                result_row["raz_context"] = _to_str(first.get("context"))
        else:
            print(f"Error parsing generated JSON: {parse_error}")
            result_row["extraction_error"] = str(parse_error)
        return result_row

    result_df = run_vllm_inference(
        df=df,
        cfg=cfg,
        preprocess=_preprocess,
        postprocess=_postprocess,
        stage_name="norm_extraction",
    )

    print(f"[norm_extraction] Completed inference, {len(result_df)} results")
    result_df = _clean_for_parquet(result_df)
    return result_df
