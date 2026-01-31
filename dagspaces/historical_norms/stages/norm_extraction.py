# Fix uvloop/Ray event loop conflict in Python 3.12+
# Must be done before importing Ray or any async libraries
import asyncio
try:
    _policy = asyncio.get_event_loop_policy()
    if 'uvloop' in type(_policy).__module__:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except Exception:
    pass

import pandas as pd
import json
import os
from omegaconf import OmegaConf
from typing import Any, Dict, List
from ..schema import InstitutionalStatement

try:
    from json_repair import repair_json
    _JSON_REPAIR_OK = True
except ImportError:
    _JSON_REPAIR_OK = False

try:
    import ray
    from ray.data.llm import build_processor, vLLMEngineProcessorConfig
    _RAY_OK = True
except Exception:
    _RAY_OK = False

def _clean_dataframe_for_ray(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame to avoid PyArrow schema conflicts in Ray Data.
    
    Ray Data uses Arrow internally, which requires consistent types across all rows.
    This function:
    1. Drops columns with complex/nested types that cause schema issues
    2. Converts boolean columns to ensure no null/bool mismatches
    3. Keeps only columns needed for extraction
    """
    # Columns needed for extraction
    keep_cols = [
        "article_text", "norm_snippet", "reasoning_trace", "potential_type",
        "chunk_id", "gutenberg_id", "batch_uuid", "norm_index", "norm_count",
        "has_norms", "is_historical_context_present"
    ]
    
    # Drop problematic columns that cause Arrow type conflicts
    drop_cols = [
        "reasoning_data",  # Nested dict with varying array lengths
        "json_repaired",   # Mixed null/bool
        "embeddings",      # Large arrays
        "generated_text", "generated_tokens", "prompt_token_ids",
        "logprobs", "prompt_logprobs", "messages", "params", "prompt",
        "metrics", "__inference_error__"
    ]
    
    # Filter to keep columns that exist
    cols_to_keep = [c for c in keep_cols if c in df.columns]
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    
    # If we have specific columns to keep, use those; otherwise drop problematic ones
    if cols_to_keep:
        df_clean = df[cols_to_keep].copy()
    else:
        df_clean = df.drop(columns=cols_to_drop, errors='ignore').copy()
    
    # Ensure boolean columns have consistent types (no null/bool mismatch)
    bool_cols = ["has_norms", "is_historical_context_present"]
    for col in bool_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(False).astype(bool)
    
    return df_clean


def run_norm_extraction_stage(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """
    Stage 2 of Norm Extraction: Parameterized IG 2.0 Extraction.
    Uses constrained decoding to map reasoning traces to structured IG 2.0 objects.
    """
    is_ray_ds = hasattr(df, "map_batches") and hasattr(df, "count") and _RAY_OK
    
    # Prefer scoped prompt configuration if available
    prompt_cfg = OmegaConf.select(cfg, "prompt_extraction") or OmegaConf.select(cfg, "prompt")

    system_prompt = str(
        OmegaConf.select(prompt_cfg, "system_prompt")
        or "You are a formal logic and institutional grammar parser. Based on the provided text and reasoning trace, extract the institutional statement in valid IG 2.0 JSON format."
    )
    prompt_template = str(
        OmegaConf.select(prompt_cfg, "prompt_template")
        or "Source Text:\n{{article_text}}\n\nReasoning Trace:\n{{reasoning_trace}}\n\nExtract the IG 2.0 statement:"
    )

    def _format_prompt(row: Dict[str, Any]) -> str:
        # Use the specific snippet from reasoning if available, otherwise fallback to full text
        text = str(row.get("norm_snippet") or row.get("article_text") or "")
        reasoning = str(row.get("reasoning_trace", ""))
        return (prompt_template
                .replace("{{article_text}}", text)
                .replace("{{reasoning_trace}}", reasoning))

    # vLLM Engine Config
    ek = dict(getattr(cfg.model, "engine_kwargs", {}))
    ek.setdefault("max_model_len", 8192)
    ek.setdefault("tensor_parallel_size", 2)
    # Note: guided_decoding_backend was removed in vLLM 1.14+
    # Guided decoding is now configured via sampling_params only
    
    engine_config = vLLMEngineProcessorConfig(
        model_source=str(getattr(cfg.model, "model_source")),
        engine_kwargs=ek,
        batch_size=getattr(cfg.model, "batch_size", 8),
    )

    # Prepare guided decoding schema
    from ..schema import ExtractionResult
    json_schema = ExtractionResult.model_json_schema()
    schema_str = json.dumps(json_schema)

    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = _format_prompt(row)
        return {
            **row,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "sampling_params": {
                "max_tokens": 4096,  # Increased for complex IG 2.0 structures
                "guided_decoding": {"json": schema_str}
            }
        }

    def _post(row: Dict[str, Any]) -> Dict[str, Any]:
        gen_text = row.get("generated_text", "{}")
        obj = None
        parse_error = None
        
        # First attempt: standard JSON parsing
        try:
            obj = json.loads(gen_text)
        except json.JSONDecodeError as e:
            parse_error = e
            # Second attempt: use json_repair for truncated/malformed JSON
            if _JSON_REPAIR_OK:
                try:
                    repaired = repair_json(gen_text, return_objects=True)
                    if isinstance(repaired, dict):
                        obj = repaired
                        row["json_repaired"] = True
                        print(f"JSON repaired successfully (original error: {parse_error})")
                except Exception as repair_err:
                    print(f"JSON repair also failed: {repair_err}")
        
        if obj is not None:
            statements = obj.get("statements", [])
            row["ig20_statements_raw"] = statements
            row["ig20_count"] = len(statements)
            
            if statements:
                # For backward compatibility, flatten the first statement
                first = statements[0]
                inner = first.get("statement", {})
                for k, v in inner.items():
                    row[f"ig20_{k}"] = v
                row["ig20_confidence"] = first.get("confidence")
        else:
            print(f"Error parsing generated JSON: {parse_error}")
            row["extraction_error"] = str(parse_error)
            
        return row

    processor = build_processor(engine_config, preprocess=_pre, postprocess=_post)
    
    if is_ray_ds:
        # Filter rows that have a reasoning trace
        df = df.filter(lambda row: bool(row.get("reasoning_trace")))
        return processor(df)
    
    # Filter pandas df
    df = df[df["reasoning_trace"].notna() & (df["reasoning_trace"] != "")]
    if len(df) == 0:
        return df
    
    # Clean DataFrame to avoid PyArrow schema conflicts
    df = _clean_dataframe_for_ray(df)
        
    ds = ray.data.from_pandas(df)
    out_ds = processor(ds).materialize()
    return out_ds.to_pandas()

