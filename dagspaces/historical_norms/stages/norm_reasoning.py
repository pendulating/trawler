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
import logging
from omegaconf import OmegaConf
from typing import Any, Dict, Optional
from ..schema import NormReasoningList

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

def run_norm_reasoning_stage(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """
    Stage 1 of Norm Extraction: Reasoning.
    Uses Qwen 2.5 72B to identify latent norms and provide reasoning traces.
    """
    is_ray_ds = hasattr(df, "map_batches") and hasattr(df, "count") and _RAY_OK
    
    # Prefer scoped prompt configuration if available
    prompt_cfg = OmegaConf.select(cfg, "prompt_reasoning") or OmegaConf.select(cfg, "prompt")
    
    system_prompt = str(
        OmegaConf.select(prompt_cfg, "system_prompt") 
        or "You are an expert in social and historical norms. Analyze the following text and identify institutional statements (norms, rules, or strategies) using the Institutional Grammar 2.0 framework. For each identified statement, provide a detailed reasoning trace explaining its context, the actors involved, and its normative force."
    )
    prompt_template = str(
        OmegaConf.select(prompt_cfg, "prompt_template")
        or "Text:\n{{article_text}}\n\nIdentify and reason about the norms in this text:"
    )

    def _format_prompt(article_text: str) -> str:
        return prompt_template.replace("{{article_text}}", str(article_text or ""))

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
    json_schema = NormReasoningList.model_json_schema()
    schema_str = json.dumps(json_schema)

    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = _format_prompt(row.get("article_text"))
        return {
            **row,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "sampling_params": {
                "max_tokens": 16384,  # Very large buffer - some outputs exceed 100k chars
                "temperature": 0.0,
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
            row["reasoning_data"] = obj
            
            # Capture is_historical_context_present from model output
            row["is_historical_context_present"] = obj.get("is_historical_context_present", False)
            
            # Extract norms array
            norms = obj.get("norms", [])
            row["has_norms"] = len(norms) > 0
            row["norm_count"] = len(norms)
            
            # Always add trace columns for schema consistency
            # First norm goes into top-level columns; runner will explode all norms
            if norms:
                row["reasoning_trace"] = norms[0].get("reasoning", "")
                row["norm_snippet"] = norms[0].get("original_text_snippet", "")
                row["potential_type"] = norms[0].get("potential_type", "")
            else:
                row["reasoning_trace"] = None
                row["norm_snippet"] = None
                row["potential_type"] = None
        else:
            print(f"Error parsing reasoning JSON: {parse_error}")
            row["reasoning_error"] = str(parse_error)
            row["has_norms"] = False
            row["is_historical_context_present"] = False
            row["norm_count"] = 0
            row["reasoning_trace"] = None
            row["norm_snippet"] = None
            row["potential_type"] = None
            
        return row

    processor = build_processor(engine_config, preprocess=_pre, postprocess=_post)
    
    if is_ray_ds:
        return processor(df)
    
    ds = ray.data.from_pandas(df)
    out_ds = processor(ds).materialize()
    return out_ds.to_pandas()

