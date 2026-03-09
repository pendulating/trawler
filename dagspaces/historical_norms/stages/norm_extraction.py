# Norm extraction stage - tensor parallelism across 2 GPUs
# Uses vLLM V0 engine (VLLM_USE_V1=0 set in launcher before Python starts)

import pandas as pd
import json
import os
from omegaconf import OmegaConf
from typing import Any, Dict, List

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

    Architecture:
    1. Launcher sets VLLM_USE_V1=0 in shell BEFORE Python starts
    2. vLLM LLM class used directly with `distributed_executor_backend="mp"`
    3. Multiprocessing executor handles tensor parallelism across GPUs

    Args:
        df: Input pandas DataFrame
        cfg: Configuration object
    """
    # STEP 1: Verify VLLM_USE_V1=0 is set
    vllm_v1_env = os.environ.get("VLLM_USE_V1", "not set")
    print(f"[norm_extraction] VLLM_USE_V1 = {vllm_v1_env}")
    if vllm_v1_env != "0":
        print("[norm_extraction] WARNING: VLLM_USE_V1 is not '0' - V1 engine may be used!")
    
    # Filter to rows that have a reasoning trace
    df = df[df["reasoning_trace"].notna() & (df["reasoning_trace"] != "")]
    if len(df) == 0:
        print("[norm_extraction] No rows with reasoning traces, returning empty")
        return df
    
    # STEP 4: Ensure V1 shared memory broadcast is NOT disabled, then import vLLM
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    print(f"[norm_extraction] vLLM imported successfully")
    
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

    # Get model configuration
    model_source = str(getattr(cfg.model, "model_source"))
    ek = dict(getattr(cfg.model, "engine_kwargs", {}))
    
    # vLLM V0 LLM class config for 72B AWQ model on 2x A6000
    llm_kwargs = {
        "model": model_source,
        "tensor_parallel_size": ek.get("tensor_parallel_size", 2),
        "max_model_len": ek.get("max_model_len", 8192),
        "gpu_memory_utilization": ek.get("gpu_memory_utilization", 0.85),
        "trust_remote_code": ek.get("trust_remote_code", True),
        "enforce_eager": ek.get("enforce_eager", True),
        # Use multiprocessing executor for single-node tensor parallelism.
        # The "ray" backend deadlocks with vLLM V1 engine (vLLM #27249).
        "distributed_executor_backend": "mp",
    }
    
    # Add quantization for AWQ models
    if "awq" in model_source.lower():
        llm_kwargs["quantization"] = "awq"
    
    print(f"[norm_extraction] Initializing vLLM V0 with config: {llm_kwargs}")
    
    # Initialize vLLM LLM directly
    llm = LLM(**llm_kwargs)
    
    # Prepare prompts with chat template
    prompts = []
    rows_list = df.to_dict('records')
    
    for row in rows_list:
        user_prompt = _format_prompt(row)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            prompt = llm.get_tokenizer().apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        
        prompts.append(prompt)
    
    # Sampling parameters with JSON schema hint
    sampling_params = SamplingParams(
        max_tokens=ek.get("max_tokens", 4096),
        temperature=0.0,
    )
    
    json_schema = PrescriptiveNormExtractionResult.model_json_schema()
    schema_hint = f"\n\nRespond with valid JSON matching this schema: {json.dumps(json_schema)}"
    prompts = [p + schema_hint for p in prompts]
    
    print(f"[norm_extraction] Running inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Process outputs
    results = []
    for idx, row in enumerate(rows_list):
        output = outputs[idx]
        gen_text = output.outputs[0].text if output.outputs else "{}"
        
        # Try to extract JSON
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
                    print(f"JSON repair also failed: {repair_err}")
        
        result_row = dict(row)
        result_row["generated_text"] = gen_text
        
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
        
        results.append(result_row)
    
    print(f"[norm_extraction] Completed inference, {len(results)} results")
    
    # Clean DataFrame for parquet serialization
    result_df = pd.DataFrame(results)
    result_df = _clean_for_parquet(result_df)
    
    return result_df
