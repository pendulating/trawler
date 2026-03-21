# Norm reasoning stage using the shared direct vLLM helper.

import pandas as pd
import json
from omegaconf import OmegaConf
from typing import Any, Dict

from dagspaces.common.vllm_inference import run_vllm_inference
from ..ci_schema import RazNormReasoningList


def _clean_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame to avoid PyArrow serialization issues.
    
    Removes or converts columns that cause parquet write errors:
    - Empty struct columns (e.g., 'metadata' with {})
    - Complex nested types that Arrow can't handle
    """
    # Columns that commonly cause issues
    problematic_cols = [
        "metadata", "reasoning_data", "__inference_error__", "embeddings"
    ]
    
    for col in problematic_cols:
        if col in df.columns:
            # Check if column contains empty dicts/structs
            try:
                sample = df[col].dropna().head(1)
                if len(sample) > 0:
                    val = sample.iloc[0]
                    if val == {} or val == []:
                        df = df.drop(columns=[col])
                        print(f"[norm_reasoning] Dropped empty struct column: {col}")
                        continue
            except Exception:
                pass
            
            # Convert complex objects to JSON strings for safe parquet storage
            try:
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
            except Exception:
                df = df.drop(columns=[col])
                print(f"[norm_reasoning] Dropped problematic column: {col}")
    
    return df

try:
    from json_repair import repair_json
    _JSON_REPAIR_OK = True
except ImportError:
    _JSON_REPAIR_OK = False


def run_norm_reasoning_stage(df, cfg: Any) -> pd.DataFrame:
    """
    Stage 1 of Norm Extraction: Raz Norm Reasoning.
    Uses Qwen 2.5 72B to identify norms (per Raz's anatomy) and provide reasoning traces.

    Args:
        df: Input pandas DataFrame
        cfg: Configuration object
    """
    if df is None or len(df) == 0:
        print("[norm_reasoning] Empty input, returning empty")
        return pd.DataFrame()
    
    prompt_cfg = OmegaConf.select(cfg, "prompt_reasoning") or OmegaConf.select(cfg, "prompt")
    if prompt_cfg is None:
        raise RuntimeError(
            "[norm_reasoning] No prompt config found at 'prompt_reasoning' or 'prompt'. "
            "Check config.yaml defaults and pipeline overrides."
        )

    system_prompt = OmegaConf.select(prompt_cfg, "system_prompt")
    prompt_template = OmegaConf.select(prompt_cfg, "prompt_template")
    if not system_prompt or not prompt_template:
        raise RuntimeError(
            f"[norm_reasoning] Prompt config is missing required fields. "
            f"system_prompt={'present' if system_prompt else 'MISSING'}, "
            f"prompt_template={'present' if prompt_template else 'MISSING'}"
        )
    system_prompt = str(system_prompt)
    prompt_template = str(prompt_template)
    print(f"[norm_reasoning] Loaded prompt from config "
          f"(system_prompt: {len(system_prompt)} chars, prompt_template: {len(prompt_template)} chars)",
          flush=True)

    def _format_prompt(row: Dict[str, Any]) -> str:
        article_text = str(row.get("article_text", ""))
        book_context = ""
        title = row.get("book_title", "")
        author = row.get("book_author", "")
        summary = row.get("book_summary", "")
        if title:
            book_context = f'Novel Context:\nThe text below is a short excerpt from the novel "{title}"'
            if author:
                book_context += f" by {author}"
            book_context += (
                ". It is one of many consecutive chunks extracted from the full novel. "
                "The excerpt may begin or end mid-scene. Use the summary below to "
                "understand the broader societal context of the novel when identifying norms.\n"
            )
            if summary:
                book_context += f"\nNovel summary: {summary}\n\n---\n\n"
            else:
                book_context += "\n---\n\n"
        return (prompt_template
                .replace("{{book_context}}", book_context)
                .replace("{{article_text}}", article_text))

    sampling_params = dict(
        OmegaConf.to_container(
            OmegaConf.select(cfg, "sampling_params"),
            resolve=True,
        ) or {}
    )
    sampling_params.setdefault("temperature", 0.0)
    sampling_params.setdefault("max_tokens", 4096)
    json_schema = RazNormReasoningList.model_json_schema()
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
            result_row["reasoning_data"] = obj
            result_row["has_prescriptive_content"] = obj.get("has_prescriptive_content", False)

            norms = obj.get("norms", [])
            result_row["has_norms"] = len(norms) > 0
            result_row["norm_count"] = len(norms)

            if norms:
                result_row["reasoning_trace"] = norms[0].get("reasoning", "")
                result_row["norm_snippet"] = norms[0].get("original_text_snippet", "")
                result_row["preliminary_normative_force"] = norms[0].get("preliminary_normative_force", "")
                result_row["governs_information_flow"] = norms[0].get("governs_information_flow", None)
            else:
                result_row["reasoning_trace"] = None
                result_row["norm_snippet"] = None
                result_row["preliminary_normative_force"] = None
                result_row["governs_information_flow"] = None
        else:
            print(f"Error parsing reasoning JSON: {parse_error}")
            result_row["reasoning_error"] = str(parse_error)
            result_row["has_norms"] = False
            result_row["has_prescriptive_content"] = False
            result_row["norm_count"] = 0
            result_row["reasoning_trace"] = None
            result_row["norm_snippet"] = None
            result_row["preliminary_normative_force"] = None
            result_row["governs_information_flow"] = None
        return result_row

    result_df = run_vllm_inference(
        df=df,
        cfg=cfg,
        preprocess=_preprocess,
        postprocess=_postprocess,
        stage_name="norm_reasoning",
    )

    print(f"[norm_reasoning] Completed inference, {len(result_df)} results")

    # Explode rows so each identified norm gets its own row.
    # Must happen before _clean_for_parquet which drops reasoning_data.
    if "reasoning_data" in result_df.columns:
        rows = []
        for _, row in result_df.iterrows():
            data = row.get("reasoning_data")
            if isinstance(data, dict):
                norms = data.get("norms", [])
                if norms and len(norms) > 0:
                    added = 0
                    for i, norm in enumerate(norms):
                        reasoning_text = norm.get("reasoning", "")
                        if not str(reasoning_text).strip():
                            print(f"[norm_reasoning] Skipping norm {i} in chunk "
                                  f"{row.get('chunk_id', '?')}: empty reasoning")
                            continue
                        new_row = row.to_dict()
                        new_row["norm_index"] = i
                        new_row["reasoning_trace"] = reasoning_text
                        new_row["norm_snippet"] = norm.get("original_text_snippet", "")
                        new_row["preliminary_normative_force"] = norm.get("preliminary_normative_force", "")
                        new_row["governs_information_flow"] = norm.get("governs_information_flow", None)
                        rows.append(new_row)
                        added += 1
                    if added == 0:
                        # All norms had empty reasoning; keep chunk row
                        rows.append(row.to_dict())
                else:
                    rows.append(row.to_dict())
            else:
                rows.append(row.to_dict())
        pre_count = len(result_df)
        result_df = pd.DataFrame(rows)
        print(f"[norm_reasoning] Exploded {pre_count} rows -> {len(result_df)} rows (one per norm)")

    result_df = _clean_for_parquet(result_df)
    return result_df
