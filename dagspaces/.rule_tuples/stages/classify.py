from typing import Any, Dict, List
import pandas as pd
import os
import json
from omegaconf import OmegaConf

from dagspaces.common.vllm_inference import run_vllm_inference
from dagspaces.common.stage_utils import (
    maybe_silence_vllm_logs,
    to_json_str,
    serialize_arrow_unfriendly_in_row,
)


def run_classification_stage(df: pd.DataFrame, cfg):
    """Classification stage.

    Heuristic baseline by default; LLM classification when cfg.runtime.use_llm_classify is True.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["name", "rule_text", "is_relevant"])
    out = df.copy()

    def _heuristic(text: Any) -> bool:
        s = str(text or "").lower()
        if not s:
            return False
        keywords = ["privacy", "consent", "data", "information", "share", "disclose", "leak", "dox"]
        return any(k in s for k in keywords)

    use_llm = bool(getattr(cfg.runtime, "use_llm_classify", False))
    if not use_llm:
        out["is_relevant"] = out["rule_text"].apply(_heuristic)
        out["classification_mode"] = "heuristic"
        return out

    # LLM classification via direct vLLM
    system_prompt = str(getattr(cfg.prompt, "system_prompt", ""))
    prompt_template = str(getattr(cfg.prompt, "prompt_template", ""))

    def _format_prompt(rule_text: str) -> str:
        return prompt_template.replace("{{rule_text}}", str(rule_text or ""))

    # Prefer stage-specific sampling params when present
    try:
        sp_src = getattr(cfg, "sampling_params_classify", getattr(cfg, "sampling_params", {}))
        sampling_params = OmegaConf.to_container(sp_src, resolve=True) if isinstance(sp_src, (dict,)) or hasattr(sp_src, "_get_node") else dict(sp_src)
    except Exception:
        sampling_params = dict(getattr(cfg, "sampling_params", {}))

    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        maybe_silence_vllm_logs()
        user = _format_prompt(row.get("rule_text"))
        from datetime import datetime as _dt
        return {
            **row,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            "sampling_params": sampling_params,
            "ts_start": _dt.now().timestamp(),
        }

    def _post(row: Dict[str, Any]) -> Dict[str, Any]:
        from datetime import datetime as _dt
        text = str(row.get("generated_text") or "").strip().upper()
        is_rel = text.startswith("YES") or ("YES" in text and "NO" not in text)
        ts_end = _dt.now().timestamp()
        usage = row.get("usage") or row.get("token_counts") or None
        try:
            serialize_nested = bool(getattr(cfg.runtime, "serialize_nested_json", True))
        except Exception:
            serialize_nested = True
        if serialize_nested:
            serialize_arrow_unfriendly_in_row(row, [
                "messages",
                "sampling_params",
                "usage",
                "token_counts",
            ])
        return {
            **row,
            "is_relevant": bool(is_rel),
            "llm_output": row.get("generated_text"),
            "classification_mode": "llm",
            "latency_s": (float(ts_end) - float(row.get("ts_start", ts_end))),
            "token_usage_prompt": ((usage or {}).get("prompt_tokens") or (usage or {}).get("input_tokens")),
            "token_usage_output": ((usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens")),
            "token_usage_total": ((usage or {}).get("total_tokens")),
            "_progress_row": 1,
        }

    result_df = run_vllm_inference(out, cfg, _pre, _post, "classify")
    return result_df


