from typing import Any, Dict, Optional
import pandas as pd
import json
import os
from omegaconf import OmegaConf
from dagspaces.uair.schema_builders import (
    object_schema,
    string_or_null,
    array_of_strings,
)

from dagspaces.common.vllm_inference import run_vllm_inference
from dagspaces.common.stage_utils import (
    maybe_silence_vllm_logs,
    to_json_str,
    serialize_arrow_unfriendly_in_row,
    extract_last_json,
)


def run_decomposition_stage(df: pd.DataFrame, cfg):
    """Decomposition stage.

    Produces CI tuple columns with None defaults; uses LLM extraction when
    cfg.runtime.use_llm_decompose is True.
    Columns: ci_subject, ci_sender, ci_receiver, ci_information, ci_transmission_principle, ci_missing_elements
    """
    base_cols = [
        "ci_subject",
        "ci_sender",
        "ci_receiver",
        "ci_information",
        "ci_transmission_principle",
        "ci_missing_elements",
    ]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["name", "rule_text"] + base_cols)
    out = df.copy()

    use_llm = bool(getattr(cfg.runtime, "use_llm_decompose", False))
    # Whether to JSON-serialize nested columns for Arrow/Parquet friendliness
    try:
        serialize_nested = bool(getattr(cfg.runtime, "serialize_nested_json", True))
    except Exception:
        serialize_nested = True

    if not use_llm:
        for c in base_cols:
            out[c] = None
        out["ci_missing_elements"] = out.apply(
            lambda r: ["subject", "sender", "receiver", "information", "transmission_principle"], axis=1
        )
        if serialize_nested:
            out["ci_missing_elements"] = out["ci_missing_elements"].map(to_json_str)
        return out

    system_prompt = str(getattr(cfg.prompt_decompose, "system_prompt", ""))
    prompt_template = str(getattr(cfg.prompt_decompose, "prompt_template", ""))

    def _format_prompt(rule_text: str) -> str:
        return prompt_template.replace("{{rule_text}}", str(rule_text or ""))

    # Prefer stage-specific sampling params when present
    try:
        sp_src = getattr(cfg, "sampling_params_decompose", getattr(cfg, "sampling_params", {}))
        sampling_params = OmegaConf.to_container(sp_src, resolve=True) if isinstance(sp_src, (dict,)) or hasattr(sp_src, "_get_node") else dict(sp_src)
    except Exception:
        sampling_params = dict(getattr(cfg, "sampling_params", {}))

    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        maybe_silence_vllm_logs()
        user = _format_prompt(row.get("rule_text"))
        sp = dict(sampling_params)
        sp.setdefault("max_tokens", 256)
        sp.setdefault("detokenize", False)
        # Optional guided decoding hook (future-proof; requires vLLM support)
        try:
            if bool(getattr(cfg.runtime, "guided_decoding_decompose", False)):
                schema = object_schema(
                    properties={
                        "subject": string_or_null(),
                        "sender": string_or_null(),
                        "receiver": string_or_null(),
                        "information": string_or_null(),
                        "transmission_principle": string_or_null(),
                        "missing": array_of_strings(),
                    },
                    required=["missing"],
                    additional_properties=False,
                )
                sp["guided_decoding"] = {"json": schema}
        except Exception:
            pass
        # Remove classification artifacts that might override our messages
        base = {k: v for k, v in row.items() if k not in {"messages", "sampling_params", "generated_text", "llm_output"}}
        base.update({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            "sampling_params": sp,
        })
        return base

    def _post(row: Dict[str, Any]) -> Dict[str, Any]:
        txt = row.get("generated_text")
        obj = extract_last_json(txt if isinstance(txt, str) else "") or {}

        def _norm_key(k: Any) -> str:
            s = str(k).strip().lower()
            return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in s)

        norm_obj: Dict[str, Any] = {}
        try:
            for k, v in (obj.items() if isinstance(obj, dict) else []):
                norm_obj[_norm_key(k)] = v
        except Exception:
            pass

        def _first_key(keys):
            for k in keys:
                if k in norm_obj:
                    return norm_obj.get(k)
            return None

        subject = _first_key(["subject", "data_subject", "person", "individual"])
        sender = _first_key(["sender", "source", "discloser", "actor_sender"])
        receiver = _first_key(["receiver", "recipient", "audience", "actor_receiver", "target"])
        information = _first_key(["information", "info", "data", "content", "pii"])
        tp = _first_key(["transmission_principle", "transmissionprinciple", "principle", "condition", "constraint", "rule"])
        missing = _first_key(["missing", "missing_elements", "missing_fields"]) or []
        if not isinstance(missing, list):
            try:
                missing = list(missing) if missing is not None else []
            except Exception:
                missing = []

        if serialize_nested:
            serialize_arrow_unfriendly_in_row(row, [
                "messages",
                "sampling_params",
                "usage",
                "token_counts",
            ])
            missing_out = to_json_str(missing)
        else:
            missing_out = missing

        def _norm_ci_value(v: Any) -> Any:
            if isinstance(v, (list, tuple)):
                return to_json_str(v) if serialize_nested else ", ".join([str(x) for x in v if x is not None])
            if isinstance(v, dict):
                return to_json_str(v) if serialize_nested else str(v)
            return v

        return {
            **row,
            "ci_subject": _norm_ci_value(subject),
            "ci_sender": _norm_ci_value(sender),
            "ci_receiver": _norm_ci_value(receiver),
            "ci_information": _norm_ci_value(information),
            "ci_transmission_principle": _norm_ci_value(tp),
            "ci_missing_elements": missing_out,
            "llm_output": row.get("generated_text"),
            "_progress_row": 1,
        }

    result_df = run_vllm_inference(out, cfg, _pre, _post, "decompose")
    for c in base_cols:
        if c not in result_df.columns:
            result_df[c] = None
    return result_df


