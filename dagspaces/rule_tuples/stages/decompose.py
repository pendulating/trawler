from typing import Any, Dict, Optional
import pandas as pd
import json
import os
import logging
from omegaconf import OmegaConf
from dagspaces.uair.schema_builders import (
    object_schema,
    string_or_null,
    array_of_strings,
)

from dagspaces.common.vllm_inference import run_vllm_inference

_VLLM_LOGS_SILENCED = False

def _maybe_silence_vllm_logs() -> None:
    global _VLLM_LOGS_SILENCED
    if _VLLM_LOGS_SILENCED:
        return
    try:
        if os.environ.get("RULE_TUPLES_SILENT"):
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
            for name in ("vllm", "vllm.logger", "vllm.engine", "vllm.core", "vllm.worker"):
                lg = logging.getLogger(name)
                lg.setLevel(logging.ERROR)
                lg.propagate = False
        _VLLM_LOGS_SILENCED = True
    except Exception:
        pass


def _to_json_str(value: Any):
    """Serialize Python objects to JSON string for Arrow/Parquet friendliness.

    Returns None for None input; falls back to str(value) on failure.
    """
    try:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        try:
            return str(value)
        except Exception:
            return None


def _serialize_arrow_unfriendly_in_row(row: Dict[str, Any], columns):
    """In-place convert nested/dict/list columns to JSON strings in a row dict."""
    for col in columns:
        if col in row:
            val = row.get(col)
            if isinstance(val, (dict, list, tuple)):
                row[col] = _to_json_str(val)


def _filter_vllm_engine_kwargs(ek: Dict[str, Any]) -> Dict[str, Any]:
    """Drop engine kwargs unsupported by the installed vLLM version.

    We try to introspect vllm.AsyncEngineArgs for accepted fields. If that
    fails, conservatively drop known newer flags.
    """
    try:
        import vllm as _v
        accepted = None
        try:
            fields = getattr(getattr(_v, "AsyncEngineArgs", None), "__dataclass_fields__", None)
            if isinstance(fields, dict) and fields:
                accepted = set(fields.keys())
        except Exception:
            accepted = None
        if accepted is None:
            try:
                import inspect as _inspect
                sig = _inspect.signature(_v.AsyncEngineArgs.__init__)
                accepted = set(k for k in sig.parameters.keys() if k != "self")
            except Exception:
                accepted = None
        if accepted:
            filtered = {k: v for k, v in ek.items() if k in accepted}
            if len(filtered) != len(ek):
                try:
                    if not os.environ.get("RULE_TUPLES_SILENT"):
                        dropped = [k for k in ek.keys() if k not in accepted]
                        print(f"Filtering unsupported vLLM engine kwargs: {dropped}")
                except Exception:
                    pass
            return filtered
    except Exception:
        pass
    ek = dict(ek)
    for k in ("use_v2_block_manager",):
        ek.pop(k, None)
    return ek

def _extract_last_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    s = text
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    import re
    snippets = re.findall(r"\{[\s\S]*\}", s)
    for snip in reversed(snippets or []):
        try:
            obj = json.loads(snip)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


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
            out["ci_missing_elements"] = out["ci_missing_elements"].map(_to_json_str)
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
        _maybe_silence_vllm_logs()
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
        obj = _extract_last_json(txt if isinstance(txt, str) else "") or {}

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
            _serialize_arrow_unfriendly_in_row(row, [
                "messages",
                "sampling_params",
                "usage",
                "token_counts",
            ])
            missing_out = _to_json_str(missing)
        else:
            missing_out = missing

        def _norm_ci_value(v: Any) -> Any:
            if isinstance(v, (list, tuple)):
                return _to_json_str(v) if serialize_nested else ", ".join([str(x) for x in v if x is not None])
            if isinstance(v, dict):
                return _to_json_str(v) if serialize_nested else str(v)
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


