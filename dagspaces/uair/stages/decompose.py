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
    """Urban AI risk decomposition stage.

    Produces structured fields as defined by prompt_decompose:
    - deployment_domain
    - deployment_purpose
    - deployment_capability
    - identity_of_ai_deployer
    - identity_of_ai_subject
    - identity_of_ai_developer
    - location_of_ai_deployer
    - location_of_ai_subject
    - date_and_time_of_event
    - missing (list of missing/uncertain fields)
    """
    base_cols = [
        "deployment_domain",
        "deployment_purpose",
        "deployment_capability",
        "identity_of_ai_deployer",
        "identity_of_ai_subject",
        "identity_of_ai_developer",
        "location_of_ai_deployer",
        "location_of_ai_subject",
        "date_and_time_of_event",
        "missing",
    ]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["article_text"] + base_cols)
    out = df.copy()

    use_llm = bool(getattr(cfg.runtime, "use_llm_decompose", False))
    # Whether to JSON-serialize nested columns for Arrow/Parquet friendliness
    try:
        serialize_nested = bool(getattr(cfg.runtime, "serialize_nested_json", True))
    except Exception:
        serialize_nested = True

    if not use_llm:
        missing_keys = [
            "deployment_domain",
            "deployment_purpose",
            "deployment_capability",
            "identity_of_ai_deployer",
            "identity_of_ai_subject",
            "identity_of_ai_developer",
            "location_of_ai_deployer",
            "location_of_ai_subject",
            "date_and_time_of_event",
        ]
        for c in base_cols:
            out[c] = None
        out["missing"] = out.apply(lambda r: list(missing_keys), axis=1)
        if serialize_nested:
            out["missing"] = out["missing"].map(_to_json_str)
        return out

    try:
        system_prompt = str(
            OmegaConf.select(cfg, "prompt_decompose.system_prompt")
            or OmegaConf.select(cfg, "prompt.system_prompt")
            or ""
        )
    except Exception:
        system_prompt = ""
    try:
        prompt_template = str(
            OmegaConf.select(cfg, "prompt_decompose.prompt_template")
            or OmegaConf.select(cfg, "prompt.prompt_template")
            or ""
        )
    except Exception:
        prompt_template = ""

    def _format_prompt(article_text: str) -> str:
        return (
            prompt_template
            .replace("{{rule_text}}", str(article_text or ""))
            .replace("{{article_text}}", str(article_text or ""))
        )

    # Prefer stage-specific sampling params when present; convert nested DictConfig -> dict
    try:
        sp_src = getattr(cfg, "sampling_params_decompose", getattr(cfg, "sampling_params", {}))
        sampling_params = OmegaConf.to_container(sp_src, resolve=True) if isinstance(sp_src, (dict,)) or hasattr(sp_src, "_get_node") else dict(sp_src)
    except Exception:
        sampling_params = dict(getattr(cfg, "sampling_params", {}))

    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        _maybe_silence_vllm_logs()
        user = _format_prompt(row.get("article_text"))
        sp = dict(sampling_params)
        # Encourage short, structured JSON-only outputs
        sp.setdefault("max_tokens", 256)
        sp.setdefault("detokenize", False)
        # Optional guided decoding hook (future-proof; requires vLLM support)
        try:
            if bool(getattr(cfg.runtime, "guided_decoding_decompose", False)):
                schema = object_schema(
                    properties={
                        # Canonical string-or-null fields
                        "deployment_domain": string_or_null(),
                        "deployment_purpose": string_or_null(),
                        "deployment_capability": string_or_null(),
                        "identity_of_ai_deployer": string_or_null(),
                        "identity_of_ai_subject": string_or_null(),
                        "identity_of_ai_developer": string_or_null(),
                        "location_of_ai_deployer": string_or_null(),
                        "location_of_ai_subject": string_or_null(),
                        "date_and_time_of_event": string_or_null(),
                        # Lists
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

        # Robust key normalization and synonyms mapping
        def _norm_key(k: Any) -> str:
            s = str(k).strip().lower()
            out = []
            for ch in s:
                out.append(ch if (ch.isalnum() or ch == "_") else "_")
            return "".join(out)

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

        deployment_domain = _first_key(["deployment_domain", "domain", "use_domain"])
        deployment_purpose = _first_key(["deployment_purpose", "purpose", "goal", "objective"])
        deployment_capability = _first_key(["deployment_capability", "capability", "capabilities", "function", "ability"])
        identity_of_ai_deployer = _first_key(["identity_of_ai_deployer", "deployer", "operator", "implementer", "user", "agency", "organization_deployer"])
        identity_of_ai_subject = _first_key(["identity_of_ai_subject", "subject", "data_subject", "affected_party", "individual", "group"])
        identity_of_ai_developer = _first_key(["identity_of_ai_developer", "developer", "vendor", "builder", "provider", "manufacturer"])
        location_of_ai_deployer = _first_key(["location_of_ai_deployer", "deployer_location", "operator_location", "location_deployer"])
        location_of_ai_subject = _first_key(["location_of_ai_subject", "subject_location", "location_subject", "where"])
        date_and_time_of_event = _first_key(["date_and_time_of_event", "datetime", "date_time", "date", "time", "event_time", "when"])
        missing = _first_key(["missing", "missing_elements", "missing_fields"]) or []
        if not isinstance(missing, list):
            try:
                missing = list(missing) if missing is not None else []
            except Exception:
                missing = []

        # Optionally serialize nested columns for Arrow/Parquet compatibility
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

        # Ensure scalar columns are Arrow-friendly (no lists/dicts in columns)
        def _norm_ci_value(v: Any) -> Any:
            if isinstance(v, (list, tuple)):
                return _to_json_str(v) if serialize_nested else ", ".join([str(x) for x in v if x is not None])
            if isinstance(v, dict):
                return _to_json_str(v) if serialize_nested else str(v)
            return v

        deployment_domain_out = _norm_ci_value(deployment_domain)
        deployment_purpose_out = _norm_ci_value(deployment_purpose)
        deployment_capability_out = _norm_ci_value(deployment_capability)
        identity_of_ai_deployer_out = _norm_ci_value(identity_of_ai_deployer)
        identity_of_ai_subject_out = _norm_ci_value(identity_of_ai_subject)
        identity_of_ai_developer_out = _norm_ci_value(identity_of_ai_developer)
        location_of_ai_deployer_out = _norm_ci_value(location_of_ai_deployer)
        location_of_ai_subject_out = _norm_ci_value(location_of_ai_subject)
        date_and_time_of_event_out = _norm_ci_value(date_and_time_of_event)

        return {
            **row,
            "deployment_domain": deployment_domain_out,
            "deployment_purpose": deployment_purpose_out,
            "deployment_capability": deployment_capability_out,
            "identity_of_ai_deployer": identity_of_ai_deployer_out,
            "identity_of_ai_subject": identity_of_ai_subject_out,
            "identity_of_ai_developer": identity_of_ai_developer_out,
            "location_of_ai_deployer": location_of_ai_deployer_out,
            "location_of_ai_subject": location_of_ai_subject_out,
            "date_and_time_of_event": date_and_time_of_event_out,
            "missing": missing_out,
            "llm_output": row.get("generated_text"),
            "_progress_row": 1,
        }

    from dagspaces.common.vllm_inference import run_vllm_inference
    result_df = run_vllm_inference(out, cfg, _pre, _post, "decompose")
    for c in base_cols:
        if c not in result_df.columns:
            result_df[c] = None
    # Stage-scoped logging is handled by the orchestrator
    return result_df
