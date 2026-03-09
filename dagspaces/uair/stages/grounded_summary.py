"""Grounded summary stage: connect tuple extraction, verification, and EU AI Act classification back to article text."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence
import copy
import hashlib
import json
import logging
import os

import pandas as pd
from omegaconf import OmegaConf


from dagspaces.common.vllm_inference import run_vllm_inference


_VLLM_LOGS_SILENCED = False

TUPLE_FIELDS = [
    "deployment_domain",
    "deployment_purpose",
    "deployment_capability",
    "identity_of_ai_deployer",
    "identity_of_ai_developer",
    "identity_of_ai_subject",
    "location_of_ai_deployer",
    "location_of_ai_subject",
    "date_and_time_of_event",
    "list_of_harms_that_occurred",
    "list_of_risks_that_occurred",
    "list_of_benefits_that_occurred",
    "missing",
]

VERIFICATION_SUFFIXES = ["_verified"]
DOC_LEVEL_VERIFICATION_FIELDS = ["core_tuple_verified"]
VERIFICATION_PROMPT_FIELDS = [
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

CLASSIFICATION_KEEP_COLUMNS = {
    "article_id",
    "article_text",
    "chunk_text",
    "article_path",
    "article_excerpt",
    "article_title",
    "eu_ai_label",
    "eu_ai_desc",
    "eu_ai_reason",
    "eu_ai_relevant_text",
    "eu_ai_raw_json",
}

DECOMPOSE_KEEP_COLUMNS = {"article_id", *TUPLE_FIELDS}

VERIFICATION_KEEP_COLUMNS = {"article_id", "core_tuple_verified"}
for _field in VERIFICATION_PROMPT_FIELDS:
    VERIFICATION_KEEP_COLUMNS.add(f"ver_tuple_{_field}_verified")

ADDITIONAL_STRING_COLUMNS = {
    "article_title",
    "article_path",
    "article_excerpt",
    "chunk_text",
    "company_or_startup",
    "data_used",
    "deployment_stage",
    "deployment_location",
    "mitigations_or_controls",
}

STRING_COLUMNS = (
    (CLASSIFICATION_KEEP_COLUMNS - {"article_id"})
    | set(TUPLE_FIELDS)
    | ADDITIONAL_STRING_COLUMNS
    | {"verification_snapshot"}
)

FLOAT_COLUMNS: set[str] = set()

BOOL_COLUMNS = {"core_tuple_verified"}

INPUT_COLUMNS = (
    {
        "article_id",
        "article_text",
        "chunk_text",
        "article_title",
        "article_path",
        "article_excerpt",
        "eu_ai_label",
        "eu_ai_desc",
        "eu_ai_reason",
        "eu_ai_relevant_text",
        "eu_ai_raw_json",
        "verification_snapshot",
        "core_tuple_verified",
        "company_or_startup",
        "data_used",
        "deployment_stage",
        "deployment_location",
        "mitigations_or_controls",
    }
    | set(TUPLE_FIELDS)
)


def _maybe_silence_vllm_logs() -> None:
    """Best-effort suppression of noisy vLLM logs."""
    global _VLLM_LOGS_SILENCED
    if _VLLM_LOGS_SILENCED:
        return
    try:
        if os.environ.get("RULE_TUPLES_SILENT"):
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
            for name in ("vllm", "vllm.logger", "vllm.engine", "vllm.worker"):
                lg = logging.getLogger(name)
                lg.setLevel(logging.ERROR)
                lg.propagate = False
        _VLLM_LOGS_SILENCED = True
    except Exception:
        pass




def _to_json_str(value: Any) -> Optional[str]:
    try:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        try:
            return str(value)
        except Exception:
            return None


def _serialize_arrow_unfriendly_in_row(row: Dict[str, Any], columns: Iterable[str]) -> None:
    for col in columns:
        if col in row and isinstance(row[col], (dict, list, tuple)):
            row[col] = _to_json_str(row[col])


def _extract_last_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        import re

        snippets = re.findall(r"\{[\s\S]*\}", text)
        for snippet in reversed(snippets or []):
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    except Exception:
        pass
    return None


def _sanitize_for_json(value: Any) -> Any:
    try:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): _sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_sanitize_for_json(v) for v in value]
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return _sanitize_for_json(tolist())
        return str(value)
    except Exception:
        return None


def _decode_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        lower = s.lower()
        if lower in {"none", "null", "nan"}:
            return None
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                pass
        return s
    return value


def _ensure_article_id(df: pd.DataFrame) -> pd.DataFrame:
    if "article_id" not in df.columns:
        df = df.copy()
        df["article_id"] = df.apply(_hash_article_id, axis=1)
        return df
    mask_missing = df["article_id"].isna()
    try:
        normalized = df["article_id"].astype(str).str.strip().str.lower()
        mask_missing = mask_missing | normalized.isin({"", "none", "null", "nan"})
    except Exception:
        pass
    if mask_missing.any():
        df.loc[mask_missing, "article_id"] = df.loc[mask_missing].apply(_hash_article_id, axis=1)
    return df


def _hash_article_id(row: pd.Series) -> str:
    src = row.get("article_id")
    if isinstance(src, str) and src.strip():
        return src.strip()
    text = row.get("article_text") or row.get("chunk_text") or row.get("article_path") or ""
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def _prepare_aux_dataframe(
    df_like: Any,
    drop_columns: Iterable[str],
    keep_columns: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
    if df_like is None:
        return None
    if hasattr(df_like, "to_pandas"):
        try:
            df_like = df_like.to_pandas()
        except Exception:
            return None
    if not isinstance(df_like, pd.DataFrame):
        return None
    if len(df_like) == 0:
        return None
    df_copy = df_like.copy()
    df_copy = _ensure_article_id(df_copy)
    for col in drop_columns:
        if col in df_copy.columns:
            df_copy = df_copy.drop(columns=[col])
    if keep_columns is not None:
        keep_set = {"article_id"} | {col for col in keep_columns if col in df_copy.columns}
        if keep_set:
            df_copy = df_copy[[col for col in df_copy.columns if col in keep_set]]
    _normalize_object_columns(df_copy)
    return df_copy


def _normalize_object_columns(df: pd.DataFrame) -> None:
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_object_dtype(series):
            continue
        if series.map(lambda v: isinstance(v, (dict, list, tuple, set))).any():
            df[col] = series.map(
                lambda v: _to_json_str(_sanitize_for_json(v)) if v is not None else None
            )
        try:
            df[col] = df[col].astype("string")
        except Exception:
            pass


def _ensure_column_types(df: pd.DataFrame) -> None:
    if df is None or len(df) == 0:
        return
    for col in df.columns:
        if col == "article_id":
            df[col] = df[col].astype("string[pyarrow]")
        elif col in STRING_COLUMNS:
            try:
                df[col] = df[col].astype("string[pyarrow]")
            except Exception:
                df[col] = df[col].astype("string")
        elif col in FLOAT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
        elif col in BOOL_COLUMNS:
            df[col] = df[col].astype("boolean")
        elif pd.api.types.is_object_dtype(df[col]):
            try:
                df[col] = df[col].astype("string[pyarrow]")
            except Exception:
                df[col] = df[col].astype("string")


def _debug_dataframe_schema(df: pd.DataFrame, label: str, max_samples: int = 3) -> None:
    try:
        print(
            f"[grounded_summary][DEBUG] {label} columns ({len(df.columns)}): {list(df.columns)}",
            flush=True,
        )
        dtype_map = {col: str(df[col].dtype) for col in df.columns}
        print(
            f"[grounded_summary][DEBUG] {label} dtypes: {dtype_map}",
            flush=True,
        )
        for col in df.columns:
            series = df[col]
            if series.dtype in ("string", "string[pyarrow]", "object") or col in (
                STRING_COLUMNS | BOOL_COLUMNS | FLOAT_COLUMNS
            ):
                sample = series.dropna().head(max_samples).tolist()
                sample_types = [type(x).__name__ for x in sample]
                print(
                    f"[grounded_summary][DEBUG] {label} sample {col}: types={sample_types} values={sample}",
                    flush=True,
                )
    except Exception as exc:
        print(f"[grounded_summary][DEBUG] Failed to log schema for {label}: {exc}", flush=True)


def _limit_text(text: str, max_chars: int = 1800) -> str:
    try:
        s = str(text or "").strip()
    except Exception:
        s = ""
    if len(s) <= max_chars:
        return s
    return f"{s[:max_chars].rstrip()}…"


def _list_from_row(row: Dict[str, Any], keys: Iterable[str]) -> Iterable[str]:
    values: list[str] = []
    for key in keys:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            values.append(val.strip())
        elif isinstance(val, (list, tuple, set)):
            for item in val:
                item_str = str(item).strip()
                if item_str:
                    values.append(item_str)
    return values


def _format_tuple_section(row: Dict[str, Any]) -> str:
    data: Dict[str, Any] = {}
    for field in TUPLE_FIELDS:
        data[field] = _decode_jsonish(row.get(field))
    sanitized = _sanitize_for_json(data) or {}
    return json.dumps(sanitized, ensure_ascii=False, indent=2)


def _format_verification_section(row: Dict[str, Any]) -> str:
    snapshot = row.get("verification_snapshot")
    info: Dict[str, Any] = {}
    if isinstance(snapshot, str) and snapshot.strip():
        try:
            info = json.loads(snapshot)
        except Exception:
            info = {}
    lines: list[str] = []
    core = info.get("core_tuple_verified")
    if core is not None:
        lines.append(f"- Core tuple verified: {bool(core)}")
    fields = info.get("fields") if isinstance(info.get("fields"), dict) else {}
    for field in VERIFICATION_PROMPT_FIELDS:
        if field not in fields:
            continue
        entry = fields.get(field) or {}
        value = entry.get("verified")
        if value is None:
            status = "unknown"
        else:
            status = "verified" if bool(value) else "unverified"
        lines.append(f"- {field.replace('_', ' ')}: {status}")
    if not lines:
        return "No verification signals were available."
    return "\n".join(lines)


def _default_guided_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "use_case_summary": {"type": "string", "minLength": 1},
            "risk_classification": {"type": "string"},
            "risk_rationale": {"type": "string"},
            "actors_involved": {"type": "array", "items": {"type": "string"}},
            "company_or_startup": {"type": "string"},
            "government_or_regulators": {"type": "array", "items": {"type": "string"}},
            "data_used": {"type": "string"},
            "deployment_stage": {"type": "string"},
            "deployment_location": {"type": "string"},
            "mitigations_or_controls": {"type": "string"},
            "open_questions": {"type": "string"},
            "supporting_quotes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "use_case_summary",
            "risk_classification",
            "risk_rationale",
            "actors_involved",
            "company_or_startup",
            "government_or_regulators",
            "data_used",
            "deployment_stage",
            "deployment_location",
            "mitigations_or_controls",
            "open_questions",
            "supporting_quotes",
        ],
        "additionalProperties": False,
    }


DEFAULT_SYSTEM_PROMPT = (
    "You are an AI policy analyst. Produce a concise, article-grounded summary that links the extracted use-case tuple, "
    "verification signals, and EU AI Act classification back to the original article text. Only respond with JSON that "
    "matches the provided schema."
)

DEFAULT_USER_TEMPLATE = (
    "Article ID: {article_id}\n"
    "Existing EU AI Act classification: {eu_ai_label}\n"
    "Classification description: {eu_ai_desc}\n"
    "Classification reasoning: {eu_ai_reason}\n"
    "Verification overview:\n{verification_summary}\n\n"
    "Tuple fields and actors:\n{tuple_summary}\n\n"
    "Verified evidence highlights:\n{verification_evidence}\n\n"
    "Article excerpt:\n\"\"\"\n{article_excerpt}\n\"\"\"\n\n"
    "Using this information, craft a two-to-three sentence use case summary that explicitly cites the entities and activities involved. "
    "Reaffirm or refine the risk classification using the provided label. Capture key actors, the company or startup named, any government or "
    "regulatory bodies, the data used, deployment stage and location, mitigations or controls mentioned, and any open questions or gaps. "
    "Return JSON that conforms to the schema."
)


def _resolve_prompt(cfg, key: str, fallback: str) -> str:
    try:
        value = OmegaConf.select(cfg, key)
        if value:
            return str(value)
    except Exception:
        pass
    return fallback


def _prepare_sampling_params(cfg, schema: Dict[str, Any]) -> Dict[str, Any]:
    try:
        raw = getattr(cfg, "sampling_params_grounded_summary", getattr(cfg, "sampling_params", {}))
        sampling_params = OmegaConf.to_container(raw, resolve=True) if hasattr(raw, "_get_node") or isinstance(raw, (dict,)) else dict(raw)
    except Exception:
        sampling_params = dict(getattr(cfg, "sampling_params", {}))
    sampling_params = _sanitize_for_json(sampling_params) or {}
    if not isinstance(sampling_params, dict):
        sampling_params = {}
    guided = {
        "json": schema,
        "disable_fallback": True,
        "disable_additional_properties": True,
    }
    sampling_params.pop("guided_decoding", None)
    sampling_params["guided_decoding"] = guided
    sampling_params.setdefault("max_tokens", 768)
    sampling_params.setdefault("temperature", 0.2)
    return sampling_params




def _build_preprocessor(
    system_prompt: str,
    user_template: str,
    schema: Dict[str, Any],
    sampling_params: Dict[str, Any],
) -> Any:
    sampling_params_template = copy.deepcopy(sampling_params)

    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        _maybe_silence_vllm_logs()
        article_id = row.get("article_id") or "unknown"
        eu_label = row.get("eu_ai_label") or row.get("risk_classification") or "Unknown"
        eu_desc = row.get("eu_ai_desc") or ""
        eu_reason = row.get("eu_ai_reason") or ""

        tuple_summary = _format_tuple_section(row)
        verification_summary = _format_verification_section(row)
        verification_block = verification_summary
        excerpt = _limit_text(row.get("article_text") or row.get("chunk_text") or "", 400)

        try:
            user_content = user_template.format(
                article_id=article_id,
                eu_ai_label=eu_label or "Unknown",
                eu_ai_desc=eu_desc or "Not provided",
                eu_ai_reason=eu_reason or "Not provided",
                verification_summary=verification_summary,
                tuple_summary=tuple_summary,
                verification_evidence=verification_block,
                article_excerpt=excerpt,
            )
        except Exception:
            user_content = (
                f"Article ID: {article_id}\n"
                f"Existing EU AI Act classification: {eu_label}\n"
                f"Article excerpt:\n{excerpt}\n\n"
                "Provide the structured summary."
            )

        base = {
            k: v
            for k, v in row.items()
            if k not in {"messages", "sampling_params", "generated_text", "guidance", "response_format"}
        }
        # Ensure string fields are never None to prevent PyArrow from inferring null type
        # when batches have all None values. Convert None to empty string for string fields.
        for key in base.keys():
            if key not in BOOL_COLUMNS and key not in FLOAT_COLUMNS:
                if base[key] is None:
                    base[key] = ""
        base["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        base["sampling_params"] = sampling_params_template
        base["article_excerpt"] = excerpt
        base["provided_risk_label"] = eu_label or ""
        return base

    return _pre


def _build_postprocessor(schema_fields: Iterable[str]) -> Any:
    fields = list(schema_fields)

    def _post(row: Dict[str, Any]) -> Dict[str, Any]:
        raw = row.get("generated_text") or ""
        parsed = _extract_last_json(raw)
        out: Dict[str, Any] = {
            k: v
            for k, v in row.items()
            if k not in {
                "messages",
                "sampling_params",
                "generated_text",
                "json",
                "guided_decoding",
                "response_format",
                "structured_output",
                "usage",
                "token_counts",
            }
        }
        for key in fields:
            value = parsed.get(key) if isinstance(parsed, dict) else None
            if isinstance(value, (dict, list, tuple)):
                out[key] = _to_json_str(value)
            else:
                out[key] = value
        # Ensure string fields are never None to prevent PyArrow from inferring null type
        # when batches have all None values. Convert None to empty string for string fields.
        for key in fields:
            if key not in BOOL_COLUMNS and key not in FLOAT_COLUMNS:
                if out.get(key) is None:
                    out[key] = ""
        if not out.get("risk_classification"):
            out["risk_classification"] = out.get("provided_risk_label") or ""
        out["llm_output"] = raw
        _serialize_arrow_unfriendly_in_row(out, ["actors_involved", "government_or_regulators", "supporting_quotes"])
        return out

    return _post


def _fallback_grounded_summary(row: pd.Series) -> Dict[str, Any]:
    summary_bits = []
    if isinstance(row.get("deployment_purpose"), str) and row.get("deployment_purpose").strip():
        summary_bits.append(row["deployment_purpose"].strip())
    if isinstance(row.get("deployment_capability"), str) and row.get("deployment_capability").strip():
        summary_bits.append(row["deployment_capability"].strip())
    if isinstance(row.get("identity_of_ai_deployer"), str) and row.get("identity_of_ai_deployer").strip():
        summary_bits.append(f"driven by {row['identity_of_ai_deployer'].strip()}")
    excerpt = _limit_text(row.get("article_text") or row.get("chunk_text") or "", 400)
    if not summary_bits:
        summary_text = f"No LLM summary; article excerpt: {excerpt}"
    else:
        summary_text = " / ".join(summary_bits)
    risk_label = row.get("eu_ai_label") or row.get("risk_classification") or "Unknown"
    actors_list = list(
        dict.fromkeys(
            _list_from_row(
                row,
                [
                    "identity_of_ai_deployer",
                    "identity_of_ai_developer",
                    "identity_of_ai_subject",
                    "company_or_startup",
                ],
            )
        )
    )
    supporting_quotes = [excerpt] if excerpt else []
    return {
        "article_id": row.get("article_id"),
        "use_case_summary": summary_text,
        "risk_classification": risk_label,
        "risk_rationale": row.get("eu_ai_reason"),
        "actors_involved": actors_list,
        "company_or_startup": row.get("company_or_startup") or row.get("identity_of_ai_deployer"),
        "government_or_regulators": [],
        "data_used": row.get("data_used"),
        "deployment_stage": row.get("deployment_stage"),
        "deployment_location": row.get("deployment_location") or row.get("location_of_ai_deployer"),
        "mitigations_or_controls": row.get("mitigations_or_controls"),
        "open_questions": None,
        "supporting_quotes": supporting_quotes,
        "article_excerpt": row.get("article_excerpt") or excerpt,
        "verification_snapshot": row.get("verification_snapshot"),
        "core_tuple_verified": row.get("core_tuple_verified"),
        "llm_output": None,
        "generation_mode": "fallback",
    }


def _extract_schema_fields(schema: Dict[str, Any]) -> Iterable[str]:
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    return list(props.keys())




def run_grounded_summary_stage(
    classification_df: Any,
    cfg,
    *,
    decompose_df: Any = None,
    verify_df: Any = None,
    logger: Any = None,
) -> pd.DataFrame:
    """
    Grounded summary stage.

    Args:
        classification_df: Output of classify_eu_act stage (Ray Dataset or pandas DataFrame).
            Expected columns include `article_id` (or fields to derive it), `article_text`,
            `eu_ai_label`, `eu_ai_desc`, and `eu_ai_reason`.
        cfg: Hydra/OmegaConf configuration used to resolve prompts, schema overrides, and sampling params.
        decompose_df: Optional output of decompose_nbl stage (merged by article_id). Tuple columns such as
            `deployment_domain`, `identity_of_ai_deployer`, etc. are incorporated into the prompt context.
        verify_df: Optional output of verify_nbl stage (merged by article_id). Verification signals prefixed
            with `ver_tuple_` are surfaced to the LLM and the fallback summarizer.
        logger: Optional wandb logger for table export.
    Returns:
        pandas.DataFrame with structured grounded summaries per article.
    """

    if classification_df is None:
        return pd.DataFrame([])

    if hasattr(classification_df, "to_pandas"):
        try:
            base_df = classification_df.to_pandas()
        except Exception:
            base_df = None
    elif isinstance(classification_df, pd.DataFrame):
        base_df = classification_df.copy()
    else:
        base_df = None

    if base_df is None or len(base_df) == 0:
        return pd.DataFrame([])

    base_df = _ensure_article_id(base_df)

    verify_prepared = _prepare_aux_dataframe(
        verify_df,
        drop_columns=[
            "article_text",
            "chunk_text",
            "messages",
            "sampling_params",
            "generated_text",
        ],
        keep_columns=VERIFICATION_KEEP_COLUMNS,
    )
    if verify_prepared is not None:
        verify_agg = verify_prepared.groupby("article_id", as_index=False).agg({"core_tuple_verified": "max"})
        base_df = base_df.merge(verify_agg, on="article_id", how="left", suffixes=("", "_from_verify"))
        if "core_tuple_verified_from_verify" in base_df.columns:
            if "core_tuple_verified" not in base_df.columns:
                base_df["core_tuple_verified"] = pd.NA
            base_df["core_tuple_verified"] = base_df["core_tuple_verified"].fillna(
                base_df["core_tuple_verified_from_verify"]
            )
            base_df = base_df.drop(columns=["core_tuple_verified_from_verify"], errors="ignore")

    def _build_verification_snapshot(row: pd.Series) -> Optional[str]:
        info: Dict[str, Any] = {
            "core_tuple_verified": bool(row.get("core_tuple_verified")),
            "fields": {},
        }
        for field in VERIFICATION_PROMPT_FIELDS:
            col = f"ver_tuple_{field}_verified"
            if col in row.index:
                val = row.get(col)
                if pd.isna(val):
                    info["fields"][field] = {"verified": None}
                else:
                    info["fields"][field] = {"verified": bool(val)}
        return _to_json_str(info)

    base_df["verification_snapshot"] = base_df.apply(_build_verification_snapshot, axis=1)
    mask_verified = base_df["core_tuple_verified"].fillna(False)
    base_df = base_df[mask_verified]
    if len(base_df) == 0:
        print("[grounded_summary] No rows with core_tuple_verified=True; exiting early.", flush=True)
        return pd.DataFrame([])
    if "article_id" in base_df.columns:
        base_df = base_df.sort_values("article_id").drop_duplicates(subset=["article_id"], keep="first")

    allowed_columns = [col for col in base_df.columns if col in INPUT_COLUMNS]
    base_df = base_df[allowed_columns].copy()

    base_df = base_df.dropna(axis=1, how="all")

    _normalize_object_columns(base_df)
    _ensure_column_types(base_df)
    base_df = base_df.dropna(axis=1, how="all")
    
    # Apply sample_n limit if specified (regardless of debug flag)
    try:
        sample_n = getattr(cfg.runtime, "sample_n", None)
        if isinstance(sample_n, int) and sample_n > 0:
            n = min(int(sample_n), int(len(base_df)))
            if n < len(base_df):
                try:
                    seed_env = os.environ.get("UAIR_SAMPLE_SEED", "777")
                    seed = int(seed_env) if seed_env is not None else 777
                except Exception:
                    seed = 777
                try:
                    base_df = base_df.sample(n=n, random_state=seed).reset_index(drop=True)
                    print(f"[grounded_summary] Applied sample_n={n} (seed={seed}), processing {len(base_df)} rows", flush=True)
                except Exception:
                    base_df = base_df.head(n)
                    print(f"[grounded_summary] Applied sample_n={n} (head), processing {len(base_df)} rows", flush=True)
    except Exception:
        pass
    
    try:
        if bool(getattr(cfg.runtime, "debug", False)):
            _debug_dataframe_schema(base_df, "base_dataframe")
    except Exception:
        pass

    system_prompt = _resolve_prompt(cfg, "prompt_grounded_summary.system_prompt", DEFAULT_SYSTEM_PROMPT)
    user_template = _resolve_prompt(cfg, "prompt_grounded_summary.user_template", DEFAULT_USER_TEMPLATE)
    schema = _default_guided_schema()
    schema_override = _resolve_prompt(cfg, "prompt_grounded_summary.schema_path", "")
    if schema_override and os.path.exists(schema_override):
        try:
            with open(schema_override, "r", encoding="utf-8") as f:
                schema = json.load(f)
        except Exception as exc:
            try:
                print(f"[grounded_summary] Failed to load schema override: {exc}", flush=True)
            except Exception:
                pass

    sampling_params = _prepare_sampling_params(cfg, schema)
    schema_fields = _extract_schema_fields(schema)

    use_llm = bool(getattr(cfg.runtime, "use_llm_grounded_summary", True))

    if use_llm:
        _pre = _build_preprocessor(system_prompt, user_template, schema, sampling_params)
        _post = _build_postprocessor(schema_fields)
        result_df = run_vllm_inference(base_df, cfg, _pre, _post, "grounded_summary")
        result_df = result_df.drop(columns=["messages", "sampling_params"], errors="ignore")
        result_df["generation_mode"] = "llm"
    else:
        rows = []
        for _, row in base_df.iterrows():
            rows.append(_fallback_grounded_summary(row))
        result_df = pd.DataFrame(rows)

    result_df["article_id"] = result_df["article_id"].fillna(base_df["article_id"])
    result_df["risk_classification"] = result_df["risk_classification"].fillna(base_df.get("eu_ai_label"))
    result_df["article_excerpt"] = result_df.get("article_excerpt", base_df.get("article_excerpt"))

    if logger is not None and len(result_df):
        try:
            logger.log_table(
                result_df,
                "grounded_summary/articles",
                prefer_cols=[
                    "article_id",
                    "use_case_summary",
                    "risk_classification",
                    "company_or_startup",
                    "actors_involved",
                ],
                panel_group="inspect_results",
            )
        except Exception as exc:
            try:
                print(f"[grounded_summary] Warning: failed to log to wandb: {exc}", flush=True)
            except Exception:
                pass

    print(f"[grounded_summary] Completed summaries for {len(result_df)} articles", flush=True)
    return result_df



