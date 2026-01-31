"""EU AI Act classification profile implementation - standalone module following decompose_nbl.py pattern."""

from typing import Any, Dict, List
import pandas as pd
import json
import os
import logging
from omegaconf import OmegaConf

try:
    import ray  # noqa: F401
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig  # type: ignore
    _RAY_OK = True
except Exception:
    _RAY_OK = False

from vllm.sampling_params import GuidedDecodingParams  # type: ignore

from .classify_shared import (
    ensure_ray_init,
    coerce_boolish_row,
    coerce_boolish_df,
    detect_num_gpus,
    detect_gpu_type,
    apply_gpu_aware_batch_settings,
    filter_vllm_engine_kwargs,
    to_json_str,
    serialize_arrow_unfriendly_in_row,
    sanitize_for_json,
    extract_last_json,
    get_required_input_columns,
    normalize_profile_columns,
    merge_result_parts,
    prune_result_columns,
    dedupe_by_article_id,
    inject_prompt_from_file,
    EU_INPUT_KEYS,
)

_VLLM_LOGS_SILENCED = False
_EU_TOTAL_INPUTS = 9


def _maybe_silence_vllm_logs() -> None:
    global _VLLM_LOGS_SILENCED
    if _VLLM_LOGS_SILENCED:
        return
    try:
        # Configure vLLM logging level from environment or default
        vllm_log_level = os.environ.get("VLLM_LOGGING_LEVEL", "WARNING")
        if os.environ.get("RULE_TUPLES_SILENT"):
            vllm_log_level = "ERROR"
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        
        # Configure vLLM loggers to ensure they work properly
        for name in ("vllm", "vllm.logger", "vllm.engine", "vllm.core", "vllm.worker"):
            lg = logging.getLogger(name)
            # Set level based on environment
            if vllm_log_level == "ERROR":
                lg.setLevel(logging.ERROR)
            elif vllm_log_level == "WARNING":
                lg.setLevel(logging.WARNING)
            elif vllm_log_level == "INFO":
                lg.setLevel(logging.INFO)
            elif vllm_log_level == "DEBUG":
                lg.setLevel(logging.DEBUG)
            else:
                lg.setLevel(logging.WARNING)
            # Ensure logs propagate unless we're silencing
            if not os.environ.get("RULE_TUPLES_SILENT"):
                lg.propagate = True
            else:
                lg.propagate = False
        _VLLM_LOGS_SILENCED = True
    except Exception:
        pass


def get_eu_guided_json_schema() -> Dict[str, Any] | None:
    """Get JSON schema for EU AI Act guided decoding."""
    try:
        return {
            "type": "object",
            "properties": {
                "Description": {"type": "string"},
                "Classification": {"enum": ["Prohibited", "High Risk", "Limited or Low Risk"]},
                "Relevant Text from the EU AI Act": {"type": "string"},
                "Reasoning": {"type": "string"},
            },
            "required": ["Description", "Classification", "Relevant Text from the EU AI Act", "Reasoning"],
            "additional_properties": False,
        }
    except Exception:
        return None


def is_nan_like(v: Any, missing_placeholder: str) -> bool:
    """Check if value is NaN-like."""
    try:
        if v is None:
            return True
        if isinstance(v, float):
            return v != v
        return str(v).strip().lower() in {"nan", "na", "none"}
    except Exception:
        return False


def norm_str(val: Any, missing_placeholder: str) -> str:
    """Normalize string value, replacing NaN-like values with placeholder."""
    try:
        if is_nan_like(val, missing_placeholder):
            return missing_placeholder
        if isinstance(val, (list, tuple, set)):
            vals = [str(x).strip() for x in val if x is not None and str(x).strip() != ""]
            return ", ".join(vals) if vals else missing_placeholder
        s = str(val).strip()
        return s if s else missing_placeholder
    except Exception:
        return missing_placeholder


def first_present(row: Dict[str, Any], keys: List[str]) -> Any:
    """Get first present value from row for given keys."""
    for k in keys:
        try:
            v = row.get(k)
            if isinstance(v, str) and v.strip() != "":
                return v
            if v not in (None, "") and not isinstance(v, str):
                return v
        except Exception:
            pass
    return None


def format_user_prompt(row: Dict[str, Any], user_template: str, missing_placeholder: str) -> str:
    """Format user prompt for EU AI Act classification."""
    domain = norm_str(first_present(row, [
        "deployment_domain", "domain", "use_domain"
    ]), missing_placeholder)
    purpose = norm_str(first_present(row, [
        "deployment_purpose", "purpose", "goal", "objective"
    ]), missing_placeholder)
    capability = norm_str(first_present(row, [
        "deployment_capability", "capability", "capabilities", "function", "ability"
    ]), missing_placeholder)
    ai_developer = norm_str(first_present(row, [
        "identity_of_ai_developer", "ai_developer", "developer", "vendor", "builder", "provider", "manufacturer"
    ]), missing_placeholder)
    ai_deployer = norm_str(first_present(row, [
        "identity_of_ai_deployer", "deployer", "operator", "implementer", "user", "agency", "organization_deployer"
    ]), missing_placeholder)
    ai_deployer_location = norm_str(first_present(row, [
        "location_of_ai_deployer", "deployer_location", "operator_location", "location_deployer"
    ]), missing_placeholder)
    ai_subject = norm_str(first_present(row, [
        "identity_of_ai_subject", "ai_subject", "subject", "data_subject", "affected_party", "individual", "group"
    ]), missing_placeholder)
    ai_subject_location = norm_str(first_present(row, [
        "location_of_ai_subject", "subject_location", "location_subject", "where"
    ]), missing_placeholder)
    date_time = norm_str(first_present(row, [
        "date_and_time_of_event", "date___time_of_event", "datetime", "date_time", "date", "time", "event_time", "when"
    ]), missing_placeholder)
    
    return user_template.format(
        domain,
        purpose,
        capability,
        ai_developer,
        ai_deployer,
        ai_deployer_location,
        ai_subject,
        ai_subject_location,
        date_time,
    )


def is_valid_eu_value(v: Any, missing_placeholder: str) -> bool:
    """Check if value is valid (not missing placeholder)."""
    try:
        s = norm_str(v, missing_placeholder)
        return bool(s) and s != missing_placeholder
    except Exception:
        return False


def add_eu_vague_flags(r: Dict[str, Any], missing_placeholder: str) -> Dict[str, Any]:
    """Add EU input completeness flags to row."""
    cnt = 0
    for k in EU_INPUT_KEYS:
        try:
            if is_valid_eu_value(r.get(k), missing_placeholder):
                cnt += 1
        except Exception:
            continue
    
    # If decompose provided a 'missing' column, use it
    try:
        miss_candidates = ("missing", "missing_fields", "eu_missing_fields")
        missing_list = None
        for mk in miss_candidates:
            if mk in r and r.get(mk) is not None:
                mv = r.get(mk)
                if isinstance(mv, (list, tuple, set)):
                    missing_list = list(mv)
                    break
                if isinstance(mv, str):
                    sv = mv.strip()
                    if sv:
                        try:
                            import json as _json
                            if sv.startswith("[") or sv.startswith("{"):
                                parsed = _json.loads(sv)
                            else:
                                import ast as _ast
                                parsed = _ast.literal_eval(sv)
                        except Exception:
                            body = sv.strip().strip("[](){}")
                            parsed = [x.strip() for x in body.split(",") if x.strip() != ""]
                        if isinstance(parsed, (list, tuple, set)):
                            missing_list = list(parsed)
                            break
        if missing_list is not None:
            try:
                miss_cnt = int(len(missing_list))
            except Exception:
                miss_cnt = 0
            if miss_cnt < 0:
                miss_cnt = 0
            if miss_cnt > _EU_TOTAL_INPUTS:
                miss_cnt = _EU_TOTAL_INPUTS
            cnt_from_missing = max(0, _EU_TOTAL_INPUTS - miss_cnt)
            cnt = min(int(cnt), int(cnt_from_missing))
    except Exception:
        pass
    
    r["eu_valid_input_count"] = int(cnt)
    r["too_vague_to_process"] = bool(int(cnt) < 5)
    return r


def _get_tokenizer(model_source: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
        return AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)
    except Exception:
        return None


_TOK_CACHED = None


def _get_tokenizer_cached(model_source: str):
    global _TOK_CACHED
    try:
        if _TOK_CACHED is None:
            _TOK_CACHED = _get_tokenizer(model_source)
        return _TOK_CACHED
    except Exception:
        return None


def _get_max_user_input_tokens(tokenizer, system_text: str, sampling_params: Dict[str, Any], cfg) -> int:
    try:
        system_tokens = len(tokenizer.encode(system_text, add_special_tokens=False)) if tokenizer else 0
        max_model_len = int(getattr(cfg.model, "engine_kwargs", {}).get("max_model_len", 4096))
        try:
            if hasattr(sampling_params, "get"):
                max_output = int(sampling_params.get("max_tokens", 512) or 512)
            else:
                max_output = int((sampling_params or {}).get("max_tokens", 512))
        except Exception:
            max_output = 512
        safety = 512
        return max(512, max_model_len - max_output - system_tokens - safety)
    except Exception:
        return 2048


def _trim_text_for_prompt(text: str, tokenizer, system_text: str, sampling_params: Dict[str, Any], cfg) -> str:
    """Token-aware trimming for prompts, matching original classify.py behavior."""
    # Tokenizer-aware trimming when available; otherwise conservative char-based fallback
    if tokenizer:
        try:
            ids = tokenizer.encode(text or "", add_special_tokens=False)
            max_user = _get_max_user_input_tokens(tokenizer, system_text, sampling_params, cfg)
            if len(ids) <= max_user:
                return text
            ids = ids[:max_user]
            return tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            pass
    # Fallback: approximate 4 chars per token budget
    try:
        max_user = _get_max_user_input_tokens(tokenizer, system_text, sampling_params, cfg)
    except Exception:
        max_user = 2048
    approx_chars_per_token = 4
    max_chars = int(max_user) * approx_chars_per_token
    try:
        return text if len(text or "") <= max_chars else str(text or "")[:max_chars]
    except Exception:
        return text


def _filter_vllm_sampling_params(sp: Dict[str, Any]) -> Dict[str, Any]:
    """Remove known problematic params that cause issues with vLLM/Ray.
    
    Note: We don't use introspection-based filtering because:
    1. It's unreliable (Ray's vLLM integration may accept params introspection doesn't show)
    2. It was incorrectly filtering out valid params like max_tokens
    3. vLLM/Ray will handle unsupported params gracefully (ignore or raise clear error)
    
    We only remove params that are known to cause issues.
    """
    sp2 = dict(sp)
    # Remove params that are known to cause issues with vLLM/Ray
    # These are OpenAI API params that vLLM doesn't support
    for k in ("early_stopping", "length_penalty", "response_format", "structured_output"):
        sp2.pop(k, None)
    # Always preserve guided_decoding (Ray's vLLM integration handles this)
    if "guided_decoding" in sp:
        sp2["guided_decoding"] = sp["guided_decoding"]
    return sp2


def _stabilize_sampling_params(sp: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure sampling_params have a stable schema across rows.
    
    This function ensures schema stability for Arrow serialization by:
    1. Normalizing types (e.g., making sure 'stop' is always a list)
    2. Removing known problematic params
    
    It does NOT add any default values - all values must come from config.
    """
    try:
        # Start with the actual config values, not defaults
        out = dict(sp or {})
        # Only normalize 'stop' to ensure it's a list (schema stability for Arrow)
        stop_val = out.get("stop")
        if stop_val is None:
            out["stop"] = []
        elif not isinstance(stop_val, list):
            out["stop"] = [str(stop_val)]
        # Remove known problematic params (but preserve all valid ones)
        filtered = _filter_vllm_sampling_params(out)
        return filtered
    except Exception as e:
        print(f"[classify_eu_act] DEBUG: _stabilize_sampling_params exception: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return dict(sp or {})


def run_classification_eu_act(df: Any, cfg) -> Any:
    """
    EU AI Act classification implementation - standalone module.
    
    Follows the decompose_nbl.py pattern with complete implementation including:
    - Ray initialization
    - Engine config setup
    - EU vague flags filtering
    - Guided decoding with EU JSON schema
    - Deduplication before processing
    - LLM processing with _pre/_post functions
    - Merging vague rows back
    """
    # Ensure profile is set and inject prompt
    from .classify_shared import ensure_profile
    ensure_profile(cfg, "eu_ai_act")
    # Support prompt subdirectory override (e.g., 'general_ai' for general_ai/eu_ai_act_classification.yaml)
    prompt_subdir = getattr(cfg.runtime, "prompt_subdirectory", None)
    prompt_filename = "eu_ai_act_classification.yaml"
    if prompt_subdir:
        prompt_filename = f"{prompt_subdir}/{prompt_filename}"
    inject_prompt_from_file(cfg, prompt_filename)
    
    # Deduplicate before processing
    in_df = df
    if hasattr(df, "to_pandas"):
        in_df = df.to_pandas()
    if isinstance(in_df, pd.DataFrame):
        in_df = dedupe_by_article_id(in_df)
    
    # Ensure Ray is initialized
    ensure_ray_init(cfg)
    
    # Check if Ray Dataset or pandas DataFrame
    is_ray_ds = hasattr(in_df, "map_batches") and hasattr(in_df, "count") and _RAY_OK
    if not is_ray_ds:
        if in_df is None or len(in_df) == 0:
            return pd.DataFrame(columns=["article_id", "article_text"])
        out = in_df.copy()
        # Ensure article_id exists
        try:
            import hashlib as _hash
            if "article_id" not in out.columns:
                out["article_id"] = None
            def _gen_id_row(r: pd.Series) -> str:
                try:
                    src = r.get("article_path")
                    if not isinstance(src, str) or src.strip() == "":
                        src = r.get("article_text") or r.get("chunk_text") or ""
                    return _hash.sha1(str(src).encode("utf-8")).hexdigest()
                except Exception:
                    return _hash.sha1(str(r.get("article_text") or r.get("chunk_text") or "").encode("utf-8")).hexdigest()
            def _need_id(val: Any) -> bool:
                return not (isinstance(val, str) and val.strip() != "")
            try:
                mask_missing = out["article_id"].apply(_need_id)
            except Exception:
                mask_missing = True
            if isinstance(mask_missing, bool):
                out["article_id"] = out.apply(_gen_id_row, axis=1)
            else:
                out.loc[mask_missing, "article_id"] = out.loc[mask_missing].apply(_gen_id_row, axis=1)
        except Exception:
            pass
        out = coerce_boolish_df(out)
    
    use_llm = bool(getattr(cfg.runtime, "use_llm_classify", True))  # EU profile defaults to LLM
    
    if not use_llm or not _RAY_OK:
        # Fallback: return empty results with proper schema
        if is_ray_ds:
            def _empty_batch(pdf: pd.DataFrame) -> pd.DataFrame:
                pdf = pdf.copy()
                pdf["eu_ai_desc"] = None
                pdf["eu_ai_label"] = None
                pdf["eu_ai_relevant_text"] = None
                pdf["eu_ai_reason"] = None
                pdf["eu_ai_raw_json"] = None
                pdf["classification_mode"] = "heuristic_fallback"
                pdf["too_vague_to_process"] = False
                return pdf
            return in_df.map_batches(_empty_batch, batch_format="pandas")
        out["eu_ai_desc"] = None
        out["eu_ai_label"] = None
        out["eu_ai_relevant_text"] = None
        out["eu_ai_reason"] = None
        out["eu_ai_raw_json"] = None
        out["classification_mode"] = "heuristic_fallback"
        out["too_vague_to_process"] = False
        return out
    
    # Get prompts
    try:
        system_prompt = (
            getattr(getattr(cfg, "prompts", {}), "eu_ai_act", {}).get("system")  # type: ignore
            if hasattr(cfg, "prompts") else None
        )
    except Exception:
        system_prompt = None
    if not system_prompt:
        system_prompt = str(getattr(cfg.prompt, "system_prompt", ""))
    
    try:
        user_template = (
            getattr(getattr(cfg, "prompts", {}), "eu_ai_act", {}).get("user_template")  # type: ignore
            if hasattr(cfg, "prompts") else None
        )
    except Exception:
        user_template = None
    if not user_template:
        user_template = str(getattr(cfg.prompt, "prompt_template", ""))
    
    # Get missing placeholder
    try:
        missing_placeholder = str(getattr(cfg.runtime, "eu_missing_placeholder", "Not known/specified"))
    except Exception:
        missing_placeholder = "Not known/specified"
    
    # Build engine config - use ONLY values from config, no defaults
    ek = dict(getattr(cfg.model, "engine_kwargs", {}))
    # Only set tensor_parallel_size from environment if not in config (this is a runtime override, not a default)
    tp_env = os.environ.get("UAIR_TENSOR_PARALLEL_SIZE")
    if "tensor_parallel_size" not in ek and tp_env:
        try:
            tp_val = max(1, int(tp_env))
            ek["tensor_parallel_size"] = tp_val
            if not os.environ.get("RULE_TUPLES_SILENT"):
                print(f"Using tensor_parallel_size={tp_val} from UAIR_TENSOR_PARALLEL_SIZE", flush=True)
        except Exception:
            pass
    # Only auto-detect tensor_parallel_size if not in config and not in env
    if "tensor_parallel_size" not in ek:
        try:
            num_gpus = detect_num_gpus()
            ek["tensor_parallel_size"] = num_gpus
            if not os.environ.get("RULE_TUPLES_SILENT"):
                print(f"Auto-detected {num_gpus} GPU(s) for tensor parallelism", flush=True)
        except Exception:
            pass  # Let vLLM use its own defaults if not specified
    gpu_settings = apply_gpu_aware_batch_settings(ek, cfg)
    # Set guided decoding backend at engine initialization (not request level)
    ek.setdefault("guided_decoding_backend", "xgrammar")
    # Filter to only vLLM-supported kwargs (this doesn't add defaults, just filters)
    ek = filter_vllm_engine_kwargs(ek)
    
    # Get batch_size from config only - no defaults
    batch_size = None
    try:
        batch_size_cfg = getattr(cfg.model, "batch_size", None)
        if batch_size_cfg is not None:
            batch_size = int(batch_size_cfg)
        elif gpu_settings and "batch_size" in gpu_settings:
            batch_size = gpu_settings["batch_size"]
            if not os.environ.get("RULE_TUPLES_SILENT"):
                print(f"Auto-set batch_size={batch_size} for {detect_gpu_type()}", flush=True)
    except Exception:
        batch_size = None
    # Use 1 as fallback only if batch_size is required and not set (vLLM will use its own default)
    if batch_size is None:
        batch_size = 1  # Minimal fallback for vLLMEngineProcessorConfig
    
    engine_config = vLLMEngineProcessorConfig(
        model_source=str(getattr(cfg.model, "model_source")),
        runtime_env={
            "env_vars": {
                "VLLM_LOGGING_LEVEL": str(os.environ.get("VLLM_LOGGING_LEVEL", "WARNING")),
                "WANDB_DISABLE_SERVICE": str(os.environ.get("WANDB_DISABLE_SERVICE", "true")),
                "WANDB_SILENT": str(os.environ.get("WANDB_SILENT", "true")),
            }
        },
        engine_kwargs=ek,
        concurrency=int(getattr(cfg.model, "concurrency", 1) or 1),
        batch_size=int(batch_size),
    )
    
    # Sampling params - use ONLY values from config, no defaults
    try:
        sp_src = getattr(cfg, "sampling_params_eu_ai_act", getattr(cfg, "sampling_params_classify", getattr(cfg, "sampling_params", {})))
        print(f"[classify_eu_act] DEBUG: sp_src type={type(sp_src)}, value={sp_src}", flush=True)
        sampling_params = OmegaConf.to_container(sp_src, resolve=True) if isinstance(sp_src, (dict,)) or hasattr(sp_src, "_get_node") else dict(sp_src)
        print(f"[classify_eu_act] DEBUG: After to_container, sampling_params={sampling_params}, max_tokens={sampling_params.get('max_tokens', 'NOT SET')}", flush=True)
    except Exception as e:
        print(f"[classify_eu_act] DEBUG: Exception getting sampling_params: {e}", flush=True)
        sampling_params = dict(getattr(cfg, "sampling_params", {}))
    
    # Log actual max_tokens being used for debugging
    try:
        actual_max_tokens = sampling_params.get("max_tokens", "NOT SET")
        print(f"[classify_eu_act] Using max_tokens={actual_max_tokens} for LLM generation (from config)", flush=True)
    except Exception:
        pass
    
    # EU guided JSON schema
    EU_GUIDED_JSON_SCHEMA = get_eu_guided_json_schema()
    
    # Internal columns to drop
    _INTERNAL_DROP_COLS = [
        "messages",
        "sampling_params",
        "usage",
        "token_counts",
        "generated_text",
        "json",
        "guided_decoding",
        "response_format",
        "structured_output",
        "prompt",
        "prompt_token_ids",
        "request_id",
        "params",
        "llm_json",
        "generated_tokens",
        "num_generated_tokens",
        "num_input_tokens",
        "time_taken_llm",
        "embeddings",
    ]
    
    def _normalize_na_blanks(r: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize EU input fields
        eu_keys = [
            "deployment_domain",
            "deployment_purpose",
            "deployment_capability",
            "identity_of_ai_developer",
            "identity_of_ai_deployer",
            "location_of_ai_deployer",
            "identity_of_ai_subject",
            "location_of_ai_subject",
            "date_and_time_of_event",
            "date___time_of_event",
        ]
        for k in eu_keys:
            try:
                v = r.get(k)
                s = norm_str(v, missing_placeholder)
                r[k] = s
            except Exception:
                try:
                    r[k] = missing_placeholder
                except Exception:
                    pass
        # Ensure article_text/chunk_text are strings
        try:
            r["article_text"] = str(r.get("article_text") or "")
        except Exception:
            pass
        try:
            if r.get("chunk_text") is not None:
                r["chunk_text"] = str(r.get("chunk_text") or "")
        except Exception:
            pass
        return coerce_boolish_row(r)
    
    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        _maybe_silence_vllm_logs()
        # Ensure article_id is present
        try:
            import hashlib as _hash
            r = dict(row)
            aid = r.get("article_id")
            if not (isinstance(aid, str) and aid.strip() != ""):
                src = r.get("article_path")
                if not isinstance(src, str) or src.strip() == "":
                    src = r.get("article_text") or r.get("chunk_text") or ""
                r["article_id"] = _hash.sha1(str(src).encode("utf-8")).hexdigest()
            row = r
        except Exception:
            pass
        
        # Normalize EU input fields
        row = _normalize_na_blanks(dict(row))
        
        # Format user prompt
        user_raw = format_user_prompt(row, user_template, missing_placeholder)
        
        # Token-aware trimming
        tok_local = _get_tokenizer_cached(str(getattr(cfg.model, "model_source", "")))
        try:
            txt0 = row.get("article_text") or ""
            row["article_text"] = _trim_text_for_prompt(str(txt0), tok_local, system_prompt, sampling_params, cfg)
        except Exception:
            pass
        
        user = _trim_text_for_prompt(user_raw, tok_local, system_prompt, sampling_params, cfg)
        
        # Log prompt length for debugging truncation issues
        try:
            if tok_local:
                try:
                    system_tokens = len(tok_local.encode(system_prompt, add_special_tokens=False)) if tok_local else 0
                    user_tokens = len(tok_local.encode(user, add_special_tokens=False)) if tok_local else 0
                    total_prompt_tokens = system_tokens + user_tokens
                    print(f"[classify_eu_act] Prompt length: system={system_tokens}, user={user_tokens}, total={total_prompt_tokens} tokens", flush=True)
                except Exception:
                    # Fallback to character count if tokenizer fails
                    system_chars = len(system_prompt)
                    user_chars = len(user)
                    total_chars = system_chars + user_chars
                    print(f"[classify_eu_act] Prompt length (chars): system={system_chars}, user={user_chars}, total={total_chars} chars", flush=True)
            else:
                system_chars = len(system_prompt)
                user_chars = len(user)
                total_chars = system_chars + user_chars
                print(f"[classify_eu_act] Prompt length: system={system_chars}, user={user_chars}, total={total_chars} chars", flush=True)
        except Exception as e:
            print(f"[classify_eu_act] Could not calculate prompt length: {e}", flush=True)
        
        # Sanitize and stabilize sampling params
        sp_local = sanitize_for_json(dict(sampling_params or {}))
        print(f"[classify_eu_act] DEBUG: After sanitize_for_json, sp_local max_tokens={sp_local.get('max_tokens', 'NOT SET')}", flush=True)
        
        # Log max_tokens being used in this row (for debugging truncation issues)
        try:
            row_max_tokens = sp_local.get("max_tokens", "not set")
            if isinstance(row_max_tokens, (int, float)):
                print(f"[classify_eu_act] Row max_tokens={row_max_tokens}", flush=True)
        except Exception:
            pass
        
        # Ensure any existing guided_decoding is removed before we add our own
        # (in case config had a GuidedDecodingParams object that got stringified)
        sp_local.pop("guided_decoding", None)
        
        # Add guided decoding
        if EU_GUIDED_JSON_SCHEMA is not None:
            try:
                # Use dict format directly - vLLM will reconstruct GuidedDecodingParams internally
                # This avoids PyArrow serialization issues since dicts are PyArrow-compatible
                # NOTE: Backend selection is done at engine initialization, not request level
                sp_local["guided_decoding"] = {
                    "json": EU_GUIDED_JSON_SCHEMA,
                    "disable_fallback": True,
                    "disable_additional_properties": True,
                }
            except Exception as exc:
                try:
                    print(f"[classify_eu_act] Failed to build guided_decoding dict: {exc}", flush=True)
                except Exception:
                    pass
                try:
                    schema_json_str = json.dumps(EU_GUIDED_JSON_SCHEMA, ensure_ascii=False)
                except Exception:
                    schema_json_str = ""
                sp_local["guided_decoding"] = {"json": schema_json_str or ""}
        
        print(f"[classify_eu_act] DEBUG: Before _stabilize_sampling_params, sp_local max_tokens={sp_local.get('max_tokens', 'NOT SET')}", flush=True)
        sp_local = _stabilize_sampling_params(sp_local)
        print(f"[classify_eu_act] DEBUG: After _stabilize_sampling_params, sp_local max_tokens={sp_local.get('max_tokens', 'NOT SET')}", flush=True)
        
        from datetime import datetime as _dt
        base = {k: v for k, v in row.items() if k not in {"messages", "sampling_params", "generated_text", "llm_output", "json", "guided_decoding", "response_format", "structured_output"}}
        return {
            **base,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            "sampling_params": sp_local,
            "ts_start": _dt.now().timestamp(),
        }
    
    def _post(row: Dict[str, Any]) -> Dict[str, Any]:
        from datetime import datetime as _dt
        ts_end = _dt.now().timestamp()
        usage = row.get("usage") or row.get("token_counts") or None
        raw = str(row.get("generated_text") or "").strip()
        
        # Log token usage and finish reason for debugging truncation issues
        # Always log something, even if usage is None, to debug why logging isn't appearing
        print(f"[classify_eu_act] _post called: usage={usage is not None}, raw_len={len(raw)}", flush=True)
        print(f"[classify_eu_act] _post row keys: {list(row.keys())[:20]}", flush=True)  # Log first 20 keys to see what's available
        try:
            if usage:
                prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
                max_tokens = row.get("sampling_params", {}).get("max_tokens") if isinstance(row.get("sampling_params"), dict) else None
                finish_reason = row.get("finish_reason") or usage.get("finish_reason") or "unknown"
                total_tokens = prompt_tokens + output_tokens
                print(
                    f"[classify_eu_act] Token usage: prompt={prompt_tokens}, output={output_tokens}, "
                    f"max_tokens={max_tokens}, total={total_tokens}, finish_reason={finish_reason}, raw_len={len(raw)}",
                    flush=True,
                )
                # Check if truncation is due to context length limit
                if finish_reason == "length":
                    print(
                        f"[classify_eu_act] WARNING: Output truncated due to length limit! "
                        f"prompt_tokens={prompt_tokens} + max_tokens={max_tokens} may exceed max_model_len",
                        flush=True,
                    )
            else:
                # Check alternative locations for usage info
                print(f"[classify_eu_act] WARNING: usage is None or empty. Row keys: {list(row.keys())[:20]}", flush=True)
                # Try to find usage in other possible locations
                if "token_usage_prompt" in row or "token_usage_output" in row:
                    print(f"[classify_eu_act] Found token_usage columns: prompt={row.get('token_usage_prompt')}, output={row.get('token_usage_output')}", flush=True)
        except Exception as e:
            print(f"[classify_eu_act] ERROR in token usage logging: {e}", flush=True)
            import traceback
            traceback.print_exc()
        
        parsed = extract_last_json(raw)
        
        eu_desc = None
        eu_label = None
        eu_reltext = None
        eu_reason = None
        
        if isinstance(parsed, dict):
            eu_desc = parsed.get("Description")
            lab = parsed.get("Classification")
            if isinstance(lab, list) and lab:
                eu_label = lab[0]
            elif isinstance(lab, str):
                eu_label = lab
            eu_reltext = parsed.get("Relevant Text from the EU AI Act")
            eu_reason = parsed.get("Reasoning")
        else:
            try:
                print(
                    "[classify_eu_act] Failed to parse LLM JSON output; full raw output:\n"
                    f"{raw}",
                    flush=True,
                )
            except Exception:
                pass
        
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
                "matched_keywords",
                "json",
                "structured_output",
                "guided_decoding",
                "response_format",
            ])
        
        try:
            eu_ai_raw_json_str = to_json_str(parsed)
        except Exception:
            eu_ai_raw_json_str = None
        
        return {
            **row,
            "eu_ai_desc": eu_desc,
            "eu_ai_label": eu_label,
            "eu_ai_relevant_text": eu_reltext,
            "eu_ai_reason": eu_reason,
            "eu_ai_raw_json": eu_ai_raw_json_str,
            "llm_output": row.get("generated_text"),
            "classification_mode": "eu_ai_act",
            "too_vague_to_process": False,
            "latency_s": (float(ts_end) - float(row.get("ts_start", ts_end))),
            "token_usage_prompt": ((usage or {}).get("prompt_tokens") or (usage or {}).get("input_tokens")),
            "token_usage_output": ((usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens")),
            "token_usage_total": ((usage or {}).get("total_tokens")),
            "_progress_row": 1,
        }
    
    # Ray Dataset path
    if is_ray_ds:
        ds_in = in_df
        ds_vague = None
        
        # Debug: inspect incoming schema before any pruning to track problematic columns
        try:
            schema = ds_in.schema()
            cols = list(schema.names)
            print(f"[classify_eu_act] Incoming Ray dataset columns ({len(cols)}): {cols}", flush=True)
            try:
                import pyarrow as pa  # type: ignore
                schema_desc = [
                    f"{field.name}:{field.type}" if isinstance(field, pa.Field) else str(field)
                    for field in schema
                ]
                print(
                    "[classify_eu_act] Incoming schema field types: " + ", ".join(schema_desc),
                    flush=True,
                )
            except Exception:
                pass
        except Exception:
            pass
        
        # Prune columns
        try:
            current_cols = set(ds_in.schema().names)
        except Exception:
            current_cols = set()
        allowed_columns = get_required_input_columns(True, False)  # EU profile
        allowed_columns = allowed_columns & current_cols
        allowed_columns.update({col for col in ("article_id", "article_text") if col in current_cols})
        try:
            if allowed_columns and allowed_columns != current_cols:
                ds_in = ds_in.select_columns(sorted(allowed_columns))
                print(
                    f"[classify_eu_act] Pruned dataset columns to {sorted(allowed_columns)}",
                    flush=True,
                )
        except Exception:
            pass
        
        # After pruning, log the updated schema for confirmation
        try:
            new_schema = ds_in.schema()
            new_cols = list(new_schema.names)
            print(f"[classify_eu_act] Columns after pruning ({len(new_cols)}): {new_cols}", flush=True)
            try:
                import pyarrow as pa  # type: ignore
                new_schema_desc = [
                    f"{field.name}:{field.type}" if isinstance(field, pa.Field) else str(field)
                    for field in new_schema
                ]
                print(
                    "[classify_eu_act] Post-prune schema field types: "
                    + ", ".join(new_schema_desc),
                    flush=True,
                )
            except Exception:
                pass
        except Exception:
            pass
        
        # Ensure article_id
        def _ensure_article_id_map(r: Dict[str, Any]) -> Dict[str, Any]:
            try:
                aid = r.get("article_id")
                if not (isinstance(aid, str) and aid.strip() != ""):
                    src = r.get("article_path")
                    if not isinstance(src, str) or src.strip() == "":
                        src = r.get("article_text") or r.get("chunk_text") or ""
                    import hashlib as _hash
                    r["article_id"] = _hash.sha1(str(src).encode("utf-8")).hexdigest()
            except Exception:
                pass
            return r
        try:
            ds_in = ds_in.map(_ensure_article_id_map)
        except Exception:
            pass
        
        def _coerce_boolish_map(r: Dict[str, Any]) -> Dict[str, Any]:
            return coerce_boolish_row(dict(r))
        try:
            ds_in = ds_in.map(_coerce_boolish_map)
        except Exception:
            pass
        
        # EU vague flags filtering
        try:
            def _add_vague_flags_map(r: Dict[str, Any]) -> Dict[str, Any]:
                return add_eu_vague_flags(dict(r), missing_placeholder)
            ds_in = ds_in.map(_add_vague_flags_map)
            ds_vague = ds_in.filter(lambda r: bool(r.get("too_vague_to_process", False)))
            ds_in = ds_in.filter(lambda r: not bool(r.get("too_vague_to_process", False)))
            try:
                vague_count = ds_vague.count()
                print(f"[classify_eu_act] Filtered {vague_count} too-vague rows before LLM processing", flush=True)
            except Exception:
                pass
        except Exception:
            ds_vague = None
        
        # Trim text
        sys_text = system_prompt
        def _trim_row(r: Dict[str, Any]) -> Dict[str, Any]:
            txt = r.get("article_text") or ""
            # Use char-based trimming to avoid heavy tokenizer in Ray workers.
            r["article_text"] = _trim_text_for_prompt(str(txt), None, sys_text, sampling_params, cfg)
            return r
        ds_in = ds_in.map(_trim_row)
        
        # Process with LLM
        processor = build_llm_processor(engine_config, preprocess=_pre, postprocess=_post)
        ds_llm_results = processor(ds_in)
        
        # Drop internal columns
        try:
            schema = ds_llm_results.schema()
            existing_cols = schema.names if hasattr(schema, 'names') else []
            cols_to_drop = [col for col in _INTERNAL_DROP_COLS if col in existing_cols]
            if cols_to_drop:
                ds_llm_results = ds_llm_results.drop_columns(cols_to_drop)
        except Exception as e:
            print(f"[classify_eu_act] Warning: Failed to drop internal columns: {e}", flush=True)
        
        # Merge vague rows back
        skip_merge_filtered = bool(getattr(cfg.runtime, "skip_merge_filtered", False))
        print(
            f"[classify_eu_act] Checking merge conditions: ds_vague={ds_vague is not None}, skip_merge_filtered={skip_merge_filtered}",
            flush=True,
        )
        if skip_merge_filtered:
            df_llm = ds_llm_results.to_pandas().drop(columns=_INTERNAL_DROP_COLS, errors="ignore")
            if "too_vague_to_process" not in df_llm.columns:
                df_llm["too_vague_to_process"] = False
            else:
                df_llm["too_vague_to_process"] = df_llm["too_vague_to_process"].fillna(False)
            print(f"[classify_eu_act] skip_merge_filtered enabled; returning {len(df_llm)} rows without merging vague", flush=True)
            return df_llm
        
        if ds_vague is not None:
            print(f"[classify_eu_act] Starting merge process...", flush=True)
            try:
                df_llm = ds_llm_results.to_pandas()
                df_vague_pd = ds_vague.to_pandas().copy()
                
                result_parts = [normalize_profile_columns(df_llm, True, False)]
                
                print(f"[classify_eu_act] Merging back {len(df_vague_pd)} too-vague rows to output", flush=True)
                df_vague_pd["classification_mode"] = "too_vague_to_process"
                df_vague_pd["too_vague_to_process"] = True
                df_vague_pd = normalize_profile_columns(df_vague_pd, True, False)
                result_parts.append(df_vague_pd)
                
                result_df = merge_result_parts(result_parts)
                if "too_vague_to_process" not in result_df.columns:
                    result_df["too_vague_to_process"] = False
                else:
                    result_df["too_vague_to_process"] = result_df["too_vague_to_process"].fillna(False)
                result_df = result_df.drop(columns=_INTERNAL_DROP_COLS, errors="ignore")
                result_df = prune_result_columns(result_df, True, False)
                try:
                    num_vague = len(df_vague_pd) if 'df_vague_pd' in locals() else 0
                except Exception:
                    num_vague = 0
                print(f"[classify_eu_act] Merged results: {len(df_llm)} LLM-processed + {num_vague} too-vague = {len(result_df)} total", flush=True)
                return result_df
            except Exception as e:
                print(f"[classify_eu_act] ERROR: Failed to merge vague rows: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return ds_llm_results
        
        print(f"[classify_eu_act] No merge needed, returning LLM results directly", flush=True)
        return ds_llm_results
    
    # Pandas path
    try:
        total_rows = int(len(out))
    except Exception:
        total_rows = 0
    try:
        desired_rows_per_block = int(getattr(cfg.runtime, "rows_per_block", 2000) or 2000)
    except Exception:
        desired_rows_per_block = 2000
    step = max(1, desired_rows_per_block)
    if total_rows > step:
        try:
            dfs_list = [out.iloc[i:i + step] for i in range(0, total_rows, step)]
            ds = ray.data.from_pandas(dfs_list)
        except Exception:
            ds = ray.data.from_pandas(out)
    else:
        ds = ray.data.from_pandas(out)
    
    ds_in = ds
    ds_vague = None
    
    # Debug instrumentation for pandas path: capture schema prior to pruning
    try:
        schema = ds_in.schema()
        cols = list(schema.names)
        print(f"[classify_eu_act] (pandas path) incoming dataset columns ({len(cols)}): {cols}", flush=True)
        try:
            import pyarrow as pa  # type: ignore
            schema_desc = [
                f"{field.name}:{field.type}" if isinstance(field, pa.Field) else str(field)
                for field in schema
            ]
            print(
                "[classify_eu_act] (pandas path) incoming schema field types: "
                + ", ".join(schema_desc),
                flush=True,
            )
        except Exception:
            pass
    except Exception:
        cols = []
        schema = None
    
    # Prune columns
    try:
        current_cols = set(schema.names) if schema else set()
    except Exception:
        current_cols = set()
    allowed_columns = get_required_input_columns(True, False)
    allowed_columns = allowed_columns & current_cols
    allowed_columns.update({col for col in ("article_id", "article_text") if col in current_cols})
    try:
        if allowed_columns and allowed_columns != current_cols:
            ds_in = ds_in.select_columns(sorted(allowed_columns))
            print(
                f"[classify_eu_act] (pandas path) pruned dataset columns to {sorted(allowed_columns)}",
                flush=True,
            )
            try:
                new_schema = ds_in.schema()
                new_cols = list(new_schema.names)
                print(
                    f"[classify_eu_act] (pandas path) columns after pruning ({len(new_cols)}): {new_cols}",
                    flush=True,
                )
                try:
                    import pyarrow as pa  # type: ignore
                    new_schema_desc = [
                        f"{field.name}:{field.type}" if isinstance(field, pa.Field) else str(field)
                        for field in new_schema
                    ]
                    print(
                        "[classify_eu_act] (pandas path) post-prune schema field types: "
                        + ", ".join(new_schema_desc),
                        flush=True,
                    )
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass
    
    # EU vague flags filtering
    try:
        def _add_vague_flags_pandas(r: Dict[str, Any]) -> Dict[str, Any]:
            row = add_eu_vague_flags(dict(r), missing_placeholder)
            # Serialize complex nested structures to JSON strings for PyArrow compatibility
            # This prevents PyArrow serialization errors when Ray Dataset tries to convert the data
            serialize_arrow_unfriendly_in_row(row, ["missing", "missing_fields", "eu_missing_fields"])
            return row
        ds_in = ds_in.map(_add_vague_flags_pandas)
        ds_vague = ds_in.filter(lambda r: bool(r.get("too_vague_to_process", False)))
        ds_in = ds_in.filter(lambda r: not bool(r.get("too_vague_to_process", False)))
        try:
            vague_count = ds_vague.count()
            print(f"[classify_eu_act] Filtered {vague_count} too-vague rows before LLM processing", flush=True)
        except Exception:
            pass
    except Exception:
        ds_vague = None
    
    # Trim text
    sys_text = system_prompt
    def _trim_row(r: Dict[str, Any]) -> Dict[str, Any]:
        txt = r.get("article_text") or ""
        # Use char-based trimming to avoid heavy tokenizer in Ray workers.
        r["article_text"] = _trim_text_for_prompt(str(txt), None, sys_text, sampling_params, cfg)
        return r
    ds_in = ds_in.map(_trim_row)
    
    # Process with LLM
    processor = build_llm_processor(engine_config, preprocess=_pre, postprocess=_post)
    out_df = processor(ds_in).to_pandas()
    
    # Drop internal columns
    try:
        out_df = out_df.drop(columns=_INTERNAL_DROP_COLS, errors="ignore")
    except Exception:
        pass
    
    # Merge vague rows back
    print(f"[classify_eu_act] Checking merge conditions: ds_vague={ds_vague is not None}", flush=True)
    if ds_vague is not None:
        print(f"[classify_eu_act] Starting merge process...", flush=True)
        try:
            df_llm = out_df
            df_vague_pd = ds_vague.to_pandas().copy()
            
            result_parts = [normalize_profile_columns(df_llm, True, False)]
            
            print(f"[classify_eu_act] Merging back {len(df_vague_pd)} too-vague rows to output", flush=True)
            df_vague_pd["classification_mode"] = "too_vague_to_process"
            df_vague_pd["too_vague_to_process"] = True
            df_vague_pd = normalize_profile_columns(df_vague_pd, True, False)
            result_parts.append(df_vague_pd)
            
            out_df = merge_result_parts(result_parts)
            if "too_vague_to_process" not in out_df.columns:
                out_df["too_vague_to_process"] = False
            else:
                out_df["too_vague_to_process"] = out_df["too_vague_to_process"].fillna(False)
            out_df = out_df.drop(columns=_INTERNAL_DROP_COLS, errors="ignore")
            try:
                num_vague = len(df_vague_pd) if 'df_vague_pd' in locals() else 0
            except Exception:
                num_vague = 0
            print(f"[classify_eu_act] Merged results: {len(df_llm)} LLM-processed + {num_vague} too-vague = {len(out_df)} total", flush=True)
        except Exception as e:
            print(f"[classify_eu_act] ERROR: Failed to merge vague rows: {e}", flush=True)
            import traceback
            traceback.print_exc()
    else:
        print(f"[classify_eu_act] No merge needed, returning LLM results directly", flush=True)
    
    out_df = prune_result_columns(out_df, True, False)
    return out_df
