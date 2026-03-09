"""Relevance classification profile implementation - standalone module following decompose_nbl.py pattern."""

from typing import Any, Dict, List
import pandas as pd
import json
import os
import logging
from omegaconf import OmegaConf

from .classify_shared import (
    build_relevant_regex,
    generate_relevant_blocks,
    coerce_boolish_row,
    coerce_boolish_df,
    detect_num_gpus,
    detect_gpu_type,
    apply_gpu_aware_batch_settings,
    filter_vllm_engine_kwargs,
    to_json_str,
    serialize_arrow_unfriendly_in_row,
    sanitize_for_json,
    get_required_input_columns,
    normalize_profile_columns,
    merge_result_parts,
    prune_result_columns,
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


def heuristic_relevance(text: Any) -> bool:
    """Simple AI/news relevance heuristic."""
    s = str(text or "").lower()
    if not s:
        return False
    keywords = [
        "artificial intelligence", " ai ", "machine learning", "ml ", "neural network", "deep learning",
        "large language model", "llm", "chatgpt", "gpt-", "openai", "anthropic", "claude", "gemini", "qwen",
        "transformer model", "fine-tuning", "inference", "prompting", "agents", "autonomous agent", "model weights",
    ]
    return any(k in s for k in keywords)


def format_user_prompt(article_text: str, row: Dict[str, Any], user_template: str) -> str:
    """Format user prompt for relevance classification."""
    text_val = str((row.get("chunk_text") if row.get("chunk_text") else article_text) or "")
    # Provide basic ids to satisfy experiments-style templates when present
    try:
        art_id = row.get("article_id") or row.get("name") or None
    except Exception:
        art_id = None
    if not art_id:
        try:
            import hashlib as _hash
            art_id = _hash.sha1(text_val.encode("utf-8")).hexdigest()[:12]
        except Exception:
            art_id = "unknown"
    # If template uses '{chunk_text}' style, format with .format; else support UAIR '{{rule_text}}'
    if "{chunk_text}" in user_template or "{article_id}" in user_template:
        try:
            return user_template.format(article_id=art_id, chunk_id=0, num_chunks=1, chunk_text=text_val)
        except Exception:
            pass
    # Maintain UAIR template compatibility
    return user_template.replace("{{rule_text}}", text_val).replace("{{article_text}}", text_val)


def _get_tokenizer(model_source: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
        return AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)
    except Exception:
        return None


_TOK_CACHED = None  # lazy per-process tokenizer cache


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
                max_output = int(sampling_params.get("max_tokens", 8) or 8)
            else:
                max_output = int((sampling_params or {}).get("max_tokens", 8))
        except Exception:
            max_output = 8
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


def _extract_matched_keywords(text: Any, kw_regex) -> tuple[bool, list[str], int]:
    """Extract which keywords matched in the text.

    Returns:
        tuple: (has_keyword, matched_keywords_list, match_count)
    """
    if kw_regex is None:
        return True, [], 0
    try:
        text_str = str(text or "").lower()
        if not text_str:
            return False, [], 0

        # Use finditer() and .group() to get full matches (not capture groups)
        matches = [m.group() for m in kw_regex.finditer(text_str)]
        if not matches:
            return False, [], 0

        # Deduplicate and sort for consistency
        unique_keywords = sorted(set(matches))
        match_count = len(matches)  # Total matches (including duplicates)

        return True, unique_keywords, match_count
    except Exception:
        return True, [], 0


def _filter_vllm_sampling_params(sp: Dict[str, Any]) -> Dict[str, Any]:
    """Remove known problematic params that cause issues with vLLM.

    Note: We don't use introspection-based filtering because:
    1. It's unreliable (vLLM integration may accept params introspection doesn't show)
    2. It was incorrectly filtering out valid params like max_tokens
    3. vLLM will handle unsupported params gracefully (ignore or raise clear error)

    We only remove params that are known to cause issues.
    """
    sp2 = dict(sp)
    # Remove params that are known to cause issues with vLLM
    # These are OpenAI API params that vLLM doesn't support
    for k in ("early_stopping", "length_penalty", "response_format", "structured_output"):
        sp2.pop(k, None)
    # Always preserve guided_decoding
    if "guided_decoding" in sp:
        sp2["guided_decoding"] = sp["guided_decoding"]
    return sp2


# Stable baseline for sampling params to avoid Arrow struct field drift across batches
_SAMPLING_DEFAULTS: Dict[str, Any] = {
    "max_tokens": 8,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "use_beam_search": False,
    "num_beams": 1,
    "early_stopping": False,
    "detokenize": True,
    "logprobs": 0,
    "n": 1,
    "best_of": 1,
    "stop": [],
    "stop_token_ids": [],
    "ignore_eos_token": False,
}


def _stabilize_sampling_params(sp: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure sampling_params have a stable schema across rows.

    This function only ensures schema stability (e.g., making sure 'stop' is a list).
    It does NOT add any default values - all values must come from config.
    """
    try:
        # Start with the actual config values, not defaults
        out = dict(sp or {})
        # Only normalize 'stop' to ensure it's a list (schema stability)
        stop_val = out.get("stop")
        if stop_val is None:
            out["stop"] = []
        elif not isinstance(stop_val, list):
            out["stop"] = [str(stop_val)]
        # Filter to only vLLM-supported params (doesn't add defaults)
        return _filter_vllm_sampling_params(out)
    except Exception:
        return dict(sp or {})


def run_classification_relevance(df: Any, cfg) -> Any:
    """
    Relevance classification implementation - standalone module.

    Pure pandas + run_vllm_inference path with keyword pre-gating support.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["article_id", "article_text", "is_relevant"])
    out = df.copy()

    # Ensure article_id exists for all rows
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

    use_llm = bool(getattr(cfg.runtime, "use_llm_classify", False))

    # Heuristic path
    if not use_llm:
        out["is_relevant"] = out["article_text"].apply(heuristic_relevance)
        out["classification_mode"] = "heuristic"
        return out

    # Keyword-based buffering and gating config
    try:
        enable_kw_buf = bool(getattr(cfg.runtime, "keyword_buffering", True))
    except Exception:
        enable_kw_buf = True

    try:
        prefilter_mode = str(getattr(cfg.runtime, "prefilter_mode", "pre_gating")).strip().lower()
    except Exception:
        prefilter_mode = "pre_gating"

    try:
        window_words = int(getattr(cfg.runtime, "keyword_window_words", 100) or 100)
    except Exception:
        window_words = 100

    kw_regex = build_relevant_regex() if enable_kw_buf else None

    # Get prompts
    # Support prompt subdirectory override (e.g., 'general_ai' for general_ai/classify.yaml)
    prompt_subdir = getattr(cfg.runtime, "prompt_subdirectory", None)
    if prompt_subdir:
        try:
            from .classify_shared import inject_prompt_from_file
            inject_prompt_from_file(cfg, f"{prompt_subdir}/classify.yaml")
        except Exception:
            pass  # Fall back to config-based loading

    try:
        system_prompt = (
            getattr(getattr(cfg, "prompts", {}), "relevance_v1", {}).get("system")  # type: ignore
            if hasattr(cfg, "prompts") else None
        )
    except Exception:
        system_prompt = None
    if not system_prompt:
        system_prompt = str(getattr(cfg.prompt, "system_prompt", ""))

    try:
        user_template = (
            getattr(getattr(cfg, "prompts", {}), "relevance_v1", {}).get("user_template")  # type: ignore
            if hasattr(cfg, "prompts") else None
        )
    except Exception:
        user_template = None
    if not user_template:
        user_template = str(getattr(cfg.prompt, "prompt_template", ""))

    # Sampling params - use ONLY values from config, no defaults
    try:
        sp_src = getattr(cfg, "sampling_params_classify", getattr(cfg, "sampling_params", {}))
        sampling_params = OmegaConf.to_container(sp_src, resolve=True) if isinstance(sp_src, (dict,)) or hasattr(sp_src, "_get_node") else dict(sp_src)
    except Exception:
        sampling_params = dict(getattr(cfg, "sampling_params", {}))

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

    def _attach_chunk_text(row: Dict[str, Any]) -> Dict[str, Any]:
        if not enable_kw_buf or kw_regex is None:
            return row
        text_val = row.get("article_text")
        try:
            blocks = generate_relevant_blocks(text_val, kw_regex, window_words)
        except Exception:
            blocks = []
        if blocks:
            row["chunk_text"] = "\n\n".join(blocks)
        else:
            row.setdefault("chunk_text", str(text_val or ""))
        return row

    def _normalize_na_blanks(r: Dict[str, Any]) -> Dict[str, Any]:
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

        if enable_kw_buf and kw_regex is not None:
            row = _attach_chunk_text(dict(row))

        row = _normalize_na_blanks(dict(row))

        # Format user prompt
        user_raw = format_user_prompt(row.get("article_text"), row, user_template)

        # Token-aware trimming
        tok_local = _get_tokenizer_cached(str(getattr(cfg.model, "model_source", "")))
        try:
            txt0 = row.get("chunk_text") or row.get("article_text")
            row["chunk_text"] = _trim_text_for_prompt(str(txt0 or ""), tok_local, system_prompt, sampling_params, cfg)
        except Exception:
            pass

        user = _trim_text_for_prompt(user_raw, tok_local, system_prompt, sampling_params, cfg)

        # Sanitize and stabilize sampling params
        sp_local = sanitize_for_json(dict(sampling_params or {}))
        sp_local = _stabilize_sampling_params(sp_local)

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
        text = str(row.get("generated_text") or "").strip().upper()
        is_rel = text.startswith("YES") or ("YES" in text and "NO" not in text)

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

        return {
            **row,
            "relevance_answer": row.get("generated_text"),
            "is_relevant": bool(is_rel),
            "llm_output": row.get("generated_text"),
            "classification_mode": "llm_relevance",
            "latency_s": (float(ts_end) - float(row.get("ts_start", ts_end))),
            "token_usage_prompt": ((usage or {}).get("prompt_tokens") or (usage or {}).get("input_tokens")),
            "token_usage_output": ((usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens")),
            "token_usage_total": ((usage or {}).get("total_tokens")),
            "_progress_row": 1,
        }

    # Keyword pre-gating in pure pandas
    df_filtered = pd.DataFrame()
    if prefilter_mode in ("pre_gating", "post_gating"):
        # Apply keyword extraction to every row
        kw_results = out["article_text"].apply(
            lambda text: _extract_matched_keywords(text, kw_regex)
        )
        out["relevant_keyword"] = kw_results.apply(lambda t: t[0])
        out["matched_keywords"] = kw_results.apply(lambda t: t[1])
        out["keyword_match_count"] = kw_results.apply(lambda t: t[2])

        if prefilter_mode == "pre_gating":
            df_for_llm = out[out["relevant_keyword"]].copy()
            df_filtered = out[~out["relevant_keyword"]].copy()
            df_filtered["classification_mode"] = "filtered_by_keyword"
            if "matched_keywords" in df_filtered.columns:
                df_filtered["matched_keywords"] = df_filtered["matched_keywords"].apply(to_json_str)
        else:
            df_for_llm = out
    else:
        df_for_llm = out

    # Attach chunk_text and trim (operates on rows destined for LLM)
    if enable_kw_buf and kw_regex is not None and len(df_for_llm):
        df_for_llm = df_for_llm.copy()
        df_for_llm = pd.DataFrame(
            [_attach_chunk_text(dict(r)) for r in df_for_llm.to_dict(orient="records")]
        )

    sys_text = system_prompt
    if len(df_for_llm):
        def _trim_row_text(r: Dict[str, Any]) -> Dict[str, Any]:
            txt = r.get("chunk_text") or r.get("article_text")
            r["chunk_text"] = _trim_text_for_prompt(str(txt or ""), None, sys_text, sampling_params, cfg)
            return r
        df_for_llm = pd.DataFrame(
            [_trim_row_text(dict(r)) for r in df_for_llm.to_dict(orient="records")]
        )

    # LLM inference
    from dagspaces.common.vllm_inference import run_vllm_inference
    result_df = run_vllm_inference(df_for_llm, cfg, _pre, _post, "classify_relevance")

    # Drop internal columns
    try:
        result_df = result_df.drop(columns=_INTERNAL_DROP_COLS, errors="ignore")
    except Exception:
        pass

    # Merge with keyword-filtered rows (pre_gating only)
    if prefilter_mode == "pre_gating" and len(df_filtered):
        print(
            f"[classify_relevance] Merging: {len(result_df)} LLM-processed + "
            f"{len(df_filtered)} keyword-filtered = {len(result_df) + len(df_filtered)} total",
            flush=True,
        )
        result_parts = [
            normalize_profile_columns(result_df, False, False),
            normalize_profile_columns(df_filtered, False, False),
        ]
        result_df = merge_result_parts(result_parts)
        result_df = result_df.drop(columns=_INTERNAL_DROP_COLS, errors="ignore")

    result_df = prune_result_columns(result_df, False, False)
    return result_df
