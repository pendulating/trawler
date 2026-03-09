"""
Classification stage module - DEPRECATED.

This module contains the legacy implementation of the classification stage.
It is maintained for internal reference only.

DEPRECATION NOTICE:
    This module is deprecated. Use profile-specific modules instead:
    - classify_relevance: Relevance classification
    - classify_eu_act: EU AI Act classification  
    - classify_risks_benefits: Risks and Benefits classification
    
    The legacy run_classification_stage() function will be removed in a future version.
    Profile-specific modules call run_classification_core() directly.
    
    NOTE: _run_classification_stage_impl is kept as an internal reference only.
    It is not part of the public API and should not be used by external code.
"""

import warnings
from typing import Any, Dict, List, Set

# Import shared utilities for backward compatibility
from .classify_shared import (
    BASE_INPUT_COLUMNS,
    EU_INPUT_KEYS,
    EU_OUTPUT_COLUMNS,
    EU_PROFILE_FLAG_COLUMNS,
    RELEVANCE_OUTPUT_COLUMNS,
    RESULT_BASE_COLUMNS,
    RISKS_BENEFITS_OUTPUT_COLUMNS,
    get_allowed_result_columns,
    get_required_input_columns,
    prune_result_columns,
)

# Deprecation warning flag
_DEPRECATION_WARNING_SHOWN = False

import pandas as pd

# Public API exports
__all__ = [
    "run_classification_stage",
    "run_classification_core",
    "get_required_input_columns",
    "get_allowed_result_columns",
    "BASE_INPUT_COLUMNS",
    "EU_INPUT_KEYS",
    "EU_OUTPUT_COLUMNS",
    "EU_PROFILE_FLAG_COLUMNS",
    "RELEVANCE_OUTPUT_COLUMNS",
    "RESULT_BASE_COLUMNS",
    "RISKS_BENEFITS_OUTPUT_COLUMNS",
]


def run_classification_stage(df: pd.DataFrame, cfg) -> Any:
    """
    DEPRECATED: Classification stage function.
    
    This function is deprecated. Use profile-specific runners instead:
    - ClassificationRelevanceRunner for relevance classification
    - ClassificationEUActRunner for EU AI Act classification
    - ClassificationRisksBenefitsRunner for risks and benefits classification
    """
    global _DEPRECATION_WARNING_SHOWN
    if not _DEPRECATION_WARNING_SHOWN:
        warnings.warn(
            "classify.run_classification_stage is deprecated. "
            "Use profile-specific runners (ClassificationRelevanceRunner, "
            "ClassificationEUActRunner, ClassificationRisksBenefitsRunner) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _DEPRECATION_WARNING_SHOWN = True
    
    # Determine profile and call appropriate function directly
    try:
        classification_profile = str(
            getattr(cfg.runtime, "classification_profile", "relevance") or "relevance"
        ).strip().lower()
    except Exception:
        classification_profile = "relevance"
    
    if classification_profile == "eu_ai_act":
        from .classify_eu_act import run_classification_eu_act
        return run_classification_eu_act(df, cfg)
    elif classification_profile == "risks_and_benefits":
        from .classify_risks_benefits import run_classification_risks_benefits
        return run_classification_risks_benefits(df, cfg)
    else:
        from .classify_relevance import run_classification_relevance
        return run_classification_relevance(df, cfg)

# Column definitions and utility functions have been moved to classify_shared.py
# These are re-exported above for convenience

from vllm.sampling_params import SamplingParams, GuidedDecodingParams  # type: ignore
import re
from bisect import bisect_right
import numpy as np
import os
import logging
import json
from omegaconf import OmegaConf

from dagspaces.common.vllm_inference import run_vllm_inference

# Import shared utilities
from .classify_shared import (
    maybe_silence_vllm_logs,
    to_json_str,
    serialize_arrow_unfriendly_in_row,
    sanitize_for_json,
    extract_last_json,
    coerce_bool_like,
    coerce_boolish_row,
    coerce_boolish_df,
    build_relevant_regex,
    generate_relevant_blocks,
    normalize_profile_columns,
    merge_result_parts,
    prune_result_columns,
)


# Local duplicate utility function definitions have been removed.
# The underscore-prefixed names below (e.g. _coerce_boolish_df) are aliases
# set just above, pointing to the canonical implementations in classify_shared.








def run_classification_core(df: pd.DataFrame, cfg):
    """
    Core classification implementation.
    
    This is the actual implementation that performs classification.
    Profile-specific modules call this function after setting up their profile-specific configuration.
    
    NOTE: This is the legacy implementation preserved for backward compatibility.
    The full implementation (2965 lines) is kept here to maintain functionality
    while we transition to profile-specific modules.
    
    Article relevance classification stage:
    - Heuristic: detect AI-related keywords in text
    - LLM: borrow prompt shape from experiments/prompts/base.yaml (relevance_v1) when available,
      else fall back to UAIR classify prompt. Produces both `relevance_answer` and `is_relevant`.
    Uses `article_text` as the canonical input text column.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["article_id", "article_text", "is_relevant"])  # minimal
    out = df.copy()
    if hasattr(out, "to_pandas") and not isinstance(out, pd.DataFrame):
        out = out.to_pandas()
    if out is None or len(out) == 0:
        return pd.DataFrame(columns=["article_id", "article_text", "is_relevant"])
    # Ensure article_id exists for all rows (deterministic hash of path or text)
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
            # Treat non-strings, empty strings, and NaNs as missing IDs
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
    out = _coerce_boolish_df(out)
    def _heuristic(text: Any) -> bool:
        s = str(text or "").lower()
        if not s:
            return False
        # Simple AI/news relevance heuristic aligned to experiments.relevance_stage intent
        keywords = [
            "artificial intelligence"," ai ","machine learning","ml ","neural network","deep learning",
            "large language model","llm","chatgpt","gpt-","openai","anthropic","claude","gemini","qwen",
            "transformer model","fine-tuning","inference","prompting","agents","autonomous agent","model weights",
        ]
        return any(k in s for k in keywords)
    use_llm = bool(getattr(cfg.runtime, "use_llm_classify", False))
    # Select classification profile (Hydra overrideable): 'relevance' (default), 'eu_ai_act', or 'risks_and_benefits'
    try:
        classification_profile = str(getattr(cfg.runtime, "classification_profile", "relevance") or "relevance").strip().lower()
    except Exception:
        classification_profile = "relevance"
    is_eu_profile = (classification_profile == "eu_ai_act")
    is_risks_benefits_profile = (classification_profile == "risks_and_benefits")
    try:
        print(
            f"[classify] classification_profile={classification_profile} eu={is_eu_profile} rb={is_risks_benefits_profile}",
            flush=True,
        )
    except Exception:
        pass
    # Prefilter gating: pre_gating (filter before LLM), post_gating (compute flag only), off
    try:
        prefilter_mode = str(getattr(cfg.runtime, "prefilter_mode", "pre_gating")).strip().lower()
    except Exception:
        prefilter_mode = "pre_gating"
    if not use_llm:
        out["is_relevant"] = out["article_text"].apply(_heuristic)
        out["classification_mode"] = "heuristic"
        return out

    # LLM classification via vLLM
    # Optional keyword-based buffering (±window words around matches)
    try:
        enable_kw_buf = bool(getattr(cfg.runtime, "keyword_buffering", True))
    except Exception:
        enable_kw_buf = True

    # EU and risks_benefits profiles: disable keyword buffering/gating entirely
    if is_eu_profile or is_risks_benefits_profile:
        prefilter_mode = "off"
        enable_kw_buf = False
    try:
        window_words = int(getattr(cfg.runtime, "keyword_window_words", 100) or 100)
    except Exception:
        window_words = 100
    kw_regex = _build_relevant_regex() if enable_kw_buf else None
    # Compute keyword prefilter flag for gating
    def _kw_flag(text: Any) -> bool:
        if kw_regex is None:
            return True
        try:
            return bool(kw_regex.search(str(text or "")))
        except Exception:
            return True
    
    # Extract specific keywords that matched in the text
    def _extract_matched_keywords(text: Any) -> tuple[bool, list[str], int]:
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
            # This avoids the issue where findall() with capturing groups returns tuples
            matches = [m.group() for m in kw_regex.finditer(text_str)]
            if not matches:
                return False, [], 0
            
            # Deduplicate and sort for consistency
            unique_keywords = sorted(set(matches))
            match_count = len(matches)  # Total matches (including duplicates)
            
            return True, unique_keywords, match_count
        except Exception:
            return True, [], 0
    # Prefer experiments-style prompts when available; otherwise fall back to UAIR prompt
    if is_eu_profile:
        # EU AI Act classification: try nested prompts.eu_ai_act first; else cfg.prompt.* (Hydra-composed)
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
            user_template_eu = (
                getattr(getattr(cfg, "prompts", {}), "eu_ai_act", {}).get("user_template")  # type: ignore
                if hasattr(cfg, "prompts") else None
            )
        except Exception:
            user_template_eu = None
        if not user_template_eu:
            user_template_eu = str(getattr(cfg.prompt, "prompt_template", ""))
    elif is_risks_benefits_profile:
        # Risks and Benefits classification: try nested prompts.risks_and_benefits first; else cfg.prompt.* (Hydra-composed)
        try:
            system_prompt = (
                getattr(getattr(cfg, "prompts", {}), "risks_and_benefits", {}).get("system")  # type: ignore
                if hasattr(cfg, "prompts") else None
            )
        except Exception:
            system_prompt = None
        if not system_prompt:
            system_prompt = str(getattr(cfg.prompt, "system_prompt", ""))

        try:
            user_template_risks_benefits = (
                getattr(getattr(cfg, "prompts", {}), "risks_and_benefits", {}).get("user_template")  # type: ignore
                if hasattr(cfg, "prompts") else None
            )
        except Exception:
            user_template_risks_benefits = None
        if not user_template_risks_benefits:
            user_template_risks_benefits = str(getattr(cfg.prompt, "prompt_template", ""))
    else:
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

    def _format_user(article_text: str, row: Dict[str, Any]) -> str:
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

    # Utilities for EU AI Act prompt formatting
    try:
        _EU_MISSING_PH = str(getattr(cfg.runtime, "eu_missing_placeholder", "Not known/specified"))
    except Exception:
        _EU_MISSING_PH = "Not known/specified"

    def _is_nan_like(v: Any) -> bool:
        try:
            # pandas/NumPy NaN/NA checks
            if v is None:
                return True
            if isinstance(v, float):
                return v != v
            # pandas NA scalar string repr
            return str(v).strip().lower() in {"nan", "na", "none"}
        except Exception:
            return False

    def _norm_str(val: Any) -> str:
        try:
            if _is_nan_like(val):
                return _EU_MISSING_PH
            # Flatten common containers
            if isinstance(val, (list, tuple, set)):
                vals = [str(x).strip() for x in val if x is not None and str(x).strip() != ""]
                return ", ".join(vals) if vals else _EU_MISSING_PH
            s = str(val).strip()
            return s if s else _EU_MISSING_PH
        except Exception:
            return _EU_MISSING_PH

    def _first_present(row: Dict[str, Any], keys: List[str]) -> Any:
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

    def _format_user_eu(row: Dict[str, Any]) -> str:
        # Map decompose outputs to EU fields with graceful fallbacks
        domain = _norm_str(_first_present(row, [
            "deployment_domain", "domain", "use_domain"
        ]))
        purpose = _norm_str(_first_present(row, [
            "deployment_purpose", "purpose", "goal", "objective"
        ]))
        capability = _norm_str(_first_present(row, [
            "deployment_capability", "capability", "capabilities", "function", "ability"
        ]))
        ai_developer = _norm_str(_first_present(row, [
            "identity_of_ai_developer", "ai_developer", "developer", "vendor", "builder", "provider", "manufacturer"
        ]))
        ai_deployer = _norm_str(_first_present(row, [
            "identity_of_ai_deployer", "deployer", "operator", "implementer", "user", "agency", "organization_deployer"
        ]))
        ai_deployer_location = _norm_str(_first_present(row, [
            "location_of_ai_deployer", "deployer_location", "operator_location", "location_deployer"
        ]))
        ai_subject = _norm_str(_first_present(row, [
            "identity_of_ai_subject", "ai_subject", "subject", "data_subject", "affected_party", "individual", "group"
        ]))
        ai_subject_location = _norm_str(_first_present(row, [
            "location_of_ai_subject", "subject_location", "location_subject", "where"
        ]))
        date_time = _norm_str(_first_present(row, [
            "date_and_time_of_event", "date___time_of_event", "datetime", "date_time", "date", "time", "event_time", "when"
        ]))
        # Attempt to fill extended 9-field template first; fallback to 5-field template; else plain text

        return user_template_eu.format(
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

    def _format_user_risks_benefits(row: Dict[str, Any]) -> str:
        # Map decompose outputs to risks_benefits fields (same 9-field structure as EU)
        domain = _norm_str(_first_present(row, [
            "deployment_domain", "domain", "use_domain"
        ]))
        purpose = _norm_str(_first_present(row, [
            "deployment_purpose", "purpose", "goal", "objective"
        ]))
        capability = _norm_str(_first_present(row, [
            "deployment_capability", "capability", "capabilities", "function", "ability"
        ]))
        ai_developer = _norm_str(_first_present(row, [
            "identity_of_ai_developer", "ai_developer", "developer", "vendor", "builder", "provider", "manufacturer"
        ]))
        ai_deployer = _norm_str(_first_present(row, [
            "identity_of_ai_deployer", "deployer", "operator", "implementer", "user", "agency", "organization_deployer"
        ]))
        ai_deployer_location = _norm_str(_first_present(row, [
            "location_of_ai_deployer", "deployer_location", "operator_location", "location_deployer"
        ]))
        ai_subject = _norm_str(_first_present(row, [
            "identity_of_ai_subject", "ai_subject", "subject", "data_subject", "affected_party", "individual", "group"
        ]))
        ai_subject_location = _norm_str(_first_present(row, [
            "location_of_ai_subject", "subject_location", "location_subject", "where"
        ]))
        date_time = _norm_str(_first_present(row, [
            "date_and_time_of_event", "date___time_of_event", "datetime", "date_time", "date", "time", "event_time", "when"
        ]))

        return user_template_risks_benefits.format(
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


    # Tokenizer-based trimming helpers
    def _get_tokenizer(model_source: str):
        try:
            from transformers import AutoTokenizer  # type: ignore
            return AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)
        except Exception:
            return None

    _TOK_CACHED = None  # lazy per-process tokenizer cache
    def _get_tokenizer_cached(model_source: str):
        nonlocal _TOK_CACHED
        try:
            if _TOK_CACHED is None:
                _TOK_CACHED = _get_tokenizer(model_source)
            return _TOK_CACHED
        except Exception:
            return None

    def _get_max_user_input_tokens(tokenizer, system_text: str) -> int:
        try:
            system_tokens = len(tokenizer.encode(system_text, add_special_tokens=False)) if tokenizer else 0
            max_model_len = int(getattr(cfg.model, "engine_kwargs", {}).get("max_model_len", 4096))
            try:
                sp = getattr(cfg, "sampling_params_classify", getattr(cfg, "sampling_params", {}))
                if hasattr(sp, "max_tokens"):
                    max_output = int(getattr(sp, "max_tokens"))
                else:
                    max_output = int((sp or {}).get("max_tokens", 8))
            except Exception:
                max_output = 8
            safety = 512
            return max(512, max_model_len - max_output - system_tokens - safety)
        except Exception:
            return 2048

    def _trim_text_for_prompt(text: str, tokenizer, system_text: str) -> str:
        # Tokenizer-aware trimming when available; otherwise conservative char-based fallback
        if tokenizer:
            try:
                ids = tokenizer.encode(text or "", add_special_tokens=False)
                max_user = _get_max_user_input_tokens(tokenizer, system_text)
                if len(ids) <= max_user:
                    return text
                ids = ids[:max_user]
                return tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                pass
        # Fallback: approximate 4 chars per token budget
        try:
            max_user = _get_max_user_input_tokens(tokenizer, system_text)
        except Exception:
            max_user = 2048
        approx_chars_per_token = 4
        max_chars = int(max_user) * approx_chars_per_token
        try:
            return text if len(text or "") <= max_chars else str(text or "")[:max_chars]
        except Exception:
            return text

    # Prefer stage/profile-specific sampling params when present; convert nested DictConfig -> dict
    try:
        if is_eu_profile:
            sp_src = getattr(cfg, "sampling_params_eu_ai_act", getattr(cfg, "sampling_params_classify", getattr(cfg, "sampling_params", {})))
        elif is_risks_benefits_profile:
            sp_src = getattr(cfg, "sampling_params_risks_and_benefits", getattr(cfg, "sampling_params_classify", getattr(cfg, "sampling_params", {})))
        else:
            sp_src = getattr(cfg, "sampling_params_classify", getattr(cfg, "sampling_params", {}))
        sampling_params = OmegaConf.to_container(sp_src, resolve=True) if isinstance(sp_src, (dict,)) or hasattr(sp_src, "_get_node") else dict(sp_src)
    except Exception:
        sampling_params = dict(getattr(cfg, "sampling_params", {}))
    # Defaults: EU and risks_benefits need larger output; relevance keeps short YES/NO (or rationales)
    try:
        if is_eu_profile:
            try:
                eu_default = int(getattr(cfg.runtime, "eu_max_tokens_default", 512) or 512)
            except Exception:
                eu_default = 512
            sampling_params.setdefault("max_tokens", eu_default)
        elif is_risks_benefits_profile:
            try:
                rb_default = int(getattr(cfg.runtime, "risks_benefits_max_tokens_default", 2048) or 2048)
            except Exception:
                rb_default = 2048
            sampling_params.setdefault("max_tokens", rb_default)
        else:
            if bool(getattr(cfg.runtime, "log_rationales", False)):
                sampling_params.setdefault("max_tokens", 32)
            else:
                sampling_params.setdefault("max_tokens", 8)
    except Exception:
        sampling_params.setdefault("max_tokens", 8)

    # JSON schema for EU AI Act structured decoding (ensure valid JSON and constrained label)
    EU_GUIDED_JSON_SCHEMA = None
    if is_eu_profile:
        try:
            EU_GUIDED_JSON_SCHEMA = {
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
            EU_GUIDED_JSON_SCHEMA = None

    # JSON schema for Risks and Benefits structured decoding (complex nested structure)
    RISKS_BENEFITS_GUIDED_JSON_SCHEMA = None
    if is_risks_benefits_profile:
        try:
            RISKS_BENEFITS_GUIDED_JSON_SCHEMA = {
                "type": "object",
                "properties": {
                    "Description": {"type": "string"},
                    "Assessment of impact on Human Rights": {
                        "type": "object",
                        "properties": {
                            "Positive Impacts": {"type": "array", "items": {"type": "object"}},
                            "Negative Impacts": {"type": "array", "items": {"type": "object"}},
                        },
                    },
                    "Assessment of impact on Sustainable Development Goals": {
                        "type": "object",
                        "properties": {
                            "Positive Impacts": {"type": "array", "items": {"type": "object"}},
                            "Negative Impacts": {"type": "array", "items": {"type": "object"}},
                        },
                    },
                    "Assessment of additional impacts": {
                        "type": "object",
                        "properties": {
                            "Positive Impacts": {"type": "array", "items": {"type": "object"}},
                            "Negative Impacts": {"type": "array", "items": {"type": "object"}},
                        },
                    },
                },
                "required": ["Description"],
                "additional_properties": False,
            }
        except Exception:
            RISKS_BENEFITS_GUIDED_JSON_SCHEMA = None

    # Columns to drop from final stage outputs (internal vLLM mechanics)
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
    
    # Additional columns to drop for risks_and_benefits profile (keep only structured fields)
    # Note: rb_raw_json is no longer created, so no need to drop it
    _RISKS_BENEFITS_DROP_COLS = []

    def _attach_chunk_text(row: Dict[str, Any]) -> Dict[str, Any]:
        if not enable_kw_buf or kw_regex is None:
            return row
        text_val = row.get("article_text")
        try:
            blocks = _generate_relevant_blocks(text_val, kw_regex, window_words)
        except Exception:
            blocks = []
        if blocks:
            row["chunk_text"] = "\n\n".join(blocks)
        else:
            row.setdefault("chunk_text", str(text_val or ""))
        return row

    # Normalize NA/blank inputs for EU and risks_benefits fields so ChatTemplate sees consistent strings
    def _normalize_na_blanks(r: Dict[str, Any]) -> Dict[str, Any]:
        if not is_eu_profile and not is_risks_benefits_profile:
            # Ensure article_text/chunk_text are strings at minimum
            try:
                r["article_text"] = str(r.get("article_text") or "")
            except Exception:
                pass
            try:
                if r.get("chunk_text") is not None:
                    r["chunk_text"] = str(r.get("chunk_text") or "")
            except Exception:
                pass
            return _coerce_boolish_row(r)

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
                s = _norm_str(v)
                r[k] = s
            except Exception:
                try:
                    r[k] = _EU_MISSING_PH
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
        return _coerce_boolish_row(r)

    # EU input completeness flags
    # Authoritative count of expected EU input fields (date field may appear under either of two names)
    _EU_TOTAL_INPUTS = 9

    def _is_valid_eu_value(v: Any) -> bool:
        try:
            s = _norm_str(v)
            return bool(s) and s != _EU_MISSING_PH
        except Exception:
            return False

    def _add_eu_vague_flags(r: Dict[str, Any]) -> Dict[str, Any]:
        if not is_eu_profile and not is_risks_benefits_profile:
            return r
        cnt = 0
        for k in EU_INPUT_KEYS:
            try:
                if _is_valid_eu_value(r.get(k)):
                    cnt += 1
            except Exception:
                continue
        # If decompose provided a 'missing' column (possibly as a stringified list), use it as a stronger signal
        # Valid count from 'missing' is TOTAL_INPUTS - len(missing)
        try:
            miss_candidates = ("missing", "missing_fields", "eu_missing_fields")
            missing_list = None
            for mk in miss_candidates:
                if mk in r and r.get(mk) is not None:
                    mv = r.get(mk)
                    # Already a list/tuple
                    if isinstance(mv, (list, tuple, set)):
                        missing_list = list(mv)
                        break
                    # Try JSON then AST literal_eval
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
                                # Fallback: naive comma-split
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
                # Clamp to [0, _EU_TOTAL_INPUTS]
                if miss_cnt < 0:
                    miss_cnt = 0
                if miss_cnt > _EU_TOTAL_INPUTS:
                    miss_cnt = _EU_TOTAL_INPUTS
                cnt_from_missing = max(0, _EU_TOTAL_INPUTS - miss_cnt)
                # Use the stricter interpretation: take the minimum valid count
                cnt = min(int(cnt), int(cnt_from_missing))
        except Exception:
            pass
        r["eu_valid_input_count"] = int(cnt)
        r["too_vague_to_process"] = bool(int(cnt) < 5)
        return r

    # Stable baseline for sampling params to avoid Arrow struct field drift across batches
    _SAMPLING_DEFAULTS: Dict[str, Any] = {
        # Core generation controls
        "max_tokens": 8,               # int
        "temperature": 1.0,            # float
        "top_p": 1.0,                  # float
        "top_k": -1,                   # int (-1 disables)
        "min_p": 0.0,                  # float
        # Penalties
        "presence_penalty": 0.0,       # float
        "frequency_penalty": 0.0,      # float
        "repetition_penalty": 1.0,     # float
        "length_penalty": 1.0,         # float (beam search)
        # Decoding strategy
        "use_beam_search": False,      # bool
        "num_beams": 1,                # int
        "early_stopping": False,       # bool
        # Output formatting/logprobs
        "detokenize": True,            # bool
        "logprobs": 0,                 # int
        # Multi-sample
        "n": 1,                        # int
        "best_of": 1,                  # int
        # Stopping conditions
        "stop": [],                    # list[str]
        "stop_token_ids": [],          # list[int]
        # EOS handling
        "ignore_eos_token": False,     # bool
    }

    def _filter_vllm_sampling_params(sp: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sampling params to only those supported by installed vLLM.

        Uses introspection of vllm.SamplingParams signature; if that fails,
        conservatively drops known problematic/optional keys.
        """
        try:
            import inspect as _inspect
            import vllm as _v
            try:
                sig = _inspect.signature(_v.SamplingParams.__init__)
                accepted = {k for k in sig.parameters.keys() if k != "self"}
            except Exception:
                accepted = None
            if accepted:
                filtered = {k: v for k, v in sp.items() if k in accepted}
                if "guided_decoding" in sp and "guided_decoding" not in filtered:
                    filtered["guided_decoding"] = sp["guided_decoding"]
                    if not os.environ.get("RULE_TUPLES_SILENT"):
                        try:
                            print("[classify] Warning: guided_decoding not in SamplingParams signature; preserving manually", flush=True)
                        except Exception:
                            pass
                # Debug print of dropped keys (best-effort)
                try:
                    dropped = [k for k in sp.keys() if k not in filtered]
                    if dropped and not os.environ.get("RULE_TUPLES_SILENT"):
                        print(f"Filtering unsupported vLLM sampling params: {dropped}")
                except Exception:
                    pass
                return filtered
        except Exception:
            pass
        # Conservative fallback: drop keys commonly missing in older vLLM
        sp2 = dict(sp)
        for k in (
            "early_stopping",
            "length_penalty",
            "response_format",
            "structured_output",
        ):
            sp2.pop(k, None)
        # Ensure guided_decoding survives fallback path
        if "guided_decoding" in sp:
            sp2["guided_decoding"] = sp["guided_decoding"]
        return sp2

    # Ensure sampling_params have a stable schema across rows to avoid Arrow concat issues
    def _stabilize_sampling_params(sp: Dict[str, Any]) -> Dict[str, Any]:
        try:
            out = dict(_SAMPLING_DEFAULTS)
            out.update(sp or {})
            stop_val = out.get("stop")
            if stop_val is None:
                out["stop"] = []
            elif not isinstance(stop_val, list):
                out["stop"] = [str(stop_val)]
            for key, value in out.items():
                if isinstance(_SAMPLING_DEFAULTS.get(key), bool) and value is None:
                    out[key] = bool(_SAMPLING_DEFAULTS.get(key))
            # Finally, drop any keys not supported by the installed vLLM
            return _filter_vllm_sampling_params(out)
        except Exception:
            return dict(sp or {})

    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        _maybe_silence_vllm_logs()
        global _DEBUG_GUIDED_LOG
        # Ensure article_id is present on every row
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
        # Normalize any NA/blank values before building messages
        row = _normalize_na_blanks(dict(row))
        # Construct user content and perform token-aware trimming to model context
        if is_eu_profile:
            user_raw = _format_user_eu(row)
        elif is_risks_benefits_profile:
            user_raw = _format_user_risks_benefits(row)
        else:
            user_raw = _format_user(row.get("article_text"), row)
        tok_local = _get_tokenizer_cached(str(getattr(cfg.model, "model_source", "")))
        # First, ensure chunk_text itself is trimmed defensively
        try:
            txt0 = row.get("chunk_text") or row.get("article_text")
            row["chunk_text"] = _trim_text_for_prompt(str(txt0 or ""), tok_local, system_prompt)
        except Exception:
            pass
        user = _trim_text_for_prompt(user_raw, tok_local, system_prompt)
        # Sanitize sampling params; then stabilize to a fixed-key schema
        sp_local = _sanitize_for_json(dict(sampling_params or {}))
        if is_eu_profile and EU_GUIDED_JSON_SCHEMA is not None:
            # Always include guided_decoding payload to keep Arrow schema stable across blocks
            try:
                guided_params = GuidedDecodingParams(
                    json=EU_GUIDED_JSON_SCHEMA,
                    disable_fallback=True,
                    disable_additional_properties=True,
                )
                sp_local["guided_decoding"] = guided_params
                if is_eu_profile and _DEBUG_GUIDED_LOG["pre"] < 5:
                    try:
                        print(
                            "[classify][eu_ai_act] Injected GuidedDecodingParams into sampling params "
                            f"(type={type(guided_params)} keys={list(sp_local.keys())})",
                            flush=True,
                        )
                    except Exception:
                        pass
                    _DEBUG_GUIDED_LOG["pre"] += 1
            except Exception as exc:
                try:
                    print(
                        f"[classify][eu_ai_act] Failed to build GuidedDecodingParams: {exc}",
                        flush=True,
                    )
                except Exception:
                    pass
                try:
                    schema_json_str = json.dumps(EU_GUIDED_JSON_SCHEMA, ensure_ascii=False)
                except Exception:
                    schema_json_str = ""
                sp_local["guided_decoding"] = {"json": schema_json_str or ""}
        elif is_risks_benefits_profile and RISKS_BENEFITS_GUIDED_JSON_SCHEMA is not None:
            # Always include guided_decoding payload to keep Arrow schema stable across blocks
            try:
                guided_params = GuidedDecodingParams(
                    json=RISKS_BENEFITS_GUIDED_JSON_SCHEMA,
                    disable_fallback=True,
                    disable_additional_properties=True,
                )
                sp_local["guided_decoding"] = guided_params
                if is_risks_benefits_profile and _DEBUG_GUIDED_LOG["pre"] < 5:
                    try:
                        print(
                            "[classify][risks_benefits] Injected GuidedDecodingParams into sampling params "
                            f"(type={type(guided_params)} keys={list(sp_local.keys())})",
                            flush=True,
                        )
                    except Exception:
                        pass
                    _DEBUG_GUIDED_LOG["pre"] += 1
            except Exception as exc:
                try:
                    print(
                        f"[classify][risks_benefits] Failed to build GuidedDecodingParams: {exc}",
                        flush=True,
                    )
                except Exception:
                    pass
                try:
                    schema_json_str = json.dumps(RISKS_BENEFITS_GUIDED_JSON_SCHEMA, ensure_ascii=False)
                except Exception:
                    schema_json_str = ""
                sp_local["guided_decoding"] = {"json": schema_json_str or ""}
        # Stabilize schema to avoid Arrow concat errors across batches
        sp_local = _stabilize_sampling_params(sp_local)
        from datetime import datetime as _dt
        # Drop any conflicting keys from input row to avoid clobbering our constructed fields
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
        global _DEBUG_GUIDED_LOG
        ts_end = _dt.now().timestamp()
        usage = row.get("usage") or row.get("token_counts") or None
        if is_risks_benefits_profile:
            raw = str(row.get("generated_text") or "").strip()
            # Robust JSON extraction similar to decompose stage
            parsed = _extract_last_json(raw)
            # Extract main fields and immediately serialize to JSON strings to avoid Arrow schema conflicts
            rb_desc = None
            rb_human_rights = None
            rb_sdgs = None
            rb_additional = None
            if isinstance(parsed, dict):
                rb_desc = parsed.get("Description")
                # Serialize nested structures immediately to JSON strings for Arrow compatibility
                # Variable-length arrays cause schema conflicts, so we convert to strings
                try:
                    hr = parsed.get("Assessment of impact on Human Rights")
                    rb_human_rights = _to_json_str(hr) if hr is not None else None
                except Exception:
                    rb_human_rights = None
                try:
                    sdgs = parsed.get("Assessment of impact on Sustainable Development Goals")
                    rb_sdgs = _to_json_str(sdgs) if sdgs is not None else None
                except Exception:
                    rb_sdgs = None
                try:
                    additional = parsed.get("Assessment of additional impacts")
                    rb_additional = _to_json_str(additional) if additional is not None else None
                except Exception:
                    rb_additional = None
            # Serialize other nested columns for Arrow/Parquet compatibility
            try:
                serialize_nested = bool(getattr(cfg.runtime, "serialize_nested_json", True))
            except Exception:
                serialize_nested = True
            if serialize_nested:
                _serialize_arrow_unfriendly_in_row(row, [
                    "messages",
                    "sampling_params",
                    "usage",
                    "token_counts",
                    "matched_keywords",
                    # Include potential engine outputs that may appear as structs
                    "json",
                    "structured_output",
                    "guided_decoding",
                    "response_format",
                ])
            return {
                **row,
                "rb_desc": rb_desc,
                "rb_human_rights": rb_human_rights,
                "rb_sdgs": rb_sdgs,
                "rb_additional": rb_additional,
                "classification_mode": "risks_and_benefits",
                "too_vague_to_process": False,  # LLM-processed rows are not too vague
                "latency_s": (float(ts_end) - float(row.get("ts_start", ts_end))),
                "token_usage_prompt": ((usage or {}).get("prompt_tokens") or (usage or {}).get("input_tokens")),
                "token_usage_output": ((usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens")),
                "token_usage_total": ((usage or {}).get("total_tokens")),
                "_progress_row": 1,
            }
        elif is_eu_profile:
            raw = str(row.get("generated_text") or "").strip()
            # Robust JSON extraction similar to decompose stage
            parsed = _extract_last_json(raw)
            # Extract fields with robustness to variations
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
                snippet = raw.replace("\n", " ").strip()
                if snippet:
                    snippet = snippet[:400]
                try:
                    print(
                        "[classify][eu_ai_act] Failed to parse LLM JSON output; raw snippet: "
                        f"{snippet}",
                        flush=True,
                    )
                    if _DEBUG_GUIDED_LOG["fail"] < 5:
                        print(
                            "[classify][eu_ai_act][debug] Full generated_text dump due to parse failure:\n"
                            f"{raw}",
                            flush=True,
                    )
                        _DEBUG_GUIDED_LOG["fail"] += 1
                except Exception:
                    pass
            # Optionally serialize nested columns for Arrow/Parquet compatibility
            try:
                serialize_nested = bool(getattr(cfg.runtime, "serialize_nested_json", True))
            except Exception:
                serialize_nested = True
            if serialize_nested:
                _serialize_arrow_unfriendly_in_row(row, [
                    "messages",
                    "sampling_params",
                    "usage",
                    "token_counts",
                    "matched_keywords",
                    # Include potential engine outputs that may appear as structs
                    "json",
                    "structured_output",
                    "guided_decoding",
                    "response_format",
                ])
            # Serialize raw parsed JSON explicitly to string to avoid struct schema drift
            try:
                eu_ai_raw_json_str = _to_json_str(parsed)
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
                "too_vague_to_process": False,  # LLM-processed rows are not too vague
                "latency_s": (float(ts_end) - float(row.get("ts_start", ts_end))),
                "token_usage_prompt": ((usage or {}).get("prompt_tokens") or (usage or {}).get("input_tokens")),
                "token_usage_output": ((usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens")),
                "token_usage_total": ((usage or {}).get("total_tokens")),
                "_progress_row": 1,
            }
        else:
            text = str(row.get("generated_text") or "").strip().upper()
            is_rel = text.startswith("YES") or ("YES" in text and "NO" not in text)
            # Optionally serialize nested columns for Arrow/Parquet compatibility
            try:
                serialize_nested = bool(getattr(cfg.runtime, "serialize_nested_json", True))
            except Exception:
                serialize_nested = True
            if serialize_nested:
                _serialize_arrow_unfriendly_in_row(row, [
                    "messages",
                    "sampling_params",
                    "usage",
                    "token_counts",
                    "matched_keywords",  # List of matched keyword strings
                    # Include potential engine outputs that may appear as structs
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
                # Signal a single-row progress marker for downstream logging
                "_progress_row": 1,
            }

    # Prune columns to required inputs
    allowed_columns = get_required_input_columns(is_eu_profile, is_risks_benefits_profile)
    current_cols = set(out.columns)
    allowed_columns = allowed_columns & current_cols
    allowed_columns.update({col for col in ("article_id", "article_text") if col in current_cols})
    if allowed_columns and allowed_columns != current_cols:
        out = out[[col for col in out.columns if col in allowed_columns]].copy()
        print(f"[classify] Pruned dataset columns to {sorted(allowed_columns)}", flush=True)

    # Keyword pre-gating for relevance profile
    df_all = None  # Track ALL articles (including keyword-filtered ones)
    if prefilter_mode in ("pre_gating", "post_gating"):
        def _add_keyword_info_row(r: Dict[str, Any]) -> Dict[str, Any]:
            has_kw, matched_kws, match_count = _extract_matched_keywords(r.get("article_text"))
            return {
                **r,
                "relevant_keyword": has_kw,
                "matched_keywords": matched_kws,
                "keyword_match_count": match_count,
            }
        out = out.apply(lambda r: _add_keyword_info_row(r.to_dict()), axis=1, result_type="expand")
        if prefilter_mode == "pre_gating":
            df_all = out.copy()
            out = out[out["relevant_keyword"].astype(bool)].copy()

    # EU and risks_benefits: compute input completeness flags and split off too-vague rows
    df_vague = None
    if is_eu_profile or is_risks_benefits_profile:
        try:
            out = out.apply(lambda r: _add_eu_vague_flags(r.to_dict()), axis=1, result_type="expand")
            for col in ["missing", "missing_fields", "eu_missing_fields"]:
                if col in out.columns:
                    out[col] = out[col].apply(
                        lambda v: _to_json_str(v) if isinstance(v, (dict, list, tuple, set)) else v
                    )
            mask_vague = out["too_vague_to_process"].astype(bool)
            df_vague = out[mask_vague].copy()
            out = out[~mask_vague].copy()
            print(f"[classify] Filtered {len(df_vague)} too-vague rows before LLM processing", flush=True)
        except Exception:
            df_vague = None

    # Attach chunk_text and trim for relevance profile
    if enable_kw_buf and kw_regex is not None:
        out = out.apply(lambda r: _attach_chunk_text(r.to_dict()), axis=1, result_type="expand")
    sys_text = system_prompt
    try:
        col_to_trim = "chunk_text" if "chunk_text" in out.columns else "article_text"
        out[col_to_trim] = out[col_to_trim].apply(
            lambda txt: _trim_text_for_prompt(str(txt or ""), None, sys_text)
        )
    except Exception:
        pass

    # Run LLM inference
    out_df = run_vllm_inference(out, cfg, _pre, _post, "classify")
    out_df = out_df.drop(columns=_INTERNAL_DROP_COLS, errors="ignore")
    out_df = out_df.drop(columns=_RISKS_BENEFITS_DROP_COLS, errors="ignore")

    # Merge filtered/vague rows back
    skip_merge_filtered = bool(getattr(cfg.runtime, "skip_merge_filtered", False))
    print(
        f"[classify] Checking merge conditions: df_all={df_all is not None}, df_vague={df_vague is not None}, skip_merge_filtered={skip_merge_filtered}",
        flush=True,
    )
    if skip_merge_filtered:
        if is_eu_profile or is_risks_benefits_profile:
            if "too_vague_to_process" not in out_df.columns:
                out_df["too_vague_to_process"] = False
            else:
                out_df["too_vague_to_process"] = out_df["too_vague_to_process"].fillna(False)
        print(f"[classify] skip_merge_filtered enabled; returning {len(out_df)} rows without merging filtered/vague", flush=True)
        return _prune_result_columns(out_df, is_eu_profile, is_risks_benefits_profile)

    if df_all is not None or (df_vague is not None and (is_eu_profile or is_risks_benefits_profile)):
        print(f"[classify] Starting merge process...", flush=True)
        try:
            df_llm = out_df
            result_parts = [_normalize_profile_columns(df_llm, is_eu_profile, is_risks_benefits_profile)]

            if df_all is not None:
                processed_ids = set(df_llm["article_id"].unique())
                vague_ids: set = set()
                try:
                    if df_vague is not None and "article_id" in df_vague.columns:
                        vague_ids = set(df_vague["article_id"].unique())
                except Exception:
                    pass
                excluded_ids = processed_ids.union(vague_ids)
                df_filtered = df_all[~df_all["article_id"].isin(list(excluded_ids))].copy()
                df_filtered["classification_mode"] = "filtered_by_keyword"
                df_filtered = _normalize_profile_columns(df_filtered, is_eu_profile, is_risks_benefits_profile)
                if "matched_keywords" in df_filtered.columns:
                    df_filtered["matched_keywords"] = df_filtered["matched_keywords"].apply(_to_json_str)
                result_parts.append(df_filtered)

            if df_vague is not None and (is_eu_profile or is_risks_benefits_profile):
                print(f"[classify] Merging back {len(df_vague)} too-vague rows to output", flush=True)
                df_vague["classification_mode"] = "too_vague_to_process"
                df_vague["too_vague_to_process"] = True
                df_vague = _normalize_profile_columns(df_vague, is_eu_profile, is_risks_benefits_profile)
                result_parts.append(df_vague)
            else:
                print(f"[classify] No too-vague rows to merge", flush=True)

            out_df = _merge_result_parts(result_parts)
            if is_eu_profile or is_risks_benefits_profile:
                if "too_vague_to_process" not in out_df.columns:
                    out_df["too_vague_to_process"] = False
                else:
                    out_df["too_vague_to_process"] = out_df["too_vague_to_process"].fillna(False)
            out_df = out_df.drop(columns=_INTERNAL_DROP_COLS, errors="ignore")
            out_df = out_df.drop(columns=_RISKS_BENEFITS_DROP_COLS, errors="ignore")

            try:
                num_kw = len(df_filtered) if "df_filtered" in locals() else 0
            except Exception:
                num_kw = 0
            try:
                num_vague = len(df_vague) if df_vague is not None else 0
            except Exception:
                num_vague = 0
            print(f"[classify] Merged results: {len(df_llm)} LLM-processed + {num_kw} keyword-filtered + {num_vague} too-vague = {len(out_df)} total", flush=True)
        except Exception as e:
            print(f"[classify] ERROR: Failed to merge filtered articles back into results: {e}", flush=True)
            import traceback
            traceback.print_exc()
    else:
        print(f"[classify] No merge needed, returning LLM results directly", flush=True)

    # Stage-scoped logging is handled by the orchestrator
    out_df = _prune_result_columns(out_df, is_eu_profile, is_risks_benefits_profile)
    return out_df

def _normalize_profile_columns(df: pd.DataFrame, is_eu_profile: bool, is_risks_benefits_profile: bool) -> pd.DataFrame:
    df = df.copy()
    if is_eu_profile:
        defaults = {
            "eu_ai_desc": None,
            "eu_ai_label": None,
            "eu_ai_relevant_text": None,
            "eu_ai_reason": None,
            "eu_ai_raw_json": None,
            "too_vague_to_process": False,
            "classification_mode": "filtered_by_keyword",
        }
    elif is_risks_benefits_profile:
        defaults = {
            "rb_desc": None,
            "rb_human_rights": None,
            "rb_sdgs": None,
            "rb_additional": None,
            "too_vague_to_process": False,
            "classification_mode": "filtered_by_keyword",
        }
    else:
        defaults = {
            "is_relevant": False,
            "relevance_answer": None,
            "classification_mode": "filtered_by_keyword",
        }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        elif isinstance(default, bool):
            df[col] = df[col].fillna(default).astype(bool)
        else:
            # Only fill when default is not None; pandas fillna(None) raises
            if default is not None:
                df[col] = df[col].fillna(default)
    if is_risks_benefits_profile:
        for col in ("rb_human_rights", "rb_sdgs", "rb_additional"):
            if col in df.columns:
                df[col] = df[col].apply(lambda v: _to_json_str(v) if not isinstance(v, str) else v)
    return df


def _merge_result_parts(parts: List[pd.DataFrame]) -> pd.DataFrame:
    """Safely merge heterogeneous DataFrame parts without relying on pd.concat.

    Ray/pandas interoperability can yield mixes of numpy-backed and Arrow-backed
    blocks. In those cases, pd.concat may raise dimensionality errors when the
    underlying array managers disagree on ndim. Converting each part to records
    ensures uniform row-wise structures before rebuilding the final DataFrame.
    """

    records: List[Dict[str, Any]] = []
    column_order: List[str] = []

    for part in parts:
        if part is None:
            continue

        if isinstance(part, pd.Series):
            part_df = part.to_frame().T
        elif not isinstance(part, pd.DataFrame):
            try:
                part_df = pd.DataFrame(part)
            except Exception:
                part_df = pd.DataFrame([part])
        else:
            part_df = part

        for col in part_df.columns:
            if col not in column_order:
                column_order.append(col)

        records.extend(part_df.to_dict(orient="records"))

    if not records:
        return pd.DataFrame(columns=column_order)

    merged = pd.DataFrame.from_records(records)
    if column_order:
        merged = merged.reindex(columns=column_order)
    return merged


# Internal reference implementation - kept for reference only, not for external use
# Profile modules should use run_classification_core() instead
_run_classification_stage_impl = run_classification_core
