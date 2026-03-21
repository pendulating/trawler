"""Shared utilities for classification stages."""

from __future__ import annotations

import json
import logging
import os
import re
from bisect import bisect_right
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from vllm.sampling_params import GuidedDecodingParams, SamplingParams  # type: ignore

# Column requirements shared across classification profiles
BASE_INPUT_COLUMNS: Set[str] = {
    "article_id",
    "article_text",
    "article_path",
    "chunk_text",
    "country",
    "year",
    "relevant_keyword",
    "matched_keywords",
    "keyword_match_count",
}

EU_PROFILE_FLAG_COLUMNS: Set[str] = {
    "core_tuple_verified",
    "doc_any_component_verified",
    "missing",
    "missing_fields",
    "eu_missing_fields",
}

EU_INPUT_KEYS: List[str] = [
    "deployment_domain",
    "deployment_purpose",
    "deployment_capability",
    "identity_of_ai_deployer",
    "identity_of_ai_developer",
    "identity_of_ai_subject",
    "location_of_ai_deployer",
    "location_of_ai_subject",
    "date_and_time_of_event",
    "date___time_of_event",
]

RESULT_BASE_COLUMNS: Set[str] = {
    "article_id",
    "article_text",
    "article_path",
    "country",
    "year",
    "classification_mode",
    "too_vague_to_process",
    "latency_s",
    "token_usage_prompt",
    "token_usage_output",
    "token_usage_total",
    "relevant_keyword",
    "matched_keywords",
    "keyword_match_count",
    "core_tuple_verified",
    "doc_any_component_verified",
    "eu_valid_input_count",
}

EU_OUTPUT_COLUMNS: Set[str] = {
    "eu_ai_desc",
    "eu_ai_label",
    "eu_ai_relevant_text",
    "eu_ai_reason",
    "eu_ai_raw_json",
}

RISKS_BENEFITS_OUTPUT_COLUMNS: Set[str] = {
    "rb_desc",
    "rb_human_rights",
    "rb_sdgs",
    "rb_additional",
}

RELEVANCE_OUTPUT_COLUMNS: Set[str] = {
    "relevance_answer",
    "is_relevant",
    "llm_output",
}


def get_required_input_columns(
    is_eu_profile: bool, is_risks_benefits_profile: bool
) -> Set[str]:
    """Return the minimal column set needed for downstream classification."""
    cols: Set[str] = set(BASE_INPUT_COLUMNS)
    if is_eu_profile or is_risks_benefits_profile:
        cols.update(EU_INPUT_KEYS)
        cols.update(EU_PROFILE_FLAG_COLUMNS)
    return cols


def get_allowed_result_columns(
    is_eu_profile: bool, is_risks_benefits_profile: bool
) -> Set[str]:
    """Return allowed result columns for a given profile."""
    cols: Set[str] = set(RESULT_BASE_COLUMNS)
    if is_eu_profile:
        cols.update(EU_OUTPUT_COLUMNS)
    elif is_risks_benefits_profile:
        cols.update(RISKS_BENEFITS_OUTPUT_COLUMNS)
    else:
        cols.update(RELEVANCE_OUTPUT_COLUMNS)
    return cols


def prune_result_columns(
    df: pd.DataFrame, is_eu_profile: bool, is_risks_benefits_profile: bool
) -> pd.DataFrame:
    """Prune DataFrame to only allowed columns for the given profile."""
    try:
        allowed = get_allowed_result_columns(is_eu_profile, is_risks_benefits_profile)
        existing = [c for c in df.columns if c in allowed]
        if existing:
            return df[existing].copy()
    except Exception:
        pass
    return df


def ensure_profile(cfg, profile: str) -> None:
    """Ensure the classification profile is set in config."""
    try:
        OmegaConf.update(cfg, "runtime.classification_profile", str(profile), merge=True)
        # Prefer LLM for EU/RB; relevance can be heuristic or LLM via cfg
        if profile in ("eu_ai_act", "risks_and_benefits"):
            OmegaConf.update(cfg, "runtime.use_llm_classify", True, merge=True)
    except Exception:
        pass


def inject_prompt_from_file(cfg, prompt_filename: str) -> None:
    """Inject prompt from YAML file into cfg.prompt.

    Supports subdirectory paths like 'general_ai/classify.yaml' or just 'classify.yaml'.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))  # dagspaces/uair
        # Support subdirectory paths (e.g., 'general_ai/classify.yaml')
        prompt_path = os.path.join(base_dir, "conf", "prompt", prompt_filename)
        if os.path.exists(prompt_path):
            prompt_cfg = OmegaConf.load(prompt_path)
            sys_p = prompt_cfg.get("system_prompt")
            usr_p = prompt_cfg.get("prompt_template")
            if sys_p:
                OmegaConf.update(cfg, "prompt.system_prompt", sys_p, merge=True)
            if usr_p:
                OmegaConf.update(cfg, "prompt.prompt_template", usr_p, merge=True)
    except Exception:
        pass


def dedupe_by_article_id(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate DataFrame by article_id, keeping first occurrence."""
    try:
        if "article_id" in df.columns and len(df):
            return df.sort_values(by=["article_id"]).drop_duplicates(subset=["article_id"], keep="first")
    except Exception:
        pass
    return df


# Shared utility functions
_VLLM_LOGS_SILENCED = False


def maybe_silence_vllm_logs() -> None:
    """Silence verbose vLLM logs."""
    global _VLLM_LOGS_SILENCED
    if _VLLM_LOGS_SILENCED:
        return
    try:
        from dagspaces.uair.logging_filters import PatternModuloFilter  # local import for worker
        lg = logging.getLogger("vllm")
        try:
            n = int(os.environ.get("UAIR_VLLM_LOG_EVERY", "10") or "10")
        except Exception:
            n = 10
        lg.setLevel(logging.INFO)
        try:
            existing_filters = getattr(lg, "filters", [])
            if not any(getattr(f, "__class__", object).__name__ == "PatternModuloFilter" for f in existing_filters):
                lg.addFilter(PatternModuloFilter(mod=n, pattern="Elapsed time for batch"))
        except Exception:
            pass
        if os.environ.get("RULE_TUPLES_SILENT"):
            lg.setLevel(logging.ERROR)
        _VLLM_LOGS_SILENCED = True
    except Exception:
        pass


def to_json_str(value: Any) -> Any:
    """Serialize Python objects to JSON string for Arrow/Parquet friendliness."""
    try:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        try:
            return str(value)
        except Exception:
            return None


def serialize_arrow_unfriendly_in_row(row: Dict[str, Any], columns: List[str]) -> None:
    """In-place convert nested/dict/list columns to JSON strings in a row dict."""
    for col in columns:
        if col in row:
            val = row.get(col)
            if isinstance(val, (dict, list, tuple)):
                row[col] = to_json_str(val)
            elif isinstance(val, GuidedDecodingParams):
                row[col] = str(val)
            elif isinstance(val, SamplingParams):
                row[col] = str(val)


def sanitize_for_json(value: Any) -> Any:
    """Recursively convert value to JSON-serializable builtins."""
    try:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        # Handle GuidedDecodingParams and SamplingParams objects explicitly
        if isinstance(value, GuidedDecodingParams):
            # Convert to dict format that vLLM can use
            # NOTE: Backend selection is done at engine initialization, not request level
            try:
                return {
                    "json": getattr(value, "json", None),
                    "disable_fallback": getattr(value, "disable_fallback", True),
                    "disable_additional_properties": getattr(value, "disable_additional_properties", True),
                }
            except Exception:
                return str(value)
        if isinstance(value, SamplingParams):
            return str(value)
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                try:
                    key = str(k)
                except Exception:
                    key = repr(k)
                out[key] = sanitize_for_json(v)
            return out
        if isinstance(value, (list, tuple, set)):
            return [sanitize_for_json(v) for v in value]
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                return sanitize_for_json(tolist())
            except Exception:
                pass
        return str(value)
    except Exception:
        return None


_BOOLISH_SUFFIXES = ("_verified",)
_BOOLISH_EXACT = {
    "core_tuple_verified",
    "doc_any_component_verified",
    "ver_tuples_any_verified",
    "ver_tuple_overall_pass",
    "relevant_keyword",
    "too_vague_to_process",
    "is_relevant",
}


def coerce_bool_like(value: Any) -> bool:
    """Coerce a value to boolean."""
    try:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except Exception:
            pass
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                return coerce_bool_like(value.item())
            except Exception:
                pass
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(int(value))
        if isinstance(value, str):
            lv = value.strip().lower()
            if lv in {"", "none", "null", "nan"}:
                return False
            if lv in {"true", "1", "yes", "y", "t"}:
                return True
            if lv in {"false", "0", "no", "n", "f"}:
                return False
        return bool(value)
    except Exception:
        return False


def coerce_boolish_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce boolean-like columns in a row dict."""
    for key in list(r.keys()):
        if key in _BOOLISH_EXACT or key.endswith(_BOOLISH_SUFFIXES):
            try:
                r[key] = coerce_bool_like(r.get(key))
            except Exception:
                r[key] = False
        else:
            val = r.get(key)
            if isinstance(val, np.ndarray):
                try:
                    val = val.tolist()
                except Exception:
                    val = list(val)
                r[key] = val
            elif hasattr(val, "tolist") and not isinstance(val, (str, bytes)):
                try:
                    candidate = val.tolist()
                    r[key] = candidate
                except Exception:
                    pass
    return r


def coerce_boolish_df(pdf: pd.DataFrame) -> pd.DataFrame:
    """Coerce boolean-like columns in a DataFrame."""
    if not isinstance(pdf, pd.DataFrame):
        return pdf
    cols = list(pdf.columns)
    for col in cols:
        if col in _BOOLISH_EXACT or col.endswith(_BOOLISH_SUFFIXES):
            try:
                pdf[col] = pdf[col].apply(coerce_bool_like)
            except Exception:
                try:
                    pdf[col] = pdf[col].fillna(False).astype(bool)
                except Exception:
                    pass
        else:
            try:
                sample = next(
                    (v for v in pdf[col].values if v is not None and not (isinstance(v, float) and pd.isna(v))),
                    None,
                )
            except Exception:
                sample = None
            if isinstance(sample, np.ndarray) or (
                hasattr(sample, "tolist") and not isinstance(sample, (str, bytes))
            ):
                try:
                    pdf[col] = pdf[col].apply(
                        lambda v: v.tolist() if isinstance(v, np.ndarray) else (
                            v.tolist() if (hasattr(v, "tolist") and not isinstance(v, (str, bytes))) else v
                        )
                    )
                except Exception:
                    pass
    return pdf


def build_relevant_regex() -> re.Pattern:
    """Build regex pattern for AI-related keywords."""
    phrases = [
        # Core AI Technologies & Acronyms
        r"\bai\b", r"\bml\b", r"\bnlp\b", r"\bllm\b", r"\bagi\b", r"\bxai\b", r"\biot\b",
        r"artificial\s+intelligence", r"machine\s+learning", r"neural\s+network",
        r"large\s+language\s+model", r"transformer", r"chatgpt|gpt-\d+|gpt-",
        r"openai|anthropic|claude|gemini|qwen", r"fine-?tuning|inference|prompt(ing)?|agent(s)?",
        # Journalist-Friendly AI Terms
        r"computer", r"computers", r"software", r"program", r"programs", r"programming",
        r"coded", r"coding", r"app", r"apps", r"application", r"applications", r"tool", r"tools",
        r"technology", r"tech", r"innovation", r"innovations", r"breakthrough", r"breakthroughs",
        r"advancement", r"advancements",
        # Anthropomorphic/Accessible Descriptions
        r"robot", r"robots", r"robotic", r"bot", r"bots", r"chatbot", r"chatbots",
        r"virtual\s+assistant", r"digital\s+assistant", r"smart\s+assistant",
        r"machine", r"machines", r"device", r"devices", r"smart\s+device",
        r"intelligent\s+system", r"thinking\s+machine", r"electronic\s+brain",
        r"digital\s+brain", r"computer\s+brain",
        # Process-Oriented Terms
        r"automated", r"automation", r"automatic", r"automatically", r"self-?learning",
        r"self-?teaching", r"self-?improving", r"adaptive", r"smart", r"intelligent",
        r"cognitive", r"thinking", r"reasoning", r"decision-?making", r"processing",
        r"analyzing", r"analysis", r"pattern\s+recognition", r"image\s+recognition",
        r"voice\s+recognition", r"language\s+processing",
        # Capability Descriptions
        r"learns", r"learning", r"teaches\s+itself", r"trains", r"training", r"trained",
        r"understands", r"understanding", r"recognizes", r"recognition", r"identifies",
        r"identification", r"predicts", r"prediction", r"predictions", r"forecasts",
        r"forecasting", r"generates", r"generation", r"creates", r"creation", r"produces",
        r"mimics", r"simulates", r"replicates",
        # Business/Industry Terms
        r"silicon\s+valley", r"tech\s+company", r"tech\s+companies", r"tech\s+giant",
        r"tech\s+giants", r"startup", r"startups", r"big\s+tech", r"platform", r"platforms",
        r"service", r"services", r"product", r"products", r"solution", r"solutions",
        r"ecosystem", r"infrastructure",
        # Buzzword/Hype Terms
        r"revolutionary", r"game-?changing", r"cutting-?edge", r"state-?of-?the-?art",
        r"next-?generation", r"futuristic", r"advanced", r"sophisticated", r"powerful",
        r"disruptive", r"transformative", r"groundbreaking", r"innovative", r"emerging",
        r"novel", r"pioneering",
        # Comparison/Analogy Terms
        r"human-?like", r"human-?level", r"superhuman", r"brain-?like", r"mimicking\s+humans?",
        r"replacing\s+humans?", r"outsmarting\s+humans?", r"beating\s+humans?",
        r"surpassing\s+humans?", r"artificial\s+brain", r"electronic\s+mind",
        r"digital\s+worker", r"virtual\s+employee",
        # AI Safety & Governance
        r"safety", r"alignment", r"governance", r"responsible", r"trustworthy", r"ethics",
        r"ethical", r"bias", r"biased", r"fairness", r"explainable", r"transparency",
        r"transparent", r"accountability", r"accountable", r"regulation", r"oversight",
        r"compliance",
        # Risk & Harm Terms
        r"risk", r"risks", r"harm", r"harms", r"harmful", r"danger", r"dangerous",
        r"threat", r"threats", r"vulnerability", r"vulnerabilities", r"attack", r"attacks",
        r"exploitation", r"manipulation", r"weaponization",
        # Societal Impact Terms
        r"deepfake", r"misinformation", r"disinformation", r"fake\s+news", r"discrimination",
        r"discriminatory", r"surveillance", r"capitalism", r"privacy", r"invasion",
        r"facial\s+recognition", r"predictive\s+policing", r"social\s+credit",
        r"filter\s+bubble", r"echo\s+chamber", r"polarization", r"radicalization",
        # Economic & Labor
        r"displacement", r"unemployment", r"automation", r"automated", r"replacement",
        r"disruption", r"workforce", r"labor", r"labour", r"jobs", r"employment",
        r"gig\s+economy", r"platform\s+workers?", r"algorithmic\s+management",
        # AI Systems & Applications
        r"algorithm", r"algorithms", r"algorithmic", r"autonomous", r"self-?driving",
        r"recommendation", r"moderation", r"computer\s+vision", r"speech\s+recognition",
        r"sentiment\s+analysis", r"predictive\s+analytics", r"decision\s+support",
        r"synthetic\s+media", r"voice\s+cloning",
        # Major Tech Companies
        r"microsoft", r"google", r"amazon", r"meta", r"facebook", r"nvidia", r"intel",
        r"apple", r"tesla", r"deepmind", r"hugging\s?face", r"stability\s+ai",
        r"midjourney", r"dall-?e", r"baidu", r"alibaba", r"tencent", r"bytedance",
        # Regulatory & Legal
        r"gdpr", r"ccpa", r"regulation", r"regulatory", r"policy", r"policies",
        r"governance", r"antitrust", r"monopoly", r"section\s+230", r"digital\s+rights",
        r"data\s+protection", r"impact\s+assessment",
        # Social & Cultural Harms
        r"manipulation", r"identity\s+theft", r"cyberbullying", r"harassment",
        r"hate\s+speech", r"democratic", r"democracy", r"election", r"elections",
        r"voting", r"interference",
        # Technical Risks
        r"adversarial", r"poisoning", r"backdoor", r"inference", r"inversion",
        r"leakage", r"robustness", r"brittleness", r"hallucination", r"confabulation",
        r"distribution\s+shift", r"out-?of-?distribution",
        # Long-term & Existential
        r"superintelligence", r"existential", r"x-?risk", r"control\s+problem",
        r"value\s+alignment", r"mesa-?optimization", r"instrumental\s+convergence",
        r"orthogonality",
        # Domain Applications
        r"city", r"cities", r"urban", r"climate", r"earth", r"environment",
        r"environmental", r"transport", r"transportation", r"smart\s+grid",
        r"infrastructure", r"connected\s+device", r"monitoring", r"carbon", r"energy",
        r"sustainable", r"sustainability", r"green",
        # Healthcare
        r"medical", r"healthcare", r"health", r"diagnostic", r"clinical",
        r"telemedicine", r"therapeutics", r"equity",
        # Financial
        r"trading", r"advisor", r"credit", r"scoring", r"financial", r"inclusion",
        r"redlining", r"lending", r"predatory",
        # Education
        r"education", r"educational", r"edtech", r"learning", r"student", r"academic",
        r"integrity", r"cheating", r"proctoring",
        # Criminal Justice
        r"criminal", r"justice", r"recidivism", r"bail", r"sentencing", r"policing",
        r"enforcement",
        # Emerging Tech
        r"quantum", r"neuromorphic", r"edge\s+computing", r"distributed", r"swarm",
        r"multi-?agent",
        # General Tech Terms
        r"technology", r"technological", r"digital", r"cyber", r"online", r"internet",
        r"platform", r"platforms", r"data", r"dataset", r"datasets", r"model", r"models",
        r"system", r"systems",
    ]
    pattern = r"(" + r"|".join(phrases) + r")"
    return re.compile(pattern, flags=re.IGNORECASE)


def generate_relevant_blocks(text: str, compiled_regex: re.Pattern, window_words: int = 100) -> List[str]:
    """Generate relevant text blocks around keyword matches."""
    if not isinstance(text, str) or not text:
        return []
    token_matches = list(re.finditer(r"\S+", text))
    if not token_matches:
        return []
    token_starts = [m.start() for m in token_matches]
    intervals: List[List[int]] = []
    for m in compiled_regex.finditer(text):
        idx = max(0, min(len(token_starts) - 1, bisect_right(token_starts, m.start()) - 1))
        start_token = max(0, idx - window_words)
        end_token = min(len(token_matches) - 1, idx + window_words)
        start_char = token_matches[start_token].start()
        end_char = token_matches[end_token].end()
        intervals.append([start_char, end_char])
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged: List[List[int]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [text[s:e] for s, e in merged]


def extract_last_json(text: str) -> Dict[str, Any] | None:
    """Extract the last JSON object from text, if any."""
    try:
        if not isinstance(text, str) or not text.strip():
            return None
        s = text
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            import re as _re
            snippets = _re.findall(r"\{[\s\S]*\}", s)
            for snip in reversed(snippets or []):
                try:
                    obj = json.loads(snip)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    continue
        except Exception:
            pass
        return None
    except Exception:
        return None


def merge_result_parts(parts: List[pd.DataFrame]) -> pd.DataFrame:
    """Safely merge heterogeneous DataFrame parts."""
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


def normalize_profile_columns(df: pd.DataFrame, is_eu_profile: bool, is_risks_benefits_profile: bool) -> pd.DataFrame:
    """Normalize DataFrame columns for a given profile."""
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
            if default is not None:
                df[col] = df[col].fillna(default)
    if is_risks_benefits_profile:
        for col in ("rb_human_rights", "rb_sdgs", "rb_additional"):
            if col in df.columns:
                df[col] = df[col].apply(lambda v: to_json_str(v) if not isinstance(v, str) else v)
    return df
