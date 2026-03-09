from typing import Any, Dict, List
import os
import json
import logging
import re
from bisect import bisect_right
import pandas as pd
from omegaconf import OmegaConf
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

from dagspaces.common.vllm_inference import run_vllm_inference

_VLLM_LOGS_SILENCED = False

def _maybe_silence_vllm_logs() -> None:
    global _VLLM_LOGS_SILENCED
    if _VLLM_LOGS_SILENCED:
        return
    try:
        from dagspaces.uair.logging_filters import PatternModuloFilter
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



def _build_relevant_regex() -> re.Pattern:
    phrases = [
        r"artificial\s+intelligence",
        r"machine\s+learning",
        r"neural\s+network",
        r"large\s+language\s+model",
        r"transformer",
        r"chatgpt|gpt-\d+|gpt-",
        r"openai|anthropic|claude|gemini|qwen",
        r"fine-?tuning|inference|prompt(ing)?|agent(s)?",
        r"(?:city|cities)",
        r"urban",
        r"climate",
        r"earth",
        r"environment",
        r"transport",
    ]
    return re.compile(r"(" + r"|".join(phrases) + r")", flags=re.IGNORECASE)


def _generate_relevant_blocks(text: str, compiled_regex: re.Pattern, window_words: int = 100) -> List[str]:
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




def _get_tokenizer(model_source: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
        return AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)
    except Exception:
        return None


def _get_max_user_input_tokens(tokenizer, system_text: str, cfg) -> int:
    try:
        system_tokens = len(tokenizer.encode(system_text, add_special_tokens=False)) if tokenizer else 0
        max_model_len = int(getattr(cfg.model, "engine_kwargs", {}).get("max_model_len", 4096))
        try:
            sp = getattr(cfg, "sampling_params_taxonomy", getattr(cfg, "sampling_params", {}))
            if hasattr(sp, "max_tokens"):
                max_output = int(getattr(sp, "max_tokens"))
            else:
                max_output = int((sp or {}).get("max_tokens", 16))
        except Exception:
            max_output = 16
        safety = 512
        return max(512, max_model_len - max_output - system_tokens - safety)
    except Exception:
        return 2048


def _trim_text_for_prompt(text: str, tokenizer, system_text: str, cfg) -> str:
    # Tokenizer-aware trimming when available; otherwise conservative char-based fallback
    if tokenizer:
        try:
            ids = tokenizer.encode(text or "", add_special_tokens=False)
            max_user = _get_max_user_input_tokens(tokenizer, system_text, cfg)
            if len(ids) <= max_user:
                return text
            ids = ids[:max_user]
            return tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            pass
    try:
        max_user = _get_max_user_input_tokens(tokenizer, system_text, cfg)
    except Exception:
        max_user = 2048
    approx_chars_per_token = 4
    max_chars = int(max_user) * approx_chars_per_token
    try:
        return text if len(text or "") <= max_chars else str(text or "")[:max_chars]
    except Exception:
        return text


## NOTE: parsing helper is defined inside run_taxonomy_stage where mappings exist


def run_taxonomy_stage(df, cfg):
    # Convert to pandas if needed
    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["article_id", "article_path", "chunk_text", "answer", "chunk_label"])
    out = df.copy()

    # Controls
    try:
        gate_on_rel = bool(getattr(cfg.runtime, "gate_on_relevance", True))
    except Exception:
        gate_on_rel = True
    try:
        prefilter_mode = str(getattr(cfg.runtime, "prefilter_mode", "pre_gating")).strip().lower()
    except Exception:
        prefilter_mode = "pre_gating"
    try:
        enable_kw_buf = bool(getattr(cfg.runtime, "keyword_buffering", True))
    except Exception:
        enable_kw_buf = True
    try:
        window_words = int(getattr(cfg.runtime, "keyword_window_words", 100) or 100)
    except Exception:
        window_words = 100
    kw_regex = _build_relevant_regex() if enable_kw_buf else None

    # Prompt content
    taxonomy_path = str(getattr(cfg, "taxonomy_json", "") or "")
    try:
        taxonomy = {}
        if taxonomy_path and os.path.exists(taxonomy_path):
            if taxonomy_path.endswith((".yaml", ".yml")) and yaml is not None:
                with open(taxonomy_path, "r") as f:
                    data = yaml.safe_load(f)
            else:
                with open(taxonomy_path, "r") as f:
                    data = json.load(f)
            taxonomy = data.get("taxonomy", data) if isinstance(data, (dict,)) else {}
    except Exception:
        taxonomy = {}
    taxonomy_str = "\n".join(["\n".join(subcats) for _, subcats in (taxonomy.items() if isinstance(taxonomy, dict) else [])])
    taxonomy_str = "\n".join([f"{line}" for line in taxonomy_str.split("\n") if line.strip()])
    # Flatten category names for guided decoding and parsing
    try:
        category_names: List[str] = []
        if isinstance(taxonomy, dict):
            for _, subcats in taxonomy.items():
                if isinstance(subcats, list):
                    for s in subcats:
                        if isinstance(s, str) and s.strip():
                            category_names.append(s.strip())
    except Exception:
        category_names = []
    # Build normalization helpers for mapping back names -> indices
    def _norm(s: str) -> str:
        try:
            return re.sub(r"\s+", " ", str(s or "").strip().lower())
        except Exception:
            return str(s or "").strip().lower()
    category_index_map: Dict[str, int] = {name: (idx + 1) for idx, name in enumerate(category_names)}
    norm_to_name: Dict[str, str] = {_norm(name): name for name in category_names}

    system_template = str(getattr(cfg.prompt_taxonomy, "system_template", ""))
    user_template = str(getattr(cfg.prompt_taxonomy, "user_template", ""))
    system_prompt = system_template.format(taxonomy_str=taxonomy_str)

    def _format_user(row: Dict[str, Any]) -> str:
        return user_template.format(
            article_id=row.get("article_id"),
            chunk_id=row.get("chunk_id", 0),
            num_chunks=row.get("num_chunks", 1),
            chunk_text=row.get("chunk_text", ""),
        )


    # Sampling params
    try:
        sp_src = getattr(cfg, "sampling_params_taxonomy", getattr(cfg, "sampling_params", {}))
        sampling_params = OmegaConf.to_container(sp_src, resolve=True) if isinstance(sp_src, (dict,)) or hasattr(sp_src, "_get_node") else dict(sp_src)
    except Exception:
        sampling_params = dict(getattr(cfg, "sampling_params", {}))
    # Enforce guided decoding choice set to category names + 'None'
    try:
        enforce_guided = bool(getattr(cfg.runtime, "guided_decoding_taxonomy", True))
    except Exception:
        enforce_guided = True
    if enforce_guided and category_names:
        try:
            choices = [*category_names, "None"]
            sampling_params["guided_decoding"] = {"choice": choices}
        except Exception:
            pass

    # Helpers
    def _ensure_ids(row: Dict[str, Any]) -> Dict[str, Any]:
        import hashlib as _hash
        art_id = row.get("article_id")
        if not (isinstance(art_id, str) and art_id):
            p = row.get("article_path", "")
            if isinstance(p, str) and p:
                art_id = _hash.sha1(p.encode("utf-8")).hexdigest()
            else:
                txt = str(row.get("chunk_text") or row.get("article_text") or "")
                art_id = _hash.sha1(txt.encode("utf-8")).hexdigest()
            row["article_id"] = art_id
        row.setdefault("chunk_id", 0)
        row.setdefault("num_chunks", 1)
        return row

    def _kw_flag(text: Any) -> bool:
        if kw_regex is None:
            return True
        try:
            return bool(kw_regex.search(str(text or "")))
        except Exception:
            return True

    def _attach_chunk_text(row: Dict[str, Any]) -> Dict[str, Any]:
        if row.get("chunk_text"):
            return row
        if not enable_kw_buf or kw_regex is None:
            row["chunk_text"] = str(row.get("article_text") or "")
            return row
        try:
            blocks = _generate_relevant_blocks(row.get("article_text"), kw_regex, int(window_words))
        except Exception:
            blocks = []
        row["chunk_text"] = ("\n\n".join(blocks) if blocks else str(row.get("article_text") or ""))
        return row

    _TOK_CACHED = None
    def _get_tokenizer_cached():
        nonlocal _TOK_CACHED
        try:
            if _TOK_CACHED is None:
                _TOK_CACHED = _get_tokenizer(str(getattr(cfg.model, "model_source", "")))
            return _TOK_CACHED
        except Exception:
            return None

    def _trim_row(row: Dict[str, Any]) -> Dict[str, Any]:
        txt = row.get("chunk_text") or row.get("article_text")
        row["chunk_text"] = _trim_text_for_prompt(str(txt or ""), _get_tokenizer_cached(), system_prompt, cfg)
        return row

    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        _maybe_silence_vllm_logs()
        row = _ensure_ids(dict(row))
        row = _attach_chunk_text(row)
        row = _trim_row(row)
        # Conservative char clamp as a last-resort safety net
        try:
            try:
                mm_len = int(getattr(cfg.model, "engine_kwargs", {}).get("max_model_len", 4096))
            except Exception:
                mm_len = 4096
            try:
                sp = dict(getattr(cfg, "sampling_params_taxonomy", getattr(cfg, "sampling_params", {})))
            except Exception:
                sp = {}
            max_out = int(sp.get("max_tokens", 16) or 16)
            approx_chars = max(2048, (mm_len - max_out - 512) * 4)
        except Exception:
            approx_chars = 4096 * 4
        # Remove any conflicting heavy columns carried over from prior stages/parquet
        try:
            for k in ("messages", "sampling_params", "usage", "token_counts"):
                if k in row:
                    row.pop(k, None)
        except Exception:
            pass
        # Ensure our constructed messages take precedence
        out = {
            **row,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _trim_text_for_prompt(_format_user(row), _get_tokenizer_cached(), system_prompt, cfg)[:int(approx_chars)]},
            ],
            "sampling_params": sampling_params,
        }
        return out

    def _post(row: Dict[str, Any]) -> Dict[str, Any]:
        return {**row, "answer": row.get("generated_text")}

    # Define parser with closure over category maps
    def _parse_answer_name_index(answer_text: str) -> Dict[str, Any]:
        if not isinstance(answer_text, str) or not answer_text.strip():
            return {"name": None, "index": None}
        if re.search(r"\bnone\b", answer_text, flags=re.IGNORECASE):
            return {"name": None, "index": None}
        nrm = _norm(answer_text)
        nm = norm_to_name.get(nrm)
        if nm:
            return {"name": nm, "index": category_index_map.get(nm)}
        # fallback: number
        try:
            m = re.search(r"\b(\d{1,6})\b", answer_text)
            if m:
                v = int(m.group(1))
                names_by_idx = sorted(category_index_map.items(), key=lambda kv: kv[1])
                if 1 <= v <= len(names_by_idx):
                    return {"name": names_by_idx[v-1][0], "index": v}
        except Exception:
            pass
        return {"name": None, "index": None}

    # Gate on is_relevant when present
    try:
        if gate_on_rel and "is_relevant" in out.columns:
            out = out[out["is_relevant"].astype(bool) == True]
    except Exception:
        pass
    try:
        fake_llm = bool(getattr(cfg.runtime, "fake_llm", False))
    except Exception:
        fake_llm = False
    if prefilter_mode in ("pre_gating", "post_gating"):
        try:
            out["relevant_keyword"] = out.apply(lambda rr: _kw_flag(rr.get("chunk_text") or rr.get("article_text")), axis=1)
        except Exception:
            out["relevant_keyword"] = True
        if prefilter_mode == "pre_gating":
            out = out[out["relevant_keyword"] == True]
    if fake_llm:
        import hashlib as _h
        def _fake_answer_row(row: Dict[str, Any]) -> Dict[str, Any]:
            row = _ensure_ids(_trim_row(_attach_chunk_text(dict(row))))
            txt = str(row.get("chunk_text", ""))
            h = int(_h.sha1(txt.encode("utf-8")).hexdigest(), 16)
            row["answer"] = str((h % 5) + 1)
            return row
        out_df = pd.DataFrame([_fake_answer_row(r) for r in out.to_dict("records")])
    else:
        out_df = run_vllm_inference(out, cfg, _pre, _post, "taxonomy")
    if len(out_df):
        try:
            def _apply_label(s: Any) -> Dict[str, Any]:
                return _parse_answer_name_index(str(s))
            res_series = out_df["answer"].apply(_apply_label)
            try:
                out_df["chunk_label_name"] = res_series.apply(lambda d: (d.get("name") if isinstance(d, dict) else None) or "None")
            except Exception:
                pass
            try:
                out_df["chunk_label"] = res_series.apply(lambda d: (str(d.get("index")) if isinstance(d, dict) and isinstance(d.get("index"), int) else "None"))
            except Exception:
                pass
        except Exception:
            pass
    # Stage-scoped logging is handled by the orchestrator
    return out_df


