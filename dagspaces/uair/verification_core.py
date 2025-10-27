import re
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch  # type: ignore
    from torch.nn import functional as F  # type: ignore
except Exception:  # pragma: no cover - torch is required at runtime
    torch = None  # type: ignore
    F = None  # type: ignore

try:
    from transformers import (  # type: ignore
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
    )
except Exception:  # pragma: no cover - transformers is required at runtime
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore


# ------------------------------
# Global caches (initialized via init_verification)
# ------------------------------
_EMBED_TOKENIZER = None
_EMBED_MODEL = None
_NLI_TOKENIZER = None
_NLI_MODEL = None
_DEVICE: Optional[str] = None
_CLAIM_MAP: Dict[str, str] = {}
_DEBUG: bool = False
_LAST_EMBED_ERROR: Optional[str] = None
_LAST_NLI_ERROR: Optional[str] = None


@dataclass
class VerificationConfig:
    method: str = "combo"  # one of: off|embed|nli|combo|combo_judge
    top_k: int = 3
    sim_threshold: float = 0.55
    entail_threshold: float = 0.85
    contra_max: float = 0.05
    embed_model_name: str = "intfloat/multilingual-e5-base"
    nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    device: Optional[str] = None


_CONFIG = VerificationConfig()


def _ensure_imports():
    if torch is None or AutoTokenizer is None or AutoModel is None:
        raise RuntimeError("verification requires torch and transformers to be installed")


def _detect_device(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    try:
        import torch as _t  # type: ignore
        return "cuda" if _t.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_thresholds_string(s: str) -> Tuple[float, float, float]:
    """Parse a thresholds string like 'sim=0.55,ent=0.85,contra=0.05'."""
    sim = 0.55
    ent = 0.85
    contra = 0.05
    try:
        parts = [p.strip() for p in str(s).split(",") if p.strip()]
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                k = k.strip().lower()
                v = float(v.strip())
                if k == "sim":
                    sim = v
                elif k == "ent":
                    ent = v
                elif k == "contra":
                    contra = v
    except Exception:
        pass
    return sim, ent, contra


def _build_label_to_text_map(taxonomy: Dict[str, List[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    idx = 1
    # Preserve insertion order of taxonomy JSON
    for _category, subcats in taxonomy.items():
        for sub in subcats:
            mapping[str(idx)] = str(sub)
            idx += 1
    return mapping


def _mean_pool(last_hidden_state: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_embeddings(texts: List[str], is_query: bool) -> np.ndarray:
    assert _EMBED_MODEL is not None and _EMBED_TOKENIZER is not None and _DEVICE is not None
    # E5-style prefix improves performance for E5 models; harmless for others
    prefix = "query: " if is_query else "passage: "
    proc_texts = [(prefix + t) for t in texts]
    with torch.inference_mode():
        inputs = _EMBED_TOKENIZER(proc_texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
        outputs = _EMBED_MODEL(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            emb = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])  # type: ignore[attr-defined]
        else:
            # Fallback: some models expose different output keys
            last = getattr(outputs, "last_hidden_state", None)
            if last is None:
                raise RuntimeError("Unexpected embedding model outputs; last_hidden_state not found")
            emb = _mean_pool(last, inputs["attention_mask"])  # type: ignore[index]
        emb = F.normalize(emb, p=2, dim=1)
        return emb.detach().cpu().numpy()


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T)


def _split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    # Simple rule-based splitter; avoids heavy deps
    pieces = re.split(r"(?<=[\.!?])\s+|[\n\r]+", text)
    sentences = [p.strip() for p in pieces if isinstance(p, str) and p.strip()]
    # Filter very short fragments
    sentences = [s for s in sentences if len(s) >= 5]
    return sentences[:256]


def _nli_probs(premises: List[str], hypotheses: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert _NLI_MODEL is not None and _NLI_TOKENIZER is not None and _DEVICE is not None
    with torch.inference_mode():
        enc = _NLI_TOKENIZER(premises, hypotheses, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(_DEVICE) for k, v in enc.items()}
        outputs = _NLI_MODEL(**enc)
        logits = outputs.logits  # type: ignore[attr-defined]
        probs = F.softmax(logits, dim=-1)
        # Map to entailment/neutral/contradiction indices robustly
        cfg = getattr(_NLI_MODEL, "config", None)
        cfg_id2label = getattr(cfg, "id2label", {}) if cfg is not None else {}
        cfg_label2id = getattr(cfg, "label2id", {}) if cfg is not None else {}
        def _resolve_idx(label_name: str) -> int:
            lname = label_name.lower()
            # Prefer label2id if present (keys are label strings, values are ints)
            try:
                if isinstance(cfg_label2id, dict) and cfg_label2id:
                    for k, v in cfg_label2id.items():
                        if str(k).lower() == lname:
                            return int(v)
            except Exception:
                pass
            # Fallback: scan id2label (keys are ids, values are label strings)
            try:
                if isinstance(cfg_id2label, dict) and cfg_id2label:
                    for i, lab in cfg_id2label.items():
                        if str(lab).lower() == lname:
                            try:
                                return int(i)
                            except Exception:
                                return int(str(i))
            except Exception:
                pass
            # Final fallback to common ordering
            default_order = {"entailment": 2, "neutral": 1, "contradiction": 0}
            return int(default_order.get(lname, 0))
        ent_idx = _resolve_idx("entailment")
        neu_idx = _resolve_idx("neutral")
        con_idx = _resolve_idx("contradiction")
        ent = probs[:, ent_idx].detach().cpu().numpy()
        neu = probs[:, neu_idx].detach().cpu().numpy()
        con = probs[:, con_idx].detach().cpu().numpy()
        return ent, neu, con


def init_verification(
    taxonomy: Dict[str, List[str]],
    method: str = "combo",
    top_k: int = 3,
    embed_model_name: str = "intfloat/multilingual-e5-base",
    nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    thresholds: Optional[Dict[str, float]] = None,
    device: Optional[str] = None,
    debug: Optional[bool] = None,
) -> None:
    """Initialize global models and configuration for verification.

    Call once per Ray worker process.
    """
    _ensure_imports()
    global _EMBED_TOKENIZER, _EMBED_MODEL, _NLI_TOKENIZER, _NLI_MODEL, _CONFIG, _DEVICE, _CLAIM_MAP, _DEBUG
    _CLAIM_MAP = _build_label_to_text_map(taxonomy)
    _DEVICE = _detect_device(device)
    # Resolve debug flag from argument or env var UAIR_VERIFY_DEBUG
    try:
        _DEBUG = bool(debug) if debug is not None else (str(os.environ.get("UAIR_VERIFY_DEBUG", "")).strip().lower() in {"1","true","yes","on"})
    except Exception:
        _DEBUG = False
    _CONFIG = VerificationConfig(
        method=str(method),
        top_k=int(top_k),
        sim_threshold=float(thresholds.get("sim", 0.55) if thresholds else 0.55),
        entail_threshold=float(thresholds.get("ent", 0.85) if thresholds else 0.85),
        contra_max=float(thresholds.get("contra", 0.05) if thresholds else 0.05),
        embed_model_name=embed_model_name,
        nli_model_name=nli_model_name,
        device=_DEVICE,
    )
    # Embedding model
    if _EMBED_MODEL is None:
        try:
            _EMBED_TOKENIZER = AutoTokenizer.from_pretrained(_CONFIG.embed_model_name)
            _EMBED_MODEL = AutoModel.from_pretrained(_CONFIG.embed_model_name)
            _EMBED_MODEL.to(_DEVICE)
            _EMBED_MODEL.eval()
            if _DEBUG:
                try:
                    print(f"[UAIR][verify] Loaded embedding model: {_CONFIG.embed_model_name} on {_DEVICE}", flush=True)
                except Exception:
                    pass
        except Exception as e:
            global _LAST_EMBED_ERROR
            _LAST_EMBED_ERROR = str(e)
            if _DEBUG:
                print(f"[UAIR][verify] Failed to load embedding model '{_CONFIG.embed_model_name}': {e}", flush=True)
    # NLI model (only if needed)
    if _CONFIG.method in {"nli", "combo", "combo_judge"} and _NLI_MODEL is None:
        try:
            _NLI_TOKENIZER = AutoTokenizer.from_pretrained(_CONFIG.nli_model_name)
            _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(_CONFIG.nli_model_name)
            _NLI_MODEL.to(_DEVICE)
            _NLI_MODEL.eval()
            if _DEBUG:
                try:
                    print(f"[UAIR][verify] Loaded NLI model: {_CONFIG.nli_model_name} on {_DEVICE}", flush=True)
                except Exception:
                    pass
        except Exception as e:
            global _LAST_NLI_ERROR
            _LAST_NLI_ERROR = str(e)
            if _DEBUG:
                print(f"[UAIR][verify] Failed to load NLI model '{_CONFIG.nli_model_name}': {e}", flush=True)


def _verify_one(chunk_text: str, chunk_label: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ver_sim_max": None,
        "ver_top_sent": None,
        "ver_evidence_topk": None,
        "ver_nli_ent_max": None,
        "ver_nli_label_max": None,
        "ver_nli_evidence": None,
        "ver_verified_chunk": False,
    }
    if _DEBUG:
        out.update({
            "ver_dbg_method": getattr(_CONFIG, "method", None),
            "ver_dbg_device": _DEVICE,
            "ver_dbg_has_embed": bool(_EMBED_MODEL is not None),
            "ver_dbg_has_nli": bool(_NLI_MODEL is not None),
            "ver_dbg_last_embed_error": _LAST_EMBED_ERROR,
            "ver_dbg_last_nli_error": _LAST_NLI_ERROR,
            "ver_dbg_reason": None,
            "ver_dbg_num_sentences": None,
            "ver_dbg_label_text": None,
            "ver_dbg_embed_error": None,
            "ver_dbg_nli_skipped": None,
            "ver_dbg_nli_skip_reason": None,
            "ver_dbg_nli_error": None,
        })
    # Check if verification is properly initialized
    if _EMBED_MODEL is None or _DEVICE is None:
        if _DEBUG:
            out["ver_dbg_reason"] = "not_initialized"
        return out
    if not isinstance(chunk_label, str) or (chunk_label.strip().lower() == "none"):
        if _DEBUG:
            out["ver_dbg_reason"] = "no_label"
        return out
    label_text = _CLAIM_MAP.get(str(chunk_label))
    if not isinstance(label_text, str) or not label_text:
        if _DEBUG:
            out["ver_dbg_reason"] = "label_text_missing"
        return out
    sentences = _split_sentences(str(chunk_text or ""))
    if not sentences:
        sentences = [str(chunk_text or "").strip()]
    if _DEBUG:
        out["ver_dbg_num_sentences"] = len(sentences)
        out["ver_dbg_label_text"] = label_text
    try:
        query_emb = _encode_embeddings([label_text], is_query=True)
        sent_emb = _encode_embeddings(sentences, is_query=False)
        sims = _cosine_sim_matrix(query_emb, sent_emb)[0]
        order = np.argsort(-sims)
        k = int(max(1, _CONFIG.top_k))
        top_idx = order[:k]
        top_sents = [sentences[i] for i in top_idx]
        top_sims = [float(sims[i]) for i in top_idx]
        out["ver_sim_max"] = float(max(top_sims) if len(top_sims) > 0 else float(sims.max()))
        if len(top_sents) > 0:
            out["ver_top_sent"] = top_sents[0]
            out["ver_evidence_topk"] = top_sents
    except Exception as e:
        # Embedding errors → leave defaults
        if _DEBUG:
            out["ver_dbg_embed_error"] = str(e)
    # Pure embedding method
    if _CONFIG.method == "embed":
        out["ver_verified_chunk"] = bool((out["ver_sim_max"] or 0.0) >= _CONFIG.sim_threshold)
        if _DEBUG:
            out["ver_dbg_nli_skipped"] = True
            out["ver_dbg_nli_skip_reason"] = "method=embed"
        return out
    # NLI or combo
    try:
        if _NLI_MODEL is not None and out.get("ver_evidence_topk"):
            premises = list(out["ver_evidence_topk"])  # type: ignore[arg-type]
            hypotheses = [f"This article is about {label_text}." for _ in premises]
            ent, neu, con = _nli_probs(premises, hypotheses)
            best = int(np.argmax(ent)) if ent.size > 0 else 0
            ent_max = float(ent[best]) if ent.size > 0 else None
            con_best = float(con[best]) if con.size > 0 else None
            out["ver_nli_ent_max"] = ent_max
            out["ver_nli_label_max"] = ("entailment" if ent_max is not None and ent_max >= _CONFIG.entail_threshold else "neutral")
            out["ver_nli_evidence"] = premises[best] if premises else None
            out["ver_verified_chunk"] = bool(
                (ent_max is not None and ent_max >= _CONFIG.entail_threshold) and (con_best is not None and con_best <= _CONFIG.contra_max)
            )
        else:
            # If NLI not available, fall back to embedding threshold
            if _DEBUG:
                out["ver_dbg_nli_skipped"] = True
                out["ver_dbg_nli_skip_reason"] = ("nli_model_none" if _NLI_MODEL is None else "no_evidence_topk")
            out["ver_verified_chunk"] = bool((out["ver_sim_max"] or 0.0) >= _CONFIG.sim_threshold)
    except Exception as e:
        out["ver_verified_chunk"] = bool((out["ver_sim_max"] or 0.0) >= _CONFIG.sim_threshold)
        if _DEBUG:
            out["ver_dbg_nli_error"] = str(e)
    return out


def verify_batch_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Ray Data map_batches-compatible function (batch_format='pandas').

    Requires init_verification() to have been called on the worker.
    """
    # Ensure expected columns exist; create defaults if missing
    if "chunk_text" not in df.columns:
        df["chunk_text"] = ""
    if "chunk_label" not in df.columns:
        df["chunk_label"] = "None"
    
    # Early return with defaults if verification not initialized
    if _EMBED_MODEL is None or _DEVICE is None:
        df["ver_verified_chunk"] = False
        df["ver_sim_max"] = None
        df["ver_nli_ent_max"] = None
        df["ver_nli_evidence"] = None
        if _DEBUG:
            try:
                df["ver_dbg_reason"] = "not_initialized"
                df["ver_dbg_method"] = getattr(_CONFIG, "method", None)
                df["ver_dbg_device"] = _DEVICE
                df["ver_dbg_has_embed"] = bool(_EMBED_MODEL is not None)
                df["ver_dbg_has_nli"] = bool(_NLI_MODEL is not None)
            except Exception:
                pass
        return df
    
    # Work on a copy to avoid index-alignment issues during assignment
    df = df.copy()
    results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            res = _verify_one(str(row.get("chunk_text", "")), str(row.get("chunk_label", "None")))
        except Exception:
            # Use defaults on error
            res = {
                "ver_sim_max": None,
                "ver_top_sent": None,
                "ver_evidence_topk": None,
                "ver_nli_ent_max": None,
                "ver_nli_label_max": None,
                "ver_nli_evidence": None,
                "ver_verified_chunk": False,
            }
        results.append(res)
    
    try:
        res_df = pd.DataFrame(results)
        # Assign by position to avoid misalignment with arbitrary indices
        for col in res_df.columns:
            df[col] = res_df[col].to_numpy()
    except Exception:
        # Add default columns if DataFrame creation fails
        df["ver_verified_chunk"] = False
        df["ver_sim_max"] = None
        df["ver_nli_ent_max"] = None
        df["ver_nli_evidence"] = None
    
    return df




def verify_tuple_claims_batch_pandas(
    df: pd.DataFrame,
    *,
    windowing_enabled: Optional[bool] = None,
    window_size: int = 1,
    window_stride: int = 1,
) -> pd.DataFrame:
    """Verify free-text tuple claims (from NBL decomposition) against article text.

    Expects row text in 'chunk_text' (falls back to 'article_text'). Optionally
    verifies these scalar fields when present and non-empty:
      - 'deployment_domain'
      - 'deployment_purpose'
      - 'deployment_capability'
      - 'deployment_space'

    And these list fields when present (either Python list or JSON-encoded list):
      - 'list_of_harms_that_occurred'
      - 'list_of_risks_that_occurred'
      - 'list_of_benefits_that_occurred'

    Produces per-field verification columns with prefix 'ver_tuple_<field>_' and
    an overall 'ver_tuples_any_verified' boolean.
    """
    # Early return if models are not initialized
    if _EMBED_MODEL is None or _DEVICE is None:
        try:
            df = df.copy()
            # Ensure expected text column exists to avoid downstream surprises
            if "chunk_text" not in df.columns:
                try:
                    df["chunk_text"] = df.get("article_text", "").fillna("").astype(str)
                except Exception:
                    df["chunk_text"] = ""
            # Pre-create scalar metric columns so they always appear
            scalar_fields = [
                ("deployment_domain", "The AI use is in the domain of {claim}."),
                ("deployment_purpose", "The purpose of the AI use is {claim}."),
                ("deployment_capability", "The capability of the AI use is {claim}."),
                ("deployment_space", "The AI use operates in {claim}."),
                ("identity_of_ai_deployer", "The AI system is deployed by {claim}."),
                ("identity_of_ai_subject", "The AI system affects {claim}."),
                ("identity_of_ai_developer", "The AI system is developed by {claim}."),
                ("location_of_ai_deployer", "The AI deployer is located in {claim}."),
                ("location_of_ai_subject", "The AI subject is located in {claim}."),
                ("date_and_time_of_event", "The event occurred on {claim}."),
            ]
            for field, _ in scalar_fields:
                base = f"ver_tuple_{field}"
                df[f"{base}_sim_max"] = float("nan")
                df[f"{base}_nli_ent_max"] = float("nan")
                df[f"{base}_evidence"] = None
                df[f"{base}_verified"] = True
            # Pre-create list summary columns
            list_fields = [
                "list_of_harms_that_occurred",
                "list_of_risks_that_occurred",
                "list_of_benefits_that_occurred",
            ]
            for field in list_fields:
                base = f"ver_tuple_{field}"
                df[f"{base}_verified_count"] = 0
                df[f"{base}_any"] = False
                df[f"{base}_verified_bools"] = [[] for _ in range(len(df))]
                df[f"{base}_details_json"] = "[]"
            # Overall flags
            df["ver_tuple_overall_pass"] = False
            df["ver_tuples_any_verified"] = False
            # Optional debug surfaces
            if _DEBUG:
                try:
                    df["ver_tuple_dbg_not_initialized"] = True
                    df["ver_tuple_dbg_method"] = getattr(_CONFIG, "method", None)
                    df["ver_tuple_dbg_device"] = _DEVICE
                    df["ver_tuple_dbg_has_embed"] = bool(_EMBED_MODEL is not None)
                    df["ver_tuple_dbg_has_nli"] = bool(_NLI_MODEL is not None)
                except Exception:
                    pass
        except Exception:
            pass
        return df

    def _text_from_row(row: Any) -> str:
        try:
            txt = row.get("chunk_text")
            if not isinstance(txt, str) or not txt.strip():
                txt2 = row.get("article_text")
                return str(txt2 or "")
            return str(txt)
        except Exception:
            return ""

    def _non_empty_str(x: Any) -> Optional[str]:
        try:
            s = str(x).strip()
            if not s:
                return None
            # Treat sentinel strings like 'None'/'null' as empty/missing
            if s.lower() in {"none", "null"}:
                return None
            return s
        except Exception:
            return None

    def _parse_list(v: Any) -> List[str]:
        # Accept list/tuple or JSON-encoded string; drop empties
        try:
            if isinstance(v, (list, tuple)):
                raw = list(v)
            elif isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    raw = list(parsed) if isinstance(parsed, (list, tuple)) else [v]
                except Exception:
                    raw = [v]
            else:
                raw = []
        except Exception:
            raw = []
        out: List[str] = []
        for it in raw:
            s = _non_empty_str(it)
            if s is not None:
                out.append(s)
        return out

    def _to_json_str_like(value: Any) -> Optional[str]:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            try:
                return str(value)
            except Exception:
                return None

    def _build_text_units(text: str, use_windows: bool, w_size: int, w_stride: int) -> List[str]:
        try:
            sentences = _split_sentences(text)
            if not sentences:
                return [text.strip()]
            if not use_windows or w_size <= 1:
                return sentences
            units: List[str] = []
            i = 0
            n = len(sentences)
            s = max(1, int(w_stride))
            w = max(2, int(w_size))
            while i < n:
                win = sentences[i : i + w]
                if not win:
                    break
                units.append(" ".join(win))
                i += s
            return units
        except Exception:
            return [text.strip()]

    def _verify_claim_over_units(units: List[str], query_text: str, hypothesis: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sim_max": None,
            "evidence": None,
            "nli_ent_max": None,
            "verified": False,
        }
        try:
            # Embedding similarity (top-k evidence)
            try:
                query_emb = _encode_embeddings([query_text], is_query=True)
                sent_emb = _encode_embeddings(units, is_query=False)
                sims = _cosine_sim_matrix(query_emb, sent_emb)[0]
                order = np.argsort(-sims)
                k = int(max(1, _CONFIG.top_k))
                top_idx = order[:k]
                top_sents = [units[i] for i in top_idx]
                top_sims = [float(sims[i]) for i in top_idx]
                out["sim_max"] = float(max(top_sims) if len(top_sims) > 0 else float(sims.max()))
                if len(top_sents) > 0:
                    out["evidence"] = top_sents[0]
            except Exception:
                top_sents = []
            # Decide verification mode
            try:
                mode = getattr(_CONFIG, "method", "combo")
            except Exception:
                mode = "combo"
            if _DEBUG:
                out_dbg = {
                    "dbg_units_count": len(units),
                    "dbg_topk_used": int(max(1, _CONFIG.top_k)),
                    "dbg_method": mode,
                    "dbg_has_nli": bool(_NLI_MODEL is not None),
                }
                try:
                    out.update(out_dbg)
                except Exception:
                    pass

            # Embed-only mode
            if str(mode) == "embed":
                out["verified"] = bool((out.get("sim_max") or 0.0) >= _CONFIG.sim_threshold)
                return out

            # NLI or combo mode: run NLI when available; do NOT fall back to sim if NLI computed but fails
            try:
                if _NLI_MODEL is not None and (top_sents or units):
                    premises = (top_sents if top_sents else units)[: int(max(1, _CONFIG.top_k))]
                    hypotheses = [hypothesis for _ in premises]
                    ent, neu, con = _nli_probs(premises, hypotheses)
                    best = int(np.argmax(ent)) if ent.size > 0 else 0
                    ent_max = float(ent[best]) if ent.size > 0 else None
                    con_best = float(con[best]) if con.size > 0 else None
                    out["nli_ent_max"] = ent_max
                    if premises:
                        out["evidence"] = premises[best]
                        try:
                            order = list(np.argsort(-ent)) if hasattr(np, "argsort") else list(range(len(premises)))
                            out["evidence_topk"] = [premises[i] for i in order]
                        except Exception:
                            out["evidence_topk"] = premises
                    out["verified"] = bool(
                        (ent_max is not None and ent_max >= _CONFIG.entail_threshold)
                        and (con_best is not None and con_best <= _CONFIG.contra_max)
                    )
                    if _DEBUG:
                        try:
                            out["dbg_nli_premises_count"] = len(premises)
                            out["dbg_nli_hypothesis"] = hypothesis
                            out["dbg_nli_ent_max"] = ent_max
                            out["dbg_nli_con_best"] = con_best
                        except Exception:
                            pass
                else:
                    # If NLI unavailable or no usable premises, fall back to embedding threshold
                    out["verified"] = bool((out.get("sim_max") or 0.0) >= _CONFIG.sim_threshold)
            except Exception as e:
                # On NLI error, fall back to embedding threshold
                if _DEBUG:
                    try:
                        out["dbg_nli_exception"] = str(e)
                    except Exception:
                        pass
                out["verified"] = bool((out.get("sim_max") or 0.0) >= _CONFIG.sim_threshold)
        except Exception:
            pass
        return out

    # Ensure expected text column exists; create defaults if missing
    if "chunk_text" not in df.columns:
        try:
            df = df.copy()
            df["chunk_text"] = df.get("article_text", "").fillna("").astype(str)
        except Exception:
            pass

    df = df.copy()
    any_verified_col = []

    # Scalar tuple fields and their hypothesis templates (canonical names)
    scalar_fields = [
        ("deployment_domain", "This article is about {claim}."),
        ("deployment_purpose", "The purpose is {claim}."),
        ("deployment_capability", "The system is capable of {claim}."),
        ("deployment_space", "The system operates in {claim}."),
        ("identity_of_ai_deployer", "The AI system is deployed by {claim}."),
        ("identity_of_ai_subject", "The AI system affects {claim}."),
        ("identity_of_ai_developer", "The AI system is developed by {claim}."),
        ("location_of_ai_deployer", "The AI deployer is located in {claim}."),
        ("location_of_ai_subject", "The AI subject is located in {claim}."),
        ("date_and_time_of_event", "The event occurred on {claim}."),
    ]
    # Accept legacy/synonym column names from older decomposition outputs
    scalar_aliases: Dict[str, List[str]] = {
        "deployment_domain": ["deployment_domain", "domain", "use_domain"],
        "deployment_purpose": ["deployment_purpose", "purpose", "goal", "objective"],
        "deployment_capability": ["deployment_capability", "capability", "capabilities", "function", "ability"],
        "deployment_space": ["deployment_space", "space"],
        "identity_of_ai_deployer": ["identity_of_ai_deployer", "ai_deployer", "deployer", "operator", "implementer", "user", "agency", "organization_deployer"],
        "identity_of_ai_subject": ["identity_of_ai_subject", "ai_subject", "subject", "data_subject", "affected_party", "individual", "group"],
        "identity_of_ai_developer": ["identity_of_ai_developer", "ai_developer", "developer", "vendor", "builder", "provider", "manufacturer"],
        "location_of_ai_deployer": ["location_of_ai_deployer", "deployer_location", "operator_location", "location_deployer"],
        "location_of_ai_subject": ["location_of_ai_subject", "subject_location", "location_subject", "where"],
        "date_and_time_of_event": ["date_and_time_of_event", "date___time_of_event", "datetime", "date_time", "date", "time", "event_time", "when"],
    }
    # List tuple fields and their hypothesis templates
    list_fields = [
        ("list_of_harms_that_occurred", "This article reports {claim}."),
        ("list_of_risks_that_occurred", "This article reports {claim}."),
        ("list_of_benefits_that_occurred", "This article reports {claim}."),
    ]

    results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        row_out: Dict[str, Any] = {}
        text = _text_from_row(row)
        # Build sentence windows once per row
        units = _build_text_units(
            text,
            bool(windowing_enabled) and int(window_size) > 1,
            int(window_size),
            int(window_stride),
        )
        row_any_verified = False

        # Pre-create scalar metric columns so they always appear in outputs
        try:
            for field, _ in scalar_fields:
                base = f"ver_tuple_{field}"
                # Use NaN for numeric defaults to preserve float dtype across batches/files
                row_out[f"{base}_sim_max"] = float("nan")
                row_out[f"{base}_nli_ent_max"] = float("nan")
                row_out[f"{base}_evidence"] = None
                row_out[f"{base}_evidence_topk_json"] = "[]"
                row_out[f"{base}_verified"] = True
        except Exception:
            pass

        # Scalars (support legacy aliases): write outputs under canonical names
        for canonical, hyp_tmpl in scalar_fields:
            # Find the first alias present with a non-empty value
            val = None
            try:
                for alias in scalar_aliases.get(canonical, [canonical]):
                    if alias in df.columns:
                        v = _non_empty_str(row.get(alias))
                        if v is not None:
                            val = v
                            break
            except Exception:
                val = None
            if val is not None:
                res = _verify_claim_over_units(units, val, hyp_tmpl.format(claim=val))
                base = f"ver_tuple_{canonical}"
                row_out[f"{base}_sim_max"] = res.get("sim_max")
                row_out[f"{base}_nli_ent_max"] = res.get("nli_ent_max")
                row_out[f"{base}_evidence"] = res.get("evidence")
                try:
                    evk = res.get("evidence_topk")
                    row_out[f"{base}_evidence_topk_json"] = _to_json_str_like(evk) if evk is not None else "[]"
                except Exception:
                    pass
                row_out[f"{base}_verified"] = bool(res.get("verified"))
                if _DEBUG:
                    try:
                        for k, v in res.items():
                            if isinstance(k, str) and k.startswith("dbg_"):
                                # strip single dbg_ prefix for cleaner column names
                                col_k = k[4:] if k.startswith("dbg_") else k
                                row_out[f"{base}_dbg_{col_k}"] = v
                    except Exception:
                        pass
                row_any_verified = row_any_verified or bool(res.get("verified"))

        # Lists
        for field, hyp_tmpl in list_fields:
            if field in df.columns:
                items = _parse_list(row.get(field))
                details = []
                verified_count = 0
                verified_bools: List[bool] = []
                for item in items:
                    res = _verify_claim_over_units(units, item, hyp_tmpl.format(claim=item))
                    details.append({
                        "item": item,
                        "sim_max": res.get("sim_max"),
                        "nli_ent_max": res.get("nli_ent_max"),
                        "evidence": res.get("evidence"),
                        "verified": bool(res.get("verified")),
                    })
                    if bool(res.get("verified")):
                        verified_count += 1
                    verified_bools.append(bool(res.get("verified")))
                row_out[f"ver_tuple_{field}_verified_count"] = int(verified_count)
                row_out[f"ver_tuple_{field}_any"] = bool(verified_count > 0)
                # Per-item verification booleans (same length as input list)
                row_out[f"ver_tuple_{field}_verified_bools"] = verified_bools
                row_out[f"ver_tuple_{field}_details_json"] = _to_json_str_like(details)
                row_any_verified = row_any_verified or bool(verified_count > 0)

        # Overall tuple pass metric: require all present scalar components to exceed thresholds
        try:
            ent_thr = float(getattr(_CONFIG, "entail_threshold", 0.85) or 0.85)
        except Exception:
            ent_thr = 0.85
        try:
            sim_thr = float(getattr(_CONFIG, "sim_threshold", 0.55) or 0.55)
        except Exception:
            sim_thr = 0.55
        all_pass = True
        considered = 0
        for field, _ in scalar_fields:
            try:
                # Consider only components that are present in the row and non-empty
                raw_val = _non_empty_str(row.get(field)) if field in df.columns else None
                if raw_val is None:
                    continue
                nli_val = row_out.get(f"ver_tuple_{field}_nli_ent_max")
                sim_val = row_out.get(f"ver_tuple_{field}_sim_max")
                if nli_val is not None:
                    component_pass = bool(float(nli_val) >= ent_thr)
                elif sim_val is not None:
                    component_pass = bool(float(sim_val) >= sim_thr)
                else:
                    component_pass = False
                considered += 1
                if not component_pass:
                    all_pass = False
            except Exception:
                considered += 1
                all_pass = False
        # If no components were considered, default to True (nothing to verify)
        if considered == 0:
            row_out["ver_tuple_overall_pass"] = True
        else:
            row_out["ver_tuple_overall_pass"] = bool(all_pass)

        row_out["ver_tuples_any_verified"] = bool(row_any_verified)
        any_verified_col.append(bool(row_any_verified))
        results.append(row_out)

    # Assign columns back to df
    try:
        res_df = pd.DataFrame(results)
        for col in res_df.columns:
            df[col] = res_df[col].to_numpy()
    except Exception:
        try:
            df["ver_tuples_any_verified"] = any_verified_col
        except Exception:
            pass
    return df

