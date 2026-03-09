from typing import Any, Dict, List
import os
import json
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# Use in-repo verification core implementation (ported from .OLD/experiments/verification.py)
from ..verification_core import (
    init_verification,
    verify_batch_pandas,
    parse_thresholds_string,
)


def run_verification_stage(df, cfg):
    """
    Runs verification over a pandas DataFrame.

    Expects taxonomy JSON at cfg.taxonomy_json and chunk rows with
    at least: article_id, chunk_text, chunk_label (string index or "None").
    """
    # Imports are resolved at module import time; functions are available here

    # Resolve taxonomy (YAML or JSON)
    taxonomy_path = str(getattr(cfg, "taxonomy_json", "") or "")
    taxonomy = {}
    try:
        if taxonomy_path and os.path.exists(taxonomy_path):
            if taxonomy_path.endswith((".yaml", ".yml")) and yaml is not None:
                with open(taxonomy_path, "r") as f:
                    data = yaml.safe_load(f)
            else:
                with open(taxonomy_path, "r") as f:
                    data = json.load(f)
            taxonomy = data.get("taxonomy", data) if isinstance(data, dict) else {}
    except Exception:
        taxonomy = {}

    # Verify configuration
    try:
        method = str(getattr(cfg.verify, "method", "combo"))
    except Exception:
        method = "combo"
    try:
        top_k = int(getattr(cfg.verify, "top_k", 3) or 3)
    except Exception:
        top_k = 3
    try:
        thr_s = str(getattr(cfg.verify, "thresholds", "sim=0.55,ent=0.85,contra=0.05"))
    except Exception:
        thr_s = "sim=0.55,ent=0.85,contra=0.05"
    sim_thr, ent_thr, con_thr = parse_thresholds_string(thr_s)
    thresholds = {"sim": sim_thr, "ent": ent_thr, "contra": con_thr}
    try:
        device = getattr(cfg.verify, "device", None)
    except Exception:
        device = None

    pdf = df.copy() if df is not None else pd.DataFrame([])
    try:
        rows_in_total = int(len(pdf))
    except Exception:
        rows_in_total = None  # type: ignore
    init_verification(
        taxonomy=taxonomy,
        method=method,
        top_k=top_k,
        thresholds=thresholds,
        device=device,
    )
    if len(pdf) == 0:
        return pdf
    # Keep only rows WITH chunk_label_name (present and non-empty), and ensure chunk_text
    try:
        if len(pdf):
            try:
                s = pdf.get("chunk_label_name")
                labeled_mask = (~s.isna()) & (~s.astype(str).str.strip().str.strip('"').str.strip("'").str.lower().isin(["", "none", "nan"]))
            except Exception:
                labeled_mask = (~pdf.get("chunk_label_name").isna())
            pdf = pdf[labeled_mask]
            if "chunk_text" not in pdf.columns:
                pdf["chunk_text"] = pdf.get("article_text", pd.Series([""] * len(pdf))).fillna("").astype(str)
            else:
                pdf["chunk_text"] = pdf["chunk_text"].fillna(pdf.get("article_text", "")).astype(str)
        rows_kept = int(len(pdf))
    except Exception:
        rows_kept = None  # type: ignore

    try:
        out = verify_batch_pandas(pdf)
    except Exception:
        out = pdf
    # Pandas doc-level aggregation
    try:
        def _reduce_verify_pdf(df_in: pd.DataFrame) -> pd.DataFrame:
            if df_in.empty:
                return pd.DataFrame([])
            verified_chunk = bool(df_in.get("ver_verified_chunk", pd.Series([], dtype=bool)).any())
            best_ent = None
            best_evi = None
            max_sim = None
            max_sim_evi = None
            if "ver_nli_ent_max" in df_in.columns:
                try:
                    idx = int(df_in["ver_nli_ent_max"].astype(float).fillna(-1).idxmax())
                    best_ent = float(df_in.loc[idx, "ver_nli_ent_max"]) if pd.notna(df_in.loc[idx, "ver_nli_ent_max"]) else None
                    best_evi = df_in.loc[idx, "ver_nli_evidence"] if "ver_nli_evidence" in df_in.columns else None
                except Exception:
                    pass
            if "ver_sim_max" in df_in.columns:
                try:
                    idx2 = int(df_in["ver_sim_max"].astype(float).fillna(-1).idxmax())
                    max_sim = float(df_in.loc[idx2, "ver_sim_max"]) if pd.notna(df_in.loc[idx2, "ver_sim_max"]) else None
                    max_sim_evi = df_in.loc[idx2, "ver_top_sent"] if "ver_top_sent" in df_in.columns else None
                except Exception:
                    pass
            out_row = {
                "article_id": df_in.get("article_id", pd.Series([None])).iloc[0],
                "verified_doc": bool(verified_chunk),
                "ver_doc_best_ent": best_ent,
                "ver_doc_best_evidence": best_evi,
                "ver_doc_max_sim": max_sim,
                "ver_doc_max_sim_evidence": max_sim_evi,
            }
            return pd.DataFrame([out_row])

        ver_docs_pdf = out.groupby("article_id", dropna=False).apply(_reduce_verify_pdf).reset_index(drop=True)
    except Exception:
        ver_docs_pdf = pd.DataFrame([])

    # Stage-scoped logging is handled by the orchestrator

    # Persist side outputs if requested
    try:
        out_dir = str(getattr(cfg.runtime, "output_dir", "") or "")
    except Exception:
        out_dir = ""
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        try:
            out.to_parquet(os.path.join(out_dir, "chunks_verification.parquet"), index=False)
        except Exception:
            pass
        try:
            if len(ver_docs_pdf):
                ver_docs_pdf.to_parquet(os.path.join(out_dir, "docs_verification.parquet"), index=False)
        except Exception:
            pass
    return out


