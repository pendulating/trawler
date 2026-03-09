from typing import Any, Dict
import os
import json
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

from ..verification_core import (
    init_verification,
    verify_tuple_claims_batch_pandas,
    parse_thresholds_string,
)


def run_verification_stage_nbl(df, cfg):
    """
    Verify decomposed NBL tuple claims (domain/purpose/capability/space and lists) against article text.

    Inputs are expected to be the outputs of decompose_nbl stage. Operates on
    pandas DataFrames only.
    """
    # Resolve taxonomy (optional; used only to keep interface parity)
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

    # Optional model overrides
    try:
        embed_model_name = str(getattr(cfg.verify, "embed_model_name", "intfloat/multilingual-e5-base"))
    except Exception:
        embed_model_name = "intfloat/multilingual-e5-base"
    try:
        nli_model_name = str(getattr(cfg.verify, "nli_model_name", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"))
    except Exception:
        nli_model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    pdf = df.copy() if df is not None else pd.DataFrame([])
    try:
        rows_in_total = int(len(pdf))
    except Exception:
        rows_in_total = None  # type: ignore

    init_verification(
        taxonomy=taxonomy,
        method=method,
        top_k=top_k,
        embed_model_name=embed_model_name,
        nli_model_name=nli_model_name,
        thresholds=thresholds,
        device=device,
    )
    if len(pdf) == 0:
        return pdf
    # Ensure chunk_text exists
    try:
        if "chunk_text" not in pdf.columns:
            pdf["chunk_text"] = pdf.get("article_text", pd.Series([""] * len(pdf))).fillna("").astype(str)
        else:
            pdf["chunk_text"] = pdf["chunk_text"].fillna(pdf.get("article_text", "")).astype(str)
    except Exception:
        pass

    try:
        # Windowing config
        try:
            w_cfg = getattr(cfg.verify, "windowing", None)
        except Exception:
            w_cfg = None
        w_enabled = None
        w_size = 1
        w_stride = 1
        try:
            if w_cfg is not None:
                w_enabled = bool(getattr(w_cfg, "enabled", None)) if getattr(w_cfg, "enabled", None) is not None else None
                w_size = int(getattr(w_cfg, "window_size", 1) or 1)
                w_stride = int(getattr(w_cfg, "stride", getattr(w_cfg, "window_stride", 1)) or 1)
        except Exception:
            pass
        out = verify_tuple_claims_batch_pandas(
            pdf,
            windowing_enabled=w_enabled,
            window_size=w_size,
            window_stride=w_stride,
        )
    except Exception as e:
        # Preserve input and align schema; print error for debugging
        try:
            print(f"[verify_nbl] verify_tuple_claims_batch_pandas failed: {e}", flush=True)
        except Exception:
            pass
        out = pdf

    # Align schema to ensure scalar tuple verification columns are present
    try:
        scalar_fields = [
            "deployment_domain",
            "deployment_purpose",
            "deployment_capability",
            "deployment_space",
            "identity_of_ai_deployer",
            "identity_of_ai_subject",
            "identity_of_ai_developer",
            "location_of_ai_deployer",
            "location_of_ai_subject",
            "date_and_time_of_event",
        ]
        for field in scalar_fields:
            base = f"ver_tuple_{field}"
            if f"{base}_sim_max" not in out.columns:
                out[f"{base}_sim_max"] = float("nan")
            if f"{base}_nli_ent_max" not in out.columns:
                out[f"{base}_nli_ent_max"] = float("nan")
            if f"{base}_evidence" not in out.columns:
                out[f"{base}_evidence"] = None
            if f"{base}_verified" not in out.columns:
                out[f"{base}_verified"] = False
    except Exception:
        pass

    # Emit diagnostics for scalar NLI columns
    try:
        scalar_nli_cols = [
            "ver_tuple_deployment_domain_nli_ent_max",
            "ver_tuple_deployment_purpose_nli_ent_max",
            "ver_tuple_deployment_capability_nli_ent_max",
            "ver_tuple_deployment_space_nli_ent_max",
            "ver_tuple_identity_of_ai_deployer_nli_ent_max",
            "ver_tuple_identity_of_ai_subject_nli_ent_max",
            "ver_tuple_identity_of_ai_developer_nli_ent_max",
            "ver_tuple_location_of_ai_deployer_nli_ent_max",
            "ver_tuple_location_of_ai_subject_nli_ent_max",
            "ver_tuple_date_and_time_of_event_nli_ent_max",
        ]
        totals = len(out) if hasattr(out, "__len__") else None
        counts = {}
        for c in scalar_nli_cols:
            try:
                counts[c] = int(out[c].notna().sum()) if c in out.columns else None
            except Exception:
                counts[c] = None
        print(
            f"[verify_nbl] scalar nli_ent_max non-null (of {totals}): "
            + ", ".join([f"{k}={counts[k]}" for k in scalar_nli_cols]),
            flush=True,
        )
    except Exception:
        pass

    # Pandas doc-level aggregation
    try:
        def _reduce_verify_pdf(df_in: pd.DataFrame) -> pd.DataFrame:
            if df_in.empty:
                return pd.DataFrame([])
            any_tuple = bool(df_in.get("ver_tuples_any_verified", pd.Series([], dtype=bool)).any())
            best_ent = None
            if "ver_tuple_deployment_domain_nli_ent_max" in df_in.columns:
                try:
                    idx = int(df_in["ver_tuple_deployment_domain_nli_ent_max"].astype(float).fillna(-1).idxmax())
                    best_ent = float(df_in.loc[idx, "ver_tuple_deployment_domain_nli_ent_max"]) if pd.notna(df_in.loc[idx, "ver_tuple_deployment_domain_nli_ent_max"]) else None
                except Exception:
                    pass
            # Per-field best evidences (domain/purpose/capability)
            field_best_evidence: Dict[str, Any] = {}
            for f in ("deployment_domain", "deployment_purpose", "deployment_capability"):
                try:
                    col_ent = f"ver_tuple_{f}_nli_ent_max"
                    col_evi = f"ver_tuple_{f}_evidence"
                    if col_ent in df_in.columns:
                        i = int(df_in[col_ent].astype(float).fillna(-1).idxmax())
                        field_best_evidence[f] = df_in.loc[i, col_evi] if col_evi in df_in.columns else None
                    else:
                        field_best_evidence[f] = None
                except Exception:
                    field_best_evidence[f] = None
            # Core tuple verified at doc level: inputs must be present AND verified
            try:
                core_fields = [
                    "deployment_domain",
                    "deployment_purpose",
                    "deployment_capability",
                ]
                def _filled_series(s: pd.Series) -> pd.Series:
                    try:
                        ss = s.astype(str).str.strip()
                        ss_l = ss.str.lower()
                        return (ss != "") & (~ss_l.isin(["none", "null"]))
                    except Exception:
                        return pd.Series([False] * len(s))
                core_ok = True
                for f in core_fields:
                    input_present_any = False
                    try:
                        if f in df_in.columns:
                            input_present_any = bool(_filled_series(df_in[f]).any())
                    except Exception:
                        input_present_any = False
                    ver_col = f"ver_tuple_{f}_verified"
                    verified_any = False
                    try:
                        if ver_col in df_in.columns:
                            verified_any = bool(df_in[ver_col].astype(bool).any())
                    except Exception:
                        verified_any = False
                    core_ok = core_ok and input_present_any and verified_any
            except Exception:
                core_ok = False
            out_row = {
                "article_id": df_in.get("article_id", pd.Series([None])).iloc[0],
                "doc_any_component_verified": any_tuple,
                "ver_doc_best_ent": best_ent,
                "ver_doc_best_evidence_deployment_domain": field_best_evidence.get("deployment_domain"),
                "ver_doc_best_evidence_deployment_purpose": field_best_evidence.get("deployment_purpose"),
                "ver_doc_best_evidence_deployment_capability": field_best_evidence.get("deployment_capability"),
                "core_tuple_verified": bool(core_ok),
            }
            return pd.DataFrame([out_row])

        ver_docs_pdf = out.groupby("article_id", dropna=False).apply(_reduce_verify_pdf).reset_index(drop=True)
        try:
            # Move verified_doc and core_tuple_verified to the end
            doc_cols = list(ver_docs_pdf.columns)
            tail = [c for c in ("verified_doc", "core_tuple_verified") if c in doc_cols]
            head = [c for c in doc_cols if c not in tail]
            ver_docs_pdf = ver_docs_pdf[head + tail]
        except Exception:
            pass
    except Exception:
        ver_docs_pdf = pd.DataFrame([])

    # Side outputs
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
            out.to_parquet(os.path.join(out_dir, "tuples_verification.parquet"), index=False)
        except Exception:
            pass
        try:
            if len(ver_docs_pdf):
                ver_docs_pdf.to_parquet(os.path.join(out_dir, "docs_verification.parquet"), index=False)
        except Exception:
            pass
    return out


