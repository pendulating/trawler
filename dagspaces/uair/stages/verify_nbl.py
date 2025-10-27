from typing import Any, Dict
import os
import json
import pandas as pd

try:
    import ray  # type: ignore
    _RAY_OK = True
except Exception:
    _RAY_OK = False

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

    Inputs are expected to be the outputs of decompose_nbl stage. This stage works on
    Ray Datasets or pandas DataFrames, mirroring the structure of stages/verify.py but
    operating on tuple fields instead of taxonomy chunk labels.
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

    # Ensure Ray is initialized with SLURM-aware CPU caps
    try:
        import ray  # type: ignore
        if not ray.is_initialized():
            cpus_alloc = None
            try:
                cpt = os.environ.get("SLURM_CPUS_PER_TASK")
                if cpt and str(cpt).strip() != "":
                    cpus_alloc = int(cpt)
                elif os.environ.get("SLURM_CPUS_ON_NODE"):
                    v = os.environ.get("SLURM_CPUS_ON_NODE")
                    try:
                        if v and "," in v:
                            cpus_alloc = sum(int(p) for p in v.split(",") if p.strip())
                        elif v and "(x" in v and v.endswith(")"):
                            import re as _re
                            m = _re.match(r"^(\d+)\(x(\d+)\)$", v)
                            if m:
                                cpus_alloc = int(m.group(1)) * int(m.group(2))
                        elif v:
                            cpus_alloc = int(v)
                    except Exception:
                        cpus_alloc = None
            except Exception:
                cpus_alloc = None
            try:
                if cpus_alloc and int(cpus_alloc) > 0:
                    ray.init(log_to_driver=True, num_cpus=int(cpus_alloc))
                else:
                    ray.init(log_to_driver=True)
            except Exception:
                pass
            # Constrain Ray Data CPUs
            try:
                if cpus_alloc and int(cpus_alloc) > 0:
                    ctx = ray.data.DataContext.get_current()
                    ctx.execution_options.resource_limits = ctx.execution_options.resource_limits.copy(cpu=int(cpus_alloc))
            except Exception:
                pass
    except Exception:
        pass

    is_ray_ds = hasattr(df, "map_batches") and hasattr(df, "count") and _RAY_OK
    if is_ray_ds:
        try:
            rows_in_total = int(df.count())
        except Exception:
            rows_in_total = None  # type: ignore
        ds_in = df

        # Ensure chunk_text exists
        def _ensure_chunk_text(row: Dict[str, Any]) -> Dict[str, Any]:
            r = dict(row)
            if not r.get("chunk_text"):
                r["chunk_text"] = str(r.get("article_text") or "")
            return r

        try:
            ds_in = ds_in.map(_ensure_chunk_text)
        except Exception:
            pass
        try:
            rows_kept = int(ds_in.count())
        except Exception:
            rows_kept = None  # type: ignore

        def _init_fn() -> None:
            init_verification(
                taxonomy=taxonomy,
                method=method,
                top_k=top_k,
                embed_model_name=embed_model_name,
                nli_model_name=nli_model_name,
                thresholds=thresholds,
                device=device,
                debug=bool(getattr(getattr(cfg, "runtime", object()), "debug", False)),
            )

        def _verify_with_init(pdf: pd.DataFrame) -> pd.DataFrame:
            try:
                init_verification(
                    taxonomy=taxonomy,
                    method=method,
                    top_k=top_k,
                    embed_model_name=embed_model_name,
                    nli_model_name=nli_model_name,
                    thresholds=thresholds,
                    device=device,
                    debug=bool(getattr(getattr(cfg, "runtime", object()), "debug", False)),
                )
            except Exception:
                pass
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
            return verify_tuple_claims_batch_pandas(
                pdf,
                windowing_enabled=w_enabled,
                window_size=w_size,
                window_stride=w_stride,
            )

        try:
            ds_in = ds_in.map_batches(lambda x: x, batch_format="pandas", fn_constructor=_init_fn)  # type: ignore[arg-type]
        except Exception:
            pass
        ds_out = ds_in.map_batches(_verify_with_init, batch_format="pandas")

        # Ensure scalar tuple verification columns exist across all batches
        def _ensure_scalar_cols(pdf: pd.DataFrame) -> pd.DataFrame:
            pdf = pdf.copy()
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
                    if f"{base}_sim_max" not in pdf.columns:
                        pdf[f"{base}_sim_max"] = float("nan")
                    if f"{base}_nli_ent_max" not in pdf.columns:
                        pdf[f"{base}_nli_ent_max"] = float("nan")
                    if f"{base}_evidence" not in pdf.columns:
                        pdf[f"{base}_evidence"] = None
                    if f"{base}_verified" not in pdf.columns:
                        pdf[f"{base}_verified"] = False
            except Exception:
                pass
            return pdf

        try:
            ds_out = ds_out.map_batches(_ensure_scalar_cols, batch_format="pandas")
        except Exception:
            pass

        # Reorder columns: group ver_ columns by input; place overall flags at end
        def _reorder_tuple_cols(pdf: pd.DataFrame) -> pd.DataFrame:
            try:
                cols = list(pdf.columns)
                non_ver = [c for c in cols if not c.startswith("ver_")]
                field_order = [
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
                    "list_of_harms_that_occurred",
                    "list_of_risks_that_occurred",
                    "list_of_benefits_that_occurred",
                ]
                suffix_priority = [
                    "_verified",
                    "_nli_ent_max",
                    "_sim_max",
                    "_evidence",
                    "_evidence_topk_json",
                ]
                grouped: list[str] = []
                for f in field_order:
                    prefix = f"ver_tuple_{f}_"
                    f_cols = [c for c in cols if c.startswith(prefix)]
                    if not f_cols:
                        continue
                    # priority ordering
                    for suf in suffix_priority:
                        target = prefix + suf.lstrip("_") if suf.startswith("_") else prefix + suf
                        for c in f_cols:
                            if c == target and c not in grouped:
                                grouped.append(c)
                    # then dbg_* and any others
                    dbg_cols = sorted([c for c in f_cols if "_dbg_" in c and c not in grouped])
                    other_cols = sorted([c for c in f_cols if c not in grouped and c not in dbg_cols])
                    grouped.extend(other_cols)
                    grouped.extend(dbg_cols)
                # overall flags at end if present
                tail = []
                for c in ("ver_tuples_any_verified", "ver_tuple_overall_pass"):
                    if c in cols:
                        tail.append(c)
                # Build final order without duplicates
                seen = set()
                ordered = []
                for c in non_ver + grouped + tail:
                    if c in cols and c not in seen:
                        ordered.append(c)
                        seen.add(c)
                # Append any remaining columns
                for c in cols:
                    if c not in seen:
                        ordered.append(c)
                        seen.add(c)
                return pdf[ordered]
            except Exception:
                return pdf

        try:
            ds_out = ds_out.map_batches(_reorder_tuple_cols, batch_format="pandas")
        except Exception:
            pass

        # Doc-level aggregation over tuple verifications
        def _reduce_verify(pdf: pd.DataFrame) -> pd.DataFrame:
            if pdf is None or len(pdf) == 0:
                return pd.DataFrame([])
            def _first(col: str):
                return pdf.get(col, pd.Series([None])).iloc[0]
            any_tuple = bool(pdf.get("ver_tuples_any_verified", pd.Series([], dtype=bool)).any())
            best_ent = None
            # Prefer domain signal if present
            if "ver_tuple_deployment_domain_nli_ent_max" in pdf.columns:
                try:
                    idx = int(pdf["ver_tuple_deployment_domain_nli_ent_max"].astype(float).fillna(-1).idxmax())
                    best_ent = float(pdf.loc[idx, "ver_tuple_deployment_domain_nli_ent_max"]) if pd.notna(pdf.loc[idx, "ver_tuple_deployment_domain_nli_ent_max"]) else None
                except Exception:
                    pass
            # Per-field best evidences (domain/purpose/capability)
            field_best_evidence: Dict[str, Any] = {}
            for f in ("deployment_domain", "deployment_purpose", "deployment_capability"):
                try:
                    col_ent = f"ver_tuple_{f}_nli_ent_max"
                    col_evi = f"ver_tuple_{f}_evidence"
                    if col_ent in pdf.columns:
                        i = int(pdf[col_ent].astype(float).fillna(-1).idxmax())
                        field_best_evidence[f] = pdf.loc[i, col_evi] if col_evi in pdf.columns else None
                    else:
                        field_best_evidence[f] = None
                except Exception:
                    field_best_evidence[f] = None
            # Core tuple: domain, purpose, capability must be present AND verified (per article)
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
                        if f in pdf.columns:
                            input_present_any = bool(_filled_series(pdf[f]).any())
                    except Exception:
                        input_present_any = False
                    ver_col = f"ver_tuple_{f}_verified"
                    verified_any = False
                    try:
                        if ver_col in pdf.columns:
                            verified_any = bool(pdf[ver_col].astype(bool).any())
                    except Exception:
                        verified_any = False
                    core_ok = core_ok and input_present_any and verified_any
            except Exception:
                core_ok = False
            out = {
                "article_id": _first("article_id"),
                "doc_any_component_verified": any_tuple,
                "ver_doc_best_ent": best_ent,
                "ver_doc_best_evidence_deployment_domain": field_best_evidence.get("deployment_domain"),
                "ver_doc_best_evidence_deployment_purpose": field_best_evidence.get("deployment_purpose"),
                "ver_doc_best_evidence_deployment_capability": field_best_evidence.get("deployment_capability"),
                "core_tuple_verified": bool(core_ok),
            }
            return pd.DataFrame([out])

        try:
            ver_docs_ds = ds_out.groupby("article_id").map_groups(_reduce_verify, batch_format="pandas")
        except Exception:
            ver_docs_ds = None

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
                ds_out.write_parquet(os.path.join(out_dir, "tuples_verification"))
            except Exception:
                pass
            try:
                if ver_docs_ds is not None:
                    ver_docs_ds.write_parquet(os.path.join(out_dir, "docs_verification"))
            except Exception:
                pass
        return ds_out

    # Pandas fallback
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


