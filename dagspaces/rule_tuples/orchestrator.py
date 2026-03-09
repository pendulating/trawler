from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import json
import pandas as pd
from datetime import datetime
import platform
import socket
import sys
import threading
import time as _time


def _load_parquet_dataset(parquet_path: str, columns: Dict[str, str], debug: bool, sample_n: Optional[int]) -> pd.DataFrame:
    if not isinstance(parquet_path, str) or not parquet_path:
        raise ValueError("data.parquet_path is required")
    if not os.path.isabs(parquet_path):
        # Resolve relative to repo root if possible
        parquet_path = os.path.abspath(parquet_path)
    df = pd.read_parquet(parquet_path)
    # Ensure required columns exist; coerce types sanely
    required = [
        columns.get("name", "name"),
        columns.get("public_description", "public_description"),
        columns.get("subscribers", "subscribers"),
        columns.get("rule_text", "rule_text"),
        columns.get("rule_index", "rule_index"),
        columns.get("total_rules_count", "total_rules_count"),
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Parquet missing expected columns: {missing}")
    # Normalize a minimal working set
    def _safe_str(x: Any) -> str:
        try:
            return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x).strip()
        except Exception:
            return str(x) if x is not None else ""
    df[columns.get("name", "name")] = df[columns.get("name", "name")].apply(_safe_str)
    df[columns.get("public_description", "public_description")] = df[columns.get("public_description", "public_description")].apply(_safe_str)
    # Keep subscribers as string for now; downstream can coerce
    df[columns.get("subscribers", "subscribers")] = df[columns.get("subscribers", "subscribers")].apply(_safe_str)
    # Subset and rename to canonical names used in stages
    df = df.rename(columns={
        columns.get("name", "name"): "name",
        columns.get("public_description", "public_description"): "public_description",
        columns.get("subscribers", "subscribers"): "subscribers",
        columns.get("rule_text", "rule_text"): "rule_text",
        columns.get("rule_index", "rule_index"): "rule_index",
        columns.get("total_rules_count", "total_rules_count"): "total_rules_count",
    })
    if debug and isinstance(sample_n, int) and sample_n > 0:
        df = df.head(sample_n)
    return df


def run_experiment(cfg) -> None:
    """Entry point for the experiment.

    Current skeleton:
    - Loads flattened rules parquet into a pandas DataFrame
    - Routes by stage: classify | decompose | pipeline (pipeline = classify then decompose subset)
    - LLM wiring to be added in subsequent steps
    """
    stage = str(getattr(cfg.runtime, "stage", "classify")).strip().lower()
    debug = bool(getattr(cfg.runtime, "debug", True))
    sample_n = getattr(cfg.runtime, "sample_n", 100)

    parquet_path = str(getattr(cfg.data, "parquet_path"))
    columns = dict(getattr(cfg.data, "columns", {}))

    df = _load_parquet_dataset(parquet_path, columns, debug=debug, sample_n=sample_n)

    # Optional W&B init
    use_wandb = False
    try:
        use_wandb = bool(getattr(cfg.wandb, "enabled", False))
    except Exception:
        use_wandb = False
    wb = None
    if use_wandb:
        try:
            import wandb as _wandb
            wb = _wandb
            run_name = f"{getattr(cfg.experiment, 'name', 'UAIR')}-{stage}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            # Build rich experiment config
            try:
                model_source = str(getattr(cfg.model, "model_source", ""))
            except Exception:
                model_source = ""
            try:
                engine_kwargs = dict(getattr(cfg.model, "engine_kwargs", {}))
            except Exception:
                engine_kwargs = {}
            try:
                batch_size = int(getattr(cfg.model, "batch_size", getattr(cfg, "model_runtime", {}).get("batch_size", 16)))
            except Exception:
                batch_size = 16
            try:
                concurrency = int(getattr(cfg.model, "concurrency", getattr(cfg, "model_runtime", {}).get("concurrency", 1)))
            except Exception:
                concurrency = 1
            classify_sp = dict(getattr(cfg, "sampling_params", {}))
            decomp_sp = dict(getattr(cfg, "sampling_params", {}))
            # Prompts
            classify_sys = str(getattr(cfg.prompt, "system_prompt", ""))
            classify_tpl = str(getattr(cfg.prompt, "prompt_template", ""))
            try:
                decomp_sys = str(getattr(cfg.prompt_decompose, "system_prompt", ""))
                decomp_tpl = str(getattr(cfg.prompt_decompose, "prompt_template", ""))
            except Exception:
                decomp_sys = ""
                decomp_tpl = ""
            run_config = {
                "stage": stage,
                "parquet_path": parquet_path,
                "data_columns_map": columns,
                "sample_n": sample_n,
                "output_csv": str(getattr(cfg.runtime, "output_csv", "") or ""),
                "use_llm_classify": bool(getattr(cfg.runtime, "use_llm_classify", False)),
                "use_llm_decompose": bool(getattr(cfg.runtime, "use_llm_decompose", False)),
                "guided_decoding_decompose": bool(getattr(cfg.runtime, "guided_decoding_decompose", False)),
                "max_errored_blocks": int(getattr(cfg.runtime, "max_errored_blocks", 0) or 0),
                # Model/runtime
                "model_source": model_source,
                "engine_kwargs": engine_kwargs,
                "batch_size": batch_size,
                "concurrency": concurrency,
                "sampling_params": dict(getattr(cfg, "sampling_params", {})),
                # Prompts
                "classify_system_prompt": classify_sys,
                "classify_prompt_template": classify_tpl,
                "decompose_system_prompt": decomp_sys,
                "decompose_prompt_template": decomp_tpl,
                # System metadata
                "python_version": sys.version.split()[0],
                "os": platform.platform(),
                "hostname": socket.gethostname(),
            }
            try:
                proj = str(getattr(cfg.wandb, "project", "UAIR") or "UAIR")
            except Exception:
                proj = "UAIR"
            try:
                ent = getattr(cfg.wandb, "entity", None)
                ent = str(ent) if (ent is not None and str(ent).strip() != "") else None
            except Exception:
                ent = None
            wb.init(project=proj, entity=ent, job_type=stage, name=run_name, config=run_config)
            try:
                wb.log({"dataset/loaded_rows": int(len(df))})
            except Exception:
                pass
            try:
                wb.log({"status/started": 1})
            except Exception:
                pass
        except Exception:
            wb = None

    def _wb_log(data: Dict[str, Any]) -> None:
        if wb is not None:
            try:
                wb.log(data)
            except Exception:
                pass

    def _wb_log_table(df: pd.DataFrame, key: str, max_rows: int = 1000) -> None:
        if wb is None:
            return
        try:
            cols_pref = [
                "name", "public_description", "subscribers", "rule_index", "rule_text",
                "is_relevant", "llm_output",
                "ci_subject", "ci_sender", "ci_receiver", "ci_information", "ci_transmission_principle", "ci_missing_elements",
            ]
            cols = [c for c in cols_pref if c in df.columns]
            if not cols:
                # Fallback: take first N columns to ensure something is logged
                cols = list(df.columns)[:12]
            table = wb.Table(columns=cols, log_mode="MUTABLE")
            def _to_str(v: Any) -> str:
                try:
                    import json as _json
                    if isinstance(v, (dict, list, tuple)):
                        return _json.dumps(v, ensure_ascii=False)
                except Exception:
                    pass
                try:
                    return str(v) if v is not None else ""
                except Exception:
                    return ""
            # Fixed-seed random sampling when exceeding max_rows
            try:
                total_rows = int(len(df))
            except Exception:
                total_rows = None
            sample_n = int(max_rows)
            seed = 777
            try:
                env_seed = os.environ.get("UAIR_WB_TABLE_SEED") or os.environ.get("UAIR_TABLE_SAMPLE_SEED")
                if env_seed is not None:
                    seed = int(env_seed)
            except Exception:
                seed = 777
            try:
                if total_rows is not None and total_rows > sample_n:
                    df_iter = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
                else:
                    df_iter = df.reset_index(drop=True)
            except Exception:
                df_iter = df.reset_index(drop=True).head(sample_n)
            n = 0
            for _, r in df_iter.iterrows():
                table.add_data(*[_to_str(r.get(c)) for c in cols])
                n += 1
            wb.log({key: table, f"{key}/rows": n, f"{key}/total_rows": (total_rows if total_rows is not None else n)})
        except Exception:
            pass

    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
        print(json.dumps({
            "loaded_rows": int(len(df)),
            "stage": stage,
            "columns": list(df.columns)[:12],
            "streaming": False,
        }, indent=2))

    if stage == "classify":
        from .stages.classify import run_classification_stage
        out_df = run_classification_stage(df, cfg)
        out_path = str(getattr(cfg.runtime, "output_csv", "") or "")
        try:
            if "is_relevant" in out_df.columns:
                rel_count = int(out_df["is_relevant"].astype(bool).sum())
                total = int(len(out_df))
                ratio = float(rel_count) / float(total) if total > 0 else 0.0
                avg_lat = None
                try:
                    lat = [float(v) for v in out_df.get("latency_s", []).tolist() if isinstance(v, (int, float))]
                    avg_lat = (sum(lat) / len(lat)) if lat else None
                except Exception:
                    pass
                _wb_log({
                    "classify/rows": total,
                    "classify/relevant_count": rel_count,
                    "classify/relevant_ratio": ratio,
                    "classify/avg_latency_s": avg_lat,
                })
        except Exception:
            pass
        if out_path:
            try:
                out_df.to_csv(out_path, index=False)
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"classified_rows": int(len(out_df)), "output_csv": out_path}, indent=2))
            except Exception:
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"classified_rows": int(len(out_df)), "output_csv": None}, indent=2))
        else:
            if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                print(json.dumps({"classified_rows": int(len(out_df))}, indent=2))
        _wb_log_table(out_df, key="inspection/classify")
        return

    if stage == "decompose":
        from .stages.decompose import run_decomposition_stage
        out_df = run_decomposition_stage(df, cfg)
        out_path = str(getattr(cfg.runtime, "output_csv", "") or "")
        try:
            total = int(len(out_df))
            have_any = int((out_df[[
                "ci_subject","ci_sender","ci_receiver","ci_information","ci_transmission_principle"
            ]].notna().any(axis=1)).sum()) if total > 0 else 0
            _wb_log({
                "decompose/rows": total,
                "decompose/any_tuple_present": have_any,
            })
        except Exception:
            pass
        if out_path:
            try:
                out_df.to_csv(out_path, index=False)
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"decomposed_rows": int(len(out_df)), "output_csv": out_path}, indent=2))
            except Exception:
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"decomposed_rows": int(len(out_df)), "output_csv": None}, indent=2))
        else:
            if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                print(json.dumps({"decomposed_rows": int(len(out_df))}, indent=2))
        _wb_log_table(out_df, key="inspection/decompose")
        return

    if stage == "pipeline":
        from .stages.classify import run_classification_stage
        from .stages.decompose import run_decomposition_stage
        cls_df = run_classification_stage(df, cfg)
        # Filter to relevant
        rel_df = cls_df[cls_df.get("is_relevant", False) == True] if "is_relevant" in cls_df.columns else cls_df
        # Log classification metrics in pipeline stage as well
        try:
            if "is_relevant" in cls_df.columns:
                rel_count_p = int(cls_df["is_relevant"].astype(bool).sum())
                total_p = int(len(cls_df))
                ratio_p = float(rel_count_p) / float(total_p) if total_p > 0 else 0.0
                avg_lat_p = None
                try:
                    lat_p = [float(v) for v in cls_df.get("latency_s", []).tolist() if isinstance(v, (int, float))]
                    avg_lat_p = (sum(lat_p) / len(lat_p)) if lat_p else None
                except Exception:
                    pass
                _wb_log({
                    "classify/rows": total_p,
                    "classify/relevant_count": rel_count_p,
                    "classify/relevant_ratio": ratio_p,
                    "classify/avg_latency_s": avg_lat_p,
                })
                _wb_log_table(cls_df, key="inspection/classify")
        except Exception:
            pass
        dec_df = run_decomposition_stage(rel_df, cfg)
        out_path = str(getattr(cfg.runtime, "output_csv", "") or "")
        try:
            total = int(len(df))
            rel = int(len(rel_df))
            dec = int(len(dec_df))
            ratio = float(rel) / float(total) if total > 0 else 0.0
            _wb_log({
                "pipeline/rows_in": total,
                "pipeline/relevant": rel,
                "pipeline/decomposed": dec,
                "pipeline/relevant_ratio": ratio,
            })
        except Exception:
            pass
        if out_path:
            try:
                dec_df.to_csv(out_path, index=False)
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({
                        "pipeline_rows_in": int(len(df)),
                        "pipeline_relevant": int(len(rel_df)),
                        "pipeline_decomposed": int(len(dec_df)),
                        "output_csv": out_path,
                    }, indent=2))
            except Exception:
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({
                        "pipeline_rows_in": int(len(df)),
                        "pipeline_relevant": int(len(rel_df)),
                        "pipeline_decomposed": int(len(dec_df)),
                        "output_csv": None,
                    }, indent=2))
        else:
            if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                print(json.dumps({
                    "pipeline_rows_in": int(len(df)),
                    "pipeline_relevant": int(len(rel_df)),
                    "pipeline_decomposed": int(len(dec_df)),
                }, indent=2))
        _wb_log_table(dec_df, key="inspection/pipeline")
        return

    raise ValueError(f"Unknown runtime.stage: {stage}")
