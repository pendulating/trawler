"""Verification NBL stage runner."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _convert_to_pandas_if_needed,
    _safe_log_table,
    prepare_stage_input,
)
from ..stages.verify_nbl import run_verification_stage_nbl
from .base import StageRunner, _compute_doc_level_verification


class VerificationNBLRunner(StageRunner):
    """Runner for the verify_nbl stage."""
    
    stage_name = "verify_nbl"

    def run(self, context: StageExecutionContext) -> StageResult:
        """Execute the verify_nbl stage."""
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        out = run_verification_stage_nbl(in_obj, cfg)
        
        out = _convert_to_pandas_if_needed(out)
        
        # Prepare optional doc-level aggregation and merge core flags into row-level output
        docs_df = None
        if isinstance(out, pd.DataFrame):
            results_path = context.output_paths.get("results")
            docs_df = _compute_doc_level_verification(out, results_path)

            # Merge core flags into row-level frame for downstream gating
            if docs_df is not None and "article_id" in out.columns and "article_id" in docs_df.columns:
                try:
                    out = out.merge(docs_df[["article_id", "core_tuple_verified"]], on="article_id", how="left")
                except Exception:
                    pass

        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            for output_name, output_path in context.output_paths.items():
                if output_name == "docs":
                    # Emit doc-level table if available
                    try:
                        if docs_df is not None and len(docs_df):
                            docs_df.to_parquet(output_path, index=False)
                        else:
                            # If not available, synthesize minimal docs view with article_id and core flag
                            if "article_id" in out.columns and "core_tuple_verified" in out.columns:
                                tmp_docs = out[["article_id", "core_tuple_verified"]].drop_duplicates()
                                tmp_docs.to_parquet(output_path, index=False)
                    except Exception:
                        pass
                else:
                    out.to_parquet(output_path, index=False)
        
        # Log verify_nbl results table with all columns (logger handles filtering of heavy/internal)
        _safe_log_table(context.logger, out, "verify_nbl/results", prefer_cols=None, panel_group="inspect_results")
        
        # Log document-level aggregation in inspect_results; build from memory if needed
        if docs_df is None and isinstance(out, pd.DataFrame) and "article_id" in out.columns:
            try:
                import pandas as _pd
                docs_df = out[["article_id", "core_tuple_verified"]].drop_duplicates()
            except Exception:
                docs_df = None

        if docs_df is not None and len(docs_df):
            _safe_log_table(context.logger, docs_df, "verify_nbl/docs", prefer_cols=None, panel_group="inspect_results")

        # Plot run-level verification rates by input and doc-level
        try:
            import pandas as _pd
            import matplotlib.pyplot as _plt  # type: ignore

            df_res = out
            df_docs = docs_df if docs_df is not None else None

            # Helper: presence check for scalars/lists
            def _present_series(s: _pd.Series) -> _pd.Series:
                try:
                    ss = s.astype(str).str.strip()
                    ssl = ss.str.lower()
                    not_empty = ss != ""
                    not_none = ~ssl.isin(["none", "null"])
                    not_empty_json = ~ss.isin(["[]", "{}"])
                    return not_empty & not_none & not_empty_json
                except Exception:
                    return _pd.Series([False] * len(s))

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
            list_fields = [
                "list_of_harms_that_occurred",
                "list_of_risks_that_occurred",
                "list_of_benefits_that_occurred",
            ]

            labels = []
            percents = []
            # Scalars: percent verified among present inputs
            for f in scalar_fields:
                try:
                    if f in df_res.columns:
                        present = _present_series(df_res[f])
                        ver_col = f"ver_tuple_{f}_verified"
                        if ver_col in df_res.columns:
                            verified = present & df_res[ver_col].astype(bool)
                            present_n = int(present.sum())
                            verified_n = int(verified.sum())
                            pct = (verified_n / present_n * 100.0) if present_n > 0 else None
                            labels.append(f)
                            percents.append(pct)
                except Exception:
                    pass
            # Lists: percent any verified among present lists
            for f in list_fields:
                try:
                    if f in df_res.columns:
                        present = _present_series(df_res[f])
                        any_col = f"ver_tuple_{f}_any"
                        if any_col in df_res.columns:
                            verified = present & df_res[any_col].astype(bool)
                            present_n = int(present.sum())
                            verified_n = int(verified.sum())
                            pct = (verified_n / present_n * 100.0) if present_n > 0 else None
                            labels.append(f)
                            percents.append(pct)
                except Exception:
                    pass

            # Build bar plot for input percentages (filter out None)
            try:
                labels_plot = []
                percents_plot = []
                for l, p in zip(labels, percents):
                    if p is not None:
                        labels_plot.append(l)
                        percents_plot.append(p)
                if labels_plot:
                    fig, ax = _plt.subplots(figsize=(10, max(3, int(len(labels_plot) * 0.4))))
                    ax.barh(labels_plot, percents_plot, color="#4e79a7")
                    ax.set_xlabel("Percent verified (%)")
                    ax.set_title("Verification rate by input")
                    for i, v in enumerate(percents_plot):
                        ax.text(v + 1, i, f"{v:.1f}%", va='center')
                    _plt.tight_layout()
                    context.logger.log_plot("verify_nbl/percent_verified_by_input", fig)
            except Exception as e:
                print(f"Warning: failed to log percent_verified_by_input plot: {e}", flush=True)

            # Doc-level percentages
            try:
                if df_docs is not None and len(df_docs):
                    vals = {}
                    for key in ("doc_any_component_verified", "core_tuple_verified"):
                        if key in df_docs.columns:
                            try:
                                vals[key] = float(df_docs[key].astype(bool).mean() * 100.0)
                            except Exception:
                                pass
                    if vals:
                        fig2, ax2 = _plt.subplots(figsize=(6, 3))
                        keys = list(vals.keys())
                        vals_list = [vals[k] for k in keys]
                        ax2.bar(keys, vals_list, color="#59a14f")
                        ax2.set_ylabel("Percent of articles (%)")
                        ax2.set_title("Doc-level verification rates")
                        for i, v in enumerate(vals_list):
                            ax2.text(i, v + 1, f"{v:.1f}%", ha='center')
                        _plt.tight_layout()
                        context.logger.log_plot("verify_nbl/doc_level_percentages", fig2)
            except Exception as e:
                print(f"Warning: failed to log doc_level_percentages plot: {e}", flush=True)
        except Exception as e:
            print(f"Warning: verify_nbl run-level plots error: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)

