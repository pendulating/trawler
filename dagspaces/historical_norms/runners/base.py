"""Base classes and shared utilities for stage runners."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import pandas as pd

# Re-export StageRunner from common module for backward compatibility
from dagspaces.common.runners.base import StageRunner

if TYPE_CHECKING:
    from ..orchestrator import StageExecutionContext, StageResult

__all__ = ["StageRunner", "_compute_doc_level_verification"]


def _compute_doc_level_verification(
    out: pd.DataFrame,
    results_path: Optional[str]
) -> Optional[pd.DataFrame]:
    """Compute document-level verification aggregation from row-level results.
    
    Returns:
        DataFrame with columns: article_id, doc_any_component_verified, core_tuple_verified
        or None if computation fails
    """
    import pandas as _pd
    
    # Preferred: read docs_verification written by stage implementation (if present)
    docs_df = None
    try:
        if results_path:
            out_dir = os.path.dirname(results_path)
            cand_file = os.path.join(out_dir, "docs_verification.parquet")
            cand_dir = os.path.join(out_dir, "docs_verification")
            if os.path.exists(cand_file):
                docs_df = _pd.read_parquet(cand_file)
            elif os.path.isdir(cand_dir):
                docs_df = _pd.read_parquet(cand_dir)
    except Exception:
        docs_df = None
    
    # Fallback: compute simple doc-level view from the results DataFrame
    if docs_df is None and "article_id" in out.columns:
        try:
            def _reduce(df_in: _pd.DataFrame) -> _pd.DataFrame:
                if df_in.empty:
                    return _pd.DataFrame([])
                any_tuple = bool(df_in.get("ver_tuples_any_verified", _pd.Series([], dtype=bool)).any())
                # core tuple verified: require per-field verified flags
                core_ok = True
                for f in ("deployment_domain", "deployment_purpose", "deployment_capability"):
                    col = f"ver_tuple_{f}_verified"
                    if col in df_in.columns:
                        try:
                            v_any = bool(df_in[col].astype(bool).any())
                        except Exception:
                            v_any = False
                        core_ok = core_ok and v_any
                    else:
                        core_ok = False
                return _pd.DataFrame([
                    {
                        "article_id": df_in.get("article_id", _pd.Series([None])).iloc[0],
                        "doc_any_component_verified": any_tuple,
                        "core_tuple_verified": bool(core_ok),
                    }
                ])
            docs_df = out.groupby("article_id", dropna=False).apply(_reduce).reset_index(drop=True)
        except Exception:
            docs_df = None
    
    return docs_df

