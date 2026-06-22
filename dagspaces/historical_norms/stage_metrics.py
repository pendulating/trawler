"""Per-stage data-quality scalars for W&B (2026-06-09 logging review).

Every extraction stage already logs a sampled table; this module adds the
numbers a reader actually wants on the run page: parse-failure rates,
label distributions, and chunk statistics. One flat dict per stage, all
keys namespaced ``data_quality/``, all values bounded scalars — verbose
enough to audit a run from W&B alone, with zero per-row noise.

A ``chunk_len_max`` over budget here would have surfaced the 2026-06-09
chunk-overflow finding (F3) the day the corpus was built.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

_PREFIX = "data_quality"


def _frac(series: pd.Series) -> float:
    """Fraction of truthy values, ignoring nulls in the denominator."""
    s = series.dropna()
    if len(s) == 0:
        return 0.0
    return float(s.astype(bool).mean())


def _error_rate(df: pd.DataFrame, col: str) -> float:
    """Fraction of rows whose error column holds a real error value."""
    s = df[col]
    return float((s.notna() & (s.astype(str).str.strip() != "")
                  & (s.astype(str) != "None") & (s.astype(str) != "False")).mean())


def _value_fracs(df: pd.DataFrame, col: str, values: tuple) -> Dict[str, float]:
    s = df[col].dropna().astype(str).str.strip().str.lower()
    n = max(len(s), 1)
    return {f"{_PREFIX}/{col}_{v}_frac": float((s == v).sum()) / n for v in values}


def _len_stats(series: pd.Series, key: str) -> Dict[str, float]:
    lens = series.dropna().astype(str).map(len)
    if lens.empty:
        return {}
    return {
        f"{_PREFIX}/{key}_mean": round(float(lens.mean()), 1),
        f"{_PREFIX}/{key}_p50": float(lens.quantile(0.50)),
        f"{_PREFIX}/{key}_p95": float(lens.quantile(0.95)),
        f"{_PREFIX}/{key}_max": float(lens.max()),
    }


def compute_stage_quality_metrics(stage: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Quality scalars for one stage's output dataframe.

    Unknown stages return the base counts only; missing columns are
    skipped, never raised — logging must not fail a pipeline.
    """
    m: Dict[str, Any] = {f"{_PREFIX}/rows": len(df)}
    if df.empty:
        return m
    if "gutenberg_id" in df.columns:
        m[f"{_PREFIX}/books"] = int(df["gutenberg_id"].nunique())
        per_book = df.groupby("gutenberg_id").size()
        m[f"{_PREFIX}/rows_per_book_min"] = int(per_book.min())
        m[f"{_PREFIX}/rows_per_book_max"] = int(per_book.max())

    if stage == "fetch_gutenberg":
        if "article_text" in df.columns:
            m.update(_len_stats(df["article_text"], "chunk_len"))

    elif stage == "ci_reasoning":
        if "ci_reasoning_parse_error" in df.columns:
            m[f"{_PREFIX}/parse_error_rate"] = _error_rate(df, "ci_reasoning_parse_error")
        if "has_information_exchange" in df.columns:
            m[f"{_PREFIX}/has_exchange_rate"] = _frac(df["has_information_exchange"])
        if "ci_flow_count" in df.columns:
            fc = pd.to_numeric(df["ci_flow_count"], errors="coerce").dropna()
            if not fc.empty:
                m[f"{_PREFIX}/flows_per_chunk_mean"] = round(float(fc.mean()), 3)
                m[f"{_PREFIX}/flows_per_chunk_max"] = float(fc.max())
                m[f"{_PREFIX}/zero_flow_frac"] = float((fc == 0).mean())

    elif stage == "ci_extraction":
        if "extraction_error" in df.columns:
            m[f"{_PREFIX}/extraction_error_rate"] = _error_rate(df, "extraction_error")
        if "ci_appropriateness" in df.columns:
            m.update(_value_fracs(df, "ci_appropriateness",
                                  ("appropriate", "inappropriate", "ambiguous")))
        if "flow_quality_passed" in df.columns:
            m[f"{_PREFIX}/flow_quality_passed_rate"] = _frac(df["flow_quality_passed"])

    elif stage == "norm_reasoning":
        if "reasoning_error" in df.columns:
            m[f"{_PREFIX}/parse_error_rate"] = _error_rate(df, "reasoning_error")
        if "has_prescriptive_content" in df.columns:
            m[f"{_PREFIX}/prescriptive_content_rate"] = _frac(df["has_prescriptive_content"])
        if "norm_count" in df.columns:
            nc = pd.to_numeric(df["norm_count"], errors="coerce").dropna()
            if not nc.empty:
                m[f"{_PREFIX}/norms_per_chunk_mean"] = round(float(nc.mean()), 3)
                m[f"{_PREFIX}/norms_per_chunk_max"] = float(nc.max())

    elif stage in ("norm_extraction", "norm_role_abstraction"):
        if "extraction_failed" in df.columns:
            m[f"{_PREFIX}/extraction_error_rate"] = _frac(df["extraction_failed"])
        if "role_abstraction_failed" in df.columns:
            m[f"{_PREFIX}/role_abstraction_error_rate"] = _frac(df["role_abstraction_failed"])
        if "norm_quality_passed" in df.columns:
            m[f"{_PREFIX}/norm_quality_passed_rate"] = _frac(df["norm_quality_passed"])
        if "raz_governs_info_flow" in df.columns:
            m[f"{_PREFIX}/governs_info_flow_rate"] = _frac(df["raz_governs_info_flow"])
        if "raz_normative_force" in df.columns:
            m.update(_value_fracs(
                df, "raz_normative_force",
                ("obligatory", "prohibited", "permitted", "recommended", "discouraged"),
            ))
        if "raz_confidence_quant" in df.columns:
            cq = pd.to_numeric(df["raz_confidence_quant"], errors="coerce").dropna()
            if not cq.empty:
                m[f"{_PREFIX}/confidence_mean"] = round(float(cq.mean()), 3)

    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
