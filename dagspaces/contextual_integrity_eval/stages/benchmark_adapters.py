"""Benchmark normalization adapters for CI evaluation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BenchmarkSchema:
    benchmark_name: str
    id_col: str
    seed_col: str
    vignette_col: str
    trajectory_col: str


SCHEMAS: dict[str, BenchmarkSchema] = {
    "SALT-NLP/PrivacyLens": BenchmarkSchema("PrivacyLens", "record_id", "S", "V", "T"),
    "GoldCoin-HIPAA": BenchmarkSchema("GoldCoin-HIPAA", "record_id", "S", "V", "T"),
    "VLM-GeoPrivacy": BenchmarkSchema("VLM-GeoPrivacy", "record_id", "S", "V", "T"),
    "CIMemories": BenchmarkSchema("CIMemories", "record_id", "S", "V", "T"),
    "facebook/CIMemories": BenchmarkSchema("CIMemories", "record_id", "S", "V", "T"),
}


def normalize_benchmark_frame(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Normalize any benchmark frame into shared evaluator columns."""
    schema = SCHEMAS.get(dataset_name, SCHEMAS["SALT-NLP/PrivacyLens"])
    out = df.copy()
    aliases: dict[str, list[str]] = {
        "S": ["S", "seed", "scenario", "case_seed"],
        "V": ["V", "vignette", "case_text", "narrative_vignette"],
        "T": ["T", "trajectory", "agent_trajectory", "actions"],
        "record_id": ["record_id", "id", "uid", "example_id"],
    }
    for canonical, candidates in aliases.items():
        if canonical in out.columns:
            continue
        for candidate in candidates:
            if candidate in out.columns:
                out[canonical] = out[candidate]
                break
        if canonical not in out.columns:
            if canonical == "record_id":
                out[canonical] = out.index.astype(str)
            elif canonical == "T" and "V" in out.columns:
                # PrivacyLens releases may omit explicit trajectory; use vignette/story text as trajectory proxy.
                out[canonical] = out["V"]
            else:
                out[canonical] = ""

    out["benchmark_name"] = schema.benchmark_name
    return out

