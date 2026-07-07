"""Load traversal cases from the three corpus tiers into a unified DataFrame.

Output columns: case_id, tier, practice_input, contaminated, gold_path.
Tier A gold JSON files double as both inputs (meta.practice_input) and
scoring targets (gold_path); Tier C comes from practices.yaml; Tier B from
generated parquet shards.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import List

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CORPUS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "corpus")


def _load_tier_a(corpus_root: str, include_contaminated: bool) -> List[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(corpus_root, "tier_a", "*.json"))):
        gold = json.load(open(path))
        meta = gold["meta"]
        practice_input = meta.get("practice_input") or ""
        if not practice_input:
            logger.warning(f"Tier A case {meta['case_id']} has no practice_input; skipping")
            continue
        if meta.get("contaminated", True) and not include_contaminated:
            logger.info(f"Excluding contaminated Tier A case {meta['case_id']} (held-out mode)")
            continue
        rows.append({
            "case_id": meta["case_id"],
            "tier": "a",
            "practice_input": practice_input,
            "contaminated": bool(meta.get("contaminated", True)),
            "gold_path": path,
        })
    return rows


def _load_tier_c(corpus_root: str) -> List[dict]:
    path = os.path.join(corpus_root, "tier_c", "practices.yaml")
    if not os.path.exists(path):
        return []
    doc = yaml.safe_load(open(path))
    return [
        {
            "case_id": f"tier_c_{p['id']}",
            "tier": "c",
            "practice_input": str(p["practice_input"]).strip(),
            "contaminated": False,
            "gold_path": "",
        }
        for p in doc.get("practices", [])
    ]


# Tier B label/flow columns carried through for scorers and the TP probe.
TIER_B_EXTRA_COLS = [
    "context_key", "departed_parameter", "prima_facie", "multi_tp",
    "sender_is_subject", "norm_statement", "flow_statement",
]


def _load_tier_b(corpus_root: str) -> List[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(corpus_root, "tier_b", "*.parquet"))):
        df = pd.read_parquet(path)
        for _, r in df.iterrows():
            row = {
                "case_id": str(r["case_id"]),
                "tier": "b",
                "practice_input": str(r["practice_input"]),
                "contaminated": False,
                "gold_path": path,
            }
            for col in TIER_B_EXTRA_COLS:
                if col in df.columns:
                    row[col] = r[col]
            rows.append(row)
    return rows


def load_cases(
    tiers: List[str],
    include_contaminated: bool = True,
    corpus_root: str | None = None,
    sample_n: int | None = None,
) -> pd.DataFrame:
    """Load cases for the requested tiers ("a", "b", "c")."""
    root = corpus_root or CORPUS_ROOT
    rows: List[dict] = []
    for tier in tiers:
        if tier == "a":
            rows.extend(_load_tier_a(root, include_contaminated))
        elif tier == "b":
            rows.extend(_load_tier_b(root))
        elif tier == "c":
            rows.extend(_load_tier_c(root))
        else:
            raise ValueError(f"Unknown tier {tier!r}; expected a/b/c")

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No cases loaded for tiers={tiers} from {root}")
    if df["case_id"].duplicated().any():
        dupes = df[df["case_id"].duplicated()]["case_id"].tolist()
        raise ValueError(f"Duplicate case_ids across tiers: {dupes}")

    if sample_n and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} cases (tiers={tiers}): {df['tier'].value_counts().to_dict()}")
    return df
