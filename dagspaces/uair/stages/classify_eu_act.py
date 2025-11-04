from __future__ import annotations

from typing import Any

import pandas as pd

from .classify import run_classification_stage as _run
from .classify_shared import ensure_profile, inject_prompt_from_file, dedupe_by_article_id


def run_classification_eu_act(df: Any, cfg) -> Any:
    """EU AI Act wrapper with per-article dedupe.

    Assumes caller performs gating by core_tuple_verified when needed.
    """
    ensure_profile(cfg, "eu_ai_act")
    inject_prompt_from_file(cfg, "eu_ai_act_classification.yaml")
    in_df = df
    if hasattr(df, "to_pandas"):
        in_df = df.to_pandas()
    if isinstance(in_df, pd.DataFrame):
        in_df = dedupe_by_article_id(in_df)
    return _run(in_df, cfg)


