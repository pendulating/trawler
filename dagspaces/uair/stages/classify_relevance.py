from __future__ import annotations

from typing import Any

import pandas as pd

from .classify import run_classification_stage as _run  # reuse underlying implementation
from .classify_shared import ensure_profile


def run_classification_relevance(df: Any, cfg) -> Any:
    """Relevance-only wrapper to avoid EU/RB behaviors.

    Ensures profile is 'relevance' and delegates to the existing classifier.
    """
    ensure_profile(cfg, "relevance")
    return _run(df, cfg)


