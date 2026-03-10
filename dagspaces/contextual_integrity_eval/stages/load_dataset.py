"""Load and prepare the PrivacyLens benchmark dataset."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .benchmark_adapters import normalize_benchmark_frame


def load_dataset(
    hf_dataset: str = "SALT-NLP/PrivacyLens",
    hf_config: Optional[str] = None,
    split: str = "train",
    max_examples: int = 0,
    hf_token: Optional[str] = None,
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    """Load the PrivacyLens dataset from HuggingFace.

    Args:
        hf_dataset: HuggingFace dataset name.
        hf_config: Optional dataset configuration.
        split: Dataset split to load.
        max_examples: Max rows to load (0 = all).
        hf_token: Optional HuggingFace API token.
        sample_n: Optional number of rows to sample for debug runs.

    Returns:
        DataFrame with normalized columns (S, V, T, record_id, benchmark_name).
    """
    from datasets import load_dataset as hf_load

    load_kwargs = {}
    if hf_token:
        load_kwargs["token"] = hf_token

    ds = hf_load(hf_dataset, hf_config, split=split, **load_kwargs)
    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))
    df = ds.to_pandas()

    print(f"[load_dataset] Loaded {len(df)} rows from {hf_dataset} (split={split})")

    df = normalize_benchmark_frame(df, hf_dataset)
    if "split" not in df.columns:
        df["split"] = split

    if sample_n is not None and sample_n > 0 and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"[load_dataset] Sampled {sample_n} rows")

    print(f"[load_dataset] Final: {len(df)} rows, columns: {list(df.columns)}")
    return df
