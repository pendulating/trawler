"""Load and prepare the GoldCoin HIPAA test dataset."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def load_dataset(
    csv_path: str,
    task: str,
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    """Load a GoldCoin HIPAA test CSV and add ground truth labels.

    Args:
        csv_path: Path to test CSV (e.g. test_real_cases_hipaa_compliance.csv).
        task: "compliance" or "applicability".
        sample_n: Optional number of rows to sample.

    Returns:
        DataFrame with columns including generate_background, ground_truth, case_id.
    """
    df = pd.read_csv(csv_path)
    print(f"[load_dataset] Loaded {len(df)} rows from {csv_path}")

    # Add case_id
    df["case_id"] = range(len(df))

    # Add ground_truth column
    if task == "compliance":
        # Direct mapping: "Permit" or "Forbid"
        df["ground_truth"] = df["generate_HIPAA_type"].str.strip()
    elif task == "applicability":
        # Permit/Forbid -> "Applicable", "Not Applicable" stays
        def map_applicability(val):
            val = str(val).strip()
            if val.lower() in ("permit", "forbid"):
                return "Applicable"
            elif val.lower() == "not applicable":
                return "Not Applicable"
            return val
        df["ground_truth"] = df["generate_HIPAA_type"].apply(map_applicability)
    else:
        raise ValueError(f"Unknown task: {task!r}")

    # Normalize ground truth to title case
    df["ground_truth"] = df["ground_truth"].str.strip()

    print(f"[load_dataset] Task={task}, label distribution:")
    print(df["ground_truth"].value_counts().to_string())

    if sample_n is not None and sample_n > 0 and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"[load_dataset] Sampled {sample_n} rows")

    return df
