"""Shared utilities for historical_norms stage implementations."""

import json
from typing import Any, Dict, Optional, Sequence

import pandas as pd

try:
    from json_repair import repair_json
    _JSON_REPAIR_OK = True
except ImportError:
    _JSON_REPAIR_OK = False


def extract_json(gen_text: str) -> tuple[dict | None, str | None]:
    """Parse JSON from LLM output, with optional repair.

    Extracts the outermost ``{…}`` block from *gen_text*, parses it with
    :func:`json.loads`, and falls back to ``json_repair`` if available.

    Returns ``(parsed_dict, None)`` on success or ``(None, error_message)``
    on failure.
    """
    obj = None
    parse_error = None
    json_text = gen_text

    if "{" in gen_text:
        start = gen_text.find("{")
        end = gen_text.rfind("}") + 1
        if start < end:
            json_text = gen_text[start:end]

    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as e:
        parse_error = e
        if _JSON_REPAIR_OK:
            try:
                repaired = repair_json(json_text, return_objects=True)
                if isinstance(repaired, dict):
                    obj = repaired
                    parse_error = None
            except Exception as repair_err:
                parse_error = f"JSON repair failed: {repair_err}"
        else:
            parse_error = str(parse_error)

    return obj, parse_error


# Columns that commonly cause Arrow serialization failures across all stages.
_BASE_PROBLEMATIC_COLS = [
    "metadata", "__inference_error__", "embeddings",
]


def clean_for_parquet(
    df: pd.DataFrame,
    extra_cols: Sequence[str] = (),
    stage_name: str = "stage",
) -> pd.DataFrame:
    """Clean a DataFrame to avoid PyArrow serialization issues.

    Removes or JSON-serializes columns that cause parquet write errors
    (empty structs, complex nested types).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to clean (modified in-place where possible).
    extra_cols : Sequence[str]
        Stage-specific column names to check in addition to the base set.
    stage_name : str
        Label used in log messages.
    """
    problematic_cols = list(_BASE_PROBLEMATIC_COLS) + list(extra_cols)

    for col in problematic_cols:
        if col not in df.columns:
            continue
        try:
            sample = df[col].dropna().head(1)
            if len(sample) > 0:
                val = sample.iloc[0]
                if val == {} or val == [] or (isinstance(val, list) and all(v == {} for v in val)):
                    df = df.drop(columns=[col])
                    print(f"[{stage_name}] Dropped empty struct column: {col}")
                    continue
        except Exception:
            pass
        try:
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
        except Exception:
            df = df.drop(columns=[col])
            print(f"[{stage_name}] Dropped problematic column: {col}")

    return df
