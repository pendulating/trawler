"""Load the google/simpleqa-verified dataset from HuggingFace.

Mirrors the pattern in :mod:`dagspaces.privacylens.stages.load_dataset`:
hf_load → pandas → normalize to a unified schema → optional sample.

Output columns:
    question_id (int)   — stable index for joins (0..N-1 by row order)
    question    (str)   — the input question shown to the model
    gold_answer (str)   — reference answer used by the grader
    topic       (str)   — optional topic metadata; "" if absent
    split       (str)   — split name, propagated for provenance
"""

from __future__ import annotations

import pandas as pd

# Canonical column aliases on google/simpleqa-verified (and OpenAI's
# upstream SimpleQA). We accept any of these and normalize to the unified
# schema above. Add new aliases here when HF schema drifts rather than
# patching downstream stages.
_QUESTION_ALIASES = ("problem", "question", "Question")
_ANSWER_ALIASES = ("answer", "Answer", "gold_target", "target")


def _pick_first_column(df: pd.DataFrame, aliases: tuple) -> str | None:
    for name in aliases:
        if name in df.columns:
            return name
    return None


def _extract_topic(row: pd.Series) -> str:
    """Best-effort extraction of a topic string from row metadata."""
    if "topic" in row.index and pd.notna(row["topic"]):
        return str(row["topic"])
    md = row.get("metadata") if "metadata" in row.index else None
    if isinstance(md, dict):
        for key in ("topic", "category", "domain"):
            val = md.get(key)
            if val:
                return str(val)
    return ""


def load_dataset(
    hf_dataset: str = "google/simpleqa-verified",
    hf_config: str | None = None,
    split: str = "eval",
    hf_token: str | None = None,
    sample_n: int | None = None,
) -> pd.DataFrame:
    """Pull simpleqa-verified, normalize columns, optionally sample.

    Args:
        hf_dataset: HuggingFace dataset id.
        hf_config: Optional dataset configuration.
        split: Split to load. The dataset card lists ``eval`` as the
            canonical split; we keep this overridable so a schema drift
            (eval → test → validation) is a one-line cfg fix instead of
            a code change.
        hf_token: Optional HF API token (e.g., for gated datasets).
        sample_n: If positive and < N, sample this many rows (seeded).
    """
    from datasets import load_dataset as hf_load

    load_kwargs = {}
    if hf_token:
        load_kwargs["token"] = hf_token

    ds = hf_load(hf_dataset, hf_config, split=split, **load_kwargs)
    df = ds.to_pandas()
    print(
        f"[load_dataset] Loaded {len(df)} rows from {hf_dataset} "
        f"(split={split}); columns: {list(df.columns)}",
        flush=True,
    )

    q_col = _pick_first_column(df, _QUESTION_ALIASES)
    a_col = _pick_first_column(df, _ANSWER_ALIASES)
    if q_col is None or a_col is None:
        raise ValueError(
            f"[load_dataset] {hf_dataset} schema missing question/answer columns. "
            f"Expected one of question={_QUESTION_ALIASES} and answer={_ANSWER_ALIASES}; "
            f"actual columns: {list(df.columns)}. Add the new alias to "
            f"_QUESTION_ALIASES / _ANSWER_ALIASES in stages/load_dataset.py."
        )

    out = pd.DataFrame({
        "question_id": range(len(df)),
        "question": df[q_col].astype(str),
        "gold_answer": df[a_col].astype(str),
        "topic": df.apply(_extract_topic, axis=1),
        "split": split,
    })

    if sample_n is not None and sample_n > 0 and sample_n < len(out):
        out = out.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"[load_dataset] Sampled {sample_n} rows", flush=True)

    print(
        f"[load_dataset] Final: {len(out)} rows | "
        f"unique topics: {out['topic'].nunique()}",
        flush=True,
    )
    return out
