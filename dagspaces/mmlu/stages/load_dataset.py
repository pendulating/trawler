"""Load cais/mmlu and (optionally) cache the dev-split few-shot exemplars.

Output schema (parquet):
    question_id (int)   — stable 0..N-1 row index
    subject     (str)   — MMLU subject slug
    question    (str)   — MCQ stem
    choices     (list)  — 4 strings in [A, B, C, D] order
    answer      (int)   — correct index 0-3
    answer_letter (str) — convenience: LETTERS[answer]
    split       (str)   — propagated for provenance
    few_shot_examples (list[dict] | None) — populated when k_shot > 0,
        each dict has {question, choices, answer} for one dev example
        from the SAME subject.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..prompts import LETTERS


def _build_fewshot_pools(
    hf_load, hf_dataset: str, hf_token: Optional[str], k_shot: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Pull up to ``k_shot`` exemplars per subject from the ``dev`` split.

    Returns ``{}`` when k_shot is 0 (caller short-circuits).
    """
    if k_shot <= 0:
        return {}
    load_kwargs = {}
    if hf_token:
        load_kwargs["token"] = hf_token
    dev = hf_load(hf_dataset, "all", split="dev", **load_kwargs).to_pandas()

    pools: Dict[str, List[Dict[str, Any]]] = {}
    for subj, group in dev.groupby("subject"):
        group = group.head(k_shot)
        examples: List[Dict[str, Any]] = []
        for _, r in group.iterrows():
            examples.append({
                "question": str(r["question"]),
                "choices": list(r["choices"]),
                "answer": int(r["answer"]),
            })
        pools[str(subj)] = examples
    return pools


def load_dataset(
    hf_dataset: str = "cais/mmlu",
    hf_config: Optional[str] = "all",
    split: str = "test",
    hf_token: Optional[str] = None,
    sample_n: Optional[int] = None,
    k_shot: int = 0,
) -> pd.DataFrame:
    """Pull MMLU, normalize columns, optionally attach few-shot exemplars.

    The ``cais/mmlu`` repository on HuggingFace requires the ``"all"``
    config to retrieve all 57 subjects; per-subject configs also exist
    (one per subject) but are harder to iterate over.
    """
    from datasets import load_dataset as hf_load

    load_kwargs = {}
    if hf_token:
        load_kwargs["token"] = hf_token

    ds = hf_load(hf_dataset, hf_config, split=split, **load_kwargs)
    df = ds.to_pandas()
    print(
        f"[load_dataset] Loaded {len(df)} rows from {hf_dataset} "
        f"(config={hf_config}, split={split}); columns: {list(df.columns)}",
        flush=True,
    )

    required = {"question", "subject", "choices", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[load_dataset] {hf_dataset} schema missing columns: {sorted(missing)}. "
            f"Actual columns: {list(df.columns)}."
        )

    out = pd.DataFrame({
        "question_id": range(len(df)),
        "subject": df["subject"].astype(str),
        "question": df["question"].astype(str),
        # `choices` is an array column; keep as list so parquet stores it
        # natively. Cast each element to str for safety.
        "choices": df["choices"].apply(lambda xs: [str(x) for x in xs]),
        "answer": df["answer"].astype(int),
        "split": split,
    })
    out["answer_letter"] = out["answer"].apply(lambda i: LETTERS[int(i)])

    # Few-shot pools attached per row from the same subject.
    pools = _build_fewshot_pools(hf_load, hf_dataset, hf_token, k_shot)
    if pools:
        out["few_shot_examples"] = out["subject"].apply(
            lambda s: pools.get(s, [])
        )
    else:
        out["few_shot_examples"] = [[] for _ in range(len(out))]

    print(
        f"[load_dataset] {len(out)} questions across {out['subject'].nunique()} "
        f"subjects (k_shot={k_shot})",
        flush=True,
    )

    if sample_n is not None and sample_n > 0 and sample_n < len(out):
        # Stratified-ish sample by subject so we still see most subjects
        # in a small debug run rather than 5 rows all in 'anatomy'.
        out = (
            out.groupby("subject", group_keys=False)
            .apply(lambda g: g.head(max(1, sample_n // out["subject"].nunique())))
            .reset_index(drop=True)
        )
        if len(out) > sample_n:
            out = out.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"[load_dataset] Sampled to {len(out)} rows (stratified by subject)", flush=True)

    return out
