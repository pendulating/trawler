"""Extract the chosen letter (A/B/C/D) from each generated response.

Two paths, tried in order:

1. **Structured JSON** — vLLM with ``guided_decoding`` emits
   ``{"answer": "<letter>"}``; we read ``answer`` directly. This is the
   intended path under the default config; it's deterministic.

2. **Bare-letter fallback** — for the (rare) case where a model dodges
   guided decoding (e.g. a transformers-backend fallback path), scan
   for a word-boundary A/B/C/D in the raw text. Case-sensitive uppercase
   only — case-folding would false-match the ``a`` in ``"answer"``.

Anything that fails both yields ``"unparseable"``, surfaced separately
in metrics so unparseable rows don't silently inflate or deflate
accuracy.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import pandas as pd

from dagspaces.common.vllm_inference import _strip_think_blocks


_BOUNDED_LETTER = re.compile(r"\b([ABCD])\b")


def _try_json_answer(response: str) -> Optional[str]:
    text = _strip_think_blocks(response).strip()
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            obj = json.loads(text[start:end])
            if isinstance(obj, dict):
                a = str(obj.get("answer", "")).strip().upper()
                if a in ("A", "B", "C", "D"):
                    return a
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def parse_letter(response: str) -> str:
    """Return one of ``A``/``B``/``C``/``D``/``"unparseable"``."""
    s = str(response or "").strip()
    if not s:
        return "unparseable"

    j = _try_json_answer(s)
    if j is not None:
        return j

    s_clean = _strip_think_blocks(s)
    m = _BOUNDED_LETTER.search(s_clean)
    if m:
        return m.group(1)
    return "unparseable"


def parse_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``prediction_letter`` + ``prediction`` + ``parse_status`` columns.

    Args:
        df: Must have ``generated_text`` (from llm_inference) and
            ``answer`` (ground truth int 0-3).

    Output columns:
        prediction_letter — A/B/C/D or "unparseable"
        prediction        — 0/1/2/3 or -1 for unparseable
        parse_status      — "empty" | "unparseable" | "parsed"
        is_correct        — bool; False whenever unparseable
    """
    df = df.copy()
    df["prediction_letter"] = df["generated_text"].apply(lambda x: parse_letter(str(x)))
    df["prediction"] = df["prediction_letter"].apply(
        lambda l: ("ABCD".index(l) if l in ("A", "B", "C", "D") else -1)
    )
    df["parse_status"] = df.apply(
        lambda r: (
            "empty" if not str(r["generated_text"]).strip()
            else (
                "unparseable" if r["prediction_letter"] == "unparseable"
                else "parsed"
            )
        ),
        axis=1,
    )
    df["is_correct"] = df.apply(
        lambda r: (r["prediction"] == int(r["answer"])) if r["prediction"] != -1 else False,
        axis=1,
    )

    total = len(df)
    n_unp = int((df["prediction_letter"] == "unparseable").sum())
    n_empty = int(
        df["generated_text"].apply(lambda x: len(str(x).strip()) == 0).sum()
    )
    print(
        f"[parse_responses] {total} rows | unparseable={n_unp} "
        f"({(n_unp / max(total, 1)) * 100:.1f}%) | empty={n_empty} | "
        f"distribution: {df['prediction_letter'].value_counts().to_dict()}",
        flush=True,
    )
    return df
