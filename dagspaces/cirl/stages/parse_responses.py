"""Parse LLM responses for the CIRL-729 action benchmark.

**Reconstruction first.** The shared inference layer
(``dagspaces/common/vllm_inference.py``) splits every model's output into
``generated_text`` (content) and ``generated_reasoning`` — literal
``<think>`` blocks, vLLM family parsers (qwen3, deepseek), and harmony
channels alike. The CIRL reward, however, scores the *raw* output, whose
strict extractor requires a ``</think>`` marker. So before extraction we
rebuild the paper's ``solution_str``::

    <think>{generated_reasoning}</think>\\n{generated_text}   # if reasoning
    {generated_text}                                          # otherwise

For literal-``<think>`` models this reproduces the raw output up to
whitespace inside the tags — same verdicts as the reference. For families
whose CoT is not literally ``<think>``-tagged (gpt-oss harmony channels,
Gemma-4 ``thought`` blocks) the reference run on raw text would score −1 on
a formatting technicality; we instead treat harness-captured reasoning as
satisfying the ``</think>`` requirement (the same reasoning-format
equivalence the repo applies everywhere — see ``wiki/thinking-modes.md``).

Two extractions per reconstructed response:

* **strict** — faithful to the CIRL reward's ``extract_solution``
  (``verl-supp/verl/utils/reward_score/contextual_integrity_reward.py``):
  requires BOTH ``</think>`` and ``<answer>...</answer>`` present, else the
  row is unparseable and scores −1.0 in ``compute_metrics``. This is the
  paper-parity headline path.
* **lenient** — a diagnostic fallback that extracts ``<answer>`` even
  without a ``</think>`` block, and if no ``<answer>`` tag exists strips
  think blocks and uses the remaining text (finally the full text). Lets a
  clean, non-reasoning model's message still be scored, mirroring the
  ``accuracy`` vs ``accuracy_among_parseable`` split used elsewhere in this
  project (``wiki/metric-trust.md``).

Reference: https://github.com/EricGLan/CI-RL (arXiv:2506.04245)
"""

from __future__ import annotations

import re

import pandas as pd

from dagspaces.common.vllm_inference import _strip_think_blocks

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def reconstruct_solution(content: str, reasoning: str) -> str:
    """Rebuild the raw ``solution_str`` the CIRL reward would have seen.

    The harness diverts reasoning into ``generated_reasoning``; re-wrap it in
    ``<think>`` tags so the strict extractor's ``</think>`` requirement is
    evaluated against the full output, not the post-split content.
    """
    if reasoning:
        return f"<think>{reasoning}</think>\n{content}"
    return content


def extract_answer_strict(response: str) -> str | None:
    """CIRL ``extract_solution``: needs ``</think>`` AND ``<answer>...</answer>``."""
    if "</think>" not in response or "<answer>" not in response or "</answer>" not in response:
        return None
    m = _ANSWER_RE.search(response)
    if m:
        return m.group(1).strip("\n").strip()
    return None


def extract_answer_lenient(response: str) -> str:
    """Best-effort answer: ``<answer>`` if present, else de-thought text, else raw."""
    m = _ANSWER_RE.search(response)
    if m:
        return m.group(1).strip("\n").strip()
    stripped = _strip_think_blocks(response).strip()
    if stripped:
        return stripped
    return response.strip()


def parse_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``answer_strict`` / ``answer_lenient`` / ``prediction`` / ``parse_status``."""
    df = df.copy()

    content = df["generated_text"].astype(str)
    if "generated_reasoning" in df.columns:
        reasoning = df["generated_reasoning"].fillna("").astype(str)
    else:
        reasoning = pd.Series([""] * len(df), index=df.index, dtype=str)

    solution = [
        reconstruct_solution(c, r) for c, r in zip(content, reasoning, strict=True)
    ]

    strict = [extract_answer_strict(s) for s in solution]
    df["answer_strict"] = [a if a is not None else "" for a in strict]
    df["strict_parsed"] = [a is not None for a in strict]
    # Lenient works off the content column: post-split, the answer (tagged or
    # bare) lives there; the reconstruction only matters for the strict
    # ``</think>`` requirement.
    df["answer_lenient"] = [extract_answer_lenient(c) for c in content]

    # ``prediction`` / ``parse_status`` feed the shared parse-health sanity
    # check. There is no discrete class label for a generation task, so the
    # "label" is simply whether the strict format was produced.
    df["prediction"] = df["strict_parsed"].map(
        lambda ok: "answered" if ok else "unparseable"
    )

    # "empty" means the model produced nothing at all — a truncated
    # all-reasoning output (content empty, reasoning long) is "unparseable",
    # not "empty"; check finish_reason before blaming the engine.
    raw_empty = [
        not (c.strip() or r.strip()) for c, r in zip(content, reasoning, strict=True)
    ]

    def _status(is_empty: bool, ok: bool) -> str:
        if is_empty:
            return "empty"
        return "parsed" if ok else "unparseable"

    df["parse_status"] = [
        _status(e, ok) for e, ok in zip(raw_empty, df["strict_parsed"], strict=True)
    ]

    total = len(df)
    unparseable = int((~df["strict_parsed"]).sum())
    empty = int(sum(raw_empty))
    rate = unparseable / total if total else 0.0

    print(
        f"[parse_responses] {total} responses, {unparseable} strict-unparseable "
        f"({rate*100:.1f}%), {empty} empty",
        flush=True,
    )

    if rate > 0.2:
        msg = (
            f"WARNING: {unparseable}/{total} ({rate:.0%}) responses lack the "
            "required </think> + <answer>...</answer> format and score -1.0 "
            "under the strict (paper-parity) metric. Non-reasoning models may "
            "still be scored via the lenient diagnostic. Check max_tokens / the "
            "model's ability to follow the <think>/<answer> format."
        )
        print(f"\n{'!'*60}\n  {msg}\n{'!'*60}\n", flush=True)
        import warnings

        warnings.warn(msg, stacklevel=2)

    return df
