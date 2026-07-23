"""Load and prepare the CIRL-729 evaluation dataset.

The CIRL benchmark is the 729-example synthetic Contextual-Integrity dataset
from Lan, Inan, Abdelnabi, Kulkarni, Wutschitz, Shokri, Brinton, Sim,
*Contextual Integrity in LLMs via Reasoning and Reinforcement Learning*
(NeurIPS 2025, arXiv:2506.04245). Source:
``huseyinatahaninan/ContextualIntegritySyntheticDataset`` on HuggingFace.

Each raw row has two JSON string columns:

- ``seed``: ``{scenario, domain, user_intention, sender, recipient,
  data_subject, information_type:{allowed:[...], disallowed:[...]},
  transmission_principle}``
- ``dataset_item``: ``{user_task, information:{attr:val}, past_emails,
  annotation:{allowed:{attr:val}, disallowed:{attr:val}}}``

This is an **action-generation** task, NOT a yes/no probe (contrast the old
``cirl_vignettes`` dagspace, which evaluated a vendored copy of PrivacyLens).
The model is given the scenario + ``user_task`` + a shuffled attribute list and
must write the message completing the task, disclosing the *allowed* attributes
while withholding the *disallowed* ones.

**Display vs. score.** The attribute list shown to the model uses the FULL
values from ``information[key]``, but scoring (see ``compute_metrics.py``)
matches the SHORT ``annotation`` values — these differ in ~73% of items (the
model is shown ``MedicalNotes: Currently taking RX2019 for hypertension`` but
leakage is scored on the substring ``RX2019``). This mirrors the CIRL reward
(``verl-supp/verl/utils/reward_score/contextual_integrity_reward.py``, which
uses ``extra_info['*_short']``) and their data-prep
(``verl-supp/examples/data_preprocess/contextual_integrity.py``).

``past_emails`` is intentionally unused: the canonical prompt omits it and it is
present in only 1 of 729 rows.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import pandas as pd

_HF_REPO = "huseyinatahaninan/ContextualIntegritySyntheticDataset"
_HF_FILE = "data/train-00000-of-00001.parquet"

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "data", "ci_benchmarks", "CIRL729",
)


def _ensure_data_file(parquet_path: str | None) -> str:
    """Return a path to the CIRL-729 parquet, downloading from HF if needed."""
    if parquet_path and os.path.isfile(parquet_path):
        return parquet_path

    cache_dir = os.path.normpath(_DEFAULT_CACHE_DIR)
    cached = os.path.join(cache_dir, "cirl729.parquet")
    if os.path.isfile(cached):
        return cached

    print(f"[load_dataset] Downloading {_HF_FILE} from HF:{_HF_REPO} …", flush=True)
    from huggingface_hub import hf_hub_download

    src = hf_hub_download(_HF_REPO, _HF_FILE, repo_type="dataset")
    os.makedirs(cache_dir, exist_ok=True)
    # Copy into the repo-local cache so eval is reproducible without HF creds.
    pd.read_parquet(src).to_parquet(cached, index=False)
    print(f"[load_dataset] Cached to {cached}", flush=True)
    return cached


def _build_attribute_list(
    information: dict[str, Any],
    allowed: dict[str, Any],
    disallowed: dict[str, Any],
    rng: random.Random,
) -> tuple[str, list[list[str]], list[list[str]]]:
    """Render the shuffled ``Key: value`` attribute list shown to the model.

    Faithful to ``contextual_integrity.py``: iterate the annotation keys
    (allowed then disallowed), display the FULL ``information[key]`` value, then
    shuffle. Returns ``(rendered_list, allowed_pairs, disallowed_pairs)`` where
    the pairs carry the SHORT annotation values used for scoring.
    """
    allowed_pairs = [[k, str(v)] for k, v in allowed.items()]
    disallowed_pairs = [[k, str(v)] for k, v in disallowed.items()]

    display: list[tuple[str, str]] = []
    for k in list(allowed.keys()) + list(disallowed.keys()):
        if k in information:
            display.append((k, str(information[k])))
        else:
            # 0/729 rows hit this in practice; fall back to the annotation
            # value so a stray missing key never drops a row silently.
            fallback = allowed.get(k, disallowed.get(k, ""))
            print(
                f"[load_dataset] WARNING: annotation key {k!r} absent from "
                f"'information'; displaying annotation value instead.",
                flush=True,
            )
            display.append((k, str(fallback)))

    rng.shuffle(display)
    rendered = "\n".join(f"{k}: {v}" for k, v in display)
    return rendered, allowed_pairs, disallowed_pairs


def load_dataset(
    parquet_path: str | None = None,
    sample_n: int | None = None,
    shuffle_seed: int = 42,
) -> pd.DataFrame:
    """Load CIRL-729 and expand into one flat row per example.

    Args:
        parquet_path: Local parquet override. Downloaded from HF if absent.
        sample_n: Optional number of examples to sample (debug).
        shuffle_seed: Base seed for the per-row attribute-list shuffle. The
            shuffle is seeded ``shuffle_seed + row_index`` so the rendered
            prompt is deterministic and reproducible across runs.

    Returns:
        DataFrame with one row per example and the columns consumed by
        ``llm_inference`` (prompt fields) and ``compute_metrics``
        (``allowed_scored`` / ``disallowed_scored`` JSON pairs).
    """
    path = _ensure_data_file(parquet_path)
    raw = pd.read_parquet(path)
    print(f"[load_dataset] Loaded {len(raw)} raw CIRL-729 rows from {path}", flush=True)

    if sample_n is not None and 0 < sample_n < len(raw):
        raw = raw.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"[load_dataset] Sampled {sample_n} examples", flush=True)

    rows = []
    for idx, rec in raw.reset_index(drop=True).iterrows():
        seed = json.loads(rec["seed"])
        item = json.loads(rec["dataset_item"])
        ann = item.get("annotation", {})
        allowed = ann.get("allowed", {}) or {}
        disallowed = ann.get("disallowed", {}) or {}
        information = item.get("information", {}) or {}

        rng = random.Random(shuffle_seed + int(idx))
        attr_list, allowed_pairs, disallowed_pairs = _build_attribute_list(
            information, allowed, disallowed, rng
        )

        rows.append(
            {
                "row_id": int(idx),
                "domain": seed.get("domain", ""),
                "scenario": seed.get("scenario", ""),
                "transmission_principle": seed.get("transmission_principle", ""),
                "user_intention": seed.get("user_intention", ""),
                "sender": seed.get("sender", ""),
                "recipient": seed.get("recipient", ""),
                "data_subject": seed.get("data_subject", ""),
                "user_task": item.get("user_task", ""),
                "attr_list": attr_list,
                # SHORT annotation values, JSON [[attr, value], ...] — scored.
                "allowed_scored": json.dumps(allowed_pairs),
                "disallowed_scored": json.dumps(disallowed_pairs),
                "n_allowed": len(allowed_pairs),
                "n_disallowed": len(disallowed_pairs),
            }
        )

    df = pd.DataFrame(rows)

    print(f"[load_dataset] {len(df)} CIRL-729 rows prepared", flush=True)
    print("[load_dataset] Domain distribution:", flush=True)
    print(df["domain"].value_counts().to_string(), flush=True)
    print(
        f"[load_dataset] allowed items: {int(df['n_allowed'].sum())}, "
        f"disallowed items: {int(df['n_disallowed'].sum())}",
        flush=True,
    )
    return df
