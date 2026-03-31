"""Load and prepare the CIRL evaluation dataset (main_data.json).

Downloads from the CI-RL GitHub repo if not available locally, then
expands each case into per-probing-level rows (seed + vignette).
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import pandas as pd


_GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/EricGLan/CI-RL/main/"
    "posttraining-research-ci-supp/components/privacylens/data/main_data.json"
)

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "data", "ci_benchmarks", "CIRL",
)


def _ensure_data_file(json_path: Optional[str]) -> str:
    """Return path to main_data.json, downloading if needed."""
    if json_path and os.path.isfile(json_path):
        return json_path

    cache_dir = os.path.normpath(_DEFAULT_CACHE_DIR)
    cached = os.path.join(cache_dir, "main_data.json")
    if os.path.isfile(cached):
        return cached

    print(f"[load_dataset] Downloading main_data.json from GitHub …")
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    urllib.request.urlretrieve(_GITHUB_RAW_URL, cached)
    print(f"[load_dataset] Saved to {cached}")
    return cached


def _strip_article(text: str) -> str:
    """Remove leading 'a ' or 'an ' article, matching the paper's preprocessing."""
    return re.sub(r"^(an?)\s+", "", text, count=1)


def _to_gerund(transmission_principle: str) -> str:
    """Convert first word to gerund form, matching the paper's preprocessing.

    'send a facebook post' → 'sending a facebook post'
    """
    parts = transmission_principle.split(None, 1)
    if not parts:
        return transmission_principle
    verb = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    if verb.endswith("e") and not verb.endswith("ee"):
        gerund = verb[:-1] + "ing"
    else:
        gerund = verb + "ing"
    return f"{gerund} {rest}".strip()


def load_dataset(
    json_path: Optional[str] = None,
    probing_levels: Optional[list[str]] = None,
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    """Load main_data.json and expand to per-probing-level rows.

    Args:
        json_path: Path to main_data.json. Downloaded from GitHub if absent.
        probing_levels: Which probing levels to include. Default: ["seed", "vignette"].
        sample_n: Optional number of *cases* to sample before expansion.

    Returns:
        DataFrame with one row per (case, probing_level). Ground truth is
        always 'B' (reject) — all cases are privacy-violating scenarios.
    """
    if probing_levels is None:
        probing_levels = ["seed", "vignette"]

    path = _ensure_data_file(json_path)
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    print(f"[load_dataset] Loaded {len(cases)} cases from {path}")

    if sample_n is not None and 0 < sample_n < len(cases):
        import random
        rng = random.Random(42)
        cases = rng.sample(cases, sample_n)
        print(f"[load_dataset] Sampled {sample_n} cases")

    rows = []
    for case in cases:
        name = case.get("name", "")
        seed = case.get("seed", {})
        vignette = case.get("vignette", {})

        # Shared seed fields (preprocessed per paper)
        data_type = _strip_article(seed.get("data_type", ""))
        data_subject = seed.get("data_subject", "")
        data_sender = seed.get("data_sender", "")
        data_sender_name = seed.get("data_sender_name", "")
        data_recipient = seed.get("data_recipient", "")
        transmission_principle_raw = seed.get("transmission_principle", "")
        transmission_principle = _to_gerund(transmission_principle_raw)

        base = {
            "case_name": name,
            "data_type": data_type,
            "data_subject": data_subject,
            "data_sender": data_sender,
            "data_sender_name": data_sender_name,
            "data_recipient": data_recipient,
            "transmission_principle": transmission_principle,
            "ground_truth": "B",
        }

        if "seed" in probing_levels:
            row = dict(base)
            row["probing_level"] = "seed"
            rows.append(row)

        if "vignette" in probing_levels:
            story = vignette.get("story", "")
            dt_concrete = _strip_article(
                vignette.get("data_type_concrete", data_type)
            ).lower().strip(".")
            ds_concrete = vignette.get("data_subject_concrete", data_subject)
            sender_concrete = vignette.get("data_sender_concrete", data_sender)
            recipient_concrete = vignette.get("data_recipient_concrete", data_recipient)

            row = dict(base)
            row["probing_level"] = "vignette"
            row["story"] = story
            row["data_type_concrete"] = dt_concrete
            row["data_subject_concrete"] = ds_concrete
            row["data_sender_concrete"] = sender_concrete
            row["data_recipient_concrete"] = recipient_concrete
            rows.append(row)

        if "trajectory" in probing_levels:
            traj = case.get("trajectory", {})
            row = dict(base)
            row["probing_level"] = "trajectory"
            row["user_name"] = traj.get("user_name", "")
            row["user_email"] = traj.get("user_email", "")
            row["user_instruction"] = traj.get("user_instruction", "")
            row["toolkits"] = json.dumps(traj.get("toolkits", []))
            row["executable_trajectory"] = traj.get("executable_trajectory", "")
            row["final_action"] = traj.get("final_action", "")
            row["sensitive_info_items"] = json.dumps(
                traj.get("sensitive_info_items", [])
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    df["row_id"] = range(len(df))

    print(f"[load_dataset] {len(df)} probing rows "
          f"({len(cases)} cases × {len(probing_levels)} levels)")
    print(f"[load_dataset] Probing level distribution:")
    print(df["probing_level"].value_counts().to_string())

    return df
