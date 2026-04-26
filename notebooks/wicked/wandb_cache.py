"""Load and filter cached W&B run data.

Usage in notebooks::

    from wandb_cache import load_runs

    # All runs as a DataFrame
    df = load_runs()

    # Filter by tags, model, dagspace, date, or arbitrary fields
    df = load_runs(
        tags=["base"],                          # runs that have ALL these tags
        tags_any=["grpo:grounded", "finetuned"], # runs that have ANY of these
        dagspace="goldcoin_hipaa",              # exact dagspace match
        checkpoint="Qwen3.5-9B",               # substring match on checkpoint_name
        after="2026-03-15",                     # created_at >= this date
        before="2026-03-30",                    # created_at < this date
        state="finished",                       # run state filter
        has_metrics=True,                       # only rows with at least one metric
    )

    # Full enriched dicts (config, summary, etc.)
    runs = load_runs_raw()
    runs = load_runs_raw(tags=["base"], dagspace="privacylens")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

_CACHE_DIR = Path(__file__).parent / "wandb_cache"

_PRECOMPUTED_METRIC_COLS = [
    "gc_applicability_f1", "gc_compliance_f1",
    "gc_applicable_f1", "gc_not_applicable_f1",
    "gc_permit_f1", "gc_forbid_f1",
    "pl_qa_accuracy", "pl_leakage_rate",
    "pl_adjusted_leakage_rate", "pl_helpful_rate",
    "vlm_q7_accuracy",
    "ca_pearson_r", "ca_accuracy", "ca_leak_rate",
    "cirl_accuracy", "cirl_integrity", "cirl_utility", "cirl_complete",
]

# Prefixes used to extract all eval metrics from the W&B summary
_EVAL_METRIC_PREFIXES = [
    "compute_metrics/eval/",
    "compute_trajectory_metrics/eval/",
]


def _load_json(cache_dir: Path = _CACHE_DIR) -> list[dict]:
    path = cache_dir / "runs.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached runs at {path}. Run `python fetch_wandb_runs.py` first."
        )
    with open(path) as f:
        return json.load(f)


def _parse_date(s: str | None) -> datetime | None:
    if s is None:
        return None
    if isinstance(s, datetime):
        if s.tzinfo is None:
            return s.replace(tzinfo=timezone.utc)
        return s
    # Normalise trailing "Z" → "+00:00" so %z can parse it
    text = s.strip().replace("Z", "+00:00")
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(text, fmt)
            # Ensure all results are timezone-aware (UTC)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    dt = pd.Timestamp(s).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _match_run(
    run: dict,
    *,
    tags: Sequence[str] | None = None,
    tags_any: Sequence[str] | None = None,
    tags_exclude: Sequence[str] | None = None,
    dagspace: str | None = None,
    checkpoint: str | None = None,
    after: str | datetime | None = None,
    before: str | datetime | None = None,
    state: str | None = None,
    has_metrics: bool = False,
    where: dict[str, Any] | None = None,
) -> bool:
    """Return True if the run matches all provided filters."""
    run_tags = set(run.get("tags", []))

    # Tag filters
    if tags and not all(t in run_tags for t in tags):
        return False
    if tags_any and not any(t in run_tags for t in tags_any):
        return False
    if tags_exclude and any(t in run_tags for t in tags_exclude):
        return False

    # Dagspace
    if dagspace is not None and run.get("dagspace") != dagspace:
        return False

    # Checkpoint (substring match)
    if checkpoint is not None and checkpoint not in (run.get("checkpoint_name") or ""):
        return False

    # Date range
    if after is not None:
        after_dt = _parse_date(after)
        run_dt = _parse_date(run.get("created_at"))
        if run_dt is None or (after_dt and run_dt < after_dt):
            return False
    if before is not None:
        before_dt = _parse_date(before)
        run_dt = _parse_date(run.get("created_at"))
        if run_dt is None or (before_dt and run_dt >= before_dt):
            return False

    # State
    if state is not None and run.get("state") != state:
        return False

    # Has at least one non-null metric (check precomputed + raw summary)
    if has_metrics:
        has_precomputed = any(run.get(k) is not None for k in _PRECOMPUTED_METRIC_COLS)
        has_summary = any(
            k.startswith(tuple(_EVAL_METRIC_PREFIXES))
            for k in run.get("summary", {})
        )
        if not has_precomputed and not has_summary:
            return False

    # Arbitrary field filters: {"config.model.checkpoint_name": "Qwen3.5-9B"}
    if where:
        for dotpath, expected in where.items():
            val = run
            for part in dotpath.split("."):
                if isinstance(val, dict):
                    val = val.get(part)
                else:
                    val = None
                    break
            if val != expected:
                return False

    return True


def load_runs_raw(
    cache_dir: Path = _CACHE_DIR,
    **filters,
) -> list[dict]:
    """Load enriched run dicts, optionally filtered.

    Accepts all keyword arguments documented in :func:`load_runs`.
    """
    runs = _load_json(cache_dir)
    if filters:
        runs = [r for r in runs if _match_run(r, **filters)]
    return runs


def load_runs(
    cache_dir: Path = _CACHE_DIR,
    **filters,
) -> pd.DataFrame:
    """Load cached runs as a flat DataFrame, optionally filtered.

    Keyword Args:
        tags: list of tags the run must have (AND logic).
        tags_any: list of tags the run must have at least one of (OR logic).
        tags_exclude: list of tags the run must NOT have.
        dagspace: exact dagspace string (e.g. "goldcoin_hipaa").
        checkpoint: substring match on checkpoint_name.
        after: only runs created at or after this date (YYYY-MM-DD or ISO).
        before: only runs created before this date.
        state: run state (e.g. "finished").
        has_metrics: if True, only rows with at least one benchmark metric.
        where: dict of dotted-path → value for arbitrary nested field matching.
    """
    runs = load_runs_raw(cache_dir=cache_dir, **filters)

    if not runs:
        return pd.DataFrame()

    rows = []
    for r in runs:
        row = {
            "run_id": r["run_id"],
            "run_name": r["run_name"],
            "state": r["state"],
            "created_at": r["created_at"],
            "dagspace": r["dagspace"],
            "checkpoint_name": r["checkpoint_name"],
            "tags": r["tags"],  # keep as list for easy filtering in notebooks
        }
        # Pre-computed composite metrics
        for k in _PRECOMPUTED_METRIC_COLS:
            row[k] = r.get(k)
        # All eval metrics from the W&B summary, flattened
        for k, v in r.get("summary", {}).items():
            if not isinstance(v, (int, float)):
                continue
            for prefix in _EVAL_METRIC_PREFIXES:
                if k.startswith(prefix):
                    col = k[len(prefix):]  # strip prefix
                    row[f"eval/{col}"] = v
                    break
        rows.append(row)

    df = pd.DataFrame(rows)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def cache_info(cache_dir: Path = _CACHE_DIR) -> dict:
    """Return metadata about the cached data."""
    meta_path = cache_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}
