"""m-series stratified prescreen + m1 cache signature (redesign item 6).

The v-era prescreen (``prompt_screening.py``, a frozen keeper surface) is
**variance-only and force-blind**: it ranks every candidate prompt by its
SFT-group reward std and keeps the global top set. That silently doubled the
vignette force skew — pool 3.07:1 became realized 5.2:1 (v10) — because whole
low-variance strata were dropped wholesale
(``wiki/grpo_redesign/prescreen-and-gates.md``).

The m-series fixes this structurally (principle 6, README.md): variance ranks
candidates **within strata, never across**. A stratum is
``(task_type x gold_class x force_class)``; ``target_n`` is a *pre-registered*
constant allocated across strata proportionally to the configured mix
(``task_mix`` for the task dimension, the pool's own gold/force composition
within each task), with a hard floor of >=1 per non-empty stratum so a minority
force can never be crowded out by a high-variance majority. The realized
composition is reported for ``training_metadata.json`` /
``prescreen_report.json``.

Two public entry points:

* :func:`stratified_prescreen` — the selection itself.
* :func:`m1_cache_signature` — the ablation-protocol cell-key contract. The
  prescreen scores are expensive (answerer + judge calls), so they are cached;
  ``reward_auxiliaries`` (the module list) and ``task_mix`` are part of the
  cache key, so every grid cell automatically gets its own screen and a config
  typo misses cleanly instead of reusing a neighbor's cache. ``formula_version``
  is pinned to ``m1`` — a new namespace that never collides with the v-era
  ``rground_formula_version`` keys.

This is **new, additive code**: it imports shared helpers from
``prompt_screening.py`` but does not edit that frozen v-era surface.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

# Reuse the v-era population-std helper rather than re-deriving it. Any spread
# score already living on the rows is accepted directly (``variance_col``); this
# is only the reference definition, kept in one place.
from .prompt_screening import _std  # noqa: F401  (re-exported for callers/tests)

# The m-series cache namespace. Deliberately discontinuous with the v-era
# ``rground_formula_version`` strings (v8/v9/v12a...) so an m1 signature can
# never collide with a v-era cache entry.
M1_FORMULA_VERSION = "m1"

# Default name of the per-row variance/spread column. The v-era screen scored
# each prompt by the population std of its SFT-group composite rewards (the
# ``stds`` dict in ``prompt_screening.py``); when that per-prompt std is carried
# on the prescreen DataFrame the natural column name is ``reward_std``. Callers
# with a differently-named spread score pass ``variance_col=...``.
DEFAULT_VARIANCE_COL = "reward_std"

# The three columns that define a stratum.
_STRATUM_COLS = ("task_type", "gold_class", "force_class")


def _normalize(weights: dict[Any, float]) -> dict[Any, float]:
    """Normalize non-negative weights to sum 1; all-zero maps to all-zero."""
    total = float(sum(max(0.0, w) for w in weights.values()))
    if total <= 0.0:
        return {k: 0.0 for k in weights}
    return {k: max(0.0, w) / total for k, w in weights.items()}


def stratified_prescreen(
    rows,
    *,
    target_n: int,
    seed: int,
    task_mix: Mapping[str, float] | None = None,
    variance_col: str = DEFAULT_VARIANCE_COL,
    id_col: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Select a stratified, variance-carrying, mix-preserving prompt set.

    Args:
        rows: a ``pandas.DataFrame`` of candidate prompts. Must carry at least
            ``task_type`` (``"extract"``/``"vignette"``), ``gold_class``
            (``"yes"``/``"no"``/``"none"``), ``force_class`` (dominant force or
            ``"mixed"``), and a per-row variance/spread score in ``variance_col``.
        target_n: the pre-registered prompt-set size (README principle: a
            declared constant, not "whatever survived"). If the eligible pool is
            smaller, the whole eligible pool is returned.
        seed: seeds the RNG used only as a last-resort within-stratum tie-break
            (variance ties are already broken deterministically by row id, so in
            practice this never fires for unique ids).
        task_mix: configured task-dimension mix, e.g.
            ``{"extract": 0.7, "vignette": 0.3}``. A task set to ``0.0`` (or
            omitted while other tasks are listed) receives **no** selections —
            this is how the ``-vignette`` cell mixes vignettes to zero. When
            ``None``, the task dimension falls back to the pool's own task
            proportions.
        variance_col: name of the spread-score column (default ``reward_std``).
        id_col: column holding a stable row id for tie-breaking; when ``None``
            the DataFrame index is used.

    Returns:
        ``(selected_rows, report)`` where ``selected_rows`` is a DataFrame slice
        of ``rows`` (original row order preserved) and ``report`` is the
        principle-6 realized-mix accounting (per-stratum pool/selected counts,
        configured vs realized task/gold/force mix) destined for
        ``prescreen_report.json`` / ``training_metadata.json``.

    Selection rule (the v10 fix): allocate ``target_n`` across strata
    proportionally to ``task_mix`` x within-task pool composition, floor >=1 per
    non-empty eligible stratum, then rank **within each stratum** by
    ``variance_col`` descending (ties by row id ascending). Variance never ranks
    across strata, so a high-variance majority stratum cannot displace a minority
    one — the configured mix survives filtering.
    """
    import numpy as np
    import pandas as pd

    if not isinstance(rows, pd.DataFrame):  # defensive; contract says DataFrame
        raise TypeError("stratified_prescreen expects a pandas DataFrame")
    for col in _STRATUM_COLS:
        if col not in rows.columns:
            raise KeyError(f"rows is missing required stratum column {col!r}")
    if variance_col not in rows.columns:
        raise KeyError(f"rows is missing variance column {variance_col!r}")

    n_pool = len(rows)
    rng = np.random.default_rng(seed)

    # Positional working frame so we can map selections back to `rows` order.
    work = rows.reset_index(drop=True).copy()
    work["_pos"] = np.arange(n_pool)
    if id_col is not None:
        if id_col not in rows.columns:
            raise KeyError(f"rows is missing id_col {id_col!r}")
        work["_rowid"] = rows[id_col].astype(str).to_numpy()
    else:
        work["_rowid"] = [str(x) for x in rows.index.tolist()]
    # RNG last-resort tie-break: only decides between rows with identical
    # variance AND identical row id (degenerate for unique ids), so the seed is
    # used "only where explicitly needed" while keeping selection deterministic.
    work["_jitter"] = rng.random(n_pool)
    # Stratum columns as strings (avoids NaN-drop / mixed-type keys in groupby).
    for col in _STRATUM_COLS:
        work[col] = work[col].astype(str)

    empty = rows.iloc[0:0]
    if n_pool == 0:
        return empty, _empty_report(target_n, seed, variance_col, task_mix)

    grouped = work.groupby(list(_STRATUM_COLS), sort=True)
    pool_counts: dict[tuple, int] = {
        key: len(idx) for key, idx in grouped.groups.items()
    }

    # --- Task-dimension weights -------------------------------------------
    task_pool: dict[str, int] = {}
    for key, cnt in pool_counts.items():
        task_pool[key[0]] = task_pool.get(key[0], 0) + cnt
    present_tasks = sorted(task_pool)

    if task_mix:
        tw_raw = {t: float(task_mix.get(t, 0.0)) for t in present_tasks}
        if sum(tw_raw.values()) <= 0.0:
            # configured mix zeroes every present task -> nothing to select
            tw = {t: 0.0 for t in present_tasks}
        else:
            tw = _normalize(tw_raw)
    else:
        tw = _normalize({t: float(task_pool[t]) for t in present_tasks})

    # --- Per-stratum ideal (real-valued) target ---------------------------
    # weight[s] = tw[task] * (pool_count[s] / pool_count_in_task[task]).
    # Sum over strata of a positively-weighted task == tw[task]; total == 1.
    weight: dict[tuple, float] = {}
    raw: dict[tuple, float] = {}
    for key, cnt in pool_counts.items():
        t = key[0]
        w = tw.get(t, 0.0) * (cnt / task_pool[t]) if task_pool[t] else 0.0
        weight[key] = w
        raw[key] = target_n * w

    # --- Integer apportionment: floor>=1, cap at pool, greedy by deficit ---
    eligible = [s for s in pool_counts if raw[s] > 0.0]
    alloc: dict[tuple, int] = {s: 0 for s in pool_counts}
    goal = min(int(target_n), sum(pool_counts[s] for s in eligible))
    # Hard floor of 1 per non-empty eligible stratum.
    for s in eligible:
        alloc[s] = min(pool_counts[s], 1)
    used = sum(alloc.values())
    # Distribute the remainder to the strata most under their ideal share.
    while used < goal:
        cands = [s for s in eligible if alloc[s] < pool_counts[s]]
        if not cands:
            break
        # Largest deficit first; deterministic tie-break by stratum key.
        best = min(cands, key=lambda s: (-(raw[s] - alloc[s]), s))
        alloc[best] += 1
        used += 1

    # --- Within-stratum ranking (variance desc, id asc, jitter) -----------
    selected_positions: list[int] = []
    for s in pool_counts:
        k = alloc[s]
        if k <= 0:
            continue
        sub = grouped.get_group(s)
        sub_sorted = sub.sort_values(
            by=[variance_col, "_rowid", "_jitter"],
            ascending=[False, True, True],
            kind="mergesort",  # stable
        )
        selected_positions.extend(sub_sorted["_pos"].head(k).tolist())

    sel_pos_sorted = sorted(set(int(p) for p in selected_positions))
    selected = rows.iloc[sel_pos_sorted]

    report = _build_report(
        rows=rows,
        selected=selected,
        target_n=target_n,
        seed=seed,
        variance_col=variance_col,
        pool_counts=pool_counts,
        alloc=alloc,
        weight=weight,
        raw=raw,
        tw=tw,
        task_pool=task_pool,
        n_pool=n_pool,
    )
    return selected, report


def _col_share(frame, col: str) -> dict[str, float]:
    """Value -> share of ``frame`` rows, as a plain dict of Python floats."""
    n = len(frame)
    if n == 0:
        return {}
    counts = frame[col].astype(str).value_counts()
    return {str(k): float(v) / n for k, v in counts.items()}


def _build_report(
    *,
    rows,
    selected,
    target_n: int,
    seed: int,
    variance_col: str,
    pool_counts: dict[tuple, int],
    alloc: dict[tuple, int],
    weight: dict[tuple, float],
    raw: dict[tuple, float],
    tw: dict[str, float],
    task_pool: dict[str, int],
    n_pool: int,
) -> dict[str, Any]:
    """Assemble the principle-6 realized-mix accounting dict."""
    strata: dict[str, Any] = {}
    for s in sorted(pool_counts):
        strata["|".join(s)] = {
            "pool": int(pool_counts[s]),
            "selected": int(alloc[s]),
            "configured_share": round(float(weight[s]), 6),
            "configured_target": round(float(raw[s]), 4),
        }
    n_selected = int(sum(alloc.values()))

    def _task_share(frame) -> dict[str, float]:
        return _col_share(frame, "task_type")

    return {
        "formula_version": M1_FORMULA_VERSION,
        "target_n": int(target_n),
        "seed": int(seed),
        "variance_col": variance_col,
        "n_pool": int(n_pool),
        "n_selected": n_selected,
        "n_dropped": int(n_pool - n_selected),
        "strata": strata,
        "configured_task_mix": {t: round(float(tw.get(t, 0.0)), 6) for t in sorted(task_pool)},
        "pool_task_mix": _task_share(rows),
        "realized_task_mix": _task_share(selected),
        "pool_gold_mix": _col_share(rows, "gold_class"),
        "realized_gold_mix": _col_share(selected, "gold_class"),
        "pool_force_mix": _col_share(rows, "force_class"),
        "realized_force_mix": _col_share(selected, "force_class"),
    }


def _empty_report(
    target_n: int, seed: int, variance_col: str,
    task_mix: Mapping[str, float] | None,
) -> dict[str, Any]:
    return {
        "formula_version": M1_FORMULA_VERSION,
        "target_n": int(target_n),
        "seed": int(seed),
        "variance_col": variance_col,
        "n_pool": 0,
        "n_selected": 0,
        "n_dropped": 0,
        "strata": {},
        "configured_task_mix": dict(task_mix or {}),
        "pool_task_mix": {},
        "realized_task_mix": {},
        "pool_gold_mix": {},
        "realized_gold_mix": {},
        "pool_force_mix": {},
        "realized_force_mix": {},
    }


# --------------------------------------------------------------------------
# m1 cache signature
# --------------------------------------------------------------------------

def _m1_signature_payload(
    *,
    module_list: Sequence[str],
    task_mix: Mapping[str, float],
    seed: int,
    data_fingerprint: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical signature payload (hashed by :func:`m1_cache_signature`).

    Exposed so tests can assert ``formula_version == "m1"`` is embedded and that
    only the declared ingredients participate. The module list is **sorted**
    (canonicalized) so the reward-auxiliary *set*, not its ordering, is what the
    cache keys on; ``task_mix``/``extra`` are hashed key-sorted at dump time.
    """
    return {
        "formula_version": M1_FORMULA_VERSION,
        "module_list": sorted(str(m) for m in module_list),
        "task_mix": {str(k): float(v) for k, v in dict(task_mix).items()},
        "seed": int(seed),
        "data_fingerprint": str(data_fingerprint),
        # Belt-and-braces bucket for the derived weight-rule output, probe/battery
        # seeds, answerer/judge identity, routing constants (prescreen-and-gates.md).
        # `None` and `{}` are canonicalized to the same empty payload.
        "extra": dict(extra) if extra else {},
    }


def m1_cache_signature(
    *,
    module_list: Sequence[str],
    task_mix: Mapping[str, float],
    seed: int,
    data_fingerprint: str,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Stable cache key for an m-series prescreen (the cell-key contract).

    ``formula_version`` is pinned to ``m1`` inside the payload. Any change to the
    module list (content — order is canonicalized away), ``task_mix``, ``seed``,
    or ``data_fingerprint`` — or to any ``extra`` ingredient — changes the
    returned signature; keys are dumped sorted so insertion order is irrelevant.
    Hash: sorted-key JSON -> SHA1 hexdigest.
    """
    payload = _m1_signature_payload(
        module_list=module_list,
        task_mix=task_mix,
        seed=seed,
        data_fingerprint=data_fingerprint,
        extra=extra,
    )
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()
