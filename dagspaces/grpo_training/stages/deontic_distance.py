"""Deontic-distance scorer for the m-series `T-VIGNETTE` battery task.

A battery is a K-item test: K scenarios (same book + context cluster, mixed gold
polarity) that the policy answers with a 5-way deontic force, brief reasoning,
and a statement of the governing norm. The reward is one linear distance formula
(wiki/grpo_redesign/task-vignettes.md, "Reward: deontic-distance scoring"):

    forces sit on the axis  obligatory +2 · recommended +1 · permitted 0 ·
    discouraged −1 · prohibited −2, and per item

        s_i = 1 − |axis(model_i) − axis(gold_i)| / 2

so an exact force scores 1.0, an adjacent degree 0.5, a hedge (permitted /
missing) against a decisive gold 0.0, and the **antithesis** (prohibited vs an
obligatory gold, or vice versa) −1.0. Battery = the mean rescaled to [0,1];
`R_vig = 0.7·battery + 0.3·cite`, where `cite` is per-item Jaccard token overlap
with the withheld source articulation.

This is additive m-series code (the parallel-stack rule,
wiki/grpo_redesign/migration.md item 5): it lives beside — and imports the force
vocabulary from — :mod:`deontic` (the declared single source of truth for the
force → gold/axis mapping), and edits none of the keeper surfaces. A later
``ModularReward`` agent imports the exact names frozen here:
``AXIS``, :func:`item_score`, :func:`battery_score`, :func:`cite_score`,
:func:`parse_battery_completion`, :func:`score_battery`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from dagspaces.common.json_extraction import extract_json_from_text

# The standard deontic axis. Shared vocabulary with FORCE_TO_GOLD in deontic.py
# (obligatory/recommended → gold yes; prohibited/discouraged → gold no;
# permitted is the neutral hedge point).
AXIS: dict[str, int] = {
    "obligatory": 2,
    "recommended": 1,
    "permitted": 0,
    "discouraged": -1,
    "prohibited": -2,
}

# The hedge / undetermined axis value. A missing, unparseable, or non-force
# answer is scored as "permitted" (axis 0) — the same neutral point the v9
# lineage used, now derived from the distance metric rather than a tier ladder.
_HEDGE_AXIS = AXIS["permitted"]

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Lowercased alphanumeric token set (for Jaccard overlap)."""
    return set(_WORD_RE.findall(str(text).lower()))


def axis_of(force: object) -> int:
    """Deontic axis value for a force label.

    A ``None``, unparseable, or non-force label collapses to the hedge point
    (``permitted``, axis 0) — the single convention the whole formula rests on.
    """
    if force is None:
        return _HEDGE_AXIS
    return AXIS.get(str(force).strip().lower(), _HEDGE_AXIS)


def item_score(model_force: object, gold_force: object) -> float:
    """Per-item deontic-distance score ``1 − |axis(m) − axis(g)| / 2`` ∈ [−1, 1].

    ``model_force`` that is ``None``, unparseable, or not one of the five forces
    is treated as ``permitted`` (the hedge point, axis 0). ``gold_force`` is a
    battery item's stored decisive force; a stray non-force gold likewise
    collapses to the hedge point.

    Worked cells (gold = obligatory): exact 1.0 · adjacent (recommended) 0.5 ·
    hedge (permitted/missing) 0.0 · mild antithesis (discouraged) −0.5 · full
    antithesis (prohibited) −1.0.
    """
    return 1.0 - abs(axis_of(model_force) - axis_of(gold_force)) / 2.0


def battery_score(item_scores: Sequence[float]) -> float:
    """Battery score = mean per-item score rescaled from [−1, 1] to [0, 1].

    ``(mean(item_scores) + 1) / 2``. An **empty** sequence is a caller bug
    (a battery always has items) and raises ``ValueError`` rather than
    returning a silent neutral — the ambiguity would mask a build/routing error.
    """
    items = list(item_scores)
    if not items:
        raise ValueError("battery_score requires a non-empty item_scores sequence")
    mean = sum(float(s) for s in items) / len(items)
    return (mean + 1.0) / 2.0


def cite_score(model_norm_text: str, articulation: str) -> float:
    """Jaccard token overlap between a governing-norm statement and the source.

    ``|A ∩ B| / |A ∪ B|`` over lowercased ``[a-z0-9]+`` token *sets*, in [0, 1].
    Credit requires reproducing the withheld articulation's content (it is never
    in the prompt), not parroting it. An empty string on either side → 0.0 (no
    overlap is defined, and a blank citation earns nothing).
    """
    a = _tokens(model_norm_text)
    b = _tokens(articulation)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def parse_battery_completion(text: str, k: int) -> list[dict | None]:
    """Parse a policy battery completion into ``k`` id-aligned item dicts.

    The policy emits ``{"items": [{"id", "force", "reasoning",
    "governing_norm"}, ...]}``. Parsed via
    :func:`~dagspaces.common.json_extraction.extract_json_from_text` with
    ``repair=True`` — the 2026-07-23 build lesson is that ``max_tokens``
    truncation otherwise masquerades as hedging (json_repair salvages the
    partial list).

    Returns a list of **exactly ``k``** entries, aligned by the 1-indexed
    ``id`` field: position ``i`` holds the item whose ``id`` is ``i + 1``, or
    ``None`` when that id is missing or the completion is unparseable. Items with
    a missing / non-integer / out-of-range id are dropped; the first item seen
    for an id wins (later duplicates ignored).
    """
    slots: list[dict | None] = [None] * k
    if k <= 0:
        return slots
    obj, _err = extract_json_from_text(text, repair=True)
    if not isinstance(obj, dict):
        return slots
    items = obj.get("items")
    if not isinstance(items, list):
        return slots
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        try:
            idx = int(raw_id) - 1
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= k:
            continue
        if slots[idx] is None:  # first-id-wins
            slots[idx] = item
    return slots


def score_battery(
    parsed_items: Sequence[dict | None],
    gold_items: Sequence[Mapping],
) -> dict:
    """Score a parsed battery against its gold, returning the reward + forensics.

    ``gold_items`` is the battery's stored gold, one per position:
    ``{"gold_force", "articulation"}``. ``parsed_items`` is the output of
    :func:`parse_battery_completion` (same length as ``gold_items``; a shorter
    list is right-padded with ``None``).

    Returns a dict:

      * ``battery`` — rescaled mean per-item deontic-distance score, [0, 1].
      * ``cite`` — mean per-item Jaccard citation score; an unparsed item
        contributes 0.
      * ``r_vig`` — ``0.7·battery + 0.3·cite`` (the fixed vignette reward).
      * ``hedge_frac`` — fraction of items answered at the hedge point (axis 0:
        ``permitted``, missing, or unparseable).
      * ``antithesis_frac`` — fraction of items whose model/gold axes have
        opposite polarity (``axis(model)·axis(gold) < 0``); the redesign's
        headline forensic (should be rare and falling).
      * ``parsed_frac`` — fraction of items that parsed (non-``None``).
    """
    gold = list(gold_items)
    k = len(gold)
    if k == 0:
        raise ValueError("score_battery requires a non-empty gold_items sequence")

    parsed = list(parsed_items)
    if len(parsed) < k:
        parsed = parsed + [None] * (k - len(parsed))

    item_scores: list[float] = []
    cites: list[float] = []
    n_hedge = 0
    n_antithesis = 0
    n_parsed = 0

    for i in range(k):
        g = gold[i]
        gold_force = g.get("gold_force")
        articulation = g.get("articulation", "") or ""
        item = parsed[i]

        if item is None:
            model_force = None
        else:
            n_parsed += 1
            model_force = item.get("force")

        item_scores.append(item_score(model_force, gold_force))

        m_axis = axis_of(model_force)
        g_axis = axis_of(gold_force)
        if m_axis == _HEDGE_AXIS:
            n_hedge += 1
        if m_axis * g_axis < 0:
            n_antithesis += 1

        if item is None:
            cites.append(0.0)
        else:
            cites.append(cite_score(item.get("governing_norm", "") or "", articulation))

    return {
        "battery": battery_score(item_scores),
        "cite": sum(cites) / k,
        "r_vig": 0.7 * battery_score(item_scores) + 0.3 * (sum(cites) / k),
        "hedge_frac": n_hedge / k,
        "antithesis_frac": n_antithesis / k,
        "parsed_frac": n_parsed / k,
    }
