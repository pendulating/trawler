"""Deontic-distance scorer for the m-series `T-VIGNETTE` battery task.

A battery is a K-item test: K scenarios (same book + context cluster, mixed gold
polarity) that the policy answers with a 5-way deontic force, brief reasoning,
and a statement of the governing norm.

**Per-item scale (2026-07-28 re-anchor; wave-2 semantics).** The m1 wave's
linear axis distance rescaled to [0,1] put the hedge point at 0.5 with zero
downside and adjacent misses at 0.75 — a hedge sanctuary the policy measurably
drifted into (hedge_frac 0.217→0.311 over the full cell while the battery term
sat pinned at 0.60; m1 post-mortem R3). Items now score directly in [0, 1]:

    exact force                     1.0
    same-side commit, not exact     0.4   (obligatory vs recommended, etc.)
    hedge (permitted/missing/
        unparseable) vs decisive    0.15
    decisive vs a permitted gold    0.4 if |axis| == 1 else 0.15
    cross-side commit (antithesis)  0.0

preserving the design ordering exact > same-side > hedge > antithesis while
moving hedge from mid-range to near the floor and widening exact-vs-adjacent
from 0.25 to 0.6. Battery = plain mean (already [0,1]); ``R_vig = battery`` —
the citation Jaccard is DEMOTED to a logged diagnostic (its near-constant
0.1-0.3 range at 0.3 weight diluted the gradient; same post-mortem).

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

from .deontic import canonical_force

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

    Synonyms are canonicalised first (audit 2026-07-28: a future universe
    emitting "forbidden"/"must" would otherwise score at the hedge point
    while the battery balancer counted it decisively — desynchronising the
    invariant checker from the reward). A ``None``, unparseable, or
    non-force label collapses to the hedge point (``permitted``, axis 0).
    """
    if force is None:
        return _HEDGE_AXIS
    raw = str(force).strip().lower()
    canon = canonical_force(raw)
    return AXIS.get(canon if canon is not None else raw, _HEDGE_AXIS)


# The re-anchored per-item scale (2026-07-28). Module-level constants so the
# realized values are visible in configs/forensics discussions, not buried in
# a formula.
ITEM_SCORE_EXACT = 1.0
ITEM_SCORE_SAME_SIDE = 0.4
ITEM_SCORE_HEDGE = 0.15
ITEM_SCORE_CROSS = 0.0


def item_score(model_force: object, gold_force: object) -> float:
    """Per-item score on the re-anchored [0, 1] scale (2026-07-28).

    ``model_force`` that is ``None``, unparseable, or not one of the five forces
    is treated as ``permitted`` (the hedge point, axis 0). ``gold_force`` is a
    battery item's stored force; a stray non-force gold likewise collapses to
    the hedge point.

    Worked cells (gold = obligatory): exact 1.0 · recommended (same side) 0.4 ·
    hedge (permitted/missing) 0.15 · discouraged/prohibited (cross side) 0.0.
    Gold = permitted: permitted 1.0 · recommended/discouraged 0.4 ·
    obligatory/prohibited 0.15. The ordering exact > same-side > hedge >
    antithesis is the invariant; the WIDTHS are the fix — the m1 scale made
    hedging the risk-averse optimum and the policy learned exactly that.
    """
    # A NON-ANSWER (missing / unparseable / non-force string) is not the
    # same as answering "permitted": axis-collapsing both let an unparsed
    # slot score a full 1.0 on a permitted-gold item (audit 2026-07-28).
    # A non-answer earns hedge credit against EVERY gold, permitted included.
    if model_force is None:
        return ITEM_SCORE_HEDGE
    raw = str(model_force).strip().lower()
    canon = canonical_force(raw)
    if (canon if canon is not None else raw) not in AXIS:
        return ITEM_SCORE_HEDGE

    m = axis_of(model_force)
    g = axis_of(gold_force)
    if m == g:
        return ITEM_SCORE_EXACT
    if m == _HEDGE_AXIS:
        return ITEM_SCORE_HEDGE  # hedging against a decisive gold
    if g == _HEDGE_AXIS:
        # Decisive commit on a permitted act: mild (±1) is a near-miss,
        # extreme (±2) is as wrong as hedging on a decisive gold.
        return ITEM_SCORE_SAME_SIDE if abs(m) == 1 else ITEM_SCORE_HEDGE
    if m * g > 0:
        return ITEM_SCORE_SAME_SIDE
    return ITEM_SCORE_CROSS


def battery_score(item_scores: Sequence[float]) -> float:
    """Battery score = plain mean of per-item scores (each already in [0, 1]).

    The m1-era rescale from [−1, 1] is GONE with the re-anchored item scale —
    it was what parked the hedge point at 0.5 of full range. An **empty**
    sequence is a caller bug (a battery always has items) and raises
    ``ValueError`` rather than returning a silent neutral — the ambiguity
    would mask a build/routing error.
    """
    items = list(item_scores)
    if not items:
        raise ValueError("battery_score requires a non-empty item_scores sequence")
    return sum(float(s) for s in items) / len(items)


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

      * ``battery`` — mean per-item score on the re-anchored [0, 1] scale.
      * ``cite`` — mean per-item Jaccard citation score; an unparsed item
        contributes 0. DIAGNOSTIC ONLY since 2026-07-28 (logged, not
        rewarded): its near-constant range at 0.3 weight diluted the
        battery gradient in the m1 wave.
      * ``r_vig`` — ``battery`` (the vignette reward).
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

    battery = battery_score(item_scores)
    return {
        "battery": battery,
        "cite": sum(cites) / k,
        "r_vig": battery,  # cite demoted to diagnostic (2026-07-28)
        "hedge_frac": n_hedge / k,
        "antithesis_frac": n_antithesis / k,
        "parsed_frac": n_parsed / k,
    }
