"""Deterministic deontic-force → appropriateness reasoning.

The reranker judge (:class:`~.clients.RerankerJudgeClient`) measures
*relevance* — norm awareness and flow governance — but is structurally blind
to **appropriateness consistency**: whether the model's appropriate /
inappropriate verdict agrees with the deontic direction the governing norm
prescribes. A flow can be maximally relevant to a norm while the model draws
the opposite normative conclusion, which is the worst failure mode for the
paper's thesis.

That check needs no LLM. The retrieved norms carry a structured Raz
``normative_force``, so a flow governed by a *prohibiting* norm ought to be
judged ``inappropriate``, by an *obligating* norm ``appropriate``, etc. This
module is the single source of truth for that force → judgment mapping, shared
with the GRPO judgment-vignette gold labels
(:func:`grpo_training._build_vignettes`).
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

# normative_force → expected appropriateness of a flow the norm governs.
# obligatory/recommended ⇒ the flow ought to occur (appropriate);
# prohibited/discouraged ⇒ it ought not (inappropriate); "permitted" and
# unknown carry no directional expectation.
FORCE_TO_APPROPRIATENESS = {
    "obligatory": "appropriate",
    "recommended": "appropriate",
    "prohibited": "inappropriate",
    "discouraged": "inappropriate",
}

# The yes/no view of the same mapping (appropriate→yes), used by the judgment
# vignettes. Kept here so the two never drift apart.
FORCE_TO_GOLD = {
    "obligatory": "yes",
    "recommended": "yes",
    "prohibited": "no",
    "discouraged": "no",
}

# Returned when the axis can't be adjudicated (permitted/unknown force, missing
# or "ambiguous" model label). Deliberately neutral: never punish a flow for an
# appropriateness call we have no ground to check.
NEUTRAL_CONSISTENCY = 0.5


def expected_appropriateness(force: str | None) -> str | None:
    """Expected appropriateness label for a flow governed by ``force``.

    Returns None for "permitted"/unknown/missing — no directional expectation.
    """
    if not force:
        return None
    return FORCE_TO_APPROPRIATENESS.get(str(force).strip().lower())


def appropriateness_consistency(
    model_label: str | None, force: str | None
) -> float:
    """Graded agreement between a model appropriateness label and a norm force.

    1.0 = the label matches the norm's deontic direction;
    0.0 = it contradicts it;
    ``NEUTRAL_CONSISTENCY`` (0.5) = undetermined (permitted/unknown force, or a
    missing/"ambiguous" model label) — neither rewarded nor penalized.
    """
    expected = expected_appropriateness(force)
    if expected is None or not model_label:
        return NEUTRAL_CONSISTENCY
    lab = str(model_label).strip().lower()
    if lab == "ambiguous":
        return NEUTRAL_CONSISTENCY
    if lab == expected:
        return 1.0
    if lab in ("appropriate", "inappropriate"):
        return 0.0
    return NEUTRAL_CONSISTENCY


def _norm_force(norm: dict) -> str | None:
    """Read normative_force from a norm dict (cleaned or raz_-prefixed)."""
    return norm.get("normative_force") or norm.get("raz_normative_force") or None


def governing_norm_force(norm_universe_json: Any) -> str | None:
    """The governing norm's force = that of the top (most-similar) retrieved norm.

    ``NormRetriever`` returns norms in descending cosine-similarity order, so
    index 0 is the most relevant — the norm most plausibly governing the flow.
    Accepts a JSON string or an already-parsed list.
    """
    norms = norm_universe_json
    if isinstance(norm_universe_json, str):
        try:
            norms = json.loads(norm_universe_json)
        except json.JSONDecodeError:
            return None
    if not isinstance(norms, list) or not norms:
        return None
    first = norms[0]
    return _norm_force(first) if isinstance(first, dict) else None


def _iter_flow_dicts(candidate_doc: Any) -> Iterator[dict]:
    """Yield flow/extraction dicts from a candidate document (JSON list or dict)."""
    data = candidate_doc
    if isinstance(candidate_doc, str):
        try:
            data = json.loads(candidate_doc)
        except json.JSONDecodeError:
            return
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return
    for item in data:
        if isinstance(item, dict):
            yield item


def flow_appropriateness_labels(candidate_doc: Any) -> list[str]:
    """Extract appropriateness labels from a candidate's flows.

    Handles the SFT-schema extraction shape (``appropriateness`` on the
    extraction, or nested under ``flow``) and the reasoning-schema
    ``potential_appropriateness`` fallback.
    """
    labels: list[str] = []
    for item in _iter_flow_dicts(candidate_doc):
        lab = item.get("appropriateness")
        if lab is None and isinstance(item.get("flow"), dict):
            lab = item["flow"].get("appropriateness")
        if lab is None:
            lab = item.get("potential_appropriateness")
        if lab:
            labels.append(str(lab))
    return labels


def candidate_appropriateness_consistency(
    candidate_doc: Any, force: str | None
) -> float:
    """Mean appropriateness-consistency over a candidate's flows.

    Returns ``NEUTRAL_CONSISTENCY`` when the candidate exposes no
    appropriateness labels (e.g. a no-flow declaration), so it neither gains
    nor loses on this axis.
    """
    labels = flow_appropriateness_labels(candidate_doc)
    if not labels:
        return NEUTRAL_CONSISTENCY
    vals = [appropriateness_consistency(lab, force) for lab in labels]
    return sum(vals) / len(vals)


def direction_multiplier(consistency: float, floor: float = 0.4) -> float:
    """Map appropriateness-consistency [0,1] to a reward multiplier [floor,1].

    The v9 two-sided reward uses this to *multiply* R_ground by how well the
    model's appropriate/inappropriate verdict agrees with the governing norm's
    deontic force, instead of the v8 additive blend that left mis-judgment
    nearly free. Affine: ``floor + (1-floor)*consistency``, so

      consistency 1.0 (correct direction)        → 1.0
      consistency 0.5 (hedge / "ambiguous" / no label) → (1+floor)/2  (0.7 at floor=0.4)
      consistency 0.0 (wrong direction)          → floor (0.4)

    The floor (>0) *discounts* a wrong/hedged verdict rather than annihilating
    the reward, so deontic-retrieval noise (a wrong top-norm) cannot zero an
    otherwise well-grounded extraction (cf. the v8 symmetric-clamp clamp-to-zero
    failure mode).
    """
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"direction floor must be in [0, 1], got {floor}")
    c = max(0.0, min(1.0, float(consistency)))
    return floor + (1.0 - floor) * c


def appropriateness_multiplier(
    model_label: str | None,
    force: str | None,
    floor: float = 0.4,
    floor_prohibit: float | None = None,
    hedge_prohibit: float | None = None,
) -> float:
    """Cost-sensitive (asymmetric) direction multiplier for one flow (v10/v12a).

    Identical to ``direction_multiplier(appropriateness_consistency(label, force),
    floor)`` *except* it can punish a **false-permit** — the model calling a
    prohibited/discouraged-governed flow ``appropriate`` — harder than a
    false-forbid, via ``floor_prohibit`` (v10), and a **hedge on a
    prohibited/discouraged-governed flow** harder than a hedge elsewhere, via
    ``hedge_prohibit`` (v12a):

      correct verdict (either direction)            → 1.0
      hedge, no directional force                   → (1+floor)/2  (0.7 at floor 0.4)
      hedge, norm prohibits                             → hedge_prohibit (0.5)
      false-forbid (said inappropriate, norm obligates) → floor          (0.4)
      false-permit (said appropriate, norm prohibits)   → floor_prohibit  (0.1)

    A "hedge" is any non-committed verdict on an extracted flow: "ambiguous", a
    missing label, or an unrecognized one (so garbage labels can't keep the
    neutral tier on prohibited flows).

    The fiction-derived governing norms are ~4:1 appropriate:inappropriate, so
    under the symmetric v9 floor the EV-optimal verdict when unsure is the
    permissive one — the measured cause of the 30% Forbid commit-accuracy. A
    lower ``floor_prohibit`` steepens the within-group gradient on prohibited
    flows toward the (correct) ``inappropriate`` verdict. v10/v11 forensics then
    showed the floor punished the false-permit *tail* but left hedging the safe
    optimum — prohibited-flow hedge mass froze at ~72% across both runs because
    under R = base × direction a well-grounded hedge (×0.7) routinely outscores
    a mediocre-grounded correct commit (×1.0). ``hedge_prohibit`` (< the neutral
    0.7, > ``floor_prohibit``) widens the commit-vs-hedge gap exactly where it
    binds while leaving hedges on non-prohibited flows at the neutral tier.
    Both knobs ``None`` reproduces the symmetric v9 multiplier exactly;
    ``hedge_prohibit=None`` alone reproduces v10.
    """
    fp = floor if floor_prohibit is None else floor_prohibit
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}")
    if not 0.0 <= fp <= 1.0:
        raise ValueError(f"floor_prohibit must be in [0, 1], got {fp}")
    if hedge_prohibit is not None and not 0.0 <= hedge_prohibit <= 1.0:
        raise ValueError(f"hedge_prohibit must be in [0, 1], got {hedge_prohibit}")
    expected = expected_appropriateness(force)
    if expected is None:
        return direction_multiplier(NEUTRAL_CONSISTENCY, floor)

    def _hedge() -> float:
        if expected == "inappropriate" and hedge_prohibit is not None:
            return hedge_prohibit
        return direction_multiplier(NEUTRAL_CONSISTENCY, floor)

    if not model_label:
        return _hedge()
    lab = str(model_label).strip().lower()
    if lab == "ambiguous":
        return _hedge()
    if lab == expected:
        return 1.0
    if lab in ("appropriate", "inappropriate"):
        # Wrong direction. A false-permit (model said the *appropriate* label
        # while the norm prohibits) gets the steeper floor; a false-forbid keeps
        # the general floor.
        return fp if expected == "inappropriate" else floor
    return _hedge()


def candidate_appropriateness_multiplier(
    candidate_doc: Any,
    force: str | None,
    floor: float = 0.4,
    floor_prohibit: float | None = None,
    hedge_prohibit: float | None = None,
) -> float:
    """Mean cost-sensitive multiplier over a candidate's flows (v10/v12a).

    The cost-sensitive map is non-linear in the per-flow verdict, so unlike the
    v9 path (mean *consistency* then one ``direction_multiplier``) this maps each
    flow to its multiplier first, then means. Single-flow candidates (the common
    case) are identical either way. A candidate with no appropriateness labels
    (e.g. a no-flow declaration) returns the neutral multiplier, so it neither
    gains nor loses on this axis — deliberately ALSO under ``hedge_prohibit``:
    the v12a hedge tier prices a hedged verdict on an *extracted* flow, never
    abstention itself (no-flow economics are owned by no_flow_reward and the
    judge's unjustified-no-flow ranking; the no_flow promotion gate already runs
    hot at ~0.6 and must not be pushed further toward extraction).
    """
    labels = flow_appropriateness_labels(candidate_doc)
    if not labels:
        return direction_multiplier(NEUTRAL_CONSISTENCY, floor)
    vals = [
        appropriateness_multiplier(lab, force, floor, floor_prohibit, hedge_prohibit)
        for lab in labels
    ]
    return sum(vals) / len(vals)
