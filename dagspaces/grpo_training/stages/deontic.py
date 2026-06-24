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
from typing import Any, Iterator, List, Optional

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


def expected_appropriateness(force: Optional[str]) -> Optional[str]:
    """Expected appropriateness label for a flow governed by ``force``.

    Returns None for "permitted"/unknown/missing — no directional expectation.
    """
    if not force:
        return None
    return FORCE_TO_APPROPRIATENESS.get(str(force).strip().lower())


def appropriateness_consistency(
    model_label: Optional[str], force: Optional[str]
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


def _norm_force(norm: dict) -> Optional[str]:
    """Read normative_force from a norm dict (cleaned or raz_-prefixed)."""
    return norm.get("normative_force") or norm.get("raz_normative_force") or None


def governing_norm_force(norm_universe_json: Any) -> Optional[str]:
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


def flow_appropriateness_labels(candidate_doc: Any) -> List[str]:
    """Extract appropriateness labels from a candidate's flows.

    Handles the SFT-schema extraction shape (``appropriateness`` on the
    extraction, or nested under ``flow``) and the reasoning-schema
    ``potential_appropriateness`` fallback.
    """
    labels: List[str] = []
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
    candidate_doc: Any, force: Optional[str]
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
