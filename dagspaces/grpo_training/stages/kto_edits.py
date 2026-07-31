"""Edit surgery for the k-series KTO dataset (wiki/2026-07-31_kto_plan.md §4).

The supervision-depth ladder builds *desirable* counterfactuals from a policy
completion by progressively deeper edits, each derived from the chunk-gold
index (the audited m2 gold chain — never a model opinion):

  * :func:`apply_verdict_edit`    (R-VERDICT)    — flip the matched flows'
    ``appropriateness`` to gold. One enum value per flow.
  * :func:`apply_citation_edit`   (R-CITATION)   — verdict + REPLACE the
    corrected flow's ``norms_invoked`` with the governing norm's articulation
    (the norm whose force decided the gold at k=1).
  * :func:`apply_scrutinize_edit` (R-SCRUTINIZE) — citation + append a
    deterministic norm→judgment rationale to the completion's top-level
    ``reasoning`` field (template variant; the teacher-generated variant is
    produced by the K0/K1 pipeline and validated against
    :func:`rationale_is_valid`).

All functions are PURE: they take the gate-parsed completion object (the
``GateResult.parsed`` dict) plus a list of :class:`Correction` records, and
return a NEW dict — inputs are never mutated. :func:`serialize_completion`
renders the edited object back to the SFT completion format (compact JSON,
``ensure_ascii=False``) and every edit is expected to round-trip through
``valid_gate`` — the K1 build asserts it per row.

Additive k-series code (parallel-stack rule): imports m-series/keeper
surfaces, edits none.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Correction:
    """One matched, mislabeled flow to correct.

    ``flow_index`` indexes the completion's ``flows`` list (the gate-parsed
    order). ``gold`` is the norm-derived appropriateness. ``norm`` is the
    governing norm's info dict from the chunk-gold index
    (``make_direct_chunk_gold(..., keep_norm_info=True)``): keys
    ``articulation`` / ``normative_force`` / ``act_polarity`` /
    ``norm_subject`` / ``norm_act`` / ``condition_of_application`` /
    ``context``. ``match_sim`` is the greedy-match cosine (edit eligibility
    is thresholded upstream by ``kto.min_edit_sim``).
    """

    flow_index: int
    gold: str
    norm: dict[str, Any]
    match_sim: float


def _check_corrections(parsed: dict, corrections: list[Correction]) -> None:
    flows = parsed.get("flows")
    if not isinstance(flows, list):
        raise ValueError("parsed completion has no flows list (gate should "
                         "have rejected it)")
    for c in corrections:
        if not 0 <= c.flow_index < len(flows):
            raise ValueError(f"correction flow_index {c.flow_index} out of "
                             f"range for {len(flows)} flows")
        if c.gold not in ("appropriate", "inappropriate"):
            raise ValueError(f"correction gold {c.gold!r} is not a decisive "
                             "label")


def apply_verdict_edit(parsed: dict, corrections: list[Correction]) -> dict:
    """R-VERDICT: flip corrected flows' ``appropriateness`` to gold."""
    _check_corrections(parsed, corrections)
    out = copy.deepcopy(parsed)
    for c in corrections:
        out["flows"][c.flow_index]["appropriateness"] = c.gold
    return out


def apply_citation_edit(parsed: dict, corrections: list[Correction]) -> dict:
    """R-CITATION: verdict + governing-norm articulation as the citation.

    The corrected flow's ``norms_invoked`` is REPLACED with the single gold
    articulation (plan §4: the desirable behavior is citing the norm that
    actually decides the judgment — keeping the policy's zero-shot citations
    alongside would blur the contrast). ``norm_source`` is set to
    ``"explicit"``: the articulation is the book's own stated rule.
    Corrections without an articulation fall back to the verdict edit for
    that flow (counted by the caller).
    """
    out = apply_verdict_edit(parsed, corrections)
    for c in corrections:
        art = (c.norm or {}).get("articulation")
        if not art:
            continue
        flow = out["flows"][c.flow_index]
        flow["norms_invoked"] = [str(art)]
        if "norm_source" in flow:
            flow["norm_source"] = "explicit"
    return out


#: Deterministic rationale template (R-SCRUTINIZE, template variant).
#: {force} is the canonical force word; {act_polarity_clause} handles the
#: refraining inversion in prose so the stated inference matches
#: deontic.flow_appropriateness exactly.
_RATIONALE_TEMPLATE = (
    'Flow {n}: the governing norm here is "{articulation}" — it holds that '
    "{subject} {force_phrase} {act}{condition_clause}. "
    "{link_clause} Therefore this flow is {gold}."
)

_FORCE_PHRASE = {
    "obligatory": "is obligated to",
    "recommended": "is encouraged to",
    "permitted": "is permitted to",
    "discouraged": "is discouraged from",
    "prohibited": "is prohibited from",
}


def render_rationale(flow_index: int, c: Correction) -> str:
    """The deterministic norm→judgment rationale for one corrected flow."""
    n = c.norm or {}
    force = str(n.get("normative_force") or "").strip().lower()
    subject = str(n.get("norm_subject") or "the actor").strip()
    act = str(n.get("norm_act") or "act this way").strip().rstrip(".")
    cond = str(n.get("condition_of_application") or "").strip().rstrip(".")
    pol = str(n.get("act_polarity") or "performing").strip().lower()
    link = ("The flow enacts what the norm addresses"
            if pol == "performing"
            else "The norm concerns refraining from this act, so performing "
                 "it runs against the norm's direction")
    return _RATIONALE_TEMPLATE.format(
        n=flow_index + 1,
        articulation=str(n.get("articulation") or "").strip(),
        subject=subject,
        force_phrase=_FORCE_PHRASE.get(force, f"is {force} to"),
        act=act,
        condition_clause=f", {cond[0].lower()}{cond[1:]}" if cond else "",
        link_clause=f"{link}.",
        gold=c.gold,
    )


def apply_scrutinize_edit(
    parsed: dict, corrections: list[Correction],
    rationales: "list[str] | None" = None,
) -> dict:
    """R-SCRUTINIZE: citation edit + norm→judgment rationale in ``reasoning``.

    ``rationales`` overrides the deterministic template per correction (the
    teacher-generated variant); each must pass :func:`rationale_is_valid`
    upstream. Rationales are APPENDED to the existing ``reasoning`` text —
    the policy's narrative stays (the edit teaches the added inference, not
    a rewrite of everything it said), separated by a single space, in
    flow-index order.
    """
    out = apply_citation_edit(parsed, corrections)
    parts = [str(out.get("reasoning") or "").strip()]
    ordered = sorted(corrections, key=lambda c: c.flow_index)
    for i, c in enumerate(ordered):
        text = (rationales[i] if rationales is not None
                else render_rationale(c.flow_index, c))
        parts.append(str(text).strip())
    out["reasoning"] = " ".join(p for p in parts if p)
    return out


def rationale_is_valid(text: str, c: Correction) -> bool:
    """Cheap string validation for a (teacher-generated) rationale.

    Must (a) reference the governing norm — at least a 6-word contiguous
    fragment of the articulation appears, case-insensitive; (b) conclude the
    gold label; (c) NOT conclude the opposite label more prominently (the
    opposite word may appear inside the argument, but the gold label must
    appear after its last occurrence — i.e. the conclusion is the gold).
    """
    t = " ".join(str(text).lower().split())
    art_words = " ".join(str((c.norm or {}).get("articulation") or "")
                         .lower().split()).split(" ")
    if len(art_words) >= 6:
        found = any(" ".join(art_words[i:i + 6]) in t
                    for i in range(len(art_words) - 5))
    else:
        found = " ".join(art_words) in t if art_words != [""] else False
    if not found:
        return False
    gold, other = c.gold, (
        "inappropriate" if c.gold == "appropriate" else "appropriate")
    # 'inappropriate' contains 'appropriate' — count word-boundary matches.
    import re
    gold_pos = [m.start() for m in re.finditer(rf"\b{gold}\b", t)]
    other_pos = [m.start() for m in re.finditer(rf"\b{other}\b", t)]
    if not gold_pos:
        return False
    return max(gold_pos) > (max(other_pos) if other_pos else -1)


def serialize_completion(parsed: dict) -> str:
    """Render an edited completion back to the SFT completion format.

    Same convention as ``sft_data_prep`` (line ~334): compact
    ``json.dumps(..., ensure_ascii=False)``. The K1 build asserts every
    serialized edit round-trips through ``valid_gate`` unchanged in flow
    count and labels.
    """
    return json.dumps(parsed, ensure_ascii=False)
