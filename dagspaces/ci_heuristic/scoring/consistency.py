"""Consistency checks [P5]: is the step-9 recommendation entailed by the
steps 6-8 findings, and do verdicts survive normatively-irrelevant
perturbations (comparators for the E7 harness)?

Consistency is necessary, not sufficient: a consistent traversal can still
be wrong, but an inconsistent one is not doing the heuristic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def check_entailment(state: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based contradictions between s6-s8 findings and the s9 decision.

    Rules (each yields a named violation when triggered):
      R1 reject_without_negatives — decision=reject but no harm factor and no
         'undermines' meaning (probe e's condition, kept here as the
         consistency view of it).
      R2 continue_over_undermining — decision=continue (no conditions) while
         s8 says the practice undermines a contextual end.
      R3 no_violation_but_reject — s6=no violation, no harms raised, yet reject.
      R4 modify_without_conditions — decision=modify but conditions empty.
      R5 carrying_findings_empty — a decision with no carrying findings.
    """
    s6 = state.get("s6") or {}
    s9 = state.get("s9") or {}
    factors = (state.get("s7") or {}).get("factors") or []
    meanings = (state.get("s8") or {}).get("meanings") or []
    decision = s9.get("decision")

    if decision not in ("continue", "modify", "reject"):
        return {"assessable": False, "violations": [], "consistent": None}

    any_harm = any(f.get("direction") in ("harm", "mixed") for f in factors if isinstance(f, dict))
    any_undermines = any(m.get("advances_or_undermines") in ("undermines", "mixed")
                          for m in meanings if isinstance(m, dict))

    violations: List[str] = []
    if decision == "reject" and not (any_harm or any_undermines):
        violations.append("reject_without_negatives")
    if decision == "continue" and not (s9.get("conditions") or []) and any_undermines:
        violations.append("continue_over_undermining")
    if decision == "reject" and s6.get("violation") == "no" and not any_harm:
        violations.append("no_violation_but_reject")
    if decision == "modify" and not (s9.get("conditions") or []):
        violations.append("modify_without_conditions")
    if not (s9.get("carrying_findings") or []):
        violations.append("carrying_findings_empty")

    return {"assessable": True, "violations": violations, "consistent": not violations}


def verdict_of(state: Dict[str, Any]) -> Optional[str]:
    return (state.get("s9") or {}).get("decision")


def flip_rate(states_a: List[Dict[str, Any]], states_b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verdict flip rate between two aligned runs (invariance/sensitivity
    comparator: paraphrase runs should NOT flip; parameter-flip runs SHOULD)."""
    pairs = [
        (verdict_of(a), verdict_of(b))
        for a, b in zip(states_a, states_b)
        if verdict_of(a) in ("continue", "modify", "reject")
        and verdict_of(b) in ("continue", "modify", "reject")
    ]
    if not pairs:
        return {"flip_rate": None, "n_paired": 0}
    flips = sum(a != b for a, b in pairs)
    return {"flip_rate": round(flips / len(pairs), 6), "n_paired": len(pairs)}
