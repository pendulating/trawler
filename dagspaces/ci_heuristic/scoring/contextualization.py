"""Contextualization score for step 8 [point (4)]: is each s8 entry argued
relative to a SPECIFIC contextual end, or is it generic-ethics boilerplate?

Three-part LLM-judge rubric per entry:
  (i)  specific_end   — a specific end of THIS context is named (not
       "society", "privacy", "ethics").
  (ii) argued_relative — the factor's significance is argued relative to that
       end, not merely restated next to it.
  (iii) transplant_survives — would the argument read unchanged for a
       different context? If yes, it is generic (scores 0 overall).

Entry score = 1 only if (i) and (ii) and NOT (iii). The judge runs on the
existing judge-server infra (cluster); this module owns the prompt, schema,
parsing, and aggregation so the judge stage is thin. Calibrate against a
hand-scored subset before trusting (TODOS Phase 4).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

JUDGE_SYS = (
    "You are auditing one step of a Contextual Integrity analysis. Judge "
    "strictly by the rubric; do not reward eloquence."
)

JUDGE_TEMPLATE = """The analysis concerns the context: {context_domain}
(stated contextual values/ends: {values_ends}).

A step-8 entry claims the following about a moral/political factor:
- factor: {factor}
- contextual end invoked: {contextual_end}
- direction: {direction}
- argument: {argument}

Answer three questions:
1. specific_end: Does the entry name a SPECIFIC end, value, or purpose of THIS
   context (e.g., "managing illness" for health care)? Vague universals
   ("society", "privacy", "people's wellbeing", "ethics") do not count.
2. argued_relative: Is the factor's significance ARGUED in relation to that
   end (a claim about how the factor advances/undermines it), rather than the
   factor merely being restated next to the end?
3. transplant_survives: If you replaced the context with a completely
   different one (say, {transplant_context}), would this argument read just as
   well unchanged? (If yes, the argument is generic.)

Return JSON: {{"specific_end": true/false, "argued_relative": true/false,
"transplant_survives": true/false, "rationale": "<one sentence>"}}"""

# Context used in the transplant test — rotated to avoid a fixed anchor.
TRANSPLANT_CONTEXTS = ["competitive sailing", "municipal waste management", "amateur astronomy"]


class ContextualizationVerdict(BaseModel):
    """Judge output for one s8 entry."""
    specific_end: bool = Field(...)
    argued_relative: bool = Field(...)
    transplant_survives: bool = Field(...)
    rationale: str = Field(...)


def build_judge_prompts(state: Dict[str, Any], case_idx: int = 0) -> List[Tuple[str, str]]:
    """One (sys, usr) prompt per s8 entry in a traversal state."""
    s2 = state.get("s2") or {}
    factors = (state.get("s7") or {}).get("factors") or []
    prompts: List[Tuple[str, str]] = []
    for i, m in enumerate((state.get("s8") or {}).get("meanings") or []):
        if not isinstance(m, dict):
            continue
        ref = m.get("factor_ref")
        factor = ""
        if isinstance(ref, int) and 0 <= ref < len(factors):
            factor = str(factors[ref].get("factor", ""))
        usr = JUDGE_TEMPLATE.format(
            context_domain=s2.get("domain", "(unstated)"),
            values_ends=", ".join(s2.get("values_ends") or []) or "(unstated)",
            factor=factor or "(unresolved factor_ref)",
            contextual_end=m.get("contextual_end", ""),
            direction=m.get("advances_or_undermines", ""),
            argument=m.get("argument", ""),
            transplant_context=TRANSPLANT_CONTEXTS[(case_idx + i) % len(TRANSPLANT_CONTEXTS)],
        )
        prompts.append((JUDGE_SYS, usr))
    return prompts


def entry_score(verdict: Dict[str, Any]) -> int:
    """1 iff specific + argued + fails the transplant test."""
    return int(
        bool(verdict.get("specific_end"))
        and bool(verdict.get("argued_relative"))
        and not bool(verdict.get("transplant_survives"))
    )


def aggregate(verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Case-level contextualization metrics from per-entry judge verdicts."""
    if not verdicts:
        return {"contextualization_score": None, "n_entries": 0}
    scores = [entry_score(v) for v in verdicts]
    return {
        "contextualization_score": round(sum(scores) / len(scores), 6),
        "generic_rate": round(sum(bool(v.get("transplant_survives")) for v in verdicts) / len(verdicts), 6),
        "n_entries": len(verdicts),
    }


def parse_judge_reply(text: str) -> Dict[str, Any] | None:
    """Parse a judge completion; None when unparseable."""
    raw = str(text or "").strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    if not (0 <= start < end):
        return None
    try:
        obj = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "specific_end" not in obj:
        return None
    return obj
