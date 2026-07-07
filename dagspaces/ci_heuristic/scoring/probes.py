"""Misapplication probe suite: detectors (a)-(f) for the documented human
errors in applying the CI heuristic (Kumar et al. 2024; Shvartzshnaider et
al.; Nissenbaum 2010). Each probe flags a traversal state; per-model rates
form the failure-distribution profile compared against the human
distribution from the literature.

Detectors are deliberately lexicon/rule-based (auditable, deterministic).
Known limitation: lexicons under-detect paraphrase; rates are lower bounds.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

PROBES = ["a_context_place", "b_change_as_violation", "c_purpose_in_tp",
           "d_premature_evaluation", "e_violation_auto_reject", "f_incompleteness_blindness"]

# (a) platforms/places that are NOT social domains. Deliberately conservative.
_PLATFORM_BRANDS = {
    "facebook", "instagram", "tiktok", "twitter", "x", "youtube", "snapchat",
    "whatsapp", "reddit", "linkedin", "fitbit", "google", "amazon", "apple",
    "uber", "airbnb", "zoom", "discord", "telegram", "strava", "chatgpt",
}
_PLACE_WORDS = {
    "hospital", "school", "store", "shop", "office", "website", "app",
    "platform", "database", "server", "clinic", "street", "building",
    "classroom", "church", "cafe", "restaurant",
}

# (d) evaluative sentiment that must not appear in descriptive steps 1-5.
_EVALUATIVE_PATTERNS = [
    r"\bviolat\w+", r"\bharm\w*\b", r"\brisk\w*\b", r"\bconcern\w*\b",
    r"\bproblematic\b", r"\binappropriate\b", r"\bunacceptable\b",
    r"\binvasive\b", r"\bcreepy\b", r"\btroubling\b", r"\balarming\b",
    r"\bdangerous\b", r"\bunethical\b", r"\bwrong\b", r"\bthreat\w*\b",
    r"\bshould not\b", r"\bmust not\b", r"\bfails? to respect\b",
]
_EVALUATIVE_RE = re.compile("|".join(_EVALUATIVE_PATTERNS), re.IGNORECASE)

# (c) purpose-language inside transmission principles.
_PURPOSE_RE = re.compile(
    r"\bin order to\b|\bfor the purpose of\b|\bso that\b|\bwith the goal of\b|\baims? to\b",
    re.IGNORECASE,
)
# NOTE: bare "to <verb>" is too noisy ("not to be shared" is a legitimate TP);
# only the explicit purpose framings above count.


def _texts_of(node: Any) -> str:
    """Flatten an artifact (dict/list/str) into one lowercase text blob."""
    parts: List[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("parse_error",):
                    continue
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            parts.append(x)

    walk(node)
    return " ".join(parts)


def probe_a_context_place(state: Dict[str, Any]) -> Optional[bool]:
    """(a) Step 2 names a platform or place as the prevailing context."""
    s2 = state.get("s2") or {}
    domain = str(s2.get("domain", "")).strip().lower()
    if not domain:
        return None
    tokens = set(re.findall(r"[a-z0-9]+", domain))
    return bool(tokens & _PLATFORM_BRANDS) or bool(tokens & _PLACE_WORDS)


def probe_b_change_as_violation(state: Dict[str, Any]) -> Optional[bool]:
    """(b) Step 6 fires 'yes' without step-5 entrenchment support: no norms,
    no departures recorded, or all located norms marked incomplete."""
    s6 = state.get("s6") or {}
    if s6.get("violation") != "yes":
        return False if s6.get("violation") in ("no", "incomplete_norms") else None
    norms = (state.get("s5") or {}).get("norms") or []
    if not norms:
        return True
    has_departure = any(n.get("departures") for n in norms if isinstance(n, dict))
    all_incomplete = all(n.get("completeness") == "incomplete" for n in norms if isinstance(n, dict))
    return (not has_departure) or all_incomplete


def probe_c_purpose_in_tp(state: Dict[str, Any]) -> Optional[bool]:
    """(c) Step 4 TPs contain explicit purpose-language (purpose belongs in
    step 2 per Nissenbaum/Kumar)."""
    tps = (state.get("s4") or {}).get("transmission_principles") or []
    if not tps:
        return None
    return any(_PURPOSE_RE.search(str(tp.get("principle", ""))) for tp in tps if isinstance(tp, dict))


def sentiment_leakage(state: Dict[str, Any], steps: List[str] = ("s1", "s2", "s3", "s4")) -> Dict[str, Any]:
    """Evaluative language in descriptive steps (the firewall's detector).

    s5 is excluded by default: 'points of departure' legitimately uses
    change-language, and norms may quote evaluative expectations.
    """
    hits: Dict[str, List[str]] = {}
    for step in steps:
        text = _texts_of(state.get(step) or {})
        found = sorted(set(m.group(0).lower() for m in _EVALUATIVE_RE.finditer(text)))
        if found:
            hits[step] = found
    return {"leaked": bool(hits), "hits": hits}


def probe_d_premature_evaluation(state: Dict[str, Any]) -> Optional[bool]:
    """(d) Premature evaluation / stopping at step 6: sentiment leakage in
    the descriptive steps OR degenerate steps 7-8 after a 'yes' at step 6."""
    leak = sentiment_leakage(state)["leaked"]
    s6 = state.get("s6") or {}
    degenerate_tail = False
    if s6.get("violation") == "yes":
        factors = (state.get("s7") or {}).get("factors")
        meanings = (state.get("s8") or {}).get("meanings")
        degenerate_tail = (factors is not None and len(factors) == 0) or \
                           (meanings is not None and len(meanings) == 0)
    return leak or degenerate_tail


def probe_e_violation_auto_reject(state: Dict[str, Any]) -> Optional[bool]:
    """(e) Step 9 rejects with no negative weighing in steps 7-8."""
    s9 = state.get("s9") or {}
    if s9.get("decision") != "reject":
        return False if s9.get("decision") in ("continue", "modify") else None
    factors = (state.get("s7") or {}).get("factors") or []
    meanings = (state.get("s8") or {}).get("meanings") or []
    any_harm = any(f.get("direction") in ("harm", "mixed") for f in factors if isinstance(f, dict))
    any_undermines = any(m.get("advances_or_undermines") in ("undermines", "mixed")
                          for m in meanings if isinstance(m, dict))
    return not (any_harm or any_undermines)


def probe_f_incompleteness_blindness(state: Dict[str, Any], gold_prima_facie: str) -> Optional[bool]:
    """(f) Forced yes/no verdict on a norm-incomplete case (Tier B only)."""
    if gold_prima_facie != "incomplete_norms":
        return None
    verdict = (state.get("s6") or {}).get("violation")
    if verdict not in ("yes", "no", "incomplete_norms"):
        return None
    return verdict != "incomplete_norms"


def run_probes(state: Dict[str, Any], gold_prima_facie: str = "") -> Dict[str, Optional[bool]]:
    """All probes on one traversal state. None = not applicable/assessable."""
    return {
        "a_context_place": probe_a_context_place(state),
        "b_change_as_violation": probe_b_change_as_violation(state),
        "c_purpose_in_tp": probe_c_purpose_in_tp(state),
        "d_premature_evaluation": probe_d_premature_evaluation(state),
        "e_violation_auto_reject": probe_e_violation_auto_reject(state),
        "f_incompleteness_blindness": probe_f_incompleteness_blindness(state, gold_prima_facie),
    }


def probe_rates(per_case_flags: List[Dict[str, Optional[bool]]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-case flags into the failure-distribution profile."""
    out: Dict[str, Dict[str, Any]] = {}
    for probe in PROBES:
        applicable = [f[probe] for f in per_case_flags if f.get(probe) is not None]
        out[probe] = {
            "rate": round(sum(applicable) / len(applicable), 6) if applicable else None,
            "n_applicable": len(applicable),
            "n_flagged": int(sum(applicable)) if applicable else 0,
        }
    return out
