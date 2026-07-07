"""Surface-form matching for extraction scoring.

Gold and model artifacts name the same entities with different surface forms
("their doctor" vs "the patient's physician"). Matching is deliberately
lenient-but-auditable: normalized containment OR token-Jaccard >= threshold.
All scorers report through these functions so the leniency is centralized.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_STOPWORDS = {
    "the", "a", "an", "their", "its", "his", "her", "of", "for", "to", "and",
    "or", "in", "on", "with", "by", "that", "this", "who", "which", "s",
}

# Common actor/TP synonym groups: normalization collapses within a group.
_SYNONYMS = {
    "doctor": {"doctor", "physician", "gp", "clinician"},
    "patient": {"patient",},
    "consent": {"consent", "permission", "authorization", "authorisation"},
    "confidentiality": {"confidentiality", "confidence", "confidential", "secrecy", "secret"},
    "need": {"need", "necessity", "required", "requires", "needs"},
    "voluntary": {"voluntary", "voluntarily", "choice", "chooses", "choosing", "willingly"},
    "mandatory": {"mandatory", "compulsion", "compulsory", "mandated", "required-by-law", "legally-mandated"},
    "notice": {"notice", "notification", "informed", "transparency"},
    "company": {"company", "corporation", "firm", "business"},
    "insurer": {"insurer", "insurance"},
}
_CANON = {w: k for k, ws in _SYNONYMS.items() for w in ws}


def normalize(text: str) -> List[str]:
    """Lowercase, strip punctuation, drop stopwords, collapse synonyms."""
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    out = []
    for t in tokens:
        if t in _STOPWORDS:
            continue
        out.append(_CANON.get(t, t))
    return out


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def alias_match(gold: str, pred: str, threshold: float = 0.5) -> bool:
    """True if pred plausibly names the same thing as gold."""
    g, p = normalize(gold), normalize(pred)
    if not g or not p:
        return False
    gs, ps = " ".join(g), " ".join(p)
    if gs in ps or ps in gs:
        return True
    return jaccard(g, p) >= threshold


def set_prf(gold_items: List[str], pred_items: List[str], threshold: float = 0.5) -> Dict[str, float]:
    """Greedy one-to-one P/R/F1 between two string lists under alias_match."""
    gold_items = [g for g in gold_items if str(g).strip()]
    pred_items = [p for p in pred_items if str(p).strip()]
    if not gold_items and not pred_items:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_gold": 0, "n_pred": 0, "n_matched": 0}

    unmatched_pred = list(range(len(pred_items)))
    matched = 0
    for g in gold_items:
        hit = next((i for i in unmatched_pred if alias_match(g, pred_items[i], threshold)), None)
        if hit is not None:
            unmatched_pred.remove(hit)
            matched += 1

    precision = matched / len(pred_items) if pred_items else 0.0
    recall = matched / len(gold_items) if gold_items else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6),
        "n_gold": len(gold_items), "n_pred": len(pred_items), "n_matched": matched,
    }
