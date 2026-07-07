"""Step-7 factor coverage vs. Tier A expert checklists, plus viewpoint
diversity for ensemble outputs."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List

from .matchers import alias_match, jaccard, normalize


def _factor_key(f: Dict[str, Any]) -> str:
    return f"{f.get('kind', '')}: {f.get('factor', '')}"


def score_factor_coverage(gold_factors: List[Dict[str, Any]], pred_factors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Recall of expert-raised factors; kind-level recall as a coarser signal.

    A gold factor counts as covered if any predicted factor alias-matches its
    description OR matches its kind AND shares affected-party surface forms.
    """
    gold = [f for f in gold_factors if isinstance(f, dict)]
    pred = [f for f in pred_factors if isinstance(f, dict)]
    if not gold:
        return {"factor_recall": None, "kind_recall": None, "n_gold": 0, "n_pred": len(pred)}

    covered = 0
    for g in gold:
        hit = False
        for p in pred:
            if alias_match(_factor_key(g), _factor_key(p), threshold=0.34):
                hit = True
                break
            same_kind = str(g.get("kind", "")).lower() == str(p.get("kind", "")).lower()
            parties_overlap = any(
                alias_match(gp, pp)
                for gp in g.get("affected_parties") or []
                for pp in p.get("affected_parties") or []
            )
            if same_kind and parties_overlap:
                hit = True
                break
        covered += hit

    gold_kinds = {str(g.get("kind", "")).lower() for g in gold}
    pred_kinds = {str(p.get("kind", "")).lower() for p in pred}
    kind_recall = len(gold_kinds & pred_kinds) / len(gold_kinds) if gold_kinds else None

    return {
        "factor_recall": round(covered / len(gold), 6),
        "kind_recall": round(kind_recall, 6) if kind_recall is not None else None,
        "n_gold": len(gold),
        "n_pred": len(pred),
    }


def viewpoint_diversity(texts: List[str]) -> Dict[str, Any]:
    """Mean pairwise token-Jaccard *distance* across ensemble member outputs.

    1.0 = every member said something lexically disjoint; 0.0 = clones.
    Crude but deterministic; embedding-based diversity can replace this once
    the embedding server is in the loop.
    """
    tokenized = [normalize(t) for t in texts if str(t).strip()]
    if len(tokenized) < 2:
        return {"mean_pairwise_distance": None, "n_members": len(tokenized)}
    dists = [1.0 - jaccard(a, b) for a, b in combinations(tokenized, 2)]
    return {
        "mean_pairwise_distance": round(sum(dists) / len(dists), 6),
        "n_members": len(tokenized),
    }
