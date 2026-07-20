"""Extraction scoring (steps 1-4) against Tier A gold lists or Tier B
constructed values. Per-parameter P/R/F1 via the shared alias matcher."""

from __future__ import annotations

import ast
from typing import Any

from .matchers import alias_match, set_prf


def _artifact_lists(state: dict[str, Any]) -> dict[str, list[str]]:
    """Pull the scoreable string lists out of a traversal state's s1/s3/s4."""
    out: dict[str, list[str]] = {"senders": [], "recipients": [], "subjects": [],
                                   "information_types": [], "transmission_principles": []}
    s3 = state.get("s3") or {}
    for k in ("senders", "recipients", "subjects"):
        vals = s3.get(k) or []
        out[k] = [str(v) for v in vals if str(v).strip()]
    for flow in (state.get("s1") or {}).get("flows") or []:
        it = str(flow.get("information_type", "")).strip()
        if it:
            out["information_types"].append(it)
    for tp in (state.get("s4") or {}).get("transmission_principles") or []:
        p = str(tp.get("principle", "")).strip()
        if p:
            out["transmission_principles"].append(p)
    return out


def score_vs_tier_a(gold: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Per-parameter P/R/F1 of a traversal against a Tier A gold file."""
    pred = _artifact_lists(state)
    g3 = gold.get("s3_actors") or {}
    gold_lists = {
        "senders": g3.get("senders") or [],
        "recipients": g3.get("recipients") or [],
        "subjects": g3.get("subjects") or [],
        "information_types": [f.get("information_type", "") for f in gold.get("s1_flows") or []],
        "transmission_principles": [t.get("principle", "") for t in gold.get("s4_transmission_principles") or []],
    }
    return {param: set_prf(gold_lists[param], pred[param]) for param in gold_lists}


def score_vs_tier_b(gold_values: str, state: dict[str, Any]) -> dict[str, Any]:
    """Hit/miss per parameter against a Tier B row's constructed flow values.

    `gold_values` is the generator's stringified dict of the (possibly
    departing) flow values; each parameter scores 1 if any extracted surface
    form alias-matches the constructed value.
    """
    try:
        values = ast.literal_eval(gold_values or "{}")
    except (ValueError, SyntaxError):
        values = {}
    if not values:
        return {}

    pred = _artifact_lists(state)
    param_to_list = {
        "sender": "senders",
        "recipient": "recipients",
        "subject": "subjects",
        "information_type": "information_types",
        "transmission_principle": "transmission_principles",
    }
    out = {}
    for param, list_key in param_to_list.items():
        gold_val = str(values.get(param, "")).strip()
        if not gold_val:
            continue
        out[param] = {
            "hit": int(any(alias_match(gold_val, p) for p in pred[list_key])),
            "n_pred": len(pred[list_key]),
        }
    return out
