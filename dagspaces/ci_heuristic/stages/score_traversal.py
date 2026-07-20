"""Score traversal artifacts: extraction F1, prima facie, misapplication
probes, entailment consistency, factor coverage — everything that needs no
LLM judge (contextualization runs as its own judge stage on the cluster).

Input: the long-format traverse output + the cases dataset (for tier labels
and gold paths). Output: per-case flags parquet + aggregate metrics with
provenance.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from dagspaces.common.metric_provenance import MetricEmitter

from ..scoring.consistency import check_entailment
from ..scoring.coverage import score_factor_coverage
from ..scoring.extraction import score_vs_tier_a, score_vs_tier_b
from ..scoring.prima_facie import score_prima_facie
from ..scoring.probes import probe_rates, run_probes, sentiment_leakage

logger = logging.getLogger(__name__)

EXTRACTION_PARAMS = ["senders", "recipients", "subjects", "information_types", "transmission_principles"]


def _reconstruct_states(traverse_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """case_id -> {step: artifact} from the long-format traverse output.

    Monolithic (L0/L1) rows: an L1 artifact is exploded into per-step keys
    (s1_flows -> s1, ...); L0 verdicts map onto a pseudo-s9 for probe/
    consistency purposes only.
    """
    states: dict[str, dict[str, Any]] = {}
    for _, row in traverse_df.iterrows():
        cid = str(row["case_id"])
        artifact = json.loads(row["artifact_json"])
        state = states.setdefault(cid, {})
        step = str(row["step"])
        if ":" in step:
            continue  # per-member deliberation rows; only merged artifacts score
        if step != "monolithic":
            state[step] = artifact
            continue
        if "s1_flows" in artifact:  # L1
            for k, v in artifact.items():
                if len(k) > 1 and k[0] == "s" and k[1].isdigit():
                    state[k[:2]] = v
        elif "decision" in artifact:  # L0
            state["s9"] = {"decision": artifact.get("decision"),
                            "conditions": [], "carrying_findings": ["l0_zero_shot"]}
    return states


def score_traversals(traverse_df: pd.DataFrame, cases_df: pd.DataFrame) -> dict[str, Any]:
    """Full no-judge scoring pass. Returns (metrics dict, per-case DataFrame)."""
    states = _reconstruct_states(traverse_df)
    cases = cases_df.set_index("case_id")

    em = MetricEmitter()
    em.emit_raw("n_cases", len(states))
    per_case_rows: list[dict[str, Any]] = []
    probe_flags: list[dict[str, Any]] = []
    tier_b_pf_cases: list[dict[str, Any]] = []
    tier_a_extraction: list[dict[str, Any]] = []
    tier_b_extraction_hits: dict[str, list[int]] = {}
    coverage_scores: list[dict[str, Any]] = []
    entailment_results: list[dict[str, Any]] = []

    for cid, state in states.items():
        if cid not in cases.index:
            logger.warning(f"case {cid} missing from cases dataset; skipping")
            continue
        case = cases.loc[cid]
        tier = str(case.get("tier", ""))
        gold_pf = str(case.get("prima_facie", "") or "")

        flags = run_probes(state, gold_prima_facie=gold_pf)
        probe_flags.append(flags)
        leak = sentiment_leakage(state)
        entail = check_entailment(state)
        entailment_results.append(entail)

        row: dict[str, Any] = {
            "case_id": cid, "tier": tier,
            "sentiment_leaked": leak["leaked"],
            "leak_hits": json.dumps(leak["hits"]),
            "entailment_consistent": entail["consistent"],
            "entailment_violations": json.dumps(entail["violations"]),
            **{f"probe_{k}": v for k, v in flags.items()},
        }

        if tier == "b":
            tier_b_pf_cases.append({
                "gold_prima_facie": gold_pf,
                "gold_departed_parameter": str(case.get("departed_parameter", "") or ""),
                "state": state,
            })
            for param, res in score_vs_tier_b(str(case.get("gold_values", "") or ""), state).items():
                tier_b_extraction_hits.setdefault(param, []).append(res["hit"])
                row[f"extract_hit_{param}"] = res["hit"]

        if tier == "a" and case.get("gold_path"):
            gold = json.load(open(case["gold_path"]))
            prf = score_vs_tier_a(gold, state)
            tier_a_extraction.append(prf)
            for param in EXTRACTION_PARAMS:
                row[f"extract_f1_{param}"] = prf[param]["f1"]
            if "s7" in (gold.get("steps_present") or []):
                cov = score_factor_coverage(gold.get("s7_factors") or [],
                                             (state.get("s7") or {}).get("factors") or [])
                coverage_scores.append(cov)
                row["factor_recall"] = cov["factor_recall"]
                row["kind_recall"] = cov["kind_recall"]

        per_case_rows.append(row)

    # ── Aggregates ────────────────────────────────────────────────────
    for probe, stats in probe_rates(probe_flags).items():
        em.emit_raw(f"probes.{probe}.n_applicable", stats["n_applicable"])
        if stats["rate"] is not None:
            em.emit_simple(f"probes.{probe}.rate", stats["rate"], n_total=stats["n_applicable"])

    assessable = [e for e in entailment_results if e["assessable"]]
    if assessable:
        em.emit(
            "consistency.entailment_rate",
            round(sum(e["consistent"] for e in assessable) / len(assessable), 6),
            n_total=len(entailment_results), n_real=len(assessable),
            n_defaulted=len(entailment_results) - len(assessable),
            default_reason="unassessable_s9_dropped" if len(assessable) < len(entailment_results) else None,
        )

    leak_flags = [r["sentiment_leaked"] for r in per_case_rows]
    if leak_flags:
        em.emit_simple("firewall.sentiment_leak_rate",
                        round(sum(leak_flags) / len(leak_flags), 6), n_total=len(leak_flags))

    if tier_b_pf_cases:
        em.emit_raw("prima_facie", score_prima_facie(tier_b_pf_cases))

    for param, hits in tier_b_extraction_hits.items():
        em.emit_simple(f"extraction.tier_b.{param}.hit_rate",
                        round(sum(hits) / len(hits), 6), n_total=len(hits))

    if tier_a_extraction:
        for param in EXTRACTION_PARAMS:
            f1s = [prf[param]["f1"] for prf in tier_a_extraction]
            em.emit_simple(f"extraction.tier_a.{param}.mean_f1",
                            round(sum(f1s) / len(f1s), 6), n_total=len(f1s))

    covs = [c["factor_recall"] for c in coverage_scores if c["factor_recall"] is not None]
    if covs:
        em.emit_simple("coverage.mean_factor_recall",
                        round(sum(covs) / len(covs), 6), n_total=len(covs))

    return em.to_dict(), pd.DataFrame(per_case_rows)
