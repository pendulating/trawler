"""Step-6 prima facie scoring against Tier B constructed ground truth.

Sensitivity/specificity per departed parameter, attribution accuracy,
incompleteness recognition, and the presumption-direction asymmetry
(misses on true departures vs. false alarms on conforming flows)."""

from __future__ import annotations

from typing import Any, Dict, List

from dagspaces.common.metric_provenance import MetricEmitter

PARAMETERS = ["sender", "recipient", "subject", "information_type", "transmission_principle"]


def _s6(state: Dict[str, Any]) -> Dict[str, Any]:
    return state.get("s6") or {}


def score_prima_facie(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score step-6 artifacts against Tier B labels.

    Args:
        cases: dicts with keys `gold_prima_facie`, `gold_departed_parameter`,
            and `state` (the traversal state dict incl. s6). Unparseable s6
            artifacts count as wrong (they ARE failures of the traversal) but
            are also reported separately via provenance.
    """
    em = MetricEmitter()
    n_total = len(cases)
    em.emit_raw("n_cases", n_total)
    if n_total == 0:
        return em.to_dict()

    parseable = [c for c in cases if _s6(c["state"]).get("violation") in ("yes", "no", "incomplete_norms")]
    em.emit_simple("s6_parseable_rate", round(len(parseable) / n_total, 6), n_total=n_total)

    # Violation detection: gold departures should be flagged yes
    departures = [c for c in parseable if c["gold_prima_facie"] == "yes"]
    controls = [c for c in parseable if c["gold_prima_facie"] == "no"]
    incompletes = [c for c in parseable if c["gold_prima_facie"] == "incomplete_norms"]

    if departures:
        hits = sum(_s6(c["state"])["violation"] == "yes" for c in departures)
        em.emit(
            "violation_sensitivity", round(hits / len(departures), 6),
            n_total=len([c for c in cases if c["gold_prima_facie"] == "yes"]),
            n_real=len(departures),
            n_defaulted=len([c for c in cases if c["gold_prima_facie"] == "yes"]) - len(departures),
            default_reason="unparseable_dropped" if len(departures) < len([c for c in cases if c["gold_prima_facie"] == "yes"]) else None,
        )
        # Per-parameter sensitivity
        for param in PARAMETERS:
            sub = [c for c in departures if c["gold_departed_parameter"] == param]
            if sub:
                em.emit_simple(
                    f"sensitivity_by_param.{param}",
                    round(sum(_s6(c["state"])["violation"] == "yes" for c in sub) / len(sub), 6),
                    n_total=len(sub),
                )
        # Attribution: did it name the right departed parameter?
        flagged = [c for c in departures if _s6(c["state"])["violation"] == "yes"]
        if flagged:
            correct_attr = sum(
                c["gold_departed_parameter"] in (_s6(c["state"]).get("departed_parameters") or [])
                for c in flagged
            )
            em.emit_simple("attribution_accuracy", round(correct_attr / len(flagged), 6), n_total=len(flagged))
            exact_attr = sum(
                (_s6(c["state"]).get("departed_parameters") or []) == [c["gold_departed_parameter"]]
                for c in flagged
            )
            em.emit_simple("attribution_exact_single", round(exact_attr / len(flagged), 6), n_total=len(flagged))

    if controls:
        # Specificity: conforming flows should NOT be flagged
        true_neg = sum(_s6(c["state"])["violation"] == "no" for c in controls)
        em.emit_simple("violation_specificity", round(true_neg / len(controls), 6), n_total=len(controls))

    if incompletes:
        recognized = sum(_s6(c["state"])["violation"] == "incomplete_norms" for c in incompletes)
        em.emit_simple("incompleteness_recognition_rate", round(recognized / len(incompletes), 6),
                        n_total=len(incompletes))

    # Presumption-direction asymmetry: CI's presumption favors the entrenched
    # practice, so errors should skew toward false alarms on controls rather
    # than misses on departures ONLY if the model over-triggers; report both.
    if departures and controls:
        miss_rate = sum(_s6(c["state"])["violation"] != "yes" for c in departures) / len(departures)
        false_alarm_rate = sum(_s6(c["state"])["violation"] != "no" for c in controls) / len(controls)
        em.emit_simple("miss_rate_on_departures", round(miss_rate, 6), n_total=len(departures))
        em.emit_simple("false_alarm_rate_on_controls", round(false_alarm_rate, 6), n_total=len(controls))
        em.emit_raw("presumption_asymmetry",
                     round(false_alarm_rate - miss_rate, 6))

    return em.to_dict()
