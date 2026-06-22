"""Synthetic 5-row pipelines per benchmark — validates compute_metrics
end-to-end with the new provenance + FAIL-tier wiring.

These tests are deliberately small (no GPU, no vLLM, no judge sidecar):
they exercise the metric layer that *consumes* parse / judge output.
The point is to catch the next yesterday-bug-class issue before it
ships, not to validate model behavior.

Each benchmark gets two tests:
1. Healthy input → metrics shape is correct, ``metric_provenance`` is
   present, no FAIL-severity warnings would be emitted.
2. Corrupt input (high default-rate scenario from the audit) →
   conditional metrics differ from overall metrics, provenance flags
   the defaulted_rate, and the format-health gate would FAIL the run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf

from dagspaces.common.eval_sanity import (
    SanityFailure,
    compute_format_health,
    compute_parse_health,
)
from dagspaces.common.runners.sanity import log_sanity_to_context


# ---------------------------------------------------------------------------
# privacylens — covered exhaustively in test_privacylens_judge_skip_provenance.
# Here we add a quick smoke that healthy data produces clean metrics.
# ---------------------------------------------------------------------------

class TestPrivacyLensHealthy:
    def test_healthy_run_no_failures(self):
        from dagspaces.privacylens.stages.compute_metrics import compute_metrics

        n = 50
        leakage_df = pd.DataFrame({
            "record_id": [f"r{i}" for i in range(n)],
            "generated_action": ["Thought: x.\nAction: send"] * n,
            "agent_action_format_status": ["valid"] * n,
            "leakage_judged": [True] * n,
            "leakage_skip_reason": [""] * n,
            "leak_flag": [False] * n,
            "leak_probability": [0.0] * n,
            "leak_judge_text": ["Answer: No."] * n,
            "parse_status": ["parsed"] * n,
        })
        helpfulness_df = pd.DataFrame({
            "record_id": leakage_df["record_id"],
            "generated_action": leakage_df["generated_action"],
            "agent_action_format_status": ["valid"] * n,
            "helpfulness_judged": [True] * n,
            "helpfulness_score": [2] * n,
            "helpfulness_binary": [True] * n,
            "helpfulness_judge_text": ["Answer: Good (2)."] * n,
            "parse_status": ["parsed"] * n,
        })
        qa_df = pd.DataFrame({
            "_qa_axis": ["S"] * n + ["T"] * n + ["V"] * n,
            "predicted_label": ["no"] * (3 * n),
            "correct": [True] * (3 * n),
            "generated_text": ["No."] * (3 * n),
            "parse_status": ["parsed"] * (3 * n),
        })

        m = compute_metrics(qa_df, leakage_df, helpfulness_df)
        assert m["leakage"]["agent_action_format_rate"] == 1.0
        assert m["leakage"]["leakage_rate_among_parseable"] == 0.0
        assert "metric_provenance" in m

        report = compute_format_health(
            leakage_df, dagspace="privacylens", stage="agent_action_format",
            format_col="agent_action_format_status",
        )
        assert not report.has_failures()


# ---------------------------------------------------------------------------
# confaide — corruption regression for tier3_free + tier3_list, healthy
# regression for tier2 + tier3_control.
# ---------------------------------------------------------------------------

class TestConfaide:
    def test_tier3_free_corruption_split(self):
        """50% empty rows: rule defaults all to 'no_leak'. Conditional
        metric reflects only the 50% with real text; overall reflects
        the historical (misleading) value with provenance flag."""
        from dagspaces.confaide.stages.compute_metrics import compute_metrics

        df = pd.DataFrame({
            "prediction": ["no_leak"] * 50 + ["leak"] * 30 + ["no_leak"] * 20,
            "parse_status": ["empty"] * 50 + ["parsed"] * 50,
            "generated_text": [""] * 50 + ["text"] * 50,
            "subject_agent": ["X"] * 100,
        })
        m = compute_metrics(df, "3_free")

        # Among parseable: 30 leak / 50 parsed = 0.6
        assert m["leak_rate_among_parseable"] == pytest.approx(0.6)
        # Overall: 30 leak / 100 total = 0.3 (the misleading historical number)
        assert m["leak_rate_overall_with_default_zero"] == pytest.approx(0.3)
        # Provenance reflects the 50% default-padding
        ovr_prov = m["metric_provenance"]["leak_rate_overall_with_default_zero"]
        assert ovr_prov["n_defaulted"] == 50
        assert ovr_prov["defaulted_rate"] == pytest.approx(0.5)
        assert ovr_prov["default_reason"] == "empty_or_unparseable_default_no_leak"

    def test_tier3_list_corruption_split(self):
        from dagspaces.confaide.stages.compute_metrics import compute_metrics

        df = pd.DataFrame({
            "prediction": ["error"] * 50 + ["no_error"] * 30 + ["error"] * 20,
            "parse_status": ["empty"] * 50 + ["parsed"] * 50,
            "generated_text": [""] * 50 + ["text"] * 50,
        })
        m = compute_metrics(df, "3_info")
        # Among parseable: 20 error / 50 parsed = 0.4
        assert m["error_rate_among_parseable"] == pytest.approx(0.4)
        # Overall: 70 error / 100 total = 0.7 (inflated by empty rows)
        assert m["error_rate_overall_with_default_zero"] == pytest.approx(0.7)

    def test_tier2_provenance(self):
        from dagspaces.confaide.stages.compute_metrics import compute_metrics

        df = pd.DataFrame({
            "prediction": [50, 100, 0, -50, -100, None],
            "ground_truth": [50, 100, 0, -50, -100, 0],
            "parse_status": ["parsed"] * 5 + ["unparseable"],
            "generated_text": ["x"] * 6,
            "tier": ["2a"] * 6,
        })
        m = compute_metrics(df, "2a")
        assert "metric_provenance" in m
        assert m["unparseable_count"] == 1
        # Pearson r should be exactly 1.0 (perfect agreement on parsed rows)
        assert m["pearson_r"] == pytest.approx(1.0)

    def test_tier3_control_clean(self):
        from dagspaces.confaide.stages.compute_metrics import compute_metrics

        df = pd.DataFrame({
            "prediction": ["No"] * 80 + ["Yes"] * 15 + ["unparseable"] * 5,
            "parse_status": ["parsed"] * 95 + ["unparseable"] * 5,
            "generated_text": ["x"] * 100,
        })
        m = compute_metrics(df, "3_control")
        assert m["accuracy"] == pytest.approx(80 / 95, abs=1e-5)


# ---------------------------------------------------------------------------
# cirl_vignettes — trajectory metrics with the helpfulness_score=0 corruption.
# ---------------------------------------------------------------------------

class TestCirlTrajectory:
    def test_among_judged_vs_overall(self):
        from dagspaces.cirl_vignettes.stages.compute_trajectory_metrics import (
            compute_trajectory_metrics,
        )

        n_total = 100
        n_judged = 50
        df = pd.DataFrame({
            "agent_action_format_status": ["valid"] * n_judged + ["no_action_format"] * (n_total - n_judged),
            "leakage_judged": [True] * n_judged + [False] * (n_total - n_judged),
            "helpfulness_judged": [True] * n_judged + [False] * (n_total - n_judged),
            "has_leakage": [True] * 10 + [False] * 40 + [False] * 50,
            "helpfulness_score": [2] * 30 + [0] * 20 + [0] * 50,
        })
        m = compute_trajectory_metrics(df)

        # Real signal (among judged):
        assert m["leakage_rate_among_judged"] == pytest.approx(0.2)  # 10/50
        assert m["utility_among_judged"] == pytest.approx(0.6)  # 30/50

        # Misleading historical (overall, default-zero):
        assert m["leakage_rate_overall_with_default_zero"] == pytest.approx(0.1)
        assert m["utility_overall_with_default_zero"] == pytest.approx(0.3)

        # Provenance flags the 50% default-padding
        ut_ovr = m["metric_provenance"]["utility_overall_with_default_zero"]
        assert ut_ovr["n_defaulted"] == 50
        assert ut_ovr["default_reason"] == "judge_skipped_default_score_zero"

    def test_format_health_warns_only_at_50pct_by_default(self):
        # format_adherence_rate has no default fail threshold (mirrors
        # upstream SALT-NLP/PrivacyLens — no adherence gate). At 50%
        # adherence the report warns loudly but does NOT raise, so the
        # pipeline reaches compute_metrics and emits the conditioned
        # leakage/helpfulness rates rather than disqualifying the run.
        from dagspaces.cirl_vignettes.stages.compute_trajectory_metrics import (
            compute_trajectory_metrics,
        )

        n_total, n_judged = 100, 50
        df = pd.DataFrame({
            "agent_action_format_status": ["valid"] * n_judged + ["no_action_format"] * (n_total - n_judged),
            "leakage_judged": [True] * n_judged + [False] * (n_total - n_judged),
            "helpfulness_judged": [True] * n_judged + [False] * (n_total - n_judged),
            "has_leakage": [False] * n_total,
            "helpfulness_score": [2] * n_judged + [0] * (n_total - n_judged),
            "final_action_generated": (["Action: x"] * n_judged + ["nope"] * (n_total - n_judged)),
        })
        m = compute_trajectory_metrics(df)
        assert m["agent_action_format_rate"] == pytest.approx(0.5)

        report = compute_format_health(
            df, dagspace="cirl_vignettes", stage="agent_action_format",
            format_col="agent_action_format_status",
        )
        assert report.has_warnings(), "low adherence must surface a warning"
        assert not report.has_failures(), (
            "format_adherence_rate is warn-only by default — halting weak "
            "instruction followers would silently disqualify them from "
            "the table"
        )
        ctx = SimpleNamespace(cfg=OmegaConf.create({}), logger=None)
        # No SanityFailure: log_sanity_to_context just emits the warning.
        log_sanity_to_context(ctx, report, metadata={})

    def test_format_health_halts_at_50pct_with_explicit_fail_override(self):
        # Operators who DO want a hard adherence gate can opt in per-run
        # via a severity-keyed threshold override on compute_format_health.
        from dagspaces.cirl_vignettes.stages.compute_trajectory_metrics import (
            compute_trajectory_metrics,  # noqa: F401 — imported for parity
        )

        n_total, n_judged = 100, 50
        df = pd.DataFrame({
            "agent_action_format_status": ["valid"] * n_judged + ["no_action_format"] * (n_total - n_judged),
            "leakage_judged": [True] * n_judged + [False] * (n_total - n_judged),
            "helpfulness_judged": [True] * n_judged + [False] * (n_total - n_judged),
            "has_leakage": [False] * n_total,
            "helpfulness_score": [2] * n_judged + [0] * (n_total - n_judged),
            "final_action_generated": (["Action: x"] * n_judged + ["nope"] * (n_total - n_judged)),
        })
        report = compute_format_health(
            df, dagspace="cirl_vignettes", stage="agent_action_format",
            format_col="agent_action_format_status",
            thresholds={"format_adherence_rate:lt:fail": 0.9},
        )
        assert report.has_failures()
        ctx = SimpleNamespace(cfg=OmegaConf.create({}), logger=None)
        with pytest.raises(SanityFailure):
            log_sanity_to_context(ctx, report, metadata={})


class TestCirlProbing:
    def test_provenance_and_conditional(self):
        from dagspaces.cirl_vignettes.stages.compute_metrics import compute_metrics

        df = pd.DataFrame({
            "prediction": ["B"] * 80 + ["A"] * 15 + ["unparseable"] * 5,
            "probing_level": ["seed"] * 50 + ["vignette"] * 50,
        })
        m = compute_metrics(df)
        # Headline (paper-quoted, CI-RL parity): 80 B / 100 total
        assert m["accuracy"] == pytest.approx(0.8)
        # Conditional: 80 B / 95 parseable
        assert m["accuracy_among_parseable"] == pytest.approx(80 / 95, abs=1e-5)
        # Provenance flags 5 unparseable
        prov = m["metric_provenance"]["accuracy"]
        assert prov["n_defaulted"] == 5
        assert prov["default_reason"] == "unparseable_counted_as_wrong"


# ---------------------------------------------------------------------------
# goldcoin_hipaa — clean per audit; verify provenance shape.
# ---------------------------------------------------------------------------

class TestGoldcoin:
    def test_provenance_attached_to_accuracy(self):
        from dagspaces.goldcoin_hipaa.stages.compute_metrics import compute_metrics

        df = pd.DataFrame({
            "ground_truth": ["Permit"] * 50 + ["Forbid"] * 50,
            "prediction": (
                ["Permit"] * 45 + ["unparseable"] * 5 + ["Forbid"] * 50
            ),
        })
        m = compute_metrics(df, "compliance")
        assert "metric_provenance" in m
        prov = m["metric_provenance"]["accuracy"]
        assert prov["n_total"] == 100
        assert prov["n_real"] == 95  # parseable
        assert prov["n_defaulted"] == 5
        assert prov["default_reason"] == "unparseable_dropped"
        # Accuracy = 95/95 (all parseable rows correct in this synth)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_zero_unparseable_clean_provenance(self):
        from dagspaces.goldcoin_hipaa.stages.compute_metrics import compute_metrics

        df = pd.DataFrame({
            "ground_truth": ["Permit"] * 50 + ["Forbid"] * 50,
            "prediction": ["Permit"] * 50 + ["Forbid"] * 50,
        })
        m = compute_metrics(df, "compliance")
        prov = m["metric_provenance"]["accuracy"]
        assert prov["n_defaulted"] == 0
        assert prov["default_reason"] is None
        assert m["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# vlm_geoprivacy_bench — clean per audit; verify per-question provenance
# splits properly when one question has unparseable predictions.
# ---------------------------------------------------------------------------

class TestVlmGeoprivacy:
    def test_q7_unparseable_recorded_in_provenance(self):
        from dagspaces.vlm_geoprivacy_bench.stages.compute_metrics import compute_metrics

        n = 50
        df = pd.DataFrame({
            "Q1_true": ["A"] * n,
            "Q1_pred": ["A"] * n,
            "Q2_true": ["A"] * n,
            "Q2_pred": ["A"] * n,
            "Q3_true": ["B"] * n,
            "Q3_pred": ["B"] * n,
            "Q4_true": ["A"] * n,
            "Q4_pred": ["A"] * n,
            "Q5_true": ["B"] * n,
            "Q5_pred": ["B"] * n,
            "Q6_true": ["A"] * n,
            "Q6_pred": ["A"] * n,
            "Q7_true": ["B"] * n,
            "Q7_pred": ["B"] * 45 + [None] * 5,
        })
        m = compute_metrics(df, free_form=False)
        # Q1 has all 50 rows valid
        q1_acc = m["metric_provenance"]["per_question.Q1.accuracy"]
        assert q1_acc["n_defaulted"] == 0
        # Q7 has 5 unparseable
        q7_acc = m["metric_provenance"]["per_question.Q7.accuracy"]
        assert q7_acc["n_defaulted"] == 5
        assert q7_acc["default_reason"] == "unparseable_dropped"
        assert q7_acc["n_real"] == 45
