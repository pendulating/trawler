"""Tests for the vignette verdict-mix logging (2026-07-01 forensics follow-up).

The v10→v11 iterations steer on the judgment-vignette force mix and per-class
verdict behaviour, which previously had to be mined from reward_traces.jsonl
post hoc. Covers:

* ``_vignette_gold_counts`` — realised yes/no mix over dataset/metadata rows
  (feeds the prescreen report and training_metadata.json)
* ``CompositeRewardFunction._push_vignette_health`` — per-call ``vignette/*``
  accuracy + drift stats (``last_vignette_health``; W&B push is fail-safe)
* ``scripts.analyze_grpo_verdict_traces.analyze`` — the offline forensics
  aggregation over raw trace entries

See wiki/grpo_training_field_notes/2026-07-01_v11_probe_midrun_forensics.md.
"""

from __future__ import annotations

from dagspaces.grpo_training.stages.prompt_screening import _vignette_gold_counts
from dagspaces.grpo_training.stages.rewards import CompositeRewardFunction

WEIGHTS = [0.10, 0.05, 0.05, 0.20, 0.10, 0.50]


def _vig_row(gold, task="norm_judgment"):
    return {"task_type": task, "gold_judgment": gold}


class TestVignetteGoldCounts:
    def test_counts_yes_no(self):
        rows = [_vig_row("yes"), _vig_row("no"), _vig_row("yes")]
        assert _vignette_gold_counts(rows) == {"yes": 2, "no": 1}

    def test_ignores_ci_extraction_rows(self):
        rows = [_vig_row("yes"),
                {"task_type": "ci_extraction", "gold_has_exchange": True}]
        assert _vignette_gold_counts(rows) == {"yes": 1, "no": 0}

    def test_ignores_missing_or_malformed_gold(self):
        rows = [_vig_row(None), _vig_row("maybe"), _vig_row("no")]
        assert _vignette_gold_counts(rows) == {"yes": 0, "no": 1}


_YES = '{"judgment": "yes", "reasoning": "sharing is expected here", "norms_considered": []}'
_NO = '{"judgment": "no", "reasoning": "the norm prohibits disclosure", "norms_considered": []}'


class TestVignetteHealth:
    def _fn(self):
        fn = CompositeRewardFunction(weights=WEIGHTS)
        fn.prompt_metadata = {
            "py": {"task_type": "norm_judgment", "gold_judgment": "yes",
                   "source_norm_articulation": "one must share the news"},
            "pn": {"task_type": "norm_judgment", "gold_judgment": "no",
                   "source_norm_articulation": "one must not reveal the secret"},
        }
        return fn

    def test_per_class_accuracy_and_drift(self):
        fn = self._fn()
        # gold-yes answered yes (correct); gold-no answered yes (the
        # over-permit drift the says_yes_gold_no metric watches).
        fn(prompts=["py", "pn"], completions=[_YES, _YES])
        h = fn.last_vignette_health
        assert h["vignette/n_yes"] == 1.0
        assert h["vignette/n_no"] == 1.0
        assert h["vignette/acc_gold_yes"] == 1.0
        assert h["vignette/acc_gold_no"] == 0.0
        assert h["vignette/says_yes_gold_no"] == 1.0
        assert h["vignette/says_no_gold_yes"] == 0.0
        assert h["vignette/unparsed_frac"] == 0.0

    def test_over_forbid_direction(self):
        fn = self._fn()
        fn(prompts=["py"], completions=[_NO])
        h = fn.last_vignette_health
        assert h["vignette/acc_gold_yes"] == 0.0
        assert h["vignette/says_no_gold_yes"] == 1.0
        # No gold-no completions this call → per-class keys absent, not 0.
        assert "vignette/acc_gold_no" not in h

    def test_unparsed_counted_but_not_drift(self):
        fn = self._fn()
        fn(prompts=["pn"], completions=["not json at all"])
        h = fn.last_vignette_health
        assert h["vignette/n_no"] == 1.0
        assert h["vignette/acc_gold_no"] == 0.0
        assert h["vignette/unparsed_frac"] == 1.0
        # A parse failure is not an affirmative "yes" — drift stays 0.
        assert h["vignette/says_yes_gold_no"] == 0.0

    def test_no_vignettes_no_push(self):
        fn = CompositeRewardFunction(weights=WEIGHTS)
        fn.prompt_metadata = {}
        fn(prompts=["x"], completions=[
            '{"reasoning": "r", "has_information_exchange": false, "flows": []}'])
        assert fn.last_vignette_health == {}


class TestVerdictTraceAnalysis:
    def _entries(self):
        # Two calls (max_call=1 → with bins=2, call 0 = bin0, call 1 = bin1).
        return [
            {"call": 0, "task_type": "norm_judgment", "gold_judgment": "no",
             "completion": _NO},
            {"call": 0, "task_type": "ci_extraction", "rground_flows": [
                {"norm_force": "prohibited", "app_direction": 0.7}]},
            {"call": 1, "task_type": "norm_judgment", "gold_judgment": "no",
             "completion": _YES},
            {"call": 1, "task_type": "norm_judgment", "gold_judgment": "yes",
             "completion": _YES},
            {"call": 1, "task_type": "ci_extraction", "rground_flows": [
                {"norm_force": "prohibited", "app_direction": 1.0},
                {"norm_force": "obligatory", "app_direction": 1.0}]},
        ]

    def test_analyze_aggregates(self):
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "analyze_grpo_verdict_traces",
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "scripts", "analyze_grpo_verdict_traces.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        r = mod.analyze(self._entries(), n_bins=2)
        assert r["gold_mix"] == {"no": 2, "yes": 1}
        # bin0: gold-no answered "no" (correct); bin1: gold-no answered "yes".
        assert r["verdicts"][0]["no"]["no"] == 1
        assert r["verdicts"][1]["no"]["yes"] == 1
        assert r["verdicts"][1]["yes"]["yes"] == 1
        # Direction tiers on prohibited-governed flows only (obligatory excluded).
        assert r["tiers"][0]["hedge"] == 1
        assert r["tiers"][1]["correct"] == 1
        # Exploration guard: bin0 group has no committer, bin1 group does.
        assert r["explore"][0] == [False]
        assert r["explore"][1] == [True]
