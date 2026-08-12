# Changelog

Dated records of changes that altered results, invalidated earlier runs, or fixed a defect whose
effects are visible in the artifacts. Newest first. `wiki/README.md` links only a handful of
these; this index covers all of them.

## 2026-07 — benchmark parity reviews

Each of the five benchmarks was audited against its upstream reference implementation. Several
flipped a metric definition, which makes pre-review numbers non-comparable.

- [2026-07-21_privacylens_parity_review.md](2026-07-21_privacylens_parity_review.md) — the
  helpfulness parser corrupted 21.5% of judgments from 2026-04-26 onward; rescue is a
  re-finalize from `output.jsonl`. Tool-pin and Thought-strip restored per upstream, so
  post-review runs are not comparable with earlier ones.
- [2026-07-21_goldcoin_parity_review.md](2026-07-21_goldcoin_parity_review.md) — headline flipped
  to the upstream forced-wrong denominator; some cells move by up to 12 points.
- [2026-07-21_confaide_parity_review.md](2026-07-21_confaide_parity_review.md) — three of six
  primary metrics were silently `None`; control parser widened, tier-3 question rebuilt.
- [2026-07-21_vlm_geoprivacy_parity_review.md](2026-07-21_vlm_geoprivacy_parity_review.md) —
  clean; accuracy denominator flipped for parity (value-neutral).
- [2026-07-21_cirl_benchmark_swap.md](2026-07-21_cirl_benchmark_swap.md) — the `cirl` dagspace
  now holds the real CIRL-729 action set; PrivacyLens-under-CIRL-protocol moved into
  `privacylens` as `pipeline=privacylens_cirl_protocol`.

## 2026-07 — infrastructure

- [2026-07-20_grpo_methodology_congruency_review.md](2026-07-20_grpo_methodology_congruency_review.md)
  — methodology-vs-code audit; §3.1 records the judge-model default mismatch (fixed in code
  2026-08-12).
- [2026-07-13_canonical_models_and_harmony.md](2026-07-13_canonical_models_and_harmony.md) — the
  canonical 13-model set; the gpt-oss harmony bug where the model scored its own chain of thought.

## 2026-06 — GRPO redesign and logging

These describe the v9-lineage reward, **removed 2026-08-12**. Retained as the historical record;
see [../grpo_redesign/](../grpo_redesign/README.md) for the current m-series design.

- [2026-06-10_judge_response_format_fix.md](2026-06-10_judge_response_format_fix.md)
- [2026-06-09_grpo_phase1_optimizer_revision.md](2026-06-09_grpo_phase1_optimizer_revision.md)
- [2026-06-09_grpo_phase2-5_reward_redesign.md](2026-06-09_grpo_phase2-5_reward_redesign.md)
- [2026-06-09_code_review_norms_grpo.md](2026-06-09_code_review_norms_grpo.md)
- [2026-06-09_wandb_logging_rationalization.md](2026-06-09_wandb_logging_rationalization.md)
- [2026-06-09_ner_quality_checks.md](2026-06-09_ner_quality_checks.md)

## 2026-05

- [2026-05-12_privacylens_action_prompt_react.md](2026-05-12_privacylens_action_prompt_react.md)
