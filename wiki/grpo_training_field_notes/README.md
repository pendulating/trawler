# GRPO training field notes

Dated, scratch-level working notes from analyzing GRPO training runs (reward
traces, hyperparameter sweeps, gold-label behavior). Less polished than the
method deep-dives in [grpo-reward.md](../grpo-reward.md); kept for the
camera-ready generative stage.

- [2026-06-19_redesign_v2-v5_gold_label.md](2026-06-19_redesign_v2-v5_gold_label.md)
  — v2→v5 "grpo_redesign_full" runs: config evolution diff + gold-label
  completion metrics (abstention ~69% flat, judgment accuracy below majority
  baseline). Mechanics fixed; behavior invariant to tuning.
- [2026-06-22_v6_v7_optimizer_stability.md](2026-06-22_v6_v7_optimizer_stability.md)
  — v6 optimizer-instability diagnosis (beta=0 entropy 10× runaway → logprob
  mismatch 80× → IS collapse → gradient masked) + v7 beta=0.02 pilot that *fixed*
  the instability yet reward stayed flat. Confirms v2–v5: bottleneck is
  reward-gradient coupling + SFT prior, not RL mechanics.
- [2026-06-22_v8_plan.md](2026-06-22_v8_plan.md)
  — v8 plan (implemented, pre-launch): two complementary levers for the
  "advantage present but unfollowed" failure. (1) `num_iterations=2` + `beta=0.02`
  — μ>1 doubles updates/rollout *judge-cost-neutral* and finally activates the
  `epsilon_high` Clip-Higher that was inert at μ=1 (ratio ≡ 1). (2) symmetric
  contrastive clamp — un-zeroes ~1/3 of well-grounded extractions the
  rank-blend-vs-full-grounding asymmetry was clamping. 530 tests green.
