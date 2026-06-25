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
- [2026-06-23_v8_results_groundtruth.md](2026-06-23_v8_results_groundtruth.md)
  — v8 results + kill. First *real* held-out movement (GoldCoin applicability
  0.921→0.972, saturated at the clean ckpt200, not entropy noise) — but an
  indiscriminate "engage/permit" bias: compliance Forbid recall collapses
  0.70→0.35. Killed at step 400/1080. Argues the **reward causes the entropy
  collapse** (more update strength diffuses instead of concentrates ⇒
  non-concentrating, one-directional reward), so no (β,μ,lr) fixes it → **v9 =
  reward redesign** (two-sided/selective). Keeper: ckpt200.
- [2026-06-23_v9_plan.md](2026-06-23_v9_plan.md)
  — v9 plan (**implemented + 548 tests green**). Two-sided reward via the
  *appropriateness direction* — gold-independent (deontic Raz force), the
  CI-faithful analog of GoldCoin Permit/Forbid. v8 traces: 97% of flows have a
  directional governing norm but the model **hedges 73%** (the reward made
  "ambiguous" the safe optimum). v9 promotes appropriateness from a 0.3 additive
  blend to a *multiplier-with-floor* on R_ground (correct ×1.0, hedge ×0.7, wrong
  ×0.4), and **simplifies** the composite to `gate · content · direction` +
  gold-aware abstention (drops the post-hoc penalty + gated `gate×disc`). μ→1,
  β=0.02 held. **ckpt-100 ground truth (2026-06-24): all predictions confirmed** —
  hedge fraction 73.5%→~6% (stable, no v8 breakout), GoldCoin Forbid recall
  0.35→0.55 vs v8 (best compliance of all arms), ConfAIDE 2a/2b + VLM-GeoPrivacy
  flat (no out-of-domain regression). **ckpt-200 (2026-06-24): plateau** — every
  GoldCoin axis flat-to-down vs ckpt-100 (Forbid recall 0.55→0.50, did NOT climb
  to SFT 0.65), ConfAIDE/GeoPrivacy flat, and the entropy↔IS coupling re-emerged
  in epoch 2 (corr −0.90). **Cancelled at step ~290**; ckpt-100 is the keeper.
  **PrivacyLens (judge-based, 2026-06-25): paper-ready** — v9-ckpt100 leaks least
  (0.463 vs SFT 0.473 vs base 0.473) AND is most helpful (0.497 vs 0.485 vs 0.473),
  adjusted leakage (leak-among-helpful) lowest (0.522 vs 0.531 vs 0.571) — a Pareto
  win on the privacy-utility frontier vs both base and SFT. Caveats: v9-vs-SFT raw
  leakage margin within noise (~1pp at n=493); QA-probe accuracy dips 0.94→0.92
  (T-axis). The full benchmark roster is now complete on ckpt-100.
- [2026-06-24_v10_plan.md](2026-06-24_v10_plan.md)
  — v10 plan (**implemented + 558 tests green, launched**). A verdict-balance
  diagnostic on v9 traces found the plateau's cause: the governing norms are
  **3.98:1 appropriate:inappropriate**, so under v9's *symmetric* floor the
  EV-optimal verdict when unsure is permissive — the v9 policy commits the correct
  "inappropriate" verdict on prohibited-governed flows **only 30%** of the time
  (53% false-permits), vs 84% on appropriate-governed; **H2 retrieval-noise ruled
  out** (top-sim 0.68 either way). v10 makes the multiplier **cost-sensitive**: a
  false-permit floors at **0.1** (false-forbid keeps 0.4), widening the Forbid
  within-group gradient 0.6→0.9. Single reward variable; everything else held from
  v9. Regime: `save_steps 50` + kill-at-peak. Falsifiable: Forbid commit-accuracy
  rises off 30%; GoldCoin Forbid recall off 0.55 toward SFT 0.65.
