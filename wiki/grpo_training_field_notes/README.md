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
  β=0.02 held. **ckpt-100 ground truth (2026-06-24): held-out predictions confirmed** —
  GoldCoin Forbid recall 0.35→0.55 vs v8, Forbid precision 0.50→0.65, best compliance
  macro-F1 of all arms, ConfAIDE 2a/2b + VLM-GeoPrivacy flat (no out-of-domain
  regression). (A training-trace "hedge 73.5%→6% collapse" claim was later
  **retracted** as a measurement artifact — v8/v9 hedge identically; the win is
  held-out. See the v9 plan's correction banner.) **ckpt-200 (2026-06-24): plateau** — every
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
  **Result (2026-06-27 held-out sweep ckpt 100→500): prediction falsified on the
  binding axis.** Forbid recall peaked at 0.55 (= v9) and never reached SFT 0.65;
  best arm ckpt-250 (acc 0.860, macro-F1 0.755) is a marginal, in-noise win over v9
  with *identical* Forbid cells; ckpt-350 ≡ ckpt-500 (byte-identical verdicts on all
  107 cases — verdict policy froze by epoch 2) and both regress to 0.45. The 0.1
  floor fired (penalised false-permits) but did not convert the 72% Forbid-flow
  hedging mass into correct commits → the marginal ~5:1 permissive prior is the
  unbroken constraint → **v11**.
- [2026-06-27_v11_plan.md](2026-06-27_v11_plan.md)
  — v11 plan (**lever (a) implemented via the top100 pivot; probe RUNNING since
  2026-06-30, job 488187**). v10 showed floor-tuning fixes the
  *reward per flow* but cannot cancel the **marginal ~5:1 permissive base rate**
  (norm_judgment prompts 118 yes : 21 no = 5.6:1; extraction governing flows
  4.8:1). v11 adds a **force-balanced prescreen**: after the existing reward-std
  variance filter, a force-balancing pass that oversamples/up-weights (a)
  inappropriate ("no") judgment prompts and (b) prohibited-rich extraction prompts
  toward force parity. Distribution change, not a reward change — composes with
  v10's 0.1 floor, low bug surface. The 2026-06-30 pivot implements (a) as a
  *natural* rebalance instead: judgment vignettes drawn from the top100 universe
  (1.72:1) via `VIGNETTE_NORM_UNIVERSES_PATH`, everything else v10-identical.
  Falsifiable: Forbid recall finally off 0.55 toward SFT 0.65 without tanking
  Permit recall / applicability (over-correction = indiscriminate-forbid, the mirror
  of v8).
- [2026-07-01_v11_probe_midrun_forensics.md](2026-07-01_v11_probe_midrun_forensics.md)
  — mid-run trace forensics on the probe (pre-registered before the held-out
  sweep; **run COMPLETED 2026-07-02**, full-trace forensics confirmed, GoldCoin
  7-arm + ConfAIDE 5-arm ckpt sweeps launched — see the note's update banner).
- [2026-07-08_top100_flows_plan.md](2026-07-08_top100_flows_plan.md)
  — top100-flows run plan (**scripts staged**: `run_extract_top100_flows.sh` →
  `run_grpo_top100_flows.sh`; launch gated on the v12a verdict). The data
  lever: fresh top100 extraction prompts (~5.3× chunks, prohibited-richer),
  all universes from the top100 build (contrastive pool 9→96), 1 epoch over a
  seeded ~1400-flow-chunk cap (≈625 steps ≈ 44h), reward pinned at v11
  (HEDGE_PROHIBIT=0.5 composes v12a per the decision gate). Corpus forensics:
  the "97 books" = the has_norms gate zeroing ALL chunks of books 35/215/6133
  (cache is 100/100; those 3 are excluded from training data — no universe).
  Blocking artifact is only stage-1 flows reasoning (~3h, new
  `COLM_flows_reasoning_prefetched_qwen36` pipeline); the ~19h tuple
  extraction resumes later via `ci_extraction_from_reasoning_fiction`.
- [2026-07-08_critical_assessment.md](2026-07-08_critical_assessment.md)
  — harsh external-reviewer assessment of the v1→v11 evidence (written with the
  same-day paper sync). Survives: v8→v9 causal flip, replicated +3 GoldCoin
  macro-F1 (×3 runs), MMLU flat, judge-validation honesty. Does not: PrivacyLens
  "Pareto win" (≈ batch noise, ordering flips between batches), grounding
  *attribution* (programmatic-only ablation never run; λ=0 ≈ λ=1 undercuts the
  mechanism), base→full-pipeline ≈ wash, KL 0.04 ≈ barely-trained policy, SFT
  format toggle (+4.7) rivals the whole RL stage. Ranked next steps:
  programmatic-only ablation > seed error-bars > v12a (expect null) > top100
  flows run > PrivacyLens reframe.
- [2026-07-03_v12a_plan.md](2026-07-03_v12a_plan.md)
  — v12a plan (**implemented + 585 tests green, launch staged**). The v11
  held-out sweeps confirmed the pre-registration: Forbid recall ceiling 0.55
  for the third iteration (best arm ckpt-350 ≡ v10's 0.860/0.755), while
  lever (a)'s narrow promise delivered (ConfAIDE-2b gap −5.4→−2.7pts, 2a
  flips positive, CIRL holds SFT level where v10 eroded to 0.880, PrivacyLens
  ckpt-200 ties the v9 keeper / ckpt-350 Pareto-dominated by drift). v12a =
  cost-sensitive **hedge** tier: `rground_app_hedge_prohibit: 0.5` drops a
  hedged verdict on a prohibited-governed flow below the neutral 0.7,
  widening the commit-vs-hedge gap 0.3→0.5 where the ~72% frozen hedge mass
  binds. Single variable vs the v11 probe; formula version bumped. The rebalance **landed** (realised vignette mix
  2.08:1 vs v10's 5.2:1 — the force-blind variance screen had roughly doubled
  v10's skew) and **halts the verdict erosion** v10's mix caused (gold-"no"
  yes-rate 0.12→0.20 in v10; 0.01→0.05 in the probe). BUT the vignette task was
  **never permissive-biased** (gold-"no" acc 0.94 pre-GRPO; the low class is
  gold-"yes" 0.64, so the vignette gradient teaches "say yes more"), and the
  extraction-side prohibited-flow **hedge mass is frozen at ~72%** (= v10) with
  a committer present in ~half the traced groups — hedge EV, not exploration,
  is the binding constraint. Prediction: Forbid recall stays ≈0.55 (n=20 —
  judge on compliance macro-F1 + ConfAIDE-2b instead); v12 pivot ranked
  (v12a cost-sensitive hedge tier > v12b extraction balancing > top100 flows
  run). Ships `scripts/analyze_grpo_verdict_traces.py`, per-call `vignette/*`
  W&B verdict stats, and realised force-mix fields in
  training_metadata/prescreen_report.
