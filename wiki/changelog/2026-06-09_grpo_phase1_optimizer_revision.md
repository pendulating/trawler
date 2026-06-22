# 2026-06-09 — GRPO Phase-1 optimizer revision (G=8, KL anchor, no std-scaling)

**Status:** in working tree. Files:
`dagspaces/grpo_training/conf/training/grpo/online_rground_external.yaml`,
`dagspaces/grpo_training/stages/grpo_training.py`.

## Why

A design review of the May 2026 λ×ρ sweep found the production GRPO cells
were close to a learning no-op:

- Training reward was **flat across all 96 optimizer steps** (λ=1.0 cell:
  0.458 → 0.405, no upward trend; entropy and clip ratios constant). See
  `multirun/2026-05-13_lambda_axis_sweep/12-31-50/lambda=1.0/grpo_only_online_external/outputs/grpo/checkpoint/checkpoint-96/trainer_state.json`.
- With `num_generations: 2`, **60% of groups had identical R_ground**
  (median within-group gap exactly 0 → zero advantage) and 62% had a
  composite gap < 0.05; TRL's default `scale_rewards="group"` std-scaled
  the remaining near-ties to full ±1 advantages (judge-noise gradient).
- **77.6% of sampled completions were no-flow declarations** despite the
  50/50 chunk balance — most groups were (no-flow, no-flow) ties scored by
  the coarse coverage judge (bimodal: 0.0 or 0.85–1.0).
- TRL 0.29.1 defaults applied silently: `beta=0.0` meant **no KL anchor to
  the SFT policy**, so the small downstream regressions vs SFT-CI (CIRL
  −0.9%, PrivacyLens QA, ConfAIde tier2a — see
  `notebooks/normative-simulacra/tables/lambda_sweep_rebuttal_2026_05_31.md`)
  were unconstrained drift + noise, which also explains the λ-sweep
  flatness.

## What changed

`online_rground_external.yaml` (production config for the sweep cells):

| Knob | Old | New | Rationale |
|---|---|---|---|
| `num_generations` | 2 | 8 | non-degenerate group statistics |
| `learning_rate` | 1e-6 | 1e-5 | LoRA-scale lr; 1e-6 never moved the policy |
| `gradient_accumulation_steps` | 16 | 32 | 4 prompts/step at G=8 |
| `num_epochs` | 1 | 3 | 772 prompts × 1 epoch was too little signal |
| `beta` | (unset → 0.0) | 0.01 | KL anchor to the SFT reference |
| `scale_rewards` | (unset → "group") | "none" | Dr. GRPO: no std-scaling of advantages |
| `mask_truncated_completions` | (unset → false) | true | DAPO: truncated → parse-fail → 0 reward otherwise rewards the short no-flow path |

`stages/grpo_training.py`: forwards `beta` / `scale_rewards` /
`mask_truncated_completions` / `num_iterations` from `training.grpo` to
`GRPOConfig` **only when set** (configs that omit them keep TRL defaults);
logs them at startup; records them in `training_metadata.json`.

Note: because the trainer re-wraps the merged SFT model with a fresh LoRA,
TRL computes reference logprobs by disabling the adapter — the KL anchor is
**exactly the SFT checkpoint**, with no separate reference model in memory.

## Caveats / follow-ups

- Cost: G=8 quadruples generation + judge calls per prompt, and 3 epochs
  triples passes. Per-cell wall-time will rise well beyond the old ~6 h;
  consider `num_iterations: 2` (pass-through already wired) to amortize.
- Old sweep checkpoints (`grpo-lXXX-rYYY` model yamls) trained under the
  old recipe; results are not comparable to new-recipe runs.
- Promotion gates before any new sweep (Phase 5 of the review): reward must
  trend up, `frac_reward_zero_std` < ~0.2, KL bounded, no-flow rate near
  the gold base rate.
- The seed-variance result (final-reward CV 3.5%) was measured under the
  flat-reward recipe — it shows pipeline reproducibility, not learning
  robustness; re-run under the new recipe before citing it.
- Phases 2–4 of the review remain: zero-variance prompt pre-screening,
  within-group ranking judge, re-enabling the judgment-vignette mix.
