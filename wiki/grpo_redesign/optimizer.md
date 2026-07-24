# Optimizer preset — fixed across all m-series cells

**Parent:** [README.md](README.md) · **Date:** 2026-07-16 · **Status:** drafted
· **Kind:** infrastructure, not a module. One block, identical in every grid
cell; optimizer experiments are a separate axis and never mix with reward
ablations (principle 5).

## The preset

Inherited wholesale from the v6–v8 stability diagnosis — every value below
was set by a falsified alternative, not taste. TRL 0.29.1.

| knob | value | evidence (field notes) |
|---|---|---|
| `num_generations` G | 8 | G=2 gave 60% tied groups (May sweep) |
| `learning_rate` | 2e-5, `cosine_with_min_lr` floor | v5: default cosine decayed to ~0 before the policy moved |
| `num_epochs` | 3 | 772 prompts × 1 epoch was too little signal |
| `beta` (KL to SFT ref) | 0.02 | v6: β=0 + hot lr → entropy runaway 10×, IS collapse; v7 pilot fixed it single-variable. Memory-R1 also retains KL. |
| `num_iterations` μ | 1 | v8: μ=2 moved held-out metrics but broke entropy ~step 240; v9 reverted. See "v13a" below. |
| `epsilon_high` | 0.28 | configured (DAPO clip-higher) but **inert at μ=1** — ratio ≡ 1; kept so a μ change activates it without a second edit |
| `scale_rewards` | `"none"` | Dr. GRPO: group-std scaling amplifies near-tie judge noise |
| `loss_type` | `"dapo"` (TRL default, unset) | no per-response length normalization |
| `mask_truncated_completions` | true | truncated output fails JSON parse and would reward the short no-flow path |
| `vllm_importance_sampling_mode` | `token_truncate` | v5: default `sequence_mask` zeroed exactly the long-extraction gradients (THE v1–v4 binding constraint) |
| rollout temperature | 1.0 (thinking **off** during GRPO rollouts — `enable_thinking_grpo: false`; think-blocks are stripped at scoring regardless; verified 2026-07-24) | exploration; greedy at eval (Memory-R1 discipline) |
| `save_steps` | 50 | pre-breakout checkpoint cadence |
| seed | 42 (single master seed; the 5-seed protocol varies only this) | seed-variance study |
| vLLM | colocate, gpu_mem 0.45, sleep mode | 1-GPU cell economics (gpu_mem verified against online_rground_external.yaml + m_series.yaml, 2026-07-24) |

The m-series training config is one new yaml (`training/grpo/m_series.yaml`)
that carries this block verbatim plus the module keys — it must **not**
inherit from `online_rground_external.yaml` (that file is keeper-frozen and
carries the legacy reward keys the m-series deletes).

## Relation to the v13a arm

v13a (`scripts/run_grpo_v13a_mu2.sh`, staged on the **old** stack) tests
μ=2 + active clip-higher against the v11 probe. Its result feeds this preset
exactly once: if μ=2 moves held-out metrics *without* the v8 entropy breakout
on a concentrating reward, the m-series preset adopts μ=2 **before** the grid
launches (one preset for all cells, as always); if it breaks or does nothing,
μ stays 1 and the question is closed. What is not allowed: running some grid
cells at μ=1 and others at μ=2.

## What the preset does not decide

Reward shape (the modules), task mix, prompt-set size, universes — all
elsewhere. If a future optimizer question arises mid-grid, it waits for the
grid to finish: a preset change invalidates cross-cell comparability, which
is the grid's entire product.
