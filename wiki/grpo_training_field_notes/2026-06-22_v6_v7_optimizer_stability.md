# GRPO redesign field notes — v6 instability diagnosis + v7 beta-anchor pilot

**Date:** 2026-06-22 · **Author:** field analysis from reward traces + trainer_state · **Status:** scratch / working notes (camera-ready generative stage)

Continues [2026-06-19_redesign_v2-v5_gold_label.md](2026-06-19_redesign_v2-v5_gold_label.md).
Same task / pipeline (`qwen3.5-9b` SFT base, `pipeline=grpo_only_online_external`,
`training/grpo=online_rground_external`). Two runs:

- **v6** — full `grpo_redesign_full_v6` on the new **contentless-curated SFT base**
  (`sft_contentless_v6`); the run where we first instrumented the *optimizer*
  internals (entropy / logprob-mismatch / IS ratios), not just gold-label behavior.
- **v7 pilot** — single-variable `beta` 0.0 → 0.02 against the same base, 150-step
  diagnostic, to test whether the v6 optimizer instability was *causing* the flat
  reward.

## TL;DR

- **v6 added a mechanism for the flat reward seen in v2–v5: optimizer instability.**
  With `beta=0`, policy **entropy ran away ~10×** (0.6 → 6.0 nats), which blew the
  vLLM-rollout vs HF-trainer **logprob mismatch up ~80×** (0.09 → 7.5), collapsing the
  **importance-sampling ratios** (min → 1e-15) so `token_truncate` **masked the
  gradient**. corr(entropy, logp_diff)=+0.92, corr(entropy, IS)=−0.96. The clean
  reward advantage never reached the weights.
- **v7 (beta=0.02) FIXED the instability exactly as predicted** — entropy bounded
  (0.58→0.69), logp_diff flat (0.08→0.15), IS≈1.0, KL gate PASS, zero masking, every
  group carrying real reward spread (std ≈0.37).
- **…and reward STILL did not grow.** 0.254 → 0.235 over 150 steps, no trend (gain
  −0.0014); gold-YES abstention if anything *worse* (tail no_flow_rate 0.73 vs gold
  base 0.30). **promote=false.**
- **Decisive conclusion:** stability was *necessary but not sufficient*. This confirms
  the v2–v5 verdict from a new angle — the bottleneck is **not** RL mechanics
  (ties, group size, masking, entropy) but **reward-gradient coupling + the SFT
  abstention prior**. A clean +0.72 within-group advantage still nets to a ~0 gradient
  (grad_norm ~0.1) because contested gold-YES groups are diluted by homogeneous ones.

## Data sources

| run | dir | job | notes |
|---|---|---|---|
| v6 | `multirun/2026-06-19_grpo_redesign_full_v6/` | 537141 | full run, 537 steps; contentless-curated SFT base; promote=false |
| v7 pilot | `multirun/2026-06-21_grpo_v7pilot_beta/20-18-12/` | 717976 | beta=0.02, max_steps=150; 2-GPU request so sanitizer drops busted klara GPU0 |

SFT base (both): `multirun/2026-06-19_sft_contentless_v6/17-40-37/sft_only/outputs/sft/checkpoint`.
Optimizer signals read from `<run>/.../outputs/grpo/checkpoint/checkpoint-150/trainer_state.json`
(`log_history`); reward traces alongside in `reward_traces.jsonl`; gate verdict in
`promotion_gates.json`.

## v6 — the optimizer-instability diagnosis

The reward *signal* is clean and strong: within gold-YES **mixed** groups, extracting
the flow beats abstaining by **+0.72** composite, in 100% of mixed groups. Yet reward
stayed flat ~0.24 over all 537 steps and gold-YES abstention flat ~0.50 (promote=false),
just like v2–v5. v6's new instrumentation shows *why the clean advantage never lands*:

| signal | v6 (beta=0) trajectory |
|---|---|
| entropy | 0.6 → 6.0 nats (**~10× runaway**) |
| sampling_logp_difference (mean) | 0.09 → 7.5 (**~80×**) |
| importance_sampling_ratio (min) | → **1e-15** (collapsed) |
| net effect | `token_truncate` masks the collapsed-ratio tokens → gradient eaten |

Correlations over the run: **corr(entropy, logp_diff)=+0.92**, **corr(entropy,
IS_ratio)=−0.96**. So entropy runaway is the driver; the logprob divergence and IS
collapse follow it.

NB on history: **beta and token_truncate had never been combined before.** v4 ran
beta=0.01 but with the *old* sequence_mask (whole-sequence zeroing); v5/v6 fixed the
masking with `token_truncate` but ran beta=0. v7 tests the untried stable regime.

**Hypothesis:** a moderate KL anchor (beta>0) bounds the entropy runaway → keeps
rollout/trainer in agreement (IS≈1, no masking) → the clean gradient flows → reward
grows.

## v7 pilot — beta=0.02, single variable vs v6

### The optimizer fix worked, precisely as predicted

| signal | v6 (beta=0) | v7 (beta=0.02) | verdict |
|---|---|---|---|
| entropy | 0.6 → 6.0 (**10×**) | 0.58 → 0.69 | ✅ bounded |
| logp_difference (mean) | 0.09 → 7.5 (**80×**) | 0.08 → 0.15 | ✅ flat |
| IS ratio (mean) | → 1e-15 | 0.997 → 0.979 (**≈1.0**) | ✅ healthy |
| IS ratio (min) | ~1e-15 | 0.026–0.06 | ✅ not collapsed |
| KL | unbounded | mean 0.22, lone spike 2.30 @ step 70 | ✅ gate PASS |
| clipped/masked tokens | gradient masked | `completions/clipped_ratio` = 0 | ✅ nothing masked |
| frac_reward_zero_std | — | 0.0 (no collapsed groups) | ✅ |
| reward_std | — | ~0.37–0.39 | ✅ real spread |

beta·KL ≈ 0.02 × 0.06 ≈ **0.001** → the anchor bounded entropy *without* dominating
the advantage. Not an over-anchoring regime.

### …but reward still did not grow

- **reward** 0.254 (step 10) → 0.235 (step 150), bouncing 0.20–0.29, **no trend**.
  Gate `reward_trend`: gain **−0.0014** → **FAIL**.
- **gold-YES abstention worse:** tail `no_flow_rate` **0.732** vs gold base **0.303**
  (deviation 0.43). Gate `no_flow_rate`: **FAIL**.
- **grad_norm** settled ~0.08–0.2 (one 47 spike) → near-zero net gradient.
- **promote = false.**

Per-step (step / reward / entropy / kl / mean_len):
```
 10  0.254  0.584  0.004  179.1
 50  0.279  0.606  0.051  152.9
 70  0.217  0.614  2.301  187.7   <- lone KL spike
100  0.243  0.679  0.291  144.2
150  0.235  0.692  0.060  151.1
```

## Interpretation — confirms v2–v5 from a new angle

v2–v5 concluded the gold-label behavior is invariant to RL tuning and the bottleneck
is the SFT prior / update strength / judge ceiling, **not** reward shape. v6→v7 closes
the last RL-mechanics escape hatch:

- v6 raised a real *new* mechanical fault (entropy/IS instability) that could have
  explained the flat reward.
- v7 **fixed it cleanly** and reward *still* didn't move. So instability was a real
  bug but **not the reason reward is flat.** With a fully healthy optimizer (IS≈1, no
  masking, every group with reward spread), the +0.72 within-group advantage nets to
  ~0 gradient because contested gold-YES groups are diluted by homogeneous ones — the
  reward landscape is locally flat around the SFT policy, and the policy drifts back
  toward the SFT abstention prior (no_flow 0.73).

**Ruled out:** lowering beta further — beta·KL ≈ 0.001 already negligible; lower beta
only re-risks the entropy runaway for no behavioral upside.

## Levers to try next (single-variable, same base, ~150-step diagnostic; keep beta=0.02)

> **Superseded by [2026-06-22_v8_plan.md](2026-06-22_v8_plan.md).** v8 keeps
> beta=0.02 but takes a *different* pair of levers than the two below: `num_iterations=2`
> (activate the inert Clip-Higher + 2× updates) and a symmetric contrastive clamp
> (recover the under-rewarded extraction). The v8 plan argues *against* lever 1 here —
> steepening `abstention_penalty` was already falsified across v2–v5 (P=0/0.2/0.4 dead
> on no-flow rate), and lever 2 here would *lower* `gold_base_rate` and tighten the
> binding gate. See the plan's "Why NOT the other candidate levers".

1. **Steepen the abstention penalty** (0.4 → ~0.7–1.0). Most directly explains the
   0.73 over-abstention: punish abstaining on gold-YES hard enough to make a
   one-directional within-group gradient that overcomes the SFT prior. *(recommended
   first — cleanest single variable; cf. v2–v5 lever #2.)*
2. **Concentrate the training distribution on contested chunks.** Tighten the prescreen
   so split abstain-vs-extract groups dominate instead of homogeneous (zero-net-gradient)
   groups. Attacks "advantages cancel across groups" directly.

Both are reward-gradient-coupling levers, consistent with the v2–v5 diagnosis that the
remaining problem is *update strength against the SFT prior*, not RL mechanics. The
ground-truth check remains the v2–v5 lever #4: **eval the final checkpoint vs the SFT
baseline** — a flat per-step trace can still coincide with a moved policy.

## Reproduction

Every table above regenerates from a single committed command (works on any run dir):

```bash
python scripts/grpo_field_metrics.py multirun/2026-06-21_grpo_v7pilot_beta/20-18-12
python scripts/grpo_field_metrics.py multirun/2026-06-19_grpo_redesign_full_v6
```

It discovers the latest-step `trainer_state.json`, the `reward_traces.jsonl`, and
`promotion_gates.json` under the run dir and prints the optimizer-signal block (incl.
the entropy~logp_diff / entropy~IS correlations), the gold-label table, the within-group
extract-vs-abstain advantage, and the gate verdict. `--json` for machine-readable.

The metric logic is the pure, unit-tested module
[`dagspaces/grpo_training/trace_metrics.py`](../../dagspaces/grpo_training/trace_metrics.py)
(tests: `tests/grpo_training/test_trace_metrics.py`); pass/fail gates remain in
[`dagspaces/grpo_training/gates.py`](../../dagspaces/grpo_training/gates.py). Verified to
reproduce the figures in this note exactly — v6: corr(entropy,logp_diff)=+0.92,
corr(entropy,IS)=−0.96, within-group advantage +0.72 over 227 mixed groups (100%);
v7: reward gain −0.0014, entropy 0.58→0.69, IS≈1.0, advantage +0.68 (100%).

## Related

- [2026-06-19_redesign_v2-v5_gold_label.md](2026-06-19_redesign_v2-v5_gold_label.md) — prior note this continues
- [grpo-reward.md](../grpo-reward.md) — composite reward components & ranked R_ground scoring
- [[project_grpo_flat_reward]] · [[project_reranker_judge_ablation]] · [[project_seed_variance]]
