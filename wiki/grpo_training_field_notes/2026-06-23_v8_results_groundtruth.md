# GRPO redesign field notes — v8 results: ground-truth eval + the reward is the binding constraint

**Date:** 2026-06-23 · **Author:** field analysis from reward traces + GoldCoin eval · **Status:** scratch / working notes (camera-ready generative stage)

Closes out [2026-06-22_v8_plan.md](2026-06-22_v8_plan.md) (the plan this reports on)
and continues [2026-06-22_v6_v7_optimizer_stability.md](2026-06-22_v6_v7_optimizer_stability.md).
Same task / pipeline (`qwen3.5-9b` contentless-v6 SFT base,
`pipeline=grpo_only_online_external`, `training/grpo=online_rground_external`).

## TL;DR

- **v8 ran the two planned levers** — `num_iterations=2` (μ=2) + `beta=0.02`
  (Lever 1) and the symmetric contrastive clamp (Lever 2) — and was **killed at
  step 400 / 1080 (epoch ~1.1)** after the ground-truth eval came in.
- **For the first time, GRPO moved the policy on held-out ground truth — and it
  is *real*, not entropy noise.** GoldCoin-HIPAA **applicability accuracy
  0.921 → 0.972** (+5.1 pts), recovering 11 of the SFT base's 17 over-conservative
  "Not Applicable" misses. The gain is **fully present at the clean ckpt200**
  (epoch 0.55, pre-instability) and **identical at ckpt400**. The flat per-step
  reward trace *did* hide a moved policy — settling the v2–v7 ambiguity.
- **…but the movement is an indiscriminate "engage / permit" bias, not
  context-calibrated CI reasoning.** The same directional push *helps*
  applicability and *hurts* compliance: **Forbid (non-compliant) recall collapses
  0.70 → 0.40 → 0.35**, dropping compliance macro-F1 **0.741 → 0.696 → 0.660**.
  Overall accuracy hides it (Permit is the majority class).
- **The entropy breakout bought nothing.** ckpt400 ≈ ckpt200 on applicability and
  is *slightly worse* on compliance. All useful learning was captured by epoch
  0.55; everything after the step-240 breakout was neutral-to-degrading.
- **Decisive conclusion: the reward is the binding constraint, and it is the
  *cause* of the entropy collapse — not the optimizer.** No (β, μ, lr) setting
  fixes it (see "Why v9 must change the reward"). **Keeper: ckpt200.** v9 = reward
  redesign.

## Data sources

| run | dir | job | notes |
|---|---|---|---|
| v8 training | `multirun/2026-06-22_grpo_redesign_full_v8/15-24-43/` | 725705 | β=0.02, μ=2, symmetric clamp; **killed @ step 400/1080**; promote n/a |
| v8 GoldCoin eval | `multirun/2026-06-23_goldcoin_v8_vs_sft/13-29-58/` | 758331 (array) | 3 arms: 0=sft-base, 1=ckpt200, 2=ckpt400; judge-free |

Keeper checkpoint: `…/grpo/checkpoint/checkpoint-200` (LoRA adapter on
`…/checkpoint/_merged_sft`). Eval arms share that merged base, so the comparison
isolates the GRPO LoRA.

## v8 training — the trajectory splits in two halves

μ=2 doubled the step count (1080 vs v6's 537). The run looked like stable v7 for
~240 steps, then the v6 instability fingerprint re-emerged — milder and slower,
but unmistakably the same coupling.

| signal | v7 (β=.02, μ=1) | v8 first ~240 steps | v8 steps 240→400 |
|---|---|---|---|
| entropy | 0.58→0.69 flat | 0.59→0.79 (stable) | **0.97→1.71** (climbing +0.0046/step) |
| kl | ~0.22 | <0.16 | 0.30→0.53 |
| IS-ratio mean | ~1.0 | ~0.95 | 0.92→0.85 (drifting) |
| logp_diff (mean) | 0.08→0.15 | ~0.14 | 0.26→0.40 |
| corr(entropy, logp_diff) | flat | — | **+0.96** |
| corr(entropy, IS) | flat | — | **−0.97** |
| reward_trend (gain) | −0.0014 | — | **−0.0366** (whole run) |
| within-group adv (gold-YES mixed) | +0.68 (100%) | — | **+0.685 over 89 groups (100%)** |
| gold-YES abstention | ~0.62 (tail no-flow .73) | — | **0.539** (moved down) |
| R_ground \| extract | 0.265 | — | 0.332 (Lever 2 partial) |

Extrapolating the +0.0046/step entropy slope to step 1080 → **~4.8 nats** (v6
catastrophe was 6.0). The back half of the run was on track to collapse.

## GoldCoin-HIPAA ground truth — the new, decisive data

Judge-free (sklearn accuracy / macro-F1). Three arms, shared merged-SFT base.

**Applicability** (n=214, balanced; "Applicable" = the *active*, non-dismissive call)

| arm | acc | macro-F1 | recall[Applicable] | recall[Not-App] |
|---|---|---|---|---|
| sft-base | 0.921 | 0.920 | 0.841 (misses 17) | 1.00 |
| **v8-ckpt200** | **0.972** | **0.972** | **0.944** (misses 6) | 1.00 |
| **v8-ckpt400** | **0.972** | **0.972** | **0.944** (misses 6) | 1.00 |

**Compliance** (n=107, imbalanced 87 Permit / 20 Forbid; "Forbid" = flag the non-compliant flow)

| arm | acc | macro-F1 | recall[Permit] | recall[Forbid] |
|---|---|---|---|---|
| sft-base | 0.822 | **0.741** | 0.851 | **0.70** (14/20) |
| v8-ckpt200 | 0.841 | 0.696 | 0.943 | 0.40 (8/20) |
| v8-ckpt400 | 0.822 | **0.660** | 0.931 | **0.35** (7/20) |

**Caveats:** compliance Forbid is n=20, so ckpt200→ckpt400 (8→7) is ~1 case =
noise; the sft→GRPO drop (14→8) is 6 cases — real but small-n. Applicability
(214) is robust.

## Interpretation

The two tasks have *opposite* structure, and GRPO pushes the **same direction**
on both: toward the engaged/permissive answer (Applicable; Permit). On
applicability the active answer is more often correct → +5 pts. On compliance the
identical push *overshoots*, converting correct refusals into wrong permits →
Forbid recall craters. This is the training-trace story made literal: GRPO is
reducing abstention / shifting toward the active-flow answer — **globally and
indiscriminately**, not by improving context-sensitivity. It rewards a *behavioral
direction* ("extract / don't abstain"), not a *specific correct completion*.

Because the applicability gain is **saturated at the clean ckpt200 and unchanged
at ckpt400**, the useful learning is genuine and pre-instability; the entropy
breakout adds nothing. Hence the kill at step 400: ckpt200 is the best held-out
checkpoint and further training only risked collapse.

## Why v9 must change the reward — and why the reward *causes* the entropy collapse

Think carefully about the *direction* of the instability. A **healthy** RL reward
(one with a clear, consistently-higher-reward best completion) under *increased*
update strength (μ=2) makes the policy **converge faster onto that mode → entropy
DECREASES**. That is the normal GRPO picture; μ=2–4 with a verifiable reward
sharpens the policy. Ours did the opposite: more update strength produced entropy
*increase*. **That sign flip is diagnostic.** The policy, when pushed harder, does
not sharpen — it diffuses. That happens only when the reward gradient is, on net,
a *diffusing* (unlearning) force rather than a *concentrating* one.

The causal chain (entropy is the driver; the rest is downstream — confirmed by the
v6 correlations, re-confirmed in v8):

```
reward gradient geometry (non-concentrating)
   → under update strength sufficient to move the policy, entropy ↑ (diffusion)
   → policy puts mass on lower-prob / longer-tail tokens
   → vLLM-rollout vs HF-trainer logprob mismatch ↑  (corr +0.96)
   → IS ratio ↓ / collapse                          (corr −0.97)
   → token_truncate masks gradient
```

The optimizer knobs (β, μ, lr) only set **how fast** and **via which symptom** the
collapse surfaces — they do not create its direction. The (β, μ) sweep proves the
reward is the binding constraint, because **stability and learning are
anti-correlated across it** — the signature of a reward problem, not an optimizer
problem:

- **v7** (β=0.02, μ=1): strong anchor + weak update → **stable but inert** (reward
  flat, abstention unchanged). The policy sits still because the reward gives
  nothing to climb.
- **v6** (β=0, μ=1) and **v8** (β=0.02, μ=2): weak anchor or strong update → the
  reward's diffusing gradient wins → **entropy runaway**.
- **No (β, μ, lr) gives both stability and learning at once**, because the
  optimizer can only scale/anchor the gradient that exists; it cannot manufacture
  a concentrating gradient the reward does not provide.

**Why the reward's gradient is non-concentrating** — three reinforcing causes,
all already in the traces, now corroborated by GoldCoin:

1. **It is one-directional.** The composite *rewards* extraction (R_ground, weight
   0.50) and *penalizes* abstention (`abstention_penalty=0.4`), but gold=False is a
   prescriptive-norm label, so there is **no penalty for inappropriate extraction**
   and **no reward for correct refusal**. A one-sided "extract more" reward cannot
   identify a best completion — it can only inflate extraction probability broadly.
   That is intrinsically entropy-increasing under strong updates (many ways to
   "extract more"; the policy spreads across all of them). The **GoldCoin Forbid
   recall collapse (0.70→0.35) is this one-sidedness made manifest**: nothing in
   the reward says "correctly forbidding a bad flow is good."
2. **The concentrating signal is sparse and diluted.** The clean +0.685 advantage
   lives only in *mixed* gold-YES groups; homogeneous groups have ~0 centered
   advantage (no gradient). Net gradient = small signal + much zero-group noise →
   grad_norm ~0.1, locally flat.
3. **It is high-variance.** Judge noise + the residual R_ground=0 clamping
   (still 0.44 on extractors at ckpt400) add variance that, step-to-step, flattens
   the distribution rather than sharpening it.

**Therefore v9 changes the reward, not another optimizer knob.** The fix must make
the gradient *point at a specific correct completion* rather than push a direction:
make the reward **two-sided and selective** — reward correct refusal as much as
correct extraction, and penalize inappropriate extraction (permitting a flow that
should be forbidden), so "be right in context" becomes a confident target the
policy can concentrate onto (entropy ↓), instead of "extract more" (entropy ↑).
The GoldCoin compliance regression is the concrete, falsifiable target for v9: a
correct reward should *raise* Forbid recall back toward/above the SFT 0.70 while
keeping the applicability gain.

## Reproduction

Training-side table (works on any run dir; the committed, unit-tested path):

```bash
python scripts/grpo_field_metrics.py multirun/2026-06-22_grpo_redesign_full_v8/15-24-43
```

GoldCoin arms (each arm writes `compute_metrics_{applicability,compliance}/metrics.parquet`):

```python
import pandas as pd, json
ROOT = "multirun/2026-06-23_goldcoin_v8_vs_sft/13-29-58"
for arm, label in {"0":"sft-base","1":"ckpt200","2":"ckpt400"}.items():
    for task in ("applicability","compliance"):
        r = pd.read_parquet(f"{ROOT}/{arm}/goldcoin_hipaa/outputs/compute_metrics_{task}/metrics.parquet").iloc[0]
        print(label, task, round(r.accuracy,4), round(r.macro_f1,4), json.loads(r.per_class))
```

Eval re-run (judge-free; 3-arm sweep): `scripts/run_eval_v8_goldcoin.sh`.
Arm model configs: `dagspaces/common/conf/model/qwen3.5-9b/v8-ckpt{200,400}.yaml`.

## Related

- [2026-06-22_v8_plan.md](2026-06-22_v8_plan.md) — the plan this reports on
- [2026-06-22_v6_v7_optimizer_stability.md](2026-06-22_v6_v7_optimizer_stability.md) — the instability diagnosis v8 inherited
- [2026-06-19_redesign_v2-v5_gold_label.md](2026-06-19_redesign_v2-v5_gold_label.md) — the gold-label invariance v8 finally broke (on ground truth)
- [grpo-reward.md](../grpo-reward.md) — composite reward components & ranked R_ground scoring
- [[project_grpo_flat_reward]] · [[project_reranker_judge_ablation]]
