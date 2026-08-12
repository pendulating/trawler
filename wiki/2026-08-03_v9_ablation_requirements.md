# v9 GRPO — design scrutiny and camera-ready ablation requirements

**Date:** 2026-08-03 · **Status:** ⚠️ **SUPERSEDED 2026-08-05** — v9 is
deprecated and the ablations proposed here will not be run. The camera-ready
reports the m2 `full` GRPO cell and the k3 `verdict` KTO arm
([2026-07-31_kto_plan.md](2026-07-31_kto_plan.md) §19). §2's evidence audit
remains live and load-bearing: it is the record of which v9-era support numbers
must NOT be reused, and §2.5's power analysis of GoldCoin's 20-case Forbid
split is part of the benchmark-distrust argument behind the §19 ruling.
· **Owner:** Matt

Written after the k-series pivot ended with v9-ckpt100 still the keeper. Every
claim below is verified against the **run configs on disk**, not against wiki
prose or the manuscript — two of the paper's three quantitative support runs
turned out to characterize a different design than the one we report, and
prose did not catch it.

---

## 1 · What v9 actually is

From `multirun/2026-06-23_grpo_redesign_full_v9/20-09-18/0/.hydra/config.yaml`
(the authoritative record for the keeper):

| axis | value |
|---|---|
| `num_generations` (G) | 8 |
| `learning_rate` | 2.0e-5, cosine w/ min_lr_rate 0.3 |
| `num_epochs` | 3 (reported checkpoint is step 100, epoch ~0.58) |
| `grad_accum` × `per_device_bs` | 32 × 1 |
| `beta` (KL) | 0.02 |
| `reward_composition` | **directional** (`R = gate × content`) |
| `reward_weights` | [0.10, 0.05, 0.05, 0.20, 0.10, 0.50] |
| gate / content partition | gate = idx {0,1,2,4}; content = idx {3,5} |
| `rground_scoring` | **ranked** (listwise), `rank_top_k` 5, `rank_weight` 0.5 |
| `rground_app_mode` | **multiplicative**, `rground_app_weight` 0.3 |
| `rground_app_floor` | 0.4 (`app_floor_prohibit` absent → symmetric v9) |
| `contrastive_lambda` (λ) | 1.0 |
| `contrastive_ratio` (ρ) | **0.0** — off in production |
| `vignette_ratio` | 0.3 |
| `no_flow_scoring` | independent |
| training judge | **Qwen3.6-27B** (keeper-era; NOT the Gemma-4-31B-it that serves today) |

Composition detail that constrains ablation design
(`stages/rewards.py`, `directional` branch):

```python
gate_w    = sum(weights[k] for k in (0,1,2,4)) or 1.0
content_w = sum(weights[k] for k in (3,5))     or 1.0
gate    = sum(weights[k]*components[k] for k in (0,1,2,4)) / gate_w
content = sum(weights[k]*components[k] for k in (3,5))     / content_w
return gate * content
```

The `or 1.0` rescues the **denominator**, not the numerator. So zeroing *all*
gate weights yields `gate = 0.0` and **R ≡ 0 for every completion** — a
degenerate run with no gradient, not an "ungated" run. Same for content.
Consequence: single-component ablations are config-only; a whole-factor
ablation needs code.

---

## 2 · Evidence audit — what is actually evidenced *on v9*

### 2.1 The λ × ρ sweep is NOT v9 (15 cells, May 13–18)

`multirun/2026-05-{13,15,17,18}_*_sweep`:

| | λ/ρ sweep | v9 |
|---|---|---|
| `num_generations` | **2** | 8 |
| `learning_rate` | **1.0e-6** | 2.0e-5 (20×) |
| `num_epochs` | **1** | 3 |
| `vignette_ratio` | **0.0** | 0.3 |
| `reward_composition` | *absent* → pre-directional | directional |
| `rground_scoring` | *absent* → absolute | ranked |
| direction multiplier | *absent* | multiplicative |
| `reward_weights` | same | same |

Only the weight vector and ρ match. The sweep predates **every mechanism that
defines v9**. And it ran at **G=2**, where the wiki records ~60% of groups tied
exactly under absolute R_ground scoring → near-zero advantage from the
0.71-mass component. Flatness measured there is close to uninformative.

**⇒ λ is unablated for v9.** The methods claim that the wrong-universe margin
penalty "discourag[es] memorization of source-specific norms" has no v9
support.

### 2.2 The 5-seed variance result is NOT v9 (May 28)

`multirun/2026-05-28_seed_variance_sweep`: `num_generations 2`,
`learning_rate 1.0e-6`, `num_epochs 1`, `contrastive_lambda 0.5`,
`vignette_ratio 0.0`, no directional composition, no listwise scoring, no
direction multiplier — the same May regime as §2.1.

**⇒ the headline stability claim (final-reward CV 3.5%), used to rebut a
reviewer's variance concern, does not describe the reported model.** The same
applies to `tab:seed-variance-components` (per-component CV: r_consist 1.1%,
r_context 17.6%) — same sweep, same provenance.

The appendix TODO says the CV is "quotable only for the superseded v9 recipe",
which assumes it *was* measured on v9. It was not. That note is wrong about
provenance in the other direction.

### 2.3 The v8 → v9 comparison IS v9-comparable — but is a 2-factor change

`multirun/2026-06-22_grpo_redesign_full_v8` vs v9. **Identical** on G=8,
LR 2.0e-5, 3 epochs, beta 0.02, λ=1.0, ranked scoring, rank_weight 0.5,
app_weight 0.3, vignette 0.3. Differs in exactly two:

| | v8 | v9 |
|---|---|---|
| `reward_composition` | gated | **directional** |
| `rground_app_mode` | *absent* → additive | **multiplicative** |

This is a genuinely controlled comparison — far better than §2.1/§2.2. But it
changes **two** factors, while `§sec:direction` attributes the effect to the
direction multiplier alone. That attribution is confounded with the
composition change. One cell (directional + additive) disentangles it.

### 2.4 Checkpoint selection is defensible but undocumented in the paper

`2026-06-23_v9_plan.md:166` — ckpt100 was chosen as the **matched-epoch
counterpart to v8-ckpt200** (epoch ~0.55), a rule fixed *before* the
comparison. ckpt200 and ckpt300 exist; ckpt200 was evaluated and "the ckpt-200
plateau did not deliver" further gains. So this is **not** eval-set
cherry-picking. But the paper reports an epoch-0.58 checkpoint of a 3-epoch run
without stating the rule or the ckpt200 result — a reviewer will ask, and the
answer should be in the paper, not only here.

### 2.5 The headline GoldCoin result is underpowered

Compliance split: 107 cases, **20 Forbid**. From the field notes, and tested
here (Fisher exact, two-sided):

| arm | Forbid recall | 95% CI |
|---|---|---|
| SFT base | 13/20 = 0.650 | [0.44, 0.86] |
| v8-ckpt200 | 7/20 = 0.350 | [0.14, 0.56] |
| v9-ckpt100 | 11/20 = 0.550 | [0.33, 0.77] |

| comparison | p |
|---|---|
| v8 → v9 (the "repair") | **0.341** |
| SFT → v8 (the "collapse") | **0.113** |
| SFT → v9 | 0.748 |

**None is significant.** `fig:goldcoin-breakdown`'s caption — "v8's
one-directional reward collapsed Forbid recall to 0.35; the v9 two-sided reward
recovers it to 0.55" — describes a 4-case and a 6-case swing on a 20-case
split, with all three CIs overlapping.

Two consequences, and the second governs the whole grid:

1. The claim as written is not defensible on this evidence alone.
2. **Forbid recall cannot adjudicate ablation cells.** Binomial SE at n=20 is
   ±0.11; only differences >~0.3 would clear it. Any ablation powered on this
   metric is uninterpretable before it starts.

### 2.6 Summary

| support | on v9? |
|---|---|
| λ × ρ sweep (15 cells) | ✗ different reward AND optimizer regime |
| 5-seed variance, CV 3.5% | ✗ same May regime |
| per-component seed CV table | ✗ same May regime |
| v8 vs v9 GoldCoin | ✓ controlled, but 2-factor + underpowered |
| SFT pair-format 2⁴ | n/a — SFT stage, not the RL reward |
| PL judge ablation | n/a — eval-side |
| **per-component reward LOO** | **✗ never run, on any design** |

---

## 3 · Ablations to run

Baselines are free: the 2026-08-03 `eval_rl_stage_keeper` sweep supplies
`full` (v9-ckpt100) and `sft` (v9-base) under the camera-ready protocol.

### Tier A — the paper's claims rest on these (4 cells, config-only)

| cell | override | claim tested | kill criterion |
|---|---|---|---|
| `−ground` | `reward_weights[5]=0.0` | R_ground (0.71 of content) is the mechanism | ≈ `full` ⇒ the mechanism claim fails; the paper must say the gains are not attributable to the grounding judge |
| `−direction` | `rground_app_weight: 0.0` | the two-sided multiplier is load-bearing | ≈ `full` ⇒ §sec:direction is decorative |
| `−context` | `reward_weights[3]=0.0` | completes the 2-way content LOO | — |
| `λ=0` | `contrastive_lambda: 0.0` | wrong-universe penalty discourages memorization (**unablated at v9**, §2.1) | ≈ `full` ⇒ restate as "not a sensitive knob", now honestly |

`−ground` + `−context` fully decompose the content factor — a complete, tidy
table rather than a grab-bag.

### Tier B — expected reviewer asks (3 cells, config-only)

| cell | override | why |
|---|---|---|
| `direction-additive` | `rground_app_mode: additive` | Disentangles §2.3's 2-factor confound: isolates multiplicative-vs-additive at fixed directional composition. Directly tests the sentence in `app:reward`. |
| `−vignette` | `vignette_ratio: 0.0` | The 0.3 judgment mix is an untested recipe choice; should hit ConfAIde-2b / CIRL specifically. |
| `ground-absolute` | `rground_scoring: absolute` | The listwise justification rests on a **G=2** tie observation; production is G=8, where ties are far rarer. May no longer hold at the shipped setting. |

### Tier C — standard-practice controls (3 cells)

| cell | what | why a reviewer asks |
|---|---|---|
| `seed-variance-v9` | rerun `full` at seeds {43,44}, ≥3 total | §2.2 — the reported model has **no** measured seed variance. This is the single most likely "reject" trigger of anything here. |
| `compute-matched SFT` | continue SFT for the same optimizer steps as v9-ckpt100, no RL | "Is the gain RL, or just more gradient steps on the same data?" We currently cannot answer. Standard control; cheap. |
| `checkpoint curve` | eval ckpt {100, 200, 300} on the current protocol | Makes §2.4 auditable in the paper instead of the wiki. |

### Tier D — needs a code change; run only if the gate claim is kept

`−gate` is **not** expressible in config (§1: zeroing all gate weights gives
R ≡ 0). Needs a `composition: content_only` branch (~10 lines + a test) to set
`gate ≡ 1.0`. Tests "malformed output cannot bank credit". If we don't run it,
soften that sentence to a description of the design rather than a claim.

### Deliberately NOT run

- **Individual gate components** (`r_complete`, `r_consist`): recorded as
  saturated post-SFT. A near-constant component has ~no within-group variance,
  so it cannot move a group-relative advantage **by construction** — a
  predictable null at ~1 GPU-day each.
- **ρ (`contrastive_ratio`)**: already 0.0 in production. Off, not untested.
- **`confidence_fallthrough`**: a keeper-repro bug toggle, not a paper claim.

---

## 4 · Protocol requirements (get these wrong and the grid is void)

1. **Training judge must be Qwen3.6-27B**, matching the keeper. The judge
   server currently serves Gemma-4-31B-it; training ablation cells against it
   would change a second variable. Alternative: retrain `full` under Gemma-4
   and compare only within the new set — cleaner going forward, one extra cell.
   *(The eval judge stays Gemma-4-31B-it for every row; that is independent.)*
2. **Matched checkpoint.** Every cell read at **step 100**, not at its own
   best-on-eval — otherwise it is selection on the evaluation set, exactly what
   the λ argument is meant to rule out. Pre-register.
3. **Primary endpoint must be adequately powered.** Per §2.5, do **not**
   adjudicate on GoldCoin Forbid recall (n=20). Pre-register the endpoint
   before running — GoldCoin compliance macro-F1 (n=107) and PrivacyLens QA
   accuracy (n=1479) are the defensible choices; CIRL Net (n=729) and VLM Q7
   (n=783) as secondary.
4. **Noise bar** = the camera-ready table's measured per-cell band, *not* the
   3.5% figure (which is training-reward CV, and per §2.2 is not even v9's).
5. **Prescreen cache**: every knob here is in the cache signature, so each cell
   re-runs prescreen. Budget it.
6. The §17.2 gold-validity finding does **not** contaminate this — the grid is
   read on external benchmarks, which the construct mismatch does not touch.

---

## 5 · Manuscript corrections needed regardless of what we run

These are **provenance errors, not preferences**; they hold even if zero
ablations are run.

1. **§4.2 λ sentence** — "even λ=0 lands within 0.011 … so our primary choice
   does not reflect tuning on the evaluation set" reads as a statement about
   the reported model. It is not (§2.1). Either scope it explicitly to the
   v-era configuration or replace it with the v9 `λ=0` cell.
2. **`app:seed-variance`** — the CV 3.5% and the per-component CV table are not
   v9 (§2.2). Scope or re-run. Given it is the reviewer-variance rebuttal,
   re-running (Tier C) is strongly preferred to scoping.
3. **`fig:goldcoin-breakdown` caption** — state n=20 for the Forbid split and
   that the differences are not individually significant (§2.5), or demote the
   figure from evidence to illustration.
4. **Checkpoint rule** — state in the paper that ckpt100 was pre-specified as
   the matched-epoch counterpart to v8-ckpt200 and that ckpt200 plateaued
   (§2.4).

---

## 6 · Compute and sequencing

~1 GPU-day training per cell + one eval sweep per cell; prescreen re-runs per
cell.

| stage | cells | rough |
|---|---|---|
| Tier A | 4 | 4 GPU-days + 4 eval sweeps |
| Tier C seed-variance | 2 | 2 GPU-days + 2 eval sweeps |
| Tier B | 3 | 3 GPU-days + 3 eval sweeps |
| Tier C others | 2 | ~1 GPU-day + 2 eval sweeps |
| Tier D | 1 | code + 1 GPU-day |

**Suggested order.** Tier A `−ground` first — it is the single cell that can
invalidate the paper's mechanism claim, and everything else is worth less if it
comes back null. Then Tier C seed-variance in parallel (it is the most likely
reviewer trigger and needs no new code). Then Tier A remainder, then Tier B.

**Honest expectation.** I previously argued the λ/ρ flatness implied reward
knobs would not move v9 benchmarks. That inference is withdrawn (§2.1): it was
measured on a different reward in a near-zero-gradient regime. **We have no
evidence either way about v9's sensitivity to its reward composition.** The
grid is genuinely unmeasured, and `−ground` is a real risk to the mechanism
claim, not a formality.
