# GRPO redesign field notes — v2→v5 gold-label completion analysis

**Date:** 2026-06-19 · **Author:** field analysis from reward traces · **Status:** scratch / working notes (camera-ready generative stage)

Subject: the four "grpo_redesign_full" runs (v2–v5) of `qwen3.5-9b/sft-ci` under
`pipeline=grpo_only_online_external`, `training/grpo=online_rground_external`.
Focus is the **gold-label completion metrics** — does the model produce the
gold-correct judgment / actually extract flows — and how they responded (or
didn't) to the hyperparameter tuning across versions.

## TL;DR

- The redesign **fixed the GRPO mechanics** (G=8, ranked R_ground scoring, 0%
  fully-tied groups, prescreen guarantees within-group reward variance →
  non-zero advantages every step). Confirmed in v5: within-group composite std
  ≈0.21, **0/498 tied groups** (was ~60% pre-redesign).
- But **none of the v2→v5 tuning moved the gold-label behavior.** Abstention,
  grounding-on-extraction, and judgment accuracy are *flat across all four runs*
  to within noise. The policy's actual decision behavior is invariant to every
  knob we turned.
- The reward *gradient points the right way* (extract = +0.37 composite, abstain
  = −0.13; the `abstention_penalty` is correctly wired and biting). The policy
  just isn't following it — strong evidence the bottleneck is **not reward shape
  / RL hyperparameters** but the SFT prior + update strength + judge ceiling.

## Data sources

| run | dir | reward_traces |
|---|---|---|
| v2 | `multirun/2026-06-14_grpo_redesign_full_v2/` | 3232 rec / 404 steps |
| v3 | `multirun/2026-06-15_grpo_redesign_full_v3/` | 4176 rec / 522 steps |
| v4 | `multirun/2026-06-17_grpo_redesign_full_v4/` | 2968 rec / 371 steps |
| v5 | `multirun/2026-06-18_grpo_redesign_full_v5/` | 4048 rec / 506 steps (RUNNING, ~97%, jobs 496867/496868) |

Traces at `<run>/*/0/grpo_only_online_external/outputs/grpo/checkpoint/reward_traces.jsonl`.
Each record: `call` (GRPO step), `idx` (candidate in group, G=8), `task_type`
(`ci_extraction` | `norm_judgment`), `gold_judgment` (yes/no for norm_judgment;
None for extraction), `completion`, `composite`, `components`.

## Config evolution (resolved hydra config diffs)

Overrides are **identical** across all four runs
(`contrastive_lambda=1.0`, `contrastive_ratio=0.0`, `abstention_penalty=0.4`
[see note], `prescreen.require_flow_variance=True`). The real changes were edits
to the base config `training/grpo/online_rground_external.yaml` between runs:

| step | change |
|---|---|
| v2 → v3 | `abstention_penalty: 0.2 → 0.4` · `dev_fraction: 0.05 → 0.0` |
| v3 → v4 | `beta: 0.01 → 0.0` (drop KL anchor) · `epsilon_high: (unset) → 0.28` (asymmetric clip) |
| v4 → v5 | `vllm_importance_sampling_mode: (unset) → token_truncate` |

(NB: the v4/v5 override files *also* list `abstention_penalty=0.4`; in v2 the
base default was 0.2 and the override was not yet present — net effect is the
v2→v3 bump above.)

**Constant across all runs:** `num_generations=8`, `learning_rate=1e-5`,
`per_device_batch_size=1`, `gradient_accumulation_steps=32`, `num_epochs=3`,
`max_completion_length=3072`, `rground_scoring=ranked` (`rank_top_k=5`,
`rank_weight=0.5`), `reward_composition=gated`, `scale_rewards=none`,
`no_flow_scoring=independent`, `prescreen.enabled=true`
(`num_samples=8`, `reward_std_min=0.05`, `min_keep=8`, `require_flow_variance=true`),
`reward_weights=[0.1,0.05,0.05,0.2,0.1,0.5]` (R_ground=0.50 dominant),
`judgment_reward_weights=[0.5,0.25,0.25]`, `vignette_ratio=0.3`.

## Gold-label completion metrics across runs

Computed from reward traces (parse = regex on completion JSON;
abstention = `has_information_exchange:false` / `flows:[]`).

| run | abstain% | R_ground \| extract | R_ground=0 on extractors | gold-match | recall(yes) | recall(no) | parse% | majority baseline |
|---|---|---|---|---|---|---|---|---|
| v2 | 69.8% | 0.269 | 34.1% | 64.2% | 54.8% | 93.5% | 99.8% | 75.7% |
| v3 | 70.4% | 0.263 | 37.2% | 65.6% | 59.9% | 86.9% | 99.9% | 78.9% |
| v4 | 69.1% | 0.262 | 35.1% | 64.9% | 57.9% | 87.9% | 100.0% | 76.6% |
| v5 | 69.3% | 0.265 | 34.8% | 66.5% | 60.5% | 89.1% | 99.9% | 79.1% |

Everything is flat to within run-to-run noise. The tuning (penalty 0.2→0.4,
KL off, asymmetric clip, importance sampling) produced **no measurable shift**
in any gold-label metric.

## Precise gold-conditional abstention (the smoking-gun metric)

The overall abstention rate above mixes gold-has-flow and gold-no-flow chunks.
The metric that actually diagnoses over-abstention is **conditional abstention
on gold-HAS-flow chunks** (these *should* be extracted, so abstaining is wrong)
vs. gold-NO-flow chunks (abstaining is correct). The traces don't store
`gold_has_exchange`, so it was reconstructed by matching each trace back to its
source chunk:

- Gold source: `/share/pierson/matt/n2s4cir/data/fiction10/ci_reasoning.parquet`
  (`gold_has_exchange = has_information_exchange`; base rate 386/2993 = 12.9%).
- Trace prompts are **truncated to 4000 chars** (W&B trace size cap), so full
  `article_text` is not a substring. Matched on a chunk-prefix fingerprint
  (`article_text[40:240]`) within the same `source_id` (=`gutenberg_id`).
  Match rate **99.2–99.6%** across runs; gold-positive fraction ~69% (the
  prescreen `require_flow_variance` enriches the training mix from the 13% raw
  base rate up to ~69% gold-has-flow).

| run | abst \| gold=YES (wrong) | abst \| gold=NO (correct) | extract-rate gap (YES−NO) |
|---|---|---|---|
| v2 | **64.4%** | 84.6% | 20.2 pp |
| v3 | **63.8%** | 85.2% | 21.4 pp |
| v4 | **62.0%** | 85.4% | 23.4 pp |
| v5 | **62.2%** | 85.8% | 23.6 pp |

- **The wrong-abstention metric is dead flat at ~62–64% across all four runs.**
  An ideal policy would abstain ~0% on gold-YES; the model misses ~62% of real
  flows, and no tuning moved it. v5's `token_truncate` fix did **not** bend it
  down (within-run windows: 64.5 / 62.2 / 59.0 / 59.4 / 64.1 / 61.3 / 63.1 /
  64.1 — noisy, ends where it started). This is the clean falsification of the
  "extraction gradients were being length-masked" hypothesis: with `token_truncate`
  un-masking them, behavior is unchanged.
- **Correct abstention (gold=NO) stays high and stable at ~85%** — no collateral
  damage from the abstention penalty; the model is not false-extracting more on
  no-flow chunks.
- The model **does discriminate** — it extracts ~2.6× more often on true-flow
  chunks (37.8%) than no-flow chunks (14.2%), and that gap is *stable/slightly
  widening* across runs. So the conditioning signal exists; the policy just sits
  at a heavily abstention-biased operating point it won't leave.

## Finding 1 — `norm_judgment` accuracy is conservative and below baseline

- Gold-match holds at **64–66%** across all runs, **below the ~76–79%
  majority-class ("yes") baseline** in every run. Parse rate ~100%, so this is
  not a formatting problem.
- The shortfall is a **conservative restrict-flow prior**: recall(no) is very
  high (87–94%) while recall(yes) is low (55–61%). The model over-says
  "no / inappropriate." This matches the paper's SFT narrative
  (SFT installs a conservative prior) — GRPO is **not** correcting it.
- No upward trend within any run; the per-window curve is noisy (54–78% in v5).
  Beware tiny end-of-run windows (v5's last window n=40 reads 87.5% — noise).

## Finding 2 — `ci_extraction` abstention is pinned at ~69% despite a correct penalty gradient

This is the headline. The v2→v3 `abstention_penalty` bump and the v3→v4/v5
changes were aimed at the v4-era ~75% abstention artifact. Result:

- Abstention sits at **69–70% in every run** — at most ~6pp below the old
  ~75%, and **flat across training steps** (no downward trend within a run).
  The model defaults to *"the text is descriptive, not prescriptive → no flows"*
  on ~7 of 10 chunks.
- **The penalty is correctly wired and biting** (verified on v5):
  abstaining completions get composite mean **−0.128** (floor −0.40);
  extracting completions get **+0.368** (max 0.91). A ~0.5 reward gap
  unambiguously favoring extraction. The gradient direction is right.
- **Extraction is under-rewarded, which blunts the pull.** When the model does
  extract, **~35% of extractions still score R_ground=0** and mean R_ground on
  extractors is only **~0.265** across all runs. So "extract" is high-variance
  and frequently unrewarded by the grounding judge, while "abstain" is a
  certain −0.4. In expectation extraction still wins (0.37 ≫ −0.13), but not by
  enough to overcome the SFT prior at lr=1e-5, beta=0, 3 epochs.

## Interpretation

The redesign solved the problem it was scoped to solve (reward *shape*: ties,
group size, advantage signal — see also the v4 contrastive/abstention artifact
in [[project_grpo_flat_reward]] and the W&B health surfacing in
`changelog/2026-06-09_wandb_logging_rationalization.md`). But the **gold-label
behavior is invariant to all four runs of tuning**, which points the bottleneck
elsewhere:

1. **Update strength.** Gradient direction is correct but the policy doesn't
   move — consistent with lr=1e-5 + beta=0 + 3 epochs being too weak to shift a
   strong SFT prior. The KL anchor was already dropped (v3→v4) with no effect.
2. **Judge ceiling.** ~35% of genuine extractions earn R_ground=0. If the
   grounding judge zeroes good-faith extractions a third of the time, it makes
   extraction look risky vs. the safe −0.4 floor, capping how hard the reward
   can push toward extraction. (Cross-ref the cross-encoder ablation
   [[project_reranker_judge_ablation]] — the judge encodes structure shallow
   scorers can't replicate; here it may be *too* strict on the positive side.)
3. **SFT prior dominance.** The conservative no/abstain prior is the common
   factor across both tasks and survives every RL config we tried.

## Levers to try next (ordered by directness to the evidence)

1. **Stronger / scheduled update.** Raise lr (e.g. 2–5e-5) or epochs; the
   gradient already points the right way, the policy just isn't following.
2. **Asymmetric / ramped abstention penalty.** Increase the penalty or ramp it
   over training so "safe abstain" stops dominating; current −0.4 is flat.
3. **Loosen the grounding judge on the positive side** so good-faith
   extractions aren't zeroed ~35% of the time (reduces the risk premium on
   extracting). Audit a sample of R_ground=0 extractions first to confirm
   they're false zeros vs. genuinely ungrounded.
4. **Before reading too much into traces:** compare the final checkpoint's eval
   metrics vs. the SFT baseline. GRPO normalizes advantage within-group, so a
   flat per-step trace mean can still coincide with a moved policy — the eval is
   the ground truth.

## Reproduction

Metrics regenerated with inline python over the four `reward_traces.jsonl`
files (no script committed yet — candidate to migrate into `scripts/` /
`tests/` if we keep tracking this). Key derivations:
- abstention: `'"has_information_exchange": false' in completion or '"flows": []' in completion`
- judgment parse: `re.search(r'"judgment"\s*:\s*"([^"]+)"', completion)`
- gold-match: parsed judgment == `gold_judgment.lower()` over `task_type=='norm_judgment'`
- group = `call`; within-group spread = `pstdev(composite per call)`

## Related

- [grpo-reward.md](../grpo-reward.md) — composite reward components & contrastive scoring
- [changelog/2026-06-09_code_review_norms_grpo.md](../changelog/2026-06-09_code_review_norms_grpo.md)
- [changelog/2026-06-09_wandb_logging_rationalization.md](../changelog/2026-06-09_wandb_logging_rationalization.md)
