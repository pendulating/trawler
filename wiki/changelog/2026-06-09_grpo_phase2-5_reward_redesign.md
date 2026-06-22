# 2026-06-09 — GRPO Phases 2–5: ranking judge, pre-screening, vignette mix, promotion gates

**Status:** in working tree. Follows the Phase-1 optimizer revision
(`2026-06-09_grpo_phase1_optimizer_revision.md`) — same review, same
evidence base (May λ×ρ sweep was a learning no-op).

## Phase 2 — variance pre-screening (`stages/prompt_screening.py`)

Before training, sample `prescreen.num_samples` completions per prompt
from the **merged SFT policy** (the GRPO initialization), score them with
the exact training reward, and drop prompts whose group reward std <
`prescreen.reward_std_min`. Rationale: 62% of May-sweep groups had
composite gap < 0.05 — zero-advantage groups that only burn generation +
judge throughput. The screen result is cached (keyed on SFT checkpoint +
prompt set + reward signature) via `GRPO_PRESCREEN_CACHE`, so the 15-cell
sweep screens once. Report → `prescreen_report.json` (incl. the SFT
policy's no-flow rate, feeding gate d). vLLM engine for sampling is
created and torn down inside the stage before TRL's colocated engine
starts.

## Phase 3a — listwise ranking judge (`rground_scoring: ranked`)

Absolute judge scores are quantized (observed mass at 0.0 and 0.85–1.0),
so same-prompt completions tie: 60% of groups had identical R_ground.
Ranked mode groups completions by prompt and makes **one listwise judge
call per (group, universe)**: all G candidates, one shared norm set
(retrieved once per group from the mean of chunk + flow-query embeddings,
`rank_top_k` norms), strict ranking + per-candidate absolute
grounding_score.

```
R_i = clamp( rank_weight·(G−rank_i)/(G−1) + (1−rank_weight)·grounding_i
             − λ·wrong_grounding_i , 0, 1)
```

- Wrong-universe side uses grounding scores only (ranking against the
  wrong universe is forced-choice noise).
- No-flow declarations are judged candidates (prompt instructs: rank high
  iff the passage genuinely lacks governed flows). Gold-label correctness
  stays in the programmatic components.
- Parse failures score 0.0 unjudged. Judge failure → uniform 0.5 for the
  group (zero advantage, not spurious gradient).
- Judge cost per group: 2 calls (was 2·G·flows absolute calls).

New: `schemas.CompletionRankingJudgment`, `prompt/reward_judge_ranking.yaml`
(composed as `prompt_reward_judge_ranking` in `conf/config.yaml`),
`JudgeClient.judge_ranking_batch`, `OnlineRGround._call_ranked`,
`NormRetriever.retrieve(top_k=...)` override. `rground_scoring: absolute`
preserves May-sweep behavior exactly.

## Phase 3b — gated composition + r_context consistency

- `reward_composition: gated` → `R = gate × disc` with gate =
  weight-normalized mean of (r_uncert, r_complete, r_consist), disc = same
  over (r_context, r_cohere, r_ground). Saturated gating components stop
  diluting the discriminative signal as an additive ~0.20 offset.
  `additive` remains the default for configs that don't set the knob.
- `r_context` no longer returns 0.0 for **correct** no-flow completions —
  now gold-aware (0.9 / 0.0 / 0.4), mirroring `r_complete`. The old
  behavior punished correct no-flow on a 0.20-weight component while
  r_complete/r_consist rewarded it.

## Phase 4 — judgment-vignette mix re-enabled

`vignette_ratio: 0.3` in `online_rground_external.yaml` (was 0.0 in the
May sweep). The vignette machinery already existed (v3 variants); GRPO's
only consistent benchmark wins were judgment-flavored (GoldCoin
compliance, PrivacyLens helpfulness) — this is the train→eval bridge.

## Phase 5a — held-out dev split

`dev_fraction: 0.05` carves a seeded dev split (post-screen); TRL
generates + scores it every `eval_steps: 50`, logging `eval_reward` — the
held-out reward curve. `per_device_eval_batch_size` is pinned to
`num_generations` (TRL divisibility constraint).

## Phase 5b — promotion gates (`dagspaces/grpo_training/gates.py`)

`scripts/check_grpo_promotion_gates.py CHECKPOINT_DIR` (exits non-zero on
failure; writes `promotion_gates.json`):

| Gate | Pass condition | Default |
|---|---|---|
| reward_trend | last-third mean reward − first-third > min_reward_gain | > 0 |
| zero_std_groups | mean frac_reward_zero_std below threshold | < 0.2 |
| kl_bounded | mean KL to SFT ref below threshold (skips if beta=0) | < 1.0 |
| no_flow_rate | trace no-flow rate within tolerance of gold base rate | ±0.15 |

Every May-sweep cell fails reward_trend. Run gates before spending eval
compute on a cell.

`training_metadata.json` now also records: rground_scoring,
reward_composition, n_screened_out, dev_fraction, n_dev_rows; and
n_flow_chunks / n_no_flow_chunks now describe the **final post-screen
training set** (the gates' base rate), not the pre-screen pool.

## Tests

`tests/grpo_training/test_reward_improvements.py` (27 tests): rank→reward
conversion, gated composition, gold-aware r_context, screening selection,
promotion gates (incl. a synthesized May-sweep flat curve failing the
trend gate), and ranked OnlineRGround end-to-end with faked clients.

## Not yet validated on cluster

Single-cell smoke run still required before sweeping: prescreen vLLM
engine teardown → TRL colocate startup on one GPU; ranking-judge latency
at G=8 (2 calls/group vs 16 absolute calls — expected net win); eval-pass
cost at dev_fraction=0.05.

Smoke harness: `./scripts/smoke_grpo_redesign.sh` (or `LOCAL=1` on a GPU
node) runs `training/grpo=online_rground_external_smoke` — 6 optimizer
steps over a 200-chunk sample exercising every new code path — then
asserts all redesign artifacts (prescreen report, ranked-judge trace
diagnostics, dev-split eval_reward, metadata knobs) and runs the gates
checker (HOLD expected at 6 steps; the artifact assertions are the
pass/fail).
