# `R-GROUND` — judge-scored normative grounding (auxiliary)

**Parent:** [README.md](README.md) · **Date:** 2026-07-16 · **Status:** drafted
· **Kind:** ablatable additive auxiliary (weight 0.25 in `full` by the 2:1
rule). Removal = deletion + renormalization; the `−ground` cell.

## One-liner

One listwise judge call per group: the frozen judge (**Gemma-4-31B-it** —
the canonical teacher/judge family, revised 2026-07-23; the Qwen3.6-27B
deployment reference was stale, see reward-outcome.md D1)
sees all G completions plus the norms retrieved from the chunk's **own
book's** universe, and must rank them by how well their extracted flows are
governed by those norms — rank blended with an absolute grounding score into
a scalar per completion.

This is the published mechanism (the v9 lineage's R_ground correct-universe
pass), kept as an *auxiliary*: the `−ground` cell asks whether process-quality
judging adds anything once outcome supervision exists — Memory-R1's Table 2
comparison run in our domain, from the other side.

## What was hoisted out (the legibility delta vs v9)

The v9–v12a R_ground was three mechanisms in a trench coat. In the redesign
each lives elsewhere or nowhere:

| formerly inside R_ground | now |
|---|---|
| appropriateness **direction** multiplier (`rground_app_*`, v9) + cost tiers (v10/v12a) | subsumed by `R-OUTCOME` (probes *are* direction questions); survives only as the `diag/direction_consistency` metric |
| **contrastive** wrong-universe clamp (`contrastive_lambda`) | its own module, [`R-CONTRAST`](reward-contrast.md) |
| gold-blind **no-flow** ranked participation | gone — abstentions never reach any scored module ([`A-ABSTAIN`](reward-abstain.md)) |

What remains is one question: *are these flows governed by this book's
norms?* — which is the only thing a grounding judge is uniquely positioned to
answer.

## Judge rubric — slimmed to grounding only

The legacy absolute-mode rubric scored three criteria: norm awareness, flow
governance, and **appropriateness consistency**. The third is deleted from
the judge prompt: direction is now scored verifiably by the outcome core,
and keeping it in the judge rubric would double-count direction *and* let
judge opinion re-enter a channel the redesign moved to EM. The m-series
rubric is two criteria:

1. **Norm awareness** — the completion's invoked norms match the retrieved
   norms.
2. **Flow governance** — the extracted flows are actually governed by those
   norms.

## Protocol (inherited, validated — no redesign)

- **Retrieval:** k = 3 norms per flow from the own-book universe,
  Qwen3-Embedding-8B + cosine (the same retriever/embeddings the probe
  builder uses). The m1 run pins this at k = 3 (`rank_top_k: 3` in
  `m_series.yaml`); the keeper listwise path used k = 5 (the code default),
  so keeper ground scores are not numerically comparable on this axis
  (2026-07-24).
- **Listwise scoring** (production since 2026-06): one judge call per group;
  strict ranking (no ties) + an absolute grounding score per candidate;
  `s_i = w_r·(n−rank_i)/(n−1) + (1−w_r)·grounding_i`, `rank_weight` w_r = 0.5.
  Rationale unchanged: LLM judges are badly calibrated at absolute scoring
  (the May-era absolute judge tied 60% of groups — zero advantage from the
  dominant component) but sharp at comparison; the absolute blend keeps a
  uniformly-bad group from being rewarded for winning a weak contest.
- **Failure fallback:** judge-failed group → uniform 0.5 (deliberate zero
  advantage, never noise); `reward/ground/judge_failed_group_frac` on W&B
  with a stdout WARNING.

**A v9-era worry, downgraded:** the deferred concern that `rank_weight`
guarantees spread *even when every candidate is wrong* (a non-concentrating
signal) mattered when R_ground was 50% of the reward. As a 0.25-weight
auxiliary beside a concentrating verifiable core, the rank component's
relative-position signal is bounded in influence; w_r stays 0.5 (lineage
continuity) and is not a grid axis.

## Cost

One judge call per group per step (the wrong-universe pass now bills to
`R-CONTRAST`). Input ≈ all G completions + retrieved norms (~5–6k tokens);
this is the most expensive module per call — which is exactly why `−ground`
is a valuable cell: if it ties `full`, the stack's costliest component is
dead weight.

## Diagnostics (`reward/ground/*`)

`judge_failed_group_frac`, `rank_spread` (within-group std of s_i),
`grounding_abs_mean` (drift of the absolute anchor), judge latency.

## `−ground` prediction

Pre-registered in [ablation-protocol.md](ablation-protocol.md): ≈ `full`
within seed noise (Memory-R1 predicts the quality judge is redundant given
outcome supervision). A consistent drop means the judge carries unique
signal — most plausibly on chunks whose probes under-cover the flow set —
and the paper reports the judge as earning its cost.
