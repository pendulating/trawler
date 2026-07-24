# Ablation protocol — the m-series grid

**Parent:** [README.md](README.md) · **Date:** 2026-07-16 · **Status:** drafted

## Naming

The modular-stack lineage is the **m-series** (`experiment.name=grpo_m1_<cell>`),
deliberately discontinuous with v1–v13: v-numbers are the old stack's
debugging arms; m-numbers are grids of the redesigned stack. m1 = the first
canonical grid on fiction10-gemma4.

## Cell mechanics

A cell is fully defined by two config keys; **nothing else varies across the
grid** (optimizer preset, data, splits, prescreen policy, servers, seeds all
pinned):

```yaml
training:
  grpo:
    reward_auxiliaries: [ground, contrast]    # subset of {ground, contrast}
    task_mix: {extract: 0.7, vignette: 0.3}   # −vignette: {extract: 1.0, vignette: 0.0}
```

Changing either key bumps the prescreen cache signature automatically (both
are part of the cache key). Cell configs live in one sweep yaml
(`conf/sweep/grpo_m1_grid.yaml` — written 2026-07-24).

**Recorded deviation (2026-07-24):** the `−outcome` cell requires a third
toggle, `training.grpo.reward_core: false` — R-OUTCOME is core, not a
member of `reward_auxiliaries`, so no setting of the two cell keys can
remove it. `reward_core: true` for every other cell; this is the single
exception to "two keys define a cell," flagged in the grid file header.

## Weights — resolved (master decision 4)

One fixed ratio, no tuning: **the outcome core weighs 2× each active
auxiliary**, auxiliaries equal among themselves, weights normalized to sum 1.

| active modules | outcome | ground | contrast |
|---|---|---|---|
| full | 0.50 | 0.25 | 0.25 |
| −ground | 0.67 | — | 0.33 |
| −contrast | 0.67 | 0.33 | — |
| core | 1.00 | — | — |
| −outcome | — | 0.50 | 0.50 |

Rationale: the verifiable signal must dominate wherever it is present
(principle 3; Memory-R1's Table 2), and a single arithmetic rule keeps the
grid free of a weight axis — the LOO cells, not weight tuning, carry the
evidence. If a reviewer asks "why 2:1", the answer is "so that no judge
opinion can outvote the verifiable outcome, and the exact value is not
load-bearing — see the `core` cell for the 1:0 extreme."

## The grid (7 cells + 2 free baselines)

Run on the canonical policy base — **qwen3.5-9b `sft-canonical`** (the
2026-07-15 canonical DFT sweep on fiction10-gemma4 flows; the earlier
"SFT-contentless" naming here predated that sweep — corrected 2026-07-24).
Every cell gets the identical eval matrix. Execution sequencing, model
scope, and readiness gating live in [m1-run-plan.md](m1-run-plan.md).

| cell | reward_auxiliaries | vignette mix | pre-registered prediction (falsifiable) |
|---|---|---|---|
| `core` | [] | 0.3 | `cannot_determine_frac` falls through training; GoldCoin Forbid recall moves off the 0.55 plateau toward SFT 0.65 with Permit recall/applicability held. **Kill:** cannot_determine plateaus >0.5 with gold-no EM flat (reward-outcome.md). |
| `full` | [ground, contrast] | 0.3 | ≥ `core` on every benchmark; strictly better on at least one (else auxiliaries are dead weight). |
| `−outcome` | [ground, contrast] | 0.3 | **The Table 2 replication, adverse direction:** judge-only content reproduces the v10/v11 pathology — extraction hedging stays high, Forbid recall pinned. If instead it matches `full`, outcome supervision is NOT the active ingredient and the redesign's core claim fails. |
| `−ground` | [contrast] | 0.3 | ≈ `full` within seed noise (Memory-R1 predicts the quality judge is redundant given outcome). Any consistent drop = the judge carries unique signal worth its cost — report either way. |
| `−contrast` | [ground] | 0.3 | Benchmarks ≈ `full`; trace diagnostic moves instead: wrong-book grounding score rises (book-specificity erodes without the penalty). |
| `−vignette` | [ground, contrast] | 0.0 | Judgment-format benchmarks (ConfAIde tier-2, CIRL) degrade vs `full` or v10-style drift re-opens; extraction metrics hold (task-vignettes.md — removal expected to hurt). |
| `+answerer-B` | (offline) | — | Not a training cell: re-score a sample of `full`'s reward traces with a second answerer (**a non-Gemma model, e.g. Qwen3.6-27B** — revised 2026-07-24 with D1: the primary answerer is now gemma-4-31b, so the robustness check must cross families); Spearman vs primary ≥ 0.8, else the outcome reward is answerer-idiosyncratic (reward-outcome.md D1 check). |
| `sft` | — | — | baseline row, no training cost |
| `0-shot` | — | — | baseline row, no training cost |

## Run order and early-exit logic

1. **`core` and `full` first** — they bracket the design. Gates + benchmarks
   on both before anything else.
2. **Branch on the bracket:**
   - `core` ≈ `full` everywhere → auxiliaries add nothing; **skip `−ground`
     and `−contrast`** (their answer is implied), run only `−vignette` and
     `−outcome`. Paper reports the simplicity result.
   - `full` > `core` → run all LOO cells to attribute the gap.
   - `core` fails its kill criterion → stop the grid; the redesign core needs
     rethinking before more compute (honest failure, documented).
3. `−outcome` runs in either branch — it is the control that shows outcome
   supervision is load-bearing, the paper's methodological claim.

## Seeds and noise

The canonical cell (`full`) quotes the existing 5-seed variance measurement
(final-reward CV 3.5%, the reviewer-rebuttal protocol) as its noise bar; LOO
cells run **one seed** and are flagged as such in the reporting table. Two
cells differing by less than the 5-seed CV on a benchmark are reported as
tied. If compute allows after the grid, `core` gets 2 extra seeds (it is the
cell most likely to be quoted alone).

## Gates before eval spend (house rule, unchanged)

`scripts/check_grpo_promotion_gates.py` must pass on a cell's training run
before any benchmark eval is launched for it. A cell that fails gates is
reported as failed-at-gates, not silently rerun.

## Eval matrix (identical for every cell)

1. **Benchmarks (zero-shot):** GoldCoin-HIPAA (applicability + compliance
   Forbid/Permit recall), PrivacyLens, ConfAIde (tiers 2a/2b), CIRL.
2. **Held-out books:** extraction + probe-EM on the 93 top100-only books
   (data.md) — the generalization column no benchmark provides.
3. **Trace forensics:** `reward/outcome/*` (incl. `cannot_determine_frac`,
   `em_mean_by_force/no`), `vignette/*` (incl. `antithesis_frac`),
   `diag/direction_consistency`, realized-mix fields. Forensics are
   per-cell appendix material; benchmarks are the main table.

## Reporting table (the one table a reviewer reads)

| cell | GoldCoin appl. | GoldCoin Forbid/Permit | PrivacyLens | ConfAIde-2 | CIRL | held-out probe-EM |
|---|---|---|---|---|---|---|
| 0-shot / sft / core / full / −outcome / −ground / −contrast / −vignette | … | | | | | |

Ties within the 5-seed CV marked as ties; single-seed cells footnoted;
`+answerer-B` reported as a Spearman in the caption, not a row.

## Pre-registration discipline

Each cell's row above **is** its pre-registration; the table is frozen in
this file before the first m1 launch, and any post-hoc deviation (new cell,
changed prediction, extra seed) is recorded here with a date. This replaces
the per-arm field-note plans of the v-era with one document — the grid is
the plan.

## Compute envelope (rough, for scheduling)

Per training cell: ~1 GPU-day (1× A6000 policy training; the m-series aux
load is the answerer + judge on the shared server — `core` needs answerer
only). 6 training cells ≈ one GPU-week plus the shared servers, evals on the
usual eval launchers. The universe-build prerequisite chains
(`scripts/run_universe_build_gemma4.sh`, launched 2026-07-16) gate the start;
the null-answerability calibration pass runs between universe completion and
the first cell.
