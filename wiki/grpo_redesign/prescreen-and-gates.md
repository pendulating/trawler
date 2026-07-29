# Stratified prescreen and promotion gates

**Parent:** [README.md](README.md) · **Date:** 2026-07-16 · **Status:** drafted
· **Kind:** infrastructure, identical across cells.

## Prescreen — what it is for

GRPO learns only from **within-group reward spread**; a prompt where the SFT
base's G samples all score the same is a dead row at step 0. The prescreen
(Phase 2–5 machinery, kept) samples G completions per candidate prompt with
the SFT base, scores them with the *cell's own reward*, and keeps prompts
with spread. The m-series changes one thing and formalizes another:

1. **Stratification (the v10 fix, now structural).** The legacy screen was
   variance-only and force-blind — it silently doubled the vignette force
   skew (pool 3.07:1 → realized 5.2:1) because low-variance strata were
   dropped wholesale. The m-series screen selects **within strata**:
   `(task type × gold class)` for extraction rows (gold-YES / gold-NO), and
   `(battery composition class)` for vignette rows. Stratum floors come from
   the configured mix (`task_mix`, gold-NO floor, battery minority share);
   variance ranks candidates *within* a stratum, never across.
2. **Pre-registered N.** The prompt-set size (N ≈ 500–800, final value fixed
   in ablation-protocol.md before m1) is a declared constant, not "whatever
   survived." Memory-R1's 152-example result is the argument that small,
   well-chosen N is a feature. Realized composition per stratum is reported
   in `training_metadata.json` / `prescreen_report.json` (principle 6).

## Cache-key contract

Prescreen scoring is expensive (it runs the cell's reward, including answerer
and judge calls) and cached. The signature must include everything that
changes a score or the candidate set:

- `formula_version` = `m1` (new namespace; no collision with the v-era keys),
- the **module list** (`reward_auxiliaries`) and `task_mix`,
- the weight rule output (derived, but hashed anyway — belt and braces),
- probe sets (per-chunk probe ids + gold), battery compositions, and their
  seeds,
- answerer/judge identity and prompt hashes,
- the routing-table constants.

Consequence: every grid cell automatically gets its own cache entries; a config typo
that would silently reuse a neighbor cell's screen instead misses cleanly.
Per-cell `cache_path` files (the v-era convention) are no longer load-bearing
but remain the convention for tidiness.

## Promotion gates (UPDATED 2026-07-28 — m1 gave them teeth)

`scripts/check_grpo_promotion_gates.py` runs on the training output **before
any benchmark eval is spent** (house rule; see ablation-protocol.md). The m1
wave exposed two holes: `min_reward_gain=0.0` promoted four flat cells
(core's gain: +0.0027), and no gate consulted discrimination — core held
reward 0.73 at the blanket floor (balanced accuracy 0.56) and promoted.

| gate | criterion | note |
|---|---|---|
| `reward_trend` | last-third − first-third gain **> 0.02** (was 0.0) | 0.02 sits above the m1 per-bin wobble; a run must beat launch noise |
| `zero_std` | groups carry spread | prescreen guarantees this at step 0; the gate catches mid-run collapse |
| `kl_bounded` | KL to reference bounded | trivial at β=0.02 |
| `no_flow_rate` | `\|tail no-flow − gold_base_rate\| ≤ 0.15` | the abstention-drift alarm. **Was DEAD on every modular run** (keeper-only trace keys; m1's gates all show it skipped) — fixed 2026-07-28 to read both schemas |
| `direct_discrimination` **(new)** | **LABEL-only** pooled Youden's J over the trace tail (last 100 calls' `direct_flows`, matched flows only) **≥ 0.05** | disk-only (W&B-crash-proof). Misses are REPORTED (`miss_frac`) not gated — a recall-priced J has blanket floor m−1 ≈ −0.23 at launch and would fail every cell. A stale direct tail (core silently stopped scoring) FAILS loudly. Skips on cells without a direct core (−outcome) |

**Caveat (audit 2026-07-28): gate verdicts are RUN-scoped.** `_find_trainer_state`
reads the highest checkpoint and the trace gates read the END of the traces —
if the protocol promotes an early checkpoint (the v9-ckpt100 precedent), the
discrimination verdict describes a later policy. Historical note: the v9
keeper's recorded verdict ("PROMOTE on +0.015") predates the 0.02 threshold —
re-running the script on it now yields HOLD; the realized thresholds are
recorded inside each `promotion_gates.json`.

New m-series signals (`cannot_determine_frac`, `antithesis_frac`,
`gate_fail_frac`) are **per-cell kill criteria** (pre-registered in
ablation-protocol.md), not global gates — a gate is a property every healthy
run must have; a kill criterion is a bet a specific cell made.
