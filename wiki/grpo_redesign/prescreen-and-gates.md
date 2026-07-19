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

## Promotion gates (carried over unchanged)

`scripts/check_grpo_promotion_gates.py` runs on the training output **before
any benchmark eval is spent** (house rule; see ablation-protocol.md):

| gate | criterion | note |
|---|---|---|
| `reward_trend` | train (or dev, when `dev_fraction>0`) reward not flat/declining | |
| `zero_std` | groups carry spread | prescreen guarantees this at step 0; the gate catches mid-run collapse |
| `kl_bounded` | KL to reference bounded | trivial at β=0.02 |
| `no_flow_rate` | `\|tail no-flow − gold_base_rate\| ≤ 0.15` | the abstention-drift alarm; gold base rate now comes from the stratified set, so the target is exact by construction |

New m-series signals (`cannot_determine_frac`, `antithesis_frac`,
`gate_fail_frac`) are **per-cell kill criteria** (pre-registered in
ablation-protocol.md), not global gates — a gate is a property every healthy
run must have; a kill criterion is a bet a specific cell made.
