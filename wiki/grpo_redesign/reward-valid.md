# `R-VALID` — validity gate (core)

**Parent:** [README.md](README.md) · **Date:** 2026-07-16 · **Status:** drafted
· **Kind:** core binary gate; always on. Not an ablation axis (a reward over
unparseable output is undefined — no identity element).

## One-liner

A completion scores zero unless it parses, carries the schema, is internally
consistent, and keeps its fields within length caps; otherwise the gate
passes and the scored path runs. Binary — there is no partial credit for
being almost-parseable.

## The criteria (all must hold)

1. **Parses**: a single JSON object (after `<think>`-strip), no trailing
   prose.
2. **Schema**: top-level `reasoning`, `has_information_exchange` (bool),
   `flows` (list) present.
3. **Consistency**: `has_information_exchange == (len(flows) > 0)`. (A "no
   exchange" claim with flows attached, or vice versa, is contradictory
   output, not a judgment call.)
4. **Core fields**: every flow has non-empty `sender`, `recipient`,
   `subject`, `information_type`, `transmission_principle`. Optional fields
   (`norms_invoked`, `confidence`, …) are not gated.
5. **Field caps**: no per-flow field exceeds the token cap (set at
   implementation, order-of-64 tokens/field) — the anti-content-stuffing
   guard from [reward-outcome.md](reward-outcome.md); violations counted, not
   truncated.

Pass → route per [`A-ABSTAIN`](reward-abstain.md) / the scored path.
Fail → R = 0, beneath every entry in the abstention table (the ordering
*invalid < wrong-but-valid* is deliberate).

## What was merged, and what was dropped (the 0.20 that became a gate)

Three legacy graded components (combined weight 0.20, all flagged "saturated
post-SFT" since the v9-era docs) collapse into this gate:

| legacy component (weight) | disposition |
|---|---|
| `r_uncert` (0.10) — schema validity facet | → criteria 1–2 |
| `r_uncert` — construct-discrimination facet | **dropped** — graded heuristic, no isolated evidence |
| `r_uncert` — confidence facet | **dropped entirely** — see below |
| `r_complete` (0.05) — proportion of non-null substantive fields | → criterion 4, binary on the five core fields |
| `r_consist` (0.05) — reasoning ↔ extraction non-contradiction | → criterion 3 (the checkable kernel; the fuzzy text-overlap remainder dropped) |

**Why binary is safe:** saturated components contribute ~zero within-group
variance — GRPO's advantage never saw their graded structure anyway. A gate
preserves their only real function (protecting downstream modules from
garbage) at zero explanation cost.

**The confidence lineage ends here.** The 2026-07-14 review found `r_uncert`'s
confidence facet had an unreachable documented fallback, and the
`confidence_fallthrough` knob exists solely so the old stack can reproduce
v9-ckpt100 bit-for-bit versus run the corrected semantics. The m-series
resolves that fork by **not scoring confidence at all**: the field stays in
the schema (SFT continuity), no reward reads it, and the knob has no m-series
counterpart. One less thing to explain, one less thing to get wrong.

## Diagnostics (`reward/valid/*`)

`gate_fail_frac` (should be ≈0 on an SFT'd policy — a rise is a policy-
degeneration alarm, the earliest one available), per-criterion failure
counts, `field_cap_violation_frac`.
