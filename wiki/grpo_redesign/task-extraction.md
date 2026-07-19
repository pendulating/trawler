# `T-EXTRACT` — CI flow extraction task (core)

**Parent:** [README.md](README.md) · **Date:** 2026-07-16 · **Status:** drafted
· **Kind:** the core task; always on. `task_mix.extract` is the complement of
the vignette mix, not an independent knob.

## One-liner

The policy reads a fiction chunk and emits a structured extraction of its
information flows under the Contextual Integrity framework — the artifact
every reward module scores.

## Prompt and schema (unchanged from the SFT stage — deliberately)

The prompt is `conf/prompt/ci_extraction.yaml`, **the same template used by
SFT data prep for the teacher's gold completions**. This identity is a
property, not laziness: the GRPO policy is trained on the distribution it was
SFT'd on, and the probe anchor (teacher flows) was produced by the same
instruction. Changing the extraction prompt would silently invalidate all
three at once — it is frozen per m-series.

Completion schema (top level): `reasoning` (free text),
`has_information_exchange` (bool), `flows` (list). Per flow:

| field | type | scored by |
|---|---|---|
| `sender`, `recipient`, `subject`, `information_type`, `transmission_principle` | the five CI core fields | `R-VALID` gate (presence); `R-OUTCOME` (content, via the answerer); `R-GROUND`/`R-CONTRAST` (via retrieval + judge) |
| `context` | free text | answerer input; no direct score (legacy `R-CONTEXT` did not migrate) |
| `appropriateness` | appropriate/inappropriate/ambiguous | answerer input — an *earned* direction channel (a correct label that helps the answerer answer correctly is the direction signal working); also the `diag/direction_consistency` metric |
| `norms_invoked`, `norm_source` | list / enum | answerer input; `R-GROUND`'s norm-awareness criterion |
| `is_new_flow` | bool | unscored, kept for schema stability |
| `confidence` | 1–10 | **unscored in the m-series** (see reward-valid.md — the confidence-scoring lineage does not migrate) |

## What each consumer sees

- **The answerer (`R-OUTCOME`):** the `flows` array verbatim — every per-flow
  field — and **never** the top-level `reasoning`. Free-text reasoning is the
  content-smuggling channel (paste the chunk, let the answerer read the
  source); structured fields are capped by the gate. Per-flow fields that
  carry the policy's own judgments (`appropriateness`, `norms_invoked`) stay
  in: parroting them into flows is not a hack, it is the behavior being
  trained, earned through an outcome.
- **The judge (`R-GROUND`/`R-CONTRAST`):** the full completion plus retrieved
  norms (the judge evaluates grounding of the whole extraction, reasoning
  included — its rubric is about the flows' governance, and the reasoning
  gives it context; the judge is an auxiliary, so this channel is bounded).
- **The gate (`R-VALID`):** structure only.

## Row lifecycle

1. Chunk selected at build (stratified prescreen from the fiction10-gemma4
   pool; gold-NO floor per [reward-abstain.md](reward-abstain.md)).
2. Probes attached at build ([reward-outcome.md](reward-outcome.md)); chunks
   with reference flows but empty probe pools are excluded.
3. At each generation step: G completions → `R-VALID` gate → routing
   (`A-ABSTAIN` for no-flow rows and all gold-NO rows) → scored path
   `gate · [outcome + auxiliaries]`.

## Gold signals available per chunk (summary)

| signal | source | consumed by |
|---|---|---|
| `has_information_exchange` (gold) | teacher reasoning parquet | `A-ABSTAIN` routing |
| reference flows | teacher `ci_flows.parquet` | probe retrieval anchor |
| probes + gold answers | build-time (norms × `FORCE_TO_GOLD`) | `R-OUTCOME` |
| own-book universe | universe build | `R-GROUND`, probe pool |
| wrong-book universe (seeded) | universe build | `R-CONTRAST` |
