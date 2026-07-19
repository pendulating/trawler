# `A-ABSTAIN` — gold-aware abstention routing (core)

**Parent:** [README.md](README.md) · **Date:** 2026-07-16 · **Status:** drafted
· **Kind:** core routing rule. **Not ablatable, by construction:** removal has
no identity element — *some* rule must define the reward for rows where the
outcome term is undefined, so "no abstention rule" is not a cell, it's an
unspecified reward. (Contrast the auxiliaries, whose identity is deletion.)

## One-liner

Rows where outcome probes cannot run — no-flow declarations, and every
completion on a gold-NO chunk — are scored by a fixed four-entry table keyed
on (gold label, declared behavior); no server calls, no judge, no knobs.

## The routing table

`gold` = the chunk's `has_information_exchange` from the teacher reasoning
parquet; `no-flow` = the completion declares no information exchange
(schema-valid, zero flows).

| gold | completion | score | reads as |
|---|---|---|---|
| YES | extraction | — | **normal path**: `gate(valid) · [outcome + auxiliaries]` |
| YES | no-flow | **0.1** | wrong abstention — a flow exists and the policy declined to look |
| NO | no-flow | **0.6** | correct abstention |
| NO | extraction | **0.4** | unverifiable engagement — neutral, see below |
| unknown / unparsable gold | anything | 0.4 | neutral; zero group spread ⇒ the stratified prescreen drops these rows naturally |

Constants are the v9 lineage's `NO_FLOW_REWARD_{WRONG,CORRECT,UNKNOWN}` =
0.1 / 0.6 / 0.4, kept deliberately: they have three arms of run history, and
what matters is their *ordering and gaps* against typical outcome scores, not
their absolute values (with `scale_rewards: none`, advantage = r − group
mean, so gaps are the gradient).

## The within-group economics (why these numbers work)

- **Gold-YES mixed group** (some completions extract, some abstain):
  extraction earns an outcome score (realistically 0.4–0.9 with auxiliaries)
  vs 0.1 for abstaining → a large, consistent advantage for engaging. This is
  the anti-over-abstention pressure the v1–v8 saga spent seven arms trying to
  produce, now a corollary of the table.
- **Gold-NO mixed group**: correct abstention 0.6 vs unverifiable extraction
  0.4 → a modest advantage for abstaining. Two-sided by construction: the
  policy is pushed to engage exactly where gold says there is something to
  find, and to decline where gold says there is not.
- **Why wrong abstention is 0.1, not 0:** 0 is the invalid-output gate. A
  schema-valid no-flow declaration on a gold-YES chunk is wrong but
  well-formed; keeping it strictly above the gate preserves the ordering
  *invalid < wrong-but-valid < everything else*, so the gate never becomes
  preferable to an honest wrong answer.
- **Why correct abstention is 0.6, not 1.0:** on a gold-NO chunk 0.6 is
  already the top of the table — raising it buys nothing within-group and
  invites abstention-mode drift on any gold-mislabeled chunk. The 0.2 gap
  over neutral is the signal; the ceiling is intentional humility about the
  gold label (next section).

## Gold-NO extractions: neutral, and why (resolves master decision 5)

Two reasons an extraction on a gold-NO chunk scores a flat 0.4 rather than
being judged or penalized:

1. **The label is miss-prone in exactly this direction.** The historical
   audit (`scripts/audit_goldno_labels.py`) found many gold-NO chunks contain
   real flows — gold-NO means *the teacher found nothing*, not *nothing is
   there*. Penalizing engagement on gold-NO punishes the policy for
   out-extracting its teacher. (This is why the v6-era symmetric
   false-extraction penalty was rejected, and the reasoning still holds.)
2. **Nothing can verify it.** Gold-NO chunks have no reference flows, hence
   no probes; letting the judge auxiliaries score these rows would make the
   *reward formula depend on the row type* (aux-only for gold-NO extractions,
   outcome-weighted elsewhere) — a legibility cost with no verifiable signal
   behind it. So gold-NO chunks are scored **entirely** by this table, for
   every completion, with zero server calls.

**Build-time check (added to the data.md job list):** the old audit ran on
the qwen-era label, which derived from the norms corpus; the fiction10-gemma4
label is the Gemma-4 teacher's `has_information_exchange` from the *flows*
pipeline — plausibly a much better flow label (126 gold-NO of 2,993 chunks,
4.2%). Re-run the audit on the new corpus before m1. If gold-NO turns out
trustworthy (<~10% of audited gold-NO chunks contain real flows), a mild
penalty for gold-NO extraction becomes defensible — but that is an m2
consideration, not an m1 knob. m1 ships the neutral table.

## Why wrong abstentions are not scored through the outcome path

An empty extraction would score ≈ 0 on probes anyway — the null-answerability
filter guarantees probes are unanswerable without extraction content — so
routing no-flow completions to the answerer would spend G answerer calls per
abstention-heavy group to compute a constant. The table short-circuits what
the outcome path would conclude, at zero cost, and keeps the 0.1 floor
deliberate rather than emergent.

## What changed vs v9

| | v9–v12a | redesign |
|---|---|---|
| no-flow completions | gold-aware `abstention_score` bypass (same constants) | same table (kept) |
| extractions on gold-NO chunks | full composite ran anyway (judge scored unverifiable flows) | constant 0.4, no calls |
| post-hoc `abstention_penalty` knob | existed (0.0 since v9) | does not exist |
| no-flow R_ground participation | version-dependent (ranked-last convention) | n/a — abstentions never reach any scored module |

## Diagnostics

- `abstain/no_flow_rate` vs the gold base rate — the old promotion gate (d)
  (`|tail no-flow − gold_base_rate| ≤ 0.15`) carries over unchanged.
- `abstain/wrong_abstention_frac` (no-flow on gold-YES) — should fall.
- `abstain/goldno_extraction_frac` — watched, not rewarded/punished; a large
  rise flags either teacher misses (good) or flow invention (bad), and only
  the audit distinguishes them.
- Realized gold-NO share of the prompt set (stratified prescreen keeps a
  floor so the two-sided signal exists; reported per principle 6).
