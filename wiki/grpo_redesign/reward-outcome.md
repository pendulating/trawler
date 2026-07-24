# `R-OUTCOME` — outcome-grounded extraction reward (core)

**Parent:** [README.md](README.md) · **Date:** 2026-07-16 · **Status:** drafted
· **Kind:** core (always on; not an ablation axis — the `−outcome` grid cell
removes it *from `full`* to measure its contribution, but no proposed keeper
config omits it)

## One-liner

A frozen answerer must answer K probe questions about information-flow
appropriateness **given only the completion's structured extraction** — no
source text, no norms; the reward is mean exact-match against gold answers
derived from the governing norms' deontic force.

This is Memory-R1's Memory-Manager reward transplanted: structured artifact →
frozen downstream consumer → EM. The universe supervises through verifiable
consequences, not judged appearances.

## Interface

- **Input:** completion's extraction (structured fields only — see "What the
  answerer sees"), the chunk's pre-built probe set.
- **Output:** scalar in [0,1] (mean EM over K probes).
- **Applies to:** `T-EXTRACT` rows with a flow-bearing completion. No-flow
  declarations route to `A-ABSTAIN` and never reach the answerer.
- **Infra:** one batched HTTP call per completion to the frozen answerer
  (reuses the judge server plumbing, `dagspaces/common/judge_client.py`).

## Probe generation (build-time, per prompt — not per completion)

The v9-lineage direction machinery resolves the governing norm **per flow,
post-completion** (`deontic.governing_norm_force`: top-1 cosine over the
completion's flow text). Probes must instead be a **property of the prompt**,
fixed before any generation, so all G completions in a group face identical
questions — otherwise within-group advantages compare answers to different
tests.

**Query anchor: reference flows, not chunk text** (design revision
2026-07-16). Retrieving norms against the raw chunk text was considered and
rejected: whole-chunk embeddings match norms about the chunk's *themes*, not
the norms governing the *information flows in it*, so probes would test
judgments about flows that may not occur in the passage. Instead, retrieval
is anchored to the chunk's **reference flows** — the Gemma-4 teacher
extraction (`ci_extraction/ci_flows.parquet`), the same artifact SFT
supervises on. One supervision source anchors both SFT and the outcome
probes; and flow-text queries are exactly the query distribution the v9
`NormRetriever` has used per-flow since v8 (validated in traces), just moved
from post-completion to build time.

1. At dataset build, for **each reference flow** in the chunk, retrieve the
   top-k norms from the chunk's own book universe (same Qwen3-Embedding-8B
   retriever and k ≈ 3 that `R-GROUND` uses; query = the flow's rendered
   text, the existing per-flow format).
2. **Union over all the chunk's reference flows**, dedupe by norm identity;
   filter to `governs_info_flow: true` with decisive force (`FORCE_TO_GOLD`
   in `deontic.py`: obligatory/recommended → **yes**, prohibited/discouraged
   → **no**; `permitted` skipped — the single source of truth shared with
   `T-VIGNETTE`). This union is the chunk's probe pool.
3. **Force-stratified sample of K** (decision resolved below: K = min(4,
   pool), at least one gold-**no** probe whenever the pool has one),
   deterministic RNG seeded by `chunk_id` — probes are reproducible and
   enter the prescreen cache signature.
4. Each sampled norm becomes a probe via the **same template as the vignette
   builder** (`_generate_vignettes`, `stages/grpo_training.py:28`): scenario
   from the norm's fields (context, subject, condition/act), articulation
   withheld. One mechanism, two uses (training task / measurement instrument).
5. **Null-answerability filter (calibration step):** ask the frozen answerer
   each candidate probe with an **empty extraction**. Probes it already
   answers correctly ≥ p_null (default 0.8 over a small vote) carry no signal
   about the extraction — drop them before sampling. This is the analog of
   Memory-R1 rewarding *memory-dependent* QA: their questions are
   unanswerable without the memory bank; ours must be unanswerable without
   the extraction. Report the drop rate in `training_metadata.json`.

Routing consequences, all inherited rather than invented: chunks with **no
reference flows** are the gold-no-exchange chunks — they never bear probes
and are exactly the rows `A-ABSTAIN` exists for. Chunks *with* reference
flows whose filtered probe pool is empty are **excluded from the `T-EXTRACT`
prompt set at build time** (no neutral-reward tier to explain). Prior: v8
traces found ~97% of flows had a directional governing norm — and that stat
was measured on flow-text queries, i.e. the same query type used here — so
expected exclusion is a few percent; the realized count is reported
(principle 6).

**Known bias, accepted:** the probe pool inherits the teacher's flow
coverage — a real flow the teacher missed generates no probe, and a policy
that extracts it earns no outcome credit for it. This is the *same* coverage
bias SFT already has (it trains on the same teacher flows), so the reward
does not introduce a new bias axis; it declines to correct an existing one.
Noted for the limitations paragraph.

## What the answerer sees

The completion's **structured fields only**: the flow tuples
(subject/sender/recipient/information_type/transmission_principle/**context**
— added by decision 2026-07-24: flows and norms are context-relative, so the
answerer cannot judge a flow without the extraction's context field; it is a
bounded structured field, covered by `R-VALID`'s length caps like the rest)
and any per-flow appropriateness labels. **Not** the free-text reasoning
trace, and never the source chunk. Rationale:

- The artifact under evaluation is the structured extraction (as Memory-R1's
  answer agent reads memory *entries*, not dialogue transcripts).
- Free text is the smuggling channel: a policy could paste the whole chunk
  into its reasoning and let the answerer read the answer off the source.
  Structured-fields-only + `R-VALID`'s per-field length caps close it.

Call shape — one batched request per completion:

```
system: You answer questions using ONLY the structured information-flow
        extraction provided. If the extraction does not determine an answer,
        reply "cannot_determine".
user:   EXTRACTION: {flows: [...]}
        Q1: <probe 1>  Q2: <probe 2> ... QK: <probe K>
        Reply as JSON: {"answers": ["yes"|"no"|"cannot_determine", ...]}
```

Scoring: `EM = 1` iff the answer string equals gold; `cannot_determine`
scores **0**. That zero is the module's tooth: an extraction too hedged or
too empty to determine an answer is *priced identically to a wrong one* —
this is where the 72%-hedge equilibrium is attacked at the incentive root,
with no tier ladder (see "Subsumption" below).

`R_outcome = mean(EM over the K probes)`.

## Failure handling

Same convention as the listwise judge (deliberate-neutrality, never noise):
if the answerer's reply fails to parse after one retry, the **whole group**
gets uniform 0.5 for this term — zero advantage — and
`reward/outcome/answerer_failed_frac` streams to W&B with a stdout WARNING.
Per-completion (not per-group) failures are not given special treatment;
a completion whose extraction provokes unparseable answerer output has, in
practice, produced a degenerate extraction — but we do not *rely* on that:
the retry + group-neutral fallback keeps failures out of the gradient.

## Anti-gaming analysis

| Attack | Defense |
|---|---|
| Probe leaks gold ("should he *not* share…") | Template never includes the norm articulation or force word — inherited property of the vignette builder, verified by the existing no-leak test pattern (`tests/`) |
| Blanket labels ("everything inappropriate") | Force-stratified probes: blanket-no aces prohibited probes and zeroes obligatory ones — EV ≤ base rate by construction |
| Content stuffing (chunk text pasted into fields) | Answerer sees structured fields only; `R-VALID` caps field lengths; `reward/outcome/extraction_token_len` streamed as a drift diagnostic |
| Answerer answers from world knowledge, extraction irrelevant | Null-answerability filter (build step 5) removes exactly these probes |
| Parroting appropriateness labels into flows | Not an attack — a correct appropriateness label that lets the answerer answer correctly *is* the direction signal, now earned through an outcome |

## Resolved design decisions

**D1 — frozen answerer: Gemma-4-31B-it** (REVISED 2026-07-23; the original
resolution named Qwen3.6-27B and was stale against the canonical-model
decision — [canonical-models.md](../canonical-models.md): **Gemma-4 is the
gold-label family**, `gemma-4-31b/instruct` is the teacher AND the judge for
the camera-ready, and the paper already credits Gemma-4-31B-it on every
judged column). One family supervises everything: teacher flows, judged
auxiliaries, and the outcome answerer — no cross-family drift inside the
reward definition. Memory-R1's precedent removes the same-family anxiety:
their frozen Answer Agent *is* the same backbone as the trained Memory
Manager — what matters is that the counterpart is frozen and capable, not
foreign. Two hard rules: (a) the answerer is **identical across every grid
cell and every policy model** (it is part of the reward definition, like the
judge); (b) a robustness spot-check with a second answerer — now a
**non-Gemma** model (e.g. Qwen3.6-27B) since the primary is Gemma — on a
sample of traces is run once, offline, before the grid is trusted — reported
in [ablation-protocol.md](ablation-protocol.md).

**D2 — K = min(4, |filtered pool|), force-stratified, seeded by `chunk_id`.**
K=4 keeps the call short (~600 in / ~40 out tokens) while making blanket
strategies EV-negative (a single gold-no probe among gold-yes probes suffices,
and vice versa). The stratification guarantee: if the pool contains both gold
classes, the sample contains both. Larger K raises cost linearly for
diminishing discrimination; revisit only with trace evidence.

## Cost model

Per group per generation step: G = 8 answerer calls (vs 2 listwise-judge
calls in the v9 lineage), but each call is ~10× smaller — the listwise call
carries all 8 full completions plus retrieved norms (~5–6k tokens in); the
outcome call carries one extraction's structured fields plus 4 one-line
probes. Rough per-group tokens: outcome ≈ 8·(600+40) ≈ 5.1k vs listwise ≈
2·(5500+300) ≈ 11.6k. **The core is cheaper per group than the judge it
displaces**; a `full`-stack cell (outcome + ground + contrast) costs roughly
the old stack + ~45%. The null-answerability calibration is a one-time
build-time pass over the probe pool (≈ N·M short calls, cacheable).

## Subsumption of `m-DIRECTION` (and its tiers)

| v9–v12a mechanism | Outcome equivalent |
|---|---|
| correct verdict → ×1.0 | correct probe answers → EM 1 |
| wrong verdict → ×0.4 (v10 false-permit floor ×0.1) | wrong probe answer → EM 0 — already maximal, no tail-pricing knob needed |
| hedge → ×0.7 (v12a: ×0.5 on prohibited) | `cannot_determine` → EM 0 — hedging is priced at the floor *everywhere*, not only on prohibited flows, with zero configuration |
| tier ladder needs 3 knobs + a formula-version bump each | no knobs |

The graded tiers existed because a *process* multiplier had to guess how bad
each failure mode was. An *outcome* score does not guess — an extraction
either supports the right decision or it does not. Direction agreement
(`appropriateness_consistency`) survives as a free W&B **diagnostic**
(`diag/direction_consistency`), giving continuity with v8–v12 forensics
without being a reward term.

## Diagnostics (W&B namespace `reward/outcome/*`)

- `em_mean`, `em_mean_by_force/{yes,no}` — watch the gold-no line: it is the
  Forbid-recall proxy that was pinned at 0.55 for three arms
- `group_spread` — within-group std of the term (the advantage carrier; the
  v1-era failure mode was zero spread)
- `cannot_determine_frac` — the hedge-mass successor metric
- `answerer_failed_frac`, `null_filter_drop_rate`, `extraction_token_len`

**Streaming note (2026-07-24):** `extraction_token_len` streams to W&B (the
drift diagnostic in the anti-gaming table — added to the code now).
`null_filter_drop_rate` does **not** stream to W&B: the null-answerability
filter is a no-op online (0.0% drop, verified in the 2026-07-24 calibration
pass — see [data.md](data.md)), so it is recorded once in
`training_metadata.json` at build time rather than per-step to W&B.

## Pre-registered predictions for the `core` cell (draft — finalize in ablation-protocol.md)

1. `group_spread` > 0 in ≥90% of groups from step 1 (probes discriminate
   where the absolute judge tied 60% of groups).
2. `cannot_determine_frac` **falls** over training — the first reward in the
   lineage where hedging is dominated for every force class. If it plateaus
   above ~0.5 with `em_mean_by_force/no` flat, outcome supervision has failed
   its central promise and the redesign's core needs rethinking — that is the
   honest kill criterion.
3. Held-out: GoldCoin Forbid recall moves off 0.55 toward SFT's 0.65 without
   the v8 indiscriminate-permit mirror (Permit recall and applicability hold).
