# `R-OUTCOME` v2 — PROPOSED spec diff (flow-anchored, norm-classified)

**Parent:** [README.md](README.md) · **Date:** 2026-07-25 · **Status:**
PROPOSAL — for Matt's review. Supersedes nothing until approved;
[reward-outcome.md](reward-outcome.md) remains the active spec. No code has
been changed against this document.

## Why v1 has to be replaced

The m1 wave was killed 2026-07-25 after the outcome core was shown to be
inert. Three *independent* defects, each measured:

1. **Contentless probes (template).** The shared scenario template preferred
   `condition_of_application` over `norm_act`
   (`if condition: "is in a situation where {condition}"`). On this corpus
   **100% of eligible norms have an act and 99.1% also have a condition**, so
   the condition branch fired for **99.2% of the 11,218 built probes** and the
   act — the thing being judged — was silently dropped. The result is
   ungrammatical and contentless: *"a wife is in a situation where unless the
   husband authorizes such a disclosure. Should this information be shared?"*
   No act, no information type, no sender or recipient. Measured: 44.8%
   `cannot_determine` on these versus **0% on the act-branch form**, even when
   the answerer was shown the *teacher's own* reference flows. The answerer was
   right to refuse.
2. **Wrong prescriptive object (conceptual).** v1 applies `FORCE_TO_GOLD`
   (yes/no "should this be shared?") to a **norm-derived hypothetical**. CI
   formulates prescription as **appropriateness of a flow**; the Raz force
   gradient is the *input* to that classification, not the question. The norm
   must **classify the extracted flow**, not supply the scenario.
   `deontic.FORCE_TO_APPROPRIATENESS` already exists for exactly this and is
   unused by the probe path.
3. **No correspondence.** Because the probe is built from the norm and the
   answerer is shown the chunk's *flows*, the two describe different
   situations. Retrieval relates them; it does not make the extraction
   *evidence about* the probe. Consequence: teacher-perfect extractions scored
   **EM 0.104** (v1 prompt) / **0.385** (best re-worded prompt) while a
   degenerate policy inducing blanket answers scores **0.5–0.625**. **The
   honest ceiling sat below the gaming floor** — the reward paid more for
   degenerate extraction than for faithful extraction.

Fixing the template alone was tested and **rejected**: it helps the strict
prompt (macro 0.094 → 0.150) but *hurts* the best arm (0.362 → 0.281) and
flags 31% of probes as leaky. v1's apparent "best" score was guessing on
vacuous questions, not signal.

## v2 — the design

### Probe construction (build time, per reference flow)

For each **reference flow** in the chunk (the Gemma-4 teacher extraction, the
same artifact SFT supervises on):

1. Retrieve the governing norm by flow-text similarity (`flow_to_query`,
   unchanged — the validated v8 query distribution).
2. **Gold** = `FORCE_TO_APPROPRIATENESS(force)` → `appropriate` /
   `inappropriate`. `permitted`/unknown → flow is not probe-eligible.
3. **Probe** = **context as a field** + a referring expression naming *which*
   flow, then the appropriateness question:

   > "In the context of *{context}*, is it appropriate for *{information_type}*
   > to be shared?"

   Context is **mandatory**, not optional: appropriateness in CI is defined
   only relative to a context — the same transfer flips polarity across
   contexts, so a context-free probe is *ill-posed*, not merely hard, and
   `cannot_determine` would be correct for reasons unrelated to extraction
   quality. `condition_of_application` is **not** rendered (clause fragment;
   also a gold-leak channel — "unless the husband authorizes…" telegraphs a
   prohibition). `norm_articulation` is never rendered.

### What the answerer sees (the key change)

**Whitelist:** `sender, recipient, subject, information_type,
transmission_principle, context, norms_invoked`
**Withheld:** `appropriateness`, `reasoning`, chunk text.

Rationale — this is the crux. Appropriateness **cannot be judged without the
governing norms**; a model judging from weights alone is doing *zero-shot*
normative grounding, which is what SFT already gives and what GRPO exists to
replace. Therefore:

- **`appropriateness` is withheld** — it is the *verdict*. Showing it lets the
  answerer read the answer off a label, collapsing R-OUTCOME into v9's
  `m-DIRECTION` (direct label matching), which the redesign explicitly claims
  to subsume.
- **`norms_invoked` is shown** — it is the *rule* the policy claims governs
  the flow. The answerer applies the stated rule to the flow and derives a
  verdict.

So the reward asks: **did the policy identify a norm that, applied to this
flow, yields the correct judgment?** That is normative grounding measured by
verifiable consequence — the paper's thesis, instantiated. It is not circular:
gold comes from the book's *actual* governing norm, so a wrong or vacuous
`norms_invoked` yields the wrong verdict or `cannot_determine` (scored 0).

**Division of labour with R-GROUND** (previously blurred): R-GROUND scores
**fidelity** — do the invoked norms match the retrieved ones (its rubric
criterion 1). R-OUTCOME scores **consequence** — does the invoked norm produce
the right judgment. Neither subsumes the other; the `−ground` cell now tests a
genuinely distinct question.

### Gold: k = 1, no aggregation

Measured over 2,000 reference flows (`outputs/2026-07-25_norm_agreement/`):

| | |
|---|---|
| top-3 unanimous on polarity | 80.5% |
| top-1 ≠ majority | 6.6% |
| unanimity, top-1 `appropriate` | 87.4% |
| unanimity, top-1 `inappropriate` | **5.4%** |
| inappropriate share after agreement-gating | **0.6%** (from 8.4%) |

**Agreement-gating and majority-vote are both rejected.** The universe is
~89% appropriate-polarity, so any top-k neighbourhood is appropriate-dominated
*by base rate*; "consensus" is a base-rate artifact. Gating would delete **95%
of the minority class** and majority-vote would systematically flip
`inappropriate` labels to `appropriate` — erasing precisely the Forbid-recall
signal the paper rests on. **k = 1** (top-1 similarity) is the only scheme that
preserves the minority class.

Retrieval confidence should instead be captured **class-neutrally** — top-1
similarity, and the top-1/top-2 margin — recorded per probe as a diagnostic
(not a filter) so gold quality can be analysed post-hoc.

**Provisional-rate warning.** The observed 91.6/8.4 gold split is *not* an
established property. Top-1 lands *below* the 10.8% pool base rate, consistent
with retrieval being roughly uninformative about direction. The true rate is an
**output** of a working pipeline, reported in `training_metadata.json` — never
an input. Nothing in v2 may be calibrated to an assumed skew.

### Scoring

- `EM = 1` iff the answer equals gold; **`cannot_determine` → 0** (the tooth,
  unchanged).
- **Macro-EM**: mean of per-class EM over the classes present in the row
  (`AnswererClient.em_macro`, landed 2026-07-25). Rate-agnostic by
  construction — no global prior. A blanket answer scores 0.5 on any row
  carrying both classes.
- Failure handling unchanged: one retry, then whole-group uniform 0.5.

### Sampling

K = min(4, eligible flows), **adaptively force-stratified**: include a
minority-class-governed flow whenever the chunk has one; never tuned to an
assumed ratio. Realized composition reported per principle 6.

### Null-answerability

Re-runs and becomes meaningful for the first time. Its 0% drop in the
2026-07-24 calibration was an artifact of the answerer refusing *everything*,
including empty extractions. With well-posed, context-bearing probes, the
filter measures whether **context + referring expression alone** determines the
answer. A large drop rate is the signal that the referring expression must
narrow — an empirical question the filter answers, not one to guess now.

## Acceptance bar before any relaunch

Validate on the existing 40-chunk balanced harness
(`scripts/ab_probe_template.py` pattern) **before** rebuilding all 11k probes:

1. **Teacher-perfect extraction must decisively beat the blanket baseline** —
   macro-EM ≫ 0.5. v1 never did (0.362). This is the gate; if v2 fails it, the
   design is wrong and we do not relaunch.
2. Empty-extraction EM ≈ 0 (no world-knowledge leakage) — v1 achieved this and
   it must be preserved.
3. Mismatched-extraction EM ≪ matched (extraction specificity).
4. `cannot_determine` on teacher-perfect extractions < 0.2.

## T-VIGNETTE — same defect, different fix (measured 2026-07-25)

`batteries._scenario_text` reuses `probes.template_probe`, so **99.1% of
battery scenarios (2,764 / 2,789) carry defect 1**. For vignettes this is worse
than for probes: the policy is asked to assign a *five-way deontic force* to a
scenario containing no act —

> "In a setting involving professional conduct, an author of non-fiction **is in
> a situation where when presenting accounts of events as true**."

This is the most likely explanation of the killed wave's vignette forensics
(`antithesis_frac` 0.24, `hedge_frac` 0.34–0.44).

**Contaminated evidence, retracted.** The SFT per-force accuracy baseline
(`outputs/2026-07-23_mseries_premeasure/sft_force_accuracy/`, 2026-07-24) was
measured with the same templating via `scripts/measure_sft_force_accuracy.py`.
Its headline — a commit gradient at init, mean-s 0.31 > always-hedge 0.19 — was
cited in [data.md](data.md) and used as evidence *against* expecting the `core`
kill branch. **It is void** and must be re-measured on well-formed scenarios.

**Fix (differs from the probe fix — the condition is KEPT).** A vignette
scenario is the *object of judgment*, not something that must correspond to an
extraction, so there is no reason to strip content and the condition often
determines the force:

```
In a setting involving {context}, {subject} is considering whether to {act}, {condition}.
```

Always render the act; **comma-join the condition verbatim** with no connector
— a bare comma composes correctly for clause conditions ("when …", "unless …")
and prepositional ones ("during formal court proceedings") alike. A connector
heuristic was tried and produced "in circumstances where *during* formal court
proceedings"; the bare comma is strictly better.

**Open risk — scenario self-revelation.** Acts are extracted from directional
norms and are often phrased directionally ("**ensure** the factual accuracy",
"**avoid** misleading the audience"), which can telegraph obligatory/prohibited
without any knowledge of the book. `probe_leaks` catches force *words*, not
this. Settle it empirically with the vignette analog of null-answerability:
score the batteries with a model shown **only the scenarios** (no book, no
norms). Well above chance ⇒ the task is measuring phrasing, not normative
knowledge, and the battery builder needs act neutralisation.

## Resolved since this proposal was drafted

- **`reward_traces.jsonl` — FIXED 2026-07-25.** `ModularReward` wrote no traces
  while `grpo_training.py:769` printed "Reward traces → …" for *both* branches,
  so the m1 wave announced traces on every cell and produced none. Trace
  plumbing added (keeper conventions: append-JSONL, bounded file, failures
  swallowed), wired through `make_modular_reward_from_cfg` and the dispatch
  site. One row per completion covering every route (`gate_fail` with reason /
  `abstain_table` / `scored` / `vignette`), carrying cell identity, weights,
  final score, outcome + per-aux terms, and — for scored rows — `probe_ids`,
  `golds`, `answers`, so micro-EM / per-class EM / re-derived gold stay
  recomputable post-hoc without re-running the answerer. Four tests, incl. a
  post-hoc-rescoring pin and an unwritable-path guard. Suite 1,315 green.

## Open items

- Referring-expression choice (`information_type` vs `subject`) — pick by the
  null-filter drop rate, not by argument.
- Re-measure the SFT per-force baseline once battery scenarios are well-formed.
