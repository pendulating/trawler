# `R-DIRECT` — norm-classified appropriateness (replaces `R-OUTCOME`)

**Parent:** [README.md](README.md) · **Date:** 2026-07-25 · **Status:** SPEC —
for review. Supersedes [reward-outcome.md](reward-outcome.md) and
[reward-outcome-v2-proposal.md](reward-outcome-v2-proposal.md) as the m-series
verifiable core if approved.

## One-liner

For each extracted flow, the **norm classifies the flow** — the governing
norm's Raz force plus its act polarity determine whether that flow is
appropriate or inappropriate — and the reward is the agreement between the
policy's own `appropriateness` label and that norm-derived gold. No frozen
answerer, no judge, no server call.

## Why the answerer is gone

`R-OUTCOME` asked a frozen answerer to judge appropriateness from the
extraction alone and scored EM against norm-derived gold. Three defects were
found and fixed (contentless probes, wrong prescriptive object, no
correspondence — [v2 proposal](reward-outcome-v2-proposal.md)); v2 fixed probe
construction on every diagnostic:

| | v1 | v2 |
|---|---|---|
| `cannot_determine` | 0.368 | **0.006** |
| specificity (matched − mismatched) | — | **+0.612** |
| empty-extraction EM (leakage) | 0.000 | 0.000 |

But the answerer then became the wall. Measured on 40 chunks / 157 probes with
polarity-corrected gold:

| arm | macro-EM | gold-yes | **gold-no** |
|---|---|---|---|
| teacher's zero-shot `norms_invoked` | 0.585 | 0.874 | **0.130** |
| **ORACLE — the book's actual governing norm, verbatim** | 0.627 | 0.901 | **0.174** |
| blanket-"appropriate" floor | 0.600 | — | — |

**Handed the prohibiting norm itself, the answerer still calls ~83% of
prohibited flows appropriate.** The entire learnable signal — the gap between
zero-shot grounding and perfect knowledge of the book's norms — is **+0.042
macro**, smaller than per-group reward noise. There is no gradient to climb.
This is a property of the answerer (over-permissive in every configuration
tested: it refused *everything* under the strict prompt, then assented to
everything under the calibrated one), not of the probes.

Recorded as a negative result for the paper's limitations: **appropriateness
judgment cannot be outsourced to a norm-blind reader.** That is not a quirk of
one model; it is the same claim the project makes about SFT — a model judging
from weights alone is doing zero-shot normative grounding, which is exactly
what the reward exists to correct. The frozen answerer was always doing that.

## The module

**Applies to:** `T-EXTRACT` rows, gold-YES, gate-passing, flow-bearing. No-flow
declarations and gold-NO chunks route to [`A-ABSTAIN`](reward-abstain.md)
unchanged.

**Per flow in the completion's extraction:**

1. Retrieve the governing norm — `flow_to_query(flow)` against the chunk's own
   book universe, **k = 1** (top-1 similarity).
2. `gold = deontic.flow_appropriateness(force, act_polarity)` →
   `appropriate` / `inappropriate`. **`permitted` → appropriate** (decision
   2026-07-25): a permission is not a violation, so a flow the norms permit is
   appropriate. It is polarity-**invariant** — "you may refrain from disclosing"
   does not forbid disclosing, and inverting a permission would manufacture a
   prohibition. Only an unknown/missing force leaves a flow **unscored** (it
   neither earns nor loses). Measured effect: scorable flow-governing norms go
   2,789 → **2,870 (100%)** with class balance essentially unchanged
   (74.6/25.4 → **72.5/27.5**) — more coverage, no new skew.
3. Compare to the policy's own `appropriateness` label for that flow:
   - exact match → **1.0**
   - opposite → **0.0**
   - `ambiguous` / missing / unparseable → **0.0** (the hedge is priced at the
     floor, inheriting `R-OUTCOME`'s tooth — hedging must not beat committing)
4. `R_direct` = **macro-average over the gold classes present** in the
   completion's scored flows (per-class mean, then mean of classes), so a
   blanket label scores 0.5 on any completion carrying both classes rather
   than riding the corpus skew.

Completion with **no scorable flows** (all `permitted`/unretrievable) → the
term is undefined; the completion takes `GROUP_NEUTRAL` (0.5) for this term
only, exactly as the answerer-failure convention did.

## What this preserves

- **Normative grounding** — gold still comes solely from the text's own
  normative universe. Unchanged claim.
- **Verifiable core, no judge opinion** (principle 3) — *strengthened*: the
  scored path is now pure lookup + string comparison, with no model in it at
  all. The one model call `R-OUTCOME` had is gone.
- **The LOO grid** — `R-GROUND` / `R-CONTRAST` / `T-VIGNETTE` are untouched;
  the `−outcome` cell becomes `−direct` and asks the same question (does the
  verifiable core carry the result, or do the judges?).
- **Polarity-corrected gold** — the 19%-inverted-label fix and the resulting
  2.5:1 class balance carry over intact.

## What it costs, stated plainly

This **is** the v9 lineage's `m-DIRECTION` mechanism, which
[reward-outcome.md](reward-outcome.md) claimed `R-OUTCOME` would subsume. The
Memory-R1-style "outcome supervision" novelty is dropped. Two mitigations, both
honest rather than cosmetic:

1. `m-DIRECTION` was a **multiplier inside `R_ground`** with a tier ladder
   (`floor` / `floor_prohibit` / `hedge_prohibit`) tuned across v9→v12a.
   `R-DIRECT` is a **standalone additive module** with **no knobs**: one lookup,
   one comparison, macro-averaged. The tier ladder is gone because the class
   imbalance that motivated it was largely the act-polarity artifact.
2. It is scored against **polarity-corrected** gold. Every v-era direction
   number was computed on labels inverted for ~19% of norms, so this is not a
   re-run of the old mechanism — it is the first time it has been measured
   correctly.

The paper reports the outcome-supervision attempt as a negative result with the
oracle measurement as evidence. That is a genuine contribution: it shows *why*
process-free outcome supervision fails in this domain, which is not obvious a
priori and is the opposite of Memory-R1's finding in theirs.

## Cost

Zero server calls per group. Retrieval only (embeddings already cached per
book). Strictly cheaper than every prior version of the core.

## Diagnostics (`reward/direct/*`)

`agreement_mean`, `agreement_by_class/{appropriate,inappropriate}` (the
Forbid-recall proxy — the line to watch), `hedge_frac` (`ambiguous`/missing),
`antithesis_frac` (opposite-polarity commits), `unscored_flow_frac`,
`group_spread`, plus `diag/retrieval_margin` (top-1 minus top-2 cosine) as a
class-neutral gold-confidence signal.

## Pre-registered predictions

1. `group_spread` > 0 in ≥90% of groups from step 1 — G=8 completions on the
   same chunk will disagree about appropriateness far more often than they
   disagreed about probe answers (the answerer collapsed to one label; the
   policy does not).
2. `agreement_by_class/inappropriate` **rises** over training. This is the
   claim the whole project rests on, and it is now measured against gold that
   is correct. **Kill criterion:** if it stays flat while
   `agreement_by_class/appropriate` climbs, the policy is learning the base
   rate rather than the norms, and the verifiable core has failed on its
   central promise.
3. Held-out: GoldCoin Forbid recall moves off the 0.55 plateau without a
   compensating collapse in Permit recall or applicability.

## Acceptance bar before launch

Same discipline as v2 — validate before rebuilding anything:

1. **Teacher-flow agreement** must sit well above the blanket-label floor on
   the balanced 40-chunk set. Unlike the answerer arms, this is a *pure
   computation* — no model in the loop — so it can be measured offline in
   seconds with no server.
2. Gold-class balance and `unscored_flow_frac` reported before launch.
3. `group_spread` confirmed non-degenerate in the m0 smoke.

**Note the asymmetry that makes this checkable:** because there is no answerer,
the "ceiling" is not a model's willingness to commit — it is simply how often
the teacher's own appropriateness labels agree with the norm-derived gold. That
number is a fact about the data, computable immediately, and it is the honest
prior on how much room GRPO has to improve.

## Acceptance bar — MEASURED 2026-07-25 (offline, no model in the loop)

1,200 teacher flows, k=1 retrieval, polarity-corrected gold:

| | |
|---|---|
| flows scorable | **100%** (0% permitted/unretrievable) |
| gold balance | 74.6% appropriate / **25.4% inappropriate** |
| teacher agreement — appropriate | 0.863 (n=895) |
| teacher agreement — **inappropriate** | **0.102** (n=305) |
| teacher hedge (`ambiguous`/missing) | 5.2% |
| **macro agreement (zero-shot baseline)** | **0.482** |
| blanket-label floor | 0.500 |

**The SFT teacher's zero-shot normative grounding is no better than a constant
label** — 86% correct on permissive flows, **10%** on prohibited ones. This is
the project's motivating claim, quantified: a model judging appropriateness from
weights alone is not doing normative grounding.

Contrast with the abandoned answerer path, which is why this module is worth
running: there the headroom between zero-shot and perfect norm knowledge was
**+0.042 macro**; here it is **0.482 → 1.0**. Roughly a twelvefold larger
gradient, with no permissive model in the scoring path to flatten it, and
within-group disagreement (the advantage carrier) arising naturally because G=8
completions genuinely differ on appropriateness.

**Caveat — gold noise.** Some of the 0.102 is retrieval error rather than
teacher error: gold rests on k=1, and top-1 disagrees with the top-3 majority on
6.6% of flows. `diag/retrieval_margin` exists to test this post-hoc (do
low-margin flows carry the disagreement?). 0.482 is therefore a *floor* on
zero-shot quality, not a point estimate — but the direction and the size of the
available gradient are not in doubt.
