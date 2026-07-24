# `T-VIGNETTE` — deontic battery task (ablatable task-mix module)

**Parent:** [README.md](README.md) · **Date:** 2026-07-16 · **Revised:**
2026-07-16 (single vignettes → K-item batteries with deontic-distance scoring)
· **Status:** drafted · **Kind:** task-mix module — removal =
`task_mix.vignette: 0`; nothing else in the stack changes.

## One-liner

A training row is an **K-item test** (default K=8): K scenarios templated from
K different norms of the **same book and same context**, mixed in gold
polarity; the policy assigns each a deontic force with brief reasoning and a
statement of the governing norm; the reward is mean **deontic-distance**
score — full credit for the right force, zero for hedging, and **negative
credit for the antithesis** (calling an obligation prohibited or vice versa).

`T-EXTRACT` teaches the policy to *describe* a text's normative world;
`T-VIGNETTE` teaches it to *discriminate between norms within one context*
and apply them as behavioral constraints. The `−vignette` cell tests the
complementarity claim.

## Why a battery instead of single vignettes (design revision 2026-07-16)

1. **GRPO signal density.** A single binary vignette gives R ∈ {0,1}; with
   G=8 completions, groups frequently tie (all correct or all wrong) → zero
   advantage → dead row. This is the v1 tied-group pathology (60% tied groups
   under the absolute judge) in miniature. A K=8 battery yields a dense
   scalar (≥9 levels before penalties), so every group carries within-group
   spread — the quantity GRPO's advantage actually consumes.
2. **Context stratification kills the context shortcut.** All K norms share
   one context but mix gold polarities, so a context-level heuristic
   ("medical setting → refuse") scores at base rate; only norm-level
   discrimination scores high. Single vignettes let context priors
   masquerade as norm knowledge.
3. **External precedent (URPO).** Recasting evaluation as a *set* task scored
   by graded agreement is URPO's validated preference-data trick (N-way
   ranking prompt, Kendall-τ reward) — a set task turns a brittle binary
   signal into a smooth verifiable one.

Elegance cost, stated honestly: `R-OUTCOME` probes remain *single* yes/no
questions to the frozen answerer, so the earlier "one mechanism, two uses"
tightens to **shared scenario templating, different aggregation** — the
builder and anti-leak properties are still common code.

## Battery construction (build-time)

From the vignette universe (default: the grounding universe — see
"Universe choice" below):

1. **Group** decisive-force, `governs_info_flow: true` norms by
   (book, context cluster). Contexts are free-text fields; cluster them with
   the small sentence embedder (the one `R-CONTEXT` used) + a similarity
   threshold, so "marriage negotiations" and "arranging a marriage" pool.
   Cluster granularity is a build-time report field, not a training knob.
2. **Compose** batteries of K = 8 items from one cluster: **both polarities
   present whenever the cluster has them**, with a hard floor of ≥1 minority
   item (target 2), bounded by pool — realized composition reported. This is
   the resolved decision in [data.md](data.md) ("hard floor ≥1 minority item
   per battery, target 2"), which the code implements as `minority_floor: 1` /
   `minority_target: 2` (2026-07-24). Clusters with 4 ≤ n < 8 eligible norms form smaller batteries
   (variable K, score is a mean so scale is unaffected); clusters with n < 4
   are skipped.
3. **Scenario per item** via the shared builder (`_generate_vignettes`,
   `stages/grpo_training.py:28`): templated from the norm's fields (context,
   subject, condition/act); the `articulation` is withheld; force words never
   appear (same anti-leak contract as probes).
4. Deterministic composition (RNG seeded by cluster id) — batteries enter the
   prescreen cache signature.

Output schema per item (JSON list, one entry per scenario):

```json
{"items": [
  {"id": 1,
   "force": "obligatory|recommended|permitted|discouraged|prohibited",
   "reasoning": "one or two sentences",
   "governing_norm": "the rule the model believes governs this scenario"},
  ...
]}
```

## Reward: deontic-distance scoring (one formula, no tier ladder)

Forces map onto the standard deontic axis:

```
obligatory +2 · recommended +1 · permitted 0 · discouraged −1 · prohibited −2
```

Per item: `s_i = 1 − |axis(model_i) − axis(gold_i)| / 2`, with a missing,
unparseable, or non-force answer scored as `permitted` (axis 0 — the hedge
point). This single linear formula produces the entire cost structure the
v9–v12a lineage needed four knobs for:

| model vs gold | s_i |
|---|---|
| exact force | **1.0** |
| adjacent degree, same polarity (recommended ↔ obligatory) | 0.5 |
| hedge (permitted / missing) vs a decisive gold | **0.0** |
| mild antithesis (discouraged vs gold obligatory) | −0.5 |
| **full antithesis (prohibited vs gold obligatory)** | **−1.0** |

Battery score `= (mean(s_i) + 1) / 2 ∈ [0,1]`. Hedge economics fall out
rather than being tuned: committing beats hedging iff per-item accuracy
> 50% (commit EV `2p−1` vs hedge 0), and a blanket-polarity strategy on a
mixed battery is driven *negative* by the antithesis penalty — the same
pressure the v12a hedge tier hand-crafted, derived from a distance metric
instead.

Full vignette reward:

```
R_vig = 0.7 · battery  +  0.3 · cite
```

- **`cite`** — per item, Jaccard token overlap between the model's
  `governing_norm` statement and the withheld source articulation
  (`r_norm_cite` generalized per-item, then averaged). The articulation is
  the citation *target* but absent from the prompt, so credit requires
  producing the rule's content, not parroting it. This keeps the battery from
  collapsing into force-classification-by-vibes: the model must say *which
  rule* it is applying, per scenario.
- **Dropped: `r_judgment_reasoning`** (0.25 in v9–v12a) — its mechanism
  (`rewards.py:655`) is a keyword counter (20 privacy words, score =
  min(1, hits/5)), gameable by one boilerplate sentence; Memory-R1's Table 2
  verbosity failure in miniature. The `reasoning` field stays in the output
  schema (it disciplines the generation and aids trace forensics) but earns
  no reward.

**Gold-degree noise, acknowledged:** the teacher's force labels are
polarity-reliable but degree-noisy (obligatory vs recommended is judgment).
The distance formula prices degree errors at ±0.5 — half the polarity
stakes — so label noise on degree costs little; polarity, which
`FORCE_TO_GOLD` treats as the trustworthy bit, carries the reward.

## What this task actually does — the v10/v11 evidence

Evidence from the single-vignette era; the battery format inherits its
motivation and must re-verify its effects:

- **Vignettes are load-bearing for judgment behavior.** The v11 probe changed
  *only* vignette composition (rebalanced vs v10's skewed mix) and **halted
  v10's verdict erosion** — gold-"no" yes-rate drift 0.12→0.20 (v10) vs
  0.01→0.05 (v11); ConfAIde tier-2b gap halved; CIRL held at SFT level where
  v10 eroded. An inert module cannot do that.
- **They never moved extraction hedging** (~72% frozen in both runs) — and
  were never the mechanism for it. In the redesign that is `R-OUTCOME`'s job;
  the division of labor is clean.
- **The gradient is asymmetric at init** (qwen-era): SFT was already strong
  on gold-"no" (0.94) and weak on gold-"yes" (0.64), so the task mostly
  teaches "engage more". **Re-measure per-class accuracy on the Gemma-4 SFT
  base at build** (now per force class, given the 5-way output) and record it
  in `training_metadata.json`.

## Mix and realized-composition accounting (principle 6)

`task_mix: {extract: 0.7, vignette: 0.3}` — fixed across every grid cell
(`−vignette` sets it to 0; no other cell varies it). One battery = one row.

The v10 lesson: the force-blind variance prescreen silently doubled the
vignette skew (pool 3.07:1 → realized 5.2:1). The redesign's prescreen is
stratified by (task × battery composition class), and the realized
composition is still *reported*: per-battery polarity counts pre/post screen
in `training_metadata.json` / `prescreen_report.json`; per-force accuracy and
antithesis rate stream to W&B under `vignette/*` (`antithesis_frac` is the
new headline forensic — it should be rare and falling).

*Caption on `antithesis_frac` (2026-07-24):* the forensic counts polarity
**flips** (axis sign product < 0, i.e. `axis(model)·axis(gold) < 0` in
`deontic_distance.py`), which includes the cross-polarity distance-2 cell
(recommended ↔ discouraged) that scores `s_i = 0.0` — reward-*neutral*, not
negative. So the metric name and the per-item reward diverge on exactly that
cell. This is intentional: the reward is pure axis-distance, the forensic is
pure polarity sign; the caption just prevents misreading the metric.

## Universe choice and balance — decoupled

The v11 lever conflated *which universe* with *what balance*; on Gemma-4 data
the conflation no longer works (top100 8.5% vs fiction10 7.3% prohibited).
The redesign separates them:

- **Balance** is a property of battery composition (step 2 above: ≥1
  minority item per battery, target 2, bounded by pool — 2026-07-24, matching
  the construction section and Config keys), not of corpus choice.
- **Universe** defaults to **the grounding universe** — same book set as
  `T-EXTRACT` and `R-OUTCOME`, one artifact fewer, and `−vignette` stays a
  pure task ablation. `VIGNETTE_NORM_UNIVERSES_PATH` survives as an override
  for corpus experiments only.
- The binding constraint is the **gold-"no" pool per context cluster**;
  count check in [data.md](data.md).

## Interactions and discipline

- **With `R-OUTCOME`:** shared scenario templating and anti-leak contract;
  different aggregation (battery answered by the policy vs single probes
  answered by the frozen answerer from the extraction). The policy never sees
  which norms serve as probes. Battery training on norms of context C
  legitimately transfers to extraction rows whose flows those norms govern —
  that transfer is the point, and `−vignette` measures it.
- **Split discipline:** batteries only from training books; held-out books
  contribute no batteries, probes, or chunks ([data.md](data.md)).
- **Leak test (migration test plan):** automated assertion that no battery
  **scenario text** contains a source articulation token sequence or any force
  word — the PrivacyLens canary pattern. The guarantee is scoped to scenario
  text only: force words legitimately appear in the *instruction* portion of
  the prompt (the force vocabulary the model must choose among), so the
  anti-leak contract binds `scenario_text` — which is exactly what the code's
  test asserts on (`tests/grpo_training/test_batteries.py`; 2026-07-24).

## `−vignette` LOO cell — pre-registered prediction

**Expected direction: removal hurts.** Based on v11's causal evidence,
`−vignette` degrades the judgment-format benchmarks relative to `full` —
ConfAIde tier-2 and/or CIRL drop, or v10-style verdict drift re-opens —
while GoldCoin applicability and extraction-trace metrics stay within noise
of `full` (extraction is carried by the outcome core).

*Falsification branch* (stated per protocol, not expected): if `−vignette`
matches `full` on all four benchmarks, the battery task is redundant under
outcome supervision — the outcome core's judgment pressure on extractions
suffices — and the paper reports that as a finding. One caveat raises this
branch's probability above the v11-era baseline: `R-OUTCOME` did not exist
then, and it now carries judgment-format signal into extraction rows, so the
battery's *marginal* value is genuinely more open than v11 alone implies.
That is precisely why the cell is worth its compute.

## Config keys

```yaml
training:
  grpo:
    task_mix: {extract: 0.7, vignette: 0.3}  # −vignette cell: vignette: 0
    battery:
      k: 8                    # items per battery (4..8 realized, pool-bound)
      min_k: 4                # min eligible norms for a cluster to form a battery
      minority_floor: 1       # hard floor: >=1 minority-polarity item per battery
      minority_target: 2      # target minority items (bounded by pool)
      # minority_share (0.25) was never implemented — the code reads integer
      # counts (min_k / minority_floor / minority_target), not a share;
      # schema above matches m_series.yaml (2026-07-24).
      # context clustering threshold fixed at build; reported, not swept
```
