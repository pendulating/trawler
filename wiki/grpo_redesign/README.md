# GRPO ground-up redesign: the modular training stack

**Date started:** 2026-07-16 · **Revised:** 2026-08-05 · **Status:** design
master, and **the shipped camera-ready design** — the m2 `full` cell
(`multirun/2026-07-28_grpo_m2_full/21-31-11/cell=full`, step 450) is the
paper's GRPO model as of 2026-08-05, promoted by author ruling over its own
failed promotion gates; see [../2026-07-31_kto_plan.md](../2026-07-31_kto_plan.md)
§19 for the decision, the red gates it ships with, and the benchmark-distrust
rationale. NB the shipped core is **`R-DIRECT`** ([reward-direct-spec.md](reward-direct-spec.md)),
not the `R-OUTCOME` frozen answerer this page's diagram and inventory describe:
the answerer was measured and abandoned (spec §"Why the answerer is gone").
· **Owner:** Matt + assistant sessions

## Why (read this first)

Three forces converge on a rewrite:

1. **Legibility.** The current reward (v9→v12a lineage) is the sediment of a
   12-arm debugging saga. Its mechanisms are *nested*, not composed: the
   appropriateness direction is a multiplier **inside** R_ground, the
   contrastive penalty is a clamp **inside** R_ground, cost-sensitive tiers are
   knobs **inside** the multiplier, vignette rows **bypass** the composite
   entirely, and no-flow completions route differently per component. Nobody —
   author or reviewer — can state the reward in three sentences.
2. **The retrain is happening anyway.** All fiction norm/flow data was
   re-extracted with the Gemma-4 teacher after the prompt-wiring fix; every
   GRPO run on the corrected data starts from a fresh SFT base (canonical
   sweeps of 2026-07-15). There is no continuity to preserve on the new data
   path.
3. **The per-component reward ablation is the main outstanding experiment**
   for the camera-ready. In the current code an honest take-one-out is nearly
   impossible: removing "direction" changes R_ground's formula; removing
   "contrast" changes its clamp; removing "vignettes" changes the prescreen
   mix. In the redesign, **an ablation cell is created by literally deleting
   one module from a config list** — nothing else moves.

**What is frozen:** the v9-ckpt100 keeper and its exact code path
(`reward_composition: directional` + `online_rground_external.yaml`) stay
untouched and byte-reproducible for the paper's existing results. The redesign
is a parallel stack, not an edit of the old one.

## The external template: Memory-R1's GRPO (ACL 2026)

The redesign deliberately copies the *validated* structure of Memory-R1
(`2026.acl-long.583.md`; see also the
[2026-07-16 paper synthesis](../grpo_training_field_notes/2026-07-16_rl_papers_synthesis.md)),
whose setting is structurally ours:

| Memory-R1 | This project |
|---|---|
| Memory Manager emits **structured memory operations** | Policy emits **structured CI flow extractions** |
| Operations judged by their **effect on a frozen Answer Agent** | Extractions judged by whether a **frozen answerer** can reach the correct privacy judgment *from the extraction alone* |
| Reward = **EM(answer, gold)** — one verifiable scalar, no quality judge | Reward core = **EM(probe answers, gold-from-norm-force)** — one verifiable scalar |
| Gold derived from existing QA pairs (152 suffice) | Gold derived from the book's **normative universe** (force → yes/no), zero manual labels |
| Counterpart **frozen during training** (no co-evolution; clean attribution) | Answerer + all servers frozen per run |
| KL to reference policy retained | β=0.02 to the SFT reference (kept) |
| T=1.0 rollouts / greedy eval | same discipline |
| Trained on one benchmark, **zero-shot transfer** to two others | Train on fiction, transfer to GoldCoin / PrivacyLens / ConfAIde / CIRL |
| Ablations: each component removed **in isolation**, same eval matrix | The LOO grid below |

Their decisive negative result is our design constraint: **rewarding with an
LLM-judge score produced verbose, judge-pleasing outputs that gamed the judge
metric while degrading every string metric** (their Table 2). Our v10–v12
forensics found the same equilibrium under our quality judge — hedge mass
frozen at ~72%, Forbid recall pinned at 0.55. The fix they validated is not a
better judge; it is **outcome supervision**: the reward asks whether the
artifact *works*, not whether it *looks good*.

### What this changes about "normative grounding"

The paper's thesis — reward grounded in each text's normative universe — is
**kept, re-operationalized**. Two operationalizations of the same principle:

- **Grounding-by-judge** (published, v9 lineage): a judge reads the extraction
  and the retrieved norms and scores quality. Process supervision; hackable by
  plausible-looking hedges.
- **Grounding-by-outcome** (new core): the universe generates **probe
  questions** with force-derived gold answers; a frozen answerer must answer
  them *given only the extraction*. The universe is still the sole source of
  supervision — it now supervises through verifiable consequences instead of
  judged appearances. A hedged or contentless extraction cannot support
  correct answers, so the hedge equilibrium is priced at its root.

Both live in the module inventory; the grid measures their relative
contribution directly (`full` vs `−outcome` vs `−ground`). That comparison —
process vs outcome supervision for normative grounding — is itself a
publishable ablation, and it is Memory-R1's Table 2 replicated in our domain.

## Design principles

1. **One module = one sentence** — what it rewards, how computed, what
   removing it tests. If it needs a paragraph, split it.
2. **Removal is the identity operation.** Additive modules delete +
   renormalize; task modules go to mix 0. No module's presence changes another
   module's formula.
3. **Verifiable core, judged auxiliaries.** The always-on reward path contains
   *no LLM-judge opinion* — only parseability and EM against universe-derived
   gold. Judge-based signals exist only as removable auxiliaries.
4. **Fewest components that can carry the claims.** Legacy components with no
   demonstrated signal do not migrate (see "What does not migrate").
5. **Fixed optimizer, moving reward.** One optimizer preset, identical across
   all cells; optimizer experiments (v13a μ=2) are a separate axis.
6. **The realized mix is the reported mix.** Prescreen is stratified by
   (task × gold class × force) so the configured mix survives filtering — the
   v10 lesson (3.07:1 pool silently became 5.2:1 realized).

## The stack in one diagram

```
DATA                             PROMPTS                     REWARD
────                             ───────                     ──────
fiction chunks ────────────────► T-EXTRACT rows ─┐
                                                 ├─ stratified ──► G completions
norm universes ──┬─────────────► T-VIGNETTE rows ┘   prescreen        │
(per book)       │               (8-item batteries)                   ▼
teacher flows ───┴► probe questions + gold ────►  T-EXTRACT:  gate(valid) · [ EM-outcome + auxiliaries ]
(SFT's reference    (retrieve norms per flow,     T-VIGNETTE: deontic-distance battery + norm-cite
 extraction)         force → yes/no)                                  │
                                                                      ▼
                                                    GRPO update (fixed optimizer preset, β=0.02, T=1.0)
                                                                      │
                                                    promotion gates → checkpoint → zero-shot benchmark evals
```

The reward in three sentences: *An extraction scores zero unless it parses and
is schema-complete; a valid one is scored by whether a frozen answerer,
reading only the extraction, correctly answers probe questions whose gold
answers come from the book's governing norms. Optional auxiliary terms
(judge-scored grounding, wrong-book contrast) add to that core and can each be
deleted independently. A vignette row is an 8-question test over norms from
one book and context, scored by how deontically close each answer is to the
norm's force — hedging earns nothing and the antithesis is penalized — plus a
citation check.*

## Module inventory

### Core (always on — not ablation axes)

| ID | One-liner | Computed by | Lineage |
|---|---|---|---|
| `R-VALID` | Parseable, schema-complete, internally consistent output — else R = 0. | programmatic | merges r_uncert + r_complete + r_consist (all saturated post-SFT) |
| `R-OUTCOME` | Frozen answerer answers K probe questions from the extraction alone; reward = mean EM vs force-derived gold. | frozen LLM, EM-scored | new (Memory-R1 transplant); **subsumes** v9's `m-DIRECTION` — direction agreement is *implied* by answering direction probes correctly |
| `A-ABSTAIN` | Rows where probes can't run — no-flow declarations, and *all* completions on gold-NO chunks — score from a fixed table (0.1 wrong-abstain / 0.6 correct-abstain / 0.4 unverifiable); no server calls. | programmatic | v9 `abstention_score`, extended to gold-NO extractions |

`R-OUTCOME` mechanics (details in [reward-outcome.md](reward-outcome.md)):
probes are built per prompt at dataset build — norms are retrieved **per
reference flow** (the Gemma-4 teacher extraction the SFT stage already trains
on; the same flow-text query type v8 traces showed carries a directional
governing norm ~97% of the time), unioned over the chunk, and templated into
probes. The frozen answerer receives *only* the completion's extraction JSON
plus the probes, one batched call per completion (G calls/group; short
inputs; cheaper per group than the listwise judge it displaces).

### Ablatable auxiliary modules

| ID | Kind | One-liner | Computed by | LOO question | Lineage |
|---|---|---|---|---|---|
| `R-GROUND` | additive | Listwise judge: are the extracted flows governed by this book's retrieved norms? | LLM judge + retrieval | Does process-quality judging add anything over outcome supervision? (Memory-R1 Table 2, our domain) | R_ground correct-universe pass |
| `R-CONTRAST` | additive | Reward for *not* also matching a random wrong book's norms (1 − wrong-universe score). | LLM judge | Does book-specificity matter, or would generic privacy norms do? | `contrastive_lambda` clamp |
| `T-VIGNETTE` | task mix | An 8-item test: assign deontic forces to 8 scenarios from the same book+context (mixed polarity); scored by deontic distance, antithesis answers negative. | programmatic (no judge) | Does *applying and discriminating* norms add anything beyond extraction with outcome probes? | `vignette_ratio: 0.3` bypass rows |

### What does *not* migrate (principle 4)

- **`R-CONTEXT` (0.20) and `R-COHERE` (0.10)** — process-quality heuristics
  with no isolated evidence of signal (R_ground carried the variance; these
  rode along). Cut. If a reviewer asks, a supplementary add-one-in cell can
  resurrect either — documented in [migration.md](migration.md).
- **`m-DIRECTION` and its cost-sensitive tiers** (v9 multiplier, v10 floor,
  v12a hedge tier) — subsumed by `R-OUTCOME`: a hedged answer to a
  prohibited-norm probe simply scores EM = 0, which *is* the cost-sensitivity,
  with no tier ladder to explain. The tiers were never validated (v12a unrun).
  Direction agreement survives as a **diagnostic metric** streamed to W&B, not
  a reward term.
- **The 6-weight vector.** The core has no weights. Auxiliaries get equal
  weight within their sum; the LOO grid, not weight tuning, is the evidence.
- **`contrastive_ratio`** (legacy additive wrong-source rows) — already 0.

### Vignette reward (fixed, two terms)

`R_vig = 0.7·battery + 0.3·cite`, where `battery` is the mean **deontic
distance** score over the K items: forces sit on the axis obligatory +2 …
prohibited −2, each item scores `1 − |model − gold|/2` → exact force 1.0,
adjacent degree 0.5, hedge 0, **antithesis −1** (the double penalty, from one
distance formula instead of a tier ladder), rescaled to [0,1]. `cite` is
per-item token overlap with the withheld articulation. Verifiable end-to-end.
(The legacy keyword-counter "judgment reasoning" term is dropped: principle
3.) Details + battery construction: [task-vignettes.md](task-vignettes.md).

### What a vignette actually is (worked example)

From each norm in the vignette universe with `governs_info_flow: true` and a
decisive force (`obligatory`/`recommended` → gold **yes**;
`prohibited`/`discouraged` → gold **no**; `permitted` skipped), the builder
templates a scenario from the norm's *fields* — its articulation is withheld
so the answer doesn't leak:

> Norm: `{subject: "a family's patriarch", act: "disclose his daughter's
> private circumstances to a suitor", context: "marriage negotiations between
> families", normative_force: "prohibited"}`
>
> Vignette prompt: *"In a setting involving marriage negotiations between
> families, a family's patriarch is considering whether to disclose his
> daughter's private circumstances to a suitor. Should he share this
> information? Judge whether this flow is appropriate and cite the norm that
> governs it."* Gold = **no**.

In `T-VIGNETTE` rows, eight such scenarios — all from the same book and
context cluster, mixed in polarity — are posed together as one test, and the
policy assigns each a deontic force (not just yes/no) with reasoning and a
statement of the governing rule.

The same scenario template generates `R-OUTCOME`'s probe questions — shared
templating, used in two places: as a *training task* (batteries the policy
answers) and as a *measurement instrument* (single probes the frozen answerer
answers from the policy's extraction). Builder: `_generate_vignettes`
(`stages/grpo_training.py:28`); force→gold map / deontic axis: `deontic.py`
`FORCE_TO_GOLD` (single source of truth).

### Not modules (fixed infrastructure, identical across all cells)

- **Optimizer preset** — one block: G=8, lr, β=0.02 (KL to SFT reference, as
  Memory-R1 retains), `scale_rewards: none`, `mask_truncated_completions`,
  `token_truncate`, μ=1, clips; T=1.0 rollouts, greedy eval. Inherits the
  v6–v8 stability diagnosis wholesale; [optimizer.md](optimizer.md).
- **Stratified prescreen** — variance screening preserved, stratified per
  principle 6.
- **Frozen counterparts** — answerer, judge, embedder never update mid-run.
- **Promotion gates, W&B logging** — per-module namespaces
  (`reward/<module_id>/*`).

## Scientific protocol (imported wholesale from Memory-R1)

1. **Splits before anything runs.** Chunk-level train/dev/test split, plus a
   *book-level* held-out set (books whose universes never supervise training)
   for the generalization claim. Memory-R1: 1:1:8 on 152/81/1307.
2. **Zero-shot transfer is the headline eval.** No benchmark data touches
   training; GoldCoin / PrivacyLens / ConfAIde / CIRL are reported for every
   grid cell — one table, rows = cells, columns = benchmarks.
3. **Every cell pre-registers a falsifiable prediction** before launch (house
   style since v8; now mandatory in [ablation-protocol.md](ablation-protocol.md)).
4. **Baselines in the same table:** SFT-only, and the 0-shot base model —
   Memory-R1's "vanilla + SFT-variant" pattern (their Memory-SFT ablation).
5. **Seed variance quoted** from the existing 5-seed protocol (CV 3.5%) for
   the canonical cell; single seed for LOO cells, flagged as such.
6. **Data efficiency is a feature.** Memory-R1 trained on 152 examples; we do
   not need every chunk. The stratified prescreen selects a *small,
   variance-carrying, force-balanced* prompt set of pre-registered size, and
   the realized composition is reported from `training_metadata.json`.

## The ablation, concretely

A cell is a config list:

```yaml
training:
  grpo:
    reward_auxiliaries: [ground, contrast]   # additive; weights by the fixed
                                             # 2:1 outcome:aux rule (ablation-protocol.md)
    task_mix: {extract: 0.7, vignette: 0.3}
```

Canonical grid (7 cells, one sweep yaml):

| cell | definition | question answered |
|---|---|---|
| `core` | valid gate × outcome only, extract-only | Is Memory-R1-style outcome supervision alone sufficient? |
| `full` | core + ground + contrast + vignette | the canonical checkpoint candidate |
| `−outcome` | full minus the outcome core term | Does the verifiable core matter, or do the judges carry it? |
| `−ground` | full minus judge grounding | process-vs-outcome supervision (the Table 2 replication) |
| `−contrast` | full minus wrong-book term | book-specificity |
| `−vignette` | full, task_mix vignette→0 | application-task complementarity |
| `sft` / `0-shot` | no GRPO | baselines (no training cost) |

Removing a module renormalizes remaining auxiliary weights, bumps the
prescreen signature automatically (the module list is part of the cache key),
and changes **nothing else**.

## Table of contents (subpages)

Status: ☐ to write · ◐ drafted · ☑ done

- ◐ [data.md](data.md) — corpus decision (fiction10-gemma4 canonical, top100
  = 93-book holdout), artifact inventory (**universe builds = gating
  prerequisite, not built**), module→artifact consumption map, splits,
  measured battery feasibility (tight but workable; clustering headroom),
  blocking job list
- ◐ [task-extraction.md](task-extraction.md) — `T-EXTRACT`: the frozen
  SFT-shared prompt, per-field schema→consumer map, answerer projection
  (flows verbatim, never reasoning), row lifecycle, gold-signal summary
- ◐ [task-vignettes.md](task-vignettes.md) — `T-VIGNETTE`: 8-item deontic
  batteries (same book+context, mixed polarity), deontic-distance scoring
  (antithesis −1 from one formula), cite term, battery construction +
  context clustering, v10/v11 evidence, balance-vs-universe decoupling,
  `−vignette` prediction (removal expected to hurt)
- ◐ [reward-outcome.md](reward-outcome.md) — **`R-OUTCOME` (core)**: probe
  generation (build-time, per prompt), frozen-answerer choice + prompt,
  null-answerability filter, cost model, anti-gaming table, subsumption of
  m-DIRECTION, pre-registered predictions
- ◐ [reward-valid.md](reward-valid.md) — `R-VALID` gate: five binary
  criteria, the 0.20-of-weight merge table, why binary is safe (saturation),
  the confidence lineage ends (no `confidence_fallthrough` counterpart)
- ◐ [reward-abstain.md](reward-abstain.md) — `A-ABSTAIN`: four-entry routing
  table, within-group economics, gold-NO neutrality + the audit caveat (and
  its re-run on the gemma4 flow label), why not ablatable
- ◐ [reward-ground.md](reward-ground.md) — `R-GROUND` (auxiliary): what was
  hoisted out, rubric slimmed to grounding-only (appropriateness criterion
  deleted — no double-counting direction), listwise protocol inherited,
  rank-weight worry downgraded, `−ground` prediction
- ◐ [reward-contrast.md](reward-contrast.md) — `R-CONTRAST` (auxiliary):
  clamp→module (why the v8 asymmetry bug becomes unrepresentable), seeded
  wrong-book sampling, universality bias stated, λ-sweep bridging note,
  `−contrast` prediction
- ◐ [optimizer.md](optimizer.md) — the fixed preset (every value =
  a falsified alternative, table + evidence), the v13a adoption rule (μ=2
  in or out *before* the grid, never per-cell), what the preset doesn't decide
- ◐ [prescreen-and-gates.md](prescreen-and-gates.md) — stratified prescreen
  (variance ranks within strata, never across), m1 cache-key contract,
  pre-registered N, the four gates carried over, gates vs kill criteria
- ◐ [ablation-protocol.md](ablation-protocol.md) — the m-series grid: cell
  mechanics, the 2:1 weight rule, per-cell pre-registered predictions,
  run-order/early-exit logic, seeds + noise bars, eval matrix, reporting
  table, compute envelope
- ☑ [m1-run-plan.md](m1-run-plan.md) — execution sequencing (2026-07-24):
  current readiness, model scope (qwen grid + gemma confirmation cell),
  phases A–G from implementation to transfer
- ◐ [migration.md](migration.md) — the parallel-stack rule (keeper surfaces
  never edited), old→new map, implementation checklist (7 components), test
  plan incl. a keeper-freeze regression guard, resurrection path,
  post-camera-ready deletion list

## Open design decisions (to resolve in subpages)

1. ~~**Frozen answerer choice for `R-OUTCOME`**~~ **RESOLVED**
   (reward-outcome.md D1, **revised 2026-07-23**): **Gemma-4-31B-it** — the
   canonical teacher/judge family for the camera-ready
   ([canonical-models.md](../canonical-models.md)); the original Qwen3.6-27B
   resolution was stale against that decision. Identical answerer across all
   cells + one offline second-answerer (non-Gemma) robustness check.
2. ~~**K (probes per chunk) and probe sampling**~~ **RESOLVED**
   (reward-outcome.md D2): K = min(4, pool), force-stratified (both gold
   classes whenever available), seeded by `chunk_id`; plus a
   null-answerability filter dropping probes answerable with an empty
   extraction.
3. ~~**Vignette/probe universe default**~~ **RESOLVED** (task-vignettes.md +
   data.md): balance is a battery-composition property (≥1 minority item hard
   floor, 2 target), universe = the grounding universe; the grounding corpus
   is **fiction10-gemma4** (matches the SFT bases and the probe anchor), with
   the 93 non-overlapping top100 books as the book-level holdout. Feasibility
   measured: ~100–200 batteries at exact-string contexts (tight), with large
   clustering headroom (median exact cluster = 1). **Universe builds for both
   corpora are the gating prerequisite jobs** (data.md job list).
4. ~~**Auxiliary weight scale**~~ **RESOLVED** (ablation-protocol.md): one
   arithmetic rule, no tuning — outcome weighs 2× each active auxiliary,
   auxiliaries equal among themselves, normalized (full = 0.50/0.25/0.25).
   The verifiable signal can never be outvoted by judge opinion; the exact
   ratio is deliberately not load-bearing (the `core` cell is the 1:0
   extreme).
5. ~~**Gold-NO abstention semantics**~~ **RESOLVED** (reward-abstain.md):
   gold-NO chunks are scored *entirely* by the four-entry abstention table
   (correct abstention 0.6 vs unverifiable extraction 0.4; no judge, no
   calls) — gold-NO is a teacher-miss-prone label, so engagement there is
   neutral, not punished. The historical gold-NO audit re-runs on the
   fiction10-gemma4 label before m1 (data.md job list); a penalty is an m2
   consideration at most.
