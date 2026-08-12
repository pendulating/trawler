# Manuscript stale-content audit — DRY RUN

**Date:** 2026-08-08
**Scope:** every `.tex` file the `00_main.tex` build touches, plus the table files it `\input`s.
**Status:** items #1, #2, #3, #7 are **RESOLVED** (see the ruling below). Everything else is
still a proposal.

---

## RULING 2026-08-08 — one SFT sweep, no DFT

**Decision:** report the non-DFT (2026-07-15) SFT results, keep GRPO and KTO as they are, and
remove all DFT content from the paper.

This is the clean resolution because the 07-15 sweep is what *everything* reported descends from.
Verified from the frozen launch records: both `multirun/2026-07-28_grpo_m2_{full,core}/.../.hydra/
config.yaml` resolve `lora_path` to `2026-07-15_sft_canonical_gemma4/00-07-44/2/...`. The DFT switch
(2026-07-18) landed before those runs, but the alias `qwen3.5-9b/sft-canonical` was never repointed
— it was last modified 2026-07-19 and still targets the 07-15 artifact — so the RL arms trained on
the pre-DFT SFT regardless of date. Reporting the 07-15 rows therefore makes the SFT rows, the RL
arms' base, and the weight-delta analysis one lineage, with no cross-era claim anywhere.

**Applied:**
- `A_additional-methods.tex:68` — the "Two protocol eras" paragraph is gone, replaced by one
  sentence stating the objective actually used and the fact that every reported model descends
  from this sweep.
- `B_additional-results.tex` — removed "Three epochs do not overfit the extraction task" and
  "But in-format generalization and benchmark behavior come apart", both of which read the later
  sweep. Their surviving content (the OpenThinker3-7B format-contamination finding, which is
  measured on the reported adapter) is folded into one replacement paragraph.
- Removed `fig:sft-protocol` and `fig:sft-epochs` entirely: panels (b, c) of the first and all of
  the second are DFT-sweep data, and item #2 established that `fig:sft-epochs` measured 07-19
  adapters while the table reports 07-15 ones. The one citation to `fig:sft-protocol`(a) now points
  at `fig:sft-training`(b), which carries the same gradient-norm evidence; the exact clip
  percentages were already stated numerically in the prose, so nothing is lost.
- `B_additional-results.tex:739` — the stage-weight table row said **"SFT (DFT cross-entropy)"**.
  It analyses the 07-15 adapter, whose frozen config has no `loss_type`. Corrected to
  "SFT (cross-entropy)", and `stage_weight_deltas.py:103` fixed at the source: it was reading the
  live `conf/training/sft/default.yaml`, which gained `loss_type: dft` on 2026-07-18, *after* the
  run it describes.
- `normsim.bib` — `@inproceedings{piper2021narrative}` appeared twice, byte-identical, from merge
  `7617844`. This was breaking `bibtex` on any run that could not reuse a cached `.bbl`. Second
  copy deleted.

Build: clean, 67 pages, no new overfull boxes. A grep for `DFT`, `dynamic fine-tuning`,
`protocol era`, `earlier sweep`, and `later sweep` across the manuscript now returns nothing.

**Consequences to be aware of:**
- The paper no longer justifies the choice of three epochs; that argument rested on the held-out
  curves, which only existed under the later protocol.
- **Item #3 is untouched and still open.** The Gemma-4 chat-template defect is a property of the
  07-15 sweep, so keeping that sweep keeps the caveat on three reported rows, including the
  Gemma-4-12B row cited at `04_results.tex:29` as a 98.1% SOTA. This ruling does not resolve it.
- No CIRL-729 re-run is needed; the sweep drafted for it was deleted unused.

Two things prompted this pass: removing content that is simply obsolete, and removing the
manuscript's habit of narrating *its own history* ("an earlier variant…", "the later sweep…",
"two protocol eras…"). A camera-ready should state what was done, not how the protocol evolved,
**unless** a reported number actually depends on the older thing.

Items are graded:

| Grade | Meaning |
|---|---|
| **REMOVE** | Obsolete or self-narrating; nothing reported depends on it. Safe to cut. |
| **REWRITE** | Content is wrong or stale, but the underlying result is still needed. |
| **⚠️ FLAG** | **A reported result genuinely depends on older work.** Do not cut without a decision. |

---

## Summary

| # | Item | File | Grade |
|---|---|---|---|
| 1 | "Two protocol eras" SFT paragraph | `A_additional-methods.tex:68` | ✅ **RESOLVED** |
| 2 | `fig:sft-epochs` measures a *different* SFT lineage than Table S4 | `B_additional-results.tex:435` | ✅ **RESOLVED** (figure removed) |
| 3 | Gemma-4 chat-template defect caveat on three reported rows | `B_additional-results.tex:351-374` | ✅ **CLOSED** — removed per ruling; marginal effect |
| 4 | Table S4's `‡` pre-parser-fix marker | `tables/benchmark_results.tex:45` | **⚠️ FLAG** (small) |
| 5 | Reranker appendix: v9-era premise, wrong weights, wrong judge | `B_reranker_judge_ablation.tex` | ✅ **RESOLVED** |
| 6 | Judge model contradiction: Qwen3.6-27B vs Gemma-4-31B-it | `A_additional-methods.tex:36` | ✅ **RESOLVED** (+ PL judge, n=36→51) |
| 7 | Cross-era language in `fig:sft-protocol` caption | `B_additional-results.tex:423-426` | ✅ **RESOLVED** (figure removed) |
| 8 | "an earlier variant" (SFT negatives) | `A_additional-methods.tex:64` | **REMOVE** |
| 9 | "This replaces an earlier composition…" (reward nesting) | `A_additional-methods.tex:85` | **REMOVE** |
| 10 | "We deliberately deleted the appropriateness criterion this judge previously carried" | `A_additional-methods.tex:100` | **REMOVE** |
| 11 | "replaces a per-flow absolute variant…" | `A_additional-methods.tex:100` | **REMOVE** |
| 12 | Ablation cell names disagree across two tables | `t1_grpo_cell_summary` vs `benchmark_results_grpo` | ✅ **RESOLVED** |
| 13 | Dead legend entries (`‡`, `†`, `N/A`, `--`) in GRPO/KTO tables | `benchmark_results_{grpo,kto}.tex` | ✅ **RESOLVED** (legend now per-table) |
| 14 | Author-note macros | `05_discussion.tex:16,24` | **2 REMAIN — need your content** |
| 15 | `[TODO: cite]` / `\caption{TODO}` markers | 3 files | ✅ **RESOLVED** |
| 16 | Orphaned appendix subsections (never cross-referenced) | `B_additional-results.tex:31,47` | ✅ **RESOLVED** |
| 17 | Four evaluation traces defined, never cited | `C_evaluation_traces.tex` | ✅ **RESOLVED** (all four cited) |
| 18 | Orphaned table files on disk | `tables/` ×7 | ✅ **4 deleted, 1 wired in, 2 kept** |
| 19 | Commented-out template boilerplate and dead figure blocks | `00_main.tex`, `B_additional-results.tex:12-28` | ✅ **RESOLVED** |

---

## ⚠️ FLAG — reported results that depend on older work

### 1. The "Two protocol eras" SFT paragraph says the opposite of what you recalled

`A_additional-methods.tex:68`, full paragraph. You said *"the 'older SFT era' before we added DFT
does not need to be mentioned, since no results use it."* The manuscript asserts the reverse, and
the artifacts agree with the manuscript:

> The SFT rows of `\autoref{tab:benchmark_results}` come from a sweep that **predates** three
> protocol changes we made subsequently, and we report the **earlier** sweep rather than silently
> mixing eras.

Verified on disk:

| | reported (Table S4) | later |
|---|---|---|
| adapter | `multirun/2026-07-15_sft_canonical_gemma4/00-07-44/2` | `multirun/2026-07-19_sft_canonical_gemma4/00-11-59/0` |
| loss | TRL stock NLL | DFT (`loss_type: dft`) |
| held-out split | none | grouped, whole novels |
| chat template | Gemma-3 applied to Gemma-4 | per-family, verified |

`qwen3.5-9b/sft-canonical.yaml:11` points at the **07-15** artifact, and that adapter is what every
SFT row and every RL arm's base is built from. `notebooks/.../sft_training_diagnostics.py:94-98`
labels the two sweeps `reported` (07-15) and `repaired` (07-19).

**So the pre-DFT era is not vestigial: it is the reported SFT stage.** Cutting this paragraph would
make the paper silently mix two sweeps, which is the exact failure the paragraph exists to prevent.

**Decision needed.** Three options:
- **(a) Keep as-is.** Honest, but spends a paragraph on protocol archaeology.
- **(b) Compress to one sentence** naming only what a reader needs: the SFT rows use stock NLL with
  no held-out split and the last epoch kept. Drop the comparison to the later protocol, and
  drop the held-out curves that depend on it (see #2).
- **(c) Re-run.** Promote the 07-19 DFT sweep to the reported SFT stage, which retires this
  paragraph *and* items #2 and #3 at once. This is the only option that actually removes the old
  era rather than describing it more briefly, and it is a full re-eval of every SFT row.

### 2. `fig:sft-epochs` evaluates a different SFT lineage than Table S4, and does not say so

`B_additional-results.tex:435-449`. This is the more serious version of #1, because here the eras
**are** mixed and the paper claims they are not.

`A_additional-methods.tex:68` states the later sweep is used *"for that purpose alone"*, meaning the
held-out curve. That is not accurate. Traced through the notebook:

- `sft_training_diagnostics.py:60` sources the epoch trajectory from
  `sft_per_checkpoint_longitudinal_2026_07_20/cells.parquet`.
- That parquet's `sweep_dir` resolves to `multirun/2026-07-19_eval_sft_per_checkpoint_all`.
- Its overrides are `model=qwen3.5-2b/sft-canonical-ckpt171`, `…-ckpt513`, etc.
- `sft-canonical-ckpt513.yaml:13` → `multirun/2026-07-19_sft_canonical_gemma4/…/checkpoint-513`,
  and its own header says *"template-overhaul + DFT era"*.

So the figure that shows *what SFT buys on benchmarks* (+2.4 GoldCoin Appl., +3.8 Comp., +2.1
ConfAIde, −1.7 MMLU) is measured on **07-19 DFT adapters**, while Table S4's SFT rows are **07-15
pre-DFT adapters**. `benchmark_results.py:146` independently rules these "**not protocol-comparable**
with the keeper-era `sft-canonical` rows."

**Nothing in the figure caption or the surrounding prose discloses this.** The prose at
`B:384-403` reads as though it characterizes the reported SFT stage.

**Decision needed:**
- **(a) Disclose.** Add one sentence to the caption and amend `A:68`'s "for that purpose alone",
  which is currently a false statement.
- **(b) Cut `fig:sft-epochs`** and the paragraph at `B:384-403`. This loses the
  in-format/out-of-format dissociation argument, which is a genuinely good result.
- **(c) Re-run** the epoch trajectory on the 07-15 checkpoints so it matches Table S4.

### 3. The Gemma-4 chat-template defect caveat sits on three reported rows

`B_additional-results.tex:351-374`, and the claim is explicit:

> **The three Gemma-4 SFT rows of `\autoref{tab:benchmark_results}` come from this lineage and
> should be read with that in mind.**

The Gemma-4 SFT adapters in the reported 07-15 sweep were trained under the *Gemma-3* template,
whose delimiters are absent from the Gemma-4 vocabulary, then served under the native template.
This is corroborated by the saved note `project_gemma4_sft_template_bug` ("all pre-2026-07-18
gemma-4 SFT adapters … those SFT rows are INVALID") and by the training signal itself
(Gemma-4-12B: median gradient norm 6.00 vs 0.37–0.76, clipped on 100% of steps, initial loss 3.12
vs 0.86–1.38).

This is era content that **cannot** be deleted, because three reported rows depend on it — including
the Gemma-4-12B row that `04_results.tex:29` cites as setting a **SOTA of 98.1%** on GoldCoin
applicability.

**Decision needed.** Note the saved record calls these rows *invalid*, which is stronger than the
manuscript's "read with that in mind." Either:
- **(a)** Keep the caveat and soften the §4.3 SOTA claim, which currently presents 98.1% without it.
- **(b)** Re-run the three Gemma-4 SFT cells on the repaired 07-19 adapters and drop the caveat.
- **(c)** Withdraw the Gemma-4 SFT rows.

I did not touch `04_results.tex:29`; the SOTA sentence and the appendix caveat currently sit ~20
pages apart with no cross-reference between them.

### 4. Table S4's `‡` marker — small, real, and nearly resolved

You asked specifically about Table S4. `tab:benchmark_results` **is** S4 (`00_main.aux:235`).

The caption defines `‡: finalized before the PrivacyLens judge-parser fix`. Counting body cells
versus caption legend, `‡` appears on exactly **one row**: `tables/benchmark_results.tex:45`, the
Gemma-4-31B-it teacher/judge reference row, on its two PrivacyLens cells (Adj Lk 41.7, Helpful 51.9).

Your instinct is 23/24 correct. `benchmark_results.py:2838` confirms:

> **PrivacyLens (‡) — mostly RESOLVED.** The 2026-07-21 F1 rescue re-finalized 23/24 judged cells
> from the raw judge `output.jsonl`, clearing their ‡ automatically; the teacher row is the
> remaining un-rescued cell and still carries ‡.

The marker is computed automatically from file mtime against the fix date, so it is not stale text
— it is live and correct. The affected row is the self-judged reference row, already excluded from
bolding by `†`.

**Recommendation:** keep, or re-finalize that one cell to clear it. Do not delete the legend while
the marker is still emitted.

---

## REWRITE — wrong content, needed result

### 5. The reranker appendix is written against the v9 reward

`B_reranker_judge_ablation.tex`. This is a **live** appendix (`\input` at `00_main.tex:243`,
cross-referenced from `03_methods.tex:65`), and its opening paragraph is false under the m-series:

| Claim (`:23-26`) | Reality |
|---|---|
| "$R_{\text{ground}}$ … is the **dominant ($w{=}0.50$)**" | `R_ground` is $w{=}0.25$; the 0.50 core is `R_DIRECT` (`A:91`) |
| "and **only** LLM-judged reward component" | Two judged auxiliaries: `R_ground` **and** `R_contrast` (`A:100-102`) |
| "scored by the **Qwen3.6-27B** listwise judge" | Gemma-4-31B-it (`server.env:87`, `00_main.tex:213`) |

The file's own header comment (`:3-16`) already diagnoses this and says the fix is a first-paragraph
rewrite only. I agree: **the negative result stands** (best reranker $\rho=0.19$ over n=432 pairs)
and needs no re-run. Note the header calls the hot-path call `R-OUTCOME`, which is itself out of date
— see #12.

Also `:35` and `:37` render literal **`[TODO: cite Qwen3-Reranker / Qwen3-Embedding report]`** and
**`[TODO: cite DeBERTaV3 + the mDeBERTa-XNLI model]`** in the compiled PDF. The header says these
were resolved once and regressed through an Overleaf round-trip; the keys are
`zhang2025qwen3embedding`, `he2023debertav3`, `laurer_less_2024`.

### 6. Judge model: the paper contradicts itself

- `00_main.tex:213` (LLM disclosure): "$R_\text{ground}$ uses **Gemma4-31B** as a judge" ✅
- `A_additional-methods.tex:36`: "We use **Qwen3.6-27B** (dense) for reward judging" ❌
- `B_reranker_judge_ablation.tex:26`: "the **Qwen3.6-27B** listwise judge" ❌

Ground truth: `server.env:87` sets `JUDGE_MODEL_PATH=…/Gemma-4-31B-it`, and every benchmark table
caption says "all judged columns use Gemma-4-31B-it."

This is item #1 of `CONGRUENCE.md`, whose own superseded banner says the going-forward judge is
Gemma-4-31B-it and that "Qwen3.6-27B was the keeper-era (v9–v12a) judge only." The correction was
applied to `00_main.tex` and never propagated to Appendix A.

**Caveat worth checking before editing:** `A:36` also credits Qwen3-Embed-8B for embeddings, which
*is* still correct per `project_gemma_stack_migration` ("Qwen kept ONLY for embeddings"). Change the
judge clause only.

### 7. Cross-era language in the `fig:sft-protocol` caption

`B_additional-results.tex:423-426`:

> This measure stays inside one protocol **era** deliberately: the later sweep also changed the
> training objective … so a **cross-era** comparison of gradient magnitude would not isolate the
> template repair.

This is the paper explaining its own bookkeeping. Whether it can go depends entirely on #2: if the
held-out panels (b, c) are cut, the whole caveat goes with them. If they stay, the sentence should
be recast as a plain scope statement without the era vocabulary.

### 12. The same ablation cell has two different names in two tables

Not "old content", but it originates in the same naming drift and will confuse a reader who reads
both tables:

| m2 grid cell (`grpo_m2_grid.yaml`) | `tab:grpo-ablation` | `tab:benchmark_results_grpo` |
|---|---|---|
| `full` | `\textsc{Full}` | "GRPO (full stack)" |
| `core` | $-$`\textsc{aux}` | — |
| `minus_outcome` | $-$`\textsc{core}` | "$-$ R-OUTCOME" |
| `minus_vignette` | $-$`\textsc{judg}` | "$-$ T-VIGNETTE" |

`minus_outcome` is printed as **$-$core** in one table and **$-$R-OUTCOME** in the other. The root
cause is in the code: `modular_reward.py:159` keys the core's weight as `"outcome"` while
`core_mode` defaults to `"direct"`, so the term that runs is R-DIRECT. The paper calls it
$R_{\text{direct}}$ everywhere in `app:reward`. **Recommend standardizing on $-R_{\text{direct}}$
in both tables.**

Separately, `tab:benchmark_results_grpo`'s caption distinguishes "all six" from "full stack", but
`app:reward` documents three modules plus routing. Worth confirming what "six" counts before a
reviewer asks.

### 15. TODO markers that render in the compiled PDF

- `03_methods.tex:65` — "For flow judging, we the deontic `\matt{TODO}`." Also an incomplete
  sentence ("we the deontic").
- `B_additional-results.tex:36` — bare `\matt{TODO}` and `:39` `\caption{TODO}` on
  `fig:retrieval-concentration`, which `04_results.tex:69` cites.
- `B_reranker_judge_ablation.tex:35,37` — two `[TODO: cite]`.
- `04_results.tex:51` — `\caption{}` (empty) on `fig:norm-flow-map`, a **main-text** figure.

`\matt{}` renders bold red in the PDF, so these are visible.

### 16. Two appendix subsections nobody can reach

`app:normative-grounding` (`B:31`) and `app:embedding-space` (`B:47`) are live subsections with
labels that **no `\autoref` anywhere points to**. `tab:hub-norms`, `\input` inside the first, is
likewise never referenced. Either cite them from §4.3 or fold their content into the sections that
are cited.

---

## REMOVE — safe, nothing depends on it

### 8-11. Four "we used to do it differently" asides in Appendix A

None of these is load-bearing; each narrates a design iteration a reader does not need.

- **`:64`** — "…; **an earlier variant** that admitted every negative chunk, including those that
  contain an exchange but no governing norm, taught a marked over-abstention prior."
  → Cut after the semicolon. State the rule, not the rejected alternative.
- **`:85`** — "**This replaces an earlier composition** in which the appropriateness direction was a
  multiplier *inside* the grounding term and the contrastive penalty was a clamp inside it, a
  nesting under which no single component could be removed in isolation."
  → Delete. This is a description of the **v9 reward**, which is deprecated. The preceding sentence
  already states the modularity principle positively.
- **`:100`** — "We **deliberately deleted** the appropriateness criterion this judge **previously
  carried**, since that verdict is now scored by $R_{\text{direct}}$…"
  → Recast as "The rubric scores grounding only; the appropriateness verdict is scored by
  $R_{\text{direct}}$." Same information, no history.
- **`:100`** — "The listwise design **replaces a per-flow absolute variant** whose quantized scores
  left roughly 60% of small groups exactly tied with zero group-relative advantage."
  → Cut, or keep only as a forward-looking justification ("Listwise scoring avoids the ties that
  quantized per-flow scores produce in small groups").

### 13. Dead legend entries in the GRPO and KTO tables

Both captions were copied from Table S4 and define markers their bodies never use:

| Marker | `benchmark_results_grpo` | `benchmark_results_kto` |
|---|---|---|
| `‡` pre-parser-fix | 0 uses | 0 uses |
| `†` self-judged | 0 uses | 0 uses |
| `N/A` | 0 uses | 0 uses |
| `--` not reported | 0 uses | 0 uses |

Four dead sentences per caption, in two already-dense captions. (In Table S4 all four **are** used
— `N/A` ×6, `--` ×18 — so leave that caption alone.)

### 14. Author-note macros

`00_main.tex:151-153` defines `\hal{#1}` and `\madiha{#1}` as **no-ops that silently swallow their
argument**, and `\matt{#1}` as bold red inline text. Five live call sites:

| Site | Content |
|---|---|
| `04_results.tex:67` | `\matt{What do we do?}` — inside the `fig:norm-flow-map` methods sentence |
| `05_discussion.tex:16` | `\matt{(X, Y, and Z. Find citations)}` |
| `05_discussion.tex:24` | `\matt{fiction is suprising and rich...}` (also a typo) |
| `03_methods.tex:65` | `\matt{TODO}` |
| `B_additional-results.tex:36` | `\matt{TODO}` |

All five need **content decisions from you**, not deletion — `04_results.tex:67` in particular is an
unanswered methods question ("To make each book's space more visually comparable, we ⟨?⟩") in a
sentence describing a main-text figure. Worth grepping for `\hal{` and `\madiha{` before submission
too: because they expand to nothing, any prose left inside them has already vanished from the PDF
without a trace.

### 17. Four of five evaluation traces are never cited

`C_evaluation_traces.tex` defines `trace:pl-46`, `pl-71`, `pl-75`, `pl-117`, `pl-225`. Only
`trace:pl-46` is cited (`04_results.tex:35`). The other four are full-page figures with detailed
captions that no text points to. Either cite them where the qualitative argument is made, or cut.

### 18. Orphaned table files

Not `\input` by any build path:

| File | Note |
|---|---|
| `tables/reward_ablation.tex` | v9-era placeholder; I marked it dead on 2026-08-08. Superseded by `t1_grpo_cell_summary` + block 2 of `benchmark_results_grpo`. **Delete.** |
| `tables/lambda_rho_sweep.tex` | v9-era λ–ρ sweep; header carries `TODO(m-series): LEGACY TABLE`. **Delete.** |
| `tables/judge_human_agreement.tex` | Superseded by the figure in `app:judge-validation`. Verify before deleting. |
| `tables/sft_pair_ablation_main_effects.tex`, `…_per_variant.tex` | SFT pair ablation; never referenced. Confirm this result was dropped deliberately. |
| `tables/top100_scaling.tex` | Superseded by `tables/corpus_scaling.tex`. |
| `tables/zero-shot-vs-sft-cross-model.tex` | Superseded by `tables/benchmark_results.tex`. |

### 19. Commented-out blocks

- `00_main.tex:89-107, 114-134, 136-142` — COLM template boilerplate (Antiquus S. Hippocampus et
  al.) plus a superseded author block. Harmless, ~47 lines of noise.
- `01_intro.tex:8-15` — the commented-out `fig:overall_diagram` block, which now carries the
  m-series provenance note added 2026-08-08. Decide whether that figure is coming back; if not,
  delete the block and the note with it.
- `B_additional-results.tex:12-28` — two commented-out figure blocks
  (`fig:zero-shot-cross-benchmark`, `fig:zero-shot-vs-sft`) and the commented
  `\subsection{Zero-Shot Cross-Benchmark Ablation}`. Their captions still assert
  *"SFT consistently improves applicability detection … reduces information leakage"*, which is a
  stronger claim than §4.2 now makes. **Delete rather than leave for someone to un-comment.**

---

## What I checked and found clean

- **No `v9` remains** anywhere the build touches, apart from the deprecation notices added
  2026-08-08. `app:grpo-evolution` and `app:verdict-forensics` are gone; the λ–ρ and
  `reward_ablation` tables are orphaned, not rendered.
- **`app:reward` and `app:grpo-hyperparams` are m-series-faithful.** The formula matches
  `grpo_m2_grid.yaml`, and "three of the five, failing the first and the last" matches m2 `Full`'s
  gate strip in `tab:grpo-ablation` exactly.
- **The corpus-scaling noise-floor negatives** (`B:280-296`) are current results, not history. The
  "quality-screen improvement is not real" finding is a live negative result — keep.
- **Table S4's `†`, `×`, `∼`, `○`, `N/A`, `--` legends** are all live, with body uses.
- **The five undefined references** (`prompt:norm-reasoning-fiction` and four others) are a
  *separate* pre-existing bug: the labels exist in `prompts/*.tex`, but `E_prompts.tex` never
  `\input`s `prompts/all_prompts.tex`. Not stale content; a missing include. Worth fixing in the
  same pass.

---

## Suggested order

1. **Answer the four ⚠️ FLAGs first** — #1/#2/#3 are one decision (do the reported SFT rows stay on
   the 07-15 sweep?), and it determines how much of `A:68`, `fig:sft-epochs`, `fig:sft-protocol`,
   and the Gemma-4 caveat survives. Everything else is independent of it.
2. **Then the factual corrections** (#5, #6, #12), which are wrong-as-printed regardless.
3. **Then the safe removals** (#8-11, #13, #17-19), which are mechanical.
4. **Then the author notes** (#14, #15), which need your content.

Nothing in this document has been applied.


---

## RULING 2026-08-08 (second pass) — items 3, 5, 6, 12–19

**#3 Gemma-4 template defect — CLOSED, removed rather than disclosed.** Ruled a marginal effect not
worth the added complexity. The defect paragraph, its trace in the `fig:sft-training` caption, and
the sentence in the `tab:sft-training` caption naming the three affected rows are all gone.

**#5 reranker appendix — reframed as a judge-validity check.** Motivation is now that retrieval
hands the judge norms already lexically close to the flow, so a grounding score a relevance model
could reconstruct would carry no information beyond retrieval. Two model families failing to
reconstruct it is evidence `R_ground` scores something retrieval does not encode. Corrected
`w=0.50`→`w=0.25`, "only judged component"→one of two, and the judge name. Data and table untouched.

**#6 judge model — Gemma-4-31B-it everywhere.** Wider than first scoped. Beyond `A:36` and the
reranker caption, the *PrivacyLens* judge claims were also stale: `04_results.tex:35` and the whole
judge-validation section reported the adopted judge as Qwen3.6-27B at κ=0.79 on n=36. The notebook
had already established both wrong (n=51 on disk; production switched 2026-07-16), and the corrected
table existed as `tables/judge_human_agreement.tex` but was never `\input`. Swapped in; it brings
bootstrap CIs and exact McNemar tests. Revised: adopted κ 0.79→**0.65** [0.38, 0.87]; incumbent κ
0.47→**0.34**; incumbent misses **19 of 37** leaks. All movement is against the method.

**#12 cell names — standardized on the pre-registered grid.** `tab:benchmark_results_grpo` now uses
`Full` / `−aux` / `−core` / `−judg`, matching `tab:grpo-ablation`. The old `− R-OUTCOME` label was
doubly wrong: it named a different cell *and* a reward mode the runs never used (`core_mode=direct`).

**#13 dead legends — legend is now computed per table.** `_caption_tail_for()` emits only the
glossary clauses whose markers appear in that table's body. GRPO/KTO captions lose four dead
clauses each; the main table is unchanged because it uses all of them.

**GRPO/KTO tables condensed (new instruction).** The two-block "first batch / second batch" framing
is gone; each table is one grouping. Duplicated rows dropped: the ablation batch's own SFT base and
`Full` (GRPO) and its SFT base and label-only arm (KTO). Each duplicated pair agrees to within
**1.4 points** on every column, stated in the caption. Also removed "camera-ready" from both
captions and from the notebook's Phase C2 blurb.

**#14 — two author notes remain**, both needing content only you can supply:
`05_discussion.tex:16` `\matt{(X, Y, and Z. Find citations)}` and `:24`
`\matt{fiction is suprising and rich...}` (also a typo: *suprising*).

**#15, #16, #17, #19 — done.** TODO markers cleared; `app:normative-grounding` and
`app:embedding-space` now referenced from §4.3 and App A respectively; all four uncited traces cited
where the helpfulness-rubric failure is described, with `trace:pl-117` as the aligned-judges
contrast; the commented zero-shot block and 40 lines of COLM template boilerplate deleted.

**#18 — four deleted** (`lambda_rho_sweep`, `reward_ablation`, `top100_scaling`,
`zero-shot-vs-sft-cross-model`), all git-tracked so recoverable. `judge_human_agreement.tex` was not
dead but the corrected replacement, now wired in. **Kept:** `sft_pair_ablation_{main_effects,
per_variant}.tex`, a real SFT pair-format ablation never included in the paper. Decide whether that
result should be reported or the files dropped.

### Undefined `prompt:*` references — RESOLVED as a non-issue, fix is safe

Five `prompt:*` references are undefined because `E_prompts.tex` never `\input`s the prompt bodies.

**I briefly flagged this as a correctness problem and was wrong; retracted 2026-08-08.** The claim
was that norm extraction ran the *prescriptive* prompts while the paper cites the fiction ones. That
came from reading `prompt_reasoning` / `prompt_extraction` out of the **flows** run's
`.hydra/config.yaml`, where those keys are inert defaults for stages that pipeline never executes.
Checking each build against the stages it actually ran:

| build | stages run | prompts used |
|---|---|---|
| `2026-07-12_fiction10_norms_gemma4/18-36-28` (10,034 norms; `ABSTRACTED_NORMS_PATH`) | `norm_reasoning`, `norm_extraction` | `${prompt_norm_reasoning_fiction}`, `${prompt_norm_extraction_fiction}` ✅ |
| `2026-07-12_fiction10_flows_gemma4/23-14-17` (16,200 flows) | `ci_reasoning`, `ci_extraction` | `${prompt_ci_reasoning_fiction}`, `${prompt_ci_extraction_fiction}` ✅ |

Both used the fiction prompts for the stages they ran, and `03_methods.tex:36` cites them correctly.
The older `project_flows_prompt_bug` record describes a state that does not hold for these builds.

**What the actual fix requires** (still not applied, because it is not a one-line include):
- `prompts/all_prompts.tex` supplies `prompt:norm-reasoning-fiction` and
  `prompt:norm-extraction-fiction`, the two cited from `03_methods.tex:36`.
- It does **not** contain the three cited from `E_prompts.tex` (`prompt:extraction-instruction`,
  `prompt:vignette-battery`, `prompt:grpo-ground-judge`); those are standalone files under
  `prompts/`.
- It **does** contain four v9-era reward prompts (`grpo-reward-judge`, `grpo-no-flow-judge`,
  `grpo-ci-extraction`, `grpo-norm-judgment`) that contradict `E_prompts.tex`'s "three prompts in
  total" and should not be rendered.

So the fix is a selective set of `\input`s, not `\input{prompts/all_prompts}`.

**APPLIED 2026-08-08.** `E_prompts.tex` now `\input`s the five standalone files directly, under two
subsections: a new **Norm Extraction** subsection carrying `norm-reasoning-fiction` and
`norm-extraction-fiction`, and the existing **Training** subsection carrying
`extraction-instruction`, `vignette-battery`, and `grpo-ground-judge`.
`prompts/all_prompts.tex` is deliberately NOT included, so its four v9-era reward prompts never
render and no label is defined twice. All five references resolve (Prompts E.1--E.5); the build
reports **zero** undefined references for the first time.

Two rendering defects surfaced because these boxes had never actually been typeset before:
- A 62-character slash-joined token in `norm-extraction-fiction.tex` overflowed the margin. Fixed in
  the `prompttext` environment (`\scriptsize` plus `\sloppy`/`\emergencystretch`) rather than by
  editing the auto-generated file, which is marked DO NOT EDIT. Overfull count is back to the
  pre-existing 4.
- `promptbox` used tcolorbox's default white title on a `gray!50` title bar, i.e. white-on-light-gray.
  Set `coltitle=black`.

The appendix adds ~9 pages (68 -> 77).
