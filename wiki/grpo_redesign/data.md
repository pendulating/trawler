# Data — corpora, universes, splits, feasibility

**Parent:** [README.md](README.md) · **Date:** 2026-07-16 · **Status:** drafted
· All counts below measured on the actual artifacts (commands preserved in
the session log of 2026-07-16); battery counts are **lower bounds** (exact
context strings, no clustering).

## Corpus decision (resolves the rest of master decision 3)

**Canonical grid trains on fiction10-gemma4.** Reasons: it is the corpus the
canonical SFT bases were just trained on (2026-07-15 sweeps), it matches the
published paper scope (10 novels), and `R-OUTCOME`'s probe anchor — the
teacher flows — must be the *same artifact* SFT supervised on, which for the
canonical checkpoints is the fiction10 teacher extraction. **top100-gemma4
serves as the book-level holdout** (below) and, optionally later, a scale
arm; it is not in the canonical grid.

Force balance no longer differentiates the corpora (the qwen-era rationale
died with the teacher swap):

| universe / extraction | books | eligible norms¹ | gold-no share |
|---|---|---|---|
| fiction10 **qwen** (Mar, keeper grounding) | 10 | 11,554 total | 15.1% prohibited |
| top100 **qwen** (06-30, v11 vignettes) | 97 | 21,510 total | 26.1% prohibited |
| fiction10 **gemma4** (pre-abstraction) | 10 | 2,711 | **10.9%** |
| top100 **gemma4** (pre-abstraction) | 100 | 14,991 | **11.8%** |

¹ eligible = quality-passed ∧ `governs_info_flow` ∧ decisive force ∧ non-empty
context — the vignette/probe-eligible pool, stricter than raw norm counts
(fiction10-gemma4 raw: 10,034).

## Artifact inventory and status

| artifact | fiction10-gemma4 | top100-gemma4 |
|---|---|---|
| chunk reasoning (`reasoning.parquet`) | ✅ `outputs/2026-07-12_fiction10_flows_gemma4/23-14-17/...` (2,993 chunks, 126 no-exchange) | ✅ `outputs/2026-07-13_top100_flows_gemma4/16-23-09/...` (15,875 chunks) |
| teacher flows (`ci_flows.parquet`) — SFT supervision + **probe anchor** | ✅ same run (16,200 flows) | ✅ same run (90,091 flows) |
| structured norms (`structured_norms.parquet`) | ✅ `outputs/2026-07-12_fiction10_norms_gemma4/18-36-28/...` | ✅ `outputs/2026-07-13_top100_norms_extraction_gemma4/16-23-09/...` (53,494) |
| abstracted norms (role abstraction) | **skipped by decision** (2026-07-17) | **skipped by decision** (2026-07-17) |
| `norm_universes.json` + embeddings | ✅ **built 2026-07-23** `multirun/2026-07-23_universe_fiction10_gemma4/15-43-41/norm_universe_only/outputs/norm_universe/` (10 books, 10,032 norms) | ✅ **built 2026-07-17** `multirun/2026-07-17_universe_top100_gemma4/17-41-33/norm_universe_only/outputs/norm_universe/` (100 books, 53,476 norms) |

**Universe builds: DONE (2026-07-23).** Role abstraction was deliberately
skipped — the gemma-4 extractor already emits name-free functional-role
subjects, so the qwen-era abstraction pass is redundant and its
name-stripping prompt would degrade them (rationale recorded in the header
of `scripts/run_universe_build_gemma4.sh`, review 2026-07-17). The build is
therefore a single job per corpus, structured_norms fed directly:

```bash
scripts/run_universe_build_gemma4.sh fiction10   # or top100
```

Two bugs fixed en route (2026-07-23; see the script log lineage): a
`numpy.bool_` JSON-serialization crash in `runners/norm_universe.py`
(manifests only when a parquet bool column has zero nulls — fiction10 but
not top100), and a torchcodec-breaks-`import sentence_transformers` landmine
on FFmpeg-broken nodes, shimmed via
`stage_utils.ensure_importable_sentence_transformers()`.

The only existing universes are **qwen-teacher** (March fiction10 = the
keeper's grounding; 06-30 top100 = v11's vignettes). They stay for legacy
reproducibility but must not supervise gemma4-based training — teacher
mismatch on the reward side.

## Who consumes what

| module | chunks | teacher flows | own-book universe | other-book universes | embeddings | judge/answerer |
|---|---|---|---|---|---|---|
| `T-EXTRACT` prompts | ● | | | | | |
| `R-VALID` / `A-ABSTAIN` | gold_has_exchange | | | | | |
| `R-OUTCOME` probes | | ● (queries) | ● (pool) | | ● | answerer |
| `T-VIGNETTE` batteries | | | ● (pool) | | ● (context clustering²) | |
| `R-GROUND` | | | ● (retrieval) | | ● | judge |
| `R-CONTRAST` | | | | ● (one sampled) | ● | judge |

² context clustering uses the small sentence embedder (`all-MiniLM-L6-v2`
class), not Qwen3-Embedding-8B.

## Splits (protocol item 1 — fixed before any run)

- **Chunk-level:** within fiction10, train / dev split of the prompt set
  (dev = held-out reward eval, the existing `dev_fraction` mechanism). The
  prompt set itself is small by design (protocol item 6) — pre-registered
  size N ≈ 500–800 chunks selected by the stratified prescreen from the
  2,993-chunk pool.
- **Book-level holdout:** fiction10 has only 10 books — holding any out is
  ruinously expensive for training. Instead the holdout is **the 93
  top100-gemma4 books not in fiction10** (measured overlap: 7 of 10 fiction10
  books are in top100 — gutenberg ids 11, 135, 145, 1023, 1184, 1342, 1399;
  fiction10-only: 541, 1984, 4078). Held-out books contribute **no chunks, no
  batteries, no probes** to training; they supply a zero-leakage
  generalization eval: extraction + probe-EM on never-seen normative
  universes. Requires the top100-gemma4 universe build (table above).
- Benchmarks (GoldCoin, PrivacyLens, ConfAIde, CIRL) remain fully zero-shot —
  no benchmark contact with training at any point.

## Battery feasibility (re-measured 2026-07-23 on the **built** universes)

Grouping eligible norms by (book, **exact** context string); measured by
`outputs/2026-07-23_mseries_premeasure/battery_feasibility.json` (the 07-16
pre-build lower bounds in parentheses where they differ — abstraction was
skipped, so the build only dropped exact duplicates and the numbers moved
marginally):

| | fiction10-gemma4 | top100-gemma4 |
|---|---|---|
| eligible norms / gold-no | 2,789 / 302 (10.8%) *(was 2,711 / 295)* | 15,049 / 1,774 (11.8%) *(was 14,991 / 1,765)* |
| (book, context) clusters | 502 (median size **1**) | 4,045 (median **1**) |
| clusters ≥8 norms, force-mixed | **46** (1,884 norms; 209 gold-no) | 259 (8,267; 1,034) |
| clusters ≥4 norms, force-mixed | 65 (237 gold-no) | 389 (1,229 gold-no) |
| batteries @ 2-minority target, no reuse | **~104** | ~517 |
| batteries @ ≥1-minority floor, no reuse | ~209 | ~1,034 |

Eligible force distribution (fiction10): obligatory 1,498 · recommended 989
· prohibited 264 · **discouraged 38** — the `discouraged` class is thin;
degree-mixed gold-no batteries will be mostly `prohibited`.

Arithmetic for the canonical corpus: with K=8 and the 2-item minority target,
gold-no is the binding resource — 209 gold-no norms in mixed clusters ⇒
~**104 batteries** without norm reuse; at the ≥1 hard floor, ~209. A 0.3
vignette mix over N = 500–800 prompts needs 150–240 battery rows. So
feasibility is **tight but workable**, with three levers in order of
preference:

1. **Context clustering headroom is huge** — median exact-string cluster size
   is 1, i.e. near-identical contexts are currently splintered; embedding
   clustering consolidates them and pulls singleton norms into usable
   mixed clusters. (This is why the numbers above are lower bounds.)
2. **Composition reuse:** the same norm may appear in more than one battery
   with different companions (batteries are distinct rows; norms repeat
   across epochs anyway). Reported as `battery_norm_reuse_factor`.
3. **Prompt-set sizing:** N = 500 needs only 150 batteries — and Memory-R1's
   152-example result argues small-N is a feature, not a compromise.

Decision recorded: hard floor **≥1 minority item per battery**, target 2;
realized per-battery composition reported (principle 6). top100 has no
feasibility problem (≥500 batteries at target composition) — relevant if a
top100 scale arm is ever run, and for holdout-eval battery construction.

## Prerequisite job list (blocking, in order)

1. ~~`role_abstraction_standalone`~~ — **skipped by decision 2026-07-17**
   (gemma-4 extractor already emits functional-role subjects; see script
   header).
2. ~~`norm_universe_only` on each~~ — **done** (fiction10 2026-07-23, top100
   2026-07-17; artifact table above).
3. Re-measure on the **built universes**: battery-feasibility table —
   **done 2026-07-23** (table above;
   `outputs/2026-07-23_mseries_premeasure/battery_feasibility.json`).
   Gold-NO audit re-run on the fiction10-gemma4 `has_information_exchange`
   label — **done 2026-07-23, PASS**
   (`outputs/2026-07-23_mseries_premeasure/goldno_audit/`): 126/2,993
   gold-NO (4.2%, as expected); advisory read of the 24 most suspicious
   cases found 0 clear missed flows (≤8% generous upper bound vs the <10%
   threshold in [reward-abstain.md](reward-abstain.md)) — m1 ships the
   neutral table; a mild m2 penalty is defensible pending hand-confirmation
   of `evidence_pack.md`. Caveat: the label is definitionally
   `ci_flow_count > 0`, so cross-parquet checks are consistent by
   construction; only reading chunks can surface teacher misses.
   Per-force-class SFT-base accuracies (task-vignettes.md build item) —
   **done 2026-07-23**
   (`outputs/2026-07-23_mseries_premeasure/sft_force_accuracy/`, use
   `summary_reparsed.json` — the first-pass qwen numbers were a
   truncation-parsing artifact). Headline (488 scenarios, 5-way force,
   greedy): gemma-4-12b force-acc 0.250 / polarity 0.418 / hedge 0.395 /
   antithesis 0.186 / mean-s 0.307; qwen3.5-9b 0.217 / 0.371 / 0.426 /
   0.203 / 0.266. Both beat the always-hedge mean-s baseline (0.193), so a
   positive commit gradient exists at init; the qwen-era gold-no≫gold-yes
   asymmetry is gone (gemma 0.415 vs 0.420 per-polarity). ~20% antithesis
   rate at init makes `antithesis_frac` the metric to watch. **Build-time
   lesson:** battery-output parsing must use json_repair or a larger
   completion budget — max_tokens truncation otherwise masquerades as
   hedging (355/488 qwen completions). All go in `training_metadata.json`
   conventions at build time.
4. Null-answerability calibration pass for the probe pool
   ([reward-outcome.md](reward-outcome.md) step 5) — **done 2026-07-24**
   (`outputs/2026-07-23_mseries_premeasure/probe_calibration/`; harness
   `scripts/run_probe_calibration.py`, answerer **gemma-4-31b** per the
   revised D1). Results, fiction10 probe pools built from real retrieval
   (top-3 per reference flow, mirrors R-GROUND's query distribution):
   - Candidate pools: 11,218 probe rows over 2,867 chunks; 1,199 unique
     probes; **158 chunks (5.5%) with empty pools** → excluded from
     T-EXTRACT at build (matches the "few percent" prior).
   - **Null-answerability drop rate: 0/1,199 = 0.0%** (both gold classes).
     Verified genuine, not a parse artifact: raw completions captured;
     5,995/5,995 votes parse; the answerer emits `cannot_determine` on the
     empty extraction every time. The filter is therefore a **no-op** for
     this answerer/probe design — every probe is extraction-dependent, the
     ideal Memory-R1-analog outcome. *Human-glance caveat:* the answerer
     never attempts world-knowledge answers at all; if a nonzero drop was
     expected as a sanity signal, consider whether the empty-extraction
     instruction is too strong before m2.
   - Realized K = min(4, pool): {1: 321, 2: 432, 3: 457, 4: 1,499}, mean
     3.16 over 2,709 retained chunks; stratification coverage 100% (both
     classes when available 788/788; ≥1 gold-no 839/839).
   - **14,464 retrieved candidate norms were leak-skipped** (mostly
     force-stem words like "obliged"/"permission" in narrative-derived norm
     fields) — conservative by design, but the skip distribution deserves a
     look before m1 if probe coverage feels thin anywhere.
   - Ops note: vLLM's custom all-reduce kernel crashes on pierson's PCIe
     A6000s at TP>1 (`custom_all_reduce.cuh 'invalid argument'`) — any
     direct `LLM()` use must set `disable_custom_all_reduce=True` (the
     shared `vllm_inference` util already does).

**All four prerequisite jobs are done (2026-07-23/24).** The redesign's
data side is cleared for m1; remaining blockers are the implementation
checklist components ([migration.md](migration.md) — probe builder ✅ built
2026-07-23 as `stages/probes.py`, items 2–7 outstanding).
