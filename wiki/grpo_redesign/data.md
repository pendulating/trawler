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
| abstracted norms (role abstraction) | ❌ **not built** | ❌ **not built** |
| `norm_universes.json` + embeddings | ❌ **not built** | ❌ **not built** |

**The universe builds are the gating prerequisite for the entire redesign** —
every reward module consumes the universe or its embeddings. Two-job chain
per corpus, both existing pipelines:

```bash
# 1) role abstraction over structured_norms
python -m dagspaces.historical_norms.cli pipeline=role_abstraction_standalone  # (gemma4 source override)
# 2) universe + Qwen3-Embedding-8B embeddings
ABSTRACTED_NORMS_PATH=<step-1 output> \
  python -m dagspaces.grpo_training.cli pipeline=norm_universe_only model=qwen3.5-9b/sft-contentless-v6
```

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

## Battery feasibility (measured 2026-07-16, lower bounds)

Grouping eligible norms by (book, **exact** context string):

| | fiction10-gemma4 | top100-gemma4 |
|---|---|---|
| eligible norms / gold-no | 2,711 / 295 (10.9%) | 14,991 / 1,765 (11.8%) |
| (book, context) clusters | 489 (median size **1**) | 4,025 (median **1**) |
| clusters ≥8 norms, force-mixed | **46** (1,840 norms; 206 gold-no) | 257 (8,238; 1,029) |
| clusters ≥4 norms, force-mixed | 64 | 387 |

Arithmetic for the canonical corpus: with K=8 and the 2-item minority target,
gold-no is the binding resource — 206 gold-no norms in mixed clusters ⇒
~**100 batteries** without norm reuse; at the ≥1 hard floor, ~200. A 0.3
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

1. `role_abstraction_standalone` on fiction10-gemma4 `structured_norms` →
   abstracted norms. Same for top100-gemma4 (holdout).
2. `norm_universe_only` on each → `norm_universes.json` + embeddings.
3. Re-measure on the **built universes** (abstraction dedupes): the
   battery-feasibility table above, and the per-force-class SFT-base
   accuracies (task-vignettes.md build item). Both go in
   `training_metadata.json` conventions. Also re-run the gold-NO audit
   (`scripts/audit_goldno_labels.py`) on the fiction10-gemma4
   `has_information_exchange` label — it gates whether `A-ABSTAIN`'s neutral
   treatment of gold-NO extractions stays warranted
   ([reward-abstain.md](reward-abstain.md)).
4. Null-answerability calibration pass for the probe pool
   ([reward-outcome.md](reward-outcome.md) step 5) — after 2, before the
   first cell launches.

Nothing else in the redesign can launch before these; they are CPU/1-GPU
jobs and can queue behind the SFT sweeps now.
