# How to run COLM norm + flow extraction on the top-100 fiction corpus

End-to-end plan for scaling the COLM norm-extraction flow from the original
10-novel curated set to the **top-100 fiction corpus** built per
[build-gutenberg-corpus.md](./build-gutenberg-corpus.md). All wall-clock
estimates below are scaled from a measured 10-novel run
(`outputs/2026-03-21_historical_norms/09-20-04/` for norms,
`outputs/2026-03-22_historical_norms/13-06-49/` for flows) and projected
onto the new model + parallelism config (see "Wall-clock estimate" below
for the derivation).

**Model + parallelism for this run**: every stage uses
`model=qwen3.6-27b/instruct` on 4 A6000s with `data_parallel_size=2 ×
tensor_parallel_size=2` (two replicas, each TP-sharded across 2 GPUs).
This is a model swap from the 10-novel baseline (qwen2.5-72b/awq at TP=2)
and a 2× parallelism bump. The three `*_qwen36.yaml` pipeline variants
described here pin this configuration so all three artefacts come from
the same judge.

## Inputs

| Artifact | Path |
|---|---|
| Chunks parquet | `/share/pierson/matt/zoo/datasets/gutenberg_cache/chunks_top100_fiction_en.parquet` |
| Selection YAML (lockfile) | `/share/pierson/matt/zoo/datasets/gutenberg_cache/selections/top100_fiction_en.yaml` |
| Materialize manifest | `/share/pierson/matt/zoo/datasets/gutenberg_cache/chunks_top100_fiction_en.manifest.json` |

Corpus stats: **100 books, 15,875 chunks** (6000-char chunks, 1000-char overlap).
Schema: `gutenberg_id, chunk_id, article_text, chunk_size, book_title, book_author, book_summary`. `book_summary` is empty (no
`--summaries-json` was passed at materialize time).

## The three pipelines

Two integrated DAGs plus one standalone — all consume `$FICTION_CHUNKS_PATH`.
The model **must be passed on the CLI** as `model=qwen3.6-27b/instruct`
(Hydra defaults order in `historical_norms/conf/config.yaml` loads
`pipeline` before `model`, so the pipeline yamls can't pin the model).

| Pipeline | Stages | Per-stage launcher | Output dir |
|---|---|---|---|
| `COLM_norms_fiction_prefetched_qwen36` | norm_reasoning → norm_extraction | slurm_gpu_4x ×2 | `…/COLM_norms_fiction_qwen36/outputs/{reasoning,extraction}/` |
| `COLM_flows_fiction_prefetched_qwen36` | ci_reasoning → ci_extraction | slurm_gpu_4x ×2 | `…/COLM_flows_fiction_qwen36/outputs/{ci_reasoning,ci_extraction}/` |
| `role_abstraction_standalone_qwen36` | norm_role_abstraction (single stage) | slurm_gpu_4x | `…/role_abstraction_standalone_qwen36/outputs/role_abstraction/abstracted_norms.parquet` |

**Why standalone role abstraction?** The prior 10-novel run executed exactly
this pattern: reasoning + extraction integrated, then role abstraction as a
separate pipeline reading the extraction parquet. This lets you iterate on
the role-abstraction prompt without paying the ~16h reasoning+extraction
cost again. Stage 3 is therefore **not in** the
`COLM_norms_fiction_prefetched_qwen36` YAML — run
`role_abstraction_standalone_qwen36` after extraction finishes.

The norms and flows pipelines are independent (no cross-pipeline deps) and
can run in parallel if GPU budget allows — at peak that's 8 A6000s held
simultaneously (4 per pipeline).

The 2-GPU sibling pipelines (`COLM_norms_fiction_prefetched`,
`COLM_flows_fiction_prefetched`, `role_abstraction_standalone`) still exist
and remain valid fallbacks if 4-GPU allocations are unavailable. They use
the default `qwen2.5-72b/awq` model.

## Pre-flight checklist

```bash
# 1. Bind the corpus path — MUST be written to .env, NOT just exported.
#
#    Gotcha: stage and orchestrator SLURM jobs source /share/pierson/matt/UAIR/.env
#    via ensure_dotenv() at process startup, AND the values in .env win over
#    the SLURM job's inherited environment for any var the shell didn't set.
#    A `export FICTION_CHUNKS_PATH=...` in your interactive shell is invisible
#    to slurm_monitor / slurm_gpu_4x jobs, so the on-disk .env value is the
#    actual source of truth. Edit it directly:
grep -n FICTION_CHUNKS_PATH /share/pierson/matt/UAIR/.env
# If the value is stale (e.g. points at outputs/2026-03-... legacy chunks),
# edit /share/pierson/matt/UAIR/.env in place to:
#   FICTION_CHUNKS_PATH=/share/pierson/matt/zoo/datasets/gutenberg_cache/chunks_top100_fiction_en.parquet
# Then for the current shell only:
export FICTION_CHUNKS_PATH=/share/pierson/matt/zoo/datasets/gutenberg_cache/chunks_top100_fiction_en.parquet

# 2. Sanity-check corpus (verifies the path .env points at, not just $FICTION_CHUNKS_PATH)
.venv/bin/python -c "
import pandas as pd, os
from dagspaces.common.stage_utils import ensure_dotenv
ensure_dotenv()
path = os.environ['FICTION_CHUNKS_PATH']
df = pd.read_parquet(path)
print(f'path  : {path}')
print(f'chunks: {len(df):,}  books: {df[\"gutenberg_id\"].nunique()}')
assert len(df) == 15875 and df['gutenberg_id'].nunique() == 100, 'corpus mismatch'
"

# 3. Smoke-test each pipeline locally on 20 chunks (~5 min each)
#    NOTE: local smoke test still loads the full model — needs 4 GPUs
#    accessible on the login/dev box. If you only have CPU/1-GPU dev access,
#    skip the smoke test and submit a small SLURM job with sample_n=20 instead.
python -m dagspaces.historical_norms.cli \
  pipeline=COLM_norms_fiction_prefetched_qwen36 \
  model=qwen3.6-27b/instruct \
  runtime.debug=true runtime.sample_n=20 \
  hydra/launcher=null

python -m dagspaces.historical_norms.cli \
  pipeline=COLM_flows_fiction_prefetched_qwen36 \
  model=qwen3.6-27b/instruct \
  runtime.debug=true runtime.sample_n=20 \
  hydra/launcher=null

# 4. Freeze the selection lockfile (so a stray regeneration doesn't change
#    the input mid-run)
git -C /share/pierson/matt/UAIR add \
  ../zoo/datasets/gutenberg_cache/selections/top100_fiction_en.yaml || true
# (or copy it into the repo if the cache dir isn't tracked)
```

## Execution

Open two terminals — pipelines run in parallel. Each holds 4 GPUs while
active, for a peak of 8 A6000s allocated.

```bash
# Terminal 1 — norms (reasoning + extraction)
python -m dagspaces.historical_norms.cli \
  pipeline=COLM_norms_fiction_prefetched_qwen36 \
  model=qwen3.6-27b/instruct

# Terminal 2 — flows (ci_reasoning + ci_extraction)
python -m dagspaces.historical_norms.cli \
  pipeline=COLM_flows_fiction_prefetched_qwen36 \
  model=qwen3.6-27b/instruct
```

Each invocation submits its stages sequentially to SLURM via `slurm_monitor`.
Outputs land in
`/share/pierson/matt/UAIR/outputs/YYYY-MM-DD_historical_norms/HH-MM-SS/`.

After **norms** finishes extraction (look for
`…/COLM_norms_fiction_qwen36/outputs/extraction/structured_norms.parquet`),
kick off role abstraction:

```bash
# Terminal 3 — role abstraction on the extraction output
export EXTRACTION_PARQUET=/share/pierson/matt/UAIR/outputs/<DATE>_historical_norms/<TIME>/COLM_norms_fiction_qwen36/outputs/extraction/structured_norms.parquet

python -m dagspaces.historical_norms.cli \
  pipeline=role_abstraction_standalone_qwen36 \
  model=qwen3.6-27b/instruct
```

## Wall-clock estimate

Projection assumes two compounding speedups vs the 10-novel baseline:

1. **Model swap qwen2.5-72b/awq → qwen3.6-27b**: ~2× faster (smaller model,
   memory-bound decode scales roughly inversely with param count). Hybrid
   mamba arch in qwen3.6 may shift this — confirm against the first stage's
   measured throughput.
2. **DP=2 → DP=2 for stages that were DP=1**: ~2× throughput (linear).
   `ci_reasoning` was already DP=2 in the 10-novel baseline, so it only
   gets the model-swap speedup.

| Stage | 10-novel measured | Effective speedup | 100-novel projection |
|---|---|---|---|
| norm_reasoning | ~4 h (TP=2, qwen2.5-72b) | ~4× | **~7 h** |
| norm_extraction | ~5 h (TP=2, qwen2.5-72b) | ~4× | **~9 h** |
| ci_reasoning | ~7 h (DP=2/TP=2, qwen2.5-72b) | ~2× | **~25 h** |
| ci_extraction | ~4 h (TP=2, qwen2.5-72b) | ~4× | **~7 h** |
| norm_role_abstraction (standalone) | <1 h (TP=2, qwen2.5-72b) | ~4× | **~2 h** |

- **Norms pipeline serial**: 7 + 9 = **~16 h**
- **Flows pipeline serial**: 25 + 7 = **~32 h** (bottleneck)
- **Role abstraction**: 2 h, runs after norm_extraction finishes

If norms + flows run in parallel: **~32 h wall** (≈ 1.3 days), bounded by
ci_reasoning. The model-swap speedup is the load-bearing assumption — if
qwen3.6 turns out only 1.3× faster than qwen2.5-72b at this workload,
the bottleneck stretches to ~40 h. Watch the first hour of ci_reasoning
throughput before committing to the projection.

## Monitoring

```bash
# SLURM queue for our jobs
squeue -u $USER

# Per-stage W&B runs (project: historical-norms-extraction)
# Each stage emits a child run tagged with wandb_suffix
#   reasoning / extraction / role_abstraction / ci_reasoning / ci_extraction

# Tail the live submitit log of an in-flight stage
tail -f outputs/<DATE>_historical_norms/<TIME>/.slurm_jobs/<stage>/*.out
```

## Post-run verification

```bash
RUN_DIR=outputs/<DATE>_historical_norms/<TIME>

# 1. Pipeline manifests show every stage completed
jq '.nodes | map({name: .name, status: .status})' \
  $RUN_DIR/COLM_norms_fiction_qwen36/pipeline_manifest.json
jq '.nodes | map({name: .name, status: .status})' \
  $RUN_DIR/COLM_flows_fiction_qwen36/pipeline_manifest.json

# 2. Row-count sanity (expectations scaled from 10-novel run)
.venv/bin/python <<'PY'
import pandas as pd
# Norms
r = pd.read_parquet("$RUN_DIR/COLM_norms_fiction_qwen36/outputs/reasoning/reasoning.parquet")
e = pd.read_parquet("$RUN_DIR/COLM_norms_fiction_qwen36/outputs/extraction/structured_norms.parquet")
print(f"reasoning rows : {len(r):,}  (expect ≈ 15,875)")
print(f"extraction rows: {len(e):,}  (expect ≈ 85,000 — exploded by raz_norm_count)")
# Flows
cr = pd.read_parquet("$RUN_DIR/COLM_flows_fiction_qwen36/outputs/ci_reasoning/reasoning.parquet")
cx = pd.read_parquet("$RUN_DIR/COLM_flows_fiction_qwen36/outputs/ci_extraction/ci_flows.parquet")
print(f"ci_reasoning   : {len(cr):,}  (expect ≈ 15,875)")
print(f"ci_extraction  : {len(cx):,}  (expect ≈ 45,000 — ratio ~6.4k/2.2k from 10-novel)")
PY

# 3. Failure-rate floors (sanity thresholds from 10-novel run)
#    - reasoning: <5% empty `reasoning_trace`
#    - extraction: <40% `extraction_failed` (10-novel hit ~30%)
#    - role_abstraction: <10% empty `abstracted_norm_articulation`
```

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `.env` overrides shell export for `FICTION_CHUNKS_PATH` | `ensure_dotenv()` is called inside every SLURM job at startup and seeds env vars from `/share/pierson/matt/UAIR/.env`. A `export FICTION_CHUNKS_PATH=...` in your interactive shell does NOT propagate to slurm_monitor / slurm_gpu_4x jobs, so the on-disk `.env` value is what actually drives the run. **Symptom**: pipeline reports an unexpected total prompt count (e.g. 2,993 instead of 15,875) — that's the prior corpus, not the current shell value. **Fix**: edit `.env` in place. Cross-check by reading the dataset path printed in each stage's manifest before the run goes long. |
| Corpus regenerated mid-run | Freeze `selections/top100_fiction_en.yaml` in git before kickoff. Do NOT re-run the gutenberg CLI until the COLM run finishes. |
| `book_summary` is empty | Known limitation of the corpus build. Norms stage still works (the 10-novel run had summaries; effect on quality is bounded by prompt design). If quality looks degraded, backfill summaries from Wikipedia and re-materialize before scaling further. |
| Stage dies at hour 30 | No mid-stage checkpointing. Restart from the last completed stage. Consider splitting the chunks parquet into two halves and running them as separate sources if a stage repeatedly OOMs or times out. |
| GPU contention | Each pipeline holds 4 GPUs (DP=2 × TP=2) per active stage. Worst case = **8 A6000s** held simultaneously when norms + flows run in parallel. If shared, stagger: kick off norms first, start flows after norm_reasoning finishes. Or fall back to the 2-GPU sibling pipelines if 4-GPU allocations are unavailable. |
| qwen3.6 speedup smaller than projected | Wall-clock projection assumes ~2× from the qwen2.5-72b → qwen3.6-27b model swap. Hybrid mamba arch may not deliver this. Measure the first hour of ci_reasoning throughput (target ~50 chunks/h per replica = ~100 chunks/h total) and adjust the rest of the schedule accordingly. |
| `EXTRACTION_PARQUET` path drift | Standalone role abstraction reads `$EXTRACTION_PARQUET`. Set it explicitly from the actual extraction output path (under `…_qwen36/outputs/extraction/`); don't rely on a default. |
| Mixed-model artefacts | All three qwen36 pipelines pin the same model (`qwen3.6-27b/instruct`) — do NOT mix-and-match with the 2-GPU sibling pipelines mid-run, as they default to `qwen2.5-72b/awq` and would produce artefacts from a different judge. |

## When to update this doc

- Bumping K from 100 → 1000: triple-check the wall-clock estimates; even with the qwen3.6 + DP=2 path, ci_reasoning at 1000 novels = ~250h and likely needs sharding.
- Adding `book_summary` backfill: document the summaries source and the new materialize command.
- Measuring actual throughput on the first stage: replace the projection table with measured numbers and update the model-swap risk row.
- Switching back to the qwen2.5-72b/awq path: revert to the 2-GPU sibling pipelines and double the wall-clock estimates back to the original 63h + 78h.
- If the integrated role_abstraction stage in `COLM_norms_fiction_prefetched_qwen36` is added back and shown to work: remove the `role_abstraction_standalone_qwen36` step.
