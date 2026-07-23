# Local ↔ W&B parity

The 2026-07-20 parity pass makes local `metrics.json` files and W&B two
redundant copies of the same record — each restorable from the other —
instead of two loosely-related logging systems.

## Why it exists

Analyzing the 2026-07-19 per-checkpoint SFT sweep exposed three gaps:

1. **W&B runs could not be scoped to a multirun dir.** The `eval_all_run:`
   tag and resumable run ids only existed when `WANDB_GROUP` happened to be
   exported at launch, which it wasn't. Analysis had to fall back to
   lora-path substrings plus timezone-converted created-at cutoffs.
2. **Key drift.** Every dagspace hand-curated a `_log_eval_metrics`
   formatter that cherry-picked and *renamed* keys (disk
   `qa_probing.accuracy` → W&B `compute_metrics/eval/qa_accuracy`). Curated
   key lists rot: the cirl trajectory formatter's list predated the metrics
   schema and silently logged almost nothing for months.
3. **No provenance parity.** The served judge lives in judge-batch
   manifests on disk; W&B had only the config value — which is a lie (the
   stale `${oc.env:JUDGE_MODEL,...}` default).

## The contract

`dagspaces/common/metrics_sync.py`, wired into the shared eval loop in
`common/orchestrator.py` (`_mirror_stage_metrics`, called at both
`log_eval_metrics` sites). Applies to every dagspace using the shared loop
— no per-dagspace code.

| Guarantee | Mechanism |
|---|---|
| Every numeric leaf of metrics.json is in W&B under **byte-identical dotted keys** | `<subdir>/metrics_json/<dotted.key>` summary keys, where `<subdir>` is the metrics.json parent dir (`compute_metrics_tier2b`, …). `flatten_numeric` walks the dict mechanically — no key list to rot. Provenance counts (`metric_provenance.*`) are included. |
| W&B holds a **byte-exact copy** of the file | `wandb.save` with the `outputs/` root as `base_path`, so the run stores `compute_metrics_tier2b/metrics.json` etc. |
| Disk knows its run | `wandb_run.json` sidecar next to each metrics.json (entity/project/run id/url/group/tags). Re-runs into the same dir append to `previous_runs` instead of orphaning the old linkage. |
| The run knows its disk location | run config `local_output_dir` + `local_sweep_dir`. |
| Every run is scopeable to its sweep dir | `WANDB_GROUP` fallback: when neither cfg nor env provides a group, it is derived from the hydra output path (`multirun/<date>_<name>/<HH-MM-SS>/…` → `<date>_<name>/<HH-MM-SS>`). eval_all additionally pins the derived group into every child benchmark's env and the judge sidecar. Filter in W&B by group or the `eval_all_run:<group>` tag. |
| Judge provenance travels | `judge:<model>` run tags read from `*judge*batch*/manifest.json` — never from config. |

Curated legacy keys (`<stage>/eval/...`) are **unchanged** — dashboards and
`fetch_wandb_runs.py` keep working. The mirror namespace is additive.

Side effect to know about: deriving a group where none existed means
`wandb.single_run` + `pipeline_run_id` now engage by default — all stages
of one pipeline (and its async finalize) collapse into a single resumable
W&B run per (sweep, dagspace, model), as originally designed, instead of
per-stage runs.

## The sync tool

`scripts/wandb_local_sync.py` — the two-way backup workflow:

```bash
# Read-only parity report (per-value: in-parity / missing / mismatch / unlinked)
python -m scripts.wandb_local_sync verify multirun/<sweep>/<time>

# Backfill W&B from disk: resumes the sidecar-linked run when present,
# else creates a run tagged `backfill` with group/bench/model identity
python -m scripts.wandb_local_sync push multirun/<sweep>/<time> [--dry-run]

# Restore metrics.json trees from W&B (never writes into an existing
# multirun dir — restores under --dest, with ORIGIN.txt pointers)
python -m scripts.wandb_local_sync pull --group "<sweep>/<time>" --dest restored/
```

`push` is the recovery path for runs that predate this pass, ran with
`WANDB_DISABLED`, or whose logger crashed; `pull` is the recovery path for
local data loss.

## Scope and non-goals

- Applies to the **shared eval loop** dagspaces (goldcoin_hipaa,
  privacylens, confaide, cirl_vignettes, vlm_geoprivacy_bench, mmlu,
  simpleqa_verified, …). `grpo_training` / `historical_norms` don't emit
  benchmark metrics.json and are out of scope.
- The mirror never *reads* W&B at eval time — disk stays the source of
  truth; W&B is the redundant copy.
- Tables/parquets are not mirrored (size); `metrics.json` + scalars are the
  backup contract. Use `log_artifact` explicitly for anything bigger.

Tests: `tests/common/test_metrics_sync.py`.
