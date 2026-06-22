# 2026-06-09 — W&B logging rationalization (grpo_training + historical_norms)

**Status:** in working tree. Follows the same-day code review
(`2026-06-09_code_review_norms_grpo.md`). Goal: every number needed to
audit a run lives on its W&B page, under a stable namespace, at a bounded
cadence — and stale/dead logging paths are gone.

## Namespaces (the contract)

| Namespace | What | Where it lands | Cadence |
|---|---|---|---|
| `rground/*` | Reward-health: judge failures, parse fails, no-flow rate, correct/wrong score means, embedding-server failures | metrics (merged into TRL's step commits via `commit=False`) | 1 set per reward call |
| `prescreen/*` | Screen outcome: n_in/kept/dropped, SFT no-flow rate, cache_hit | summary + full report in `config.prescreen` | once |
| `gates/*` | Promotion verdict: per-gate status + numbers, `gates/promote` | summary | once, end of training |
| `grpo_runtime` (config) | THE full `training_metadata.json` dict | config | once |
| `data_quality/*` | historical_norms per-stage QA: parse-error rates, label distributions, chunk-length stats | metrics | once per stage |
| `norm_universe/*` | Universe shape: books, total norms, dropped invalid/duplicate | metrics + `norm_universes` artifact | once |

## What changed

### grpo_training

- **`grpo_runtime` config now mirrors `training_metadata.json` exactly**
  (`stages/grpo_training.py`). The old hand-copied subset silently omitted
  every redesign knob (rground_scoring, reward_composition, n_screened_out,
  vignette counts, beta, …) — the W&B UI could not distinguish a
  ranked-gated cell from a May-recipe cell.
- **Promotion gates run automatically at the end of training** and write
  `promotion_gates.json` + `gates/*` summary keys. A failing cell is
  visible in the sweep table without running
  `scripts/check_grpo_promotion_gates.py` (which still works for re-checks).
- **`OnlineRGround` pushes `rground/*` health per reward call** (both
  absolute and ranked modes), kept on `self.last_health` for tests. Key
  signals: `judge_failed_group_frac`, `parse_fail_frac`, `no_flow_frac`,
  `mean_correct` / `mean_wrong`, `consecutive_zero_batches`,
  `embedding_consecutive_failures`. A judge that dies mid-run is now a
  visible step change, not a stdout line in a SLURM log.
- **Prescreen report logged** (`prompt_screening.py`): full report into
  `config.prescreen`, headline numbers into summary; report gains a
  `cache_hit` field.
- **`norm_universe` runner logs universe shape + a versioned
  `norm_universes` artifact** (per-book counts in artifact metadata) so a
  GRPO run can be traced to the exact universe build it trained against.
- **`reward_traces.jsonl` is size-capped** (`rewards.py`, default 256 MB,
  `trace_max_bytes`): when exceeded, the newest half is kept (whole
  lines). Gates only read the tail, so nothing they need is lost.
- **Stage jobs log a sampled output table for parquet datasets**
  (`orchestrator.py`) — this activates the previously dead
  `full_column_log_stages` config for `sft_data_prep` / `reward_prep`.

### historical_norms

- **New `stage_metrics.py` → `data_quality/*` scalars per stage**, logged
  by the orchestrator next to the existing sampled table:
  - `fetch_gutenberg`: chunk-length mean/p50/p95/max (a `chunk_len_max`
    over 6000 is the F3 alarm), rows/books.
  - `ci_reasoning`: parse-error rate, has-exchange rate, flows/chunk,
    zero-flow fraction.
  - `ci_extraction`: extraction-error rate, appropriateness distribution.
  - `norm_reasoning`: parse-error rate, prescriptive-content rate,
    norms/chunk.
  - `norm_extraction` / `norm_role_abstraction`: extraction/abstraction
    failure rates, `norm_quality_passed` rate, normative-force
    distribution, governs-info-flow rate, mean confidence.

### Both orchestrators

- **Human-readable W&B group per pipeline invocation**:
  `<experiment.name>-<timestamp>` when `wandb.group` / `WANDB_GROUP` are
  unset, propagated to stage SLURM jobs by the existing `WANDB_GROUP`
  export. Previously each job fell back to its own opaque `slurm-<jobid>`
  group, so a pipeline's runs never clustered in the UI.

### Architecture note (verified, no change needed)

TRL does **not** fork a second W&B run: `execute_stage_job` opens the
WandbLogger context before `runner.run()`, so TRL's `report_to="wandb"`
attaches to the already-active run. All `rground/*`, `prescreen/*`,
`gates/*` data therefore lands on the same run as TRL's reward/kl curves.

## Tests

`tests/historical_norms/test_stage_metrics.py` (9),
`tests/grpo_training/test_trace_cap.py` (2), plus `rground/*` health
assertions in `test_reward_improvements.py::TestRankedOnlineRGround`.
Full suite: 428 passed.
