# m1 run plan — execution order, model scope, readiness gating

**Parent:** [README.md](README.md) · **Date:** 2026-07-24 · **Revised:**
2026-07-24 (rewritten for 4× training concurrency; 6-GPU budget = 4 policy
lanes + 2 judge/answerer) · **Status:** active plan · Cell definitions,
predictions, weights, and the reporting table are frozen in
[ablation-protocol.md](ablation-protocol.md); this page sequences the work
from the current state to a complete grid.

## GPU budget

**6 GPUs total: 2 pinned to the shared gemma-4-31b server** (TP=2; serves
BOTH the R-OUTCOME answerer and the R-GROUND judge — one deployment, two
roles) **+ up to 4 concurrent policy-training lanes** (1× A6000 each).
Freed lanes absorb benchmark evals as cells finish. Server discipline:
`judge_sidecar` global `max_inflight=16` (the KV-bound lesson); if answerer
latency visibly stalls ≥2 trainers, drop to 3 lanes before touching server
config.

## Current state (2026-07-24 morning)

**Done:** all four data.md prerequisite jobs (universes, feasibility,
gold-NO audit PASS, null-answerability calibration — 0% drop, 158 chunks
excluded, K̄=3.16); implementation checklist item 1 (`stages/probes.py`, 47
tests); answerer/judge = **gemma-4-31b** everywhere (revised D1).

**Not built:** migration.md items 2–7 (battery builder, answerer client,
`ModularReward`, deontic-distance scorer, stratified prescreen + m1 cache
signature, `training/grpo/m_series.yaml` + `conf/sweep/grpo_m1_grid.yaml`),
the keeper-guard test, and Gemma-4 support in the GRPO trainer (SFT got the
attn fix; the GRPO/vLLM-colocate path has zero gemma handling).

## Model scope (decision, overridable)

- **The grid runs once, on qwen3.5-9b `sft-canonical`** — the protocol's
  canonical base, the keeper lineage's family (v9-ckpt100 continuity), and
  the trainer path with three months of run history.
- **gemma-4-12b `sft-canonical` gets a confirmation cell, not a grid**: after
  the grid picks a winning cell, replicate that one cell on gemma-4-12b.
  Rationale: 2× grid cost buys little (the LOO questions are about the
  *reward*, not the base), but the paper's best-SFT story includes gemma, so
  the winning recipe must be shown to transfer.

## Phases

### Phase A — build the stack (items 2–7, blocking everything; no GPU)

Order chosen so each piece is testable without the next:

1. **Deontic-distance scorer** (axis map + `1 − |Δ|/2`, antithesis −1) +
   the 25-pair table test. Pure CPU, no deps.
2. **Battery builder** (context clustering with the small embedder,
   composition floors ≥1/target-2 minority, seeded composition; shares
   scenario templating with probes.py) + composition/leak/determinism tests.
   Parsing uses `json_repair` (the 07-23 truncation lesson).
3. **Answerer client** (gemma-4-31b; reuse judge plumbing; batched
   per-completion call, retry-then-group-neutral fallback; MUST set
   `disable_custom_all_reduce=True` on any TP>1 path — the 07-24 PCIe
   crash).
4. **`ModularReward`** (gate → A-ABSTAIN routing table → outcome +
   auxiliaries, 2:1 weight rule, per-module W&B namespaces) + routing/weight
   renormalization tests.
5. **Stratified prescreen** (task × gold class × force strata, `formula_version=m1`
   cache key incl. module list + task_mix + seeds) + signature tests.
6. **`training/grpo/m_series.yaml`** (optimizer preset copied verbatim per
   optimizer.md) + `conf/sweep/grpo_m1_grid.yaml` (7 cells).
7. **Keeper-guard test** (frozen-surface checksum regression) — owed since
   the 07-19 refactors touched keeper files.

All additive (parallel-stack rule); full suite green before phase B.

### Phase B — smokes (2 lanes in parallel, hours not days)

Server up first (2 GPUs), then concurrently:

- **Lane 1 — m0 core smoke** on qwen3.5-9b sft-canonical:
  `runtime.sample_n`-limited prompt set, ~20 steps, real answerer. Pass:
  probes load from built pools; answerer calls parse
  (`answerer_failed_frac` ≈ 0); `reward/outcome/*` namespaces stream;
  group_spread > 0 in most groups; gates script runs. Wiring bugs die here.
- **Lane 2 — gemma-4-12b trainer smoke** (phase-G prerequisite pulled
  forward): same m0 recipe on gemma-4-12b sft-canonical — exercises the
  gemma attn path, vLLM colocate, and LoRA keys under the GRPO trainer for
  the first time. Fixing gemma trainer bugs now means the confirmation cell
  launches without a stall later.

m0 results are never reported.

### Phase C — wave 1: four cells at once (the grid's branch-invariant set)

`core`, `full`, `−outcome`, `−vignette` — one per lane, simultaneously.
This is the key concurrency win: all four run in **every** non-kill branch
of the pre-registered logic (`−ground`/`−contrast` are the only conditional
cells), so launching them together spends no compute the branch logic could
have saved — *except* in the kill branch:

- **Accepted risk:** if `core` hits its kill criterion
  (`cannot_determine_frac` plateaus > 0.5 with gold-no EM flat), up to 3
  sibling cell-days are sunk. Accepted because the calibration evidence
  (100% extraction-dependent probes) and the SFT-base commit gradient
  (mean-s 0.31 > hedge 0.19) both point away from the kill branch.
- **Mitigation:** a mid-run forensics checkpoint at ~⅓ of training on
  `core` — if `cannot_determine_frac` is flat-high and `em_mean_by_force/no`
  is dead, abort the wave early and stop the grid (the honest-failure exit).

Gates (`scripts/check_grpo_promotion_gates.py`) per cell on completion;
benchmark evals launch on lanes as they free.

### Phase D — wave 2: branch-dependent fill (up to 4 lanes)

Composition depends on the wave-1 bracket read (`core` vs `full` on the
full eval matrix):

- **`full` > `core`:** lanes 1–2 = `−ground`, `−contrast`; lanes 3–4 =
  `core` extra seeds ×2 (phase F pulled in — they are branch-independent
  once `core` survives).
- **`core` ≈ `full`:** `−ground`/`−contrast` are skipped (answer implied);
  lanes run `core` seeds ×2 + the gemma-4-12b confirmation cell immediately
  (winner = `core` by parsimony, per the simplicity result).
- In the `full`-branch case the gemma confirmation cell launches as soon as
  a wave-2 lane frees and the winner is unambiguous.

### Phase E — offline checks (server + freed lanes, no policy training)

1. `+answerer-B`: re-score a sample of `full`'s traces with a **non-Gemma**
   answerer (Qwen3.6-27B), Spearman ≥ 0.8 gate. One lane, brief.
2. Held-out books eval (93 top100-only books): extraction + probe-EM for
   every surviving cell — requires probe pools built on the top100 universe
   (same harness, `scripts/run_probe_calibration.py` pointed at the top100
   artifacts; the null filter is currently a no-op so this is build-only).
3. `sft` / `0-shot` baseline rows — eval-only, cheapest rows; schedule
   opportunistically on any freed lane from phase C onward.

### Phase F — seeds

Absorbed into wave 2 (above). `full` quotes the existing 5-seed CV (3.5%);
`core` gets +2 seeds; nothing else is re-seeded.

### Phase G — gemma-4-12b confirmation cell

Trainer risk was retired in phase B lane 2; the cell itself runs in wave 2
(or immediately after) once the winning cell is unambiguous. Reported as a
transfer row, not a grid row.

**2026-07-24 — TP=2 colocate lane wired (Phase-G trainer unblock).** The m0
lane-2 smoke found gemma-4-12b loads/merges/builds the modular dataset with
zero gemma-specific errors but OOMs the 1-GPU colocate lane at the vLLM
sleep/wake boundary — weights-bound (~24 GB training + ~24 GB engine on one
48 GB A6000; identical OOM across gpu_mem 0.45→0.30 and max_len 16384→8192).
Fix wired (all additive; keeper surfaces untouched): (1) a model-agnostic
`disable_custom_all_reduce` injection into TRL's colocate `vllm.LLM.__init__`
(`grpo_training.py`) — MANDATORY at TP>1 on this PCIe/P2P-disabled cluster,
mirroring `common/vllm_inference.py`; TP=2 itself already reaches TRL via
`GRPOConfig.vllm_tensor_parallel_size`. (2) `conf/training/grpo/m_series_2gpu.yaml`
composing `m_series` + TP=2 + the all-reduce knob + gemma memory profile
(gpu_mem 0.40, max_len 16384); the existing `slurm_train_2x` launcher (2 GPUs,
pierson, PCIe NCCL env) is reused. TP=2 halves the engine weight shard to
~12 GB/GPU, dropping the wake peak to ~36 GB. (3) A gemma-4 weight-sync branch:
**analysis shows gemma needs NO manual `_push_param_to_vllm` prefix** — unlike
qwen, `AutoModelForCausalLM` returns the full composite
`Gemma4UnifiedForConditionalGeneration` (names already `model.language_model.*`,
tied embeddings ⇒ no `lm_head`), and vLLM's Gemma4Unified `hf_to_vllm_mapper`
already maps `model.language_model.`→`language_model.model.`. Copying qwen's
prefix would double-prefix and break it. The branch instead validates the
composite and installs a one-shot sync diagnostic (names passed through). Full
`tests/grpo_training/` green (548). **Smoke (job 364491)** confirmed all three
patches fire — merge loads the composite `Gemma4UnifiedForConditionalGeneration`,
modular stack `auxiliaries=[], reward_core=True, weights={'outcome': 1.0}`,
MiniLM embedder loads LOCALLY (the down klara:8001 embedding server does NOT
block the core-only recipe), and the run reached "Starting GRPO" — but then TRL
raised `tensor_parallel_size (2) must divide world size (1) evenly`. **This is
the real Phase-G plumbing gap:** TRL colocate vLLM at TP>1 requires the
accelerate world size to equal the TP degree, i.e. the trainer must run as
`tp` distributed PROCESSES, not one. Fixed by a 4th additive piece mirroring
the SFT DDP path: `runners/grpo_training.py` now spawns
`accelerate launch --num_processes <tp>` (via new
`stages/_grpo_accelerate_entry.py`) whenever TP>1 colocate + ≥tp GPUs, so
world_size==tp; the single-GPU keeper path stays the byte-identical direct
call (gated on TP>1). The stage's LoRA-merge disk writes are made rank-local
when `WORLD_SIZE>1` (barrier-free, since the torch PG isn't up pre-trainer).
Re-submitted as a second smoke (`_phaseG_gemma2gpu_smoke2`). **Verdict pending
the re-run** — see the session report for the job id and train-past-step-0
read; residual multi-rank race points (prescreen cache, reward_traces append)
are noted there as follow-ups if they surface.

## Standing rules (inherited, not new)

- Gates before eval spend (`scripts/check_grpo_promotion_gates.py`).
- The reporting table in ablation-protocol.md is the pre-registration;
  deviations get dated entries there.
- T=1.0 rollouts / greedy eval; optimizer preset never varies per cell.
- Every cell's realized prompt mix / battery composition lands in
  `training_metadata.json` (principle 6).
- One shared server for all cells (answerer + judge identity per cell is
  part of the reward definition and never varies).

## Envelope (4-lane)

Phase A is the long pole (build + tests, no GPU). Training wall-clock:
wave 1 ≈ **1 GPU-day** (4 cells in parallel), bracket evals overlap wave-2
launch, wave 2 ≈ 1 further day. **Grid + seeds + gemma confirmation ≈ 2–3
calendar days after phase B clears** (vs 6–7 sequential), at the same total
GPU-day cost. The gemma-4-31b server runs continuously through phases B–E;
it is the single shared dependency — treat its uptime as part of every
cell's provenance.
