# SLURM & Environment

## `server.env`

Site-specific settings. Copy `server.env.example` → `server.env` and edit. Loaded automatically by `ensure_dotenv()` before Hydra resolves configs.

```bash
SLURM_PARTITION=pierson
TRAWLER_PROJECT_ROOT=/share/pierson/matt/UAIR
TRAWLER_VENV_ACTIVATE=/share/pierson/matt/UAIR/.venv/bin/activate

# NCCL: P2P/IB off for PCIe-only (no NVLink). SHM must stay ON — see below.
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
NCCL_SHM_DISABLE=0
NCCL_CUMEM_HOST_ENABLE=0
```

### NCCL: disable P2P, but **not** SHM

NCCL's intra-node transports are P2P (GPU-to-GPU), SHM (shared host memory), and
NET (TCP sockets). Turning off *both* P2P and SHM leaves only TCP — every
tensor-parallel all-reduce then round-trips through the kernel network stack.

Benchmarked on klara (8×A6000, PCIe, no NVLink) on 2026-07-12:

| all-reduce | P2P off + **SHM off** | P2P off + **SHM on** |
|---|---|---|
| decode-shaped (16 tok) | 0.245 ms | **0.069 ms** (3.5× faster) |
| prefill-shaped (8192 tok) | 33.2 ms | **16.9 ms** (2× faster) |
| peak bus bandwidth | 2.5 GB/s | **5.2 GB/s** |

Worth **21–29%** of end-to-end vLLM throughput. `NCCL_SHM_DISABLE=1` was a
long-standing misconfiguration (the launcher comment read "SHM transport can
hang on PCIe").

**P2P, however, genuinely hangs**: with `NCCL_P2P_DISABLE=0`, a 4-rank all-reduce
on klara never returns (TP=2 survives; TP=4 hangs silently until killed). So keep
P2P disabled. Enabling P2P also buys nothing when SHM is on — the topology is
PCIe host-bridge, not NVLink, and the two measure identically at TP=2.

Note the PCIe topology is asymmetric: GPUs 1/2/3 share a switch (`PIX`) while
GPU0 is a bridge hop away (`NODE`). A TP=2 replica on GPUs (0,1) runs ~5% slower
than one on (2,3) — minor, but it makes DP replicas finish out of step.

Plus pipeline input paths (set after running earlier phases):
```
GUTENBERG_CACHE_ROOT=/share/pierson/matt/zoo/datasets/gutenberg_cache
FICTION_CHUNKS_PATH=...
CI_REASONING_PATH=...
CI_EXTRACTION_PATH=...
ABSTRACTED_NORMS_PATH=...
SFT_PAIRS_PATH=...
NORM_UNIVERSES_PATH=...
SFT_CHECKPOINT_PATH=...
REWARD_CACHE_PATH=...
JUDGE_SERVER_URL=http://...
```

`GUTENBERG_CACHE_ROOT` is the durable on-disk cache for the Gutenberg corpus
tool (catalog snapshots, raw texts, per-book chunks). See
[howto/build-gutenberg-corpus.md](howto/build-gutenberg-corpus.md).

`JUDGE_SERVER_URL` is the standardized env var across all dagspaces (see commit 474b694).

## SLURM launchers

`dagspaces/common/conf/hydra/launcher/*.yaml` — or equivalently `launchers/*.yaml` via the root symlink. Override with `hydra/launcher=<name>`.

| Launcher | GPUs | Typical use |
|---|---|---|
| `slurm_monitor` | 0 | Orchestrator itself (submits child jobs) |
| `slurm_cpu` | 0 | Data prep, fetching |
| `slurm_cpu_beefy` | 0 | Heavy CPU (8 cores, 64 GB) |
| `slurm_gpu_1x` | 1 | 9B inference, LoRA serving |
| `slurm_gpu_2x` | 2 | 72B inference |
| `slurm_gpu_3x` | 3 | Large inference TP=2 |
| `slurm_gpu_4x` | 4 | 4-GPU inference |
| `slurm_gpu_5x` | 5 | — |
| `slurm_train_1x`..`4x` | 1–4 | SFT / GRPO training |

Per-node override inside a pipeline YAML:
```yaml
pipeline.graph.nodes.sft_training.launcher: slurm_train_2x
```

## ⚠️ Running a driver under `sbatch`: the `srun` trap

**If you submit a pipeline driver as an `sbatch` job, you must fix `PATH` or the GPU
stages silently run on the CPU.**

```bash
export PATH="$HOME/.local/bin:$PATH:/usr/local/slurm/current/bin"
```

Two independent traps, both hit on 2026-07-13 (see
`scripts/run_resume_top100_norms_extraction_gemma4.sh`):

1. **The SLURM clients are ssh-forwarding shims.** Compute nodes have the binaries but
   no `slurm.conf`/munge, so `~/.local/bin/{sacct,sbatch,scancel,sinfo,squeue}` are
   symlinks to `_slurm_ssh_shim`, which forwards to `unicorn-login-04`. An `sbatch` job
   does **not** get `~/.local/bin` on `PATH` — without it the driver dies immediately
   with `FileNotFoundError: 'squeue'`. Loud, at least.

2. **There is no `srun` shim — and this failure is silent.** submitit's backend
   detection is literally:

   ```python
   # submitit/slurm/slurm.py
   def affinity(cls) -> int:
       return -1 if shutil.which("srun") is None else 2
   ```

   With no `srun` on `PATH`, `AutoExecutor` scores SLURM as unavailable and **falls back
   to its LOCAL executor** — running the stage as a plain subprocess on whatever node the
   driver is on. No error, no warning. On 2026-07-13 this quietly launched
   **Gemma-4-31B on a GPU-less CPU node**; the only symptom was
   `CUDA_VISIBLE_DEVICES=''` buried in a stage log, and later a confusing
   `RuntimeError: Device string must not be empty` from deep inside vLLM.

   A login shell has `/usr/local/slurm/current/bin` on `PATH`, which supplies a real
   `srun` — which is why interactive runs work and `sbatch` drivers do not. Append that
   dir **last** so the working shims still win for `sbatch`/`squeue`; `srun` only needs
   to *exist* for the affinity check (the launchers set `slurm_use_srun=False`, so it is
   never executed).

`_create_submitit_executor()` in `dagspaces/common/orchestrator.py` now **raises** if
`srun` is missing, rather than degrading to CPU. Do not remove that check.

Diagnosing it after the fact: a real submitit SLURM job leaves a `*_submission.sh` in
`.slurm_jobs/<node>/` and its `job_id` appears in `sacct`. A local-fallback job leaves
neither — the "job id" is a PID.

## Local execution

Add `hydra/launcher=null` to run without submitit (good for debugging a single node). Combine with `runtime.debug=true runtime.sample_n=N` for sampled runs.

## GPU sanitization

Orchestrator probes each allocated GPU in a subprocess before stage execution:
- Broken GPUs are removed from the visible set
- Tensor parallelism is automatically adjusted

Bypass with `UAIR_SKIP_GPU_SANITIZE=1`.

## W&B integration

- Auto-tags: `bench:<dagspace>`, `family:<model>`, `finetuned` | `base`, `task:<eval_task>`
- Per-stage or single-run mode configurable
- Example queries:

```python
api.runs("goldcoin-hipaa", filters={"tags": {"$in": ["family:qwen3.5", "finetuned"]}})
api.runs("goldcoin-hipaa", filters={"config.checkpoint_name": {"$regex": "grpo"}})
```
