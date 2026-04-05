# SLURM & Environment

## `server.env`

Site-specific settings. Copy `server.env.example` → `server.env` and edit. Loaded automatically by `ensure_dotenv()` before Hydra resolves configs.

```bash
SLURM_PARTITION=pierson
TRAWLER_PROJECT_ROOT=/share/pierson/matt/UAIR
TRAWLER_VENV_ACTIVATE=/share/pierson/matt/UAIR/.venv/bin/activate

# NCCL: set to 1 for PCIe-only (no NVLink)
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_CUMEM_HOST_ENABLE=0
```

Plus pipeline input paths (set after running earlier phases):
```
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

`JUDGE_SERVER_URL` is the standardized env var across all dagspaces (see commit 474b694).

## SLURM launchers

`dagspaces/common/conf/hydra/launcher/*.yaml`. Override with `hydra/launcher=<name>`.

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
