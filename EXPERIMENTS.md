# COLM 2026 Experiment Guide

Running guide for "Reinforcing privacy reasoning in LLMs via normative simulacra."

## Prerequisites

```bash
uv pip install -e .
source .env   # or ensure LD_PRELOAD and model paths are exported
```

All pipelines use SLURM via Hydra submitit. Add `hydra/launcher=null` for local execution. Add `runtime.debug=true runtime.sample_n=50` for quick debug runs.

## Phase 1: Norm Extraction (Historical Norms)

Extract Raz normative tuples from 10 fiction novels.

```bash
python -m dagspaces.historical_norms.cli pipeline=COLM_norms_fiction
```

**DAG**: fetch (chunk text) &rarr; reasoning (LLM norm analysis) &rarr; extraction (Raz tuples) &rarr; role_abstraction (character &rarr; social roles)

**Output**: `structured_norms.parquet`, `abstracted_norms.parquet`

**Model**: Qwen2.5-72B-AWQ (2-GPU, judge-quality extraction)

## Phase 2: Training

All training uses the `grpo_training` dagspace. Set env vars or pass Hydra overrides for input paths.

### Full pipeline (SFT + GRPO)

```bash
python -m dagspaces.grpo_training.cli pipeline=full_training model=qwen3-8b
```

**DAG**: norm_universe &rarr; sft_data_prep &rarr; sft_training &rarr; reward_prep &rarr; grpo_training

### SFT-only baseline

```bash
python -m dagspaces.grpo_training.cli pipeline=sft_only model=qwen3-8b
```

### GRPO-only (reuse existing SFT checkpoint)

Requires `SFT_CHECKPOINT_PATH`, `REWARD_CACHE_PATH`, `NORM_UNIVERSES_PATH`, `SFT_PAIRS_PATH` in `.env`.

```bash
python -m dagspaces.grpo_training.cli pipeline=grpo_only model=qwen3-8b
```

### Ablation: programmatic-only GRPO (no judge reward)

```bash
python -m dagspaces.grpo_training.cli pipeline=grpo_programmatic_only model=qwen3-8b
```

R_ground is zeroed out; reward weights redistributed to the 5 programmatic components.

### Large model: Qwen3.5-27B SFT

```bash
python -m dagspaces.grpo_training.cli pipeline=sft_only_27b model=qwen3.5-27b
```

Uses 4-GPU with QLoRA 4-bit quantization.

### Qwen3.5-9B variants

```bash
# SFT
SFT_CHECKPOINT_PATH=/share/pierson/matt/UAIR/outputs/2026-03-14/14-18-47/sft_only/outputs/sft/checkpoint \
  python -m dagspaces.grpo_training.cli pipeline=sft_only model=qwen3.5-9b

# GRPO (vLLM disabled — Qwen3.5 + vLLM + TRL has unresolved compat issues)
SFT_CHECKPOINT_PATH=<path> \
  python -m dagspaces.grpo_training.cli pipeline=grpo_only model=qwen3.5-9b training.grpo.use_vllm=false
```

## Phase 3: Benchmark Evaluations

Run each benchmark with the trained checkpoint. Override `model=` to swap between base/SFT/GRPO variants.

### GoldCoin-HIPAA (CI applicability + compliance)

```bash
python -m dagspaces.goldcoin_hipaa.cli pipeline=full_eval model=qwen3-8b-grpo-ci
```

Two parallel branches: applicability (214 cases) and compliance (107 cases). Metrics: accuracy, macro F1, per-class precision/recall, confusion matrix.

**W&B project**: `goldcoin-hipaa`

### PrivacyLens (QA probing + leakage judgment)

```bash
python -m dagspaces.privacylens.cli pipeline=privacylens_clean model=qwen3-8b-grpo-ci
```

QA probing across 3 axes (Subject/Vector/Target, 493 prompts each). Agent action generation + leakage judgment. Metrics: QA accuracy per axis, leakage rate.

**W&B project**: `privacylens-eval`

### VLM-GeoPrivacy (visual privacy, requires VLM)

```bash
python -m dagspaces.vlm_geoprivacy_bench.cli pipeline=mcq_eval model=qwen3-vl-8b-instruct
```

MCQ evaluation on geoprivacy image scenarios. Requires a vision-language model (Qwen3-VL or Qwen2.5-VL).

**W&B project**: `vlm-geoprivacy-bench`

## Shared Model Configs

All model configs live in `dagspaces/common/conf/model/` and are resolved via Hydra searchpath. Override on the CLI with `model=<name>`.

| Config | Model | Use |
|--------|-------|-----|
| `qwen3-8b` | Qwen3-8B | Base for training + eval |
| `qwen3-8b-sft-ci` | Qwen3-8B + SFT LoRA | Eval SFT baseline |
| `qwen3-8b-grpo-ci` | Qwen3-8B + GRPO LoRA | Eval GRPO model |
| `qwen3.5-9b` | Qwen3.5-9B | Base for training + eval |
| `qwen3.5-9b-sft-ci` | Qwen3.5-9B + SFT LoRA | Eval SFT baseline |
| `qwen2.5-72b-awq` | Qwen2.5-72B-AWQ | Judge / extraction |
| `llama3.1-8b-instruct` | Llama 3.1-8B | Baseline comparison |
| `llama3.3-70b-instruct` | Llama 3.3-70B | Baseline comparison |

## SLURM Launchers

All in `dagspaces/common/conf/hydra/launcher/`. Override with `hydra/launcher=<name>`.

| Launcher | GPUs | Typical use |
|----------|------|-------------|
| `slurm_monitor` | 0 | Orchestrator (submits child jobs) |
| `g2_slurm_cpu` | 0 | Data prep, fetch |
| `g2_slurm_gpu_1x` | 1 | Inference (8B/9B models) |
| `g2_slurm_pierson` | 2 | Inference (72B models) |
| `g2_slurm_pierson_4x` | 4 | Large inference |
| `g2_slurm_train_2x` | 2 | Training (small) |
| `g2_slurm_train_4x` | 4 | Training (SFT/GRPO) |

## W&B Querying

Runs are auto-tagged with `bench:<dagspace>`, `family:<model>`, `finetuned`/`base`, `task:<eval_task>`. Query examples:

```python
import wandb
api = wandb.Api()

# All finetuned Qwen3 runs on GoldCoin
api.runs("goldcoin-hipaa", filters={"tags": {"$in": ["family:qwen3", "finetuned"]}})

# Compare compliance task across models
api.runs("goldcoin-hipaa", filters={"config.eval_task": "compliance"})

# Find runs by checkpoint
api.runs("goldcoin-hipaa", filters={"config.checkpoint_name": {"$regex": "grpo"}})
```

## Paper Ablation Matrix

| Row | Model | Training | Pipeline |
|-----|-------|----------|----------|
| Zero-shot baseline | Qwen3-8B | None | `model=qwen3-8b` |
| SFT baseline | Qwen3-8B + SFT | `sft_only` | `model=qwen3-8b-sft-ci` |
| SFT + GRPO (full) | Qwen3-8B + GRPO | `full_training` | `model=qwen3-8b-grpo-ci` |
| GRPO w/o judge | Qwen3-8B + Prog-GRPO | `grpo_programmatic_only` | Custom model config |
| Large model | Qwen3.5-27B + SFT | `sft_only_27b` | Custom model config |

Each row is evaluated on all 3 benchmarks (GoldCoin, PrivacyLens, VLM-GeoPrivacy where applicable).
