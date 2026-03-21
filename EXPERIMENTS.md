# COLM 2026 Experiment Guide

Running guide for "Reinforcing privacy reasoning in LLMs via normative simulacra."

## Prerequisites

```bash
uv pip install -e .
cp server.env.example server.env  # edit SLURM_PARTITION, paths, NCCL settings
# .env is auto-loaded for pipeline input paths
```

All pipelines use SLURM via Hydra submitit. Add `hydra/launcher=null` for local execution. Add `runtime.debug=true runtime.sample_n=50` for quick debug runs.

## Phase 1: Norm Extraction (Historical Norms)

Extract Raz normative tuples from 10 fiction novels.

### Fetch novels (run once)

```bash
python -m dagspaces.historical_norms.cli pipeline=COLM_fetch_fiction
```

Downloads and chunks all 10 novels from Project Gutenberg (6000-char chunks, 1000-char overlap), enriches with Wikipedia plot summaries. Output: `chunks.parquet`. Set `FICTION_CHUNKS_PATH` in `.env` to the output path.

### Extract norms (from prefetched chunks)

```bash
python -m dagspaces.historical_norms.cli pipeline=COLM_norms_fiction_prefetched
```

**DAG**: reasoning (LLM norm analysis) &rarr; extraction (Raz tuples) &rarr; role_abstraction (character &rarr; social roles)

**Output**: `structured_norms.parquet`, `abstracted_norms.parquet`

**Model**: Qwen2.5-72B-AWQ (2-GPU, judge-quality extraction with guided decoding)

### Full pipeline (fetch + extract in one run)

```bash
python -m dagspaces.historical_norms.cli pipeline=COLM_norms_fiction
```

## Phase 2: Training

All training uses the `grpo_training` dagspace. Set env vars or pass Hydra overrides for input paths.

### Full pipeline (SFT + GRPO)

```bash
python -m dagspaces.grpo_training.cli pipeline=full_training model=qwen3.5-9b
```

**DAG**: norm_universe &rarr; sft_data_prep &rarr; sft_training &rarr; reward_prep &rarr; grpo_training

### SFT-only baseline

```bash
python -m dagspaces.grpo_training.cli pipeline=sft_only model=qwen3.5-9b
```

### GRPO-only (reuse existing SFT checkpoint)

Requires `SFT_CHECKPOINT_PATH`, `REWARD_CACHE_PATH`, `NORM_UNIVERSES_PATH`, `SFT_PAIRS_PATH` in `.env`.

```bash
# Qwen3.5-9B (vLLM disabled — Qwen3.5 + vLLM + TRL has unresolved compat issues)
python -m dagspaces.grpo_training.cli pipeline=grpo_only model=qwen3.5-9b training.grpo.use_vllm=false
```

### Ablation: programmatic-only GRPO (no judge reward)

```bash
python -m dagspaces.grpo_training.cli pipeline=grpo_programmatic_only model=qwen3.5-9b training.grpo.use_vllm=false
```

R_ground is zeroed out; reward weights redistributed to the 5 programmatic components.

### Model size variants

```bash
# Qwen3.5-4B (small, fast iteration)
python -m dagspaces.grpo_training.cli pipeline=sft_only model=qwen3.5-4b

# Qwen3.5-27B (large model ablation, 4-GPU QLoRA)
python -m dagspaces.grpo_training.cli pipeline=sft_only_27b model=qwen3.5-27b
```

## Phase 3: Benchmark Evaluations

Run each benchmark with the trained checkpoint. Override `model=` to swap between base/SFT/GRPO variants.

### GoldCoin-HIPAA (CI applicability + compliance)

```bash
python -m dagspaces.goldcoin_hipaa.cli pipeline=full_eval model=qwen3.5-9b-sft-ci
```

Two parallel branches: applicability (214 cases) and compliance (107 cases). Metrics: accuracy, macro F1, per-class precision/recall, confusion matrix.

**W&B project**: `goldcoin-hipaa`

### PrivacyLens (QA probing + leakage judgment)

```bash
python -m dagspaces.privacylens.cli pipeline=privacylens_clean model=qwen3.5-9b-sft-ci
```

QA probing across 3 axes (Subject/Vector/Target, 493 prompts each). Agent action generation + leakage judgment. Metrics: QA accuracy per axis, leakage rate.

**W&B project**: `privacylens-eval`

### VLM-GeoPrivacy (visual privacy, requires VLM)

```bash
python -m dagspaces.vlm_geoprivacy_bench.cli pipeline=mcq_eval model=qwen3-vl-8b-instruct
```

MCQ evaluation on geoprivacy image scenarios. Requires a vision-language model.

**W&B project**: `vlm-geoprivacy-bench`

## Model Configs

Shared configs in `dagspaces/common/conf/model/`, resolved via Hydra searchpath. Override with `model=<name>`.

### Training & evaluation (Qwen3.5 family)

| Config | Model | Use |
|--------|-------|-----|
| `qwen3.5-9b` | Qwen3.5-9B | Primary base for SFT/GRPO + eval |
| `qwen3.5-9b-sft-ci` | Qwen3.5-9B + SFT LoRA | Eval SFT baseline |
| `qwen3.5-4b` | Qwen3.5-4B | Small model ablation (local in grpo_training) |
| `qwen3.5-27b` | Qwen3.5-27B | Large model ablation (local in grpo_training) |

### Extraction & judging

| Config | Model | Use |
|--------|-------|-----|
| `qwen2.5-72b-awq` | Qwen2.5-72B-Instruct-AWQ | Norm extraction + GRPO judge (2-GPU) |
| `qwen2.5-72b` | Qwen2.5-72B-Instruct | Unquantized variant |

### Zero-shot baselines (external models)

| Config | Model | Use |
|--------|-------|-----|
| `gpt-oss-20b` | GPT-OSS-20B (OpenAI) | MoE baseline (3.6B active, single GPU) |
| `phi-4` | Phi-4 (Microsoft, 14B) | Dense baseline |
| `phi-4-multimodal-instruct` | Phi-4-Multimodal (Microsoft) | VLM baseline |
| `gemma-3-12b-it` | Gemma 3 12B-IT (Google) | Dense baseline |
| `llama3.1-8b-instruct` | Llama 3.1-8B | Small baseline |
| `llama3.3-70b-instruct` | Llama 3.3-70B | Large baseline |

### VLM-specific (local to vlm_geoprivacy_bench)

| Config | Model | Use |
|--------|-------|-----|
| `qwen3-vl-8b-instruct` | Qwen3-VL-8B-Instruct | VLM eval |
| `qwen2.5-vl-7b` | Qwen2.5-VL-7B | VLM eval |

## SLURM Launchers

All in `dagspaces/common/conf/hydra/launcher/`. Override with `hydra/launcher=<name>`. Parameterized via `server.env` (partition, paths, NCCL).

| Launcher | GPUs | Typical use |
|----------|------|-------------|
| `slurm_monitor` | 0 | Orchestrator (submits child jobs) |
| `slurm_cpu` | 0 | Data prep, fetch |
| `slurm_cpu_beefy` | 0 | Heavy CPU tasks (8 cores, 64GB) |
| `slurm_gpu_1x` | 1 | Inference (9B models, LoRA) |
| `slurm_gpu_2x` | 2 | Inference (72B models) |
| `slurm_gpu_3x` | 3 | Large inference with TP=2 |
| `slurm_gpu_4x` | 4 | 4-GPU inference |
| `slurm_train_2x` | 2 | Training (small models) |
| `slurm_train_4x` | 4 | Training (SFT/GRPO) |

## W&B Querying

Runs are auto-tagged with `bench:<dagspace>`, `family:<model>`, `finetuned`/`base`, `task:<eval_task>`. Query examples:

```python
import wandb
api = wandb.Api()

# All finetuned Qwen3.5 runs on GoldCoin
api.runs("goldcoin-hipaa", filters={"tags": {"$in": ["family:qwen3.5", "finetuned"]}})

# Compare compliance task across models
api.runs("goldcoin-hipaa", filters={"config.eval_task": "compliance"})

# Find runs by checkpoint
api.runs("goldcoin-hipaa", filters={"config.checkpoint_name": {"$regex": "grpo"}})
```

## Paper Ablation Matrix

| Row | Model | Training | Pipeline |
|-----|-------|----------|----------|
| Zero-shot baseline | Qwen3.5-9B | None | `model=qwen3.5-9b` |
| SFT baseline | Qwen3.5-9B + SFT | `sft_only` | `model=qwen3.5-9b-sft-ci` |
| SFT + GRPO (full) | Qwen3.5-9B + GRPO | `full_training` | `model=qwen3.5-9b-grpo-ci` |
| GRPO w/o judge | Qwen3.5-9B + Prog-GRPO | `grpo_programmatic_only` | Custom model config |
| Small model | Qwen3.5-4B + SFT | `sft_only` | `model=qwen3.5-4b` |
| Large model | Qwen3.5-27B + SFT | `sft_only_27b` | `model=qwen3.5-27b` |
| GPT-OSS-20B | GPT-OSS-20B | None | `model=gpt-oss-20b` |
| Phi-4 | Phi-4 14B | None | `model=phi-4` |
| Gemma 3 | Gemma 3 12B-IT | None | `model=gemma-3-12b-it` |
| Llama 3.1 | Llama 3.1-8B | None | `model=llama3.1-8b-instruct` |
| Llama 3.3 | Llama 3.3-70B | None | `model=llama3.3-70b-instruct` |

Each row is evaluated on all 3 benchmarks (GoldCoin, PrivacyLens, VLM-GeoPrivacy where applicable).
