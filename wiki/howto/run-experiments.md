# How to run experiments

Quick-reference cheat sheet. Full matrix in `EXPERIMENTS.md`.

## Flags you'll always use

| Flag | Effect |
|---|---|
| `-m` | Use hydra-submitit launcher (SLURM) |
| `hydra/launcher=null` | Run locally (no SLURM) |
| `runtime.debug=true runtime.sample_n=50` | Debug mode on a sample |
| `UAIR_SKIP_GPU_SANITIZE=1` | Skip GPU probe |

## Phase 1 — Norm extraction (one-time)

```bash
# Fetch + chunk novels
python -m dagspaces.historical_norms.cli -m pipeline=COLM_fetch_fiction
# → set FICTION_CHUNKS_PATH in server.env

# Extract norms (Raz + role abstraction)
python -m dagspaces.historical_norms.cli -m pipeline=COLM_norms_fiction_prefetched
# → set ABSTRACTED_NORMS_PATH, CI_REASONING_PATH, CI_EXTRACTION_PATH
```

## Phase 2 — SFT

```bash
# Primary: Qwen3.5-9B
python -m dagspaces.grpo_training.cli -m pipeline=sft_only model=qwen3.5-9b/base

# Small ablation
python -m dagspaces.grpo_training.cli -m pipeline=sft_only model=qwen3.5-4b/base

# QLoRA (≥12B, single GPU)
python -m dagspaces.grpo_training.cli -m pipeline=sft_only \
  model=gemma-3-12b/it training/sft=sft_27b

# GPT-OSS-20B (dequantized, 2 GPUs via DDP)
python -m dagspaces.grpo_training.cli -m pipeline=sft_only \
  model=gpt-oss-20b/base training/sft=gpt_oss \
  pipeline.graph.nodes.sft_training.launcher=slurm_train_2x

# Cross-family
python -m dagspaces.grpo_training.cli -m pipeline=sft_only model=llama3.1-8b/instruct
python -m dagspaces.grpo_training.cli -m pipeline=sft_only model=phi-4/base
```

After SFT, the checkpoint lives at `multirun/<date>_grpo_training/<time>/sft_only/outputs/sft/checkpoint`. Create a `<family>/sft-ci.yaml` with `lora_path` set to it (or update existing).

## Phase 3 — Norm universe + reward cache (one-time)

```bash
python -m dagspaces.grpo_training.cli -m pipeline=norm_universe_and_reward_prep
# → set NORM_UNIVERSES_PATH, REWARD_CACHE_PATH
```

## Phase 4 — GRPO

```bash
# Full pipeline (SFT → GRPO end-to-end)
python -m dagspaces.grpo_training.cli -m pipeline=full_training model=qwen3.5-9b/base

# GRPO-only (reuse SFT checkpoint + cache)
python -m dagspaces.grpo_training.cli -m pipeline=grpo_only \
  model=qwen3.5-9b/base training.grpo.use_vllm=false

# Ablation: no R_ground
python -m dagspaces.grpo_training.cli -m pipeline=grpo_programmatic_only \
  model=qwen3.5-9b/base training.grpo.use_vllm=false
```

## Phase 5 — Benchmark evaluations

```bash
# GoldCoin-HIPAA
python -m dagspaces.goldcoin_hipaa.cli -m pipeline=full_eval model=qwen3.5-9b/sft-ci

# PrivacyLens
python -m dagspaces.privacylens.cli -m pipeline=privacylens_clean model=qwen3.5-9b/sft-ci

# VLM-GeoPrivacy
python -m dagspaces.vlm_geoprivacy_bench.cli -m pipeline=mcq_eval model=qwen3-vl-8b/instruct

# ConfAIde
python -m dagspaces.confaide.cli -m pipeline=<pipeline> model=qwen3.5-9b/sft-ci

# CI-RL Vignettes
python -m dagspaces.cirl_vignettes.cli -m pipeline=<pipeline> model=qwen3.5-9b/sft-ci
```

Swap `model=` to produce zero-shot / SFT / GRPO rows.

## Paper ablation matrix (Qwen3.5-9B)

| Row | Training | Pipeline | Eval `model=` |
|---|---|---|---|
| Zero-shot | — | — | `qwen3.5-9b/base` |
| SFT only | SFT | `sft_only` | `qwen3.5-9b/sft-ci` |
| SFT + GRPO | SFT→GRPO | `full_training` | `qwen3.5-9b/grpo-ci` |
| SFT + Prog-GRPO | SFT→Prog-GRPO | `grpo_programmatic_only` | custom yaml |

## Debug recipe

```bash
python -m dagspaces.goldcoin_hipaa.cli pipeline=full_eval \
  runtime.debug=true runtime.sample_n=5 \
  hydra/launcher=null model=qwen3.5-9b/base
```
