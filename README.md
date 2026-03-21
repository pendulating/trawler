# Trawler

Experiment infrastructure for *"Reinforcing privacy reasoning in LLMs via normative simulacra"* (COLM 2026). Implements the full pipeline from norm extraction through fine-tuning to benchmark evaluation.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Paper Summary

LLM agents handle personal information but lack principled privacy reasoning. We teach LLMs to reason about privacy using Helen Nissenbaum's [Contextual Integrity](https://en.wikipedia.org/wiki/Contextual_integrity) (CI) theory, which defines privacy as the appropriate flow of information within social contexts.

**Key insight:** Fiction novels depict fully-realized societies with rich normative landscapes governing who may share what information with whom. We extract structured *normative simulacra* from these texts — CI information flow tuples paired with Raz-anatomy norms — and use them to fine-tune LLMs in two stages:

1. **SFT** teaches the model the vocabulary and format of CI-grounded reasoning
2. **GRPO** (Group Relative Policy Optimization) rewards reasoning that is structurally complete, internally coherent, and *grounded* in an explicit normative universe verified by an LLM judge

The claim: SFT alone teaches *mimicry* of CI output; GRPO teaches actual *reasoning* about the relationship between information flows and the norms that govern them.

### Research Questions

- **RQ1:** Can an LLM learn contextual privacy reasoning from structured reasoning traces extracted from fiction?
- **RQ2:** Does reinforcement learning with a normatively-grounded reward improve privacy reasoning beyond SFT alone?
- **RQ3:** Does fine-tuning on a divergent normative universe (e.g., dystopian text) introduce measurable normative bias?

### Evaluation

Three CI-aligned benchmarks spanning distinct societal contexts:

| Benchmark | Domain | What it tests |
|-----------|--------|---------------|
| [GoldCoin-HIPAA](https://arxiv.org/abs/2309.11500) | Healthcare | CI applicability + compliance on real court cases |
| [PrivacyLens](https://github.com/SALT-NLP/PrivacyLens) | Corporate/social | QA probing (subject/vector/target) + agent leakage |
| [VLM-GeoPrivacy](https://arxiv.org/abs/2411.15087) | Visual geolocation | Location disclosure granularity from images |

---

## Pipeline Overview

```
Fiction novels (10 texts)
    │
    ▼
┌─────────────────────┐
│  Norm Extraction     │  historical_norms dagspace
│  (Qwen2.5-72B-AWQ)  │  Chunk → Reason → Extract → Role Abstraction
└─────────┬───────────┘
          │  normative simulacra
          ▼
┌─────────────────────┐
│  SFT Fine-tuning    │  grpo_training dagspace
│  (LoRA on Qwen3-8B) │  CI reasoning traces as instruction-following data
└─────────┬───────────┘
          │  SFT checkpoint
          ▼
┌─────────────────────┐
│  GRPO Training      │  grpo_training dagspace
│  6-component reward  │  R_uncert + R_complete + R_consist + R_context + R_cohere + R_ground
└─────────┬───────────┘
          │  GRPO checkpoint
          ▼
┌─────────────────────┐
│  Benchmark Evals    │  goldcoin_hipaa / privacylens / vlm_geoprivacy_bench
│  3 CI benchmarks    │
└─────────────────────┘
```

---

## Architecture

```
dagspaces/
├── common/                      # Shared framework code
│   ├── conf/
│   │   ├── model/               # Shared model configs (Hydra searchpath)
│   │   └── hydra/launcher/      # Shared SLURM launchers
│   ├── orchestrator.py          # DAG execution, SLURM dispatch, artifact tracking
│   ├── vllm_inference.py        # vLLM inference (LoRA, think-block stripping)
│   ├── wandb_logger.py          # W&B tracking (auto-tags for cross-model comparison)
│   ├── stage_utils.py           # Dotenv loading, JSON utils
│   ├── config_schema.py         # Pipeline/node dataclasses
│   └── runners/base.py          # StageRunner protocol
├── historical_norms/            # Norm extraction from fiction
├── grpo_training/               # SFT + GRPO fine-tuning
├── goldcoin_hipaa/              # GoldCoin-HIPAA evaluation
├── privacylens/                 # PrivacyLens evaluation
├── vlm_geoprivacy_bench/        # VLM-GeoPrivacy evaluation
├── uair/                        # AI risk analysis (news)
└── rule_tuples/                 # Reddit rule CI classification
```

Each dagspace is a self-contained pipeline with `cli.py` (Hydra entry), `orchestrator.py` (DAG execution), `runners/` (stage implementations), `stages/` (logic), and `conf/` (YAML configs).

---

## Setup

```bash
git clone <repo-url> && cd trawler
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .

# Site-specific configuration
cp server.env.example server.env
# Edit: SLURM_PARTITION, TRAWLER_PROJECT_ROOT, TRAWLER_VENV_ACTIVATE, NCCL settings
```

Requires CUDA GPUs. Models are downloaded to a local zoo directory and referenced by `conf/model/*.yaml`.

## Running Experiments

See **[EXPERIMENTS.md](EXPERIMENTS.md)** for the full COLM execution guide with commands for every pipeline, training variant, and benchmark evaluation.

Quick start:

```bash
# Extract norms from fiction novels
python -m dagspaces.historical_norms.cli pipeline=COLM_norms_fiction

# Train (SFT → GRPO)
python -m dagspaces.grpo_training.cli pipeline=full_training model=qwen3-8b

# Evaluate on GoldCoin-HIPAA
python -m dagspaces.goldcoin_hipaa.cli pipeline=full_eval model=qwen3-8b-grpo-ci

# Debug mode (local, no SLURM, 5 samples)
python -m dagspaces.goldcoin_hipaa.cli pipeline=full_eval model=qwen3-8b \
  runtime.debug=true runtime.sample_n=5 hydra/launcher=null
```

## Configuration

- **Model configs** in `dagspaces/common/conf/model/` — shared via Hydra searchpath
- **SLURM launchers** in `dagspaces/common/conf/hydra/launcher/` — parameterized via `server.env`
- **Pipeline DAGs** in each dagspace's `conf/pipeline/` — define stage graph + per-node overrides
- Override anything from CLI: `model=qwen3.5-9b sampling_params.temperature=0.3`

---

## Key Dependencies

- **[vLLM](https://docs.vllm.ai/)** (>=0.17.0) — LLM inference with tensor parallelism and LoRA
- **[TRL](https://github.com/huggingface/trl)** — SFT and GRPO training
- **[Hydra](https://hydra.cc/)** — hierarchical YAML configuration with CLI overrides
- **[Weights & Biases](https://wandb.ai/)** — experiment tracking with auto-tagging
- **[submitit](https://github.com/facebookincubator/submitit)** — SLURM job submission

## License

MIT
