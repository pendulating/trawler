# Trawler

Experiment infrastructure for *"Reinforcing privacy reasoning in LLMs via normative simulacra"* (COLM 2026). Implements the full pipeline from norm extraction through fine-tuning to benchmark evaluation.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Paper Summary

LLM agents handle personal information but lack principled privacy reasoning. We teach LLMs to reason about privacy using Helen Nissenbaum's [Contextual Integrity](https://en.wikipedia.org/wiki/Contextual_integrity) (CI) theory, which defines privacy as the appropriate flow of information within social contexts.

**Key insight:** Fiction novels depict fully-realized societies with rich normative landscapes governing who may share what information with whom. We extract structured *normative simulacra* from these texts — CI information flow tuples paired with Raz-anatomy norms — and use them to fine-tune LLMs in two stages:

1. **SFT** teaches the model the vocabulary and format of CI-grounded reasoning
2. **GRPO** (Group Relative Policy Optimization, $G{=}4$) rewards reasoning that is structurally complete, internally coherent, and *grounded* in an explicit normative universe verified by an LLM judge — interleaved with norm judgment vignettes that exercise norm *application*

The claim: SFT alone teaches *mimicry* of CI output; GRPO teaches actual *reasoning* about the relationship between information flows and the norms that govern them.

### Research Questions

- **RQ1:** Can an LLM learn contextual privacy reasoning from structured reasoning traces extracted from fiction?
- **RQ2:** Does reinforcement learning with a normatively-grounded reward improve privacy reasoning beyond SFT alone?
- **RQ3:** Does fine-tuning on a divergent normative universe (e.g., dystopian text) introduce measurable normative bias?

### Evaluation

Five CI-aligned benchmarks spanning distinct societal contexts:

| Benchmark | Domain | What it tests |
|-----------|--------|---------------|
| [GoldCoin-HIPAA](https://arxiv.org/abs/2309.11500) | Healthcare | CI applicability + compliance on real court cases |
| [PrivacyLens](https://github.com/SALT-NLP/PrivacyLens) | Corporate/social | QA probing + agent action leakage judgment |
| [CI-RL Vignettes](https://github.com/EricGLan/CI-RL) | Privacy norms | Probing accuracy (seed/vignette) + trajectory I/U/C |
| [ConfAIDE](https://github.com/skywalker023/confaide) | Confidentiality | Likert correlation, rejection accuracy, leak/error rate |
| [VLM-GeoPrivacy](https://arxiv.org/abs/2411.15087) | Visual geolocation | Location disclosure granularity from images |

---

## Pipeline Overview

```
Fiction novels (10 texts from Project Gutenberg)
    │
    ▼
┌──────────────────────┐
│  Norm Extraction      │  historical_norms dagspace
│  (Qwen2.5-72B-AWQ)   │  Chunk → Reason → Extract → Cluster → Normative Universe
└──────────┬───────────┘
           │  normative simulacra (N̂_b per text)
           ▼
┌──────────────────────┐
│  SFT Fine-tuning     │  grpo_training dagspace
│  (LoRA)              │  CI reasoning traces as instruction-following data
└──────────┬───────────┘
           │  SFT checkpoint
           ▼
┌──────────────────────┐
│  GRPO Training       │  grpo_training dagspace
│  6-component reward   │  R_uncert + R_complete + R_consist + R_context + R_cohere + R_ground
│  + judgment vignettes │  Norm application tasks interleaved at 1:1 ratio
└──────────┬───────────┘
           │  GRPO checkpoint
           ▼
┌──────────────────────┐
│  Benchmark Evals     │  eval_all orchestrates all 5 benchmarks
│  5 CI benchmarks     │  goldcoin / privacylens / cirl_vignettes / confaide / vlm_geoprivacy
└──────────────────────┘
```

---

## Architecture

```
dagspaces/
├── common/                      # Shared framework code
│   ├── conf/
│   │   ├── model/               # Model configs: family/variant dirs (e.g. qwen3.5-9b/base.yaml)
│   │   └── hydra/launcher/      # SLURM launchers (monitor, gpu_1x/2x, train_1x/3x)
│   ├── orchestrator.py          # DAG execution, SLURM dispatch, GPU sanitization
│   ├── vllm_inference.py        # vLLM inference (LoRA, think-block stripping)
│   ├── wandb_logger.py          # W&B tracking (auto-tags: bench, family, finetuned/base)
│   ├── judge_client.py          # OpenAI-compatible judge (vLLM server, batch concurrency)
│   ├── eval_schemas.py          # Shared evaluation dataclasses
│   ├── stage_utils.py           # Dotenv loading, JSON utils
│   ├── config_schema.py         # Pipeline/node dataclasses
│   └── runners/base.py          # StageRunner protocol
│
├── historical_norms/            # Norm extraction from fiction
├── grpo_training/               # SFT + GRPO fine-tuning
│
├── goldcoin_hipaa/              # GoldCoin-HIPAA evaluation
├── privacylens/                 # PrivacyLens evaluation
├── cirl_vignettes/              # CI-RL probing + trajectory I/U/C evaluation
│   └── toolemu/                 # Vendored toolemu (Apache 2.0) for agent prompts
├── confaide/                    # ConfAIDE 6-tier evaluation
├── vlm_geoprivacy_bench/        # VLM-GeoPrivacy evaluation
│
└── eval_all/                    # Meta-orchestrator: runs all benchmarks for a model
    └── conf/sweep/              # Hydra sweep configs for multi-model SLURM array jobs
```

Each dagspace is a self-contained pipeline with `cli.py` (Hydra entry), `orchestrator.py` (DAG execution), `runners/` (stage implementations), `stages/` (logic), and `conf/` (YAML configs).

### Model Configs

Models are organized as `conf/model/{family}/{variant}.yaml`:

| Family | Variants | Params |
|--------|----------|--------|
| `qwen3.5-9b` | `base`, `sft-ci`, `grpo-v3-*` | 9B |
| `qwen3-8b` | `base`, `sft-ci`, `grpo-ci` | 8B |
| `qwen3.5-27b` | `base` | 27B |
| `qwen2.5-72b` | `base`, `awq`, `instruct-awq` | 72B |
| `gemma-3-12b` | `it`, `it-sft-ci` | 12B |
| `llama3.1-8b` | `instruct` | 8B |
| `llama3.3-70b` | `instruct` | 70B |
| `gpt-oss-20b` | `base` | 20B |
| `phi-4` | `base` | 14B |
| ... | `cirl`, `context-reasoner`, `by_book`, `openthinker*` | various |

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

Requires CUDA GPUs. Models are downloaded to a local zoo directory and referenced by model configs.

## Running Experiments

See **[EXPERIMENTS.md](EXPERIMENTS.md)** for the full COLM execution guide.

### Quick start

```bash
# Extract norms from fiction novels
python -m dagspaces.historical_norms.cli pipeline=COLM_norms_fiction

# Train (SFT → GRPO with online normative grounding)
python -m dagspaces.grpo_training.cli pipeline=grpo_only_online_external model=qwen3.5-9b/base

# Evaluate a single model on all benchmarks
python -m dagspaces.eval_all.cli -m model=qwen3.5-9b/grpo-v3-vr05-lambda10

# Sweep across models (max 3 concurrent SLURM jobs, judge on port 9015)
python -m dagspaces.eval_all.cli --multirun +sweep=colm_1gpu_judge1

# Individual benchmarks
python -m dagspaces.goldcoin_hipaa.cli pipeline=full_eval model=qwen3-8b/grpo-ci
python -m dagspaces.privacylens.cli pipeline=privacylens_clean model=qwen3-8b/grpo-ci
python -m dagspaces.cirl_vignettes.cli pipeline=cirl_vignettes_eval model=qwen3-8b/grpo-ci
python -m dagspaces.cirl_vignettes.cli pipeline=cirl_trajectory_eval model=qwen3-8b/grpo-ci
python -m dagspaces.confaide.cli pipeline=confaide_eval model=qwen3-8b/grpo-ci
python -m dagspaces.vlm_geoprivacy_bench.cli pipeline=mcq_eval model=qwen3-vl-8b/instruct

# Debug mode (local, no SLURM, 5 samples)
python -m dagspaces.goldcoin_hipaa.cli pipeline=full_eval model=qwen3-8b/base \
  runtime.debug=true runtime.sample_n=5 hydra/launcher=null
```

### Multi-model sweeps

Hydra sweeps with `submitit_slurm` submit one SLURM job per model, with `array_parallelism` capping concurrency:

```bash
# Pre-configured sweeps (see dagspaces/eval_all/conf/sweep/)
python -m dagspaces.eval_all.cli --multirun +sweep=colm_1gpu_judge1   # 6 models, judge :9015
python -m dagspaces.eval_all.cli --multirun +sweep=colm_1gpu_judge2   # 8 models, judge :9016
python -m dagspaces.eval_all.cli --multirun +sweep=colm_2gpu          # 3 models, 2-GPU

# Ad-hoc sweep
python -m dagspaces.eval_all.cli --multirun \
  'model=qwen3-8b/base,qwen3-8b/grpo-ci,qwen2.5-7b/instruct' \
  hydra.launcher.array_parallelism=3
```

## Configuration

- **Model configs** in `dagspaces/common/conf/model/{family}/{variant}.yaml` — shared via Hydra searchpath
- **SLURM launchers** in `dagspaces/common/conf/hydra/launcher/` — parameterized via `server.env`
- **Pipeline DAGs** in each dagspace's `conf/pipeline/` — define stage graph + per-node overrides
- **Judge servers** configured via `JUDGE_URL` / `JUDGE_SERVER_URL` env vars or sweep setup blocks
- Override anything from CLI: `model=qwen3.5-9b/base sampling_params.temperature=0.3`

---

## GRPO Reward Function

Six components ($R = \sum w_i R_i$), plus interleaved norm judgment vignettes:

| Component | Weight | Type | Signal |
|-----------|--------|------|--------|
| $R_\text{uncert}$ | 0.10 | Programmatic | Schema validity, construct discrimination, confidence |
| $R_\text{complete}$ | 0.05 | Programmatic | Proportion of non-null CI tuple fields |
| $R_\text{consist}$ | 0.05 | Programmatic | Internal invariant checks |
| $R_\text{context}$ | 0.20 | Embedding | Cosine similarity of stated context vs. normative universe |
| $R_\text{cohere}$ | 0.10 | Programmatic | Reasoning trace ↔ extraction coherence |
| $R_\text{ground}$ | 0.50 | LLM judge | Per-flow: 0.4 norm_match + 0.4 governance + 0.2 appropriateness |

Norm judgment vignettes (mixed at `vignette_ratio=0.5`) are scored separately: judgment accuracy (0.50) + reasoning quality (0.25) + norm citation (0.25).

---

## Key Dependencies

- **[vLLM](https://docs.vllm.ai/)** (>=0.17.0) — LLM inference with tensor parallelism and LoRA
- **[TRL](https://github.com/huggingface/trl)** — SFT and GRPO training
- **[Hydra](https://hydra.cc/)** — hierarchical YAML configuration with CLI overrides
- **[Weights & Biases](https://wandb.ai/)** — experiment tracking with auto-tagging
- **[submitit](https://github.com/facebookincubator/submitit)** — SLURM job submission
- **[procoder](https://github.com/dhh1995/PromptCoder)** — prompt templating (for vendored toolemu)
- **[langchain](https://python.langchain.com/)** — tool interface abstractions (for vendored toolemu)

## License

MIT. Vendored `toolemu` code (in `dagspaces/cirl_vignettes/toolemu/`) is Apache 2.0, from [CI-RL](https://github.com/EricGLan/CI-RL).
