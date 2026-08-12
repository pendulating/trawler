# Trawler

Experiment infrastructure for *"Reinforcing privacy reasoning in LLMs via normative simulacra from fiction"* (COLM 2026). Implements the full pipeline from norm extraction through fine-tuning to benchmark evaluation.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Paper Summary

LLM agents handle personal information but lack principled privacy reasoning. We teach LLMs to reason about privacy using Helen Nissenbaum's [Contextual Integrity](https://en.wikipedia.org/wiki/Contextual_integrity) (CI) theory, which defines privacy as the appropriate flow of information within social contexts.

**Key insight:** Fiction novels depict fully-realized societies with rich normative landscapes governing who may share what information with whom. We extract structured *normative simulacra* from these texts — CI information flow tuples paired with Raz-anatomy norms — and use them to fine-tune LLMs in two stages:

1. **SFT** teaches the model the vocabulary, format, and a conservative prior for CI-grounded reasoning
2. **GRPO** (Group Relative Policy Optimization) rewards reasoning that is structurally complete, internally coherent, and *grounded* in an explicit normative universe verified by an LLM judge

To prevent the model from memorizing source-specific norms rather than learning general CI reasoning, every normative grounding evaluation scores each completion against both the correct normative universe and a randomly selected wrong one (**per-completion contrastive scoring**).

The key claim: SFT alone teaches the *form* of CI reasoning and a prior toward caution, while GRPO calibrates that reasoning against context-specific norms.

### Research Questions

- **RQ1:** Can an LLM learn contextual privacy reasoning from structured reasoning traces extracted from fiction?
- **RQ2:** Does reinforcement learning with a normatively-grounded reward improve privacy reasoning beyond SFT alone?

### Evaluation

Five CI-aligned benchmarks spanning distinct societal contexts:

| Benchmark | Domain | What it tests |
|-----------|--------|---------------|
| [GoldCoin-HIPAA](https://arxiv.org/abs/2309.11500) | Healthcare | CI applicability + compliance on real court cases |
| [PrivacyLens](https://github.com/SALT-NLP/PrivacyLens) | Corporate/social | QA probing + agent action leakage judgment |
| [CIRL-729](https://github.com/EricGLan/CI-RL) | Privacy norms | Judge-free leakage / utility over the 729-action set |
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
│  (Gemma-4-31B-it)    │  Chunk → Reason → Extract → Cluster → Normative Universe
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
│  RL Stage            │  grpo_training dagspace
│  modular reward      │  R = max(0.50·R_DIRECT + 0.25·R_GROUND + 0.25·R_CONTRAST, 0.15)
│  GRPO + KTO arm      │  Norm-judgment vignettes interleaved with extraction
└──────────┬───────────┘
           │  m2 `full` ckpt-450 (GRPO) · k3 `verdict` (KTO)
           ▼
┌──────────────────────┐
│  Benchmark Evals     │  eval_all orchestrates all 5 benchmarks
│  5 CI benchmarks     │  goldcoin / privacylens / cirl / confaide / vlm_geoprivacy
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
├── cirl/                        # CIRL-729 judge-free leakage / utility
│   └── toolemu/                 # Vendored toolemu (Apache 2.0) for agent prompts
├── confaide/                    # ConfAIDE 6-tier evaluation
├── vlm_geoprivacy_bench/        # VLM-GeoPrivacy evaluation
│
└── eval_all/                    # Meta-orchestrator: runs all benchmarks for a model
    └── conf/sweep/              # Hydra sweep configs for multi-model SLURM array jobs
```

Each dagspace is a self-contained pipeline with `cli.py` (Hydra entry), `orchestrator.py` (DAG execution), `runners/` (stage implementations), `stages/` (logic), and `conf/` (YAML configs).

### Model Configs

Models live under `dagspaces/common/conf/model/{family}/{variant}.yaml`. For convenience, the project root has two symlinks pointing at the shared config dirs — **adding a yaml in either location is equivalent**:

- `models/` → `dagspaces/common/conf/model/`
- `launchers/` → `dagspaces/common/conf/hydra/launcher/`

So `models/qwen3.5-9b/base.yaml` and `dagspaces/common/conf/model/qwen3.5-9b/base.yaml` are the same file. Prefer whichever is more convenient to tab-complete to.

Organized as `{family}/{variant}.yaml`:

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
# Canonical venv name — all launchers/scripts expect .venv-vllm025cu129
# (see server.env TRAWLER_VENV_ACTIVATE). NB: use `uv pip install`, never
# `uv add`/`uv sync`, which prune this non-lock-managed venv.
uv venv --python 3.12 .venv-vllm025cu129 && source .venv-vllm025cu129/bin/activate
uv pip install -e .

# Site-specific configuration
cp server.env.example server.env
# Edit: SLURM_PARTITION, TRAWLER_PROJECT_ROOT, TRAWLER_VENV_ACTIVATE, NCCL settings
```

Requires CUDA GPUs. Models are downloaded to a local zoo directory and referenced by model configs.

## Running Experiments

See `EXPERIMENTS.md` (local only, not tracked in repo) for the full COLM execution guide.

### Quick start

```bash
# Extract norms from fiction novels
python -m dagspaces.historical_norms.cli pipeline=COLM_norms_fiction

# Train (SFT → GRPO with online normative grounding)
python -m dagspaces.grpo_training.cli pipeline=grpo_only_online_external model=qwen3.5-9b/base

# Evaluate a single model on all benchmarks
python -m dagspaces.eval_all.cli -m model=qwen3.5-9b/m2-full-ckpt450

# Sweep across models (max 3 concurrent SLURM jobs, judge on port 9015)
python -m dagspaces.eval_all.cli --multirun +sweep=colm_1gpu_judge1

# Individual benchmarks
python -m dagspaces.goldcoin_hipaa.cli pipeline=full_eval model=qwen3.5-9b/m2-full-ckpt450
python -m dagspaces.privacylens.cli pipeline=privacylens_clean model=qwen3.5-9b/m2-full-ckpt450
python -m dagspaces.cirl.cli pipeline=cirl_eval model=qwen3.5-9b/m2-full-ckpt450
python -m dagspaces.confaide.cli pipeline=confaide_eval model=qwen3.5-9b/m2-full-ckpt450
python -m dagspaces.vlm_geoprivacy_bench.cli pipeline=mcq_eval model=qwen3.5-9b/instruct

# Debug mode (local, no SLURM, 5 samples)
python -m dagspaces.goldcoin_hipaa.cli pipeline=full_eval model=qwen3.5-9b/base \
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
  'model=qwen3.5-9b/base,qwen3.5-9b/m2-full-ckpt450,qwen3.5-9b/instruct' \
  hydra.launcher.array_parallelism=3
```

## Configuration

- **Model configs** in `dagspaces/common/conf/model/{family}/{variant}.yaml` — shared via Hydra searchpath
- **SLURM launchers** in `dagspaces/common/conf/hydra/launcher/` — parameterized via `server.env`
- **Pipeline DAGs** in each dagspace's `conf/pipeline/` — define stage graph + per-node overrides
- **Judge servers** configured via `JUDGE_URL` / `JUDGE_SERVER_URL` env vars or sweep setup blocks
- Override anything from CLI: `model=qwen3.5-9b/base sampling_params.temperature=0.3`

---

## The RL-stage reward (m-series)

`training.grpo.reward_composition: modular` selects `stages/modular_reward.py`. A completion is
routed exactly once. It scores 0 unless it parses and is schema-complete (**R-VALID**);
abstentions and gold-no-flow chunks take a fixed **A-ABSTAIN** table with no server calls;
everything else takes the scored path

$$R = \max(0.50 \cdot R_\text{DIRECT} + 0.25 \cdot R_\text{GROUND} + 0.25 \cdot R_\text{CONTRAST},\; 0.15)$$

| Term | Weight | Type | Signal |
|------|--------|------|--------|
| $R_\text{DIRECT}$ | 0.50 | Verifiable | Retrieve the nearest `governs_info_flow` norm, derive a reference label from deontic force × act polarity, score the policy's own appropriateness label against it (chunk-fixed denominator, $\tau{=}0.55$) |
| $R_\text{GROUND}$ | 0.25 | Listwise judge | Normative grounding against the text's own universe $\hat{\mathcal{N}}_b$ |
| $R_\text{CONTRAST}$ | 0.25 | Listwise judge | Discrimination against a randomly selected wrong universe $\hat{\mathcal{N}}_{b'}$ |

One verifiable core, two **removable** listwise-judge auxiliaries. Weights are not tuned: the core
weighs 2× each active auxiliary, auxiliaries weigh equally, and removal renormalises — so
`-ground` gives 0.67/0.33 and `-aux` gives a core-only 1.00. A cell is defined by
`reward_auxiliaries` + `task_mix` + the `reward_core` toggle; `conf/sweep/grpo_m2_grid.yaml` is
the pre-registration record.

**Naming trap.** The core's weight key is `"outcome"`, but `core_mode` defaults to `"direct"`, so
the term that actually runs is R-DIRECT. Confirm from the launch log, which prints
`[modular_reward] chunk-denominator R-DIRECT active (tau=..., gold_k=...)`.

A **KTO arm** (`pipeline=kto_only`) trains on offline prospect-theoretic binary desirability
labels derived from the same reward stack. The reported checkpoints are the m2 `full` GRPO cell
at checkpoint-450 and the k3 `verdict` KTO arm, which share a merged SFT base.

> The earlier six-component `directional` reward (v9 lineage) was deprecated 2026-08-05 and
> removed 2026-08-12. See `wiki/grpo_redesign/` for the current design and
> `wiki/grpo_training_field_notes/` for the v2→v12a history.

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

MIT. Vendored `toolemu` code (in `dagspaces/common/toolemu/`) is Apache 2.0, from [CI-RL](https://github.com/EricGLan/CI-RL).
