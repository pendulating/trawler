# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Trawler** is the experiment infrastructure for a COLM 2026 paper: *"Reinforcing privacy reasoning in LLMs via normative simulacra."* The paper proposes a method to teach LLMs contextual integrity (CI) reasoning by: (1) extracting structured normative representations from fiction novels, (2) fine-tuning via SFT then GRPO with a composite reward grounded in each text's normative universe, and (3) evaluating on three CI-aligned benchmarks. Trawler implements the full pipeline as a DAG-based framework using vLLM for multi-GPU inference, Hydra for configuration, and SLURM for cluster execution. Python 3.12 only.

## Common Commands

```bash
# Install
uv pip install -e .

# Norm extraction from fiction (COLM pipeline)
python -m dagspaces.historical_norms.cli pipeline=COLM_norms_fiction

# Training: SFT + GRPO
python -m dagspaces.grpo_training.cli pipeline=full_training model=qwen3-8b
python -m dagspaces.grpo_training.cli pipeline=grpo_only model=qwen3-8b

# Benchmark evaluations
python -m dagspaces.goldcoin_hipaa.cli pipeline=full_eval model=qwen3-8b-grpo-ci
python -m dagspaces.privacylens.cli pipeline=privacylens_clean model=qwen3-8b-grpo-ci
python -m dagspaces.vlm_geoprivacy_bench.cli pipeline=mcq_eval model=qwen3-vl-8b-instruct

# Debug mode (local, sampled)
python -m dagspaces.goldcoin_hipaa.cli pipeline=full_eval runtime.debug=true runtime.sample_n=5 hydra/launcher=null
```

See `EXPERIMENTS.md` for the full COLM execution guide.

No formal test suite exists. Ad-hoc tests live in `test_clean_stage.py` and `scripts/test_*.py`.

## Architecture

### Dagspace Pattern
Each domain pipeline is a self-contained "dagspace" under `dagspaces/{name}/` with:
- `cli.py` - Hydra entry point (`python -m dagspaces.{name}.cli`)
- `orchestrator.py` - DAG execution (imports shared utilities from `common/orchestrator.py`): loads pipeline graph, topologically sorts nodes, resolves inputs via ArtifactRegistry, submits jobs to SLURM or runs locally
- `conf/` - Hydra configs: `config.yaml` (base), `pipeline/` (DAG definitions), `prompt/` (LLM templates)
- `runners/` - Stage runner classes extending `StageRunner` base; lazy-loaded via `get_stage_registry()`
- `stages/` - Stage implementation functions

Five active dagspaces:
- **historical_norms** — Norm extraction from fiction novels (CI tuples + Raz norms → normative simulacra)
- **grpo_training** — SFT + GRPO fine-tuning with composite reward (6 components including normative grounding judge)
- **goldcoin_hipaa** — GoldCoin-HIPAA benchmark evaluation (healthcare CI)
- **privacylens** — PrivacyLens benchmark evaluation (QA probing + leakage judgment)
- **vlm_geoprivacy_bench** — VLM-GeoPrivacy benchmark evaluation (visual geolocation CI)

Deprecated (dot-prefixed, not used for COLM): `.uair`, `.rule_tuples`.

### Shared Configuration

Model configs and SLURM launchers live in `dagspaces/common/conf/` and are resolved by all dagspaces via Hydra searchpath (`pkg://dagspaces.common.conf`). Dagspace-local `conf/model/` or `conf/hydra/launcher/` overrides take precedence when present.

Site-specific settings (SLURM partition, project paths, NCCL) are in `server.env`, loaded automatically by `ensure_dotenv()`. Copy `server.env.example` to `server.env` for a new cluster.

### Key Abstractions

**Common modules** (`dagspaces/common/`): `orchestrator.py` (shared DAG utilities, dataclasses, SLURM helpers, `build_run_config()`), `vllm_inference.py` (direct LLM inference with LoRA support and think-block stripping), `wandb_logger.py` (experiment tracking with auto-tagging for cross-model comparison), `stage_utils.py` (`ensure_dotenv`, `extract_last_json`, `sanitize_for_json`), `config_schema.py` (pipeline dataclasses), `runners/base.py` (StageRunner protocol).

**StageRunner** (`dagspaces/common/runners/base.py`): Base class all runners extend. Receives `StageExecutionContext` with cfg, node spec, resolved inputs, output paths, and optional WandbLogger. Returns `StageResult`.

**Pipeline Graph** (`dagspaces/common/config_schema.py`): `PipelineGraphSpec` / `PipelineNodeSpec` dataclasses define DAG structure. Nodes declare stage name, dependencies, inputs (referencing sources or other node outputs), outputs, per-node config overrides, and optional SLURM launcher override.

**ArtifactRegistry**: Tracks sources and node outputs. Cross-node input references resolve as `node_name.output_key` or `source_name`.

### LLM / GPU Patterns
- Stages use vLLM directly via the shared utility `dagspaces/common/vllm_inference.py` (`run_vllm_inference()`)
- The `LLM` class handles multi-GPU tensor parallelism internally via `distributed_executor_backend="mp"`
- GPU sanitization: orchestrator probes each GPU in a subprocess before stage execution, removes broken ones, adjusts tensor parallelism automatically
- Set `UAIR_SKIP_GPU_SANITIZE=1` to bypass GPU checks
- NCCL env vars for PCIe GPUs (P2P/IB/SHM disable) are configured via `server.env`
- When `chat_template_kwargs.enable_thinking` is `false`, vLLM inference automatically strips `<think>` blocks from model output
- Prompts loaded from YAML configs under `conf/prompt/`

### Hydra Configuration
- Version base: `1.3`
- Uses config composition, `@package` directives, `${oc.env:VAR}` interpolation
- Pipeline YAML defines the full DAG including per-node overrides and launcher selection
- Shared configs resolved via `hydra.searchpath: [pkg://dagspaces.common.conf]`
- Outputs go to `outputs/YYYY-MM-DD_experiment_name/HH-MM-SS/` with `.hydra/` metadata

### W&B Integration
`common/wandb_logger.py` is the canonical implementation; each dagspace has a thin shim with dagspace-specific defaults (project name, column exclusions). Runs are auto-tagged with `bench:<dagspace>`, `family:<model>`, `finetuned`/`base`, `task:<eval_task>`. Supports single-run or per-stage runs, table sampling.

## Adding a New Stage
1. Create stage function in `dagspaces/{name}/stages/mystage.py`
2. Create runner class in `dagspaces/{name}/runners/mystage.py` extending `StageRunner`
3. Register in `dagspaces/{name}/runners/__init__.py` inside `get_stage_registry()`
4. Add node to a pipeline YAML in `conf/pipeline/`
5. Optionally add prompt config in `conf/prompt/`

## Adding a New Model
1. Download to `/share/pierson/matt/zoo/models/<ModelName>/`
2. Create `dagspaces/common/conf/model/<model-name>.yaml` with `@package _global_` format
3. Use with any dagspace: `model=<model-name>`

## Key Dependencies
hydra-core, hydra-submitit-launcher, omegaconf, vllm (>=0.17.0), pandas, pyarrow, wandb, torch, transformers, sentence-transformers, trl, peft
