# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Trawler** is a DAG-based pipeline framework for processing large text datasets with LLM integration. It uses vLLM for multi-GPU inference, Hydra for configuration, and SLURM for cluster execution. Python 3.12 only.

## Common Commands

```bash
# Install
uv pip install -e .

# Run a pipeline (each dagspace has its own CLI)
python -m dagspaces.uair.cli pipeline=full_event_pipeline data.parquet_path=/path/to/data.parquet
python -m dagspaces.historical_norms.cli pipeline=norm_extraction data.parquet_path=/path/to/data.parquet
python -m dagspaces.rule_tuples.cli pipeline=classify_decompose data.parquet_path=/path/to/data.parquet

# Debug mode (local, sampled)
python -m dagspaces.uair.cli pipeline=full_event_pipeline runtime.debug=true runtime.sample_n=100

# Local execution (no SLURM)
python -m dagspaces.uair.cli pipeline=full_event_pipeline hydra/launcher=null

# Run with specific SLURM launcher
python -m dagspaces.uair.cli pipeline=full_event_pipeline hydra/launcher=g2_slurm_gpu_4x
```

No formal test suite exists. Ad-hoc tests live in `test_clean_stage.py` and `scripts/test_*.py`.

## Architecture

### Dagspace Pattern
Each domain pipeline is a self-contained "dagspace" under `dagspaces/{name}/` with:
- `cli.py` - Hydra entry point (`python -m dagspaces.{name}.cli`)
- `orchestrator.py` - DAG execution (imports shared utilities from `common/orchestrator.py`): loads pipeline graph, topologically sorts nodes, resolves inputs via ArtifactRegistry, submits jobs to SLURM or runs locally
- `conf/` - Hydra configs: `config.yaml` (base), `pipeline/` (DAG definitions), `prompt/` (LLM templates), `model/`, `hydra/launcher/` (SLURM specs)
- `runners/` - Stage runner classes extending `StageRunner` base; lazy-loaded via `get_stage_registry()`
- `stages/` - Stage implementation functions

Four dagspaces: **uair** (news AI risk analysis, most mature), **historical_norms** (literature norm extraction via contextual integrity), **rule_tuples** (Reddit rule CI classification), **privacylens** (benchmarking LLM CI understanding via PrivacyLens).

### Key Abstractions

**Common modules** (`dagspaces/common/`): `orchestrator.py` (shared DAG utilities, dataclasses, SLURM helpers), `vllm_inference.py` (direct LLM inference), `wandb_logger.py` (experiment tracking), `stage_utils.py` (shared `maybe_silence_vllm_logs`, `to_json_str`, `serialize_arrow_unfriendly_in_row`, `extract_last_json`, `sanitize_for_json`), `config_schema.py` (pipeline dataclasses), `runners/base.py` (StageRunner protocol).

**StageRunner** (`dagspaces/common/runners/base.py`): Base class all runners extend. Receives `StageExecutionContext` with cfg, node spec, resolved inputs, output paths, and optional WandbLogger. Returns `StageResult`.

**Pipeline Graph** (`dagspaces/common/config_schema.py`): `PipelineGraphSpec` / `PipelineNodeSpec` dataclasses define DAG structure. Nodes declare stage name, dependencies, inputs (referencing sources or other node outputs), outputs, per-node config overrides, and optional SLURM launcher override.

**ArtifactRegistry**: Tracks sources and node outputs. Cross-node input references resolve as `node_name.output_key` or `source_name`.

**Config cloning**: Each node gets a deep-copied config via `OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))` to prevent mutation across stages.

### LLM / GPU Patterns
- Stages use vLLM directly via the shared utility `dagspaces/common/vllm_inference.py` (`run_vllm_inference()`)
- The `LLM` class handles multi-GPU tensor parallelism internally via `distributed_executor_backend="mp"`
- GPU sanitization: orchestrator probes each GPU in a subprocess before stage execution, removes broken ones, adjusts tensor parallelism automatically
- Set `UAIR_SKIP_GPU_SANITIZE=1` to bypass GPU checks
- NCCL env vars for PCIe GPUs (P2P/IB/SHM disable) are set automatically before vLLM init
- Prompts loaded from YAML configs under `conf/prompt/`

### Hydra Configuration
- Version base: `1.3`
- Uses config composition, `@package` directives, `${oc.env:VAR}` interpolation
- Pipeline YAML defines the full DAG including per-node overrides and launcher selection
- Outputs go to `outputs/YYYY-MM-DD/HH-MM-SS/` with `.hydra/` metadata

### W&B Integration
`common/wandb_logger.py` is the canonical implementation; each dagspace has a thin shim with dagspace-specific defaults (project name, column exclusions). `WandbLogger` context manager handles run lifecycle; `_NoOpLogger` fallback when disabled. Supports single-run or per-stage runs, table sampling.

## Adding a New Stage
1. Create stage function in `dagspaces/{name}/stages/mystage.py`
2. Create runner class in `dagspaces/{name}/runners/mystage.py` extending `StageRunner`
3. Register in `dagspaces/{name}/runners/__init__.py` inside `get_stage_registry()`
4. Add node to a pipeline YAML in `conf/pipeline/`
5. Optionally add prompt config in `conf/prompt/`

## Key Dependencies
hydra-core, hydra-submitit-launcher, omegaconf, vllm (>=0.11.0), pandas, pyarrow, wandb, torch, transformers, sentence-transformers
