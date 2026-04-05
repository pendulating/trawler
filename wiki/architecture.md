# Architecture

## The Dagspace pattern

A **dagspace** is a self-contained pipeline module under `dagspaces/{name}/`. Each has the same shape:

```
dagspaces/{name}/
├── cli.py              # Hydra entry point: python -m dagspaces.{name}.cli
├── orchestrator.py     # DAG execution (imports from common/orchestrator.py)
├── conf/
│   ├── config.yaml     # base config
│   ├── pipeline/*.yaml # DAG definitions (nodes, edges, launchers)
│   └── prompt/*.yaml   # LLM prompt templates
├── runners/
│   ├── __init__.py     # get_stage_registry() — lazy-loaded stage → runner map
│   └── {stage}.py      # StageRunner subclass per stage
└── stages/             # stage implementation functions
```

The orchestrator loads a pipeline YAML → topologically sorts nodes → resolves inputs via the ArtifactRegistry → submits each node to SLURM (via hydra-submitit) or runs it locally.

## Six active dagspaces

| Dagspace | Role |
|---|---|
| `historical_norms` | Extract CI tuples + Raz norms from fiction → normative simulacra |
| `grpo_training` | SFT + GRPO with composite reward (includes norm-universe build) |
| `goldcoin_hipaa` | Healthcare CI benchmark (HIPAA court cases) |
| `privacylens` | Agent action / QA leakage benchmark |
| `vlm_geoprivacy_bench` | Visual geolocation CI benchmark |
| `confaide` | ConfAIde tiers 1–2 paired with crowdsourced humans |
| `cirl_vignettes` | Structured CI-vignette benchmark |

Deprecated (dot-prefixed, ignore for COLM): `.uair`, `.rule_tuples`.

## Shared modules (`dagspaces/common/`)

| Module | Provides |
|---|---|
| `orchestrator.py` | Shared DAG utilities, dataclasses, SLURM helpers, `build_run_config()` |
| `vllm_inference.py` | `run_vllm_inference()` — LLM calls with LoRA support and auto `<think>` stripping |
| `wandb_logger.py` | Auto-tagging W&B runs for cross-model comparison |
| `stage_utils.py` | `ensure_dotenv()`, `extract_last_json()`, `sanitize_for_json()` |
| `config_schema.py` | `PipelineGraphSpec`, `PipelineNodeSpec` dataclasses |
| `runners/base.py` | `StageRunner` protocol |
| `judge_client.py` | HTTP client for external judge server (used by `R_ground`) |
| `eval_schemas.py` | Eval output dataclasses shared across benchmarks |
| `conf/model/*` | Shared model configs resolved via Hydra searchpath |
| `conf/hydra/launcher/*` | SLURM launcher configs |

## Key abstractions

### `StageRunner`
Base class in `dagspaces/common/runners/base.py`. Subclasses set `stage_name` and implement `run(context) → StageResult`. The context carries `cfg`, `node`, resolved `inputs`, `output_paths`, `output_dir`, and a `WandbLogger`.

### `PipelineGraphSpec` / `PipelineNodeSpec`
Dataclasses in `dagspaces/common/config_schema.py`. A pipeline YAML declares nodes with `stage`, `depends_on`, `inputs`, `outputs`, `launcher`, and per-node `config` overrides.

### `ArtifactRegistry`
Tracks sources and node outputs. Input references resolve as `node_name.output_key` (cross-node) or `source_name` (root inputs from parquet/JSON/dir sources).

### Stage registry
Each dagspace's `runners/__init__.py` exports `get_stage_registry()` which **lazy-imports** runner classes (so adding a new runner doesn't incur an import at module load time).

## Hydra conventions

- Version base: `1.3`
- Shared configs resolved via `hydra.searchpath: [pkg://dagspaces.common.conf]`
- Model yamls use `# @package _global_` so they merge into the top-level `model:` key
- Outputs go to `outputs/YYYY-MM-DD_name/HH-MM-SS/` with `.hydra/` metadata
- Env interpolation: `${oc.env:VAR,default}` is standard
- `-m` flag invokes hydra-submitit (SLURM); `hydra/launcher=null` bypasses for local runs

## vLLM / GPU patterns

- All stages call vLLM via the shared `run_vllm_inference()` helper
- Multi-GPU tensor parallelism: `tensor_parallel_size` in model config + `distributed_executor_backend="mp"`
- **GPU sanitization**: orchestrator probes each GPU in a subprocess before stage execution, removes broken ones, adjusts TP automatically. Bypass with `UAIR_SKIP_GPU_SANITIZE=1`.
- NCCL env vars for PCIe GPUs live in `server.env` (P2P/IB/SHM disable for non-NVLink)
- When `model.chat_template_kwargs.enable_thinking: false`, `<think>` blocks are stripped automatically
