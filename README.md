# Trawler

A DAG-based pipeline framework for large-scale text analysis with LLM-driven extraction, classification, and synthesis. Built for research on AI governance, contextual integrity, and norm extraction from text corpora.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What Trawler Does

Trawler orchestrates multi-stage NLP pipelines over document collections, using LLMs for structured information extraction. Each pipeline is defined as a directed acyclic graph (DAG) in YAML. Stages run sequentially or are dispatched to SLURM, with vLLM handling GPU inference via tensor parallelism.

The framework currently supports four research domains, each implemented as a self-contained **dagspace**:

### UAIR — AI Risk Assessment in News Media

Processes global news articles to extract structured records of real-world AI deployments, incidents, risks, and benefits. The full pipeline:

1. **Classify Relevance** — keyword pre-gating + LLM binary filter for AI-related content
2. **Decompose** — extracts AI use-case tuples: deployment domain, purpose, capability, deployer, subject, location, date, harms, risks, benefits
3. **Verify** — embedding similarity + entailment scoring to validate extracted tuples against source text
4. **Classify EU AI Act** — maps each AI use case to EU AI Act risk tiers (Prohibited / High / Limited / Minimal)
5. **Classify Risks & Benefits** — fine-grained categorization of specific risks and benefits mentioned
6. **Taxonomy / Topic / Synthesis** — clustering and cross-article synthesis of extracted patterns

### Historical Norms — Norm Extraction from Literature

Extracts societal norms about information flows from historical and prescriptive texts (Project Gutenberg), grounded in Helen Nissenbaum's [Contextual Integrity](https://en.wikipedia.org/wiki/Contextual_integrity) (CI) framework:

1. **Fetch Gutenberg** — retrieves and chunks books by Gutenberg ID
2. **Norm/CI Reasoning** — LLM analysis of text chunks for societal norms and information flow patterns
3. **Norm/CI Extraction** — structures output as formal CI 5-tuples: subject, sender, recipient, information type, transmission principle

### Rule Tuples — Privacy Norms in Online Communities

Applies the same CI framework to Reddit community rules:

1. **Classify** — identifies rules governing privacy and information flows (vs. content moderation rules)
2. **Decompose** — extracts CI tuples from relevant community governance rules

### Contextual Integrity Eval — Benchmarking LLM Understanding of CI

Evaluates how well LLMs understand contextual integrity through QA probing, agent action evaluation, judge calibration, active prompting ablation, and context collapse diagnostics. Based on the [PrivacyLens](https://github.com/SALT-NLP/PrivacyLens) dataset.

---

## Architecture

```
dagspaces/
├── common/                      # Shared framework code
│   ├── orchestrator.py          # DAG execution, SLURM dispatch, artifact tracking
│   ├── vllm_inference.py        # Direct vLLM inference (GPU detection, NCCL config)
│   ├── wandb_logger.py          # W&B experiment tracking
│   ├── stage_utils.py           # Shared stage utilities
│   ├── config_schema.py         # Pipeline/node dataclasses
│   └── runners/base.py          # StageRunner protocol
├── uair/                        # AI risk analysis dagspace
├── historical_norms/            # Literature norm extraction dagspace
├── rule_tuples/                 # Reddit rule analysis dagspace
└── contextual_integrity_eval/   # CI evaluation dagspace
```

Each dagspace follows a consistent structure:

```
dagspaces/{name}/
├── cli.py              # Hydra entry point
├── orchestrator.py     # Pipeline-specific DAG logic + re-exports from common
├── wandb_logger.py     # Thin shim over common/wandb_logger.py
├── conf/
│   ├── config.yaml     # Base config (model, runtime, sampling, wandb)
│   ├── pipeline/       # DAG definitions (sources → nodes → outputs)
│   ├── prompt/         # LLM prompt templates
│   ├── model/          # vLLM engine configs (model path, TP size, memory)
│   └── hydra/launcher/ # SLURM launcher configs
├── runners/            # StageRunner implementations (one per stage)
└── stages/             # Stage logic (_pre/_post transforms + run function)
```

---

## Setup

```bash
git clone <repo-url> && cd trawler

# Install with uv (recommended)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
```

Requires CUDA GPUs. Models are loaded from local paths configured in `conf/model/*.yaml` (e.g., Qwen 2.5-72B-AWQ, Qwen 3-30B).

## Running Pipelines

```bash
# UAIR: full news analysis pipeline
python -m dagspaces.uair.cli \
  pipeline=full_event_pipeline \
  data.parquet_path=/path/to/articles.parquet

# Historical norms: CI extraction from Gutenberg texts
python -m dagspaces.historical_norms.cli \
  pipeline=ci_extraction \
  data.parquet_path=/path/to/texts.parquet

# Rule tuples: classify + decompose Reddit rules
python -m dagspaces.rule_tuples.cli \
  runtime.stage=pipeline \
  data.parquet_path=/path/to/rules.parquet

# Debug mode: sample 100 rows
python -m dagspaces.uair.cli \
  pipeline=full_event_pipeline \
  runtime.debug=true runtime.sample_n=100 \
  data.parquet_path=/path/to/data.parquet
```

### SLURM Dispatch

```bash
python -m dagspaces.uair.cli \
  pipeline=full_event_pipeline \
  hydra/launcher=g2_slurm_gpu_1x \
  data.parquet_path=/path/to/data.parquet
```

Launcher configs in `conf/hydra/launcher/` define GPU count, memory, partition, and setup commands.

---

## Pipeline Configuration

Pipelines are Hydra-composed YAML DAGs:

```yaml
# conf/pipeline/my_pipeline.yaml
pipeline:
  sources:
    articles:
      path: ${data.parquet_path}
  graph:
    nodes:
      classify:
        stage: classify_relevance
        inputs: {articles: articles}
        outputs: [classified]
      extract:
        stage: decompose_nbl
        depends_on: [classify]
        inputs: {articles: classified}
        outputs: [extracted]
```

Override anything from CLI: `model=qwen2.5-72b-awq sampling_params.temperature=0.3 runtime.sample_n=500`

## Adding a Stage

1. Implement in `dagspaces/{name}/stages/mystage.py`:

```python
from dagspaces.common.vllm_inference import run_vllm_inference

def _pre(row):
    row["messages"] = [{"role": "user", "content": row["text"]}]
    row["sampling_params"] = {"max_tokens": 512, "temperature": 0.0}
    return row

def _post(row):
    row["result"] = row["generated_text"]
    return row

def run_mystage(df, cfg):
    return run_vllm_inference(df, cfg, _pre, _post, "mystage")
```

2. Create runner in `dagspaces/{name}/runners/mystage.py` implementing `StageRunner.run(context)`.
3. Register in `runners/__init__.py`.

---

## Key Dependencies

- **[vLLM](https://docs.vllm.ai/)** — LLM inference with tensor parallelism (`distributed_executor_backend="mp"`)
- **[Hydra](https://hydra.cc/)** — hierarchical YAML configuration with CLI overrides
- **[Weights & Biases](https://wandb.ai/)** — experiment tracking, table logging, run grouping
- **[submitit](https://github.com/facebookincubator/submitit)** — SLURM job submission from Hydra

## License

MIT
