# Trawler: Large-Scale Text Pipeline Framework

A configurable DAG-based pipeline framework for processing large text datasets with LLM integration.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

Trawler is a scalable pipeline framework for running multi-stage text processing workflows over large datasets. Built on Ray Data for distributed processing and vLLM for efficient LLM inference, Trawler enables researchers and engineers to:

- **Define** complex multi-stage pipelines as YAML DAGs
- **Process** millions of documents using distributed computing
- **Integrate** LLMs for classification, extraction, and synthesis
- **Track** experiments automatically with Weights & Biases
- **Deploy** on local machines or SLURM clusters

### Key Features

- **Configuration-Driven**: Define pipelines in YAML, no code changes needed
- **Dagspace Architecture**: Modular domain-specific pipeline configurations
- **Scalable**: Process millions of records using Ray Data and SLURM clusters
- **LLM-Integrated**: Built-in vLLM support with automatic GPU management
- **Extensible**: Easy to add custom processing stages
- **Tracked**: Automatic experiment logging with Weights & Biases

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Definition (YAML)                    │
├─────────────────────────────────────────────────────────────────┤
│  Sources → Stage 1 → Stage 2 → Stage 3 → Outputs               │
│  (Data)    (filter)  (extract) (synth)   (Parquet)             │
├─────────────────────────────────────────────────────────────────┤
│              Orchestrator (DAG Execution Engine)                 │
├─────────────────────────────────────────────────────────────────┤
│  Ray Data (Distributed) | vLLM (GPU) | SLURM (Cluster) | W&B   │
├─────────────────────────────────────────────────────────────────┤
│            Hydra (Configuration Management)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Dagspaces

Trawler organizes domain-specific pipelines into **dagspaces** - self-contained modules with their own stages, configurations, and prompts:

| Dagspace | Domain | Description |
|----------|--------|-------------|
| **uair** | News Analysis | AI risk assessment in news media coverage |
| **historical_norms** | Literature | Norm extraction from historical texts |
| **rule_tuples** | Social Media | Rule classification from Reddit communities |

Each dagspace follows a consistent structure:

```
dagspaces/{name}/
├── cli.py                 # Hydra CLI entry point
├── orchestrator.py        # Pipeline execution engine
├── conf/                  # Configuration files
│   ├── config.yaml        # Base config
│   ├── pipeline/          # Pipeline DAG definitions
│   ├── prompt/            # LLM prompt templates
│   └── model/             # Model configurations
├── runners/               # Stage runner classes
└── stages/                # Stage implementations
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/trawler.git
cd trawler

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with uv (recommended)
uv pip install -e .
```

### Run a Pipeline

```bash
# Run the UAIR news analysis pipeline
python -m dagspaces.uair.cli \
  pipeline=full_event_pipeline \
  data.parquet_path=/path/to/articles.parquet

# Run historical norms extraction
python -m dagspaces.historical_norms.cli \
  pipeline=norm_extraction \
  data.parquet_path=/path/to/texts.parquet
```

### Debug Mode

```bash
# Run with sampling for quick iteration
python -m dagspaces.uair.cli \
  runtime.debug=true \
  runtime.sample_n=100 \
  data.parquet_path=/path/to/data.parquet
```

---

## Documentation

Complete documentation is available in `docs/`:

| Document | Description |
|----------|-------------|
| [User Guide](docs/USER_GUIDE.md) | Complete introduction and Quick Start |
| [Configuration Guide](docs/CONFIGURATION_GUIDE.md) | Pipeline recipes and config patterns |
| [Custom Stages Guide](docs/CUSTOM_STAGES_GUIDE.md) | Building custom processing stages |
| [Quick Reference](docs/QUICK_REFERENCE.md) | Command cheat sheet |

---

## Project Structure

```
trawler/
├── dagspaces/                    # Domain-specific pipelines
│   ├── uair/                     # News AI analysis
│   │   ├── cli.py                # CLI entry point
│   │   ├── orchestrator.py       # Pipeline orchestrator
│   │   ├── conf/                 # Configuration files
│   │   ├── runners/              # Stage runners
│   │   └── stages/               # Stage implementations
│   ├── historical_norms/         # Historical text analysis
│   └── rule_tuples/              # Social media rules
├── docs/                         # Documentation
├── notebooks/                    # Analysis notebooks
├── scripts/                      # Utility scripts
├── viz/                          # Visualization projects
└── pyproject.toml                # Project configuration
```

---

## Pipeline Configuration

Trawler uses [Hydra](https://hydra.cc/) for hierarchical configuration. Pipelines are defined as DAGs in YAML:

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
        stage: decompose
        depends_on: [classify]
        inputs: {articles: classified}
        outputs: [extracted]
```

**Override from command line:**

```bash
python -m dagspaces.uair.cli \
  pipeline=my_pipeline \
  model.batch_size=16 \
  runtime.sample_n=1000
```

---

## Deployment

### Local Execution

```bash
python -m dagspaces.uair.cli \
  hydra/launcher=null \
  runtime.sample_n=100
```

### SLURM Cluster

```bash
python -m dagspaces.uair.cli \
  pipeline=full_event_pipeline \
  hydra/launcher=g2_slurm_gpu_4x
```

---

## Development

### Creating a Custom Stage

1. **Implement stage function** in `dagspaces/{name}/stages/mystage.py`:

```python
def run_mystage(df, cfg):
    """Process dataframe with custom logic."""
    # Your processing logic
    return df
```

2. **Create runner** in `dagspaces/{name}/runners/mystage.py`:

```python
from .base import StageRunner

class MyStageRunner(StageRunner):
    stage_name = "mystage"
    
    def run(self, context):
        from ..stages.mystage import run_mystage
        return run_mystage(context.input_df, context.cfg)
```

3. **Register** in `dagspaces/{name}/runners/__init__.py`:

```python
STAGE_REGISTRY["mystage"] = MyStageRunner()
```

See [Custom Stages Guide](docs/CUSTOM_STAGES_GUIDE.md) for details.

---

## Example Dagspaces

### UAIR: News AI Analysis

Analyze AI-related risks and benefits in news coverage:

```bash
python -m dagspaces.uair.cli \
  pipeline=classify_risks_and_benefits_from_decompose \
  data.parquet_path=/data/news_articles.parquet
```

Stages: `classify_relevance` → `decompose` → `verify` → `taxonomy` → `topic` → `synthesis`

### Historical Norms: Literature Analysis

Extract structured norms from historical texts:

```bash
python -m dagspaces.historical_norms.cli \
  pipeline=norm_extraction \
  data.parquet_path=/data/gutenberg_texts.parquet
```

Stages: `fetch_gutenberg` → `norm_reasoning` → `norm_extraction`

---

## Contributing

We welcome contributions! Areas of interest:

- **New Dagspaces**: Domain-specific pipeline configurations
- **Stages**: Additional processing capabilities  
- **Optimizations**: Performance improvements
- **Documentation**: Examples, tutorials, guides

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Built with:

- [Hydra](https://hydra.cc/) - Configuration management
- [Ray Data](https://docs.ray.io/en/latest/data/data.html) - Distributed processing
- [vLLM](https://docs.vllm.ai/) - LLM inference
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [SLURM](https://slurm.schedmd.com/) - Cluster scheduling
