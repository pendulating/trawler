# How to add a new dagspace

Use this when you need a fundamentally new pipeline (e.g. a new extraction source or training regime). For a new CI benchmark specifically, see [add-eval-benchmark.md](add-eval-benchmark.md).

## Scaffold

```
dagspaces/<name>/
├── __init__.py
├── cli.py
├── orchestrator.py
├── conf/
│   ├── config.yaml
│   ├── pipeline/<default>.yaml
│   └── prompt/<stage>.yaml
├── runners/
│   ├── __init__.py       # get_stage_registry()
│   └── <stage>.py
└── stages/
    └── <stage>.py
```

## Files

### `cli.py`

```python
import hydra
from omegaconf import DictConfig
from dagspaces.common.stage_utils import ensure_dotenv

ensure_dotenv()

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from .orchestrator import run
    run(cfg)

if __name__ == "__main__":
    main()
```

### `conf/config.yaml`

```yaml
defaults:
  - _self_
  - pipeline: default
  - model: qwen3.5-9b/base
  - override hydra/launcher: slurm_monitor

hydra:
  searchpath:
    - pkg://dagspaces.common.conf

runtime:
  debug: false
  sample_n: 0
```

### `orchestrator.py`

Thin wrapper around `dagspaces/common/orchestrator.py`. Define a dagspace-local `StageExecutionContext` and `StageResult` and call the shared DAG runner. Simplest path: look at `dagspaces/goldcoin_hipaa/orchestrator.py` or `dagspaces/grpo_training/orchestrator.py` and copy the structure.

### `runners/__init__.py`

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import StageRunner

_STAGE_REGISTRY = None

def get_stage_registry():
    global _STAGE_REGISTRY
    if _STAGE_REGISTRY is None:
        from .my_stage import MyStageRunner
        _STAGE_REGISTRY = {"my_stage": MyStageRunner()}
    return _STAGE_REGISTRY.copy()
```

### `conf/pipeline/default.yaml`

```yaml
# @package _global_
defaults:
  - override /hydra/launcher: slurm_monitor

pipeline:
  output_root: ${hydra:run.dir}/<name>

  sources:
    input_data:
      path: ${oc.env:MY_INPUT_PATH}
      type: parquet

  graph:
    nodes:
      my_stage:
        stage: my_stage
        depends_on: []
        inputs: {dataset: input_data}
        outputs:
          dataset: {path: outputs/my_stage/out.parquet, type: parquet}
        launcher: slurm_gpu_1x
        wandb_suffix: my_stage
```

## Run

```bash
python -m dagspaces.<name>.cli pipeline=default \
  hydra/launcher=null runtime.debug=true runtime.sample_n=5
```

## Parallels

All existing dagspaces share identical scaffolding. The fastest path is: **copy the closest existing dagspace and rename**. For eval work → copy `goldcoin_hipaa`. For training work → copy `grpo_training`. For extraction work → copy `historical_norms`.
