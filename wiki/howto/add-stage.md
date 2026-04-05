# How to add a stage to an existing dagspace

A "stage" is one node in the pipeline DAG. Adding one is 4 files + 1 pipeline edit.

## 1. Stage function

`dagspaces/{name}/stages/mystage.py`:

```python
from __future__ import annotations
import pandas as pd

def run_mystage(
    input_path: str,
    output_path: str,
    cfg,              # OmegaConf node-level config (per-node overrides merged in)
    wandb_logger=None,
) -> dict:
    df = pd.read_parquet(input_path)
    # ... do work ...
    df.to_parquet(output_path)
    return {"n_rows": len(df)}
```

## 2. Runner class

`dagspaces/{name}/runners/mystage.py`:

```python
from __future__ import annotations
from dagspaces.common.runners.base import StageRunner
from ..stages.mystage import run_mystage

class MyStageRunner(StageRunner):
    stage_name = "mystage"

    def run(self, context):
        input_path = context.inputs["dataset"]
        output_path = context.output_paths["dataset"]
        metadata = run_mystage(
            input_path=input_path,
            output_path=output_path,
            cfg=context.cfg,
            wandb_logger=getattr(context, "wandb_logger", None),
        )
        # Return dagspace's StageResult (each dagspace defines its own)
        from ..orchestrator import StageResult
        return StageResult(
            outputs={"dataset": output_path},
            metadata=metadata,
        )
```

## 3. Register in stage registry

`dagspaces/{name}/runners/__init__.py` — add to the lazy-import block in `get_stage_registry()`:

```python
def get_stage_registry():
    global _STAGE_REGISTRY
    if _STAGE_REGISTRY is None:
        # ... existing imports ...
        from .mystage import MyStageRunner
        _STAGE_REGISTRY = {
            # ... existing entries ...
            "mystage": MyStageRunner(),
        }
    return _STAGE_REGISTRY.copy()
```

Keep imports **inside** the function — the registry is lazy by design.

## 4. Add to a pipeline YAML

`dagspaces/{name}/conf/pipeline/my_pipeline.yaml`:

```yaml
pipeline:
  graph:
    nodes:
      mystage:
        stage: mystage
        depends_on: [upstream_node]
        inputs:
          dataset: upstream_node.dataset     # or a source name
        outputs:
          dataset:
            path: outputs/mystage/out.parquet
            type: parquet
        launcher: slurm_cpu                  # pick from common/conf/hydra/launcher/
        config:                              # optional per-node overrides
          mystage:
            my_param: 42
        wandb_suffix: mystage
```

## 5. Prompt config (only if using an LLM)

`dagspaces/{name}/conf/prompt/mystage.yaml`:

```yaml
# @package _global_
prompt:
  system: |
    You are ...
  user_template: |
    Given {input}, produce ...
```

Load via Hydra group defaults in the pipeline's base `config.yaml`.

## Test

```bash
python -m dagspaces.{name}.cli pipeline=my_pipeline \
  runtime.debug=true runtime.sample_n=5 hydra/launcher=null
```

Check outputs at `outputs/YYYY-MM-DD_*/HH-MM-SS/mystage/`.
