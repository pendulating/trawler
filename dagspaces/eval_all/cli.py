"""Run all evaluation benchmarks for a given model.

Hydra dagspace that dispatches to each eval dagspace as a subprocess.
Uses slurm_monitor launcher so `-m` submits the orchestrator itself
to SLURM, matching the pattern of all other dagspaces.

Usage:
    python -m dagspaces.eval_all.cli -m model=qwen3.5-9b/base
    python -m dagspaces.eval_all.cli -m model=qwen3.5-9b/base runtime.skip_vlm=true
    python -m dagspaces.eval_all.cli -m model=qwen3.5-9b/base runtime.debug=true runtime.sample_n=5

    # Local (no SLURM):
    python -m dagspaces.eval_all.cli model=qwen3.5-9b/base hydra/launcher=null
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.stage_utils import ensure_dotenv

from .orchestrator import run_eval_all


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Evaluate a model on all benchmarks."""
    # Idempotent. Must run inside main() so it fires in submitit-launched
    # workers too (those import the module rather than executing __main__,
    # so a __main__-guarded call would skip them and downstream
    # ${oc.env:...}-style interpolations would fail.)
    ensure_dotenv()
    print(OmegaConf.to_yaml(cfg))
    run_eval_all(cfg)


if __name__ == "__main__":
    main()
