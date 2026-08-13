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

from dagspaces.common.cli import make_cli

from .orchestrator import run_eval_all

# Explicit description: this module's docstring carries usage lines too.
main = make_cli(run_eval_all, description="Evaluate a model on all benchmarks.")


if __name__ == "__main__":
    main()
