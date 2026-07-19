"""One-shot migration helper: rewrite the 7 remaining copied eval orchestrators
as thin stubs over the shared run loop, extracting each dagspace's
``_log_eval_metrics`` (and privacylens's ``_perturb_qualified_dagspace``)
verbatim via AST so the metric code is preserved byte-for-byte.

Run: python scripts/migrate_eval_orchestrators.py
Idempotent-ish: overwrites the target orchestrator.py files.
"""
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# dagspace -> (dagspace_name, output_subdir, job_prefix, wandb_dagspace_expr, extra_funcs)
# wandb_dagspace_expr is Python source for a callable taking cfg.
SPECS = {
    "goldcoin_hipaa": ("goldcoin_hipaa", "goldcoin_hipaa", "GoldCoin", 'lambda cfg: "goldcoin"', []),
    "vlm_geoprivacy_bench": ("vlm_geoprivacy_bench", "vlm_geoprivacy_bench", "VLM", 'lambda cfg: "vlm_geoprivacy"', []),
    "vlm_geoprivacy_aug": ("vlm_geoprivacy_aug", "vlm_geoprivacy_aug", "VLM", 'lambda cfg: "vlm_geoprivacy_aug"', []),
    "confaide": ("confaide", "confaide", "CONFAIDE", 'lambda cfg: "confaide"', []),
    "cirl_vignettes": ("cirl_vignettes", "cirl_vignettes", "CIRLVignettes", 'lambda cfg: "cirl_vignettes"', []),
    "simpleqa_verified": ("simpleqa_verified", "simpleqa_verified", "SimpleQAVerified", 'lambda cfg: "simpleqa_verified"', []),
    "ci_heuristic": ("ci_heuristic", "ci_heuristic", "CIH", 'lambda cfg: "ci_heuristic"', []),
    "privacylens": ("privacylens", "privacylens_eval", "PLens", "_perturb_qualified_dagspace", ["_perturb_qualified_dagspace"]),
}

TEMPLATE = '''"""Orchestrator for the {ds} dagspace.

Thin wrapper over the shared eval run loop in ``dagspaces/common/orchestrator.py``
(see ``wiki/jul19_orchestrator_unification_plan.md``).  The only {ds}-specific
code below is the metric formatter{extra_note}; everything else is the generic
loop parameterized by ``ORCHESTRATOR_HOOKS``.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from dagspaces.common.orchestrator import (
    OrchestratorHooks,
    run_experiment as _run_experiment,
)

# Re-exported for the generic SLURM worker, which recovers the registry via
# ``importlib.import_module("dagspaces.{ds}.orchestrator").get_stage_registry()``.
from .runners import get_stage_registry  # noqa: F401

_CONF_DIR = os.path.join(os.path.dirname(__file__), "conf")


{extra_funcs}
{log_metrics}

ORCHESTRATOR_HOOKS = OrchestratorHooks(
    dagspace_module="dagspaces.{ds}.orchestrator",
    dagspace_name="{name}",
    output_subdir="{subdir}",
    job_prefix="{prefix}",
    config_dir=_CONF_DIR,
    log_eval_metrics=_log_eval_metrics,
    wandb_dagspace={wandb},
    use_srun=False,
)


def run_experiment(cfg: DictConfig) -> None:
    """Execute the {ds} evaluation pipeline."""
    _run_experiment(cfg, ORCHESTRATOR_HOOKS)
'''


def _func_source(src: str, name: str) -> str:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise SystemExit(f"function {name} not found")


def migrate(ds: str):
    name, subdir, prefix, wandb, extra = SPECS[ds]
    path = os.path.join(REPO, "dagspaces", ds, "orchestrator.py")
    with open(path) as f:
        src = f.read()

    log_metrics = _func_source(src, "_log_eval_metrics")
    extra_funcs = ""
    extra_note = ""
    if extra:
        parts = [_func_source(src, fn) for fn in extra]
        extra_funcs = "\n\n\n".join(parts) + "\n\n"
        extra_note = " (and " + ", ".join(extra) + ")"

    out = TEMPLATE.format(
        ds=ds, name=name, subdir=subdir, prefix=prefix, wandb=wandb,
        extra_funcs=extra_funcs, log_metrics=log_metrics, extra_note=extra_note,
    )
    with open(path, "w") as f:
        f.write(out)
    print(f"migrated {ds}: {len(src.splitlines())} -> {len(out.splitlines())} lines")


if __name__ == "__main__":
    targets = sys.argv[1:] or list(SPECS)
    for ds in targets:
        migrate(ds)
