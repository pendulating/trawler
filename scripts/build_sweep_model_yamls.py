#!/usr/bin/env python3
"""Post-training helper for the (λ × ρ) sweep.

Scans the multirun output directories produced by
``scripts/launch_lambda_ratio_sweep.sh`` and writes:

1. One eval-time model yaml per trained cell, under
   ``dagspaces/common/conf/model/qwen3.5-9b/grpo-l<L>-r<R>.yaml``,
   pointing at the cell's merged base + LoRA checkpoint paths.

2. A single Hydra sweep config at
   ``dagspaces/eval_all/conf/sweep/contrastive_lambda_ratio.yaml``
   that runs ``eval_all`` over every generated model yaml.

Usage:
    # Auto-discover the latest sweep runs
    python scripts/build_sweep_model_yamls.py

    # Specify run dirs explicitly
    python scripts/build_sweep_model_yamls.py \\
        --run-dir multirun/2026-05-15_lambda_axis_sweep/14-12-00 \\
        --run-dir multirun/2026-05-15_ratio_axis_sweep/22-30-00 \\
        --run-dir multirun/2026-05-16_offaxis_sweep/09-45-00

Naming convention: λ=0.25, ρ=0.10 → ``grpo-l025-r010.yaml`` (3-digit padded
percent representation, no decimal point — avoids dots in hydra config keys).

Refuses to overwrite an existing yaml unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
MODEL_DIR = PROJECT_ROOT / "dagspaces/common/conf/model/qwen3.5-9b"
EVAL_SWEEP_PATH = (
    PROJECT_ROOT
    / "dagspaces/eval_all/conf/sweep/contrastive_lambda_ratio.yaml"
)

MODEL_YAML_TEMPLATE = """# @package _global_
# Auto-generated from the COLM (λ × ρ) sweep.
# λ = {lam}, ρ = {rho}.
# Source training run: {run_dir}
# Judge during training: Qwen3.6-27B (klara:8002).
model:
  model_source: {merged_dir}
  model_family: qwen3.5
  lora_path: {lora_dir}
  chat_template_kwargs:
    enable_thinking: false
  engine_kwargs:
    max_model_len: 16384
    max_num_seqs: 16
    tensor_parallel_size: 1
    trust_remote_code: true
    enforce_eager: true
    enable_lora: true
    max_lora_rank: 64
  batch_size: 0
  concurrency: 1
"""


def fmt_pct(x: float) -> str:
    """Format λ or ρ as 3-digit zero-padded percent (0.25 → '025')."""
    return f"{int(round(x * 100)):03d}"


def discover_cells(run_dirs: list[Path]) -> list[dict]:
    """Walk each run dir and collect (lam, rho, merged_dir, lora_dir, run_dir)."""
    cells = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"  ! skipping (does not exist): {run_dir}", file=sys.stderr)
            continue
        for cell_dir in sorted(run_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            ckpt_dir = (
                cell_dir
                / "grpo_only_online_external"
                / "outputs"
                / "grpo"
                / "checkpoint"
            )
            meta_path = ckpt_dir / "training_metadata.json"
            if not meta_path.exists():
                print(
                    f"  ! skipping {cell_dir.name}: no training_metadata.json",
                    file=sys.stderr,
                )
                continue
            with meta_path.open() as f:
                meta = json.load(f)
            lam = float(meta.get("contrastive_lambda", -1))
            rho = float(meta.get("contrastive_ratio", -1))
            merged_dir = ckpt_dir / "_merged_sft"
            if not merged_dir.exists():
                print(
                    f"  ! skipping λ={lam}, ρ={rho}: no _merged_sft dir",
                    file=sys.stderr,
                )
                continue
            cells.append(
                dict(
                    lam=lam,
                    rho=rho,
                    merged_dir=str(merged_dir),
                    lora_dir=str(ckpt_dir),
                    run_dir=str(cell_dir),
                )
            )
    return cells


def autodiscover_run_dirs() -> list[Path]:
    """Find the most-recent sweep dirs by experiment.name pattern."""
    multirun = PROJECT_ROOT / "multirun"
    candidates = []
    for sweep_name in (
        "lambda_axis_sweep",
        "ratio_axis_sweep",
        "offaxis_sweep",
    ):
        matches = sorted(multirun.glob(f"*_{sweep_name}/*"))
        if matches:
            # take the most recent timestamp
            candidates.append(max(matches))
    return candidates


def write_model_yaml(cell: dict, force: bool) -> Path:
    name = f"grpo-l{fmt_pct(cell['lam'])}-r{fmt_pct(cell['rho'])}.yaml"
    path = MODEL_DIR / name
    if path.exists() and not force:
        print(f"  ! exists (use --force to overwrite): {name}")
        return path
    content = MODEL_YAML_TEMPLATE.format(
        lam=cell["lam"],
        rho=cell["rho"],
        run_dir=cell["run_dir"],
        merged_dir=cell["merged_dir"],
        lora_dir=cell["lora_dir"],
    )
    path.write_text(content)
    print(f"  ✓ wrote {name}")
    return path


def write_eval_sweep_yaml(cells: list[dict]) -> Path:
    EVAL_SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_keys = sorted(
        f"qwen3.5-9b/grpo-l{fmt_pct(c['lam'])}-r{fmt_pct(c['rho'])}"
        for c in cells
    )
    indented = ",\n        ".join(model_keys)
    body = f"""# @package _global_
# Auto-generated COLM eval sweep — runs `all_benchmarks` for every cell
# of the (λ × ρ) sweep produced by `scripts/launch_lambda_ratio_sweep.sh`.
#
# {len(cells)} cells total. Regenerate with:
#   python scripts/build_sweep_model_yamls.py
#
# Usage:
#   python -m dagspaces.eval_all.cli --multirun +sweep=contrastive_lambda_ratio

hydra:
  mode: MULTIRUN
  launcher:
    array_parallelism: 2
  sweeper:
    params:
      model: >-
        {indented}
"""
    EVAL_SWEEP_PATH.write_text(body)
    print(f"  ✓ wrote {EVAL_SWEEP_PATH.relative_to(PROJECT_ROOT)}")
    return EVAL_SWEEP_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        default=[],
        help="Sweep run dir (multirun/<date>_<sweep>_sweep/<time>). "
        "Can be passed multiple times. If omitted, auto-discovers the "
        "latest dirs for lambda_axis_sweep, ratio_axis_sweep, offaxis_sweep.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model yamls.",
    )
    args = parser.parse_args()

    run_dirs = args.run_dir or autodiscover_run_dirs()
    if not run_dirs:
        print("ERROR: no run dirs given and none auto-discovered.", file=sys.stderr)
        return 1

    print("Scanning sweep run dirs:")
    for d in run_dirs:
        print(f"  - {d.relative_to(PROJECT_ROOT) if d.is_absolute() else d}")

    cells = discover_cells(run_dirs)
    if not cells:
        print("ERROR: no completed cells found.", file=sys.stderr)
        return 2

    print(f"\nFound {len(cells)} cells.")
    print("Writing model yamls:")
    for cell in cells:
        write_model_yaml(cell, force=args.force)

    print("\nWriting eval sweep yaml:")
    write_eval_sweep_yaml(cells)

    print(
        f"\nDone. Launch evaluation with:\n"
        f"  python -m dagspaces.eval_all.cli --multirun +sweep=contrastive_lambda_ratio"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
