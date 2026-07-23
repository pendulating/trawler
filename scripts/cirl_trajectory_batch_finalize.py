"""Finalize a CIRL-Vignettes trajectory batch-export run.

Thin CLI wrapper — the logic lives in
``dagspaces.privacylens.cirl_protocol.stages.finalize_async`` (shared with the
eval_all async-judge flow's ``cirl_finalize_async`` stage; this script is
the manual path for OpenAI-Batch runs where you fetched ``output.jsonl``
yourself via ``python -m dagspaces.common.batch_api fetch``).

Usage::

    python scripts/cirl_trajectory_batch_finalize.py \\
        --run-dir outputs/2026-04-10_cirl_vignettes/14-22-01/cirl_vignettes
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", required=True,
        help="pipeline.output_root from a cirl_trajectory_batch run, "
             "e.g. outputs/YYYY-MM-DD_cirl_vignettes/HH-MM-SS/cirl_vignettes",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"error: run-dir not found: {run_dir}", file=sys.stderr)
        return 1

    from dagspaces.privacylens.cirl_protocol.stages.finalize_async import (
        finalize_trajectory_async,
    )

    result = finalize_trajectory_async(run_dir)
    metrics = result["metrics"]

    print("=" * 60)
    print("  CIRL-VIGNETTES BATCH-FINALIZE RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
