#!/usr/bin/env python3
"""Remove GRPO merged-model copies from /scratch/$USER.

During GRPO training, ``dagspaces/grpo_training/stages/grpo_training.py``
copies the SFT-merged base model to ``/scratch/$USER/grpo_merged_sft_<job_id>/``
so vLLM can load it from local SSD. These copies are ~15–20 GB each and
are not cleaned up after training.

This script deletes those directories in one go. By default it targets
``/scratch/$USER`` and matches any directory whose name starts with
``grpo_merged_sft``.

Usage:
    python -m scripts.clean_scratch_models --dry-run
    python -m scripts.clean_scratch_models -y

    # Protect a job's directory (e.g. a currently running SLURM job):
    python -m scripts.clean_scratch_models --keep-job-id 712345 -y

    # Different scratch root / prefix:
    python -m scripts.clean_scratch_models --root /scratch/me --prefix foo_
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import sys
from typing import List, Tuple


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.lstat(fp).st_size
            except OSError:
                pass
    return total


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _collect(root: str, prefix: str, keep_ids: set[str]) -> List[str]:
    if not os.path.isdir(root):
        raise SystemExit(f"not a directory: {root}")
    hits: List[str] = []
    for name in sorted(os.listdir(root)):
        if not name.startswith(prefix):
            continue
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        # Skip if the trailing suffix after the prefix matches a kept id.
        # Accepts names like ``grpo_merged_sft_712345``.
        tail = name[len(prefix):].lstrip("_")
        if tail and tail in keep_ids:
            continue
        hits.append(full)
    return hits


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_root = f"/scratch/{getpass.getuser()}"
    ap.add_argument("--root", default=default_root,
                    help=f"scratch root to clean (default: {default_root})")
    ap.add_argument("--prefix", default="grpo_merged_sft",
                    help="directory-name prefix to match (default: grpo_merged_sft)")
    ap.add_argument("--keep-job-id", action="append", default=[], metavar="ID",
                    help="job id to protect; can be repeated")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be deleted, delete nothing")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="skip confirmation prompt")
    args = ap.parse_args(argv)

    keep_ids = set(args.keep_job_id)
    hits = _collect(args.root, args.prefix, keep_ids)

    if not hits:
        print(f"no directories with prefix {args.prefix!r} under {args.root!r}")
        return 0

    print(f"Scanning sizes for {len(hits)} dir(s) under {args.root!r}...")
    total = 0
    sized: List[Tuple[str, int]] = []
    for path in hits:
        sz = _dir_size_bytes(path)
        sized.append((path, sz))
        total += sz

    for path, sz in sized:
        print(f"  {_human(sz):>10s}  {path}")
    print(f"TOTAL: {len(sized)} dir(s), {_human(total)}")
    if keep_ids:
        print(f"  (kept job ids: {sorted(keep_ids)})")

    if args.dry_run:
        print("dry-run: nothing deleted")
        return 0

    if not args.yes:
        reply = input(f"DELETE {len(sized)} dir(s) ({_human(total)})? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("aborted")
            return 1

    n_ok = 0
    for path, _sz in sized:
        try:
            shutil.rmtree(path)
            n_ok += 1
        except Exception as e:
            print(f"  FAILED {path}: {e}", file=sys.stderr)
    print(f"removed {n_ok}/{len(sized)} dirs")
    return 0 if n_ok == len(sized) else 2


if __name__ == "__main__":
    raise SystemExit(main())
