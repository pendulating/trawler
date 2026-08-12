#!/usr/bin/env python3
"""Remove multirun/ date folders on or before a cutoff date.

Top-level folders under ``multirun/`` are named ``YYYY-MM-DD`` (one per
date a sweep was launched). This script deletes every such folder whose
date is ``<= --before``.

Usage:
    python -m scripts.clean_multirun --before 2026-03-19 --dry-run
    python -m scripts.clean_multirun --before 2026-03-19 -y

    # Different target directory:
    python -m scripts.clean_multirun --before 2026-03-19 --path outputs
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import sys
from typing import List, Tuple

# Hydra writes bare ``YYYY-MM-DD`` dirs, but the sweep launchers name theirs
# ``YYYY-MM-DD_<experiment>``. Anchoring on ``$`` matched 1 of 168 entries under
# multirun/, so this script silently reclaimed almost nothing until 2026-08-12.
_DATE_DIR_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[_-].*)?$")


def _parse_date(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except ValueError as e:
        raise SystemExit(f"invalid date {s!r}: expected YYYY-MM-DD") from e


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


def _load_protected(path: str | None) -> set[str]:
    """Read a newline-delimited protect list into a set of basenames.

    Entries may be bare names (``2026-07-28_grpo_m2_full``) or paths
    (``multirun/2026-07-28_grpo_m2_full/21-31-11/...``); only the run-directory
    component is compared, so a deep artifact path protects its whole run dir.
    Blank lines and ``#`` comments are ignored.
    """
    if not path:
        return set()
    out: set[str] = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.replace("\\", "/").split("/") if p]
            for i, p in enumerate(parts):
                if _DATE_DIR_RE.match(p):
                    out.add(p)
                    break
            else:
                if parts:
                    out.add(parts[-1])
    return out


def _collect(root: str, cutoff: dt.date,
             protected: set[str] | None = None) -> List[Tuple[str, dt.date]]:
    if not os.path.isdir(root):
        raise SystemExit(f"not a directory: {root}")
    protected = protected or set()
    hits: List[Tuple[str, dt.date]] = []
    skipped: List[str] = []
    for name in sorted(os.listdir(root)):
        m = _DATE_DIR_RE.match(name)
        if not m:
            continue
        try:
            d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        if d > cutoff:
            continue
        if name in protected:
            skipped.append(name)
            continue
        hits.append((os.path.join(root, name), d))
    if skipped:
        print(f"protected ({len(skipped)} kept despite matching the cutoff):")
        for name in skipped:
            print(f"  KEEP  {name}")
    return hits


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--before", required=True, metavar="YYYY-MM-DD",
                    help="remove date-folders ON OR BEFORE this date (inclusive)")
    ap.add_argument("--path", default="multirun",
                    help="target directory (default: multirun)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be deleted, delete nothing")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="skip confirmation prompt")
    ap.add_argument("--protect-from", metavar="FILE", default=None,
                    help="newline-delimited list of run dirs (or paths into "
                         "them) that must never be deleted")
    args = ap.parse_args(argv)

    cutoff = _parse_date(args.before)
    hits = _collect(args.path, cutoff, _load_protected(args.protect_from))

    if not hits:
        print(f"no date-folders in {args.path!r} with date <= {cutoff}")
        return 0

    print(f"Scanning sizes for {len(hits)} folder(s) under {args.path!r}...")
    total = 0
    sized: List[Tuple[str, dt.date, int]] = []
    for path, d in hits:
        sz = _dir_size_bytes(path)
        sized.append((path, d, sz))
        total += sz

    # Print table
    for path, d, sz in sized:
        print(f"  {d}  {_human(sz):>10s}  {path}")
    print(f"TOTAL: {len(sized)} folder(s), {_human(total)} on or before {cutoff}")

    if args.dry_run:
        print("dry-run: nothing deleted")
        return 0

    if not args.yes:
        reply = input(f"DELETE {len(sized)} folder(s) ({_human(total)})? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("aborted")
            return 1

    n_ok = 0
    for path, _d, _sz in sized:
        try:
            shutil.rmtree(path)
            n_ok += 1
        except Exception as e:
            print(f"  FAILED {path}: {e}", file=sys.stderr)
    print(f"removed {n_ok}/{len(sized)} folders")
    return 0 if n_ok == len(sized) else 2


if __name__ == "__main__":
    raise SystemExit(main())
