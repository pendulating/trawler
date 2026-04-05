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

_DATE_DIR_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


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


def _collect(root: str, cutoff: dt.date) -> List[Tuple[str, dt.date]]:
    if not os.path.isdir(root):
        raise SystemExit(f"not a directory: {root}")
    hits: List[Tuple[str, dt.date]] = []
    for name in sorted(os.listdir(root)):
        m = _DATE_DIR_RE.match(name)
        if not m:
            continue
        try:
            d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        if d <= cutoff:
            hits.append((os.path.join(root, name), d))
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
    args = ap.parse_args(argv)

    cutoff = _parse_date(args.before)
    hits = _collect(args.path, cutoff)

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
