#!/usr/bin/env python3
"""Merge the act_polarity backfill INTO a norm universe (R-DIRECT prerequisite).

R-DIRECT derives gold from the retrieved norm itself, and NormRetriever returns
norm dicts (not indices) — so `act_polarity` must live ON the norms rather than
in a side lookup keyed by index. This writes a new universe directory with the
field applied, leaving the original untouched.

Without this, flow_appropriateness falls back to "performing" and 19% of gold
labels are inverted (measured 2026-07-25).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

SRC = ("multirun/2026-07-23_universe_fiction10_gemma4/15-43-41/"
       "norm_universe_only/outputs/norm_universe")
POL = "outputs/2026-07-25_act_polarity_backfill/act_polarity.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--polarity", default=POL)
    ap.add_argument("--out", default="outputs/2026-07-25_universe_fiction10_polarity")
    args = ap.parse_args()

    universe = json.load(open(f"{args.src}/norm_universes.json"))
    polarity = json.load(open(args.polarity))
    os.makedirs(args.out, exist_ok=True)

    applied = missing = 0
    for gid, norms in universe.items():
        for i, n in enumerate(norms):
            pol = polarity.get(gid, {}).get(str(i))
            if pol:
                n["act_polarity"] = pol
                applied += 1
            else:
                missing += 1
    json.dump(universe, open(f"{args.out}/norm_universes.json", "w"), indent=2)

    # Embeddings are index-aligned and unchanged — copy so the directory is a
    # drop-in replacement for NORM_UNIVERSES_PATH / NORM_EMBEDDINGS_PATH.
    if os.path.isdir(f"{args.src}/embeddings") and not os.path.isdir(f"{args.out}/embeddings"):
        shutil.copytree(f"{args.src}/embeddings", f"{args.out}/embeddings")

    tot = applied + missing
    print(f"[apply] act_polarity set on {applied}/{tot} norms "
          f"({applied/tot:.1%}); {missing} left unlabelled (non-eligible)")
    print(f"[apply] wrote {args.out}/norm_universes.json (+ embeddings/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
