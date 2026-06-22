#!/usr/bin/env python3
"""Prototype + validate a 'truly contentless' classifier for SFT negative curation.

gold=False chunks split into:
  (a) truly contentless  — no inter-agent information transfer (preface, criticism,
      scene/landscape description). SHOULD remain an SFT negative ("abstain on empty").
  (b) has-exchange-no-norm — a real disclosure/conversation, just no prescriptive
      norm. Should NOT be an SFT negative (it taught "abstain on real flows").

The gold reasoning ALWAYS contains the boilerplate "...regulates the exchange of
information between agents" (the reason it's gold=False), so we cannot key on
"exchange". Instead detect POSITIVE story-interaction language. Absence => contentless.
"""
from __future__ import annotations

import sys
import pandas as pd

# Single source of truth — validate the PRODUCTION classifier, not a copy.
from dagspaces.grpo_training.stages.sft_data_prep import _is_contentless_chunk

GOLD = "/share/pierson/matt/n2s4cir/data/fiction10/ci_reasoning.parquet"


def is_contentless(reasoning_text: str) -> bool:
    return _is_contentless_chunk(reasoning_text)


def main():
    df = pd.read_parquet(GOLD)
    nf = df[df["has_information_exchange"] == False].copy()
    nf["rt"] = nf["ci_reasoning_text"].fillna("").astype(str)
    nf["contentless"] = nf["rt"].map(is_contentless)

    n = len(nf)
    nc = int(nf["contentless"].sum())
    n_pos = int((df["has_information_exchange"] == True).sum())
    print(f"gold=False chunks: {n}   (gold=True positives: {n_pos})")
    print(f"  CONTENTLESS keep-as-negative: {nc} ({100*nc/n:.0f}%)  -> capped at n_pos={n_pos}")
    print(f"  HAS-EXCHANGE drop-from-neg:   {n-nc} ({100*(n-nc)/n:.0f}%)")
    print()
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    keep = nf[nf["contentless"]].sample(min(14, nc), random_state=seed)
    drop = nf[~nf["contentless"]].sample(min(14, n-nc), random_state=seed)
    print("=== KEPT as contentless negative — want: NO real transfer (essay/scene/expository) ===")
    for i, (_, r) in enumerate(keep.iterrows()):
        print(f"  K{i}: {r['rt'][:200]}")
    print()
    print("=== DROPPED as has-exchange — want: a real conversation/disclosure between agents ===")
    for i, (_, r) in enumerate(drop.iterrows()):
        print(f"  D{i}: {r['rt'][:200]}")


if __name__ == "__main__":
    main()
