#!/usr/bin/env python3
"""How much do the top-k retrieved norms agree about a flow's appropriateness?

Settles the k question for the realigned R-OUTCOME (2026-07-25): the norm is the
CLASSIFIER of an extracted flow (Raz force -> FORCE_TO_APPROPRIATENESS), so each
scored flow needs ONE gold label. That forces a choice:

  k=1              top-1 similarity supplies the gold  (retrieval noise -> bad gold)
  k=3 majority     polarity vote over the top 3        (robust to one bad hit)
  k=3 agreement    score the flow only when all k agree (highest-precision gold,
                   fewer scored flows; disagreement becomes an explicit
                   abstention instead of silent gold corruption)

If agreement is high, k=1 is fine and gating is free. If it is low, k=1 has been
feeding noise into the gold label and gating is mandatory.

Uses the running embedding server (:8001) — no training GPU needed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, "/share/pierson/matt/UAIR")
from dagspaces.common.stage_utils import ensure_dotenv  # noqa: E402
from dagspaces.grpo_training.stages.clients import EmbeddingClient  # noqa: E402
from dagspaces.grpo_training.stages.deontic import (  # noqa: E402
    FORCE_TO_APPROPRIATENESS,
)
from dagspaces.grpo_training.stages.probes import flow_to_query  # noqa: E402

UNIV_DIR = ("multirun/2026-07-23_universe_fiction10_gemma4/15-43-41/"
            "norm_universe_only/outputs/norm_universe")
FLOWS = ("outputs/2026-07-12_fiction10_flows_gemma4/23-14-17/"
         "COLM_flows_fiction_gemma4/outputs/ci_extraction/ci_flows.parquet")
FIELD_MAP = {
    "ci_subject": "subject", "ci_sender": "sender", "ci_recipient": "recipient",
    "ci_information_type": "information_type",
    "ci_transmission_principle": "transmission_principle",
    "ci_context": "context",
}


def eligible_idx(norms: list[dict]) -> list[int]:
    out = []
    for i, n in enumerate(norms):
        force = str(n.get("normative_force") or "").strip().lower()
        if (n.get("governs_info_flow") is True
                and force in FORCE_TO_APPROPRIATENESS
                and str(n.get("context") or "").strip()):
            out.append(i)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/2026-07-25_norm_agreement")
    args = ap.parse_args()

    ensure_dotenv()
    url = os.environ.get("EMBEDDING_SERVER_URL")
    print(f"[agree] embedding server: {url}")

    universe = json.load(open(f"{UNIV_DIR}/norm_universes.json"))
    flows_df = pd.read_parquet(FLOWS)

    # Sample flows, grouped by book so we embed/retrieve per book.
    rows = flows_df.to_dict("records")
    random.Random(args.seed).shuffle(rows)
    sample = rows[: args.n]
    print(f"[agree] sampled {len(sample)} reference flows")

    # The served model id must match exactly: vLLM 404s on model lookup, and
    # EmbeddingClient's default model_name ("default") never matches.
    import requests
    served = requests.get(f"{url.rstrip('/')}/v1/models", timeout=10).json()
    model_name = served["data"][0]["id"]
    print(f"[agree] served embedding model: {model_name}")
    client = EmbeddingClient(base_url=url, model_name=model_name)
    by_book: dict[str, list[dict]] = {}
    for r in sample:
        by_book.setdefault(str(r["gutenberg_id"]), []).append(r)

    stats = Counter()
    per_flow = []
    for gid, frows in by_book.items():
        norms = universe.get(gid) or []
        idx = eligible_idx(norms)
        if not idx:
            continue
        emb = np.load(f"{UNIV_DIR}/embeddings/{gid}.npy")
        emb = emb[idx]                      # eligible-only matrix
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)

        queries = []
        for r in frows:
            flow = {d: str(r[s]) for s, d in FIELD_MAP.items()
                    if r.get(s) is not None and str(r.get(s)).strip()
                    and str(r.get(s)) != "nan"}
            queries.append(flow_to_query(flow))
        q = client.encode_batch(queries)
        q = np.asarray(q, dtype=np.float32)
        q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-9, None)

        sims = q @ emb.T
        topk = np.argsort(-sims, axis=1)[:, : args.k]
        for row_i, tk in enumerate(topk):
            pols = []
            for j in tk:
                force = str(norms[idx[j]].get("normative_force") or "").strip().lower()
                pols.append(FORCE_TO_APPROPRIATENESS.get(force))
            c = Counter(pols)
            top1 = pols[0]
            majority, maj_n = c.most_common(1)[0]
            unanimous = len(c) == 1
            stats["total"] += 1
            stats["unanimous"] += unanimous
            stats[f"maj_{maj_n}_of_{args.k}"] += 1
            stats["top1_differs_from_majority"] += (top1 != majority)
            stats[f"top1_{top1}"] += 1
            # Validation: does retrieval-derived gold agree with the TEACHER's
            # own appropriateness judgment for this same flow? If it agrees at
            # chance, the norm→gold premise is broken independent of k.
            teacher = str(frows[row_i].get("ci_appropriateness") or "").strip().lower()
            if teacher in ("appropriate", "inappropriate"):
                stats[f"teacher_{teacher}"] += 1
                stats[f"pair_{top1}_vs_{teacher}"] += 1
                stats["teacher_labeled"] += 1
                stats["agree_with_teacher"] += (top1 == teacher)
            per_flow.append({"gutenberg_id": gid, "polarities": pols,
                             "top1": top1, "majority": majority,
                             "unanimous": bool(unanimous), "teacher": teacher})

    tot = stats["total"] or 1
    print(f"\n=== top-{args.k} polarity agreement over {tot} flows ===")
    print(f"unanimous (all {args.k} agree) : {stats['unanimous']:5d} "
          f"({stats['unanimous']/tot:.1%})   <- gating keeps these")
    for m in range(args.k, 0, -1):
        key = f"maj_{m}_of_{args.k}"
        if stats[key]:
            print(f"  majority {m}/{args.k}            : {stats[key]:5d} ({stats[key]/tot:.1%})")
    print(f"top-1 != majority              : {stats['top1_differs_from_majority']:5d} "
          f"({stats['top1_differs_from_majority']/tot:.1%})  <- k=1 would have been wrong here")
    print("\ntop-1 gold polarity mix:")
    for pol in ("appropriate", "inappropriate", None):
        k2 = f"top1_{pol}"
        if stats[k2]:
            print(f"  {str(pol):15s}: {stats[k2]:5d} ({stats[k2]/tot:.1%})")

    # Class-conditional agreement: does gating preferentially delete the
    # minority (inappropriate) class — the Forbid-recall signal?
    print("\nunanimity BY top-1 polarity (does gating hurt the minority class?):")
    for pol in ("appropriate", "inappropriate"):
        grp = [r for r in per_flow if r["top1"] == pol]
        if grp:
            u = sum(r["unanimous"] for r in grp)
            print(f"  {pol:15s}: n={len(grp):5d}  unanimous {u/len(grp):.1%}")
    kept = [r for r in per_flow if r["unanimous"]]
    if kept:
        inap = sum(r["top1"] == "inappropriate" for r in kept)
        print(f"\nafter agreement-gating: {len(kept)} flows kept, "
              f"inappropriate share {inap/len(kept):.1%} "
              f"(vs {stats['top1_inappropriate']/tot:.1%} ungated)")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "agreement.json"), "w") as f:
        json.dump({"stats": dict(stats), "k": args.k, "n": tot}, f, indent=2)
    print(f"\n[agree] wrote {args.out}/agreement.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
