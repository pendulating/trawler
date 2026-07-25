#!/usr/bin/env python3
"""Why does the answerer still refuse 44% of probes given TEACHER gold flows?

Hypothesis (P3): probe–extraction *correspondence*. Probes are templated from a
retrieved NORM's (subject, act, context); the extraction describes the chunk's
FLOWS. Retrieval relates them, but the probe scenario often is not the same
situation as any extracted flow — so the answerer legitimately cannot map the
question onto the evidence and returns cannot_determine.

Test: run the winning (`role`) prompt on teacher flows, then compare
probe↔extraction token overlap for ANSWERED vs REFUSED probes. If refusals have
systematically lower overlap, the residual is structural (probe construction),
not prompt calibration. Also dumps refusal cases for reading.

Read-only; live server; writes under outputs/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

import pandas as pd

sys.path.insert(0, "/share/pierson/matt/UAIR")
from dagspaces.common.stage_utils import ensure_dotenv  # noqa: E402
from dagspaces.grpo_training.stages.answerer_client import (  # noqa: E402
    AnswererClient,
    _parse_answers,
)
from scripts.select_answerer_prompt import (  # noqa: E402
    CANDIDATES,
    build_balanced_eval,
    FLOWS,
    POOLS,
)

_STOP = set("a an the is are was were be been being to of in on at for with "
            "and or if this that his her their its it he she they them from "
            "should whether answer yes no considering situation where setting "
            "involving information shared share".split())


def toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", str(s).lower())
            if w not in _STOP and len(w) > 2}


def overlap(probe: str, flows: list[dict]) -> float:
    """Max Jaccard-ish coverage of the probe by any single flow tuple."""
    p = toks(probe)
    if not p:
        return 0.0
    best = 0.0
    for f in flows:
        ft = toks(" ".join(str(v) for v in f.values()))
        if ft:
            best = max(best, len(p & ft) / len(p))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", type=int, default=12)
    ap.add_argument("--out", default="outputs/2026-07-25_probe_correspondence")
    args = ap.parse_args()

    ensure_dotenv()
    base = os.environ.get("VLLM_SERVER_URL") or os.environ.get("JUDGE_SERVER_URL")
    model = os.environ.get("JUDGE_MODEL_PATH", "")
    client = AnswererClient(base_url=base, model=model, temperature=0.0)
    system = CANDIDATES["role"]

    pools = pd.read_parquet(POOLS)
    flows_df = pd.read_parquet(FLOWS)
    evalset = build_balanced_eval(pools, flows_df, args.chunks, args.seed)
    print(f"[corr] {len(evalset)} chunks, role prompt, teacher flows\n")

    records = []
    for row in evalset:
        messages = [{"role": "system", "content": system},
                    {"role": "user",
                     "content": client.build_user(row["flows"], row["probes"])}]
        try:
            raw = client._post(client._request_body(messages))
            answers = _parse_answers(raw, len(row["probes"])) or \
                ["cannot_determine"] * len(row["probes"])
        except Exception as exc:
            print(f"  transport error: {exc}")
            continue
        for probe, gold, ans in zip(row["probes"], row["golds"], answers):
            records.append({
                "gutenberg_id": row["gutenberg_id"], "chunk_id": row["chunk_id"],
                "probe": probe, "gold": gold, "answer": ans,
                "overlap": overlap(probe, row["flows"]),
                "n_flows": len(row["flows"]),
                "flows": row["flows"],
            })

    answered = [r for r in records if r["answer"] != "cannot_determine"]
    refused = [r for r in records if r["answer"] == "cannot_determine"]
    correct = [r for r in answered if r["answer"] == r["gold"]]

    def med(rs):
        return statistics.median([r["overlap"] for r in rs]) if rs else float("nan")

    print(f"probes: {len(records)} | answered {len(answered)} "
          f"({len(answered)/max(1,len(records)):.1%}) | refused {len(refused)}")
    print(f"of answered, correct: {len(correct)}/{len(answered)} "
          f"= {len(correct)/max(1,len(answered)):.1%}   <-- accuracy WHEN it answers")
    print("\nprobe↔extraction overlap (median):")
    print(f"  answered : {med(answered):.3f}")
    print(f"  refused  : {med(refused):.3f}")
    print(f"  correct  : {med(correct):.3f}")

    # Refusal rate by overlap quartile — the structural signal.
    if records:
        ov = sorted(r["overlap"] for r in records)
        qs = [ov[int(len(ov) * f)] for f in (0.25, 0.5, 0.75)]
        print("\nrefusal rate by probe↔extraction overlap quartile:")
        bounds = [(-1, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 2)]
        for lo, hi in bounds:
            grp = [r for r in records if lo < r["overlap"] <= hi]
            if grp:
                rr = sum(r["answer"] == "cannot_determine" for r in grp) / len(grp)
                print(f"  overlap ({lo:.2f},{hi:.2f}]: n={len(grp):3d} "
                      f"refusal {rr:.1%}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "correspondence.json"), "w") as f:
        json.dump({"records": [{k: v for k, v in r.items() if k != "flows"}
                               for r in records]}, f, indent=2)

    lines = ["# Refusal cases — probe vs teacher extraction\n"]
    for r in sorted(refused, key=lambda r: r["overlap"])[: args.dump]:
        lines.append(f"\n## {r['gutenberg_id']}#{r['chunk_id']} "
                     f"(gold={r['gold']}, overlap={r['overlap']:.2f})")
        lines.append(f"**PROBE:** {r['probe']}")
        lines.append(f"**EXTRACTION ({r['n_flows']} flows):**")
        for fl in r["flows"][:3]:
            lines.append("  - " + "; ".join(f"{k}={v}" for k, v in fl.items()))
    with open(os.path.join(args.out, "refusal_cases.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"\n[corr] wrote {args.out}/  (correspondence.json, refusal_cases.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
