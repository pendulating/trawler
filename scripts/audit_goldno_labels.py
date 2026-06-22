#!/usr/bin/env python3
"""Audit gold=NO label quality before adding a false-extraction penalty.

The penalty fires on (model extracts a flow) AND (gold_has_exchange=False).
Risk: if gold=False chunks actually contain real flows the norm-extraction
pipeline missed, the penalty would punish CORRECT extraction. This script:
  1. Clarifies what has_information_exchange means (gold reasoning on True/False).
  2. Matches v5 gold-NO *extraction* completions back to the gold chunk, and
     prints gold-reasoning vs model-extraction for the highest-grounding cases
     (the ones the penalty would bite hardest) so they can be classified by hand.
"""
from __future__ import annotations

import glob
import json
import statistics as st

import pandas as pd

GOLD = "/share/pierson/matt/n2s4cir/data/fiction10/ci_reasoning.parquet"


def fp(text):
    return (text or "")[40:240]


def main():
    df = pd.read_parquet(GOLD)
    df["gid"] = df["gutenberg_id"].astype(str)

    print("=" * 90)
    print("PART 1 — what does has_information_exchange mean? (gold reasoning samples)")
    print("=" * 90)
    for label in [True, False]:
        sub = df[df["has_information_exchange"] == label]
        print(f"\n--- gold={label}  (n={len(sub)}, mean ci_flow_count={sub['ci_flow_count'].mean():.2f}) ---")
        for _, r in sub.head(3).iterrows():
            print(f"  [{r['gid']}#{r['chunk_id']}] flows={r['ci_flow_count']}: "
                  f"{str(r['ci_reasoning_text'])[:260]}")

    # ---- Load v5 gold-NO extraction traces ----
    p = glob.glob("multirun/*grpo_redesign_full_v5*/**/reward_traces.jsonl", recursive=True)[0]
    rows = [json.loads(l) for l in open(p)]
    ci = [r for r in rows if r.get("task_type") == "ci_extraction"]
    goldno_ext = [
        r for r in ci
        if r.get("gold_has_exchange") is False and not r.get("is_no_flow")
    ]
    print()
    print("=" * 90)
    print(f"PART 2 — v5 gold-NO EXTRACTIONS: {len(goldno_ext)} completions "
          f"(these are what the penalty would hit)")
    print("=" * 90)

    # grounding distribution + model confidence/appropriateness
    grnd = []
    confs = []
    appr = {"appropriate": 0, "inappropriate": 0, "ambiguous": 0, "other": 0}
    for r in goldno_ext:
        d = (r.get("rground_flows") or [{}])[0]
        g = d.get("grounding_score")
        if g is not None:
            grnd.append(float(g))
        try:
            obj = json.loads(r["completion"])
            for fl in obj.get("flows", []):
                if isinstance(fl, dict):
                    c = fl.get("confidence")
                    if c is not None:
                        confs.append(float(c))
                    a = str(fl.get("appropriateness", "")).lower()
                    appr[a if a in appr else "other"] += 1
        except Exception:
            pass
    if grnd:
        qs = sorted(grnd)
        print(f"  correct grounding_score: mean={st.mean(grnd):.3f}  "
              f"p25={qs[len(qs)//4]:.2f} p50={qs[len(qs)//2]:.2f} p75={qs[3*len(qs)//4]:.2f}  "
              f"frac>0.6={sum(1 for x in grnd if x>0.6)/len(grnd):.2f}")
    if confs:
        print(f"  model self-confidence on these flows: mean={st.mean(confs):.2f}/10")
    print(f"  appropriateness labels: {appr}")
    print("  (high grounding + high confidence + 'appropriate' => more likely a REAL flow gold missed)")

    # ---- Match highest-grounding gold-NO extractions back to gold chunk ----
    print()
    print("=" * 90)
    print("PART 3 — highest-grounding gold-NO extractions: gold-reasoning vs model (hand-classify)")
    print("=" * 90)
    by_gid = {}
    for gid, sub in df[df["has_information_exchange"] == False].groupby("gid"):
        by_gid[gid] = sub[["chunk_id", "article_text", "ci_reasoning_text"]].to_dict("records")

    def match(trace):
        sid = str(trace.get("source_id", ""))
        prompt = trace.get("prompt", "") or ""
        for rec in by_gid.get(sid, []):
            if fp(rec["article_text"]) and fp(rec["article_text"]) in prompt:
                return rec
        return None

    ranked = sorted(
        goldno_ext,
        key=lambda r: (r.get("rground_flows") or [{}])[0].get("grounding_score") or 0,
        reverse=True,
    )
    shown = 0
    for r in ranked:
        if shown >= 8:
            break
        rec = match(r)
        if rec is None:
            continue
        try:
            obj = json.loads(r["completion"])
        except Exception:
            continue
        flows = obj.get("flows", [])
        if not flows:
            continue
        d = (r.get("rground_flows") or [{}])[0]
        shown += 1
        print(f"\n### Case {shown}  [{r['source_id']}#{rec['chunk_id']}]  "
              f"correct_grounding={d.get('grounding_score')}  wrong={d.get('wrong_grounding')}")
        print(f"  GOLD says NO-FLOW because: {str(rec['ci_reasoning_text'])[:300]}")
        print(f"  MODEL reasoning: {str(obj.get('reasoning',''))[:240]}")
        for fl in flows[:2]:
            if isinstance(fl, dict):
                print(f"  MODEL flow: sender={fl.get('sender')!r} -> recipient={fl.get('recipient')!r} | "
                      f"info={fl.get('information_type')!r} | tp={fl.get('transmission_principle')!r} | "
                      f"appr={fl.get('appropriateness')!r} conf={fl.get('confidence')}")
        print(f"  CHUNK excerpt: {str(rec['article_text'])[200:560]}")
    print(f"\n(matched+shown {shown} cases)")


if __name__ == "__main__":
    main()
