#!/usr/bin/env python3
"""Answerer-prompt selection harness (m-series R-OUTCOME repair, 2026-07-25).

Context: the production answerer prompt makes gemma-4-31b refuse — the TEACHER's
own reference flows score EM 0.104 with 83.6% ``cannot_determine``, so the reward
has no headroom to reward good extraction (P1). Separately the training probe set
is 88.2% gold-yes, so an always-yes answerer scores EM 0.882 (P2) — which means
**raw EM cannot select a prompt**: softening the prompt buys yes-bias that looks
like signal.

Selection metric is therefore the yes-bias-immune

    discriminative gap = EM(matched teacher extraction) - EM(empty extraction)

evaluated on a CLASS-BALANCED probe subset (the natural 88/12 makes gold-no
estimates meaningless), with a mismatched-extraction arm as the anti-gaming
control and an always-yes constant as the bias baseline.

Arms per candidate prompt (identical probes throughout):
  matched     — the chunk's own teacher reference flows (headroom / upper bound)
  empty       — {"flows": []}                       (world-knowledge leakage)
  mismatched  — ANOTHER chunk's teacher flows       (extraction specificity)
  always-yes  — computed, no calls                  (bias baseline)

Reports micro EM (production today) and macro EM (mean of per-class EM — the
approved P2 fix, under which blanket-yes scores 0.5 on both-class rows).

Read-only: uses the live answerer server, writes only under outputs/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, "/share/pierson/matt/UAIR")
from dagspaces.common.stage_utils import ensure_dotenv  # noqa: E402
from dagspaces.grpo_training.stages.answerer_client import (  # noqa: E402
    ANSWERER_SYSTEM,
    AnswererClient,
    _parse_answers,
)

POOLS = ("outputs/2026-07-23_mseries_premeasure/probe_calibration/"
         "probe_pools_filtered.parquet")
FLOWS = ("outputs/2026-07-12_fiction10_flows_gemma4/23-14-17/"
         "COLM_flows_fiction_gemma4/outputs/ci_extraction/ci_flows.parquet")
FIELD_MAP = {
    "ci_subject": "subject", "ci_sender": "sender", "ci_recipient": "recipient",
    "ci_information_type": "information_type",
    "ci_transmission_principle": "transmission_principle",
    "ci_context": "context", "ci_appropriateness": "appropriateness",
}

# ── Candidate prompts ────────────────────────────────────────────────────────
CANDIDATES: dict[str, str] = {
    # 1. The frozen production prompt (control).
    "frozen": ANSWERER_SYSTEM,

    # 2. Explicit-grounds: same contract, but states that reasoning FROM the
    #    fields is expected and reserves cannot_determine for silence.
    "grounds": (
        "You answer questions using the structured information-flow extraction "
        "provided. Reason from the extraction's fields; if the extraction gives "
        "you reasonable grounds to judge, answer yes or no. Reply "
        '"cannot_determine" only when the extraction is truly silent on the '
        "question."
    ),

    # 3. Role-framed: names the judgment task (contextual integrity over a flow
    #    tuple) so the model knows what counts as sufficient grounds.
    "role": (
        "You are judging information flows using the contextual-integrity "
        "framework. You are given a structured extraction of the flows in a "
        "passage (sender, recipient, subject, information type, transmission "
        "principle, context) and questions about whether a flow should occur. "
        "Judge each question from the extraction's fields alone — the tuple is "
        "your only evidence, and it is normally sufficient to judge "
        "appropriateness. Answer yes or no. Reply \"cannot_determine\" only "
        "when the extraction contains no flow relevant to the question."
    ),

    # 4. Few-shot: one worked yes and one worked no, to fix calibration without
    #    a bias-inducing instruction change.
    "fewshot": (
        ANSWERER_SYSTEM + "\n\n"
        "Worked examples of the judgment expected:\n"
        'EXTRACTION: {"flows": [{"sender": "a physician", "recipient": "the '
        'patient\'s insurer", "information_type": "diagnosis", '
        '"transmission_principle": "without consent", "context": "medical '
        'treatment"}]}\n'
        "Q1: Should this information be shared? → no (a medical confidence "
        "passed on without consent violates the flow's norm).\n"
        'EXTRACTION: {"flows": [{"sender": "a physician", "recipient": "the '
        'treating surgeon", "information_type": "diagnosis", '
        '"transmission_principle": "for the patient\'s care", "context": '
        '"medical treatment"}]}\n'
        "Q1: Should this information be shared? → yes (transmission within the "
        "care context serves the norm).\n"
        "Judge the extraction below the same way; reserve \"cannot_determine\" "
        "for extractions with no relevant flow."
    ),
}


# ── EM variants ──────────────────────────────────────────────────────────────
def em_micro(answers: list[str], golds: list[str]) -> float:
    if not golds:
        return 0.0
    return sum(1.0 for a, g in zip(answers, golds) if a == g) / len(golds)


def em_macro(answers: list[str], golds: list[str]) -> float:
    """Mean of per-gold-class EM over the classes present (the P2 fix).

    Blanket-yes scores 0.5 on a row carrying both classes instead of ~1.0.
    Rows with a single class present are unchanged (macro == micro).
    """
    by: dict[str, list[float]] = {}
    for a, g in zip(answers, golds):
        by.setdefault(g, []).append(1.0 if a == g else 0.0)
    if not by:
        return 0.0
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


def teacher_flows(df: pd.DataFrame, gid: str, cid) -> list[dict]:
    sub = df[(df["gutenberg_id"].astype(str) == str(gid))
             & (df["chunk_id"].astype(str) == str(cid))]
    out = []
    for _, r in sub.iterrows():
        flow = {d: str(r[s]) for s, d in FIELD_MAP.items()
                if r.get(s) is not None and str(r.get(s)).strip()
                and str(r.get(s)) != "nan"}
        if flow:
            out.append(flow)
    return out


def build_balanced_eval(pools: pd.DataFrame, flows_df: pd.DataFrame,
                        n_chunks: int, seed: int):
    """Per-chunk probe lists balanced across gold classes (production call shape).

    Only chunks carrying BOTH classes can be balanced, so those are used: they
    are also the only rows where macro-EM differs from micro-EM, i.e. exactly
    where the anti-gaming property is measurable.
    """
    by_chunk: dict[tuple, list[dict]] = {}
    for rec in pools.to_dict("records"):
        by_chunk.setdefault(
            (str(rec["gutenberg_id"]), str(rec["chunk_id"])), []).append(rec)

    both = []
    for key, ps in by_chunk.items():
        yes = [p for p in ps if p["gold"] == "yes"]
        no = [p for p in ps if p["gold"] == "no"]
        if yes and no:
            both.append((key, yes, no))
    random.Random(seed).shuffle(both)

    evalset = []
    for (gid, cid), yes, no in both:
        tf = teacher_flows(flows_df, gid, cid)
        if not tf:
            continue
        picked = no[:2] + yes[:2]          # K<=4, class-balanced
        evalset.append({
            "gutenberg_id": gid, "chunk_id": cid, "flows": tf,
            "probes": [p["prompt_text"] for p in picked],
            "golds": [p["gold"] for p in picked],
        })
        if len(evalset) >= n_chunks:
            break
    return evalset


def ask(client: AnswererClient, system: str, flows: list[dict],
        probes: list[str]) -> dict:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": client.build_user(flows, probes)}]
    try:
        raw = client._post(client._request_body(messages))
    except Exception as exc:
        return {"answers": ["cannot_determine"] * len(probes), "failed": True,
                "raw": f"<transport: {exc}>"}
    parsed = _parse_answers(raw, len(probes))
    if parsed is None:
        return {"answers": ["cannot_determine"] * len(probes), "failed": True,
                "raw": raw}
    return {"answers": parsed, "failed": False, "raw": raw}


def score_arm(client, system, evalset, mode: str) -> dict:
    micro, macro, ans_counter = [], [], Counter()
    cls: dict[str, list[float]] = {"yes": [], "no": []}
    n = len(evalset)
    for i, row in enumerate(evalset):
        if mode == "matched":
            flows = row["flows"]
        elif mode == "empty":
            flows = []
        else:                                     # mismatched
            flows = evalset[(i + 1) % n]["flows"]
        res = ask(client, system, flows, row["probes"])
        a, g = res["answers"], row["golds"]
        ans_counter.update(a)
        micro.append(em_micro(a, g))
        macro.append(em_macro(a, g))
        for ai, gi in zip(a, g):
            cls[gi].append(1.0 if ai == gi else 0.0)
    tot = sum(ans_counter.values()) or 1
    return {
        "em_micro": sum(micro) / max(1, len(micro)),
        "em_macro": sum(macro) / max(1, len(macro)),
        "em_gold_yes": sum(cls["yes"]) / max(1, len(cls["yes"])),
        "em_gold_no": sum(cls["no"]) / max(1, len(cls["no"])),
        "n_gold_yes": len(cls["yes"]), "n_gold_no": len(cls["no"]),
        "cannot_determine_frac": ans_counter["cannot_determine"] / tot,
        "yes_frac": ans_counter["yes"] / tot, "no_frac": ans_counter["no"] / tot,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/2026-07-25_answerer_prompt_selection")
    ap.add_argument("--only", default="", help="comma-separated candidate subset")
    args = ap.parse_args()

    ensure_dotenv()
    base = os.environ.get("VLLM_SERVER_URL") or os.environ.get("JUDGE_SERVER_URL")
    model = os.environ.get("JUDGE_MODEL_PATH", "")
    if not base:
        print("ERROR: VLLM_SERVER_URL unset", file=sys.stderr)
        return 2

    pools = pd.read_parquet(POOLS)
    flows_df = pd.read_parquet(FLOWS)
    evalset = build_balanced_eval(pools, flows_df, args.chunks, args.seed)
    n_no = sum(g == "no" for r in evalset for g in r["golds"])
    n_yes = sum(g == "yes" for r in evalset for g in r["golds"])
    print(f"[sel] answerer={os.path.basename(model)} @ {base}")
    print(f"[sel] balanced eval: {len(evalset)} chunks | "
          f"gold-yes {n_yes} / gold-no {n_no}")

    # Always-yes baseline (analytic, no calls).
    ay_micro = sum(em_micro(["yes"] * len(r["golds"]), r["golds"]) for r in evalset) / len(evalset)
    ay_macro = sum(em_macro(["yes"] * len(r["golds"]), r["golds"]) for r in evalset) / len(evalset)
    print(f"[sel] ALWAYS-YES baseline: micro {ay_micro:.3f} | macro {ay_macro:.3f}\n")

    client = AnswererClient(base_url=base, model=model, temperature=0.0)
    names = [c for c in (args.only.split(",") if args.only else CANDIDATES)
             if c in CANDIDATES]

    results = {"baseline_always_yes": {"em_micro": ay_micro, "em_macro": ay_macro},
               "eval": {"n_chunks": len(evalset), "n_gold_yes": n_yes,
                        "n_gold_no": n_no, "seed": args.seed},
               "candidates": {}}

    for name in names:
        sysmsg = CANDIDATES[name]
        arms = {m: score_arm(client, sysmsg, evalset, m)
                for m in ("matched", "empty", "mismatched")}
        gap_micro = arms["matched"]["em_micro"] - arms["empty"]["em_micro"]
        gap_macro = arms["matched"]["em_macro"] - arms["empty"]["em_macro"]
        spec = arms["matched"]["em_micro"] - arms["mismatched"]["em_micro"]
        results["candidates"][name] = {
            "arms": arms, "gap_micro": gap_micro, "gap_macro": gap_macro,
            "specificity": spec, "prompt": sysmsg,
        }
        m = arms["matched"]
        print(f"=== {name}")
        print(f"    matched   micro {m['em_micro']:.3f} macro {m['em_macro']:.3f} "
              f"| gold-yes {m['em_gold_yes']:.3f} gold-no {m['em_gold_no']:.3f} "
              f"| cd {m['cannot_determine_frac']:.3f}")
        print(f"    empty     micro {arms['empty']['em_micro']:.3f} "
              f"| mismatched micro {arms['mismatched']['em_micro']:.3f}")
        print(f"    GAP micro {gap_micro:+.3f}  GAP macro {gap_macro:+.3f}  "
              f"specificity {spec:+.3f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "selection.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[sel] wrote {args.out}/selection.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
