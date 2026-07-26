#!/usr/bin/env python3
"""Teacher-ceiling re-measurement on polarity-corrected gold (2026-07-25).

The finding that stopped the m1 relaunch — "the honest ceiling sits below the
gaming floor" — was measured before the act-polarity bug was known, i.e. on gold
that was inverted for 19.0% of norms, against a blanket-answer floor computed
from a skew (88/12) that the bug had itself inflated.

This decomposes the failure. The answerer runs ONCE per arm on the TEACHER's own
reference flows (the best extraction that can exist); the identical responses are
then scored against BOTH gold labelings:

    old gold : FORCE_TO_GOLD[force]                      (pre-fix)
    new gold : flow_appropriateness(force, act_polarity)  (polarity-corrected)

Same probes, same extractions, same model outputs — only the labels differ. So
any change is attributable to gold correction alone, and whatever gap remains is
attributable to probe design (the contentless-template + no-correspondence
defects that v2 addresses).

Probes here are still the ORIGINAL pool probes; this is deliberate. v2's
flow-anchored probes are not built yet, and the question this answers is how
much of the ceiling problem the gold fix alone removes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, "/share/pierson/matt/UAIR")
from dagspaces.common.stage_utils import ensure_dotenv  # noqa: E402
from dagspaces.grpo_training.stages.answerer_client import (  # noqa: E402
    AnswererClient, _parse_answers,
)
from dagspaces.grpo_training.stages.deontic import (  # noqa: E402
    flow_appropriateness,
)
from scripts.select_answerer_prompt import (  # noqa: E402
    CANDIDATES, FLOWS, POOLS, em_macro, em_micro, teacher_flows,
)

UNIV = ("multirun/2026-07-23_universe_fiction10_gemma4/15-43-41/"
        "norm_universe_only/outputs/norm_universe/norm_universes.json")
POLARITY = "outputs/2026-07-25_act_polarity_backfill/act_polarity.json"


def corrected_gold(universe, polarity, gid: str, norm_index: int) -> str | None:
    norm = universe[gid][int(norm_index)]
    force = str(norm.get("normative_force") or "").strip().lower()
    pol = polarity.get(gid, {}).get(str(int(norm_index)), "performing")
    app = flow_appropriateness(force, pol)
    if app is None:
        return None
    return "yes" if app == "appropriate" else "no"


def ask(client, system, flows, probes):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": client.build_user(flows, probes)}]
    try:
        raw = client._post(client._request_body(msgs))
    except Exception:
        return ["cannot_determine"] * len(probes)
    return _parse_answers(raw, len(probes)) or ["cannot_determine"] * len(probes)


def score(rows, answers_by_row, gold_key):
    mic, mac = [], []
    cls = {"yes": [], "no": []}
    cd = tot = 0
    for r, ans in zip(rows, answers_by_row):
        g = r[gold_key]
        mic.append(em_micro(ans, g))
        mac.append(em_macro(ans, g))
        for a, gi in zip(ans, g):
            tot += 1
            cd += a == "cannot_determine"
            cls[gi].append(1.0 if a == gi else 0.0)
    blanket_mic = sum(em_micro(["yes"] * len(r[gold_key]), r[gold_key]) for r in rows) / len(rows)
    blanket_mac = sum(em_macro(["yes"] * len(r[gold_key]), r[gold_key]) for r in rows) / len(rows)
    return {
        "em_micro": sum(mic) / len(mic), "em_macro": sum(mac) / len(mac),
        "gold_yes": sum(cls["yes"]) / max(1, len(cls["yes"])),
        "gold_no": sum(cls["no"]) / max(1, len(cls["no"])),
        "n_gold_yes": len(cls["yes"]), "n_gold_no": len(cls["no"]),
        "cannot_determine_frac": cd / max(1, tot),
        "blanket_micro": blanket_mic, "blanket_macro": blanket_mac,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/2026-07-25_teacher_ceiling_corrected")
    args = ap.parse_args()

    ensure_dotenv()
    url = os.environ.get("VLLM_SERVER_URL") or os.environ.get("JUDGE_SERVER_URL")
    model = os.environ.get("JUDGE_MODEL_PATH", "")
    client = AnswererClient(base_url=url, model=model, temperature=0.0)

    universe = json.load(open(UNIV))
    polarity = json.load(open(POLARITY))
    pools = pd.read_parquet(POOLS)
    flows_df = pd.read_parquet(FLOWS)

    by_chunk: dict[tuple, list[dict]] = {}
    for rec in pools.to_dict("records"):
        by_chunk.setdefault(
            (str(rec["gutenberg_id"]), str(rec["chunk_id"])), []).append(rec)

    # Balance on CORRECTED gold — the population that matters going forward.
    cand = []
    for (gid, cid), ps in by_chunk.items():
        for p in ps:
            p["gold_new"] = corrected_gold(universe, polarity, gid, p["norm_index"])
        ok = [p for p in ps if p["gold_new"] is not None]
        yes = [p for p in ok if p["gold_new"] == "yes"]
        no = [p for p in ok if p["gold_new"] == "no"]
        if yes and no:
            cand.append(((gid, cid), yes, no))
    random.Random(args.seed).shuffle(cand)

    rows = []
    for (gid, cid), yes, no in cand:
        tf = teacher_flows(flows_df, gid, cid)
        if not tf:
            continue
        picked = no[:2] + yes[:2]
        rows.append({
            "flows": tf,
            "probes": [p["prompt_text"] for p in picked],
            "gold_old": [p["gold"] for p in picked],
            "gold_new": [p["gold_new"] for p in picked],
        })
        if len(rows) >= args.chunks:
            break

    n_probes = sum(len(r["probes"]) for r in rows)
    flipped = sum(a != b for r in rows for a, b in zip(r["gold_old"], r["gold_new"]))
    print(f"[ceiling] {len(rows)} chunks / {n_probes} probes on TEACHER flows")
    print(f"[ceiling] labels flipped by the polarity fix: {flipped}/{n_probes} "
          f"({flipped/n_probes:.1%})\n")

    results = {}
    for pname in ("frozen", "role"):
        answers = [ask(client, CANDIDATES[pname], r["flows"], r["probes"]) for r in rows]
        for gold_key, label in (("gold_old", "OLD gold"), ("gold_new", "CORRECTED gold")):
            s = score(rows, answers, gold_key)
            results[f"{pname}/{gold_key}"] = s
            beat = "  <-- BEATS blanket (macro)" if s["em_macro"] > s["blanket_macro"] else ""
            print(f"{pname:7s} {label:15s} micro {s['em_micro']:.3f} "
                  f"macro {s['em_macro']:.3f} | y {s['gold_yes']:.3f} n {s['gold_no']:.3f} "
                  f"| cd {s['cannot_determine_frac']:.3f}")
            print(f"{'':7s} {'':15s} blanket-yes floor: micro {s['blanket_micro']:.3f} "
                  f"macro {s['blanket_macro']:.3f}{beat}")
        print()

    os.makedirs(args.out, exist_ok=True)
    json.dump(results, open(os.path.join(args.out, "ceiling.json"), "w"), indent=2)
    print(f"[ceiling] wrote {args.out}/ceiling.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
