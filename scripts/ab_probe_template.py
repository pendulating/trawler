#!/usr/bin/env python3
"""A/B the probe template fix (2026-07-25) on identical chunks.

OLD template preferred ``condition_of_application`` over ``norm_act`` and so
dropped the act on 99.1% of norms, producing contentless questions. NEW template
always renders the act and drops the condition.

Same chunks, same golds, same teacher flows, same server — only the probe text
differs. Run under BOTH the frozen (strict) and `role` prompts: with
well-formed probes the strict prompt may suffice, and strictness is preferable
for the anti-gaming property.

Decision bar (pre-stated): teacher-quality extraction must decisively clear the
blanket-answer baseline. Before the fix it did NOT (macro 0.362 vs blanket
0.500).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "/share/pierson/matt/UAIR")
from dagspaces.common.stage_utils import ensure_dotenv  # noqa: E402
from dagspaces.grpo_training.stages.answerer_client import (  # noqa: E402
    AnswererClient, _parse_answers,
)
from dagspaces.grpo_training.stages.probes import (  # noqa: E402
    _QUESTION_SUFFIX, probe_leaks, template_probe,
)
from scripts.select_answerer_prompt import (  # noqa: E402
    CANDIDATES, FLOWS, POOLS, em_macro, em_micro, teacher_flows,
)

UNIVERSE = ("multirun/2026-07-23_universe_fiction10_gemma4/15-43-41/"
            "norm_universe_only/outputs/norm_universe/norm_universes.json")


def old_template(norm: dict) -> str:
    """The pre-fix template, reproduced for the A arm."""
    subject = norm.get("norm_subject") or "a person"
    act = norm.get("norm_act") or "share this information"
    condition = norm.get("condition_of_application") or ""
    context = norm.get("context") or ""
    parts = []
    if context:
        parts.append(f"In a setting involving {context},")
    parts.append(str(subject))
    parts.append(f"is in a situation where {condition}." if condition
                 else f"is considering whether to {act}.")
    return f"{' '.join(parts)} {_QUESTION_SUFFIX}"


def ask(client, system, flows, probes):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": client.build_user(flows, probes)}]
    try:
        raw = client._post(client._request_body(msgs))
    except Exception:
        return ["cannot_determine"] * len(probes)
    return _parse_answers(raw, len(probes)) or ["cannot_determine"] * len(probes)


def score(client, system, evalset, key):
    mic, mac, cd, n_ans, n_tot = [], [], 0, 0, 0
    cls = {"yes": [], "no": []}
    for row in evalset:
        a = ask(client, system, row["flows"], row[key])
        g = row["golds"]
        mic.append(em_micro(a, g)); mac.append(em_macro(a, g))
        for ai, gi in zip(a, g):
            n_tot += 1
            if ai == "cannot_determine":
                cd += 1
            else:
                n_ans += 1
            cls[gi].append(1.0 if ai == gi else 0.0)
    return {
        "em_micro": sum(mic) / len(mic), "em_macro": sum(mac) / len(mac),
        "gold_yes": sum(cls["yes"]) / max(1, len(cls["yes"])),
        "gold_no": sum(cls["no"]) / max(1, len(cls["no"])),
        "cannot_determine_frac": cd / max(1, n_tot),
        "answer_rate": n_ans / max(1, n_tot),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/2026-07-25_probe_template_ab")
    args = ap.parse_args()

    ensure_dotenv()
    base = os.environ.get("VLLM_SERVER_URL") or os.environ.get("JUDGE_SERVER_URL")
    model = os.environ.get("JUDGE_MODEL_PATH", "")
    client = AnswererClient(base_url=base, model=model, temperature=0.0)

    pools = pd.read_parquet(POOLS)
    flows_df = pd.read_parquet(FLOWS)
    universe = json.load(open(UNIVERSE))

    by_chunk: dict[tuple, list[dict]] = {}
    for rec in pools.to_dict("records"):
        by_chunk.setdefault(
            (str(rec["gutenberg_id"]), str(rec["chunk_id"])), []).append(rec)

    import random
    both = []
    for key, ps in by_chunk.items():
        yes = [p for p in ps if p["gold"] == "yes"]
        no = [p for p in ps if p["gold"] == "no"]
        if yes and no:
            both.append((key, yes, no))
    random.Random(args.seed).shuffle(both)

    evalset, n_leak_new = [], 0
    for (gid, cid), yes, no in both:
        tf = teacher_flows(flows_df, gid, cid)
        if not tf:
            continue
        picked = no[:2] + yes[:2]
        norms = [universe[gid][int(p["norm_index"])] for p in picked]
        new_probes = [template_probe(n) for n in norms]
        n_leak_new += sum(probe_leaks(t, n) for t, n in zip(new_probes, norms))
        evalset.append({
            "flows": tf, "golds": [p["gold"] for p in picked],
            "old": [old_template(n) for n in norms], "new": new_probes,
        })
        if len(evalset) >= args.chunks:
            break

    n_probes = sum(len(r["golds"]) for r in evalset)
    print(f"[ab] {len(evalset)} chunks / {n_probes} probes "
          f"(gold-no {sum(g=='no' for r in evalset for g in r['golds'])})")
    print(f"[ab] leak-flagged under NEW template: {n_leak_new}/{n_probes}")
    ay_mic = sum(em_micro(["yes"] * len(r["golds"]), r["golds"]) for r in evalset) / len(evalset)
    ay_mac = sum(em_macro(["yes"] * len(r["golds"]), r["golds"]) for r in evalset) / len(evalset)
    print(f"[ab] BLANKET-YES baseline (the bar to beat): "
          f"micro {ay_mic:.3f} | macro {ay_mac:.3f}\n")
    print(f"[ab] EXAMPLE old: {evalset[0]['old'][0]}")
    print(f"[ab] EXAMPLE new: {evalset[0]['new'][0]}\n")

    res = {"baseline": {"micro": ay_mic, "macro": ay_mac}, "arms": {}}
    for prompt_name in ("frozen", "role"):
        for tmpl in ("old", "new"):
            s = score(client, CANDIDATES[prompt_name], evalset, tmpl)
            res["arms"][f"{prompt_name}/{tmpl}"] = s
            flag = "  <-- beats blanket (macro)" if s["em_macro"] > ay_mac else ""
            print(f"{prompt_name:7s} {tmpl:3s}  micro {s['em_micro']:.3f} "
                  f"macro {s['em_macro']:.3f} | y {s['gold_yes']:.3f} "
                  f"n {s['gold_no']:.3f} | answer-rate {s['answer_rate']:.3f}{flag}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ab.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[ab] wrote {args.out}/ab.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
