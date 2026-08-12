#!/usr/bin/env python
"""Probe alternative reranker framings against the LLM-judge grounding.

Zero-shot Qwen3-Reranker relevance correlated only ~0.19 with the 27B judge's
grounding under the production framing (query=instruction+norms-JSON,
document=flow-JSON). Before concluding distillation is required, this script
sweeps several framings on the SAME flow-bearing pairs (full norms) to see if
any orientation/representation lifts the correlation toward the ~0.6 bar.

Run with the reranker server up (RERANKER_SERVER_URL) — read-only probing.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from dagspaces.grpo_training.stages.clients import (
    RerankerJudgeClient,
    RERANKER_GROUNDING_INSTRUCTION,
)
from dagspaces.grpo_training.stages.parsing import parse_completion as _parse_completion
from scripts.validate_reranker_judge import _spearman, _pearson

TRACES = ("multirun/2026-06-17_grpo_redesign_full_v4/13-42-30/0/"
          "grpo_only_online_external/outputs/grpo/checkpoint/reward_traces.jsonl")
UNIVERSE = ("/share/pierson/matt/UAIR/multirun/2026-03-23_grpo_training/"
            "12-17-43/norm_universe_and_reward_prep/outputs/norm_universe/"
            "norm_universes.json")
NF = "NO information flows"


def render_flow_nl(flow: dict) -> str:
    """Natural-language rendering of a CI flow tuple (no appropriateness — that
    is the separate deontic axis)."""
    f = flow.get("flow", flow) if isinstance(flow.get("flow"), dict) else flow
    s = f.get("sender", "someone")
    r = f.get("recipient", "someone")
    u = f.get("subject", "a person")
    a = f.get("information_type", f.get("attribute", "information"))
    t = f.get("transmission_principle", "")
    ctx = f.get("context", "")
    out = f"{s} discloses {a} about {u} to {r}"
    if t:
        out += f" under the principle of {t}"
    if ctx:
        out += f" (context: {ctx})"
    return out


def load_pairs(limit=100000):
    nu = json.load(open(UNIVERSE))
    idx = {sid: {str(n.get("norm_articulation", ""))[:120]: n
                 for n in norms if n.get("norm_articulation")}
           for sid, norms in nu.items()}
    pairs = []
    for line in open(TRACES):
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        rf = e.get("rground_flows")
        if not rf:
            continue
        sid = str(e.get("source_id", ""))
        parsed = _parse_completion(e.get("completion", ""))
        if not parsed:
            continue
        flows = [x for x in (parsed.get("extraction") or []) if isinstance(x, dict)]
        if not flows:
            continue
        for d in rf:
            if d.get("type") != "ranked":
                continue
            norms = [idx.get(sid, {}).get(s[:120])
                     for s in (d.get("correct_norm_snippets") or [])]
            norms = [n for n in norms if n]
            t = d.get("grounding_score")
            if norms and t is not None:
                pairs.append({"norms": norms, "flows": flows, "teacher": float(t)})
        if len(pairs) >= limit:
            break
    return pairs


def main():
    url = os.environ.get("RERANKER_SERVER_URL")
    if not url:
        print("set RERANKER_SERVER_URL", file=sys.stderr)
        return 2
    client = RerankerJudgeClient(
        base_url=url,
        model_name="/share/pierson/matt/zoo/models/Qwen3-Reranker-8B",
        max_workers=16,
    )
    pairs = load_pairs()
    print(f"flow-bearing pairs: {len(pairs)}")
    teacher = [p["teacher"] for p in pairs]

    def par(fn):
        with ThreadPoolExecutor(max_workers=16) as pool:
            return list(pool.map(fn, pairs))

    framings = {}

    # A. baseline: q = instruction + norms-JSON, d = flows-JSON
    def f_baseline(p):
        q = RERANKER_GROUNDING_INSTRUCTION + "\nGoverning norms: " + json.dumps(p["norms"], ensure_ascii=False)
        s = client._rerank(q, [json.dumps(p["flows"], ensure_ascii=False)])
        return s[0] if s else 0.0
    framings["A: instr+normJSON q / flowJSON doc (baseline)"] = par(f_baseline)

    # B. NL flow doc, NL norm query (articulations)
    def f_nl(p):
        arts = " ".join(str(n.get("norm_articulation", "")) for n in p["norms"])
        q = RERANKER_GROUNDING_INSTRUCTION + "\nNorms: " + arts
        d = " ; ".join(render_flow_nl(fl) for fl in p["flows"])
        s = client._rerank(q, [d])
        return s[0] if s else 0.0
    framings["B: instr+NLnorms q / NLflow doc"] = par(f_nl)

    # C. orientation swap: q = NL flow, d = concat norm articulations
    def f_swap(p):
        q = " ; ".join(render_flow_nl(fl) for fl in p["flows"])
        d = " ".join(str(n.get("norm_articulation", "")) for n in p["norms"])
        s = client._rerank(q, [d])
        return s[0] if s else 0.0
    framings["C: NLflow q / concatNorm doc (swap)"] = par(f_swap)

    # D/E. per-norm: q = NL flow, docs = each norm articulation → max / mean
    def f_pernorm(p):
        q = " ; ".join(render_flow_nl(fl) for fl in p["flows"])
        docs = [str(n.get("norm_articulation", "")) for n in p["norms"]]
        s = client._rerank(q, docs)
        return s or [0.0]
    pernorm = par(f_pernorm)
    framings["D: NLflow q / per-norm docs MAX"] = [max(s) for s in pernorm]
    framings["E: NLflow q / per-norm docs MEAN"] = [sum(s) / len(s) for s in pernorm]

    # F. sharper grounding instruction
    SHARP = ("Judge whether the document's information flow is APPROPRIATE "
             "according to the social norms in the query: does the flow comply "
             "with what these norms require or prohibit?")
    def f_sharp(p):
        q = SHARP + "\nNorms: " + " ".join(str(n.get("norm_articulation", "")) for n in p["norms"])
        d = " ; ".join(render_flow_nl(fl) for fl in p["flows"])
        s = client._rerank(q, [d])
        return s[0] if s else 0.0
    framings["F: sharp-instr+NLnorms q / NLflow doc"] = par(f_sharp)

    client.close()
    print("\n=== Spearman / Pearson vs teacher grounding (n=%d) ===" % len(pairs))
    rows = []
    for name, scores in framings.items():
        rho = _spearman(scores, teacher)
        r = _pearson(scores, teacher)
        rows.append((rho if rho is not None else -9, name, rho, r))
    for _, name, rho, r in sorted(rows, reverse=True):
        rho_s = f"{rho:.4f}" if rho is not None else "n/a"
        r_s = f"{r:.4f}" if r is not None else "n/a"
        print(f"  rho={rho_s}  pearson={r_s}   {name}")
    print("\nViability bar ~0.6. Baseline was ~0.19.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
