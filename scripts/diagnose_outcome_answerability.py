#!/usr/bin/env python3
"""Discriminator: is R-OUTCOME's ~93% cannot_determine a probe/answerer artifact
or a genuine policy-extraction failure?

Feeds the frozen answerer the TEACHER's own reference flows — the best possible
extraction for each chunk — through the exact production call shape, and scores
their probes. Interpretation:

  * teacher extractions ALSO mostly cannot_determine  → the probes/answerer are
    structurally unanswerable (artifact; the reward cannot reward anything) →
    fixable without abandoning the design.
  * teacher extractions answer well                   → the reward works; the
    policy's extractions are genuinely uninformative → the pre-registered kill
    read stands.

Arm 2 (same chunks, softened system prompt) separates "probe unanswerable" from
"prompt-strictness jaw-lock": if the soft prompt answers where the strict one
refuses, the tooth is a jaw-lock and the fix is the answerer prompt.

Read-only w.r.t. training: uses the live judge/answerer server, writes only to
outputs/. No SLURM.
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

SOFT_SYSTEM = (
    "You answer questions using the structured information-flow extraction "
    "provided. Reason from the extraction's fields; if the extraction gives you "
    "reasonable grounds to judge, answer yes or no. Reply "
    '"cannot_determine" only when the extraction is truly silent on the question.'
)

POOLS = (
    "outputs/2026-07-23_mseries_premeasure/probe_calibration/"
    "probe_pools_filtered.parquet"
)
FLOWS = (
    "outputs/2026-07-12_fiction10_flows_gemma4/23-14-17/"
    "COLM_flows_fiction_gemma4/outputs/ci_extraction/ci_flows.parquet"
)
FIELD_MAP = {
    "ci_subject": "subject",
    "ci_sender": "sender",
    "ci_recipient": "recipient",
    "ci_information_type": "information_type",
    "ci_transmission_principle": "transmission_principle",
    "ci_context": "context",
    "ci_appropriateness": "appropriateness",
}


def teacher_flows(df: pd.DataFrame, gid: str, cid) -> list[dict]:
    sub = df[(df["gutenberg_id"].astype(str) == str(gid))
             & (df["chunk_id"].astype(str) == str(cid))]
    out = []
    for _, r in sub.iterrows():
        flow = {}
        for src, dst in FIELD_MAP.items():
            v = r.get(src)
            if v is not None and str(v) != "nan" and str(v).strip():
                flow[dst] = str(v)
        if flow:
            out.append(flow)
    return out


def _answer_with_system(client, system: str, flows, probes) -> dict:
    """Soft-arm only: same transport/parse path, overridden system prompt.

    The production client's system prompt is deliberately frozen (it is part of
    the reward definition), so the override lives here rather than in the
    client. Reuses the client's own user-turn builder, request body and parser
    so the ONLY difference vs the strict arm is the system string.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": client.build_user(flows, probes)},
    ]
    try:
        raw = client._post(client._request_body(messages))
    except Exception as exc:  # transport — count as failed, mirrors client
        return {"answers": ["cannot_determine"] * len(probes), "failed": True,
                "raw": f"<transport error: {exc}>"}
    parsed = _parse_answers(raw, len(probes))
    if parsed is None:
        return {"answers": ["cannot_determine"] * len(probes), "failed": True,
                "raw": raw}
    return {"answers": parsed, "failed": False, "raw": raw}


def run_arm(client, system, chunks, label):
    per_chunk, answers_all, ems = [], Counter(), []
    for gid, cid, flows, probes, golds in chunks:
        res = _answer_with_system(client, system, flows, probes) if system else \
            client.answer_probes(flows, probes)
        ans = res.get("answers", [])
        answers_all.update(ans)
        em = AnswererClient.em(ans, golds) if ans else 0.0
        ems.append(em)
        per_chunk.append({
            "gutenberg_id": gid, "chunk_id": str(cid), "n_flows": len(flows),
            "n_probes": len(probes), "answers": ans, "golds": golds,
            "em": em, "failed": bool(res.get("failed")),
        })
    tot = sum(answers_all.values()) or 1
    summary = {
        "arm": label,
        "n_chunks": len(per_chunk),
        "mean_em": sum(ems) / max(1, len(ems)),
        "cannot_determine_frac": answers_all["cannot_determine"] / tot,
        "yes_frac": answers_all["yes"] / tot,
        "no_frac": answers_all["no"] / tot,
        "answer_counts": dict(answers_all),
    }
    return summary, per_chunk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/2026-07-25_outcome_discriminator")
    args = ap.parse_args()

    ensure_dotenv()
    base = os.environ.get("VLLM_SERVER_URL") or os.environ.get("JUDGE_SERVER_URL")
    model = os.environ.get("JUDGE_MODEL_PATH", "")
    if not base:
        print("ERROR: VLLM_SERVER_URL/JUDGE_SERVER_URL unset", file=sys.stderr)
        return 2
    print(f"[disc] answerer={model} @ {base}")

    pools = pd.read_parquet(POOLS)
    flows_df = pd.read_parquet(FLOWS)

    grouped = {}
    for rec in pools.to_dict("records"):
        grouped.setdefault(
            (str(rec["gutenberg_id"]), str(rec["chunk_id"])), []
        ).append(rec)
    keys = sorted(grouped)
    random.Random(args.seed).shuffle(keys)

    chunks = []
    for gid, cid in keys:
        tf = teacher_flows(flows_df, gid, cid)
        if not tf:
            continue
        probes = [p["prompt_text"] for p in grouped[(gid, cid)][:4]]
        golds = [p["gold"] for p in grouped[(gid, cid)][:4]]
        chunks.append((gid, cid, tf, probes, golds))
        if len(chunks) >= args.n:
            break
    print(f"[disc] sampled {len(chunks)} chunks with teacher flows + probes")

    client = AnswererClient(base_url=base, model=model, temperature=0.0)

    results = {}
    s1, d1 = run_arm(client, None, chunks, "strict (production prompt)")
    print(json.dumps(s1, indent=2))
    results["strict"] = {"summary": s1, "detail": d1}

    try:
        s2, d2 = run_arm(client, SOFT_SYSTEM, chunks, "soft (relaxed prompt)")
        print(json.dumps(s2, indent=2))
        results["soft"] = {"summary": s2, "detail": d2}
    except TypeError:
        print("[disc] client has no system override; skipping soft arm")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "discriminator.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[disc] wrote {args.out}/discriminator.json")

    st = results["strict"]["summary"]
    print("\n=== VERDICT INPUT ===")
    print(f"teacher-extraction mean EM      : {st['mean_em']:.3f}")
    print(f"teacher-extraction cannot_det.  : {st['cannot_determine_frac']:.3f}")
    print("(policy in-run: EM 0.052, cannot_determine 0.934)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
