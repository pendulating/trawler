#!/usr/bin/env python3
"""v2 acceptance bar — validate flow-anchored probes BEFORE rebuilding 11k.

Bar (reward-outcome-v2-proposal.md), all on polarity-corrected gold:
  1. teacher-perfect extraction macro-EM  >  blanket-answer floor   [THE GATE]
  2. empty-extraction EM ~ 0            (no world-knowledge leakage)
  3. mismatched-extraction EM << matched (extraction specificity)
  4. cannot_determine on teacher-perfect < 0.2

v1 never cleared (1): on corrected gold its ceiling was macro 0.356 against a
0.500 floor (deficit -0.144). If v2 does not clear it either, the redesign is
wrong and nothing gets rebuilt or relaunched.

Arms are scored on the answerer's response to each extraction variant, using the
production call shape (appropriateness withheld, norms_invoked shown).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/share/pierson/matt/UAIR")
from dagspaces.common.stage_utils import ensure_dotenv  # noqa: E402
from dagspaces.grpo_training.stages.answerer_client import (  # noqa: E402
    ANSWERER_SYSTEM, AnswererClient, _parse_answers,
)
from dagspaces.grpo_training.stages.clients import EmbeddingClient  # noqa: E402
from dagspaces.grpo_training.stages.probes import (  # noqa: E402
    build_flow_probe_pool, sample_flow_probes,
)
from scripts.select_answerer_prompt import (  # noqa: E402
    CANDIDATES, FLOWS, em_macro, em_micro,
)

UNIV_DIR = ("multirun/2026-07-23_universe_fiction10_gemma4/15-43-41/"
            "norm_universe_only/outputs/norm_universe")
POLARITY = "outputs/2026-07-25_act_polarity_backfill/act_polarity.json"
FIELD_MAP = {
    "ci_subject": "subject", "ci_sender": "sender", "ci_recipient": "recipient",
    "ci_information_type": "information_type",
    "ci_transmission_principle": "transmission_principle", "ci_context": "context",
    "ci_norms_invoked": "norms_invoked",
}


def eligible_idx(norms):
    out = []
    for i, n in enumerate(norms):
        if (n.get("governs_info_flow") is True
                and str(n.get("normative_force") or "").strip().lower()
                in ("obligatory", "recommended", "prohibited", "discouraged")
                and str(n.get("context") or "").strip()):
            out.append(i)
    return out


def ask(client, system, flows, probes):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": client.build_user(flows, probes)}]
    try:
        raw = client._post(client._request_body(msgs))
    except Exception:
        return ["cannot_determine"] * len(probes)
    return _parse_answers(raw, len(probes)) or ["cannot_determine"] * len(probes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="frozen,role")
    ap.add_argument("--out", default="outputs/2026-07-25_v2_validation")
    args = ap.parse_args()

    ensure_dotenv()
    aurl = os.environ.get("VLLM_SERVER_URL") or os.environ.get("JUDGE_SERVER_URL")
    eurl = os.environ.get("EMBEDDING_SERVER_URL")
    amodel = os.environ.get("JUDGE_MODEL_PATH", "")
    import requests
    emodel = requests.get(f"{eurl.rstrip('/')}/v1/models", timeout=15).json()["data"][0]["id"]
    aclient = AnswererClient(base_url=aurl, model=amodel, temperature=0.0)
    eclient = EmbeddingClient(base_url=eurl, model_name=emodel)

    universe = json.load(open(f"{UNIV_DIR}/norm_universes.json"))
    polarity = json.load(open(POLARITY))
    flows_df = pd.read_parquet(FLOWS)

    # chunks with >=2 teacher flows, sampled deterministically
    grp = flows_df.groupby(["gutenberg_id", "chunk_id"]).size()
    keys = [k for k, v in grp.items() if v >= 2]
    random.Random(args.seed).shuffle(keys)

    rows = []
    for gid, cid in keys:
        gid = str(gid)
        norms = universe.get(gid) or []
        idx = eligible_idx(norms)
        if not idx:
            continue
        emb = np.load(f"{UNIV_DIR}/embeddings/{gid}.npy")[idx]
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)

        sub = flows_df[(flows_df.gutenberg_id.astype(str) == gid)
                       & (flows_df.chunk_id == cid)]
        tf = []
        for _, r in sub.iterrows():
            f = {d: str(r[s]) for s, d in FIELD_MAP.items()
                 if r.get(s) is not None and str(r.get(s)).strip() and str(r.get(s)) != "nan"}
            if f:
                tf.append(f)
        if not tf:
            continue

        def retrieve(query, k, _emb=emb, _idx=idx):
            q = np.asarray(eclient.encode_batch([query]), dtype=np.float32)[0]
            q = q / max(1e-9, float(np.linalg.norm(q)))
            return [_idx[j] for j in np.argsort(-(_emb @ q))[:k]]

        pol = {str(i): polarity.get(gid, {}).get(str(i)) for i in idx}
        pool, _ = build_flow_probe_pool(tf, norms, retrieve, polarity_lookup=pol)
        sampled = sample_flow_probes(pool, str(cid), k_max=4)
        if not sampled:
            continue
        rows.append({"gid": gid, "cid": str(cid), "flows": tf,
                     "probes": [p["prompt_text"] for p in sampled],
                     "golds": [p["gold"] for p in sampled],
                     # ORACLE arm: the book's ACTUAL governing norms, replacing
                     # the teacher's zero-shot norms_invoked. Tests whether a
                     # policy that HAS learned the norms can score — i.e.
                     # whether the reward is learnable at all.
                     "oracle_norms": [str(p["norm"].get("norm_articulation") or "")
                                      for p in sampled]})
        if len(rows) >= args.chunks:
            break

    n_p = sum(len(r["probes"]) for r in rows)
    n_no = sum(g == "no" for r in rows for g in r["golds"])
    print(f"[v2] {len(rows)} chunks / {n_p} probes | gold-no {n_no} ({n_no/n_p:.1%})")
    print(f"[v2] EXAMPLE probe: {rows[0]['probes'][0]}\n")

    results = {}
    for pname in args.prompt.split(","):
        system = CANDIDATES[pname]
        arms = {}
        for mode in ("matched", "oracle", "empty", "mismatched"):
            mic, mac, cd, tot = [], [], 0, 0
            cls = {"yes": [], "no": []}
            for i, r in enumerate(rows):
                if mode == "matched":
                    flows = r["flows"]
                elif mode == "oracle":
                    flows = [dict(f, norms_invoked=r["oracle_norms"]) for f in r["flows"]]
                elif mode == "empty":
                    flows = []
                else:
                    flows = rows[(i + 1) % len(rows)]["flows"]
                a = ask(aclient, system, flows, r["probes"])
                mic.append(em_micro(a, r["golds"])); mac.append(em_macro(a, r["golds"]))
                for ai, gi in zip(a, r["golds"]):
                    tot += 1; cd += ai == "cannot_determine"
                    cls[gi].append(1.0 if ai == gi else 0.0)
            arms[mode] = {
                "em_micro": sum(mic)/len(mic), "em_macro": sum(mac)/len(mac),
                "gold_yes": sum(cls["yes"])/max(1, len(cls["yes"])),
                "gold_no": sum(cls["no"])/max(1, len(cls["no"])),
                "cannot_determine_frac": cd/max(1, tot),
            }
        bl_mic = sum(em_micro(["yes"]*len(r["golds"]), r["golds"]) for r in rows)/len(rows)
        bl_mac = sum(em_macro(["yes"]*len(r["golds"]), r["golds"]) for r in rows)/len(rows)
        m = arms["matched"]
        gate = m["em_macro"] > bl_mac
        results[pname] = {"arms": arms, "blanket_micro": bl_mic,
                          "blanket_macro": bl_mac, "gate_passed": bool(gate)}
        print(f"=== {pname}")
        print(f"  matched     micro {m['em_micro']:.3f} macro {m['em_macro']:.3f} "
              f"| y {m['gold_yes']:.3f} n {m['gold_no']:.3f} | cd {m['cannot_determine_frac']:.3f}")
        o = arms["oracle"]
        print(f"  ORACLE      micro {o['em_micro']:.3f} macro {o['em_macro']:.3f} "
              f"| y {o['gold_yes']:.3f} n {o['gold_no']:.3f} | cd {o['cannot_determine_frac']:.3f}"
              f"   {'*** ORACLE CLEARS FLOOR ***' if o['em_macro'] > bl_mac else ''}")
        print(f"  empty       micro {arms['empty']['em_micro']:.3f} "
              f"| mismatched micro {arms['mismatched']['em_micro']:.3f}")
        print(f"  blanket floor macro {bl_mac:.3f} -> deficit {m['em_macro']-bl_mac:+.3f}"
              f"   {'*** GATE PASSED ***' if gate else 'gate FAILED'}")
        print(f"  (2) empty~0 {'OK' if arms['empty']['em_micro'] < 0.1 else 'FAIL'} | "
              f"(3) specificity {m['em_micro']-arms['mismatched']['em_micro']:+.3f} | "
              f"(4) cd<0.2 {'OK' if m['cannot_determine_frac'] < 0.2 else 'FAIL'}\n")

    os.makedirs(args.out, exist_ok=True)
    json.dump(results, open(os.path.join(args.out, "validation.json"), "w"), indent=2)
    print(f"[v2] wrote {args.out}/validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
