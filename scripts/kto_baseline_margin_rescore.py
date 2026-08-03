#!/usr/bin/env python3
"""Wiki §17 confirmatory test: is the appropriateness gold salvageable on the
high-confidence retrieval subset?

The K3 probe scored the baseline policy at Youden J = -0.0344 (below chance).
The 2026-08-03 audit showed the gold agrees with the teacher's own
``ci_appropriateness`` at kappa 0.053 overall, but kappa 0.230 / J 0.299 where
the top1-top2 retrieval margin >= 0.10. If the gold is merely *noisy* rather
than *wrong*, the policy's measured discrimination should climb on that subset.

This regenerates the BASELINE SLICE ONLY (1 of the probe's 45) with identical
generation settings, and retains per-flow retrieval margin so J can be
recomputed under margin filters.

FAITHFULNESS GUARD: the unfiltered J recomputed here must reproduce the probe's
-0.0344. If it does not, this script is measuring something else and its
filtered numbers mean nothing — it raises rather than reporting.

Usage:
    python scripts/kto_baseline_margin_rescore.py --out outputs/2026-08-03_margin_rescore
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/share/pierson/matt/UAIR")
sys.path.insert(0, str(ROOT))

from dagspaces.common.stage_utils import ensure_dotenv  # noqa: E402

K1_DIR = ROOT / "outputs/2026-07-31_k1_full"
EMB_MODEL = "/share/pierson/matt/zoo/models/Qwen3-Embedding-8B"
N_SAMPLES = 8        # identical to the probe
GEN_SEED = 7         # identical to the probe
TAU = 0.55
MIN_EDIT_SIM = 0.55
PROBE_BASELINE_J = -0.034379   # the number this must reproduce
J_TOLERANCE = 0.004            # generation is seeded; allow only server jitter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-screen", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_dotenv()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from dagspaces.grpo_training.stages.aux_scorers import (
        DirectChunkGold,
        _build_retrieval,
        _flatten_flow,
        _flow_to_query,
        majority_gold,
        make_direct_chunk_gold,
    )
    from dagspaces.grpo_training.stages.clients import EmbeddingClient
    from dagspaces.grpo_training.stages.kto_probe import select_probe_chunks
    from dagspaces.grpo_training.stages.modular_reward import (
        match_flows,
        valid_gate,
    )

    # ---- probe subset (identical selection to the K3 probe) --------------
    metadata = json.load(open(K1_DIR / "kto_metadata.json"))
    pop = pd.read_parquet(K1_DIR / "population.parquet")
    pop["chunk_key"] = pop["k0"].astype(str) + "|" + pop["k1"].astype(str)
    gold_yes = dict(zip(pop["chunk_key"], pop["gold_yes"]))
    prompts = dict(zip(pop["chunk_key"], pop["prompt"]))
    heldout = [k for k in metadata["heldout_keys"] if k in prompts]
    probe_keys = select_probe_chunks(
        heldout, gold_yes, "screen", args.n_screen, args.seed)

    ref_keys = json.load(
        open(ROOT / "outputs/2026-08-02_k3_probe/merged/probe_meta.json")
    )["probe_keys"]
    if list(probe_keys) != list(ref_keys):
        raise RuntimeError(
            "probe subset differs from the K3 probe — not comparable")
    print(f"[rescore] probe subset matches K3 exactly ({len(probe_keys)} chunks)")

    universes = json.load(open(os.environ["NORM_UNIVERSES_PATH"]))
    emb_client = EmbeddingClient(
        base_url=os.environ["EMBEDDING_SERVER_URL"], model_name=EMB_MODEL)
    yes_keys = {tuple(k.split("|", 1)) for k in probe_keys if gold_yes[k]}

    # ---- ONE index, carrying margin + the teacher's own label ------------
    # Single retrieval pass is the authority for golds, embeddings AND margin,
    # so there is no cross-pass alignment to get wrong (a two-index version of
    # this script diverged from production for exactly that reason). The build
    # loop mirrors aux_scorers.make_direct_chunk_gold; production is rebuilt
    # below purely as a cross-check, and the end-to-end guard is the J
    # reproduction at the bottom.
    _, retriever = _build_retrieval(
        None, {"embeddings_dir": os.environ.get("NORM_EMBEDDINGS_PATH", "")},
        universes, emb_client, None,
        norm_filter=lambda n: n.get("governs_info_flow") is True)

    df = pd.read_parquet(os.environ["CI_EXTRACTION_PATH"])
    sid = "gutenberg_id" if "gutenberg_id" in df.columns else "source_id"
    df["_key"] = list(zip(df[sid].astype(str), df["chunk_id"].astype(str)))
    df = df[df["_key"].isin(yes_keys)].reset_index(drop=True)

    field_cols = {
        "ci_sender": "sender", "ci_recipient": "recipient",
        "ci_subject": "subject", "ci_information_type": "information_type",
        "ci_transmission_principle": "transmission_principle",
        "ci_context": "context",
    }
    tflows = [{v: (r.get(c) if pd.notna(r.get(c)) else None)
               for c, v in field_cols.items()} for _, r in df.iterrows()]
    tvecs = np.asarray(
        emb_client.encode_batch(
            [_flow_to_query(_flatten_flow(f)) for f in tflows]),
        dtype=np.float32)
    if int((~tvecs.any(axis=1)).sum()):
        raise RuntimeError("zero teacher embeddings — embedding-server fault")
    tvecs /= np.maximum(np.linalg.norm(tvecs, axis=1, keepdims=True), 1e-9)

    side: dict = {}
    for i, (key, vec) in enumerate(zip(df["_key"], tvecs)):
        raw, sims = retriever.retrieve(vec, key[0], return_scores=True, top_k=2)
        norms = json.loads(raw) if isinstance(raw, str) else (raw or [])
        gold = majority_gold(norms, k=1)
        if gold is None:
            continue  # same drop rule as production
        e = side.setdefault(key, {"golds": [], "emb": [], "texts": [],
                                  "margin": [], "top_sim": [],
                                  "teacher_appr": []})
        e["golds"].append(gold)
        e["emb"].append(vec)
        e["texts"].append(_flow_to_query(_flatten_flow(tflows[i])))
        e["top_sim"].append(float(sims[0]) if len(sims) else float("nan"))
        e["margin"].append(
            float(sims[0] - sims[1]) if len(sims) > 1 else float("nan"))
        e["teacher_appr"].append(df["ci_appropriateness"].iloc[i])

    index = {k: {**v, "emb": np.stack(v["emb"])} for k, v in side.items()}
    chunk_gold = DirectChunkGold(index, emb_client.encode_batch)
    print(f"[rescore] index: {len(index)} chunks, "
          f"{sum(len(v['golds']) for v in index.values())} teacher flows")

    # Cross-check against the production builder. Soft: a divergence does not
    # by itself invalidate the run (the J guard below is the real gate), but it
    # must be visible if it happens.
    prod_gold = make_direct_chunk_gold(
        None, {"embeddings_dir": os.environ.get("NORM_EMBEDDINGS_PATH", "")},
        universes, yes_keys, keep_norm_info=True, embedding_client=emb_client)
    diverged = [k for k, v in index.items()
                if list((prod_gold.get(*k) or {}).get("golds", []))
                != list(v["golds"])]
    if diverged:
        print(f"[rescore] WARNING: {len(diverged)}/{len(index)} chunks differ "
              f"from the production builder, e.g. {diverged[:5]}")
    else:
        print(f"[rescore] cross-check: golds identical to production on all "
              f"{len(index)} chunks")

    # ---- generate the baseline slice -------------------------------------
    from vllm import LLM, SamplingParams

    llm = LLM(model=metadata["generation"]["merged_sft"], dtype="bfloat16",
              max_model_len=8192, gpu_memory_utilization=0.85,
              disable_custom_all_reduce=True, max_num_batched_tokens=2048)
    sp = SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=1.0,
                        max_tokens=3072, seed=GEN_SEED)
    outs = llm.generate([prompts[k] for k in probe_keys], sp)

    # ---- score, retaining per-flow retrieval confidence -------------------
    rows = []
    for key, o in zip(probe_keys, outs):
        if not gold_yes[key]:
            continue
        entry = chunk_gold.get(*key.split("|", 1))
        if entry is None:
            continue
        e = side[tuple(key.split("|", 1))]
        for ci, comp in enumerate(o.outputs):
            g = valid_gate(comp.text)
            if not g.passed or not g.flows:
                continue
            p_emb = np.asarray(chunk_gold.embed_flows(g.flows))
            if p_emb.ndim != 2 or not p_emb.any(axis=1).all():
                continue
            for t, p, sim in match_flows(entry["emb"] @ p_emb.T, TAU):
                label = str(
                    g.flows[p].get("appropriateness") or "").strip().lower()
                rows.append({
                    "chunk_key": key, "completion": ci,
                    "gold": entry["golds"][t],
                    "policy": label,
                    "correct": label == entry["golds"][t],
                    "match_sim": float(sim),
                    "margin": e["margin"][t],
                    "top_sim": e["top_sim"][t],
                    "teacher_appr": e["teacher_appr"][t],
                })

    res = pd.DataFrame(rows)
    res.to_parquet(out_dir / "baseline_per_flow.parquet")
    print(f"[rescore] {len(res)} matched per-flow judgments")

    # ---- faithfulness guard: reproduce the probe's baseline J -------------
    def youden(d: pd.DataFrame) -> float:
        viol = d[d.gold == "inappropriate"]
        appr = d[d.gold == "appropriate"]
        if not len(viol) or not len(appr):
            return float("nan")
        return float(viol.correct.mean() + appr.correct.mean() - 1)

    j_all = youden(res)
    print(f"\n[guard] recomputed baseline J = {j_all:+.4f} "
          f"(probe reported {PROBE_BASELINE_J:+.4f})")
    if not abs(j_all - PROBE_BASELINE_J) <= J_TOLERANCE:
        raise RuntimeError(
            f"FAITHFULNESS GUARD FAILED: recomputed J {j_all:+.4f} does not "
            f"reproduce the probe's {PROBE_BASELINE_J:+.4f} within "
            f"{J_TOLERANCE}. This re-score is measuring something else; its "
            "margin-filtered numbers are meaningless. Do not report them.")
    print("[guard] PASSED — filtered numbers below are trustworthy\n")

    # ---- the actual question ---------------------------------------------
    def boot(d: pd.DataFrame, n: int = 4000, seed: int = 0) -> tuple:
        rng = np.random.default_rng(seed)
        keys = d.chunk_key.unique()
        idx = {k: g for k, g in d.groupby("chunk_key")}
        vals = []
        for _ in range(n):
            samp = pd.concat([idx[k] for k in rng.choice(keys, len(keys))])
            v = youden(samp)
            if not np.isnan(v):
                vals.append(v)
        return (float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5)))

    print("=== baseline discrimination vs retrieval confidence ===")
    out_rows = []
    for name, sub in [
        ("all", res),
        ("margin >= 0.05", res[res.margin >= 0.05]),
        ("margin >= 0.08", res[res.margin >= 0.08]),
        ("margin >= 0.10", res[res.margin >= 0.10]),
        ("top_sim >= 0.65", res[res.top_sim >= 0.65]),
        ("top_sim >= 0.70", res[res.top_sim >= 0.70]),
    ]:
        if len(sub) < 50:
            print(f"{name:18s} n={len(sub):5d}  (too few — skipped)")
            continue
        j = youden(sub)
        lo, hi = boot(sub)
        flag = "  <-- above chance" if lo > 0 else ""
        print(f"{name:18s} n={len(sub):5d}  J={j:+.4f}  "
              f"CI[{lo:+.4f},{hi:+.4f}]{flag}")
        out_rows.append({"subset": name, "n": len(sub), "J": j,
                         "ci_lo": lo, "ci_hi": hi})

    pd.DataFrame(out_rows).to_csv(out_dir / "margin_summary.csv", index=False)
    print(f"\n[rescore] wrote {out_dir}/margin_summary.csv")


if __name__ == "__main__":
    main()
