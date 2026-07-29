#!/usr/bin/env python3
"""Calibrate `grpo.direct_match_threshold` (tau) for chunk-denominator R-DIRECT.

Samples the SFT policy under the SFT-ALIGNED extract prompt (R1 fix — the
distribution wave 2 will actually train on), parses flows with the production
`valid_gate`, embeds them, and measures cosine similarity against the chunk's
teacher flows from the PRODUCTION chunk-gold index (`make_direct_chunk_gold`,
restricted norm index, k=3 gold) — never a reimplementation (see
project memory: validate through production code paths).

Two distributions decide tau:
  * signal: per policy flow, best cosine to its OWN chunk's teacher flows
  * null:   per policy flow, best cosine to a DIFFERENT chunk's teacher flows
            (same book — the hardest negatives)
tau goes where the null's upper tail ends / the signal's mass begins.

Run:
  sbatch (1 GPU) — see outputs/2026-07-28_tau_calibration/tau.sub
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path("/share/pierson/matt/UAIR")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from dagspaces.common.stage_utils import ensure_dotenv

TRACES = ROOT / ("multirun/2026-07-26_grpo_m1_core/00-13-20/cell=core/"
                 "grpo_only_online_external/outputs/grpo/checkpoint/"
                 "reward_traces.jsonl")
MERGED_SFT = ROOT / ("multirun/2026-07-26_grpo_m1_core/00-13-20/cell=core/"
                     "grpo_only_online_external/outputs/grpo/checkpoint/"
                     "_merged_sft")
OUT_DIR = ROOT / "outputs/2026-07-28_tau_calibration"
EMB_MODEL = "/share/pierson/matt/zoo/models/Qwen3-Embedding-8B"
N_CHUNKS = 120
N_SAMPLES = 4
SEED = 0


def main() -> int:
    ensure_dotenv()
    import os

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from dagspaces.grpo_training.stages.aux_scorers import (
        _flatten_flow,
        _flow_to_query,
        make_direct_chunk_gold,
    )
    from dagspaces.grpo_training.stages.clients import EmbeddingClient
    from dagspaces.grpo_training.stages.modular_reward import valid_gate
    from dagspaces.grpo_training.stages.sft_data_prep import (
        sft_aligned_extract_template,
    )

    # ---- chunks: the m1 extract population, gold-YES only ----------------
    seen = set()
    for line in open(TRACES):
        o = json.loads(line)
        if (o.get("task_type") == "extract" and o.get("chunk_id") is not None
                and o.get("gold_has_exchange") is True):
            seen.add((str(o["source_id"]), str(o["chunk_id"])))
    picked = random.Random(SEED).sample(sorted(seen), min(N_CHUNKS, len(seen)))

    reasoning = pd.read_parquet(os.environ["CI_REASONING_PATH"])
    lut = {(str(g), str(int(c))): t for g, c, t in zip(
        reasoning["gutenberg_id"].astype(str),
        reasoning["chunk_id"].astype(int),
        reasoning["article_text"])}
    rows = [{"key": k, "chunk_text": lut[k]} for k in picked
            if isinstance(lut.get(k), str) and lut[k].strip()]
    print(f"[tau] {len(rows)} chunks resolved")

    # ---- production chunk-gold index -------------------------------------
    universes = json.load(open(os.environ["NORM_UNIVERSES_PATH"]))
    emb_client = EmbeddingClient(
        base_url=os.environ["EMBEDDING_SERVER_URL"], model_name=EMB_MODEL)
    grpo_cfg = {"embeddings_dir": os.environ.get("NORM_EMBEDDINGS_PATH", "")}
    chunk_gold = make_direct_chunk_gold(
        None, grpo_cfg, universes, {tuple(r["key"]) for r in rows},
        embedding_client=emb_client)

    # ---- sample the policy under the ALIGNED prompt ----------------------
    template = sft_aligned_extract_template(OmegaConf.create({}))

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(str(MERGED_SFT), trust_remote_code=True)
    prompts = [tok.apply_chat_template(
        [{"role": "user",
          "content": template.replace("{{chunk_text}}", r["chunk_text"]).strip()}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for r in rows]

    llm = LLM(model=str(MERGED_SFT), dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.9, disable_custom_all_reduce=True)
    outs = llm.generate(prompts, SamplingParams(
        n=N_SAMPLES, temperature=1.0, top_p=1.0, max_tokens=3072, seed=SEED))

    # ---- flows -> embeddings -> signal/null cosine pools -----------------
    signal, null, gate_pass = [], [], 0
    flow_texts, flow_keys = [], []
    for r, out in zip(rows, outs):
        for comp in out.outputs:
            g = valid_gate(comp.text)
            if not g.passed or not g.flows:
                continue
            gate_pass += 1
            for f in g.flows:
                if isinstance(f, dict):
                    flow_texts.append(_flow_to_query(_flatten_flow(f)))
                    flow_keys.append(tuple(r["key"]))
    print(f"[tau] {gate_pass} gate-passing completions, "
          f"{len(flow_texts)} policy flows")

    V = np.asarray(emb_client.encode_batch(flow_texts), dtype=np.float32)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)

    entries = {k: chunk_gold.get(*k) for k in {tuple(r["key"]) for r in rows}}
    by_book: dict[str, list] = {}
    for k, e in entries.items():
        if e is not None:
            by_book.setdefault(k[0], []).append((k, e))

    rng = random.Random(SEED)
    for v, key in zip(V, flow_keys):
        own = entries.get(key)
        if own is not None:
            signal.append(float(np.max(own["emb"] @ v)))
        others = [e for k2, e in by_book.get(key[0], []) if k2 != key]
        if others:
            null.append(float(np.max(rng.choice(others)["emb"] @ v)))

    sig, nul = np.asarray(signal), np.asarray(null)
    qs = [5, 10, 25, 50, 75, 90, 95]
    result = {
        "n_chunks": len(rows), "n_policy_flows": len(flow_texts),
        "n_signal": len(sig), "n_null": len(nul),
        "signal_pcts": dict(zip(qs, np.percentile(sig, qs).round(4).tolist())),
        "null_pcts": dict(zip(qs, np.percentile(nul, qs).round(4).tolist())),
        "frac_signal_above": {},
        "frac_null_above": {},
    }
    for tau in (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        result["frac_signal_above"][tau] = float((sig >= tau).mean())
        result["frac_null_above"][tau] = float((nul >= tau).mean())

    (OUT_DIR / "tau_calibration.json").write_text(json.dumps(result, indent=2))
    np.savez(OUT_DIR / "tau_pools.npz", signal=sig, null=nul)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
