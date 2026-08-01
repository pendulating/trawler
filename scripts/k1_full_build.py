#!/usr/bin/env python3
"""K1 — ONE clean job: sample + label + edit + build the k-series dataset.

Ruled 2026-07-31: no piecemeal cache assembly. This job, on 4 GPUs:

  phase 1  parent computes THE population from the corpus — every gold-YES
           chunk with >=1 scorable teacher flow (via the production
           chunk-gold index) + all gold-NO chunks — and writes
           population.parquet with the byte-exact rollout prompts;
  phase 2  4 data-parallel workers (one vLLM engine per GPU; DP not TP —
           a 9B model on PCIe gets ~4x from sharding, ~1.5x from TP=4)
           generate a UNIFORM N=8 samples/chunk at the m2 rollout params,
           each writing its shard; parent merges -> samples.parquet;
  phase 3  parent runs `build_kto_dataset` (labeler -> D1' -> ladder edits
           -> teacher rationales via an 8-worker pool -> split/composition
           invariants) -> kto_rows.parquet + kto_metadata.json.

Everything the dataset depends on is produced or recorded inside this one
job directory: population, samples, per-shard seeds, dataset, metadata.

Worker mode: K1_WORKER=<i> K1_NUM_WORKERS=<n> (parent sets
CUDA_VISIBLE_DEVICES per worker).

Run: sbatch outputs/2026-07-31_k1_full/k1full.sub  (gpu:4)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/share/pierson/matt/UAIR")
sys.path.insert(0, str(ROOT))

import pandas as pd
from omegaconf import OmegaConf

from dagspaces.common.stage_utils import ensure_dotenv

MERGED_SFT = ROOT / ("multirun/2026-07-28_grpo_m2_core/21-31-11/cell=core/"
                     "grpo_only_online_external/outputs/grpo/checkpoint/"
                     "_merged_sft")
OUT = ROOT / "outputs/2026-07-31_k1_full"
EMB_MODEL = "/share/pierson/matt/zoo/models/Qwen3-Embedding-8B"
N_SAMPLES = 8          # uniform, every chunk
BASE_SEED = 1000       # worker i uses BASE_SEED + i


# ===========================================================================
# WORKER: generate one shard on one GPU
# ===========================================================================
def worker() -> int:
    i = int(os.environ["K1_WORKER"])
    n = int(os.environ["K1_NUM_WORKERS"])
    pop = pd.read_parquet(OUT / "population.parquet").sort_values(
        ["k0", "k1"]).reset_index(drop=True)
    shard = pop.iloc[i::n]
    print(f"[k1w{i}] {len(shard)} chunks on GPU "
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')}")

    from vllm import LLM, SamplingParams

    llm = LLM(model=str(MERGED_SFT), dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85, disable_custom_all_reduce=True,
              max_num_batched_tokens=2048)
    outs = llm.generate(list(shard["prompt"]), SamplingParams(
        n=N_SAMPLES, temperature=1.0, top_p=1.0, max_tokens=3072,
        seed=BASE_SEED + i))
    rows = [{"k0": r.k0, "k1": r.k1, "sample": j, "text": c.text,
             "shard": i, "seed": BASE_SEED + i}
            for r, o in zip(shard.itertuples(), outs)
            for j, c in enumerate(o.outputs)]
    (OUT / "shards").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT / "shards" / f"shard_{i}.parquet")
    print(f"[k1w{i}] wrote {len(rows)} completions. DONE")
    return 0


# ===========================================================================
# PARENT
# ===========================================================================
def main() -> int:
    ensure_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    from dagspaces.grpo_training.stages.aux_scorers import (
        make_direct_chunk_gold,
    )
    from dagspaces.grpo_training.stages.clients import EmbeddingClient
    from dagspaces.grpo_training.stages.kto_data_prep import build_kto_dataset
    from dagspaces.grpo_training.stages.sft_data_prep import (
        sft_aligned_extract_template,
    )

    # ---- phase 1: population + prompts + chunk-gold index ----------------
    reasoning = pd.read_parquet(os.environ["CI_REASONING_PATH"])
    lut, gold_yes_flag = {}, {}
    for r in reasoning.itertuples():
        k = (str(r.gutenberg_id), str(int(r.chunk_id)))
        if isinstance(r.article_text, str) and r.article_text.strip():
            lut[k] = r.article_text
            gold_yes_flag[k] = bool(r.has_information_exchange)

    fl = pd.read_parquet(os.environ["CI_EXTRACTION_PATH"])
    flow_keys = {(str(g), str(int(c))) for g, c in zip(
        fl["gutenberg_id"].astype(str), fl["chunk_id"].astype(int))}

    universes = json.load(open(os.environ["NORM_UNIVERSES_PATH"]))
    emb_client = EmbeddingClient(
        base_url=os.environ["EMBEDDING_SERVER_URL"], model_name=EMB_MODEL)
    chunk_gold = make_direct_chunk_gold(
        None, {"embeddings_dir": os.environ.get("NORM_EMBEDDINGS_PATH", "")},
        universes, {k for k in flow_keys if k in lut},
        keep_norm_info=True, embedding_client=emb_client)

    yes_keys = sorted(k for k in flow_keys
                      if k in lut and chunk_gold.get(*k) is not None)
    no_keys = sorted(k for k, v in gold_yes_flag.items() if not v)
    print(f"[k1] population: {len(yes_keys)} gold-YES (scorable) + "
          f"{len(no_keys)} gold-NO; uniform N={N_SAMPLES}")

    tok = AutoTokenizer.from_pretrained(str(MERGED_SFT), trust_remote_code=True)
    template = sft_aligned_extract_template(OmegaConf.create({}))

    def fmt(k):
        up = template.replace("{{chunk_text}}", lut[k]).strip()
        return tok.apply_chat_template(
            [{"role": "user", "content": up}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    prompts = {k: fmt(k) for k in [*yes_keys, *no_keys]}
    pd.DataFrame([{"k0": k[0], "k1": k[1],
                   "gold_yes": k in set(yes_keys), "prompt": prompts[k]}
                  for k in [*yes_keys, *no_keys]]).to_parquet(
        OUT / "population.parquet")

    # ---- phase 2: data-parallel generation -------------------------------
    gpus = [g.strip() for g in
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if g.strip()]
    n_workers = len(gpus)
    print(f"[k1] spawning {n_workers} generation workers on GPUs {gpus}")
    procs = []
    for i, gpu in enumerate(gpus):
        env = {**os.environ, "K1_WORKER": str(i),
               "K1_NUM_WORKERS": str(n_workers),
               "CUDA_VISIBLE_DEVICES": gpu}
        log = open(OUT / f"worker_{i}.log", "w")
        procs.append((i, subprocess.Popen(
            [sys.executable, __file__], env=env, stdout=log, stderr=log)))
    fails = [i for i, p in procs if p.wait() != 0]
    if fails:
        raise RuntimeError(f"[k1] generation workers failed: {fails} "
                           f"(see worker_*.log)")
    samples = pd.concat(
        [pd.read_parquet(OUT / "shards" / f"shard_{i}.parquet")
         for i in range(n_workers)], ignore_index=True)
    expected = (len(yes_keys) + len(no_keys)) * N_SAMPLES
    if len(samples) != expected:
        raise RuntimeError(f"[k1] merged {len(samples)} completions, "
                           f"expected {expected}")
    samples.to_parquet(OUT / "samples.parquet")
    print(f"[k1] merged {len(samples)} completions from {n_workers} shards")

    # ---- phase 3: label -> edits -> rationales -> dataset ----------------
    chunk_info = {}
    for k in yes_keys:
        e = chunk_gold.get(*k)
        chunk_info[k] = {"book": k[0], "gold_yes": True,
                         "mixed": len(set(e["golds"])) > 1}
    for k in no_keys:
        chunk_info[k] = {"book": k[0], "gold_yes": False, "mixed": False}

    import requests
    from concurrent.futures import ThreadPoolExecutor

    judge_url = os.environ["VLLM_SERVER_URL"]
    judge_model = os.environ["JUDGE_MODEL_PATH"]
    n_calls = {"ok": 0, "err": 0}

    def _one_rationale(c, parsed):
        flow = parsed["flows"][c.flow_index]
        prompt = (
            "You are explaining a Contextual Integrity judgment. "
            f"Governing norm: \"{(c.norm or {}).get('articulation')}\" "
            f"(force: {(c.norm or {}).get('normative_force')}). "
            "Information flow: "
            + json.dumps({k2: flow.get(k2) for k2 in (
                "sender", "recipient", "subject", "information_type",
                "transmission_principle", "context")}, ensure_ascii=False)
            + f". In one or two sentences, explain why this norm makes the "
              f"flow {c.gold}. Quote the norm verbatim once and end with "
              f"the word '{c.gold}'.")
        try:
            resp = requests.post(
                f"{judge_url}/v1/chat/completions",
                json={"model": judge_model, "temperature": 0.3,
                      "max_tokens": 160,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=90)
            n_calls["ok"] += 1
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            n_calls["err"] += 1
            return None

    def rationale_batch_fn(items):
        print(f"[k1] generating {len(items)} teacher rationales (8 workers)")
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(lambda it: _one_rationale(*it), items))

    rows, metadata = build_kto_dataset(
        samples[["k0", "k1", "sample", "text"]], chunk_gold, chunk_info,
        prompts, rationale_batch_fn=rationale_batch_fn)
    metadata["teacher_calls"] = n_calls
    metadata["generation"] = {
        "n_samples_per_chunk": N_SAMPLES, "base_seed": BASE_SEED,
        "n_workers": n_workers, "merged_sft": str(MERGED_SFT),
        "population": {"gold_yes": len(yes_keys), "gold_no": len(no_keys)},
    }
    rows.to_parquet(OUT / "kto_rows.parquet")
    (OUT / "kto_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str))
    print(f"[k1] wrote {len(rows)} rows -> {OUT}")
    print(json.dumps({k: v for k, v in metadata.items()
                      if k != "heldout_keys"}, indent=2, default=str))
    print("[k1] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(worker() if "K1_WORKER" in os.environ else main())
