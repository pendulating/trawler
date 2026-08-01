#!/usr/bin/env python3
"""k-series held-out probe driver (plan §5/§8, K2; logic in kto_probe.py).

One vLLM engine serving the m2 ``_merged_sft`` base; each arm checkpoint
attaches as a LoRA adapter (no per-checkpoint engine reloads). The
epoch-0 baseline (no adapter) runs in the same pass, on the same probe
subset, with the same seed — every §8 gate is a within-probe comparison.

Scoring goes through the PRODUCTION labeler (`label_completion` on the
chunk-gold index) — the same instrument that labeled the training data.

  python scripts/k...probe.py --tier screen \
      --arm citation=multirun/.../checkpoint [--arm ...] \
      --out outputs/2026-08-01_k3_probe

Tier "screen" = fixed seeded ~150-chunk subset per save; tier "full" =
all held-out chunks (gate-passing checkpoints only). GPU + embedding
server. LoRA adapters go through ``_remap_lora_keys_for_vlm`` (the
Qwen3.5 VLM-arch key trap).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("/share/pierson/matt/UAIR")
sys.path.insert(0, str(ROOT))

import pandas as pd

from dagspaces.common.stage_utils import ensure_dotenv

K1_DIR = ROOT / "outputs/2026-07-31_k1_full"
EMB_MODEL = "/share/pierson/matt/zoo/models/Qwen3-Embedding-8B"
N_SAMPLES = 8       # k per chunk — the noise floor (0.011) was measured at 8
GEN_SEED = 7        # one seed for every checkpoint: curves compare like-for-like


def find_checkpoints(arm_dir: Path) -> list[tuple[str, Path]]:
    """(name, path) for each intermediate save + the final adapter."""
    out = sorted(
        ((p.name, p) for p in arm_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda t: int(t[0].rsplit("-", 1)[1]))
    if (arm_dir / "adapter_model.safetensors").exists() or \
       (arm_dir / "adapter_config.json").exists():
        out.append(("final", arm_dir))
    if not out:
        raise FileNotFoundError(f"no checkpoints under {arm_dir}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    metavar="NAME=CKPT_DIR", help="repeatable")
    ap.add_argument("--tier", choices=("screen", "full"), default="screen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k1-dir", default=str(K1_DIR))
    ap.add_argument("--n-screen", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_dotenv()
    k1 = Path(args.k1_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from dagspaces.grpo_training.stages.aux_scorers import (
        make_direct_chunk_gold,
    )
    from dagspaces.grpo_training.stages.clients import EmbeddingClient
    from dagspaces.grpo_training.stages.kto_data_prep import label_completion
    from dagspaces.grpo_training.stages.kto_probe import (
        probe_row,
        select_probe_chunks,
    )
    from dagspaces.grpo_training.stages.modular_reward import valid_gate

    # ---- probe subset (deterministic) ------------------------------------
    metadata = json.load(open(k1 / "kto_metadata.json"))
    pop = pd.read_parquet(k1 / "population.parquet")
    pop["chunk_key"] = pop["k0"].astype(str) + "|" + pop["k1"].astype(str)
    gold_yes = dict(zip(pop["chunk_key"], pop["gold_yes"]))
    prompts = dict(zip(pop["chunk_key"], pop["prompt"]))
    heldout = [k for k in metadata["heldout_keys"] if k in prompts]
    if len(heldout) != len(metadata["heldout_keys"]):
        raise RuntimeError("held-out keys missing from population.parquet")
    probe_keys = select_probe_chunks(
        heldout, gold_yes, args.tier, args.n_screen, args.seed)
    n_no = sum(1 for k in probe_keys if not gold_yes[k])
    print(f"[probe] tier={args.tier}: {len(probe_keys)} chunks "
          f"({len(probe_keys) - n_no} gold-YES, {n_no} gold-NO)")

    # ---- chunk-gold index over probe gold-YES chunks ---------------------
    universes = json.load(open(os.environ["NORM_UNIVERSES_PATH"]))
    emb_client = EmbeddingClient(
        base_url=os.environ["EMBEDDING_SERVER_URL"], model_name=EMB_MODEL)
    yes_keys = {tuple(k.split("|", 1)) for k in probe_keys if gold_yes[k]}
    chunk_gold = make_direct_chunk_gold(
        None, {"embeddings_dir": os.environ.get("NORM_EMBEDDINGS_PATH", "")},
        universes, yes_keys, keep_norm_info=True,
        embedding_client=emb_client)

    # ---- engine + adapters -----------------------------------------------
    base_model = metadata["generation"]["merged_sft"]
    arms = [("baseline", None)]
    for spec in args.arm:
        name, _, d = spec.partition("=")
        for ckpt_name, ckpt_path in find_checkpoints(Path(d)):
            arms.append((f"{name}/{ckpt_name}", ckpt_path))
    print(f"[probe] {len(arms)} (arm, checkpoint) slices incl. baseline")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from dagspaces.common.vllm_inference import _remap_lora_keys_for_vlm

    llm = LLM(model=base_model, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85, disable_custom_all_reduce=True,
              max_num_batched_tokens=2048, enable_lora=True,
              max_lora_rank=64)
    sp = SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=1.0,
                       max_tokens=3072, seed=GEN_SEED)
    prompt_list = [prompts[k] for k in probe_keys]

    rows = []
    for i, (slice_name, ckpt) in enumerate(arms):
        lora_req = None
        if ckpt is not None:
            path = _remap_lora_keys_for_vlm(
                str(ckpt), base_model, stage_name="kto_probe")
            lora_req = LoRARequest(slice_name, i, path)
        print(f"[probe] generating {slice_name} "
              f"({len(prompt_list)} prompts x {N_SAMPLES})")
        outs = llm.generate(prompt_list, sp, lora_request=lora_req)
        for key, o in zip(probe_keys, outs):
            entry = (chunk_gold.get(*key.split("|", 1))
                     if gold_yes[key] else None)
            n_teacher = len(entry["golds"]) if entry else 0
            for comp in o.outputs:
                g = valid_gate(comp.text)
                if gold_yes[key]:
                    res = (label_completion(comp.text, entry,
                                            chunk_gold.embed_flows,
                                            0.55, 0.55))
                else:  # gold-NO: gate + abstention is the whole measurement
                    res = {"status": ("gate_fail" if not g.passed
                                      else "scored")}
                r = probe_row(key, gold_yes[key], res,
                              no_flow=bool(g.passed and g.no_flow),
                              n_teacher_flows=n_teacher)
                r["slice"] = slice_name
                rows.append(r)
        # Incremental save: a crash late in the sweep loses nothing.
        pd.DataFrame(rows).to_parquet(out_dir / "probe_results.parquet")

    meta = {
        "tier": args.tier, "n_chunks": len(probe_keys),
        "probe_keys": probe_keys, "n_samples": N_SAMPLES,
        "gen_seed": GEN_SEED, "subset_seed": args.seed,
        "base_model": base_model,
        "slices": [name for name, _ in arms],
        "dataset_fingerprint": metadata["fingerprint"],
    }
    (out_dir / "probe_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[probe] {len(rows)} rows -> {out_dir}/probe_results.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
