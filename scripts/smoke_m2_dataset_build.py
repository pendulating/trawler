#!/usr/bin/env python3
"""Pre-launch smoke: run the PRODUCTION wave-2 dataset build end-to-end.

Composes the real m2 config (hydra compose, cell=full), then executes the
exact build path the stage runs — SFT-aligned template, battery build with
the both-sides floor, stratified prescreen with the realized-mix invariant,
chunk-gold index over the restricted norm index — against the real corpus,
universes, embeddings, and the live embedding server. Finishes by scoring
two synthetic completions through the reward callable (gate -> chunk-
denominator core) so the score-time path is exercised too.

This validates ON the production code what the pre-launch audits validated
with offline reproductions: n_batteries >= 108, realized mix 0.82/0.18
within tolerance, ~68% mixed-gold chunks at gold_k=1, full index coverage,
zero one-sided batteries. Any invariant violation raises.

CPU + embedding server only (no GPU). Run from the repo root.
"""
from __future__ import annotations

import json
import sys
import tempfile

sys.path.insert(0, "/share/pierson/matt/UAIR")

from dagspaces.common.stage_utils import ensure_dotenv


def main() -> int:
    ensure_dotenv()
    import os

    import pandas as pd
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(
        config_dir="/share/pierson/matt/UAIR/dagspaces/grpo_training/conf",
        version_base="1.3",
    ):
        cfg = compose(config_name="config",
                      overrides=["+sweep=grpo_m2_grid", "cell=full"])
    grpo_cfg = OmegaConf.to_container(cfg.training.grpo, resolve=True)
    print(f"[smoke] composed cell=full: task_mix={grpo_cfg['task_mix']} "
          f"gold_k={grpo_cfg['direct_gold_k']} tau={grpo_cfg['direct_match_threshold']} "
          f"floor={grpo_cfg['battery']['minority_floor']}")

    # chunks_df exactly as the stage builds it (grpo_training.py:405-432)
    chunks_df = pd.read_parquet(os.environ["CI_REASONING_PATH"])
    chunks_df = chunks_df.rename(columns={"article_text": "chunk_text"})
    chunks_df["source_id"] = chunks_df["gutenberg_id"].astype(str)
    chunks_df = chunks_df[chunks_df["chunk_text"].notna()].reset_index(drop=True)

    universes = json.load(open(os.environ["NORM_UNIVERSES_PATH"]))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "/share/pierson/matt/UAIR/multirun/2026-07-26_grpo_m1_core/00-13-20/"
        "cell=core/grpo_only_online_external/outputs/grpo/checkpoint/_merged_sft",
        trust_remote_code=True)

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(
        str(grpo_cfg.get("context_embedding_model", "all-MiniLM-L6-v2")))
    embed_fn = lambda t: st.encode(t, normalize_embeddings=True)  # noqa: E731

    from dagspaces.grpo_training.stages.modular_reward import (
        build_modular_dataset,
        make_modular_reward_from_cfg,
    )
    from dagspaces.grpo_training.stages.sft_data_prep import (
        sft_aligned_extract_template,
    )

    reward_fn = make_modular_reward_from_cfg(cfg, grpo_cfg, universes)
    assert reward_fn.answerer is None, "answerer built in direct mode!"
    template = sft_aligned_extract_template(cfg)

    with tempfile.TemporaryDirectory() as tmp:
        dataset, metadata = build_modular_dataset(
            cfg=cfg, grpo_cfg=grpo_cfg, chunks_df=chunks_df,
            norm_universes=universes, reward_fn=reward_fn, tokenizer=tok,
            ci_prompt_template=template, output_dir=tmp, seed=42,
            embed_fn=embed_fn,
        )
        meta_on_disk = json.load(open(os.path.join(tmp, "training_metadata.json")))

    # --- assertions on the audit expectations -----------------------------
    n_vig = sum(1 for r in dataset if r["task_type"] == "vignette")
    n_ext = len(dataset) - n_vig
    print(f"[smoke] dataset: {len(dataset)} rows = {n_ext} extract + {n_vig} vignette")
    assert len(dataset) == 600, len(dataset)
    assert abs(n_vig / len(dataset) - 0.18) <= 0.03, n_vig

    comps = meta_on_disk.get("battery_compositions") or []
    assert comps, "battery_compositions missing from metadata"
    one_sided = [c for c in comps
                 if c.get("n_gold_no", 0) == 0 or c.get("n_gold_yes", 0) == 0]
    assert not one_sided, f"{len(one_sided)} one-sided batteries!"
    print(f"[smoke] batteries built: {len(comps)}, one-sided: 0")

    cg = reward_fn._direct_chunk_gold
    assert cg is not None, "chunk-gold index not attached"
    n_mixed = sum(1 for v in cg.index.values() if len(set(v["golds"])) > 1)
    frac_mixed = n_mixed / len(cg.index)
    print(f"[smoke] chunk-gold: {len(cg.index)} chunks, "
          f"{sum(len(v['golds']) for v in cg.index.values())} teacher flows, "
          f"mixed-gold {frac_mixed:.1%}")
    assert 0.60 <= frac_mixed <= 0.78, frac_mixed

    # --- score-time path: 2 synthetic completions through the callable ----
    extract_prompts = [k for k, v in reward_fn.prompt_metadata.items()
                       if v.get("task_type") == "extract"
                       and v.get("gold_has_exchange") is True]
    pk = extract_prompts[0]
    meta = reward_fn.prompt_metadata[pk]
    entry = cg.get(str(meta["source_id"]), str(meta["chunk_id"]))
    assert entry is not None, "sampled chunk missing from index"
    good = json.dumps({
        "reasoning": "r",
        "has_information_exchange": True,
        "flows": [{
            "sender": "a", "recipient": "b", "subject": "c",
            "information_type": entry["texts"][0][:80],
            "transmission_principle": "t", "context": "x",
            "appropriateness": entry["golds"][0],
        }],
    })
    scores = reward_fn(prompts=[pk, pk], completions=[good, "not json at all"])
    print(f"[smoke] reward call: valid={scores[0]:.3f} gate_fail={scores[1]:.3f}")
    assert scores[1] == 0.0
    assert scores[0] >= 0.15  # valid-path floor
    lm = reward_fn.last_metrics
    print(f"[smoke] live metrics sample: "
          f"miss_frac={lm.get('reward/direct/miss_frac')}, "
          f"spurious={lm.get('reward/direct/spurious_flow_frac')}, "
          f"gate_frac={lm.get('reward/valid/gate_frac')}")

    print("[smoke] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
