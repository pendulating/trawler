#!/usr/bin/env python3
"""A/B probe: R-VALID gate pass rate of the SFT policy under the GRPO
extraction prompt vs the SFT training prompt (2026-07-28).

Motivation: the m1 wave measured a flat 16-25% `valid_gate` failure rate on
extract rows, consuming 63-79% of extract advantage mass (see
notebooks/normative-simulacra/grpo_m1_wave_analysis_2026_07_28.py, section 4).
The GRPO rollout prompt (`conf/prompt/ci_extraction.yaml`) turned out to be a
completely different text from the SFT training instruction
(`sft_data_prep._build_ci_instruction`) — the policy was scored off-
distribution. This probe answers: how much of the failure rate is the prompt
mismatch vs an in-distribution format weakness?

Design (matches the m1 rollout exactly):
  * model   = the m1 core cell's `_merged_sft` (base + canonical SFT adapter,
              already merged by the trainer — no LoRA remap concerns)
  * chunks  = 200 sampled from the chunk set that actually appeared as
              extract-task rows in the core cell's reward traces
  * arm A   = GRPO prompt: run-config `prompt_ci_extraction` template,
              `{{instruction}}` substituted, chunk substituted, .strip()
              (modular_reward.build_modular_dataset:1216)
  * arm B   = SFT prompt: `_build_ci_instruction()` (all-on defaults, as the
              canonical sweep used) + "\n\n" + article_text
              (sft_data_prep.run_sft_data_prep_stage:337)
  * both arms chat-templated identically (_format_prompt: single user turn,
    add_generation_prompt, enable_thinking=False) and sampled at the exact
    TRL rollout params (temperature=1.0, top_p=1.0, n=8, max_tokens=3072).
  * scoring = modular_reward.valid_gate verbatim (repair=False).

Outputs: per-arm gate fail rate + reason breakdown + paired per-chunk
comparison, printed and saved to JSON.

Run (1 GPU is enough for Qwen3.5-9B):
  srun -p pierson --gres=gpu:1 --cpus-per-task=8 --mem=64G \
    /share/pierson/matt/UAIR/.venv-vllm025cu129/bin/python \
    scripts/ab_probe_sft_vs_grpo_prompt.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/share/pierson/matt/UAIR")
sys.path.insert(0, str(ROOT))

import pandas as pd
from omegaconf import OmegaConf

from dagspaces.common.stage_utils import ensure_dotenv
from dagspaces.grpo_training.stages.modular_reward import valid_gate
from dagspaces.grpo_training.stages.sft_data_prep import _build_ci_instruction

RUN_CFG = ROOT / ("multirun/2026-07-26_grpo_m1_core/00-13-20/cell=core/"
                  ".hydra/config.yaml")
TRACES = ROOT / ("multirun/2026-07-26_grpo_m1_core/00-13-20/cell=core/"
                 "grpo_only_online_external/outputs/grpo/checkpoint/"
                 "reward_traces.jsonl")
MERGED_SFT = ROOT / ("multirun/2026-07-26_grpo_m1_core/00-13-20/cell=core/"
                     "grpo_only_online_external/outputs/grpo/checkpoint/"
                     "_merged_sft")
OUT_DIR = ROOT / "outputs/2026-07-28_ab_prompt_probe"
N_CHUNKS = 200
N_SAMPLES = 8
SEED = 0


def main() -> int:
    ensure_dotenv()
    import os

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- prompts ---------------------------------------------------------
    cfg = OmegaConf.load(RUN_CFG)
    pc = cfg.prompt_ci_extraction
    grpo_template = str(pc.prompt_template).replace(
        "{{instruction}}", str(pc.instruction).strip())
    sft_instruction = _build_ci_instruction()  # all-on defaults = canonical

    # ---- chunk sample: the core cell's actual extract-row chunk set ------
    # (source_id, chunk_id) pairs from every extract-task trace row — the
    # population the m1 24.5% failure rate was measured on. chunk_id alone is
    # NOT unique (per-book numbering restarts at 0).
    seen_keys_set: set[tuple[str, int]] = set()
    for line in open(TRACES):
        o = json.loads(line)
        if o.get("task_type") == "extract" and o.get("chunk_id") is not None:
            seen_keys_set.add((str(o["source_id"]), int(o["chunk_id"])))
    seen_keys = sorted(seen_keys_set)
    rng = random.Random(SEED)
    picked = rng.sample(seen_keys, min(N_CHUNKS, len(seen_keys)))

    reasoning = pd.read_parquet(os.environ["CI_REASONING_PATH"])
    reasoning["gutenberg_id"] = reasoning["gutenberg_id"].astype(str)
    reasoning["chunk_id"] = reasoning["chunk_id"].astype(int)
    lut = {
        (str(g), int(c)): t
        for g, c, t in zip(reasoning["gutenberg_id"], reasoning["chunk_id"],
                           reasoning["article_text"])
    }

    rows = []
    for key in picked:
        text = lut.get(key)
        if text and isinstance(text, str) and text.strip():
            rows.append({"key": key, "chunk_text": text})
    if len(rows) < 50:
        raise RuntimeError(
            f"only {len(rows)} chunks resolved from traces->parquet join "
            f"(keys look like {picked[:3]}); refusing to run an underpowered probe")
    print(f"[probe] {len(rows)} chunks resolved (of {len(picked)} sampled)")

    # ---- format both arms ------------------------------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MERGED_SFT), trust_remote_code=True)

    def fmt(user_prompt: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    prompts, meta = [], []
    for r in rows:
        a = grpo_template.replace("{{chunk_text}}", r["chunk_text"]).strip()
        b = f"{sft_instruction}\n\n{r['chunk_text']}"
        for arm, up in (("grpo_prompt", a), ("sft_prompt", b)):
            prompts.append(fmt(up))
            meta.append({"key": str(r["key"]), "arm": arm})

    # ---- sample at the exact m1 rollout params ---------------------------
    from vllm import LLM, SamplingParams

    llm = LLM(model=str(MERGED_SFT), dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.9, enforce_eager=False,
              disable_custom_all_reduce=True)
    sp = SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=1.0,
                        max_tokens=3072, seed=SEED)
    outs = llm.generate(prompts, sp)

    # ---- gate every completion ------------------------------------------
    per_completion = []
    for m, out in zip(meta, outs):
        for comp in out.outputs:
            g = valid_gate(comp.text)
            per_completion.append({
                **m,
                "passed": g.passed,
                "reason": g.reason,
                "finish_reason": comp.finish_reason,
                "n_tokens": len(comp.token_ids),
            })
    df = pd.DataFrame(per_completion)

    summary = {}
    for arm, sub in df.groupby("arm"):
        fail = ~sub["passed"]
        by_chunk = sub.groupby("key")["passed"].apply(lambda s: (~s).mean())
        summary[arm] = {
            "n_completions": len(sub),
            "gate_fail_rate": float(fail.mean()),
            "reasons": dict(Counter(sub.loc[fail, "reason"])),
            "truncated_frac": float((sub["finish_reason"] != "stop").mean()),
            "mean_tokens": float(sub["n_tokens"].mean()),
            "chunks_all_fail": int((by_chunk == 1.0).sum()),
            "chunks_no_fail": int((by_chunk == 0.0).sum()),
        }

    piv = df.assign(fail=~df["passed"]).pivot_table(
        index="key", columns="arm", values="fail", aggfunc="mean")
    paired = {
        "n_chunks": int(len(piv)),
        "mean_fail_grpo_prompt": float(piv["grpo_prompt"].mean()),
        "mean_fail_sft_prompt": float(piv["sft_prompt"].mean()),
        "chunks_sft_strictly_better": int((piv["sft_prompt"] < piv["grpo_prompt"]).sum()),
        "chunks_grpo_strictly_better": int((piv["grpo_prompt"] < piv["sft_prompt"]).sum()),
    }

    result = {"m1_reference_gate_fail_core": 0.245, "summary": summary,
              "paired": paired, "n_chunks": len(rows),
              "n_samples_per_prompt": N_SAMPLES, "seed": SEED,
              "sampling": {"temperature": 1.0, "top_p": 1.0,
                           "max_tokens": 3072},
              "model": str(MERGED_SFT)}
    out_path = OUT_DIR / "ab_prompt_probe_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    df.to_parquet(OUT_DIR / "per_completion.parquet")

    print(json.dumps(result, indent=2))
    print(f"\n[probe] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
