#!/usr/bin/env python3
"""K0 — the k-series mining probe (wiki/2026-07-31_kto_plan.md §9).

Measures, through PRODUCTION code paths only, everything the KTO plan left
as a pre-registered measurement:

  * D1 (primary) and D2 desirability yields over N=16 policy samples/chunk
    on the m2 gold-YES extract population; R-ABSTAIN yields on the gold-NO
    chunks (N=4);
  * gate-fail rate under the aligned prompt (abort signal if > 8%);
  * edit opportunity rate + articulation availability for the ladder
    (R-VERDICT / R-CITATION / R-SCRUTINIZE), with per-depth round-trip
    validation through `valid_gate`;
  * flip-sensitivity of desirability labels to the [0.55, 0.65) match band
    (the `kto.min_edit_sim` contingency trigger at > 10%);
  * perplexity delta of edited vs original completions (n=60 pairs, via
    vLLM prompt_logprobs) + n=20 per-depth eyeball exports;
  * teacher-generated rationale validation rate (n=50, judge server) vs
    the deterministic template (validated by construction);
  * the held-out scorer noise floor: two seeded N=8 re-samples over a 20%
    chunk subsample -> between-seed spread of minority-class accuracy;
  * realized class counts -> recommended desirable/undesirable weights for
    the TRL ratio band [1, 4/3].

Run (1 GPU): sbatch outputs/2026-07-31_k0_probe/k0.sub
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/share/pierson/matt/UAIR")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from dagspaces.common.stage_utils import ensure_dotenv

M2_TRACES = ROOT / ("multirun/2026-07-28_grpo_m2_full/21-31-11/cell=full/"
                    "grpo_only_online_external/outputs/grpo/checkpoint/"
                    "reward_traces.jsonl")
MERGED_SFT = ROOT / ("multirun/2026-07-28_grpo_m2_core/21-31-11/cell=core/"
                     "grpo_only_online_external/outputs/grpo/checkpoint/"
                     "_merged_sft")
OUT = ROOT / "outputs/2026-07-31_k0_probe"
EMB_MODEL = "/share/pierson/matt/zoo/models/Qwen3-Embedding-8B"
N_MAIN, N_NOFLOW, N_NOISE = 16, 4, 8
TAU, LOW_BAND = 0.55, (0.55, 0.65)
SEED = 0


def d1_label(per_flow):
    """per_flow: list of (gold, correct) for MATCHED flows. -> D1 verdict."""
    if not per_flow:
        return "excluded"
    viol = [c for g, c in per_flow if g == "inappropriate"]
    if viol and not all(viol):
        return "undesirable"
    n_ok = sum(c for _, c in per_flow)
    if viol and all(viol) and n_ok >= len(per_flow) / 2:
        return "desirable"
    if not viol:  # all-appropriate evidence: specificity control
        return "desirable" if n_ok == len(per_flow) else "false_alarm"
    return "neither"


def main() -> int:
    ensure_dotenv()
    import os

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "edit_samples").mkdir(exist_ok=True)

    from dagspaces.grpo_training.stages.aux_scorers import (
        make_direct_chunk_gold,
    )
    from dagspaces.grpo_training.stages.clients import EmbeddingClient
    from dagspaces.grpo_training.stages.kto_edits import (
        Correction,
        apply_citation_edit,
        apply_scrutinize_edit,
        apply_verdict_edit,
        rationale_is_valid,
        render_rationale,
        serialize_completion,
    )
    from dagspaces.grpo_training.stages.modular_reward import (
        match_flows,
        valid_gate,
    )
    from dagspaces.grpo_training.stages.sft_data_prep import (
        sft_aligned_extract_template,
    )

    # ---- populations (from the m2 full cell's realized dataset) ----------
    yes_keys, no_keys = set(), set()
    for line in open(M2_TRACES):
        o = json.loads(line)
        if o.get("task_type") != "extract" or o.get("chunk_id") is None:
            continue
        k = (str(o["source_id"]), str(o["chunk_id"]))
        if o.get("gold_has_exchange") is True:
            yes_keys.add(k)
        elif o.get("gold_has_exchange") is False:
            no_keys.add(k)
    reasoning = pd.read_parquet(os.environ["CI_REASONING_PATH"])
    lut = {(str(g), str(int(c))): t for g, c, t in zip(
        reasoning["gutenberg_id"].astype(str),
        reasoning["chunk_id"].astype(int), reasoning["article_text"])}
    yes_keys = sorted(k for k in yes_keys if isinstance(lut.get(k), str))
    no_keys = sorted(k for k in no_keys if isinstance(lut.get(k), str))
    print(f"[k0] population: {len(yes_keys)} gold-YES, {len(no_keys)} gold-NO")

    universes = json.load(open(os.environ["NORM_UNIVERSES_PATH"]))
    # Explicit client: with cfg=None the factory would fall back to
    # model_name="default", which vLLM 404s (third scripting incident of
    # this trap — the K1 stage must resolve it from cfg as production does).
    emb_client = EmbeddingClient(
        base_url=os.environ["EMBEDDING_SERVER_URL"], model_name=EMB_MODEL)
    chunk_gold = make_direct_chunk_gold(
        None, {"embeddings_dir": os.environ.get("NORM_EMBEDDINGS_PATH", "")},
        universes, set(yes_keys), keep_norm_info=True,
        embedding_client=emb_client)

    # ---- generation -------------------------------------------------------
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(str(MERGED_SFT), trust_remote_code=True)
    template = sft_aligned_extract_template(OmegaConf.create({}))

    def fmt(key):
        up = template.replace("{{chunk_text}}", lut[key]).strip()
        return tok.apply_chat_template(
            [{"role": "user", "content": up}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    # max_num_batched_tokens caps the chunked-prefill step so the ppl pass's
    # prompt_logprobs buffer (full vocab x scheduled tokens) stays ~0.5 GB —
    # an unchunked 2.4k-token prompt OOMed the first K0 run. 0.85 leaves the
    # headroom the buffer lives in.
    llm = LLM(model=str(MERGED_SFT), dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85, disable_custom_all_reduce=True,
              max_num_batched_tokens=768)

    rng = random.Random(SEED)
    noise_keys = rng.sample(yes_keys, max(1, len(yes_keys) // 5))

    GEN_CACHE = OUT / "gen_cache.parquet"

    def generate(tag, keys, n, seed, _cache={}):
        if not _cache and GEN_CACHE.exists():
            df = pd.read_parquet(GEN_CACHE)
            for (t, k0, k1), g in df.groupby(["tag", "k0", "k1"]):
                _cache.setdefault(t, {})[(k0, k1)] = list(
                    g.sort_values("sample")["text"])
        if tag in _cache and all(tuple(k) in _cache[tag] for k in keys):
            print(f"[k0] gen cache hit: {tag}")
            return {k: _cache[tag][tuple(k)] for k in keys}
        outs = llm.generate(
            [fmt(k) for k in keys],
            SamplingParams(n=n, temperature=1.0, top_p=1.0,
                           max_tokens=3072, seed=seed))
        res = {k: [c.text for c in o.outputs] for k, o in zip(keys, outs)}
        rows = [{"tag": tag, "k0": k[0], "k1": k[1], "sample": i, "text": t}
                for k, ts in res.items() for i, t in enumerate(ts)]
        old = pd.read_parquet(GEN_CACHE) if GEN_CACHE.exists() else None
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True) \
            .to_parquet(GEN_CACHE) if old is not None else \
            pd.DataFrame(rows).to_parquet(GEN_CACHE)
        return res

    main_gen = generate("main", yes_keys, N_MAIN, SEED)
    noise_gen = {s: generate(f"noise{s}", noise_keys, N_NOISE, s)
                 for s in (1, 2)}
    noflow_gen = generate("noflow", no_keys, N_NOFLOW, SEED)

    # ---- scoring ----------------------------------------------------------
    def score(key, text):
        g = valid_gate(text)
        if not g.passed:
            return {"status": "gate_fail"}
        entry = chunk_gold.get(*key)
        if entry is None or not g.flows:
            return {"status": "excluded"}
        p_emb = chunk_gold.embed_flows(g.flows)
        if not np.asarray(p_emb).any(axis=1).all():
            return {"status": "embed_fail"}
        matches = match_flows(entry["emb"] @ p_emb.T, TAU)
        per_flow, per_flow_hi, corrections = [], [], []
        for t, p, sim in matches:
            gold = entry["golds"][t]
            label = str(g.flows[p].get("appropriateness") or "").strip().lower()
            correct = label == gold
            per_flow.append((gold, correct))
            if sim >= LOW_BAND[1]:
                per_flow_hi.append((gold, correct))
            if not correct:
                corrections.append(Correction(
                    flow_index=p, gold=gold,
                    norm=(entry.get("norms") or [{}] * len(entry["golds"]))[t],
                    match_sim=float(sim)))
        macro_by = defaultdict(list)
        for gold, c in per_flow:
            macro_by[gold].append(float(c))
        d2 = (sum(sum(v) / len(v) for v in macro_by.values()) / len(macro_by)
              if macro_by else None)
        return {"status": "scored", "d1": d1_label(per_flow),
                "d1_hi": d1_label(per_flow_hi),
                "d2": d2, "parsed": g.parsed, "corrections": corrections,
                "n_matched": len(matches), "n_flows": len(g.flows)}

    records, edit_pairs = [], []
    for key in yes_keys:
        for i, text in enumerate(main_gen[key]):
            r = score(key, text)
            records.append({"key": str(key), "sample": i,
                            "status": r["status"],
                            "d1": r.get("d1"), "d1_hi": r.get("d1_hi"),
                            "d2": r.get("d2"),
                            "n_flows": r.get("n_flows"),
                            "n_matched": r.get("n_matched"),
                            "n_corr": len(r.get("corrections") or [])})
            if r["status"] == "scored" and r.get("corrections"):
                elig = [c for c in r["corrections"] if c.match_sim >= TAU]
                if elig and r["d1"] == "undesirable":
                    edit_pairs.append((key, text, r["parsed"], elig))
    rec_df = pd.DataFrame(records)
    rec_df.to_parquet(OUT / "per_completion.parquet")

    scored = rec_df[rec_df["status"] == "scored"]
    gate_rate = (rec_df["status"] == "gate_fail").mean()
    d1c = Counter(scored["d1"])
    report = {
        "population": {"gold_yes": len(yes_keys), "gold_no": len(no_keys),
                       "samples_per_chunk": N_MAIN},
        "gate_fail_rate": round(float(gate_rate), 4),
        "d1_yields": dict(d1c),
        "d1_yields_frac": {k: round(v / len(scored), 4)
                          for k, v in d1c.items()},
        "d2_ge_075_frac": round(float((scored["d2"] >= 0.75).mean()), 4),
        "edit_pairs_available": len(edit_pairs),
        "flip_sensitivity": round(float(
            (scored["d1"] != scored["d1_hi"]).mean()), 4),
    }

    # ---- edits: round-trip, articulation coverage, eyeball exports --------
    depths = {"verdict": apply_verdict_edit, "citation": apply_citation_edit,
              "scrutinize": apply_scrutinize_edit}
    rt_fail = Counter()
    art_missing = 0
    eye = rng.sample(edit_pairs, min(20, len(edit_pairs)))
    ppl_pairs = rng.sample(edit_pairs, min(60, len(edit_pairs)))
    for name, fn in depths.items():
        for j, (key, text, parsed, corrs) in enumerate(edit_pairs):
            if name == "citation":
                art_missing += sum(
                    1 for c in corrs if not (c.norm or {}).get("articulation"))
            edited = serialize_completion(fn(parsed, corrs))
            g = valid_gate(edited)
            if not g.passed:
                rt_fail[name] += 1
            if (key, text, parsed, corrs) in eye and name != "verdict":
                (OUT / "edit_samples" / f"{name}_{j}.txt").write_text(
                    f"=== ORIGINAL ===\n{text}\n\n=== {name.upper()} ===\n{edited}\n")
    report["roundtrip_gate_failures"] = dict(rt_fail)
    report["citation_articulation_missing"] = art_missing

    # ---- perplexity delta (edited vs original, n<=60, all three depths) ---
    def mean_logprob(prompt, completion):
        full = prompt + completion
        out = llm.generate([full], SamplingParams(
            max_tokens=1, temperature=0.0, prompt_logprobs=0))[0]
        n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        lps = [next(iter(d.values())).logprob
               for d in out.prompt_logprobs[n_prompt:] if d]
        return sum(lps) / max(1, len(lps))

    ppl = {}
    for name, fn in depths.items():
        deltas = []
        for key, text, parsed, corrs in ppl_pairs[:20]:
            base = mean_logprob(fmt(key), text)
            edit = mean_logprob(fmt(key), serialize_completion(fn(parsed, corrs)))
            deltas.append(edit - base)
        if deltas:
            ppl[name] = {"mean_logprob_delta": round(float(np.mean(deltas)), 4),
                         "p10": round(float(np.percentile(deltas, 10)), 4)}
    report["perplexity_delta"] = ppl

    # ---- teacher rationales (n=50 corrections via judge server) -----------
    import requests
    judge_url = os.environ.get("VLLM_SERVER_URL", "")
    judge_model = os.environ.get("JUDGE_MODEL_PATH", "")
    t_ok, t_n, t_rows = 0, 0, []
    if judge_url:
        flat = [(c, parsed) for _, _, parsed, corrs in edit_pairs
                for c in corrs][:50]
        for c, parsed in flat:
            flow = parsed["flows"][c.flow_index]
            prompt = (
                "You are explaining a Contextual Integrity judgment. "
                f"Governing norm: \"{(c.norm or {}).get('articulation')}\" "
                f"(force: {(c.norm or {}).get('normative_force')}). "
                f"Information flow: {json.dumps({k: flow.get(k) for k in ('sender', 'recipient', 'subject', 'information_type', 'transmission_principle', 'context')}, ensure_ascii=False)}. "
                f"In one or two sentences, explain why this norm makes the "
                f"flow {c.gold}. Quote the norm verbatim once and end with "
                f"the word '{c.gold}'.")
            try:
                resp = requests.post(
                    f"{judge_url}/v1/chat/completions",
                    json={"model": judge_model, "temperature": 0.3,
                          "max_tokens": 160,
                          "messages": [{"role": "user", "content": prompt}]},
                    timeout=90)
                textr = resp.json()["choices"][0]["message"]["content"]
            except Exception as exc:
                textr = f"<error: {exc}>"
            ok = rationale_is_valid(textr, c)
            t_ok += ok
            t_n += 1
            t_rows.append({"gold": c.gold, "valid": ok, "text": textr})
        (OUT / "teacher_rationales.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in t_rows))
    report["teacher_rationale"] = {
        "n": t_n, "valid_rate": round(t_ok / t_n, 3) if t_n else None}
    report["template_rationale_valid_rate"] = round(float(np.mean([
        rationale_is_valid(render_rationale(c.flow_index, c), c)
        for _, _, _, corrs in edit_pairs[:200] for c in corrs]) if edit_pairs
        else 0.0), 3)

    # ---- gold-NO (R-ABSTAIN) yields ---------------------------------------
    ab = Counter()
    for key in no_keys:
        for text in noflow_gen[key]:
            g = valid_gate(text)
            if not g.passed:
                ab["gate_fail"] += 1
            elif g.no_flow:
                ab["desirable_abstain"] += 1
            else:
                ab["undesirable_extraction"] += 1
    report["abstain_yields"] = dict(ab)

    # ---- noise floor -------------------------------------------------------
    floors = {}
    for s, gen in noise_gen.items():
        hits = Counter()
        for key in noise_keys:
            for text in gen[key]:
                r = score(key, text)
                if r["status"] != "scored":
                    continue
                # recompute per-flow minority accuracy
                entry = chunk_gold.get(*key)
                g = valid_gate(text)
                p_emb = chunk_gold.embed_flows(g.flows)
                for t, p, sim in match_flows(entry["emb"] @ p_emb.T, TAU):
                    if entry["golds"][t] == "inappropriate":
                        lab = str(g.flows[p].get("appropriateness") or "").strip().lower()
                        hits[lab == "inappropriate"] += 1
        tot = hits[True] + hits[False]
        floors[s] = round(hits[True] / tot, 4) if tot else None
    report["noise_floor"] = {
        "minority_acc_by_seed": floors,
        "abs_spread": (round(abs(floors[1] - floors[2]), 4)
                       if None not in floors.values() else None),
        "n_chunks": len(noise_keys), "n_samples": N_NOISE}

    # ---- class weights -----------------------------------------------------
    n_d = d1c.get("desirable", 0) + len(edit_pairs)      # mined + edited
    n_u = d1c.get("undesirable", 0)
    if n_d and n_u:
        target = 1.15  # midpoint of TRL's [1, 4/3]
        report["class_weights"] = {
            "n_desirable": n_d, "n_undesirable": n_u,
            "recommended": {"desirable_weight": round(target * n_u / n_d, 3),
                            "undesirable_weight": 1.0}}

    (OUT / "k0_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("[k0] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
