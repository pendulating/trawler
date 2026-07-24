#!/usr/bin/env python
"""Null-answerability calibration pass for the R-OUTCOME probe pool (m-series).

Job 4 of the m-series data prerequisites (wiki/grpo_redesign/data.md; the
"Probe generation" step 5 in wiki/grpo_redesign/reward-outcome.md). Builds the
per-chunk candidate probe pools from the fiction10-gemma4 teacher reference
flows, votes each UNIQUE probe against the frozen answerer (Qwen3.6-27B) with an
EMPTY extraction, drops the probes the answerer already answers correctly from
world knowledge (null-answerability filter), and reports the drop rate + routing
statistics the reward-outcome spec asks to carry into training_metadata.json.

This file lives under scripts/ and writes only under outputs/. It IMPORTS the
frozen probe contract from dagspaces/grpo_training/stages/probes.py (never
reimplements it) and the embedding recipe from reward_prep / norm_universe, but
modifies nothing under dagspaces/.

Pipeline (single process, 2 GPUs):
  1. Load teacher reference flows; build per-chunk flow dicts (ci_* -> contract
     keys).
  2. Embed every flow query with Qwen3-Embedding-8B (EXACTLY as reward_prep
     does: EMBED_INSTRUCTION prefix, normalize_embeddings=True, cuda:0,
     padding_side=left). Free the embed model.
  3. Per chunk, build the candidate probe pool via probes.build_probe_pool with
     an own-book cosine retriever (top-k=3). Write probe_pools.parquet + report.
  4. Vote UNIQUE probes (dedupe by probe_id) against Qwen3.6-27B (offline vLLM,
     TP=2, enable_thinking=false), empty extraction, n=5 @ T=1.0, max_tokens=64.
     Write null_answerability.parquet.
  5. apply_null_filter -> probe_pools_filtered.parquet; compute post-filter
     routing stats via probes.sample_probes. Write calibration_report.md +
     calibration_stats.json.

Usage:
  python -m scripts.run_probe_calibration              # full run (GPU)
  python -m scripts.run_probe_calibration --build-only # phases 1-3 only (CPU+1GPU)
  python -m scripts.run_probe_calibration --report-only # re-aggregate from parquets
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical inputs (the built fiction10-gemma4 artifacts).
# ---------------------------------------------------------------------------

REPO = "/share/pierson/matt/UAIR"

FLOWS_PATH = (
    "/share/pierson/matt/UAIR/outputs/2026-07-12_fiction10_flows_gemma4/"
    "23-14-17/COLM_flows_fiction_gemma4/outputs/ci_extraction/ci_flows.parquet"
)
UNIVERSE_DIR = (
    "/share/pierson/matt/UAIR/multirun/2026-07-23_universe_fiction10_gemma4/"
    "15-43-41/norm_universe_only/outputs/norm_universe"
)
UNIVERSE_PATH = os.path.join(UNIVERSE_DIR, "norm_universes.json")
EMB_DIR = os.path.join(UNIVERSE_DIR, "embeddings")

EMBED_MODEL_PATH = "/share/pierson/matt/zoo/models/Qwen3-Embedding-8B"

# Frozen answerer = Gemma-4-31B-it (Matt override 2026-07-23; reward-outcome.md
# D1's Qwen3.6-27B is stale — the teacher/judge for everything going forward is
# gemma-4-31b). Config-driven: read the canonical model yaml for model_source,
# engine kwargs, tensor_parallel_size, trust_remote_code, chat-template settings.
ANSWERER_MODEL_YAML = os.path.join(
    REPO, "dagspaces/common/conf/model/gemma-4-31b/instruct.yaml"
)
ANSWERER_NAME = "gemma-4-31b"

OUTPUT_DIR = (
    "/share/pierson/matt/UAIR/outputs/2026-07-23_mseries_premeasure/"
    "probe_calibration"
)

# Retrieval / probe params (mirror R-GROUND: top-k=3).
TOP_K = 3
# Force-stratified sample cap (D2).
K_MAX = 4
# Null-answerability drop threshold (reward-outcome default p_null=0.8).
P_NULL = 0.8
# Vote budget per unique probe.
N_VOTES = 5
VOTE_TEMPERATURE = 1.0
# ~64 per spec; bumped to 128 as truncation insurance (gemma wraps the answer in
# <|channel> markers, and the documented failure mode is max_tokens truncation
# masquerading as failure). vLLM stops at EOS (~25 tok actual), so cost is ~zero.
VOTE_MAX_TOKENS = 128
VOTE_SEED = 0

# ci_flows column -> contract flow key.
FLOW_COL_MAP = {
    "subject": "ci_subject",
    "sender": "ci_sender",
    "recipient": "ci_recipient",
    "information_type": "ci_information_type",
    "transmission_principle": "ci_transmission_principle",
    "context": "ci_context",
}

# Answerer call shape (VERBATIM from reward-outcome.md "What the answerer sees").
ANSWERER_SYSTEM = (
    "You answer questions using ONLY the structured information-flow extraction "
    "provided. If the extraction does not determine an answer, reply "
    '"cannot_determine".'
)


# ---------------------------------------------------------------------------
# Phase 1 — reference flows.
# ---------------------------------------------------------------------------


def _parse_norms_invoked(raw: Any) -> list:
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    if isinstance(raw, np.ndarray):
        return list(raw)
    s = str(raw).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        return list(obj) if isinstance(obj, list) else [str(obj)]
    except (json.JSONDecodeError, ValueError):
        return [s]


def load_chunk_flows() -> dict[tuple[str, int], list[dict]]:
    """Group teacher reference flows by (gutenberg_id, chunk_id).

    Returns {(gutenberg_id, chunk_id): [flow_dict, ...]} where each flow_dict
    carries the exact keys probes.flow_to_query consumes.
    """
    df = pd.read_parquet(FLOWS_PATH)
    print(f"[flows] loaded {len(df)} flow rows, "
          f"{df.groupby(['gutenberg_id', 'chunk_id']).ngroups} chunks", flush=True)

    chunk_flows: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in df.to_dict("records"):
        gid = str(rec["gutenberg_id"])
        cid = int(rec["chunk_id"])
        flow: dict[str, Any] = {}
        for key, col in FLOW_COL_MAP.items():
            val = rec.get(col)
            flow[key] = "" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val)
        flow["norms_invoked"] = _parse_norms_invoked(rec.get("ci_norms_invoked"))
        chunk_flows[(gid, cid)].append(flow)
    return chunk_flows


# ---------------------------------------------------------------------------
# Phase 2 — universe + embeddings.
# ---------------------------------------------------------------------------


def load_universe() -> tuple[dict[str, list], dict[str, np.ndarray]]:
    """Load per-book norm universes (with gutenberg_id injected) + embeddings.

    gutenberg_id is injected into every norm dict so probes.probe_id() (which
    reads norm["gutenberg_id"]) produces book-scoped ids. Verifies per-book
    npy row count == len(universe[book]) and that embeddings are L2-normalized.
    """
    with open(UNIVERSE_PATH, encoding="utf-8") as f:
        universe = json.load(f)

    emb_by_book: dict[str, np.ndarray] = {}
    for book in sorted(universe.keys()):
        norms = universe[book]
        # Inject book id for book-scoped probe_id (probes.probe_id reads it).
        for n in norms:
            n["gutenberg_id"] = book
        npy = np.load(os.path.join(EMB_DIR, f"{book}.npy"))
        if npy.shape[0] != len(norms):
            raise RuntimeError(
                f"[universe] ALIGNMENT FAIL {book}: npy {npy.shape[0]} rows "
                f"vs universe {len(norms)} norms"
            )
        emb_by_book[book] = npy.astype(np.float32)

    # L2-normalization spot check on the concatenation.
    sample = np.concatenate([emb_by_book[b][:5] for b in emb_by_book], axis=0)
    l2 = np.linalg.norm(sample, axis=1)
    if not np.allclose(l2, 1.0, atol=1e-2):
        print(f"[universe] WARNING: embeddings not unit-norm "
              f"(min {l2.min():.4f} max {l2.max():.4f}); normalizing.", flush=True)
        for b in emb_by_book:
            m = emb_by_book[b]
            emb_by_book[b] = m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-8, None)
    else:
        print(f"[universe] embeddings unit-norm OK "
              f"(min {l2.min():.5f} max {l2.max():.5f})", flush=True)

    total = sum(len(v) for v in universe.values())
    print(f"[universe] {len(universe)} books, {total} norms, all aligned", flush=True)
    return universe, emb_by_book


def embed_flow_queries(unique_queries: list[str]) -> dict[str, np.ndarray]:
    """Embed flow-query strings with Qwen3-Embedding-8B (reward_prep recipe)."""
    from dagspaces.grpo_training.stages.norm_universe import EMBED_INSTRUCTION
    from dagspaces.common.stage_utils import ensure_importable_sentence_transformers

    ensure_importable_sentence_transformers()
    from sentence_transformers import SentenceTransformer

    print(f"[embed] loading {EMBED_MODEL_PATH}", flush=True)
    model = SentenceTransformer(
        EMBED_MODEL_PATH,
        device="cuda:0",
        tokenizer_kwargs={"padding_side": "left"},
    )
    print(f"[embed] encoding {len(unique_queries)} unique flow queries", flush=True)
    prefixed = [EMBED_INSTRUCTION + q for q in unique_queries]
    t0 = time.time()
    embs = model.encode(
        prefixed,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embs = np.asarray(embs, dtype=np.float32)
    print(f"[embed] shape {embs.shape} in {time.time()-t0:.1f}s", flush=True)

    cache = {q: embs[i] for i, q in enumerate(unique_queries)}

    # Free GPU before the answerer loads.
    del model
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


# ---------------------------------------------------------------------------
# Phase 3 — build candidate pools.
# ---------------------------------------------------------------------------


def build_pools(
    chunk_flows: dict[tuple[str, int], list[dict]],
    universe: dict[str, list],
    emb_by_book: dict[str, np.ndarray],
    query_cache: dict[str, np.ndarray],
) -> tuple[list[dict], dict, dict[tuple[str, int], list[dict]]]:
    """Build per-chunk candidate probe pools via probes.build_probe_pool.

    Returns (probe_rows, report, pools_by_chunk) where pools_by_chunk holds the
    full probe dicts (needed for later filtering / sampling).
    """
    from dagspaces.grpo_training.stages.probes import (
        build_probe_pool_with_stats,
        flow_to_query,
    )

    probe_rows: list[dict] = []
    pools_by_chunk: dict[tuple[str, int], list[dict]] = {}
    pool_sizes: list[int] = []
    empty_pool_chunks = 0
    total_leak_skipped = 0
    chunks_missing_book = 0

    for (gid, cid), flows in chunk_flows.items():
        if gid not in universe:
            chunks_missing_book += 1
            continue
        book_norms = universe[gid]
        book_emb = emb_by_book[gid]

        def retrieve_top_k(query: str, k: int, _emb=book_emb) -> list[int]:
            qv = query_cache.get(query)
            if qv is None:
                raise KeyError(
                    f"[build] query not in embedding cache (len={len(query)}): "
                    f"{query[:80]!r}"
                )
            sims = _emb @ qv
            return list(np.argsort(sims)[-k:][::-1])

        pool, stats = build_probe_pool_with_stats(
            flows, book_norms, retrieve_top_k, k=TOP_K
        )
        total_leak_skipped += int(stats.get("n_leak_skipped", 0))
        pool_sizes.append(len(pool))
        if not pool:
            empty_pool_chunks += 1
            continue  # empty pools carry no rows and are absent from the
                      # parquet; their count lives in the build report so the
                      # full-run and --votes-only paths agree on empty counts.
        pools_by_chunk[(gid, cid)] = pool

        for p in pool:
            probe_rows.append({
                "gutenberg_id": gid,
                "chunk_id": cid,
                "probe_id": p["probe_id"],
                "norm_index": p["norm_index"],
                "gold": p["gold"],
                "prompt_text": p["prompt_text"],
            })

    sizes = np.array(pool_sizes) if pool_sizes else np.array([0])
    n_chunks = len(chunk_flows)
    report = {
        "n_chunks_with_reference_flows": n_chunks,
        "n_chunks_missing_book_universe": chunks_missing_book,
        "n_chunks_empty_candidate_pool": empty_pool_chunks,
        "frac_chunks_empty_candidate_pool": empty_pool_chunks / max(n_chunks, 1),
        "total_leak_skipped_norms": total_leak_skipped,
        "pool_size_distribution": {
            "min": int(sizes.min()),
            "p25": float(np.percentile(sizes, 25)),
            "median": float(np.median(sizes)),
            "mean": float(sizes.mean()),
            "p75": float(np.percentile(sizes, 75)),
            "max": int(sizes.max()),
        },
        "pool_size_histogram": dict(sorted(Counter(pool_sizes).items())),
        "n_candidate_probe_rows": len(probe_rows),
    }
    print(f"[build] {len(probe_rows)} candidate probe rows across {n_chunks} chunks; "
          f"{empty_pool_chunks} empty pools; {total_leak_skipped} leak-skipped", flush=True)
    return probe_rows, report, pools_by_chunk


# ---------------------------------------------------------------------------
# Phase 4 — null-answerability voting.
# ---------------------------------------------------------------------------


def unique_probes(probe_rows: list[dict]) -> pd.DataFrame:
    """Dedupe candidate probes by probe_id (null-answerability is a property of
    the probe text, so vote once per unique probe)."""
    seen: dict[str, dict] = {}
    for r in probe_rows:
        pid = r["probe_id"]
        if pid not in seen:
            seen[pid] = {
                "probe_id": pid,
                "gutenberg_id": r["gutenberg_id"],
                "gold": r["gold"],
                "prompt_text": r["prompt_text"],
            }
    return pd.DataFrame(list(seen.values()))


def _parse_vote(text: str) -> str:
    """Parse a single answerer completion into yes|no|cannot_determine.

    Primary path: the ``{"answers": [...]}`` JSON. Fallback (for models that
    answer in prose or wrap the JSON in channel markers that defeat extraction):
    scan the raw completion for a standalone answer token — ``cannot_determine``
    first (most specific), then a word-boundary yes/no. This closes the
    documented masquerade where a genuine yes/no is misparsed as
    cannot_determine. Truly empty / unparseable -> cannot_determine.
    """
    import re as _re

    from dagspaces.common.json_extraction import extract_json_from_text

    obj, _err = extract_json_from_text(text or "", repair=True)
    ans = None
    if isinstance(obj, dict):
        arr = obj.get("answers")
        if isinstance(arr, list) and arr:
            ans = arr[0]
        elif "answer" in obj:
            ans = obj.get("answer")
    if ans is not None:
        v = str(ans).strip().lower()
        if v in ("yes", "no", "cannot_determine"):
            return v
        if v.startswith("cannot"):
            return "cannot_determine"
        if v.startswith("yes"):
            return "yes"
        if v.startswith("no"):
            return "no"

    # Prose / channel-wrapped fallback: look at the completion text directly.
    low = str(text or "").lower()
    if "cannot_determine" in low or "cannot determine" in low:
        return "cannot_determine"
    if _re.search(r"\byes\b", low) and not _re.search(r"\bno\b", low):
        return "yes"
    if _re.search(r"\bno\b", low) and not _re.search(r"\byes\b", low):
        return "no"
    return "cannot_determine"


def _load_answerer_engine_kwargs() -> tuple[str, dict, dict]:
    """Read the gemma-4-31b answerer yaml -> (model_source, engine_kwargs, ctk).

    Faithful to the canonical model config (model_source, tensor_parallel_size,
    trust_remote_code, chat_template_kwargs). max_model_len is trimmed to 4096
    and max_num_seqs raised for throughput: the probes are tiny (~200 tok), so
    the config's 16384/8 (tuned for KV-hungry long-context judging) needlessly
    caps concurrency here.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ANSWERER_MODEL_YAML)
    m = cfg.model
    model_source = str(m.model_source)
    ek_raw = OmegaConf.select(m, "engine_kwargs")
    ek_cfg = OmegaConf.to_container(ek_raw, resolve=True) if ek_raw is not None else {}
    engine_kwargs = {
        "tensor_parallel_size": int(ek_cfg.get("tensor_parallel_size", 2)),
        "trust_remote_code": bool(ek_cfg.get("trust_remote_code", True)),
        "max_model_len": 4096,
        # Coordinator directive (2026-07-24): 16 (not throughput-tuned 64) and
        # 0.95 util — canonical-models.md notes the 31B wants a high fraction;
        # 0.90 left KV tight (11.71 GiB in run 298149).
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.95,
        "seed": VOTE_SEED,
        # Required for TP>1 on pierson's PCIe A6000s: the custom all-reduce
        # kernel fails there ("custom_all_reduce.cuh:455 'invalid argument'",
        # crash in run 298149). These two are exactly what
        # dagspaces.common.vllm_inference sets for TP>1 — the direct LLM() path
        # here otherwise bypasses them.
        "distributed_executor_backend": "mp",
        "disable_custom_all_reduce": True,
        # Skip torch.compile / CUDA-graph capture: run 298149 spent ~75 min
        # compiling this Gemma4 graph (TRITON_ATTN, heterogeneous head dims)
        # before the all-reduce crash. The vote workload is ~6k tiny generations
        # that gain nothing from graph capture, so eager is both faster to start
        # and more robust here.
        "enforce_eager": True,
    }
    ctk_raw = OmegaConf.select(m, "chat_template_kwargs")
    ctk = dict(OmegaConf.to_container(ctk_raw, resolve=True)) if ctk_raw is not None else {}
    # thinking_mode: off -> no reasoning trace (template defaults enable_thinking false).
    ctk.setdefault("enable_thinking", False)
    return model_source, engine_kwargs, ctk


def run_votes(uniq: pd.DataFrame) -> pd.DataFrame:
    """Vote each unique probe against the frozen answerer with empty extraction."""
    # NCCL / vLLM runtime env BEFORE importing vLLM (mirror vllm_inference).
    from dagspaces.common.vllm_inference import (
        get_pcie_nccl_env_vars,
        get_vllm_runtime_env_vars,
    )
    for k, v in {**get_pcie_nccl_env_vars(), **get_vllm_runtime_env_vars()}.items():
        os.environ.setdefault(k, v)

    from vllm import LLM, SamplingParams

    from dagspaces.common.json_extraction import extract_json_from_text

    model_source, engine_kwargs, ctk = _load_answerer_engine_kwargs()
    print(f"[vote] loading answerer {ANSWERER_NAME}: {model_source} "
          f"engine_kwargs={engine_kwargs} chat_template_kwargs={ctk}", flush=True)
    llm = LLM(model=model_source, **engine_kwargs)

    sampling = SamplingParams(
        n=N_VOTES,
        temperature=VOTE_TEMPERATURE,
        top_p=1.0,
        max_tokens=VOTE_MAX_TOKENS,
        seed=VOTE_SEED,
    )

    conversations = []
    for row in uniq.to_dict("records"):
        user = (
            'EXTRACTION: {"flows": []}\n'
            f'Q1: {row["prompt_text"]}\n'
            'Reply as JSON: {"answers": ["yes"|"no"|"cannot_determine"]}'
        )
        conversations.append([
            {"role": "system", "content": ANSWERER_SYSTEM},
            {"role": "user", "content": user},
        ])

    print(f"[vote] generating {len(conversations)} probes x {N_VOTES} votes "
          f"@ T={VOTE_TEMPERATURE}", flush=True)
    t0 = time.time()
    outputs = llm.chat(
        conversations,
        sampling,
        chat_template_kwargs=ctk,
    )
    dt = time.time() - t0
    print(f"[vote] done in {dt:.1f}s ({len(conversations)/max(dt,1e-9):.1f} probes/s)",
          flush=True)

    records = uniq.to_dict("records")
    out_rows = []
    # Raw-parse diagnostic: without saving raw text, a 100%-cannot_determine
    # result is ambiguous between genuine faithfulness and a parse artifact
    # (the documented masquerade). Capture raw completions + count how many
    # actually carried extractable JSON.
    n_json_ok = 0
    n_raw = 0
    example_raw: list[str] = []
    for row, out in zip(records, outputs):
        raw_texts = [o.text for o in out.outputs]
        votes = [_parse_vote(t) for t in raw_texts]
        for t in raw_texts:
            n_raw += 1
            obj, _e = extract_json_from_text(t or "", repair=True)
            if isinstance(obj, dict) and ("answers" in obj or "answer" in obj):
                n_json_ok += 1
        if len(example_raw) < 12:
            example_raw.append(raw_texts[0])
        # Pad defensively if the engine returned < N_VOTES.
        while len(votes) < N_VOTES:
            votes.append("cannot_determine")
        gold = row["gold"]
        n_correct = sum(1 for v in votes if v == gold)
        correct_frac = n_correct / N_VOTES
        out_rows.append({
            "probe_id": row["probe_id"],
            "gutenberg_id": row["gutenberg_id"],
            "gold": gold,
            "prompt_text": row["prompt_text"],
            "votes": votes,
            "raw_texts": raw_texts,
            "correct_frac": correct_frac,
            "dropped": bool(correct_frac >= P_NULL),
        })

    json_ok_frac = n_json_ok / max(n_raw, 1)
    print(f"[vote] raw-parse diagnostic: {n_json_ok}/{n_raw} completions carried "
          f"extractable answers JSON ({json_ok_frac:.3f})", flush=True)
    print("[vote] example raw completions (first 3):", flush=True)
    for ex in example_raw[:3]:
        print(f"  RAW: {ex[:200]!r}", flush=True)

    del llm
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(out_rows), dt


# ---------------------------------------------------------------------------
# Phase 5 — filter + routing stats + report.
# ---------------------------------------------------------------------------


def compute_stats(
    pools_by_chunk: dict[tuple[str, int], list[dict]],
    null_df: pd.DataFrame,
    build_report: dict,
    vote_seconds: float | None,
) -> tuple[dict, list[dict]]:
    """Apply the null filter and compute post-filter routing statistics."""
    from dagspaces.grpo_training.stages.probes import apply_null_filter, sample_probes

    null_correct_frac = dict(zip(null_df["probe_id"], null_df["correct_frac"]))

    # ---- Null-drop headline (overall + by gold class) ----
    n_unique = len(null_df)
    n_dropped = int(null_df["dropped"].sum())
    by_gold_total = null_df.groupby("gold").size().to_dict()
    by_gold_dropped = null_df[null_df["dropped"]].groupby("gold").size().to_dict()
    drop_rate_by_gold = {
        g: (by_gold_dropped.get(g, 0) / by_gold_total.get(g, 1))
        for g in by_gold_total
    }

    # ---- Empty-extraction answerer behaviour (sanity) ----
    all_votes = [v for votes in null_df["votes"] for v in votes]
    vote_counter = Counter(all_votes)
    n_all_votes = max(len(all_votes), 1)
    cannot_determine_base_rate = vote_counter.get("cannot_determine", 0) / n_all_votes
    yes_rate = vote_counter.get("yes", 0) / n_all_votes
    no_rate = vote_counter.get("no", 0) / n_all_votes

    # ---- Filter the pools ----
    # pools_by_chunk holds only NON-empty candidate pools (empty ones carry no
    # rows and live only in the build report). So the pre-filter empty count is
    # authoritative from the build report, and the post-filter empty count is
    # those already-empty chunks PLUS non-empty pools that the filter empties.
    filtered_rows: list[dict] = []
    pre_empty = int(build_report.get("n_chunks_empty_candidate_pool", 0))
    newly_empty_post_filter = 0
    realized_K: list[int] = []
    n_sets_both_classes = 0
    n_sets_with_gold_no = 0
    n_pools_with_both_classes = 0
    n_pools_with_gold_no = 0

    for (gid, cid), pool in pools_by_chunk.items():
        filtered = apply_null_filter(pool, null_correct_frac, p_null=P_NULL)
        if not filtered:
            newly_empty_post_filter += 1
            continue

        for p in filtered:
            filtered_rows.append({
                "gutenberg_id": gid,
                "chunk_id": cid,
                "probe_id": p["probe_id"],
                "norm_index": p["norm_index"],
                "gold": p["gold"],
                "prompt_text": p["prompt_text"],
            })

        # Realized K = min(4, |filtered|) via the frozen sampler.
        chunk_key = f"{gid}:{cid}"
        sampled = sample_probes(filtered, chunk_key, k_max=K_MAX)
        realized_K.append(len(sampled))

        golds_pool = {p["gold"] for p in filtered}
        golds_sample = {p["gold"] for p in sampled}
        if golds_pool == {"yes", "no"}:
            n_pools_with_both_classes += 1
            if golds_sample == {"yes", "no"}:
                n_sets_both_classes += 1
        if "no" in golds_pool:
            n_pools_with_gold_no += 1
            if "no" in golds_sample:
                n_sets_with_gold_no += 1

    n_nonempty_post = len(realized_K)
    post_empty = pre_empty + newly_empty_post_filter
    total_chunks = int(build_report.get("n_chunks_with_reference_flows",
                                        len(pools_by_chunk) + pre_empty))
    stats = {
        "params": {
            "top_k_retrieval": TOP_K,
            "k_max_sample": K_MAX,
            "p_null": P_NULL,
            "n_votes": N_VOTES,
            "vote_temperature": VOTE_TEMPERATURE,
            "vote_max_tokens": VOTE_MAX_TOKENS,
            "vote_seed": VOTE_SEED,
            "answerer": ANSWERER_NAME,
            "answerer_yaml": ANSWERER_MODEL_YAML,
            "embed_model": EMBED_MODEL_PATH,
        },
        "build": build_report,
        "null_answerability": {
            "n_unique_probes": n_unique,
            "n_dropped": n_dropped,
            "drop_rate_overall": n_dropped / max(n_unique, 1),
            "unique_by_gold": {k: int(v) for k, v in by_gold_total.items()},
            "dropped_by_gold": {k: int(by_gold_dropped.get(k, 0)) for k in by_gold_total},
            "drop_rate_by_gold": drop_rate_by_gold,
            "correct_frac_histogram": dict(sorted(
                Counter([round(f, 2) for f in null_df["correct_frac"]]).items()
            )),
        },
        "empty_extraction_answerer": {
            "total_votes": len(all_votes),
            "cannot_determine_base_rate": cannot_determine_base_rate,
            "yes_rate": yes_rate,
            "no_rate": no_rate,
            "yes_no_bias_ratio": (yes_rate / no_rate) if no_rate else float("inf"),
        },
        "routing_post_filter": {
            "empty_pool_chunks_pre_filter": pre_empty,
            "empty_pool_chunks_post_filter": post_empty,
            "newly_emptied_by_null_filter": newly_empty_post_filter,
            "frac_chunks_excluded_post_filter": post_empty / max(total_chunks, 1),
            "n_chunks_retained_post_filter": n_nonempty_post,
            "realized_K_histogram": dict(sorted(Counter(realized_K).items())),
            "realized_K_mean": float(np.mean(realized_K)) if realized_K else 0.0,
            "n_filtered_probe_rows": len(filtered_rows),
            "pools_with_both_gold_classes": n_pools_with_both_classes,
            "sampled_sets_with_both_gold_classes": n_sets_both_classes,
            "frac_both_classes_sampled_when_available": (
                n_sets_both_classes / n_pools_with_both_classes
                if n_pools_with_both_classes else None
            ),
            "pools_with_gold_no": n_pools_with_gold_no,
            "sampled_sets_with_gold_no": n_sets_with_gold_no,
            "frac_gold_no_sampled_when_available": (
                n_sets_with_gold_no / n_pools_with_gold_no
                if n_pools_with_gold_no else None
            ),
        },
        "runtime": {
            "vote_seconds": vote_seconds,
        },
    }
    return stats, filtered_rows


def write_report(stats: dict) -> str:
    na = stats["null_answerability"]
    ee = stats["empty_extraction_answerer"]
    rt = stats["routing_post_filter"]
    bd = stats["build"]

    def pct(x):
        return f"{100*x:.2f}%" if x is not None else "n/a"

    lines: list[str] = []
    lines.append("# R-OUTCOME probe pool — null-answerability calibration\n")
    lines.append("_m-series data prerequisite job 4 "
                 "(wiki/grpo_redesign/reward-outcome.md step 5)._\n")
    lines.append(f"- Reference flows: `{FLOWS_PATH}`")
    lines.append(f"- Universe + embeddings: `{UNIVERSE_DIR}`")
    lines.append(f"- Answerer: **{ANSWERER_NAME}** (`{ANSWERER_MODEL_YAML}`, TP=2, "
                 "enable_thinking=false) — Matt override 2026-07-23; reward-outcome.md "
                 "D1's Qwen3.6-27B is stale")
    lines.append(f"- Embedder: `{EMBED_MODEL_PATH}` (EMBED_INSTRUCTION, unit-norm cosine)")
    lines.append(f"- Params: top-k={TOP_K}, K_max={K_MAX}, p_null={P_NULL}, "
                 f"n_votes={N_VOTES} @ T={VOTE_TEMPERATURE}, max_tokens={VOTE_MAX_TOKENS}\n")

    lines.append("## HEADLINE — null-answerability drop rate\n")
    lines.append("> Report this drop rate in `training_metadata.json`.\n")
    lines.append(f"- **Overall: {na['n_dropped']}/{na['n_unique_probes']} = "
                 f"{pct(na['drop_rate_overall'])}** of unique probes dropped "
                 "(answerer scores >= p_null correct on EMPTY extraction).")
    for g in sorted(na["drop_rate_by_gold"]):
        lines.append(f"  - gold **{g}**: {na['dropped_by_gold'].get(g,0)}/"
                     f"{na['unique_by_gold'].get(g,0)} = {pct(na['drop_rate_by_gold'][g])}")
    lines.append("")

    lines.append("## Empty-extraction answerer sanity\n")
    lines.append(f"- `cannot_determine` base rate: **{pct(ee['cannot_determine_base_rate'])}** "
                 f"of all {ee['total_votes']} votes (should dominate — the extraction "
                 "is empty, so a faithful answerer cannot decide).")
    lines.append(f"- yes rate {pct(ee['yes_rate'])} · no rate {pct(ee['no_rate'])} "
                 f"· yes/no bias ratio {ee['yes_no_bias_ratio']:.2f}\n")

    lines.append("## Candidate pool (pre-filter)\n")
    lines.append(f"- chunks with reference flows: {bd['n_chunks_with_reference_flows']}")
    lines.append(f"- chunks with EMPTY candidate pool: {bd['n_chunks_empty_candidate_pool']} "
                 f"({pct(bd['frac_chunks_empty_candidate_pool'])})")
    lines.append(f"- leak-skipped norms (probe_leaks): {bd['total_leak_skipped_norms']}")
    lines.append(f"- candidate probe rows: {bd['n_candidate_probe_rows']}")
    psd = bd["pool_size_distribution"]
    lines.append(f"- pool size: min {psd['min']} · p25 {psd['p25']:.0f} · median "
                 f"{psd['median']:.0f} · mean {psd['mean']:.2f} · p75 {psd['p75']:.0f} "
                 f"· max {psd['max']}\n")

    lines.append("## Routing (post-filter)\n")
    lines.append(f"- empty-pool chunks PRE filter: {rt['empty_pool_chunks_pre_filter']}")
    lines.append(f"- empty-pool chunks POST filter (excluded from T-EXTRACT): "
                 f"**{rt['empty_pool_chunks_post_filter']}** "
                 f"({pct(rt['frac_chunks_excluded_post_filter'])})")
    lines.append(f"- chunks retained post-filter: {rt['n_chunks_retained_post_filter']}")
    lines.append(f"- realized K=min(4,|pool|) histogram: {rt['realized_K_histogram']} "
                 f"(mean {rt['realized_K_mean']:.2f})")
    lines.append(f"- both gold classes sampled when available: "
                 f"{rt['sampled_sets_with_both_gold_classes']}/"
                 f"{rt['pools_with_both_gold_classes']} "
                 f"({pct(rt['frac_both_classes_sampled_when_available'])})")
    lines.append(f"- >=1 gold-no sampled when available: "
                 f"{rt['sampled_sets_with_gold_no']}/{rt['pools_with_gold_no']} "
                 f"({pct(rt['frac_gold_no_sampled_when_available'])})\n")

    vs = stats["runtime"].get("vote_seconds")
    if vs is not None:
        lines.append("## Runtime / cost\n")
        n_uni = na["n_unique_probes"]
        lines.append(f"- vote pass: {vs:.1f}s for {n_uni} unique probes x {N_VOTES} "
                     f"votes = {n_uni*N_VOTES} generations "
                     f"({n_uni*N_VOTES/max(vs,1e-9):.1f} gen/s)\n")

    report = "\n".join(lines)
    path = os.path.join(OUTPUT_DIR, "calibration_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def _reconstruct_from_parquet(
    probe_df: pd.DataFrame, universe: dict[str, list]
) -> tuple[list[dict], dict[tuple[str, int], list[dict]]]:
    """Rebuild (probe_rows, pools_by_chunk) from an on-disk probe_pools parquet.

    Lets --votes-only / --report-only reuse a completed build phase (the
    embedding + retrieval are deterministic, so the parquet is authoritative).
    """
    probe_rows: list[dict] = []
    pools_by_chunk: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in probe_df.to_dict("records"):
        gid = str(r["gutenberg_id"])
        cid = int(r["chunk_id"])
        idx = int(r["norm_index"])
        row = {
            "gutenberg_id": gid,
            "chunk_id": cid,
            "probe_id": r["probe_id"],
            "norm_index": idx,
            "gold": r["gold"],
            "prompt_text": r["prompt_text"],
        }
        probe_rows.append(row)
        pools_by_chunk[(gid, cid)].append({
            "probe_id": r["probe_id"],
            "norm_index": idx,
            "norm": universe[gid][idx],
            "gold": r["gold"],
            "prompt_text": r["prompt_text"],
        })
    # Preserve chunks whose candidate pool was empty at build (they carry no
    # rows in the parquet); the pre-filter empty-pool count is taken from the
    # build report, so pools_by_chunk here holds only non-empty chunks — which
    # is exactly what the post-filter routing stats iterate over.
    return probe_rows, pools_by_chunk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true",
                    help="phases 1-3 only (pools + embeddings, no answerer)")
    ap.add_argument("--votes-only", action="store_true",
                    help="reuse on-disk probe_pools.parquet; run votes + stats "
                         "(skips embed/build — for relaunch after an answerer crash)")
    ap.add_argument("--report-only", action="store_true",
                    help="re-aggregate stats from existing parquets")
    args = ap.parse_args()

    try:
        from dagspaces.common.stage_utils import ensure_dotenv
        ensure_dotenv()
    except Exception as exc:
        print(f"WARNING: ensure_dotenv failed: {exc}", file=sys.stderr)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pools_path = os.path.join(OUTPUT_DIR, "probe_pools.parquet")
    null_path = os.path.join(OUTPUT_DIR, "null_answerability.parquet")
    filtered_path = os.path.join(OUTPUT_DIR, "probe_pools_filtered.parquet")
    build_report_path = os.path.join(OUTPUT_DIR, "_build_report.json")

    def _finish(pools_by_chunk, null_df, build_report, vote_seconds, tag):
        stats, filtered_rows = compute_stats(
            pools_by_chunk, null_df, build_report, vote_seconds
        )
        pd.DataFrame(filtered_rows).to_parquet(filtered_path)
        with open(os.path.join(OUTPUT_DIR, "calibration_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        rp = write_report(stats)
        print(f"[{tag}] wrote {filtered_path}", flush=True)
        print(f"[{tag}] wrote {rp}", flush=True)
        print(json.dumps(stats["null_answerability"], indent=2), flush=True)
        print(json.dumps(stats["routing_post_filter"], indent=2), flush=True)
        return stats

    if args.report_only:
        with open(build_report_path, encoding="utf-8") as f:
            build_report = json.load(f)
        universe, _ = load_universe()
        probe_df = pd.read_parquet(pools_path)
        null_df = pd.read_parquet(null_path)
        _, pools_by_chunk = _reconstruct_from_parquet(probe_df, universe)
        _finish(pools_by_chunk, null_df, build_report, None, "report-only")
        return

    if args.votes_only:
        # Reuse the completed build phase (probe_pools.parquet is deterministic);
        # run only the answerer votes + stats. For relaunch after an answerer
        # crash without paying the ~4-min re-embed.
        with open(build_report_path, encoding="utf-8") as f:
            build_report = json.load(f)
        universe, _ = load_universe()
        probe_df = pd.read_parquet(pools_path)
        probe_rows, pools_by_chunk = _reconstruct_from_parquet(probe_df, universe)
        uniq = unique_probes(probe_rows)
        print(f"[votes-only] {len(uniq)} unique probes to vote "
              f"(reused {len(probe_rows)} candidate rows from {pools_path})",
              flush=True)
        null_df, vote_seconds = run_votes(uniq)
        null_df.to_parquet(null_path)
        print(f"[votes-only] wrote {null_path}", flush=True)
        _finish(pools_by_chunk, null_df, build_report, vote_seconds, "votes-only")
        return

    # ---- Phases 1-3 ----
    chunk_flows = load_chunk_flows()
    universe, emb_by_book = load_universe()

    # Unique flow queries.
    from dagspaces.grpo_training.stages.probes import flow_to_query
    all_queries = set()
    for flows in chunk_flows.values():
        for flow in flows:
            all_queries.add(flow_to_query(flow))
    unique_q = sorted(all_queries)
    print(f"[embed] {len(unique_q)} unique flow queries", flush=True)

    query_cache = embed_flow_queries(unique_q)

    probe_rows, build_report, pools_by_chunk = build_pools(
        chunk_flows, universe, emb_by_book, query_cache
    )
    pd.DataFrame(probe_rows).to_parquet(pools_path)
    with open(build_report_path, "w") as f:
        json.dump(build_report, f, indent=2)
    print(f"[build] wrote {pools_path}", flush=True)

    if args.build_only:
        print("[build-only] done (skipping answerer votes)", flush=True)
        return

    # ---- Phase 4: votes ----
    uniq = unique_probes(probe_rows)
    print(f"[vote] {len(uniq)} unique probes to vote", flush=True)
    null_df, vote_seconds = run_votes(uniq)
    null_df.to_parquet(null_path)
    print(f"[vote] wrote {null_path}", flush=True)

    # ---- Phase 5: filter + stats + report ----
    _finish(pools_by_chunk, null_df, build_report, vote_seconds, "done")


if __name__ == "__main__":
    main()
