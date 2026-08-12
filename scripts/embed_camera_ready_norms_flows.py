#!/usr/bin/env python3
"""Embed the Gemma-4 lineage fiction10 norms + flows for the camera-ready figures.

Supersedes the ad-hoc `scripts/embed_norms_and_flows.py` run that produced
`/share/pierson/matt/n2s4cir/data/fiction10/*_embeddings_qwen3emb.npy`. That
matrix was built from the **Qwen2.5-72B-AWQ era** extraction (and, per the
2026-07-12 prompt-wiring fix, under the wrong prompts); every camera-ready
figure has to come off the Gemma-4-31B-it lineage instead.

Two instruction variants are written, because the norm/flow *displacement*
analysis and the norm/flow *separation* analysis want different spaces:

``prod``
    Production parity. Norms carry the norm instruction, flows carry the flow
    instruction — exactly the asymmetric space R-GROUND retrieval lives in, and
    the space the existing separation/silhouette numbers were computed in.

``shared``
    One instruction for both constructs. Required for any claim about the
    *direction* of the norm-minus-flow offset: under ``prod`` a large, perfectly
    coherent component of every difference vector is contributed by the differing
    instruction prefixes alone, so a "systematic transformation" measured there
    would be an artifact of how we asked for the embedding, not a property of the
    extracted content.

Serialization is byte-identical to production (`norm_universe._build_norm_text`
for norms, the flow template used by `embed_norms_and_flows.flow_to_text`).
Exact-duplicate norms are dropped per book before embedding, matching the
methods section ("~0.5% of rows, predominantly same-chunk extractor repetition").

Usage:
    export EMBEDDING_SERVER_URL=http://klara.tech.cornell.edu:8001
    .venv-vllm025cu129/bin/python scripts/embed_camera_ready_norms_flows.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path("/share/pierson/matt/UAIR")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagspaces.grpo_training.stages.norm_universe import _build_norm_text  # noqa: E402

# ---------------------------------------------------------------------------
# Provenance — the same canonical Gemma-4 artifacts the other camera-ready
# notebooks read (see notebooks/colm-camera-ready/corpus_descriptives_two_corpora.py).
# ---------------------------------------------------------------------------
SOURCES = {
    "fiction10": {
        "norms": ROOT
        / "outputs/2026-07-12_fiction10_norms_gemma4/18-36-28"
        / "COLM_norms_fiction_gemma4/outputs/extraction/structured_norms.parquet",
        "flows": ROOT
        / "outputs/2026-07-12_fiction10_flows_gemma4/23-14-17"
        / "COLM_flows_fiction_gemma4/outputs/ci_extraction/ci_flows.parquet",
    },
    "top100": {
        "norms": ROOT
        / "outputs/2026-07-13_top100_norms_extraction_gemma4/16-23-09"
        / "COLM_norms_fiction_gemma4/outputs/extraction/structured_norms.parquet",
        "flows": ROOT
        / "outputs/2026-07-13_top100_flows_gemma4/16-23-09"
        / "COLM_flows_fiction_gemma4/outputs/ci_extraction/ci_flows.parquet",
    },
}

OUT_DIR = ROOT / "outputs/camera_ready/embeddings"

EMB_MODEL = "/share/pierson/matt/zoo/models/Qwen3-Embedding-8B"

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------
# `prod` norm instruction is verbatim `norm_universe.EMBED_INSTRUCTION`.
PROD_NORM_INSTRUCTION = (
    "Instruct: Given a prescriptive social norm from a literary text, "
    "represent it for semantic matching with information flows.\nQuery: "
)
PROD_FLOW_INSTRUCTION = (
    "Instruct: Given a contextual integrity information flow from a literary text, "
    "represent it for semantic comparison with other information flows.\nQuery: "
)

# One prefix for both constructs: the difference vector then reflects content.
SHARED_INSTRUCTION = (
    "Instruct: Given a statement extracted from a literary text describing either "
    "a social norm or an information flow, represent it for semantic comparison.\n"
    "Query: "
)

VARIANTS = {
    "prod": {"norm": PROD_NORM_INSTRUCTION, "flow": PROD_FLOW_INSTRUCTION},
    "shared": {"norm": SHARED_INSTRUCTION, "flow": SHARED_INSTRUCTION},
}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Flow text variants
# ---------------------------------------------------------------------------
# How much of the norm/flow separation is a fact about the extracted content,
# and how much is normative vocabulary we ourselves wrote into the flow string?
# Three serializations answer that empirically instead of by argument:
#
#   full         CI tuple + transmission principle + appropriateness verdict.
#                What `scripts/embed_norms_and_flows.py` produced, hence what
#                the paper's existing figure was built on. Carries an explicit
#                deontic judgment ("This flow is considered inappropriate").
#   noappr       Appropriateness dropped. This is production parity: both
#                `online_rground._flow_to_query` and
#                `DirectChunkGold.MATCH_FIELDS` build retrieval queries from
#                sender/recipient/information_type/context/transmission_
#                principle/subject and never from the appropriateness verdict.
#   descriptive  Transmission principle dropped as well. In CI the transmission
#                principle *is* the normative constraint on the flow, and the
#                extracted values are frankly deontic vocabulary — discretion,
#                social obligation, confidentiality, propriety, consent,
#                coercion. What remains is purely descriptive: who sent what
#                about whom to whom, in what context.
FLOW_TEXT_VARIANTS = ("full", "noappr", "descriptive")


def flow_to_text(row: dict, variant: str = "full") -> str:
    """CI flow row -> readable sentence. See FLOW_TEXT_VARIANTS."""
    if variant not in FLOW_TEXT_VARIANTS:
        raise ValueError(f"unknown flow text variant {variant!r}")

    ctx = row.get("ci_context") or "unknown"
    sender = row.get("ci_sender") or "unknown"
    recipient = row.get("ci_recipient") or "unknown"
    info_type = row.get("ci_information_type") or "unknown"
    tp = row.get("ci_transmission_principle") or "unknown"
    subject = row.get("ci_subject")
    approp = row.get("ci_appropriateness") or ""

    parts = [f"In a {ctx} context", f"{sender} sends {info_type}"]
    if subject and str(subject).strip():
        parts.append(f"about {subject}")
    parts.append(f"to {recipient}")
    if variant != "descriptive":
        parts.append(f"via {tp}")
    text = ", ".join(parts) + "."
    if variant == "full" and approp and str(approp).strip():
        text += f" This flow is considered {approp}."
    return text


def _dedup_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_frames(corpus: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load + filter norms and flows, attaching the serialized embedding text."""
    norms = pd.read_parquet(SOURCES[corpus]["norms"])
    flows = pd.read_parquet(SOURCES[corpus]["flows"])

    art = norms["raz_norm_articulation"]
    norms = norms[art.notna() & (art.astype(str).str.strip() != "")].reset_index(drop=True)
    norms["embed_text"] = [_build_norm_text(r) for r in norms.to_dict("records")]

    # Per-book exact-duplicate removal (methods §3).
    key = norms["gutenberg_id"].astype(str) + "||" + norms["embed_text"].map(_dedup_key)
    n_before = len(norms)
    norms = norms[~key.duplicated()].reset_index(drop=True)
    print(f"  [{corpus}] norms: {n_before} -> {len(norms)} after per-book exact dedup "
          f"({(n_before - len(norms)) / max(n_before, 1):.2%} dropped)")

    _recs = flows.to_dict("records")
    for _v in FLOW_TEXT_VARIANTS:
        flows[f"embed_text_{_v}"] = [flow_to_text(r, _v) for r in _recs]
    # `embed_text` stays the `full` string so the column name means the same
    # thing on both frames and older readers keep working.
    flows["embed_text"] = flows["embed_text_full"]
    print(f"  [{corpus}] flows: {len(flows)}")
    return norms, flows


# ---------------------------------------------------------------------------
# vLLM embedding client
# ---------------------------------------------------------------------------
def embed_batch(session, server_url, texts, model_name, timeout=300.0, max_retries=4):
    for attempt in range(max_retries):
        try:
            resp = session.post(
                f"{server_url}/v1/embeddings",
                json={"model": model_name, "input": texts},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            embs = np.array([d["embedding"] for d in data], dtype=np.float32)
            n = np.linalg.norm(embs, axis=1, keepdims=True)
            return embs / np.maximum(n, 1e-9)
        except Exception as e:  # noqa: BLE001 - retried, then re-raised
            if attempt == max_retries - 1:
                raise RuntimeError(f"embed_batch failed after {max_retries} attempts: {e}") from e
            wait = 2**attempt
            print(f"    attempt {attempt + 1} failed ({e}); retrying in {wait}s")
            time.sleep(wait)


def embed_all(session, server_url, texts, instruction, model_name, batch_size=32):
    prefixed = [instruction + t for t in texts]
    out, n = [], len(prefixed)
    t0 = time.time()
    for start in range(0, n, batch_size):
        out.append(embed_batch(session, server_url, prefixed[start:start + batch_size], model_name))
        done = min(start + batch_size, n)
        if done % (batch_size * 20) == 0 or done == n:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    {done}/{n} ({100 * done / n:.0f}%) — {rate:.0f}/s", flush=True)
    return np.vstack(out)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="fiction10", choices=sorted(SOURCES))
    ap.add_argument("--variants", nargs="+", default=["shared", "prod"], choices=sorted(VARIANTS))
    ap.add_argument("--flow-text-variants", nargs="+", default=list(FLOW_TEXT_VARIANTS),
                    choices=list(FLOW_TEXT_VARIANTS))
    ap.add_argument("--server-url", default=os.environ.get("EMBEDDING_SERVER_URL", ""))
    ap.add_argument("--model-name", default=EMB_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--force", action="store_true", help="re-embed even if the .npy exists")
    args = ap.parse_args()

    if not args.server_url:
        ap.error("no --server-url and EMBEDDING_SERVER_URL unset")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    print(f"Loading {args.corpus} ...")
    norms, flows = load_frames(args.corpus)

    # The row-aligned metadata every downstream figure joins against. Written
    # once, so a stale .npy can never be paired with a re-filtered frame.
    meta_cols_norm = [
        "gutenberg_id", "chunk_id", "book_title", "book_author", "raz_norm_articulation",
        "raz_norm_subject", "raz_norm_act", "raz_normative_force", "raz_norm_source",
        "raz_governs_info_flow", "raz_context", "raz_confidence_qual",
        "norm_quality_passed", "embed_text",
    ]
    meta_cols_flow = [
        "gutenberg_id", "chunk_id", "book_title", "book_author", "ci_subject", "ci_sender",
        "ci_recipient", "ci_information_type", "ci_transmission_principle", "ci_context",
        "ci_appropriateness", "ci_confidence_qual", "flow_quality_passed", "embed_text",
        *[f"embed_text_{_v}" for _v in FLOW_TEXT_VARIANTS],
    ]
    norms[[c for c in meta_cols_norm if c in norms.columns]].to_parquet(
        args.out_dir / f"{args.corpus}_norms_meta.parquet", index=False)
    flows[[c for c in meta_cols_flow if c in flows.columns]].to_parquet(
        args.out_dir / f"{args.corpus}_flows_meta.parquet", index=False)

    # (output stem, frame, text column, instruction key)
    jobs = [(f"{args.corpus}_norms", norms, "embed_text", "norm")]
    jobs += [
        (f"{args.corpus}_flows_{tv}", flows, f"embed_text_{tv}", "flow")
        for tv in args.flow_text_variants
    ]

    for variant in args.variants:
        instr = VARIANTS[variant]
        for stem, frame, text_col, instr_key in jobs:
            out = args.out_dir / f"{stem}_{variant}.npy"
            if out.exists() and not args.force:
                cached = np.load(out, mmap_mode="r")
                if cached.shape[0] == len(frame):
                    print(f"  cached {out.name} {cached.shape} — skipping")
                    continue
                print(f"  {out.name} has {cached.shape[0]} rows but frame has "
                      f"{len(frame)}; re-embedding")
            print(f"  embedding {len(frame)} rows -> {out.name} ...", flush=True)
            emb = embed_all(session, args.server_url, frame[text_col].tolist(),
                            instr[instr_key], args.model_name, args.batch_size)
            np.save(out, emb)
            print(f"  saved {out} {emb.shape}")

    session.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
