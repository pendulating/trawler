#!/usr/bin/env python3
"""Embed all norms and flows from fiction10 via a vLLM embedding server.

Reads abstracted_norms.parquet and ci_flows.parquet, serializes each row
into an embedding-friendly text string, and encodes via the vLLM OpenAI-
compatible /v1/embeddings endpoint. Saves .npy embedding matrices alongside
the source parquets.

Usage:
    python scripts/embed_norms_and_flows.py \
        --data-dir /path/to/fiction10 \
        --server-url http://localhost:8001
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Embedding instructions (instruction-aware model) ──

NORM_INSTRUCTION = (
    "Instruct: Given a prescriptive social norm from a literary text, "
    "represent it for semantic comparison with other norms.\nQuery: "
)

FLOW_INSTRUCTION = (
    "Instruct: Given a contextual integrity information flow from a literary text, "
    "represent it for semantic comparison with other information flows.\nQuery: "
)


# ── Serialization ──

def norm_to_text(row: dict) -> str:
    """Build embedding-friendly text from a norm row."""
    art = row.get("raz_norm_articulation") or ""
    subj = row.get("raz_norm_subject") or ""
    pe = row.get("raz_prescriptive_element") or ""
    act = row.get("raz_norm_act") or ""
    cond = row.get("raz_condition_of_application") or ""
    ctx = row.get("raz_context") or ""
    force = row.get("raz_normative_force") or ""

    parts = []
    if art:
        parts.append(art)
    tuple_str = f"{subj} {pe} {act}".strip()
    if cond:
        tuple_str += f" when {cond}"
    parts.append(tuple_str)
    if ctx:
        parts.append(f"[context: {ctx}]")
    if force:
        parts.append(f"[force: {force}]")
    return " | ".join(parts)


def flow_to_text(row: dict) -> str:
    """Convert a CI flow row into a readable sentence for embedding."""
    ctx = row.get("ci_context") or "unknown"
    sender = row.get("ci_sender") or "unknown"
    recipient = row.get("ci_recipient") or "unknown"
    info_type = row.get("ci_information_type") or "unknown"
    tp = row.get("ci_transmission_principle") or "unknown"
    subject = row.get("ci_subject")
    approp = row.get("ci_appropriateness") or ""

    parts = [f"In a {ctx} context"]
    parts.append(f"{sender} sends {info_type}")
    if subject and str(subject).strip():
        parts.append(f"about {subject}")
    parts.append(f"to {recipient}")
    parts.append(f"via {tp}")
    text = ", ".join(parts) + "."

    if approp and str(approp).strip():
        text += f" This flow is considered {approp}."

    return text


# ── vLLM embedding client ──

def embed_batch(
    session: requests.Session,
    server_url: str,
    texts: list[str],
    model_name: str,
    timeout: float = 120.0,
    max_retries: int = 3,
) -> np.ndarray:
    """Send a batch of texts to vLLM /v1/embeddings and return normalized embeddings."""
    for attempt in range(max_retries):
        try:
            resp = session.post(
                f"{server_url}/v1/embeddings",
                json={"model": model_name, "input": texts},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            data_sorted = sorted(data, key=lambda d: d["index"])
            embs = np.array([d["embedding"] for d in data_sorted], dtype=np.float32)
            # L2 normalize
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return embs / norms
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed after {max_retries} attempts: {e}") from e
            wait = 2 ** attempt
            print(f"  Attempt {attempt + 1} failed ({e}), retrying in {wait}s...")
            time.sleep(wait)


def embed_all(
    session: requests.Session,
    server_url: str,
    texts: list[str],
    instruction: str,
    model_name: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed all texts in batches, returning a single (N, dim) array."""
    prefixed = [instruction + t for t in texts]
    n = len(prefixed)
    results = []
    for start in range(0, n, batch_size):
        batch = prefixed[start : start + batch_size]
        embs = embed_batch(session, server_url, batch, model_name)
        results.append(embs)
        done = min(start + batch_size, n)
        print(f"  {done}/{n} ({100 * done / n:.0f}%)")
    return np.vstack(results)


def main():
    parser = argparse.ArgumentParser(description="Embed norms and flows via vLLM server")
    parser.add_argument("--data-dir", required=True, help="Path to fiction10 data directory")
    parser.add_argument("--server-url", default="http://localhost:8001", help="vLLM embedding server URL")
    parser.add_argument("--model-name", default="/share/pierson/matt/zoo/models/Qwen3-Embedding-8B", help="Model name as registered in vLLM (must match the path passed to vllm serve)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    session = requests.Session()

    # Load data
    print("Loading norms...")
    norms = pd.read_parquet(data_dir / "abstracted_norms.parquet")
    mask = norms["raz_norm_articulation"].notna() & (norms["raz_norm_articulation"] != "")
    norms = norms[mask].reset_index(drop=True)
    print(f"  {len(norms)} valid norms")

    print("Loading flows...")
    flows = pd.read_parquet(data_dir / "ci_flows.parquet")
    print(f"  {len(flows)} flows")

    # Serialize
    print("\nSerializing texts...")
    norm_texts = [norm_to_text(row) for row in norms.to_dict("records")]
    flow_texts = [flow_to_text(row) for row in flows.to_dict("records")]

    # Embed norms
    print(f"\nEmbedding {len(norm_texts)} norms...")
    norm_emb = embed_all(session, args.server_url, norm_texts, NORM_INSTRUCTION, args.model_name, args.batch_size)
    norm_out = data_dir / "norm_embeddings_qwen3emb.npy"
    np.save(norm_out, norm_emb)
    print(f"Saved: {norm_out} {norm_emb.shape}")

    # Embed flows
    print(f"\nEmbedding {len(flow_texts)} flows...")
    flow_emb = embed_all(session, args.server_url, flow_texts, FLOW_INSTRUCTION, args.model_name, args.batch_size)
    flow_out = data_dir / "flow_embeddings_qwen3emb.npy"
    np.save(flow_out, flow_emb)
    print(f"Saved: {flow_out} {flow_emb.shape}")

    session.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
