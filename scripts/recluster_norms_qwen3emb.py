#!/usr/bin/env python3
"""Re-cluster norms using Qwen3-Embedding-8B for higher-quality consolidation.

Replaces the MiniLM embeddings in the consolidation pipeline with Qwen3-Embedding-8B
(4096-dim, instruction-aware). Runs embedding on GPU, clustering on CPU, then
feeds the new clusters to the existing LLM merge pipeline.

Usage:
    python scripts/recluster_norms_qwen3emb.py

Requires 1 GPU for the 8B embedding model (~16GB).
"""

import json
import os

import hdbscan
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── Paths ──
NORMS_PATH = "/share/pierson/matt/n2s4cir/data/fiction10/structured_norms.parquet"
MODEL_PATH = "/share/pierson/matt/zoo/models/Qwen3-Embedding-8B"
OUTPUT_DIR = "/share/pierson/matt/n2s4cir/data/fiction10"

# ── Clustering params ──
MIN_CLUSTER_SIZE = 3
MIN_SAMPLES = 2
CLUSTER_SELECTION_EPSILON = 0.0

# ── Embedding instruction ──
EMBED_INSTRUCTION = (
    "Instruct: Given a prescriptive social norm from a literary text, "
    "represent it for clustering with semantically equivalent norms.\nQuery: "
)

ID_TO_TITLE = {
    "1984": "1984", "541": "Age of Innocence", "11": "Alice",
    "1399": "Anna Karenina", "1023": "Bleak House",
    "1184": "Count of Monte Cristo", "135": "Les Mis",
    "145": "Middlemarch", "4078": "Dorian Gray", "1342": "Pride & Prejudice",
}


def build_norm_text(row: dict) -> str:
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


def cluster_within_partitions(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    epsilon: float,
    offset: int = 0,
) -> tuple[np.ndarray, int]:
    """Cluster embeddings partitioned by normative_force."""
    labels = np.full(len(df), -1, dtype=int)
    force_col = df["raz_normative_force"].fillna("unknown")

    for force_val in force_col.unique():
        mask = (force_col == force_val).values
        indices = np.where(mask)[0]
        if len(indices) < min_cluster_size:
            continue

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_epsilon=epsilon,
            cluster_selection_method="eom",
        )
        part_labels = clusterer.fit_predict(embeddings[indices])
        n_clusters = len(set(part_labels)) - (1 if -1 in part_labels else 0)
        n_noise = int((part_labels == -1).sum())
        print(f"    {force_val}: {len(indices)} norms -> {n_clusters} clusters, {n_noise} singletons")

        for i, idx in enumerate(indices):
            if part_labels[i] >= 0:
                labels[idx] = part_labels[i] + offset
        offset += max(n_clusters, 0)

    return labels, offset


def main():
    print("Loading norms...")
    df = pd.read_parquet(NORMS_PATH)
    mask = df["raz_norm_articulation"].notna() & (df["raz_norm_articulation"] != "")
    df = df[mask].reset_index(drop=True)
    print(f"  {len(df)} valid norms")

    # Build texts
    texts = [build_norm_text(row) for row in df.to_dict("records")]

    # Load Qwen3-Embedding-8B
    print(f"\nLoading Qwen3-Embedding-8B from {MODEL_PATH}...")
    model = SentenceTransformer(
        MODEL_PATH,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Embed with instruction prefix
    print(f"\nEmbedding {len(texts)} norms...")
    prefixed = [EMBED_INSTRUCTION + t for t in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Save embeddings for reuse
    emb_path = os.path.join(OUTPUT_DIR, "norm_embeddings_qwen3emb.npy")
    np.save(emb_path, embeddings)
    print(f"  Saved to {emb_path}")

    # Free GPU memory
    del model

    # Cluster per-book, per-deontic-force
    print("\nClustering...")
    groups = df["gutenberg_id"].fillna("unknown")
    labels = np.full(len(df), -1, dtype=int)
    offset = 0

    for gval in sorted(groups.unique()):
        gmask = (groups == gval).values
        gidx = np.where(gmask)[0]
        if len(gidx) == 0:
            continue
        title = ID_TO_TITLE.get(gval, gval)
        print(f"  {title} ({gval}): {len(gidx)} norms")

        gdf = df.iloc[gidx].copy()
        gemb = embeddings[gidx]
        glabels, offset = cluster_within_partitions(
            gdf, gemb, MIN_CLUSTER_SIZE, MIN_SAMPLES, CLUSTER_SELECTION_EPSILON, offset,
        )
        for i, idx in enumerate(gidx):
            labels[idx] = glabels[i]

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_singletons = int((labels == -1).sum())
    print(f"\nTotal: {n_clusters} clusters, {n_singletons} singletons")

    # Save cluster assignments
    df["qwen3emb_cluster"] = labels
    cluster_path = os.path.join(OUTPUT_DIR, "norms_with_qwen3emb_clusters.parquet")
    df.to_parquet(cluster_path)
    print(f"Saved cluster assignments to {cluster_path}")

    # Summary comparison with MiniLM
    print(f"\n=== Cluster size distribution ===")
    from collections import Counter
    size_counter = Counter()
    for cid in set(labels):
        if cid == -1:
            continue
        size_counter[int((labels == cid).sum())] += 1
    for size in sorted(size_counter.keys()):
        if size <= 10 or size_counter[size] > 1:
            print(f"  size {size}: {size_counter[size]} clusters")


if __name__ == "__main__":
    main()
