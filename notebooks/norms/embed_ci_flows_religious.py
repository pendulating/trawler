"""
Embed religious CI flows into a shared vector space using sentence-transformers,
then cluster with UMAP + HDBSCAN.

Loads a single parquet of religious-text CI flows (keyed by gutenberg_id),
serializes each flow into natural language, embeds with all-MiniLM-L6-v2,
and produces the same outputs as embed_ci_flows.py but with book = gutenberg_id.

Outputs:
  - A parquet file with the original flow data + embedding coordinates,
    cluster labels, and the serialized text.
  - A 2D UMAP scatter plot colored by text (gutenberg_id) and by cluster.
"""

import argparse
import re
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# 1. Text normalization
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    """Lowercase, strip articles/whitespace/punctuation."""
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = re.sub(r"^(a|an|the)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,;:!?]+$", "", s)
    return s.strip()


# ---------------------------------------------------------------------------
# 2. Serialize a CI flow row into natural language
# ---------------------------------------------------------------------------

FLOW_COLS = [
    "ci_context",
    "ci_sender",
    "ci_recipient",
    "ci_subject",
    "ci_information_type",
    "ci_transmission_principle",
    "ci_appropriateness",
]


def flow_to_text(row: pd.Series) -> str:
    """Convert a CI flow row into a readable sentence for embedding."""
    ctx = row.get("ci_context", "unknown")
    sender = row.get("ci_sender", "unknown")
    recipient = row.get("ci_recipient", "unknown")
    info_type = row.get("ci_information_type", "unknown")
    tp = row.get("ci_transmission_principle", "unknown")
    subject = row.get("ci_subject")
    approp = row.get("ci_appropriateness", "")

    parts = [f"In a {ctx} context"]
    parts.append(f"{sender} sends {info_type}")
    if pd.notna(subject) and str(subject).strip():
        parts.append(f"about {subject}")
    parts.append(f"to {recipient}")
    parts.append(f"via {tp}")
    text = ", ".join(parts) + "."

    if pd.notna(approp) and str(approp).strip():
        text += f" This flow is considered {approp}."

    return text


# ---------------------------------------------------------------------------
# 3. Load and prepare (religious)
# ---------------------------------------------------------------------------

def load_and_prepare(path_religious: str) -> pd.DataFrame:
    """Load religious ci_flows parquet, normalize CI fields, serialize flows."""
    df = pd.read_parquet(path_religious)
    df["book"] = df["gutenberg_id"].astype(str)

    for col in FLOW_COLS:
        if col in df.columns:
            df[col] = df[col].map(normalize_text)

    df["flow_text"] = df.apply(flow_to_text, axis=1)
    return df


def embed_flows(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
    """Embed flow_text column using sentence-transformers."""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    texts = df["flow_text"].tolist()
    print(f"Embedding {len(texts)} flows ...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings


def reduce_and_cluster(
    embeddings: np.ndarray,
    umap_n_components_cluster: int = 10,
    umap_n_components_viz: int = 2,
    hdbscan_min_cluster_size: int = 5,
    hdbscan_min_samples: int = 3,
    random_state: int = 42,
):
    """Run UMAP dimensionality reduction + HDBSCAN clustering."""
    print(f"UMAP reducing to {umap_n_components_cluster}D for clustering ...")
    reducer_cluster = umap.UMAP(
        n_components=umap_n_components_cluster,
        metric="cosine",
        random_state=random_state,
    )
    reduced = reducer_cluster.fit_transform(embeddings)

    print(
        f"HDBSCAN clustering (min_cluster_size={hdbscan_min_cluster_size}, "
        f"min_samples={hdbscan_min_samples}) ..."
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric="euclidean",
    )
    cluster_labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(cluster_labels) - {-1})
    n_noise = (cluster_labels == -1).sum()
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    print(f"UMAP reducing to {umap_n_components_viz}D for visualization ...")
    reducer_viz = umap.UMAP(
        n_components=umap_n_components_viz,
        metric="cosine",
        random_state=random_state,
    )
    umap_2d = reducer_viz.fit_transform(embeddings)
    return cluster_labels, umap_2d


def plot_results(df: pd.DataFrame, umap_2d: np.ndarray, output_dir: Path):
    """Generate scatter plots colored by text (gutenberg_id) and by cluster."""
    df = df.copy()
    df["umap_x"] = umap_2d[:, 0]
    df["umap_y"] = umap_2d[:, 1]

    # --- Plot 1: colored by book (gutenberg_id), top N in legend + Other ---
    fig, ax = plt.subplots(figsize=(12, 8))
    max_legend = 12
    book_counts = df["book"].value_counts()
    books_sorted = book_counts.index.tolist()
    in_legend = books_sorted[:max_legend]
    other_books = books_sorted[max_legend:]

    for i, book in enumerate(in_legend):
        mask = df["book"] == book
        ax.scatter(
            df.loc[mask, "umap_x"],
            df.loc[mask, "umap_y"],
            c=[plt.cm.tab20(i % 20)],
            label=book,
            alpha=0.5,
            s=10,
        )
    if other_books:
        mask = df["book"].isin(other_books)
        ax.scatter(
            df.loc[mask, "umap_x"],
            df.loc[mask, "umap_y"],
            c=[plt.cm.tab20(19)],
            label=f"Other ({len(other_books)} texts)",
            alpha=0.5,
            s=10,
        )
    ax.set_title("CI Flows (religious) — UMAP 2D projection (colored by text)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path_book = output_dir / "ci_flows_umap_by_book.png"
    fig.savefig(path_book, dpi=150, bbox_inches="tight")
    print(f"Saved: {path_book}")
    plt.close(fig)

    # --- Plot 2: colored by cluster ---
    fig, ax = plt.subplots(figsize=(12, 8))
    noise_mask = df["cluster"] == -1
    if noise_mask.any():
        ax.scatter(
            df.loc[noise_mask, "umap_x"],
            df.loc[noise_mask, "umap_y"],
            c="lightgrey",
            label="noise",
            alpha=0.3,
            s=8,
        )
    clustered = df[~noise_mask]
    scatter = ax.scatter(
        clustered["umap_x"],
        clustered["umap_y"],
        c=clustered["cluster"],
        cmap="tab20",
        alpha=0.6,
        s=10,
    )
    ax.set_title("CI Flows (religious) — UMAP 2D projection (colored by cluster)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    plt.tight_layout()
    path_cluster = output_dir / "ci_flows_umap_by_cluster.png"
    fig.savefig(path_cluster, dpi=150)
    print(f"Saved: {path_cluster}")
    plt.close(fig)


def print_cluster_summary(df: pd.DataFrame, n_examples: int = 3):
    """Print summary stats and example flows per cluster (per-text counts)."""
    clusters = sorted(df["cluster"].unique())
    for cid in clusters:
        subset = df[df["cluster"] == cid]
        label = f"Cluster {cid}" if cid != -1 else "Noise"
        book_counts = subset["book"].value_counts().head(8)
        book_summary = ", ".join(f"{b}: {n}" for b, n in book_counts.items())
        if subset["book"].nunique() > 8:
            book_summary += f" (+{subset['book'].nunique() - 8} more)"
        print(f"\n{'='*60}")
        print(f"{label}: {len(subset)} flows ({book_summary})")
        print(f"{'='*60}")
        top_ctx = subset["ci_context"].value_counts().head(3)
        print(f"  Top contexts: {dict(top_ctx)}")
        examples = subset["flow_text"].sample(min(n_examples, len(subset)), random_state=42)
        for i, txt in enumerate(examples):
            print(f"  Example {i+1}: {txt[:150]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embed and cluster religious CI flows")
    parser.add_argument(
        "--path-religious",
        default="/share/pierson/matt/UAIR/outputs/2026-02-11/15-40-07/ci_extraction_religious/outputs/ci_extraction/ci_flows.parquet",
        help="Path to religious ci_flows parquet",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--output-dir",
        default="/share/pierson/matt/UAIR/notebooks/norms/embedding_output/religious",
        help="Directory for output files",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN min_cluster_size (default: 5)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="HDBSCAN min_samples (default: 3)",
    )
    parser.add_argument(
        "--umap-cluster-dims",
        type=int,
        default=10,
        help="UMAP dims for clustering (default: 10)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(args.path_religious)
    print(f"Loaded {len(df)} total flows")

    embeddings = embed_flows(df, model_name=args.model)
    emb_path = output_dir / "ci_flow_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings: {emb_path} (shape {embeddings.shape})")

    cluster_labels, umap_2d = reduce_and_cluster(
        embeddings,
        umap_n_components_cluster=args.umap_cluster_dims,
        hdbscan_min_cluster_size=args.min_cluster_size,
        hdbscan_min_samples=args.min_samples,
    )
    df["cluster"] = cluster_labels
    df["umap_x"] = umap_2d[:, 0]
    df["umap_y"] = umap_2d[:, 1]

    out_parquet = output_dir / "ci_flows_embedded.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"Saved enriched data: {out_parquet}")

    plot_results(df, umap_2d, output_dir)
    print_cluster_summary(df)
    print("\nDone!")


if __name__ == "__main__":
    main()
