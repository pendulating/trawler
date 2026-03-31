"""
Embed CI flows into a shared vector space using sentence-transformers,
then cluster with UMAP + HDBSCAN.

Each CI flow is serialized into a natural-language template from its
structured fields (context, sender, recipient, information_type,
transmission_principle, subject, appropriateness) and embedded with
all-MiniLM-L6-v2.

Outputs:
  - A parquet file with the original flow data + embedding coordinates,
    cluster labels, and the serialized text.
  - A 2D UMAP scatter plot colored by book and by cluster.
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
# 1. Text normalization (same as notebook)
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
    """Convert a CI flow row into a readable sentence for embedding.

    Example output:
      "In a state/political context, big brother (state) sends state
       propaganda about production of pig-iron to the public via state
       mandate. This flow is considered appropriate."
    """
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
# 3. Main pipeline
# ---------------------------------------------------------------------------

def load_and_prepare(
    path_1984: str,
    path_pandp: str,
) -> pd.DataFrame:
    """Load both parquet files, normalize CI fields, serialize flows."""
    df_1984 = pd.read_parquet(path_1984)
    df_pandp = pd.read_parquet(path_pandp)

    df_1984["book"] = "1984"
    df_pandp["book"] = "Pride & Prejudice"

    df = pd.concat([df_1984, df_pandp], ignore_index=True)

    # Normalize the CI tuple fields
    for col in FLOW_COLS:
        if col in df.columns:
            df[col] = df[col].map(normalize_text)

    # Serialize each flow
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
    """Run UMAP dimensionality reduction + HDBSCAN clustering.

    Returns:
        cluster_labels: array of int cluster labels (-1 = noise)
        umap_2d: (N, 2) array for visualization
    """
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


def plot_results(
    df: pd.DataFrame,
    umap_2d: np.ndarray,
    output_dir: Path,
):
    """Generate scatter plots colored by book and by cluster."""
    df = df.copy()
    df["umap_x"] = umap_2d[:, 0]
    df["umap_y"] = umap_2d[:, 1]

    # --- Plot 1: colored by book ---
    fig, ax = plt.subplots(figsize=(12, 8))
    book_colors = {"1984": "steelblue", "Pride & Prejudice": "salmon"}
    for book, color in book_colors.items():
        mask = df["book"] == book
        ax.scatter(
            df.loc[mask, "umap_x"],
            df.loc[mask, "umap_y"],
            c=color,
            label=book,
            alpha=0.5,
            s=10,
        )
    ax.set_title("CI Flows — UMAP 2D projection (colored by book)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend()
    plt.tight_layout()
    path_book = output_dir / "ci_flows_umap_by_book.png"
    fig.savefig(path_book, dpi=150)
    print(f"Saved: {path_book}")
    plt.close(fig)

    # --- Plot 2: colored by cluster ---
    fig, ax = plt.subplots(figsize=(12, 8))
    noise_mask = df["cluster"] == -1
    # Plot noise points in grey first
    if noise_mask.any():
        ax.scatter(
            df.loc[noise_mask, "umap_x"],
            df.loc[noise_mask, "umap_y"],
            c="lightgrey",
            label="noise",
            alpha=0.3,
            s=8,
        )
    # Plot clustered points
    clustered = df[~noise_mask]
    scatter = ax.scatter(
        clustered["umap_x"],
        clustered["umap_y"],
        c=clustered["cluster"],
        cmap="tab20",
        alpha=0.6,
        s=10,
    )
    ax.set_title("CI Flows — UMAP 2D projection (colored by cluster)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    plt.tight_layout()
    path_cluster = output_dir / "ci_flows_umap_by_cluster.png"
    fig.savefig(path_cluster, dpi=150)
    print(f"Saved: {path_cluster}")
    plt.close(fig)


def print_cluster_summary(df: pd.DataFrame, n_examples: int = 3):
    """Print summary stats and example flows per cluster."""
    clusters = sorted(df["cluster"].unique())
    for cid in clusters:
        subset = df[df["cluster"] == cid]
        label = f"Cluster {cid}" if cid != -1 else "Noise"
        n_1984 = (subset["book"] == "1984").sum()
        n_pandp = (subset["book"] == "Pride & Prejudice").sum()
        print(f"\n{'='*60}")
        print(f"{label}: {len(subset)} flows (1984: {n_1984}, P&P: {n_pandp})")
        print(f"{'='*60}")
        # Show top contexts
        top_ctx = subset["ci_context"].value_counts().head(3)
        print(f"  Top contexts: {dict(top_ctx)}")
        # Show example flow texts
        examples = subset["flow_text"].sample(min(n_examples, len(subset)), random_state=42)
        for i, txt in enumerate(examples):
            print(f"  Example {i+1}: {txt[:150]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embed and cluster CI flows (1984 + Pride & Prejudice)")
    parser.add_argument(
        "--path-1984",
        default="/share/pierson/matt/UAIR/outputs/2026-02-10/08-02-59/ci_extraction_1984/outputs/ci_extraction/ci_flows.parquet",
        help="Path to 1984 ci_flows parquet",
    )
    parser.add_argument(
        "--path-pandp",
        default="/share/pierson/matt/UAIR/outputs/2026-02-09/10-03-34/ci_extraction/outputs/ci_extraction/ci_flows.parquet",
        help="Path to Pride & Prejudice ci_flows parquet",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--output-dir",
        default="/share/pierson/matt/UAIR/notebooks/norms/embedding_output",
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

    # Load and prepare
    df = load_and_prepare(args.path_1984, args.path_pandp)
    print(f"Loaded {len(df)} total flows")

    # Embed
    embeddings = embed_flows(df, model_name=args.model)

    # Save raw embeddings
    emb_path = output_dir / "ci_flow_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings: {emb_path} (shape {embeddings.shape})")

    # Cluster
    cluster_labels, umap_2d = reduce_and_cluster(
        embeddings,
        umap_n_components_cluster=args.umap_cluster_dims,
        hdbscan_min_cluster_size=args.min_cluster_size,
        hdbscan_min_samples=args.min_samples,
    )
    df["cluster"] = cluster_labels
    df["umap_x"] = umap_2d[:, 0]
    df["umap_y"] = umap_2d[:, 1]

    # Save enriched dataframe
    out_parquet = output_dir / "ci_flows_embedded.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"Saved enriched data: {out_parquet}")

    # Plots
    plot_results(df, umap_2d, output_dir)

    # Cluster summary
    print_cluster_summary(df)

    print("\nDone!")


if __name__ == "__main__":
    main()
