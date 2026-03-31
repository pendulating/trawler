"""Norm universe construction: embed raw norms per book with Qwen3-Embedding-8B.

Takes abstracted_norms.parquet (role-abstracted norms) and:
1. Groups by gutenberg_id (each book = one normative universe N_hat_b)
2. Embeds each norm's articulation with Qwen3-Embedding-8B (4096-dim)
3. Outputs a JSON universe file + .npy embeddings matrix

No consolidation step — embedding similarity replaces LLM merging.
"""

import os
from typing import Any, Dict

import numpy as np
import pandas as pd


# Instruction prefix for Qwen3-Embedding-8B (instruction-aware model)
EMBED_INSTRUCTION = (
    "Instruct: Given a prescriptive social norm from a literary text, "
    "represent it for semantic matching with information flows.\nQuery: "
)


def _build_norm_text(row: dict) -> str:
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


# Norm fields to include in the JSON universe
_NORM_FIELDS = [
    "raz_prescriptive_element",
    "raz_norm_subject",
    "raz_norm_act",
    "raz_condition_of_application",
    "raz_normative_force",
    "raz_norm_articulation",
    "raz_context",
    "raz_governs_info_flow",
    "raz_info_flow_note",
    "raz_confidence_qual",
    "raz_confidence_quant",
]


def run_norm_universe_stage(
    df: pd.DataFrame,
    cfg: Any,
    output_dir: str,
) -> Dict[str, Any]:
    """Build per-book normative universes with Qwen3-Embedding-8B embeddings.

    Args:
        df: abstracted_norms DataFrame with raz_* columns and gutenberg_id.
        cfg: Hydra config with model.embedding_model_source for Qwen3-Embedding.
        output_dir: Directory to write norm_universes.json and embeddings.

    Returns:
        Dict mapping gutenberg_id -> list of norm dicts (the N_hat_b).
    """
    from omegaconf import OmegaConf

    # Identify source column
    source_col = None
    for candidate in ("gutenberg_id", "source_id", "book_id"):
        if candidate in df.columns:
            source_col = candidate
            break
    if source_col is None:
        raise ValueError(
            f"[norm_universe] No source identifier column. "
            f"Available: {list(df.columns)}"
        )

    # Book-level filter: restrict to a single book's norms
    book_id = OmegaConf.select(cfg, "runtime.book_id", default=None)
    if book_id is not None:
        book_id_str = str(book_id)
        pre = len(df)
        df = df[df[source_col].astype(str) == book_id_str].reset_index(drop=True)
        print(f"[norm_universe] Filtered to book_id={book_id_str}: {len(df)}/{pre} norms")

    # Filter to norms with valid articulations
    mask = df["raz_norm_articulation"].notna() & (df["raz_norm_articulation"] != "")
    df = df[mask].reset_index(drop=True)
    print(f"[norm_universe] {len(df)} valid norms across "
          f"{df[source_col].nunique()} books")

    # Build embedding texts
    texts = [_build_norm_text(row) for row in df.to_dict("records")]

    # Load embedding model
    embedding_model_path = str(
        OmegaConf.select(cfg, "embedding_model.model_source", default=None)
        or OmegaConf.select(cfg, "model.embedding_model_source", default=None)
        or OmegaConf.select(cfg, "norm_universe.embedding_model", default=None)
        or ""
    )
    if not embedding_model_path:
        raise ValueError(
            "[norm_universe] No embedding model configured. "
            "Set model.embedding_model_source or norm_universe.embedding_model"
        )

    print(f"[norm_universe] Loading embedding model: {embedding_model_path}")
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer(
        embedding_model_path,
        device="cuda:0",
        tokenizer_kwargs={"padding_side": "left"},
    )
    embed_dim = embed_model.get_sentence_embedding_dimension()
    print(f"[norm_universe] Embedding dim: {embed_dim}")

    # Embed all norms with instruction prefix
    print(f"[norm_universe] Embedding {len(texts)} norms...")
    prefixed = [EMBED_INSTRUCTION + t for t in texts]
    embeddings = embed_model.encode(
        prefixed,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings)
    print(f"[norm_universe] Embeddings shape: {embeddings.shape}")

    # Free GPU memory
    del embed_model
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build per-book universes and save per-book embeddings
    os.makedirs(output_dir, exist_ok=True)
    emb_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    available_fields = [f for f in _NORM_FIELDS if f in df.columns]
    norm_universes: Dict[str, list] = {}

    for source_id, group in df.groupby(source_col):
        source_key = str(source_id)
        indices = group.index.tolist()
        norms = []
        for idx in indices:
            row = df.iloc[idx]
            norm = {}
            for field in available_fields:
                val = row.get(field)
                clean_key = field.replace("raz_", "", 1)
                norm[clean_key] = val if pd.notna(val) else None
            norms.append(norm)
        norm_universes[source_key] = norms

        # Save per-book embedding matrix
        book_emb = embeddings[indices]
        np.save(os.path.join(emb_dir, f"{source_key}.npy"), book_emb)

    # Also save full embeddings + index for convenience
    np.save(os.path.join(output_dir, "all_embeddings.npy"), embeddings)
    index_df = df[[source_col]].copy()
    index_df.to_parquet(os.path.join(output_dir, "embedding_index.parquet"))

    total_norms = sum(len(v) for v in norm_universes.values())
    print(f"[norm_universe] Built {len(norm_universes)} universes "
          f"({total_norms} norms, {embed_dim}-dim embeddings)")
    for sid, norms in sorted(norm_universes.items()):
        print(f"  {sid}: {len(norms)} norms")

    return norm_universes
