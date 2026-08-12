#!/usr/bin/env python3
"""Per-book nearest-norm retrieval for CI flows.

Given a norms frame and a CI-flows frame (the parquet outputs of the
`historical_norms` dagspace), embed both sides against the standing
Qwen3-Embedding-8B vLLM server and return, for each flow, its top-K nearest
norms **from the same source text**. Stratifying by book is not a nicety: the
normative universe R-GROUND retrieves against is per-source, so a cross-book
neighbour is not a thing production could ever surface.

Two retrieval spaces are available, and they are genuinely different spaces —
the neighbours differ:

``rground`` (the default)
    Production parity with what the reward actually retrieves. The norm side is
    ``norm_universe._build_norm_text``; the flow side is
    ``online_rground._flow_to_query`` — a bare concatenation of sender,
    recipient, information_type, context, transmission_principle, subject and
    any invoked norms. **Both** sides carry ``norm_universe.EMBED_INSTRUCTION``,
    because that is what `reward_prep.py` and `online_rground.py` do: the query
    is embedded with the norm instruction, not with a flow instruction.

``shared``
    Camera-ready notebook parity. Flow side is the prose sentence from
    ``embed_camera_ready_norms_flows.flow_to_text``; both sides carry that
    script's ``SHARED_INSTRUCTION``. Reproduces the pairing behind
    `notebooks/colm-camera-ready/norm_flow_embedding_space.py`.

Embeddings are cached by a content hash of (model, instruction, texts), so
re-running against the same frames costs nothing and a changed frame can never
silently reuse a stale matrix.

Used by `scripts/norms_inspector.py --neighbors K`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagspaces.grpo_training.stages.norm_universe import (  # noqa: E402
    EMBED_INSTRUCTION,
    _build_norm_text,
)
from dagspaces.grpo_training.stages.online_rground import _flow_to_query  # noqa: E402
from scripts.embed_camera_ready_norms_flows import (  # noqa: E402
    EMB_MODEL,
    SHARED_INSTRUCTION,
    embed_all,
    flow_to_text,
)

DEFAULT_CACHE_DIR = ROOT / "outputs/norms_inspector/embeddings"

# --- retrieval spaces -------------------------------------------------------
# (flow-query builder key, instruction applied to BOTH constructs)
SPACES = {
    "rground": EMBED_INSTRUCTION,
    "shared": SHARED_INSTRUCTION,
}

# `production` is `_flow_to_query`; the other three are the camera-ready prose
# serializations from embed_camera_ready_norms_flows.FLOW_TEXT_VARIANTS.
FLOW_QUERY_MODES = ("production", "noappr", "full", "descriptive")

# ci_* parquet column -> the flat key `_flow_to_query` expects.
_CI_TO_FLAT = {
    "ci_sender": "sender",
    "ci_recipient": "recipient",
    "ci_information_type": "information_type",
    "ci_context": "context",
    "ci_transmission_principle": "transmission_principle",
    "ci_subject": "subject",
    "ci_norms_invoked": "norms_invoked",
}


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Row dicts with missing values as None rather than NaN.

    Both serializers guard with ``row.get(col) or ""``, and float('nan') is
    truthy — a NaN cell would sail through that guard and end up either
    concatenated as the literal string "nan" or, in `_build_norm_text`, raising
    a TypeError inside `" | ".join`. Production is fed JSON-parsed dicts where
    absent means None, so cleaning here restores parity rather than changing it.
    """
    out = []
    for rec in df.to_dict("records"):
        clean: dict[str, Any] = {}
        for k, v in rec.items():
            clean[k] = None if _is_missing(v) else v
        out.append(clean)
    return out


def _is_missing(v: Any) -> bool:
    """Scalar-safe NA test. `pd.isna` returns an array for list/ndarray cells."""
    if v is None:
        return True
    if isinstance(v, (list, tuple, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _flat_flow(row: dict[str, Any]) -> dict[str, Any]:
    """Map a ci_flows parquet row onto the flat keys production uses."""
    flat: dict[str, Any] = {}
    for col, key in _CI_TO_FLAT.items():
        val = row.get(col)
        # Parquet list columns arrive as ndarray, and `_flow_to_query` gates the
        # invoked-norms branch on `isinstance(invoked, list)` — an ndarray would
        # be silently dropped and the query would differ from production's.
        if isinstance(val, np.ndarray):
            val = val.tolist()
        flat[key] = val
    return flat


def norm_texts(norms: pd.DataFrame) -> list[str]:
    """Serialize norms exactly as the norm universe does."""
    return [_build_norm_text(r) for r in _records(norms)]


def flow_texts(flows: pd.DataFrame, mode: str) -> list[str]:
    """Serialize flows into retrieval queries. See FLOW_QUERY_MODES."""
    if mode not in FLOW_QUERY_MODES:
        raise ValueError(f"unknown flow query mode {mode!r}")
    recs = _records(flows)
    if mode == "production":
        return [_flow_to_query(_flat_flow(r)) for r in recs]
    return [flow_to_text(r, mode) for r in recs]


# --- embedding with a content-addressed cache -------------------------------
def _cache_key(texts: list[str], instruction: str, model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(b"\x00")
    h.update(instruction.encode())
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8", errors="replace"))
    return h.hexdigest()[:24]


def embed_cached(
    texts: list[str],
    instruction: str,
    *,
    server_url: str,
    model_name: str = EMB_MODEL,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    batch_size: int = 32,
    label: str = "texts",
) -> np.ndarray:
    """L2-normalised embeddings for `texts`, cached by content hash."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(texts, instruction, model_name)
    path = cache_dir / f"{key}.npy"

    if path.exists():
        emb = np.load(path)
        if emb.shape[0] == len(texts):
            print(f"    [{label}] cache hit {path.name} {emb.shape}")
            return emb
        print(f"    [{label}] cache {path.name} has {emb.shape[0]} rows, "
              f"expected {len(texts)} — re-embedding")

    if not server_url:
        raise RuntimeError(
            f"no embedding for {label} in {cache_dir} and no server URL. "
            "Start scripts/embedding_server.sub and set EMBEDDING_SERVER_URL, "
            "or pass --embed-server-url."
        )

    import requests

    print(f"    [{label}] embedding {len(texts)} rows via {server_url} ...", flush=True)
    session = requests.Session()
    try:
        emb = embed_all(session, server_url, texts, instruction, model_name, batch_size)
    finally:
        session.close()
    np.save(path, emb)
    (cache_dir / f"{key}.json").write_text(
        json.dumps(
            {"label": label, "n": len(texts), "model": model_name,
             "instruction": instruction, "example": texts[0] if texts else ""},
            indent=2, ensure_ascii=False,
        ) + "\n"
    )
    print(f"    [{label}] saved {path.name} {emb.shape}")
    return emb


# --- per-book top-K ---------------------------------------------------------
def topk_same_book(
    flow_emb: np.ndarray,
    norm_emb: np.ndarray,
    flow_books: np.ndarray,
    norm_books: np.ndarray,
    norm_mask: np.ndarray,
    k: int,
    block: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-K nearest norms per flow, restricted to the flow's own book.

    Embeddings are already L2-normalised, so the dot product is cosine.

    Returns (idx, sim), both (n_flows, k). Slots with no candidate — a book with
    flows but no eligible norms, or fewer than k of them — are -1 / NaN.
    """
    n_flows = len(flow_emb)
    idx = np.full((n_flows, k), -1, dtype=np.int64)
    sim = np.full((n_flows, k), np.nan, dtype=np.float32)

    flows_by_book: dict[Any, list[int]] = {}
    for i, b in enumerate(flow_books):
        flows_by_book.setdefault(b, []).append(i)
    norms_by_book: dict[Any, list[int]] = {}
    for j, b in enumerate(norm_books):
        if norm_mask[j]:
            norms_by_book.setdefault(b, []).append(j)

    for book, fis in flows_by_book.items():
        cand = norms_by_book.get(book)
        if not cand:
            continue
        cand = np.asarray(cand)
        fis = np.asarray(fis)
        ne = norm_emb[cand]
        kk = min(k, len(cand))
        cols = np.arange(kk)[None, :]
        for start in range(0, len(fis), block):
            blk = fis[start:start + block]
            sims = flow_emb[blk] @ ne.T
            rows = np.arange(len(blk))[:, None]
            if kk < sims.shape[1]:
                part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
            else:
                part = np.tile(np.arange(kk), (len(blk), 1))
            top = part[rows, np.argsort(-sims[rows, part], axis=1)]
            idx[blk[:, None], cols] = cand[top]
            sim[blk[:, None], cols] = sims[rows, top]
    return idx, sim


def book_keys(df: pd.DataFrame) -> np.ndarray:
    """Per-row book identity: gutenberg_id when present, else book_title."""
    col = "gutenberg_id" if "gutenberg_id" in df.columns else "book_title"
    return df[col].astype(str).to_numpy()


def eligible_norm_mask(norms: pd.DataFrame, pool: str) -> np.ndarray:
    """Which norms may be retrieved. `pool` is 'all' or 'governs'."""
    art = norms.get("raz_norm_articulation")
    # `.to_numpy()` can hand back a read-only view of the underlying block, so
    # copy before any in-place &=.
    mask = (
        np.ones(len(norms), dtype=bool)
        if art is None
        else (art.notna() & (art.astype(str).str.strip() != "")).to_numpy(dtype=bool, copy=True)
    )
    if pool == "governs":
        gov = norms.get("raz_governs_info_flow")
        if gov is None:
            raise KeyError("--neighbor-pool governs needs a raz_governs_info_flow column")
        mask &= gov.fillna(False).astype(bool).to_numpy(dtype=bool, copy=True)
    elif pool != "all":
        raise ValueError(f"unknown norm pool {pool!r}")
    return mask


# --- the one call the inspector makes ---------------------------------------
# Fields carried into each neighbour card, so a flow row stays readable even
# when the norm it points at was filtered out of the rendered norms stage.
NEIGHBOR_FIELDS = [
    ("art", "raz_norm_articulation"),
    ("subj", "raz_norm_subject"),
    ("pe", "raz_prescriptive_element"),
    ("act", "raz_norm_act"),
    ("cond", "raz_condition_of_application"),
    ("force", "raz_normative_force"),
    ("ctx", "raz_context"),
    ("gov", "raz_governs_info_flow"),
    ("chunk", "chunk_id"),
    ("book", "book_title"),
]


def compute_neighbors(
    norms: pd.DataFrame,
    flows: pd.DataFrame,
    *,
    k: int = 3,
    space: str = "rground",
    flow_query: str = "production",
    pool: str = "all",
    server_url: str = "",
    model_name: str = EMB_MODEL,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    batch_size: int = 32,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Top-K same-book norms per flow, as JSON-ready records.

    `norms` must be the *same* frame whose row positions the caller renders —
    each neighbour's ``i`` is a positional index into it.

    Returns (per-flow neighbour lists, a metadata dict describing the run).
    """
    if space not in SPACES:
        raise ValueError(f"unknown retrieval space {space!r}; choose from {sorted(SPACES)}")
    instruction = SPACES[space]

    ntexts = norm_texts(norms)
    ftexts = flow_texts(flows, flow_query)
    mask = eligible_norm_mask(norms, pool)
    print(f"  norm pool '{pool}': {int(mask.sum()):,} / {len(norms):,} eligible")
    print(f"  flow query [{flow_query}]: {ftexts[0][:160] if ftexts else '(none)'}")
    print(f"  norm text            : {ntexts[0][:160] if ntexts else '(none)'}")

    ne = embed_cached(ntexts, instruction, server_url=server_url, model_name=model_name,
                      cache_dir=cache_dir, batch_size=batch_size, label="norms")
    fe = embed_cached(ftexts, instruction, server_url=server_url, model_name=model_name,
                      cache_dir=cache_dir, batch_size=batch_size, label="flows")

    idx, sim = topk_same_book(fe, ne, book_keys(flows), book_keys(norms), mask, k)

    norm_recs = _records(norms)
    out: list[list[dict[str, Any]]] = []
    for r in range(len(flows)):
        nbrs = []
        for rank in range(k):
            j = int(idx[r, rank])
            if j < 0:
                continue
            src = norm_recs[j]
            rec: dict[str, Any] = {"i": j, "r": rank + 1, "s": round(float(sim[r, rank]), 4)}
            for short, col in NEIGHBOR_FIELDS:
                val = src.get(col)
                if val is None:
                    continue
                if isinstance(val, (np.bool_, bool)):
                    rec[short] = bool(val)
                elif isinstance(val, (np.integer, np.floating, int, float)):
                    num = val.item() if hasattr(val, "item") else val
                    # A non-finite number would serialize as bare NaN/Infinity,
                    # which is not valid JSON.
                    if isinstance(num, float) and not np.isfinite(num):
                        continue
                    rec[short] = num
                else:
                    text = str(val)
                    if text.strip():
                        rec[short] = text
            nbrs.append(rec)
        out.append(nbrs)

    n_empty = sum(1 for n in out if not n)
    meta = {
        "k": k,
        "space": space,
        "flow_query": flow_query,
        "pool": pool,
        "model": model_name,
        "n_norms_eligible": int(mask.sum()),
        "n_norms_total": len(norms),
        "n_flows_unmatched": n_empty,
    }
    print(f"  paired {len(out) - n_empty:,} / {len(out):,} flows "
          f"(top-1 cos mean {np.nanmean(sim[:, 0]):.4f})")
    if n_empty:
        print(f"  WARNING: {n_empty} flows have no same-book norm in this pool")
    return out, meta
