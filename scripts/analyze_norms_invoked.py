#!/usr/bin/env python3
"""Do the norms a policy CITES belong to the book's actual normative universe?

Plan: wiki/2026-08-05_distilled_grounding_plan.md §4.2

The appropriateness analysis (``build_grounding_disagreement.py``) asks what a
model *concludes*. This asks what it *cites*. Each extracted flow carries
``ci_norms_invoked`` — a JSON list of norm articulations the model produced with
no universe in context — so we can ask whether fine-tuning moved those citations
toward the norms the source text actually contains.

Three measurements per arm, per book:

``hit@tau``      fraction of cited norms whose top-1 cosine against the book's
                 own governing-norm index clears ``--tau``.
``concordance``  fraction of flows where the cited norm's nearest universe norm
                 IS the norm retrieval picked for that flow. A model can cite
                 something that lives in the universe while still not citing the
                 governing one; this separates the two.
``hit@tau (wrong book)``
                 THE CONTROL, and the reason this script is worth running. The
                 same citations scored against a *different* novel's universe,
                 paired by ``seeded_wrong_book`` — the same deterministic
                 pairing R-CONTRAST uses. **If the hit rate rises equally
                 against the wrong book, the model learned norm-shaped phrasing,
                 not this universe.** Read ``hit_margin = hit - hit_wrong``, not
                 ``hit``.

Input is the per-arm table ``build_grounding_disagreement.py`` writes, so the
retrieval columns are already joined and the flow-side work is not redone.

Needs a live embedding server on first run for any arm: cited-norm strings are
new text and will miss the ``rground`` cache.

Usage:
    python scripts/analyze_norms_invoked.py \
        --labels outputs/2026-08-05_distilled_grounding/m2-full/flow_grounding_labels.parquet \
        --labels outputs/2026-08-05_distilled_grounding/teacher/flow_grounding_labels.parquet \
        --embed-server-url http://klara:8001 \
        --out outputs/2026-08-05_distilled_grounding/norms_invoked
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagspaces.common.stage_utils import ensure_dotenv  # noqa: E402
from dagspaces.grpo_training.stages.aux_scorers import seeded_wrong_book  # noqa: E402
from dagspaces.grpo_training.stages.clients import NormRetriever  # noqa: E402
from dagspaces.grpo_training.stages.norm_universe import EMBED_INSTRUCTION  # noqa: E402
from scripts.norm_neighbors import DEFAULT_CACHE_DIR, embed_cached  # noqa: E402

UNIVERSE = ROOT / "outputs/2026-07-25_universe_fiction10_polarity"
EMB_MODEL = "/share/pierson/matt/zoo/models/Qwen3-Embedding-8B"


def parse_cited(raw) -> list[str]:
    """The cited-norm strings for one flow.

    ``ci_norms_invoked`` is a JSON-encoded list of articulations (teacher
    corpus: 15,713 flows cite exactly one, 480 cite two, 3 cite three, 4 cite
    none). Anything unparseable yields [] and is counted, never guessed at.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, (list, tuple, np.ndarray)):
        vals = list(raw)
    else:
        try:
            vals = json.loads(str(raw))
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(vals, list):
        return []
    return [str(v).strip() for v in vals if str(v).strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="append", required=True,
                    help="flow_grounding_labels.parquet, repeatable (one per arm)")
    ap.add_argument("--universe", default=str(UNIVERSE))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.60,
                    help="cosine threshold for a citation counting as a hit")
    ap.add_argument("--embed-server-url", default="")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    args = ap.parse_args()

    ensure_dotenv()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    universe = Path(args.universe)
    universes = json.loads((universe / "norm_universes.json").read_text())
    books = sorted(universes)

    retriever = NormRetriever(
        norm_universes=universes,
        embeddings_dir=str(universe / "embeddings"),
        embedding_client=None,
        top_k=1,
        norm_filter=lambda n: n.get("governs_info_flow") is True,
    )

    # ---- flatten every arm's citations into one embed batch ---------------
    # One batch across arms so shared strings embed once and the cache is
    # populated for every later run.
    frames = []
    for path in args.labels:
        df = pd.read_parquet(path)
        arm = str(df["arm"].iloc[0]) if "arm" in df.columns else Path(path).parent.name
        for i, row in df.reset_index(drop=True).iterrows():
            cited = parse_cited(row.get("ci_norms_invoked"))
            sid = str(row["gutenberg_id"])
            wrong = seeded_wrong_book(row["chunk_id"], sid, books)
            if not cited:
                frames.append({"arm": arm, "gutenberg_id": sid, "flow_row": i,
                               "cited": None, "wrong_book": wrong,
                               "top_articulation": row.get("top_articulation")})
                continue
            for c in cited:
                frames.append({"arm": arm, "gutenberg_id": sid, "flow_row": i,
                               "cited": c, "wrong_book": wrong,
                               "top_articulation": row.get("top_articulation")})
    cites = pd.DataFrame(frames)
    scored = cites[cites["cited"].notna()].copy()
    print(f"[cites] {len(cites)} rows over {cites['arm'].nunique()} arms; "
          f"{len(scored)} with a citation, "
          f"{len(cites) - len(scored)} with none")

    texts = scored["cited"].tolist()
    emb = embed_cached(texts, EMBED_INSTRUCTION,
                       server_url=args.embed_server_url, model_name=EMB_MODEL,
                       cache_dir=Path(args.cache_dir), label="cited-norms")
    emb = np.asarray(emb, dtype=np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)

    # ---- score each citation against its own book and its paired wrong book
    rows = []
    for pos, (_, r) in enumerate(scored.iterrows()):
        own, sims = retriever.retrieve(emb[pos], r["gutenberg_id"],
                                       return_scores=True, top_k=1)
        own = json.loads(own) if isinstance(own, str) else (own or [])
        own_sim = float(sims[0]) if sims else float("nan")
        own_art = own[0].get("norm_articulation") if own else None

        wrong_sim = float("nan")
        if r["wrong_book"]:
            w, wsims = retriever.retrieve(emb[pos], r["wrong_book"],
                                          return_scores=True, top_k=1)
            wrong_sim = float(wsims[0]) if wsims else float("nan")

        rows.append({
            "own_sim": own_sim,
            "wrong_sim": wrong_sim,
            "nearest_own": own_art,
            # Concordance: the citation's nearest universe norm is the SAME
            # norm retrieval surfaced for the flow. Compared on articulation
            # text because that is the norm's identity in the universe json.
            "concordant": (own_art is not None
                           and r["top_articulation"] is not None
                           and str(own_art) == str(r["top_articulation"])),
        })
    scored = pd.concat([scored.reset_index(drop=True),
                        pd.DataFrame(rows)], axis=1)
    scored["hit"] = scored["own_sim"] >= args.tau
    scored["hit_wrong"] = scored["wrong_sim"] >= args.tau

    per_cite = out_dir / "cited_norm_scores.parquet"
    scored.to_parquet(per_cite, index=False)

    # ---- summaries --------------------------------------------------------
    def summarise(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_citations": len(g),
            "hit": g["hit"].mean(),
            "hit_wrong": g["hit_wrong"].mean(),
            # The headline. `hit` alone rises with generic norm-ish phrasing;
            # only the margin is evidence about THIS universe.
            "hit_margin": g["hit"].mean() - g["hit_wrong"].mean(),
            "concordance": g["concordant"].mean(),
            "own_sim_median": g["own_sim"].median(),
            "wrong_sim_median": g["wrong_sim"].median(),
        })

    by_arm = scored.groupby("arm").apply(summarise, include_groups=False)
    by_arm_book = (scored.groupby(["arm", "gutenberg_id"])
                   .apply(summarise, include_groups=False))
    # Flows citing nothing are an arm-level behaviour, not a missing value —
    # a policy that stops citing would otherwise post a great hit rate on a
    # shrinking denominator.
    no_cite = (cites.assign(none=cites["cited"].isna())
               .groupby("arm")["none"].mean().rename("no_citation_rate"))
    by_arm = by_arm.join(no_cite)

    by_arm.to_csv(out_dir / "by_arm.csv")
    by_arm_book.to_csv(out_dir / "by_arm_book.csv")
    (out_dir / "params.json").write_text(json.dumps({
        "tau": args.tau, "universe": str(universe),
        "embedding_model": EMB_MODEL, "instruction": EMBED_INSTRUCTION,
        "labels": args.labels,
    }, indent=2) + "\n")

    print("\n=== by arm ===")
    print(by_arm.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\n[write] {per_cite}")
    print(f"[write] {out_dir / 'by_arm.csv'}")
    print(f"[write] {out_dir / 'by_arm_book.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
