import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Norm-yield gap: qwen3.6-27b vs. qwen2.5-72b-awq on fiction10

    Built 2026-05-26. The 2026-05-23 norm-extraction run on `chunks_top100_fiction_en`
    (qwen3.6-27b/instruct) produced **222 norms/book × 97 books = 21,550 norms** —
    roughly 1/5 of the **1,187 norms/book** baseline observed on
    `fiction10/abstracted_norms.parquet` (qwen2.5-72b-awq).

    Possible explanations:

    1. **Chunking changed** — earlier diagnosis blamed a 1,800 → 6,000 char chunk-size
       shift. *Falsified* by inspection: the aggregated `abstracted_norms.parquet`
       uses 6,000-char chunks for all 10 books (the 1,800-char files under
       `by_book_norms/` are legacy). Both runs use the same chunking.
    2. **Prompt / sampling changed** — user reports no change.
    3. **Model behaviour at the `has_norms` gate** — qwen3.6-27b says "no
       norms" on chunks that qwen2.5-72b would have flagged.

    This notebook isolates (3) by comparing the two extractions on the
    intersection of books between the fiction10 set and the current top100
    extraction.

    **Book inventory:**

    | source | books |
    |---|---|
    | fiction10 (reference) | 10 |
    | top100 fiction (current) | 100 |
    | intersection | 7 — *not* 10 |

    The 3 fiction10 books missing from `chunks_top100_fiction_en` are:

    | gutenberg_id | title | author | reason |
    |---|---|---|---|
    | 1984 | Nineteen Eighty-Four | Orwell | not in chunks_top100 *and* not in chunks_top1000 (likely custom add to fiction10) |
    | 4078 | The Picture of Dorian Gray | Wilde | in top1000, dropped from top100 by download_count cut |
    | 541 | The Age of Innocence | Wharton | in top1000, dropped from top100 by download_count cut |

    The comparison below uses the 7 overlapping books. For the missing 3, we'd
    have to re-extract from raw text under qwen3.6-27b (raw text is cached for
    4078 and 541; 1984 has no raw text in the gutenberg cache).
    """)
    return


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
    NB_DIR = PROJECT_ROOT / "notebooks/normative-simulacra"

    REF_ABSTRACTED = Path(
        "/share/pierson/matt/n2s4cir/data/fiction10/abstracted_norms.parquet"
    )
    CUR_RUN = Path(
        "/share/pierson/matt/UAIR/multirun/2026-05-23_historical_norms/15-23-59/0/COLM_norms_fiction_qwen36"
    )
    CUR_REASONING = CUR_RUN / "outputs/reasoning/reasoning.parquet"
    CUR_STRUCTURED = CUR_RUN / "outputs/extraction/structured_norms.parquet"
    TOP100_CHUNKS = Path(
        "/share/pierson/matt/zoo/datasets/gutenberg_cache/chunks_top100_fiction_en.parquet"
    )

    pd.set_option("display.max_columns", 80)
    pd.set_option("display.float_format", "{:.2f}".format)

    sys.path.insert(0, str(NB_DIR.parent))
    try:
        from font_utils import load_ibm_plex_sans

        load_ibm_plex_sans()
    except Exception as _e:
        print(f"[font_utils] skipped: {_e}")
    return (
        CUR_REASONING,
        CUR_STRUCTURED,
        REF_ABSTRACTED,
        TOP100_CHUNKS,
        mticker,
        np,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Load datasets

    - `ref` — `fiction10/abstracted_norms.parquet` (qwen2.5-72b-awq, post role-abstraction). One row per extracted norm.
    - `cur_reasoning` — 2026-05-23 run, reasoning stage output (one row per chunk; `has_norms` field).
    - `cur_structured` — 2026-05-23 run, extraction stage output (one row per structured norm).
    - `top100_chunks` — input parquet to the current run; supplies the *total* chunk count per book (the extraction stage drops `has_norms=False` chunks, so we need this to recover the denominator).
    """)
    return


@app.cell
def _(
    CUR_REASONING,
    CUR_STRUCTURED,
    REF_ABSTRACTED,
    TOP100_CHUNKS,
    pd,
):
    ref = pd.read_parquet(REF_ABSTRACTED)
    ref["gutenberg_id"] = ref["gutenberg_id"].astype(str)

    cur_reasoning = pd.read_parquet(CUR_REASONING)
    cur_reasoning["gutenberg_id"] = cur_reasoning["gutenberg_id"].astype(str)

    cur_structured = pd.read_parquet(CUR_STRUCTURED)
    cur_structured["gutenberg_id"] = cur_structured["gutenberg_id"].astype(str)

    top100_chunks = pd.read_parquet(TOP100_CHUNKS)
    top100_chunks["gutenberg_id"] = top100_chunks["gutenberg_id"].astype(str)

    print(
        f"ref (fiction10 abstracted): {len(ref):,} rows, {ref['gutenberg_id'].nunique()} books"
    )
    print(
        f"cur_reasoning (qwen3.6 reasoning): {len(cur_reasoning):,} rows (exploded), "
        f"{cur_reasoning['gutenberg_id'].nunique()} books, "
        f"{cur_reasoning[['gutenberg_id','chunk_id']].drop_duplicates().shape[0]} unique chunks"
    )
    print(
        f"cur_structured (qwen3.6 extracted): {len(cur_structured):,} rows, "
        f"{cur_structured['gutenberg_id'].nunique()} books"
    )
    print(
        f"top100_chunks (current source): {len(top100_chunks):,} chunks, "
        f"{top100_chunks['gutenberg_id'].nunique()} books"
    )
    return cur_reasoning, cur_structured, ref, top100_chunks


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Confirm chunking is identical

    If chunk_size distributions differ between `ref` and `cur`, the rest of the
    comparison is invalid — different chunk sizes produce different
    norm-density baselines. Verify by inspecting per-book chunk_size median
    and max from both sources.
    """)
    return


@app.cell
def _(cur_structured, pd, ref, top100_chunks):
    ref_chunking = (
        ref.groupby("gutenberg_id")
        .agg(
            ref_chunk_size_med=("chunk_size", "median"),
            ref_chunk_size_max=("chunk_size", "max"),
            ref_chunk_id_max=("chunk_id", "max"),
        )
        .reset_index()
    )
    cur_chunking = (
        top100_chunks.groupby("gutenberg_id")
        .agg(
            cur_chunk_size_med=("chunk_size", "median"),
            cur_chunk_size_max=("chunk_size", "max"),
            cur_chunk_id_max=("chunk_id", "max"),
        )
        .reset_index()
    )
    chunking_cmp = ref_chunking.merge(cur_chunking, on="gutenberg_id", how="inner")
    chunking_cmp["delta_med"] = (
        chunking_cmp["cur_chunk_size_med"] - chunking_cmp["ref_chunk_size_med"]
    )
    print(
        "Per-book chunk_size comparison (only books present in BOTH ref and "
        "current top100):"
    )
    print(
        chunking_cmp.sort_values("gutenberg_id").to_string(index=False)
    )
    return (chunking_cmp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Per-book norm yield

    For each overlapping book, compute:

    - `ref_norms` — count from `abstracted_norms.parquet` (qwen2.5-72b-awq)
    - `ref_chunks_with_norms` — distinct chunks in ref (≈ chunks the reference model flagged `has_norms=True`)
    - `cur_total_chunks` — total chunks in current run (from `top100_chunks`)
    - `cur_has_norms_chunks` — distinct chunks marked `has_norms=True` by qwen3.6-27b
    - `cur_has_rate` — `cur_has_norms_chunks / cur_total_chunks`
    - `cur_norms` — extracted-norm count from qwen3.6-27b
    - `*_norms_per_chunk` — norm density on chunks that *did* yield norms

    The key signal: if `*_norms_per_chunk` is similar but `*_has_rate` differs,
    the entire gap is at the gate, not in the per-chunk extraction depth.
    """)
    return


@app.cell
def _(cur_reasoning, cur_structured, pd, ref):
    overlap_books = sorted(
        set(ref["gutenberg_id"]) & set(cur_reasoning["gutenberg_id"]), key=int
    )

    rows = []
    for b in overlap_books:
        ref_b = ref[ref["gutenberg_id"] == b]
        rea_b = cur_reasoning[cur_reasoning["gutenberg_id"] == b]
        cur_b = cur_structured[cur_structured["gutenberg_id"] == b]

        ref_norms = len(ref_b)
        ref_chunks_with_norms = ref_b[["chunk_id"]].drop_duplicates().shape[0]

        cur_total_chunks = rea_b[["chunk_id"]].drop_duplicates().shape[0]
        cur_has_norms = (
            rea_b[rea_b["has_norms"] == True][["chunk_id"]].drop_duplicates().shape[0]
        )
        cur_norms = len(cur_b)
        cur_chunks_with_norms = cur_b[["chunk_id"]].drop_duplicates().shape[0]

        rows.append(
            {
                "book": b,
                "title": ref_b["book_title"].iloc[0],
                "REF_norms": ref_norms,
                "REF_chunks_w_norms": ref_chunks_with_norms,
                "REF_norms_per_chunk": ref_norms / max(ref_chunks_with_norms, 1),
                "CUR_total_chunks": cur_total_chunks,
                "CUR_has_norms_chunks": cur_has_norms,
                "CUR_has_rate": cur_has_norms / max(cur_total_chunks, 1),
                "CUR_norms": cur_norms,
                "CUR_chunks_w_norms": cur_chunks_with_norms,
                "CUR_norms_per_chunk": cur_norms / max(cur_chunks_with_norms, 1),
                "CUR_REF_norm_ratio": cur_norms / max(ref_norms, 1),
            }
        )
    per_book = pd.DataFrame(rows)
    print(f"Overlap books used: {len(per_book)} of 10 fiction10 books")
    print(per_book.to_string(index=False))
    return overlap_books, per_book


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Aggregate totals on the overlap

    Roll up the per-book table into headline numbers. Key ratios to read:

    - **CUR/REF norm ratio** — if model is unchanged, ≈ 1.0; observed value tells us the magnitude of the gap.
    - **REF inferred has_rate** — `ref_chunks_with_norms / cur_total_chunks`. Valid only because chunking is identical, so the reference run *would have* operated on the same denominator if it had been re-run on the current chunks parquet.
    - **CUR has_rate** — direct from the qwen3.6-27b reasoning stage.
    - **Norms-per-chunk gap** — `CUR_norms_per_chunk / REF_norms_per_chunk`. Tells us whether the per-chunk extraction depth changed.

    If `has_rate` gap fully explains `CUR/REF norm ratio` (with norms-per-chunk
    ≈ unchanged), the bottleneck is the gate, not the structurer.
    """)
    return


@app.cell
def _(per_book):
    tot_ref_norms = per_book["REF_norms"].sum()
    tot_cur_norms = per_book["CUR_norms"].sum()
    tot_ref_chunks_with_norms = per_book["REF_chunks_w_norms"].sum()
    tot_cur_has_norms = per_book["CUR_has_norms_chunks"].sum()
    tot_cur_total_chunks = per_book["CUR_total_chunks"].sum()

    ref_inferred_has_rate = tot_ref_chunks_with_norms / max(tot_cur_total_chunks, 1)
    cur_has_rate = tot_cur_has_norms / max(tot_cur_total_chunks, 1)
    ref_npc = tot_ref_norms / max(tot_ref_chunks_with_norms, 1)
    cur_npc = tot_cur_norms / max(tot_cur_has_norms, 1)

    print(f"=== TOTALS on {len(per_book)} overlapping books ===")
    print(f"REF (qwen2.5-72b-awq) norms:            {tot_ref_norms:>7,}")
    print(f"CUR (qwen3.6-27b)     norms:            {tot_cur_norms:>7,}")
    print(f"CUR / REF norm ratio:                   {tot_cur_norms/tot_ref_norms:>7.3f}")
    print()
    print(f"Total chunks (top100 source):           {tot_cur_total_chunks:>7,}")
    print(f"REF chunks_with_norms:                  {tot_ref_chunks_with_norms:>7,}")
    print(f"CUR has_norms chunks:                   {tot_cur_has_norms:>7,}")
    print(
        f"REF inferred has_rate (vs same chunks): {ref_inferred_has_rate:>7.1%}"
    )
    print(f"CUR has_rate:                           {cur_has_rate:>7.1%}")
    print(
        f"Gate ratio CUR/REF:                     {cur_has_rate/ref_inferred_has_rate:>7.3f}"
    )
    print()
    print(f"REF norms / yes-chunk:                  {ref_npc:>7.2f}")
    print(f"CUR norms / yes-chunk:                  {cur_npc:>7.2f}")
    print(f"Density ratio CUR/REF:                  {cur_npc/ref_npc:>7.3f}")
    print()
    print("Decomposition: CUR/REF norm ratio = (gate ratio) × (density ratio)")
    print(
        f"  {tot_cur_norms/tot_ref_norms:.3f}  ≈  "
        f"{cur_has_rate/ref_inferred_has_rate:.3f}  ×  {cur_npc/ref_npc:.3f}  =  "
        f"{(cur_has_rate/ref_inferred_has_rate) * (cur_npc/ref_npc):.3f}"
    )
    return cur_has_rate, ref_inferred_has_rate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Visual — per-book yield gap

    Three side-by-side bar plots:

    - **Left**: total norms per book (REF vs CUR) — log scale to span 100→3000.
    - **Center**: has_norms rate per book — REF inferred (`ref_chunks_w_norms / cur_total_chunks`) vs CUR direct.
    - **Right**: norms per yes-chunk — the depth metric that should be similar across models if the prompt/sampling are unchanged.
    """)
    return


@app.cell
def _(np, per_book, plt):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    pb = per_book.copy()
    pb["short"] = pb["title"].str.slice(0, 30)
    pb = pb.sort_values("REF_norms", ascending=True)
    y = np.arange(len(pb))
    bar_h = 0.38

    ax = axes[0]
    ax.barh(y - bar_h / 2, pb["REF_norms"], bar_h, label="REF (qwen2.5-72b)", color="#1f77b4")
    ax.barh(y + bar_h / 2, pb["CUR_norms"], bar_h, label="CUR (qwen3.6-27b)", color="#ff7f0e")
    ax.set_yticks(y)
    ax.set_yticklabels(pb["short"], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("norms (log)")
    ax.set_title("Norms per book")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
    ref_rate = pb["REF_chunks_w_norms"] / pb["CUR_total_chunks"]
    cur_rate = pb["CUR_has_norms_chunks"] / pb["CUR_total_chunks"]
    ax.barh(y - bar_h / 2, ref_rate, bar_h, label="REF inferred", color="#1f77b4")
    ax.barh(y + bar_h / 2, cur_rate, bar_h, label="CUR direct", color="#ff7f0e")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("has_norms rate")
    ax.set_title("Reasoning-gate pass rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[2]
    ax.barh(
        y - bar_h / 2,
        pb["REF_norms_per_chunk"],
        bar_h,
        label="REF",
        color="#1f77b4",
    )
    ax.barh(
        y + bar_h / 2,
        pb["CUR_norms_per_chunk"],
        bar_h,
        label="CUR",
        color="#ff7f0e",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("norms / yes-chunk")
    ax.set_title("Per-chunk extraction depth")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Spot-check: chunks REF would have flagged but CUR did not

    The has_rate gap is the candidate explanation. We can sanity-check whether
    qwen3.6-27b's "no norms" calls are correct by looking at chunks where:

    - REF (qwen2.5-72b) extracted ≥1 norm (chunk_id present in `ref`)
    - CUR (qwen3.6-27b) marked `has_norms=False`

    Caveat: REF's "chunks_with_norms" comes from the *abstracted* parquet,
    which is rows-per-norm. We can infer the chunk IDs but cannot recover
    *which* chunks REF rejected (those were filtered out of the abstracted
    output). So the comparison below is asymmetric — REF positives vs CUR
    rejections — which is exactly what we want for spot-checking the gate.
    """)
    return


@app.cell
def _(cur_reasoning, pd, ref):
    ref_chunk_keys = set(
        zip(
            ref["gutenberg_id"].astype(str),
            ref["chunk_id"].astype(int),
        )
    )
    cur_no_norms = cur_reasoning[cur_reasoning["has_norms"] == False].copy()
    cur_no_norms["key"] = list(
        zip(
            cur_no_norms["gutenberg_id"].astype(str),
            cur_no_norms["chunk_id"].astype(int),
        )
    )
    disputed = cur_no_norms[cur_no_norms["key"].isin(ref_chunk_keys)].copy()
    print(
        f"Disputed chunks (REF says 'has norms', CUR says 'no norms'): "
        f"{len(disputed):,}"
    )
    print(
        f"  / out of {len(cur_no_norms):,} CUR no_norms calls overall on overlap books"
    )
    disputed_by_book = (
        disputed.groupby("gutenberg_id").size().reset_index(name="n_disputed")
    )
    print("\nDisputed chunk count per book:")
    print(disputed_by_book.to_string(index=False))
    return (disputed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sample five disputed chunks. Read the `article_text` (longer excerpt — 1,500
    chars) and judge: *would you say there's a prescriptive/normative claim
    here?*

    Note: the current run uses `thinking_mode: off`, so the model returns a
    bare `{"norms": [], "has_prescriptive_content": false}` with no rationale —
    `generated_reasoning` is empty and `reasoning_trace` is null. The judgment
    has to be made from the article text alone.

    - If most look obviously normative → qwen3.6-27b's gate is mis-calibrated; consider prompt-tuning, enabling thinking, or model swap.
    - If most look descriptive or ambiguous → qwen3.6-27b is the better discriminator; the prior 1,400 norms/book number was inflated.
    """)
    return


@app.cell
def _(disputed):
    sample = disputed.sample(
        n=min(5, len(disputed)), random_state=42
    ).reset_index(drop=True)
    for i, row in sample.iterrows():
        print("=" * 80)
        print(
            f"[{i+1}/{len(sample)}] book={row['gutenberg_id']} ({row.get('book_title','?')}) "
            f"chunk_id={row['chunk_id']}"
        )
        print(f"CUR generated_text: {row.get('generated_text','')!r}")
        print("-" * 80)
        print("ARTICLE_TEXT (first 1,500 chars):")
        print(str(row["article_text"])[:1500])
        print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Summary

    Run the cells above and inspect §4 in particular. The decomposition

    ```
    CUR / REF norm ratio  =  (gate ratio)  ×  (density ratio)
    ```

    isolates whether the gap is at the `has_norms` gate or in per-chunk
    extraction depth. The §6 spot-check tells you whether the gate is
    operating sensibly or whether qwen3.6-27b is under-flagging.

    **What this notebook does NOT do:**

    - Re-extract for the 3 missing fiction10 books (1984, 4078, 541) under
      qwen3.6-27b. Raw text is cached for 4078 and 541 but not 1984.
    - Compare role-abstracted norms (current run hasn't finished its
      abstraction stage at notebook authorship time).
    - Make a quality judgement on the extracted norms themselves.
    """)
    return


if __name__ == "__main__":
    app.run()
