import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Row 1106 — live norm retrieval from the *1984* universe

    This notebook takes the contextual-integrity information flow rendered in
    `slides/google/normsim/figures/row1106_ci_flows.html` and runs the **real**
    norm-embedding retrieval pipeline against the *1984* (Orwell) normative
    universe it was extracted from — returning the **k = 3 nearest norms**.

    It reuses the production code paths verbatim so the result matches GRPO's
    `R_ground` retrieval:

    - `EMBED_INSTRUCTION` + `_build_norm_text` — `dagspaces/grpo_training/stages/norm_universe.py`
    - `_flow_to_query` / `_flatten_flow` — `dagspaces/grpo_training/stages/online_rground.py`
    - `NormRetriever.retrieve` cosine top-k — `dagspaces/grpo_training/stages/clients.py`

    **§8** then runs the *contrastive* half — the identical flow scored against
    the **wrong** book (*Pride and Prejudice*) — the mechanism behind GRPO's
    contrastive `R_ground` (`r_correct − λ·r_wrong`).

    The flow's CI tuple:
    *children → Thought Police* disclose *compromising information* about
    *parents*, under the transmission principle of *mandatory reporting and
    loyalty to the Party* (judged **appropriate** in-universe).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Configuration — paths, source book, embedding model""")
    return


@app.cell
def _():
    import os
    import sys

    # Repo root so `dagspaces.*` imports resolve when run from notebooks/.
    REPO_ROOT = "/share/pierson/matt/UAIR"
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    # The HTML figure this notebook reproduces.
    HTML_PATH = os.path.join(
        REPO_ROOT,
        "papers/colm26_normative-simulacra/slides/google/normsim/figures/"
        "row1106_ci_flows.html",
    )

    # Pre-built norm universe + per-book embeddings (Qwen3-Embedding-8B, 4096-d,
    # L2-normalized). 1984 = Project Gutenberg-style source_id "1984".
    NORM_UNIVERSE_DIR = os.path.join(
        REPO_ROOT,
        "multirun/2026-03-23_grpo_training/12-32-48/"
        "norm_universe_and_reward_prep/outputs/norm_universe",
    )
    SOURCE_ID = "1984"
    TOP_K = 3

    # Contrastive (wrong-universe) source for §8: Pride and Prejudice
    # (Jane Austen) = Project Gutenberg "1342".
    CONTRASTIVE_SOURCE_ID = "1342"

    # Same embedding model used to build the .npy matrices (see config.yaml).
    EMBEDDING_MODEL_PATH = os.environ.get(
        "EMBEDDING_MODEL_PATH",
        "/share/pierson/matt/zoo/models/Qwen3-Embedding-8B",
    )

    return (
        CONTRASTIVE_SOURCE_ID,
        EMBEDDING_MODEL_PATH,
        HTML_PATH,
        NORM_UNIVERSE_DIR,
        SOURCE_ID,
        TOP_K,
        os,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Parse the CI flow out of the HTML figure

    The figure encodes the 5-tuple in `.ci-tuple-cell` blocks and the flow
    metadata in `.field-row` blocks. We parse both and remap the display labels
    to the CI-schema field names the retrieval pipeline expects.
    """
    )
    return


@app.cell
def _(HTML_PATH):
    import html as _html
    import json

    from bs4 import BeautifulSoup

    with open(HTML_PATH, "r", encoding="utf-8") as _fh:
        _soup = BeautifulSoup(_fh.read(), "html.parser")

    # The 5-tuple cells (Subject / Sender / Recipient / Info Type / Transmission).
    _tuple_cells = {}
    for _cell in _soup.select(".ci-tuple-cell"):
        _label = _cell.select_one(".cell-label")
        _value = _cell.select_one(".cell-value")
        if _label and _value:
            _tuple_cells[_label.get_text(strip=True)] = _value.get_text(strip=True)

    # The flow-context rows (Snippet / Context / Direction / Appropriateness / ...).
    _field_rows = {}
    for _row in _soup.select(".field-row"):
        _label = _row.select_one(".field-label")
        _value = _row.select_one(".field-value")
        if _label and _value:
            _field_rows[_label.get_text(strip=True)] = _value.get_text(strip=True)

    # Remap to the CI-schema field names used by _flow_to_query.
    raw_flow = {
        "subject": _tuple_cells.get("Subject"),
        "sender": _tuple_cells.get("Sender"),
        "recipient": _tuple_cells.get("Recipient"),
        "information_type": _tuple_cells.get("Info Type"),
        "transmission_principle": _tuple_cells.get("Transmission"),
        "context": _field_rows.get("Context"),
        "appropriateness": _field_rows.get("Appropriateness"),
        "snippet": _field_rows.get("Snippet"),
    }

    # norms_invoked is rendered as a JSON array string; parse it back to a list.
    _invoked_raw = _field_rows.get("Norms Invoked", "[]")
    try:
        raw_flow["norms_invoked"] = json.loads(_html.unescape(_invoked_raw))
    except (json.JSONDecodeError, TypeError):
        raw_flow["norms_invoked"] = [_invoked_raw] if _invoked_raw else []

    raw_flow
    return json, raw_flow


@app.cell(hide_code=True)
def _(mo, raw_flow):
    _t = raw_flow
    mo.md(
        f"""
    | CI field | value |
    |---|---|
    | **subject** | {_t['subject']} |
    | **sender** | {_t['sender']} |
    | **recipient** | {_t['recipient']} |
    | **information_type** | {_t['information_type']} |
    | **transmission_principle** | {_t['transmission_principle']} |
    | **context** | {_t['context']} |
    | **appropriateness** | {_t['appropriateness']} |
    | **norms_invoked** | {_t['norms_invoked']} |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Build the retrieval query (production `_flow_to_query`)

    Concatenates sender, recipient, information_type, context,
    transmission_principle, subject, then any invoked norms — exactly as the
    online R_ground stage does before embedding.
    """
    )
    return


@app.cell
def _(raw_flow):
    from dagspaces.grpo_training.stages.online_rground import (
        _flatten_flow,
        _flow_to_query,
    )

    # _flatten_flow merges a nested {"flow": {...}} tuple with metadata; our
    # parsed dict is already flat, so it passes through unchanged.
    flat_flow = _flatten_flow(raw_flow)
    query_text = _flow_to_query(flat_flow)
    query_text
    return (query_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Load the *1984* normative universe + pre-computed embeddings

    589 norms, each a `(4096,)` L2-normalized Qwen3-Embedding-8B vector built by
    the `norm_universe` stage. Row order in `1984.npy` aligns 1:1 with the norm
    list in `norm_universes.json["1984"]`.
    """
    )
    return


@app.cell
def _(NORM_UNIVERSE_DIR, SOURCE_ID, json, os):
    import numpy as np

    with open(os.path.join(NORM_UNIVERSE_DIR, "norm_universes.json")) as _fh:
        _universes = json.load(_fh)
    norms_1984 = _universes[SOURCE_ID]

    norm_embeddings = np.load(
        os.path.join(NORM_UNIVERSE_DIR, "embeddings", f"{SOURCE_ID}.npy")
    )
    assert norm_embeddings.shape[0] == len(norms_1984), (
        f"embedding/norm count mismatch: {norm_embeddings.shape[0]} "
        f"vs {len(norms_1984)}"
    )
    f"{len(norms_1984)} norms, embeddings {norm_embeddings.shape} ({norm_embeddings.dtype})"
    return norm_embeddings, norms_1984, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Embed the query into the same space

    Loads Qwen3-Embedding-8B via `SentenceTransformer` (CUDA if available, else
    CPU) and applies the identical `EMBED_INSTRUCTION` prefix + L2 normalization
    used when the norm matrix was built — so the query lands in the same space.
    """
    )
    return


@app.cell
def _(EMBEDDING_MODEL_PATH, np, query_text):
    import torch
    from sentence_transformers import SentenceTransformer

    from dagspaces.grpo_training.stages.norm_universe import EMBED_INSTRUCTION

    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _embed_model = SentenceTransformer(
        EMBEDDING_MODEL_PATH,
        device=_device,
        tokenizer_kwargs={"padding_side": "left"},
    )

    query_embedding = _embed_model.encode(
        [EMBED_INSTRUCTION + query_text],
        normalize_embeddings=True,
    )[0].astype(np.float32)

    del _embed_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    f"query embedding {query_embedding.shape}, device used: {_device}"
    return (query_embedding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6. Retrieve the k = 3 nearest norms (production `NormRetriever`)

    Cosine similarity = matrix–vector product (both sides L2-normalized), then
    top-k by descending similarity — identical to `NormRetriever.retrieve`.
    """
    )
    return


@app.cell
def _(
    NORM_UNIVERSE_DIR,
    SOURCE_ID,
    TOP_K,
    json,
    norms_1984,
    os,
    query_embedding,
):
    from dagspaces.grpo_training.stages.clients import NormRetriever

    retriever = NormRetriever(
        norm_universes={SOURCE_ID: norms_1984},
        embeddings_dir=os.path.join(NORM_UNIVERSE_DIR, "embeddings"),
        embedding_client=None,  # use the pre-computed .npy matrix
        top_k=TOP_K,
    )

    _result_json, top_sims = retriever.retrieve(
        query_embedding,
        source_id=SOURCE_ID,
        return_scores=True,
        top_k=TOP_K,
    )
    retrieved_norms = json.loads(_result_json)
    list(zip(top_sims, [n["norm_articulation"] for n in retrieved_norms]))
    return retrieved_norms, top_sims


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 7. Results — the 3 nearest *1984* norms to row 1106's flow""")
    return


@app.cell(hide_code=True)
def _(mo, query_text, retrieved_norms, top_sims):
    _md = [f"**Query:** `{query_text}`\n"]
    for _rank, (_sim, _norm) in enumerate(zip(top_sims, retrieved_norms), start=1):
        _gov = _norm.get("governs_info_flow")
        _flow_note = _norm.get("info_flow_note") or "—"
        _md.append(
            f"""
### #{_rank} · cosine = {_sim:.4f}

> *{_norm.get('norm_articulation')}*

| | |
|---|---|
| prescriptive_element | {_norm.get('prescriptive_element')} |
| norm_subject | {_norm.get('norm_subject')} |
| norm_act | {_norm.get('norm_act')} |
| condition_of_application | {_norm.get('condition_of_application')} |
| normative_force | {_norm.get('normative_force')} |
| context | {_norm.get('context')} |
| governs_info_flow | {_gov} |
| info_flow_note | {_flow_note} |
| confidence | {_norm.get('confidence_qual')} ({_norm.get('confidence_quant')}) |
"""
        )
    mo.md("\n".join(_md))
    return


@app.cell(hide_code=True)
def _(mo, np, norm_embeddings, query_embedding, retrieved_norms, top_sims):
    # Sanity: recompute the full similarity distribution to confirm the top-k
    # really are the maxima and to show where they sit in the universe.
    _sims = norm_embeddings @ query_embedding
    mo.md(
        f"""
    **Retrieval sanity check** — over all {len(_sims)} norms in the *1984* universe:

    - max cosine = `{float(_sims.max()):.4f}`  (matches top hit `{top_sims[0]:.4f}`)
    - mean cosine = `{float(_sims.mean()):.4f}`,  std = `{float(_sims.std()):.4f}`
    - the top-{len(retrieved_norms)} retrieved sims `{[round(s, 4) for s in top_sims]}`
      are the {len(retrieved_norms)} largest values in the distribution.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 8. Contrastive retrieval — the *same* flow against *Pride and Prejudice*

    GRPO's grounding reward is **contrastive**: every flow is scored against its
    true book's norms *and* a random **wrong** book's norms, and rewarded for
    matching the right universe more than the wrong one —
    `R_ground = clamp(r_correct − λ · r_wrong, 0, 1)`. `NormRetriever.retrieve`
    takes a `contrastive_source` argument for exactly this.

    Here we run the wrong-universe half: the **identical** 1984 query embedding
    from §5, retrieved against Jane Austen's *Pride and Prejudice*
    (Gutenberg #1342, 666 norms). A faithful model's 1984 flow should match its
    own *1984* universe far better than Austen's drawing-room norms — so the top
    P&P cosines should sit well below the §6 1984 cosines. **That gap is the
    contrastive signal.**
    """
    )
    return


@app.cell
def _(CONTRASTIVE_SOURCE_ID, NORM_UNIVERSE_DIR, json, np, os):
    # Pride and Prejudice universe, mirroring the §4 load of 1984.
    with open(os.path.join(NORM_UNIVERSE_DIR, "norm_universes.json")) as _fh:
        _universes = json.load(_fh)
    norms_pp = _universes[CONTRASTIVE_SOURCE_ID]

    pp_embeddings = np.load(
        os.path.join(NORM_UNIVERSE_DIR, "embeddings", f"{CONTRASTIVE_SOURCE_ID}.npy")
    )
    assert pp_embeddings.shape[0] == len(norms_pp), (
        f"embedding/norm count mismatch: {pp_embeddings.shape[0]} "
        f"vs {len(norms_pp)}"
    )
    f"Pride and Prejudice: {len(norms_pp)} norms, embeddings {pp_embeddings.shape}"
    return (norms_pp,)


@app.cell
def _(
    CONTRASTIVE_SOURCE_ID,
    NORM_UNIVERSE_DIR,
    SOURCE_ID,
    TOP_K,
    json,
    norms_1984,
    norms_pp,
    os,
    query_embedding,
):
    from dagspaces.grpo_training.stages.clients import NormRetriever as _NormRetriever

    # One retriever indexing BOTH universes — the production setup, where the
    # retriever holds every book and `contrastive_source` selects which universe
    # a query is scored against. (Loads 1984.npy + 1342.npy from disk.)
    contrastive_retriever = _NormRetriever(
        norm_universes={SOURCE_ID: norms_1984, CONTRASTIVE_SOURCE_ID: norms_pp},
        embeddings_dir=os.path.join(NORM_UNIVERSE_DIR, "embeddings"),
        embedding_client=None,
        top_k=TOP_K,
    )

    # Same 1984 query, retrieved from the WRONG universe via contrastive_source.
    _pp_json, contrast_sims = contrastive_retriever.retrieve(
        query_embedding,
        source_id=SOURCE_ID,
        contrastive_source=CONTRASTIVE_SOURCE_ID,
        return_scores=True,
        top_k=TOP_K,
    )
    contrast_norms = json.loads(_pp_json)
    list(zip(contrast_sims, [n["norm_articulation"] for n in contrast_norms]))
    return contrast_norms, contrast_sims


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### The 3 nearest *Pride and Prejudice* norms (wrong universe)""")
    return


@app.cell(hide_code=True)
def _(contrast_norms, contrast_sims, mo):
    _md = []
    for _rank, (_sim, _norm) in enumerate(zip(contrast_sims, contrast_norms), start=1):
        _gov = _norm.get("governs_info_flow")
        _flow_note = _norm.get("info_flow_note") or "—"
        _md.append(
            f"""
#### #{_rank} · cosine = {_sim:.4f}

> *{_norm.get('norm_articulation')}*

| | |
|---|---|
| prescriptive_element | {_norm.get('prescriptive_element')} |
| norm_subject | {_norm.get('norm_subject')} |
| norm_act | {_norm.get('norm_act')} |
| context | {_norm.get('context')} |
| governs_info_flow | {_gov} |
| info_flow_note | {_flow_note} |
"""
        )
    mo.md("\n".join(_md))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Side-by-side — correct (*1984*) vs contrastive (*P&P*)""")
    return


@app.cell(hide_code=True)
def _(contrast_norms, contrast_sims, mo, np, retrieved_norms, top_sims):
    _k = max(len(top_sims), len(contrast_sims))
    _lines = [
        "| rank | *1984* cos | *1984* norm | *P&P* cos | *P&P* norm |",
        "|---|---|---|---|---|",
    ]
    for _i in range(_k):
        _cs = f"{top_sims[_i]:.4f}" if _i < len(top_sims) else "—"
        _ws = f"{contrast_sims[_i]:.4f}" if _i < len(contrast_sims) else "—"
        _cn = (retrieved_norms[_i]["norm_articulation"][:50] + "…"
               if _i < len(retrieved_norms) else "")
        _wn = (contrast_norms[_i]["norm_articulation"][:50] + "…"
               if _i < len(contrast_norms) else "")
        _lines.append(f"| {_i + 1} | {_cs} | {_cn} | {_ws} | {_wn} |")

    _r_correct = float(np.mean(top_sims))
    _r_wrong = float(np.mean(contrast_sims))
    _lines += [
        "",
        f"- mean top-{len(top_sims)} cosine — **1984 (correct): {_r_correct:.4f}**, "
        f"**P&P (wrong): {_r_wrong:.4f}**",
        f"- **contrastive margin (correct − wrong): {_r_correct - _r_wrong:+.4f}** "
        f"— the raw retrieval gap that GRPO's `R_ground` rewards (before the λ "
        f"weight and the LLM judge). A positive margin means the flow is better "
        f"explained by its own *1984* universe than by Austen's.",
    ]
    mo.md("\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()
