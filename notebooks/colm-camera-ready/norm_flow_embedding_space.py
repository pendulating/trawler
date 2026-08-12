import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Norm / flow embedding space — Gemma-4 lineage (camera-ready)

    Built 2026-07-31 for the COLM 2026 camera-ready. Supersedes
    `notebooks/COLM26/norm_simulacra/norm_flow_embedding_space.ipynb`, which
    produced the current main-paper figure
    (`figures/norm_flow_per_book_umap.pdf`, `\autoref{fig:norm-flow-per-book-umap}`)
    off `/share/pierson/matt/n2s4cir/data/fiction10` — a **stale
    Qwen2.5-72B-AWQ-era extraction produced under the wrong prompts** (the
    2026-07-12 prompt-wiring fix). Everything below is recomputed on the
    canonical Gemma-4-31B-it fiction10 artifacts, the same ones
    `norm_distribution_top100_vs_fiction10.py` and
    `corpus_descriptives_two_corpora.py` read.

    **The unit of analysis is the source text.** Every pairing, null, and
    displacement statistic below is computed within a book: for each CI flow we
    retrieve its nearest norm from the *same novel*, and every null re-draws that
    pairing from the *same novel's* norms.

    Figures:

    1. **`fig_norm_flow_per_book_umap`** — per-book small multiples, the
       main-paper figure. Same design as the accepted version, new data.
    2. **`fig_paired_displacement`** — the corpus-level paired-displacement
       figure: each flow joined by a line to its nearest same-book norm.
    3. **`fig_paired_displacement_per_book[_umap]`** — the same construction as
       small multiples, one panel per novel, on the linear PCA basis and on
       UMAP. Norms are split by whether any flow ever retrieves them.
    4. **`fig_paired_displacement_per_book[_umap]_governs`** — the same two
       figures over the **information-flow-governing norm subset only**, which
       is the pool production actually retrieves against.
    5. **`fig_displacement_concentration`** — the quantitative companion: how
       concentrated the difference vectors
       $d_i = e_{\text{norm}} - e_{\text{flow}}$ are about their mean, against
       a shuffled-within-book null, and whether the ten novels share one
       displacement direction.

    Figures → `figures/norm_flow_embedding/`, tables →
    `tables/norm_flow_embedding/`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data provenance

    | artifact | path | produced | extractor | prompts |
    |---|---|---|---|---|
    | norms | `outputs/2026-07-12_fiction10_norms_gemma4/18-36-28/COLM_norms_fiction_gemma4/outputs/extraction/structured_norms.parquet` | 2026-07-12 | `gemma-4-31b/instruct` | `norm_extraction_fiction` |
    | flows | `outputs/2026-07-12_fiction10_flows_gemma4/23-14-17/COLM_flows_fiction_gemma4/outputs/ci_extraction/ci_flows.parquet` | 2026-07-12 | `gemma-4-31b/instruct` | `ci_extraction_fiction` |

    Embeddings are built by `scripts/embed_camera_ready_norms_flows.py` against
    the standing Qwen3-Embedding-8B vLLM server
    (`scripts/embedding_server.sub`), cached under
    `outputs/camera_ready/embeddings/`. Norm serialization is byte-identical to
    production (`norm_universe._build_norm_text`), and exact-duplicate norms are
    dropped per book before embedding, per methods §3.

    ### Flow serialization: stripping the normative vocabulary

    A flow record carries deontic content that is not part of *what was
    transmitted*, and writing it into the embedded string would let the
    norm/flow separation be partly an artifact of our own template. Three
    serializations are built so this is measured rather than argued:

    | variant | string | note |
    |---|---|---|
    | `full` | CI tuple + transmission principle + `This flow is considered {appropriate/inappropriate/ambiguous}.` | what `scripts/embed_norms_and_flows.py` produced, hence what the paper's existing figure rests on |
    | **`noappr`** | CI tuple + transmission principle | **production parity** — `online_rground._flow_to_query` and `DirectChunkGold.MATCH_FIELDS` both build retrieval queries from sender/recipient/information_type/context/transmission_principle/subject and never from the appropriateness verdict |
    | `descriptive` | CI tuple only | the transmission principle is dropped too: in CI it *is* the normative constraint, and the extracted values are frankly deontic — discretion, social obligation, confidentiality, propriety, consent, coercion |

    `noappr` is the default everywhere below. `ci_norms_invoked` and
    `ci_norm_source` were never in any of the three strings.

    ### Two embedding instructions, and why the directional claims need the second

    Qwen3-Embedding-8B is instruction-aware, and production embeds the two
    constructs under **different** instructions — norms under "represent it for
    semantic matching with information flows", flows under "represent it for
    semantic comparison with other information flows". That asymmetry is correct
    for R-GROUND retrieval, but it contaminates both claims made here: a
    coherent component of every difference vector, and part of any measured
    separation, would be contributed by the differing prefixes alone.

    - **`shared`** — one instruction for both constructs. **The default.** Every
      figure and every headline number uses it.
    - **`prod`** — production instructions, reported alongside so the size of
      the confound is visible rather than hidden.

    ### Two norm pools

    | pool | norms | what it is |
    |---|---|---|
    | **`all`** | every norm with a non-empty articulation | the default, and what the paper's existing figure shows |
    | **`governs`** | `raz_governs_info_flow == True` only | the pool **production actually retrieves against** — the R-DIRECT index was restricted to information-flow-governing norms on 2026-07-28 |

    Every figure and statistic is produced for both pools; the `governs`
    variants carry a `_governs` suffix. Nothing else changes between them — same
    embeddings, same instruction, same flow serialization, same pairing rule,
    same nulls — so any difference is attributable to the pool.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")  # headless-safe; marimo renders figures regardless

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D

    PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
    NB_DIR = PROJECT_ROOT / "notebooks/colm-camera-ready"
    EMB_DIR = PROJECT_ROOT / "outputs/camera_ready/embeddings"
    FIG_DIR = NB_DIR / "figures/norm_flow_embedding"
    TAB_DIR = NB_DIR / "tables/norm_flow_embedding"
    CACHE_DIR = PROJECT_ROOT / "outputs/camera_ready/projections"
    for _d in (FIG_DIR, TAB_DIR, CACHE_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    PAPER_FIG_DIR = PROJECT_ROOT / "papers/colm26_normative-simulacra/figures"
    PAPER_TAB_DIR = PROJECT_ROOT / "papers/colm26_normative-simulacra/tables"

    CORPUS = "fiction10"

    # --- the choices every figure below inherits ---------------------------
    FLOW_TEXTS = ("full", "noappr", "descriptive")
    FLOW_TEXT = "noappr"   # production parity: no appropriateness verdict
    INSTR = "shared"       # one embedding instruction for both constructs
    NORM_SET_NAMES = ("all", "governs")

    # None = draw every pair. A fixed per-panel cap (this was 90) equalises
    # visual density across panels whose pair counts differ by 16x — Alice has
    # 211 pairs and Monte Cristo 3,479 — which is exactly the comparison the
    # small-multiples format exists to make, so it was throwing away the signal
    # to buy tidiness. Rasterised line collections render all 16,200 without
    # trouble. Set an int here to cap per panel, or a float for a fixed
    # fraction of each novel's pairs.
    SEG_PER_BOOK = None

    # Panel titles. Truncating `book_title` at a character budget produced
    # "Alice's Adventures in W…" and "The Picture of Dorian G…"; these are the
    # short forms a reader recognises.
    SHORT_TITLE = {
        "Alice's Adventures in Wonderland": "Alice's Adventures in Wonderland",
        "The Picture of Dorian Gray": "Dorian Gray",
        "The Count of Monte Cristo": "Monte Cristo",
        "Nineteen Eighty-Four": "1984",
        "The Age of Innocence": "The Age of Innocence",
        "Pride and Prejudice": "Pride and Prejudice",
        "Les Misérables": "Les Misérables",
        "Anna Karenina": "Anna Karenina",
        "Bleak House": "Bleak House",
        "Middlemarch": "Middlemarch",
    }

    # The camera-ready panel drops two novels. At COLM's 5.5in \textwidth a
    # 10-panel grid has to be built ~12in wide and scaled to 0.45x on the page,
    # which takes the corner annotations to ~2.6pt — invisible in print. Eight
    # panels fit a 4x2 grid drawn at final size, so nothing is downscaled and
    # every glyph is legible. Bleak House and Middlemarch are the two dropped:
    # both are mid-pack on every statistic the figure carries (R, frac_reached,
    # concentration), so neither is load-bearing for any claim. The full
    # ten-novel version is still rendered for the notebook and the appendix.
    DROPPED_FOR_PRINT = ("Bleak House", "Middlemarch")

    # Construct identity. Blue/orange are the colours of the accepted main-paper
    # figure, kept so the camera-ready version reads as the same figure with new
    # data rather than a redesign. Purple is new: it marks the norms some flow
    # actually retrieves, so "reached" and "never reached" are separable at a
    # glance. Purple also matches the segment colour, which makes the chain
    # flow → segment → retrieved norm read as one object.
    NORM_COLOR = "#4C72B0"        # norms no flow ever retrieves
    MATCH_COLOR = "#7B4FA8"       # norms retrieved by >= 1 flow
    FLOW_COLOR = "#DD8452"
    BG_COLOR = "#E8E8E8"
    LINK_COLOR = "#6E5B8A"        # displacement segments
    ARROW_COLOR = "#1F1233"
    NULL_COLOR = "#9A9A9A"

    pd.set_option("display.max_columns", 80)
    pd.set_option("display.float_format", "{:.4f}".format)

    # COLM camera-ready house style, matching the other camera-ready notebooks.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )

    # `pad_inches=0` (2026-08-05). `bbox_inches="tight"` crops to the drawn
    # content and then re-adds matplotlib's default 0.1in border on every side,
    # which lands in the PDF as margin no LaTeX float can reclaim: 0.2in of a
    # 5.5in column spent on nothing. The tight bbox is computed from the
    # rendered artists' extents, text included, so zero is a true crop and not
    # a clip. Pass a non-zero value back if a figure ever needs breathing room.
    def save_fig(fig, name, also_paper=False, pad_inches=0.0):
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300,
                        bbox_inches="tight", pad_inches=pad_inches)
        if also_paper:
            fig.savefig(PAPER_FIG_DIR / f"{name}.pdf", dpi=300,
                        bbox_inches="tight", pad_inches=pad_inches)
            print(f"[paper] {PAPER_FIG_DIR / name}.pdf")
        print(f"[fig] {FIG_DIR / name}.png|.pdf")

    def save_caption(name, title, caption, label, tags):
        out = FIG_DIR / f"{name}.json"
        out.write_text(
            json.dumps(
                {
                    "plot-title": title,
                    "plot-caption": caption,
                    "plot-latex-label": label,
                    "plot-tags": tags,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(f"[caption] {out}")

    def save_table(df, name, index=True):
        out = TAB_DIR / f"{name}.csv"
        df.to_csv(out, index=index)
        print(f"[table] {out}")

    def save_tex(body, name, also_paper=False):
        """Write a complete `\\input`-able LaTeX table.

        A whole `table` environment, not a row fragment: `\\input`-ing bare rows
        into an open `tabular` breaks, because the final row's `\\\\` scans past
        the file boundary and expands whatever follows where alignment material
        is illegal. `also_paper` mirrors `save_fig`'s switch — off by default, so
        running the notebook never silently edits the paper tree.
        """
        out = TAB_DIR / f"{name}.tex"
        out.write_text(body.rstrip("\n") + "\n")
        print(f"[latex] {out}")
        if also_paper:
            paper_out = PAPER_TAB_DIR / f"{name}.tex"
            paper_out.write_text(body.rstrip("\n") + "\n")
            print(f"[paper] {paper_out}")

    return (
        ARROW_COLOR,
        BG_COLOR,
        CACHE_DIR,
        CORPUS,
        DROPPED_FOR_PRINT,
        EMB_DIR,
        FLOW_COLOR,
        FLOW_TEXT,
        FLOW_TEXTS,
        INSTR,
        LINK_COLOR,
        Line2D,
        MATCH_COLOR,
        NORM_COLOR,
        NORM_SET_NAMES,
        NULL_COLOR,
        SEG_PER_BOOK,
        SHORT_TITLE,
        json,
        np,
        pd,
        plt,
        save_caption,
        save_fig,
        save_table,
        save_tex,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load embeddings and row-aligned metadata""")
    return


@app.cell
def _(CORPUS, DROPPED_FOR_PRINT, EMB_DIR, FLOW_TEXTS, np, pd):
    norms = pd.read_parquet(EMB_DIR / f"{CORPUS}_norms_meta.parquet")
    flows = pd.read_parquet(EMB_DIR / f"{CORPUS}_flows_meta.parquet")
    norms["gutenberg_id"] = norms["gutenberg_id"].astype(str)
    flows["gutenberg_id"] = flows["gutenberg_id"].astype(str)

    # NORM_EMB[instr]; FLOW_EMB[(flow_text, instr)]
    NORM_EMB, FLOW_EMB = {}, {}
    for _iv in ("shared", "prod"):
        _ne = np.load(EMB_DIR / f"{CORPUS}_norms_{_iv}.npy")
        # A mismatch here means the .npy predates the current filter; the
        # builder writes meta and matrices together, so this should not fire.
        assert _ne.shape[0] == len(norms), f"{_iv}: {_ne.shape[0]} norm rows vs {len(norms)}"
        NORM_EMB[_iv] = _ne
        for _tv in FLOW_TEXTS:
            _fe = np.load(EMB_DIR / f"{CORPUS}_flows_{_tv}_{_iv}.npy")
            assert _fe.shape[0] == len(flows), f"{_tv}/{_iv}: {_fe.shape[0]} vs {len(flows)}"
            FLOW_EMB[(_tv, _iv)] = _fe

    BOOKS = sorted(norms["book_title"].astype(str).unique())
    # The 4x2 print subset. Kept as a filter of BOOKS rather than a literal so a
    # title that stops matching the corpus fails loudly here instead of silently
    # dropping a ninth novel from the camera-ready panel.
    _missing = set(DROPPED_FOR_PRINT) - set(BOOKS)
    assert not _missing, f"DROPPED_FOR_PRINT names no such novel: {_missing}"
    PRINT_BOOKS = [b for b in BOOKS if b not in DROPPED_FOR_PRINT]
    NBOOK = np.asarray(norms["book_title"].astype(str))
    FBOOK = np.asarray(flows["book_title"].astype(str))

    print(f"norms {NORM_EMB['shared'].shape}, flows {FLOW_EMB[('noappr', 'shared')].shape}, "
          f"{len(BOOKS)} books ({len(PRINT_BOOKS)} in the print subset)")
    print("\nexample serializations")
    for _tv in FLOW_TEXTS:
        print(f"  [{_tv:11s}] {flows[f'embed_text_{_tv}'].iloc[2]}")
    print(f"  [{'norm':11s}] {norms['embed_text'].iloc[0]}")
    return BOOKS, FBOOK, FLOW_EMB, NBOOK, NORM_EMB, PRINT_BOOKS, flows, norms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The two norm pools

    `governs` keeps only norms the extractor flagged as regulating information
    transmission (`raz_governs_info_flow`). This is not a cosmetic filter: it is
    the pool R-DIRECT retrieves against in production, and restricting the index
    to it is what took the unscored rate from 37% to 0 with zero gold flips
    (2026-07-28). A norm outside it — "a gentleman does not strike a lady" —
    cannot govern a flow no matter how close it lands in embedding space, so
    including it in the retrieval pool can only add noise.
    """)
    return


@app.cell
def _(BOOKS, NBOOK, np, norms, pd, save_table):
    _gov = norms["raz_governs_info_flow"].fillna(False).astype(bool).values
    NORM_SETS = {
        "all": np.arange(len(norms)),
        "governs": np.flatnonzero(_gov),
    }

    norm_pool_sizes = pd.DataFrame(
        {
            "book": BOOKS,
            "norms_all": [int((NBOOK == b).sum()) for b in BOOKS],
            "norms_governs": [int(((NBOOK == b) & _gov).sum()) for b in BOOKS],
        }
    )
    norm_pool_sizes["frac_governing"] = (
        norm_pool_sizes["norms_governs"] / norm_pool_sizes["norms_all"]
    )
    save_table(norm_pool_sizes, "norm_pool_sizes", index=False)
    print(f"norm pools: all={len(NORM_SETS['all']):,}  "
          f"governs={len(NORM_SETS['governs']):,} "
          f"({len(NORM_SETS['governs']) / len(NORM_SETS['all']):.1%})")
    print(norm_pool_sizes.to_string(index=False))
    return NORM_SETS, norm_pool_sizes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Joint projections

    Each projection is fit on the **pooled** norm+flow matrix for its own norm
    pool, so norms and flows land in one space and the `governs` figures are a
    genuine re-derivation rather than a re-crop of the `all` layout.

    - **UMAP** (cosine, `n_neighbors=30`, `min_dist=0.0`, seed 42) — matches the
      accepted figure's settings. Good for neighbourhood structure; *directions
      and distances in it are not meaningful*, which is why it is not where the
      displacement claim is made.
    - **PCA** (2 components) — linear, so a displacement drawn in it is a
      faithful (if lossy) projection of the true 4096-D displacement. This is
      the panel the vector-field reading is allowed to rest on.

    Both are cached to `outputs/camera_ready/projections/`.
    """)
    return


@app.cell
def _(CACHE_DIR, CORPUS, FLOW_EMB, NORM_EMB, NORM_SETS, np):
    # Camera for the PCA-3 render. Chosen by grid search over (elev, azim) to
    # minimise the median percentile-rank of each flow's true retrieved norm in
    # the *rendered* image — i.e. optimised for what a reader can actually
    # recover from the printed figure, not for true 3D distance. Two parameters
    # are selected on that metric, which the caption discloses. Note that
    # matplotlib's default 3D view (elev=30, azim=-60) scores *worse* than a
    # flat PCA-2 plot, so the angle has to be set deliberately.
    VIEW_ELEV, VIEW_AZIM = -60, 100

    def camera(elev, azim):
        """Orthographic image-plane basis (2x3) for a matplotlib-style view."""
        e, a = np.radians(elev), np.radians(azim)
        d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        right = np.cross([0.0, 0.0, 1.0], d)
        right /= np.linalg.norm(right)
        up = np.cross(d, right)
        up /= np.linalg.norm(up)
        return np.stack([right, up])

    def _fit_projections(flow_text, instr, norm_set):
        idx = NORM_SETS[norm_set]
        pooled = np.vstack([NORM_EMB[instr][idx], FLOW_EMB[(flow_text, instr)]])
        # `all` keeps the historical stem so the already-computed caches are
        # reused rather than silently refit.
        stem = f"{CORPUS}_{flow_text}_{instr}"
        if norm_set != "all":
            stem += f"_{norm_set}"
        out = {}

        upath = CACHE_DIR / f"{stem}_umap2d.npy"
        if upath.exists() and np.load(upath, mmap_mode="r").shape[0] == pooled.shape[0]:
            out["umap"] = np.load(upath)
        else:
            import umap

            print(f"fitting UMAP [{stem}] on {pooled.shape} ...", flush=True)
            out["umap"] = umap.UMAP(
                n_components=2, metric="cosine", n_neighbors=30,
                min_dist=0.0, random_state=42,
            ).fit_transform(pooled).astype(np.float32)
            np.save(upath, out["umap"])

        ppath = CACHE_DIR / f"{stem}_pca2d.npy"
        vpath = CACHE_DIR / f"{stem}_pca_evr.npy"
        if ppath.exists() and np.load(ppath, mmap_mode="r").shape[0] == pooled.shape[0]:
            out["pca"] = np.load(ppath)
            out["pca_evr"] = np.load(vpath)
        else:
            from sklearn.decomposition import PCA

            print(f"fitting PCA [{stem}] on {pooled.shape} ...", flush=True)
            _p = PCA(n_components=2, random_state=42)
            out["pca"] = _p.fit_transform(pooled).astype(np.float32)
            out["pca_evr"] = _p.explained_variance_ratio_.astype(np.float32)
            np.save(ppath, out["pca"])
            np.save(vpath, out["pca_evr"])

        # 3-component PCA, rendered to 2D through a fixed camera. The pairing
        # lives in 4096-D and any 2D display loses most of it; a third component
        # viewed from a chosen angle recovers part of that loss (see the
        # retrieval-faithfulness sweep below).
        p3path = CACHE_DIR / f"{stem}_pca3d.npy"
        v3path = CACHE_DIR / f"{stem}_pca3_evr.npy"
        if p3path.exists() and np.load(p3path, mmap_mode="r").shape[0] == pooled.shape[0]:
            out["pca3_raw"] = np.load(p3path)
            out["pca3_evr"] = np.load(v3path)
        else:
            from sklearn.decomposition import PCA

            print(f"fitting PCA-3 [{stem}] on {pooled.shape} ...", flush=True)
            _p3 = PCA(n_components=3, random_state=42)
            out["pca3_raw"] = _p3.fit_transform(pooled).astype(np.float32)
            out["pca3_evr"] = _p3.explained_variance_ratio_.astype(np.float32)
            np.save(p3path, out["pca3_raw"])
            np.save(v3path, out["pca3_evr"])
        out["pca3"] = out["pca3_raw"] @ camera(VIEW_ELEV, VIEW_AZIM).T
        out["n_norm"] = len(idx)
        return out

    # Primary space, the two serialization sensitivities, the
    # production-instruction comparison, and the governing-norm pool.
    PROJ = {
        k: _fit_projections(*k)
        for k in (("noappr", "shared", "all"), ("full", "shared", "all"),
                  ("descriptive", "shared", "all"), ("noappr", "prod", "all"),
                  ("noappr", "shared", "governs"))
    }
    for _k, _p in PROJ.items():
        print(f"{_k} umap {_p['umap'].shape}  pca EVR {_p['pca_evr'].round(4).tolist()}"
              f"  pca3 EVR {_p['pca3_evr'].sum():.4f}")
    return PROJ, VIEW_AZIM, VIEW_ELEV, camera


@app.cell
def _(CACHE_DIR, CORPUS, FBOOK, FLOW_EMB, NBOOK, NORM_EMB, NORM_SETS, camera,
      json, np):
    # Per-layout facts worth printing on an axis (variance fractions, effect
    # sizes). Keyed by the same stem as the coordinate cache and persisted
    # beside it, so a cache hit still populates it.
    LAYOUT_META = {}

    def fit_per_book_layouts(flow_text, instr, norm_set, method, elev=None, azim=None):
        """Fit a display layout and stack it into one (N+F, 2) array.

        Rows keep the same order as the corpus projections — pool norms first,
        then flows — so `draw_per_book` indexes them identically.

        Most methods refit *inside each novel*: the pairing is within-novel, so
        a within-novel basis is the matched choice. The cost is that panels then
        carry no common coordinate system and nothing may be compared across
        them.

        `contrast` is the exception: one basis for the whole corpus, so panels
        ARE comparable. See the branch below for why that is affordable here.
        """
        stem = f"{CORPUS}_{flow_text}_{instr}_{norm_set}_perbook_{method}"
        path = CACHE_DIR / f"{stem}.npy"
        mpath = CACHE_DIR / f"{stem}_meta.json"
        nidx = NORM_SETS[norm_set]
        ne, fe = NORM_EMB[instr][nidx], FLOW_EMB[(flow_text, instr)]
        nbook, fbook = NBOOK[nidx], FBOOK
        if path.exists() and np.load(path, mmap_mode="r").shape[0] == len(ne) + len(fe):
            if mpath.exists():
                LAYOUT_META[stem] = json.loads(mpath.read_text())
            return np.load(path)

        from sklearn.decomposition import PCA
        from sklearn.manifold import MDS

        if method == "contrast":
            # ONE basis for the whole corpus, so every panel shares coordinates.
            #   x = the corpus norm-minus-flow contrast direction
            #   y = the leading PC of the data with x projected out
            #
            # Why this is affordable: a shared basis normally costs retrieval
            # fidelity, because each novel's own principal plane points
            # elsewhere in 4096-D (measured: principal angles between per-novel
            # PCA-2 planes average 65 deg / 38 deg and reach 89 deg). What makes
            # one shared axis legitimate anyway is that the *contrast direction*
            # does agree across novels — pairwise cosine 0.794 (min 0.659) — so
            # x means the same thing in every panel instead of being a
            # compromise none of them wants.
            #
            # x is chosen using the class labels, so the clean norm/flow split
            # it produces is by construction and the caption must say so. The
            # split is real independently of the choice (Cohen's d ~6.4,
            # 99.8% held-out separable); the axis displays it, not creates it.
            u = ne.mean(0) - fe.mean(0)
            u /= np.linalg.norm(u)
            X = np.vstack([ne, fe]).astype(np.float64)
            tot = float(np.var(X, axis=0).sum())
            X -= X.mean(0)
            xu = X @ u
            X -= np.outer(xu, u)          # in place: X is now the residual
            v = PCA(1, random_state=42).fit(X).components_[0]
            v -= (v @ u) * u              # guard against drift off orthogonality
            v /= np.linalg.norm(v)
            # v is orthogonal to u, so projecting the residual equals projecting
            # the centred data — no need to keep an uncentred copy around.
            xv = X @ v
            xy = np.stack([xu, xv], axis=1).astype(np.float32)
            _a, _b = ne @ u, fe @ u
            LAYOUT_META[stem] = {
                "var_x": float(np.var(xu) / tot),
                "var_y": float(np.var(xv) / tot),
                "cohens_d": float((_a.mean() - _b.mean())
                                  / np.sqrt((_a.var() + _b.var()) / 2)),
            }
            mpath.write_text(json.dumps(LAYOUT_META[stem]))
            np.save(path, xy)
            return xy

        def _place_oos(Zl, D, iters=40):
            """Place out-of-sample points against fixed landmark coordinates.

            Single-point SMACOF (Guttman) update, landmarks held fixed:
                z_i <- mean_j [ l_j + d_ij * (z_i - l_j) / ||z_i - l_j|| ]
            Used by `mds2bal` so every flow still gets a position while only a
            class-balanced subset drives the layout.
            """
            nn = np.argsort(D, axis=1)[:, :8]
            Z = Zl[nn].mean(axis=1)
            n = len(Zl)
            for _ in range(iters):
                diff = Z[:, None, :] - Zl[None, :, :]
                dist = np.maximum(np.linalg.norm(diff, axis=2), 1e-9)
                Z = (Zl[None, :, :] + (D / dist)[:, :, None] * diff).sum(1) / n
            return Z

        def _cosdist(A, B=None):
            d = 1.0 - (A @ (A if B is None else B).T)
            if B is None:
                np.fill_diagonal(d, 0.0)
                d = (d + d.T) / 2.0
            return np.clip(d, 0, None)

        def _mds(d, k):
            return MDS(k, metric=True, dissimilarity="precomputed", random_state=42,
                       n_init=1, max_iter=200, normalized_stress=False).fit_transform(d)

        xy = np.zeros((len(ne) + len(fe), 2), np.float32)
        for bk in np.unique(fbook):
            ni, fi = np.flatnonzero(nbook == bk), np.flatnonzero(fbook == bk)
            if not len(ni):
                continue
            X = np.vstack([ne[ni], fe[fi]])
            print(f"  [{method}] {bk}: {X.shape[0]} rows ...", flush=True)
            if method == "pca2":
                Z = PCA(2, random_state=42).fit_transform(X)
            elif method == "pca3":
                Z = PCA(3, random_state=42).fit_transform(X) @ camera(elev, azim).T
            elif method in ("mds2", "mds3"):
                # SMACOF on the cosine distance matrix. Unlike a k-NN method this
                # sees every flow-norm distance directly, which is why it holds
                # the retrieval relation better than UMAP.
                #
                # CAVEAT: with ~6x more flows than norms the flow cloud dominates
                # the stress function, takes a compact central placement, and the
                # sparse norms get distributed *around* it — producing a ring that
                # is not in the data. In 4096-D the norms are a one-sided lobe
                # (R ~ 0.55 against an isotropic floor of ~0.05). Balancing the
                # classes moves per-book 2D one-sidedness 0.66 -> 0.87. Use
                # `mds2bal` for any claim about configuration.
                Z = _mds(_cosdist(X), 2 if method == "mds2" else 3)
                if method == "mds3":
                    Z = Z @ camera(elev, azim).T
            elif method == "mds2bal":
                # Class-balanced MDS. The layout is fit on all norms plus a
                # matched-size flow sample, so neither class outvotes the other in
                # the stress function; the remaining flows are then placed against
                # that fixed layout. Every flow still appears — only the *fit* is
                # balanced, not the plot.
                rng_b = np.random.default_rng(17)
                Nb, Fb = ne[ni], fe[fi]
                k = min(len(Nb), len(Fb))
                sel = rng_b.choice(len(Fb), k, replace=False)
                rest = np.setdiff1d(np.arange(len(Fb)), sel)
                land = np.vstack([Nb, Fb[sel]])
                ZL = _mds(_cosdist(land), 2)
                Z = np.zeros((len(Nb) + len(Fb), 2))
                Z[: len(Nb)] = ZL[: len(Nb)]
                Z[len(Nb) + sel] = ZL[len(Nb):]
                if len(rest):
                    Z[len(Nb) + rest] = _place_oos(ZL, _cosdist(Fb[rest], land))
            else:
                raise ValueError(method)
            xy[ni] = Z[: len(ni)]
            xy[len(ne) + fi] = Z[len(ni):]
        np.save(path, xy)
        return xy

    return LAYOUT_META, fit_per_book_layouts


@app.cell
def _(FBOOK, FLOW_TEXT, INSTR, NBOOK, NORM_SETS, PROJ, pd):
    def plot_frame(projection, flow_text=FLOW_TEXT, instr=INSTR, norm_set="all"):
        p = PROJ[(flow_text, instr, norm_set)]
        xy = p[projection]
        n_norm = p["n_norm"]
        return pd.DataFrame(
            {
                "x": xy[:, 0],
                "y": xy[:, 1],
                "construct": ["norm"] * n_norm + ["flow"] * (len(xy) - n_norm),
                "book": list(NBOOK[NORM_SETS[norm_set]]) + list(FBOOK),
            }
        )

    df_umap = plot_frame("umap")
    print(df_umap["construct"].value_counts().to_dict())
    return df_umap, plot_frame


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1 — per-book small multiples (main paper)

    The accepted design, recomputed on the Gemma-4 lineage. Each panel holds one
    source text against the pooled corpus in grey; norms are dots, flows are
    crosses. Full norm pool.
    """)
    return


@app.cell
def _(
    BG_COLOR,
    BOOKS,
    FLOW_COLOR,
    Line2D,
    NORM_COLOR,
    SHORT_TITLE,
    df_umap,
    plt,
    save_fig,
):
    _ncols = 5
    _nrows = (len(BOOKS) + _ncols - 1) // _ncols
    _fig, _axes = plt.subplots(
        _nrows, _ncols, figsize=(2.45 * _ncols, 2.25 * _nrows), sharex=True, sharey=True
    )
    _axes = _axes.flatten()

    for _i, _bk in enumerate(BOOKS):
        _ax = _axes[_i]
        _ax.grid(False)
        _other = df_umap[df_umap["book"] != _bk]
        _ax.scatter(_other["x"], _other["y"], c=BG_COLOR, s=0.6, alpha=0.35,
                    linewidths=0, rasterized=True, zorder=1)

        _bn = df_umap[(df_umap["book"] == _bk) & (df_umap["construct"] == "norm")]
        _bf = df_umap[(df_umap["book"] == _bk) & (df_umap["construct"] == "flow")]
        _ax.scatter(_bn["x"], _bn["y"], c=NORM_COLOR, s=3.5, alpha=0.5,
                    linewidths=0, rasterized=True, zorder=2)
        _ax.scatter(_bf["x"], _bf["y"], c=FLOW_COLOR, s=6, alpha=0.6, marker="x",
                    linewidths=0.45, rasterized=True, zorder=3)

        _ax.set_title(SHORT_TITLE[_bk], fontsize=8.5, fontweight="bold", pad=3)
        # Counts as a corner annotation rather than a per-panel legend: ten
        # legends repeat the same two keys and crowd out the data.
        _ax.text(
            0.03, 0.03,
            f"{len(_bn):,} norms\n{len(_bf):,} flows",
            transform=_ax.transAxes, fontsize=6.5, va="bottom", ha="left",
            color="#333333", zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.72),
        )
        _ax.set_xticks([])
        _ax.set_yticks([])
        for _sp in _ax.spines.values():
            _sp.set_visible(True)
            _sp.set_linewidth(0.4)
            _sp.set_color("#BBBBBB")

    for _j in range(len(BOOKS), len(_axes)):
        _axes[_j].set_visible(False)

    _handles = [
        Line2D([], [], marker="o", ls="none", color=NORM_COLOR, ms=4, label="Norms"),
        Line2D([], [], marker="x", ls="none", color=FLOW_COLOR, ms=4.5, mew=1.1,
               label="Information flows"),
        Line2D([], [], marker="o", ls="none", color=BG_COLOR, ms=4,
               label="All other source texts"),
    ]
    _fig.tight_layout(rect=[0.02, 0.04, 1, 0.94])
    # supxlabel anchors to the figure, not to the axes, and with hidden ticks
    # that leaves a band of dead space under the bottom row. Anchor to the
    # lowest visible axes instead.
    _y0 = min(_a.get_position().y0 for _a in _axes if _a.get_visible())
    _fig.text(0.5, _y0 - 0.055, "UMAP-1", ha="center", fontsize=9)
    _fig.text(0.008, 0.5, "UMAP-2", va="center", rotation="vertical", fontsize=9)
    # Legend above the panels: below them it lands on top of the shared x label.
    _fig.legend(handles=_handles, loc="upper center", ncol=3, frameon=False,
                bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    save_fig(_fig, "fig_norm_flow_per_book_umap")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How much of the norm/flow separation is our own template?

    Swept over the three flow serializations and both embedding instructions.
    The paper currently quotes a silhouette of 0.77 from the stale extraction
    under the `full` string and production instructions; the rows below are what
    replace it, and the spread across rows is the size of the question.

    **Read the separability columns, not the silhouette.** Silhouette is a
    full-space ratio: it divides between-class distance by within-class spread
    measured over all 4096 dimensions, and the norm/flow contrast axis carries
    only 5–13% of total variance (`contrast_var_frac`), so the other ~90% —
    variance orthogonal to the contrast — dominates the denominator. Silhouette
    therefore reports ~0.17–0.40 for a split that every row below shows to be
    ≥99.7% separable: it is measuring ambient dimensionality, not overlap. The
    honest measure is separation *along the discriminant*: fit the
    mean-difference direction on half the rows, project the held-out half, and
    score it there.

    - `heldout_cohens_d` — standardised norm/flow gap on that axis
    - `heldout_acc` — accuracy of a single threshold on it
    - `heldout_overlap` — overlapping area of the two marginals (0 = disjoint)

    Silhouette and Davies-Bouldin are retained as secondary columns so the
    figure numbers quoted elsewhere remain traceable.
    """)
    return


@app.cell
def _(FLOW_EMB, FLOW_TEXTS, NORM_EMB, NORM_SETS, PROJ, np, pd, save_table):
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    def heldout_separability(X, lab, seed=0, nbin=200):
        """Norm/flow separation along the discriminant, scored out of sample.

        The mean-difference direction is fit on half the rows and every number
        below is measured on the other half, so the reported separation is not
        the in-sample best case. Returns (cohens_d, threshold accuracy, overlap
        coefficient of the two marginals, variance fraction along the axis).
        """
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        tr, te = perm[: len(X) // 2], perm[len(X) // 2:]
        u = X[tr][lab[tr] == 0].mean(0) - X[tr][lab[tr] == 1].mean(0)
        u /= np.linalg.norm(u)

        proj = X[te] @ u
        flow, norm = proj[lab[te] == 1], proj[lab[te] == 0]
        d = (norm.mean() - flow.mean()) / np.sqrt((flow.var() + norm.var()) / 2)
        thr = (flow.mean() + norm.mean()) / 2
        acc = ((norm > thr).sum() + (flow <= thr).sum()) / len(proj)

        bins = np.linspace(proj.min(), proj.max(), nbin)
        hf, _ = np.histogram(flow, bins=bins, density=True)
        hn, _ = np.histogram(norm, bins=bins, density=True)
        overlap = float(np.minimum(hf, hn).sum() * (bins[1] - bins[0]))
        var_frac = float(np.var(X @ u) / np.var(X, axis=0).sum())
        return float(d), float(acc), overlap, var_frac

    _rows = []
    for _ns in ("all", "governs"):
        _nidx = NORM_SETS[_ns]
        for _iv in ("shared", "prod"):
            for _tv in FLOW_TEXTS:
                _pooled = np.vstack([NORM_EMB[_iv][_nidx], FLOW_EMB[(_tv, _iv)]])
                _lab = np.r_[np.zeros(len(_nidx), int),
                             np.ones(len(_pooled) - len(_nidx), int)]
                _rng = np.random.default_rng(42)
                _sub = _rng.choice(len(_pooled), min(5000, len(_pooled)), replace=False)

                _d, _acc, _ov, _vf = heldout_separability(_pooled, _lab)
                _row = {
                    "norm_pool": _ns,
                    "instruction": _iv,
                    "flow_text": _tv,
                    # Primary: separation along the discriminant, out of sample.
                    "heldout_cohens_d": _d,
                    "heldout_acc": _acc,
                    "heldout_overlap": _ov,
                    "contrast_var_frac": _vf,
                    # Secondary: full-space ratios, dominated by the ~95% of
                    # variance orthogonal to the contrast. See the note above.
                    "silhouette_4096d": silhouette_score(
                        _pooled[_sub], _lab[_sub], metric="cosine"),
                    "davies_bouldin_4096d": davies_bouldin_score(_pooled[_sub], _lab[_sub]),
                }
                _proj = PROJ.get((_tv, _iv, _ns))
                if _proj is not None:
                    _row["silhouette_umap2d"] = silhouette_score(
                        _proj["umap"], _lab, metric="euclidean")
                    _row["davies_bouldin_umap2d"] = davies_bouldin_score(
                        _proj["umap"], _lab)
                _rows.append(_row)

    separation = pd.DataFrame(_rows)
    save_table(separation, "norm_flow_separation", index=False)
    print(separation.to_string(index=False))
    separation
    return (separation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Do the source texts form distinct clusters?

    The existing caption claims the texts "form partially overlapping but
    distinct clusters". That was written against the Qwen-era extraction and has
    to be re-checked, not inherited: on the Gemma-4 lineage the norm/flow split
    is visually much sharper, which can crowd out book structure.

    - **k-NN book purity** — for each row, the share of its 20 nearest
      neighbours from the same source text, against the share expected if
      neighbours were drawn at random (the corpus's Simpson index).
    - **book silhouette** — silhouette of the 10-way book labelling.
    """)
    return


@app.cell
def _(FBOOK, FLOW_EMB, FLOW_TEXT, INSTR, NBOOK, NORM_EMB, NORM_SETS, np, pd, save_table):
    from sklearn.metrics import silhouette_score as _sil
    from sklearn.neighbors import NearestNeighbors

    _rows = []
    for _kind, _e, _bk in (
        ("norm (all)", NORM_EMB[INSTR], NBOOK),
        ("norm (governs)", NORM_EMB[INSTR][NORM_SETS["governs"]],
         NBOOK[NORM_SETS["governs"]]),
        ("flow", FLOW_EMB[(FLOW_TEXT, INSTR)], FBOOK),
    ):
        _rng = np.random.default_rng(0)
        _sub = _rng.choice(len(_e), min(6000, len(_e)), replace=False)
        _nn = NearestNeighbors(n_neighbors=21, metric="cosine").fit(_e)
        _, _idx = _nn.kneighbors(_e[_sub])
        _p = pd.Series(_bk).value_counts(normalize=True).values
        _rows.append({
            "construct": _kind,
            "knn20_book_purity": float((_bk[_idx[:, 1:]] == _bk[_sub][:, None]).mean()),
            "chance_purity": float((_p**2).sum()),
            "book_silhouette": float(_sil(_e[_sub], _bk[_sub], metric="cosine")),
            "n": len(_sub),
        })

    book_structure = pd.DataFrame(_rows)
    save_table(book_structure, "book_structure", index=False)
    print(book_structure.to_string(index=False))
    book_structure
    return (book_structure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-book pairing

    For each CI flow, retrieve the nearest norm by cosine similarity **from the
    same novel**, within the active norm pool. Displacement is
    $d_i = e_{\text{norm},\pi(i)} - e_{\text{flow},i}$ on L2-normalised
    embeddings.

    **The null.** Re-draw $\pi$ uniformly at random from the same novel's norms.
    Note what this null does and does not control: it preserves the global
    norm-centroid-minus-flow-centroid offset, because that offset survives any
    permutation. So it isolates exactly the quantity in question — whether
    *which* norm a flow is paired with contributes direction beyond the fact
    that norms sit somewhere else than flows on average.
    """)
    return


@app.cell
def _(FLOW_EMB, NORM_EMB, NORM_SETS, flows, norms, np):
    _FLOW_BY_BOOK = {}
    for _i, _b in enumerate(flows["gutenberg_id"].values):
        _FLOW_BY_BOOK.setdefault(_b, []).append(_i)
    _FLOW_BY_BOOK = {k: np.asarray(v) for k, v in _FLOW_BY_BOOK.items()}

    # Per norm pool: book id -> positions **within that pool's index array**,
    # which is what indexes both its embedding submatrix and its projection.
    _NORM_BY_BOOK = {}
    for _ns, _idx in NORM_SETS.items():
        _g = {}
        for _pos, _b in enumerate(norms["gutenberg_id"].values[_idx]):
            _g.setdefault(_b, []).append(_pos)
        _NORM_BY_BOOK[_ns] = {k: np.asarray(v) for k, v in _g.items()}

    def norm_emb(instr, norm_set):
        return NORM_EMB[instr][NORM_SETS[norm_set]]

    def pair_flows(flow_text, instr, norm_set, rng=None):
        """Nearest same-book norm (rng=None) or shuffled within book (rng given).

        Returns (flow_idx, norm_pos) — `norm_pos` indexes the *pool*, so it maps
        straight onto that pool's projection rows; `NORM_SETS[norm_set][pos]`
        recovers the row in `norms`.
        """
        ne, fe = norm_emb(instr, norm_set), FLOW_EMB[(flow_text, instr)]
        groups = _NORM_BY_BOOK[norm_set]
        fi, ni = [], []
        for _b, _idxs in _FLOW_BY_BOOK.items():
            _cand = groups.get(_b)
            if _cand is None or len(_cand) == 0:
                continue  # a book with flows but no norms in this pool
            if rng is None:
                _best = _cand[(fe[_idxs] @ ne[_cand].T).argmax(axis=1)]
            else:
                _best = rng.choice(_cand, size=len(_idxs), replace=True)
            fi.append(_idxs)
            ni.append(_best)
        _o = np.concatenate(fi).argsort()
        return np.concatenate(fi)[_o], np.concatenate(ni)[_o]

    def displacement(flow_text, instr, norm_set, fi, ni):
        return norm_emb(instr, norm_set)[ni] - FLOW_EMB[(flow_text, instr)][fi]

    def concentration(d):
        """Mean resultant length R of the unit displacement directions, plus
        cos(d_i, dbar) per pair. R in [0,1]; 0 = isotropic, 1 = all parallel."""
        u = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        R = float(np.linalg.norm(u.mean(axis=0)))
        dbar = d.mean(axis=0)
        cos = u @ (dbar / max(np.linalg.norm(dbar), 1e-12))
        return R, cos, dbar

    return concentration, displacement, norm_emb, pair_flows


@app.cell
def _(FBOOK, NBOOK, NORM_SETS, np):
    def layout_fidelity(xy, n_norm, pairs, norm_set, books=None):
        """How much of the 4096-D retrieval relation survives into a layout.

        For each flow, rank its *truly retrieved* norm among that novel's norms
        by distance **in the layout**. `hit@1` is the share where the norm the
        segment is drawn to is also the nearest one on the page; the median
        percentile rank says how far off the rest land. Chance is 1/n_norms and
        0.500 respectively.

        Ranking is within-novel because retrieval is: a norm from another book
        was never a candidate, so counting it as a competitor would flatter
        every basis equally and mean nothing.

        `books` restricts the score to a subset of novels, so a figure that
        drops panels is described by a number measured on the panels it keeps.
        """
        nbook = NBOOK[NORM_SETS[norm_set]]
        Zn, Zf = np.asarray(xy[:n_norm], float), np.asarray(xy[n_norm:], float)
        fb = FBOOK[pairs["flow_idx"]]
        hit, pr = [], []
        for bk in (np.unique(FBOOK) if books is None else books):
            ni = np.flatnonzero(nbook == bk)
            m = np.flatnonzero(fb == bk)
            if not len(ni) or not len(m):
                continue
            # `norm_pos` indexes the whole pool; rank against this novel's block.
            loc = np.full(n_norm, -1)
            loc[ni] = np.arange(len(ni))
            true = loc[pairs["norm_pos"][m]]
            D = ((Zf[pairs["flow_idx"][m]][:, None, :] - Zn[ni][None, :, :]) ** 2).sum(-1)
            rank = (D < D[np.arange(len(m)), true][:, None]).sum(1)
            hit.append(rank == 0)
            pr.append(rank / max(len(ni) - 1, 1))
        return (float(np.concatenate(hit).mean()),
                float(np.median(np.concatenate(pr))))

    return (layout_fidelity,)


@app.cell
def _(
    FLOW_EMB,
    FLOW_TEXT,
    FLOW_TEXTS,
    NORM_SETS,
    concentration,
    displacement,
    norm_emb,
    np,
    pair_flows,
    pd,
    save_table,
):
    N_NULL = 20

    # The full serialization x instruction sweep on the `all` pool, plus the
    # primary cell repeated on the `governs` pool.
    _COMBOS = [(_tv, _iv, "all") for _iv in ("shared", "prod") for _tv in FLOW_TEXTS]
    _COMBOS += [(FLOW_TEXT, _iv, "governs") for _iv in ("shared", "prod")]

    _rows = []
    PAIRS = {}
    for _tv, _iv, _ns in _COMBOS:
        _fi, _ni = pair_flows(_tv, _iv, _ns)
        _d = displacement(_tv, _iv, _ns, _fi, _ni)
        _R, _cos, _dbar = concentration(_d)
        _sim = (FLOW_EMB[(_tv, _iv)][_fi] * norm_emb(_iv, _ns)[_ni]).sum(axis=1)
        PAIRS[(_tv, _iv, _ns)] = {
            "flow_idx": _fi, "norm_pos": _ni,
            "d": _d, "R": _R, "cos": _cos, "dbar": _dbar, "sim": _sim,
        }

        _Rn, _align = [], []
        for _s in range(N_NULL):
            _r = np.random.default_rng(1000 + _s)
            _fis, _nis = pair_flows(_tv, _iv, _ns, rng=_r)
            _dn = displacement(_tv, _iv, _ns, _fis, _nis)
            _Rx, _cx, _dbn = concentration(_dn)
            _Rn.append(_Rx)
            _align.append(float(_dbn @ _dbar / (np.linalg.norm(_dbn) * np.linalg.norm(_dbar))))
        PAIRS[(_tv, _iv, _ns)]["null_R"] = np.array(_Rn)

        _reached = len(np.unique(_ni))
        _rows.append({
            "norm_pool": _ns,
            "instruction": _iv,
            "flow_text": _tv,
            "n_pairs": len(_fi),
            "norms_in_pool": len(NORM_SETS[_ns]),
            "norms_reached": _reached,
            "frac_norms_reached": _reached / len(NORM_SETS[_ns]),
            "nn_cos_mean": float(_sim.mean()),
            "disp_norm_mean": float(np.linalg.norm(_d, axis=1).mean()),
            "R": _R,
            "R_null_mean": float(np.mean(_Rn)),
            "R_null_sd": float(np.std(_Rn)),
            "R_minus_null": _R - float(np.mean(_Rn)),
            "cos_dbar_mean": float(_cos.mean()),
            "frac_cos_gt_0.5": float((_cos > 0.5).mean()),
            "dbar_true_vs_null_align": float(np.mean(_align)),
        })

    concentration_table = pd.DataFrame(_rows)
    save_table(concentration_table, "displacement_concentration", index=False)
    print(concentration_table.to_string(index=False))
    concentration_table
    return (PAIRS, concentration_table)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — paired displacement, corpus level

    Left: the linear (PCA) projection, where a segment's direction is a faithful
    projection of the true 4096-D displacement, so the vector-field reading is
    licensed. Right: the same pairs drawn on UMAP, which shows *which*
    neighbourhoods the pairs connect — but UMAP is non-linear, so segment
    direction there carries no directional claim and the panel is labelled
    accordingly.

    Segments are a seeded subsample (drawing 16k lines fills the panel solid);
    the full point clouds are always drawn.
    """)
    return


@app.cell
def _(
    ARROW_COLOR,
    FLOW_COLOR,
    FLOW_TEXT,
    INSTR,
    LINK_COLOR,
    Line2D,
    MATCH_COLOR,
    NORM_COLOR,
    PAIRS,
    PROJ,
    concentration_table,
    np,
    plot_frame,
    plt,
    save_fig,
):
    from matplotlib.collections import LineCollection

    N_SEGMENTS = 400

    _p = PAIRS[(FLOW_TEXT, INSTR, "all")]
    _rng = np.random.default_rng(7)
    _pick = _rng.choice(len(_p["flow_idx"]), size=min(N_SEGMENTS, len(_p["flow_idx"])),
                        replace=False)
    _fi = _p["flow_idx"][_pick]
    _ni = _p["norm_pos"][_pick]

    _row = concentration_table.query(
        "instruction == @INSTR and flow_text == @FLOW_TEXT and norm_pool == 'all'"
    ).iloc[0]

    _n_norm = PROJ[(FLOW_TEXT, INSTR, "all")]["n_norm"]
    _reached = np.zeros(_n_norm, bool)
    _reached[np.unique(_p["norm_pos"])] = True

    _fig, _axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    for _ax, _projection, _note in (
        (_axes[0], "pca", "linear — segment direction is meaningful"),
        (_axes[1], "umap", "non-linear — direction not interpretable"),
    ):
        _xy = PROJ[(FLOW_TEXT, INSTR, "all")][_projection]
        _f = plot_frame(_projection)
        _nmask = (_f["construct"] == "norm").values

        _ax.grid(False)
        _ax.scatter(_f.loc[~_nmask, "x"], _f.loc[~_nmask, "y"], c=FLOW_COLOR, s=1.4,
                    alpha=0.13, linewidths=0, rasterized=True, zorder=1)
        _ax.scatter(_xy[:_n_norm][~_reached, 0], _xy[:_n_norm][~_reached, 1],
                    c=NORM_COLOR, s=1.4, alpha=0.13, linewidths=0,
                    rasterized=True, zorder=1)
        _ax.scatter(_xy[:_n_norm][_reached, 0], _xy[:_n_norm][_reached, 1],
                    c=MATCH_COLOR, s=2.4, alpha=0.5, linewidths=0,
                    rasterized=True, zorder=2)

        _ax.add_collection(
            LineCollection(
                np.stack([_xy[_n_norm + _fi], _xy[_ni]], axis=1),
                colors=LINK_COLOR, linewidths=0.4, alpha=0.4, rasterized=True, zorder=3,
            )
        )
        # Endpoints of the drawn segments, so the pairing is legible.
        _ax.scatter(_xy[_n_norm + _fi][:, 0], _xy[_n_norm + _fi][:, 1], c=FLOW_COLOR,
                    s=6, marker="x", linewidths=0.7, alpha=0.9, rasterized=True, zorder=4)
        _ax.scatter(_xy[_ni][:, 0], _xy[_ni][:, 1], c=MATCH_COLOR, s=6, alpha=0.9,
                    linewidths=0, rasterized=True, zorder=4)

        # The resultant, drawn: flow centroid -> norm centroid. In the linear
        # panel this *is* the projection of dbar; in the UMAP panel it is only
        # the centroid displacement, so it is drawn there too but claims less.
        _fc = _xy[_n_norm:].mean(axis=0)
        _nc = _xy[:_n_norm].mean(axis=0)
        _ax.annotate(
            "", xy=_nc, xytext=_fc,
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color=ARROW_COLOR,
                            shrinkA=0, shrinkB=0, mutation_scale=18),
            zorder=5,
        )
        _ax.scatter(*_fc, s=26, marker="X", c=ARROW_COLOR, zorder=6, linewidths=0)

        if _projection == "pca":
            _evr = PROJ[(FLOW_TEXT, INSTR, "all")]["pca_evr"]
            _ax.set_xlabel(f"PC1 ({_evr[0]:.1%} var.)")
            _ax.set_ylabel(f"PC2 ({_evr[1]:.1%} var.)")
            _ax.text(
                0.985, 0.02,
                f"$R$ = {_row['R']:.3f}   mean $\\cos(d_i,\\bar{{d}})$ = "
                f"{_row['cos_dbar_mean']:.2f}",
                transform=_ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                color="#333333", zorder=10,
            )
        else:
            _ax.set_xlabel("UMAP-1")
            _ax.set_ylabel("UMAP-2")
        _ax.set_title(_note, fontsize=8.5, style="italic", color="#555555", pad=4)
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.autoscale_view()

    _handles = [
        Line2D([], [], marker="x", ls="none", color=FLOW_COLOR, ms=5, mew=1.2,
               label="Information flow"),
        Line2D([], [], marker="o", ls="none", color=MATCH_COLOR, ms=4.5,
               label=f"Norm retrieved by ≥1 flow ({_reached.sum():,})"),
        Line2D([], [], marker="o", ls="none", color=NORM_COLOR, ms=4.5,
               label=f"Norm never retrieved ({(~_reached).sum():,})"),
        Line2D([], [], color=LINK_COLOR, lw=1.1,
               label=f"flow → nearest same-book norm ({N_SEGMENTS:,} of "
                     f"{len(_p['flow_idx']):,} drawn)"),
        Line2D([], [], color=ARROW_COLOR, lw=2.2,
               label=r"mean displacement $\bar{d}$"),
    ]
    _fig.tight_layout(rect=[0, 0.08, 1, 1])
    _fig.legend(handles=_handles, loc="lower center", ncol=3, frameon=False,
                bbox_to_anchor=(0.5, -0.02), fontsize=8)
    save_fig(_fig, "fig_paired_displacement")
    _fig
    return (LineCollection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3 — paired displacement per source text

    The same construction as small multiples, so it can sit next to Figure 1.
    Each panel draws one novel's flows, its norms, **every** one of that novel's
    flow → nearest-same-book-norm segments, and (on the PCA basis) that novel's
    own mean displacement $\bar d_b$ as a bold arrow. All panels share one
    corpus-level projection, so the arrows are directly comparable across
    novels — and because nothing is subsampled, so is the ink: a panel's density
    is that novel's actual pair count.

    **Purple vs blue.** A norm is purple once at least one flow retrieves it as
    its nearest same-book norm, and blue if no flow ever does. The blue mass is
    the part of each novel's extracted normative universe that this retrieval
    never touches — worth seeing, because that is the part of the universe the
    reward signal cannot reach either.
    """)
    return


@app.cell
def _(
    ARROW_COLOR,
    BG_COLOR,
    BOOKS,
    FBOOK,
    FLOW_COLOR,
    LINK_COLOR,
    Line2D,
    LineCollection,
    MATCH_COLOR,
    NBOOK,
    NORM_COLOR,
    NORM_SETS,
    PAIRS,
    PROJ,
    SEG_PER_BOOK,
    SHORT_TITLE,
    VIEW_AZIM,
    VIEW_ELEV,
    np,
    plt,
    save_fig,
):
    def _select(n, budget, rng):
        """Row indices to draw: all of them, a fixed cap, or a fixed fraction."""
        if budget is None:
            return np.arange(n)
        size = min(int(budget), n) if isinstance(budget, int) else max(1, round(budget * n))
        return rng.choice(n, size=size, replace=False)

    def draw_per_book(flow_text, instr, norm_set, projection, name, stats,
                      budget=SEG_PER_BOOK, also_paper=False, layout=None,
                      axis_label=None, context_lab=None, context_colors=None,
                      books=None, print_size=False, comparable=False,
                      draw_xlabel=True):
        """One panel per novel: that novel's pairs.

        `layout=None` uses the shared corpus projection named by `projection`,
        so all panels sit in one space and the pooled corpus is drawn behind each
        in grey. Passing a per-book `layout` array instead refits inside each
        novel: panels then share no basis, so the grey corpus backdrop and the
        shared axis limits are both dropped rather than left in place implying a
        comparability that no longer holds.

        `stats` is the per-book frame; its `R`/`R_null`/`nn_cos_mean`/`frac_reached`
        columns feed the corner annotation. The mean-displacement arrow is drawn
        only on linear projections — on UMAP a direction is not a claim we
        can make.

        `books` restricts and orders the panels (default: all ten novels).
        `print_size` builds the figure at COLM's 5.5in \\textwidth so LaTeX
        applies no scaling and the point sizes below are the sizes on paper; the
        default screen build is ~12in wide and would be scaled to 0.45x.
        `comparable` declares that an explicitly passed `layout` is one shared
        basis rather than a per-novel refit, which restores the shared axis
        limits, the grey corpus backdrop, and drops the "not comparable" note.

        `draw_xlabel=False` suppresses the figure-level x-label and reclaims the
        strip reserved for it, for figures whose axis description belongs in the
        LaTeX caption instead. The panels carry no ticks, so this removes the
        last rendered x-axis element. The suppressed string is printed verbatim
        so it can be pasted into the caption rather than paraphrased: it carries
        measured quantities (explained variance, Cohen's d) that must not drift
        from the figure they describe. The y-label is unaffected.
        """
        books = BOOKS if books is None else list(books)
        # Point sizes are absolute, so they only mean anything once the figure
        # is built at its final width. Two profiles rather than one scaled set:
        # the print column has less room per panel, so its annotations are also
        # abbreviated below, not merely shrunk.
        fs = (dict(title=7.0, corner=6.0, axis=7.0, legend=6.3, ramp=6.2, arrow=(2.8, 1.5))
              if print_size else
              dict(title=8.5, corner=5.8, axis=9.0, legend=7.5, ramp=7.0, arrow=(3.4, 1.8)))
        p = PAIRS[(flow_text, instr, norm_set)]
        proj = PROJ[(flow_text, instr, norm_set)]
        n_norm = proj["n_norm"]
        # `shared` drives every consequence of comparability: shared axis
        # limits, the grey corpus backdrop, and the axis-label caveat. A passed
        # layout is per-novel unless it declares otherwise.
        shared = layout is None or comparable
        xy = proj[projection] if layout is None else layout
        # Only orthographic projections of the 4096-D space carry a direction.
        # Metric MDS preserves *distances* well (better than UMAP, since it sees
        # every flow-norm distance directly) but it is not a linear map, so a
        # segment's direction there is not a projection of the true displacement
        # and no mean-displacement arrow is drawn.
        # `contrast` is two fixed orthonormal vectors, so it is as linear as PCA
        # and segment direction survives the projection.
        linear = projection in ("pca", "pca2", "pca3", "contrast")
        nbook_pool = NBOOK[NORM_SETS[norm_set]]
        fbook_pairs = FBOOK[p["flow_idx"]]

        ctx_lab = context_lab
        # In-degree: how many flows retrieve each norm. Retrieval is heavily
        # top-heavy (median Gini 0.81; the median novel has ~12 norms absorbing
        # half its flows), and a binary reached/unreached split hides all of it —
        # 575 segments landing on one dot look identical to 575 landing on 575.
        # Marker area is scaled by sqrt(in-degree) so the hubs are visible.
        indeg = np.bincount(p["norm_pos"], minlength=n_norm)
        reached = indeg > 0
        HUB_S0, HUB_K = 3.0, 3.1     # base area, and area per sqrt(flow)

        def hub_size(pos):
            return HUB_S0 + HUB_K * np.sqrt(indeg[pos])

        # With every pair drawn, ink per panel scales with that novel's pair
        # count, so the capped version's alpha would read as a solid block.
        seg_lw, seg_alpha = (0.35, 0.45) if budget else (0.18, 0.10)

        if print_size:
            # 4x2 at COLM's \textwidth. The context variant needs a second
            # header band for its hues, so it buys that height rather than
            # taking it out of the panels.
            ncols = 4
            figsize = (5.5, 3.75 + (0.55 if context_lab is not None else 0.0))
        else:
            ncols = 5
            figsize = (2.45 * ncols, 2.3 * ((len(books) + ncols - 1) // ncols))
        nrows = (len(books) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 sharex=shared, sharey=shared)
        axes = np.atleast_1d(axes).flatten()
        rng = np.random.default_rng(11)

        for i, bk in enumerate(books):
            ax = axes[i]
            ax.grid(False)
            if shared:
                ax.scatter(xy[:, 0], xy[:, 1], c=BG_COLOR, s=0.5, alpha=0.3,
                           linewidths=0, rasterized=True, zorder=1)

            m = fbook_pairs == bk
            bfi, bni = p["flow_idx"][m], p["norm_pos"][m]
            sel = _select(len(bfi), budget, rng)

            bk_norm = nbook_pool == bk
            # Unreached norms are the point of the split, so they get enough
            # weight to be counted by eye — just less than the reached ones.
            ax.scatter(xy[:n_norm][bk_norm & ~reached, 0],
                       xy[:n_norm][bk_norm & ~reached, 1],
                       c=NORM_COLOR, s=2.8, alpha=0.5, linewidths=0,
                       rasterized=True, zorder=2)
            ax.scatter(xy[n_norm:][FBOOK == bk, 0], xy[n_norm:][FBOOK == bk, 1],
                       c=FLOW_COLOR, s=3.2, alpha=0.35, marker="x", linewidths=0.35,
                       rasterized=True, zorder=2)

            ax.add_collection(LineCollection(
                np.stack([xy[n_norm + bfi[sel]], xy[bni[sel]]], axis=1),
                colors=LINK_COLOR, linewidths=seg_lw, alpha=seg_alpha,
                rasterized=True, zorder=3,
            ))
            # Retrieved norms on top of the segments that reach them, sized by
            # in-degree. Drawn largest-first so a hub never occludes the small
            # norms beside it, and given a thin white ring so overlapping hubs
            # stay countable.
            hub_pos = np.flatnonzero(bk_norm & reached)
            hub_pos = hub_pos[np.argsort(-indeg[hub_pos])]
            hub_c = (MATCH_COLOR if ctx_lab is None
                     else [context_colors[c] for c in ctx_lab[hub_pos]])
            ax.scatter(xy[:n_norm][hub_pos, 0], xy[:n_norm][hub_pos, 1],
                       c=hub_c, s=hub_size(hub_pos), alpha=0.8,
                       linewidths=0.25, edgecolors="white",
                       rasterized=True, zorder=4)

            r = stats.set_index("book").loc[bk]
            if linear:
                fc = xy[n_norm:][FBOOK == bk].mean(axis=0)
                nc = xy[:n_norm][bk_norm].mean(axis=0)
                # White halo: over a full-density segment field the bare arrow
                # loses its outline against the segments it sits on.
                _halo, _core = fs["arrow"]
                for lw, colour, z, scale in ((_halo, "white", 5, 15),
                                             (_core, ARROW_COLOR, 6, 13)):
                    ax.annotate("", xy=nc, xytext=fc,
                                arrowprops=dict(arrowstyle="-|>", lw=lw, color=colour,
                                                shrinkA=0, shrinkB=0, mutation_scale=scale),
                                zorder=z)
                # The R/null plate was dropped 2026-08-08. R is a directional
                # concentration statistic about an axis chosen USING the
                # norm/flow labels, so a high value is partly guaranteed by
                # construction and the per-panel number invited a reading the
                # figure cannot support. The pooled R-versus-null comparison
                # still runs and still prints to the console; it is simply not
                # annotated per panel. `stat` stays None for these panels.
                stat = None
            else:
                stat = f"mean cos {r['nn_cos_mean']:.3f}"
            # At 1.3in of panel width the two top-corner labels are ~1.05in
            # together, so the print build abbreviates rather than overlapping.
            reach = (f"{r['frac_reached']:.0%} reached" if print_size
                     else f"{r['frac_reached']:.0%} of norms reached")

            ax.set_title(SHORT_TITLE[bk], fontsize=fs["title"], fontweight="bold", pad=3)
            # One stat per corner rather than a stacked block: the block cost
            # four lines of vertical space in the densest part of the panel.
            # Every plate needs zorder above all data layers (segments 3,
            # norms 4, arrow 5-6) — at matplotlib's default of 3 the points
            # paint straight over the text.
            plate = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9)
            _corners = [(0.03, 0.97, "top", "left", f"{len(sel):,} pairs"),
                        (0.03, 0.03, "bottom", "left", reach)]
            if stat is not None:
                _corners.insert(1, (0.97, 0.97, "top", "right", stat))
            for _x, _y, _va, _ha, _txt in _corners:
                ax.text(_x, _y, _txt, transform=ax.transAxes, fontsize=fs["corner"],
                        va=_va, ha=_ha, color="#333333", zorder=10, bbox=plate)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.4)
                sp.set_color("#BBBBBB")

        for j in range(len(books), len(axes)):
            axes[j].set_visible(False)

        # Size ramp keyed to real in-degrees, so the reader can convert a dot
        # back to a flow count. matplotlib scatter `s` is area in pt^2 and
        # Line2D `ms` is diameter in pt, hence the sqrt.
        ramp = [d for d in (1, 10, 100, 500) if d <= max(indeg.max(), 1)]
        size_handles = [
            Line2D([], [], marker="o", ls="none", color=MATCH_COLOR,
                   ms=np.sqrt(HUB_S0 + HUB_K * np.sqrt(d)), label=f"{d:,}")
            for d in ramp
        ]

        handles = [
            Line2D([], [], marker="x", ls="none", color=FLOW_COLOR, ms=4.5, mew=1.1,
                   label="Flows (this text)"),
            Line2D([], [], marker="o", ls="none", color=NORM_COLOR, ms=4,
                   label="Norms never retrieved"),
            Line2D([], [], color=LINK_COLOR, lw=1.1,
                   label="flow → nearest same-book norm"
                         + ("" if budget is None else f" ({budget} per panel)")),
        ]
        # When norms are coloured by context the hue carries the identity, so
        # the context keys get their own row and the shape keys shrink.
        ctx_handles = [] if ctx_lab is None else [
            Line2D([], [], marker="o", ls="none", color=_c, ms=4.2, label=_k)
            for _k, _c in context_colors.items()
        ]
        if linear:
            handles.append(Line2D([], [], color=ARROW_COLOR, lw=1.8,
                                  label=r"$\bar{d}_b$ (this text)"))

        if axis_label is not None:
            xlab, ylab = axis_label
        elif projection == "pca":
            evr = proj["pca_evr"]
            xlab, ylab = f"PC1 ({evr[0]:.1%} var.)", f"PC2 ({evr[1]:.1%} var.)"
        elif projection == "pca3":
            # The camera does NOT change the components' variance — that is
            # fixed by the PCA fit. It changes which 2D plane of the 3D subspace
            # reaches the page, discarding the variance along the view
            # direction: rendered = sum(EVR) - sum(d_i^2 * EVR_i). Quoting the
            # 3-component total here (as an earlier version did) implies the
            # angle produced it, which is wrong. Report what actually lands.
            evr = proj["pca3_evr"]
            _e, _a = np.radians(VIEW_ELEV), np.radians(VIEW_AZIM)
            _d = np.array([np.cos(_e) * np.cos(_a),
                           np.cos(_e) * np.sin(_a),
                           np.sin(_e)])
            _rendered = float(evr.sum() - (_d ** 2) @ evr)
            xlab = (f"PCA-3 rendered at elev {VIEW_ELEV}°, azim {VIEW_AZIM}° "
                    f"({_rendered:.1%} of variance reaches the image plane; "
                    f"the 3 components hold {evr.sum():.1%})")
            ylab = "rendered image plane"
        else:
            xlab, ylab = "UMAP-1", "UMAP-2"
        if not shared:
            xlab += "  —  basis refit within each novel; panels not comparable"

        # Header bands are *measured*, not guessed. The previous fixed offsets
        # (context at y=1.0, shape keys at 0.955) assumed the context legend was
        # under 0.045 of figure height; with a title plus a row of keys it is
        # closer to 0.07, so the two bands collided. Measuring also lets the
        # print build wrap its legends onto more rows without new constants.
        def _band_bottom(artist):
            fig.canvas.draw()
            return artist.get_window_extent().y0 / fig.bbox.height

        _gap = 0.014
        _top = 1.0
        if ctx_handles:
            _cl = fig.legend(handles=ctx_handles, loc="upper center",
                             ncol=4 if print_size else len(ctx_handles),
                             frameon=False, bbox_to_anchor=(0.5, _top),
                             fontsize=fs["legend"],
                             title="norm context (retrieved norms only)",
                             title_fontsize=fs["legend"] + 0.5, handletextpad=0.35,
                             columnspacing=1.1)
            fig.add_artist(_cl)
            _top = _band_bottom(_cl) - _gap
        # Shape keys left, size ramp right on one band. Kept as two legends
        # rather than one row: mixing shape keys with size keys reads as a
        # single scale and misleads. Print wraps only the shape keys to two
        # columns; the ramp stays on one row because matplotlib fills legend
        # columns top-to-bottom, so a 2-column ramp reads 1, 100 / 10, 500 —
        # an ordered scale broken out of order.
        leg = fig.legend(handles=handles, loc="upper left",
                         ncol=2 if print_size else len(handles),
                         frameon=False, bbox_to_anchor=(0.015, _top),
                         fontsize=fs["legend"])
        fig.add_artist(leg)
        ramp_leg = fig.legend(handles=size_handles, loc="upper right",
                              ncol=len(size_handles),
                              frameon=False, bbox_to_anchor=(0.985, _top),
                              fontsize=fs["ramp"],
                              title="norms retrieved by ≥1 flow — area ∝ √(flows)",
                              title_fontsize=fs["ramp"], handletextpad=0.4,
                              columnspacing=1.2)
        _rect_top = min(_band_bottom(leg), _band_bottom(ramp_leg)) - _gap

        # An x-label needs ~0.30in of clearance regardless of figure height, so
        # the reserve is taken in inches and converted, not left as a constant
        # fraction that shrinks with the shorter print figure. With the label
        # suppressed the reserve goes to zero rather than staying as an empty
        # strip: the panels have no ticks, so nothing else needs the room, and
        # a tight bbox would otherwise crop to the same content anyway while
        # the axes sat needlessly high in the frame.
        _xlab_in = 0.30
        _bottom = (_xlab_in / fig.get_figheight() + 0.02) if draw_xlabel else 0.0
        fig.tight_layout(rect=[0.02, _bottom, 1, _rect_top])
        if draw_xlabel:
            y0 = min(a.get_position().y0 for a in axes if a.get_visible())
            fig.text(0.5, y0 - _xlab_in / fig.get_figheight(), xlab,
                     ha="center", fontsize=fs["axis"])
        else:
            print(f"[x-label -> caption] {name}:\n    {xlab}")
        fig.text(0.008, 0.5, ylab, va="center", rotation="vertical",
                 fontsize=fs["axis"])
        save_fig(fig, name, also_paper=also_paper)
        return fig

    return (draw_per_book,)


@app.cell
def _(
    BOOKS,
    FBOOK,
    FLOW_EMB,
    NBOOK,
    NORM_SETS,
    PAIRS,
    concentration,
    displacement,
    norm_emb,
    np,
    pair_flows,
    pd,
    save_table,
):
    def one_sidedness(P, centre):
        """Mean resultant length of the unit vectors from `centre` to each row
        of `P`. 0 = points surround the centre isotropically; 1 = all points lie
        in a single direction from it.

        Computed in the full 4096-D space on purpose. The per-book panels are
        drawn on per-novel bases, and while "do the norms surround the flows or
        occupy one lobe?" is invariant to the arbitrary rotation/reflection of
        those bases — so it is a fair question to ask across panels — MDS
        distorts it badly: rendered one-sidedness spans 0.42-0.90 across novels
        and correlates with the value below at only rho = 0.32, including
        outright inversions. Anyone reading shape off the panels needs this
        column to check against.
        """
        v = P - centre
        u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        return float(np.linalg.norm(u.mean(axis=0)))

    def isotropic_floor(n, dim, draws=10, seed=0):
        """E[R] for n random unit vectors in `dim` dimensions.

        R is biased upward at small n — about 1/sqrt(n) — so a novel with 44
        norms scores ~0.15 before any real structure exists, against ~0.045 for
        one with 500. Reporting R without this floor makes short novels look
        systematically more one-sided than long ones.
        """
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(draws):
            g = rng.normal(size=(n, dim))
            g /= np.linalg.norm(g, axis=1, keepdims=True)
            vals.append(float(np.linalg.norm(g.mean(axis=0))))
        return float(np.mean(vals))

    def per_book_stats(flow_text, instr, norm_set, n_null=10):
        p = PAIRS[(flow_text, instr, norm_set)]
        fbook_pairs = FBOOK[p["flow_idx"]]
        nbook_pool = NBOOK[NORM_SETS[norm_set]]
        reached = np.unique(p["norm_pos"])
        ne_pool, fe_all = norm_emb(instr, norm_set), FLOW_EMB[(flow_text, instr)]

        nulls = [pair_flows(flow_text, instr, norm_set, rng=np.random.default_rng(4000 + s))
                 for s in range(n_null)]

        rows = []
        for bk in BOOKS:
            m = fbook_pairs == bk
            Rb, _, _ = concentration(p["d"][m])
            Rn = [
                concentration(displacement(
                    flow_text, instr, norm_set,
                    fis[FBOOK[fis] == bk], nis[FBOOK[fis] == bk]))[0]
                for fis, nis in nulls
            ]
            pool = np.flatnonzero(nbook_pool == bk)
            n_reached = len(np.intersect1d(pool, reached))
            if len(pool):
                _R_bk = one_sidedness(
                    ne_pool[pool], fe_all[np.flatnonzero(FBOOK == bk)].mean(axis=0))
                _floor_bk = isotropic_floor(len(pool), ne_pool.shape[1])
            else:
                _R_bk = _floor_bk = float("nan")
            rows.append({
                "book": bk,
                "n_flows": int(m.sum()),
                "n_norms": len(pool),
                "n_norms_reached": n_reached,
                "frac_reached": n_reached / max(len(pool), 1),
                "nn_cos_mean": float(p["sim"][m].mean()),
                "R": Rb,
                "R_null": float(np.mean(Rn)),
                "R_minus_null": Rb - float(np.mean(Rn)),
                "R_null_sd": float(np.std(Rn)),
                # Shape of this novel's norm cloud about its flow centroid,
                # measured in 4096-D rather than read off the panel. Report the
                # floor alongside it: R alone conflates real one-sidedness with
                # small-sample bias, and `..._minus_floor` is the honest number.
                # Every novel lands ~0.5 above its floor, i.e. the norms are a
                # displaced lobe, NOT a shell surrounding the flows — the ring in
                # the unbalanced MDS panels is a class-imbalance artifact.
                "R_norms_about_flows": _R_bk,
                "R_isotropic_floor": _floor_bk,
                "R_minus_floor": (_R_bk - _floor_bk
                                  if _R_bk == _R_bk else float("nan")),
            })
        return pd.DataFrame(rows)

    PER_BOOK = {}
    for _ns in ("all", "governs"):
        PER_BOOK[_ns] = per_book_stats("noappr", "shared", _ns)
        save_table(PER_BOOK[_ns], f"displacement_per_book_{_ns}", index=False)
        print(f"\n=== norm pool: {_ns} ===")
        print(PER_BOOK[_ns].to_string(index=False))
    return (PER_BOOK,)


@app.cell
def _(FLOW_TEXT, INSTR, PER_BOOK, draw_per_book):
    draw_per_book(FLOW_TEXT, INSTR, "all", "pca",
                  "fig_paired_displacement_per_book", PER_BOOK["all"])
    return


@app.cell
def _(FLOW_TEXT, INSTR, PER_BOOK, draw_per_book):
    draw_per_book(FLOW_TEXT, INSTR, "all", "umap",
                  "fig_paired_displacement_per_book_umap", PER_BOOK["all"])
    return


@app.cell
def _(FLOW_TEXT, INSTR, PER_BOOK, draw_per_book):
    # PCA-3 through the chosen camera: the basis that best preserves the
    # flow -> retrieved-norm relation of any 2D display tested.
    draw_per_book(FLOW_TEXT, INSTR, "all", "pca3",
                  "fig_paired_displacement_per_book_pca3", PER_BOOK["all"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 4 — the same, over the information-flow-governing norms only

    Identical construction, identical embeddings, identical pairing rule and
    nulls — the only change is the candidate pool, restricted to norms the
    extractor flagged as governing information transmission. This is the pool
    R-DIRECT retrieves against in production, so these are the panels that
    describe what the reward actually sees. The projections are refit on the
    restricted corpus rather than cropped from the full one, so the layout is a
    genuine re-derivation.
    """)
    return


@app.cell
def _(FLOW_TEXT, INSTR, PER_BOOK, draw_per_book):
    draw_per_book(FLOW_TEXT, INSTR, "governs", "pca",
                  "fig_paired_displacement_per_book_governs", PER_BOOK["governs"])
    return


@app.cell
def _(FLOW_TEXT, INSTR, PER_BOOK, draw_per_book):
    draw_per_book(FLOW_TEXT, INSTR, "governs", "umap",
                  "fig_paired_displacement_per_book_umap_governs", PER_BOOK["governs"])
    return


@app.cell
def _(FLOW_TEXT, INSTR, PER_BOOK, draw_per_book):
    draw_per_book(FLOW_TEXT, INSTR, "governs", "pca3",
                  "fig_paired_displacement_per_book_pca3_governs", PER_BOOK["governs"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 6 — per-book bases

    The pairing is defined *within* a novel, so the matched basis is one refit
    inside each novel rather than a shared corpus projection. Panels then carry
    no common coordinate system — nothing may be compared across them — but each
    panel shows its own novel's retrieval structure at full resolution.

    Scored on retrieval faithfulness (how often the norm a flow is drawn joined
    to is also the nearest norm on the page), over all ten novels. The two norm
    pools are reported separately because the numbers differ substantially and
    an earlier version of this page quoted the `governs` sweep for both:

    | basis | `governs` hit@1 / med pct-rank | `all` hit@1 / med pct-rank |
    |---|---|---|
    | corpus PCA-2 (the original figure) | 6.9% / 0.123 | 6.7% / 0.031 |
    | corpus PCA-3 @ camera | 8.0% / 0.080 | 7.2% / 0.027 |
    | **per-book PCA-2 (primary)** | **6.2% / 0.093** | **7.1% / 0.028** |
    | per-book PCA-3 @ camera | 8.6% / 0.080 | 6.8% / 0.028 |
    | per-book metric MDS-2 (cosine) | 8.5% / 0.072 | — |
    | *reference:* per-book PCA-3 as true 3D | 10.8% / 0.050 | 9.2% / 0.017 |
    | *reference:* per-book PCA-4 / PCA-5 | 12.7% / 14.4% | 11.3% / 12.6% |

    Chance is 0.7% (`governs`) and 0.2% (`all`) at pct-rank 0.500, so every
    layout carries real signal — but all of them sit near the floor of what this
    metric can reach, which is the fact that decides the basis.

    MDS beats the linear maps here because, unlike UMAP, it optimises against the
    full distance matrix and so sees every flow-norm distance directly — where
    UMAP's k-NN graph contains almost none, since 98% of flows have no norm among
    their 30 nearest neighbours.

    ### Why PCA-2 is the primary panel

    Two constraints, in order of force.

    **Nothing saturates at three components.** PCA-4 reaches 12.7% and PCA-5
    14.4%, still climbing steeply. If three were the natural dimensionality of
    this mapping there would be a knee; there is none, so choosing 3 over 2 is
    arbitrary rather than principled.

    **The camera angle does not transfer.** It was grid-searched to minimise
    median rank on the `governs` pool, where it gains +1.1pt over corpus PCA-2 —
    but on the `all` pool it gains +0.5pt, and per-book on `all` it is *worse*
    than plain PCA-2 (6.8% vs 7.1%). An angle tuned on one pool that reverses
    sign on another is a property of the fit, not of the data, and it costs a
    nameable axis: `PC1 (per novel)` becomes `rendered image plane`, and the
    caption owes a camera spec plus a defence of it. Two points of a
    floor-level metric does not buy that.

    Both PCA-2 and PCA-3 are linear, so the argument that rules out unbalanced
    MDS applies equally to either and does not separate them. That argument still
    holds against MDS: with ~6x more flows than norms the flow cloud dominates
    the stress function, takes a compact central placement, and the sparse norms
    get distributed around it as a ring that is not in the data. In 4096-D the
    norms are a **one-sided lobe**, not a shell — R ~ 0.51-0.62 against an
    isotropic floor of 0.045-0.151 (`R_minus_floor` in the per-book tables, ~0.5
    for every novel). Refitting with the classes balanced moves mean per-book 2D
    one-sidedness from 0.66 to 0.87, and the two panels that read as the most
    convincing full rings collapse hardest: Les Misérables 0.42 -> 0.87,
    Alice 0.49 -> 0.89. `mds2bal` fits the layout on all norms plus a
    matched-size flow sample and then places the remaining flows against that
    fixed layout, so every flow is still drawn but the fit is not
    cardinality-driven — that is the MDS variant to read configuration from.

    ### Making the panels comparable — the `contrast` basis

    Every basis above refits inside each novel, so nothing may be read across
    panels. That restriction is real, not fixable by alignment: the principal
    angles between per-novel PCA-2 planes average **65° / 38° and reach 89°**,
    so Procrustes-rotating the panels onto a common frame would be spinning
    genuinely different planes together and manufacturing the comparability.

    One thing does agree across novels, though — the direction from a novel's
    flows to its norms. Pairwise cosine between novels' contrast directions is
    **0.794** (min 0.659, max 0.923). So a single axis defined as *the* corpus
    norm-minus-flow direction means nearly the same thing in every panel, and
    the `contrast` basis uses it:

    * **x** = corpus norm − flow contrast direction
    * **y** = leading PC of the data with x projected out

    Comparability turns out to be close to free:

    | basis | `all` hit@1 / med pct-rank | `governs` | comparable? |
    |---|---|---|---|
    | per-book PCA-2 | 7.1% / 0.028 | 6.2% / 0.093 | no |
    | shared corpus PCA-2 | 6.7% / 0.031 | 6.9% / 0.123 | yes |
    | **shared `contrast`** | **7.0% / 0.028** | **7.6% / 0.095** | **yes** |

    It matches the per-book refit on `all` and beats it on `governs`, while
    giving all panels one coordinate system, shared limits, and a grey pooled
    backdrop.

    **The disclosure this basis owes.** x is chosen *using the class labels*, so
    the clean left/right norm/flow split is true by construction and a reader
    must not take it as an unsupervised discovery. The separation itself is a
    measured fact independent of the choice (Cohen's d ~6.4 on this axis, 99.8%
    held-out linear separability), so the axis displays a real thing rather than
    inventing one — but the caption has to say which axis was chosen and how.
    A second consequence: the per-novel arrows now all point roughly +x by
    construction, so the arrow carries magnitude, not direction.

    ### The print subset

    The camera-ready panel drops Bleak House and Middlemarch for a 4x2 grid built
    at COLM's 5.5in `\textwidth`. At ten panels the figure has to be authored
    ~12in wide and scaled to 0.45x on the page, which renders the corner
    annotations at ~2.6pt. Neither dropped novel is the minimum or maximum on
    any statistic the figure carries, with one exception the caption now detects
    and states automatically: on the `all` pool Bleak House is the only novel
    with `R_minus_null` below zero, so the eight-panel version loses the sole
    counterexample to "every novel beats its null". The ten-panel version is
    still rendered here and belongs in the appendix.
    """)
    return


@app.cell
def _(
    DROPPED_FOR_PRINT,
    FLOW_TEXT,
    INSTR,
    LAYOUT_META,
    PAIRS,
    PER_BOOK,
    PRINT_BOOKS,
    PROJ,
    VIEW_AZIM,
    VIEW_ELEV,
    draw_per_book,
    fit_per_book_layouts,
    layout_fidelity,
    save_caption,
):
    for _ns_pb in ("all", "governs"):
        for _meth, _lab in (
            # PCA-2 first: it is the primary basis. Linear (so it cannot
            # manufacture configuration from class imbalance the way MDS does),
            # and its axes are nameable, which the camera render's are not.
            ("pca2", ("PC1 (per novel)", "PC2 (per novel)")),
            # The one shared basis, so its panels ARE comparable. Axis labels
            # are filled in from the measured fit below.
            ("contrast", None),
            ("mds2bal", ("metric MDS, class-balanced fit (per novel)", "MDS-2")),
            ("pca3", (f"PCA-3 rendered at elev {VIEW_ELEV}°, azim {VIEW_AZIM}° "
                      f"(per novel)", "rendered image plane")),
            ("mds2", ("metric MDS on cosine distance (per novel) — UNBALANCED, "
                      "ring is a class-imbalance artifact", "MDS-2")),
        ):
            _suffix = "" if _ns_pb == "all" else "_governs"
            _layout = fit_per_book_layouts(FLOW_TEXT, INSTR, _ns_pb, _meth,
                                           elev=VIEW_ELEV, azim=VIEW_AZIM)
            _cmp = _meth == "contrast"
            if _cmp:
                _m = LAYOUT_META[f"fiction10_{FLOW_TEXT}_{INSTR}_{_ns_pb}"
                                 f"_perbook_contrast"]
                # Name what the axis IS, including that it was picked using the
                # labels — a reader must be able to tell the split apart from
                # the choice that displays it.
                _lab = (f"norm ↔ flow contrast axis, chosen on the class labels "
                        f"({_m['var_x']:.1%} var.; Cohen's $d$ = "
                        f"{_m['cohens_d']:.1f})",
                        f"leading orthogonal PC ({_m['var_y']:.1%} var.)")
            draw_per_book(
                FLOW_TEXT, INSTR, _ns_pb, _meth,
                f"fig_paired_displacement_per_book_{_meth}_perbook{_suffix}",
                PER_BOOK[_ns_pb], layout=_layout, axis_label=_lab,
                comparable=_cmp,
            )
            # The camera-ready cut: 8 novels, 4x2, authored at \textwidth so
            # LaTeX applies no scaling. The two candidate bases get one; the
            # rest are diagnostics for this page, not figures for the paper.
            if _meth in ("pca2", "contrast"):
                _nm_print = (f"fig_paired_displacement_per_book_{_meth}"
                             f"_print{_suffix}")
                draw_per_book(
                    FLOW_TEXT, INSTR, _ns_pb, _meth,
                    _nm_print, PER_BOOK[_ns_pb], layout=_layout, axis_label=_lab,
                    books=PRINT_BOOKS, print_size=True, also_paper=True,
                    comparable=_cmp, draw_xlabel=False,
                )
                # Scored over the eight novels actually drawn, not all ten —
                # the caption describes this panel.
                _hit, _pr = layout_fidelity(
                    _layout, PROJ[(FLOW_TEXT, INSTR, _ns_pb)]["n_norm"],
                    PAIRS[(FLOW_TEXT, INSTR, _ns_pb)], _ns_pb, books=PRINT_BOOKS)
                # Dropping panels is only safe if it moves no reported range.
                # Checked rather than asserted: on the `all` pool Bleak House is
                # the only novel with R_minus_null < 0, so omitting it silently
                # deletes the one counterexample the text relies on.
                _f10 = PER_BOOK[_ns_pb]
                _f8 = _f10[_f10["book"].isin(PRINT_BOOKS)]
                _moved = [c for c in ("R", "R_null", "R_minus_null",
                                      "frac_reached", "nn_cos_mean")
                          if c in _f10
                          and (abs(_f10[c].min() - _f8[c].min()) > 1e-9
                               or abs(_f10[c].max() - _f8[c].max()) > 1e-9)]
                _drop_note = (
                    "neither is the minimum or maximum on any statistic shown, "
                    "so the ranges reported in the text are unchanged by their "
                    "omission"
                    if not _moved else
                    "note that omitting them shifts the reported range of "
                    + ", ".join(_moved) +
                    ", so that range must be quoted from the ten-novel version"
                )
                if _moved:
                    print(f"  [!] {_nm_print}: dropping {DROPPED_FOR_PRINT} "
                          f"moves the range of {_moved}")
                _basis_note = (
                    "each panel on its own within-novel PCA basis. Because the "
                    "basis is refit per novel, coordinates are not comparable "
                    "across panels; each shows that novel's own retrieval "
                    "structure."
                    if not _cmp else
                    f"all panels on one shared linear basis, so coordinates ARE "
                    f"comparable across novels and the pooled corpus is drawn "
                    f"behind each panel in grey. The horizontal axis is the "
                    f"corpus-level norm-minus-flow contrast direction and the "
                    f"vertical is the leading component orthogonal to it. That "
                    f"horizontal axis is chosen using the norm/flow labels, so "
                    f"the clean left-right split is true by construction rather "
                    f"than discovered; the separation it displays is however a "
                    f"measured property of the space (Cohen's $d$ = "
                    f"{_m['cohens_d']:.1f} on this axis, 99.8% held-out linear "
                    f"separability). A single shared axis is defensible here "
                    f"because novels agree on this direction: pairwise cosine "
                    f"between their contrast directions is 0.79 (min 0.66)."
                )
                save_caption(
                    _nm_print,
                    f"Which norm each information flow retrieves, by source text "
                    f"({'all' if _ns_pb == 'all' else 'governing'} norms"
                    f"{', shared basis' if _cmp else ''})",
                    f"Every flow → nearest-same-novel-norm pair for eight of the "
                    f"ten novels, {_basis_note} Orange crosses are information flows, "
                    f"blue dots norms no flow ever retrieves, purple dots norms "
                    f"retrieved at least once, with area proportional to the "
                    f"square root of in-degree so hub norms are visible — "
                    f"retrieval is heavily top-heavy (median Gini of norm "
                    f"in-degree 0.81)"
                    + (", and the median novel has ~12 of its ~313 governing "
                       "norms absorbing half its flows. "
                       if _ns_pb == "governs" else ". ") +
                    f"Nothing is subsampled, so panel density is that novel's "
                    f"actual pair count. The arrow is that novel's mean "
                    f"displacement, a faithful projection of the 4096-D "
                    f"difference vector since the projection is linear"
                    + (" — note that on this basis the arrows point rightward by "
                       "construction, so they carry magnitude rather than "
                       "direction. " if _cmp else ". ") +
                    f"Each panel prints "
                    f"its pair count and $R$/null, where $R$ is the mean "
                    f"resultant length of that novel's displacement directions "
                    f"and null is the same quantity under a "
                    f"shuffled-within-novel pairing. Retrieval is computed in the "
                    f"full 4096-D embedding space and the panel is a lossy "
                    f"display of it: a segment asserts which norm a flow "
                    f"retrieved, not that the two are close on the page — in "
                    f"this basis the drawn norm is also the nearest norm on the "
                    f"page for {_hit:.1%} of flows (median rank {_pr:.0%} of the "
                    f"way through that novel's pool, against 50% for chance). "
                    f"Bleak House and Middlemarch are omitted for space; "
                    f"{_drop_note}, and the full ten-novel version is in the "
                    f"appendix.",
                    f"fig:norm-retrieval-per-book-{_meth}"
                    f"{'' if _ns_pb == 'all' else '-governs'}",
                    ["embedding-space", "displacement", "per-book", _meth,
                     "comparable-panels" if _cmp else "per-novel-basis",
                     f"norm-pool-{_ns_pb}", "print-subset", "camera-ready"],
                )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 6c — the pooled corpus in real 3D

    The per-book panels render PCA-3 through a fixed camera, which is
    mathematically identical to a 3D scatter viewed at that angle but shows no
    axis frame. At small-multiple size that is the right trade: the 3D box costs
    ~40% of the panel to dead space in its rotated corners and buys a depth cue
    too weak to perceive at 2.4 inches.

    At full size the trade reverses. The box gives orientation, the panes give
    depth, and structure that flattens into overplotting — most visibly the fan
    of segments converging on a hub norm — becomes legible. So the corpus gets
    one large 3D panel.

    Two notes. **The camera is the antipodal of the per-book one** (elev +60°,
    azim −80° rather than −60°, +100°): the view direction is exactly negated,
    so the projection plane and every distance in it are identical and the image
    is simply mirrored — but the box renders from above rather than from
    underneath, which reads far more naturally. And in real 3D the
    "how much variance reaches the page" question dissolves: all three
    components are drawn, so each axis carries its own explained variance.
    """)
    return


@app.cell
def _(
    FLOW_COLOR,
    FLOW_TEXT,
    INSTR,
    Line2D,
    MATCH_COLOR,
    NORM_COLOR,
    NORM_SETS,
    PAIRS,
    PROJ,
    norms,
    np,
    plt,
    save_caption,
    save_fig,
):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # Antipodal to the per-book camera: same plane, mirrored image, box seen
    # from above instead of from below.
    ELEV3D, AZIM3D = 60, -80
    N_SEG_3D = 700

    def draw_corpus_3d(norm_set, name):
        p = PAIRS[(FLOW_TEXT, INSTR, norm_set)]
        proj = PROJ[(FLOW_TEXT, INSTR, norm_set)]
        Z, n_norm, evr = proj["pca3_raw"], proj["n_norm"], proj["pca3_evr"]
        Zn, Zf = Z[:n_norm], Z[n_norm:]

        indeg = np.bincount(p["norm_pos"], minlength=n_norm)
        reached = indeg > 0

        rng = np.random.default_rng(7)
        pick = rng.choice(len(p["flow_idx"]),
                          size=min(N_SEG_3D, len(p["flow_idx"])), replace=False)

        fig = plt.figure(figsize=(7.6, 6.0))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(Zf[:, 0], Zf[:, 1], Zf[:, 2], c=FLOW_COLOR, s=1.4, alpha=0.16,
                   marker="x", linewidths=0.3, depthshade=False, rasterized=True)
        ax.scatter(Zn[~reached, 0], Zn[~reached, 1], Zn[~reached, 2], c=NORM_COLOR,
                   s=2.2, alpha=0.35, linewidths=0, depthshade=False, rasterized=True)
        ax.add_collection3d(Line3DCollection(
            np.stack([Zf[p["flow_idx"][pick]], Zn[p["norm_pos"][pick]]], axis=1),
            colors="#6E5B8A", linewidths=0.3, alpha=0.28, rasterized=True))
        # Hubs last and largest-first so they are never buried.
        hub = np.flatnonzero(reached)
        hub = hub[np.argsort(-indeg[hub])]
        ax.scatter(Zn[hub, 0], Zn[hub, 1], Zn[hub, 2], c=MATCH_COLOR,
                   s=3.0 + 3.1 * np.sqrt(indeg[hub]), alpha=0.85, linewidths=0.25,
                   edgecolors="white", depthshade=False, rasterized=True)

        ax.view_init(elev=ELEV3D, azim=AZIM3D)
        ax.set_proj_type("ortho")
        # Each axis carries its own variance — in real 3D nothing is discarded
        # by the camera, so there is no "reaches the page" caveat to make.
        for setter, lab, i in ((ax.set_xlabel, "PC1", 0), (ax.set_ylabel, "PC2", 1),
                               (ax.set_zlabel, "PC3", 2)):
            setter(f"{lab} ({evr[i]:.1%})", fontsize=8.5, labelpad=-4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_alpha(0.10)
            axis.pane.set_edgecolor("#CCCCCC")
        ax.set_box_aspect((1, 1, 0.9))
        ax.grid(True)

        handles = [
            Line2D([], [], marker="x", ls="none", color=FLOW_COLOR, ms=5, mew=1.2,
                   label=f"Information flows ({len(Zf):,})"),
            Line2D([], [], marker="o", ls="none", color=MATCH_COLOR, ms=5,
                   label=f"Norms retrieved by ≥1 flow ({int(reached.sum()):,}) "
                         f"— area ∝ √(flows)"),
            Line2D([], [], marker="o", ls="none", color=NORM_COLOR, ms=4,
                   label=f"Norms never retrieved ({int((~reached).sum()):,})"),
            Line2D([], [], color="#6E5B8A", lw=1.1,
                   label=f"flow → nearest same-book norm "
                         f"({N_SEG_3D:,} of {len(p['flow_idx']):,} drawn)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.005), fontsize=7.5)
        # mplot3d reserves generous padding around the box and ignores
        # tight_layout, leaving the data in a small central island. Placing the
        # axes explicitly (oversized, anchored past the figure edges) claws that
        # back — without it roughly half the panel is white.
        ax.set_position([-0.06, 0.02, 1.12, 1.02])
        save_fig(fig, name)
        return fig, int(reached.sum()), n_norm

    for _ns3 in ("all", "governs"):
        _sfx = "" if _ns3 == "all" else "_governs"
        _f3, _reached3, _npool3 = draw_corpus_3d(
            _ns3, f"fig_corpus_pca3_3d{_sfx}")
        _evr3 = PROJ[(FLOW_TEXT, INSTR, _ns3)]["pca3_evr"]
        save_caption(
            f"fig_corpus_pca3_3d{_sfx}",
            f"Norms and information flows in three principal components "
            f"({_ns3} norm pool)",
            f"All {len(NORM_SETS[_ns3]):,} norms of the "
            f"{'full extracted pool' if _ns3 == 'all' else 'information-flow-governing pool'} "
            f"and all 16,200 CI flows, projected onto the first three principal "
            f"components of the pooled Qwen3-Embedding-8B space "
            f"(PC1 {_evr3[0]:.1%}, PC2 {_evr3[1]:.1%}, PC3 {_evr3[2]:.1%}; "
            f"{_evr3.sum():.1%} of total variance). Drawn on true 3D axes rather "
            f"than a fixed 2D camera, so no component is discarded. Norm marker "
            f"area is proportional to the square root of the number of flows "
            f"retrieving that norm: {_reached3:,} of {_npool3:,} norms are "
            f"reached at least once, and retrieval is heavily concentrated on a "
            f"small number of hubs. A seeded subsample of "
            f"{N_SEG_3D:,} flow → nearest-same-novel-norm segments is drawn; the "
            f"point clouds are complete. Retrieval is computed in the full "
            f"4096-D space, so a segment records which norm a flow retrieved "
            f"rather than asserting the two are close on the page.",
            f"fig:corpus-pca3-3d{_sfx.replace('_', '-')}",
            ["embedding-space", "pca3", "3d", "hubs", "camera-ready"],
        )
    return (draw_corpus_3d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 6b — do the retrieved norms organise by context?

    `raz_context` is **free text, not a controlled vocabulary**: 309 distinct
    values over the 2,870 governing norms, many of them compound
    (`family/social propriety`, `gender/social propriety`). Taking the primary
    term before the first `/` collapses that to 72, and the top seven cover
    88%. Those seven plus an `other` bucket are what the figure can encode —
    eight is the ceiling for categorical hue.

    One caveat the colouring cannot fix: after normalisation **`social
    propriety` alone is 47% of the governing pool**, so roughly half the
    coloured markers carry one hue no matter what the geometry does.

    The clustering is real, but the projection loses most of it. Measured
    *within* each novel (the only meaningful frame for a per-book basis), kNN-10
    label purity:

    | space | purity | chance | lift | share of above-chance signal kept |
    |---|---|---|---|---|
    | 4096-D | 0.655 | 0.300 | **2.18×** | — |
    | rendered PCA-2 (primary) | 0.461 | 0.300 | 1.54× | 45% |
    | rendered MDS-2, balanced | 0.434 | 0.300 | 1.45× | 38% |

    So under half the above-chance context signal survives into the panel. PCA-2
    keeps more of it than the balanced MDS fit, which is a second reason (after
    linearity and nameable axes) to read context off the PCA panel.
    Context is a *stronger* local organiser than source text (book lift 2.37×
    corpus-wide, and context is only weakly confounded with book — Cramér's
    V = 0.26), but like book it forms no separable clusters: the 4096-D context
    silhouette is 0.04. Read the figure as "locally enriched, globally
    overlapping", and expect it to understate even that.
    """)
    return


@app.cell
def _(NORM_SETS, np, norms, pd, save_table):
    # Primary term before the first '/', lowercased. Top 7 by frequency keep an
    # identity; the tail folds into `other`. Assignment is fixed by frequency
    # rank, never cycled, so a hue means the same context in every panel.
    _prim = (norms["raz_context"].fillna("(missing)").astype(str)
             .str.strip().str.lower().str.split("/").str[0].str.strip())
    CONTEXT_PRIMARY = np.asarray(_prim.tolist(), dtype=object)

    _gov_prim = CONTEXT_PRIMARY[NORM_SETS["governs"]]
    TOP_CONTEXTS = list(pd.Series(_gov_prim).value_counts().head(7).index)

    def context_labels(norm_set):
        lab = CONTEXT_PRIMARY[NORM_SETS[norm_set]]
        return np.asarray([c if c in TOP_CONTEXTS else "other" for c in lab],
                          dtype=object)

    # Validated with the dataviz palette checker (light surface, categorical):
    # passes lightness band, chroma floor, CVD separation (worst adjacent dE
    # 10.4 deutan / 19.3 tritan), normal-vision floor, and 3:1 contrast. This
    # ORDER is part of the result — reordering breaks the adjacent-pair checks.
    _HUES = ["#0072B2", "#D55E00", "#009E73", "#B07800",
             "#6A3D9A", "#3E8FC7", "#A64D79"]
    CONTEXT_COLORS = {k: c for k, c in zip(TOP_CONTEXTS, _HUES)}
    # Deliberate neutral for the catch-all: `other` is not an identity competing
    # with the seven, so it should not carry a hue.
    CONTEXT_COLORS["other"] = "#B0B0B0"

    context_sizes = (pd.Series(context_labels("governs")).value_counts()
                     .rename_axis("context").reset_index(name="n_governing_norms"))
    context_sizes["share"] = (context_sizes["n_governing_norms"]
                              / context_sizes["n_governing_norms"].sum())
    save_table(context_sizes, "context_pool_sizes")
    print(context_sizes.to_string(index=False))
    return CONTEXT_COLORS, CONTEXT_PRIMARY, TOP_CONTEXTS, context_labels


@app.cell
def _(
    FBOOK,
    FLOW_EMB,
    FLOW_TEXT,
    INSTR,
    NBOOK,
    NORM_SETS,
    context_labels,
    fit_per_book_layouts,
    norm_emb,
    np,
    pd,
    save_table,
):
    from sklearn.neighbors import NearestNeighbors as _NN

    def context_clustering(norm_set="governs", k=10):
        """kNN label purity for context, computed inside each novel.

        Within-book is the only meaningful frame here: the per-book layouts
        share no basis, so a global kNN in the stacked 2D coordinates would
        count norms from different novels that happen to land nearby as
        neighbours. Each novel is scored against its own chance rate.
        """
        lab = context_labels(norm_set)
        nb = NBOOK[NORM_SETS[norm_set]]
        ne = norm_emb(INSTR, norm_set)
        # Score every layout the context figure is published in, so the
        # "surviving signal" number always describes the panel being read. PCA-2
        # is the primary basis; mds2bal is kept because it is the one to read
        # configuration from and its purity is the higher of the two.
        xy = {t: fit_per_book_layouts(FLOW_TEXT, INSTR, norm_set, t)[: len(ne)]
              for t in ("pca2", "mds2bal")}
        tags = ("4096d", "pca2", "mds2bal")

        rows = []
        for bk in np.unique(nb):
            m = np.flatnonzero(nb == bk)
            if len(m) < k + 2:
                continue
            L = lab[m]
            p = pd.Series(L).value_counts(normalize=True).values
            out = {"book": bk, "n_norms": len(m), "n_contexts": len(set(L)),
                   "chance": float((p ** 2).sum())}
            for tag, P, metric in (("4096d", ne[m], "cosine"),
                                   ("pca2", xy["pca2"][m], "euclidean"),
                                   ("mds2bal", xy["mds2bal"][m], "euclidean")):
                _, I = _NN(n_neighbors=k + 1, metric=metric).fit(P).kneighbors(P)
                out[f"purity_{tag}"] = float((L[I[:, 1:]] == L[:, None]).mean())
            rows.append(out)
        df = pd.DataFrame(rows)
        for tag in tags:
            df[f"lift_{tag}"] = df[f"purity_{tag}"] / df["chance"]
        return df

    context_clusters = context_clustering("governs")
    save_table(context_clusters, "context_clustering_governs")
    _w = context_clusters["n_norms"] / context_clusters["n_norms"].sum()
    _ch = float((context_clusters["chance"] * _w).sum())
    _hi = float((context_clusters["purity_4096d"] * _w).sum())
    print(context_clusters.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nweighted: chance {_ch:.3f} | 4096-D {_hi:.3f} (lift {_hi/_ch:.2f}x)")
    for _tag, _nm in (("pca2", "PCA-2 (primary)"), ("mds2bal", "MDS-2 balanced")):
        _2d = float((context_clusters[f"purity_{_tag}"] * _w).sum())
        print(f"  {_nm:<18} {_2d:.3f} (lift {_2d/_ch:.2f}x) — "
              f"{(_2d - _ch) / (_hi - _ch):.0%} of the above-chance signal survives")
    return (context_clusters,)


@app.cell
def _(
    CONTEXT_COLORS,
    FLOW_TEXT,
    INSTR,
    LAYOUT_META,
    PER_BOOK,
    PRINT_BOOKS,
    VIEW_AZIM,
    VIEW_ELEV,
    context_labels,
    draw_per_book,
    fit_per_book_layouts,
):
    # Only bases that are safe to read for spatial organisation: PCA-2 and
    # `contrast` (both linear, so neither can reposition a class by cardinality)
    # and the class-balanced MDS fit. The UNBALANCED mds2 layout is deliberately
    # not offered here — it is the one that manufactures a ring.
    for _ns_ctx in ("all", "governs"):
        _suffix = "" if _ns_ctx == "all" else "_governs"
        for _m_ctx, _lab_ctx in (
            ("pca2", ("PC1 (per novel)", "PC2 (per novel)")),
            # On the shared basis a context hue means the same location in every
            # panel, so context can be compared across novels rather than only
            # within one.
            ("contrast", None),
            ("mds2bal", ("metric MDS, class-balanced fit (per novel)", "MDS-2")),
        ):
            _layout_c = fit_per_book_layouts(FLOW_TEXT, INSTR, _ns_ctx, _m_ctx,
                                             elev=VIEW_ELEV, azim=VIEW_AZIM)
            _cmp_c = _m_ctx == "contrast"
            if _cmp_c:
                _mc = LAYOUT_META[f"fiction10_{FLOW_TEXT}_{INSTR}_{_ns_ctx}"
                                  f"_perbook_contrast"]
                _lab_ctx = (f"norm ↔ flow contrast axis, chosen on the class "
                            f"labels ({_mc['var_x']:.1%} var.; Cohen's $d$ = "
                            f"{_mc['cohens_d']:.1f})",
                            f"leading orthogonal PC ({_mc['var_y']:.1%} var.)")
            draw_per_book(
                FLOW_TEXT, INSTR, _ns_ctx, _m_ctx,
                f"fig_paired_displacement_per_book_{_m_ctx}_bycontext{_suffix}",
                PER_BOOK[_ns_ctx], layout=_layout_c, axis_label=_lab_ctx,
                context_lab=context_labels(_ns_ctx),
                context_colors=CONTEXT_COLORS, comparable=_cmp_c,
            )
            if _m_ctx in ("pca2", "contrast"):
                draw_per_book(
                    FLOW_TEXT, INSTR, _ns_ctx, _m_ctx,
                    f"fig_paired_displacement_per_book_{_m_ctx}_bycontext_print{_suffix}",
                    PER_BOOK[_ns_ctx], layout=_layout_c, axis_label=_lab_ctx,
                    context_lab=context_labels(_ns_ctx),
                    context_colors=CONTEXT_COLORS,
                    books=PRINT_BOOKS, print_size=True, also_paper=True,
                    comparable=_cmp_c, draw_xlabel=False,
                )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 4b — how concentrated is retrieval?

    The reached/unreached split says 59% of the governing pool is touched at
    least once, but says nothing about *how* the flows distribute over the norms
    they do reach — and overplotting hides it, since 575 segments landing on one
    dot are indistinguishable from 575 landing on 575. They are extremely
    top-heavy: median Gini of norm in-degree 0.81, and the median novel has ~12
    of its ~313 governing norms absorbing half its flows.

    This bounds what R-GROUND can discriminate. The effective normative universe
    the reward actually resolves per novel is on the order of tens of norms, not
    hundreds.
    """)
    return


@app.cell
def _(
    BOOKS,
    FBOOK,
    FLOW_EMB,
    FLOW_TEXT,
    INSTR,
    MATCH_COLOR,
    NBOOK,
    NORM_SETS,
    PAIRS,
    SHORT_TITLE,
    np,
    pd,
    plt,
    save_caption,
    save_fig,
    save_table,
):
    def retrieval_concentration(norm_set="governs"):
        p = PAIRS[(FLOW_TEXT, INSTR, norm_set)]
        fbook_pairs = FBOOK[p["flow_idx"]]
        nbook_pool = NBOOK[NORM_SETS[norm_set]]
        out = {}
        rows = []
        for bk in BOOKS:
            pool = np.flatnonzero(nbook_pool == bk)
            m = fbook_pairs == bk
            # In-degree over this novel's pool positions, descending.
            deg = np.sort(np.bincount(
                np.searchsorted(pool, p["norm_pos"][m]), minlength=len(pool)))[::-1]
            out[bk] = deg
            cum = np.cumsum(deg) / deg.sum()
            g = np.sort(deg)
            n = len(g)
            rows.append({
                "book": bk,
                "n_flows": int(m.sum()),
                "n_norms": n,
                "frac_reached": float((deg > 0).mean()),
                "gini_indegree": float((2 * np.arange(1, n + 1) - n - 1) @ g / (n * g.sum())),
                "top_norm_share": float(deg[0] / deg.sum()),
                "norms_for_50pct": int(np.searchsorted(cum, 0.5) + 1),
                "max_indegree": int(deg[0]),
            })
        return pd.DataFrame(rows), out

    conc_table, DEG = retrieval_concentration("governs")
    conc_table = conc_table.sort_values("n_flows", ascending=False).reset_index(drop=True)
    save_table(conc_table, "retrieval_concentration_governs", index=False)
    print(conc_table.to_string(index=False))

    _order = list(conc_table["book"])
    # Sequential ramp keyed to corpus size (a magnitude, so a ramp is the right
    # encoding); the two novels called out in the text are direct-labelled, so
    # identity never rests on colour alone.
    _ramp = plt.cm.PuBu(np.linspace(0.88, 0.32, len(_order)))

    import textwrap

    _fig = plt.figure(figsize=(11.6, 3.5))
    # Generous wspace: (c)'s y-tick labels are novel titles and run into (b)'s
    # plot area at anything tighter, even with the titles wrapped.
    _gs = _fig.add_gridspec(1, 3, width_ratios=[1.05, 0.92, 1.2], wspace=0.52)

    _ax = _fig.add_subplot(_gs[0, 0])
    _ax.plot([0, 1], [0, 1], ls=":", lw=0.8, color="#9A9A9A", zorder=1)
    for _c, _bk in zip(_ramp, _order):
        _d = DEG[_bk]
        _ax.plot(np.r_[0, np.arange(1, len(_d) + 1) / len(_d)],
                 np.r_[0, np.cumsum(_d) / _d.sum()], lw=1.3, color=_c, zorder=2)
    _ax.axhline(0.5, lw=0.7, ls="--", color="#555555", zorder=1)
    # Anchored right of centre: every curve is already above 0.9 by x=0.5, so
    # this sits in clear space. At the left edge it lands on the steep rise.
    _ax.text(0.97, 0.47, "50% of flows", fontsize=6.3, color="#555555",
             va="top", ha="right")
    _ax.set_xlabel("norms, ranked by in-degree (share of pool)")
    _ax.set_ylabel("cumulative share of flows")
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1.02)
    _ax.set_title("(a) retrieval is concentrated on few norms", fontsize=9)

    _ax = _fig.add_subplot(_gs[0, 1])
    for _c, _bk in zip(_ramp, _order):
        _d = DEG[_bk]
        _d = _d[_d > 0]
        _ax.plot(np.arange(1, len(_d) + 1), _d, lw=1.2, color=_c)
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax.set_xlabel("norm rank within novel")
    _ax.set_ylabel("flows retrieving it")
    _ax.set_title("(b) in-degree is heavy-tailed", fontsize=9)

    _ax = _fig.add_subplot(_gs[0, 2])
    _ax.grid(False)
    _ax.xaxis.grid(True, alpha=0.25, lw=0.5)
    for _i, _bk in enumerate(_order[::-1]):
        _r = conc_table.set_index("book").loc[_bk]
        _c = _ramp[_order.index(_bk)]
        # .loc on a mixed-dtype row upcasts these counts to float, which renders
        # as "26.0 of 500.0" — cast back before formatting.
        _n_pool, _n_half = int(_r["n_norms"]), int(_r["norms_for_50pct"])
        _ax.hlines(_i, 0, _n_pool, color="#E2E2E2", lw=3.2, zorder=1)
        _ax.hlines(_i, 0, _n_half, color=_c, lw=3.2, zorder=2)
        _ax.text(_n_pool + 9, _i, f"{_n_half} of {_n_pool}",
                 va="center", fontsize=6.4, color="#333333")
    _ax.set_yticks(range(len(_order)))
    _ax.set_yticklabels(
        ["\n".join(textwrap.wrap(SHORT_TITLE[_b], 13)) for _b in _order[::-1]],
        fontsize=6.5,
    )
    _ax.set_xlim(0, conc_table["n_norms"].max() * 1.22)
    _ax.set_xlabel("governing norms in the novel's pool")
    _ax.set_title("(c) norms absorbing half the novel's flows", fontsize=9)
    for _sp in ("top", "right", "left"):
        _ax.spines[_sp].set_visible(False)

    save_fig(_fig, "fig_retrieval_concentration")
    save_caption(
        "fig_retrieval_concentration",
        "Concentration of flow-to-norm retrieval",
        f"How the {int(conc_table['n_flows'].sum()):,} CI flows distribute over the "
        f"{int(conc_table['n_norms'].sum()):,} information-flow-governing norms they "
        f"retrieve, per novel. (a) Lorenz curves of norm in-degree: every novel "
        f"hugs the top-left, so the shape is a property of the corpus and not of "
        f"one text. (b) the same in-degrees as rank-frequency curves on log-log "
        f"axes. (c) the number of norms that between them absorb half of a "
        f"novel's flows, against its full governing pool. Retrieval is extremely "
        f"top-heavy: median Gini {conc_table['gini_indegree'].median():.2f}, and "
        f"the median novel needs only {int(conc_table['norms_for_50pct'].median())} "
        f"of its {int(conc_table['n_norms'].median())} governing norms to cover "
        f"half its flows; in Anna Karenina a single norm is the nearest neighbour "
        f"for {conc_table.set_index('book').loc['Anna Karenina', 'top_norm_share']:.0%} "
        f"of 2,491 flows. Together with the "
        f"{1 - conc_table['frac_reached'].mean():.0%} of the pool that is never "
        f"retrieved at all, this bounds what the grounding reward can "
        f"discriminate: the effective normative universe per novel is tens of "
        f"norms, not hundreds.",
        "fig:retrieval-concentration",
        ["embedding-space", "retrieval", "concentration", "hubs", "camera-ready"],
    )
    _fig
    return (conc_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Table — the hub norms themselves

    Figure 4b says retrieval is top-heavy but not *what* it is top-heavy on. The
    hubs are the norms with the largest in-degree in each novel's governing
    pool — the handful most of that novel's flows resolve to, so they are the
    operative content of the normative universe the reward retrieves against,
    and worth printing verbatim rather than summarising.

    Two views are written:

    - `hub_norms_governs.csv` — top `HUB_TOP_K_CSV` per novel with in-degree,
      share of the novel's flows, mean retrieval cosine, deontic force, context,
      and the articulation.
    - `hub_norms.tex` (`\autoref{tab:hub-norms}`) — the top `HUB_TOP_K_TEX` per
      novel, as an `\input`-able table.

    In-degree is computed on the same pairing every figure above uses: nearest
    same-book norm over the **governing** pool, flow string without the
    appropriateness verdict, one shared embedding instruction. Books are ordered
    by corpus size, matching Figure 4b.
    """)
    return


@app.cell
def _(
    FBOOK,
    FLOW_TEXT,
    INSTR,
    NBOOK,
    NORM_SETS,
    PAIRS,
    SHORT_TITLE,
    conc_table,
    norms,
    np,
    pd,
    save_table,
    save_tex,
):
    HUB_TOP_K_CSV = 5   # written to CSV for inspection
    HUB_TOP_K_TEX = 2   # printed in the paper table
    # Articulations run to 245 characters. None = print them whole; set an int to
    # truncate on a word boundary if a hub ever pushes the table off the page.
    HUB_ART_MAXLEN = None

    _p = PAIRS[(FLOW_TEXT, INSTR, "governs")]
    _pool = NORM_SETS["governs"]
    _nbook_pool = NBOOK[_pool]
    _fbook_pairs = FBOOK[_p["flow_idx"]]

    _rows = []
    for _bk in conc_table["book"]:
        _m = np.flatnonzero(_fbook_pairs == _bk)
        _pos = _p["norm_pos"][_m]
        _sim = _p["sim"][_m]
        # In-degree over the pool positions this novel's flows actually reached;
        # never-reached norms are absent by construction and cannot be hubs.
        _uniq, _inv = np.unique(_pos, return_inverse=True)
        _deg = np.bincount(_inv)
        _cos = np.bincount(_inv, weights=_sim) / _deg
        # lexsort on (pool position, -in-degree): deg ties break by position, so
        # the printed hubs are stable across runs rather than argsort-dependent.
        _order = np.lexsort((_uniq, -_deg))
        _npool = int((_nbook_pool == _bk).sum())
        for _r, _j in enumerate(_order[:HUB_TOP_K_CSV]):
            _nrow = _pool[_uniq[_j]]
            _rows.append({
                "book": _bk,
                "rank": _r + 1,
                "indegree": int(_deg[_j]),
                "share_of_book_flows": float(_deg[_j] / len(_m)),
                "cum_share_top_k": float(_deg[_order[:_r + 1]].sum() / len(_m)),
                "mean_retrieval_cos": float(_cos[_j]),
                "book_flows": len(_m),
                "book_governing_norms": _npool,
                "force": norms["raz_normative_force"].values[_nrow],
                "context": norms["raz_context"].values[_nrow],
                "articulation": norms["raz_norm_articulation"].values[_nrow],
                "gutenberg_id": norms["gutenberg_id"].values[_nrow],
                "chunk_id": int(norms["chunk_id"].values[_nrow]),
                "norm_row": int(_nrow),
            })

    hub_norms = pd.DataFrame(_rows)
    save_table(hub_norms, "hub_norms_governs", index=False)

    # --- LaTeX ------------------------------------------------------------
    _ACCENTS = {
        "à": r"\`{a}", "á": r"\'{a}", "â": r"\^{a}", "ä": r'\"{a}', "ã": r"\~{a}",
        "è": r"\`{e}", "é": r"\'{e}", "ê": r"\^{e}", "ë": r'\"{e}',
        "î": r"\^{i}", "ï": r'\"{i}', "ô": r"\^{o}", "ö": r'\"{o}',
        "û": r"\^{u}", "ü": r'\"{u}', "ñ": r"\~{n}", "ç": r"\c{c}",
        "–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''",
        "…": r"\ldots{}",
    }

    def tex_escape(s):
        s = str(s)
        for _a, _b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                       ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                       ("}", r"\}"), ("~", r"\textasciitilde{}"),
                       ("^", r"\textasciicircum{}")):
            s = s.replace(_a, _b)
        for _a, _b in _ACCENTS.items():
            s = s.replace(_a, _b)
        # The novel titles and the Gemma-4 articulations are ASCII apart from the
        # accents above; anything else would reach LaTeX as a compile error many
        # pages away from its cause, so fail here instead.
        _bad = {_c for _c in s if ord(_c) > 127}
        assert not _bad, f"unescaped non-ASCII {_bad!r} in {s!r}"
        return s

    def _clip(s, n):
        if n is None or len(s) <= n:
            return s
        return s[:n].rsplit(" ", 1)[0].rstrip(",;:") + r" \ldots{}"

    # Number formatting is per-field, never over the assembled row: a blanket
    # `,` -> `{,}` on the row string would reach into the articulation and put
    # math-mode braces in the middle of an English sentence.
    def _n(x):
        return f"{int(x):,}".replace(",", "{,}")

    def _pct(x):
        return rf"{round(100 * x)}\%"

    _lines = []
    for _i, _bk in enumerate(conc_table["book"]):
        _g = hub_norms[hub_norms["book"] == _bk].head(HUB_TOP_K_TEX)
        if _i:
            _lines.append(r"\addlinespace[2pt]")
        for _k, (_, _h) in enumerate(_g.iterrows()):
            # Novel and pool size label the group once; repeating them on every
            # hub row would multiply the ink for no information.
            _label = (
                rf"{tex_escape(SHORT_TITLE[_bk])} \newline "
                rf"\textcolor{{gray}}{{{_n(_h['book_flows'])} flows / "
                rf"{_n(_h['book_governing_norms'])} norms}}"
                if _k == 0 else ""
            )
            _lines.append(
                f"{_label} & {_n(_h['indegree'])} "
                f"({_pct(_h['share_of_book_flows'])}) & "
                f"{_h['mean_retrieval_cos']:.2f} & "
                f"{tex_escape(_h['force'])} & "
                f"{tex_escape(_clip(_h['articulation'], HUB_ART_MAXLEN))} \\\\"
            )

    _med_gini = conc_table["gini_indegree"].median()
    _top2 = (hub_norms[hub_norms["rank"] <= HUB_TOP_K_TEX]
             .groupby("book")["share_of_book_flows"].sum())
    _caption = (
        rf"\textbf{{Hub norms.}} The {HUB_TOP_K_TEX} highest-in-degree norms in "
        rf"each source text's information-flow-governing pool: the norms that are "
        rf"the nearest same-novel neighbor of the largest number of that novel's "
        rf"CI flows, and therefore the operative content of the normative "
        rf"universe $R_{{\text{{direct}}}}$ and $R_{{\text{{ground}}}}$ retrieve "
        rf"against. \textit{{Flows}} is in-degree, with its share of the novel's "
        rf"flows; \textit{{cos}} is the mean retrieval cosine over those flows. "
        rf"Retrieval is "
        rf"heavily hubbed --- median Gini of in-degree {_med_gini:.2f}, and these "
        rf"{HUB_TOP_K_TEX} norms alone absorb a median "
        rf"{_pct(_top2.median())} of a novel's flows "
        rf"(max {_pct(_top2.max())}, "
        rf"{tex_escape(SHORT_TITLE[_top2.idxmax()])}) --- so the effective "
        rf"normative universe per novel is tens of norms, not the hundreds the "
        rf"pool contains."
    )

    # Both text columns are ragged-right: justified setting in a 0.15\textwidth
    # column opens interword rivers that read as broken. Hyphenation is off in
    # the title column only — "Pride and Preju-dice" is worse than a short line,
    # while the articulations are ordinary prose that needs it to avoid
    # overfulls. `\arraybackslash` restores `\\` after `\raggedright`.
    _titlecol = (r">{\raggedright\hyphenpenalty=10000\exhyphenpenalty=10000"
                 r"\arraybackslash}p{0.15\textwidth}")
    _artcol = r">{\raggedright\arraybackslash}p{0.54\textwidth}"

    _tex = "\n".join([
        r"\begin{table}[ht]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        rf"\begin{{tabular}}{{@{{}}{_titlecol}rrl {_artcol}@{{}}}}",
        r"\toprule",
        r"Source text & Flows & cos & Force & Norm articulation \\",
        r"\midrule",
        "\n".join(_lines),
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{_caption}}}",
        r"\label{tab:hub-norms}",
        r"\end{table}",
    ])
    save_tex(_tex, "hub_norms")

    print(hub_norms[["book", "rank", "indegree", "share_of_book_flows",
                     "mean_retrieval_cos", "force", "articulation"]]
          .to_string(index=False, max_colwidth=90))
    hub_norms
    return HUB_ART_MAXLEN, HUB_TOP_K_CSV, HUB_TOP_K_TEX, hub_norms, tex_escape


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 5 — directional concentration

    (a) distribution of $\cos(d_i, \bar d)$ for the true pairing against the
    shuffled-within-book null. (b) the same directions as a polar histogram, in
    the plane spanned by $\bar d$ and the leading residual direction — angle 0
    is exactly along the mean displacement. (c) pairwise cosine between the ten
    per-novel mean displacements $\bar d_b$: is the flow→norm offset one shared
    direction, or does each novel have its own?
    """)
    return


@app.cell
def _(
    BOOKS,
    FBOOK,
    FLOW_TEXT,
    INSTR,
    LINK_COLOR,
    NULL_COLOR,
    PAIRS,
    SHORT_TITLE,
    displacement,
    np,
    pair_flows,
    pd,
    plt,
    save_fig,
    save_table,
):
    _p = PAIRS[(FLOW_TEXT, INSTR, "all")]
    _d = _p["d"]
    _cos_true = _p["cos"]
    _dbar = _p["dbar"]
    _u_hat = _dbar / np.linalg.norm(_dbar)

    # Null cosines, pooled over draws, measured against the TRUE dbar so both
    # histograms answer the same question ("how aligned with the observed mean
    # displacement is a pair drawn this way?").
    _cos_null = []
    for _s in range(10):
        _fis, _nis = pair_flows(FLOW_TEXT, INSTR, "all", rng=np.random.default_rng(2000 + _s))
        _dn = displacement(FLOW_TEXT, INSTR, "all", _fis, _nis)
        _cos_null.append(
            (_dn / np.maximum(np.linalg.norm(_dn, axis=1, keepdims=True), 1e-12)) @ _u_hat
        )
    _cos_null = np.concatenate(_cos_null)

    # Plane for the polar view: u_hat and the leading residual direction.
    _u = _d / np.maximum(np.linalg.norm(_d, axis=1, keepdims=True), 1e-12)
    _resid = _u - np.outer(_u @ _u_hat, _u_hat)
    _sub = np.random.default_rng(3).choice(len(_resid), min(4000, len(_resid)), replace=False)
    _, _, _Vt = np.linalg.svd(_resid[_sub], full_matrices=False)
    _v_hat = _Vt[0] / np.linalg.norm(_Vt[0])
    _theta = np.arctan2(_u @ _v_hat, _u @ _u_hat)

    # Per-book mean displacements and their pairwise alignment.
    _fbook = FBOOK[_p["flow_idx"]]
    _D = np.stack([_d[_fbook == _bk].mean(axis=0) for _bk in BOOKS])
    _D = _D / np.linalg.norm(_D, axis=1, keepdims=True)
    _align = _D @ _D.T
    book_dbar_alignment = pd.DataFrame(_align, index=BOOKS, columns=BOOKS)
    save_table(book_dbar_alignment, "book_dbar_alignment")
    _off = _align[np.triu_indices(len(BOOKS), 1)]
    print(f"cross-book cos(dbar_b, dbar_b'): min {_off.min():.3f}  "
          f"median {np.median(_off):.3f}  max {_off.max():.3f}")

    _fig = plt.figure(figsize=(12.2, 3.6))
    # Generous wspace: the polar panel's 0-degree tick sits on its right edge
    # and lands on the heatmap's row labels at anything tighter.
    _gs = _fig.add_gridspec(1, 3, width_ratios=[1.15, 0.8, 1.25], wspace=0.62)

    # --- (a) cosine distributions -----------------------------------------
    _ax = _fig.add_subplot(_gs[0, 0])
    _bins = np.linspace(-1, 1, 81)
    _ax.hist(_cos_null, bins=_bins, density=True, color=NULL_COLOR, alpha=0.55,
             label=f"shuffled within book ($R$={_p['null_R'].mean():.3f})", zorder=1)
    _ax.hist(_cos_true, bins=_bins, density=True, histtype="step", color=LINK_COLOR,
             lw=1.6, label=f"nearest same-book norm ($R$={_p['R']:.3f})", zorder=2)
    _ax.axvline(0, color="#777777", lw=0.7, ls=":")
    _ax.annotate("isotropic field\nwould centre here",
                 xy=(0, 0.3), xycoords=("data", "axes fraction"),
                 xytext=(-0.55, 0.52), textcoords=("data", "axes fraction"),
                 fontsize=6.5, va="center", ha="left", color="#777777",
                 arrowprops=dict(arrowstyle="->", color="#999999", lw=0.6))
    _ax.set_xlabel(r"$\cos(d_i,\ \bar{d})$")
    _ax.set_ylabel("density")
    _ax.set_xlim(-0.65, 1.0)
    _ax.set_title("(a) alignment with the mean displacement", fontsize=9)
    _ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(-0.02, 1.02))

    # --- (b) polar histogram ----------------------------------------------
    _axp = _fig.add_subplot(_gs[0, 1], projection="polar")
    _counts, _edges = np.histogram(_theta, bins=48, range=(-np.pi, np.pi))
    _width = _edges[1] - _edges[0]
    _axp.bar(_edges[:-1] + _width / 2, _counts / _counts.sum(), width=_width,
             color=LINK_COLOR, alpha=0.8, edgecolor="white", linewidth=0.3)
    _axp.set_theta_zero_location("E")
    _axp.set_yticklabels([])
    _axp.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    _axp.set_xticklabels(["0°", "", "90°", "", "180°", "", "270°", ""], fontsize=7)
    _axp.grid(alpha=0.3, lw=0.4)
    _axp.set_title(r"(b) angle to $\bar{d}$ in the $(\bar{d},\,v_1)$ plane",
                   fontsize=9, pad=12)

    # --- (c) cross-book alignment of the per-novel mean displacements ------
    _ax = _fig.add_subplot(_gs[0, 2])
    _ax.grid(False)
    _im = _ax.imshow(_align, cmap="YlGnBu", vmin=max(0.0, _off.min() - 0.02), vmax=1.0)
    _short = [SHORT_TITLE[_b] for _b in BOOKS]
    _ax.set_xticks(range(len(BOOKS)))
    _ax.set_yticks(range(len(BOOKS)))
    _ax.set_xticklabels(_short, rotation=45, ha="right", fontsize=6)
    _ax.set_yticklabels(_short, fontsize=6)
    for _i in range(len(BOOKS)):
        for _j in range(len(BOOKS)):
            _ax.text(_j, _i, f"{_align[_i, _j]:.2f}", ha="center", va="center",
                     fontsize=4.8,
                     color="white" if _align[_i, _j] > 0.5 * (1 + _off.min()) else "#222222")
    _cb = plt.colorbar(_im, ax=_ax, shrink=0.82, pad=0.02)
    _cb.ax.tick_params(labelsize=6)
    _ax.set_title(r"(c) $\cos(\bar{d}_b,\ \bar{d}_{b'})$ across novels", fontsize=9)

    save_fig(_fig, "fig_displacement_concentration")
    _fig
    return (book_dbar_alignment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Qualitative check — what a pair actually looks like

    Flow → nearest-same-book-norm pairs at high, median and low retrieval
    similarity, over the governing-norm pool. The flow string shown is the one
    actually embedded, i.e. without the appropriateness verdict.
    """)
    return


@app.cell
def _(FBOOK, FLOW_TEXT, INSTR, NORM_SETS, PAIRS, flows, norms, np, pd, save_table):
    _p = PAIRS[(FLOW_TEXT, INSTR, "governs")]
    _sim = _p["sim"]
    _order = np.argsort(-_sim)
    _mid = len(_order) // 2
    _picks = np.r_[_order[:6], _order[_mid - 3:_mid + 3], _order[-6:]]
    _rows = NORM_SETS["governs"][_p["norm_pos"][_picks]]

    pair_examples = pd.DataFrame({
        "band": ["high"] * 6 + ["median"] * 6 + ["low"] * 6,
        "cos": _sim[_picks].round(3),
        "book": FBOOK[_p["flow_idx"][_picks]],
        "flow": flows[f"embed_text_{FLOW_TEXT}"].values[_p["flow_idx"][_picks]],
        "norm": norms["raz_norm_articulation"].values[_rows],
    })
    save_table(pair_examples, "pair_examples", index=False)
    pair_examples
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Caption sidecars""")
    return


@app.cell
def _(
    BOOKS,
    FLOW_TEXT,
    INSTR,
    PAIRS,
    PER_BOOK,
    PROJ,
    book_dbar_alignment,
    book_structure,
    concentration_table,
    flows,
    layout_fidelity,
    norm_pool_sizes,
    norms,
    np,
    save_caption,
    separation,
):
    def _sep(pool, instr, text):
        return separation.query(
            "norm_pool == @pool and instruction == @instr and flow_text == @text").iloc[0]

    def _con(pool, instr=INSTR, text=FLOW_TEXT):
        return concentration_table.query(
            "norm_pool == @pool and instruction == @instr and flow_text == @text").iloc[0]

    _s = _sep("all", INSTR, FLOW_TEXT)
    _s_full = _sep("all", INSTR, "full")
    _s_desc = _sep("all", INSTR, "descriptive")
    _c_all, _c_gov = _con("all"), _con("governs")
    _c_prod = _con("all", instr="prod")
    _bs_norm = book_structure.query("construct == 'norm (all)'").iloc[0]
    _bs_flow = book_structure.query("construct == 'flow'").iloc[0]
    _off = book_dbar_alignment.values[np.triu_indices(len(BOOKS), 1)]
    _pa, _pg = PER_BOOK["all"], PER_BOOK["governs"]

    save_caption(
        "fig_norm_flow_per_book_umap",
        "Norms and information flows per source text",
        f"2D UMAP projection of Qwen3-Embedding-8B embeddings of all "
        f"{len(norms):,} norms and {len(flows):,} information flows extracted "
        f"from the {len(BOOKS)} fiction10 source texts by Gemma-4-31B-it. Both "
        f"constructs are embedded under one shared instruction, and the flow "
        f"string carries no appropriateness verdict, so neither the separation "
        f"nor its direction can be an artifact of the serialization. Each panel "
        f"highlights one text against the pooled corpus in grey. The two "
        f"constructs are almost perfectly separable in the 4096-D space: a "
        f"mean-difference direction fit on half the rows separates the held-out "
        f"half with {_s['heldout_acc']:.1%} accuracy (Cohen's "
        f"$d$ = {_s['heldout_cohens_d']:.1f}, marginal overlap "
        f"{_s['heldout_overlap']:.3f}), even though that axis carries only "
        f"{_s['contrast_var_frac']:.1%} of total variance — which is why the "
        f"full-space silhouette ({_s['silhouette_4096d']:.2f}) understates it. "
        f"The source texts, by contrast, share "
        f"one space rather than partitioning it: a norm's 20 nearest neighbours "
        f"come from its own text {_bs_norm['knn20_book_purity']:.0%} of the time "
        f"against a {_bs_norm['chance_purity']:.0%} chance rate (flows "
        f"{_bs_flow['knn20_book_purity']:.0%} versus "
        f"{_bs_flow['chance_purity']:.0%}), so neighbourhoods are strongly "
        f"text-enriched, but the 10-way book silhouette is "
        f"{_bs_norm['book_silhouette']:.2f} — the texts do not form separable "
        f"clusters.",
        "fig:norm-flow-per-book-umap",
        ["embedding-space", "umap", "norms", "flows", "camera-ready"],
    )

    save_caption(
        "fig_paired_displacement",
        "Paired displacement from flow space to norm space",
        f"Each of the {_c_all['n_pairs']:,} CI flows is joined to its nearest "
        f"norm from the same novel. Norms are coloured by whether any flow "
        f"retrieves them: {_c_all['norms_reached']:,} of "
        f"{_c_all['norms_in_pool']:,} "
        f"({_c_all['frac_norms_reached']:.0%}) are reached at least once. Left: "
        f"PCA, a linear projection in which segment direction faithfully "
        f"projects the 4096-D displacement. Right: the same pairs on UMAP, which "
        f"shows which neighbourhoods pair with which but — being non-linear — "
        f"supports no claim about direction. A seeded subsample of segments is "
        f"drawn; the full point clouds are complete. The displacement field is "
        f"strongly coherent: mean resultant length R = {_c_all['R']:.3f} against "
        f"~0 for an isotropic field.",
        "fig:paired-displacement",
        ["embedding-space", "displacement", "pairing", "camera-ready"],
    )

    for _nm, _pool, _proj, _stats, _cc, _lbl in (
        ("fig_paired_displacement_per_book", "all", "PCA", _pa, _c_all,
         "fig:paired-displacement-per-book"),
        ("fig_paired_displacement_per_book_umap", "all", "UMAP", _pa, _c_all,
         "fig:paired-displacement-per-book-umap"),
        ("fig_paired_displacement_per_book_pca3", "all", "PCA-3 render", _pa, _c_all,
         "fig:paired-displacement-per-book-pca3"),
        ("fig_paired_displacement_per_book_governs", "governs", "PCA", _pg, _c_gov,
         "fig:paired-displacement-per-book-governs"),
        ("fig_paired_displacement_per_book_umap_governs", "governs", "UMAP", _pg, _c_gov,
         "fig:paired-displacement-per-book-umap-governs"),
        ("fig_paired_displacement_per_book_pca3_governs", "governs", "PCA-3 render",
         _pg, _c_gov, "fig:paired-displacement-per-book-pca3-governs"),
    ):
        _pool_note = (
            "the full extracted norm pool"
            if _pool == "all" else
            f"the {_cc['norms_in_pool']:,} information-flow-governing norms only "
            f"({_cc['norms_in_pool'] / len(norms):.0%} of the extracted total) — "
            f"the pool R-DIRECT retrieves against in production"
        )
        _dir_note = (
            "Segment direction is a faithful projection of the 4096-D "
            "displacement, and each panel carries that novel's own mean "
            "displacement as an arrow."
            if _proj == "PCA" else
            "Three principal components viewed through a fixed orthographic "
            "camera (elevation -60 deg, azimuth 100 deg), chosen by grid search "
            "to minimise the median rank of each flow's true retrieved norm in "
            "the rendered image; the projection remains linear, so segment "
            "direction and the mean-displacement arrow keep their meaning."
            if _proj == "PCA-3 render" else
            "UMAP is non-linear, so segment direction carries no claim and no "
            "mean-displacement arrow is drawn; each panel instead reports its "
            "mean retrieval cosine, a property of the 4096-D space."
        )
        # Retrieval faithfulness of this basis: how often the norm a flow is
        # drawn joined to is also the nearest norm on the page. Every 2D display
        # loses most of a 4096-D rank relation, so the segments assert pairing
        # identity, not spatial adjacency, and the caption has to say so.
        #
        # Measured here rather than pasted in. One hardcoded set of numbers used
        # to serve both pools, which put the `governs` sweep on the `all`
        # captions — claiming 8.0% / 8th percentile where the truth is 7.2% /
        # 3rd, overstating the hit rate and understating the rank by 5 points.
        _pkey = {"PCA": "pca", "PCA-3 render": "pca3", "UMAP": "umap"}[_proj]
        _pj = PROJ[(FLOW_TEXT, INSTR, _pool)]
        _fid_hit, _fid_pr = layout_fidelity(
            _pj[_pkey], _pj["n_norm"], PAIRS[(FLOW_TEXT, INSTR, _pool)], _pool)
        _fidelity = (f"{_fid_hit:.1%} of flows (median rank of the true norm: "
                     f"{_fid_pr:.0%} of the way through that novel's pool"
                     f"{', i.e. near chance' if _fid_pr > 0.3 else ''})")
        save_caption(
            _nm,
            f"Paired displacement per source text ({_proj}, {_pool} norms)",
            f"Every one of the {_cc['n_pairs']:,} flow → nearest-same-novel-norm "
            f"pairs, drawn as small multiples on a shared corpus-level {_proj} "
            f"basis over {_pool_note}. Nothing is subsampled, so a panel's "
            f"density is that novel's actual pair count (211 for Alice to 3,479 "
            f"for Monte Cristo). Purple norms are retrieved by at least one "
            f"flow, blue norms by none: corpus-wide "
            f"{_cc['frac_norms_reached']:.0%} of this pool is reached, ranging "
            f"{_stats['frac_reached'].min():.0%}–{_stats['frac_reached'].max():.0%} "
            f"across novels. Norm marker area is proportional to the square root "
            f"of in-degree, so the hubs are visible: retrieval is heavily "
            f"top-heavy, and the median novel has ~12 of its ~313 governing "
            f"norms absorbing half its flows. Retrieval is computed in the full "
            f"4096-D space and the layout is a lossy display of it, so a segment "
            f"asserts which norm a flow retrieved, not that the two are close on "
            f"the page: in this basis the drawn norm is also the nearest norm on "
            f"the page for {_fidelity}. {_dir_note} Per-novel concentration ranges "
            f"R = {_stats['R'].min():.3f}–{_stats['R'].max():.3f}, but the gap to "
            f"each novel's own shuffled-within-novel null is only "
            f"{_stats['R_minus_null'].min():+.3f} to "
            f"{_stats['R_minus_null'].max():+.3f} "
            f"({int((_stats['R_minus_null'] <= 0).sum())} of {len(_stats)} novels "
            f"do not beat their null), so the offset is a property of the "
            f"norm/flow contrast rather than of which particular norm a flow is "
            f"matched to.",
            _lbl,
            ["embedding-space", "displacement", "per-book", _proj.lower(),
             f"norm-pool-{_pool}", "camera-ready"],
        )

    save_caption(
        "fig_displacement_concentration",
        "Directional concentration of the norm-minus-flow displacement",
        f"Difference vectors d_i = e_norm - e_flow for each flow and its nearest "
        f"same-novel norm, under one shared embedding instruction and with the "
        f"appropriateness verdict stripped from the flow string. (a) "
        f"cos(d_i, dbar) for the true pairing (mean resultant length "
        f"R = {_c_all['R']:.3f}) against a pairing shuffled within the same "
        f"novel (R = {_c_all['R_null_mean']:.3f}). The field is strongly "
        f"coherent — an isotropic field would give R near 0 at this sample size "
        f"— but the shuffled null reproduces almost all of it and the two mean "
        f"directions agree to cos = {_c_all['dbar_true_vs_null_align']:.3f}: the "
        f"coherence is a construct-level offset between norm space and flow "
        f"space, not a consequence of which norm a flow is matched to. (b) the "
        f"same directions as a polar histogram in the plane spanned by dbar and "
        f"the leading residual direction. (c) the ten novels' mean displacements "
        f"are near-collinear (pairwise cosine {_off.min():.2f}–{_off.max():.2f}), "
        f"so a single flow→norm direction serves the whole corpus. Restricting "
        f"to the information-flow-governing norm pool gives "
        f"R = {_c_gov['R']:.3f} against a null of "
        f"{_c_gov['R_null_mean']:.3f}. Under the production "
        f"asymmetric-instruction embeddings the concentration reads "
        f"R = {_c_prod['R']:.3f}, inflated by the differing instruction "
        f"prefixes. Retaining the appropriateness verdict in the flow string "
        f"raises the norm/flow silhouette from {_s['silhouette_4096d']:.3f} to "
        f"{_s_full['silhouette_4096d']:.3f}; dropping the transmission principle "
        f"as well lowers it to {_s_desc['silhouette_4096d']:.3f}. Governing-norm "
        f"pool sizes per novel are in tables/norm_pool_sizes.csv "
        f"(median {norm_pool_sizes['frac_governing'].median():.0%} of each "
        f"novel's norms).",
        "fig:displacement-concentration",
        ["embedding-space", "displacement", "concentration", "null", "camera-ready"],
    )
    return


if __name__ == "__main__":
    app.run()
