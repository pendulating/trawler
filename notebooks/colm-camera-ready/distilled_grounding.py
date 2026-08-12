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
    # Was normative grounding distilled into the weights?

    Built 2026-08-06 for the COLM 2026 camera-ready.
    Plan: `wiki/2026-08-05_distilled_grounding_plan.md`.

    ## The question

    §5.2 measures normative grounding on the **teacher's** flows: grounding
    reclassifies 30.9% of them, 21.1% appropriate→inappropriate against 9.8%
    the other way. That describes Gemma-4-31B-it. It says nothing about whether
    our fine-tuning moved a *policy's* ungrounded judgment toward the grounded
    one.

    So we re-ran the same two-stage `historical_norms` flow pipeline — **same
    fiction prompts, same 2,993 chunks** — with each fine-tuned checkpoint
    substituted for the teacher, and recomputed the rate per arm. Only the
    weights change, which is what makes each arm's number comparable to 30.9%.

    **Hypothesis:** the fine-tuned arms reclassify *less*, because training
    pushed their ungrounded judgment toward the grounded one.

    ## Why the numbers here are the same instrument as the paper's

    The metric code below reproduces every published §5.2 figure from the
    teacher's own table: D 30.9%, A→I 21.1%, I→A 9.8%, asymmetry 2.15, class
    priors 14.9% → 26.1%, and the *Nineteen Eighty-Four* exception at 15.5%
    each. That check runs as an assertion in the metrics cell — if it ever
    fails, nothing below is comparable to the paper and the notebook says so
    rather than quietly reporting a different quantity.

    ## Reading the table: three traps

    1. **D alone is not evidence.** Each arm extracts its *own* flows, so D can
       fall through extraction shift, label-prior shift, or retrieval shift with
       no alignment gain. `kappa` — Cohen's κ against a chance model fixed
       within each novel — is the quantity immune to a policy simply moving its
       base rate toward 26% inappropriate. **Read κ and its CI, not D.**
    2. **The RL arms were trained toward this target.** R-DIRECT's gold *is*
       `flow_appropriateness` over this universe, and KTO's K1 labels derive
       from the same gold. On chunks they trained on, a low D is close to
       tautological — which is why `double-heldout` is the primary cell and
       `fiction10-all` is labelled training fit.
    3. **SFT saw all of fiction-10.** Arm S is training fit everywhere here. It
       is kept as the §7.2 sanity check: SFT trained on the teacher's
       *ungrounded* labels, so D(SFT) should track D(teacher). If it does not,
       the run is broken.

    Tables → `tables/distilled_grounding/`. Figures → `figures/distilled_grounding/`.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
    NB_DIR = PROJECT_ROOT / "notebooks/colm-camera-ready"
    TAB_DIR = NB_DIR / "tables/distilled_grounding"
    FIG_DIR = NB_DIR / "figures/distilled_grounding"
    for _d in (TAB_DIR, FIG_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    # Per-arm grounding tables, written by
    #   scripts/build_grounding_disagreement.py --arm <name> --flows <parquet>
    GROUND_DIR = PROJECT_ROOT / "outputs/2026-08-06_distilled_grounding"

    # Training records — the contamination bookkeeping is DERIVED here, never
    # hardcoded, so it tracks the artifacts if they change.
    M2_TRACES = PROJECT_ROOT / (
        "multirun/2026-07-28_grpo_m2_full/21-31-11/cell=full/"
        "grpo_only_online_external/outputs/grpo/checkpoint/reward_traces.jsonl"
    )
    K1_META = PROJECT_ROOT / "outputs/2026-07-31_k1_full/kto_metadata.json"

    #: Published §5.2 values, used as a self-check on the metric code.
    PAPER = {"D": 0.309, "a2i": 0.211, "i2a": 0.098,
             "own_inappr": 0.149, "grounded_inappr": 0.261}

    ARM_ORDER = ["teacher", "base", "sft", "m2-full", "k3-verdict"]
    ARM_LABEL = {
        "teacher": "Gemma-4-31B (teacher)",
        "base": "Qwen3.5-9B",
        "sft": "+ SFT",
        "m2-full": "+ GRPO (m2 full)",
        "k3-verdict": "+ KTO (k3 verdict)",
    }

    def save_table(df, name, index=True):
        out = TAB_DIR / f"{name}.csv"
        df.to_csv(out, index=index)
        print(f"[table] {out}")
        return df

    return (
        ARM_LABEL,
        ARM_ORDER,
        FIG_DIR,
        GROUND_DIR,
        K1_META,
        M2_TRACES,
        PAPER,
        json,
        np,
        pd,
        save_table,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chunk sets

    Derived from the two training records, not remembered. Keys are
    `(gutenberg_id, chunk_id)` — `chunk_id` is **per-book**, so keying on it
    alone silently merges chunks across novels.
    """)
    return


@app.cell
def _(K1_META, M2_TRACES, json, pd):
    def _grpo_chunks():
        keys = set()
        with M2_TRACES.open() as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("task_type") == "extract":
                    keys.add(f"{rec.get('source_id')}|{rec.get('chunk_id')}")
        return keys

    GRPO_SEEN = _grpo_chunks()
    KTO_HELDOUT = set(json.loads(K1_META.read_text())["heldout_keys"])
    GRPO_BOOKS = {k.split("|", 1)[0] for k in GRPO_SEEN}
    DOUBLE_HELDOUT = KTO_HELDOUT - GRPO_SEEN

    # These are the plan §3.1 numbers. Assert rather than trust.
    assert len(GRPO_SEEN) == 492, len(GRPO_SEEN)
    assert len(KTO_HELDOUT) == 599, len(KTO_HELDOUT)
    assert len(DOUBLE_HELDOUT) == 503, len(DOUBLE_HELDOUT)
    assert GRPO_BOOKS == {"1023", "11", "1342", "135"}, GRPO_BOOKS

    def chunk_sets(df: pd.DataFrame) -> dict[str, pd.Series]:
        """Boolean masks over a per-flow table, keyed by (book, chunk)."""
        key = (df["gutenberg_id"].astype(str) + "|"
               + df["chunk_id"].astype(str))
        book = df["gutenberg_id"].astype(str)
        return {
            # Same chunks + same prompts as the teacher, so directly comparable
            # to 30.9% — but every arm has seen some or all of it.
            "fiction10-all": pd.Series(True, index=df.index),
            # PRIMARY for the RL arms: unseen by GRPO and by KTO, all 10 books.
            "double-heldout": key.isin(DOUBLE_HELDOUT),
            # Book-level rather than chunk-level holdout — the stronger form,
            # but only for GRPO (KTO trained on ~80% of these).
            "grpo-unseen-books": ~book.isin(GRPO_BOOKS),
            "kto-heldout": key.isin(KTO_HELDOUT),
        }

    print(f"GRPO saw {len(GRPO_SEEN)} chunks across books {sorted(GRPO_BOOKS)}")
    print(f"KTO held out {len(KTO_HELDOUT)}; doubly held out {len(DOUBLE_HELDOUT)}")
    return (chunk_sets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metrics

    `teacher` / `teacher_ci` are the column names
    `build_grounding_disagreement.py` writes for **whichever extractor produced
    the flows** — for a policy arm they hold that policy's own ungrounded
    label, not Gemma's. The names are shared so one table shape serves every
    arm.
    """)
    return


@app.cell
def _(np, pd):
    def kappa_wb(own: np.ndarray, gr: np.ndarray, book: np.ndarray) -> float:
        """Cohen's κ with the chance model fixed WITHIN book.

        `own`/`gr` are boolean (True = inappropriate); `book` is a 0-based
        integer code. Chance disagreement is the expected D when the arm's own
        labels are shuffled inside each novel — which has a closed form, since
        a uniform permutation inside book b leaves every flow in b with
        P(label = inappropriate) = p_b:

            E[D_null] = 1 − Σ_b (n_b/N) · [p_b·q_b + (1−p_b)·(1−q_b)]

        so no sampling and no seed. Within-book rather than global because the
        grounded label's class prior varies by novel, and a global chance model
        would credit an arm for "its books happen to be the permissive ones".

        κ = 1 − D/E[D_null]. This is the notebook's headline: alignment beyond
        what the label prior alone buys, on a scale that is comparable across
        arms (their null rates range 0.28–0.33, so the raw difference is not).
        """
        n_b = np.bincount(book)
        p = np.bincount(book, weights=own) / n_b
        q = np.bincount(book, weights=gr) / n_b
        d_null = float(1.0 - (n_b / len(book) * (p * q + (1 - p) * (1 - q))).sum())
        if d_null <= 0:
            return np.nan
        return (d_null - float((own != gr).mean())) / d_null

    def kappa_ci(own, gr, book, n: int = 2000, seed: int = 0):
        """95% CI on κ resampling whole novels.

        Flows inside one book share a norm universe and an extraction pass, so
        they are not independent draws; a flow-level interval would be far too
        narrow. Ten books is a small resampling frame — read these as coarse.
        """
        rng = np.random.default_rng(seed)
        n_books = int(book.max()) + 1
        idx = [np.flatnonzero(book == b) for b in range(n_books)]
        vals = np.empty(n)
        for i in range(n):
            pick = rng.integers(0, n_books, n_books)
            sel = np.concatenate([idx[b] for b in pick])
            rebook = np.repeat(np.arange(n_books), [len(idx[b]) for b in pick])
            vals[i] = kappa_wb(own[sel], gr[sel], rebook)
        return (float(np.nanpercentile(vals, 2.5)),
                float(np.nanpercentile(vals, 97.5)))

    VALID = {"appropriate", "inappropriate"}

    def metrics(g: pd.DataFrame, n_boot: int = 2000) -> pd.Series:
        g = g[g["scored"]]
        if len(g) == 0:
            return pd.Series(dtype=float)
        # A few flows carry NO appropriateness label from the extractor
        # (0.12-0.13% on the two RL arms, 0 elsewhere). They cannot enter an
        # agreement statistic, but dropping them silently would shrink the
        # denominator invisibly — so they are excluded here AND reported as
        # `missing_own_rate` so the drop is always on the page.
        has_own = g["teacher_ci"].isin(VALID)
        missing_own = float((~has_own).mean())
        g = g[has_own]
        if len(g) == 0:
            return pd.Series(dtype=float)
        own, gr = g["teacher_ci"], g["grounded"]
        a2i = float(((own == "appropriate") & (gr == "inappropriate")).mean())
        i2a = float(((own == "inappropriate") & (gr == "appropriate")).mean())
        own_i = (own == "inappropriate").to_numpy()
        gr_i = (gr == "inappropriate").to_numpy()
        book = pd.factorize(g["gutenberg_id"].astype(str))[0]
        lo, hi = kappa_ci(own_i, gr_i, book, n=n_boot) if n_boot else (
            np.nan, np.nan)
        return pd.Series({
            "n_flows": len(g),
            "n_chunks": g.groupby(["gutenberg_id", "chunk_id"]).ngroups,
            "D": float((own != gr).mean()),
            "A->I": a2i,
            "I->A": i2a,
            # nan not inf when i2a == 0, so it aggregates without poisoning.
            "asymmetry": (a2i / i2a) if i2a > 0 else np.nan,
            # THE headline: alignment beyond what the label prior alone buys.
            "kappa": kappa_wb(own_i, gr_i, book),
            "kappa_lo": lo,
            "kappa_hi": hi,
            "own_inappr": float((own == "inappropriate").mean()),
            "grounded_inappr": float((gr == "inappropriate").mean()),
            "ambiguous_rate": float(
                (g["teacher"].astype(str) == "ambiguous").mean()),
            "missing_own_rate": missing_own,
            "top_sim_median": float(g["top_sim"].median()),
        })

    return (metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Self-check: reproduce the published §5.2 numbers
    """)
    return


@app.cell
def _(GROUND_DIR, PAPER, metrics, pd):
    _t = GROUND_DIR / "teacher" / "flow_grounding_labels.parquet"
    if _t.exists():
        _m = metrics(pd.read_parquet(_t), n_boot=0)
        _checks = {
            "D": (_m["D"], PAPER["D"]),
            "A->I": (_m["A->I"], PAPER["a2i"]),
            "I->A": (_m["I->A"], PAPER["i2a"]),
            "own_inappr": (_m["own_inappr"], PAPER["own_inappr"]),
            "grounded_inappr": (_m["grounded_inappr"], PAPER["grounded_inappr"]),
        }
        _bad = {k: v for k, v in _checks.items() if abs(v[0] - v[1]) > 0.002}
        for _k, (_got, _want) in _checks.items():
            print(f"  {_k:18s} got {_got:.4f}  paper {_want:.4f}  "
                  f"{'OK' if _k not in _bad else 'MISMATCH'}")
        assert not _bad, (
            f"metric code does not reproduce the paper: {_bad}. Nothing below "
            f"is comparable to the published 30.9%."
        )
        print("\nSELF-CHECK PASSED — same instrument as §5.2.")
    else:
        print(f"teacher table not built yet: {_t}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-arm results
    """)
    return


@app.cell
def _(ARM_ORDER, GROUND_DIR, chunk_sets, metrics, pd, save_table):
    _rows = []
    _loaded = []
    for _arm in ARM_ORDER:
        _p = GROUND_DIR / _arm / "flow_grounding_labels.parquet"
        if not _p.exists():
            print(f"[skip] {_arm}: not built ({_p})")
            continue
        _df = pd.read_parquet(_p)
        _loaded.append(_arm)
        for _cs, _mask in chunk_sets(_df).items():
            _m = metrics(_df[_mask])
            if _m.empty:
                continue
            _m["arm"] = _arm
            _m["chunk_set"] = _cs
            _rows.append(_m)

    RESULTS = pd.DataFrame(_rows)
    if not RESULTS.empty:
        RESULTS = RESULTS.set_index(["chunk_set", "arm"]).sort_index()
        save_table(RESULTS, "per_arm_by_chunk_set")
        print(f"\narms loaded: {_loaded}")
    RESULTS
    return (RESULTS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Primary cell — `double-heldout` (503 chunks, unseen by GRPO and KTO)

    Read `kappa` and its book-level CI first. `D` is reported because it is
    what the hypothesis was stated in, but it moves with the label prior and
    the flow population; κ does not. A CI straddling zero means the arm's
    agreement with grounding is what its label prior alone would produce.

    SFT is contaminated even here and is shown only as the sanity anchor.
    """)
    return


@app.cell
def _(RESULTS, save_table):
    if not RESULTS.empty and "double-heldout" in RESULTS.index.get_level_values(0):
        PRIMARY = RESULTS.loc["double-heldout"][
            ["n_flows", "n_chunks", "D", "A->I", "I->A", "asymmetry",
             "kappa", "kappa_lo", "kappa_hi", "own_inappr", "grounded_inappr"]
        ]
        save_table(PRIMARY, "primary_double_heldout")
    else:
        PRIMARY = None
        print("primary cell not available yet")
    PRIMARY
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Extraction descriptives

    Guards against reading D wrong. A drop in D that comes with a collapse in
    flows-per-chunk, or with a jump in retrieval similarity, is an extraction
    shift rather than an alignment gain. The teacher's corpus rate is 5.4
    flows/chunk.
    """)
    return


@app.cell
def _(RESULTS, save_table):
    if not RESULTS.empty:
        DESC = RESULTS.assign(
            flows_per_chunk=RESULTS["n_flows"] / RESULTS["n_chunks"]
        )[["n_chunks", "n_flows", "flows_per_chunk", "ambiguous_rate",
           "top_sim_median"]]
        save_table(DESC, "extraction_descriptives")
    else:
        DESC = None
    DESC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure
    """)
    return


@app.cell
def _(ARM_LABEL, ARM_ORDER, FIG_DIR, RESULTS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "text.usetex": False,
    })

    def save_fig(fig, name, pad_inches=0.0):
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300,
                        bbox_inches="tight", pad_inches=pad_inches)
        print(f"[fig] {FIG_DIR / name}.png|.pdf")

    FIG = None
    if RESULTS is not None and not RESULTS.empty:
        _sets = [s for s in ("double-heldout", "grpo-unseen-books",
                             "fiction10-all")
                 if s in RESULTS.index.get_level_values(0)]
        fig, axes = plt.subplots(1, len(_sets), figsize=(3.1 * len(_sets), 2.7),
                                 sharey=True)
        axes = [axes] if len(_sets) == 1 else list(axes)
        for ax, cs in zip(axes, _sets):
            sub = RESULTS.loc[cs]
            arms = [a for a in ARM_ORDER if a in sub.index]
            x = range(len(arms))
            a2i = [sub.loc[a, "A->I"] * 100 for a in arms]
            i2a = [sub.loc[a, "I->A"] * 100 for a in arms]
            # Stacked so the bar height IS D and the split shows the direction
            # — the asymmetry can collapse while D holds flat.
            ax.bar(x, a2i, color="#B4443B", label="appropriate → inappropriate")
            ax.bar(x, i2a, bottom=a2i, color="#4C72B0",
                   label="inappropriate → appropriate")
            ax.set_xticks(list(x))
            ax.set_xticklabels([ARM_LABEL.get(a, a) for a in arms],
                               rotation=40, ha="right")
            ax.set_title(f"{cs}  (n={int(sub.iloc[0]['n_chunks'])} chunks)")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("flows reclassified by grounding (%)")
        axes[-1].legend(frameon=False, loc="upper right")
        fig.tight_layout()
        save_fig(fig, "fig_distilled_grounding")
        FIG = fig
    FIG
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this can and cannot license

    It measures whether a fine-tuned policy's *own* appropriateness judgment
    has moved toward the norm-grounded one, on flows that policy extracted
    itself, under the teacher's prompts. It does **not** establish that the
    grounded label is more correct — that needs external gold, which is what
    the benchmark table is for.

    Specific limits, carried from the plan:

    - **top100 was not run.** Nothing here supports a cross-corpus
      generalization claim. fiction-10 is the only corpus measured, and SFT
      trained on all of it.
    - **The RL arms were trained toward this target.** A low D in-domain is
      near-tautological; only `double-heldout` and `grpo-unseen-books` speak to
      transfer, and even those share the corpus and the universe.
    - **The policies are off their prompt distribution.** They were trained on
      the terse one-stage extraction instruction, not the two-stage fiction
      prompts used here. That is the price of holding the instrument fixed at
      the teacher's. Arm `base` is the control that establishes what the untuned
      backbone does under the same conditions; read the tuned arms against it,
      not against the teacher alone.
    - **Pre-registered null:** if D drops on `fiction10-all` but not on
      `double-heldout`, the finding is *training fit, not distillation*, and
      that is what gets reported.
    """)
    return


if __name__ == "__main__":
    app.run()
