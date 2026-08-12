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
    # Are norms and flows *semantically* distinct?

    Companion to `norm_flow_embedding_space.py`, which established the geometry.
    That notebook answered "are they separable" — decisively yes. This one asks
    the question that actually matters and that separability does **not**
    answer: *is the separation semantic, or is it our serialization template?*

    ### What is already established (and is not re-litigated here)

    | finding | value |
    |---|---|
    | held-out linear separability of norm vs flow | 99.8% (Cohen's $d$ = 6.55, marginal overlap 0.003) |
    | share of total variance on the contrast axis | 5.4% (`governs`) / 10.4% (`all`) |
    | separability across flow serializations | $d$ = 6.5 (`noappr`) / 7.2 (`full`) / 6.9 (`descriptive`) |
    | source texts form separable clusters? | no — book silhouette −0.04 |
    | kNN book purity (norms / flows) | 33.6% vs 14.0% chance / 71.2% vs 14.9% chance |
    | flow → norm displacement | construct-level, not pairing-level ($R \approx R_{\text{null}}$) |
    | governing norms ever retrieved | 59%; retrieval Gini 0.81 |

    ### Why separability is not the claim

    The two constructs are serialized in visibly different shapes:

    ```
    norm  A lady … is expected to exercise restraint … | a lady … | [context: social propriety] | [force: obligatory]
    flow  In a biographical/familial history context, Mr. Austen Leigh sends marital fate …, to the public, via public record.
    ```

    Pipe delimiters and bracketed fields versus one prose sentence; 379 versus
    151 characters on average. **Any** embedding model separates those on
    surface form. And the near-invariance of $d$ across the three flow
    serializations is weak evidence *for* the confound rather than against it:
    changing what the flow string says barely moves the number, which is what a
    template detector would do.

    So the honest status of the existing result is *separable*, not
    *semantically distinct*. This notebook tries to break that claim four ways.

    ### The four tests

    1. **Lexical / format baselines.** If TF-IDF, character n-grams, or
       function words alone separate the two classes as well as the 4096-D
       embedding does, the embedding contributes nothing semantic.
    2. **Format-controlled re-serialization.** Strip the scaffolding and match
       the surface shape, then re-measure. Two deterministic controls run here;
       the LLM paraphrase control is staged for the embedding job.
    3. **What is the discriminant axis about?** Regress position on the axis
       against interpretable covariates — deontic modals, entity density,
       length, tense — and read the extremes.
    4. **Is normativity latent in the flow tuple?** The appropriateness verdict
       is stripped from the embedded string, so it is a clean held-out label.
       If a probe recovers it, the CI tuple encodes normative content
       implicitly. (This measures one teacher's self-consistency, not ground
       truth — see `project_gold_validity_defect`.)

    Retrieval *validity* — whether the nearest norm actually governs the flow —
    is deliberately out of scope. It needs annotation and belongs in its own
    project.

    Figures → `figures/norm_flow_semantics/`, tables → `tables/norm_flow_semantics/`.
    """)
    return


@app.cell
def _():
    import json
    import re
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
    NB_DIR = PROJECT_ROOT / "notebooks/colm-camera-ready"
    EMB_DIR = PROJECT_ROOT / "outputs/camera_ready/embeddings"
    FIG_DIR = NB_DIR / "figures/norm_flow_semantics"
    TAB_DIR = NB_DIR / "tables/norm_flow_semantics"
    STAGE_DIR = PROJECT_ROOT / "outputs/camera_ready/serializations"
    for _d in (FIG_DIR, TAB_DIR, STAGE_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    CORPUS = "fiction10"
    INSTR = "shared"      # one embedding instruction for both constructs
    FLOW_TEXT = "noappr"  # production parity: no appropriateness verdict
    SEED = 42

    NORM_COLOR = "#4C72B0"
    FLOW_COLOR = "#DD8452"
    ACCENT = "#7B4FA8"
    NULL_COLOR = "#9A9A9A"

    pd.set_option("display.max_columns", 80)
    pd.set_option("display.width", 200)

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

    def save_fig(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        print(f"[fig] {FIG_DIR / name}.png|.pdf")

    def save_table(df, name, index=False):
        out = TAB_DIR / f"{name}.csv"
        df.to_csv(out, index=index)
        print(f"[table] {out}")

    def save_caption(name, title, caption, label, tags):
        (FIG_DIR / f"{name}.json").write_text(
            json.dumps(
                {"plot-title": title, "plot-caption": caption,
                 "plot-latex-label": label, "plot-tags": tags},
                indent=2, ensure_ascii=False,
            ) + "\n"
        )
        print(f"[caption] {FIG_DIR / name}.json")

    return (
        ACCENT,
        CORPUS,
        EMB_DIR,
        FLOW_COLOR,
        FLOW_TEXT,
        INSTR,
        NORM_COLOR,
        NULL_COLOR,
        SEED,
        STAGE_DIR,
        np,
        pd,
        plt,
        re,
        save_caption,
        save_fig,
        save_table,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load""")
    return


@app.cell
def _(CORPUS, EMB_DIR, FLOW_TEXT, INSTR, np, pd):
    norms = pd.read_parquet(EMB_DIR / f"{CORPUS}_norms_meta.parquet")
    flows = pd.read_parquet(EMB_DIR / f"{CORPUS}_flows_meta.parquet")

    NORM_EMB = np.load(EMB_DIR / f"{CORPUS}_norms_{INSTR}.npy")
    FLOW_EMB = np.load(EMB_DIR / f"{CORPUS}_flows_{FLOW_TEXT}_{INSTR}.npy")
    assert NORM_EMB.shape[0] == len(norms) and FLOW_EMB.shape[0] == len(flows)

    # Label: 0 = norm, 1 = flow. Every test below predicts this.
    X_EMB = np.vstack([NORM_EMB, FLOW_EMB])
    Y = np.r_[np.zeros(len(norms), int), np.ones(len(flows), int)]
    BOOK = np.r_[norms["book_title"].astype(str).values,
                 flows["book_title"].astype(str).values]

    print(f"norms {NORM_EMB.shape}  flows {FLOW_EMB.shape}  pooled {X_EMB.shape}")
    print(f"class balance: {(Y == 0).sum():,} norms / {(Y == 1).sum():,} flows "
          f"— majority baseline {max((Y == 0).mean(), (Y == 1).mean()):.1%}")
    return BOOK, FLOW_EMB, NORM_EMB, X_EMB, Y, flows, norms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §2 (built first) — the serialization variants

    Three surface forms for the same underlying records, so the ablation in §1
    can be run over all of them rather than only the production strings.

    | variant | norm string | flow string | what it removes |
    |---|---|---|---|
    | `asis` | production `embed_text` | production `embed_text_noappr` | nothing — the baseline |
    | `prose` | `raz_norm_articulation` alone | unchanged (already one sentence) | the pipes, the duplicated abstraction, the `[context:]` / `[force:]` brackets |
    | `fields` | bare field values, comma-joined | bare field values, comma-joined | *both* sentence frames — no scaffolding on either side |

    `prose` asks whether the delimiters carry the separation. `fields` is the
    stronger control: both constructs become an unpunctuated list of their own
    field values, so neither retains a grammatical frame the other lacks. What
    survives `fields` cannot be sentence-shape; it can still be vocabulary,
    which is what §1's function-word and char-n-gram baselines isolate.

    Neither control can equalise *content* vocabulary — a norm says "expected
    to" and a flow names a sender. That is the residue the LLM paraphrase
    control is for, and it is staged to disk at the end of this section.
    """)
    return


@app.cell
def _(FLOW_TEXT, flows, norms, pd):
    def _join(df, cols):
        """Comma-join the non-empty field values, with no frame or labels."""
        parts = []
        for c in cols:
            s = df[c].fillna("").astype(str).str.strip() if c in df.columns else ""
            parts.append(s)
        out = parts[0]
        for s in parts[1:]:
            out = out.str.cat(s, sep=", ", na_rep="")
        return out.str.replace(r"(, )+", ", ", regex=True).str.strip(", ")

    NORM_FIELDS = ["raz_norm_subject", "raz_norm_act", "raz_context", "raz_normative_force"]
    FLOW_FIELDS = ["ci_sender", "ci_information_type", "ci_subject", "ci_recipient",
                   "ci_context", "ci_transmission_principle"]

    TEXTS = {
        "asis": (norms["embed_text"].astype(str),
                 flows[f"embed_text_{FLOW_TEXT}"].astype(str)),
        "prose": (norms["raz_norm_articulation"].fillna("").astype(str),
                  flows[f"embed_text_{FLOW_TEXT}"].astype(str)),
        "fields": (_join(norms, NORM_FIELDS), _join(flows, FLOW_FIELDS)),
    }

    _rows = []
    for _v, (_n, _f) in TEXTS.items():
        _rows.append({
            "variant": _v,
            "norm_chars_mean": float(_n.str.len().mean()),
            "flow_chars_mean": float(_f.str.len().mean()),
            "norm_words_mean": float(_n.str.split().str.len().mean()),
            "flow_words_mean": float(_f.str.split().str.len().mean()),
            "norm_has_delim": float(_n.str.contains(r"[|\[\]]", regex=True).mean()),
            "flow_has_delim": float(_f.str.contains(r"[|\[\]]", regex=True).mean()),
        })
    serialization_shapes = pd.DataFrame(_rows)
    print(serialization_shapes.to_string(index=False))
    print()
    for _v, (_n, _f) in TEXTS.items():
        print(f"[{_v}] norm: {_n.iloc[0][:150]}")
        print(f"[{_v}] flow: {_f.iloc[0][:150]}\n")
    return TEXTS, serialization_shapes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §1 — lexical and format baselines

    Every model below predicts the same binary label (norm vs flow) with the
    same 5-fold **grouped** cross-validation, grouped by source text so no novel
    appears in both train and test. The question is not whether each separates —
    they will — but **how much of the embedding's 99.8% is already available to
    a bag of characters.**

    | feature set | what it can see |
    |---|---|
    | `length` | one feature: character count. The crudest possible format cue. |
    | `funcwords` | stopwords and punctuation only, content words deleted. Pure grammatical shape. |
    | `char3_5` | character 3–5-grams. Catches delimiters, casing, morphology. |
    | `tfidf_word` | word 1–2-grams. Vocabulary, no order beyond bigrams. |
    | `embedding` | the 4096-D Qwen3 vectors. |

    If `funcwords` or `char3_5` matches `embedding`, the separation is formal.
    If `tfidf_word` matches it but `funcwords` does not, it is lexical — real,
    but not evidence of the deep semantic contrast the paper wants to claim.
    """)
    return


@app.cell
def _(BOOK, TEXTS, X_EMB, Y, np, pd, save_table):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _STOP = set(ENGLISH_STOP_WORDS)

    def _funcwords_only(s):
        """Keep stopwords and punctuation; delete every content word.

        What remains is the grammatical skeleton — 'in a context , sends , about
        , to , via .' for a flow. If this separates the classes, the model is
        reading our template, not the text.
        """
        out = []
        for tok in s.split():
            bare = "".join(ch for ch in tok if ch.isalnum()).lower()
            punct = "".join(ch for ch in tok if not ch.isalnum())
            if bare in _STOP or bare == "":
                out.append(bare + punct if bare else punct)
            elif punct:
                out.append(punct)
        return " ".join(x for x in out if x)

    def cv_score(make_X, y, groups, name, variant, n_splits=5):
        """Grouped-by-novel CV. Returns accuracy and AUC, mean over folds."""
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
        accs, aucs = [], []
        for tr, te in cv.split(np.zeros(len(y)), y, groups):
            Xtr, Xte = make_X(tr, te)
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr, y[tr])
            accs.append(clf.score(Xte, y[te]))
            aucs.append(roc_auc_score(y[te], clf.decision_function(Xte)))
        return {"variant": variant, "features": name,
                "accuracy": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
                "auc": float(np.mean(aucs))}

    def _text_of(variant):
        n, f = TEXTS[variant]
        return np.r_[n.values, f.values]

    rows = []
    for _variant in TEXTS:
        _txt = _text_of(_variant)
        _func = np.array([_funcwords_only(s) for s in _txt])

        # length: a single standardised feature
        rows.append(cv_score(
            lambda tr, te, t=_txt: (
                StandardScaler().fit(np.array([len(s) for s in t[tr]]).reshape(-1, 1))
                .transform(np.array([len(s) for s in t[tr]]).reshape(-1, 1)),
                StandardScaler().fit(np.array([len(s) for s in t[tr]]).reshape(-1, 1))
                .transform(np.array([len(s) for s in t[te]]).reshape(-1, 1)),
            ), Y, BOOK, "length", _variant))

        for _name, _src, _kw in (
            ("funcwords", _func, dict(analyzer="word", ngram_range=(1, 3),
                                      token_pattern=r"\S+", min_df=3)),
            ("char3_5", _txt, dict(analyzer="char_wb", ngram_range=(3, 5),
                                   min_df=3, max_features=200_000)),
            ("tfidf_word", _txt, dict(analyzer="word", ngram_range=(1, 2),
                                      min_df=3, max_features=200_000)),
        ):
            def _mk(tr, te, src=_src, kw=_kw):
                v = TfidfVectorizer(**kw)
                return v.fit_transform(src[tr]), v.transform(src[te])
            rows.append(cv_score(_mk, Y, BOOK, _name, _variant))

    # The embedding, on the production strings only (the other variants have no
    # embeddings yet — see the staging cell).
    rows.append(cv_score(lambda tr, te: (X_EMB[tr], X_EMB[te]), Y, BOOK,
                         "embedding", "asis"))

    baselines = pd.DataFrame(rows)
    save_table(baselines, "lexical_baselines")
    print(baselines.pivot(index="features", columns="variant", values="accuracy")
          .reindex(["length", "funcwords", "char3_5", "tfidf_word", "embedding"])
          .to_string(float_format=lambda v: f"{v:.4f}"))
    baselines
    return baselines, cv_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Length-matched control

    Even after stripping scaffolding the two classes differ in length. Restrict
    to the overlapping band of the two length distributions and re-run: if
    accuracy holds up on rows a length-only model cannot tell apart, length is
    not doing the work.
    """)
    return


@app.cell
def _(BOOK, TEXTS, X_EMB, Y, cv_score, np, pd, save_table):
    from sklearn.feature_extraction.text import TfidfVectorizer as _TV

    _n_txt, _f_txt = TEXTS["asis"]
    _len = np.r_[_n_txt.str.len().values, _f_txt.str.len().values]
    _lo = max(np.percentile(_len[Y == 0], 5), np.percentile(_len[Y == 1], 5))
    _hi = min(np.percentile(_len[Y == 0], 95), np.percentile(_len[Y == 1], 95))
    _band = np.flatnonzero((_len >= _lo) & (_len <= _hi))

    print(f"length band [{_lo:.0f}, {_hi:.0f}] chars -> {len(_band):,} rows "
          f"({(Y[_band] == 0).sum():,} norms / {(Y[_band] == 1).sum():,} flows)")

    _rows = []
    if len(_band) > 200 and len(np.unique(Y[_band])) == 2:
        _txt_b = np.r_[_n_txt.values, _f_txt.values][_band]
        _y_b, _g_b = Y[_band], BOOK[_band]
        _rows.append(cv_score(
            lambda tr, te: (_TV(analyzer="char_wb", ngram_range=(3, 5), min_df=3)
                            .fit(_txt_b[tr]).transform(_txt_b[tr]),
                            _TV(analyzer="char_wb", ngram_range=(3, 5), min_df=3)
                            .fit(_txt_b[tr]).transform(_txt_b[te])),
            _y_b, _g_b, "char3_5", "asis_lenmatched"))
        _rows.append(cv_score(lambda tr, te: (X_EMB[_band][tr], X_EMB[_band][te]),
                              _y_b, _g_b, "embedding", "asis_lenmatched"))
        length_matched = pd.DataFrame(_rows)
        save_table(length_matched, "length_matched_baselines")
        print(length_matched.to_string(index=False))
    else:
        length_matched = pd.DataFrame(_rows)
        print("length bands do not overlap enough for a matched comparison")
    return (length_matched,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Staging the paraphrase control

    The deterministic controls cannot equalise content vocabulary. The strongest
    available control is to have the teacher rewrite **both** constructs into
    one neutral template of matched length and register, then re-embed and
    re-measure. That needs a generation pass plus the embedding server, so the
    inputs are written to disk here and the numbers land in this notebook once
    those jobs have run.
    """)
    return


@app.cell
def _(CORPUS, STAGE_DIR, TEXTS, Y, np, pd):
    _n_txt, _f_txt = TEXTS["fields"]
    stage = pd.DataFrame({
        "row_id": np.arange(len(Y)),
        "construct": np.where(Y == 0, "norm", "flow"),
        "fields_text": np.r_[_n_txt.values, _f_txt.values],
    })
    _out = STAGE_DIR / f"{CORPUS}_paraphrase_input.parquet"
    stage.to_parquet(_out, index=False)
    print(f"[stage] {_out}  ({len(stage):,} rows)")
    print("next: paraphrase both constructs into one neutral template, then embed "
          "with scripts/embed_camera_ready_norms_flows.py")
    return (stage,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §3 — what is the discriminant axis about?

    Fit the norm-minus-flow mean-difference direction on half the rows, project
    everything onto it, and ask what that coordinate correlates with. A high
    correlation with deontic modals would say the axis is tracking the
    prescriptive/descriptive contrast — which is the distinction CI theory
    actually wants. A high correlation with entity density or length would say
    it is tracking surface properties.

    Correlations are reported **within class** as well as pooled. The pooled
    number is inflated by anything that merely differs between the two classes
    on average, which is every covariate here; only the within-class column
    shows whether the axis orders records *inside* a construct.
    """)
    return


@app.cell
def _(FLOW_EMB, NORM_EMB, SEED, TEXTS, Y, np, pd, re, save_table):
    from scipy.stats import spearmanr

    _X = np.vstack([NORM_EMB, FLOW_EMB])
    _rng = np.random.default_rng(SEED)
    _perm = _rng.permutation(len(_X))
    _tr, _te = _perm[: len(_X) // 2], _perm[len(_X) // 2:]
    _u = _X[_tr][Y[_tr] == 0].mean(0) - _X[_tr][Y[_tr] == 1].mean(0)
    _u /= np.linalg.norm(_u)
    AXIS = _X @ _u   # + = norm-like, - = flow-like

    _n_txt, _f_txt = TEXTS["asis"]
    _txt = np.r_[_n_txt.values, _f_txt.values]

    DEONTIC = re.compile(
        r"\b(must|should|ought|expected|required|require|forbidden|forbid|may|"
        r"permitted|permit|obliged|obligation|shall|prohibited|prohibit|"
        r"allowed|allow|duty|improper|proper)\b", re.I)
    _cap = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}")

    COVARS = pd.DataFrame({
        "chars": [len(s) for s in _txt],
        "words": [len(s.split()) for s in _txt],
        "deontic_per_100w": [100 * len(DEONTIC.findall(s)) / max(len(s.split()), 1)
                             for s in _txt],
        "capitalised_per_100w": [100 * len(_cap.findall(s)) / max(len(s.split()), 1)
                                 for s in _txt],
        "mean_word_len": [np.mean([len(w) for w in s.split()]) if s.split() else 0.0
                          for s in _txt],
        "comma_per_100w": [100 * s.count(",") / max(len(s.split()), 1) for s in _txt],
    })

    _rows = []
    for _c in COVARS.columns:
        _v = COVARS[_c].values
        _rows.append({
            "covariate": _c,
            "spearman_pooled": float(spearmanr(AXIS, _v).statistic),
            "spearman_within_norm": float(spearmanr(AXIS[Y == 0], _v[Y == 0]).statistic),
            "spearman_within_flow": float(spearmanr(AXIS[Y == 1], _v[Y == 1]).statistic),
            "mean_norm": float(_v[Y == 0].mean()),
            "mean_flow": float(_v[Y == 1].mean()),
        })
    axis_covariates = pd.DataFrame(_rows).sort_values(
        "spearman_pooled", key=lambda s: -s.abs())
    save_table(axis_covariates, "axis_covariates")
    print(axis_covariates.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    axis_covariates
    return AXIS, COVARS, axis_covariates


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The extremes of the axis

    The most norm-like and most flow-like records, and — more diagnostically —
    the records each class places *nearest the boundary*, which is where the two
    constructs are hardest to tell apart.
    """)
    return


@app.cell
def _(AXIS, TEXTS, Y, np, pd, save_table):
    _n_txt, _f_txt = TEXTS["asis"]
    _txt = np.r_[_n_txt.values, _f_txt.values]

    def _band(mask, k, most):
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(-AXIS[idx] if most else AXIS[idx])]
        return order[:k]

    _picks, _labels = [], []
    for _m, _nm in ((Y == 0, "norm"), (Y == 1, "flow")):
        for _most, _tag in ((True, "most norm-like"), (False, "most flow-like")):
            _sel = _band(_m, 4, _most)
            _picks.append(_sel)
            _labels += [f"{_nm}: {_tag}"] * len(_sel)

    _idx = np.concatenate(_picks)
    axis_examples = pd.DataFrame({
        "band": _labels,
        "axis": AXIS[_idx].round(3),
        "text": [t[:220] for t in _txt[_idx]],
    })
    save_table(axis_examples, "axis_examples")
    axis_examples
    return (axis_examples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §4 — is normativity latent in the flow tuple?

    `ci_appropriateness` is **not** in the embedded string (that is what
    `noappr` means), so it is a held-out label. Probe: multinomial logistic
    regression on the 4096-D flow embeddings, grouped 5-fold by novel.

    Read this as a statement about the *representation*, not about the world.
    The labels are the extraction teacher's own verdicts, so a probe that
    recovers them shows the CI tuple carries enough signal for the teacher's own
    judgement to be reconstructed — self-consistency, not validated ground
    truth. The K-series established that this teacher's appropriateness labels
    disagree sharply with retrieval-derived gold (kappa 0.05), so a high score
    here does **not** license any claim about correctness.

    The class distribution is very skewed (85% appropriate), so macro-F1 against
    a stratified-random baseline is the number to read, not accuracy.
    """)
    return


@app.cell
def _(FLOW_EMB, flows, np, pd, save_table):
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import StratifiedGroupKFold as _SGKF

    _y = flows["ci_appropriateness"].fillna("missing").astype(str).values
    _g = flows["book_title"].astype(str).values
    _keep = np.flatnonzero(_y != "missing")
    _y, _g, _Xf = _y[_keep], _g[_keep], FLOW_EMB[_keep]

    print("label distribution:", pd.Series(_y).value_counts().to_dict())

    _cv = _SGKF(n_splits=5, shuffle=True, random_state=0)
    _pred = np.empty_like(_y)
    _dumb = np.empty_like(_y)
    for _tr, _te in _cv.split(_Xf, _y, _g):
        _clf = _LR(max_iter=3000, class_weight="balanced").fit(_Xf[_tr], _y[_tr])
        _pred[_te] = _clf.predict(_Xf[_te])
        _dumb[_te] = DummyClassifier(strategy="stratified", random_state=0) \
            .fit(_Xf[_tr], _y[_tr]).predict(_Xf[_te])

    appropriateness_probe = pd.DataFrame([
        {"model": "logreg on 4096-D flow embedding",
         "macro_f1": f1_score(_y, _pred, average="macro"),
         "accuracy": float((_pred == _y).mean())},
        {"model": "stratified random",
         "macro_f1": f1_score(_y, _dumb, average="macro"),
         "accuracy": float((_dumb == _y).mean())},
    ])
    save_table(appropriateness_probe, "appropriateness_probe")
    print(appropriateness_probe.to_string(index=False))
    print()
    print(classification_report(_y, _pred, digits=3))
    appropriateness_probe
    return (appropriateness_probe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary figure
    """)
    return


@app.cell
def _(
    ACCENT,
    FLOW_COLOR,
    NORM_COLOR,
    NULL_COLOR,
    AXIS,
    Y,
    baselines,
    np,
    plt,
    save_caption,
    save_fig,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 3.4))

    # (a) baseline ladder
    _ax = _axes[0]
    _ax.grid(False)
    _ax.xaxis.grid(True, alpha=0.25, lw=0.5)
    _order = ["length", "funcwords", "char3_5", "tfidf_word", "embedding"]
    _b = baselines.query("variant == 'asis'").set_index("features")
    _vals = [_b.loc[f, "accuracy"] if f in _b.index else np.nan for f in _order]
    _cols = [NULL_COLOR] * 4 + [ACCENT]
    _ax.barh(range(len(_order)), _vals, color=_cols, height=0.6)
    for _i, _v in enumerate(_vals):
        if not np.isnan(_v):
            _ax.text(_v + 0.006, _i, f"{_v:.3f}", va="center", fontsize=7,
                     color="#333333")
    _ax.axvline(max((Y == 0).mean(), (Y == 1).mean()), ls=":", lw=0.9,
                color="#555555")
    _ax.text(max((Y == 0).mean(), (Y == 1).mean()) - 0.01, -0.75, "majority",
             fontsize=6.5, color="#555555", ha="right")
    _ax.set_yticks(range(len(_order)))
    _ax.set_yticklabels(_order, fontsize=7.5)
    _ax.set_xlim(0.4, 1.06)
    _ax.set_xlabel("grouped 5-fold accuracy (norm vs flow)")
    _ax.set_title("(a) how much separability is already lexical?", fontsize=9)
    for _sp in ("top", "right", "left"):
        _ax.spines[_sp].set_visible(False)

    # (b) the discriminant axis
    _ax = _axes[1]
    _bins = np.linspace(AXIS.min(), AXIS.max(), 70)
    _ax.hist(AXIS[Y == 0], bins=_bins, color=NORM_COLOR, alpha=0.65, label="norms")
    _ax.hist(AXIS[Y == 1], bins=_bins, color=FLOW_COLOR, alpha=0.65, label="flows")
    _ax.set_xlabel("position on the held-out norm/flow discriminant")
    _ax.set_ylabel("records")
    _ax.set_title("(b) the contrast axis", fontsize=9)
    _ax.legend(frameon=False, loc="upper center")

    _fig.tight_layout()
    save_fig(_fig, "fig_semantic_distinctness")
    save_caption(
        "fig_semantic_distinctness",
        "Is the norm/flow separation semantic or formal?",
        "(a) Accuracy of a logistic classifier separating norms from information "
        "flows, under feature sets of increasing richness, 5-fold "
        "cross-validated grouped by source text so no novel appears in both "
        "train and test. `funcwords` deletes every content word, leaving only "
        "stopwords and punctuation — the grammatical skeleton of our own "
        "serialization template. The gap between it and the 4096-D embedding "
        "bounds how much of the separation is more than format. (b) The "
        "distribution of both constructs along the norm-minus-flow "
        "mean-difference direction, fit on half the rows and evaluated on the "
        "other half.",
        "fig:semantic-distinctness",
        ["semantics", "baselines", "ablation", "camera-ready"],
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
