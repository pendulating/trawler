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
    # What normative grounding does to appropriateness classification

    Built 2026-08-03 for the COLM 2026 camera-ready.

    The paper's claim is that grounding a model in a text's own normative
    universe changes how it reasons about information flows. Every measurement
    of that claim so far has been *downstream* — benchmark deltas on GoldCoin,
    PrivacyLens, ConfAIde — where the effect of grounding is entangled with
    training dynamics, prompt format, and decoding. This notebook measures it
    **directly and row-wise**, on the extraction corpus itself, with no policy
    model in the loop.

    ## The comparison

    The same 16,200 Gemma-4-31B-it fiction flows carry two appropriateness
    labelings:

    | | label | what it is |
    |---|---|---|
    | **SFT treatment** | `teacher_ci` | the teacher's own `ci_appropriateness`, emitted during CI extraction **with no norm in context**. This is literally what SFT trains on (`sft_data_prep.py:233`). |
    | **Grounding treatment** | `grounded` | the same flow re-classified **through the novel's own normative universe**: embed the six CI fields, retrieve the nearest `governs_info_flow` norm *from the same book*, and read appropriateness off that norm's `normative_force × act_polarity`. This is exactly the gold R-DIRECT consumes (`aux_scorers.make_direct_chunk_gold`). |

    The two labelings differ in **exactly one thing**: whether a norm from the
    source text was consulted. Same flows, same extractor, same deontic
    vocabulary. So the row-wise difference between them *is* the effect of
    normative grounding on appropriateness classification, isolated.

    ## Coding of the teacher's `ambiguous` class

    The teacher emits three classes; the deontic gradient expresses two. **The
    707 `ambiguous` flows (4.4%) are coded `inappropriate`, as contextual
    integrity calls for.** Under CI a flow is appropriate only insofar as it
    conforms to the informational norms in force in its context; a flow whose
    conformity cannot be established is not thereby sanctioned. "We cannot tell
    whether this transmission was licensed" is not a finding of
    appropriateness, and coding it as one would build a permissive default into
    the measurement.

    §1.1 reports the binary-intersection alternative (drop the 707) as a
    sensitivity check, and it is also the subset the wiki §17 audit measured.

    ## What this notebook can and cannot license

    It measures **what grounding changes**, and whether that change is
    systematic. It does **not** establish that the grounded label is *more
    correct* — that needs external gold, which is what the benchmark table is
    for. Every claim below is phrased as an effect, not as an accuracy gain.

    This distinction is load-bearing. On 2026-08-03 the same disagreement
    statistic, read as *validity* rather than as *effect*, retracted the m2 and
    K3 negatives (wiki `2026-07-31_kto_plan.md` §17). The numbers here are the
    same numbers; the question asked of them is different, and the §17 caveats
    are reproduced in full in the closing section rather than omitted.

    Tables → `tables/norm_grounding_disagreement/`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data provenance

    | artifact | path |
    |---|---|
    | flows (+ teacher label) | `outputs/2026-07-12_fiction10_flows_gemma4/23-14-17/.../ci_flows.parquet` |
    | norms | `outputs/2026-07-12_fiction10_norms_gemma4/18-36-28/.../structured_norms.parquet` |
    | norm universe (production index) | `outputs/2026-07-25_universe_fiction10_polarity/` |
    | per-flow join | `outputs/2026-08-03_grounding_disagreement/flow_grounding_labels.parquet` |

    The join is built by `scripts/build_grounding_disagreement.py` through
    **production code paths** — `NormRetriever` with the `governs_info_flow`
    filter over the production per-book embedding index (2,870 / 10,032 norms),
    `deontic.flow_appropriateness` for the label. The flow-side query is
    byte-identical to `make_direct_chunk_gold`'s (verified on all 16,200 rows),
    so its embeddings come out of the existing `rground` cache and the whole
    build runs offline with no GPU.

    The universe **must** be the polarity-merged one. `act_polarity` flips the
    sign of `obligatory`/`recommended` for norms about *refraining*; without it
    19% of grounded labels invert (measured 2026-07-25).

    ### Faithfulness guard

    The builder refuses to write unless it reproduces the §17 audit, which was
    measured against a **live embedding server**. Because §17 measured the
    binary intersection, the guard runs on that subset. It passes:

    | stat | §17 audit | rebuilt offline |
    |---|---|---|
    | dual-labelled flows | 15,493 | 15,493 |
    | Cohen's κ | 0.0532 | 0.0527 |
    | raw agreement | 0.7057 | 0.7059 |
    | teacher inappropriate | 11.0% | 11.0% |
    | grounded inappropriate | 25.8% | 25.7% |
    | median retrieval margin | 0.0136 | 0.0136 |

    So this notebook is reading the same object the reward saw.
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
    TAB_DIR = NB_DIR / "tables/norm_grounding_disagreement"
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    BUILD_DIR = PROJECT_ROOT / "outputs/2026-08-03_grounding_disagreement"
    UNIVERSE = PROJECT_ROOT / "outputs/2026-07-25_universe_fiction10_polarity"

    def save_table(df, name, index=True):
        out = TAB_DIR / f"{name}.csv"
        df.to_csv(out, index=index)
        print(f"[table] {out}")
        return df

    return (
        BUILD_DIR,
        PROJECT_ROOT,
        TAB_DIR,
        UNIVERSE,
        json,
        np,
        pd,
        save_table,
    )


@app.cell
def _(BUILD_DIR, json, pd):
    flows = pd.read_parquet(BUILD_DIR / "flow_grounding_labels.parquet")
    build_meta = json.loads((BUILD_DIR / "build_metadata.json").read_text())

    if not build_meta["guard"]["passed"]:
        raise RuntimeError(
            "The per-flow build did not reproduce the wiki §17 audit "
            "(guard overridden). Every number below would be uninterpretable."
        )

    # PRIMARY analysis set: every flow carrying a grounded label, with the
    # teacher's `ambiguous` coded to `inappropriate` per CI.
    ci = flows[flows["scored"]].copy()
    # SENSITIVITY set: binary intersection (drop ambiguous) = the §17 subset.
    dual = flows[flows["dual"]].copy()

    n_all = len(flows)
    n_amb = int((flows["teacher"] == "ambiguous").sum())
    n_unscored = int((~flows["scored"]).sum())

    print(f"flows                    {n_all:,}")
    print(f"  grounded unscored      {n_unscored:,}  ({n_unscored / n_all:.1%})")
    print(f"  teacher=ambiguous      {n_amb:,}  ({n_amb / n_all:.1%})  "
          "→ coded inappropriate (CI)")
    print(f"PRIMARY  (CI coding)     {len(ci):,}  ({len(ci) / n_all:.1%})")
    print(f"SENSITIVITY (drop amb.)  {len(dual):,}  ({len(dual) / n_all:.1%})")
    return build_meta, ci, dual, flows, n_all, n_amb, n_unscored


@app.cell
def _(np, pd):
    def cohen_kappa(a, b):
        """(kappa, observed agreement, chance agreement)."""
        a, b = np.asarray(a), np.asarray(b)
        labels = sorted(set(a) | set(b))
        po = float((a == b).mean())
        pe = float(sum((a == lab).mean() * (b == lab).mean() for lab in labels))
        return ((po - pe) / (1.0 - pe) if pe < 1.0 else 0.0), po, pe

    def effect_block(t, tcol="teacher_ci"):
        """The per-stratum effect summary used by every table below."""
        k, po, _ = cohen_kappa(t[tcol], t["grounded"])
        a2i = ((t[tcol] == "appropriate")
               & (t["grounded"] == "inappropriate")).mean()
        i2a = ((t[tcol] == "inappropriate")
               & (t["grounded"] == "appropriate")).mean()
        return pd.Series({
            "n": len(t),
            "disagree": 1.0 - po,
            "A→I": a2i,
            "I→A": i2a,
            "asymmetry": (a2i / i2a) if i2a > 0 else np.nan,
            "teacher_inappr": (t[tcol] == "inappropriate").mean(),
            "grounded_inappr": (t["grounded"] == "inappropriate").mean(),
            "kappa": k,
        })

    def cluster_bootstrap(t, stat, by="gutenberg_id", n=2000, seed=0):
        """95% CI resampling whole books — flows within a novel share a norm
        universe, so they are not independent draws."""
        rng = np.random.default_rng(seed)
        groups = [g for _, g in t.groupby(by, sort=True)]
        vals = []
        for _ in range(n):
            pick = rng.integers(0, len(groups), len(groups))
            vals.append(stat(pd.concat([groups[i] for i in pick], ignore_index=True)))
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    return cluster_bootstrap, cohen_kappa, effect_block


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The headline: grounding reclassifies 30.9% of flows, 2.15 : 1 toward *inappropriate*

    The paired confusion matrix over all 16,200 flows. Rows are the SFT
    treatment's label under CI coding, columns the grounding treatment's.
    """)
    return


@app.cell
def _(ci, cohen_kappa, pd, save_table):
    conf = pd.crosstab(ci["teacher_ci"], ci["grounded"])
    save_table(conf, "confusion_counts")
    save_table((conf / len(ci) * 100).round(2), "confusion_pct")

    a2i_n = int(conf.loc["appropriate", "inappropriate"])
    i2a_n = int(conf.loc["inappropriate", "appropriate"])
    n_disagree = a2i_n + i2a_n

    # Does grounding have a NET direction? The test is a paired rank test on
    # the shift in inappropriate rate, taken PER NOVEL — ten paired numbers.
    #
    # This used to be McNemar's χ² on the 16,200 flows, which reported
    # χ²(1) = 665, p = 1e-146. That is a flow-level paired test
    # (√665 = 25.8) and it is wrong because it
    # treats 16,200 flows as 16,200 independent draws when they sit inside 10
    # novels that share a norm universe and an extraction pass. The book is the
    # unit of independence, so it is the unit the test runs on. The honest
    # p is 0.004, not 1e-146, and it agrees with the cluster bootstrap below.
    # Wilcoxon signed-rank, not a paired t: the corpus comparison of
    # app:corpus-scaling tests per-book paired shifts with the same rank test,
    # and this is structurally identical data (one shift per novel, n=10). A
    # t-test here gave p=0.0040 against the rank test's 0.0039, so the choice
    # costs nothing and keeps one test family across the paper. Reported with a
    # Hodges-Lehmann shift for the same reason.
    from scipy.stats import wilcoxon

    book_shift = (
        ((ci["grounded"] == "inappropriate").astype(int)
         - (ci["teacher_ci"] == "inappropriate").astype(int))
        .groupby(ci["gutenberg_id"].astype(str)).mean()
    )
    _w = wilcoxon(book_shift.to_numpy(), zero_method="wilcox")
    w_stat, p_direction = float(_w.statistic), float(_w.pvalue)
    # Hodges-Lehmann one-sample estimate: median of the Walsh averages.
    _v = book_shift.to_numpy()
    hl_shift = float(np.median(
        [(_v[i] + _v[j]) / 2 for i in range(len(_v)) for j in range(i, len(_v))]))
    kappa_ci, agree_ci, chance_ci = cohen_kappa(ci["teacher_ci"], ci["grounded"])

    print(conf.to_string())
    print()
    print(f"disagreement   {n_disagree:,} / {len(ci):,} = {n_disagree / len(ci):.1%}")
    print(f"  appropriate → inappropriate   {a2i_n:,}  ({a2i_n / len(ci):.1%})")
    print(f"  inappropriate → appropriate   {i2a_n:,}  ({i2a_n / len(ci):.1%})")
    print(f"  asymmetry                     {a2i_n / i2a_n:.2f} : 1")
    print()
    print(f"inappropriate rate   teacher {(ci['teacher_ci'] == 'inappropriate').mean():.1%}"
          f"  →  grounded {(ci['grounded'] == 'inappropriate').mean():.1%}"
          f"   ({(ci['grounded'] == 'inappropriate').mean() / (ci['teacher_ci'] == 'inappropriate').mean():.2f}×)")
    print(f"Cohen's κ = {kappa_ci:.4f}   (agreement {agree_ci:.4f}, chance {chance_ci:.4f})")
    print()
    print(f"per-novel shift in inappropriate rate ({(book_shift > 0).sum()}/"
          f"{len(book_shift)} novels positive):")
    print("  " + book_shift.mul(100).round(1).to_string().replace("\n", "\n  "))
    print(f"Wilcoxon signed-rank W = {w_stat:.1f}   p = {p_direction:.4f}   "
          f"Hodges-Lehmann shift {hl_shift * 100:+.1f} pts")
    return (
        a2i_n,
        agree_ci,
        book_shift,
        chance_ci,
        conf,
        i2a_n,
        kappa_ci,
        n_disagree,
        hl_shift,
        p_direction,
        w_stat,
    )


@app.cell
def _(ci, cluster_bootstrap, cohen_kappa, save_table, pd):
    boot = {}
    boot["disagreement rate"] = (
        float((ci["teacher_ci"] != ci["grounded"]).mean()),
        cluster_bootstrap(ci, lambda t: float((t["teacher_ci"] != t["grounded"]).mean())),
    )
    boot["Δ inappropriate rate"] = (
        float((ci["grounded"] == "inappropriate").mean()
              - (ci["teacher_ci"] == "inappropriate").mean()),
        cluster_bootstrap(
            ci,
            lambda t: float((t["grounded"] == "inappropriate").mean()
                            - (t["teacher_ci"] == "inappropriate").mean()),
        ),
    )
    boot["Cohen's κ"] = (
        float(cohen_kappa(ci["teacher_ci"], ci["grounded"])[0]),
        cluster_bootstrap(ci, lambda t: cohen_kappa(t["teacher_ci"], t["grounded"])[0]),
    )

    boot_tab = pd.DataFrame(
        [(k, v[0], v[1][0], v[1][1]) for k, v in boot.items()],
        columns=["quantity", "estimate", "ci_lo", "ci_hi"],
    ).set_index("quantity")
    save_table(boot_tab.round(4), "bootstrap_ci")
    print("cluster bootstrap over the 10 novels (2,000 resamples)")
    print(boot_tab.round(4).to_string())
    return boot, boot_tab


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.1 · Sensitivity to the `ambiguous` coding

    The two codings, side by side. **The choice changes the size of the
    directional effect but not its existence, sign, or significance.**
    """)
    return


@app.cell
def _(ci, dual, effect_block, pd, save_table):
    sens = pd.DataFrame({
        "CI coding (ambiguous → inappropriate)": effect_block(ci, "teacher_ci"),
        "binary intersection (drop ambiguous)": effect_block(dual, "teacher"),
    })
    save_table(sens.round(4), "coding_sensitivity")
    print(sens.round(4).to_string())
    return (sens,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Under CI coding the reclassification rate is *higher* (30.9% vs 29.4%) and
    agreement with the teacher is *better* (κ 0.069 vs 0.053), but the
    directional asymmetry is *smaller* (2.15 : 1 vs 3.01 : 1) and so is the
    shift in the inappropriate rate (+11.3 pts vs +14.7 pts).

    The reason is mechanical and worth stating, because the intuition runs the
    other way. Grounding calls 35.4% of the teacher's ambiguous flows
    inappropriate. Coding those flows as `inappropriate` on the teacher side
    converts that 35.4% into **agreement** and the remaining 64.6% into
    **I→A** movement — so admitting them adds 457 rows to the I→A cell and 250
    to the agreeing diagonal, and adds nothing to A→I. That raises κ and
    lowers the asymmetry at the same time.

    Both codings give the same qualitative result: a large, one-sided shift
    toward *inappropriate*, significant at the level of the novel — paired
    Wilcoxon signed-rank $p = 0.004$ under CI coding (9/10 novels positive,
    Hodges--Lehmann shift $+15.5$ points) and $p = 0.002$ under the
    intersection (10/10, $+19.5$ points).
    The CI coding is reported as primary because its permissive alternative —
    treating "we cannot tell" as "this was fine" — is precisely the default CI
    exists to reject. Everything downstream uses it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Is the effect real, or is it the book's norm prior?

    The obvious deflationary reading: a novel's governing norms are 27.5%
    prohibiting-ish, so *any* norm you attach to a flow would flag it
    inappropriate about a quarter of the time. If that were the whole story,
    grounding would be adding a constant, not information.

    The test is the same shuffle-within-book null the embedding notebook uses:
    replace each flow's *retrieved* norm with a **random governing norm from
    the same novel**, and recompute. The null preserves the per-book norm
    prior exactly and destroys only the flow→norm correspondence.
    """)
    return


@app.cell
def _(UNIVERSE, ci, cohen_kappa, json, np, pd, save_table):
    import sys

    if "/share/pierson/matt/UAIR" not in sys.path:
        sys.path.insert(0, "/share/pierson/matt/UAIR")
    from dagspaces.grpo_training.stages.deontic import flow_appropriateness

    universes = json.loads((UNIVERSE / "norm_universes.json").read_text())
    book_pool = {}
    for gid, norms in universes.items():
        labs = [
            flow_appropriateness(str(n.get("normative_force") or ""),
                                 n.get("act_polarity"))
            for n in norms if n.get("governs_info_flow") is True
        ]
        book_pool[gid] = [x for x in labs if x]

    rng = np.random.default_rng(0)
    gid_arr = ci["gutenberg_id"].to_numpy()
    teacher_arr = ci["teacher_ci"].to_numpy()

    null_kappa, null_prior, null_agree = [], [], []
    for _ in range(500):
        draw = np.array([book_pool[g][rng.integers(len(book_pool[g]))]
                         for g in gid_arr])
        k, po, _ = cohen_kappa(teacher_arr, draw)
        null_kappa.append(k)
        null_agree.append(po)
        null_prior.append(float((draw == "inappropriate").mean()))

    obs_kappa, obs_agree, _pe = cohen_kappa(ci["teacher_ci"], ci["grounded"])
    obs_prior = float((ci["grounded"] == "inappropriate").mean())
    z_kappa = (obs_kappa - np.mean(null_kappa)) / np.std(null_kappa)

    null_tab = pd.DataFrame({
        "quantity": ["Cohen's κ vs teacher", "raw agreement",
                     "grounded inappropriate rate"],
        "retrieved": [obs_kappa, obs_agree, obs_prior],
        "shuffled null (mean)": [np.mean(null_kappa), np.mean(null_agree),
                                 np.mean(null_prior)],
        "null sd": [np.std(null_kappa), np.std(null_agree), np.std(null_prior)],
        "null 95%": [
            f"[{np.percentile(null_kappa, 2.5):+.4f}, {np.percentile(null_kappa, 97.5):+.4f}]",
            f"[{np.percentile(null_agree, 2.5):.4f}, {np.percentile(null_agree, 97.5):.4f}]",
            f"[{np.percentile(null_prior, 2.5):.4f}, {np.percentile(null_prior, 97.5):.4f}]",
        ],
    }).set_index("quantity")
    save_table(null_tab.round(4), "shuffle_null")
    print(null_tab.round(4).to_string())
    print(f"\nκ z-score vs the within-book null:  {z_kappa:+.1f}")
    return (
        book_pool,
        flow_appropriateness,
        gid_arr,
        null_agree,
        null_kappa,
        null_prior,
        null_tab,
        obs_agree,
        obs_kappa,
        obs_prior,
        rng,
        sys,
        teacher_arr,
        universes,
        z_kappa,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The effect is flow-specific, not a prior.** Retrieval-grounded labels
    agree with the teacher at κ = 0.069 against a within-book null of
    +0.001 ± 0.007 — **z ≈ +9.7**, and no null draw in 500 came close. The
    book-clustered 95% CI on κ, [0.046, 0.091], excludes zero on its own.

    The grounded inappropriate rate (26.1%) also sits *below* the null's
    (27.3%), so grounding is not simply regressing flows onto the norm pool's
    deontic mix; it is selecting norms whose force is systematically related to
    the particular flow.

    That is the licensing result for everything that follows: the 30.9%
    reclassification carries real information about individual flows.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Dose–response: the effect strengthens with retrieval quality

    If grounding works by *matching a flow to a norm that actually governs it*,
    then agreement with the teacher should rise when the match is better. It
    does, monotonically, across top-1 cosine quintiles.
    """)
    return


@app.cell
def _(ci, effect_block, pd, save_table):
    d_sim = ci.copy()
    d_sim["sim_q"] = pd.qcut(d_sim["top_sim"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    by_sim = (
        d_sim.groupby("sim_q", observed=True)
        .apply(effect_block, include_groups=False)
        .join(d_sim.groupby("sim_q", observed=True)["top_sim"].median()
              .rename("top_sim (median)"))
    )
    by_sim = by_sim[["n", "top_sim (median)", "disagree", "A→I", "I→A",
                     "teacher_inappr", "grounded_inappr", "kappa"]]
    save_table(by_sim.round(4), "by_retrieval_similarity")
    print(by_sim.round(3).to_string())
    return by_sim, d_sim


@app.cell
def _(ci, effect_block, pd, save_table):
    # The margin axis (top1 − top2) is reported alongside because §17 used it,
    # and because it behaves DIFFERENTLY from top_sim — see the note below.
    d_mar = ci.copy()
    d_mar["margin_q"] = pd.qcut(d_mar["margin"], 5,
                                labels=[f"Q{i}" for i in range(1, 6)])
    by_margin = (
        d_mar.groupby("margin_q", observed=True)
        .apply(effect_block, include_groups=False)
        .join(d_mar.groupby("margin_q", observed=True)["margin"].median()
              .rename("margin (median)"))
    )
    by_margin = by_margin[["n", "margin (median)", "disagree", "A→I", "I→A",
                           "teacher_inappr", "grounded_inappr", "kappa"]]
    save_table(by_margin.round(4), "by_retrieval_margin")
    print(by_margin.round(3).to_string())
    return by_margin, d_mar


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Two confidence axes, two different behaviours — and the difference is
    informative.**

    - **top-1 cosine** (how well the nearest norm matches the flow) gives a
      clean monotone gradient: κ climbs 0.017 → 0.143 from the worst to the
      best quintile, roughly an **8×** spread. The teacher's own inappropriate
      rate climbs with it (12.2% → 18.4%), i.e. well-matched flows are ones
      both labelings find more contentious.
    - **top1 − top2 margin** (how *decisively* one norm won) carries no trend.
      κ is non-monotonic across quintiles (0.066, 0.074, 0.038, 0.103, 0.063)
      and disagreement is flat at 30–32% throughout.

    Read together: what matters is whether the retrieved norm *fits the flow*,
    not whether it beat its runner-up. That is the expected signature when many
    norms in a novel are near-paraphrases of one another — they are
    interchangeable for the purpose of classifying the flow, so a small margin
    between them costs nothing. It also explains why §17's margin-gated salvage
    attempt was underpowered: it gated on the axis that does not carry the
    signal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Mechanism: what grounding contributes is *refraining* norms

    Decomposing the effect by the retrieved norm's deontic content shows the
    reclassification is not diffuse. It is carried almost entirely by norms
    whose act polarity is **refraining** — norms that say *withhold this*.
    """)
    return


@app.cell
def _(ci, effect_block, save_table):
    by_pol = (ci.groupby("top_polarity", observed=True)
              .apply(effect_block, include_groups=False)
              .sort_values("n", ascending=False))
    by_force = (ci.groupby("top_force", observed=True)
                .apply(effect_block, include_groups=False)
                .sort_values("n", ascending=False))
    save_table(by_pol.round(4), "by_act_polarity")
    save_table(by_force.round(4), "by_normative_force")
    print("== by act_polarity ==")
    print(by_pol.round(3).to_string())
    print("\n== by normative force ==")
    print(by_force.round(3).to_string())
    return by_force, by_pol


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The mechanism is legible, and the two polarity strata move in opposite
    directions.**

    - **Refraining** (3,160 flows, 19.5% of the corpus): 96.9% relabelled
      inappropriate against a teacher rate of 18.0%. Disagreement 80.0%,
      essentially all of it A→I (79.4% vs 0.5% I→A).
    - **Performing** (12,620 flows, 77.9%): grounding is *more permissive* than
      the teacher — 9.3% inappropriate vs the teacher's 14.0%, with I→A (11.9%)
      exceeding A→I (7.2%).

    The remaining 420 flows (2.6%) carry no `act_polarity`, and they are
    exactly the `permitted`-force rows: `permitted` expresses no directional
    expectation, so the backfill never assigned it a polarity and
    `flow_appropriateness` maps it to *appropriate* unconditionally. They
    account for the entire I→A movement in the `permitted` row of the force
    table. Within the polarity-labelled subset the split is 80 / 20
    performing / refraining.

    So normative grounding is not applying a uniform severity shift. It is
    doing one specific thing: **surfacing the novel's confidentiality and
    discretion norms — the ones phrased as obligations to withhold — and
    applying them to flows the context-free teacher waved through**, while
    *relaxing* its verdict where the governing norm is about performing an act.
    The net corpus-level movement toward inappropriate is the first effect
    outrunning the second.

    The prohibited-force stratum (n = 1,107) is the sharpest version: 99.5%
    grounded-inappropriate, 76.8% of them flips from the teacher's
    *appropriate*.

    This is the CI story in miniature. A context-free reader sees information
    moving between two characters and finds nothing wrong with it; the novel's
    own norms say a lady does not disclose a gentleman's intentions, or a ward
    does not discuss the suit with her guardian, and the same flow becomes a
    violation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Per-novel effects

    The unit of analysis stays the source text: every retrieval is
    within-book, so each novel is an independent replication of the
    experiment.
    """)
    return


@app.cell
def _(ci, effect_block, save_table):
    by_book = (ci.groupby("book_title", observed=True)
               .apply(effect_block, include_groups=False)
               .sort_values("kappa", ascending=False))
    save_table(by_book.round(4), "by_book")
    print(by_book.round(3).to_string())
    return (by_book,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Nine of the ten novels show the directional effect** (A→I exceeds I→A);
    *Nineteen Eighty-Four* is the exception, with the two flip directions
    exactly balanced (15.5% each, asymmetry 1.00). That is not a failure —
    1984 has by far the highest teacher inappropriate rate (25.7%), so there is
    much less one-directional room to move, and it simultaneously posts the
    **highest agreement of any novel (κ = 0.191, ~3× the corpus value)**. A
    novel whose explicit subject is surveillance and information control has a
    normative universe that talks directly about who may tell whom what —
    exactly the regime where grounding and a context-free reader converge on
    the same flows rather than one systematically overriding the other.

    The rest of the ordering is interpretable in the same way: *The Age of
    Innocence* (0.107), *Middlemarch* (0.087), *Anna Karenina* (0.079) and
    *Les Misérables* (0.077) are novels organised around social discretion and
    concealed information.

    ***Alice's Adventures in Wonderland*** is the floor (κ = −0.058, 62.1%
    disagreement). Its "norms" are nonsense-world etiquette with no stable
    relation to information flow, so grounding relabels nearly two-thirds of
    flows without agreeing with anything. It is the useful negative control:
    the method does not manufacture signal where the source text has none.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Where grounding overrides a confident teacher

    If grounding were merely picking up the teacher's own uncertainty, the
    reclassification would concentrate on flows the teacher hedged. It does
    not.
    """)
    return


@app.cell
def _(ci, effect_block, save_table):
    order = ["very_certain", "certain", "somewhat_certain", "uncertain"]
    by_conf = (ci.groupby("ci_confidence_qual", observed=True)
               .apply(effect_block, include_groups=False))
    by_conf = by_conf.reindex([o for o in order if o in by_conf.index])
    save_table(by_conf.round(4), "by_teacher_confidence")
    print(by_conf.round(3).to_string())

    # How much of each confidence stratum is recoded `ambiguous`? The
    # somewhat_certain row is dominated by them, so it must not be read as a
    # finding about the teacher's hedged *verdicts*.
    amb_share = (ci.assign(_a=ci["teacher"] == "ambiguous")
                 .groupby("ci_confidence_qual", observed=True)["_a"].mean()
                 .reindex([o for o in order if o in by_conf.index])
                 .rename("share recoded from ambiguous"))
    save_table(amb_share.round(4).to_frame(), "ambiguous_share_by_confidence")
    print("\n" + amb_share.round(3).to_string())
    return amb_share, by_conf, order


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **84.8% of flows are ones the teacher marked `certain`, and grounding still
    reclassifies 31.3% of them** — 22.6 points of that as *appropriate →
    inappropriate*. The effect is not concentrated in the teacher's hedges: it
    is an override of confident context-free judgments.

    Two reading notes on this table:

    - Agreement tracks confidence in the expected direction at the top
      (κ = 0.087 at `very_certain`, 0.048 at `certain`), which is a sanity
      check on both labelings.
    - The `somewhat_certain` row is **an artifact of the CI coding, not a
      finding**: 67% of those 772 flows are recoded `ambiguous`, which is why
      its teacher inappropriate rate reads 76.4%. It is reported for
      completeness and should not be interpreted as the teacher's behaviour on
      flows it hedged but still classified. The `uncertain` row is n = 2.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Retrieval is not degenerate

    A failure mode worth excluding: if a handful of norms won every retrieval,
    the "effect" would be a few strings broadcast across the corpus.
    """)
    return


@app.cell
def _(ci, pd, save_table):
    vc = ci["top_articulation"].value_counts()
    conc = pd.DataFrame({
        "quantity": ["distinct norms retrieved as top-1",
                     "governing norms in the index",
                     "coverage of the index",
                     "share of flows taking the top-10 norms",
                     "share of flows taking the top-100 norms",
                     "median flows per retrieved norm"],
        "value": [len(vc), 2870, len(vc) / 2870,
                  vc.head(10).sum() / len(ci),
                  vc.head(100).sum() / len(ci),
                  float(vc.median())],
    }).set_index("quantity")
    save_table(conc.round(4), "retrieval_concentration")
    print(conc.round(4).to_string())
    print("\nmost-retrieved norms:")
    for art, n in vc.head(5).items():
        print(f"  {n:>4}  {str(art)[:105]}")
    return art, conc, n, vc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2,198 distinct norms win at least one retrieval — **77% of the 2,870-norm
    index** — and the ten most-retrieved norms together account for only 5.7%
    of flows. The effect is spread across the normative universe, not carried
    by a few dominant strings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Viewer: inspect the flows grounding reclassified

    Every number above is a rate. This is the qualitative counterpart — pick a
    stratum and read the actual passages, so a claim like "grounding surfaces
    discretion norms the context-free reader waved through" can be checked
    against the text rather than taken on faith.

    Each row shows the flow's CI tuple, the novel passage it came from, the
    norm the **teacher itself cited** with no universe in context
    (`ci_norms_invoked`), and the norm **retrieval** surfaced — plus the two
    runners-up, because the median top1−top2 margin is 0.0136 and whether the
    winner was a near-tie is usually the thing you want to see.
    """)
    return


@app.cell
def _(ci, mo):
    FLIP_SETS = {
        "A→I  (grounding flags it)": ("appropriate", "inappropriate"),
        "I→A  (grounding clears it)": ("inappropriate", "appropriate"),
        "agree: both appropriate": ("appropriate", "appropriate"),
        "agree: both inappropriate": ("inappropriate", "inappropriate"),
        "any disagreement": ("*disagree*", None),
        "everything": (None, None),
    }

    v_flip = mo.ui.dropdown(
        options=list(FLIP_SETS), value="A→I  (grounding flags it)",
        label="transition",
    )
    v_book = mo.ui.dropdown(
        options=["(all novels)"] + sorted(ci["book_title"].dropna().unique()),
        value="(all novels)", label="novel", searchable=True,
    )
    v_pol = mo.ui.dropdown(
        options=["(any)", "refraining", "performing", "(no polarity)"],
        value="(any)", label="act polarity",
    )
    v_force = mo.ui.dropdown(
        options=["(any)"] + sorted(ci["top_force"].dropna().unique()),
        value="(any)", label="norm force",
    )
    v_sim = mo.ui.range_slider(
        start=0.0, stop=1.0, step=0.01, value=[0.0, 1.0],
        label="top-1 cosine", show_value=True,
    )
    v_margin = mo.ui.range_slider(
        start=0.0, stop=0.5, step=0.005, value=[0.0, 0.5],
        label="top1−top2 margin", show_value=True,
    )
    v_search = mo.ui.text(
        placeholder="substring in passage, CI tuple, or norm …",
        label="search", full_width=True,
    )
    v_sort = mo.ui.dropdown(
        options=["top-1 cosine (high→low)", "top-1 cosine (low→high)",
                 "margin (high→low)", "margin (low→high)", "book order"],
        value="top-1 cosine (high→low)", label="sort by",
    )

    controls = mo.vstack([
        mo.hstack([v_flip, v_book, v_sort], justify="start", gap=1),
        mo.hstack([v_pol, v_force], justify="start", gap=1),
        mo.hstack([v_sim, v_margin], justify="start", gap=1),
        v_search,
    ])
    controls
    return (
        FLIP_SETS,
        controls,
        v_book,
        v_flip,
        v_force,
        v_margin,
        v_search,
        v_sim,
        v_sort,
    )


@app.cell
def _(
    FLIP_SETS,
    ci,
    v_book,
    v_flip,
    v_force,
    v_margin,
    v_pol,
    v_search,
    v_sim,
    v_sort,
):
    view = ci

    _t, _g = FLIP_SETS[v_flip.value]
    if _t == "*disagree*":
        view = view[view["teacher_ci"] != view["grounded"]]
    elif _t is not None:
        view = view[(view["teacher_ci"] == _t) & (view["grounded"] == _g)]

    if v_book.value != "(all novels)":
        view = view[view["book_title"] == v_book.value]
    if v_pol.value == "(no polarity)":
        # The `permitted`-force rows: no directional expectation, so the
        # backfill never assigned a polarity.
        view = view[view["top_polarity"].isna()]
    elif v_pol.value != "(any)":
        view = view[view["top_polarity"] == v_pol.value]
    if v_force.value != "(any)":
        view = view[view["top_force"] == v_force.value]

    lo_s, hi_s = v_sim.value
    view = view[view["top_sim"].between(lo_s, hi_s)]
    lo_m, hi_m = v_margin.value
    view = view[view["margin"].between(lo_m, hi_m)]

    if (q := v_search.value.strip()):
        cols = ["ci_flow_snippet", "ci_sender", "ci_recipient",
                "ci_information_type", "ci_transmission_principle",
                "ci_context", "ci_subject", "top_articulation",
                "ci_norms_invoked"]
        hit = None
        for c in cols:
            m = view[c].astype("string").str.contains(q, case=False, na=False)
            hit = m if hit is None else (hit | m)
        view = view[hit]

    _sorts = {
        "top-1 cosine (high→low)": ("top_sim", False),
        "top-1 cosine (low→high)": ("top_sim", True),
        "margin (high→low)": ("margin", False),
        "margin (low→high)": ("margin", True),
        "book order": ("book_title", True),
    }
    _col, _asc = _sorts[v_sort.value]
    view = view.sort_values(_col, ascending=_asc).copy()

    # A stable per-flow key. The detail card looks the selection up by this
    # rather than by DataFrame index, so it does not depend on whether
    # `mo.ui.table` preserves the index of what it was handed.
    view["uid"] = (
        view["gutenberg_id"].astype(str) + "|"
        + view["chunk_id"].astype(str) + "|"
        + view["ci_flow_index"].astype(str)
    )
    return lo_m, lo_s, hi_m, hi_s, view


@app.cell
def _(ci, mo, view):
    mo.md(
        f"**{len(view):,}** flows match "
        f"({len(view) / len(ci):.1%} of the corpus). "
        + ("Select a row below to inspect it."
           if len(view) else "_Loosen a filter._")
    )
    return


@app.cell
def _(mo, view):
    _cols = ["uid", "book_title", "ci_sender", "ci_recipient",
             "ci_information_type", "ci_transmission_principle",
             "top_force", "top_polarity", "top_sim", "margin"]
    picker = mo.ui.table(
        view[_cols].head(500).round({"top_sim": 3, "margin": 4}),
        selection="single",
        page_size=10,
        freeze_columns_left=["uid"],
        label="matching flows (first 500 of the current filter)",
    )
    picker
    return (picker,)


@app.cell
def _(mo, picker, view):
    def _esc(x):
        return (str(x) if x is not None else "—").replace("|", "\\|")

    _sel = None if picker.value is None or len(picker.value) == 0 else picker.value
    _hit = None if _sel is None else view[view["uid"] == _sel["uid"].iloc[0]]

    if _hit is None or len(_hit) == 0:
        card = mo.md("_No row selected — click one in the table above._")
    else:
        r = _hit.iloc[0]

        _tuple_rows = "\n".join(
            f"| {k} | {_esc(r[c])} |" for k, c in [
                ("sender", "ci_sender"), ("recipient", "ci_recipient"),
                ("subject", "ci_subject"),
                ("information type", "ci_information_type"),
                ("transmission principle", "ci_transmission_principle"),
                ("context", "ci_context"),
            ]
        )

        _nbrs = "\n".join(
            f"| {i + 1} | {s:.4f} | {_esc(f)} / {_esc(p) or '—'} | "
            f"**{_esc(imp)}** | {_esc(a)} |"
            for i, (a, f, p, s, imp) in enumerate(zip(
                r["nbr_articulation"], r["nbr_force"], r["nbr_polarity"],
                r["nbr_sim"], r["nbr_implies"],
            ))
        )

        _verdict = (
            "**reclassified** — grounding "
            + ("flags this flow" if r["grounded"] == "inappropriate"
               else "clears this flow")
            if r["teacher_ci"] != r["grounded"] else "**both labelings agree**"
        )
        _amb = ("  \n_(teacher said `ambiguous`; coded `inappropriate` per CI)_"
                if r["teacher"] == "ambiguous" else "")

        card = mo.md(f"""
### {r['book_title']} · chunk {r['chunk_id']} · flow {r['ci_flow_index']}

| | |
|---|---|
| **SFT label** (teacher, no norm in context) | `{r['teacher_ci']}` |
| **grounded label** (nearest governing norm) | `{r['grounded']}` |
| | {_verdict}{_amb} |
| teacher confidence | {_esc(r['ci_confidence_qual'])} |
| top-1 cosine / margin | {r['top_sim']:.4f} / {r['margin']:.4f} |

**Passage**

> {_esc(r['ci_flow_snippet'])}

**CI tuple**

| field | value |
|---|---|
{_tuple_rows}

**Norm the teacher cited on its own** (no universe in context)

> {_esc(r['ci_norms_invoked'])}

**Norms retrieval surfaced** — rank 1 is the one that sets the grounded label

| # | cos | force / polarity | implies | articulation |
|---|---|---|---|---|
{_nbrs}
""")
    card
    return card, r


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A worked example, and what it shows

    Set the filters to **A→I**, top-1 cosine ≥ 0.70, sorted high→low, and the
    first row is *Les Misérables* chunk 239 (uid `135|239|0`), cosine 0.866:

    > the parlor, which had a window on the side of the world, had none on the
    > side of the convent. Profane eyes must see nothing of that sacred place.

    The flow is `the convent → the outside world`, information type *visual
    access to sacred spaces*, transmission principle *Confidentiality*. The
    teacher labelled it **appropriate**. The retrieved norm — *"A person of the
    secular world must not see anything of the sacred interior of the
    convent"*, prohibited/performing — makes it **inappropriate**, and both
    runners-up say the same thing at cosine 0.83 and 0.77, so the verdict does
    not hinge on a near-tie.

    The instructive part is the teacher's *own* cited norm: *"Sacred spaces and
    the activities within a convent must be shielded from the view of the
    secular or 'profane' public to maintain sanctity."* The teacher retrieved
    the right normative content from its own reading and **still returned
    `appropriate`**. So at least some of the A→I mass is not the teacher
    missing the norm; it is the teacher failing to *apply* a norm it has
    already articulated. That is a sharper claim than "grounding adds
    knowledge", and the viewer is how it gets checked case by case rather than
    asserted.

    Two honest caveats on reading the viewer this way. Selecting on high cosine
    selects the cases most flattering to the method — §3 shows κ is 8× higher
    in the top quintile than the bottom, so browse the low-cosine end too.
    And a single vivid example is not evidence of a rate; the rates are §1–§7.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Camera-ready summary table
    """)
    return


@app.cell
def _(a2i_n, ci, hl_shift, i2a_n, kappa_ci, p_direction, pd, save_table, z_kappa):
    headline = pd.DataFrame(
        [
            ("flows compared", f"{len(ci):,}"),
            ("ambiguous coded inappropriate (CI)",
             f"{(ci['teacher'] == 'ambiguous').sum():,}  "
             f"({(ci['teacher'] == 'ambiguous').mean():.1%})"),
            ("reclassified by grounding",
             f"{(a2i_n + i2a_n) / len(ci):.1%}  ({a2i_n + i2a_n:,})"),
            ("  appropriate → inappropriate",
             f"{a2i_n / len(ci):.1%}  ({a2i_n:,})"),
            ("  inappropriate → appropriate",
             f"{i2a_n / len(ci):.1%}  ({i2a_n:,})"),
            ("directional asymmetry", f"{a2i_n / i2a_n:.2f} : 1"),
            ("direction, paired rank test over 10 novels",
             f"Wilcoxon signed-rank p = {p_direction:.3f}, "
             f"HL shift {hl_shift * 100:+.1f} pts"),
            ("inappropriate rate, SFT label",
             f"{(ci['teacher_ci'] == 'inappropriate').mean():.1%}"),
            ("inappropriate rate, grounded label",
             f"{(ci['grounded'] == 'inappropriate').mean():.1%}"),
            ("Cohen's κ", f"{kappa_ci:.4f}"),
            ("κ vs within-book shuffled null", f"z = {z_kappa:+.1f}"),
            ("κ, worst → best retrieval quintile", "0.017 → 0.143"),
            ("effect carried by refraining norms",
             f"{(ci['top_polarity'] == 'refraining').mean():.1%} of flows, "
             "96.9% relabelled inappropriate"),
            ("novels showing the directional effect", "9 / 10 (1984 balanced)"),
        ],
        columns=["quantity", "value"],
    ).set_index("quantity")
    save_table(headline, "headline")
    print(headline.to_string())
    return (headline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · Prose for the camera-ready

    > We measure the effect of normative grounding directly on the extraction
    > corpus, without a policy model in the loop. Each of the 16,200 fiction
    > flows carries the teacher's own appropriateness judgment, produced with
    > no norm in context — the label SFT trains on — and a second judgment
    > obtained by retrieving the flow's nearest information-flow-governing norm
    > from the same novel and reading appropriateness off that norm's deontic
    > force and act polarity, which is the signal the grounded reward consumes.
    > The two labelings differ in exactly one respect: whether the source
    > text's normative universe was consulted. Following contextual integrity,
    > the teacher's `ambiguous` verdicts (707 flows, 4.4%) are coded
    > *inappropriate*: a flow whose conformity to the norms in force cannot be
    > established is not thereby sanctioned.
    >
    > Grounding reclassifies 30.9% of flows, and it does so with a strong
    > direction: 21.1% move from *appropriate* to *inappropriate* against 9.8%
    > moving the other way, a 2.15 : 1 asymmetry. The corpus-level rate of
    > inappropriate flows rises from 14.9% to 26.1%, a shift of +11.3 points
    > whose 95% CI under a bootstrap resampling whole novels is [+6.4, +17.6]
    > points. Because flows within a novel share a norm universe, we test the
    > direction at the level of the novel rather than the flow: the shift is
    > positive in 9 of 10 novels (Wilcoxon signed-rank p = 0.004,
    > Hodges-Lehmann shift +15.5 points). The effect is
    > flow-specific rather than a property of the norm pool: replacing each
    > flow's retrieved norm with a random governing norm from the same novel
    > drives agreement with the teacher to chance (κ = +0.001 ± 0.007 over 500
    > draws), while retrieval attains κ = 0.069, z ≈ +9.7, with a
    > book-clustered CI of [0.046, 0.091]. It also strengthens with retrieval
    > quality, κ rising monotonically from 0.017 to 0.143 across quintiles of
    > top-1 cosine similarity.
    >
    > Decomposing by the retrieved norm shows the mechanism is specific rather
    > than a uniform severity shift, and that the two polarity strata move in
    > opposite directions. Flows matched to norms of *refraining* polarity —
    > the novel's confidentiality and discretion norms, 19.5% of the corpus —
    > are relabelled inappropriate 96.9% of the time against a teacher rate of
    > 18.0%, whereas flows matched to *performing* norms become *more*
    > permissive than the teacher (9.3% vs 14.0%). Grounding is surfacing
    > obligations to withhold that a context-free reader does not apply, and
    > the net movement is the first effect outrunning the second. Nine of the
    > ten novels show the directional asymmetry, and agreement tracks how much
    > each text is *about* information control: *Nineteen Eighty-Four* attains
    > the highest agreement (κ = 0.191) while *Alice's Adventures in
    > Wonderland*, whose norms bear no stable relation to information flow,
    > attains the lowest (κ = −0.058) — a useful negative control that the
    > method does not manufacture signal where the source text has none.

    ### Statement of scope (keep this attached to the claim)

    This is an **effect** measurement, not an accuracy measurement. It shows
    that normative grounding changes appropriateness classification in a
    systematic, flow-specific, strongly directional way. It does **not** show
    that the grounded label is closer to a human notion of correctness; the two
    labelings agree only slightly above chance (κ = 0.069), so agreement with
    the context-free teacher cannot be used to argue either is right. External
    validity for the grounded direction rests on the benchmark results
    (GoldCoin / PrivacyLens / ConfAIde), which use independent gold.

    Three further limits:

    1. Because the two labelings are near-independent, **this gold must not be
       used to score a policy trained on the other one.** That is the defect
       that retracted the m2 and K3 negatives (wiki §17), and nothing here
       rehabilitates it. The effect measured in this notebook is a property of
       the *labeling procedures*, and says nothing about whether the grounded
       label is a usable training target.
    2. The median top1−top2 retrieval margin is 0.0136, so which norm ranks
       first is often a near-tie. §4 shows this costs less than it appears —
       near-tied norms tend to be paraphrases with the same polarity, and the
       signal tracks top-1 *similarity*, not margin — but any claim that rests
       on the identity of a particular retrieved norm, rather than on its
       deontic class, is not supported.
    3. The CI coding of `ambiguous` **reduces** the headline asymmetry (2.15 : 1
       against 3.01 : 1 if those flows are dropped) while **raising** agreement
       (κ 0.069 against 0.053). Both codings are reported in §1.1 and neither
       changes the sign or the significance, but the specific figures quoted
       above are the CI-coded ones and should not be mixed with §17's, which
       are computed on the binary intersection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11 · Next: the teacher re-judge arm

    The grounded label above is a **deterministic** read-off — the deontic
    force of whichever norm ranked first, applied mechanically. It cannot
    decline to apply a norm that does not in fact govern the flow, which is the
    one thing that would separate *"grounding changed the verdict"* from
    *"retrieval handed us the wrong norm"*.

    The registered next step is a third labeling: the teacher
    (`gemma-4-31b/instruct`) re-judges each flow **with its retrieved norm in
    context**, returning `appropriate` / `inappropriate` /
    `norm-does-not-govern-this-flow`. The abstain option is the point —
    §4 predicts the *refraining* stratum should largely survive re-judgment
    while the near-tied *performing* stratum should abstain heavily, and that
    is a falsifiable prediction of the mechanism claimed above.

    The per-flow table this notebook reads already carries the top-3
    neighbours' identity, force, polarity and similarity, so that arm needs
    only a prompt, an inference stage, and a `rejudged` column appended to
    `flow_grounding_labels.parquet`. Every table above is written against
    `teacher_ci` / `grounded` column names and will extend to a third arm
    without restructuring.
    """)
    return


if __name__ == "__main__":
    app.run()
