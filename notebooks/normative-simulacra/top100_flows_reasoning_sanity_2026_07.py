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
    # Sanity check: top100 flows-reasoning run (job 1 of the top100-flows GRPO plan)

    Built 2026-07-12, against the run that finished 2026-07-08:

    | | |
    |---|---|
    | **run** | `outputs/2026-07-08_top100_flows_reasoning/20-38-41` |
    | **pipeline** | `COLM_flows_reasoning_prefetched_qwen36` (stage-1 only) |
    | **model** | Qwen3.6-27B-Instruct, `enable_thinking: false`, temp 0.0 |
    | **input** | `chunks_top100_fiction_en.parquet` (100 books, 15,875 chunks) |
    | **baseline** | fiction10 `ci_reasoning.parquet` (10 books, 2,993 chunks, flow rate 0.129) |
    | **plan** | `wiki/grpo_training_field_notes/2026-07-08_top100_flows_plan.md` |

    The check runs in three tiers: **(1) structural integrity** of the artifact,
    **(2) the flow gate** (`has_information_exchange`) against the fiction10
    baseline — including a paired same-chunk comparison on the 7 books both
    corpora share — and **(3) prompt forensics**, because the gate turns out to
    be the story. A downstream-viability verdict for job 2
    (`scripts/run_grpo_top100_flows.sh`) closes the notebook.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import yaml

    ROOT = Path("/share/pierson/matt/UAIR")
    RUN_DIR = ROOT / "outputs/2026-07-08_top100_flows_reasoning/20-38-41"
    T100_PARQUET = (
        RUN_DIR / "COLM_flows_reasoning_qwen36/outputs/ci_reasoning/reasoning.parquet"
    )
    F10_PARQUET = Path("/share/pierson/matt/n2s4cir/data/fiction10/ci_reasoning.parquet")
    HYDRA_CFG = RUN_DIR / ".hydra/config.yaml"
    PROMPT_DIR = ROOT / "dagspaces/historical_norms/conf/prompt"
    CHUNK_CACHE = Path(
        "/share/pierson/matt/zoo/datasets/gutenberg_cache/chunks_top100_fiction_en.parquet"
    )

    # The three books the norms-track has_norms gate zeroed (no normative
    # universe -> excluded from GRPO training data by run_grpo_top100_flows.sh).
    EXCLUDE_BOOKS = {"35", "215", "6133"}

    # Columns the GRPO loader consumes via CI_REASONING_PATH.
    GRPO_REQUIRED_COLS = [
        "gutenberg_id",
        "chunk_id",
        "article_text",
        "has_information_exchange",
        "ci_flow_count",
        "ci_reasoning_text",
    ]

    # dataviz tokens (light surface)
    C_BLUE, C_GRAY, C_INK, C_MUTED = "#2a78d6", "#9b9a94", "#0b0b0b", "#52514e"
    return (
        CHUNK_CACHE,
        C_BLUE,
        C_GRAY,
        C_INK,
        C_MUTED,
        EXCLUDE_BOOKS,
        F10_PARQUET,
        GRPO_REQUIRED_COLS,
        HYDRA_CFG,
        PROMPT_DIR,
        T100_PARQUET,
        np,
        pd,
        plt,
        yaml,
    )


@app.cell
def _(CHUNK_CACHE, F10_PARQUET, T100_PARQUET, pd):
    import pyarrow.parquet as pq

    t100 = pd.read_parquet(T100_PARQUET)
    f10 = pd.read_parquet(F10_PARQUET)
    cache_rows = pq.ParquetFile(CHUNK_CACHE).metadata.num_rows
    return cache_rows, f10, t100


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Structural integrity

    Is the artifact well-formed and complete — before we ask whether its
    *content* is right? Every check below must pass for the parquet to even be
    loadable as GRPO training data.
    """)
    return


@app.cell
def _(GRPO_REQUIRED_COLS, cache_rows, mo, pd, t100):
    _err = t100["ci_reasoning_parse_error"].fillna("")
    _missing_cols = [c for c in GRPO_REQUIRED_COLS if c not in t100.columns]
    _dupes = int(t100.duplicated(subset=["gutenberg_id", "chunk_id"]).sum())
    _empty_gen = int((t100["generated_text"].str.len() == 0).sum())
    n_parse_err = int((_err != "").sum())
    parse_err_rate = n_parse_err / len(t100)

    _checks = [
        (
            "row count == chunk cache",
            f"{cache_rows}",
            f"{len(t100)}",
            len(t100) == cache_rows,
        ),
        (
            "book count == 100",
            "100",
            f"{t100['gutenberg_id'].astype(str).nunique()}",
            t100["gutenberg_id"].astype(str).nunique() == 100,
        ),
        (
            "GRPO loader columns present",
            ", ".join(GRPO_REQUIRED_COLS),
            "all present" if not _missing_cols else f"MISSING: {_missing_cols}",
            not _missing_cols,
        ),
        (
            "unique (gutenberg_id, chunk_id)",
            "0 duplicates",
            f"{_dupes} duplicates",
            _dupes == 0,
        ),
        (
            "no empty generations",
            "0 empty",
            f"{_empty_gen} empty",
            _empty_gen == 0,
        ),
        (
            "JSON parse-error rate < 1%",
            "< 1%",
            f"{n_parse_err} rows ({parse_err_rate:.2%})",
            parse_err_rate < 0.01,
        ),
        (
            "parse errors imply gate=False (conservative default)",
            "no True among unparsed",
            f"{int(t100.loc[_err != '', 'has_information_exchange'].sum())} True",
            int(t100.loc[_err != "", "has_information_exchange"].sum()) == 0,
        ),
    ]
    structural_df = pd.DataFrame(
        [
            {
                "check": c,
                "expected": e,
                "observed": o,
                "status": "✅ pass" if ok else "❌ FAIL",
            }
            for c, e, o, ok in _checks
        ]
    )
    structural_ok = all(ok for *_rest, ok in _checks)
    mo.vstack(
        [
            mo.ui.table(structural_df, selection=None, pagination=False),
            mo.md(
                f"**Tier 1: {'all structural checks pass' if structural_ok else 'STRUCTURAL FAILURES — stop here'}.** "
                "The artifact is complete and well-formed; the problem (below) is "
                "semantic, not mechanical."
                if structural_ok
                else "**Tier 1 FAILED — fix the artifact before reading further.**"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The flow gate

    `has_information_exchange` is the gate that selects GRPO extraction
    prompts. The plan's wall-time budget assumed the fiction10 flow rate
    (0.129) would roughly carry over — `TARGET_FLOW_CHUNKS=1400` out of an
    expected ~2,000 flow chunks.
    """)
    return


@app.cell
def _(f10, mo, pd, t100):
    def _summary(df, name):
        return {
            "corpus": name,
            "chunks": len(df),
            "books": df["gutenberg_id"].astype(str).nunique(),
            "flow chunks": int(df["has_information_exchange"].sum()),
            "flow rate": round(float(df["has_information_exchange"].mean()), 4),
            "parse errors": int((df["ci_reasoning_parse_error"].fillna("") != "").sum()),
        }

    flow_summary_df = pd.DataFrame(
        [
            _summary(t100, "top100 (this run, Qwen3.6-27B)"),
            _summary(f10, "fiction10 (March, Qwen2.5-72B-AWQ)"),
        ]
    )
    t100_rate = float(t100["has_information_exchange"].mean())
    f10_rate = float(f10["has_information_exchange"].mean())
    t100_flows = int(t100["has_information_exchange"].sum())
    mo.ui.table(flow_summary_df, selection=None, pagination=False)
    return f10_rate, t100_flows, t100_rate


@app.cell(hide_code=True)
def _(f10_rate, mo, t100_flows, t100_rate):
    mo.callout(
        mo.md(
            f"**FAIL — the flow gate collapsed.** Flow rate **{t100_rate:.4f}** vs the "
            f"fiction10 baseline **{f10_rate:.3f}** — a **{f10_rate / t100_rate:.1f}× drop**. "
            f"The whole 100-book corpus yields **{t100_flows} flow chunks**, "
            f"*fewer than fiction10's 386 from 10 books* and 6× short of the plan's "
            f"`TARGET_FLOW_CHUNKS=1400`. On this artifact, job 2 would train on a "
            f"smaller prompt pool than the run it was designed to escape."
        ),
        kind="danger",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Per-book flow rates

    Rank plot of all 100 books. The gray dashed line is the fiction10 corpus
    rate under the same gate prompt (but a different judge model — see §5).
    """)
    return


@app.cell
def _(C_BLUE, C_GRAY, C_INK, C_MUTED, EXCLUDE_BOOKS, f10_rate, np, plt, t100):
    per_book = (
        t100.assign(gid=t100["gutenberg_id"].astype(str))
        .groupby(["gid", "book_title"], as_index=False)
        .agg(
            chunks=("chunk_id", "count"),
            flow_chunks=("has_information_exchange", "sum"),
        )
    )
    per_book["flow_rate"] = per_book["flow_chunks"] / per_book["chunks"]
    per_book["excluded_no_universe"] = per_book["gid"].isin(EXCLUDE_BOOKS)
    per_book = per_book.sort_values("flow_rate", ascending=False).reset_index(drop=True)
    n_zero_books = int((per_book["flow_chunks"] == 0).sum())

    _fig, _ax = plt.subplots(figsize=(9, 3.4))
    _x = np.arange(len(per_book))
    _ax.bar(_x, per_book["flow_rate"], width=1.0, color=C_BLUE, edgecolor="none")
    _ax.axhline(f10_rate, color=C_GRAY, linestyle="--", linewidth=1.2)
    _ax.text(
        len(per_book) - 1,
        f10_rate + 0.004,
        f"fiction10 corpus rate {f10_rate:.3f}",
        ha="right",
        color=C_MUTED,
        fontsize=9,
    )
    for _i, _dy in enumerate([2, 12, -13]):
        _ax.annotate(
            per_book.loc[_i, "book_title"].split(";")[0][:22],
            (_i, per_book.loc[_i, "flow_rate"]),
            xytext=(6, _dy),
            textcoords="offset points",
            fontsize=8,
            color=C_MUTED,
        )
    _ax.set_xlim(-0.5, len(per_book) - 0.5)
    _ax.set_xlabel("book (ranked by flow rate)", color=C_MUTED)
    _ax.set_ylabel("has_information_exchange rate", color=C_MUTED)
    _ax.set_title(
        f"top100 per-book flow rate — {n_zero_books}/100 books have ZERO flow chunks",
        color=C_INK,
        fontsize=11,
        loc="left",
    )
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(axis="y", color="#eceae6", linewidth=0.8)
    _ax.set_axisbelow(True)
    _ax.tick_params(colors=C_MUTED)
    _fig.tight_layout()
    _fig
    return n_zero_books, per_book


@app.cell
def _(mo, n_zero_books, per_book):
    mo.vstack(
        [
            mo.md(
                f"**{n_zero_books}/100 books contribute nothing.** Top contributors "
                "(the run's entire signal sits in a handful of dialogue-heavy novels):"
            ),
            mo.ui.table(
                per_book.head(15).drop(columns=["excluded_no_universe"]),
                selection=None,
                pagination=False,
            ),
            mo.md(
                "The three universe-less books "
                f"(`{', '.join(sorted(per_book.loc[per_book['excluded_no_universe'], 'gid']))}` — "
                "excluded from training anyway) contribute "
                f"{int(per_book.loc[per_book['excluded_no_universe'], 'flow_chunks'].sum())} "
                "flow chunks, so the exclusion is not the cause."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Same chunks, March vs. now

    Seven fiction10 books also sit in the top100 cache **with byte-identical
    chunking** (same chunk counts per book). Pairing on
    `(gutenberg_id, chunk_id)` isolates the extraction config from the corpus:
    if the corpus were the cause, paired rates would match.
    """)
    return


@app.cell
def _(f10, mo, pd, t100):
    paired = f10[
        ["gutenberg_id", "chunk_id", "has_information_exchange"]
    ].merge(
        t100[
            [
                "gutenberg_id",
                "chunk_id",
                "book_title",
                "article_text",
                "has_information_exchange",
                "generated_text",
            ]
        ],
        on=["gutenberg_id", "chunk_id"],
        suffixes=("_f10", "_t100"),
    )
    _ct = pd.crosstab(
        paired["has_information_exchange_f10"],
        paired["has_information_exchange_t100"],
    )
    _ct_disp = pd.DataFrame(
        {
            "fiction10 ↓ · top100 →": ["gate False", "gate True"],
            "gate False": _ct[False].tolist(),
            "gate True": _ct[True].tolist(),
        }
    )
    n_lost = int(
        (paired["has_information_exchange_f10"] & ~paired["has_information_exchange_t100"]).sum()
    )
    n_gained = int(
        (~paired["has_information_exchange_f10"] & paired["has_information_exchange_t100"]).sum()
    )
    _pb = (
        paired.groupby("book_title")
        .agg(
            chunks=("chunk_id", "count"),
            f10_rate=("has_information_exchange_f10", "mean"),
            t100_rate=("has_information_exchange_t100", "mean"),
        )
        .round(3)
        .sort_values("f10_rate", ascending=False)
        .reset_index()
    )
    mo.vstack(
        [
            mo.md(
                f"**{len(paired):,} paired chunks across 7 shared books.** "
                f"The disagreement is one-directional: **{n_lost} flows lost, "
                f"{n_gained} gained** — the same chunks, the same gate question, "
                "a collapsed answer. The corpus is exonerated; the config is not."
            ),
            mo.ui.table(_ct_disp, selection=None, pagination=False),
            mo.ui.table(_pb, selection=None, pagination=False),
        ]
    )
    return (paired,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Root cause: the prompt that actually ran

    The pipeline yaml says `prompt_ci_reasoning: ${prompt_ci_reasoning_fiction}`.
    Below we compare the **resolved prompt saved in the run's `.hydra/config.yaml`**
    against the two prompt variants on disk.
    """)
    return


@app.cell
def _(HYDRA_CFG, PROMPT_DIR, mo, pd, yaml):
    def _body(d):
        # `name` is excluded: the provenance field was added to the prompt
        # yamls on 2026-07-12 (post-run), so the run's .hydra copy lacks it.
        return {
            k: v
            for k, v in d.items()
            if not k.startswith("_") and k not in ("defaults", "name")
        }

    prompt_ran = yaml.safe_load(HYDRA_CFG.read_text())["prompt_ci_reasoning"]
    prompt_presc = yaml.safe_load((PROMPT_DIR / "ci_reasoning_prescriptive.yaml").read_text())
    prompt_fict = yaml.safe_load((PROMPT_DIR / "ci_reasoning_fiction.yaml").read_text())
    ran_is_prescriptive = _body(prompt_ran) == _body(prompt_presc)
    ran_is_fiction = _body(prompt_ran) == _body(prompt_fict)

    _rows = pd.DataFrame(
        [
            {
                "prompt": "what the run USED (.hydra, resolved)",
                "system_prompt opens with…": prompt_ran["system_prompt"][:120] + "…",
            },
            {
                "prompt": "ci_reasoning_prescriptive.yaml",
                "system_prompt opens with…": prompt_presc["system_prompt"][:120] + "…",
            },
            {
                "prompt": "ci_reasoning_fiction.yaml (intended)",
                "system_prompt opens with…": prompt_fict["system_prompt"][:120] + "…",
            },
        ]
    )
    mo.vstack(
        [
            mo.ui.table(_rows, selection=None, pagination=False),
            mo.callout(
                mo.md(
                    f"resolved prompt **== prescriptive variant: `{ran_is_prescriptive}`** · "
                    f"== fiction variant: `{ran_is_fiction}`"
                    + (
                        "\n\nThe run asked a *prescriptive-and-religious-texts* judge to find "
                        "flows that are **prescribed, commanded, prohibited, or regulated** — "
                        "in novels. The model repeatedly (and correctly, per its instructions) "
                        "answers that Austen doesn't legislate gossip."
                        if ran_is_prescriptive
                        else ""
                    )
                ),
                kind="danger" if ran_is_prescriptive else "success",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why the fiction override never applies

    `dagspaces/historical_norms/conf/config.yaml` orders its defaults list:

    ```yaml
    defaults:
      - _self_                                          # 1. base config body
      - pipeline: norm_extraction                       # 2. pipeline body (@_global_)
      ...
      - prompt@prompt_ci_reasoning: ci_reasoning_prescriptive   # 3. merged LAST
    ```

    Hydra merges later entries **over** earlier ones. The pipeline body's
    `prompt_ci_reasoning: ${prompt_ci_reasoning_fiction}` (step 2) is
    clobbered by the `prompt@prompt_ci_reasoning` group default (step 3) —
    the interpolation line is **dead code**, and has been since the prompt
    groups were introduced (commit `fb7256e`, 2026-03-09). The same dead line
    sits in `COLM_flows_fiction.yaml`, `COLM_flows_fiction_prefetched.yaml`,
    and `COLM_flows_fiction_prefetched_qwen36.yaml`.

    **Corollary (checked below): fiction10's own flows data was produced with
    the prescriptive prompt too.** The 0.129 → 0.014 collapse is therefore the
    *judge-model* shift (lenient Qwen2.5-72B-AWQ → literal Qwen3.6-27B) applied
    to the *same mis-framed prompt* — the exact phenomenon already documented
    for the norms-track `has_norms` gate ("97 books").
    """)
    return


@app.cell
def _(f10, mo, pd):
    _low = f10["generated_text"].str.lower()
    _kw_rows = [
        {"framing keyword": _k, "fiction10 outputs containing it": int(_low.str.contains(_k, regex=False).sum())}
        for _k in ["prescriptive", "religious", "command", "prohibit"]
    ]
    _sample = f10.loc[
        _low.str.contains("prescriptive", regex=False), "generated_text"
    ].iloc[0]
    mo.vstack(
        [
            mo.md(
                "**Evidence that fiction10 ran under the prescriptive prompt** — its "
                "outputs echo the prescriptive framing (out of 2,993 chunks):"
            ),
            mo.ui.table(pd.DataFrame(_kw_rows), selection=None, pagination=False),
            mo.accordion(
                {
                    "example fiction10 output (March run) — note the framing": mo.md(
                        f"```json\n{_sample[:900]}\n```"
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Disagreement browser

    Chunks fiction10 gated **True** and this run gated **False** — read a few
    to confirm the refusals are prompt-framing, not content, driven.
    """)
    return


@app.cell
def _(mo, paired):
    disagree = (
        paired[
            paired["has_information_exchange_f10"]
            & ~paired["has_information_exchange_t100"]
        ]
        .reset_index(drop=True)
    )
    example_picker = mo.ui.slider(
        start=0, stop=len(disagree) - 1, step=1, value=0, label="disagreement #"
    )
    example_picker
    return disagree, example_picker


@app.cell
def _(disagree, example_picker, mo):
    _row = disagree.iloc[example_picker.value]
    mo.vstack(
        [
            mo.md(
                f"**{_row['book_title']}** — `gutenberg_id={_row['gutenberg_id']}, "
                f"chunk_id={_row['chunk_id']}` · fiction10: **True** · this run: **False**"
            ),
            mo.accordion(
                {
                    "chunk text (first 900 chars)": mo.md(
                        f"> {_row['article_text'][:900].replace(chr(10), ' ')}…"
                    )
                }
            ),
            mo.md(f"**This run's output:**\n```json\n{_row['generated_text'][:1100]}\n```"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Output lengths & the parse errors

    `max_tokens=4096`, thinking off. If the 44 parse errors were truncation,
    they'd sit at the top of the length distribution.
    """)
    return


@app.cell
def _(C_BLUE, C_INK, C_MUTED, np, plt, t100):
    _lens = t100["generated_text"].str.len()
    _err_mask = t100["ci_reasoning_parse_error"].fillna("") != ""

    _fig, _ax = plt.subplots(figsize=(9, 3))
    _bins = np.geomspace(_lens.min(), _lens.max(), 60)
    _ax.hist(_lens[~_err_mask], bins=_bins, color=C_BLUE, edgecolor="white", linewidth=0.3)
    _ax.hist(
        _lens[_err_mask],
        bins=_bins,
        color=C_MUTED,
        edgecolor="white",
        linewidth=0.3,
        label=f"parse errors (n={int(_err_mask.sum())})",
    )
    _ax.set_xscale("log")
    _ax.set_xlabel("generated_text length (chars, log scale)", color=C_MUTED)
    _ax.set_ylabel("chunks", color=C_MUTED)
    _ax.set_title(
        f"output lengths — median {int(_lens.median())} · p95 {int(_lens.quantile(0.95))} "
        f"· max {int(_lens.max())} · parse-error median {int(_lens[_err_mask].median())}",
        color=C_INK,
        fontsize=11,
        loc="left",
    )
    _ax.legend(frameon=False, fontsize=9)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(axis="y", color="#eceae6", linewidth=0.8)
    _ax.set_axisbelow(True)
    _ax.tick_params(colors=C_MUTED)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Downstream viability for job 2

    Applying the plan's own arithmetic (2 extraction prompts per flow chunk,
    +30% vignettes, v11 prescreen keep-rate 0.64, 4 prompts/step) to what this
    artifact actually yields:
    """)
    return


@app.cell
def _(EXCLUDE_BOOKS, mo, pd, t100):
    _usable = t100[~t100["gutenberg_id"].astype(str).isin(EXCLUDE_BOOKS)]
    _n_flow = int(_usable["has_information_exchange"].sum())
    _extraction = _n_flow * 2
    _prescreen = round(_extraction * 1.3)
    _postscreen = round(_prescreen * 0.64)
    _steps = round(_postscreen / 4)

    viability_df = pd.DataFrame(
        [
            {"quantity": "usable flow chunks (after book exclusion)", "planned": "1,400", "actual": f"{_n_flow}"},
            {"quantity": "extraction prompts (×2)", "planned": "~2,800", "actual": f"~{_extraction}"},
            {"quantity": "pre-screen prompts (+30% vignettes)", "planned": "~4,000", "actual": f"~{_prescreen}"},
            {"quantity": "post-screen prompts (×0.64)", "planned": "~2,500", "actual": f"~{_postscreen}"},
            {"quantity": "GRPO steps @ 4 prompts/step", "planned": "~625", "actual": f"~{_steps}"},
            {"quantity": "fiction10 reference (post-screen)", "planned": "—", "actual": "704"},
        ]
    )
    mo.vstack(
        [
            mo.ui.table(viability_df, selection=None, pagination=False),
            mo.callout(
                mo.md(
                    f"**NO-GO for `run_grpo_top100_flows.sh` on this artifact.** "
                    f"~{_postscreen} post-screen prompts is *smaller* than the fiction10 "
                    f"pool (704) this run was designed to escape — it would test nothing "
                    f"in the plan's pre-registered predictions. Do not launch job 2; "
                    f"re-run job 1 first (options below)."
                ),
                kind="danger",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Remediation — RESOLVED 2026-07-12

    The wiring bug is **fixed**: `config.yaml` now sets the active prompt
    keys as body interpolations (pipeline selections finally apply), every
    prompt yaml carries a `name:` provenance field, stages loudly log the
    active prompt and stamp `prompt_name` into their output parquets, and
    `tests/historical_norms/test_prompt_wiring.py` guards the composition
    (verified to fail on the pre-fix config).

    The chosen path (2026-07-12): regenerate ALL gold labels — norms + flows,
    fiction10 + top100 — with a fresh judge, `google/gemma-4-26B-A4B-it`,
    under the fiction prompts. Pipelines:
    `COLM_norms_fiction_prefetched_gemma4` / `COLM_flows_fiction_prefetched_gemma4`,
    launched via `scripts/run_extract_{fiction10,top100}_{norms,flows}_gemma4.sh`.
    Caveat logged in the plan notes: this changes the gate regime relative to
    the v9–v11 lineage (prompt AND judge move together) — the old artifacts
    remain the paper-as-submitted lineage; the regeneration is the corrected
    lineage going forward.

    The provenance fact — *the paper's extraction stages ran with the
    prescriptive-texts prompts while the appendix prints the fiction ones* —
    still belongs in `CONGRUENCE.md` and the camera-ready.
    """)
    return


if __name__ == "__main__":
    app.run()
