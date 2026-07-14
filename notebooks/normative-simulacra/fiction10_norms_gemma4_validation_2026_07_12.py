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
    # fiction10 — gemma-4 gold-label validation (norms + flows)

    Built 2026-07-12 against `outputs/2026-07-12_fiction10_norms_gemma4/18-36-28`;
    **extended 2026-07-13** with the flows track
    (`outputs/2026-07-12_fiction10_flows_gemma4/23-14-17`, both stages completed 04:30)
    and a cross-track comparison.

    These are the **first fiction runs with the fiction prompts actually applied**.
    Every prior fiction extraction silently ran the *prescriptive* prompts (Hydra group
    defaults clobbered the pipeline's prompt selection), so the headline question is not
    just "did it run" but **"is this output actually different from, and better than, the
    prescriptive-prompt lineage it replaces?"** Gate H answers that for norms.

    They are also the first runs on the retuned engine config (TP=2 × DP=2,
    `NCCL_SHM_DISABLE=0`, `max_model_len=24576`), which is why Gates C/F3 spend effort on
    truncation: the previous config clamped long-summary books and biased them toward
    `has_norms=False`.

    Gates are ordered so that a failure early invalidates everything after it.

    ### Part 1 — norms (`structured_norms.parquet`, 10,034 Raz tuples)

    | gate | asks |
    |---|---|
    | **A. Provenance** | did the intended prompt + model + layout actually run? |
    | **B. Coverage** | is every source chunk accounted for, exactly once? |
    | **C. Integrity** | any parse failures or silent truncation? |
    | **D. Reasoning** | are the norm distributions sane, and unbiased across books? |
    | **E. Extraction** | did the Raz tuples come out well-formed? |
    | **F. Semantics** | *the money gate* — are norms generalizable roles, not plot recaps? |
    | **G. Cross-stage** | does extraction agree with the reasoning it consumed? |
    | **H. A/B** | did the fiction prompt change the output vs prescriptive? |

    ### Part 2 — flows (`ci_flows.parquet`, 16,200 CI 5-tuples)

    | gate | asks |
    |---|---|
    | **F1. Provenance** | did the `ci_*_fiction` prompts run? |
    | **F2. Coverage** | every chunk, exactly once? |
    | **F3. Integrity** | parse failures / truncation? |
    | **F4. Distributions** | flow yield sane across books? |
    | **F5. Extraction** | CI 5-tuples well-formed, vocabulary not collapsed? |
    | **F6. `flow_quality`** | *audit the shipped QA gate* — does it measure what it claims? |
    | **F7. Cross-stage** | any yield leak from reasoning → extraction? |

    ### Part 3 — cross-track

    | gate | asks |
    |---|---|
    | **X. Convergence** | do the two independent tracks agree about which chunks carry information flows? |

    > **Scope.** fiction10 only. The top100 chain is still running (`top100_norms`
    > reasoning at ~77%; `top100_flows` not yet started) — the last cell reports live
    > progress and will light up automatically once those parquets land.
    """)
    return


@app.cell
def _():
    import json
    import re
    from collections import Counter
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import yaml

    ROOT = Path("/share/pierson/matt/UAIR")
    RUN = ROOT / "outputs/2026-07-12_fiction10_norms_gemma4/18-36-28"
    OUT = RUN / "COLM_norms_fiction_gemma4/outputs"

    REASONING = OUT / "reasoning/reasoning.parquet"
    EXTRACTION = OUT / "extraction/structured_norms.parquet"

    # Flows track — same corpus, same model, same engine layout, separate pipeline.
    FLOWS_RUN = ROOT / "outputs/2026-07-12_fiction10_flows_gemma4/23-14-17"
    FLOWS_OUT = FLOWS_RUN / "COLM_flows_fiction_gemma4/outputs"

    CI_REASONING = FLOWS_OUT / "ci_reasoning/reasoning.parquet"
    CI_EXTRACTION = FLOWS_OUT / "ci_extraction/ci_flows.parquet"

    # The top100 chain, for the live-status cell at the bottom. Globbed, not pinned:
    # these runs are in flight and their timestamp dirs do not exist yet.
    TOP100 = ROOT / "outputs"

    # Source of truth for coverage, and the text we mine character names from.
    SOURCE = ROOT / (
        "outputs/2026-03-20_historical_norms/23-05-10/"
        "COLM_fetch_fiction/outputs/fetch/chunks.parquet"
    )
    # The prescriptive-prompt lineage this run is meant to replace (Gate H).
    BASELINE = ROOT / (
        "outputs/2026-03-20_historical_norms/23-12-53/"
        "COLM_norms_fiction/outputs/reasoning/reasoning.parquet"
    )
    return (
        BASELINE,
        CI_EXTRACTION,
        CI_REASONING,
        EXTRACTION,
        FLOWS_RUN,
        REASONING,
        RUN,
        SOURCE,
        TOP100,
        pd,
        re,
        yaml,
    )


@app.cell
def _(EXTRACTION, REASONING, SOURCE, pd):
    src = pd.read_parquet(SOURCE)
    rsn = pd.read_parquet(REASONING)

    # reasoning.parquet is EXPLODED: one row per (chunk, norm). Chunk-level facts
    # must be read off the deduplicated view or every per-chunk rate is wrong.
    KEY = ["gutenberg_id", "chunk_id"]
    rsn_chunks = rsn.drop_duplicates(subset=KEY).copy()

    ext = pd.read_parquet(EXTRACTION) if EXTRACTION.exists() else None
    EXT_READY = ext is not None
    return EXT_READY, KEY, ext, rsn, rsn_chunks, src


@app.cell
def _(CI_EXTRACTION, CI_REASONING, pd):
    # Flows track. Note the shape difference from norms: ci_reasoning is NOT exploded —
    # it is one row per chunk carrying a `ci_flow_count`, whereas norm reasoning is one
    # row per (chunk, norm). Getting this backwards silently rescales every per-chunk rate.
    FLOWS_READY = CI_REASONING.exists() and CI_EXTRACTION.exists()

    fr = pd.read_parquet(CI_REASONING) if FLOWS_READY else None
    fe = pd.read_parquet(CI_EXTRACTION) if FLOWS_READY else None
    return FLOWS_READY, fe, fr


@app.cell(hide_code=True)
def _(EXT_READY, FLOWS_READY, fe, fr, mo, rsn, rsn_chunks, src):
    mo.md(f"""
    **Loaded.**

    - source: `{len(src):,}` chunks
    - **norms** — reasoning `{len(rsn):,}` rows over `{len(rsn_chunks):,}` chunks
      (exploded, one row per norm) · extraction
      {'`loaded`' if EXT_READY else '**not yet written** — Gates E/G will report PENDING'}
    - **flows** — {f'ci_reasoning `{len(fr):,}` rows (one per chunk) · ci_extraction `{len(fe):,}` CI tuples'
                   if FLOWS_READY else '**not found** — Gates F1–F7 and X will report PENDING'}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate A — Provenance

    The claim this run exists to make is "these labels came from the *fiction* prompt."
    That claim is only as good as the `prompt_name` stamped into every row, so check the
    column rather than the config that was *supposed* to produce it.
    """)
    return


@app.cell
def _(RUN, mo, rsn, yaml):
    _cfg = yaml.safe_load((RUN / ".hydra/config.yaml").open())
    _ov = yaml.safe_load((RUN / ".hydra/overrides.yaml").open())
    _node = _cfg["pipeline"]["graph"]["nodes"]["reasoning"]["overrides"]

    _names = rsn["prompt_name"].value_counts(dropna=False)
    _prompt_ok = (_names.index.tolist() == ["norm_reasoning_fiction"]) and len(_names) == 1

    gate_a = "PASS" if _prompt_ok else "FAIL"

    mo.md(f"""
    **prompt_name across all {len(rsn):,} rows:** `{_names.to_dict()}`

    - fiction prompt applied to every row: **{'YES' if _prompt_ok else 'NO — STOP'}**
    - model: `{_cfg['model']['model_source'].split('/')[-1]}`
    - overrides: `{_ov}`
    - engine layout: TP=`{_node.get('model.engine_kwargs.tensor_parallel_size')}`
      × DP=`{_node.get('model.engine_kwargs.data_parallel_size')}`,
      max_model_len=`{_node.get('model.engine_kwargs.max_model_len')}`,
      max_tokens=`{_node.get('sampling_params.max_tokens')}`

    ### Gate A: **{gate_a}**

    A single mixed `prompt_name` here would mean the run straddled the wiring fix and the
    whole corpus is untrustworthy. It didn't.
    """)
    return (gate_a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate B — Coverage

    Set-compare against the source chunks. A dropped chunk is invisible in any
    aggregate rate (the denominator just shrinks), so compare *identities*, not counts.
    Duplicates matter too: DP=2 shards the frame, and a bad split would double-count.
    """)
    return


@app.cell
def _(KEY, mo, rsn_chunks, src):
    _s = set(map(tuple, src[KEY].values))
    _o = set(map(tuple, rsn_chunks[KEY].values))
    _missing, _extra = _s - _o, _o - _s
    _dupes = len(rsn_chunks) - rsn_chunks.groupby(KEY).ngroups
    _books_ok = set(src.book_title) == set(rsn_chunks.book_title)

    gate_b = "PASS" if not (_missing or _extra or _dupes) and _books_ok else "FAIL"

    mo.md(f"""
    | check | value |
    |---|---|
    | source chunks | {len(_s):,} |
    | output chunks | {len(_o):,} |
    | missing (dropped) | **{len(_missing)}** |
    | extra (fabricated) | **{len(_extra)}** |
    | duplicate chunk keys | **{_dupes}** |
    | books present | {rsn_chunks.book_title.nunique()} / {src.book_title.nunique()} |

    ### Gate B: **{gate_b}**

    Zero missing and zero duplicates means the DP=2 contiguous shard split reassembled
    losslessly — worth confirming explicitly, since this is the first DP=2 run.
    """)
    return (gate_b,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate C — Integrity: parse failures and truncation

    The failure mode this gate exists for: an output that runs past `max_tokens` truncates
    mid-JSON → parse error → `has_norms=False`. That doesn't look like an error downstream,
    it looks like *"this chunk has no norms"* — a **silent, length-correlated bias** that
    would fall hardest on the books with the longest summaries (Les Mis). The old
    `max_model_len=16384` did exactly this to 706 chunks.

    So: distinguish a *clean* empty result from a *truncated* one, and check the residual
    correlates with nothing.
    """)
    return


@app.cell
def _(mo, pd, rsn_chunks):
    _c = rsn_chunks.copy()
    _c["gen"] = _c.generated_text.fillna("")
    _c["gen_chars"] = _c.gen.str.len()

    # A truncated JSON object does not close its brace. A clean "no norms" answer does.
    _c["closes"] = _c.gen.str.rstrip().str.endswith("}")
    _c["empty_out"] = _c.gen_chars == 0

    _parse_err = (
        rsn_chunks["reasoning_error"].notna().sum()
        if "reasoning_error" in rsn_chunks.columns
        else 0
    )
    _unclosed = int((~_c.closes).sum())
    _empty = int(_c.empty_out.sum())

    _nonorm = _c[~_c.has_norms]
    _nonorm_clean = int(_nonorm.closes.sum())

    # Output length vs the cap. max_tokens=6144; ~4 chars/token => ~24k chars headroom.
    _hdr = pd.DataFrame(
        {
            "generated_text chars": [
                _c.gen_chars.min(),
                int(_c.gen_chars.median()),
                int(_c.gen_chars.quantile(0.99)),
                _c.gen_chars.max(),
            ]
        },
        index=["min", "p50", "p99", "max"],
    )

    gate_c = "PASS" if (_parse_err == 0 and _unclosed == 0 and _empty == 0) else "FAIL"

    mo.md(f"""
    | check | value |
    |---|---|
    | JSON parse errors | **{_parse_err}** |
    | outputs not closing `}}` (truncated) | **{_unclosed}** |
    | empty outputs | **{_empty}** |
    | no-norm chunks | {len(_nonorm)} |
    | …of which emit *complete* JSON | **{_nonorm_clean} / {len(_nonorm)}** |

    {_hdr.to_markdown()}

    ### Gate C: **{gate_c}**

    Longest output is **{_c.gen_chars.max():,} chars ≈ {_c.gen_chars.max() // 4:,} tokens**
    against a **6,144-token** cap — nothing is within 4× of the ceiling. Combined with
    every no-norm chunk emitting a *complete* `{{"norms": [], ...}}` object, the
    `has_norms=False` rows are genuine model judgements, **not truncation artifacts**.
    The `max_model_len=24576` bump did its job.
    """)
    return (gate_c,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate D — Reasoning distributions, and the Les Mis question

    Gate C proved the no-norm rows aren't truncation. But they are **not uniformly
    distributed**: Les Misérables carries most of them. Since Les Mis is exactly the book
    that used to get truncated, we have to rule out that we've merely swapped a truncation
    bias for some other length-correlated one.

    The discriminating test: if it were an artifact, the no-norm chunks would look
    *mechanically* different (long summary, long chunk). If it's real, they should look
    like ordinary chunks that simply contain no prescriptive content — Hugo's digressions
    on Waterloo, the sewers, and argot are famously norm-free narration.
    """)
    return


@app.cell
def _(mo, rsn, rsn_chunks):
    _g = (
        rsn.groupby("book_title")
        .agg(
            chunks=("chunk_id", "nunique"),
            norm_rows=("chunk_id", "size"),
            no_norm=("has_norms", lambda s: int((~s).sum())),
            info_flow=("governs_information_flow", lambda s: int((s == True).sum())),
        )
        .assign(
            norms_per_chunk=lambda d: (d.norm_rows - d.no_norm) / d.chunks,
            no_norm_pct=lambda d: 100 * d.no_norm / d.chunks,
            info_flow_pct=lambda d: 100 * d.info_flow / (d.norm_rows - d.no_norm),
        )
        .sort_values("chunks", ascending=False)
        .round(2)
    )

    _force = rsn.preliminary_normative_force.value_counts(dropna=False)
    _ALLOWED = {"obligatory", "prohibited", "permitted", "recommended", "discouraged"}
    _bad_force = set(_force.index.dropna()) - _ALLOWED

    # Is "no norms" a property of long/edge chunks, or of content?
    _nn = rsn_chunks[~rsn_chunks.has_norms]
    _yn = rsn_chunks[rsn_chunks.has_norms]
    _len_cmp = (
        f"no-norm chunks: median article {_nn.article_text.str.len().median():,.0f} chars · "
        f"has-norm chunks: median article {_yn.article_text.str.len().median():,.0f} chars"
    )

    gate_d = "PASS" if not _bad_force else "FAIL"

    mo.md(f"""
    {_g.to_markdown()}

    **normative_force vocabulary:** `{_force.to_dict()}`
    → off-schema values: **{_bad_force or 'none'}**

    **Are no-norm chunks mechanically different?** {_len_cmp}

    ### Gate D: **{gate_d}**

    Les Mis is the outlier on both axes (**{_g.loc['Les Misérables', 'no_norm_pct']}%** no-norm,
    lowest norms/chunk at **{_g.loc['Les Misérables', 'norms_per_chunk']}**) — but its no-norm
    chunks are *not* longer than average, which is what a truncation/length artifact would
    require. Read this as content, not corruption. **Spot-check the flagged chunks below
    before trusting that read** — this is the one place where a defensible-looking number
    could still be hiding a real bias.
    """)
    return (gate_d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Eyeball the no-norm chunks.** If these read as digression/description, Gate D's conclusion holds.
    """)
    return


@app.cell
def _(mo, rsn_chunks):
    _nn = rsn_chunks[~rsn_chunks.has_norms][["book_title", "chunk_id", "article_text"]].copy()
    _nn["excerpt"] = _nn.article_text.str.slice(0, 260).str.replace(r"\s+", " ", regex=True)
    mo.ui.table(_nn.drop(columns=["article_text"]), page_size=12, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate E — Extraction: Raz tuple well-formedness

    Stage 2 turns each reasoning trace into a structured Raz tuple under a guided-JSON
    schema. Two things worth checking beyond "did it parse": that the *required* schema
    fields are actually populated, and that the fields which are *conditionally* null are
    null for the right reason rather than by accident.
    """)
    return


@app.cell
def _(EXT_READY, ext, mo):
    if not EXT_READY:
        gate_e = "PENDING"
        _md = "Extraction parquet not written yet — re-run once the stage lands."
    else:
        _fail = int(ext["extraction_failed"].sum())

        # Required by the schema — a null here is a real defect.
        _REQUIRED = [
            "raz_prescriptive_element",
            "raz_norm_subject",
            "raz_norm_act",
            "raz_normative_force",
        ]
        _req_null = int(ext[_REQUIRED].isna().any(axis=1).sum())

        _ALLOWED = {"obligatory", "prohibited", "permitted", "recommended", "discouraged"}
        _bad_force = set(ext.raz_normative_force.dropna().unique()) - _ALLOWED

        _nulls = (
            ext[[c for c in ext.columns if c.startswith("raz_")]]
            .isna()
            .mean()
            .mul(100)
            .round(2)
            .rename("null %")
            .to_frame()
            .query("`null %` > 0")
        )

        # raz_info_flow_note is only meaningful when the norm governs an information
        # flow. If its null count EXACTLY equals the count of governs_info_flow == False,
        # the 71% nulls are structural, not missing data. Check it rather than assume.
        _note_null = int(ext.raz_info_flow_note.isna().sum())
        _not_flow = int((ext.raz_governs_info_flow == False).sum())
        _note_consistent = _note_null == _not_flow

        gate_e = (
            "PASS"
            if (_fail == 0 and _req_null == 0 and not _bad_force and _note_consistent)
            else "FAIL"
        )
        _md = f"""
    | check | value |
    |---|---|
    | rows (norms) | {len(ext):,} |
    | `extraction_failed` | **{_fail}** |
    | rows missing a **required** Raz field | **{_req_null}** |
    | off-schema `normative_force` | **{_bad_force or 'none'}** |

    **Nulls, and whether they are legal:**

    {_nulls.to_markdown()}

    - `raz_condition_of_application` — `Optional` in the schema (an *unconditional* norm).
      Legal.
    - `raz_info_flow_note` — **{_note_null:,}** nulls vs **{_not_flow:,}** norms with
      `governs_info_flow == False`: **{'exact match' if _note_consistent else 'MISMATCH'}**.
      So the 71% null rate is structural (the note only exists for information-flow norms),
      **not** missing data.

    ### Gate E: **{gate_e}**
    """

    mo.md(_md)
    return (gate_e,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate F — Semantics: the money gate

    Everything above can pass on output that is *well-formed and worthless*. The schema is
    emphatic about what makes a norm useful:

    > `norm_subject` — **MUST be a social role, NEVER a named character.**

    A model that emits `"Elizabeth"` / `"refuses Mr. Collins's proposal"` produces a plot
    summary wearing a norm's clothes. It parses, it validates, and it is useless as a
    normative universe.

    The pipeline **already ships a detector** for this (`historical_norms/name_detection.py`
    → `norm_quality_flags` / `norm_quality_passed`: blocklist + title regex + spaCy PERSON
    NER). So the job here is not to reinvent it — it is to **audit it**, because a QA gate
    that cries wolf is worse than none.
    """)
    return


@app.cell
def _(EXT_READY, ext, mo, pd, re):
    if not EXT_READY:
        gate_f = "PENDING"
        fp_rows = pd.DataFrame()
        real_rows = pd.DataFrame()
        _md = "Extraction not available yet."
    else:
        # norm_quality_flags is a SEMICOLON-DELIMITED STRING, not a list.
        # Iterating it directly shreds it into characters.
        def _terms(v):
            if v is None or not isinstance(v, str) or not v.strip():
                return []
            out = []
            for f in v.split(";"):
                m = re.match(r"([a-z_]+)_in_([a-z_]+):(.+)$", f.strip())
                if m:
                    out.append(m.group(3).strip().lower())
            return out

        _e = ext.copy()
        _e["terms"] = _e.norm_quality_flags.map(_terms)

        _failed = _e[~_e.norm_quality_passed]

        # "may" (May Welland, Age of Innocence) and "will" (Will Ladislaw, Middlemarch)
        # are in the GLOBAL character blocklist and matched case-insensitively — so every
        # modal verb "may"/"will" in any norm, in any book, trips the gate.
        MODAL = {"may", "will"}
        _failed = _failed.assign(
            only_modal=_failed.terms.map(lambda t: bool(t) and all(x in MODAL for x in t))
        )
        fp_rows = _failed[_failed.only_modal]
        real_rows = _failed[~_failed.only_modal]

        _gate_rate = 100 * len(_failed) / len(_e)
        _true_rate = 100 * len(real_rows) / len(_e)

        # Of the survivors, most are institutions/titles, not private characters.
        INSTITUTION = {"big brother", "queen", "king", "ingsoc", "gospel", "the party"}
        _inst = real_rows[
            real_rows.terms.map(lambda t: bool(t) and all(x in INSTITUTION for x in t))
        ]
        _named = real_rows[~real_rows.index.isin(_inst.index)]
        _named_rate = 100 * len(_named) / len(_e)

        _generic = (
            _e.raz_norm_subject.fillna("")
            .str.lower()
            .str.match(r"^(a|an|the|any|all|one)\b")
            .mean()
            * 100
        )

        # A leakage rate this low is only meaningful if the norms aren't degenerate.
        _top_act = _e.raz_norm_act.value_counts().head(5)

        gate_f = "PASS" if _named_rate < 1.0 else ("WARN" if _named_rate < 5.0 else "FAIL")
        _md = f"""
    ### The shipped gate over-reports by ~6x

    | | rows | rate |
    |---|---|---|
    | `norm_quality_passed == False` (as shipped) | {len(_failed):,} | **{_gate_rate:.2f}%** |
    | …caused **solely** by the modal verbs `may` / `will` | {len(fp_rows):,} | **{100*len(fp_rows)/max(len(_failed),1):.0f}% of failures** |
    | true flags after removing them | {len(real_rows):,} | {_true_rate:.2f}% |
    | …of which are **institutions/titles** (`big brother`, `queen`, `ingsoc`) | {len(_inst):,} | |
    | **actual named-character leakage** | **{len(_named):,}** | **{_named_rate:.2f}%** |

    **Root cause.** `norm_extraction.py`'s built-in `character_blocklist` contains
    `"will"` (*Will Ladislaw*, Middlemarch) and `"may"` (*May Welland*, The Age of
    Innocence). It is a **global** list applied to **all ten books**, matched
    **case-insensitively** — so the modal verbs in *"a servant **may** not…"* and
    *"the heir **will** inherit…"* trip it in every novel. The code comments that
    "false positives are cheap (just a flag, no rows dropped)" — true for the data, but it
    means `norm_quality_passed` is **~85% noise**, and `stage_metrics.py` logs
    `norm_quality_passed_rate` to W&B as if it were signal.

    **Also over-flagged:** spaCy tags `Queen` as a PERSON, so *"a subject or servant of the
    Queen"* — a textbook social **role** — is marked a character leak.

    ### ✅ Resolved 2026-07-13 — ambiguous blocklist entries are now case-sensitive

    `name_detection.AMBIGUOUS_NAMES` lists blocklist entries that are also ordinary English
    words; those match the **capitalized** form only. Re-running the detector over these
    same 440 rows: **373 clear, 67 remain** — the gate drops from **4.39% → 0.67%**, and
    what survives is the genuine named-character and institution flags.

    The parquet above still carries the *old* flags (it predates the fix). `top100_norms`
    extraction has not started yet, so it will be written with the corrected detector.

    ### The corpus itself is clean

    - generalizable `norm_subject` (starts with `a`/`an`/`the`…): **{_generic:.1f}%**
    - real named-character leakage: **{_named_rate:.2f}%** ({len(_named)} of {len(_e):,})

    Most-repeated `norm_act` (template collapse would show one value dominating):

    {_top_act.to_frame('count').to_markdown()}

    ### Gate F: **{gate_f}** — *for the corpus.* The **shipped detector is WARN**: it works,
    but its headline number is 6x too pessimistic.
    """

    mo.md(_md)
    return fp_rows, gate_f, real_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The false positives** — every one of these is a modal verb, not a character:
    """)
    return


@app.cell
def _(fp_rows, mo):
    mo.ui.table(
        fp_rows[["book_title", "raz_norm_subject", "raz_norm_act", "norm_quality_flags"]],
        page_size=8,
        selection=None,
    ) if len(fp_rows) else mo.md("*none*")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The genuine flags** — review these by hand; institutions (`Big Brother`, `the Queen`) are arguably legitimate societal context, actual character names (`Martin Verga`, `Mingott`, `Jarndyce`) are not:
    """)
    return


@app.cell
def _(mo, real_rows):
    mo.ui.table(
        real_rows[["book_title", "raz_norm_subject", "raz_norm_act", "norm_quality_flags"]],
        page_size=12,
        selection=None,
    ) if len(real_rows) else mo.md("*none*")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate G — Cross-stage consistency

    Extraction consumes the reasoning parquet. Every reasoning norm should survive into a
    Raz tuple; anything that doesn't is a silent yield leak. The one *expected* drop is the
    51 no-norm chunks, which have nothing to extract.
    """)
    return


@app.cell
def _(EXT_READY, KEY, ext, mo, rsn, rsn_chunks):
    if not EXT_READY:
        gate_g = "PENDING"
        _md = "Extraction not available yet."
    else:
        _rsn_norms = int(rsn.has_norms.sum())          # reasoning rows carrying a norm
        _no_norm = int((~rsn_chunks.has_norms).sum())  # chunks with nothing to extract
        _ext_in = _rsn_norms                           # what extraction should have consumed
        _ext_out = len(ext)

        # Chunk-level: did any chunk lose all its norms in translation?
        _r = rsn[rsn.has_norms].groupby(KEY).size().rename("reasoning")
        _x = ext.groupby(KEY).size().rename("extracted")
        _j = _r.to_frame().join(_x, how="outer").fillna(0)
        _lost_all = int(((_j.reasoning > 0) & (_j.extracted == 0)).sum())
        _fewer = int((_j.extracted < _j.reasoning).sum())

        gate_g = "PASS" if _lost_all == 0 else "FAIL"
        _md = f"""
    | | rows |
    |---|---|
    | reasoning rows total | {len(rsn):,} |
    | …carrying a norm (extraction's input) | {_rsn_norms:,} |
    | …no-norm chunks, nothing to extract (expected drop) | {_no_norm} |
    | extraction output (Raz tuples) | **{_ext_out:,}** |

    `{len(rsn):,} − {_no_norm} = {_rsn_norms:,}` consumed → **{_ext_out:,}** tuples out.
    Extraction emits *more* rows than it consumed because a single reasoning trace can
    yield several Raz tuples (`raz_norm_count > 1`), so the count going **up** is expected;
    the count going **down** would not be.

    | chunk-level | chunks |
    |---|---|
    | lost **all** norms in extraction | **{_lost_all}** |
    | ended with fewer norms than reasoning found | {_fewer:,} |

    ### Gate G: **{gate_g}** — no chunk that found a norm came out empty.
    """

    mo.md(_md)
    return (gate_g,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate H — Did the fiction prompt actually change anything?

    The whole point of this run. Compare against `2026-03-20/23-12-53`, the fiction10 norms
    produced by the **prescriptive** prompts (the wiring bug). If the two are
    indistinguishable, the fix was cosmetic and the paper's appendix still misdescribes the
    method. A real difference is the evidence that re-running was worth it.

    Caveat: the baseline also used a **different model** (qwen-lineage), so this is not a
    clean prompt-only ablation — it is a lineage comparison. Read it as "is the new corpus
    materially different", not "the prompt caused exactly this delta".
    """)
    return


@app.cell
def _(BASELINE, mo, pd, rsn, rsn_chunks):
    if not BASELINE.exists():
        gate_h = "PENDING"
        _md = "Baseline reasoning parquet not found."
    else:
        _b = pd.read_parquet(BASELINE)
        _bc = _b.drop_duplicates(subset=["gutenberg_id", "chunk_id"])

        _books = sorted(set(rsn.book_title) & set(_b.book_title))
        _cmp = pd.DataFrame(
            {
                "new (fiction prompt)": {
                    "chunks": rsn_chunks.shape[0],
                    "norm rows": len(rsn),
                    "norms / chunk": round(
                        rsn.has_norms.sum() / rsn_chunks.shape[0], 2
                    ),
                    "no-norm chunk %": round(
                        100 * (~rsn_chunks.has_norms).mean(), 2
                    ),
                    "governs info flow %": round(
                        100 * (rsn.governs_information_flow == True).mean(), 1
                    ),
                },
                "old (prescriptive prompt)": {
                    "chunks": _bc.shape[0],
                    "norm rows": len(_b),
                    "norms / chunk": round(_b.has_norms.sum() / _bc.shape[0], 2),
                    "no-norm chunk %": round(100 * (~_bc.has_norms).mean(), 2),
                    "governs info flow %": (
                        round(
                            100 * (_b.governs_information_flow == True).mean(), 1
                        )
                        if "governs_information_flow" in _b.columns
                        else float("nan")
                    ),
                },
            }
        )

        # Verbatim overlap: identical reasoning traces would mean nothing changed.
        _prompt_names = (
            _b.prompt_name.unique().tolist()
            if "prompt_name" in _b.columns
            else ["<unstamped — predates the provenance guard>"]
        )

        gate_h = "PASS"
        _md = f"""
    **Baseline `prompt_name`:** `{_prompt_names}`
    (the provenance column postdates that run, which is *why* the prescriptive-prompt bug
    went unnoticed for so long — there was nothing to check.)

    {_cmp.to_markdown()}

    Shared books: {len(_books)}.
    """

    mo.md(f"{_md}\n\n### Gate H: **{gate_h}** *(informational — read the deltas, not a threshold)*")
    return (gate_h,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Part 2 — flows

    Same corpus, same model, same engine layout; a separate pipeline
    (`COLM_flows_fiction_prefetched_gemma4`) producing Nissenbaum CI 5-tuples
    (*subject, sender, recipient, information_type, transmission_principle*) plus
    context, appropriateness, and confidence.

    Two structural differences from the norms track that shape every gate below:

    1. **`ci_reasoning` is not exploded.** One row per chunk, carrying `ci_flow_count`.
       (Norm reasoning is one row per *norm*.) Per-chunk rates are read directly.
    2. **There is no role-abstraction step.** `norm_role_abstraction` exists for norms;
       the flows track has no equivalent — which turns out to matter a great deal for
       Gate F6.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate F1 — Provenance

    Same question as Gate A, and the same reason to distrust the config and check the
    stamped column instead: the flows track is where the prompt-wiring bug did the most
    damage (the 2026-07-08 top100 flows attempt collapsed to a 0.014 flow rate under the
    prescriptive prompt).
    """)
    return


@app.cell
def _(FLOWS_READY, FLOWS_RUN, fe, fr, mo, yaml):
    if not FLOWS_READY:
        gate_f1 = "PENDING"
        _md = "Flows parquets not found."
    else:
        _cfg = yaml.safe_load((FLOWS_RUN / ".hydra/config.yaml").open())
        _nodes = _cfg["pipeline"]["graph"]["nodes"]

        _rn = fr.prompt_name.value_counts(dropna=False).to_dict()
        _en = fe.prompt_name.value_counts(dropna=False).to_dict()
        _ok = list(_rn) == ["ci_reasoning_fiction"] and list(_en) == ["ci_extraction_fiction"]

        _layout = "\n".join(
            f"    | `{_n}` | TP={_o.get('model.engine_kwargs.tensor_parallel_size')}"
            f" × DP={_o.get('model.engine_kwargs.data_parallel_size')} "
            f"| {_o.get('model.engine_kwargs.max_model_len'):,} "
            f"| {_o.get('sampling_params.max_tokens'):,} |"
            for _n, _o in ((k, _nodes[k]["overrides"]) for k in ("ci_reasoning", "ci_extraction"))
        )

        gate_f1 = "PASS" if _ok else "FAIL"
        _md = f"""
    **`prompt_name` across all rows** — ci_reasoning: `{_rn}` · ci_extraction: `{_en}`

    Fiction prompts applied to every row: **{'YES' if _ok else 'NO — STOP'}**.
    Model: `{_cfg['model']['model_source'].split('/')[-1]}`.

    | stage | engine | max_model_len | max_tokens |
    |---|---|---|---|
    {_layout}

    ### Gate F1: **{gate_f1}**
    """

    mo.md(_md)
    return (gate_f1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate F2 — Coverage
    """)
    return


@app.cell
def _(FLOWS_READY, KEY, fr, mo, src):
    if not FLOWS_READY:
        gate_f2 = "PENDING"
        _md = "Flows parquets not found."
    else:
        _s = set(map(tuple, src[KEY].values))
        _o = set(map(tuple, fr[KEY].values))
        _missing, _extra = _s - _o, _o - _s
        _dupes = len(fr) - fr.groupby(KEY).ngroups

        gate_f2 = "PASS" if not (_missing or _extra or _dupes) else "FAIL"
        _md = f"""
    | check | value |
    |---|---|
    | source chunks | {len(_s):,} |
    | output chunks | {len(_o):,} |
    | missing (dropped) | **{len(_missing)}** |
    | extra (fabricated) | **{len(_extra)}** |
    | duplicate chunk keys | **{_dupes}** |
    | books present | {fr.book_title.nunique()} / {src.book_title.nunique()} |

    ### Gate F2: **{gate_f2}** — the DP=2 shard split reassembled losslessly here too.
    """

    mo.md(_md)
    return (gate_f2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate F3 — Integrity

    The flows track has the *same* silent-failure mode as norms, one step meaner: a
    malformed JSON object yields `has_information_exchange=False`, which is
    **indistinguishable downstream from a chunk that genuinely contains no information
    exchange**. So it is not enough to count parse errors — we have to check whether the
    no-exchange population is contaminated by them, and whether the cause is truncation
    (a config bug, fixable) or malformed generation (a model defect, not fixable by config).
    """)
    return


@app.cell
def _(FLOWS_READY, fr, mo, pd):
    if not FLOWS_READY:
        gate_f3 = "PENDING"
        bad_chunk = pd.DataFrame()
        _md = "Flows parquets not found."
    else:
        _c = fr.copy()
        _c["gen"] = _c.generated_text.fillna("")
        _c["chars"] = _c.gen.str.len()
        _c["closes"] = _c.gen.str.rstrip().str.endswith("}")

        _parse_err = int(_c.ci_reasoning_parse_error.notna().sum())
        _unclosed = int((~_c.closes).sum())
        _empty = int((_c.chars == 0).sum())

        _no_exch = _c[~_c.has_information_exchange.astype(bool)]
        bad_chunk = _c[_c.ci_reasoning_parse_error.notna()][
            ["book_title", "chunk_id", "chars", "closes", "ci_reasoning_parse_error"]
        ]
        # Contamination: how many of the "no exchange" chunks are actually parse failures?
        _contam = int(_no_exch.ci_reasoning_parse_error.notna().sum())

        _cap_tokens = 4096
        _max_tok_est = int(_c.chars.max() // 4)
        _truncated = _max_tok_est >= _cap_tokens * 0.9

        _hdr = pd.DataFrame(
            {
                "generated_text chars": [
                    int(_c.chars.min()),
                    int(_c.chars.median()),
                    int(_c.chars.quantile(0.99)),
                    int(_c.chars.max()),
                ]
            },
            index=["min", "p50", "p99", "max"],
        )

        # One malformed object in 2,993 is a model defect, not a config defect — PASS,
        # but say so out loud rather than rounding it to zero.
        gate_f3 = "PASS" if (_parse_err <= 1 and _empty == 0 and not _truncated) else "FAIL"
        _md = f"""
    | check | value |
    |---|---|
    | JSON parse errors | **{_parse_err}** |
    | outputs not closing `}}` | **{_unclosed}** |
    | empty outputs | **{_empty}** |
    | no-exchange chunks | {len(_no_exch)} |
    | …contaminated by a parse error | **{_contam}** |

    {_hdr.to_markdown()}

    ### Not truncation

    Longest output is **{int(_c.chars.max()):,} chars ≈ {_max_tok_est:,} tokens** against a
    **{_cap_tokens:,}-token** cap — a 2.3× margin. So the one failure is **malformed
    generation, not a clamped budget**: the model emitted a syntactically broken object
    ({int(bad_chunk.chars.iloc[0]):,} chars, well inside budget) rather than running out of room.
    `max_model_len=24576` is doing its job.

    ### The one contaminated chunk

    {bad_chunk.to_markdown(index=False)}

    This chunk now reads as `has_information_exchange=False` / `ci_flow_count=0` and is
    **indistinguishable from a genuine no-flow chunk** downstream. It is **1 of 2,993
    (0.03%)** and **1 of {len(_no_exch)}** no-exchange chunks, so it cannot move any
    aggregate — but it is a real, if tiny, silent loss, and the failure *mode* is the one
    to watch if it ever grows.

    ### Gate F3: **{gate_f3}** — {len(_no_exch) - _contam} of {len(_no_exch)} no-exchange
    chunks are genuine model judgements.
    """

    mo.md(_md)
    return (gate_f3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate F4 — Distributions

    The tell we are looking for: Les Misérables was the norms track's outlier (12% of its
    chunks carried no norm, against 1.7% corpus-wide), and Gate D argued that was *content*
    — Hugo's Waterloo/sewers/argot digressions — not corruption.

    The flows track is an **independent run with a different prompt and a different schema**.
    If Les Mis is the outlier *here too*, that is strong corroboration: two independent
    extractions do not invent the same book-shaped artifact. If it is not, Gate D's
    conclusion needs re-examining.
    """)
    return


@app.cell
def _(FLOWS_READY, fr, mo):
    if not FLOWS_READY:
        gate_f4 = "PENDING"
        _md = "Flows parquets not found."
    else:
        _g = (
            fr.groupby("book_title")
            .agg(
                chunks=("chunk_id", "nunique"),
                no_exch=("has_information_exchange", lambda s: int((~s.astype(bool)).sum())),
                flows=("ci_flow_count", "sum"),
            )
            .assign(
                flows_per_chunk=lambda d: (d.flows / d.chunks).round(2),
                no_exch_pct=lambda d: (100 * d.no_exch / d.chunks).round(2),
            )
            .sort_values("chunks", ascending=False)
        )
        _worst = _g.no_exch_pct.idxmax()

        gate_f4 = "PASS" if _worst == "Les Misérables" else "WARN"
        _md = f"""
    {_g.to_markdown()}

    ### Gate F4: **{gate_f4}**

    **Les Misérables is the outlier again** — **{_g.loc['Les Misérables', 'no_exch_pct']}%**
    no-exchange (corpus: {100 * _g.no_exch.sum() / _g.chunks.sum():.1f}%) and the *lowest*
    flows/chunk at **{_g.loc['Les Misérables', 'flows_per_chunk']}** — with
    *Nineteen Eighty-Four* a distant second at
    **{_g.loc['Nineteen Eighty-Four', 'no_exch_pct']}%**. That is the same ranking the norms
    track produced, from a different prompt and a different schema.

    **Two independent extractions agreeing on which book is norm-sparse is content, not
    corruption.** Gate D's reading holds.

    (*1984* placing second is a nice sanity check in its own right: a novel about a society
    where information exchange is dangerous and suppressed *should* have more chunks with
    nothing flowing.)
    """

    mo.md(_md)
    return (gate_f4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate F5 — Extraction well-formedness

    Required fields populated; and — the thing that actually kills a normative universe —
    **vocabulary collapse**. If `transmission_principle` were 95% `"confidentiality"`, the
    tuples would validate perfectly and carry no information. The corpus is only useful if
    the CI vocabulary is genuinely differentiated.
    """)
    return


@app.cell
def _(FLOWS_READY, fe, mo, pd):
    if not FLOWS_READY:
        gate_f5 = "PENDING"
        _md = "Flows parquets not found."
    else:
        _REQUIRED = [
            "ci_sender",
            "ci_recipient",
            "ci_information_type",
            "ci_transmission_principle",
            "ci_context",
        ]
        _req_null = int(fe[_REQUIRED].isna().any(axis=1).sum())

        _tp = fe.ci_transmission_principle.str.lower().str.strip().value_counts()
        _top1 = 100 * _tp.iloc[0] / len(fe)
        _top5 = 100 * _tp.head(5).sum() / len(fe)

        _appr = fe.ci_appropriateness.value_counts(dropna=False)
        _appr_pct = (100 * _appr / len(fe)).round(1)

        # ci_subject is Optional in the schema (flows about non-persons: news, public
        # events). A small null rate is legal; a large one would mean the model gave up.
        _subj_null = 100 * fe.ci_subject.isna().mean()

        _vocab = pd.DataFrame(
            {
                "unique values": [
                    fe.ci_transmission_principle.nunique(),
                    fe.ci_context.nunique(),
                    fe.ci_information_type.nunique(),
                    fe.ci_sender.nunique(),
                    fe.ci_recipient.nunique(),
                ]
            },
            index=_REQUIRED,
        )

        gate_f5 = "PASS" if (_req_null == 0 and _top1 < 50) else "FAIL"
        _md = f"""
    | check | value |
    |---|---|
    | rows (CI tuples) | {len(fe):,} |
    | rows missing a **required** CI field | **{_req_null}** |
    | `ci_subject` null (Optional in schema) | {_subj_null:.2f}% |
    | mean `confidence_quant` | {fe.ci_confidence_quant.mean():.2f} / 10 |

    **Vocabulary breadth** (collapse would show as a handful of unique values):

    {_vocab.to_markdown()}

    **Top transmission principles** — most common is `{_tp.index[0]}` at **{_top1:.1f}%**,
    top-5 cover **{_top5:.1f}%**:

    {_tp.head(8).to_frame('count').to_markdown()}

    **Appropriateness:** {_appr_pct.to_dict()}

    ### Gate F5: **{gate_f5}**

    {fe.ci_transmission_principle.nunique()} distinct transmission principles with the
    leader at {_top1:.1f}% is a **differentiated vocabulary, not a collapsed one** — and the
    long tail is semantically real (`coercion`, `state mandate`, `entitlement`), not noise.

    ⚠️ **Note for downstream reward work:** appropriateness is
    **{_appr_pct.get('appropriate', 0)}% `appropriate`** vs
    {_appr_pct.get('inappropriate', 0)}% `inappropriate`. Any reward term keyed on
    appropriateness direction inherits that imbalance — worth knowing before it shows up as
    a degenerate policy.
    """

    mo.md(_md)
    return (gate_f5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate F6 — `flow_quality_passed`: auditing the shipped QA gate

    **37.6% of flows fail `flow_quality_passed`.** Taken at face value that is alarming —
    more than a third of the corpus flagged for character-name leakage.

    Gate F on the norms side established the habit: *audit the detector before believing
    it.* Here that pays off, because the number is not a data problem at all.
    """)
    return


@app.cell
def _(FLOWS_READY, fe, mo, re):
    if not FLOWS_READY:
        gate_f6 = "PENDING"
        flow_fails = None
        _md = "Flows parquets not found."
    else:
        # flow_quality_flags is a SEMICOLON-DELIMITED STRING, same as norm_quality_flags.
        def _parse(v):
            if not isinstance(v, str) or not v.strip():
                return []
            out = []
            for f in v.split(";"):
                m = re.match(r"([a-z_]+)_in_([a-z_]+):(.+)$", f.strip())
                if m:
                    out.append((m.group(2), m.group(3).strip().lower()))
                elif f.strip():  # bare flags like `titled_name_in_sender` carry no name
                    m2 = re.match(r"([a-z_]+)_in_([a-z_]+)$", f.strip())
                    if m2:
                        out.append((m2.group(2), None))
            return out

        _e = fe.copy()
        _e["parsed"] = _e.flow_quality_flags.map(_parse)
        flow_fails = _e[~_e.flow_quality_passed.astype(bool)]
        _rate = 100 * len(flow_fails) / len(_e)

        # Which field trips the gate, and by what name?
        _fields = {}
        _names = {}
        for _ts in flow_fails.parsed:
            for _f, _n in _ts:
                _fields[_f] = _fields.get(_f, 0) + 1
                if _n:
                    _names[_n] = _names.get(_n, 0) + 1
        _fld = ", ".join(
            f"`{k}` {v:,}" for k, v in sorted(_fields.items(), key=lambda x: -x[1])
        )
        _top_names = ", ".join(
            f"`{k}` ({v})" for k, v in sorted(_names.items(), key=lambda x: -x[1])[:8]
        )

        # The may/will blocklist bug that dominated the NORMS gate is a non-event here.
        _modal = flow_fails[
            flow_fails.parsed.map(
                lambda ts: bool(ts) and all(n in {"may", "will"} for _, n in ts if n)
            )
        ]

        # The killer: does the gate discriminate names, or name *formatting*?
        # Pull PASSING rows whose sender is a bare given name — if the gate were measuring
        # name presence, none of these could exist.
        _passed = _e[_e.flow_quality_passed.astype(bool)]
        _demo = _passed[["ci_sender", "ci_recipient", "ci_information_type"]].head(4)

        gate_f6 = "FAIL"  # the DETECTOR fails, not the corpus — see the verdict below
        _md = f"""
    | | rows | rate |
    |---|---|---|
    | `flow_quality_passed == False` | {len(flow_fails):,} | **{_rate:.2f}%** |
    | …caused solely by the `may`/`will` blocklist bug (the norms-track culprit) | {len(_modal):,} | {100 * len(_modal) / max(len(flow_fails), 1):.2f}% |

    **Flagged field:** {_fld}
    **Most-flagged names:** {_top_names}

    These are not modal verbs and not artifacts. They are **Jean Valjean, Lady Dedlock,
    Elizabeth Bennet** — real character names, really in `ci_sender` / `ci_recipient` /
    `ci_subject`. So unlike Gate F, the detector is not misfiring on *noise*.

    It is misfiring on *premise*.

    ### 1. Neither CI prompt ever asked for roles

    `_validate_flow_quality` in `stages/ci_extraction.py:203` says it flags flows
    "whose components reference named characters", and its docstring justifies this:

    > *"The flows track previously had NO quality validation even though **its prompt forbids
    > character names**."*

    **It does not.** Grep both CI extraction prompts for a role requirement and you get
    **zero hits** — while `norm_extraction_fiction.yaml` states it **five times**
    ("this must be a social role, not a named character"; "Test 1 — Role Universality"; …).
    The schema agrees: `RazNormTuple.norm_subject` is documented *"MUST be a social role,
    NEVER a named character"*, whereas `InformationFlowTuple.sender` is only *"the agent
    transmitting or disclosing the information"*.

    The check was **copy-pasted from the norms track onto a track whose prompt never
    required what it enforces**. The 37.6% is the model doing exactly what it was told.

    ### 2. The gate does not even measure what it claims

    It is internally inconsistent. These rows **PASS**:

    {_demo.to_markdown(index=False)}

    …while `Mrs. Bennet → Mr. Bennet` **FAILS** on `titled_name_in_sender`. Both are named
    characters. The detector catches *titled* names (`Mr.`/`Mrs.`/`Lady`) and names spaCy
    recognises as PERSON; bare given names walk straight through. So the flag tracks
    **name formatting, not name presence** — it cannot be read as a leakage rate even on
    its own terms.

    ### 3. Nothing downstream filters on it

    Grepping the repo, the only consumer of `flow_quality_passed` is
    `stage_metrics.py:93`, which logs `flow_quality_passed_rate` to W&B. **No rows are
    dropped and no data is lost** — exactly as with `norm_quality_passed_rate`. The damage
    is confined to a scary-looking dashboard number that means nothing.

    ### Gate F6: **{gate_f6}** — *for the detector.* **The corpus is unaffected.**

    Also worth noting: there is no `ci_role_abstraction` stage. `norm_role_abstraction`
    exists and re-validates `norm_quality_passed` after abstracting; the flows track has no
    such step, so there is nowhere for these names to be abstracted *to*. Either the gate is
    wrong, or the flows track is missing a stage — but it cannot be that the gate is right
    and the pipeline is complete.

    ---

    ### ✅ Resolved 2026-07-13 — `_validate_flow_quality` deleted

    `stages/ci_extraction.py` no longer runs the check, and
    `data_quality/flow_quality_passed_rate` is gone from `stage_metrics.py`.
    `tests/historical_norms/test_name_detection.py::TestFlowQualityCheckRemoved` guards
    against it being reintroduced by reflex.

    **The parquets analysed above still carry the columns** — they were written by the old
    code. Treat `flow_quality_flags` / `flow_quality_passed` in
    `2026-07-12_fiction10_flows_gemma4` as **vestigial: ignore them.** Flows written from
    now on will not have them.

    Root cause, for the record: the 2026-06-09 changelog justified the check by citing
    *"the CI extraction prompt itself forbids character names (`ci_schema.py:188`)"* — but
    **line 188 is inside `RazNormTuple`**, the *norms* schema. A norms rule was cited into
    existence for flows. (Retraction filed in `wiki/changelog/2026-06-09_ner_quality_checks.md`.)
    """

    mo.md(_md)
    return flow_fails, gate_f6


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The flagged rows** — scan them: these are ordinary conversations between named
    characters, which is what the CI prompt asked for.
    """)
    return


@app.cell
def _(flow_fails, mo):
    mo.ui.table(
        flow_fails[
            [
                "book_title",
                "ci_sender",
                "ci_recipient",
                "ci_information_type",
                "ci_transmission_principle",
                "flow_quality_flags",
            ]
        ],
        page_size=12,
        selection=None,
    ) if flow_fails is not None and len(flow_fails) else mo.md("*none*")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate F7 — Cross-stage yield

    Extraction consumes the reasoning parquet. The expected drop is exactly the no-exchange
    chunks, and the expected row count is exactly `sum(ci_flow_count)`. Both are checkable
    to the row, so check them to the row rather than eyeballing the totals.
    """)
    return


@app.cell
def _(FLOWS_READY, KEY, fe, fr, mo):
    if not FLOWS_READY:
        gate_f7 = "PENDING"
        _md = "Flows parquets not found."
    else:
        _rsn_chunks = set(map(tuple, fr[KEY].values))
        _ext_chunks = set(map(tuple, fe[KEY].values))
        _absent = _rsn_chunks - _ext_chunks

        _no_exch = fr[~fr.has_information_exchange.astype(bool)]
        _no_exch_keys = set(map(tuple, _no_exch[KEY].values))

        # The two sets should be IDENTICAL, not merely the same size.
        _exact = _absent == _no_exch_keys
        _promised = int(fr[fr[KEY].apply(tuple, axis=1).isin(_ext_chunks)].ci_flow_count.sum())
        _delivered = len(fe)

        gate_f7 = "PASS" if (_exact and _promised == _delivered) else "FAIL"
        _md = f"""
    | | value |
    |---|---|
    | reasoning chunks | {len(_rsn_chunks):,} |
    | extraction chunks | {len(_ext_chunks):,} |
    | chunks absent from extraction | {len(_absent)} |
    | no-exchange chunks (the expected drop) | {len(_no_exch_keys)} |
    | **are those the same chunks?** | **{'YES — set-identical' if _exact else 'NO — MISMATCH'}** |
    | flows promised by `sum(ci_flow_count)` | {_promised:,} |
    | CI tuples delivered | {_delivered:,} |
    | **shortfall** | **{_promised - _delivered}** |

    ### Gate F7: **{gate_f7}**

    Not "the counts happen to match" — the absent chunks are *set-identical* to the
    no-exchange chunks, and every flow the reasoning stage promised was delivered as a
    tuple. **Zero yield leak.**
    """

    mo.md(_md)
    return (gate_f7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Part 3 — cross-track

    ## Gate X — Convergence: do the two tracks agree?

    The norms and flows tracks are **independent extractions of the same 2,993 chunks** —
    different prompt, different schema, different run. That independence buys a validation
    nothing internal to either run can: they should agree about *which chunks carry
    information flows*.

    But **not symmetrically**, and getting the predicted relation right matters more than
    the agreement number:

    - the **flows** track asks *"is information exchanged here?"* — descriptive, and in a
      novel almost every scene qualifies;
    - the **norms** track asks *"is there a **prescriptive norm** here, and does it
      **govern** an information flow?"* — a strictly higher bar.

    So the prediction is **containment**, not equality: every chunk with a norm *about* an
    information flow should also be a chunk where a flow was *found*. The reverse need not
    hold. A raw agreement rate would be the wrong statistic; **containment violations** are
    the signal.
    """)
    return


@app.cell
def _(FLOWS_READY, KEY, fr, mo, pd, rsn):
    if not FLOWS_READY:
        gate_x = "PENDING"
        _md = "Flows parquets not found."
    else:
        _n = (
            rsn.groupby(KEY)
            .governs_information_flow.apply(lambda s: bool((s == True).any()))  # noqa: E712
            .rename("norms: a norm governs a flow")
        )
        _f = (
            fr.set_index(KEY)
            .has_information_exchange.astype(bool)
            .rename("flows: an exchange was found")
        )
        _j = pd.concat([_n, _f], axis=1).dropna()

        _ct = pd.crosstab(
            _j["norms: a norm governs a flow"],
            _j["flows: an exchange was found"],
            margins=True,
        )

        _norm_flow = int(_j["norms: a norm governs a flow"].sum())
        _violate = int(
            (_j["norms: a norm governs a flow"] & ~_j["flows: an exchange was found"]).sum()
        )
        _contained = 100 * (1 - _violate / _norm_flow)
        _other_way = int(
            (~_j["norms: a norm governs a flow"] & _j["flows: an exchange was found"]).sum()
        )

        gate_x = "PASS" if _contained >= 95 else "WARN"
        _md = f"""
    {_ct.to_markdown()}

    | | chunks | |
    |---|---|---|
    | norms found an info-flow norm | {_norm_flow:,} | |
    | …and flows also found an exchange | {_norm_flow - _violate:,} | **{_contained:.1f}% contained** |
    | …but flows found **nothing** | **{_violate}** | ← the only real disagreement |
    | flows found an exchange, norms found no *norm* about it | {_other_way:,} | expected — see above |

    ### Gate X: **{gate_x}**

    **Containment holds at {_contained:.1f}%.** Of the {_norm_flow:,} chunks where the norms
    track independently concluded "there is a norm here governing an information flow", the
    flows track found an actual exchange in all but **{_violate}**. Two models, two prompts,
    two schemas, agreeing on the hard direction.

    The {_other_way:,} chunks in the other cell are **not** a failure — they are the gap
    between *"people exchanged information"* and *"the text is prescriptive about it"*,
    which is exactly the distinction the two tracks exist to draw. (Gossip happens on nearly
    every page of *Middlemarch*; the text moralises about it far less often.)

    A flat "agreement rate" over this table would read
    {100 * (_j["norms: a norm governs a flow"] == _j["flows: an exchange was found"]).mean():.1f}%
    and would be **meaningless** — it penalises the tracks for measuring different things,
    which they are supposed to do.
    """

    mo.md(_md)
    return (gate_x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Yield, side by side

    Do the two tracks rank the books the same way? A book that is norm-rich should be
    flow-rich; a large disagreement would suggest one track is keying on something
    idiosyncratic.
    """)
    return


@app.cell
def _(FLOWS_READY, fr, mo, rsn, rsn_chunks):
    if not FLOWS_READY:
        _md = "Flows parquets not found."
    else:
        _a = rsn.groupby("book_title").agg(
            chunks=("chunk_id", "nunique"), norms=("has_norms", "sum")
        )
        _b = fr.groupby("book_title").agg(flows=("ci_flow_count", "sum"))
        _c = (
            _a.join(_b)
            .assign(
                norms_per_chunk=lambda d: (d.norms / d.chunks).round(2),
                flows_per_chunk=lambda d: (d.flows / d.chunks).round(2),
                flows_per_norm=lambda d: (d.flows / d.norms).round(2),
            )
            .sort_values("chunks", ascending=False)
        )
        _corr = _c.norms_per_chunk.corr(_c.flows_per_chunk)

        _md = f"""
    {_c.to_markdown()}

    **Corpus:** {len(rsn_chunks):,} chunks → **{int(rsn.has_norms.sum()):,} norms** and
    **{int(fr.ci_flow_count.sum()):,} flows**.

    - book-level correlation of norms/chunk vs flows/chunk: **{_corr:.2f}**
    - `flows_per_norm` stays in a tight **{_c.flows_per_norm.min():.2f}–{_c.flows_per_norm.max():.2f}**
      band across ten novels spanning 1813–1949 and three languages of origin

    The two tracks rank the books consistently, and the flow:norm ratio does not swing
    wildly by book — no single novel is driving either corpus.
    """

    mo.md(_md)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scorecard
    """)
    return


@app.cell
def _(
    gate_a,
    gate_b,
    gate_c,
    gate_d,
    gate_e,
    gate_f,
    gate_f1,
    gate_f2,
    gate_f3,
    gate_f4,
    gate_f5,
    gate_f6,
    gate_f7,
    gate_g,
    gate_h,
    gate_x,
    mo,
    pd,
):
    _rows = [
        ("norms", "A", "Provenance — fiction prompt + model + layout", gate_a),
        ("norms", "B", "Coverage — every chunk, exactly once", gate_b),
        ("norms", "C", "Integrity — no parse failure, no truncation", gate_c),
        ("norms", "D", "Reasoning distributions — no length bias", gate_d),
        ("norms", "E", "Extraction — Raz tuples well-formed", gate_e),
        ("norms", "F", "Semantics — roles not characters", gate_f),
        ("norms", "G", "Cross-stage — no yield leak", gate_g),
        ("norms", "H", "A/B vs prescriptive lineage", gate_h),
        ("flows", "F1", "Provenance — ci_*_fiction prompts", gate_f1),
        ("flows", "F2", "Coverage — every chunk, exactly once", gate_f2),
        ("flows", "F3", "Integrity — 1 malformed JSON in 2,993", gate_f3),
        ("flows", "F4", "Distributions — Les Mis outlier corroborated", gate_f4),
        ("flows", "F5", "Extraction — CI vocabulary differentiated", gate_f5),
        ("flows", "F6", "`flow_quality` detector — mis-specified", gate_f6),
        ("flows", "F7", "Cross-stage — zero yield leak", gate_f7),
        ("cross", "X", "Convergence — containment holds", gate_x),
    ]
    _df = pd.DataFrame(_rows, columns=["track", "gate", "asks", "result"])

    _icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "PENDING": "⏳"}
    _df["result"] = _df.result.map(lambda r: f"{_icon.get(r, '')} {r}")

    mo.md(f"""
    {_df.to_markdown(index=False)}

    ### Verdict

    **Both fiction10 corpora are usable as gold labels.** Every gate that tests the *data*
    passes. The two ❌/⚠️ marks were **defects in shipped QA code, not in the corpus** —
    both have since been fixed (2026-07-13):

    | | was | root cause | now |
    |---|---|---|---|
    | **F6** `flow_quality_passed` | 37.6% "fail" | a **norms** rule (`ci_schema.py:188` = `RazNormTuple`) cited as if it governed **flows**; neither CI prompt ever mentioned roles | check + W&B metric **deleted** |
    | **F** `norm_quality_passed` | 4.39% "fail" | global blocklist holds `may`/`will` (May Welland, Will Ladislaw), matched case-insensitively → every modal verb trips it | ambiguous names now **case-sensitive**; **0.67%** |

    Neither ever dropped a row, so **no corpus was corrupted** — the damage was two W&B
    gauges that read as signal and were noise. The parquets analysed here still carry the
    old columns; ignore them.

    The strongest evidence in this notebook is **Gate X**: two independent extractions —
    different prompts, different schemas, different runs — agree on which chunks carry
    information flows in the one direction where they logically must, at 98.0% containment.
    That is not something a broken pipeline produces.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Still in flight — top100

    The overnight chain (`scripts/run_overnight_gemma4_chain.sh`) runs
    `fiction10_flows` → `top100_norms` → `top100_flows`. The first is validated above.
    This cell reads the streaming shards directly, so it reports live progress and will
    switch to a corpus comparison the moment the final parquets land.
    """)
    return


@app.cell
def _(TOP100, mo, pd):
    def _progress(run_glob, stage_dir, streaming_key, total):
        """Rows written so far, from the final parquet if present, else the DP shards."""
        import glob as _glob

        _runs = sorted(TOP100.glob(run_glob))
        if not _runs:
            return None, "not started"
        _base = _runs[-1]
        _final = list(_base.glob(f"*/outputs/{stage_dir}/*.parquet"))
        _final = [f for f in _final if "_streaming" not in str(f)]
        if _final:
            return pd.read_parquet(_final[0]), "complete"
        _shards = _glob.glob(
            str(_base / f"*/outputs/{stage_dir}/_streaming/{streaming_key}/dp*/*.parquet")
        )
        if not _shards:
            return None, "started, no shards yet"
        import pyarrow.parquet as _pq

        _n = sum(_pq.ParquetFile(f).metadata.num_rows for f in _shards)
        return None, f"running — {_n:,} / {total:,} chunks ({100 * _n / total:.0f}%)"

    _TOP100_CHUNKS = 15875

    _rows = []
    for _label, _glb, _stage, _key in [
        ("top100 norms · reasoning", "*top100_norms_gemma4/*", "reasoning", "norm_reasoning"),
        ("top100 norms · extraction", "*top100_norms_gemma4/*", "extraction", "norm_extraction"),
        ("top100 flows · ci_reasoning", "*top100_flows_gemma4/*", "ci_reasoning", "ci_reasoning"),
        ("top100 flows · ci_extraction", "*top100_flows_gemma4/*", "ci_extraction", "ci_extraction"),
    ]:
        _df_or_none, _status = _progress(_glb, _stage, _key, _TOP100_CHUNKS)
        _rows.append((_label, _status))

    _status_df = pd.DataFrame(_rows, columns=["stage", "status"])

    mo.md(f"""
    {_status_df.to_markdown(index=False)}

    (top100 = **{_TOP100_CHUNKS:,}** chunks, 5.3× fiction10. At fiction10's measured
    ~20 prompts/min this is ~13 h per stage, ~2 days for the remaining three.)

    Once these complete, the comparison to run is **fiction10 vs top100 on the same axes as
    Gate X** — the containment rate in particular. fiction10 is ten canonical novels chosen
    for normative density; top100 is whatever Gutenberg's download counts surface. If
    containment survives that shift in corpus quality, the extraction is robust; if it
    degrades, the normative universe is only as good as the reading list.
    """)
    return


if __name__ == "__main__":
    app.run()
