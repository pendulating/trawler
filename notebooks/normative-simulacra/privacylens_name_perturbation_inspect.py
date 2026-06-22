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
    # PrivacyLens — cultural name-perturbation inspector

    Eyes-on verification for the `privacylens` cultural-perturbation stage
    (`dagspaces/privacylens/perturb/`). The stage detects person/location
    entities with a RoBERTa-large NER model, then applies a **deterministic,
    gender-preserving, record-seeded** name-bank substitution so each vignette
    is rewritten with names/locations from a target culture while staying
    internally coherent.

    This notebook lets you:

    1. Pick a **culture** and a **record**, and see the vignette **before vs
       after** with the swapped names highlighted (story, trajectory, user
       identity, secrets) plus the exact replacement map.
    2. Inspect the **raw NER entities** that drove the swap (to spot NER
       misses / false positives).
    3. Run **aggregate verification** over the loaded sample: swap coverage,
       JSON-structure integrity of the trajectory, survival of original names,
       and the western identity-passthrough invariant.

    Run with: `marimo edit notebooks/normative-simulacra/privacylens_name_perturbation_inspect.py`
    """)
    return


@app.cell
def _():
    import functools
    import html
    import json
    import re
    import sys
    from pathlib import Path

    import pandas as pd

    PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from dagspaces.privacylens.stages.load_dataset import load_dataset
    from dagspaces.privacylens.perturb import perturb_dataset
    from dagspaces.privacylens.perturb.name_bank import available_cultures
    from dagspaces.privacylens.perturb.ner import detect_entities, get_nlp

    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_colwidth", 200)

    RED = "#ffd6d6"
    GREEN = "#d6f5d6"

    # Load the NER model up-front (first call downloads weights); the whole
    # notebook needs it except the western passthrough.
    try:
        get_nlp()
        NLP_OK = True
        NLP_MSG = "NER model `Jean-Baptiste/roberta-large-ner-english` loaded."
    except Exception as _e:  # pragma: no cover - environment guard
        NLP_OK = False
        NLP_MSG = f"NER model unavailable ({_e})."

    @functools.lru_cache(maxsize=8)
    def load_privacylens(n: int):
        """Load (and cache) a PrivacyLens sample of size n (0 = full 493)."""
        return load_dataset(sample_n=n if n and n > 0 else None)

    def asobj(x):
        """Return S/V cell as a dict (handles dict or JSON-string encodings)."""
        if isinstance(x, dict):
            return x
        try:
            return json.loads(x)
        except Exception:
            return {}

    def items_list(t):
        """sensitive_info_items -> list[str] (handles list / ndarray / scalar)."""
        v = t.get("sensitive_info_items")
        try:
            return [str(x) for x in list(v)]
        except TypeError:
            return [] if v is None else [str(v)]

    def highlight(text, phrases, color):
        """Wrap word-boundary occurrences of `phrases` in a colored mark."""
        if not text:
            return ""
        esc = html.escape(str(text), quote=False)
        phrases = sorted({p for p in phrases if p}, key=len, reverse=True)
        if not phrases:
            return esc.replace("\n", "<br>")
        pat = re.compile(
            r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(p) for p in phrases) + r")(?![A-Za-z0-9_])"
        )
        out = pat.sub(
            lambda m: f"<mark style='background:{color};padding:0 2px;border-radius:3px'>{m.group(1)}</mark>",
            esc,
        )
        return out.replace("\n", "<br>")

    def struct_counts(s):
        """Structural character counts — must be invariant under name swaps."""
        s = str(s or "")
        return tuple(s.count(c) for c in '{}[]"')

    return (
        GREEN,
        NLP_MSG,
        NLP_OK,
        RED,
        asobj,
        available_cultures,
        detect_entities,
        get_nlp,
        highlight,
        items_list,
        json,
        load_privacylens,
        pd,
        perturb_dataset,
        struct_counts,
    )


@app.cell(hide_code=True)
def _(NLP_MSG, NLP_OK, mo):
    mo.md(f"> {'✅' if NLP_OK else '⚠️'} {NLP_MSG}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Controls
    """)
    return


@app.cell
def _(available_cultures, mo):
    cultures = [c for c in available_cultures()]
    culture_ui = mo.ui.dropdown(
        options=cultures,
        value="east_asian" if "east_asian" in cultures else cultures[0],
        label="Culture",
    )
    sample_ui = mo.ui.slider(
        start=10, stop=200, step=10, value=40, label="Records to load (HF sample)"
    )
    mo.hstack([culture_ui, sample_ui], justify="start", gap=2)
    return culture_ui, sample_ui


@app.cell
def _(culture_ui, load_privacylens, mo, perturb_dataset, sample_ui):
    df_base = load_privacylens(int(sample_ui.value))
    df_pert = perturb_dataset(df_base, culture_ui.value)
    _n_changed = int((df_pert["n_persons_swapped"] > 0).sum())
    mo.md(
        f"Loaded **{len(df_base)}** records · culture **{culture_ui.value}** · "
        f"**{_n_changed}** with ≥1 person swapped · "
        f"total persons swapped **{int(df_pert['n_persons_swapped'].sum())}**, "
        f"locations **{int(df_pert['n_locations_swapped'].sum())}**"
    )
    return df_base, df_pert


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Single-record inspection
    """)
    return


@app.cell
def _(df_base, mo):
    _options = {
        f"{i}: rec {rid}": i for i, rid in enumerate(df_base["record_id"].astype(str))
    }
    record_ui = mo.ui.dropdown(options=_options, value=next(iter(_options)), label="Record")
    record_ui
    return (record_ui,)


@app.cell
def _(
    GREEN,
    RED,
    asobj,
    df_base,
    df_pert,
    highlight,
    items_list,
    json,
    mo,
    record_ui,
):
    _idx = int(record_ui.value)
    base_row = df_base.iloc[_idx]
    pert_row = df_pert.iloc[_idx]

    rmap = json.loads(pert_row["perturb_map_json"] or "{}")
    sources = list(rmap.keys())
    targets = list(rmap.values())

    _v0, _v1 = asobj(base_row["V"]), asobj(pert_row["V"])
    t0, t1 = base_row["T"], pert_row["T"]

    _map_rows = "\n".join(f"| `{k}` | `{val}` |" for k, val in rmap.items()) or "| _(none)_ | |"
    _sens_before = "<br>".join(highlight(s, sources, RED) for s in items_list(t0)) or "_(none)_"
    _sens_after = "<br>".join(highlight(s, targets, GREEN) for s in items_list(t1)) or "_(none)_"

    mo.md(
        f"""
    ### Record `{base_row['record_id']}` — culture `{pert_row['culture']}`

    **Replacement map** ({len(rmap)} entries)

    | original | replacement |
    |---|---|
    {_map_rows}

    ---

    #### Story
    <table width="100%"><tr>
    <td width="50%" valign="top"><b>BEFORE</b><br>{highlight(_v0.get('story',''), sources, RED)}</td>
    <td width="50%" valign="top"><b>AFTER</b><br>{highlight(_v1.get('story',''), targets, GREEN)}</td>
    </tr></table>

    #### User identity
    | field | before | after |
    |---|---|---|
    | user_name | {highlight(t0.get('user_name',''), sources, RED)} | {highlight(t1.get('user_name',''), targets, GREEN)} |
    | user_email | {highlight(t0.get('user_email',''), sources, RED)} | {highlight(t1.get('user_email',''), targets, GREEN)} |

    #### Sensitive info items (used by the leakage judge)
    <table width="100%"><tr>
    <td width="50%" valign="top"><b>BEFORE</b><br>{_sens_before}</td>
    <td width="50%" valign="top"><b>AFTER</b><br>{_sens_after}</td>
    </tr></table>
    """
    )
    return base_row, sources, t0, t1, targets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Executable trajectory (before / after)

    The map is applied here too, but NER is **not** run on the trajectory
    (its tool outputs spawn false positives) — names are swapped via the map
    built from the story / secrets / name-slots.
    """)
    return


@app.cell
def _(GREEN, RED, highlight, mo, sources, t0, t1, targets):
    mo.md(
        f"""
    <table width="100%"><tr>
    <td width="50%" valign="top"><b>BEFORE</b><br><div style="font-family:monospace;font-size:11px">{highlight(t0.get('executable_trajectory',''), sources, RED)}</div></td>
    <td width="50%" valign="top"><b>AFTER</b><br><div style="font-family:monospace;font-size:11px">{highlight(t1.get('executable_trajectory',''), targets, GREEN)}</div></td>
    </tr></table>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### NER entities driving this record

    What the detector found per field (PERSON seeds the swap; ORG is
    *protected*; GPE/LOC are swapped via the location bank). Use this to spot
    NER misses (a name in the story not tagged) or false positives (an
    org/word tagged PERSON).
    """)
    return


@app.cell
def _(NLP_OK, asobj, base_row, detect_entities, get_nlp, items_list, mo, pd):
    if not NLP_OK:
        ent_view = mo.md("_NER model not available._")
    else:
        _v = asobj(base_row["V"])
        _t = base_row["T"]
        _fields = {
            "story": _v.get("story", "") or "",
            "user_instruction": str(_t.get("user_instruction", "") or ""),
        }
        for _k, _s in enumerate(items_list(_t)):
            _fields[f"secret[{_k}]"] = _s

        _nlp = get_nlp()
        _rows = []
        for _field, _txt in _fields.items():
            for _e in detect_entities(_txt, _nlp):
                _rows.append({"field": _field, "label": _e.label, "text": _e.text})
        ent_view = (
            pd.DataFrame(_rows, columns=["field", "label", "text"])
            if _rows
            else mo.md("_no entities detected_")
        )
    ent_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Aggregate verification (over the loaded sample)

    Automated invariants. **All checks should pass** for a non-western culture;
    flagged records are listed so you can drill in by selecting them above.
    """)
    return


@app.cell
def _(asobj, df_base, df_pert, items_list, json, mo, struct_counts):
    import re as _re

    _n = len(df_base)
    _zero = []      # records where nothing changed (possible NER miss)
    _broken = []    # trajectory JSON structure altered (must be empty)
    _surv = []      # an original mapped source still present after swap

    for _i in range(_n):
        _b, _p = df_base.iloc[_i], df_pert.iloc[_i]
        _rid = str(_b["record_id"])
        _rmap = json.loads(_p["perturb_map_json"] or "{}")

        if _p["n_persons_swapped"] == 0 and _p["n_locations_swapped"] == 0:
            _zero.append(_rid)

        if struct_counts(_b["T"].get("executable_trajectory")) != struct_counts(
            _p["T"].get("executable_trajectory")
        ):
            _broken.append(_rid)

        _after = " ".join(
            [asobj(_p["V"]).get("story", "") or "", str(_p["T"].get("user_name", "") or "")]
            + items_list(_p["T"])
        )
        for _src in _rmap:
            if _re.search(
                r"(?<![A-Za-z0-9_])" + _re.escape(_src) + r"(?![A-Za-z0-9_])", _after
            ):
                _surv.append(f"{_rid}:{_src}")
                break

    def _fmt(lst, k=15):
        lst = list(lst)
        if not lst:
            return "_none_ ✅"
        shown = ", ".join(str(x) for x in lst[:k])
        more = f" … (+{len(lst) - k} more)" if len(lst) > k else ""
        return f"⚠️ {len(lst)}: {shown}{more}"

    mo.md(
        f"""
    | check | result |
    |---|---|
    | records loaded | {_n} |
    | **trajectory JSON structure preserved** (must be none broken) | {_fmt(_broken)} |
    | **no original mapped name survives the swap** (must be none) | {_fmt(_surv)} |
    | records with **zero swaps** (review for NER misses) | {_fmt(_zero)} |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Full-benchmark audit (all 493 records)

    Click to load the entire benchmark and re-run the structural-integrity and
    western identity-passthrough checks across every record. Slower (one HF
    load + a pass over 493 vignettes).
    """)
    return


@app.cell
def _(mo):
    run_full = mo.ui.run_button(label="Run full-benchmark audit")
    run_full
    return (run_full,)


@app.cell
def _(
    culture_ui,
    load_privacylens,
    mo,
    perturb_dataset,
    run_full,
    struct_counts,
):
    mo.stop(not run_full.value, mo.md("_Press the button above to run._"))

    _full = load_privacylens(0)
    _full_pert = perturb_dataset(_full, culture_ui.value)
    _west = perturb_dataset(_full, "western")

    _broken_n = 0
    for _i in range(len(_full)):
        if struct_counts(_full.iloc[_i]["T"].get("executable_trajectory")) != struct_counts(
            _full_pert.iloc[_i]["T"].get("executable_trajectory")
        ):
            _broken_n += 1

    _west_ok = all(
        _west.iloc[_i][c] == _full.iloc[_i][c]
        for _i in range(len(_full))
        for c in ("S", "V", "T")
    )
    _swapped = int((_full_pert["n_persons_swapped"] > 0).sum())

    mo.md(
        f"""
    **Full benchmark — {len(_full)} records, culture `{culture_ui.value}`**

    | check | result |
    |---|---|
    | records with ≥1 person swapped | {_swapped} / {len(_full)} ({_swapped / len(_full):.0%}) |
    | trajectory JSON structure broken | {'✅ 0' if _broken_n == 0 else f'⚠️ {_broken_n}'} |
    | western passthrough byte-identical (S/V/T) | {'✅ yes' if _west_ok else '⚠️ NO'} |
    | total persons / locations swapped | {int(_full_pert['n_persons_swapped'].sum())} / {int(_full_pert['n_locations_swapped'].sum())} |
    """
    )
    return


if __name__ == "__main__":
    app.run()
