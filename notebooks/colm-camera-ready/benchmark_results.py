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
    # COLM camera-ready — main benchmark-results table

    Successor to
    `notebooks/normative-simulacra/colm_benchmark_results_gemma4_2026_07.py`,
    rebuilt for the camera-ready after the **2026-07-21 parity reviews**
    (CIRL-729 swap, PrivacyLens, GoldCoin, ConfAIde, VLM-GeoPrivacy — see
    `wiki/changelog/2026-07-21_*`). Same architecture and provenance
    discipline: restrictive SWEEP_GLOBS, judge verified per cell from the
    judge-batch **manifest** (never from config — the sweeps' `multirun.yaml`
    still carries the stale `${oc.env:JUDGE_MODEL,Qwen3.6-27B}` default),
    per-cell latest-run-wins, and a provenance table written next to the
    markdown so every number traces to a metrics.json on disk.

    ### What changed vs the 2026-07 notebook (all camera-ready-blocking)

    1. **CIRL was swapped (2026-07-21); the canonical CIRL-729 re-run has
       LANDED.** The old `cirl_vignettes` dagspace was
       PrivacyLens-under-CIRL-protocol (493 rows, rejection accuracy) — NOT
       the CIRL benchmark. The `cirl` dagspace now runs the real **CIRL-729**
       action task (deterministic substring leakage/utility, judge-free;
       `wiki/changelog/2026-07-21_cirl_benchmark_swap.md`). The old
       notebook's `cirl_vignettes … accuracy` column is the retired metric
       and is **not read here at all**. The CIRL columns read
       `cirl/cirl/outputs/compute_metrics/metrics.json` →
       `leakage.leakage_rate` / `utility.utility_rate` / `net_score` (keys
       per `dagspaces/eval_all/primary_metrics.py`), sourced from the
       2026-07-22 canonical sweep (`*_eval_cirl729_canonical`, 22 cells) +
       the 07-23 requeue of its five failed arms + the teacher sweep. Per
       the paper protocol (Matt's 2026-07-22 ruling: strict-format misses
       speak to the benchmark, not the model), an action missing the strict
       `</think>`+`<answer>` format scores **−1** — recorded via
       `+runtime.allow_unreliable_metrics=true`, never dropped. So **Net
       always fills**, while **Lk↓/Util are rates conditional on
       strict-parseable actions** and are blanked ("—") whenever fewer than
       half of the 729 actions parse — a "leakage rate" over 17/729 rows is
       not a rate. That bar keeps Lk↓/Util only for the three Gemma cells
       per condition, Phi-4 zero-shot, and the teacher; every cell's
       `parseable=n/729` is recorded in the provenance `semantics` column.
    2. **PrivacyLens judged columns are stale (parser corruption).** The
       helpfulness/leakage judge-response parsers scanned free-text FIRST on
       guided-JSON responses (landed 2026-04-26), mis-scoring **21.5% of
       helpfulness judgments** on a real canonical cell (mean 2.345→1.859);
       leakage had 4/1114 per-secret flips; adjusted leakage inherits the
       helpfulness corruption. Fixed 2026-07-21, but **no re-finalize has
       been run** — every PL metrics.json in these sweeps has
       mtime ≤ 2026-07-18. The notebook detects staleness per file (mtime
       vs the fix date) and marks affected cells **‡**; if a rescue
       re-finalize regenerates a metrics.json in place, the flag clears
       automatically on re-run. Additionally, protocol fixes F3 (tool pin
       restored) + F4 (judges no longer see `[Thought]`) mean keeper-era PL
       rows are not comparable with any post-2026-07-21 re-run. Judged-rate
       keys follow the parity review's primary variants
       (`*_among_parseable`, per `primary_metrics.py`), not the
       `*_overall_with_default_zero` audit variants the old notebook quoted.
    3. **GoldCoin headline flipped to the upstream forced-wrong
       denominator.** Upstream never drops an unparseable response — it
       substitutes a deterministically wrong label. Our pre-2026-07-21
       metrics dropped them, inflating accuracy for weak-format models
       (gemma-4-E2B applicability 0.399→0.285; gpt-oss compliance
       0.756→0.636). All GoldCoin files in these sweeps are **pre-flip**;
       the parity review's exact retro-conversion
       `accuracy_upstream = accuracy_old × parseable_rate` is applied here
       (both factors read from the same metrics.json — nothing is
       fabricated), and the provenance table records the conversion per
       cell. Post-flip files (detected via `accuracy_among_parseable`) are
       used natively, so pre/post semantics can never silently mix.
       **Columns are upstream accuracy now, not macro-F1** — macro-F1 has
       no exact retro-conversion. Also: GoldCoin runs at temp 0.2 and a
       same-model re-run moved Appl. by 0.9pt on 2/214 flips — sub-1pt
       GoldCoin gaps are noise.
    4. **ConfAIde `r` stays Tier-2b Pearson** (`compute_metrics_tier2b` →
       `pearson_r`). The 2026-07-21 ConfAIde review changed tier-3 headline
       semantics and the eval_all summary keys, none of which touch this
       column.
    5. **VLM-GeoPrivacy parity review: clean.** Q7 accuracy unchanged.

    ### Excluded sweeps (deliberate — do not add without reading their yaml)

    - `*eval_judgefree_variance*` — the judge-free variance record (now
      complete, 163 arms). It is a **noise-floor instrument**, not a results
      source: repeated seeds per cell, and not the canonical protocol run.
    - `*_eval_harc_confaide_tokenfix` — diagnostic run. It established that
      harc-llama3.1-8b's remaining ConfAIde unparsed rows are explicit
      refusals (31/31 on tier2b), so the cell stays **deliberately blank**
      rather than computing r over the self-selected 68% it agreed to rate.
    - `*_eval_sft_per_checkpoint*` (07-19/20) — evaluate post-2026-07-18
      SFT checkpoints trained under the new template + DFT protocol; **not
      protocol-comparable** with the keeper-era `sft-canonical` rows here.

    ### Where each metric lives

    | Paper col | Benchmark dir | metrics.json subdir | dotted key | judged? |
    |---|---|---|---|---|
    | Appl. / Comp. | `goldcoin/goldcoin_hipaa` | `compute_metrics_{applicability,compliance}` | `accuracy` (×`parseable_rate` retro-conv on pre-flip files) | no |
    | QA Acc | `privacylens/privacylens_eval` | `compute_metrics` | `qa_probing.accuracy` | no |
    | Adj Lk ↓ | ″ | ″ | `adjusted_leakage.adjusted_leakage_rate` | **yes** |
    | Helpful | ″ | ″ | `helpfulness.helpful_rate_among_parseable` | **yes** |
    | r | `confaide/confaide` | `compute_metrics_tier2b` | `pearson_r` | no |
    | CIRL Lk↓ / Util / Net | `cirl/cirl` | `compute_metrics` | `leakage.leakage_rate` / `utility.utility_rate` / `net_score` | no |
    | Q7 | `vlm_geoprivacy/vlm_geoprivacy_bench` | `compute_metrics` | `per_question.Q7.accuracy` | no |
    | MMLU | `mmlu/mmlu` | `compute_metrics` | `overall_accuracy` | no |

    ### Known gaps carried over from the canonical sweeps (findings, not bugs)

    - **`harc-llama3.1-8b/instruct` ConfAIde r** — blank: 31/31 remaining
      tier2b unparsed rows are explicit refusals (see tokenfix note above).
    - **`openthinker3-7b/sft-canonical` PrivacyLens** — blank: emits JSON
      instead of ReAct `Thought:/Action:` on 493/493 vignettes; zero
      parseable actions, deterministic at temp 0.
    - **`gpt-oss-20b/sft-canonical` PrivacyLens** — blank: empty harmony
      final channel on 33.3% of QA probes (channel discipline, not
      truncation). Not repaired by reading the reasoning channel — that
      re-introduces scoring the model's own CoT.
    - **VLM Q7** blank for genuinely text-only models (Phi-4, Llama,
      OpenThinker, GPT-OSS). gemma-4-E2B/E4B Q7 values are single-class
      collapse base rates, not skill (E2B 782/783 one class).
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import re
    import datetime as dt

    MULTIRUN_GLOB_ROOT = Path("/share/pierson/matt/UAIR/multirun")
    REPORT_DIR = Path(__file__).resolve().parent / "tables" / "benchmark_results"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # The Gemma-4-judged canonical-set sweeps (unchanged from the 2026-07
    # notebook — they remain the canonical keeper-era corpus). Restricting
    # the scan to these keeps a stray older eval_all from silently supplying
    # a Qwen3.6-judged cell.
    #
    # DELIBERATELY EXCLUDED (see the header cell for the full rationale):
    #   *eval_judgefree_variance*        noise-floor instrument (complete,
    #                                    163 arms of repeated seeds) — not
    #                                    the canonical protocol run.
    #   *_eval_harc_confaide_tokenfix    diagnostic; its finding is that the
    #                                    harc ConfAIde cell stays blank.
    #   *_eval_sft_per_checkpoint*       post-07-18 SFT protocol (new template
    #                                    + DFT) — not comparable with the
    #                                    keeper-era sft-canonical rows.
    #   *_eval_12b_family_postreview     ditto (new-protocol 12B ckpt series
    #                                    for the longitudinal notebook).
    SWEEP_GLOBS = [
        "*_eval_canonical_instruct/*",
        "*_eval_canonical_sft_gemma4/*",
        # 2026-07-18 repair/extension sweeps. All Gemma-4-judged (or
        # judge-free), composing with the originals under per-cell
        # latest-wins.
        "*_eval_canonical_repair/*",  # llama3.1-8b GoldCoin (cancelled),
        # harc-llama3.1-8b ConfAIde (parse fail)
        "*_eval_canonical_gptoss_refix/*",  # gpt-oss SFT, after the enforce_eager fix
        "*_eval_gemma4_q7_backfill/*",  # VLM Q7 for the six gemma-4 cells
        "*_eval_teacher_gemma4_31b/*",  # the teacher/judge as a subject
        # 2026-07-22/23 canonical CIRL-729 re-run (post-swap `cirl`
        # dagspace, keeper-era model set, paper-protocol -1s via the
        # strict-format escape hatch). These sweeps contain ONLY the cirl
        # benchmark, so they cannot supersede any other column.
        "*_eval_cirl729_canonical/*",  # 22 canonical cells + 07-23 requeue of 5 failed arms
        "*_eval_cirl729_teacher/*",  # the teacher on CIRL-729
    ]

    # Judge every judged cell must have come from. Matched as a substring of
    # the served-model path recorded in the judge-batch manifest.
    EXPECTED_JUDGE = "Gemma-4-31B-it"

    # Multiruns whose judge CANNOT be verified from artifacts, but was
    # attested out-of-band. Attest only from a live observation, never from
    # config — config lies (the stale `${oc.env:JUDGE_MODEL,Qwen3.6-27B}`
    # default).
    JUDGE_ATTESTED_MULTIRUNS = {
        "2026-07-18_eval_teacher_gemma4_31b/10-04-59": "curl http://klara.tech.cornell.edu:8002/v1/models immediately "
        "before launch (2026-07-18) returned exactly one served model: "
        "/share/pierson/matt/zoo/models/Gemma-4-31B-it. The sweep pins "
        "JUDGE_SERVER_URL to that host and the server ran uninterrupted "
        "(job 920636) across the run.",
    }

    # PrivacyLens judge-response parser fix landed 2026-07-21 (parity review
    # F1). Any PL metrics.json written BEFORE this instant was finalized with
    # the corrupting free-text-first parser and its judged metrics are stale.
    # A rescue re-finalize (re-parse of the raw judge output.jsonl, no GPU)
    # would regenerate the file in place and clear the flag on the next run
    # of this notebook.
    PL_PARSER_FIX_DT = dt.datetime(2026, 7, 21, 0, 0, 0)

    # Which Tier-2 sub-metric backs the ConfAIde "r" column — Tier-2b
    # Pearson, same as the paper (Tier-2a is a different, much higher number).
    CONFAIDE_R_SUBDIR = "compute_metrics_tier2b"

    # The teacher/judge model, evaluated as a subject. Its label carries the
    # warning inline so the row cannot be copied out of the table without it.
    TEACHER_ROW = "Gemma-4-31B-it (teacher/judge — self-judged)"

    # ── Rows: the canonical 11, in size-then-family order ─────────────────
    ROW_ORDER = [
        ("Qwen3.5-2B", ["Zero-shot", "SFT"]),
        ("Qwen3.5-4B", ["Zero-shot", "SFT"]),
        ("Qwen3.5-9B", ["Zero-shot", "SFT"]),
        ("Gemma-4-E2B", ["Zero-shot", "SFT"]),
        ("Gemma-4-E4B", ["Zero-shot", "SFT"]),
        ("Gemma-4-12B", ["Zero-shot", "SFT"]),
        ("OpenThinker3-7B", ["Zero-shot", "SFT"]),
        ("Llama-3.1-8B", ["Zero-shot", "SFT"]),
        ("HARC-Llama-3.1-8B", ["Zero-shot", "SFT"]),
        ("Phi-4", ["Zero-shot", "SFT"]),
        ("GPT-OSS-20B", ["Zero-shot", "SFT"]),
        # Reference ceiling, NOT one of the canonical 11 and not a paired
        # Zero-shot/SFT contrast — this is the teacher that generated the
        # SFT data and the judge that scores every PrivacyLens row.
        (TEACHER_ROW, ["Reference"]),
    ]

    # ── model= override string  →  (display model, condition) ─────────────
    # "Zero-shot" = <family>/instruct, verified (2026-07-17 sweep yaml
    # header) to be the exact pre-SFT weights for the paired
    # <family>/sft-canonical adapter.
    _FAMILIES = [
        ("qwen3.5-2b", "Qwen3.5-2B"),
        ("qwen3.5-4b", "Qwen3.5-4B"),
        ("qwen3.5-9b", "Qwen3.5-9B"),
        ("gemma-4-e2b", "Gemma-4-E2B"),
        ("gemma-4-e4b", "Gemma-4-E4B"),
        ("gemma-4-12b", "Gemma-4-12B"),
        ("openthinker3-7b", "OpenThinker3-7B"),
        ("llama3.1-8b", "Llama-3.1-8B"),
        ("harc-llama3.1-8b", "HARC-Llama-3.1-8B"),
        ("phi-4", "Phi-4"),
        ("gpt-oss-20b", "GPT-OSS-20B"),
    ]
    OVERRIDE_TO_ROW = {}
    for _slug, _disp in _FAMILIES:
        OVERRIDE_TO_ROW[f"{_slug}/instruct"] = (_disp, "Zero-shot")
        OVERRIDE_TO_ROW[f"{_slug}/sft-canonical"] = (_disp, "SFT")
    OVERRIDE_TO_ROW["gemma-4-31b/instruct"] = (TEACHER_ROW, "Reference")

    # Judged columns for THIS row are self-judged (judge == subject) — an
    # optimistic bound, not a like-for-like score.
    SELF_JUDGED_ROWS = {TEACHER_ROW}

    # ── Column registry: paper column → where to read it ──────────────────
    # Fields: group, col, bench_dir, inner, subdir, key, judged,
    #         lower_is_better, scale ("pct" = ×100, "raw" = as-is), kind.
    # kind selects the reader:
    #   "plain"    — dotted key, no extra handling
    #   "gc_acc"   — GoldCoin upstream accuracy: detect pre/post the
    #                2026-07-21 denominator flip and retro-convert pre-flip
    #                files exactly (accuracy × parseable_rate)
    #   "pl_stale" — PL judged metric: flag cells finalized before the
    #                2026-07-21 parser fix as stale (‡)
    COLUMNS = [
        (
            "GoldCoin",
            "Appl.",
            "goldcoin",
            "goldcoin_hipaa",
            "compute_metrics_applicability",
            "accuracy",
            False,
            False,
            "pct",
            "gc_acc",
        ),
        (
            "GoldCoin",
            "Comp.",
            "goldcoin",
            "goldcoin_hipaa",
            "compute_metrics_compliance",
            "accuracy",
            False,
            False,
            "pct",
            "gc_acc",
        ),
        (
            "PrivacyLens",
            "QA Acc",
            "privacylens",
            "privacylens_eval",
            "compute_metrics",
            "qa_probing.accuracy",
            False,
            False,
            "pct",
            "plain",
        ),
        (
            "PrivacyLens",
            "Adj Lk↓",
            "privacylens",
            "privacylens_eval",
            "compute_metrics",
            "adjusted_leakage.adjusted_leakage_rate",
            True,
            True,
            "pct",
            "pl_stale",
        ),
        (
            "PrivacyLens",
            "Helpful",
            "privacylens",
            "privacylens_eval",
            "compute_metrics",
            "helpfulness.helpful_rate_among_parseable",
            True,
            False,
            "pct",
            "pl_stale",
        ),
        (
            "ConfAIde",
            "r",
            "confaide",
            "confaide",
            CONFAIDE_R_SUBDIR,
            "pearson_r",
            False,
            False,
            "pct",
            "plain",
        ),
        # CIRL-729 (post-swap dagspace `cirl`). NOT the retired
        # cirl_vignettes rejection accuracy. Lk/Util are conditional on
        # strict-parseable actions → kind "cirl_cond" suppresses them below
        # majority-parseable; Net includes the paper-protocol -1s → kind
        # "cirl_net" always reads, recording parseable=n/729 in semantics.
        (
            "CIRL",
            "Lk↓",
            "cirl",
            "cirl",
            "compute_metrics",
            "leakage.leakage_rate",
            False,
            True,
            "pct",
            "cirl_cond",
        ),
        (
            "CIRL",
            "Util",
            "cirl",
            "cirl",
            "compute_metrics",
            "utility.utility_rate",
            False,
            False,
            "pct",
            "cirl_cond",
        ),
        (
            "CIRL",
            "Net",
            "cirl",
            "cirl",
            "compute_metrics",
            "net_score",
            False,
            False,
            "raw",
            "cirl_net",
        ),
        (
            "VLM",
            "Q7",
            "vlm_geoprivacy",
            "vlm_geoprivacy_bench",
            "compute_metrics",
            "per_question.Q7.accuracy",
            False,
            False,
            "pct",
            "plain",
        ),
        (
            "MMLU",
            "Acc",
            "mmlu",
            "mmlu",
            "compute_metrics",
            "overall_accuracy",
            False,
            False,
            "pct",
            "plain",
        ),
    ]

    # Guard: the retired PrivacyLens-under-CIRL-protocol metric must never
    # be readable as a "CIRL" column again.
    assert not any(c[2] == "cirl_vignettes" for c in COLUMNS), (
        "cirl_vignettes is the RETIRED PrivacyLens-under-CIRL-protocol "
        "dagspace — its accuracy is not a CIRL-729 metric."
    )

    # Column groups expected to have NO data yet: rendered "pend." instead
    # of "—", with an explicit footnote. Empty since the canonical CIRL-729
    # re-run landed (2026-07-22/23) — kept as a mechanism for any future
    # benchmark addition that outpaces its runs.
    PENDING_GROUPS = {}
    return (
        COLUMNS,
        EXPECTED_JUDGE,
        JUDGE_ATTESTED_MULTIRUNS,
        MULTIRUN_GLOB_ROOT,
        OVERRIDE_TO_ROW,
        PENDING_GROUPS,
        PL_PARSER_FIX_DT,
        REPORT_DIR,
        ROW_ORDER,
        SELF_JUDGED_ROWS,
        SWEEP_GLOBS,
        TEACHER_ROW,
        dt,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase A — Scan the canonical-set multiruns

    Walk every in-scope `multirun/<sweep>/<HH-MM-SS>/<idx>/` sub-run. For
    each, read the `model=` override (→ row identity) and the **served judge
    from the judge-batch manifest**, then record one row per
    (sub-run × benchmark column) with the metric value, its **semantics**
    (GoldCoin retro-conversion, PL parser staleness) and provenance.
    """)
    return


@app.cell
def _(
    COLUMNS,
    EXPECTED_JUDGE,
    JUDGE_ATTESTED_MULTIRUNS,
    MULTIRUN_GLOB_ROOT,
    OVERRIDE_TO_ROW,
    PL_PARSER_FIX_DT,
    SWEEP_GLOBS,
    dt,
    re,
):
    import json as _json

    # Extracts only the <date>/<time> stamp — it must NOT also gate on the
    # sweep name. SWEEP_GLOBS above is the single place that decides which
    # sweeps are in scope.
    _MR_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_\w+/(\d{2}-\d{2}-\d{2})$")

    def _parse_mr_dt(mr_dir):
        m = _MR_RE.search(str(mr_dir))
        if not m:
            return None
        return dt.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H-%M-%S")

    def _served_judge(sub_dir):
        """The judge that actually served, from the PrivacyLens judge-batch
        manifest (`judge_export.py` writes the model it resolved from
        /v1/models). Deliberately NOT read from multirun.yaml (stale
        OmegaConf default). Returns None when no manifest exists — callers
        must treat None as "unverified", not "wrong judge"; see
        JUDGE_ATTESTED_MULTIRUNS.
        """
        for man in sorted(
            sub_dir.glob(
                "privacylens/privacylens_eval/outputs/*_judge_batch/manifest.json"
            )
        ):
            try:
                m = _json.loads(man.read_text()).get("model")
            except (ValueError, OSError):
                continue
            if m:
                return m.rstrip("/").split("/")[-1]
        return None

    def _override_model(sub_dir):
        # The model override lives in any benchmark's .hydra/overrides.yaml.
        for ov in sub_dir.glob("*/.hydra/overrides.yaml"):
            for line in ov.read_text(errors="ignore").splitlines():
                line = line.strip().lstrip("- ").strip()
                if line.startswith("model="):
                    return line.split("=", 1)[1]
        return None

    def _dotted(d, path):
        cur = d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    def _read_metric(mp, key, kind):
        """Read one metric. Returns (value, semantics, stale) or None.

        kind "gc_acc": GoldCoin upstream-parity accuracy. Post-flip files
          (>= 2026-07-21, marked by the `accuracy_among_parseable` key /
          forced-wrong provenance) already carry the upstream forced-wrong
          denominator — use `accuracy` natively. Pre-flip files dropped
          unparseable rows; the parity review's EXACT retro-conversion is
          `accuracy_upstream = accuracy_dropsem × parseable_rate` (the
          substitution never produces a correct prediction). Both factors
          come from the same metrics.json.

        kind "pl_stale": PL judged metric; stale ⇔ the metrics.json was
          finalized before the 2026-07-21 parser fix (file mtime).

        kinds "cirl_cond"/"cirl_net": CIRL-729. Net includes the
          paper-protocol -1 for every strict-format miss, so it always
          reads; Lk/Util are rates conditional on strict-parseable actions
          and are SUPPRESSED (returned as missing, logged) when fewer than
          half of the 729 actions parse. Both record parseable=n/total in
          semantics.
        """
        try:
            data = _json.loads(mp.read_text())
        except (ValueError, OSError):
            return None
        if kind in ("cirl_cond", "cirl_net"):
            val = _dotted(data, key)
            if val is None:
                return None
            _p, _t = data.get("parseable"), data.get("total")
            if kind == "cirl_cond" and _p is not None and _t and _p < _t // 2:
                _cirl_suppressed.append((str(mp), key, f"{_p}/{_t}"))
                return None
            return float(val), f"cirl:parseable={_p}/{_t}", False
        if kind == "gc_acc":
            acc = data.get("accuracy")
            pr = data.get("parseable_rate")
            if acc is None:
                return None
            if "accuracy_among_parseable" in data:
                return float(acc), "goldcoin:forced_wrong_native", False
            if pr is None:
                return None  # cannot establish semantics — refuse the cell
            return (
                float(acc) * float(pr),
                f"goldcoin:retro_converted_acc_x_parseable(pr={float(pr):.4f})",
                False,
            )
        val = _dotted(data, key)
        if val is None:
            return None
        if kind == "pl_stale":
            mtime = dt.datetime.fromtimestamp(mp.stat().st_mtime)
            stale = mtime < PL_PARSER_FIX_DT
            sem = (
                "privacylens:pre_parserfix_2026-07-21"
                if stale
                else "privacylens:post_parserfix"
            )
            return float(val), sem, stale
        return float(val), "", False

    rows = []
    _cirl_suppressed = []  # (metrics_path, key, parseable/total) — reported below
    _mr_dirs = sorted({p for g in SWEEP_GLOBS for p in MULTIRUN_GLOB_ROOT.glob(g)})
    for _mr in _mr_dirs:
        if not _mr.is_dir():
            continue
        _mrdt = _parse_mr_dt(_mr)
        if _mrdt is None:
            continue
        for _sub in sorted(p for p in _mr.iterdir() if p.is_dir() and p.name.isdigit()):
            _ov = _override_model(_sub)
            if _ov is None or _ov not in OVERRIDE_TO_ROW:
                continue
            _model, _cond = OVERRIDE_TO_ROW[_ov]
            _mr_key = str(_mr.relative_to(MULTIRUN_GLOB_ROOT))
            _judge = _served_judge(_sub)
            if _judge is None and _mr_key in JUDGE_ATTESTED_MULTIRUNS:
                _judge, _judge_src = EXPECTED_JUDGE, "attested"
            elif _judge is None:
                _judge_src = "unresolved"
            else:
                _judge_src = "manifest"
            for (
                _grp,
                _col,
                _bd,
                _inner,
                _subdir,
                _key,
                _judged,
                _lo,
                _scale,
                _kind,
            ) in COLUMNS:
                _mp = _sub / _bd / _inner / "outputs" / _subdir / "metrics.json"
                if not _mp.exists():
                    continue
                _res = _read_metric(_mp, _key, _kind)
                if _res is None:
                    continue
                _val, _sem, _stale = _res
                rows.append(
                    {
                        "model": _model,
                        "condition": _cond,
                        "group": _grp,
                        "col": _col,
                        "col_id": f"{_grp}::{_col}",
                        "value": _val,
                        "judged": _judged,
                        "judge": _judge,
                        "judge_src": _judge_src,
                        "semantics": _sem,
                        "stale": _stale,
                        "mr_dt": _mrdt,
                        "multirun": _mr_key,
                        "override": _ov,
                    }
                )

    import pandas as _pd

    scan = _pd.DataFrame(rows)
    print(
        f"{len(scan)} (sub-run × column) observations across "
        f"{scan['multirun'].nunique()} multiruns"
    )
    print(f"multiruns: {sorted(scan['multirun'].unique())}")
    print(f"judges seen: {sorted(scan['judge'].dropna().unique())}")
    print(f"judge provenance: {scan.groupby('judge_src').size().to_dict()}")
    _sems = scan[scan["semantics"] != ""].groupby("semantics").size()
    print(f"metric semantics:\n{_sems.to_string()}")

    # Loud CIRL status. The canonical re-run landed 2026-07-22/23 — Net must
    # cover every cell; a zero here means the cirl729 sweeps fell out of
    # scope (a regression, not a pending state).
    _n_cirl_net = int((scan["col_id"] == "CIRL::Net").sum())
    if _n_cirl_net == 0:
        print(
            "\n!! CIRL-729: NO observations — the canonical re-run exists "
            "(*_eval_cirl729_canonical / _teacher); check SWEEP_GLOBS."
        )
    else:
        print(f"\nCIRL-729: {_n_cirl_net} Net observations found.")
    if _cirl_suppressed:
        print(
            f"CIRL-729: {len(_cirl_suppressed)} conditional Lk/Util reads "
            "SUPPRESSED (<50% of 729 actions strict-parseable — the paper-"
            "protocol -1s live in Net; a conditional rate over a sliver of "
            "rows is not a rate):"
        )
        for _mpath, _mkey, _frac in sorted(_cirl_suppressed):
            print(f"   {_frac:>8}  {_mkey:<25} {_mpath.split('multirun/')[-1]}")

    # Stale-PL summary — these are the parser-corruption cells (‡).
    _n_stale = int(scan["stale"].sum())
    if _n_stale:
        print(
            f"\n!! {_n_stale} PrivacyLens judged observations were "
            "finalized BEFORE the 2026-07-21 parser fix (helpfulness "
            "~21.5% row corruption; adjusted leakage inherits it; leakage "
            "~0.4% of secrets). Marked ‡ in the table. Rescue: re-finalize "
            "from the raw judge output.jsonl (no GPU), which regenerates "
            "metrics.json in place and clears this flag."
        )

    # Judged observations with an unresolvable judge get dropped downstream
    # — say so loudly, a silent drop looks like "never ran".
    _unres = scan[scan["judged"] & (scan["judge_src"] == "unresolved")]
    if len(_unres):
        print(
            f"\n!! {len(_unres)} JUDGED observations have an unresolvable "
            f"judge and will be DROPPED (a provenance gap, not a data gap):"
        )
        for (_m, _c, _mr_), _g in _unres.groupby(["model", "condition", "multirun"]):
            print(f"   {_m} / {_c}  [{_mr_}]  cols={sorted(_g['col'])}")
        print(
            "   Fix: add the multirun to JUDGE_ATTESTED_MULTIRUNS with "
            "evidence, or re-run under a batch-mode pipeline that writes "
            "a judge manifest."
        )
    scan
    return (scan,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase B — Judge assertion + per-cell latest-run selection

    Every judged cell must come from a `Gemma-4-31B-it` run; the assertion
    below is expected to be a no-op and exists so a re-run against a
    differently-judged sweep fails loudly instead of silently mixing judges.
    Then, for each (model, condition, column), take the most recent
    qualifying multirun.
    """)
    return


@app.cell
def _(EXPECTED_JUDGE, scan):
    # Loud check: any judged observation not from the expected judge is
    # dropped AND reported. With the current corpus this should print nothing.
    _judged = scan[scan["judged"]]
    _bad = _judged[~_judged["judge"].fillna("").str.contains(EXPECTED_JUDGE)]
    if len(_bad):
        print(
            f"!! {len(_bad)} judged observations NOT judged by "
            f"{EXPECTED_JUDGE} — dropped. Judges: "
            f"{sorted(_bad['judge'].fillna('<none>').unique())}"
        )
        print(_bad[["model", "condition", "col", "judge", "multirun"]].to_string())
    else:
        print(f"OK: all {len(_judged)} judged observations came from {EXPECTED_JUDGE}")

    _elig = scan[
        (~scan["judged"]) | (scan["judge"].fillna("").str.contains(EXPECTED_JUDGE))
    ].copy()
    # Latest run per (model, condition, col) — repair sweeps supersede the
    # originals cell-by-cell rather than wholesale.
    picked = (
        _elig.sort_values("mr_dt")
        .drop_duplicates(subset=["model", "condition", "col_id"], keep="last")
        .copy()
    )
    print(f"{len(picked)} cells filled")
    picked[
        [
            "model",
            "condition",
            "group",
            "col",
            "value",
            "judge",
            "semantics",
            "stale",
            "multirun",
        ]
    ].sort_values(["model", "condition", "col"])
    return (picked,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase B2 — Coverage: which cells are missing, and why

    A blank cell in the table below is never a zero. "—" marks the
    documented model-level findings (refusals, format collapse), text-only
    Q7 cells, and CIRL Lk/Util rates suppressed for <50% strict-parseable
    (their paper-protocol score lives in CIRL Net, which fills everywhere).
    """)
    return


@app.cell
def _(COLUMNS, PENDING_GROUPS, ROW_ORDER, picked):
    import pandas as _pd

    _have = {(r["model"], r["condition"], r["col_id"]) for _, r in picked.iterrows()}
    _col_ids = [(f"{c[0]}::{c[1]}", f"{c[0]} {c[1]}", c[0]) for c in COLUMNS]

    _grid = []
    for _mdl, _conds in ROW_ORDER:
        for _cond in _conds:
            _row = {"model": _mdl, "condition": _cond}
            _n = 0
            for _cid, _label, _grp in _col_ids:
                _ok = (_mdl, _cond, _cid) in _have
                if _ok:
                    _row[_label] = "✓"
                elif _grp in PENDING_GROUPS:
                    _row[_label] = "pend."
                else:
                    _row[_label] = "—"
                _n += _ok
            _row["filled"] = f"{_n}/{len(_col_ids)}"
            _grid.append(_row)
    coverage = _pd.DataFrame(_grid)

    _missing = [
        f"{r['model']}/{r['condition']}" for r in _grid if r["filled"].startswith("0/")
    ]
    if _missing:
        print(f"Rows with NO data at all: {', '.join(_missing)}")
    coverage
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase C — Assemble the table + markdown

    Lay the cells out in row/column order, format (percentages ×100; CIRL
    Net on −1…1), bold the best per column, mark
    self-judged (†) and stale-parser (‡) cells, render, and save to
    `benchmark_results.md` plus a provenance parquet.
    """)
    return


@app.cell
def _(
    COLUMNS,
    EXPECTED_JUDGE,
    PENDING_GROUPS,
    REPORT_DIR,
    ROW_ORDER,
    SELF_JUDGED_ROWS,
    picked,
):
    import pandas as _pd

    # specs: (col_id, group, col, judged, lower_better, scale)
    _col_specs = [(f"{c[0]}::{c[1]}", c[0], c[1], c[6], c[7], c[8]) for c in COLUMNS]

    _lut = {
        (r["model"], r["condition"], r["col_id"]): r["value"]
        for _, r in picked.iterrows()
    }
    _stale_lut = {
        (r["model"], r["condition"], r["col_id"]): bool(r["stale"])
        for _, r in picked.iterrows()
    }
    _prov = {
        (r["model"], r["condition"], r["col_id"]): (
            r["judge"],
            r["judge_src"],
            r["multirun"],
            r["semantics"],
            bool(r["stale"]),
        )
        for _, r in picked.iterrows()
    }

    def _scaled(cid, scale):
        out = {}
        for mdl, conds in ROW_ORDER:
            for cond in conds:
                # The teacher is a reference ceiling (and self-judged on
                # judged columns) — excluded from best-per-column.
                if mdl in SELF_JUDGED_ROWS:
                    continue
                v = _lut.get((mdl, cond, cid))
                if v is None:
                    continue
                out[(mdl, cond)] = v * 100.0 if scale == "pct" else v
        return out

    _best = {}
    for cid, grp, col, judged, lo, scale in _col_specs:
        vals = _scaled(cid, scale)
        if not vals:
            continue
        _best[cid] = (min if lo else max)(vals.values())

    def _fmt(cid, grp, judged, lo, scale, mdl, cond):
        v = _lut.get((mdl, cond, cid))
        if v is None:
            # Pending benchmark groups are awaiting a re-run, not a gap in
            # a benchmark that ran.
            return "pend." if grp in PENDING_GROUPS else "—"
        sv = v * 100.0 if scale == "pct" else v
        spec = "{:.1f}" if scale == "pct" else "{:.2f}"
        s = spec.format(sv)
        if cid in _best and abs(sv - _best[cid]) < 1e-9:
            s = f"**{s}**"
        # † self-judged; ‡ finalized before the 2026-07-21 PL parser fix.
        if judged and mdl in SELF_JUDGED_ROWS:
            s = f"{s}†"
        if _stale_lut.get((mdl, cond, cid)):
            s = f"{s}‡"
        return s

    _hdr_cols = [f"{grp} {col}" for (cid, grp, col, judged, lo, scale) in _col_specs]

    _lines = []
    _lines.append("| Model | Cond. | " + " | ".join(_hdr_cols) + " |")
    _lines.append("|" + "---|" * (2 + len(_col_specs)))
    _dropped = []
    for mdl, conds in ROW_ORDER:
        _shown = 0
        for cond in conds:
            _cells = [
                _fmt(cid, grp, judged, lo, scale, mdl, cond)
                for (cid, grp, col, judged, lo, scale) in _col_specs
            ]
            # Drop (model, condition) rows with no data at all rather than
            # emitting an all-blank row sourced from nothing.
            if all(c in ("—", "pend.") for c in _cells):
                _dropped.append(f"{mdl} / {cond}")
                continue
            _mcell = mdl if _shown == 0 else ""
            _lines.append(f"| {_mcell} | {cond} | " + " | ".join(_cells) + " |")
            _shown += 1
    table_md = "\n".join(_lines)
    if _dropped:
        print("Dropped (no data for any column): " + ", ".join(_dropped))

    _legend = (
        "\n\n*Percentages (×100) except **CIRL Net** (utility − leakage, "
        "−1…1). ↓ = lower is better. "
        "Best per column in **bold**. "
        "GoldCoin Appl./Comp. = upstream-parity accuracy (unparseable "
        "counted as wrong, per the 2026-07-21 denominator flip); pre-flip "
        "metrics files are retro-converted exactly as "
        "accuracy × parseable_rate — see the provenance table. Sub-1pt "
        "GoldCoin gaps are re-run noise (temp 0.2, measured ±0.9pt). "
        "PrivacyLens: QA Acc, Adj Lk = adjusted leakage, Helpful = helpful "
        "rate (both judged, the `*_among_parseable` primary variants). "
        "ConfAIde r = Tier-2b Pearson. CIRL-729: Net = utility − leakage "
        "with every strict-format miss scored −1 per the paper protocol "
        "(recorded, not dropped); Lk/Util are rates conditional on "
        "strict-parseable actions and are blanked when fewer than half of "
        "the 729 actions parse (per-cell parseable fraction in the "
        "provenance `semantics` column). VLM Q7 = location-granularity "
        "accuracy (text-only models blank; gemma-4-E2B/E4B values are "
        "single-class-collapse base rates, not skill). "
        f"All judged columns use {EXPECTED_JUDGE}. Zero-shot = the pre-SFT "
        "`<family>/instruct` checkpoint each SFT adapter was trained from. "
        "Most recent qualifying run per cell.*"
        "\n\n*† = **self-judged**: judge and subject are the same weights — "
        "an optimistic bound, excluded from best-per-column bolding.*"
        "\n\n*‡ = **stale (pre-parser-fix)**: finalized before the "
        "2026-07-21 PrivacyLens judge-response parser fix, which corrupted "
        "~21.5% of helpfulness judgments (true 3s dragged down; mean_score "
        "2.345→1.859 on a measured cell), ~0.4% of leakage secrets, and "
        "adjusted leakage via its helpfulness dependence. Rescue = "
        "re-finalize from the raw judge output.jsonl. Additionally, the "
        "2026-07-21 protocol fixes (tool pin restored; judges no longer "
        "see [Thought]) mean these keeper-era PrivacyLens rows are NOT "
        "comparable with any post-2026-07-21 re-run.*"
    )
    _pending_note = "".join(
        f"\n\n*pend. = {note}.*" for note in PENDING_GROUPS.values()
    )
    _dropped_note = (
        f"\n\n*Rows omitted (no run produced any metric): {', '.join(_dropped)}.*"
        if _dropped
        else ""
    )
    full_md = (
        f"## COLM camera-ready benchmark results — canonical set, "
        f"{EXPECTED_JUDGE}-judged\n\n"
        + table_md
        + _legend
        + _pending_note
        + _dropped_note
        + "\n"
    )

    (REPORT_DIR / "benchmark_results.md").write_text(full_md)

    # Provenance: judge, source multirun, and metric semantics per cell.
    _prov_rows = []
    for mdl, conds in ROW_ORDER:
        for cond in conds:
            for cid, grp, col, judged, lo, scale in _col_specs:
                if (mdl, cond, cid) in _prov:
                    j, jsrc, mr, sem, stale = _prov[(mdl, cond, cid)]
                    _prov_rows.append(
                        {
                            "model": mdl,
                            "condition": cond,
                            "column": f"{grp} {col}",
                            "value": _lut[(mdl, cond, cid)],
                            "judge": j,
                            "judge_src": jsrc,
                            "multirun": mr,
                            "semantics": sem,
                            "stale_parser": stale,
                            "self_judged": bool(judged and mdl in SELF_JUDGED_ROWS),
                        }
                    )
    provenance = _pd.DataFrame(_prov_rows)
    provenance.to_parquet(
        REPORT_DIR / "benchmark_results_provenance.parquet", index=False
    )
    print(f"saved table + provenance to {REPORT_DIR}")
    print(full_md)
    return full_md, provenance


@app.cell
def _(full_md, mo):
    mo.md(full_md)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase C2 — LaTeX table

    The same cells rendered as a booktabs `tabular` for direct inclusion in the
    camera-ready (requires the `booktabs` and `graphicx` packages). Best per
    column is bold; the markers ($\\dagger$ self-judged, $\\ddagger$ stale
    pre-parser-fix) match the markdown legend. Saved to
    `tables/benchmark_results/benchmark_results.tex`.
    """)
    return


@app.cell
def _(
    COLUMNS,
    EXPECTED_JUDGE,
    PENDING_GROUPS,
    REPORT_DIR,
    ROW_ORDER,
    SELF_JUDGED_ROWS,
    picked,
):
    # Cell-local names are underscore-prefixed per marimo convention (shared
    # names must be unique across cells). specs: (col_id, group, col, judged,
    # lower_better, scale).
    _col_specs = [
        (f"{_c[0]}::{_c[1]}", _c[0], _c[1], _c[6], _c[7], _c[8]) for _c in COLUMNS
    ]
    _lut = {
        (_r["model"], _r["condition"], _r["col_id"]): _r["value"]
        for _, _r in picked.iterrows()
    }
    _stale_lut = {
        (_r["model"], _r["condition"], _r["col_id"]): bool(_r["stale"])
        for _, _r in picked.iterrows()
    }

    def _scaled(_cid, _scale):
        _out = {}
        for _mdl, _conds in ROW_ORDER:
            for _cond in _conds:
                # Teacher is a reference ceiling (self-judged): excluded from
                # best-per-column, same as the markdown table.
                if _mdl in SELF_JUDGED_ROWS:
                    continue
                _v = _lut.get((_mdl, _cond, _cid))
                if _v is None:
                    continue
                _out[(_mdl, _cond)] = _v * 100.0 if _scale == "pct" else _v
        return _out

    _best = {}
    for _cid, _grp, _col, _judged, _lo, _scale in _col_specs:
        _vals = _scaled(_cid, _scale)
        if _vals:
            _best[_cid] = (min if _lo else max)(_vals.values())

    def _esc(_s):
        # LaTeX-safe model cell: escape underscores; the teacher row carries a
        # unicode em dash, which we render as an en-dash range.
        return _s.replace("_", r"\_").replace("\u2014", "--")

    def _fmt_tex(_cid, _grp, _judged, _lo, _scale, _mdl, _cond):
        _v = _lut.get((_mdl, _cond, _cid))
        if _v is None:
            return r"\textit{pend.}" if _grp in PENDING_GROUPS else "--"
        _sv = _v * 100.0 if _scale == "pct" else _v
        _spec = "{:.1f}" if _scale == "pct" else "{:.2f}"
        _s = _spec.format(_sv)
        if _cid in _best and abs(_sv - _best[_cid]) < 1e-9:
            _s = rf"\textbf{{{_s}}}"
        if _judged and _mdl in SELF_JUDGED_ROWS:
            _s = _s + r"$^{\dagger}$"
        if _stale_lut.get((_mdl, _cond, _cid)):
            _s = _s + r"$^{\ddagger}$"
        return _s

    # Benchmark groups in column order, with their column counts (for the
    # multicolumn header + cmidrules).
    _groups = []
    for _cid, _grp, _col, _judged, _lo, _scale in _col_specs:
        if _groups and _groups[-1][0] == _grp:
            _groups[-1][1] += 1
        else:
            _groups.append([_grp, 1])

    _colspec = "@{}ll" + "c" * len(_col_specs) + "@{}"

    _L = []
    _L.append("% Requires: booktabs, graphicx (for \\resizebox).")
    _L.append(r"\begin{table*}[t]")
    _L.append(r"\centering")
    _L.append(r"\small")
    _L.append(r"\resizebox{\textwidth}{!}{%")
    _L.append(rf"\begin{{tabular}}{{{_colspec}}}")
    _L.append(r"\toprule")

    # Group header row (Model block spans the model + condition columns).
    _grp_cells = [r"\multicolumn{2}{c}{Model}"]
    for _g, _n in _groups:
        _grp_cells.append(rf"\multicolumn{{{_n}}}{{c}}{{{_g}}}")
    _L.append(" & ".join(_grp_cells) + r" \\")

    # cmidrules under each benchmark group (offset by the 2 identity columns).
    _rules = []
    _start = 3
    for _g, _n in _groups:
        _end = _start + _n - 1
        _rules.append(rf"\cmidrule(lr){{{_start}-{_end}}}")
        _start = _end + 1
    _L.append(" ".join(_rules))

    # Sub-header row.
    _sub = ["Model", "Cond."] + [
        _col.replace("\u2193", r"$\downarrow$")
        for (_cid, _grp, _col, _judged, _lo, _scale) in _col_specs
    ]
    _L.append(" & ".join(_sub) + r" \\")
    _L.append(r"\midrule")

    # Body rows (drop any model/condition with no data at all).
    for _mdl, _conds in ROW_ORDER:
        _shown = 0
        for _cond in _conds:
            _cells = [
                _fmt_tex(_cid, _grp, _judged, _lo, _scale, _mdl, _cond)
                for (_cid, _grp, _col, _judged, _lo, _scale) in _col_specs
            ]
            if all(_c in ("--", r"\textit{pend.}") for _c in _cells):
                continue
            _mcell = _esc(_mdl) if _shown == 0 else ""
            _L.append(" & ".join([_mcell, _cond] + _cells) + r" \\")
            _shown += 1
    _L.append(r"\bottomrule")
    _L.append(r"\end{tabular}}")  # closes the \resizebox argument

    _caption = (
        rf"Benchmark results for the canonical model set. We report each model "
        rf"under its zero-shot (the pre-SFT \texttt{{<family>/instruct}} "
        rf"checkpoint) and SFT condition across six benchmarks: GoldCoin-HIPAA "
        rf"(applicability and compliance, upstream-parity accuracy), PrivacyLens "
        rf"(question-answering accuracy, adjusted leakage, and helpfulness "
        rf"rate), ConfAIde (Tier-2b Pearson correlation), "
        rf"CIRL-729 (leakage, utility, and net score), VLM-GeoPrivacy (Q7 "
        rf"location-granularity accuracy), and MMLU (overall accuracy). We bold "
        rf"the best value per column, excluding the self-judged teacher. "
        rf"Percentages are scaled by 100, except CIRL net score ($-1$ to $1$); "
        rf"$\downarrow$ marks lower-is-better "
        rf"columns. All judged columns use {EXPECTED_JUDGE} as the judge. "
        rf"$\dagger$: self-judged (judge and subject share weights), an "
        rf"optimistic bound excluded from best-per-column bolding. $\ddagger$: "
        rf"stale, finalized before the 2026-07-21 PrivacyLens judge-response "
        rf"parser fix. CIRL-729 scores every strict-format miss as $-1$ (paper "
        rf"protocol); its leakage and utility rates are conditioned on "
        rf"strict-parseable actions and omitted when fewer than half of the 729 "
        rf"actions parse. --: a documented model-level finding (refusal, format "
        rf"collapse, or a suppressed conditional rate), not a zero."
    )
    _L.append(rf"\caption{{{_caption}}}")
    _L.append(r"\label{tab:benchmark_results}")
    _L.append(r"\end{table*}")

    table_tex = "\n".join(_L)
    (REPORT_DIR / "benchmark_results.tex").write_text(table_tex + "\n")
    print(f"saved LaTeX table to {REPORT_DIR / 'benchmark_results.tex'}")
    print(table_tex)
    return (table_tex,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase D — SFT effect (paired Zero-shot → SFT delta)

    Same 11 checkpoints, same benchmarks, same judge, LoRA on/off. Delta is
    signed so **positive = SFT better** on every column (lower-is-better
    columns are negated). Only cells present in *both* conditions are
    differenced — so CIRL Net pairs everywhere, while CIRL Lk/Util pair
    only where both sides cleared the majority-parseable bar (the three
    Gemma families).
    """)
    return


@app.cell
def _(COLUMNS, REPORT_DIR, ROW_ORDER, picked):
    import pandas as _pd

    _lut2 = {
        (r["model"], r["condition"], r["col_id"]): r["value"]
        for _, r in picked.iterrows()
    }
    _specs = [(f"{c[0]}::{c[1]}", f"{c[0]} {c[1]}", c[7], c[8]) for c in COLUMNS]

    _drows = []
    for _mdl2, _conds2 in ROW_ORDER:
        _row2 = {"model": _mdl2}
        for _cid2, _label2, _lo2, _scale2 in _specs:
            _z = _lut2.get((_mdl2, "Zero-shot", _cid2))
            _s = _lut2.get((_mdl2, "SFT", _cid2))
            if _z is None or _s is None:
                _row2[_label2] = None
                continue
            _mult = 100.0 if _scale2 == "pct" else 1.0
            _d = (_s - _z) * _mult
            _row2[_label2] = -_d if _lo2 else _d
        _drows.append(_row2)
    sft_delta = _pd.DataFrame(_drows).set_index("model")

    print(
        "Paired Zero-shot → SFT delta (positive = SFT better; "
        "lower-is-better columns sign-flipped). Blank = one side missing. "
        "Reminder: PL judged deltas inherit the ‡ staleness of both sides."
    )
    print(sft_delta.round(1).to_string())
    print("\nMean delta per column (over models with both conditions):")
    print(sft_delta.mean(axis=0).round(2).to_string())
    sft_delta.to_parquet(REPORT_DIR / "sft_delta.parquet")
    sft_delta.round(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Provenance — which run, judge and metric semantics back each cell
    """)
    return


@app.cell
def _(provenance):
    provenance.sort_values(["model", "condition", "column"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read the result — camera-ready checklist

    - The **markdown table** above (also at
      `tables/benchmark_results/benchmark_results.md`) is the current
      camera-ready state of the main results table.
    - **CIRL — RESOLVED (2026-07-22/23).** The canonical set + teacher ran
      on CIRL-729 (`*_eval_cirl729_canonical`, `*_eval_cirl729_teacher`);
      all 23 cells fill Net. Strict-format misses score −1 per the paper
      protocol; Lk/Util conditional rates are blanked below 50%
      strict-parseable (the scan cell lists every suppression). Do NOT
      resurrect the old `cirl_vignettes` accuracy; it was
      PrivacyLens-under-CIRL-protocol.
    - **PrivacyLens (‡) — mostly RESOLVED.** The 2026-07-21 F1 rescue
      re-finalized 23/24 judged cells from the raw judge `output.jsonl`,
      clearing their ‡ automatically; the teacher row is the remaining
      un-rescued cell and still carries ‡. The F3/F4 protocol caveat
      stands: keeper-era PL rows are not comparable with any
      post-2026-07-21 re-run.
    - **GoldCoin** — numbers here are upstream-parity accuracy via the
      exact retro-conversion (accuracy × parseable_rate); the conversion
      applied per cell is in the provenance table (`semantics` column). If
      GoldCoin is re-run post-flip, the reader detects the native
      forced-wrong file and uses it directly — pre/post semantics cannot
      silently mix.
    - **Blank cells** are documented findings, not zeros — see the coverage
      grid and the header cell (harc ConfAIde refusals, openthinker/gpt-oss
      SFT format collapse, text-only Q7, sub-majority-parseable CIRL
      conditional rates).
    - The judge-free variance record (complete, 163 arms) stays excluded:
      it is a noise-floor instrument, not a results source.
    """)
    return


if __name__ == "__main__":
    app.run()
