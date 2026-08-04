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
    6. **Variance gating (Phase B3 / C, added 2026-08-03).** Every "best"
       claim in this table is now checked against the **measured re-run
       noise floor** from the judge-free seed/rep variance record
       (`*_eval_judgefree_variance*`, 163 arms / 3–8 reps per config). The
       variance record supplies **dispersion only, never a value** — no
       number in the results table comes from it, and Phase B3 asserts
       that. A column's top value is bolded together with every other cell
       inside its noise band, so a column whose leader is not separable
       from the field renders as a **tie set**, not a winner. See the
       "reading the variance gate" cell for the full rule.
    7. **CIRL Lk/Util now report the `*_scorable` rates (2026-08-03).**
       They previously used the strict paper-parity rates and were blanked
       below majority strict-parseable, which discarded 15 of 23 cells for
       three unrelated reasons — only one of which is an evaluation failure.
       They now average over the actions that produced a *complete message*,
       filling 19 of 22 canonical cells, and the three genuine failures
       render **N/A** rather than blank. **CIRL Net is unchanged.** Phase B2b
       has the decomposition; `scripts/rescore_cirl_scorable.py` back-filled
       the metric onto existing runs (CPU only, no re-run, backups kept).

    ### Excluded sweeps (deliberate — do not add without reading their yaml)

    - `*eval_judgefree_variance*` — the judge-free variance record (163
      arms, complete). It is a **noise-floor instrument, not a results
      source**: repeated seeds per cell, non-canonical seeds, server-mode
      engines, and post-parity-review code. Since 2026-08-03 it *is* read
      by this notebook — but only for **dispersion** (Phase B3), which is
      then attached to the canonical cells as an uncertainty band. Its
      values never enter `picked`.
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

    ### What the variance gate does NOT cover (state these, don't imply them)

    - **The three PrivacyLens columns have no measured noise floor at all.**
      The variance record is judge-free by construction (no judge server,
      sidecar disabled), so QA Acc / Adj Lk / Helpful get **no** band. They
      are marked `°` and their leaders are *nominal*, not established.
    - **SFT training-seed variance is not measured anywhere.** Each SFT row
      is a single training run; the variance record reps *inference*, not
      training. The only training-seed replication in the project is the
      5-seed **GRPO** sweep (`multirun/2026-05-28_seed_variance_sweep`,
      final composite reward CV 3.5%), which says nothing about these SFT
      checkpoints' benchmark scores. The bands below are therefore a
      **lower bound** on total run-to-run variability.
    - **SFT bands are transferred from sibling checkpoints.** The variance
      record's SFT arms are `sft-canonical-ckpt{86,171,172,258,342,513}` —
      the post-2026-07-18 template+DFT era — not the keeper `sft-canonical`
      adapters scored here. Same base weights, same SFT data, different
      training-protocol era: dispersion transfers, means do not (and are
      never used).
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
        # NOT ADDED: *_eval_cirl729_tokenfix/*. A token-budget repair for the
        # phi-4/openthinker SFT cells was drafted and then killed by its own
        # smoke test (2026-08-03) — 6x the budget did not move either model's
        # truncation rate; both simply run any cap out in a degenerate loop.
        # The 20-row debug probes live in
        # `multirun/2026-08-03_eval_cirl729_tokenfix_SMOKE_NOGO_debug20/` and
        # are deliberately named so NO glob here can pick them up: they are
        # sample_n=20 runs and would otherwise supersede the real 729-row
        # cells under latest-run-wins.
        #
        # ── RL stage (2026-08-03) ─────────────────────────────────────────
        # Both sweeps are Gemma-4-31B-it-judged and post-parity-review, so
        # they compose with the canonical corpus under the same protocol.
        # Each carries its OWN SFT base cell — see the RL_* rows below for
        # why that base is not optional.
        #
        # KTO (k-series) arms + their base. Two run dirs, deliberately named
        # rather than globbed with `*`:
        #   14-28-30  the good run — goldcoin/privacylens/confaide/vlm
        #   11-46-24  supplies ONLY k3-base's confaide cell, which 14-28-30
        #             lost to a SLURM submission wedge
        #             ("sbatch: Socket timed out"). Same protocol, same
        #             code; composes under per-cell latest-run-wins.
        # NOT ADDED: 2026-08-03_k3_arms_ci_eval/09-02-18 — every task
        # request in it 404'd (a stray server.env VLLM_SERVER_URL routed
        # inference to the judge port), so its cells are empty. Its
        # vlm_geoprivacy metrics ARE real (that benchmark never routed
        # through the hijacked path), but 14-28-30 covers those cells, so
        # the run is excluded rather than relied on.
        "*_k3_arms_ci_eval/14-28-30",
        "*_k3_arms_ci_eval/11-46-24",
        # GRPO keeper + its matched SFT base, re-run under the current
        # protocol. The keeper's published 2026-06-24 numbers are NOT usable
        # here: pre-flip GoldCoin, pre-parser-fix + Qwen3.6-judged
        # PrivacyLens, and the retired `cirl_vignettes` metric.
        "*_eval_rl_stage_keeper/*",
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

    # ── CIRL Lk/Util fidelity bar ─────────────────────────────────────────
    # A CIRL conditional rate is reportable only when the model produced a
    # complete, scoreable message on at least this fraction of the 729
    # actions (`scorable_rate` from compute_metrics: non-empty extraction,
    # not truncated). Below it the cell renders **N/A** — not "—" — because
    # the failure is STRUCTURAL: the model cannot be evaluated on this
    # benchmark, as distinct from a metric we chose not to report.
    #
    # Measured 2026-08-03, this bar keeps 19 of 22 canonical cells and fails
    # exactly three, each for a documented reason:
    #   gpt-oss-20b/sft-canonical    13.4%  631/729 EMPTY harmony final channel
    #   openthinker3-7b/sft-canonical 44.2% 407/729 hit the token cap
    #   phi-4/sft-canonical           46.0% 394/729 hit the token cap
    #
    # "Hit the cap" was initially read as a token-BUDGET artifact, and a
    # repair sweep was drafted. Its own smoke test FALSIFIED that — 20-row
    # probes at max_tokens=12288 (6x / 3x the original budgets):
    #     phi-4/sft         truncation 54% -> 55%,  scorable 46.0% -> 45.0%
    #     openthinker3-7b/sft          56% -> 65%,  scorable 44.2% -> 35.0%
    # Every capped row stopped at EXACTLY 12288 tokens (44-65k chars) while
    # rows that terminate normally finish in a median 392 / 830 tokens. The
    # runaway text is a degenerate loop emitting the SFT JSON schema, not
    # CIRL's <think>/<answer> (phi-4: one 80-char phrase 82x; openthinker:
    # endless `"has_criminal_conviction_others": false, ...` fields).
    # So both are non-termination + an SFT format regression: a model
    # finding, not a budget we set, and NO larger budget repairs it.
    # The sweep was abandoned. See the Phase B2b table.
    CIRL_SCORABLE_MIN = 0.5

    # The teacher/judge model, evaluated as a subject. Its label carries the
    # warning inline so the row cannot be copied out of the table without it.
    TEACHER_ROW = "Gemma-4-31B-it (teacher/judge — self-judged)"

    # ── RL-stage rows (2026-08-03) ────────────────────────────────────────
    # Both RL blocks sit on Qwen3.5-9B but on SFT lineages that are NOT the
    # `sft-canonical` adapter behind the "Qwen3.5-9B / SFT" row above:
    #   GRPO  qwen-teacher-era SFT (2026-06-23 v9 run)
    #   KTO   the m2 merged SFT   (2026-07-28 m2 run)
    # So each block carries its OWN "SFT base" condition and every RL delta
    # must be read against that base, never against the canonical SFT row —
    # otherwise the lineage difference is charged to the RL stage. The labels
    # say "own SFT base" inline so a row cannot be lifted out of the table
    # without the caveat, the same discipline as TEACHER_ROW.
    GRPO_ROW = "Qwen3.5-9B + GRPO (own SFT base)"
    KTO_ROW = "Qwen3.5-9B + KTO (own SFT base)"

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
        # ── RL stage. Each block is self-contained: read the RL conditions
        # against the "SFT base" row directly above them, never against the
        # canonical Qwen3.5-9B/SFT row (different SFT lineage).
        (GRPO_ROW, ["SFT base", "GRPO"]),
        # SFT control = plain SFT loss on the KTO dataset's desirable rows
        # only. It isolates whether KTO's use of UNdesirable examples adds
        # anything over ordinary fine-tuning on the same corrected text.
        (
            KTO_ROW,
            [
                "SFT base",
                "KTO verdict",
                "KTO citation",
                "KTO scrutinize",
                "SFT control",
            ],
        ),
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
    # RL stage. Each block's base maps into its OWN row, so the two lineages
    # can never be silently pooled.
    OVERRIDE_TO_ROW["qwen3.5-9b/v9-base"] = (GRPO_ROW, "SFT base")
    OVERRIDE_TO_ROW["qwen3.5-9b/v9-ckpt100"] = (GRPO_ROW, "GRPO")
    OVERRIDE_TO_ROW["qwen3.5-9b/k3-base"] = (KTO_ROW, "SFT base")
    OVERRIDE_TO_ROW["qwen3.5-9b/k3-verdict"] = (KTO_ROW, "KTO verdict")
    OVERRIDE_TO_ROW["qwen3.5-9b/k3-citation"] = (KTO_ROW, "KTO citation")
    OVERRIDE_TO_ROW["qwen3.5-9b/k3-scrutinize"] = (KTO_ROW, "KTO scrutinize")
    OVERRIDE_TO_ROW["qwen3.5-9b/k3-sftctrl"] = (KTO_ROW, "SFT control")

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
        # cirl_vignettes rejection accuracy.
        #
        # Lk/Util read the *_scorable rates (2026-08-03): averages over the
        # rows where the model produced a COMPLETE message — non-empty
        # extraction, not truncated. Previously they read the strict
        # (paper-parity) rates and were blanked below majority-parseable,
        # which threw away 15 of 23 cells for three different reasons, only
        # one of which is a real evaluation failure. See the Phase B3b
        # markdown for the full decomposition. Net keeps the STRICT
        # paper-protocol -1s and is unchanged — it is the headline.
        (
            "CIRL",
            "Lk↓",
            "cirl",
            "cirl",
            "compute_metrics",
            "leakage.leakage_rate_scorable",
            False,
            True,
            "pct",
            "cirl_scorable",
        ),
        (
            "CIRL",
            "Util",
            "cirl",
            "cirl",
            "compute_metrics",
            "utility.utility_rate_scorable",
            False,
            False,
            "pct",
            "cirl_scorable",
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

    # ── Variance record: the empirical re-run noise floor ─────────────────
    # The judge-free seed/rep sweep (2026-07-21→23, 163 arms). Read for
    # DISPERSION ONLY — Phase B3 asserts none of its values reach `picked`.
    # Composition:
    #   *_eval_judgefree_variance         main N=3 server-mode half (117
    #                                     arms) + the ConfAIde Option-B
    #                                     repair + the 12b-ckpt513 cell
    #   *_eval_judgefree_variance_gptoss  in-process harmony companion
    #   *_eval_judgefree_variance_topup   GoldCoin seed top-up; reps live in
    #                                     `goldcoin_sNNN/` dirs inside one arm
    # The 2026-07-21 `..._CANCELLED_precirlswap` tree is deliberately NOT
    # matched: mixed pre-parity metric semantics, never pool it.
    VARIANCE_GLOBS = [
        "*_eval_judgefree_variance/*",
        "*_eval_judgefree_variance_gptoss/*",
        "*_eval_judgefree_variance_topup/*",
    ]

    # Which noise source a column's reps actually measure. "sampled" = reps
    # vary `sampling_params.seed` (101–108) on a temp>0 benchmark;
    # "greedy" = temp-0 benchmark, so the spread is pure engine
    # nondeterminism (batch composition, kernel numerics). Both are real
    # re-run noise; reporting them apart keeps the mechanism visible.
    VARIANCE_REP_TYPE = {
        "GoldCoin::Appl.": "sampled",
        "GoldCoin::Comp.": "sampled",
        "VLM::Q7": "sampled",
        "ConfAIde::r": "greedy",
        "CIRL::Lk↓": "greedy",
        "CIRL::Util": "greedy",
        "CIRL::Net": "greedy",
        "MMLU::Acc": "greedy",
    }

    # A cell needs at least this many reps before its own spread is usable.
    NOISE_MIN_REPS = 2

    # Noise policy, in display units (pct-points; CIRL Net raw −1…1).
    #
    #   band(cell) = max(own measured max rep-range, column median range)
    #
    # Two cells are treated as SEPARATED only when
    #   |a − b|  >=  (band(a) + band(b)) / 2
    # i.e. when their ±half-band intervals do not overlap. With equal bands
    # this reduces to the familiar "a gap smaller than the metric's rep
    # range is indistinguishable from a re-run", which is the rule the
    # longitudinal notebook's Phase D2b already uses.
    #
    # The column-median floor exists because a 3-rep range of exactly 0.0
    # (observed: gemma-4-E4B/instruct GoldCoin Appl.) is an artifact of
    # three draws, not evidence that the cell is noiseless. Never let a
    # cell claim to be quieter than its benchmark's typical cell.
    NOISE_USE_COLUMN_MEDIAN_FLOOR = True

    # Columns with NO variance data (the judged PrivacyLens trio) can't be
    # gated. True = still bold their nominal leader but mark the column `°`
    # and say in the legend that the lead is unestablished. Flip to False to
    # suppress bolding there entirely.
    BOLD_UNMEASURED_COLUMNS = True
    return (
        BOLD_UNMEASURED_COLUMNS,
        CIRL_SCORABLE_MIN,
        COLUMNS,
        EXPECTED_JUDGE,
        JUDGE_ATTESTED_MULTIRUNS,
        MULTIRUN_GLOB_ROOT,
        NOISE_MIN_REPS,
        NOISE_USE_COLUMN_MEDIAN_FLOOR,
        OVERRIDE_TO_ROW,
        PENDING_GROUPS,
        PL_PARSER_FIX_DT,
        REPORT_DIR,
        ROW_ORDER,
        SELF_JUDGED_ROWS,
        SWEEP_GLOBS,
        TEACHER_ROW,
        VARIANCE_GLOBS,
        VARIANCE_REP_TYPE,
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
    CIRL_SCORABLE_MIN,
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

    def read_metric(mp, key, kind, suppressed=None):
        """Read one metric. Returns (value, semantics, stale, na) or None.

        `na=True` means the cell is a STRUCTURAL failure: the benchmark ran,
        but the model's output cannot be scored (empty generations, truncated
        fragments). Such a cell carries `value=None` and renders **N/A**,
        which is a different statement from "—" (a metric we chose not to
        report, e.g. a documented finding or a text-only model).

        Shared with the Phase B3 variance scan so the noise floor is
        measured on EXACTLY the quantity the table reports — same
        GoldCoin denominator handling, same CIRL majority-parseable bar.
        `suppressed`, when given, collects the CIRL conditional reads this
        call dropped, as (path, key, "parseable/total").

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

        kind "cirl_net": CIRL-729 headline. Includes the paper-protocol -1
          for every strict-format miss, so it always reads; records
          parseable=n/total in semantics.

        kind "cirl_scorable": CIRL-729 Lk/Util. Rates over the rows where the
          model produced a COMPLETE message (`*_scorable` in
          dagspaces/cirl/stages/compute_metrics.py: non-empty extraction,
          not truncated). Below CIRL_SCORABLE_MIN the cell is a structural
          failure → `na=True`, rendered **N/A**. A metrics.json predating
          the 2026-08-03 rescore has no `scorable_rate`; that returns None
          (renders "—") rather than guessing — run
          `scripts/rescore_cirl_scorable.py`.
        """
        try:
            data = _json.loads(mp.read_text())
        except (ValueError, OSError):
            return None
        if kind == "cirl_net":
            val = _dotted(data, key)
            if val is None:
                return None
            _p, _t = data.get("parseable"), data.get("total")
            return float(val), f"cirl:strict_parseable={_p}/{_t}", False, False
        if kind == "cirl_scorable":
            val = _dotted(data, key)
            _sr = data.get("scorable_rate")
            _n, _t = data.get("scorable"), data.get("total")
            if val is None or _sr is None:
                # Pre-rescore artifact — no verdict is possible, and
                # inventing one would silently mislabel a cell N/A.
                return None
            _exc = data.get("scorable_exclusions", {}) or {}
            _sem = (
                f"cirl:scorable={_n}/{_t}"
                f"(empty={_exc.get('empty_answer', 0)},"
                f"trunc={_exc.get('truncated', 0)})"
            )
            if float(_sr) < CIRL_SCORABLE_MIN:
                if suppressed is not None:
                    suppressed.append((str(mp), key, f"{_n}/{_t}"))
                return None, _sem + ":STRUCTURAL_NA", False, True
            return float(val), _sem, False, False
        if kind == "gc_acc":
            acc = data.get("accuracy")
            pr = data.get("parseable_rate")
            if acc is None:
                return None
            if "accuracy_among_parseable" in data:
                return float(acc), "goldcoin:forced_wrong_native", False, False
            if pr is None:
                return None  # cannot establish semantics — refuse the cell
            return (
                float(acc) * float(pr),
                f"goldcoin:retro_converted_acc_x_parseable(pr={float(pr):.4f})",
                False,
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
            return float(val), sem, stale, False
        return float(val), "", False, False

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
                _res = read_metric(_mp, _key, _kind, _cirl_suppressed)
                if _res is None:
                    continue
                _val, _sem, _stale, _na = _res
                rows.append(
                    {
                        "model": _model,
                        "condition": _cond,
                        "group": _grp,
                        "col": _col,
                        "col_id": f"{_grp}::{_col}",
                        "value": _val,
                        "structural_na": _na,
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
    # Cells whose CIRL Lk/Util is a STRUCTURAL failure (rendered N/A, not "—").
    if _cirl_suppressed:
        print(
            f"CIRL-729: {len(_cirl_suppressed)} Lk/Util reads are STRUCTURAL "
            f"N/A (<{CIRL_SCORABLE_MIN:.0%} of 729 actions produced a "
            "complete, scoreable message — empty generations, or generations "
            "that never terminate and run the token cap out; a 12288-token "
            "probe confirmed the latter is NOT a budget artifact). These "
            "render N/A, NOT '—': the model cannot be evaluated on this "
            "benchmark, as opposed to a metric we chose not to report. Its "
            "paper-protocol score still lives in Net."
        )
        for _mpath, _mkey, _frac in sorted(_cirl_suppressed):
            print(f"   {_frac:>8}  {_mkey:<32} {_mpath.split('multirun/')[-1]}")

    # A pre-rescore metrics.json has no `scorable_rate`, so its Lk/Util read
    # returns None and the cell silently renders "—". Say so loudly — the
    # fix is a CPU-only re-score, not a re-run.
    _n_cirl_lk = int((scan["col_id"] == "CIRL::Lk↓").sum())
    if _n_cirl_net and not _n_cirl_lk:
        print(
            "\n!! CIRL Lk/Util: 0 observations while Net has "
            f"{_n_cirl_net} — every metrics.json predates the `*_scorable` "
            "metrics. Run `scripts/rescore_cirl_scorable.py` (no GPU, "
            "re-scores existing parse artifacts in place)."
        )

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
    return read_metric, scan


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

    _have = {
        (r["model"], r["condition"], r["col_id"])
        for _, r in picked.iterrows()
        if r["value"] is not None and not _pd.isna(r["value"])
    }
    _na = {
        (r["model"], r["condition"], r["col_id"])
        for _, r in picked.iterrows()
        if bool(r.get("structural_na"))
    }
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
                elif (_mdl, _cond, _cid) in _na:
                    # The benchmark RAN — the model's output was unscoreable.
                    _row[_label] = "N/A"
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
    mo.md(r"""
    ## Phase B2b — CIRL Lk/Util: what fills, and what is N/A

    Until 2026-08-03 these two columns read the **strict** (paper-parity)
    rates and were blanked below majority strict-parseable, which threw away
    **15 of 23 cells**. Inspecting the per-row parse artifacts showed those
    15 failures are three unrelated things, and only one is an evaluation
    failure:

    | Cause | Cells | What the rows look like |
    |---|---|---|
    | Missing `</think>` **only** | the 6 Qwen cells | A clean, complete `<answer>…</answer>`. qwen3.5-4b/sft: 729/729 emit `<answer>`, 728 close it — 88/729 strict. Our family yaml runs Qwen with `enable_thinking: false`, while the CIRL prompt *requires* a reasoning phase. |
    | Unclosed `</answer>` | llama, harc, gpt-oss instr. | `finish_reason=stop`, complete well-formed prose, the closing tag simply never emitted (llama 427/729). |
    | **Genuinely unscoreable** | gpt-oss/sft, openthinker/sft, phi-4/sft | An **empty** final channel (gpt-oss harmony: 631/729 rows), or generation that **never terminates** (407 and 394 rows run the token cap out). |

    So the columns now report the **scorable** rate — leakage/utility over
    the actions where the model produced a *complete message* (non-empty
    extraction, `finish_reason != length`), from
    `dagspaces/cirl/stages/compute_metrics.py`. Two properties make this
    safe to swap in:

    1. **It cannot move a compliant cell.** Where a model emits a well-formed
       `<answer>` block, strict and scorable extract the *same text*. Measured
       across every cell that already cleared the strict bar, |Δ leakage| ≤
       0.004 (gemma-4-12b, gemma-4-E4B, phi-4 zero-shot: Δ = 0.000). The
       re-score of all 160 CIRL cells changed the strict headline on **zero**
       of them.
    2. **It refuses to reward silence.** The `*_lenient` variant already on
       disk scores all 729 rows, and an empty answer substring-matches
       nothing — so it books leakage 0.0 *and* utility 0.0 with no −1 floor.
       gpt-oss-20b/sft-canonical's lenient leakage is 0.038, which reads as
       excellent privacy and is really "emitted nothing". Excluding empty and
       truncated rows is what stops that from entering the table.

    Below `CIRL_SCORABLE_MIN` the cell renders **N/A**, deliberately distinct
    from `—`: the benchmark ran and the model failed it structurally, versus a
    metric we chose not to report. Its paper-protocol score still appears in
    **CIRL Net**, which is untouched and remains the headline.

    ### The two truncating cells are NOT a token-budget artifact (tested)

    The obvious reading of "407 and 394 rows hit `max_tokens`" is that our
    budget was too small. A repair sweep was written on that premise and
    **killed by its own smoke test** (2026-08-03, 20-row probes at
    `max_tokens=12288`, i.e. 6× phi-4's budget and 3× openthinker's):

    | Cell | truncated before → after | scorable before → after |
    |---|---|---|
    | `phi-4/sft-canonical` | 54% → **55%** | 46.0% → **45.0%** |
    | `openthinker3-7b/sft-canonical` | 56% → **65%** | 44.2% → **35.0%** |

    Neither improved; openthinker got worse. Every capped row stopped at
    *exactly* 12288 tokens (44–65k characters), while rows that terminate
    normally finish in a median of 392 (phi-4) / 830 (openthinker) tokens.
    The runaway text is a degenerate loop emitting the **SFT JSON schema**
    rather than CIRL's `<think>`/`<answer>` — phi-4 repeats one 80-character
    phrase 82×; openthinker emits unbounded
    `"has_criminal_conviction_others": false, …` fields.

    So both cells are **non-termination plus an SFT format regression** — a
    property of those checkpoints, not of a budget we chose, and no larger
    budget repairs it. All three N/A cells are genuine model findings. The
    sweep was abandoned rather than run; its 20-row probes are parked in
    `multirun/2026-08-03_eval_cirl729_tokenfix_SMOKE_NOGO_debug20/`, named so
    that no `SWEEP_GLOBS` entry can pick up sample_n=20 data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase B3 — The measured re-run noise floor (seed/rep variance record)

    **Why this phase exists.** Every "best model" claim in a benchmark table
    is a claim that one number is *distinguishable* from another. On these
    benchmarks that is often false: a same-weights, same-protocol re-run
    moves GoldCoin applicability by up to 18.7 pt (qwen3.5-2b/instruct
    flips its whole Applicable/Not-Applicable policy with the sampling
    seed) and ConfAIde Tier-2b Pearson by up to 27.2 pt (openthinker
    parse-subset instability, **at temperature 0**). Bolding a 0.4 pt lead
    against that is a fabricated result.

    **The instrument.** `multirun/*_eval_judgefree_variance{,_gptoss,_topup}`
    — 163 arms, the canonical families' `instruct` and
    `sft-canonical-ckptNNN` configs re-run 3× (GoldCoin 8× after the seed
    top-up) over the judge-free benchmarks. Sampled benchmarks (GoldCoin,
    VLM Q7; temp 0.2) rep over `sampling_params.seed` 101–108; greedy ones
    (ConfAIde, CIRL-729, MMLU; temp 0) ignore the seed, so their spread is
    pure engine nondeterminism. Metrics are read through the **same**
    `read_metric` used in Phase A, so the dispersion is measured on exactly
    the quantity the table reports (same GoldCoin upstream denominator,
    same CIRL majority-parseable bar).

    **Dispersion only — never a value.** The reps use non-canonical seeds,
    server-mode engines and post-parity-review code, so their *means* sit
    slightly off the canonical points; only the *spread* transfers. The
    assertion at the end of the scan cell fails loudly if any variance
    multirun ever leaks into `picked`.

    **Band assignment.** Each table cell gets

    $$\text{band} = \max\big(\text{own max rep-range},\ \text{column median rep-range}\big)$$

    preferring the cell's own reps (`instruct` arms are the *same weights*
    as the Zero-shot rows; SFT rows borrow from their family's
    `sft-canonical-ckptNNN` siblings — see the header's transfer caveat),
    falling back to the family then the column. Two cells count as
    separated only when $|a-b| \ge (\text{band}_a + \text{band}_b)/2$, i.e.
    when their $\pm$half-band intervals do not overlap.
    """)
    return


@app.cell
def _(
    COLUMNS,
    MULTIRUN_GLOB_ROOT,
    OVERRIDE_TO_ROW,
    SWEEP_GLOBS,
    VARIANCE_GLOBS,
    picked,
    re,
    read_metric,
):
    import fnmatch as _fnmatch

    import pandas as _pd

    # slug → display name, reusing the single family registry above so a new
    # model can never be in the results table but silently absent here.
    _SLUG_TO_ROW = {
        _k.split("/")[0]: _v[0]
        for _k, _v in OVERRIDE_TO_ROW.items()
        if _k.endswith("/instruct")
    }

    # Variance-arm model overrides → (condition, variant, era-matched?).
    #   <slug>/instruct              same weights as the Zero-shot rows
    #   <slug>/sft-canonical         same weights as the SFT rows (none on
    #                                disk today; handled so a future keeper-
    #                                era rep sweep is picked up correctly)
    #   <slug>/sft-canonical-ckptNNN sibling adapter, post-2026-07-18
    #                                template+DFT era — dispersion transfers,
    #                                means do not
    _V_INSTRUCT_RE = re.compile(r"^(?P<slug>[\w.\-]+)/instruct$")
    _V_KEEPER_RE = re.compile(r"^(?P<slug>[\w.\-]+)/sft-canonical$")
    _V_CKPT_RE = re.compile(r"^(?P<slug>[\w.\-]+)/sft-canonical-ckpt(?P<step>\d+)$")

    def _variance_identity(ovr):
        _m = _V_INSTRUCT_RE.match(ovr)
        if _m:
            return _m.group("slug"), "Zero-shot", "instruct", True
        _m = _V_KEEPER_RE.match(ovr)
        if _m:
            return _m.group("slug"), "SFT", "sft-canonical", True
        _m = _V_CKPT_RE.match(ovr)
        if _m:
            return _m.group("slug"), "SFT", f"ckpt{_m.group('step')}", False
        return None

    _vdirs = sorted(
        {p for g in VARIANCE_GLOBS for p in MULTIRUN_GLOB_ROOT.glob(g) if p.is_dir()}
    )
    if not _vdirs:
        print(
            "!! no variance sweep dirs matched VARIANCE_GLOBS — the noise "
            "floor will be empty and NOTHING will be gated. Check the globs "
            "before quoting any 'best' from this table."
        )

    _vrows = []
    _vsupp = []
    _varms = 0
    for _vdir in _vdirs:
        for _vsub in sorted(
            (p for p in _vdir.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        ):
            _ovp = _vsub / ".hydra" / "overrides.yaml"
            if not _ovp.exists():
                continue
            _vovr = _vseed = _vsweep = None
            for _line in _ovp.read_text(errors="ignore").splitlines():
                _line = _line.strip().lstrip("- ").strip()
                if _line.startswith("model="):
                    _vovr = _line.split("=", 1)[1]
                elif _line.startswith("variance_seed="):
                    _vseed = _line.split("=", 1)[1]
                elif _line.startswith("+sweep="):
                    _vsweep = _line.split("=", 1)[1]
            if _vovr is None:
                continue
            _ident = _variance_identity(_vovr)
            if _ident is None:
                continue
            _slug, _vcond, _vvariant, _era_matched = _ident
            if _slug not in _SLUG_TO_ROW:
                continue
            _varms += 1
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
                # Two rep layouts: the N=3 sweeps put one rep per arm under
                # `<bench>/` (rep id = the arm's variance_seed), the GoldCoin
                # top-up puts five reps inside ONE arm as `<bench>_sNNN/`.
                _reps = []
                if _vseed is not None and (_vsub / _bd).is_dir():
                    _reps.append((_vsub / _bd, _vseed))
                for _rd in sorted(_vsub.glob(f"{_bd}_s*")):
                    _sfx = _rd.name[len(_bd) + 2 :]
                    if _rd.is_dir() and _sfx.isdigit():
                        _reps.append((_rd, _sfx))
                for _rdir, _rseed in _reps:
                    _vmp = _rdir / _inner / "outputs" / _subdir / "metrics.json"
                    if not _vmp.exists():
                        continue
                    _vres = read_metric(_vmp, _key, _kind, _vsupp)
                    if _vres is None:
                        continue
                    _vval, _vsem, _, _vna = _vres
                    # A structurally-unscoreable rep contributes no number, so
                    # it cannot contribute dispersion either.
                    if _vna or _vval is None:
                        continue
                    _vrows.append(
                        {
                            "model": _SLUG_TO_ROW[_slug],
                            "condition": _vcond,
                            "variant": _vvariant,
                            "era_matched": _era_matched,
                            "col_id": f"{_grp}::{_col}",
                            "col_label": f"{_grp} {_col}",
                            "scale": _scale,
                            "judged": _judged,
                            "value": _vval,
                            "seed": _rseed,
                            "semantics": _vsem,
                            "sweep": _vsweep,
                            "arm": f"{_vdir.relative_to(MULTIRUN_GLOB_ROOT)}/{_vsub.name}",
                        }
                    )
    variance_obs = _pd.DataFrame(_vrows)

    print(
        f"variance record: {_varms} arms, {len(variance_obs)} (rep × column) "
        f"observations across {len(_vdirs)} sweep dirs"
    )
    if len(variance_obs):
        print(
            "columns covered: "
            + ", ".join(sorted(variance_obs["col_id"].unique()))
        )
        _uncov = sorted(
            {f"{c[0]}::{c[1]}" for c in COLUMNS} - set(variance_obs["col_id"])
        )
        print(
            f"columns with NO variance data (judge-free sweep → judged "
            f"columns are unmeasurable here): {', '.join(_uncov)}"
        )
        _sft_variants = sorted(
            variance_obs[variance_obs.condition == "SFT"]["variant"].unique()
        )
        print(f"SFT variants backing the SFT bands: {_sft_variants}")
        if not variance_obs[variance_obs.condition == "SFT"]["era_matched"].any():
            print(
                "   (none is the keeper `sft-canonical` adapter — SFT bands "
                "are transferred from post-2026-07-18-era siblings; "
                "dispersion only, see the header caveat)"
            )
    if _vsupp:
        print(
            f"variance: {len(_vsupp)} CIRL conditional reads suppressed "
            "(<50% strict-parseable) — same bar as the results table."
        )

    # HARD GUARD: the variance record is a dispersion instrument. If any of
    # its multiruns ever supplies a VALUE in the results table, the two
    # roles have been confused and every number downstream is suspect.
    _leaked = sorted(
        {
            _mr
            for _mr in picked["multirun"].unique()
            if any(_fnmatch.fnmatch(_mr, _g) for _g in VARIANCE_GLOBS)
        }
    )
    assert not _leaked, (
        "variance multiruns leaked into the results table: "
        f"{_leaked}. The variance record supplies DISPERSION ONLY — its "
        "reps use non-canonical seeds, server-mode engines and "
        "post-parity-review code. Remove them from SWEEP_GLOBS "
        f"({SWEEP_GLOBS})."
    )
    variance_obs
    return (variance_obs,)


@app.cell
def _(
    COLUMNS,
    NOISE_MIN_REPS,
    NOISE_USE_COLUMN_MEDIAN_FLOOR,
    REPORT_DIR,
    ROW_ORDER,
    VARIANCE_REP_TYPE,
    variance_obs,
):
    import pandas as _pd

    _COL_IDS = [f"{_c[0]}::{_c[1]}" for _c in COLUMNS]
    _LABEL_OF = {f"{_c[0]}::{_c[1]}": f"{_c[0]} {_c[1]}" for _c in COLUMNS}

    if not len(variance_obs):
        variance_cells = _pd.DataFrame()
        noise_floor = _pd.DataFrame()
        cell_noise = {}
        print("(no variance observations — NO cell is gated, NO band exists)")
    else:
        _g = variance_obs.groupby(
            ["model", "condition", "variant", "era_matched", "col_id", "scale"],
            dropna=False,
        )
        variance_cells = (
            _g["value"]
            .agg(n_reps="count", rep_mean="mean", rep_std="std", rep_min="min", rep_max="max")
            .reset_index()
        )
        _mult = variance_cells["scale"].map(lambda s: 100.0 if s == "pct" else 1.0)
        variance_cells["rep_range_disp"] = (
            variance_cells["rep_max"] - variance_cells["rep_min"]
        ) * _mult
        variance_cells["rep_std_disp"] = variance_cells["rep_std"] * _mult
        variance_cells["rep_mean_disp"] = variance_cells["rep_mean"] * _mult
        variance_cells["rep_type"] = variance_cells["col_id"].map(VARIANCE_REP_TYPE)
        variance_cells["col_label"] = variance_cells["col_id"].map(_LABEL_OF)

        _usable = variance_cells[variance_cells.n_reps >= NOISE_MIN_REPS]
        print(
            f"{len(_usable)}/{len(variance_cells)} variance cells have "
            f"≥{NOISE_MIN_REPS} reps (spread measurable); rep counts "
            f"{_usable.n_reps.value_counts().sort_index().to_dict()}"
        )

        noise_floor = (
            _usable.groupby("col_id")
            .agg(
                rep_type=("rep_type", "first"),
                n_cells=("col_id", "count"),
                median_std=("rep_std_disp", "median"),
                max_std=("rep_std_disp", "max"),
                median_range=("rep_range_disp", "median"),
                max_range=("rep_range_disp", "max"),
            )
            .reset_index()
        )
        noise_floor["col_label"] = noise_floor["col_id"].map(_LABEL_OF)
        noise_floor = noise_floor.sort_values(
            ["rep_type", "median_range"], ascending=[True, False]
        )
        print("\nPer-column noise floor (display units; pct-points, CIRL Net raw):")
        print(
            noise_floor[
                [
                    "col_label",
                    "rep_type",
                    "n_cells",
                    "median_std",
                    "max_std",
                    "median_range",
                    "max_range",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )

        print("\nWidest single cells (max−min across reps, display units):")
        print(
            _usable.nlargest(10, "rep_range_disp")[
                ["model", "condition", "variant", "col_label", "n_reps", "rep_range_disp"]
            ]
            .round(2)
            .to_string(index=False)
        )

        # ── band per results-table cell ───────────────────────────────────
        _own = (
            _usable.groupby(["model", "condition", "col_id"])
            .agg(
                rng=("rep_range_disp", "max"),
                n=("n_reps", "sum"),
                variants=("variant", lambda s: ",".join(sorted(set(s)))),
                era=("era_matched", "any"),
            )
            .to_dict("index")
        )
        _fam = (
            _usable.groupby(["model", "col_id"])
            .agg(
                rng=("rep_range_disp", "max"),
                n=("n_reps", "sum"),
                variants=("variant", lambda s: ",".join(sorted(set(s)))),
            )
            .to_dict("index")
        )
        _colmed = dict(zip(noise_floor["col_id"], noise_floor["median_range"]))

        cell_noise = {}
        _nrows = []
        for _mdl, _conds in ROW_ORDER:
            for _cond in _conds:
                for _cid in _COL_IDS:
                    _rec, _src = None, "unmeasured"
                    if (_mdl, _cond, _cid) in _own:
                        _rec = _own[(_mdl, _cond, _cid)]
                        _src = (
                            "own:same-weights"
                            if _rec["era"]
                            else "own:sibling-ckpts"
                        )
                    elif (_mdl, _cid) in _fam:
                        _rec = _fam[(_mdl, _cid)]
                        _src = "family:other-condition"
                    elif _cid in _colmed:
                        _rec = {
                            "rng": _colmed[_cid],
                            "n": 0,
                            "variants": "",
                        }
                        _src = "column-median"
                    if _rec is None:
                        cell_noise[(_mdl, _cond, _cid)] = (None, "unmeasured", 0, "")
                    else:
                        _band = float(_rec["rng"])
                        # A 3-draw range of 0.0 is an artifact, not silence.
                        if (
                            NOISE_USE_COLUMN_MEDIAN_FLOOR
                            and _src != "column-median"
                            and _cid in _colmed
                        ):
                            _band = max(_band, float(_colmed[_cid]))
                        cell_noise[(_mdl, _cond, _cid)] = (
                            _band,
                            _src,
                            int(_rec["n"]),
                            _rec["variants"],
                        )
                    _b, _s, _n, _v = cell_noise[(_mdl, _cond, _cid)]
                    _nrows.append(
                        {
                            "model": _mdl,
                            "condition": _cond,
                            "column": _LABEL_OF[_cid],
                            "band_disp": _b,
                            "band_source": _s,
                            "n_reps": _n,
                            "variants": _v,
                        }
                    )
        _ndf = _pd.DataFrame(_nrows)
        print("\nband source over all (row × column) slots:")
        print(_ndf.band_source.value_counts().to_string())
        variance_cells.to_parquet(REPORT_DIR / "variance_cells.parquet", index=False)
        noise_floor.to_parquet(REPORT_DIR / "noise_floor.parquet", index=False)
        _ndf.to_parquet(REPORT_DIR / "cell_noise_bands.parquet", index=False)
        print(f"\nsaved variance_cells / noise_floor / cell_noise_bands to {REPORT_DIR}")
    noise_floor
    return cell_noise, noise_floor, variance_cells


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase B4 — Which column leaders survive their own noise?

    For each column: rank the canonical cells (teacher excluded, as
    everywhere), then walk down from the leader collecting every cell whose
    ±half-band interval still overlaps the leader's. That set is the
    column's **tie set** — Phase C bolds all of it, so a column with an
    unresolvable leader can no longer be read as a win. Columns with no
    variance data (the judged PrivacyLens trio) are marked `°` and their
    leader is reported as *nominal*.
    """)
    return


@app.cell
def _(
    BOLD_UNMEASURED_COLUMNS,
    COLUMNS,
    REPORT_DIR,
    SELF_JUDGED_ROWS,
    cell_noise,
    noise_floor,
    picked,
):
    import pandas as _pd

    _specs = {f"{_c[0]}::{_c[1]}": (f"{_c[0]} {_c[1]}", _c[7], _c[8]) for _c in COLUMNS}
    # Structural-N/A cells carry value=None — they are present in `picked` so
    # the table can render N/A, but they are not numbers and must never enter
    # a leader comparison.
    _vals = {
        (_r["model"], _r["condition"], _r["col_id"]): _r["value"]
        for _, _r in picked.iterrows()
        if _r["value"] is not None and not _pd.isna(_r["value"])
    }

    def _band_of(_mdl, _cond, _cid):
        return cell_noise.get((_mdl, _cond, _cid), (None, "unmeasured", 0, ""))[0]

    tie_sets = {}
    col_status = {}
    _lrows = []
    _floor_lut = (
        noise_floor.set_index("col_id").to_dict("index") if len(noise_floor) else {}
    )
    for _cid, (_label, _lo, _scale) in _specs.items():
        _mult = 100.0 if _scale == "pct" else 1.0
        _cands = [
            (_m, _c, _v * _mult)
            for (_m, _c, _k), _v in _vals.items()
            if _k == _cid and _m not in SELF_JUDGED_ROWS
        ]
        if not _cands:
            tie_sets[_cid] = set()
            col_status[_cid] = "empty"
            continue
        _cands.sort(key=lambda t: t[2], reverse=not _lo)
        _lm, _lc, _lv = _cands[0]
        _lb = _band_of(_lm, _lc, _cid)
        if _lb is None:
            # No measured band anywhere in this column: the lead cannot be
            # defended. Keep it visible but never call it separated.
            tie_sets[_cid] = {(_lm, _lc)} if BOLD_UNMEASURED_COLUMNS else set()
            col_status[_cid] = "unmeasured"
            _lrows.append(
                {
                    "column": _label,
                    "status": "unmeasured",
                    "rep_type": "—",
                    "leader": f"{_lm} / {_lc}",
                    "leader_value": round(_lv, 2),
                    "runner_up": (
                        f"{_cands[1][0]} / {_cands[1][1]}" if len(_cands) > 1 else "—"
                    ),
                    "margin": (
                        round(abs(_lv - _cands[1][2]), 2) if len(_cands) > 1 else None
                    ),
                    "band": None,
                    "tie_n": None,
                    # Matches what the table shows: the nominal leader is
                    # still bolded when BOLD_UNMEASURED_COLUMNS is on, so
                    # the report must name it rather than print "—".
                    "tie_set": (
                        f"{_lm} / {_lc} (nominal)"
                        if BOLD_UNMEASURED_COLUMNS
                        else "(nothing bolded)"
                    ),
                }
            )
            continue
        _tie = [(_lm, _lc)]
        for _m, _c, _v in _cands[1:]:
            _b = _band_of(_m, _c, _cid)
            _thresh = (_lb + (_lb if _b is None else _b)) / 2.0
            if abs(_lv - _v) < _thresh:
                _tie.append((_m, _c))
        tie_sets[_cid] = set(_tie)
        col_status[_cid] = "tied" if len(_tie) > 1 else "resolved"
        _rb = _band_of(*_cands[1][:2], _cid) if len(_cands) > 1 else None
        _lrows.append(
            {
                "column": _label,
                "status": col_status[_cid],
                "rep_type": _floor_lut.get(_cid, {}).get("rep_type", "—"),
                "leader": f"{_lm} / {_lc}",
                "leader_value": round(_lv, 2),
                "runner_up": (
                    f"{_cands[1][0]} / {_cands[1][1]}" if len(_cands) > 1 else "—"
                ),
                "margin": (
                    round(abs(_lv - _cands[1][2]), 2) if len(_cands) > 1 else None
                ),
                "band": round((_lb + (_lb if _rb is None else _rb)) / 2.0, 2)
                if len(_cands) > 1
                else round(_lb, 2),
                "tie_n": len(_tie),
                "tie_set": "; ".join(f"{_m} / {_c}" for _m, _c in _tie),
            }
        )
    leader_report = _pd.DataFrame(_lrows)

    _n_tied = sum(1 for _s in col_status.values() if _s == "tied")
    _n_res = sum(1 for _s in col_status.values() if _s == "resolved")
    _n_unm = sum(1 for _s in col_status.values() if _s == "unmeasured")
    print(
        f"column leaders: {_n_res} resolved, {_n_tied} statistically TIED, "
        f"{_n_unm} unmeasurable (judged columns, no judge-free reps)"
    )
    print(
        leader_report[
            ["column", "status", "rep_type", "leader", "leader_value", "margin", "band", "tie_n"]
        ].to_string(index=False)
    )
    if _n_tied:
        print("\nTie sets (all bolded together in the table below):")
        for _, _r in leader_report[leader_report.status == "tied"].iterrows():
            print(f"  {_r['column']}: {_r['tie_set']}")

    # A standalone noise report so the numbers can be cited without re-running
    # the notebook.
    _md = [
        "## Measured re-run noise floor and column-leader resolution",
        "",
        "Source: the judge-free seed/rep variance record "
        "(`multirun/*_eval_judgefree_variance{,_gptoss,_topup}`, 163 arms). "
        "Read for **dispersion only** — no value in the results table comes "
        "from it. Display units: percentage points, except CIRL Net (raw "
        "−1…1).",
        "",
        "### Per-column noise floor",
        "",
        "| Column | Rep type | Cells | Median σ | Max σ | Median range | Max range |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, _r in noise_floor.iterrows():
        _md.append(
            f"| {_r['col_label']} | {_r['rep_type']} | {int(_r['n_cells'])} | "
            f"{_r['median_std']:.2f} | {_r['max_std']:.2f} | "
            f"{_r['median_range']:.2f} | {_r['max_range']:.2f} |"
        )
    _md += [
        "",
        "*`sampled` = reps vary `sampling_params.seed` on a temp-0.2 "
        "benchmark; `greedy` = temp-0 benchmark, so the spread is engine "
        "nondeterminism alone. Range = max−min across a cell's reps.*",
        "",
        "### Column-leader resolution",
        "",
        "| Column | Status | Leader | Value | Runner-up | Margin | Band | Tie set |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, _r in leader_report.iterrows():
        # `band` is NaN (not None) for unmeasured columns once pandas has
        # typed the column as float — test with isna, not `is None`.
        _mg = "—" if _pd.isna(_r["margin"]) else f"{_r['margin']:.2f}"
        _bd = "—" if _pd.isna(_r["band"]) else f"{_r['band']:.2f}"
        _md.append(
            f"| {_r['column']} | **{_r['status']}** | {_r['leader']} | "
            f"{_r['leader_value']} | {_r['runner_up']} | {_mg} | {_bd} | "
            f"{_r['tie_set'] or '—'} |"
        )
    _md += [
        "",
        "*`resolved` = the leader's margin over the runner-up exceeds the "
        "half-band sum, so the two are separable. `tied` = it does not; "
        "every model in the tie set is a co-leader and the column supports "
        "no single winner. `unmeasured` = the variance record is judge-free, "
        "so this judged column has no band at all and its leader is nominal.*",
        "",
        "*Not covered: SFT **training**-seed variance (each SFT row is one "
        "training run; the only training-seed replication in the project is "
        "the 5-seed GRPO sweep). These bands are a lower bound on total "
        "run-to-run variability.*",
        "",
    ]
    (REPORT_DIR / "noise_floor.md").write_text("\n".join(_md))
    leader_report.to_parquet(REPORT_DIR / "leader_report.parquet", index=False)
    print(f"\nsaved noise_floor.md + leader_report.parquet to {REPORT_DIR}")
    leader_report
    return col_status, leader_report, tie_sets


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase C — Assemble the table + markdown

    Lay the cells out in row/column order, format (percentages ×100; CIRL
    Net on −1…1), **bold the Phase-B4 tie set** rather than the bare
    argmax, mark self-judged (†), stale-parser (‡) and unmeasured-noise (°)
    cells, render, and save to `benchmark_results.md` plus a provenance
    parquet.

    Two tables are written: `benchmark_results.md` (clean, for the paper)
    and `benchmark_results_with_noise.md` (every cell as
    `value ±half-band`, for the appendix / rebuttal).
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
    cell_noise,
    col_status,
    picked,
    tie_sets,
):
    import pandas as _pd

    # specs: (col_id, group, col, judged, lower_better, scale)
    _col_specs = [(f"{c[0]}::{c[1]}", c[0], c[1], c[6], c[7], c[8]) for c in COLUMNS]

    _lut = {
        (r["model"], r["condition"], r["col_id"]): r["value"]
        for _, r in picked.iterrows()
        if r["value"] is not None and not _pd.isna(r["value"])
    }
    # Cells the benchmark could not evaluate at all (empty or truncated
    # generations). Rendered N/A — a different claim from "—".
    _na_lut = {
        (r["model"], r["condition"], r["col_id"])
        for _, r in picked.iterrows()
        if bool(r.get("structural_na"))
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

    def _fmt(cid, grp, judged, lo, scale, mdl, cond):
        v = _lut.get((mdl, cond, cid))
        if v is None:
            # Three distinct empty states, never collapsed into one:
            #   N/A   the benchmark ran and the model's output is structurally
            #         unscoreable (empty / truncated generations)
            #   pend. the benchmark has no run yet
            #   —     a metric we deliberately do not report here
            if (mdl, cond, cid) in _na_lut:
                return "N/A"
            return "pend." if grp in PENDING_GROUPS else "—"
        sv = v * 100.0 if scale == "pct" else v
        spec = "{:.1f}" if scale == "pct" else "{:.2f}"
        s = spec.format(sv)
        # Bold the Phase-B4 TIE SET, not the bare argmax: every cell that
        # the measured re-run noise cannot separate from the leader is a
        # co-leader. A single bold cell in a column is therefore a real,
        # defended claim; several mean the column has no winner.
        if (mdl, cond) in tie_sets.get(cid, set()):
            s = f"**{s}**"
        # † self-judged; ‡ finalized before the 2026-07-21 PL parser fix;
        # ° no measured noise band for this column (judge-free variance
        # record cannot cover judged metrics).
        if judged and mdl in SELF_JUDGED_ROWS:
            s = f"{s}†"
        if _stale_lut.get((mdl, cond, cid)):
            s = f"{s}‡"
        return s

    # Column headers carry the leader-resolution status so the caveat cannot
    # be separated from the numbers by a copy-paste.
    _STATUS_MARK = {"tied": "~", "unmeasured": "°", "resolved": "", "empty": ""}
    _hdr_cols = [
        f"{grp} {col}{_STATUS_MARK.get(col_status.get(cid, ''), '')}"
        for (cid, grp, col, judged, lo, scale) in _col_specs
    ]

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
            # emitting an all-blank row sourced from nothing. An N/A cell IS
            # a result (the benchmark ran and the model failed it), so a row
            # carrying one is never dropped.
            if all(c in ("—", "pend.") for c in _cells):
                _dropped.append(f"{mdl} / {cond}")
                continue
            _mcell = mdl if _shown == 0 else ""
            _lines.append(f"| {_mcell} | {cond} | " + " | ".join(_cells) + " |")
            _shown += 1
    table_md = "\n".join(_lines)
    if _dropped:
        print("Dropped (no data for any column): " + ", ".join(_dropped))

    # ── companion table: every cell as `value ±half-band` ─────────────────
    # Same cells, uncertainty made explicit. This is the version to reach
    # for in a rebuttal, where "is this difference real?" is the question.
    def _fmt_band(cid, grp, scale, mdl, cond):
        v = _lut.get((mdl, cond, cid))
        if v is None:
            if (mdl, cond, cid) in _na_lut:
                return "N/A"
            return "pend." if grp in PENDING_GROUPS else "—"
        sv = v * 100.0 if scale == "pct" else v
        spec = "{:.1f}" if scale == "pct" else "{:.2f}"
        band = cell_noise.get((mdl, cond, cid), (None,))[0]
        if band is None:
            return spec.format(sv) + " ±?"
        # Half-bands get an extra digit: at the value's own precision MMLU
        # would print "±0.0", which reads as "noiseless" — the exact
        # misreading this table exists to prevent.
        bspec = "{:.2f}" if scale == "pct" else "{:.3f}"
        return spec.format(sv) + " ±" + bspec.format(band / 2.0)

    _blines = [
        "| Model | Cond. | " + " | ".join(_hdr_cols) + " |",
        "|" + "---|" * (2 + len(_col_specs)),
    ]
    for mdl, conds in ROW_ORDER:
        _shown = 0
        for cond in conds:
            _cells = [
                _fmt_band(cid, grp, scale, mdl, cond)
                for (cid, grp, col, judged, lo, scale) in _col_specs
            ]
            if all(c in ("—", "pend.") for c in _cells):
                continue
            _mcell = mdl if _shown == 0 else ""
            _blines.append(f"| {_mcell} | {cond} | " + " | ".join(_cells) + " |")
            _shown += 1
    band_table_md = (
        "## Benchmark results with measured re-run uncertainty\n\n"
        + "\n".join(_blines)
        + "\n\n*Each cell is `value ±half-band`, where band = the measured "
        "max rep-range for that cell in the judge-free seed/rep variance "
        "record (floored at the column's median range; see "
        "`noise_floor.md`). Two cells are separable only when their "
        "intervals do not overlap. `±?` = no measured band: the variance "
        "record is judge-free, so the three PrivacyLens columns have none. "
        "SFT rows' bands are transferred from post-2026-07-18 "
        "`sft-canonical-ckptNNN` siblings (dispersion transfers, means do "
        "not). SFT **training**-seed variance is not measured anywhere — "
        "these bands are a lower bound.*\n"
    )
    (REPORT_DIR / "benchmark_results_with_noise.md").write_text(band_table_md)

    _legend = (
        "\n\n*Percentages (×100) except **CIRL Net** (utility − leakage, "
        "−1…1). ↓ = lower is better. "
        "**Bold** = the column's statistical tie set (see below), not its "
        "argmax. "
        "GoldCoin Appl./Comp. = upstream-parity accuracy (unparseable "
        "counted as wrong, per the 2026-07-21 denominator flip); pre-flip "
        "metrics files are retro-converted exactly as "
        "accuracy × parseable_rate — see the provenance table. Sub-1pt "
        "GoldCoin gaps are re-run noise (temp 0.2, measured ±0.9pt). "
        "PrivacyLens: QA Acc, Adj Lk = adjusted leakage, Helpful = helpful "
        "rate (both judged, the `*_among_parseable` primary variants). "
        "ConfAIde r = Tier-2b Pearson. CIRL-729: Net = utility − leakage "
        "with every strict-format miss scored −1 per the paper protocol "
        "(recorded, not dropped); Lk/Util are rates over the actions where "
        "the model produced a **complete, scoreable message** (non-empty, "
        "not truncated) — per-cell scoreable fraction in the provenance "
        "`semantics` column. VLM Q7 = location-granularity "
        "accuracy (text-only models blank; gemma-4-E2B/E4B values are "
        "single-class-collapse base rates, not skill). "
        f"All judged columns use {EXPECTED_JUDGE}. Zero-shot = the pre-SFT "
        "`<family>/instruct` checkpoint each SFT adapter was trained from. "
        "Most recent qualifying run per cell.*"
        "\n\n***Bold = statistical tie set, not argmax.** A cell is bolded "
        "when the measured re-run noise cannot separate it from the "
        "column's top value: bands come from the judge-free seed/rep "
        "variance record (163 arms, 3–8 reps per config; "
        "`multirun/*_eval_judgefree_variance*`, read for dispersion only, "
        "never for a value), and two cells count as separated only when "
        "|a−b| ≥ (band_a + band_b)/2. A column with one bold cell has a "
        "defended leader; a column with several has **no winner**. "
        "Per-cell bands and the full derivation are in `noise_floor.md` "
        "and `benchmark_results_with_noise.md`.*"
        "\n\n*~ = the column's leader is **statistically tied** with the "
        "other bolded cells — do not report it as best. ° = the column has "
        "**no measured noise floor**: the variance record is judge-free by "
        "construction, so the judged PrivacyLens columns cannot be gated "
        "and their leader is nominal only.*"
        "\n\n*Bands measure **inference** re-run noise (sampling seed on "
        "temp-0.2 benchmarks; engine nondeterminism on temp-0 ones). They "
        "do NOT measure SFT **training**-seed variance — each SFT row is a "
        "single training run — and SFT bands are transferred from "
        "post-2026-07-18 `sft-canonical-ckptNNN` siblings of the same "
        "family. Treat every band as a lower bound.*"
        "\n\n***N/A vs —.** `N/A` = the benchmark ran and the model's output "
        "is **structurally unscoreable**: fewer than half of the 729 CIRL "
        "actions produced a complete message (empty generations, or "
        "fragments truncated at `max_tokens`). That is a property of the "
        "model on this benchmark, and its paper-protocol score still appears "
        "in CIRL Net. `—` is the different statement that we do not report "
        "the metric for that cell — a documented finding (refusal, format "
        "collapse) or an inapplicable column (Q7 on a text-only model). "
        "Neither is ever a zero.*"
        "\n\n*† = **self-judged**: judge and subject are the same weights — "
        "an optimistic bound, excluded from the tie-set comparison and "
        "never bolded.*"
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
                    nb, nsrc, nreps, nvars = cell_noise.get(
                        (mdl, cond, cid), (None, "unmeasured", 0, "")
                    )
                    _prov_rows.append(
                        {
                            "model": mdl,
                            "condition": cond,
                            "column": f"{grp} {col}",
                            # None for a structural-N/A cell — the reason is
                            # in `semantics` and `structural_na`.
                            "value": _lut.get((mdl, cond, cid)),
                            "structural_na": (mdl, cond, cid) in _na_lut,
                            "judge": j,
                            "judge_src": jsrc,
                            "multirun": mr,
                            "semantics": sem,
                            "stale_parser": stale,
                            "self_judged": bool(judged and mdl in SELF_JUDGED_ROWS),
                            # Variance gate (Phase B3/B4).
                            "noise_band_disp": nb,
                            "noise_band_source": nsrc,
                            "noise_n_reps": nreps,
                            "noise_variants": nvars,
                            "col_leader_status": col_status.get(cid, ""),
                            "in_tie_set": (mdl, cond) in tie_sets.get(cid, set()),
                        }
                    )
    provenance = _pd.DataFrame(_prov_rows)
    provenance.to_parquet(
        REPORT_DIR / "benchmark_results_provenance.parquet", index=False
    )
    print(f"saved table + provenance (with noise bands) to {REPORT_DIR}")
    print(full_md)
    return band_table_md, full_md, provenance


@app.cell
def _(full_md, mo):
    mo.md(full_md)
    return


@app.cell
def _(band_table_md, mo):
    mo.md(band_table_md)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase C2 — LaTeX table

    The same cells rendered as a booktabs `tabular` for direct inclusion in the
    camera-ready (requires the `booktabs` and `graphicx` packages). Bold is the
    **Phase-B4 tie set**, not the argmax; the markers ($\\dagger$ self-judged,
    $\\ddagger$ stale pre-parser-fix, $\\sim$ statistically tied column,
    $\\circ$ no measured noise floor) match the markdown legend. Saved to
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
    col_status,
    picked,
    tie_sets,
):
    # Cell-local names are underscore-prefixed per marimo convention (shared
    # names must be unique across cells). specs: (col_id, group, col, judged,
    # lower_better, scale).
    _col_specs = [
        (f"{_c[0]}::{_c[1]}", _c[0], _c[1], _c[6], _c[7], _c[8]) for _c in COLUMNS
    ]
    import pandas as _pd

    _lut = {
        (_r["model"], _r["condition"], _r["col_id"]): _r["value"]
        for _, _r in picked.iterrows()
        if _r["value"] is not None and not _pd.isna(_r["value"])
    }
    _na_lut = {
        (_r["model"], _r["condition"], _r["col_id"])
        for _, _r in picked.iterrows()
        if bool(_r.get("structural_na"))
    }
    _stale_lut = {
        (_r["model"], _r["condition"], _r["col_id"]): bool(_r["stale"])
        for _, _r in picked.iterrows()
    }

    def _esc(_s):
        # LaTeX-safe model cell: escape underscores; the teacher row carries a
        # unicode em dash, which we render as an en-dash range.
        return _s.replace("_", r"\_").replace("\u2014", "--")

    def _fmt_tex(_cid, _grp, _judged, _lo, _scale, _mdl, _cond):
        _v = _lut.get((_mdl, _cond, _cid))
        if _v is None:
            if (_mdl, _cond, _cid) in _na_lut:
                return r"\textit{N/A}"
            return r"\textit{pend.}" if _grp in PENDING_GROUPS else "--"
        _sv = _v * 100.0 if _scale == "pct" else _v
        _spec = "{:.1f}" if _scale == "pct" else "{:.2f}"
        _s = _spec.format(_sv)
        # Bold the Phase-B4 tie set, identical rule to the markdown table:
        # a bare argmax would assert a lead the measured re-run noise does
        # not support.
        if (_mdl, _cond) in tie_sets.get(_cid, set()):
            _s = rf"\textbf{{{_s}}}"
        if _judged and _mdl in SELF_JUDGED_ROWS:
            _s = _s + r"$^{\dagger}$"
        if _stale_lut.get((_mdl, _cond, _cid)):
            _s = _s + r"$^{\ddagger}$"
        return _s

    # Sub-header status marks, matching the markdown table.
    _TEX_STATUS_MARK = {
        "tied": r"$^{\sim}$",
        "unmeasured": r"$^{\circ}$",
        "resolved": "",
        "empty": "",
    }

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

    # Sub-header row, with the leader-resolution mark on each column so the
    # caveat travels with the table into the paper.
    _sub = ["Model", "Cond."] + [
        # `.replace("$$", "")` merges an adjacent arrow and status mark into
        # one math group ($\downarrow^{\circ}$) instead of two.
        (
            _col.replace("\u2193", r"$\downarrow$")
            + _TEX_STATUS_MARK.get(col_status.get(_cid, ""), "")
        ).replace("$$", "")
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
            # N/A is a result, so a row carrying one is never dropped.
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
        rf"location-granularity accuracy), and MMLU (overall accuracy). "
        rf"\textbf{{Bold marks a statistical tie set, not the argmax:}} a cell "
        rf"is bold when the measured re-run noise cannot separate it from the "
        rf"column's top value, so a column with several bold cells supports no "
        rf"single winner. Noise bands come from a judge-free seed/rep variance "
        rf"record (163 configurations re-run 3--8 times; sampling-seed variation "
        rf"on temperature-0.2 benchmarks, engine nondeterminism on greedy ones), "
        rf"and two cells count as separated only when their half-band intervals "
        rf"are disjoint; per-cell bands are reported in the appendix. The "
        rf"self-judged teacher is excluded from the comparison. "
        rf"Percentages are scaled by 100, except CIRL net score ($-1$ to $1$); "
        rf"$\downarrow$ marks lower-is-better "
        rf"columns. All judged columns use {EXPECTED_JUDGE} as the judge. "
        rf"$\sim$: the column's leader is statistically tied with the other "
        rf"bold cells. $\circ$: no measured noise floor --- the variance record "
        rf"is judge-free, so the judged PrivacyLens columns cannot be gated and "
        rf"their leader is nominal. Bands measure inference-time re-run noise "
        rf"only: SFT training-seed variance is not measured, and SFT bands are "
        rf"transferred from sibling checkpoints of the same family, so every "
        rf"band is a lower bound. "
        rf"$\dagger$: self-judged (judge and subject share weights), an "
        rf"optimistic bound excluded from best-per-column bolding. $\ddagger$: "
        rf"stale, finalized before the 2026-07-21 PrivacyLens judge-response "
        rf"parser fix. CIRL-729 scores every strict-format miss as $-1$ (paper "
        rf"protocol); its leakage and utility rates are computed over the "
        rf"actions for which the model produced a complete, scoreable message "
        rf"(non-empty and not truncated). \textit{{N/A}}: the benchmark ran but "
        rf"the model's output is structurally unscoreable --- fewer than half "
        rf"of the 729 actions yielded a complete message --- which is a "
        rf"property of the model on this benchmark; its paper-protocol score "
        rf"still appears under CIRL Net. --: a metric we do not report for that "
        rf"cell (a documented model-level finding such as a refusal or format "
        rf"collapse, or an inapplicable column). Neither is a zero."
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
    ## Phase D — SFT effect (paired Zero-shot → SFT delta), gated on noise

    Same 11 checkpoints, same benchmarks, same judge, LoRA on/off. Delta is
    signed so **positive = SFT better** on every column (lower-is-better
    columns are negated). Only cells present in *both* conditions are
    differenced — so CIRL Net pairs everywhere, while CIRL Lk/Util pair
    only where both sides cleared the majority-parseable bar (the three
    Gemma families).

    Each delta is then compared against the **combined band of its two
    cells**, (band_zero-shot + band_SFT)/2 — the same separability rule
    Phase B4 applies to column leaders. `sft_delta_real` marks the deltas
    that clear it; everything else is a delta this evidence cannot
    distinguish from re-running the same checkpoint twice. The three
    PrivacyLens columns have no band, so their deltas are reported as
    **unmeasurable** (`<NA>`), never as real.
    """)
    return


@app.cell
def _(COLUMNS, REPORT_DIR, ROW_ORDER, cell_noise, picked):
    import pandas as _pd

    # Structural-N/A cells are not numbers — a delta against one is undefined,
    # so dropping them here leaves the pair blank rather than inventing a 0.
    _lut2 = {
        (r["model"], r["condition"], r["col_id"]): r["value"]
        for _, r in picked.iterrows()
        if r["value"] is not None and not _pd.isna(r["value"])
    }
    _specs = [(f"{c[0]}::{c[1]}", f"{c[0]} {c[1]}", c[7], c[8]) for c in COLUMNS]

    _drows = []
    _brows = []
    _rrows = []
    for _mdl2, _conds2 in ROW_ORDER:
        # Rows that are not a paired Zero-shot/SFT contrast (the teacher
        # reference) have no delta at all — including them would pad the
        # "unmeasurable" headline with 11 cells that were never a comparison.
        if not {"Zero-shot", "SFT"} <= set(_conds2):
            continue
        _row2 = {"model": _mdl2}
        _rowb = {"model": _mdl2}
        _rowr = {"model": _mdl2}
        for _cid2, _label2, _lo2, _scale2 in _specs:
            _z = _lut2.get((_mdl2, "Zero-shot", _cid2))
            _s = _lut2.get((_mdl2, "SFT", _cid2))
            if _z is None or _s is None:
                _row2[_label2] = None
                _rowb[_label2] = None
                _rowr[_label2] = None
                continue
            _mult = 100.0 if _scale2 == "pct" else 1.0
            _d = (_s - _z) * _mult
            _d = -_d if _lo2 else _d
            _row2[_label2] = _d
            _bz = cell_noise.get((_mdl2, "Zero-shot", _cid2), (None,))[0]
            _bs = cell_noise.get((_mdl2, "SFT", _cid2), (None,))[0]
            if _bz is None or _bs is None:
                # No band on at least one side → the delta is UNMEASURABLE.
                # Never fall through to a comparison against 0/NaN, which
                # silently reads as "not real".
                _rowb[_label2] = None
                _rowr[_label2] = None
            else:
                _band = (_bz + _bs) / 2.0
                _rowb[_label2] = _band
                _rowr[_label2] = bool(abs(_d) >= _band)
        _drows.append(_row2)
        _brows.append(_rowb)
        _rrows.append(_rowr)
    sft_delta = _pd.DataFrame(_drows).set_index("model")
    sft_delta_band = _pd.DataFrame(_brows).set_index("model")
    sft_delta_real = _pd.DataFrame(_rrows).set_index("model").astype("boolean")

    print(
        "Paired Zero-shot → SFT delta (positive = SFT better; "
        "lower-is-better columns sign-flipped). Blank = one side missing. "
        "Reminder: PL judged deltas inherit the ‡ staleness of both sides."
    )
    print(sft_delta.round(1).to_string())
    print("\nCombined noise band per delta ((band_zs + band_sft)/2):")
    print(sft_delta_band.round(2).to_string())
    print(
        "\nDelta exceeds its band? True = separable from a re-run, "
        "False = within noise, <NA> = no band (judged column) or one side "
        "missing:"
    )
    print(sft_delta_real.to_string())

    # pandas ≥3 stack() keeps NA rows (and rejects the old dropna kwarg) —
    # the <NA>s here are load-bearing: they are the unmeasurable deltas.
    _flat = sft_delta_real.stack()
    _n_real = int((_flat == True).sum())  # noqa: E712 — pandas BooleanDtype
    _n_noise = int((_flat == False).sum())  # noqa: E712
    _n_na = int(_flat.isna().sum())
    print(
        f"\nHEADLINE: {_n_real} paired SFT deltas exceed the measured "
        f"re-run band, {_n_noise} do NOT (report those as no change), "
        f"{_n_na} are unmeasurable (no band, or one condition missing)."
    )
    print("\nMean delta per column (over models with both conditions):")
    print(sft_delta.mean(axis=0).round(2).to_string())
    print("Of which separable from noise, per column:")
    print(
        _pd.DataFrame(
            {
                "n_real": (sft_delta_real == True).sum(axis=0),  # noqa: E712
                "n_within_noise": (sft_delta_real == False).sum(axis=0),  # noqa: E712
                "n_unmeasurable": sft_delta_real.isna().sum(axis=0),
            }
        ).to_string()
    )
    sft_delta.to_parquet(REPORT_DIR / "sft_delta.parquet")
    sft_delta_band.to_parquet(REPORT_DIR / "sft_delta_band.parquet")
    sft_delta_real.astype("object").to_parquet(
        REPORT_DIR / "sft_delta_real.parquet"
    )
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
    - **CIRL Lk/Util — REPAIRED (2026-08-03).** They now report the
      `*_scorable` rates (leakage/utility over the actions that produced a
      complete message), which fills **19 of 22** canonical cells where the
      old strict bar filled 8. The three that remain are **N/A**, not blank:
      `gpt-oss-20b/sft` (631/729 empty harmony final channels),
      `openthinker3-7b/sft` and `phi-4/sft` (≈55% of rows never terminate —
      **not** a token-budget artifact: a 12288-token probe left truncation
      unchanged at 55% / worse at 65%, with every capped row degenerating
      into a repetition loop, so the repair sweep was abandoned). The swap
      cannot move a
      compliant cell (|Δ| ≤ 0.004 where the format was followed) and the
      re-score of all 160 CIRL cells changed the strict headline on zero of
      them. **CIRL Net is untouched** and remains the paper-parity headline.
    - **Three empty states, never collapsed**: `N/A` = the benchmark ran and
      the model's output is structurally unscoreable; `—` = a metric we do
      not report for that cell (harc ConfAIde refusals, openthinker/gpt-oss
      SFT PrivacyLens format collapse, text-only Q7); `pend.` = no run yet.
      None of the three is a zero.
    - The judge-free variance record (complete, 163 arms) is still excluded
      as a **value** source — `SWEEP_GLOBS` never matches it and Phase B3
      asserts that. Since 2026-08-03 it is read as a **dispersion** source
      and drives the variance gate below.

    ### Reading the variance gate (do this before quoting any "best")

    - **Bold means "co-leader", not "winner".** Phase B4 prints the
      per-column verdict: `resolved` (one bold cell, a defended lead),
      `tied` (several bold cells — the column supports no winner, header
      marked `~`), `unmeasured` (judged PrivacyLens columns, header marked
      `°`, leader nominal only). Quote a column as a win only when it is
      `resolved`.
    - **The SFT story is in Phase D's headline count**, not in the mean
      delta: how many paired Zero-shot→SFT deltas actually clear their
      combined band. Deltas that don't should be described as no change.
    - **Bands are a lower bound on run-to-run variability.** They measure
      inference re-runs (sampling seed at temp 0.2; engine nondeterminism
      at temp 0). Nobody has measured SFT **training**-seed variance for
      these checkpoints, and the SFT bands are borrowed from
      post-2026-07-18 sibling checkpoints of the same family. If a reviewer
      asks for training-seed error bars, the honest answer is that only the
      GRPO stage has them (5 seeds, reward CV 3.5%).
    - The two worst offenders are worth naming when discussing headroom:
      GoldCoin applicability moves up to ~19 pt on a seed change for weak
      instruct models (qwen3.5-2b flips its whole Applicable decision
      policy), and ConfAIde Tier-2b Pearson moves up to ~27 pt **at
      temperature 0** through parse-subset instability. Both are properties
      of the benchmark-plus-model pair, not of our training.
    - Artifacts: `noise_floor.md` (per-column floor + leader resolution),
      `benchmark_results_with_noise.md` (`value ±half-band` table),
      `variance_cells.parquet`, `cell_noise_bands.parquet`,
      `leader_report.parquet`, `sft_delta_{band,real}.parquet`.
    """)
    return


if __name__ == "__main__":
    app.run()
