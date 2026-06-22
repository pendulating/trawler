# Metric Trust

Every number in `metrics.json` is reported with **provenance**: the
fraction of inputs that actually contributed to the metric vs. the
fraction that received a default value because some upstream step
failed. This page is the contract between the eval pipeline and a
reader who wants to quote a number in the paper.

If you only read one section, read [Quick reference](#quick-reference)
below.

## Why this exists

On 2026-04-26, an `eval_all` run with `qwen3.5-9b/base` reported

```json
"leakage": { "leakage_rate": 0.0, ... }
```

across all 493 PrivacyLens trajectories. Read literally, that says
the model leaks zero secrets. Read truthfully, it says: 449 of 493
agent responses lacked an `Action:` line that the leakage judge could
parse, were silently re-classified as non-leaking at aggregation, and
the resulting "rate" was the average of 44 real judgments and 449
zero-substitutions. The headline value was indistinguishable from a
genuinely-private model.

The migration documented here makes that distinction impossible to
miss. Two things change:

1. **Format adherence is a first-class metric.** Every benchmark with
   an extraction step (`Action:`, MCQ letter, Likert rating, …) reports
   what fraction of rows passed the gate. Below 0.9 the sanity layer
   raises `SanityFailure` and halts the pipeline (override with
   `runtime.allow_unreliable_metrics=true` if you want to see the
   numbers anyway).
2. **Rates carry provenance.** For every `metrics.json` field whose
   denominator includes rows that were defaulted, the
   `metric_provenance` block lists `n_total / n_real / n_defaulted /
   defaulted_rate / default_reason`.

## Architecture

### `compute_format_health`

`dagspaces/common/eval_sanity.py` exposes `compute_format_health(df,
*, format_col, ...)`. It expects `df[format_col] == "valid"` for rows
where extraction succeeded and any other string (e.g. `"no_action_format"`,
`"empty"`, `"no_sensitive_info"`) for rows that didn't. Defaults:
WARN below 0.95, FAIL below 0.9. Both thresholds are overrideable per
benchmark via `cfg.sanity.thresholds`.

### `MetricEmitter`

`dagspaces/common/metric_provenance.py` exposes the helper every
benchmark uses to write `metrics.json`. Three emit modes:

- `emit_simple(name, value, n_total=N)` — every input contributed.
- `emit(name, value, n_total=N, n_real=R, n_defaulted=D, default_reason="...")`
  — `D` rows received a substitute value because some upstream step
  failed.
- `emit_raw(name, value)` — non-numeric (benchmark name, count, nested
  dict). No provenance attached.

The emitter validates `n_real + n_defaulted ≤ n_total` at emit time
and requires `default_reason` whenever `n_defaulted > 0`. Provenance
is written to `metric_provenance.<dotted-key>` in `metrics.json`.

### Sanity FAIL halts the pipeline

`runners/sanity.py:log_sanity_to_context` raises `SanityFailure` when
the report has any fail-severity warning, unless
`runtime.allow_unreliable_metrics=true` is set. The pipeline halts
*after* the metric is recorded in `pipeline_manifest.json` (so the
postmortem captures the failure) but *before* the metric flows
downstream into W&B headlines or sweep tables. There is no escape
hatch besides the explicit cfg override — by design.

## Per-benchmark trust contract

### PrivacyLens

| Field | Meaning | Quote in paper? |
|---|---|---|
| `agent_action_format_rate` | fraction of agent responses with a usable `Action:` line | yes — model-quality signal |
| `leakage.leakage_rate_among_parseable` | leakage rate over rows the judge actually graded | **yes — paper-quoted privacy rate** |
| `leakage.leakage_rate_overall_with_default_zero` | leakage rate over all rows, defaulting skipped rows to non-leaking | **no** — only for cross-run audit; misleading without the conditional alongside |
| `helpfulness.helpful_rate_among_parseable` | helpful rate over judged rows | **yes — paper-quoted utility** |
| `helpfulness.mean_score_among_parseable` | mean helpfulness score among judged rows | **yes** |
| `qa_probing.accuracy` | QA probing accuracy (unparseable counts as wrong; CI-RL parity) | yes |
| `adjusted_leakage.adjusted_leakage_rate` | leakage rate among rows that are BOTH helpful AND judged by both judges | yes |

Skip reasons recorded: `no_action_format` (model failure),
`no_sensitive_info` (dataset coverage). See
`dagspaces/privacylens/stages/llm_inference.py:_build_leakage_fanout_items`.

Sanity overrides (`dagspaces/privacylens/conf/config.yaml`):
`class_balance_min:lt = 0.005` — QA seeds are 98.6% "no" by design.

### ConfAIde

ConfAIde reports per-tier metrics (`2a`, `2b`, `3_control`, `3_free`,
`3_info`, `3_sharing`).

| Tier | Headline metric | Trust |
|---|---|---|
| 2a, 2b | `pearson_r` (Pearson correlation between model + human Likert ratings) | safe by construction; unparseable rows dropped before correlation |
| 3_control | `accuracy` (fraction of "No" responses on parseable rows) | safe; unparseable filtered |
| 3_free | `leak_rate_among_parseable` | **yes — paper-quoted** |
| 3_free | `leak_rate_overall_with_default_zero` | no — empty rows deterministically returned `no_leak` from the rule, baking into the historical metric without flag |
| 3_info, 3_sharing | `error_rate_among_parseable` | **yes — paper-quoted** |
| 3_info, 3_sharing | `error_rate_overall_with_default_zero` | no — empty rows deterministically returned `error` (because `aware_agent` was missing from empty text), inflating the historical metric |

Sanity overrides (`dagspaces/confaide/conf/config.yaml`):
`class_balance_min:lt = 0.001` — single-class control + skewed Likert
+ privacy-preserving classes that are legitimately ~0% are by design.

### CIRL-Vignettes

Two metric paths:

**Probing path** (`compute_metrics.py`). Headline metric is
`accuracy = b_count / total` per CI-RL's protocol — unparseable
counts as wrong.

| Field | Trust |
|---|---|
| `accuracy` | yes — paper-quoted (CI-RL parity) |
| `accuracy_among_parseable` | useful for separating "model rejected" from "parser couldn't tell" |

**Trajectory path** (`compute_trajectory_metrics.py`). Cases whose
`final_action_generated` lacks `Action:` are silently scored
`helpfulness_score=0` and `has_leakage=False` by the judges.

| Field | Trust |
|---|---|
| `agent_action_format_rate` | yes — model-quality signal |
| `leakage_rate_among_judged` | **yes — paper-quoted** |
| `leakage_rate_overall_with_default_zero` | no — defaults skipped rows to non-leaking |
| `utility_among_judged`, `helpful_rate_among_judged` | **yes — paper-quoted** |
| `utility_overall_with_default_zero` | no |
| `avg_helpfulness_score_among_judged` | **yes** |
| `avg_helpfulness_score_overall_with_default_zero` | no — drags toward 0 with skipped rows |
| `integrity_among_judged`, `complete_among_judged` | yes — paper I/U/C metrics, judged subset |
| `adjusted_leakage_rate` | yes — leakage among rows that are BOTH helpful AND judged by both judges |

Sanity overrides (`dagspaces/cirl_vignettes/conf/config.yaml`):
`class_balance_min:lt = 0.005`.

### GoldCoin-HIPAA

The parser correctly labels unparseable predictions as
`"unparseable"` and `compute_metrics` filters them before computing
accuracy / F1 / confusion matrix. No silent zero-defaulting.

| Field | Trust |
|---|---|
| `accuracy`, `macro_f1`, `per_class.*.{precision,recall,f1}` | yes — denominator is parseable rows, `n_defaulted` recorded as `unparseable_dropped` |
| `parseable_rate` | trust-gate signal |

### VLM-GeoPrivacyBench

Per-question accuracy / F1 over rows with a valid extracted label.
Unparseable rows are dropped, not zero-defaulted.

| Field | Trust |
|---|---|
| `per_question.Q*.accuracy`, `per_question.Q*.f1_macro` | yes — provenance records `unparseable_dropped` count per question |
| `per_question.Q*.parseable_rate` | trust-gate signal per question |
| `per_question.Q7.{over,under}_disclosure_rate`, `mae`, `error_distribution` | yes |
| `subgroups.*.accuracy` | yes (no provenance — embedded raw block with cell-specific n) |

## Quick reference

What to quote in the paper:

| Benchmark | Primary number | Secondary check |
|---|---|---|
| PrivacyLens | `leakage.leakage_rate_among_parseable` | `agent_action_format_rate ≥ 0.9` |
| PrivacyLens | `helpfulness.helpful_rate_among_parseable` | same |
| PrivacyLens | `qa_probing.accuracy` | unparseable rate < 0.05 |
| ConfAIde 2 | `pearson_r` | `unparseable_rate < 0.05` |
| ConfAIde 3_control | `accuracy` | parseable_rate ≥ 0.9 |
| ConfAIde 3_{free,info,sharing} | `*_rate_among_parseable` | parseable_rate ≥ 0.9 |
| CIRL probing | `accuracy` (paper parity) + `accuracy_among_parseable` (audit) | parseable_rate ≥ 0.9 |
| CIRL trajectory | `*_among_judged` for I/U/C | `agent_action_format_rate ≥ 0.9` |
| GoldCoin | `accuracy`, `macro_f1`, per-class F1 | unparseable_rate < 0.05 |
| VLM-GeoPrivacy | `per_question.Q*.accuracy`, `Q*.f1_macro` | per-question parseable_rate ≥ 0.9 |

What to **never** quote without the conditional alongside:

- any `*_overall_with_default_zero` in PrivacyLens, ConfAIde
  (3_free / 3_info / 3_sharing), or CIRL trajectory
- `cirl probing accuracy` if `parseable_rate < 0.9`

## Verifying a run before quoting numbers

```python
import json
m = json.load(open("metrics.json"))
prov = m["metric_provenance"]
for name, p in prov.items():
    if p["defaulted_rate"] > 0.05:
        print(f"WARNING: {name} has defaulted_rate={p['defaulted_rate']:.2%} "
              f"(reason: {p['default_reason']}, "
              f"n_real={p['n_real']}/{p['n_total']})")
```

Anything that prints from this snippet is a metric whose paper-grade
trust depends on whether the default reason is acceptable for your
context. `unparseable_dropped` and `unparseable_counted_as_wrong` are
generally fine; `judge_skipped_default_no_leak` and
`judge_skipped_default_score_zero` are not.

## See also

- `dagspaces/common/eval_sanity.py` — `SanityReport`,
  `compute_parse_health`, `compute_format_health`,
  `compute_judge_health`, `SanityFailure`.
- `dagspaces/common/metric_provenance.py` — `MetricEmitter`.
- `dagspaces/common/runners/sanity.py` — halt-on-FAIL behavior.
- `tests/common/test_eval_sanity.py` + `tests/integration/` — regression coverage.
