"""Shared sanity-logging helpers for parse-stage runners.

Each benchmark's parse runner does the same three things after parsing:

1. Pull per-benchmark sanity overrides (refusal patterns, threshold
   adjustments) from cfg.sanity.* if present.
2. Resolve a best-effort task-LLM identifier for the failure-row
   ``model`` column.
3. Log the SanityReport via context.logger and fold a compact summary
   into StageResult.metadata so the orchestrator's pipeline_manifest
   captures the headline sanity numbers without rerunning the parquet.

Extracted from privacylens runners so all five judged / parsed
dagspaces can call the same code instead of copy-pasting.

When a SanityReport contains ``fail``-severity warnings, the helper
raises :class:`SanityFailure` after persisting the report to logger +
metadata. This halts the pipeline so unreliable metrics never silently
flow into ``metrics.json``. Override for a single run with
``runtime.allow_unreliable_metrics=true``.
"""

from __future__ import annotations

import sys

from typing import Any

from omegaconf import OmegaConf

from dagspaces.common.eval_sanity import SanityFailure


def sanity_overrides(cfg: Any) -> tuple[dict[str, float] | None, list[str] | None]:
    """Pull per-benchmark sanity overrides from cfg, if any.

    Looks at ``cfg.sanity.thresholds`` (dict like ``{parseable_rate:lt: 0.9}``)
    and ``cfg.sanity.refusal_patterns`` (list[str]). Both are optional;
    omitted keys fall back to ``DEFAULT_THRESHOLDS`` /
    ``DEFAULT_REFUSAL_PATTERNS`` in eval_sanity.
    """
    thresholds: dict[str, float] | None = None
    patterns: list[str] | None = None
    try:
        sanity_cfg = OmegaConf.select(cfg, "sanity")
        if sanity_cfg is not None:
            t = OmegaConf.select(sanity_cfg, "thresholds")
            if t is not None:
                thresholds = {
                    str(k): float(v)
                    for k, v in OmegaConf.to_container(t, resolve=True).items()
                }
            p = OmegaConf.select(sanity_cfg, "refusal_patterns")
            if p is not None:
                patterns = [str(x) for x in OmegaConf.to_container(p, resolve=True)]
    except Exception as e:
        # Do NOT swallow silently. Falling back to the defaults here WEAKENS
        # the health gate — a malformed `sanity.thresholds` would otherwise
        # look exactly like "no overrides configured", and the stage would
        # pass checks the operator believed they had tightened.
        print(
            f"[sanity] could not read the sanity overrides from the config, "
            f"so the DEFAULT thresholds apply: {type(e).__name__}: {e}",
            file=sys.stderr,
            flush=True,
        )
    return thresholds, patterns


def task_model_name(cfg: Any) -> str:
    """Best-effort task-LLM identifier for failure-row attribution.

    Tries cfg.model.model_source, cfg.model.served_model_name,
    cfg.model.model_family in that order. Returns empty string if
    nothing usable is set.
    """
    try:
        for key in ("model.model_source", "model.served_model_name", "model.model_family"):
            v = OmegaConf.select(cfg, key)
            if v:
                return str(v)
    except Exception:
        pass
    return ""


def log_sanity_to_context(
    context: Any,
    report: Any,
    *,
    metadata: dict[str, Any],
) -> None:
    """Log a SanityReport via context.logger and fold a compact summary
    into the stage's StageResult.metadata.

    Logging itself never fails the pipeline — exceptions from
    ``log_sanity_report`` are caught and printed. The metadata fold is
    keyed by ``report.stage`` so multiple sanity reports per stage (e.g.
    judge + parse) coexist.

    If the report contains any ``fail``-severity threshold violations,
    the metadata is still recorded (so the failure is visible in the
    manifest), then :class:`SanityFailure` is raised. Override with
    ``cfg.runtime.allow_unreliable_metrics=true`` (escape hatch — only
    for triaging known-broken runs).
    """
    try:
        if context.logger is not None:
            context.logger.log_sanity_report(report)
    except Exception as exc:
        print(f"[sanity] log failure for {getattr(report, 'stage', '?')}: {exc}", flush=True)

    metadata.setdefault("sanity", {})
    failure_rows = getattr(report, "failure_rows", None)
    n_failure_rows = int(len(failure_rows)) if failure_rows is not None else 0
    all_warnings = list(getattr(report, "warnings", None) or [])
    fail_warnings = [w for w in all_warnings if getattr(w, "severity", "warn") == "fail"]
    warn_warnings = [w for w in all_warnings if getattr(w, "severity", "warn") == "warn"]
    metadata["sanity"][report.stage] = {
        "metrics": dict(getattr(report, "metrics", {}) or {}),
        "n_warnings": len(warn_warnings),
        "n_failures": len(fail_warnings),
        "n_failure_rows": n_failure_rows,
        "failures_dropped": int(getattr(report, "failures_dropped", 0) or 0),
        "warnings": [w.message() for w in warn_warnings],
        "failures": [w.message() for w in fail_warnings],
        "halted": False,  # rewritten below if we raise
    }

    if not fail_warnings:
        return

    # Fail-tier violations present. Halt the pipeline unless the user
    # has explicitly demoted fails to warnings for this run.
    allow_unreliable = False
    try:
        cfg = getattr(context, "cfg", None)
        if cfg is not None:
            v = OmegaConf.select(cfg, "runtime.allow_unreliable_metrics")
            allow_unreliable = bool(v) if v is not None else False
    except Exception:
        allow_unreliable = False

    if allow_unreliable:
        print(
            f"[sanity] {report.dagspace}.{report.stage}: {len(fail_warnings)} "
            f"fail-tier threshold(s) crossed — DEMOTED to warnings via "
            f"runtime.allow_unreliable_metrics=true. Metrics from this stage "
            f"are NOT trustworthy.",
            flush=True,
        )
        return

    metadata["sanity"][report.stage]["halted"] = True
    raise SanityFailure(report.dagspace, report.stage, fail_warnings)
