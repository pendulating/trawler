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
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf


def sanity_overrides(cfg: Any) -> "tuple[Optional[Dict[str, float]], Optional[List[str]]]":
    """Pull per-benchmark sanity overrides from cfg, if any.

    Looks at ``cfg.sanity.thresholds`` (dict like ``{parseable_rate:lt: 0.9}``)
    and ``cfg.sanity.refusal_patterns`` (list[str]). Both are optional;
    omitted keys fall back to ``DEFAULT_THRESHOLDS`` /
    ``DEFAULT_REFUSAL_PATTERNS`` in eval_sanity.
    """
    thresholds: Optional[Dict[str, float]] = None
    patterns: Optional[List[str]] = None
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
    except Exception:
        pass
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
    metadata: Dict[str, Any],
) -> None:
    """Log a SanityReport via context.logger and fold a compact summary
    into the stage's StageResult.metadata.

    Sanity logging never fails the pipeline — exceptions are caught and
    printed. The metadata fold is keyed by ``report.stage`` so multiple
    sanity reports per stage (e.g. judge + parse) coexist.
    """
    try:
        if context.logger is not None:
            context.logger.log_sanity_report(report)
    except Exception as exc:
        print(f"[sanity] log failure for {getattr(report, 'stage', '?')}: {exc}", flush=True)

    metadata.setdefault("sanity", {})
    metadata["sanity"][report.stage] = {
        "metrics": dict(getattr(report, "metrics", {}) or {}),
        "n_warnings": len(getattr(report, "warnings", []) or []),
        "n_failures": int(len(getattr(report, "failure_rows", []) or [])),
        "failures_dropped": int(getattr(report, "failures_dropped", 0) or 0),
        "warnings": [w.message() for w in (getattr(report, "warnings", []) or [])],
    }
