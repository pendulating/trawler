"""Aggregate sanity + provenance signals across a pipeline run.

The orchestrator writes ``pipeline_manifest.json`` with each node's
``metadata`` (which carries the per-stage ``sanity`` block populated by
:func:`dagspaces.common.runners.sanity.log_sanity_to_context`) and
``metrics.json`` (which carries ``metric_provenance``). This module
walks both, produces a single ``health_summary.json`` per pipeline,
and exposes a CLI:

    python -m dagspaces.common.health_summary <run_dir>

The CLI prints a colored table so you can eyeball before-paper-numbers
runs without grepping a tree of JSON files.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def collect_health(run_dir: str) -> Dict[str, Any]:
    """Walk ``run_dir`` for pipeline_manifest.json + metrics.json files.

    Returns a dict with two top-level lists:

    * ``sanity`` — one entry per (benchmark, stage) sanity report, with
      ``n_warnings``, ``n_failures``, ``halted``, ``warnings``,
      ``failures``, and the per-stage ``metrics`` dict.
    * ``metric_provenance`` — one entry per (benchmark, metric_name),
      with ``n_total``, ``n_real``, ``n_defaulted``, ``defaulted_rate``,
      ``default_reason``.

    Plus rolled-up ``totals``: number of fail-tier sanity entries, max
    ``defaulted_rate`` across any metric, list of halted stages.
    """
    run_dir = os.path.abspath(run_dir)
    sanity_entries: List[Dict[str, Any]] = []
    provenance_entries: List[Dict[str, Any]] = []

    # Manifests can live at varying depths (eval_all/<bench>/<dagspace>/...)
    # — glob recursively.
    manifest_paths = glob.glob(
        os.path.join(run_dir, "**", "pipeline_manifest.json"), recursive=True
    )
    metrics_paths = glob.glob(
        os.path.join(run_dir, "**", "metrics.json"), recursive=True
    )

    for mp in manifest_paths:
        try:
            m = json.load(open(mp))
        except Exception as exc:
            print(f"[health] skip manifest {mp}: {exc}", file=sys.stderr)
            continue
        bench = _bench_from_manifest_path(mp, run_dir)
        nodes = m.get("nodes", {}) or {}
        if not isinstance(nodes, dict):
            continue
        for node_name, node in nodes.items():
            md = (node.get("metadata") or {})
            sanity = md.get("sanity") or {}
            for stage_name, info in sanity.items():
                # Schema migration (2026-04-27): pre-migration manifests stored
                # the failure-row COUNT under ``n_failures`` and had no concept
                # of fail-severity warnings. Post-migration, ``n_failures`` is
                # the count of fail-tier warnings and ``n_failure_rows`` is the
                # row count. Detect the schema by presence of ``n_failure_rows``
                # so old runs render correctly.
                has_new_schema = (
                    "n_failure_rows" in info
                    or "halted" in info
                    or "failures" in info
                )
                if has_new_schema:
                    n_failures = int(info.get("n_failures", 0) or 0)
                    n_failure_rows = int(info.get("n_failure_rows", 0) or 0)
                    failures_list = list(info.get("failures", []) or [])
                else:
                    # Legacy: n_failures was the row count.
                    n_failures = 0
                    n_failure_rows = int(info.get("n_failures", 0) or 0)
                    failures_list = []
                sanity_entries.append({
                    "benchmark": bench,
                    "node": node_name,
                    "stage": stage_name,
                    "manifest": mp,
                    "schema": "v2" if has_new_schema else "v1_legacy",
                    "n_warnings": int(info.get("n_warnings", 0) or 0),
                    "n_failures": n_failures,
                    "n_failure_rows": n_failure_rows,
                    "halted": bool(info.get("halted", False)),
                    "warnings": list(info.get("warnings", []) or []),
                    "failures": failures_list,
                    "metrics": dict(info.get("metrics", {}) or {}),
                })

    for mp in metrics_paths:
        try:
            m = json.load(open(mp))
        except Exception as exc:
            print(f"[health] skip metrics {mp}: {exc}", file=sys.stderr)
            continue
        bench = _bench_from_metrics_path(mp, run_dir)
        prov = m.get("metric_provenance") or {}
        if not isinstance(prov, dict):
            continue
        for metric_name, p in prov.items():
            if not isinstance(p, dict):
                continue
            provenance_entries.append({
                "benchmark": bench,
                "metric": metric_name,
                "metrics_json": mp,
                "n_total": int(p.get("n_total", 0) or 0),
                "n_real": int(p.get("n_real", 0) or 0),
                "n_defaulted": int(p.get("n_defaulted", 0) or 0),
                "defaulted_rate": float(p.get("defaulted_rate", 0.0) or 0.0),
                "default_reason": p.get("default_reason"),
            })

    totals = {
        "run_dir": run_dir,
        "n_sanity_reports": len(sanity_entries),
        "n_warn_stages": sum(1 for s in sanity_entries if s["n_warnings"] > 0),
        "n_fail_stages": sum(1 for s in sanity_entries if s["n_failures"] > 0),
        "n_halted_stages": sum(1 for s in sanity_entries if s["halted"]),
        "halted_stages": [
            f"{s['benchmark']}.{s['stage']}" for s in sanity_entries if s["halted"]
        ],
        "n_provenance_records": len(provenance_entries),
        "max_defaulted_rate": max(
            (p["defaulted_rate"] for p in provenance_entries), default=0.0
        ),
        "high_default_metrics": [
            f"{p['benchmark']}.{p['metric']} ({p['defaulted_rate']:.2%}, {p['default_reason']})"
            for p in provenance_entries
            if p["defaulted_rate"] > 0.05
        ],
    }
    return {
        "totals": totals,
        "sanity": sanity_entries,
        "metric_provenance": provenance_entries,
    }


def _bench_from_manifest_path(manifest_path: str, run_dir: str) -> str:
    rel = os.path.relpath(manifest_path, run_dir)
    parts = [p for p in rel.split(os.sep) if p]
    # Best-effort: take the first directory component as the bench name.
    return parts[0] if parts else "unknown"


def _bench_from_metrics_path(metrics_path: str, run_dir: str) -> str:
    rel = os.path.relpath(metrics_path, run_dir)
    parts = [p for p in rel.split(os.sep) if p]
    return parts[0] if parts else "unknown"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def write_health_summary(run_dir: str, *, output_path: Optional[str] = None) -> str:
    """Write ``health_summary.json`` next to (or inside) ``run_dir``.

    Returns the path written.
    """
    summary = collect_health(run_dir)
    if output_path is None:
        output_path = os.path.join(run_dir, "health_summary.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _color(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


def _print_table(summary: Dict[str, Any]) -> None:
    totals = summary["totals"]
    sanity = summary["sanity"]
    prov = summary["metric_provenance"]

    print("=" * 72)
    print(f"  Evaluation Health  —  {totals['run_dir']}")
    print("=" * 72)

    color = "32" if totals["n_fail_stages"] == 0 else "31"
    print(f"  sanity: {totals['n_sanity_reports']} reports  "
          f"warn={totals['n_warn_stages']}  "
          f"{_color(f'fail={totals['n_fail_stages']}', color)}  "
          f"halted={totals['n_halted_stages']}")

    color2 = "32" if totals["max_defaulted_rate"] <= 0.05 else "33" if totals["max_defaulted_rate"] <= 0.5 else "31"
    print(f"  provenance: {totals['n_provenance_records']} metrics  "
          f"max_defaulted_rate={_color(f'{totals['max_defaulted_rate']:.2%}', color2)}")

    if totals["halted_stages"]:
        print()
        print(_color("  HALTED STAGES:", "31"))
        for s in totals["halted_stages"]:
            print(f"    - {s}")

    if totals["high_default_metrics"]:
        print()
        print(_color("  HIGH-DEFAULT METRICS (defaulted_rate > 5%):", "33"))
        for m in totals["high_default_metrics"]:
            print(f"    - {m}")

    # Per-stage sanity table
    if sanity:
        print()
        print("  Per-stage sanity (W/F = warn/fail):")
        for s in sanity:
            tag = ""
            if s["n_failures"]:
                tag = _color("[FAIL]", "31")
            elif s["n_warnings"]:
                tag = _color("[WARN]", "33")
            else:
                tag = _color("[ ok ]", "32")
            print(f"    {tag} {s['benchmark']}.{s['stage']:<32} "
                  f"W={s['n_warnings']} F={s['n_failures']} "
                  f"rows_failed={s['n_failure_rows']}")
            for w in s["failures"]:
                print(f"           - {_color(w, '31')}")
            for w in s["warnings"]:
                print(f"           - {_color(w, '33')}")

    # Top-N highest-defaulted-rate metrics
    if prov:
        print()
        print("  Highest-defaulted-rate metrics:")
        top = sorted(prov, key=lambda p: -p["defaulted_rate"])[:10]
        for p in top:
            if p["defaulted_rate"] == 0:
                break
            print(f"    {p['defaulted_rate']:>6.2%}  "
                  f"{p['benchmark']}.{p['metric']:<60} "
                  f"({p['n_real']}/{p['n_total']}, {p['default_reason']})")

    print("=" * 72)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="health_summary",
        description="Aggregate sanity + provenance signals across a pipeline run.",
    )
    parser.add_argument("run_dir", help="Pipeline run directory")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write health_summary.json into <run_dir>/.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override path for health_summary.json (implies --write).",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print the JSON summary instead of the table.",
    )
    args = parser.parse_args(argv)

    summary = collect_health(args.run_dir)
    if args.write or args.output:
        path = write_health_summary(args.run_dir, output_path=args.output)
        print(f"[health] wrote {path}", file=sys.stderr)
    if args.json_only:
        json.dump(summary, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        _print_table(summary)
    # Exit 1 if any FAIL or any defaulted_rate > 50%, so this can gate
    # downstream automation.
    if summary["totals"]["n_fail_stages"] or summary["totals"]["max_defaulted_rate"] > 0.5:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
