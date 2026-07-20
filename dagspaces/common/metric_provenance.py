"""Metric provenance helper.

Every numeric metric written to ``metrics.json`` should carry provenance:

* ``n_total``: how many rows entered the metric computation
* ``n_real``: how many rows actually contributed (i.e., upstream
  extraction + judging both succeeded — the count behind the rate)
* ``n_defaulted``: rows where a default value was substituted because
  some upstream step failed (e.g., format extraction returned no
  ``Action:`` line, judge returned an unparseable label)
* ``default_reason``: short slug categorizing the substitution
  (``no_action_format``, ``judge_unparseable``, …)

Without provenance, a reader of ``metrics.json`` cannot tell whether
``leakage_rate=0.0`` reflects real privacy preservation (judged 493
rows, 0 leaked) or silent corruption (judged 44 rows, 0 leaked,
449 rows defaulted to non-leaking because format extraction failed).

Output layout::

    {
      "benchmark": "PrivacyLens",         # via emit_raw
      "leakage": {                         # nested via dotted emit
        "leakage_rate_among_parseable": 0.0,
        "leakage_rate_overall_with_default_zero": 0.0,
        ...
      },
      "metric_provenance": {
        "leakage.leakage_rate_among_parseable": {
          "n_total": 493, "n_real": 44, "n_defaulted": 449,
          "defaulted_rate": 0.9107, "default_reason": "no_action_format"
        },
        ...
      }
    }

The dotted-key in ``metric_provenance`` mirrors the nesting in metrics
so a reader can navigate from one to the other unambiguously.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class MetricRecord:
    """One metric's provenance record."""

    name: str
    value: Any
    n_total: int
    n_real: int
    n_defaulted: int = 0
    default_reason: str | None = None

    def to_provenance_dict(self) -> dict[str, Any]:
        defaulted_rate = (
            float(self.n_defaulted) / float(self.n_total)
            if self.n_total > 0
            else 0.0
        )
        return {
            "n_total": int(self.n_total),
            "n_real": int(self.n_real),
            "n_defaulted": int(self.n_defaulted),
            "defaulted_rate": round(defaulted_rate, 6),
            "default_reason": self.default_reason,
        }


class MetricEmitter:
    """Collect metrics + provenance, write to ``metrics.json``.

    Use ``emit`` for any rate / aggregate that could have been computed
    over a subset of inputs (because some inputs failed extraction or
    judging). Use ``emit_simple`` when every input contributed (e.g.,
    accuracy of a parser that never silently defaults). Use ``emit_raw``
    for non-metric fields like ``"benchmark": "PrivacyLens"`` or count
    totals where provenance is meaningless.

    The emitter enforces ``n_real + n_defaulted <= n_total`` at emit
    time, so a callsite that miscounts trips immediately rather than
    quietly producing a misleading ``defaulted_rate``.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, Any] = {}
        self._provenance: dict[str, MetricRecord] = {}

    # -- emit -------------------------------------------------------------

    def emit(
        self,
        name: str,
        value: Any,
        *,
        n_total: int,
        n_real: int,
        n_defaulted: int = 0,
        default_reason: str | None = None,
    ) -> None:
        """Emit a numeric metric with full provenance.

        Args:
            name: Dotted path. ``"leakage.leakage_rate_among_parseable"``
                stores the value at ``metrics["leakage"]["leakage_rate_among_parseable"]``
                and the provenance at the same dotted key.
            value: The metric value.
            n_total: Total inputs entering the computation.
            n_real: Inputs that contributed to ``value``.
            n_defaulted: Inputs that were substituted with a default.
            default_reason: Short slug for the substitution reason.
        """
        if n_real < 0 or n_defaulted < 0 or n_total < 0:
            raise ValueError(
                f"emit({name!r}): n_total/n_real/n_defaulted must all be >= 0"
            )
        if n_real + n_defaulted > n_total:
            raise ValueError(
                f"emit({name!r}): n_real+n_defaulted={n_real + n_defaulted} > n_total={n_total}"
            )
        if n_defaulted > 0 and not default_reason:
            raise ValueError(
                f"emit({name!r}): n_defaulted={n_defaulted} > 0 requires default_reason"
            )
        self._set_nested(name, value)
        self._provenance[name] = MetricRecord(
            name=name,
            value=value,
            n_total=int(n_total),
            n_real=int(n_real),
            n_defaulted=int(n_defaulted),
            default_reason=default_reason,
        )

    def emit_simple(self, name: str, value: Any, *, n_total: int) -> None:
        """Emit a metric where every input row contributed (no defaults)."""
        self.emit(name, value, n_total=n_total, n_real=n_total, n_defaulted=0)

    def emit_raw(self, name: str, value: Any) -> None:
        """Set a nested key without any provenance entry.

        Use for non-metric fields (benchmark name, raw counts) or for
        nested groupings that themselves contain emitted metrics.
        """
        self._set_nested(name, value)

    # -- internal ---------------------------------------------------------

    def _set_nested(self, name: str, value: Any) -> None:
        keys = name.split(".")
        d: Any = self._metrics
        for k in keys[:-1]:
            existing = d.get(k)
            if existing is None:
                d[k] = {}
                d = d[k]
            elif isinstance(existing, dict):
                d = existing
            else:
                raise ValueError(
                    f"emit({name!r}): path conflict — key {k!r} already holds a non-dict"
                )
        d[keys[-1]] = value

    # -- output -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the merged metrics + ``metric_provenance`` block."""
        out = dict(self._metrics)
        if self._provenance:
            out["metric_provenance"] = {
                name: rec.to_provenance_dict()
                for name, rec in self._provenance.items()
            }
        return out

    def write(self, path: str) -> None:
        """Write ``metrics.json`` (creates parent dirs if needed)."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    # -- introspection ----------------------------------------------------

    def provenance(self) -> dict[str, dict[str, Any]]:
        """Return the provenance map only (no metrics)."""
        return {
            name: rec.to_provenance_dict() for name, rec in self._provenance.items()
        }

    def metrics(self) -> dict[str, Any]:
        """Return the metrics tree only (no provenance)."""
        return dict(self._metrics)

    def max_defaulted_rate(self) -> float:
        """Highest ``defaulted_rate`` across all emitted metrics.

        Useful for asserting "no metric in this run had >X% defaults"
        without iterating manually.
        """
        if not self._provenance:
            return 0.0
        return max(
            (rec.n_defaulted / rec.n_total) if rec.n_total > 0 else 0.0
            for rec in self._provenance.values()
        )
