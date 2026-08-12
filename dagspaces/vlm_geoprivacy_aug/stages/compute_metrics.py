"""Metric stage — shared with ``vlm_geoprivacy_bench``.

These are the per-question accuracy and F1 metrics of the base benchmark. The
metrics that compare a hypothetical variant against its baseline are separate:
see ``stages/hypothetical_metrics.py``.

Before 2026-08-12 this file was a copy, and the copy held the accuracy
denominator that ``vlm_geoprivacy_bench`` replaced in the parity review of
2026-07-21. The copy dropped an unparseable prediction from the denominator.
Upstream ``eval.py`` keeps it and counts it wrong, and the benchmark now
matches upstream. The old behavior stays available as the
``*_among_parseable`` diagnostic. This dagspace never got that correction,
because nobody knew that the file was a copy.

The flip was value-neutral for the benchmark, because guided JSON decoding
holds ``parseable_rate`` at 1.000 on every 2026-07 run. It is also
value-neutral here, because this dagspace has no run on disk.
"""

from __future__ import annotations

from dagspaces.vlm_geoprivacy_bench.stages.compute_metrics import (
    LABEL_ORDER,
    LABEL_TO_INT,
    _extract_first_char,
    compute_metrics,
    metrics_to_dataframe,
)

# ``_extract_first_char`` is private to the benchmark module, but
# ``stages/hypothetical_metrics.py`` needs the SAME label reader that the
# per-question metrics use. Re-export it here rather than write a second one.
__all__ = [
    "LABEL_ORDER",
    "LABEL_TO_INT",
    "_extract_first_char",
    "compute_metrics",
    "metrics_to_dataframe",
]
