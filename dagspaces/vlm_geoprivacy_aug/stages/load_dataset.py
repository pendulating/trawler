"""Dataset load stage — shared with ``vlm_geoprivacy_bench``.

This dagspace reads the same VLM-GeoPrivacy dataset as the benchmark. The
hypothetical variants attach later, in ``stages/inpaint_hypotheticals.py``,
so the load stage needs no change.

Before 2026-08-12 this file was a byte-identical copy of the benchmark file.
"""

from __future__ import annotations

from dagspaces.vlm_geoprivacy_bench.stages.load_dataset import load_dataset

__all__ = ["load_dataset"]
