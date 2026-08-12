"""Response parse stages — shared with ``vlm_geoprivacy_bench``.

A hypothetical variant changes the PROMPT, not the answer format, so both
dagspaces parse a completion the same way.

Before 2026-08-12 this file was a byte-identical copy of the benchmark file.
"""

from __future__ import annotations

from dagspaces.vlm_geoprivacy_bench.stages.parse_responses import (
    parse_freeform_responses,
    parse_mcq_responses,
)

__all__ = ["parse_freeform_responses", "parse_mcq_responses"]
