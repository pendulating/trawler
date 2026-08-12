"""Granularity judge stage — shared with ``vlm_geoprivacy_bench``.

The judge reads a free-form location answer and gives it a granularity label.
A hypothetical variant changes the prompt that produced that answer, not the
way the judge grades it, so both dagspaces use one judge.

Before 2026-08-12 this file was a copy, and the copy held a defect that
``vlm_geoprivacy_bench`` corrected on 2026-07-14 (commit 21838ab). The copy
took the first character in the set {A, B, C, D} from the judge completion.
That character is the ``a`` of "answer" for any verbose completion, so the
row got label ``A`` — the abstention class. A completion with no such
character got the default ``D``, which is also a real class. The benchmark
now anchors on an explicit answer marker and returns ``"unparseable"`` when
it finds no label. This dagspace never got that correction, because nobody
knew that the file was a copy.

No result changes: this dagspace has no run on disk, no entry in the paper,
and no consumer notebook.
"""

from __future__ import annotations

from dagspaces.vlm_geoprivacy_bench.stages.granularity_judge import (
    run_granularity_judge,
)

__all__ = ["run_granularity_judge"]
