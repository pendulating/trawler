"""NormRetriever ``norm_filter`` — the R-DIRECT index restriction (2026-07-28).

The m1 wave left 37% of policy flows unscored because R-DIRECT retrieved over
the full universe (71% non-flow norms) and discarded any flow whose single
nearest neighbour failed the ``governs_info_flow`` check post-hoc. The ruling:
restrict the index to flow-governing norms instead. The load-bearing invariant
is norms<->embedding-row alignment: pre-computed ``.npy`` rows are 1:1 with
the UNFILTERED per-source norm list, so the mask must be applied to both, and
a row-count mismatch must refuse loudly rather than misalign silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from dagspaces.grpo_training.stages.clients import NormRetriever

GOVERNS = lambda n: n.get("governs_info_flow") is True  # noqa: E731


def _universe():
    # Order matters: the non-flow norm (idx 1) is the global nearest to the
    # query below. Three flow norms keep the filtered index (3) above k=2 —
    # retrieve()'s ``len(norms) <= k`` shortcut returns norms UNRANKED, which
    # R-DIRECT never hits in production (min 44 flow norms/book).
    return {
        "book": [
            {"norm_id": "flow_a", "governs_info_flow": True},
            {"norm_id": "conduct", "governs_info_flow": False},
            {"norm_id": "flow_b", "governs_info_flow": True},
            {"norm_id": "flow_c", "governs_info_flow": True},
        ]
    }


def _embeddings_dir(tmp_path, rows):
    d = tmp_path / "emb"
    d.mkdir()
    np.save(d / "book.npy", np.asarray(rows, dtype=np.float32))
    return str(d)


# Unit vectors: query == the conduct norm's vector, flow_a nearby, flow_b far.
_CONDUCT = [1.0, 0.0]
_FLOW_A = [0.9701425, 0.2425356]   # cos to query ~0.97
_FLOW_B = [0.0, 1.0]               # cos to query 0.0
_FLOW_C = [-1.0, 0.0]              # cos to query -1.0
_QUERY = np.asarray([1.0, 0.0], dtype=np.float32)


class TestNormFilter:
    def test_unfiltered_top1_is_the_conduct_norm(self, tmp_path):
        r = NormRetriever(_universe(),
                          _embeddings_dir(tmp_path, [_FLOW_A, _CONDUCT, _FLOW_B, _FLOW_C]),
                          top_k=2)
        import json
        norms, sims = r.retrieve(_QUERY, "book", return_scores=True)
        top = json.loads(norms)[0]
        assert top["norm_id"] == "conduct"  # the failure mode being fixed

    def test_filtered_top1_is_the_nearest_flow_norm(self, tmp_path):
        r = NormRetriever(_universe(),
                          _embeddings_dir(tmp_path, [_FLOW_A, _CONDUCT, _FLOW_B, _FLOW_C]),
                          top_k=2, norm_filter=GOVERNS)
        import json
        norms, sims = r.retrieve(_QUERY, "book", return_scores=True)
        got = json.loads(norms)
        assert [n["norm_id"] for n in got] == ["flow_a", "flow_b"]  # flow_c cut by k=2
        # Alignment: flow_a's similarity, not conduct's, and margin is
        # computed within the restricted index.
        assert sims[0] == pytest.approx(0.9701425, abs=1e-4)
        assert sims[1] == pytest.approx(0.0, abs=1e-4)

    def test_filter_does_not_mutate_callers_universe(self, tmp_path):
        u = _universe()
        NormRetriever(u, _embeddings_dir(tmp_path, [_FLOW_A, _CONDUCT, _FLOW_B, _FLOW_C]),
                      top_k=2, norm_filter=GOVERNS)
        assert len(u["book"]) == 4

    def test_misaligned_npy_with_filter_raises(self, tmp_path):
        # 2 rows for 4 norms: applying a mask would silently misalign — the
        # constructor must refuse.
        with pytest.raises(ValueError, match="misaligned"):
            NormRetriever(_universe(),
                          _embeddings_dir(tmp_path, [_FLOW_A, _CONDUCT]),
                          top_k=2, norm_filter=GOVERNS)

    def test_misaligned_npy_without_filter_keeps_legacy_behavior(self, tmp_path):
        # Pre-existing (unfiltered) semantics are untouched by this change.
        NormRetriever(_universe(), _embeddings_dir(tmp_path, [_FLOW_A, _CONDUCT]),
                      top_k=2)

    def test_reembed_path_embeds_only_filtered_norms(self):
        class FakeEmb:
            def __init__(self):
                self.texts = []

            def encode_batch(self, texts):
                self.texts.extend(texts)
                v = np.zeros((len(texts), 2), dtype=np.float32)
                v[:, 0] = 1.0
                return v

        fake = FakeEmb()
        r = NormRetriever(_universe(), embeddings_dir="",
                          embedding_client=fake, top_k=2, norm_filter=GOVERNS)
        assert len(fake.texts) == 3  # flow_a/b/c, conduct excluded
        assert r._embeddings["book"].shape == (3, 2)

    def test_source_with_zero_flow_norms_returns_empty(self, tmp_path):
        u = {"book": [{"norm_id": "conduct", "governs_info_flow": False}]}
        d = tmp_path / "emb"
        d.mkdir()
        np.save(d / "book.npy", np.asarray([_CONDUCT], dtype=np.float32))
        r = NormRetriever(u, str(d), top_k=2, norm_filter=GOVERNS)
        norms, sims = r.retrieve(_QUERY, "book", return_scores=True)
        assert norms == "[]" and sims == []
