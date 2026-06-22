"""EmbeddingClient outage handling (2026-06-09 review, S2).

Zero-embedding fallbacks make every cosine similarity 0, silently zeroing
R_ground. Transient failures may degrade a batch or two, but a persistent
server outage must abort the run instead of corrupting the training cell.
"""

import numpy as np
import pytest

from dagspaces.grpo_training.stages.clients import EmbeddingClient


def _failing_client(fail_after=3):
    client = EmbeddingClient(max_retries=1, fail_after=fail_after)
    client._session.post = lambda *a, **kw: (_ for _ in ()).throw(
        ConnectionError("server down")
    )
    return client


class TestEmbeddingClientFailure:
    def test_transient_failures_return_zeros(self):
        client = _failing_client(fail_after=3)
        for _ in range(2):
            out = client.encode_batch(["a", "b"])
            assert out.shape == (2, 1)
            assert not out.any()

    def test_persistent_outage_raises(self):
        client = _failing_client(fail_after=3)
        client.encode_batch(["a"])
        client.encode_batch(["a"])
        with pytest.raises(RuntimeError, match="consecutive"):
            client.encode_batch(["a"])

    def test_success_resets_failure_counter(self):
        client = _failing_client(fail_after=2)
        client.encode_batch(["a"])  # failure 1/2

        def _ok(url, json=None, timeout=None):
            class _Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"data": [
                        {"index": i, "embedding": [1.0, 0.0]}
                        for i in range(len(json["input"]))
                    ]}
            return _Resp()

        client._session.post = _ok
        ok = client.encode_batch(["a", "b"])  # success resets counter
        assert np.allclose(np.linalg.norm(ok, axis=1), 1.0)

        client._session.post = lambda *a, **kw: (_ for _ in ()).throw(
            ConnectionError("down again")
        )
        out = client.encode_batch(["a"])  # failure 1/2 again — no raise
        assert not out.any()


def _ok_post_factory(call_log, dim=2, fail_on_calls=()):
    """POST stub returning one unit embedding per input text."""

    def _post(url, json=None, timeout=None):
        call_log.append(len(json["input"]))
        if len(call_log) in fail_on_calls:
            raise ConnectionError("transient")

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self_inner):
                return {"data": [
                    {"index": i, "embedding": [1.0] + [0.0] * (dim - 1)}
                    for i in range(len(json["input"]))
                ]}
        return _Resp()

    return _post


class TestEncodeBatchChunking:
    """Oversized encode_batch calls must split into bounded HTTP requests.

    The 2026-06-10 production launch sent one embedding request covering
    1103 prompts × 8 samples, blew the 60s read timeout on every retry,
    and crashed retrieval on the (n, 1) zero fallback.
    """

    def test_large_batch_splits_requests(self):
        calls = []
        client = EmbeddingClient(max_retries=1, max_batch_size=2)
        client._session.post = _ok_post_factory(calls)
        out = client.encode_batch(["a", "b", "c", "d", "e"])
        assert calls == [2, 2, 1]
        assert out.shape == (5, 2)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

    def test_failed_chunk_zeroed_with_real_dim(self):
        calls = []
        client = EmbeddingClient(
            max_retries=1, max_batch_size=2, fail_after=5,
        )
        client._session.post = _ok_post_factory(calls, fail_on_calls={2})
        out = client.encode_batch(["a", "b", "c", "d", "e"])
        # Failed middle chunk → zero rows, but the matrix keeps the real
        # embedding dim from the successful chunks (no (n, 1) placeholder).
        assert out.shape == (5, 2)
        assert out[0].any() and out[4].any()
        assert not out[2].any() and not out[3].any()


class TestRetrieveDegenerateQuery:
    """Zero/misshaped query embeddings must not crash the retrieval matmul."""

    def _retriever(self):
        from dagspaces.grpo_training.stages.clients import NormRetriever
        norms = [{"norm_articulation": f"norm {i}"} for i in range(5)]
        r = NormRetriever({"book": norms}, embeddings_dir="", top_k=3)
        r._embeddings["book"] = np.eye(5, 4, dtype=np.float32)
        return r

    def test_zero_query_returns_empty(self):
        r = self._retriever()
        out, sims = r.retrieve(np.zeros(4, dtype=np.float32), "book",
                               return_scores=True)
        assert out == "[]" and sims == []

    def test_dim_mismatch_returns_empty(self):
        r = self._retriever()
        out = r.retrieve(np.zeros(1, dtype=np.float32), "book")
        assert out == "[]"

    def test_healthy_query_still_retrieves(self):
        r = self._retriever()
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out, sims = r.retrieve(q, "book", return_scores=True)
        assert "norm 0" in out and sims[0] == 1.0
