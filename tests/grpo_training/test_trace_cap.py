"""reward_traces.jsonl size cap (2026-06-09 logging review, item G).

With trace_every=1 in production, the trace file grew without bound; the
promotion gates only read the most recent calls, so the cap keeps the
newest half when the file exceeds the byte budget.
"""

import json
import os

from dagspaces.grpo_training.stages.rewards import CompositeRewardFunction


def _traced_fn(tmp_path, max_bytes):
    fn = CompositeRewardFunction.__new__(CompositeRewardFunction)
    fn._trace_path = str(tmp_path / "reward_traces.jsonl")
    fn._trace_max_bytes = max_bytes
    fn._trace_writes = 0
    return fn


class TestTraceCap:
    def test_truncates_to_newest_entries(self, tmp_path):
        fn = _traced_fn(tmp_path, max_bytes=20_000)
        for call in range(400):
            fn._log_trace([{"call": call, "payload": "x" * 100}])
        assert os.path.getsize(fn._trace_path) < 20_000
        rows = [json.loads(line)
                for line in open(fn._trace_path, encoding="utf-8")]
        assert rows[-1]["call"] == 399      # newest entries kept
        assert rows[0]["call"] > 0          # oldest dropped
        calls = [r["call"] for r in rows]   # contiguous tail, no holes
        assert calls == list(range(calls[0], 400))

    def test_under_cap_untouched(self, tmp_path):
        fn = _traced_fn(tmp_path, max_bytes=10**6)
        for call in range(150):
            fn._log_trace([{"call": call}])
        rows = [json.loads(line)
                for line in open(fn._trace_path, encoding="utf-8")]
        assert [r["call"] for r in rows] == list(range(150))
