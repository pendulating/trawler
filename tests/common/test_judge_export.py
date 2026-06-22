"""Tests for ``dagspaces/common/judge_export.py``.

Cover the failure modes that previously produced silent metric
corruption:

1. Async mode + ``judge.batch.target_model`` set → must IGNORE
   target_model and resolve from the live ``/v1/models`` probe.
2. Async mode + unreachable judge server → fail fast (not silently
   write 1400+ requests destined to 404).
3. Async mode + explicit ``judge.model_name`` not in served list →
   fail fast.
4. Async mode + ``judge.model_name=default`` → resolve to first
   served id.
5. ``batch_export`` mode + missing ``target_model`` → fail fast (no
   ``gpt-5.2`` fallback).
6. ``batch_export`` mode + explicit target_model → use it.
7. Unresolved Hydra ``${...}`` interpolation in base_url → treated
   as unset (would have hidden config bugs in earlier runs).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from dagspaces.common.judge_export import (
    JudgeExportConfigError,
    resolve_export_client,
    resolve_export_endpoint,
    resolve_judge_mode,
)


def _cfg(**overrides):
    """Build a minimal cfg with judge.* overlays."""
    base = {
        "judge": {
            "mode": "live",
            "base_url": "",
            "model_name": "default",
            "temperature": 0.0,
            "max_tokens": 1024,
            "batch": {
                "target_model": None,
                "target_endpoint": "/v1/chat/completions",
            },
        },
        "judge_server_url": "",
    }
    if overrides:
        # Shallow merge; tests deep-set via dotted keys via OmegaConf.update.
        cfg = OmegaConf.create(base)
        for k, v in overrides.items():
            OmegaConf.update(cfg, k, v, merge=True)
        return cfg
    return OmegaConf.create(base)


class _FakeServedModels:
    """Mock for the OpenAI SDK ``Models.list`` return value."""

    def __init__(self, ids):
        self._ids = ids

    def __iter__(self):
        for i in self._ids:
            m = MagicMock()
            m.id = i
            yield m


@pytest.fixture
def patched_judge_client():
    """Patch JudgeClient to return a probe with a configurable model list.

    Yields the mock so tests can set ``probe._client.models.list.return_value``
    for the model-discovery probe.
    """
    with patch("dagspaces.common.judge_export.JudgeClient") as MockJudgeClient:
        instance = MagicMock()
        instance._client = MagicMock()
        MockJudgeClient.return_value = instance
        yield MockJudgeClient


class TestResolveJudgeMode:
    def test_default(self):
        assert resolve_judge_mode(_cfg()) == "live"

    def test_async(self):
        assert resolve_judge_mode(_cfg(**{"judge.mode": "async"})) == "async"

    def test_uppercase_normalized(self):
        assert resolve_judge_mode(_cfg(**{"judge.mode": "BATCH_EXPORT"})) == "batch_export"


class TestResolveExportEndpoint:
    def test_default(self):
        assert resolve_export_endpoint(_cfg()) == "/v1/chat/completions"

    def test_override(self):
        cfg = _cfg(**{"judge.batch.target_endpoint": "/v1/responses"})
        assert resolve_export_endpoint(cfg) == "/v1/responses"


class TestAsyncMode:
    def test_async_ignores_target_model_uses_served(self, patched_judge_client):
        """The bug we just shipped a fix for: async runs were stamping
        ``judge.batch.target_model`` (default 'gpt-5.2') into body.model
        even when the live judge server was serving Qwen.
        """
        cfg = _cfg(**{
            "judge.mode": "async",
            "judge.base_url": "http://klara:8002",
            "judge.model_name": "default",
            "judge.batch.target_model": "gpt-5.2",  # MUST be ignored
        })
        # First JudgeClient(...) call is the probe; second is the offline
        # export client. Both share the same MagicMock instance returned
        # by patched_judge_client(), so we configure models.list once.
        patched_judge_client.return_value._client.models.list.return_value = (
            _FakeServedModels(["Qwen3.6-27B"])
        )

        client, info = resolve_export_client(cfg, dagspace="privacylens")
        assert info["mode"] == "async"
        assert info["base_url"] == "http://klara:8002"
        assert info["model_name"] == "Qwen3.6-27B"
        # Ensure target_model was NOT used.
        assert info.get("target_model") is None

    def test_async_explicit_model_name_validated(self, patched_judge_client):
        cfg = _cfg(**{
            "judge.mode": "async",
            "judge.base_url": "http://klara:8002",
            "judge.model_name": "Qwen3.6-27B",
        })
        patched_judge_client.return_value._client.models.list.return_value = (
            _FakeServedModels(["Qwen3.6-27B", "Qwen2.5-72B"])
        )
        _client, info = resolve_export_client(cfg, dagspace="privacylens")
        assert info["model_name"] == "Qwen3.6-27B"

    def test_async_explicit_model_name_not_served_fails(self, patched_judge_client):
        cfg = _cfg(**{
            "judge.mode": "async",
            "judge.base_url": "http://klara:8002",
            "judge.model_name": "gpt-5.2",   # not actually served
        })
        patched_judge_client.return_value._client.models.list.return_value = (
            _FakeServedModels(["Qwen3.6-27B"])
        )
        with pytest.raises(JudgeExportConfigError, match="not served by"):
            resolve_export_client(cfg, dagspace="privacylens")

    def test_async_no_url_fails_fast(self, patched_judge_client):
        cfg = _cfg(**{"judge.mode": "async"})
        with pytest.raises(JudgeExportConfigError, match="no live judge endpoint"):
            resolve_export_client(cfg, dagspace="privacylens")

    def test_async_unreachable_server_fails(self, patched_judge_client):
        cfg = _cfg(**{
            "judge.mode": "async",
            "judge.base_url": "http://nonexistent:9999",
        })
        # Make the probe's models.list raise — simulates ConnectionError.
        patched_judge_client.return_value._client.models.list.side_effect = (
            ConnectionError("Connection refused")
        )
        with pytest.raises(JudgeExportConfigError, match="probe failed"):
            resolve_export_client(cfg, dagspace="privacylens")

    def test_async_zero_served_models_fails(self, patched_judge_client):
        cfg = _cfg(**{
            "judge.mode": "async",
            "judge.base_url": "http://klara:8002",
        })
        patched_judge_client.return_value._client.models.list.return_value = (
            _FakeServedModels([])
        )
        with pytest.raises(JudgeExportConfigError, match="zero served models"):
            resolve_export_client(cfg, dagspace="privacylens")

    def test_async_unresolved_interpolation_treated_as_unset(self, patched_judge_client):
        """Hydra leaves unresolved ${var} as literal text; treat as empty."""
        cfg = _cfg(**{
            "judge.mode": "async",
            "judge.base_url": "${judge_server_url}",  # unresolved
        })
        # Without the fallback to env / legacy alias being populated,
        # this should fail fast rather than try to POST to '${judge_server_url}'.
        with pytest.raises(JudgeExportConfigError, match="no live judge endpoint"):
            resolve_export_client(cfg, dagspace="privacylens")

    def test_async_legacy_alias_fallback(self, patched_judge_client):
        """``judge_server_url`` (legacy) should be picked up if judge.base_url is empty."""
        cfg = _cfg(**{
            "judge.mode": "async",
            "judge_server_url": "http://klara:8002",
        })
        patched_judge_client.return_value._client.models.list.return_value = (
            _FakeServedModels(["Qwen3.6-27B"])
        )
        _, info = resolve_export_client(cfg, dagspace="privacylens")
        assert info["base_url"] == "http://klara:8002"

    def test_async_env_var_fallback(self, patched_judge_client, monkeypatch):
        """$JUDGE_SERVER_URL should be picked up if judge.base_url is empty."""
        cfg = _cfg(**{"judge.mode": "async"})
        monkeypatch.setenv("JUDGE_SERVER_URL", "http://klara:8002")
        patched_judge_client.return_value._client.models.list.return_value = (
            _FakeServedModels(["Qwen3.6-27B"])
        )
        _, info = resolve_export_client(cfg, dagspace="privacylens")
        assert info["base_url"] == "http://klara:8002"


class TestBatchExportMode:
    def test_batch_export_uses_target_model(self, patched_judge_client):
        cfg = _cfg(**{
            "judge.mode": "batch_export",
            "judge.batch.target_model": "gpt-4o-mini",
        })
        client, info = resolve_export_client(cfg, dagspace="privacylens")
        assert info["mode"] == "batch_export"
        assert info["target_model"] == "gpt-4o-mini"
        # No live endpoint probe in batch_export mode.
        patched_judge_client.return_value._client.models.list.assert_not_called()

    def test_batch_export_missing_target_fails(self, patched_judge_client):
        cfg = _cfg(**{"judge.mode": "batch_export"})
        with pytest.raises(JudgeExportConfigError, match="target_model is unset"):
            resolve_export_client(cfg, dagspace="privacylens")

    def test_batch_export_does_not_probe(self, patched_judge_client):
        cfg = _cfg(**{
            "judge.mode": "batch_export",
            "judge.batch.target_model": "gpt-5",
            "judge.base_url": "http://nonexistent:9999",  # would error if probed
        })
        # The unreachable URL should be irrelevant here.
        _, info = resolve_export_client(cfg, dagspace="privacylens")
        assert info["mode"] == "batch_export"


class TestUnknownMode:
    def test_live_mode_refused(self, patched_judge_client):
        cfg = _cfg(**{"judge.mode": "live"})
        with pytest.raises(JudgeExportConfigError, match="does not write requests"):
            resolve_export_client(cfg, dagspace="privacylens")

    def test_unknown_mode_refused(self, patched_judge_client):
        cfg = _cfg(**{"judge.mode": "wat"})
        with pytest.raises(JudgeExportConfigError, match="unknown judge.mode"):
            resolve_export_client(cfg, dagspace="privacylens")
