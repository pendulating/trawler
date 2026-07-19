"""Guards on eval_all server-mode routing (dagspaces.common.vllm_inference).

Server mode exports VLLM_SERVER_URL to EVERY stage in an eval cell, so the
resolver must only route stages whose model matches the server's
(VLLM_SERVER_MODEL) — a judge stage running a different model must fall
back to a local engine, never be silently answered by the task model.
"""

import pytest
from omegaconf import OmegaConf

from dagspaces.common.vllm_inference import (
    _resolve_server_url,
    _sp_to_openai_kwargs,
)


def _cfg(model_source="/zoo/Qwen3.5-9B", **model_extra):
    return OmegaConf.create({"model": {"model_source": model_source, **model_extra}})


class TestResolveServerUrl:
    URL = "http://klara:8123/v1"

    def test_no_env_no_override_is_none(self, monkeypatch):
        monkeypatch.delenv("VLLM_SERVER_URL", raising=False)
        monkeypatch.delenv("VLLM_SERVER_MODEL", raising=False)
        assert _resolve_server_url(_cfg()) is None

    def test_env_with_matching_model_routes(self, monkeypatch):
        monkeypatch.setenv("VLLM_SERVER_URL", self.URL)
        monkeypatch.setenv("VLLM_SERVER_MODEL", "/zoo/Qwen3.5-9B")
        assert _resolve_server_url(_cfg("/zoo/Qwen3.5-9B")) == self.URL

    def test_trailing_slash_tolerated(self, monkeypatch):
        monkeypatch.setenv("VLLM_SERVER_URL", self.URL)
        monkeypatch.setenv("VLLM_SERVER_MODEL", "/zoo/Qwen3.5-9B/")
        assert _resolve_server_url(_cfg("/zoo/Qwen3.5-9B")) == self.URL

    def test_env_with_mismatched_model_falls_back_local(self, monkeypatch):
        # The judge-hijack case: stage runs a different model than the server.
        monkeypatch.setenv("VLLM_SERVER_URL", self.URL)
        monkeypatch.setenv("VLLM_SERVER_MODEL", "/zoo/Qwen3.5-9B")
        assert _resolve_server_url(_cfg("/zoo/Gemma-4-31B-it")) is None

    def test_env_without_model_identity_still_routes(self, monkeypatch):
        # Back-compat: a hand-launched server that never exported the
        # identity var keeps working (caller's responsibility).
        monkeypatch.setenv("VLLM_SERVER_URL", self.URL)
        monkeypatch.delenv("VLLM_SERVER_MODEL", raising=False)
        assert _resolve_server_url(_cfg()) == self.URL

    def test_explicit_cfg_override_skips_identity_check(self, monkeypatch):
        monkeypatch.setenv("VLLM_SERVER_URL", "http://other:1/v1")
        monkeypatch.setenv("VLLM_SERVER_MODEL", "/zoo/SomethingElse")
        cfg = _cfg("/zoo/Gemma-4-31B-it", vllm_server_url=self.URL)
        assert _resolve_server_url(cfg) == self.URL


class TestSpToOpenaiKwargs:
    def test_direct_params_pass_through(self):
        kwargs, extra = _sp_to_openai_kwargs(
            {"max_tokens": 64, "temperature": 0.2, "top_p": 0.9}
        )
        assert kwargs == {"max_tokens": 64, "temperature": 0.2, "top_p": 0.9}
        assert extra == {}

    def test_vllm_extensions_go_to_extra_body(self):
        _, extra = _sp_to_openai_kwargs({"top_k": 40, "repetition_penalty": 1.1})
        assert extra == {"top_k": 40, "repetition_penalty": 1.1}

    def test_guided_json_translates_to_structured_outputs(self):
        # GoldCoin's guided-JSON path must survive the server translation —
        # vLLM >= 0.19 ignores the legacy guided_* params.
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        _, extra = _sp_to_openai_kwargs({"guided_decoding": {"json": schema}})
        assert extra["structured_outputs"] == {"json": schema}

    def test_none_values_dropped(self):
        kwargs, extra = _sp_to_openai_kwargs({"max_tokens": None, "top_k": None})
        assert kwargs == {}
        assert extra == {}
