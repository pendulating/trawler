"""Guards on eval_all server-mode routing (dagspaces.common.vllm_inference).

Server mode exports VLLM_SERVER_URL to EVERY stage in an eval cell, so the
resolver must only route stages whose model matches the server's
(VLLM_SERVER_MODEL) — a judge stage running a different model must fall
back to a local engine, never be silently answered by the task model.

Corollary (2026-08-03): a URL with NO identity var is not trusted either.
eval_all always exports VLLM_SERVER_MODEL alongside the URL, so a bare URL did
not come from server mode — it is a stray env var, and server.env ships exactly
such a var pointing at the judge.
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

    def test_env_without_model_identity_refuses_to_route(self, monkeypatch):
        # INVERTED 2026-08-03. This used to assert the URL was honoured, as
        # back-compat for a hand-launched server that never exported the
        # identity var ("caller's responsibility"). That bet lost: the caller
        # is not a human choosing to route. server.env ships
        # VLLM_SERVER_URL pointing at the JUDGE server for the answerer /
        # aux-scorer clients, and ensure_dotenv() re-loads it inside EVERY
        # stage job — so the var silently applied to unrelated benchmark runs.
        # On 2026-08-03 it routed all five CI benchmarks' task inference to the
        # judge port: 404 on every request, generated_text empty,
        # parseable_rate 0.0000, the whole 5-cell sweep lost. It failed loudly
        # only because the served model name did not match; a collision would
        # have scored the JUDGE's answers as the task model's.
        #
        # Nothing regressed by inverting this: no caller in the repo reaches
        # _resolve_server_url with a bare URL (every script referencing
        # VLLM_SERVER_URL reads it directly for the judge/answerer role), and
        # the legitimate hand-launched-server workflow keeps its trusted path
        # via cfg.model.vllm_server_url — see
        # test_explicit_cfg_override_skips_identity_check below.
        monkeypatch.setenv("VLLM_SERVER_URL", self.URL)
        monkeypatch.delenv("VLLM_SERVER_MODEL", raising=False)
        assert _resolve_server_url(_cfg()) is None

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
