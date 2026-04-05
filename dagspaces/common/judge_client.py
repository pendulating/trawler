"""OpenAI-compatible judge client for evaluation dagspaces.

Talks to any service speaking the OpenAI ``/v1/chat/completions`` API:
- vLLM servers (default; launched via ``scripts/judge_server.sub``)
- OpenAI (``https://api.openai.com/v1``)
- Anthropic (``https://api.anthropic.com/v1/``, OpenAI-compat beta endpoint)
- Google Gemini (``https://generativelanguage.googleapis.com/v1beta/openai/``)
- Any other OpenAI-compatible provider (OpenRouter, Together, Groq, etc.)

The provider is auto-detected from the hostname in ``base_url``; override
via the ``provider`` argument.

Usage (vLLM — unchanged from previous versions)::

    client = JudgeClient(base_url="http://klara:8002")
    results = client.judge_batch(items, build_messages_fn, json_schema=schema)

Usage (commercial API — just add base_url + model_name + api key)::

    client = JudgeClient(
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o",
        api_key_env="OPENAI_API_KEY",
    )
    results = client.judge_batch(items, build_messages_fn, json_schema=schema)
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import requests

__all__ = ["JudgeClient"]


_PROVIDER_ENV_VARS = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
}

_COMMERCIAL_PROVIDERS = {"openai", "anthropic", "gemini", "openai_compatible"}


def _guess_provider(base_url: str) -> str:
    """Infer provider from a base URL's hostname."""
    host = base_url.lower()
    if "api.openai.com" in host:
        return "openai"
    if "api.anthropic.com" in host:
        return "anthropic"
    if "generativelanguage.googleapis.com" in host:
        return "gemini"
    if "openrouter.ai" in host or "together.xyz" in host or "groq.com" in host \
            or "fireworks.ai" in host or "deepinfra.com" in host:
        return "openai_compatible"
    # Local / internal hostnames default to vLLM.
    return "vllm"


def _resolve_api_key(
    api_key: Optional[str], api_key_env: Optional[str], provider: str,
) -> str:
    """Resolve an API key from args / env vars, in priority order."""
    if api_key:
        return api_key
    if api_key_env:
        val = os.environ.get(api_key_env, "")
        if val:
            return val
    for env_name in _PROVIDER_ENV_VARS.get(provider, []):
        val = os.environ.get(env_name, "")
        if val:
            return val
    return ""


class JudgeClient:
    """HTTP client for an OpenAI-compatible chat-completions endpoint.

    Parameters
    ----------
    base_url:
        Full base URL ending in ``/v1`` (e.g. ``http://klara:8002``,
        ``https://api.openai.com/v1``). vLLM servers accept both with and
        without the trailing ``/v1``.
    model_name:
        Model identifier. For vLLM, leave as ``"default"`` to auto-detect
        via ``/v1/models``. For commercial providers, must be explicit
        (``"gpt-4o"``, ``"claude-3-5-sonnet-20241022"``, ``"gemini-2.0-flash"``, ...).
    api_key, api_key_env:
        Commercial API key. Pass ``api_key`` directly or ``api_key_env``
        (env var name) to read from the environment. When neither is set,
        falls back to ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` /
        ``GOOGLE_API_KEY|GEMINI_API_KEY`` based on the detected provider.
    provider:
        Override the auto-detected provider. One of ``"vllm"``,
        ``"openai"``, ``"anthropic"``, ``"gemini"``, ``"openai_compatible"``.
        Only affects whether vLLM-specific fields (``chat_template_kwargs``)
        are sent and which default env var holds the API key.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8002",
        model_name: str = "default",
        timeout: float = 120.0,
        max_workers: int = 8,
        max_retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        *,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.provider = (provider or _guess_provider(self.base_url)).lower()
        self._api_key = _resolve_api_key(api_key, api_key_env, self.provider)

        # Commercial providers require an API key and an explicit model name.
        if self.provider in _COMMERCIAL_PROVIDERS and not self._api_key:
            env_list = _PROVIDER_ENV_VARS.get(self.provider, [])
            hint = f" (tried env vars: {', '.join(env_list)})" if env_list else ""
            raise ValueError(
                f"Judge provider {self.provider!r} requires an API key. "
                f"Pass api_key=... or set api_key_env=<name>{hint}."
            )

        # Build the OpenAI SDK client lazily so import cost is paid once.
        from openai import OpenAI
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self._api_key or "EMPTY",
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        # Keep a plain requests session for vLLM's /health (not in the OpenAI spec).
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def health_check(self, timeout: float = 10.0) -> bool:
        """Verify the endpoint is reachable and (for vLLM) auto-detect model name.

        For vLLM, probes ``GET /health`` then ``/v1/models`` to pick up the
        served model id when ``model_name == "default"``. For commercial
        providers, validates that an explicit model name is configured
        (probing their ``/v1/models`` would return hundreds of entries).
        """
        if self.provider == "vllm":
            try:
                r = self._session.get(
                    f"{self.base_url}/health",
                    timeout=timeout,
                    headers=self._auth_header(),
                )
                if r.status_code != 200:
                    return False
            except Exception:
                return False
            if self.model_name == "default":
                try:
                    models = list(self._client.models.list())
                    if models:
                        self.model_name = models[0].id
                except Exception:
                    pass
            return True

        # Commercial providers: require explicit model name.
        if self.model_name in ("", "default"):
            raise ValueError(
                f"Provider {self.provider!r} requires an explicit model_name "
                f"(e.g. 'gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-2.0-flash')."
            )
        return True

    def _auth_header(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    # ------------------------------------------------------------------
    # Single-call
    # ------------------------------------------------------------------
    def _call_single(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send one chat completion. Returns the assistant content string.

        Structured decoding: when ``json_schema`` is supplied, the request
        uses ``response_format={"type": "json_schema", ...}``. Providers
        that don't support this may reject the request; on 400-style
        errors we retry once without the schema so the caller still gets
        plain text it can parse defensively.
        """
        kwargs: Dict[str, Any] = dict(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": json_schema},
            }
        # vLLM-only: tell the server not to emit <think> reasoning blocks.
        if self.provider == "vllm":
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        try:
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            # If the provider rejected the structured schema, retry plain.
            if json_schema and self._looks_like_schema_rejection(e):
                try:
                    kwargs.pop("response_format", None)
                    resp = self._client.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content or ""
                except Exception as e2:
                    return json.dumps({"error": f"fallback_failed: {e2}"})
            return json.dumps({"error": str(e)})

    @staticmethod
    def _looks_like_schema_rejection(err: Exception) -> bool:
        s = str(err).lower()
        return (
            "response_format" in s
            or "json_schema" in s
            or "invalid_request_error" in s
            or "unsupported" in s
        )

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def judge_batch(
        self,
        items: List[Dict[str, Any]],
        build_messages_fn: Callable[[Dict[str, Any]], List[Dict[str, str]]],
        json_schema: Optional[Dict[str, Any]] = None,
        progress_every: int = 50,
    ) -> List[str]:
        """Judge a batch of items concurrently via a thread pool.

        Args:
            items: list of item dicts, each passed to ``build_messages_fn``.
            build_messages_fn: callable returning ``[{"role", "content"}, ...]``.
            json_schema: optional JSON Schema for structured decoding.
            progress_every: log progress every N completions.

        Returns:
            List of response strings in the same order as ``items``.
        """
        results: List[Optional[str]] = [None] * len(items)
        completed = 0

        def _process(idx: int) -> tuple[int, str]:
            messages = build_messages_fn(items[idx])
            return idx, self._call_single(messages, json_schema)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_process, i): i for i in range(len(items))}
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text
                completed += 1
                if completed % progress_every == 0 or completed == len(items):
                    print(f"  [{completed}/{len(items)}] judge responses received",
                          flush=True)

        return [r or "" for r in results]
