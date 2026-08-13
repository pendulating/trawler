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

import sys

import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

# OpenAI Batch API hard limits (docs: wiki/integrations/openai-batch-api.md).
# A single batch input file may include up to 50,000 requests and be up to
# 200 MB in size. We only warn — the API enforces both server-side.
_BATCH_MAX_REQUESTS = 50_000
_BATCH_MAX_FILE_BYTES = 200 * 1024 * 1024

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
    api_key: str | None, api_key_env: str | None, provider: str,
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
        Full base URL ending in ``/v1`` (e.g. ``http://klara:8002/v1``,
        ``https://api.openai.com/v1``). For vLLM, a bare host:port without
        ``/v1`` is accepted and normalized internally — every OpenAI-compat
        endpoint vLLM exposes lives under ``/v1/``.
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
    offline:
        When ``True``, skips the commercial-provider API-key requirement
        and does not construct the OpenAI SDK client. Used with
        ``export_batch_jsonl`` when the emitted JSONL will be submitted
        from a different account / machine, so no credentials exist
        locally. Any method that would make a network call (``judge_batch``,
        ``_call_single``, ``health_check`` for non-vLLM providers) will
        raise when ``offline=True``.
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
        api_key: str | None = None,
        api_key_env: str | None = None,
        provider: str | None = None,
        offline: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.offline = offline

        self.provider = (provider or _guess_provider(self.base_url)).lower()

        # vLLM only exposes OpenAI-compat endpoints under /v1 (plus /health at root).
        # The OpenAI SDK appends /chat/completions directly to base_url, so a
        # base_url without /v1 silently 404s on every request. Normalize here so
        # callers can pass either form.
        if self.provider == "vllm" and not self.base_url.rstrip("/").endswith("/v1"):
            self.base_url = self.base_url + "/v1"

        self._api_key = _resolve_api_key(api_key, api_key_env, self.provider)

        # Commercial providers normally require an API key. In ``offline=True``
        # mode (used for ``export_batch_jsonl`` when the JSONL will be
        # submitted from a *different* account), we skip that check and also
        # skip constructing the OpenAI SDK client — no network calls happen.
        if (
            self.provider in _COMMERCIAL_PROVIDERS
            and not self._api_key
            and not self.offline
        ):
            env_list = _PROVIDER_ENV_VARS.get(self.provider, [])
            hint = f" (tried env vars: {', '.join(env_list)})" if env_list else ""
            raise ValueError(
                f"Judge provider {self.provider!r} requires an API key. "
                f"Pass api_key=... or set api_key_env=<name>{hint}."
            )

        self._client = None
        if not self.offline:
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
        if self.offline:
            raise RuntimeError(
                "JudgeClient is in offline mode; health_check() makes no sense "
                "because no network calls are allowed. Use offline mode only "
                "with export_batch_jsonl()."
            )
        if self.provider == "vllm":
            # /health is at the server root, not under /v1.
            health_base = self.base_url
            if health_base.rstrip("/").endswith("/v1"):
                health_base = health_base.rstrip("/")[: -len("/v1")]
            try:
                r = self._session.get(
                    f"{health_base}/health",
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
                except Exception as e:
                    # Do NOT swallow silently: model_name stays "default",
                    # and that literal string then travels into every judge
                    # request, where it fails far from the cause.
                    print(
                        f"[judge_client] could not auto-discover the served "
                        f"model name at {self.base_url!r}; model_name stays "
                        f"'default' and requests will use that literally. "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
            return True

        # Commercial providers: require explicit model name.
        if self.model_name in ("", "default"):
            raise ValueError(
                f"Provider {self.provider!r} requires an explicit model_name "
                f"(e.g. 'gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-2.0-flash')."
            )
        return True

    def _auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    # ------------------------------------------------------------------
    # Single-call
    # ------------------------------------------------------------------
    def _call_single(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Send one chat completion. Returns the assistant content string.

        Structured decoding: when ``json_schema`` is supplied, the request
        uses ``response_format={"type": "json_schema", ...}``. Providers
        that don't support this may reject the request; on 400-style
        errors we retry once without the schema so the caller still gets
        plain text it can parse defensively.
        """
        if self.offline:
            raise RuntimeError(
                "JudgeClient is in offline mode; live judging is disabled. "
                "Use export_batch_jsonl() to emit a Batch API input file."
            )
        kwargs: dict[str, Any] = dict(
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
        items: list[dict[str, Any]],
        build_messages_fn: Callable[[dict[str, Any]], list[dict[str, str]]],
        json_schema: dict[str, Any] | None = None,
        progress_every: int = 50,
    ) -> list[str]:
        """Judge a batch of items concurrently via a thread pool.

        Args:
            items: list of item dicts, each passed to ``build_messages_fn``.
            build_messages_fn: callable returning ``[{"role", "content"}, ...]``.
            json_schema: optional JSON Schema for structured decoding.
            progress_every: log progress every N completions.

        Returns:
            List of response strings in the same order as ``items``.
        """
        results: list[str | None] = [None] * len(items)
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

    # ------------------------------------------------------------------
    # Batch export (OpenAI Batch API JSONL)
    # ------------------------------------------------------------------
    def export_batch_jsonl(
        self,
        items: list[dict[str, Any]],
        build_messages_fn: Callable[[dict[str, Any]], list[dict[str, str]]],
        output_path: str,
        custom_id_fn: Callable[[dict[str, Any], int], str],
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "result",
        endpoint_url: str = "/v1/chat/completions",
    ) -> dict[str, Any]:
        """Write items as an OpenAI Batch API input JSONL file.

        Produces one JSONL line per item with the exact same chat-completions
        body ``_call_single`` would send in live mode (same model, temperature,
        max_tokens, and ``response_format`` if ``json_schema`` is supplied), so
        batch and live runs are prompt-identical.

        The Batch API is an OpenAI feature; we refuse to export when the
        configured provider is ``vllm``. Anthropic and Gemini are allowed with
        a warning since they share the same OpenAI-compat JSONL shape and can
        be re-pointed at their own batch endpoints downstream.

        Args:
            items: list of item dicts, each passed to ``build_messages_fn``.
            build_messages_fn: callable returning ``[{"role", "content"}, ...]``.
            output_path: absolute path to the target ``.jsonl`` file.
            custom_id_fn: callable ``(item, idx) -> str`` returning a unique
                ``custom_id`` per request. Caller is responsible for uniqueness;
                this method verifies and raises on duplicates.
            json_schema: optional JSON Schema for structured decoding; embedded
                in ``body.response_format`` identically to live mode.
            schema_name: name field of the embedded json_schema response_format.
            endpoint_url: one of the Batch-API-supported endpoints; defaults
                to ``/v1/chat/completions``.

        Returns:
            Manifest dict with ``path``, ``count``, ``model``, ``provider``,
            ``endpoint``, ``schema_name`` (if applicable), and ``bytes``.
        """
        # The JSONL shape is OpenAI-compat, which both vLLM (consumed by
        # the async-mode judge sidecar) and OpenAI's Batch API speak. We
        # used to refuse provider=vllm here, which forced async-mode
        # exports to fake provider=openai and put OpenAI-only model
        # names like ``gpt-5.2`` into ``body.model`` — vLLM then 404'd
        # every line. Now we allow vllm explicitly; the only thing we
        # still refuse is non-vllm/non-openai providers (anthropic /
        # gemini) which would need their own batch endpoints.
        if self.provider not in ("openai", "vllm"):
            print(
                f"[judge_client] WARN: export_batch_jsonl called with provider="
                f"{self.provider!r}; the emitted JSONL follows the OpenAI Batch "
                f"shape. Make sure your downstream consumer (Batch API or "
                f"async-judge sidecar) speaks it.",
                flush=True,
            )
        if not self.model_name or self.model_name == "default":
            raise ValueError(
                "export_batch_jsonl() requires an explicit model_name. "
                "For vLLM async mode, resolve via dagspaces.common.judge_export "
                "(probes /v1/models). For OpenAI batch_export mode, set "
                "judge.batch.target_model explicitly (e.g. 'gpt-4o-mini')."
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        seen_ids: set[str] = set()
        total_bytes = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, item in enumerate(items):
                cid = custom_id_fn(item, idx)
                if not isinstance(cid, str) or not cid:
                    raise ValueError(
                        f"custom_id_fn must return a non-empty string "
                        f"(got {cid!r} at idx={idx})"
                    )
                if cid in seen_ids:
                    raise ValueError(
                        f"duplicate custom_id {cid!r} at idx={idx}; "
                        f"each Batch API request must have a unique custom_id"
                    )
                seen_ids.add(cid)

                messages = build_messages_fn(item)
                body: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if json_schema:
                    body["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": schema_name, "schema": json_schema},
                    }
                line = json.dumps(
                    {
                        "custom_id": cid,
                        "method": "POST",
                        "url": endpoint_url,
                        "body": body,
                    },
                    ensure_ascii=False,
                )
                f.write(line)
                f.write("\n")
                total_bytes += len(line) + 1

        count = len(seen_ids)
        if count > _BATCH_MAX_REQUESTS:
            print(
                f"[judge_client] WARN: wrote {count} requests to {output_path}; "
                f"OpenAI Batch API caps a single input file at "
                f"{_BATCH_MAX_REQUESTS}. Split before submission.",
                flush=True,
            )
        if total_bytes > _BATCH_MAX_FILE_BYTES:
            print(
                f"[judge_client] WARN: {output_path} is "
                f"{total_bytes / 1024 / 1024:.1f} MB; OpenAI Batch API caps "
                f"input files at 200 MB. Split before submission.",
                flush=True,
            )

        manifest: dict[str, Any] = {
            "path": output_path,
            "count": count,
            "model": self.model_name,
            "provider": self.provider,
            "endpoint": endpoint_url,
            "bytes": total_bytes,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_schema:
            manifest["schema_name"] = schema_name
        return manifest
