"""Lightweight OpenAI-compatible judge client for evaluation dagspaces.

Calls a vLLM server's /v1/chat/completions endpoint with optional
structured decoding (guided_json). The server is launched separately
via scripts/judge_server.sub.

Usage:
    from dagspaces.common.judge_client import JudgeClient
    client = JudgeClient(base_url="http://klara:8002")
    results = client.judge_batch(items, build_messages_fn, json_schema=schema)
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import requests


class JudgeClient:
    """HTTP client for a vLLM judge server (OpenAI-compatible API).

    Sends chat completion requests with optional guided JSON decoding.
    Supports concurrent batching via a thread pool.
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
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._session = requests.Session()

    def health_check(self, timeout: float = 10.0) -> bool:
        """Check if the judge server is reachable. Auto-detects model name."""
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=timeout)
            if r.status_code != 200:
                return False
            # Auto-detect model name from /v1/models if using default
            if self.model_name == "default":
                try:
                    mr = self._session.get(
                        f"{self.base_url}/v1/models", timeout=timeout
                    )
                    if mr.status_code == 200:
                        models = mr.json().get("data", [])
                        if models:
                            self.model_name = models[0]["id"]
                except Exception:
                    pass
            return True
        except Exception:
            return False

    def _call_single(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a single chat completion request with retries."""
        body: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "result",
                    "schema": json_schema,
                },
            }

        for attempt in range(self.max_retries):
            try:
                r = self._session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return json.dumps({"error": str(e)})
                time.sleep(2 ** attempt)

        return json.dumps({"error": "max retries exceeded"})

    def judge_batch(
        self,
        items: List[Dict[str, Any]],
        build_messages_fn: Callable[[Dict[str, Any]], List[Dict[str, str]]],
        json_schema: Optional[Dict[str, Any]] = None,
        progress_every: int = 50,
    ) -> List[str]:
        """Judge a batch of items concurrently.

        Args:
            items: List of dicts, each passed to build_messages_fn.
            build_messages_fn: Callable that takes an item dict and returns
                a list of chat messages [{"role": ..., "content": ...}].
            json_schema: Optional JSON Schema for structured decoding.
            progress_every: Log progress every N items.

        Returns:
            List of response strings, one per item, in the same order.
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
