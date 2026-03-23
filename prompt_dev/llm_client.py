from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LLMClientConfig:
    endpoint: str
    model: str
    timeout_sec: int = 180
    max_retries: int = 3
    retry_backoff_sec: float = 2.0


class VLLMClient:
    def __init__(self, cfg: LLMClientConfig) -> None:
        self.cfg = cfg
        self._url = f"{cfg.endpoint.rstrip('/')}/v1/chat/completions"

    def chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = requests.post(self._url, json=payload, timeout=self.cfg.timeout_sec)
                resp.raise_for_status()
                body = resp.json()
                content = body["choices"][0]["message"]["content"]
                usage = body.get("usage", {})
                return {"text": content, "usage": usage, "raw": body}
            except Exception as exc:  # pragma: no cover - network dependent
                last_exc = exc
                if attempt < self.cfg.max_retries - 1:
                    sleep_for = self.cfg.retry_backoff_sec ** (attempt + 1)
                    time.sleep(sleep_for)
        raise RuntimeError(f"vLLM chat request failed after retries: {last_exc}") from last_exc


def health_check(endpoint: str, timeout_sec: int = 10) -> bool:
    url = f"{endpoint.rstrip('/')}/health"
    try:
        resp = requests.get(url, timeout=timeout_sec)
        return resp.status_code == 200
    except Exception:  # pragma: no cover - network dependent
        return False

