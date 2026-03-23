"""HTTP clients for online R_ground evaluation during GRPO training.

Three clients:
  EmbeddingClient  — encodes flow queries via a vLLM embedding server
  JudgeClient      — evaluates normative grounding via a vLLM judge server
  NormRetriever    — in-memory top-k norm retrieval from pre-computed embeddings
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from dagspaces.common.stage_utils import extract_last_json
from .norm_universe import EMBED_INSTRUCTION


class EmbeddingClient:
    """HTTP client for a vLLM embedding server (Qwen3-Embedding-8B).

    Encodes flow queries into the same embedding space as pre-computed
    normative universe embeddings, enabling semantic retrieval.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        model_name: str = "default",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode texts into normalized embeddings.

        Prepends the instruction prefix used during norm universe
        construction so queries land in the same embedding space.

        Returns:
            np.ndarray of shape (len(texts), dim), L2-normalized.
        """
        if not texts:
            return np.empty((0, 0))

        prefixed = [EMBED_INSTRUCTION + t for t in texts]

        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"model": self.model_name, "input": prefixed},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                # vLLM returns embeddings sorted by index
                data_sorted = sorted(data, key=lambda d: d["index"])
                embs = np.array(
                    [d["embedding"] for d in data_sorted], dtype=np.float32
                )
                # L2 normalize
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                return embs / norms
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"Embedding server failed after {self.max_retries} attempts: {e}"
                    ) from e
                wait = 2 ** attempt
                print(f"[EmbeddingClient] Attempt {attempt + 1} failed ({e}), "
                      f"retrying in {wait}s...")
                time.sleep(wait)

    def close(self):
        self._session.close()


class JudgeClient:
    """HTTP client for a vLLM judge server (Qwen2.5-72B-Instruct-AWQ).

    Evaluates normative grounding of individual CI flows against
    retrieved norms from the normative universe.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8002",
        model_name: str = "default",
        system_prompt: str = "",
        prompt_template: str = "",
        json_schema: Optional[Dict[str, Any]] = None,
        timeout: float = 120.0,
        max_workers: int = 8,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.json_schema = json_schema
        self.timeout = timeout
        self.max_workers = max_workers
        self.max_retries = max_retries
        self._session = requests.Session()

    def _build_messages(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build chat messages from the judge prompt template."""
        user_prompt = (
            self.prompt_template
            .replace("{{chunk_text}}", str(item.get("chunk_text", "")))
            .replace("{{flow_json}}", str(item.get("flow_json", "{}")))
            .replace("{{norm_universe_json}}", str(item.get("norm_universe_json", "[]")))
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _judge_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Send a single judge request with retries."""
        messages = self._build_messages(item)

        request_body: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 512,
        }
        if self.json_schema:
            request_body["guided_json"] = self.json_schema

        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=request_body,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = extract_last_json(content)
                if parsed and isinstance(parsed, dict):
                    return {
                        "norm_match_score": float(
                            parsed.get("norm_match_score", 0.0)
                        ),
                        "governance_score": float(
                            parsed.get("governance_score", 0.0)
                        ),
                    }
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                print(f"[JudgeClient] Failed after {self.max_retries} attempts: {e}")

        return {"norm_match_score": 0.0, "governance_score": 0.0}

    def judge_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a batch of flows concurrently.

        Each item should have keys: chunk_text, flow_json, norm_universe_json.

        Returns:
            List of dicts with norm_match_score and governance_score.
        """
        if not items:
            return []

        results = [None] * len(items)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as pool:
            future_to_idx = {
                pool.submit(self._judge_single, item): i
                for i, item in enumerate(items)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = {
                        "norm_match_score": 0.0,
                        "governance_score": 0.0,
                    }
        return results

    def close(self):
        self._session.close()


class NormRetriever:
    """In-memory top-k norm retrieval from pre-computed embeddings.

    Loads per-book .npy embedding matrices at init and supports
    cosine-similarity retrieval for flow queries.
    """

    def __init__(
        self,
        norm_universes: Dict[str, list],
        embeddings_dir: str,
        top_k: int = 3,
    ):
        import os

        self.norm_universes = norm_universes
        self.top_k = top_k
        self._embeddings: Dict[str, np.ndarray] = {}

        for source_id in norm_universes:
            npy_path = os.path.join(embeddings_dir, f"{source_id}.npy")
            if os.path.exists(npy_path):
                self._embeddings[source_id] = np.load(npy_path)

        loaded = sum(len(v) for v in self._embeddings.values())
        print(f"[NormRetriever] Loaded embeddings for "
              f"{len(self._embeddings)} books ({loaded} vectors)")

    def retrieve(
        self,
        query_embedding: np.ndarray,
        source_id: str,
        contrastive_source: Optional[str] = None,
    ) -> str:
        """Retrieve top-k norms most relevant to a query embedding.

        Args:
            query_embedding: 1-D normalized embedding vector.
            source_id: The source book's ID.
            contrastive_source: If set, retrieve from this (wrong) source.

        Returns:
            JSON string of top-k norm dicts.
        """
        target_id = contrastive_source or source_id
        norms = self.norm_universes.get(target_id, [])
        if not norms:
            return "[]"
        if len(norms) <= self.top_k:
            return json.dumps(norms, ensure_ascii=False, indent=1)

        emb_matrix = self._embeddings.get(target_id)
        if emb_matrix is None:
            return json.dumps(norms[: self.top_k], ensure_ascii=False, indent=1)

        # Cosine similarity (both are L2-normalized)
        sims = emb_matrix @ query_embedding
        top_indices = np.argsort(sims)[-self.top_k :][::-1]
        selected = [norms[i] for i in top_indices]
        return json.dumps(selected, ensure_ascii=False, indent=1)

    def retrieve_batch(
        self,
        query_embeddings: np.ndarray,
        source_ids: List[str],
        contrastive_sources: List[Optional[str]],
    ) -> List[str]:
        """Vectorized top-k retrieval for multiple queries."""
        return [
            self.retrieve(query_embeddings[i], source_ids[i], contrastive_sources[i])
            for i in range(len(source_ids))
        ]
