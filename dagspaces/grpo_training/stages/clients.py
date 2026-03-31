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


def _build_norm_embed_text(norm: Dict[str, Any]) -> str:
    """Build embedding-friendly text from a norm dict.

    Mirrors norm_universe._build_norm_text but operates on the cleaned
    field names (without raz_ prefix) used in norm_universes.json.
    """
    art = norm.get("norm_articulation") or norm.get("canonical_norm_articulation") or ""
    subj = norm.get("norm_subject") or ""
    pe = norm.get("prescriptive_element") or ""
    act = norm.get("norm_act") or ""
    cond = norm.get("condition_of_application") or ""
    ctx = norm.get("context") or ""
    force = norm.get("normative_force") or ""

    parts = []
    if art:
        parts.append(str(art))
    tuple_str = f"{subj} {pe} {act}".strip()
    if cond:
        tuple_str += f" when {cond}"
    parts.append(tuple_str)
    if ctx:
        parts.append(f"[context: {ctx}]")
    if force:
        parts.append(f"[force: {force}]")
    return " | ".join(parts)


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
        self._embed_dim: int = 0  # cached from first successful call

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
                result = embs / norms
                self._embed_dim = result.shape[1]
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"[EmbeddingClient] Failed after {self.max_retries} "
                          f"attempts: {e}. Returning zero embeddings.")
                    # Return zero embeddings with correct dimension so
                    # retrieval produces low (but not crash-inducing) scores.
                    dim = self._embed_dim or 1
                    return np.zeros((len(texts), dim), dtype=np.float32)
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
        timeout: float = 600.0,
        max_workers: int = 4,
        max_retries: int = 4,
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
            "max_tokens": 256,
            # Suppress thinking for Qwen3 models — judge output is
            # short structured JSON, not a reasoning chain.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.json_schema:
            request_body["guided_json"] = self.json_schema

        prompt_chars = sum(len(m["content"]) for m in messages)

        for attempt in range(self.max_retries):
            t0 = time.time()
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
                        "appropriateness_consistent": bool(
                            parsed.get("appropriateness_consistent", False)
                        ),
                        "raw_response": content,
                    }
            except Exception as e:
                elapsed = time.time() - t0
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[JudgeClient] Attempt {attempt + 1}/{self.max_retries} "
                          f"failed ({elapsed:.0f}s, {prompt_chars} chars): {e}")
                    time.sleep(wait)
                    continue
                print(f"[JudgeClient] Failed after {self.max_retries} attempts "
                      f"({elapsed:.0f}s, {prompt_chars} prompt chars): {e}")

        return {"norm_match_score": 0.0, "governance_score": 0.0, "appropriateness_consistent": False}

    def judge_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a batch of flows concurrently.

        Each item should have keys: chunk_text, flow_json, norm_universe_json.

        Returns:
            List of dicts with norm_match_score and governance_score.
        """
        if not items:
            return []

        print(f"[JudgeClient] Batch: {len(items)} items, "
              f"max_workers={min(self.max_workers, len(items))}")

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
                        "appropriateness_consistent": False,
                    }
        return results

    def _coverage_single(
        self,
        item: Dict[str, Any],
        system_prompt: str,
        prompt_template: str,
        json_schema: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Send a single no-flow coverage judge request with retries."""
        user_prompt = (
            prompt_template
            .replace("{{chunk_text}}", str(item.get("chunk_text", "")))
            .replace("{{norm_universe_json}}", str(item.get("norm_universe_json", "[]")))
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        request_body: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_schema:
            request_body["guided_json"] = json_schema

        prompt_chars = sum(len(m["content"]) for m in messages)

        for attempt in range(self.max_retries):
            t0 = time.time()
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
                        "coverage_score": float(
                            parsed.get("coverage_score", 0.0)
                        ),
                        "passage_contains_governed_flows": bool(
                            parsed.get("passage_contains_governed_flows", False)
                        ),
                        "raw_response": content,
                    }
            except Exception as e:
                elapsed = time.time() - t0
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[JudgeClient] Coverage attempt {attempt + 1}/"
                          f"{self.max_retries} failed ({elapsed:.0f}s, "
                          f"{prompt_chars} chars): {e}")
                    time.sleep(wait)
                    continue
                print(f"[JudgeClient] Coverage failed after {self.max_retries} "
                      f"attempts ({elapsed:.0f}s, {prompt_chars} chars): {e}")

        return {"coverage_score": 0.0, "passage_contains_governed_flows": False}

    def judge_coverage_batch(
        self,
        items: List[Dict[str, Any]],
        system_prompt: str,
        prompt_template: str,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate no-flow coverage for a batch of chunks concurrently.

        Each item should have keys: chunk_text, norm_universe_json.

        Returns:
            List of dicts with coverage_score and
            passage_contains_governed_flows.
        """
        if not items:
            return []

        print(f"[JudgeClient] Coverage batch: {len(items)} items, "
              f"max_workers={min(self.max_workers, len(items))}")

        results = [None] * len(items)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as pool:
            future_to_idx = {
                pool.submit(
                    self._coverage_single, item,
                    system_prompt, prompt_template, json_schema,
                ): i
                for i, item in enumerate(items)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = {
                        "coverage_score": 0.0,
                        "passage_contains_governed_flows": False,
                    }
        return results

    def close(self):
        self._session.close()


class NormRetriever:
    """In-memory top-k norm retrieval with aligned embeddings.

    To guarantee query and norm embeddings are in the same space, norms
    are re-embedded via the same vLLM embedding server used for queries
    at init time.  Pre-computed .npy files are used as a fallback only.
    """

    def __init__(
        self,
        norm_universes: Dict[str, list],
        embeddings_dir: str,
        embedding_client: Optional["EmbeddingClient"] = None,
        top_k: int = 3,
    ):
        import os

        self.norm_universes = norm_universes
        self.top_k = top_k
        self._embeddings: Dict[str, np.ndarray] = {}

        # First, load any pre-computed .npy embeddings from disk.
        if embeddings_dir and os.path.isdir(embeddings_dir):
            for source_id in norm_universes:
                npy_path = os.path.join(embeddings_dir, f"{source_id}.npy")
                if os.path.exists(npy_path):
                    self._embeddings[source_id] = np.load(npy_path)
            loaded = sum(len(v) for v in self._embeddings.values())
            print(f"[NormRetriever] Loaded pre-computed embeddings for "
                  f"{len(self._embeddings)} books ({loaded} vectors)")

        # Re-embed only sources that are missing from the pre-computed set.
        missing = [
            sid for sid in norm_universes
            if sid not in self._embeddings and norm_universes[sid]
        ]
        if missing and embedding_client is not None:
            print(f"[NormRetriever] Re-embedding {len(missing)} missing "
                  f"sources via embedding server...")
            for source_id in missing:
                texts = [
                    _build_norm_embed_text(n) for n in norm_universes[source_id]
                ]
                self._embeddings[source_id] = embedding_client.encode_batch(texts)
            re_embedded = sum(
                len(norm_universes[sid]) for sid in missing
            )
            print(f"[NormRetriever] Re-embedded {re_embedded} norms across "
                  f"{len(missing)} books")
        elif missing:
            print(f"[NormRetriever] Warning: {len(missing)} sources have no "
                  f"embeddings and no embedding client available")

    def retrieve(
        self,
        query_embedding: np.ndarray,
        source_id: str,
        contrastive_source: Optional[str] = None,
        return_scores: bool = False,
    ):
        """Retrieve top-k norms most relevant to a query embedding.

        Args:
            query_embedding: 1-D normalized embedding vector.
            source_id: The source book's ID.
            contrastive_source: If set, retrieve from this (wrong) source.
            return_scores: If True, return (json_str, top_k_similarities).

        Returns:
            JSON string of top-k norm dicts, or (json_str, sims) tuple.
        """
        target_id = contrastive_source or source_id
        norms = self.norm_universes.get(target_id, [])
        if not norms:
            return ("[]", []) if return_scores else "[]"
        if len(norms) <= self.top_k:
            result = json.dumps(norms, ensure_ascii=False, indent=1)
            return (result, [1.0] * len(norms)) if return_scores else result

        emb_matrix = self._embeddings.get(target_id)
        if emb_matrix is None:
            result = json.dumps(norms[: self.top_k], ensure_ascii=False, indent=1)
            return (result, []) if return_scores else result

        # Cosine similarity (both are L2-normalized)
        sims = emb_matrix @ query_embedding
        top_indices = np.argsort(sims)[-self.top_k :][::-1]
        selected = [norms[i] for i in top_indices]
        top_sims = [round(float(sims[i]), 4) for i in top_indices]
        result = json.dumps(selected, ensure_ascii=False, indent=1)
        return (result, top_sims) if return_scores else result

    def retrieve_batch(
        self,
        query_embeddings: np.ndarray,
        source_ids: List[str],
        contrastive_sources: List[Optional[str]],
        return_scores: bool = False,
    ):
        """Vectorized top-k retrieval for multiple queries."""
        return [
            self.retrieve(
                query_embeddings[i], source_ids[i], contrastive_sources[i],
                return_scores=return_scores,
            )
            for i in range(len(source_ids))
        ]
