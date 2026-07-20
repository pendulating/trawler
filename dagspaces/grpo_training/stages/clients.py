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
from typing import Any

import numpy as np
import requests

from dagspaces.common.stage_utils import extract_last_json

from .deontic import (
    candidate_appropriateness_consistency,
    governing_norm_force,
)
from .norm_universe import EMBED_INSTRUCTION


def _json_schema_response_format(json_schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON schema in the OpenAI ``response_format`` envelope.

    vLLM >= 0.19 silently ignores the legacy ``guided_json`` extra-body
    param (no error, no enforcement) — structured output must go through
    ``response_format`` with type ``json_schema``.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": "judgment", "schema": json_schema},
    }


def _build_norm_embed_text(norm: dict[str, Any]) -> str:
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
        fail_after: int = 3,
        max_batch_size: int = 64,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        # Per-request text cap: one oversized request (e.g. a prescreen
        # pass scoring thousands of completions at once) exceeds the read
        # timeout no matter how many times it is retried (2026-06-10 launch
        # failure). Large encode_batch calls are split transparently.
        self.max_batch_size = max_batch_size
        # Consecutive fully-failed encode_batch calls before raising
        # (2026-06-09 review, S2): zero-embedding fallbacks make cosine
        # similarity 0 everywhere, silently zeroing R_ground for the rest
        # of training. A transient blip degrades one batch; a persistent
        # outage must crash the run instead of corrupting the cell.
        self.fail_after = fail_after
        self._consecutive_failures = 0
        self._session = requests.Session()
        self._embed_dim: int = 0  # cached from first successful call

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode texts into normalized embeddings.

        Prepends the instruction prefix used during norm universe
        construction so queries land in the same embedding space. Requests
        are split into chunks of ``max_batch_size`` texts; a failed chunk
        falls back to zeros (and counts toward the consecutive-failure
        abort) without poisoning the other chunks.

        Returns:
            np.ndarray of shape (len(texts), dim), L2-normalized.
        """
        if not texts:
            return np.empty((0, 0))

        prefixed = [EMBED_INSTRUCTION + t for t in texts]
        chunks = [
            prefixed[i:i + self.max_batch_size]
            for i in range(0, len(prefixed), self.max_batch_size)
        ]
        outs = [self._encode_chunk(c) for c in chunks]
        # A chunk that failed before any chunk succeeded has placeholder
        # dim 1 — re-shape zero chunks to the real dim so vstack works.
        dim = max(o.shape[1] for o in outs)
        outs = [
            o if o.shape[1] == dim
            else np.zeros((o.shape[0], dim), dtype=np.float32)
            for o in outs
        ]
        return np.vstack(outs)

    def _encode_chunk(self, prefixed: list[str]) -> np.ndarray:
        """Encode one already-prefixed chunk with retries."""
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
                self._consecutive_failures = 0
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self.fail_after:
                        raise RuntimeError(
                            f"[EmbeddingClient] {self._consecutive_failures} "
                            f"consecutive encode_batch calls failed against "
                            f"{self.base_url} (last error: {e}). The embedding "
                            f"server is down — aborting instead of training on "
                            f"zeroed R_ground."
                        ) from e
                    print(f"[EmbeddingClient] Failed after {self.max_retries} "
                          f"attempts: {e}. Returning zero embeddings "
                          f"({self._consecutive_failures}/{self.fail_after} "
                          f"consecutive failures before abort).")
                    # Return zero embeddings with correct dimension so
                    # retrieval produces low (but not crash-inducing) scores.
                    dim = self._embed_dim or 1
                    return np.zeros((len(prefixed), dim), dtype=np.float32)
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
        json_schema: dict[str, Any] | None = None,
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

    def _build_messages(self, item: dict[str, Any]) -> list[dict[str, str]]:
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

    def _judge_single(self, item: dict[str, Any]) -> dict[str, Any]:
        """Send a single judge request with retries."""
        messages = self._build_messages(item)

        request_body: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 256,
            # Suppress thinking for Qwen3 models — judge output is
            # short structured JSON, not a reasoning chain.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.json_schema:
            request_body["response_format"] = _json_schema_response_format(self.json_schema)

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

    def judge_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        item: dict[str, Any],
        system_prompt: str,
        prompt_template: str,
        json_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
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

        request_body: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_schema:
            request_body["response_format"] = _json_schema_response_format(json_schema)

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
        items: list[dict[str, Any]],
        system_prompt: str,
        prompt_template: str,
        json_schema: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
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

    def _ranking_single(
        self,
        item: dict[str, Any],
        system_prompt: str,
        prompt_template: str,
        json_schema: dict[str, Any] | None,
    ) -> list[dict[str, Any]] | None:
        """Send a single listwise ranking request with retries.

        Returns the validated rankings list (one entry per candidate, with
        candidate_index / rank / grounding_score), or None on failure —
        callers treat None as "no ranking signal" rather than all-zero.
        """
        n_candidates = int(item.get("n_candidates", 0))
        user_prompt = (
            prompt_template
            .replace("{{chunk_text}}", str(item.get("chunk_text", "")))
            .replace("{{norm_universe_json}}", str(item.get("norm_universe_json", "[]")))
            .replace("{{candidates_block}}", str(item.get("candidates_block", "")))
            .replace("{{n_candidates}}", str(n_candidates))
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        request_body: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            # Rankings list one entry per candidate plus an explanation —
            # larger than the per-flow judgments' 256-token budget.
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_schema:
            request_body["response_format"] = _json_schema_response_format(json_schema)

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
                if not (parsed and isinstance(parsed, dict)
                        and isinstance(parsed.get("rankings"), list)
                        and parsed["rankings"]):
                    raise ValueError(
                        f"no 'rankings' list in judge response: {content[:200]!r}")
                cleaned = []
                for entry in parsed["rankings"]:
                    if not isinstance(entry, dict):
                        continue
                    try:
                        cleaned.append({
                            "candidate_index": int(
                                entry.get("candidate_index",
                                          entry.get("candidate_id", -1))),
                            "rank": int(entry.get("rank", 0)),
                            "grounding_score": float(entry.get("grounding_score", 0.0)),
                        })
                    except (TypeError, ValueError):
                        continue
                valid_indices = {
                    e["candidate_index"] for e in cleaned
                    if 0 <= e["candidate_index"] < n_candidates
                }
                # Require coverage of every candidate; a partial
                # ranking would silently zero the missing ones.
                if len(valid_indices) == n_candidates:
                    return cleaned
                raise ValueError(
                    f"ranking covered {len(valid_indices)}/{n_candidates} "
                    f"candidates: {content[:200]!r}")
            except Exception as e:
                elapsed = time.time() - t0
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[JudgeClient] Ranking attempt {attempt + 1}/"
                          f"{self.max_retries} failed ({elapsed:.0f}s, "
                          f"{prompt_chars} chars): {e}")
                    time.sleep(wait)
                    continue
                print(f"[JudgeClient] Ranking failed after {self.max_retries} "
                      f"attempts ({elapsed:.0f}s, {prompt_chars} chars): {e}")
        return None

    def judge_ranking_batch(
        self,
        items: list[dict[str, Any]],
        system_prompt: str,
        prompt_template: str,
        json_schema: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]] | None]:
        """Rank candidate completions listwise, one request per group.

        Each item should have keys: chunk_text, norm_universe_json,
        candidates_block, n_candidates.

        Returns:
            One rankings list (or None on failure) per item.
        """
        if not items:
            return []

        print(f"[JudgeClient] Ranking batch: {len(items)} groups, "
              f"max_workers={min(self.max_workers, len(items))}")

        results: list[list[dict[str, Any]] | None] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as pool:
            future_to_idx = {
                pool.submit(
                    self._ranking_single, item,
                    system_prompt, prompt_template, json_schema,
                ): i
                for i, item in enumerate(items)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = None
        return results

    def close(self):
        self._session.close()


# Default grounding instruction for the reranker. Frames the relevance
# query as a normative-grounding question so the cross-encoder's yes-token
# probability approximates the LLM judge's norm_match/governance signal.
RERANKER_GROUNDING_INSTRUCTION = (
    "Given a set of social norms governing information flow, judge how well "
    "the candidate's extracted information flow is grounded in and consistent "
    "with those norms (norm awareness, flow governance, and appropriateness)."
)


class RerankerJudgeClient:
    """Drop-in replacement for :class:`JudgeClient` backed by a cross-encoder
    reranker (e.g. Qwen3-Reranker-8B) served by vLLM's ``/rerank`` endpoint.

    Duck-typed to the subset of ``JudgeClient`` that :class:`OnlineRGround`
    calls — ``judge_ranking_batch`` (the production ``rground_scoring=ranked``
    path), ``judge_batch`` (legacy absolute mode), ``judge_coverage_batch``
    (no-flow), and ``close`` — so it slots in behind the same construction
    site with no changes to the scoring/contrastive math.

    The reranker scores each candidate flow against the group's retrieved
    norm set, producing a continuous relevance score in ``[0, 1]`` used
    directly as ``grounding_score``; ranks are derived by sorting those
    scores. Because the scores are continuous (not the quantized text the
    generative judge emits), within-group ties — the pathology that forced
    listwise judging — are rare, so the rank component falls out naturally.

    Relevance ≠ the full judge rubric: the reranker covers norm_match +
    governance but cannot see *appropriateness consistency* (the deontic
    direction — does the model's appropriate/inappropriate verdict agree with
    the governing norm?). That axis is restored deterministically: the
    governing norm's Raz ``normative_force`` implies an expected appropriateness
    (see :mod:`.deontic`), compared against the candidate's own labels. The
    final per-candidate grounding blends the reranker relevance with this
    appropriateness-consistency by ``app_weight`` (default 0.2, mirroring the
    LLM judge's 0.4/0.4/0.2 norm_match/governance/appropriateness split).
    Set ``app_weight=0`` for a pure-relevance ablation.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8003",
        model_name: str = "default",
        instruction: str = RERANKER_GROUNDING_INSTRUCTION,
        timeout: float = 600.0,
        max_workers: int = 16,
        max_retries: int = 4,
        max_doc_chars: int = 4000,
        app_weight: float = 0.2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.instruction = instruction
        self.timeout = timeout
        self.max_workers = max_workers
        self.max_retries = max_retries
        # Candidate flow JSON is short, but a degenerate completion can emit
        # a huge extraction list; cap document length so one outlier doesn't
        # blow the reranker's context window.
        self.max_doc_chars = max_doc_chars
        # Blend weight of the deontic appropriateness-consistency term against
        # the reranker relevance (norm_match+governance). 0.2 ≈ the LLM judge's
        # appropriateness share; 0.0 disables it (pure-relevance ablation).
        self.app_weight = max(0.0, min(1.0, app_weight))
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Low-level rerank call
    # ------------------------------------------------------------------

    def _build_query(self, norm_universe_json: str, chunk_text: str = "") -> str:
        """Build the reranker query (the grounding rubric) from norms + context.

        vLLM's Qwen3-Reranker score adapter already wraps the query in the
        ``<Instruct>/<Query>/<Document>`` chat template, so we use plain prose
        labels here (re-using ``<Instruct>:``/``<Query>:`` would nest the tags).
        The grounding instruction is carried in the query text so the relevance
        score reflects normative grounding rather than generic topical overlap.
        """
        parts = [self.instruction]
        if chunk_text:
            parts.append(f"Context passage: {chunk_text[:self.max_doc_chars]}")
        parts.append(f"Governing norms: {norm_universe_json}")
        return "\n".join(parts)

    def _rerank(self, query: str, documents: list[str]) -> list[float] | None:
        """Score ``documents`` against ``query`` via vLLM ``/rerank``.

        Returns one relevance score per document (aligned to input order),
        or ``None`` on persistent failure so callers can treat the group as
        judge-failed (uniform neutral score) rather than all-zero.
        """
        if not documents:
            return []
        docs = [d[:self.max_doc_chars] if d else "" for d in documents]
        request_body = {
            "model": self.model_name,
            "query": query,
            "documents": docs,
        }
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                resp = self._session.post(
                    f"{self.base_url}/rerank",
                    json=request_body,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                # vLLM returns one entry per document with an ``index`` into
                # the input list and a ``relevance_score``; realign to input
                # order (the endpoint may sort by score).
                scores = [0.0] * len(docs)
                seen = 0
                for r in results:
                    idx = r.get("index")
                    if idx is None or not (0 <= idx < len(docs)):
                        continue
                    score = r.get("relevance_score", r.get("score", 0.0))
                    scores[idx] = max(0.0, min(1.0, float(score)))
                    seen += 1
                if seen == len(docs):
                    return scores
                raise ValueError(
                    f"rerank covered {seen}/{len(docs)} documents: "
                    f"{str(results)[:200]!r}")
            except Exception as e:
                elapsed = time.time() - t0
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[RerankerJudgeClient] Attempt {attempt + 1}/"
                          f"{self.max_retries} failed ({elapsed:.0f}s, "
                          f"{len(docs)} docs): {e}")
                    time.sleep(wait)
                    continue
                print(f"[RerankerJudgeClient] Failed after {self.max_retries} "
                      f"attempts ({elapsed:.0f}s, {len(docs)} docs): {e}")
        return None

    # ------------------------------------------------------------------
    # JudgeClient-compatible API
    # ------------------------------------------------------------------

    def _ranking_single(self, item: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Score one group's candidates and emit a JudgeClient-style ranking.

        Expects ``item['candidates']`` (per-candidate document text, injected
        by OnlineRGround alongside the legacy ``candidates_block``). Returns
        one entry per candidate with candidate_index / rank / grounding_score,
        or None on rerank failure.
        """
        candidates = item.get("candidates")
        if candidates is None:
            # Fall back to splitting the joined block if the structured list
            # was not provided (keeps the client usable standalone).
            candidates = _split_candidates_block(
                str(item.get("candidates_block", "")),
                int(item.get("n_candidates", 0)),
            )
        n = len(candidates)
        if n == 0:
            return None
        query = self._build_query(
            str(item.get("norm_universe_json", "[]")),
            str(item.get("chunk_text", "")),
        )
        relevance = self._rerank(query, list(candidates))
        if relevance is None:
            return None
        # Restore the appropriateness-consistency axis the reranker can't see:
        # the governing norm (top retrieved) implies an expected appropriateness;
        # each candidate's own labels are scored against it. grounding =
        # (1-app_weight)*relevance + app_weight*appropriateness_consistency.
        if self.app_weight > 0.0:
            force = governing_norm_force(item.get("norm_universe_json", "[]"))
            grounding = [
                (1.0 - self.app_weight) * relevance[i]
                + self.app_weight
                * candidate_appropriateness_consistency(candidates[i], force)
                for i in range(n)
            ]
        else:
            grounding = list(relevance)
        # Derive ranks from the blended grounding (descending); stable sort
        # gives distinct ranks even on the rare score tie.
        order = sorted(range(n), key=lambda i: grounding[i], reverse=True)
        rankings = [None] * n
        for rank_pos, cand_idx in enumerate(order, start=1):
            rankings[cand_idx] = {
                "candidate_index": cand_idx,
                "rank": rank_pos,
                "grounding_score": grounding[cand_idx],
            }
        return rankings

    def judge_ranking_batch(
        self,
        items: list[dict[str, Any]],
        system_prompt: str = "",
        prompt_template: str = "",
        json_schema: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]] | None]:
        """Rank candidates per group via reranker. Signature mirrors
        ``JudgeClient.judge_ranking_batch``; the prompt/schema args are
        accepted for interface compatibility and ignored (the reranker has
        no generative prompt)."""
        if not items:
            return []
        print(f"[RerankerJudgeClient] Ranking batch: {len(items)} groups, "
              f"max_workers={min(self.max_workers, len(items))}")
        results: list[list[dict[str, Any]] | None] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as pool:
            future_to_idx = {
                pool.submit(self._ranking_single, item): i
                for i, item in enumerate(items)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = None
        return results

    def judge_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Per-flow absolute scoring (legacy ``rground_scoring=absolute``).

        Maps the reranker relevance to norm_match_score = governance_score =
        score, and adjudicates ``appropriateness_consistent`` from the deontic
        force of the governing (top-retrieved) norm vs the flow's own
        appropriateness label (see :mod:`.deontic`). When the axis is
        undetermined (permitted/unknown force, ambiguous/missing label) it
        resolves to the neutral 0.5 → True, matching the LLM judge's bool
        contract (the consumer reads it as ``1.0 if consistent else 0.0``).
        """
        if not items:
            return []

        def _one(item: dict[str, Any]) -> dict[str, Any]:
            query = self._build_query(
                str(item.get("norm_universe_json", "[]")),
                str(item.get("chunk_text", "")),
            )
            flow_json = str(item.get("flow_json", "{}"))
            scores = self._rerank(query, [flow_json])
            if not scores:
                return {"norm_match_score": 0.0, "governance_score": 0.0,
                        "appropriateness_consistent": False}
            s = scores[0]
            force = governing_norm_force(item.get("norm_universe_json", "[]"))
            consistency = candidate_appropriateness_consistency(flow_json, force)
            return {
                "norm_match_score": s,
                "governance_score": s,
                "appropriateness_consistent": consistency >= 0.5,
            }

        results: list[dict[str, Any]] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as pool:
            future_to_idx = {pool.submit(_one, it): i for i, it in enumerate(items)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = {"norm_match_score": 0.0, "governance_score": 0.0,
                                    "appropriateness_consistent": False}
        return results

    def judge_coverage_batch(
        self,
        items: list[dict[str, Any]],
        system_prompt: str = "",
        prompt_template: str = "",
        json_schema: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """No-flow coverage via reranker: score the chunk against its norms.

        High relevance ⇒ the passage contains norm-governed flows (so a
        no-flow declaration is unjustified). Mirrors JudgeClient's coverage
        output shape; prompt/schema args ignored.
        """
        if not items:
            return []

        def _one(item: dict[str, Any]) -> dict[str, Any]:
            query = self._build_query(str(item.get("norm_universe_json", "[]")))
            scores = self._rerank(query, [str(item.get("chunk_text", ""))])
            if not scores:
                return {"coverage_score": 0.0,
                        "passage_contains_governed_flows": False}
            s = scores[0]
            return {
                "coverage_score": s,
                "passage_contains_governed_flows": s >= 0.5,
            }

        results: list[dict[str, Any]] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as pool:
            future_to_idx = {pool.submit(_one, it): i for i, it in enumerate(items)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = {"coverage_score": 0.0,
                                    "passage_contains_governed_flows": False}
        return results

    def close(self):
        self._session.close()


def _split_candidates_block(block: str, n_candidates: int) -> list[str]:
    """Recover per-candidate texts from a joined ``candidates_block``.

    OnlineRGround builds the block as ``### Candidate {i}\\n{text}`` joined by
    blank lines; this reverses it. Used only as a fallback when the structured
    ``candidates`` list is absent.
    """
    if not block:
        return []
    parts = block.split("### Candidate ")
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Drop the leading "<idx>\n" header.
        nl = p.find("\n")
        out.append(p[nl + 1:].strip() if nl != -1 else p)
    if n_candidates and len(out) != n_candidates:
        # Coverage mismatch — let the caller treat as failure rather than
        # silently mis-aligning candidates.
        return out
    return out


class NormRetriever:
    """In-memory top-k norm retrieval with aligned embeddings.

    To guarantee query and norm embeddings are in the same space, norms
    are re-embedded via the same vLLM embedding server used for queries
    at init time.  Pre-computed .npy files are used as a fallback only.
    """

    def __init__(
        self,
        norm_universes: dict[str, list],
        embeddings_dir: str,
        embedding_client: EmbeddingClient | None = None,
        top_k: int = 3,
    ):
        import os

        self.norm_universes = norm_universes
        self.top_k = top_k
        self._embeddings: dict[str, np.ndarray] = {}

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
        contrastive_source: str | None = None,
        return_scores: bool = False,
        top_k: int | None = None,
    ):
        """Retrieve top-k norms most relevant to a query embedding.

        Args:
            query_embedding: 1-D normalized embedding vector.
            source_id: The source book's ID.
            contrastive_source: If set, retrieve from this (wrong) source.
            return_scores: If True, return (json_str, top_k_similarities).
            top_k: Per-call override of the retriever's default k (used by
                group-level retrieval in ranked R_ground scoring).

        Returns:
            JSON string of top-k norm dicts, or (json_str, sims) tuple.
        """
        k = top_k if top_k is not None else self.top_k
        target_id = contrastive_source or source_id
        norms = self.norm_universes.get(target_id, [])
        if not norms:
            return ("[]", []) if return_scores else "[]"
        if len(norms) <= k:
            result = json.dumps(norms, ensure_ascii=False, indent=1)
            return (result, [1.0] * len(norms)) if return_scores else result

        emb_matrix = self._embeddings.get(target_id)
        if emb_matrix is None:
            result = json.dumps(norms[:k], ensure_ascii=False, indent=1)
            return (result, []) if return_scores else result

        # Degenerate query (zero-embedding fallback after a failed encode
        # call) — no meaningful retrieval, and a dim-1 placeholder would
        # crash the matmul.
        if (query_embedding.shape[-1] != emb_matrix.shape[1]
                or not query_embedding.any()):
            return ("[]", []) if return_scores else "[]"

        # Cosine similarity (both are L2-normalized)
        sims = emb_matrix @ query_embedding
        top_indices = np.argsort(sims)[-k:][::-1]
        selected = [norms[i] for i in top_indices]
        top_sims = [round(float(sims[i]), 4) for i in top_indices]
        result = json.dumps(selected, ensure_ascii=False, indent=1)
        return (result, top_sims) if return_scores else result

    def retrieve_batch(
        self,
        query_embeddings: np.ndarray,
        source_ids: list[str],
        contrastive_sources: list[str | None],
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
