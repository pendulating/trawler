"""Online R_ground evaluation for GRPO training.

Replaces the cached R_ground lookup with live evaluation: parses each
completion's flows, embeds queries, retrieves top-k norms, and calls
the judge server — all batched across completions within a single
reward function invocation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .clients import EmbeddingClient, JudgeClient, NormRetriever
from .rewards import _parse_completion


def _flow_to_query(flow: Dict[str, Any]) -> str:
    """Build a retrieval query from a single flow's CI tuple fields.

    Mirrors the logic in reward_prep.py so retrieval queries produce
    comparable results to the offline reward prep stage.
    """
    parts = []
    for key in (
        "sender", "recipient", "information_type",
        "context", "transmission_principle", "subject",
    ):
        val = flow.get(key, "")
        if val:
            parts.append(str(val))
    invoked = flow.get("norms_invoked", [])
    if isinstance(invoked, list):
        parts.extend(str(n) for n in invoked)
    return " ".join(parts) if parts else "information flow"


def _flatten_flow(extraction: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested flow tuple into a single dict for query building."""
    flow_tuple = extraction.get("flow", {})
    if isinstance(flow_tuple, dict):
        return {**flow_tuple, **extraction}
    return extraction


class OnlineRGround:
    """Batched online normative grounding evaluation.

    Called by CompositeRewardFunction.__call__ with all completions from
    a single training step.  Batches embedding and judge calls for
    efficiency.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        judge_client: JudgeClient,
        norm_retriever: NormRetriever,
    ):
        self.embedding_client = embedding_client
        self.judge_client = judge_client
        self.norm_retriever = norm_retriever
        self._consecutive_zero_batches = 0
        self._total_calls = 0

    def __call__(
        self,
        completions: List[str],
        prompts: List[str],
        metadata_list: List[Dict[str, Any]],
    ) -> List[float]:
        """Evaluate R_ground for a batch of completions.

        Args:
            completions: Raw completion texts (think blocks already stripped).
            prompts: Corresponding prompt texts.
            metadata_list: Per-completion dicts with source_id, prompt_id,
                is_contrastive, and optionally contrastive_source.

        Returns:
            List of R_ground scores (0.0–1.0), one per completion.
        """
        # ---------------------------------------------------------------
        # Phase 1: Parse completions and collect per-flow queries
        # ---------------------------------------------------------------
        # For each completion, track which flows belong to it
        completion_flow_ranges: List[tuple] = []  # (start_idx, count)
        all_queries: List[str] = []
        all_chunk_texts: List[str] = []
        all_flow_jsons: List[str] = []
        all_source_ids: List[str] = []
        all_contrastive_sources: List[Optional[str]] = []

        for i, completion in enumerate(completions):
            meta = metadata_list[i] if i < len(metadata_list) else {}
            source_id = meta.get("source_id", "")
            is_contrastive = meta.get("is_contrastive", False)
            contrastive_source = meta.get("contrastive_source") if is_contrastive else None

            # Use raw chunk text (pre-template) from metadata so the judge
            # sees clean source text, not chat-templated text with special tokens.
            chunk_text = meta.get("chunk_text", "")
            if not chunk_text:
                chunk_text = prompts[i] if i < len(prompts) else ""

            parsed = _parse_completion(completion)
            extractions = []
            if parsed:
                extractions = parsed.get("extraction", [])
                if not isinstance(extractions, list):
                    extractions = []

            start_idx = len(all_queries)
            flow_count = 0

            for ext in extractions:
                if not isinstance(ext, dict):
                    continue
                flat = _flatten_flow(ext)
                query = _flow_to_query(flat)
                flow_json = json.dumps(ext, ensure_ascii=False, indent=1)

                all_queries.append(query)
                all_chunk_texts.append(chunk_text)
                all_flow_jsons.append(flow_json)
                all_source_ids.append(source_id)
                all_contrastive_sources.append(contrastive_source)
                flow_count += 1

            completion_flow_ranges.append((start_idx, flow_count))

        # Short-circuit: no flows across all completions
        if not all_queries:
            return [0.0] * len(completions)

        # ---------------------------------------------------------------
        # Phase 2: Batch embed all flow queries
        # ---------------------------------------------------------------
        query_embeddings = self.embedding_client.encode_batch(all_queries)

        # ---------------------------------------------------------------
        # Phase 3: Retrieve top-k norms for each flow
        # ---------------------------------------------------------------
        retrieved_norms = self.norm_retriever.retrieve_batch(
            query_embeddings, all_source_ids, all_contrastive_sources,
        )

        # ---------------------------------------------------------------
        # Phase 4: Batch judge all flows
        # ---------------------------------------------------------------
        judge_items = [
            {
                "chunk_text": all_chunk_texts[j],
                "flow_json": all_flow_jsons[j],
                "norm_universe_json": retrieved_norms[j],
            }
            for j in range(len(all_queries))
        ]
        judge_results = self.judge_client.judge_batch(judge_items)

        # ---------------------------------------------------------------
        # Phase 5: Aggregate per-flow scores back to per-completion
        # ---------------------------------------------------------------
        scores: List[float] = []
        for start_idx, flow_count in completion_flow_ranges:
            if flow_count == 0:
                scores.append(0.0)
                continue
            total = 0.0
            for j in range(start_idx, start_idx + flow_count):
                result = judge_results[j]
                nm = result.get("norm_match_score", 0.0)
                gov = result.get("governance_score", 0.0)
                total += 0.5 * nm + 0.5 * gov
            scores.append(total / flow_count)

        # Track consecutive all-zero batches to detect server failures
        self._total_calls += 1
        if all(s == 0.0 for s in scores):
            self._consecutive_zero_batches += 1
            if self._consecutive_zero_batches >= 5:
                print(
                    f"[OnlineRGround] WARNING: {self._consecutive_zero_batches} "
                    f"consecutive all-zero batches. Embedding or judge server "
                    f"may be down. R_ground is providing a biased negative "
                    f"signal. Check server health."
                )
        else:
            self._consecutive_zero_batches = 0

        return scores
