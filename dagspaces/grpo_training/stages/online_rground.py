"""Online R_ground evaluation for GRPO training.

Replaces the cached R_ground lookup with live evaluation: parses each
completion's flows, embeds queries, retrieves top-k norms, and calls
the judge server — all batched across completions within a single
reward function invocation.

Per-completion contrastive scoring: every completion is judged against
BOTH the correct source's norms AND a random wrong source's norms.
R_ground = correct_score - λ * wrong_score, clamped to [0, 1].
This replaces the old additive-row contrastive mechanism.
"""

from __future__ import annotations

import json
import random
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


def _norm_snippet(norm_json_str: str, max_norms: int = 3) -> List[str]:
    """Extract short summaries from retrieved norm JSON for tracing."""
    try:
        norms = json.loads(norm_json_str)
        snippets = []
        for n in norms[:max_norms]:
            art = n.get("norm_articulation") or n.get("articulation", "")
            if art:
                snippets.append(art[:120])
            else:
                parts = []
                for k in ("prescriptive_element", "norm_subject", "norm_act"):
                    v = n.get(k, "")
                    if v:
                        parts.append(str(v))
                snippets.append(" ".join(parts)[:120] or "(empty norm)")
        return snippets
    except (json.JSONDecodeError, TypeError):
        return []


def _pick_wrong_source(source_id: str, all_source_ids: List[str]) -> Optional[str]:
    """Pick a random source ID different from the correct one."""
    candidates = [s for s in all_source_ids if s != source_id]
    return random.choice(candidates) if candidates else None


class OnlineRGround:
    """Batched online normative grounding evaluation with per-completion
    contrastive scoring.

    Called by CompositeRewardFunction.__call__ with all completions from
    a single training step.  For each completion:

    1. Evaluate R_ground against the **correct** source's norms.
    2. Evaluate R_ground against a **random wrong** source's norms.
    3. Compute: ``R_ground = clamp(correct - λ * wrong, 0, 1)``

    This produces within-group contrastive signal: completions whose
    extracted norms discriminate between the correct and wrong normative
    universes get higher R_ground.

    For no-flow completions, performs *coverage scoring* against both
    correct and wrong sources.

    After each call, ``last_diagnostics`` contains per-completion
    diagnostics for trace logging.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        judge_client: JudgeClient,
        norm_retriever: NormRetriever,
        all_source_ids: Optional[List[str]] = None,
        contrastive_lambda: float = 0.5,
        no_flow_judge_system_prompt: str = "",
        no_flow_judge_prompt_template: str = "",
        no_flow_judge_json_schema: Optional[Dict] = None,
    ):
        self.embedding_client = embedding_client
        self.judge_client = judge_client
        self.norm_retriever = norm_retriever
        self.all_source_ids = all_source_ids or []
        self.contrastive_lambda = contrastive_lambda
        self._no_flow_system_prompt = no_flow_judge_system_prompt
        self._no_flow_prompt_template = no_flow_judge_prompt_template
        self._no_flow_json_schema = no_flow_judge_json_schema
        self._consecutive_zero_batches = 0
        self._total_calls = 0
        self.last_diagnostics: List[List[Dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        completions: List[str],
        prompts: List[str],
        metadata_list: List[Dict[str, Any]],
    ) -> List[float]:
        """Evaluate R_ground for a batch of completions.

        Each completion is scored against the correct source AND a random
        wrong source.  Final score = clamp(correct - λ * wrong, 0, 1).

        Returns:
            List of R_ground scores (0.0–1.0), one per completion.
        """
        # ---------------------------------------------------------------
        # Phase 1: Parse completions and collect per-flow queries
        # ---------------------------------------------------------------
        completion_flow_ranges: List[tuple] = []  # (start_idx, count)
        completion_valid_no_flow: List[bool] = []
        completion_source_ids: List[str] = []
        completion_wrong_sources: List[Optional[str]] = []

        all_queries: List[str] = []
        all_chunk_texts: List[str] = []
        all_flow_jsons: List[str] = []
        all_source_ids: List[str] = []

        for i, completion in enumerate(completions):
            meta = metadata_list[i] if i < len(metadata_list) else {}
            source_id = meta.get("source_id", "")
            completion_source_ids.append(source_id)
            completion_wrong_sources.append(
                _pick_wrong_source(source_id, self.all_source_ids)
            )

            chunk_text = meta.get("chunk_text", "")
            if not chunk_text:
                chunk_text = prompts[i] if i < len(prompts) else ""

            parsed = _parse_completion(completion)
            extractions = []
            is_valid_no_flow = False
            if parsed:
                extractions = parsed.get("extraction", [])
                if not isinstance(extractions, list):
                    extractions = []
                reasoning = parsed.get("reasoning", {})
                if isinstance(reasoning, dict):
                    has_exchange = reasoning.get("has_information_exchange")
                    if has_exchange is False and len(extractions) == 0:
                        is_valid_no_flow = True
            completion_valid_no_flow.append(is_valid_no_flow)

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
                flow_count += 1

            completion_flow_ranges.append((start_idx, flow_count))

        # Short-circuit: all completions are no-flow / parse failures.
        if not all_queries:
            return self._score_no_flow_only(
                completions, metadata_list,
                completion_valid_no_flow,
                completion_source_ids, completion_wrong_sources,
            )

        # ---------------------------------------------------------------
        # Phase 2: Batch embed all flow queries
        # ---------------------------------------------------------------
        query_embeddings = self.embedding_client.encode_batch(all_queries)

        # ---------------------------------------------------------------
        # Phase 3: Retrieve norms from CORRECT source and judge
        # ---------------------------------------------------------------
        correct_sources = [None] * len(all_queries)  # None → use source_id
        correct_retrieval = self.norm_retriever.retrieve_batch(
            query_embeddings, all_source_ids, correct_sources,
            return_scores=True,
        )
        correct_norms = [r[0] for r in correct_retrieval]
        correct_sims = [r[1] for r in correct_retrieval]

        correct_judge_items = [
            {
                "chunk_text": all_chunk_texts[j],
                "flow_json": all_flow_jsons[j],
                "norm_universe_json": correct_norms[j],
            }
            for j in range(len(all_queries))
        ]
        correct_judge_results = self.judge_client.judge_batch(correct_judge_items)

        # ---------------------------------------------------------------
        # Phase 4: Retrieve norms from WRONG source and judge
        # ---------------------------------------------------------------
        # Map each flow to its completion's wrong source
        flow_wrong_sources = []
        for comp_idx, (start_idx, flow_count) in enumerate(completion_flow_ranges):
            wrong = completion_wrong_sources[comp_idx]
            for _ in range(flow_count):
                flow_wrong_sources.append(wrong)

        do_contrastive = (
            self.contrastive_lambda > 0.0
            and len(self.all_source_ids) > 1
        )

        wrong_judge_results: List[Dict[str, Any]] = []
        wrong_norms: List[str] = []
        wrong_sims: List[list] = []
        if do_contrastive:
            wrong_retrieval = self.norm_retriever.retrieve_batch(
                query_embeddings, all_source_ids, flow_wrong_sources,
                return_scores=True,
            )
            wrong_norms = [r[0] for r in wrong_retrieval]
            wrong_sims = [r[1] for r in wrong_retrieval]

            wrong_judge_items = [
                {
                    "chunk_text": all_chunk_texts[j],
                    "flow_json": all_flow_jsons[j],
                    "norm_universe_json": wrong_norms[j],
                }
                for j in range(len(all_queries))
            ]
            wrong_judge_results = self.judge_client.judge_batch(wrong_judge_items)

        # ---------------------------------------------------------------
        # Phase 5: Aggregate per-flow scores with contrastive margin
        # ---------------------------------------------------------------
        scores: List[float] = [0.0] * len(completions)
        self.last_diagnostics = [[] for _ in completions]
        no_flow_indices: List[int] = []

        for comp_idx, (start_idx, flow_count) in enumerate(completion_flow_ranges):
            if flow_count == 0:
                if completion_valid_no_flow[comp_idx]:
                    no_flow_indices.append(comp_idx)
                continue

            correct_total = 0.0
            wrong_total = 0.0
            flow_diags = []

            for j in range(start_idx, start_idx + flow_count):
                # Correct source score (0.4 norm_match + 0.4 governance + 0.2 appropriateness)
                cr = correct_judge_results[j]
                c_nm = cr.get("norm_match_score", 0.0)
                c_gov = cr.get("governance_score", 0.0)
                c_ac = 1.0 if cr.get("appropriateness_consistent", False) else 0.0
                c_score = 0.4 * c_nm + 0.4 * c_gov + 0.2 * c_ac
                correct_total += c_score

                # Wrong source score
                w_score = 0.0
                w_nm = 0.0
                w_gov = 0.0
                w_ac = 0.0
                if do_contrastive and j < len(wrong_judge_results):
                    wr = wrong_judge_results[j]
                    w_nm = wr.get("norm_match_score", 0.0)
                    w_gov = wr.get("governance_score", 0.0)
                    w_ac = 1.0 if wr.get("appropriateness_consistent", False) else 0.0
                    w_score = 0.4 * w_nm + 0.4 * w_gov + 0.2 * w_ac
                wrong_total += w_score

                diag: Dict[str, Any] = {
                    "query": all_queries[j],
                    "source_id": all_source_ids[j],
                    "correct_norm_match": round(c_nm, 4),
                    "correct_governance": round(c_gov, 4),
                    "correct_appropriateness": c_ac,
                    "correct_score": round(c_score, 4),
                    "correct_retrieval_sims": correct_sims[j],
                    "correct_norm_snippets": _norm_snippet(correct_norms[j]),
                }
                if do_contrastive:
                    diag.update({
                        "wrong_source": flow_wrong_sources[j],
                        "wrong_norm_match": round(w_nm, 4),
                        "wrong_governance": round(w_gov, 4),
                        "wrong_appropriateness": w_ac,
                        "wrong_score": round(w_score, 4),
                        "wrong_retrieval_sims": wrong_sims[j] if j < len(wrong_sims) else [],
                        "wrong_norm_snippets": _norm_snippet(wrong_norms[j]) if j < len(wrong_norms) else [],
                    })
                flow_diags.append(diag)

            avg_correct = correct_total / flow_count
            avg_wrong = wrong_total / flow_count if do_contrastive else 0.0

            # R_ground = clamp(correct - λ * wrong, 0, 1)
            raw = avg_correct - self.contrastive_lambda * avg_wrong
            scores[comp_idx] = max(0.0, min(1.0, raw))
            self.last_diagnostics[comp_idx] = flow_diags

        # Score no-flow completions via coverage judge (with contrastive)
        if no_flow_indices:
            nf_scores, nf_diags = self._score_no_flow_coverage(
                no_flow_indices, metadata_list,
                completion_source_ids, completion_wrong_sources,
            )
            for idx, sc, diag in zip(no_flow_indices, nf_scores, nf_diags):
                scores[idx] = sc
                self.last_diagnostics[idx] = diag

        # Track consecutive all-zero batches to detect server failures
        self._total_calls += 1
        if all(s == 0.0 for s in scores):
            self._consecutive_zero_batches += 1
            if self._consecutive_zero_batches >= 5:
                print(
                    f"[OnlineRGround] WARNING: {self._consecutive_zero_batches} "
                    f"consecutive all-zero batches. Embedding or judge server "
                    f"may be down."
                )
        else:
            self._consecutive_zero_batches = 0

        return scores

    # ------------------------------------------------------------------
    # No-flow coverage scoring (with contrastive)
    # ------------------------------------------------------------------

    def _coverage_score_to_rground(
        self,
        correct_coverage: float,
        wrong_coverage: float,
        gold_has_exchange: bool | None,
    ) -> float:
        """Map dual coverage scores + gold label to an R_ground value.

        correct_coverage: coverage from the correct source's norms.
        wrong_coverage: coverage from a wrong source's norms.
        gold_has_exchange: whether the passage actually contains flows.

        Uses the contrastive margin (correct - wrong) to modulate the
        base gold-label-aware score.
        """
        cc = max(0.0, min(1.0, correct_coverage))
        wc = max(0.0, min(1.0, wrong_coverage))

        if gold_has_exchange is False:
            # Correct no-flow.  Base: 0.7–0.9 depending on correct coverage.
            # Low correct coverage = fully justified → higher score.
            base = 0.7 + 0.2 * (1.0 - cc)
            # Contrastive bonus: if wrong source has HIGHER coverage than
            # correct source, the model is right to declare no-flow for the
            # correct source.  Subtle signal.
            margin = wc - cc  # positive if wrong source is more relevant
            return max(0.0, min(1.0, base + 0.1 * margin))
        elif gold_has_exchange is True:
            # Wrong no-flow.  Base: 0.0–0.2 based on correct coverage.
            base = 0.2 * (1.0 - cc)
            # Contrastive penalty: if correct source covers this well
            # but wrong source doesn't, the model missed something specific
            # to the correct source.
            margin = cc - wc  # positive if correct source is more relevant
            return max(0.0, min(1.0, base - 0.1 * margin))
        else:
            return 0.4 + 0.1 * (1.0 - cc)

    def _score_no_flow_coverage(
        self,
        indices: List[int],
        metadata_list: List[Dict[str, Any]],
        completion_source_ids: List[str],
        completion_wrong_sources: List[Optional[str]],
    ) -> tuple:
        """Score no-flow completions via coverage judge against both sources.

        Returns (scores, diagnostics) with one entry per index.
        """
        chunk_texts = []
        source_ids = []
        wrong_sources = []
        golds = []
        for idx in indices:
            meta = metadata_list[idx] if idx < len(metadata_list) else {}
            chunk_texts.append(meta.get("chunk_text", ""))
            source_ids.append(completion_source_ids[idx])
            wrong_sources.append(completion_wrong_sources[idx])
            golds.append(meta.get("gold_has_exchange"))

        embeddings = self.embedding_client.encode_batch(chunk_texts)

        # Retrieve correct norms
        correct_ret = self.norm_retriever.retrieve_batch(
            embeddings, source_ids, [None] * len(indices),
            return_scores=True,
        )
        correct_norms = [r[0] for r in correct_ret]
        correct_sims = [r[1] for r in correct_ret]

        # Judge correct coverage
        correct_items = [
            {"chunk_text": chunk_texts[j], "norm_universe_json": correct_norms[j]}
            for j in range(len(indices))
        ]
        correct_results = self.judge_client.judge_coverage_batch(
            correct_items,
            system_prompt=self._no_flow_system_prompt,
            prompt_template=self._no_flow_prompt_template,
            json_schema=self._no_flow_json_schema,
        )

        # Contrastive: retrieve and judge wrong norms
        do_contrastive = (
            self.contrastive_lambda > 0.0
            and len(self.all_source_ids) > 1
        )
        wrong_results: List[Dict[str, Any]] = []
        wrong_norms_list: List[str] = []
        wrong_sims: List[list] = []
        if do_contrastive:
            wrong_ret = self.norm_retriever.retrieve_batch(
                embeddings, source_ids, wrong_sources,
                return_scores=True,
            )
            wrong_norms_list = [r[0] for r in wrong_ret]
            wrong_sims = [r[1] for r in wrong_ret]

            wrong_items = [
                {"chunk_text": chunk_texts[j], "norm_universe_json": wrong_norms_list[j]}
                for j in range(len(indices))
            ]
            wrong_results = self.judge_client.judge_coverage_batch(
                wrong_items,
                system_prompt=self._no_flow_system_prompt,
                prompt_template=self._no_flow_prompt_template,
                json_schema=self._no_flow_json_schema,
            )

        scores = []
        diags = []
        for j in range(len(indices)):
            cc = correct_results[j].get("coverage_score", 0.0)
            wc = wrong_results[j].get("coverage_score", 0.0) if do_contrastive else 0.0
            rg = self._coverage_score_to_rground(cc, wc, golds[j])
            scores.append(rg)

            diag: Dict[str, Any] = {
                "type": "no_flow_coverage",
                "source_id": source_ids[j],
                "gold_has_exchange": golds[j],
                "correct_coverage": round(cc, 4),
                "correct_sims": correct_sims[j],
                "correct_norm_snippets": _norm_snippet(correct_norms[j]),
                "r_ground": round(rg, 4),
            }
            if do_contrastive:
                diag.update({
                    "wrong_source": wrong_sources[j],
                    "wrong_coverage": round(wc, 4),
                    "wrong_sims": wrong_sims[j] if j < len(wrong_sims) else [],
                    "wrong_norm_snippets": _norm_snippet(wrong_norms_list[j]) if j < len(wrong_norms_list) else [],
                })
            diags.append([diag])

        return scores, diags

    def _score_no_flow_only(
        self,
        completions: List[str],
        metadata_list: List[Dict[str, Any]],
        completion_valid_no_flow: List[bool],
        completion_source_ids: List[str],
        completion_wrong_sources: List[Optional[str]],
    ) -> List[float]:
        """Handle the case where ALL completions are no-flow or parse failures."""
        no_flow_indices = [
            i for i in range(len(completions))
            if completion_valid_no_flow[i]
        ]

        scores = [0.0] * len(completions)
        self.last_diagnostics = [[] for _ in completions]

        if no_flow_indices:
            nf_scores, nf_diags = self._score_no_flow_coverage(
                no_flow_indices, metadata_list,
                completion_source_ids, completion_wrong_sources,
            )
            for idx, sc, diag in zip(no_flow_indices, nf_scores, nf_diags):
                scores[idx] = sc
                self.last_diagnostics[idx] = diag

        return scores
