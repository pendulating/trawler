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
from .deontic import (
    candidate_appropriateness_consistency,
    direction_multiplier,
    governing_norm_force,
)
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


def _rankings_to_scores(
    rankings: List[Dict[str, Any]],
    n_candidates: int,
    rank_weight: float = 0.5,
) -> List[float]:
    """Convert a listwise judge ranking into per-candidate scores in [0, 1].

    score_i = rank_weight * (n - rank_i) / (n - 1) + (1 - rank_weight) * grounding_i

    The rank component guarantees within-group discrimination (the judge's
    absolute scores are heavily quantized and tie); the grounding component
    keeps reward magnitudes comparable across groups, which matters for
    logging and for scale_rewards="none". With a single candidate the rank
    component is undefined, so the grounding score is used alone.

    Args:
        rankings: Entries with candidate_index, rank, grounding_score.
        n_candidates: Number of candidates the ranking covers.
        rank_weight: Blend weight of the rank component.

    Returns:
        One score per candidate index; candidates missing from the ranking
        get 0.0 (judge_ranking_batch enforces full coverage upstream).
    """
    scores = [0.0] * n_candidates
    if n_candidates <= 0:
        return scores
    for entry in rankings:
        idx = entry.get("candidate_index", -1)
        if not 0 <= idx < n_candidates:
            continue
        grounding = max(0.0, min(1.0, float(entry.get("grounding_score", 0.0))))
        if n_candidates == 1:
            scores[idx] = grounding
            continue
        rank = max(1, min(n_candidates, int(entry.get("rank", n_candidates))))
        rank_component = (n_candidates - rank) / (n_candidates - 1)
        scores[idx] = rank_weight * rank_component + (1.0 - rank_weight) * grounding
    return scores


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
        scoring_mode: str = "absolute",
        ranking_system_prompt: str = "",
        ranking_prompt_template: str = "",
        ranking_json_schema: Optional[Dict] = None,
        rank_top_k: int = 5,
        rank_weight: float = 0.5,
        app_weight: float = 0.0,
        app_mode: str = "additive",
        app_floor: float = 0.4,
    ):
        if scoring_mode not in ("absolute", "ranked"):
            raise ValueError(f"Unknown rground scoring mode: {scoring_mode!r}")
        if not 0.0 <= app_weight <= 1.0:
            raise ValueError(f"app_weight must be in [0, 1], got {app_weight}")
        if app_mode not in ("additive", "multiplicative"):
            raise ValueError(f"Unknown app_mode: {app_mode!r}")
        if not 0.0 <= app_floor <= 1.0:
            raise ValueError(f"app_floor must be in [0, 1], got {app_floor}")
        self.embedding_client = embedding_client
        self.judge_client = judge_client
        self.norm_retriever = norm_retriever
        self.all_source_ids = all_source_ids or []
        self.contrastive_lambda = contrastive_lambda
        self._no_flow_system_prompt = no_flow_judge_system_prompt
        self._no_flow_prompt_template = no_flow_judge_prompt_template
        self._no_flow_json_schema = no_flow_judge_json_schema
        self.scoring_mode = scoring_mode
        self._ranking_system_prompt = ranking_system_prompt
        self._ranking_prompt_template = ranking_prompt_template
        self._ranking_json_schema = ranking_json_schema
        self.rank_top_k = rank_top_k
        self.rank_weight = rank_weight
        # Programmatic deontic appropriateness-consistency blend weight. The
        # ranked judge scores *grounding* (norm awareness + flow governance) but
        # is a weak, holistic signal for whether the model's appropriate /
        # inappropriate verdict agrees with the governing norm's deontic force —
        # the core of context-relative CI reasoning. When >0, blend the
        # deterministic deontic check into R_ground:
        #   R = clamp((1-app_weight)*(rank_blend - λ*wrong) + app_weight*app_cons)
        # Mirrors RerankerJudgeClient's app_weight (which is blind to direction).
        # 0.0 = legacy (grounding only).
        self.app_weight = float(app_weight)
        # app_mode (v9): "additive" = legacy 0.3 blend (R = (1-w)·base + w·app_cons);
        # "multiplicative" = the two-sided directional reward R = base · direction(app_cons),
        # where direction = app_floor + (1-app_floor)·app_cons. Multiplicative makes a
        # wrong appropriateness verdict (e.g. a violation called "appropriate") cost a
        # large fraction of R_ground instead of a diluted additive sliver.
        self.app_mode = app_mode
        self.app_floor = float(app_floor)
        self._consecutive_zero_batches = 0
        self._total_calls = 0
        self.last_diagnostics: List[List[Dict[str, Any]]] = []
        self.last_health: Dict[str, float] = {}

    def _push_health(self, metrics: Dict[str, Any]) -> None:
        """Surface per-call reward health under ``rground/*`` on the W&B run.

        One bounded scalar set per reward call. Uses ``commit=False`` so the
        values merge into TRL's next step commit (same x-axis as reward/kl)
        instead of fabricating extra steps. Also kept on ``self.last_health``
        for tests and offline inspection. A judge that dies mid-run shows up
        here immediately instead of only in stdout.
        """
        out: Dict[str, float] = {}
        for k, v in metrics.items():
            try:
                out[f"rground/{k}"] = round(float(v), 4)
            except (TypeError, ValueError):
                continue
        out["rground/consecutive_zero_batches"] = float(self._consecutive_zero_batches)
        emb_failures = getattr(self.embedding_client, "_consecutive_failures", None)
        if emb_failures is not None:
            out["rground/embedding_consecutive_failures"] = float(emb_failures)
        self.last_health = out
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(out, commit=False)
        except Exception:
            pass

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
        if self.scoring_mode == "ranked":
            return self._call_ranked(completions, prompts, metadata_list)

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
        _hl_corrects: List[float] = []
        _hl_wrongs: List[float] = []

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
            _hl_corrects.append(avg_correct)
            _hl_wrongs.append(avg_wrong)

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

        n = max(len(completions), 1)
        n_parse_fail = sum(
            1 for ci, (_, fc) in enumerate(completion_flow_ranges)
            if fc == 0 and not completion_valid_no_flow[ci]
        )
        self._push_health({
            "n_completions": len(completions),
            "parse_fail_frac": n_parse_fail / n,
            "no_flow_frac": len(no_flow_indices) / n,
            "mean_score": sum(scores) / n,
            "zero_score_frac": sum(1 for s in scores if s == 0.0) / n,
            "mean_correct": (sum(_hl_corrects) / len(_hl_corrects)) if _hl_corrects else 0.0,
            "mean_wrong": (sum(_hl_wrongs) / len(_hl_wrongs)) if _hl_wrongs else 0.0,
        })

        return scores

    # ------------------------------------------------------------------
    # Ranked (listwise) scoring
    # ------------------------------------------------------------------

    def _call_ranked(
        self,
        completions: List[str],
        prompts: List[str],
        metadata_list: List[Dict[str, Any]],
    ) -> List[float]:
        """Listwise R_ground: rank same-prompt completions against each other.

        Groups completions by prompt, retrieves ONE shared norm set per group
        (so all candidates are judged against identical evidence), and makes
        two listwise judge calls per group — one against the correct
        universe, one against a random wrong universe:

            R_i = clamp(blend(rank_i, grounding_i) - λ * wrong_grounding_i, 0, 1)

        The rank component breaks the ties that per-candidate absolute
        scoring produces; the wrong-universe side uses grounding scores only
        (ranking against the wrong universe is forced-choice noise — some
        candidate must rank first even when none is grounded there).

        No-flow declarations participate as ordinary candidates: the judge
        is instructed to rank them by whether the declaration is justified.
        Gold-label correctness is handled by the other reward components.
        Parse failures score 0.0 without judging.
        """
        n = len(completions)
        scores = [0.0] * n
        self.last_diagnostics = [[] for _ in range(n)]

        # ---- Group completions by prompt ----
        group_members: Dict[str, List[int]] = {}
        for i in range(n):
            meta = metadata_list[i] if i < len(metadata_list) else {}
            key = (prompts[i] if i < len(prompts) and prompts[i] else "") \
                or str(meta.get("prompt_id", i))
            group_members.setdefault(key, []).append(i)

        # ---- Parse all completions; collect texts to embed ----
        candidate_texts: Dict[int, str] = {}   # idx → block text for the judge
        flow_queries: Dict[int, List[str]] = {}  # idx → retrieval queries
        for i, completion in enumerate(completions):
            parsed = _parse_completion(completion)
            if parsed is None:
                continue  # parse failure: excluded, scores 0.0
            extractions = parsed.get("extraction", [])
            if not isinstance(extractions, list):
                extractions = []
            reasoning = parsed.get("reasoning", {})
            has_exchange = reasoning.get("has_information_exchange") \
                if isinstance(reasoning, dict) else None
            if extractions:
                candidate_texts[i] = json.dumps(
                    [e for e in extractions if isinstance(e, dict)],
                    ensure_ascii=False, indent=1,
                )
                flow_queries[i] = [
                    _flow_to_query(_flatten_flow(e))
                    for e in extractions if isinstance(e, dict)
                ]
            elif has_exchange is False:
                candidate_texts[i] = (
                    "This candidate declares the passage contains "
                    "NO information flows."
                )
                flow_queries[i] = []

        # ---- Batch-embed all flow queries + one chunk text per group ----
        embed_texts: List[str] = []
        query_slices: Dict[int, tuple] = {}  # idx → (start, count) in embed_texts
        group_chunk_pos: Dict[str, int] = {}  # group key → chunk-text position
        group_meta: Dict[str, Dict[str, Any]] = {}
        for key, members in group_members.items():
            lead_meta = metadata_list[members[0]] if members[0] < len(metadata_list) else {}
            chunk_text = lead_meta.get("chunk_text", "") or \
                (prompts[members[0]] if members[0] < len(prompts) else "")
            group_meta[key] = {
                "chunk_text": chunk_text,
                "source_id": lead_meta.get("source_id", ""),
                "wrong_source": _pick_wrong_source(
                    lead_meta.get("source_id", ""), self.all_source_ids
                ),
            }
            group_chunk_pos[key] = len(embed_texts)
            embed_texts.append(chunk_text or "information flow")
            for i in members:
                qs = flow_queries.get(i, [])
                query_slices[i] = (len(embed_texts), len(qs))
                embed_texts.extend(qs)

        embeddings = self.embedding_client.encode_batch(embed_texts)

        # ---- Build one judge item per (group, universe) ----
        import numpy as np

        do_contrastive = (
            self.contrastive_lambda > 0.0 and len(self.all_source_ids) > 1
        )
        correct_items: List[Dict[str, Any]] = []
        wrong_items: List[Dict[str, Any]] = []
        group_candidates: Dict[str, List[int]] = {}  # key → judged member idxs
        group_order: List[str] = []
        for key, members in group_members.items():
            judged = [i for i in members if i in candidate_texts]
            group_candidates[key] = judged
            if not judged:
                continue
            gm = group_meta[key]

            # Group query embedding: mean of the chunk-text embedding and all
            # members' flow-query embeddings, re-normalized. One retrieval per
            # group keeps the evidence identical across candidates.
            rows = [embeddings[group_chunk_pos[key]]]
            for i in judged:
                start, count = query_slices[i]
                rows.extend(embeddings[start:start + count])
            group_emb = np.mean(np.asarray(rows), axis=0)
            norm = np.linalg.norm(group_emb)
            if norm > 0:
                group_emb = group_emb / norm

            if not group_emb.any():
                # Zero-embedding fallback from a failed encode call — no
                # meaningful retrieval is possible. Route the group through
                # the judge-failed path (uniform neutral score, zero
                # advantage) instead of judging against garbage norms.
                gm["correct_norms"] = "[]"
                gm["correct_sims"] = []
                correct_items.append(None)
                wrong_items.append(None)
                group_order.append(key)
                continue

            correct_norms, correct_sims = self.norm_retriever.retrieve(
                group_emb, gm["source_id"], return_scores=True,
                top_k=self.rank_top_k,
            )
            gm["correct_norms"] = correct_norms
            gm["correct_sims"] = correct_sims

            block_lines = []
            for pos, i in enumerate(judged):
                block_lines.append(f"### Candidate {pos}\n{candidate_texts[i]}")
            candidates_block = "\n\n".join(block_lines)

            # `candidates` is the structured per-candidate list used by the
            # reranker backend (RerankerJudgeClient); the generative
            # JudgeClient ignores it and reads candidates_block instead.
            candidate_list = [candidate_texts[i] for i in judged]
            correct_items.append({
                "chunk_text": gm["chunk_text"],
                "norm_universe_json": correct_norms,
                "candidates_block": candidates_block,
                "candidates": candidate_list,
                "n_candidates": len(judged),
            })
            if do_contrastive and gm["wrong_source"]:
                wrong_norms, wrong_sims = self.norm_retriever.retrieve(
                    group_emb, gm["source_id"],
                    contrastive_source=gm["wrong_source"],
                    return_scores=True, top_k=self.rank_top_k,
                )
                gm["wrong_norms"] = wrong_norms
                gm["wrong_sims"] = wrong_sims
                wrong_items.append({
                    "chunk_text": gm["chunk_text"],
                    "norm_universe_json": wrong_norms,
                    "candidates_block": candidates_block,
                    "candidates": candidate_list,
                    "n_candidates": len(judged),
                })
            else:
                wrong_items.append(None)
            group_order.append(key)

        live_correct = [it for it in correct_items if it is not None]
        correct_results_live = self.judge_client.judge_ranking_batch(
            live_correct,
            system_prompt=self._ranking_system_prompt,
            prompt_template=self._ranking_prompt_template,
            json_schema=self._ranking_json_schema,
        ) if live_correct else []
        # Re-align correct results with group order (None where the group
        # embedding was degenerate — handled as judge-failed downstream)
        correct_results: List[Optional[List[Dict[str, Any]]]] = []
        cpos = 0
        for it in correct_items:
            if it is None:
                correct_results.append(None)
            else:
                correct_results.append(correct_results_live[cpos])
                cpos += 1
        live_wrong = [it for it in wrong_items if it is not None]
        wrong_results_live = self.judge_client.judge_ranking_batch(
            live_wrong,
            system_prompt=self._ranking_system_prompt,
            prompt_template=self._ranking_prompt_template,
            json_schema=self._ranking_json_schema,
        ) if live_wrong else []
        # Re-align wrong results with group order (None where skipped)
        wrong_results: List[Optional[List[Dict[str, Any]]]] = []
        wpos = 0
        for it in wrong_items:
            if it is None:
                wrong_results.append(None)
            else:
                wrong_results.append(wrong_results_live[wpos])
                wpos += 1

        # ---- Convert rankings to per-completion scores ----
        _hl_failed_groups = 0
        _hl_correct_sum = 0.0
        _hl_wrong_sum = 0.0
        _hl_app_sum = 0.0
        _hl_n_scored = 0
        for gpos, key in enumerate(group_order):
            judged = group_candidates[key]
            n_cand = len(judged)
            gm = group_meta[key]

            rankings = correct_results[gpos]
            judge_failed = rankings is None
            if judge_failed:
                _hl_failed_groups += 1
                # Judge failure: identical neutral scores → zero advantage
                # for this group rather than a spurious one.
                correct_scores = [0.5] * n_cand
            else:
                correct_scores = _rankings_to_scores(
                    rankings, n_cand, rank_weight=self.rank_weight,
                )

            # When the correct-side judge failed, leave wrong_grounding at 0:
            # subtracting a surviving wrong-universe score from the neutral
            # 0.5 would vary within the group, turning the failure into a
            # spurious gradient driven entirely by the wrong-universe judge.
            wrong_grounding = [0.0] * n_cand
            wr = wrong_results[gpos]
            if wr is not None and not judge_failed:
                for entry in wr:
                    idx = entry.get("candidate_index", -1)
                    if 0 <= idx < n_cand:
                        wrong_grounding[idx] = max(
                            0.0, min(1.0, float(entry.get("grounding_score", 0.0)))
                        )

            rank_by_pos = {}
            grounding_by_pos = {}
            if rankings is not None:
                for entry in rankings:
                    idx = entry.get("candidate_index", -1)
                    if 0 <= idx < n_cand:
                        rank_by_pos[idx] = entry.get("rank")
                        grounding_by_pos[idx] = entry.get("grounding_score")

            _hl_correct_sum += sum(correct_scores)
            _hl_wrong_sum += sum(wrong_grounding)
            _hl_n_scored += n_cand

            # Governing norm's deontic force for this group. Programmatic and
            # judge-independent, so it stays valid even when the ranking judge
            # failed (correct_scores=0.5 uniform) — recovering a real signal
            # rather than the deliberate zero-advantage fallback.
            _force = (governing_norm_force(gm.get("correct_norms", "[]"))
                      if self.app_weight > 0.0 else None)

            for pos, i in enumerate(judged):
                # ── Symmetric contrastive clamp (v8, 2026-06-22) ──
                # Apply the wrong-universe penalty to the GROUNDING component
                # only, leaving the rank component (within-group anti-tie
                # discrimination) contrast-free:
                #   base = w_r·rank + (1−w_r)·clamp(g_correct − λ·g_wrong, 0, 1)
                # The prior form subtracted λ·g_wrong from the full rank-BLENDED
                # correct score while g_wrong was FULL grounding — an asymmetry
                # (correct side diluted ×(1−w_r) by the rank blend, wrong side
                # not) that clamped ~1/3 of well-grounded extractions to 0 when
                # g_wrong≈g_correct and, worse, ate the rank signal so
                # tied-grounding groups collapsed toward zero advantage. The
                # contrast is now grounding-vs-grounding (symmetric); λ=1.0 and
                # the contrastive thesis are unchanged. correct_scores already
                # encodes the n_candidates==1 / judge-failed fallbacks, so
                # reconstruct the grounding term from g_correct and swap in its
                # contrasted value; fall back to the legacy form when the rank
                # blend does not apply (singleton / judge-failed / no grounding).
                g_correct = grounding_by_pos.get(pos)
                if judge_failed or g_correct is None or n_cand <= 1:
                    base = max(0.0, min(
                        1.0,
                        correct_scores[pos]
                        - self.contrastive_lambda * wrong_grounding[pos],
                    ))
                else:
                    g_correct = max(0.0, min(1.0, float(g_correct)))
                    contrasted = max(0.0, min(
                        1.0,
                        g_correct - self.contrastive_lambda * wrong_grounding[pos],
                    ))
                    # correct_scores[pos] = w_r·rank + (1−w_r)·g_correct; replace
                    # the grounding term with its contrasted value, rank intact.
                    base = max(0.0, min(
                        1.0,
                        correct_scores[pos]
                        - (1.0 - self.rank_weight) * g_correct
                        + (1.0 - self.rank_weight) * contrasted,
                    ))
                app_cons = None
                if self.app_weight > 0.0:
                    # candidate_texts[i] is the extraction JSON (or the no-flow
                    # sentinel, which carries no appropriateness labels → neutral).
                    app_cons = candidate_appropriateness_consistency(
                        candidate_texts.get(i, ""), _force)
                    if self.app_mode == "multiplicative":
                        # v9: gate grounding by appropriateness-direction. A wrong
                        # verdict floors the extraction reward at app_floor; a hedge
                        # ("ambiguous"/no label → 0.5) costs ~30%; correct keeps full.
                        raw = base * direction_multiplier(app_cons, self.app_floor)
                    else:
                        raw = (1.0 - self.app_weight) * base + self.app_weight * app_cons
                    _hl_app_sum += app_cons
                else:
                    raw = base
                scores[i] = max(0.0, min(1.0, raw))
                self.last_diagnostics[i] = [{
                    "type": "ranked",
                    "source_id": gm["source_id"],
                    "n_candidates": n_cand,
                    "rank": rank_by_pos.get(pos),
                    "grounding_score": grounding_by_pos.get(pos),
                    "correct_score": round(correct_scores[pos], 4),
                    "wrong_source": gm.get("wrong_source"),
                    "wrong_grounding": round(wrong_grounding[pos], 4),
                    "norm_force": _force,
                    "app_consistency": round(app_cons, 4) if app_cons is not None else None,
                    "correct_retrieval_sims": gm.get("correct_sims", []),
                    "correct_norm_snippets": _norm_snippet(gm.get("correct_norms", "[]")),
                    "judge_failed": judge_failed,
                    "r_ground": round(scores[i], 4),
                }]

        # Track consecutive all-zero batches to detect server failures
        self._total_calls += 1
        if all(s == 0.0 for s in scores):
            self._consecutive_zero_batches += 1
            if self._consecutive_zero_batches >= 5:
                print(
                    f"[OnlineRGround] WARNING: {self._consecutive_zero_batches} "
                    f"consecutive all-zero batches (ranked mode). Embedding or "
                    f"judge server may be down."
                )
        else:
            self._consecutive_zero_batches = 0

        n_total = max(n, 1)
        n_groups = max(len(group_order), 1)
        n_no_flow = sum(
            1 for i in candidate_texts if not flow_queries.get(i)
        )
        if _hl_failed_groups:
            # A failed group collapses to uniform 0.5 (zero advantage), so a
            # persistent failure silently turns R_ground off — make it loud.
            print(
                f"[OnlineRGround] WARNING: ranking judge failed for "
                f"{_hl_failed_groups}/{len(group_order)} groups this call; "
                f"those groups received uniform R_ground=0.5 (no signal)."
            )
        self._push_health({
            "n_completions": n,
            "n_groups": len(group_order),
            "parse_fail_frac": (n - len(candidate_texts)) / n_total,
            "no_flow_frac": n_no_flow / n_total,
            "judge_failed_group_frac": _hl_failed_groups / n_groups,
            "mean_score": sum(scores) / n_total,
            "zero_score_frac": sum(1 for s in scores if s == 0.0) / n_total,
            "mean_correct": (_hl_correct_sum / _hl_n_scored) if _hl_n_scored else 0.0,
            "mean_wrong": (_hl_wrong_sum / _hl_n_scored) if _hl_n_scored else 0.0,
            "mean_app_consistency": (
                (_hl_app_sum / _hl_n_scored) if (_hl_n_scored and self.app_weight > 0.0) else 0.0
            ),
        })

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
