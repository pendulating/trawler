"""Composite reward function for GRPO training.

Implements 6 reward components from the CoLM 2026 manuscript:
  R = sum(w_i * R_i) where:
    R_uncert   - Task clarity (schema validity, construct discrimination, confidence)
    R_complete - Structural completeness (proportion of non-null CI tuple fields)
    R_consist  - Internal consistency (boolean invariant checks)
    R_context  - Context identification (semantic similarity)
    R_cohere   - Reasoning-to-extraction coherence
    R_ground   - Normative grounding (pre-computed LLM judge scores)

All components evaluate CI information flow extraction completions.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pydantic import ValidationError

from dagspaces.common.vllm_inference import _strip_think_blocks
from ..schemas import CICompletionResult


def _parse_completion(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from completion text and normalise to the canonical schema.

    The canonical (nested) schema expected by reward components:
        {"reasoning": CIReasoningList, "extraction": [ContextualIntegrityFlow]}

    The SFT data prep produces a *flat* schema:
        {"reasoning": str, "has_information_exchange": bool, "flows": [...]}

    This function transparently converts the flat format to the nested one so
    that all reward components can assume a single structure.
    """
    if not text:
        return None
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                obj = json.loads(match.group())
            except json.JSONDecodeError:
                pass
    if obj is None:
        return None

    # --- Normalise flat SFT format → nested canonical format ---
    # Detect flat format: "reasoning" is a string (not a dict) and "flows" key exists.
    reasoning_val = obj.get("reasoning")
    if isinstance(reasoning_val, str) and "flows" in obj:
        flat_flows = obj.get("flows", [])
        has_exchange = obj.get("has_information_exchange", bool(flat_flows))

        # Build nested reasoning object
        reasoning_entries = []
        for flow in flat_flows:
            if not isinstance(flow, dict):
                continue
            reasoning_entries.append({
                "original_text_snippet": "",
                "reasoning": reasoning_val,
                "context_identified": flow.get("context", ""),
                "flow_direction": "",
                "potential_appropriateness": flow.get("appropriateness", "ambiguous"),
                "is_new_flow": flow.get("is_new_flow", False),
            })

        nested_reasoning = {
            "flows": reasoning_entries,
            "has_information_exchange": has_exchange,
        }

        # Build nested extraction objects
        extraction = []
        for flow in flat_flows:
            if not isinstance(flow, dict):
                continue
            extraction.append({
                "flow": {
                    "subject": flow.get("subject"),
                    "sender": flow.get("sender"),
                    "recipient": flow.get("recipient"),
                    "information_type": flow.get("information_type"),
                    "transmission_principle": flow.get("transmission_principle"),
                },
                "context": flow.get("context", ""),
                "appropriateness": flow.get("appropriateness", "ambiguous"),
                "norms_invoked": flow.get("norms_invoked", []),
                "norm_source": flow.get("norm_source", "implicit"),
                "is_new_flow": flow.get("is_new_flow", False),
                "confidence_qual": flow.get("confidence", "uncertain"),
                "confidence_quant": flow.get("confidence_quant", 5),
            })

        obj = {"reasoning": nested_reasoning, "extraction": extraction}

    return obj


# ---------------------------------------------------------------------------
# Individual reward components
# ---------------------------------------------------------------------------

def r_uncert(completion: str) -> float:
    """R_uncert: Task clarity reward.

    Three facets:
    1. Schema validity (gating): does output parse as CICompletionResult? (0.0 or 0.6)
    2. Flow discrimination: has_information_exchange flag present? (+0.2)
    3. Confidence calibration: self-reported confidence on extractions (+0.2)
    """
    parsed = _parse_completion(completion)
    if parsed is None:
        return 0.0

    score = 0.0

    # Facet 1: Schema validity (gating)
    try:
        CICompletionResult.model_validate(parsed)
        score += 0.6
    except (ValidationError, Exception):
        return 0.0

    # Facet 2: Flow discrimination
    reasoning = parsed.get("reasoning", {})
    if isinstance(reasoning, dict) and "has_information_exchange" in reasoning:
        score += 0.2

    # Facet 3: Confidence calibration
    extractions = parsed.get("extraction", [])
    if isinstance(extractions, list) and len(extractions) > 0:
        has_confidence = any(
            "confidence_quant" in e for e in extractions if isinstance(e, dict)
        )
        if has_confidence:
            score += 0.2

    return score


def r_complete(completion: str) -> float:
    """R_complete: Structural completeness.

    Proportion of non-null, substantive fields in extracted CI flow tuples.
    """
    parsed = _parse_completion(completion)
    if parsed is None:
        return 0.0

    extractions = parsed.get("extraction", [])
    if not isinstance(extractions, list) or len(extractions) == 0:
        return 0.0

    # CI 5-tuple required fields + metadata
    tuple_fields = ["sender", "recipient", "information_type",
                    "transmission_principle", "subject"]
    meta_fields = ["context", "appropriateness", "norm_source",
                   "confidence_qual", "confidence_quant"]
    all_fields = tuple_fields + meta_fields

    total_score = 0.0
    for extraction in extractions:
        if not isinstance(extraction, dict):
            continue
        # Flatten nested flow tuple
        flow_tuple = extraction.get("flow", {})
        if isinstance(flow_tuple, dict):
            flat = {**flow_tuple, **extraction}
        else:
            flat = extraction

        filled = sum(
            1 for f in all_fields
            if flat.get(f) is not None and str(flat.get(f, "")).strip() != ""
        )
        total_score += filled / len(all_fields)

    return total_score / len(extractions)


def r_consist(completion: str) -> float:
    """R_consist: Internal consistency.

    Checks boolean invariants between reasoning and extraction:
    - has_information_exchange=False → empty flows and empty extraction
    - has_information_exchange=True → non-empty flows
    - is_new_flow=True → appropriateness should be inappropriate or ambiguous
    """
    parsed = _parse_completion(completion)
    if parsed is None:
        return 0.0

    reasoning = parsed.get("reasoning", {})
    extractions = parsed.get("extraction", [])

    if not isinstance(reasoning, dict):
        return 0.0

    checks_passed = 0
    total_checks = 0

    # Check: has_information_exchange vs flows/extraction
    has_exchange = reasoning.get("has_information_exchange")
    flows = reasoning.get("flows", [])
    if has_exchange is not None:
        total_checks += 1
        if has_exchange is False and len(flows) == 0 and len(extractions) == 0:
            checks_passed += 1
        elif has_exchange is True and (len(flows) > 0 or len(extractions) > 0):
            checks_passed += 1

    # Check: is_new_flow → appropriateness constraint
    for ext in extractions:
        if not isinstance(ext, dict):
            continue
        if ext.get("is_new_flow") is True:
            total_checks += 1
            appropriateness = ext.get("appropriateness", "")
            if appropriateness in ("inappropriate", "ambiguous"):
                checks_passed += 1

    return checks_passed / max(total_checks, 1)


def r_context(
    completion: str,
    source_context: str,
    embedding_model: Any = None,
) -> float:
    """R_context: Context identification accuracy.

    Semantic similarity between model's stated societal context and
    source text metadata.
    """
    parsed = _parse_completion(completion)
    if parsed is None:
        return 0.0

    extractions = parsed.get("extraction", [])
    if not isinstance(extractions, list) or len(extractions) == 0:
        return 0.0

    model_contexts = []
    for ext in extractions:
        if isinstance(ext, dict):
            ctx = ext.get("context", "")
            if ctx:
                model_contexts.append(str(ctx))

    if not model_contexts or not source_context:
        return 0.0

    if embedding_model is not None:
        try:
            import numpy as np
            source_emb = embedding_model.encode([source_context], normalize_embeddings=True)
            context_embs = embedding_model.encode(model_contexts, normalize_embeddings=True)
            similarities = np.dot(context_embs, source_emb.T).flatten()
            return float(similarities.mean())
        except Exception:
            pass

    # Fallback: token overlap
    source_tokens = set(source_context.lower().split())
    total_overlap = 0.0
    for ctx in model_contexts:
        ctx_tokens = set(ctx.lower().split())
        if source_tokens and ctx_tokens:
            overlap = len(source_tokens & ctx_tokens) / len(source_tokens | ctx_tokens)
            total_overlap += overlap
    return total_overlap / len(model_contexts)


def r_cohere(completion: str) -> float:
    """R_cohere: Reasoning-to-extraction coherence.

    Checks that extracted flow tuples reference entities/concepts present
    in the reasoning trace.
    """
    parsed = _parse_completion(completion)
    if parsed is None:
        return 0.0

    reasoning = parsed.get("reasoning", {})
    extractions = parsed.get("extraction", [])

    # Collect reasoning text from all flow entries
    reasoning_text = ""
    if isinstance(reasoning, dict):
        entries = reasoning.get("flows", [])
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    reasoning_text += " " + str(entry.get("reasoning", ""))
                    reasoning_text += " " + str(entry.get("original_text_snippet", ""))
    elif isinstance(reasoning, str):
        reasoning_text = reasoning

    if not reasoning_text.strip() or not extractions:
        return 0.0

    reasoning_lower = reasoning_text.lower()
    coherence_scores = []

    for ext in extractions:
        if not isinstance(ext, dict):
            continue
        # Flatten nested flow tuple
        flow_tuple = ext.get("flow", {})
        if isinstance(flow_tuple, dict):
            flat = {**flow_tuple, **ext}
        else:
            flat = ext

        check_fields = ["sender", "recipient", "information_type", "subject"]
        matches = 0
        checked = 0
        for field in check_fields:
            val = flat.get(field)
            if val and isinstance(val, str) and len(val) > 2:
                checked += 1
                words = [w for w in val.lower().split() if len(w) > 3]
                if any(w in reasoning_lower for w in words):
                    matches += 1
        if checked > 0:
            coherence_scores.append(matches / checked)

    return sum(coherence_scores) / max(len(coherence_scores), 1)


def r_ground_cached(
    completion: str,
    prompt_id: str,
    reward_cache: pd.DataFrame,
    is_contrastive: bool = False,
) -> float:
    """R_ground: Normative grounding (cached lookup, legacy).

    Looks up pre-computed judge score from the reward cache.
    Used when online_rground is not configured.
    """
    if reward_cache is None or len(reward_cache) == 0:
        return 0.0

    mask = reward_cache["prompt_id"] == prompt_id
    if "is_contrastive" in reward_cache.columns:
        mask = mask & (reward_cache["is_contrastive"] == is_contrastive)

    matches = reward_cache[mask]
    if len(matches) == 0:
        return 0.0

    return float(matches.iloc[0].get("judge_score", 0.0))


# ---------------------------------------------------------------------------
# Composite reward function
# ---------------------------------------------------------------------------

class CompositeRewardFunction:
    """Composite reward R = sum(w_i * R_i) for GRPO training.

    All components evaluate CI information flow extraction completions.
    """

    __name__ = "composite_ci_reward"

    def __init__(
        self,
        weights: Sequence[float],
        norm_universes: Optional[Dict[str, list]] = None,
        reward_cache: Optional[pd.DataFrame] = None,
        context_embedding_model: Any = None,
        source_contexts: Optional[Dict[str, str]] = None,
        prompt_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        trace_log_path: Optional[str] = None,
        trace_every_n_calls: int = 50,
        online_rground: Optional[Any] = None,
    ):
        if len(weights) != 6:
            raise ValueError(f"Expected 6 reward weights, got {len(weights)}")
        self.weights = list(weights)
        self.norm_universes = norm_universes or {}
        self.reward_cache = reward_cache
        self.context_embedding_model = context_embedding_model
        self.source_contexts = source_contexts or {}
        # Maps prompt text -> {"source_id", "prompt_id", "is_contrastive"}
        # Used because TRL doesn't forward extra dataset columns to reward fns.
        self.prompt_metadata = prompt_metadata or {}
        # Online R_ground evaluator (replaces cached lookup when set)
        self.online_rground = online_rground
        # Periodic detailed trace logging
        self._call_count = 0
        self._trace_every = trace_every_n_calls
        self._trace_path = trace_log_path
        self._component_names = [
            "r_uncert", "r_complete", "r_consist",
            "r_context", "r_cohere", "r_ground",
        ]
        # Set by grpo_training.py for trace logging
        self.enable_thinking_grpo = None

    @staticmethod
    def _extract_text(completion) -> str:
        """Extract plain text from a completion, stripping any ``<think>`` blocks.

        Think blocks are stripped so reward components (especially
        ``_parse_completion``'s greedy JSON regex) don't choke on braces
        inside reasoning traces produced when thinking is enabled during GRPO.
        """
        if isinstance(completion, str):
            text = completion
        elif isinstance(completion, list):
            # Conversational: [{"role": "assistant", "content": "..."}]
            text = ""
            for msg in completion:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    text = msg.get("content", "")
                    break
            if not text:
                text = " ".join(
                    msg.get("content", "") for msg in completion if isinstance(msg, dict)
                )
        else:
            text = str(completion)
        return _strip_think_blocks(text)

    def _should_trace(self) -> bool:
        """Whether to log detailed traces on this call."""
        if not self._trace_path:
            return False
        return (self._call_count == 0) or (self._call_count % self._trace_every == 0)

    def _log_trace(self, entries: List[Dict[str, Any]]) -> None:
        """Append trace entries to the JSONL log file."""
        if not self._trace_path:
            return
        try:
            import json as _json
            os.makedirs(os.path.dirname(self._trace_path), exist_ok=True)
            with open(self._trace_path, "a", encoding="utf-8") as f:
                for entry in entries:
                    f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def __call__(
        self,
        *,
        prompts=None,
        completions,
        **kwargs,
    ) -> List[float]:
        """Score completions with the composite reward.

        When ``online_rground`` is set, R_ground is evaluated by batching
        all completions through the embedding + judge servers.  Otherwise
        falls back to cached lookup.
        """
        do_trace = self._should_trace()
        self._call_count += 1

        # ----- Phase 1: Extract text and metadata for all completions -----
        extracted_texts = []
        prompt_texts = []
        meta_list = []
        for i, completion in enumerate(completions):
            text = self._extract_text(completion)
            extracted_texts.append(text)

            prompt = prompts[i] if prompts else ""
            if isinstance(prompt, list):
                prompt = " ".join(
                    m.get("content", "") for m in prompt
                    if isinstance(m, dict) and m.get("role") == "user"
                )
            prompt_texts.append(prompt)
            meta_list.append(self.prompt_metadata.get(prompt, {}))

        # ----- Phase 2: Compute R_uncert through R_cohere per completion -----
        partial_components = []  # list of [r0, r1, r2, r3, r4] per completion
        for i, completion in enumerate(extracted_texts):
            meta = meta_list[i]
            source_id = meta.get("source_id", "")
            source_context = self.source_contexts.get(source_id, "")

            partial_components.append([
                r_uncert(completion),
                r_complete(completion),
                r_consist(completion),
                r_context(completion, source_context, self.context_embedding_model),
                r_cohere(completion),
            ])

        # ----- Phase 3: Compute R_ground (online or cached) -----
        use_rground = self.weights[5] > 0.0
        if use_rground and self.online_rground is not None:
            # Batch all completions through embedding + judge servers
            rground_scores = self.online_rground(
                completions=extracted_texts,
                prompts=prompt_texts,
                metadata_list=meta_list,
            )
        elif use_rground:
            # Cached fallback
            rground_scores = []
            for i in range(len(extracted_texts)):
                meta = meta_list[i]
                rground_scores.append(
                    r_ground_cached(
                        extracted_texts[i],
                        meta.get("prompt_id", ""),
                        self.reward_cache,
                        meta.get("is_contrastive", False),
                    )
                )
        else:
            rground_scores = [0.0] * len(extracted_texts)

        # ----- Phase 4: Combine and score -----
        scores = []
        trace_entries = []
        for i in range(len(extracted_texts)):
            components = partial_components[i] + [rground_scores[i]]
            r = sum(w * c for w, c in zip(self.weights, components))
            scores.append(r)

            if do_trace and i < 2:
                meta = meta_list[i]
                trace_entries.append({
                    "call": self._call_count - 1,
                    "idx": i,
                    "source_id": meta.get("source_id", ""),
                    "prompt_id": meta.get("prompt_id", ""),
                    "completion_len": len(extracted_texts[i]),
                    "completion": extracted_texts[i],
                    "components": {
                        name: round(val, 4)
                        for name, val in zip(self._component_names, components)
                    },
                    "weighted": {
                        name: round(w * val, 4)
                        for name, w, val in zip(self._component_names, self.weights, components)
                    },
                    "composite": round(r, 4),
                    "enable_thinking_grpo": self.enable_thinking_grpo,
                    "rground_mode": "online" if self.online_rground is not None else "cached",
                })

        if trace_entries:
            self._log_trace(trace_entries)

        return scores
