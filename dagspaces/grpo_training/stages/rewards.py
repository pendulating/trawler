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

# Reward for valid no-flow declarations (has_information_exchange=false,
# empty flows).  Asymmetric based on gold labels from SFT data:
#   - Gold says no flows → correct no-flow, moderate reward
#   - Gold says has flows → false no-flow (reward hacking), punitive
#   - No gold label      → cautious default
NO_FLOW_REWARD_CORRECT = 0.6   # Genuinely no flows: decent but < good extraction
NO_FLOW_REWARD_UNKNOWN = 0.4   # No gold label: hedge
NO_FLOW_REWARD_WRONG = 0.1     # Gold says flows exist: punish the lazy path


def no_flow_reward(gold_has_exchange: Optional[bool] = None) -> float:
    """Return the appropriate no-flow reward based on the gold label."""
    if gold_has_exchange is True:
        return NO_FLOW_REWARD_WRONG
    elif gold_has_exchange is False:
        return NO_FLOW_REWARD_CORRECT
    else:
        return NO_FLOW_REWARD_UNKNOWN


def _to_str(val) -> str:
    """Coerce a value to string, joining lists with ', '."""
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val)


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
    if obj is None or not isinstance(obj, dict):
        return None

    # --- Normalise flat SFT format → nested canonical format ---
    # Detect flat format: "reasoning" is a string (not a dict) and "flows" key exists.
    reasoning_val = obj.get("reasoning")
    if isinstance(reasoning_val, str) and "flows" in obj:
        flat_flows = obj.get("flows", [])
        if not isinstance(flat_flows, list):
            return None
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

def r_uncert(
    completion: str,
    gold_has_exchange: Optional[bool] = None,
    is_no_flow: bool = False,
) -> float:
    """R_uncert: Task clarity reward.

    Three facets:
    1. Schema validity (gating): valid JSON with reasoning + flows/extraction? (0.0 or 0.6)
    2. Flow discrimination: has_information_exchange flag present? (+0.2)
    3. Confidence calibration: self-reported confidence on extractions (+0.2)

    For no-flow completions, awards schema validity + discrimination (0.8)
    scaled by correctness: 1.0 if gold agrees, 0.25 if gold says flows exist.
    """
    parsed = _parse_completion(completion)
    if parsed is None:
        return 0.0

    score = 0.0

    # Facet 1: Schema validity (gating)
    # Accept either nested (extraction) or flat (flows) format.
    reasoning = parsed.get("reasoning")
    flows = parsed.get("extraction") or parsed.get("flows", [])
    if reasoning is None or not isinstance(flows, list):
        return 0.0

    # No-flow completion: valid schema + discrimination, scaled by correctness.
    has_exchange = None
    if isinstance(reasoning, dict):
        has_exchange = reasoning.get("has_information_exchange")
    elif "has_information_exchange" in (parsed if isinstance(parsed, dict) else {}):
        has_exchange = parsed.get("has_information_exchange")
    if is_no_flow or (has_exchange is False and len(flows) == 0):
        # Schema validity (0.6) + discrimination present (0.2) = 0.8 base
        base = 0.8
        if gold_has_exchange is True:
            return base * 0.25  # 0.2 — valid schema but wrong answer
        elif gold_has_exchange is False:
            return base  # 0.8 — correct no-flow
        else:
            return base * 0.625  # 0.5 — unknown

    # Require at least one flow with a sender or recipient
    ci_fields = {"sender", "recipient", "information_type", "transmission_principle"}
    valid_flows = 0
    for f in flows:
        if not isinstance(f, dict):
            continue
        # Check nested (flow.sender) or flat (sender) format
        flat = f.get("flow", f) if isinstance(f.get("flow"), dict) else f
        if any(flat.get(k) for k in ci_fields):
            valid_flows += 1
    if valid_flows > 0:
        score += 0.6
    else:
        return 0.0

    # Facet 2: Flow discrimination
    if isinstance(reasoning, dict):
        if "has_information_exchange" in reasoning:
            score += 0.2
    else:
        # Flat format: has_information_exchange is a top-level key
        if "has_information_exchange" in parsed:
            score += 0.2

    # Facet 3: Confidence calibration — scale by actual value (1-10).
    # Low self-reported confidence is penalized; high confidence rewarded.
    # The flat SFT format uses "confidence" (numeric), but _parse_completion
    # normalizes it to "confidence_qual" (stored as string) and
    # "confidence_quant" (default 5).  Check all three field names.
    conf_values = []
    for e in flows:
        if not isinstance(e, dict):
            continue
        # Try raw "confidence" first (flat format pre-normalization),
        # then "confidence_qual" (where _parse_completion stores the
        # original numeric value), then "confidence_quant" (default 5).
        c = e.get("confidence")
        if c is None:
            c = e.get("confidence_qual")
        if c is None:
            c = e.get("confidence_quant")
        if c is not None:
            try:
                conf_values.append(float(c))
            except (TypeError, ValueError):
                pass
    if conf_values:
        avg_conf = sum(conf_values) / len(conf_values)
        score += 0.2 * max(0.0, min(avg_conf, 10.0)) / 10.0

    return score


def r_complete(
    completion: str,
    gold_has_exchange: Optional[bool] = None,
    is_no_flow: bool = False,
) -> float:
    """R_complete: Structural completeness.

    Proportion of non-null, substantive fields in extracted CI flow tuples.
    For no-flow completions: 0.9 if gold agrees (nothing to fill), 0.0 if
    gold says flows exist (missed all flows), 0.4 if unknown.
    """
    parsed = _parse_completion(completion)
    if parsed is None:
        return 0.0

    extractions = parsed.get("extraction", [])
    if not isinstance(extractions, list) or len(extractions) == 0:
        # Check if this is a valid no-flow completion
        reasoning = parsed.get("reasoning", {})
        has_exchange = reasoning.get("has_information_exchange") if isinstance(reasoning, dict) else None
        if is_no_flow or has_exchange is False:
            if gold_has_exchange is True:
                return 0.0   # Missed all flows — completely incomplete
            elif gold_has_exchange is False:
                return 0.9   # Nothing to fill — near-perfect
            else:
                return 0.4   # Unknown
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


def r_consist(
    completion: str,
    gold_has_exchange: Optional[bool] = None,
    is_no_flow: bool = False,
) -> float:
    """R_consist: Internal consistency.

    Checks boolean invariants between reasoning and extraction:
    - has_information_exchange=False → empty flows and empty extraction
    - has_information_exchange=True → non-empty flows
    - is_new_flow=True → appropriateness should be inappropriate or ambiguous
    - sender != recipient (a CI flow requires distinct agents)

    For no-flow: 1.0 (internally consistent regardless of gold label —
    consistency measures self-agreement, not correctness).
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
            # No-flow is internally consistent — return 1.0.
            # Correctness vs gold label is handled by other components.
            return 1.0
        elif has_exchange is True and (len(flows) > 0 or len(extractions) > 0):
            checks_passed += 1

    for ext in extractions:
        if not isinstance(ext, dict):
            continue

        # Check: is_new_flow → appropriateness constraint
        if ext.get("is_new_flow") is True:
            total_checks += 1
            appropriateness = ext.get("appropriateness", "")
            if appropriateness in ("inappropriate", "ambiguous"):
                checks_passed += 1

        # Check: sender != recipient (CI flows require distinct agents)
        flow_tuple = ext.get("flow", {})
        if isinstance(flow_tuple, dict):
            sender = _to_str(flow_tuple.get("sender")).strip().lower()
            recipient = _to_str(flow_tuple.get("recipient")).strip().lower()
        else:
            sender = _to_str(ext.get("sender")).strip().lower()
            recipient = _to_str(ext.get("recipient")).strip().lower()
        if sender and recipient:
            total_checks += 1
            if sender != recipient:
                checks_passed += 1

    return checks_passed / max(total_checks, 1)


def r_context(
    completion: str,
    source_context_embeddings: Any,
    source_context_strings: List[str],
    embedding_model: Any = None,
    gold_has_exchange: Optional[bool] = None,
    is_no_flow: bool = False,
) -> float:
    """R_context: Context identification accuracy.

    For each model-stated context, find the most similar norm-level
    context from the source's normative universe (max cosine similarity).
    Return the mean of per-flow best-match similarities.

    For no-flow completions: 0.0 (no contexts extracted = nothing to score).

    Args:
        completion: Model completion text.
        source_context_embeddings: Pre-computed (N, D) embedding matrix
            for this source's norm contexts, or None.
        source_context_strings: List of norm-level context strings for
            token-overlap fallback.
        embedding_model: SentenceTransformer for encoding model contexts.
        gold_has_exchange: Gold label for this prompt (used by other
            components; not used here — no contexts means no score).
        is_no_flow: Whether this completion declared no information flows.
    """
    parsed = _parse_completion(completion)
    if parsed is None:
        return 0.0

    extractions = parsed.get("extraction", [])
    if not isinstance(extractions, list) or len(extractions) == 0:
        # No-flow or empty extraction: nothing to score.
        return 0.0

    model_contexts = []
    for ext in extractions:
        if isinstance(ext, dict):
            ctx = ext.get("context", "")
            if ctx:
                model_contexts.append(str(ctx))

    if not model_contexts or (source_context_embeddings is None and not source_context_strings):
        return 0.0

    # Embedding path: per-flow max similarity against source norm contexts
    if embedding_model is not None and source_context_embeddings is not None:
        try:
            import numpy as np
            # (M, D) — one row per model-stated context
            model_embs = embedding_model.encode(
                model_contexts, normalize_embeddings=True,
            )
            # (M, N) — cosine similarities against all source norm contexts
            sim_matrix = np.dot(model_embs, source_context_embeddings.T)
            # Per-flow best match, then average across flows
            best_per_flow = sim_matrix.max(axis=1)
            return float(best_per_flow.mean())
        except Exception:
            pass

    # Fallback: per-flow max token overlap against individual norm contexts
    if not source_context_strings:
        return 0.0
    ref_token_sets = [set(s.lower().split()) for s in source_context_strings]
    total_best = 0.0
    for ctx in model_contexts:
        ctx_tokens = set(ctx.lower().split())
        best = 0.0
        for ref_tokens in ref_token_sets:
            if ref_tokens and ctx_tokens:
                overlap = len(ref_tokens & ctx_tokens) / len(ref_tokens | ctx_tokens)
                if overlap > best:
                    best = overlap
        total_best += best
    return total_best / len(model_contexts)


def r_cohere(
    completion: str,
    gold_has_exchange: Optional[bool] = None,
    is_no_flow: bool = False,
) -> float:
    """R_cohere: Reasoning-to-extraction coherence.

    Checks that extracted flow tuples reference entities/concepts present
    in the reasoning trace.

    For no-flow completions: scores reasoning quality (0.0–0.5) based on
    whether the reasoning text discusses information exchange.
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
        # Also capture top-level reasoning string if present
        top_reasoning = reasoning.get("reasoning")
        if isinstance(top_reasoning, str):
            reasoning_text += " " + top_reasoning
    elif isinstance(reasoning, str):
        reasoning_text = reasoning

    if not extractions:
        if is_no_flow or (isinstance(reasoning, dict) and
                          reasoning.get("has_information_exchange") is False):
            # Score reasoning quality for no-flow declarations.
            # The flat→nested normalizer drops the reasoning string for
            # no-flow completions (no flows to attach it to), so also
            # try extracting it directly from the raw completion JSON.
            if not reasoning_text.strip():
                try:
                    raw_obj = json.loads(completion)
                    if isinstance(raw_obj, dict) and isinstance(raw_obj.get("reasoning"), str):
                        reasoning_text = raw_obj["reasoning"]
                except (json.JSONDecodeError, TypeError):
                    pass
            text = reasoning_text.strip().lower()
            if len(text) < 20:
                return 0.0
            # Reward reasoning that explicitly discusses information exchange
            _nf_keywords = [
                "no information", "no exchange", "no flow", "does not contain",
                "does not describe", "no transfer", "no sharing", "not share",
                "no personal", "no private", "no sensitive", "privacy",
                "information flow", "contextual integrity",
            ]
            keyword_hits = sum(1 for kw in _nf_keywords if kw in text)
            # 0.0 to 0.5 based on reasoning engagement
            return min(0.5, 0.15 * keyword_hits)
        return 0.0
    if not reasoning_text.strip():
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
# Judgment vignette reward components
# ---------------------------------------------------------------------------

def _parse_judgment_completion(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from a norm judgment vignette completion.

    Simpler than _parse_completion — no flat→nested normalization.
    Expects: {"judgment": "yes"/"no", "reasoning": "...", "norms_considered": [...]}
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
    if obj is None or not isinstance(obj, dict):
        return None
    return obj


def r_judgment(completion: str, gold_judgment: str) -> float:
    """Binary reward: does the model's yes/no match the gold answer?"""
    parsed = _parse_judgment_completion(completion)
    if parsed is None:
        return 0.0
    model_judgment = str(parsed.get("judgment", "")).lower().strip()
    return 1.0 if model_judgment == gold_judgment else 0.0


def r_judgment_reasoning(completion: str) -> float:
    """Keyword-based reasoning quality for judgment vignettes.

    Rewards reasoning that engages with privacy/norm concepts rather
    than producing minimal or irrelevant explanations.
    """
    parsed = _parse_judgment_completion(completion)
    if parsed is None:
        return 0.0
    reasoning = str(parsed.get("reasoning", "")).lower()
    if len(reasoning) < 20:
        return 0.0
    _keywords = [
        "norm", "appropriate", "inappropriate", "privacy", "information",
        "sharing", "context", "expectation", "principle", "obligation",
        "prohibited", "consent", "confidential", "disclose", "trust",
        "duty", "permission", "sensitive", "transmission", "social",
    ]
    hits = sum(1 for kw in _keywords if kw in reasoning)
    return min(1.0, hits / 5.0)


def r_norm_cite(completion: str, source_norm_articulation: str) -> float:
    """Token overlap between norms_considered and the source norm.

    Rewards the model for citing norms that semantically relate to the
    actual norm governing the scenario.
    """
    parsed = _parse_judgment_completion(completion)
    if parsed is None:
        return 0.0
    norms = parsed.get("norms_considered", [])
    if not isinstance(norms, list) or not norms or not source_norm_articulation:
        return 0.0
    source_tokens = set(source_norm_articulation.lower().split())
    if not source_tokens:
        return 0.0
    best_overlap = 0.0
    for n in norms:
        n_tokens = set(str(n).lower().split())
        if n_tokens:
            overlap = len(n_tokens & source_tokens) / len(n_tokens | source_tokens)
            best_overlap = max(best_overlap, overlap)
    return best_overlap


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
        source_contexts: Optional[Dict[str, List[str]]] = None,
        prompt_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        trace_log_path: Optional[str] = None,
        trace_every_n_calls: int = 50,
        online_rground: Optional[Any] = None,
        no_flow_scoring: str = "independent",
        judgment_weights: Optional[Sequence[float]] = None,
    ):
        if len(weights) != 6:
            raise ValueError(f"Expected 6 reward weights, got {len(weights)}")
        self.weights = list(weights)
        self.no_flow_scoring = no_flow_scoring
        self.judgment_weights = list(judgment_weights) if judgment_weights else [0.5, 0.25, 0.25]
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

        # Pre-compute source context embeddings so r_context doesn't
        # re-encode the same reference set on every call.
        self._source_context_embeddings: Dict[str, Any] = {}
        if context_embedding_model is not None and self.source_contexts:
            import numpy as np
            for sid, ctx_list in self.source_contexts.items():
                if ctx_list:
                    try:
                        self._source_context_embeddings[sid] = (
                            context_embedding_model.encode(
                                ctx_list, normalize_embeddings=True,
                            )
                        )
                    except Exception:
                        pass
            n_cached = sum(
                e.shape[0] for e in self._source_context_embeddings.values()
            )
            print(f"[rewards] Pre-computed context embeddings: "
                  f"{len(self._source_context_embeddings)} sources, "
                  f"{n_cached} total contexts")

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

        # ----- Phase 2: Detect task type, compute components -----
        # CI extraction completions get 5 per-component scores.
        # Judgment vignettes get a pre-computed score (sentinel None in
        # partial_components) and bypass Phases 3-4.
        partial_components = []  # list of [r0..r4] or None (judgment sentinel)
        is_no_flow = []          # True if CI completion declares no exchange
        judgment_scores: Dict[int, float] = {}  # idx → pre-computed reward

        for i, completion in enumerate(extracted_texts):
            meta = meta_list[i]
            task_type = meta.get("task_type", "ci_extraction")

            # --- Judgment vignette: separate reward path ---
            if task_type == "norm_judgment":
                gold_j = meta.get("gold_judgment", "")
                norm_art = meta.get("source_norm_articulation", "")
                jw = self.judgment_weights
                j_acc = r_judgment(completion, gold_j)
                j_reas = r_judgment_reasoning(completion)
                j_cite = r_norm_cite(completion, norm_art)
                judgment_scores[i] = jw[0] * j_acc + jw[1] * j_reas + jw[2] * j_cite
                partial_components.append(None)  # sentinel
                is_no_flow.append(False)
                continue

            # --- CI extraction: existing component scoring ---
            source_id = meta.get("source_id", "")
            gold_has_exchange = meta.get("gold_has_exchange")

            parsed = _parse_completion(completion)
            _is_nf = False
            if parsed is not None:
                _reasoning = parsed.get("reasoning", {})
                _has_ex = None
                if isinstance(_reasoning, dict):
                    _has_ex = _reasoning.get("has_information_exchange")
                elif isinstance(parsed, dict):
                    _has_ex = parsed.get("has_information_exchange")
                _flows = parsed.get("extraction") or parsed.get("flows", [])
                if _has_ex is False and (not _flows or len(_flows) == 0):
                    _is_nf = True
            is_no_flow.append(_is_nf)

            if _is_nf and self.no_flow_scoring == "flat":
                nf_r = no_flow_reward(gold_has_exchange)
                partial_components.append([nf_r, nf_r, nf_r, nf_r, nf_r])
            else:
                src_ctx_embs = self._source_context_embeddings.get(source_id)
                src_ctx_strs = self.source_contexts.get(source_id, [])
                partial_components.append([
                    r_uncert(completion, gold_has_exchange=gold_has_exchange, is_no_flow=_is_nf),
                    r_complete(completion, gold_has_exchange=gold_has_exchange, is_no_flow=_is_nf),
                    r_consist(completion, gold_has_exchange=gold_has_exchange, is_no_flow=_is_nf),
                    r_context(completion, src_ctx_embs, src_ctx_strs, self.context_embedding_model,
                              gold_has_exchange=gold_has_exchange, is_no_flow=_is_nf),
                    r_cohere(completion, gold_has_exchange=gold_has_exchange, is_no_flow=_is_nf),
                ])

        # ----- Phase 3: Compute R_ground (online or cached) -----
        # Filter to CI extraction completions only — judgment vignettes
        # bypass R_ground entirely (their scores are in judgment_scores).
        ci_indices = [i for i, pc in enumerate(partial_components) if pc is not None]
        use_rground = self.weights[5] > 0.0
        rground_scores = [0.0] * len(extracted_texts)

        if use_rground and self.online_rground is not None:
            if ci_indices:
                ci_texts = [extracted_texts[i] for i in ci_indices]
                ci_prompts = [prompt_texts[i] for i in ci_indices]
                ci_metas = [meta_list[i] for i in ci_indices]
                ci_rground = self.online_rground(
                    completions=ci_texts,
                    prompts=ci_prompts,
                    metadata_list=ci_metas,
                )
                for pos, i in enumerate(ci_indices):
                    rground_scores[i] = ci_rground[pos]
            if self.no_flow_scoring == "flat":
                for i in ci_indices:
                    if is_no_flow[i]:
                        gold = meta_list[i].get("gold_has_exchange")
                        rground_scores[i] = no_flow_reward(gold)
        elif use_rground:
            for i in ci_indices:
                if is_no_flow[i] and self.no_flow_scoring == "flat":
                    gold = meta_list[i].get("gold_has_exchange")
                    rground_scores[i] = no_flow_reward(gold)
                else:
                    meta = meta_list[i]
                    rground_scores[i] = r_ground_cached(
                        extracted_texts[i],
                        meta.get("prompt_id", ""),
                        self.reward_cache,
                        meta.get("is_contrastive", False),
                    )

        # Map global index → position in ci_indices for diagnostic lookup
        _ci_pos = {gi: pos for pos, gi in enumerate(ci_indices)}

        # ----- Phase 4: Combine and score -----
        scores = []
        trace_entries = []
        for i in range(len(extracted_texts)):
            meta = meta_list[i]

            # Judgment vignettes: use pre-computed score, skip CI combination
            if partial_components[i] is None:
                r = judgment_scores.get(i, 0.0)
                scores.append(r)

                if do_trace and i < 8:
                    gold_j = meta.get("gold_judgment", "")
                    norm_art = meta.get("source_norm_articulation", "")
                    j_acc = r_judgment(extracted_texts[i], gold_j)
                    j_reas = r_judgment_reasoning(extracted_texts[i])
                    j_cite = r_norm_cite(extracted_texts[i], norm_art)
                    trace_entries.append({
                        "call": self._call_count - 1,
                        "idx": i,
                        "task_type": "norm_judgment",
                        "source_id": meta.get("source_id", ""),
                        "gold_judgment": gold_j,
                        "completion_len": len(extracted_texts[i]),
                        "completion": extracted_texts[i],
                        "components": {
                            "r_judgment": round(j_acc, 4),
                            "r_reasoning": round(j_reas, 4),
                            "r_norm_cite": round(j_cite, 4),
                        },
                        "composite": round(r, 4),
                    })
                continue

            components = partial_components[i] + [rground_scores[i]]
            r = sum(w * c for w, c in zip(self.weights, components))
            scores.append(r)

            if do_trace and i < 8:
                entry = {
                    "call": self._call_count - 1,
                    "idx": i,
                    "task_type": "ci_extraction",
                    "source_id": meta.get("source_id", ""),
                    "prompt_id": meta.get("prompt_id", ""),
                    "is_contrastive": meta.get("is_contrastive", False),
                    "contrastive_source": meta.get("contrastive_source"),
                    "gold_has_exchange": meta.get("gold_has_exchange"),
                    "is_no_flow": is_no_flow[i],
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
                }
                # Attach per-flow R_ground diagnostics when available.
                # last_diagnostics is indexed by CI-only position, not
                # global index, so map through _ci_pos.
                if (
                    self.online_rground is not None
                    and hasattr(self.online_rground, "last_diagnostics")
                    and i in _ci_pos
                    and _ci_pos[i] < len(self.online_rground.last_diagnostics)
                ):
                    entry["rground_flows"] = self.online_rground.last_diagnostics[_ci_pos[i]]
                trace_entries.append(entry)

        if trace_entries:
            self._log_trace(trace_entries)

        return scores
