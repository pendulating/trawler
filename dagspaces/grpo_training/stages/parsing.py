"""Completion parsing shared by the GRPO reward stack.

Extracted from the retired v9 ``rewards.py`` so that the m-series modular
reward does not depend on a deprecated module. Two entry points:

  ``parse_completion``          CI flow-extraction completions, normalising the
                                flat SFT schema to the nested canonical one
  ``parse_judgment_completion`` norm-judgment vignette completions

Both return ``None`` when the completion does not yield a JSON object, which is
how callers detect a schema-invalid generation.
"""

from __future__ import annotations

import json
import re
from typing import Any


def to_str(val) -> str:
    """Coerce a value to string, joining lists with ', '."""
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val)


def parse_completion(text: str) -> dict[str, Any] | None:
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
    from dagspaces.common.json_extraction import extract_json_from_text

    obj, _ = extract_json_from_text(text)
    if obj is None:
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
            reasoning_entries.append(
                {
                    "original_text_snippet": "",
                    "reasoning": reasoning_val,
                    "context_identified": flow.get("context", ""),
                    "flow_direction": "",
                    "potential_appropriateness": flow.get(
                        "appropriateness", "ambiguous"
                    ),
                    "is_new_flow": flow.get("is_new_flow", False),
                }
            )

        nested_reasoning = {
            "flows": reasoning_entries,
            "has_information_exchange": has_exchange,
        }

        # Build nested extraction objects.
        #
        # NOTE (SFT pair-format ablation incompatibility): the per-flow reads
        # below assume the full SFT schema (context, appropriateness,
        # norms_invoked, norm_source, is_new_flow, confidence). SFT checkpoints
        # produced with any `training.sft.flow_*=false` toggle (see
        # dagspaces/grpo_training/conf/training/sft/{no_context,no_appropriateness,
        # no_norms_meta,no_confidence,minimal_tuple}.yaml) will emit completions
        # missing those fields. The `.get(..., default)` calls here silently
        # substitute defaults (e.g. confidence_quant=5, is_new_flow=False),
        # which makes the reward components degenerate without any error.
        # Running GRPO on an ablated SFT checkpoint is UNSUPPORTED — those
        # checkpoints are SFT-only artifacts for downstream CI benchmark
        # evaluation.
        extraction = []
        for flow in flat_flows:
            if not isinstance(flow, dict):
                continue
            extraction.append(
                {
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
                }
            )

        obj = {"reasoning": nested_reasoning, "extraction": extraction}

    return obj


def parse_judgment_completion(text: str) -> dict[str, Any] | None:
    """Extract JSON from a norm judgment vignette completion.

    Simpler than :func:`parse_completion` — no flat→nested normalisation.
    Expects: {"judgment": "yes"/"no", "reasoning": "...", "norms_considered": [...]}
    """
    if not text:
        return None
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                obj = json.loads(match.group())
            except json.JSONDecodeError:
                pass
    if obj is None or not isinstance(obj, dict):
        return None
    return obj


# Legacy aliases: the v9 stack exported these names with a leading underscore.
_to_str = to_str
_parse_completion = parse_completion
_parse_judgment_completion = parse_judgment_completion
