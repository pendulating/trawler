"""SFT data preparation: format CI extraction outputs into training pairs.

Takes CI reasoning and CI extraction parquets from historical_norms,
groups extraction rows back to chunk level, and produces combined
reasoning-then-extraction completions in chat messages format for TRL SFTTrainer.

Output schema per completion (flat, no nesting):
{
  "reasoning": "<narrative trace covering all flows>",
  "has_information_exchange": true,
  "flows": [
    {
      "sender": "...", "recipient": "...", "subject": "...",
      "information_type": "...", "transmission_principle": "...",
      "context": "...", "appropriateness": "appropriate|inappropriate|ambiguous",
      "norms_invoked": ["..."], "norm_source": "explicit|implicit|both",
      "is_new_flow": false, "confidence": 8
    }
  ]
}
"""

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import pandas as pd
from omegaconf import OmegaConf


# Story-level information transfer between characters. Used to curate SFT
# negatives (see `negative_selection` below). The gold reasoning ALWAYS contains
# the boilerplate "...regulates the exchange of information between agents" (the
# reason a chunk is has_information_exchange=False), so we cannot key on bare
# "exchange"/"information"; instead we positively detect a described transfer.
_STORY_INTERACTION_RE = re.compile(
    r"conversations?\b|dialogues?\b|interact|social\s+interaction|"
    r"\bdisclos|\bconfid(e|es|ed|ing|ential)|\bconfess|"
    r"\btell(s|ing|)\b|\btold\b|\breveal|\bshar(e|es|ed|ing)\b|"
    r"\bwithh(o|e)ld|\bcommunicat|\bgossip|\bconvey|\bimpart|\binform(s|ed|ing)\b|"
    r"\bletter(s)?\b|\bmessage(s)?\b|\bcorrespondence\b|\bnote(s)?\b|"
    r"\bask(s|ed|ing)?\b|\binquir|\brequest|\bwarn(s|ed|ing)?\b|"
    r"\bannounc|\bconsult|\bconverse|\bspeaks?\s+(to|with)|"
    r"\bquestion(s|ed|ing)?\b|\badvis(e|es|ed|ing)\b|\bnegotiat|"
    r"\bconceal(s|ed|ing|ment)?\b|\bsecret(s|ly)?\b|\bprivate\s+(matter|affair|information)|"
    r"information\s+(is|are|being|was|were)\s+(being\s+)?(exchang|transmit|convey|disclos|shar|pass|impart|communicat)|"
    r"instances?\s+of\s+(information\s+)?(being\s+)?(exchang|communicat|disclos|shar)|"
    r"exchanged?\s+between|exchange\s+of\s+(pleasantr|news|letters|glances|confidence|gossip|secret|ideas|greetings)|"
    r"involv(e|es|ed|ing)\b.{0,30}\bexchang|"  # "involves the exchange of information" (≠ "regulates the exchange" boilerplate)
    r"transmission\s+of\s+information|transmits|passing\s+(on\s+|along\s+)?(of\s+)?information",
    re.I,
)


def _is_contentless_chunk(reasoning_text: str) -> bool:
    """Whether a gold has_information_exchange=False chunk is *genuinely* flow-free.

    `has_information_exchange` is a **prescriptive-norm** label: a chunk is False
    when no norm *commands/prohibits/regulates* an information flow — even when a
    real (descriptive) disclosure or conversation is present (e.g. a character
    confiding financial trouble to a spouse). Using those has-exchange-but-no-norm
    chunks as SFT "abstain" negatives teaches the model to under-extract real flows
    — the over-abstention seen across the v2-v5 GRPO runs. A chunk is treated as
    contentless (a clean negative) only when its gold reasoning shows no sign of an
    inter-agent information transfer. Conservative by construction: borderline
    chunks are excluded from the negative pool rather than risk teaching abstention
    on a real flow.
    """
    t = (reasoning_text or "").strip()
    if not t:
        return True
    return not _STORY_INTERACTION_RE.search(t)


@dataclass(frozen=True)
class FieldToggles:
    """Per-flow field ablation toggles for SFT pair generation.

    The base 5-tuple (sender, recipient, subject, information_type,
    transmission_principle) and top-level reasoning + has_information_exchange
    are always emitted. These toggles drop fields from BOTH the assistant JSON
    and the user-side instruction prose.

    Note: `flow_norms_meta` bundles three fields that share a normative-source
    semantics (norms_invoked + norm_source + is_new_flow); they drop together.
    """

    flow_context: bool = True
    flow_appropriateness: bool = True
    flow_norms_meta: bool = True
    flow_confidence: bool = True


def _build_ci_instruction(
    *,
    flow_context: bool = True,
    flow_appropriateness: bool = True,
    flow_norms_meta: bool = True,
    flow_confidence: bool = True,
) -> str:
    """Build the user-side CI extraction instruction, conditioned on toggles.

    All-on returns the original verbatim text (byte-identical baseline).
    Otherwise, the metadata tail enumerates only the toggled-on fragments.
    """
    head = (
        "Analyze the following text passage for information flows using the "
        "Contextual Integrity framework. First, reason about what information "
        "exchanges are described — identify senders, recipients, subjects, "
        "information types, and transmission principles. Then provide a structured "
        "extraction of each information flow as a flat JSON object with the "
        "5-component CI tuple (sender, recipient, subject, information_type, "
        "transmission_principle)"
    )

    if flow_context and flow_appropriateness and flow_norms_meta and flow_confidence:
        return head + " and contextual metadata."

    fragments: List[str] = []
    if flow_context:
        fragments.append("the surrounding context")
    if flow_appropriateness:
        fragments.append("an appropriateness judgment")
    if flow_norms_meta:
        fragments.append(
            "the invoked norms with their source and whether each flow is new"
        )
    if flow_confidence:
        fragments.append("a confidence rating")

    if not fragments:
        return head + "."

    if len(fragments) == 1:
        joined = fragments[0]
    elif len(fragments) == 2:
        joined = f"{fragments[0]} and {fragments[1]}"
    else:
        joined = ", ".join(fragments[:-1]) + ", and " + fragments[-1]

    return head + f" along with {joined}."


# Backward-compat alias: grpo_training.py:707 imports this name.
# The all-default builder reproduces the original constant verbatim.
_CI_INSTRUCTION = _build_ci_instruction()

_NO_EXCHANGE_REASONING = (
    "This passage does not contain any identifiable information flows. "
    "While the text may describe characters, settings, or events, there is "
    "no explicit or implicit transfer of information between a sender and "
    "a recipient. No CI tuple can be constructed."
)


def _build_reasoning_trace(group_df: pd.DataFrame) -> str:
    """Build a single narrative reasoning trace from per-flow reasoning entries."""
    if "ci_flow_index" in group_df.columns:
        group_df = group_df.sort_values("ci_flow_index")

    parts = []
    for i, (_, row) in enumerate(group_df.iterrows()):
        snippet = row.get("ci_flow_snippet") or ""
        trace = row.get("ci_reasoning_trace") or ""
        if not trace.strip():
            continue
        # Number flows when there are multiple
        prefix = f"Flow {i + 1}: " if len(group_df) > 1 else ""
        if snippet.strip():
            parts.append(f'{prefix}Re: "{snippet.strip()}" — {trace.strip()}')
        else:
            parts.append(f"{prefix}{trace.strip()}")
    return "\n\n".join(parts)


def _reconstruct_flows(
    group_df: pd.DataFrame,
    toggles: FieldToggles,
) -> List[Dict[str, Any]]:
    """Reconstruct flat CI flow dicts from exploded extraction rows.

    Per-flow fields beyond the base 5-tuple are gated by `toggles`. Toggling
    `flow_norms_meta` off drops the bundled trio (norms_invoked, norm_source,
    is_new_flow) together.
    """
    if "ci_flow_index" in group_df.columns:
        group_df = group_df.sort_values("ci_flow_index")

    flows = []
    for _, row in group_df.iterrows():
        flow: Dict[str, Any] = {
            "sender": row.get("ci_sender"),
            "recipient": row.get("ci_recipient"),
            "subject": row.get("ci_subject"),
            "information_type": row.get("ci_information_type"),
            "transmission_principle": row.get("ci_transmission_principle"),
        }

        if toggles.flow_context:
            flow["context"] = row.get("ci_context")
        if toggles.flow_appropriateness:
            flow["appropriateness"] = row.get("ci_appropriateness")
        if toggles.flow_norms_meta:
            norms = row.get("ci_norms_invoked", [])
            if isinstance(norms, str):
                try:
                    norms = json.loads(norms)
                except (json.JSONDecodeError, TypeError):
                    norms = []
            if not isinstance(norms, list):
                norms = []
            flow["norms_invoked"] = norms
            flow["norm_source"] = row.get("ci_norm_source")
            flow["is_new_flow"] = bool(row.get("ci_is_new_flow", False))
        if toggles.flow_confidence:
            flow["confidence"] = row.get("ci_confidence_qual")

        flows.append(flow)
    return flows


def run_sft_data_prep_stage(
    ci_reasoning_df: pd.DataFrame,
    ci_extraction_df: pd.DataFrame,
    cfg: Any,
) -> pd.DataFrame:
    """Prepare SFT training pairs from CI reasoning and extraction outputs.

    Groups the per-flow exploded extraction rows back to chunk level and
    builds combined reasoning-then-extraction completions.

    Args:
        ci_reasoning_df: DataFrame from ci_reasoning stage with columns:
            article_text, ci_reasoning_json, has_information_exchange,
            ci_flow_count, gutenberg_id, chunk_id.
        ci_extraction_df: DataFrame from ci_extraction stage (exploded, one
            row per flow) with ci_* columns.
        cfg: Hydra config.

    Returns:
        DataFrame with columns: messages (JSON string), source_id, task_type.
    """
    # Per-flow field ablation toggles. Defaults preserve the full schema; flipping
    # any to false drops the corresponding fields from BOTH the assistant JSON
    # and the user-side instruction prose. See FieldToggles docstring above.
    toggles = FieldToggles(
        flow_context=bool(OmegaConf.select(cfg, "training.sft.flow_context", default=True)),
        flow_appropriateness=bool(OmegaConf.select(cfg, "training.sft.flow_appropriateness", default=True)),
        flow_norms_meta=bool(OmegaConf.select(cfg, "training.sft.flow_norms_meta", default=True)),
        flow_confidence=bool(OmegaConf.select(cfg, "training.sft.flow_confidence", default=True)),
    )
    instruction = _build_ci_instruction(**asdict(toggles))

    # Book-level filter: restrict to a single book's data
    book_id = OmegaConf.select(cfg, "runtime.book_id", default=None)
    if book_id is not None:
        book_id = str(book_id)
        for _df_name, _df in [("ci_reasoning", ci_reasoning_df), ("ci_extraction", ci_extraction_df)]:
            for col in ("gutenberg_id", "source_id", "book_id"):
                if col in _df.columns:
                    mask = _df[col].astype(str) == book_id
                    if _df_name == "ci_reasoning":
                        ci_reasoning_df = ci_reasoning_df[mask].reset_index(drop=True)
                    else:
                        ci_extraction_df = ci_extraction_df[mask].reset_index(drop=True)
                    break
        print(f"[sft_data_prep] Filtered to book_id={book_id}: "
              f"{len(ci_reasoning_df)} reasoning, {len(ci_extraction_df)} extraction rows")

    # Identify the chunk grouping columns present in both DataFrames
    group_cols = []
    for candidate in ("chunk_id", "gutenberg_id"):
        if candidate in ci_extraction_df.columns:
            group_cols.append(candidate)
    if not group_cols:
        raise ValueError(
            "[sft_data_prep] No chunk grouping columns (chunk_id, gutenberg_id) "
            f"found in extraction data. Available: {list(ci_extraction_df.columns)}"
        )

    # Build a lookup from chunk identifiers to reasoning metadata
    reasoning_lookup: Dict[tuple, Dict[str, Any]] = {}
    text_col = "article_text" if "article_text" in ci_reasoning_df.columns else None
    if text_col is None:
        for candidate in ("chunk_text", "text"):
            if candidate in ci_reasoning_df.columns:
                text_col = candidate
                break
    if text_col is None:
        raise ValueError("[sft_data_prep] No text column found in reasoning data")

    reasoning_group_cols = [c for c in group_cols if c in ci_reasoning_df.columns]
    for _, row in ci_reasoning_df.iterrows():
        key = tuple(row.get(c) for c in reasoning_group_cols)
        reasoning_lookup[key] = {
            "article_text": row.get(text_col),
            "has_information_exchange": bool(row.get("has_information_exchange", False)),
            "source_id": str(row.get("gutenberg_id") or row.get("source_id") or "unknown"),
            "reasoning_text": row.get("ci_reasoning_text", ""),
        }

    # Group extraction rows back to chunk level
    pairs: List[Dict[str, Any]] = []
    for group_key, group_df in ci_extraction_df.groupby(group_cols):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        # Look up reasoning for this chunk
        reasoning_key = group_key[:len(reasoning_group_cols)]
        reasoning_info = reasoning_lookup.get(reasoning_key)
        if reasoning_info is None:
            continue

        article_text = reasoning_info["article_text"]
        if not article_text or (isinstance(article_text, float) and pd.isna(article_text)):
            continue

        # Build narrative reasoning trace from per-flow reasoning entries
        reasoning_trace = _build_reasoning_trace(group_df)
        if not reasoning_trace.strip():
            continue

        # Reconstruct flat extraction flows from the exploded rows
        extraction_flows = _reconstruct_flows(group_df, toggles)
        if not extraction_flows:
            continue

        # Build the combined completion (flat structure)
        completion = {
            "reasoning": reasoning_trace,
            "has_information_exchange": reasoning_info["has_information_exchange"],
            "flows": extraction_flows,
        }
        completion_str = json.dumps(completion, ensure_ascii=False)

        messages = [
            {"role": "user", "content": f"{instruction}\n\n{article_text}"},
            {"role": "assistant", "content": completion_str},
        ]

        pairs.append({
            "messages": json.dumps(messages, ensure_ascii=False),
            "source_id": reasoning_info["source_id"],
            "task_type": "ci_extraction",
        })

    n_positive = len(pairs)

    # Include chunks with has_information_exchange=False as negative examples.
    # These teach the model to produce empty flows when no exchange is present,
    # exercising the r_consist invariant (False => empty flows).
    # Gated by training.sft.include_negative_examples (default True).
    include_negatives = True
    try:
        include_negatives = bool(OmegaConf.select(cfg, "training.sft.include_negative_examples", default=True))
    except Exception:
        pass

    if not include_negatives:
        n_negative = 0
        out_df = pd.DataFrame(pairs)
        print(
            f"[sft_data_prep] Created {len(out_df)} chunk-level SFT pairs "
            f"({n_positive} positive, {n_negative} negative — negatives disabled) "
            f"from {len(ci_extraction_df)} extraction rows"
        )
        if len(out_df) > 0:
            source_counts = out_df["source_id"].value_counts()
            print(f"  Sources: {dict(source_counts.head(10))}")
        return out_df

    # Negative selection mode (2026-06-19): "all" (legacy) samples any
    # has_information_exchange=False chunk; "contentless" keeps only genuinely
    # flow-free chunks (see _is_contentless_chunk) — the v6 fix for the
    # over-abstention prior, which traced to gold=False negatives that actually
    # describe real disclosures/conversations.
    negative_selection = "all"
    try:
        negative_selection = str(
            OmegaConf.select(cfg, "training.sft.negative_selection", default="all")
        ).lower()
    except Exception:
        pass
    if negative_selection not in ("all", "contentless"):
        raise ValueError(
            f"[sft_data_prep] unknown training.sft.negative_selection="
            f"{negative_selection!r} (expected 'all' or 'contentless')"
        )

    consumed_keys = {
        group_key[:len(reasoning_group_cols)]
        for group_key in ci_extraction_df.groupby(group_cols).groups
    }
    negative_candidates = []
    n_dropped_has_exchange = 0
    for key, info in reasoning_lookup.items():
        if key in consumed_keys:
            continue
        if info["has_information_exchange"]:
            continue
        article_text = info["article_text"]
        if not article_text or (isinstance(article_text, float) and pd.isna(article_text)):
            continue
        if (negative_selection == "contentless"
                and not _is_contentless_chunk(info.get("reasoning_text", ""))):
            n_dropped_has_exchange += 1
            continue
        negative_candidates.append((key, info))

    if negative_selection == "contentless":
        print(
            f"[sft_data_prep] negative_selection=contentless: kept "
            f"{len(negative_candidates)} contentless negatives, dropped "
            f"{n_dropped_has_exchange} has-exchange-but-no-norm chunks"
        )

    # Cap negatives so they don't overwhelm positives (at most 1:1 ratio).
    import random as _rng

    max_negatives = n_positive
    if len(negative_candidates) > max_negatives:
        _rng.shuffle(negative_candidates)
        negative_candidates = negative_candidates[:max_negatives]

    for _key, info in negative_candidates:
        reasoning = (info.get("reasoning_text") or "").strip() or _NO_EXCHANGE_REASONING
        completion = {
            "reasoning": reasoning,
            "has_information_exchange": False,
            "flows": [],
        }
        messages = [
            {"role": "user", "content": f"{instruction}\n\n{info['article_text']}"},
            {"role": "assistant", "content": json.dumps(completion, ensure_ascii=False)},
        ]
        pairs.append({
            "messages": json.dumps(messages, ensure_ascii=False),
            "source_id": info["source_id"],
            "task_type": "ci_extraction",
        })

    n_negative = len(pairs) - n_positive

    out_df = pd.DataFrame(pairs)
    print(
        f"[sft_data_prep] Created {len(out_df)} chunk-level SFT pairs "
        f"({n_positive} positive, {n_negative} negative) "
        f"from {len(ci_extraction_df)} extraction rows"
    )
    if len(out_df) > 0:
        source_counts = out_df["source_id"].value_counts()
        print(f"  Sources: {dict(source_counts.head(10))}")

    return out_df
