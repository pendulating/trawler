"""SFT-policy variance pre-screening of GRPO training prompts.

The May 2026 sweep showed most GRPO groups carried no learning signal:
60% had identical R_ground across completions and 78% of completions were
no-flow declarations, so group-relative advantages were zero or judge
noise. This module screens prompts *before* training: sample G completions
per prompt from the SFT policy (the GRPO initialization), score them with
the same composite reward used in training, and keep only prompts whose
group reward std clears a threshold — i.e. prompts where the policy's own
behavior is undecided enough for GRPO to learn from.

Screening cost is one generation + reward pass over the dataset. Because
every sweep cell starts from the same SFT checkpoint, the result is cached
(keyed on checkpoint, prompt set, and sampling params) and reused across
cells via ``training.grpo.prescreen.cache_path``.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _std(values: List[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def select_prompts_by_reward_std(
    rewards_by_prompt: Dict[str, List[float]],
    reward_std_min: float,
    min_keep: int = 8,
) -> Tuple[List[str], Dict[str, float]]:
    """Pick prompts whose sampled-completion rewards vary enough to train on.

    Args:
        rewards_by_prompt: prompt key → composite rewards of its G samples.
        reward_std_min: keep prompts with group reward std >= this.
        min_keep: if fewer prompts clear the threshold, fall back to the
            top ``min_keep`` by std (an over-aggressive threshold must not
            empty the dataset).

    Returns:
        (kept prompt keys, prompt key → std) — kept keys preserve the
        iteration order of ``rewards_by_prompt``.
    """
    stds = {k: _std(v) for k, v in rewards_by_prompt.items()}
    return _select_from_stds(stds, reward_std_min, min_keep)


def _reward_signature(reward_fn, temperature: float, max_tokens: int) -> str:
    """Serialize everything that changes sampled completions or their scores.

    Any knob missing here lets a sweep cell silently reuse a screen computed
    under different reward dynamics (2026-06-09 review, F4: rank_top_k /
    rank_weight were absent, so retrieval-depth changes reused stale caches).
    judge_model may still read "default" if the client hasn't resolved the
    served model id yet — it still distinguishes explicitly-configured judges.

    The config knobs above only guard CONFIG-level reward changes. A change to
    the scoring *formula* in code (same knobs, different math) would otherwise
    cache-hit on a stale screen — so ``rground_formula_version`` is bumped
    whenever the R_ground computation changes (v8 2026-06-22: symmetric
    contrastive clamp). See 2026-06-22_v8_plan.md.
    """
    rg = getattr(reward_fn, "online_rground", None)
    return json.dumps({
        "weights": getattr(reward_fn, "weights", []),
        "composition": getattr(reward_fn, "composition", "additive"),
        "abstention_penalty": getattr(reward_fn, "abstention_penalty", 0.0),
        "lambda": getattr(rg, "contrastive_lambda", None),
        "scoring_mode": getattr(rg, "scoring_mode", None),
        "rank_top_k": getattr(rg, "rank_top_k", None),
        "rank_weight": getattr(rg, "rank_weight", None),
        "app_weight": getattr(rg, "app_weight", None),
        "rground_formula_version": "v8_symmetric_clamp",
        "judge_model": getattr(getattr(rg, "judge_client", None),
                               "model_name", None),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }, sort_keys=True)


def _cache_key(
    sft_checkpoint: str,
    prompt_keys: List[str],
    num_samples: int,
    reward_signature: str,
) -> str:
    """Cache identity: SFT policy + exact prompt set + sampling/reward setup."""
    h = hashlib.sha256()
    h.update(str(sft_checkpoint).encode())
    h.update(str(num_samples).encode())
    h.update(reward_signature.encode())
    for k in sorted(prompt_keys):
        h.update(hashlib.sha256(k.encode()).digest())
    return h.hexdigest()[:32]


def prescreen_dataset(
    dataset,
    reward_fn,
    model_dir: str,
    grpo_cfg: Dict[str, Any],
    output_dir: str,
    cache_identity: str = "",
    composite_config_path: str = "",
) -> Any:
    """Filter a GRPO dataset down to prompts with non-degenerate reward groups.

    Samples ``prescreen.num_samples`` completions per prompt from the model
    at ``model_dir`` (the merged SFT checkpoint) with vLLM, scores them with
    ``reward_fn`` (judge servers must be reachable when R_ground is online),
    and drops prompts whose group reward std is below
    ``prescreen.reward_std_min``.

    The vLLM engine is created and torn down inside this function, before
    TRL's colocated engine starts. A screening report is written to
    ``output_dir/prescreen_report.json``; per-prompt stats are cached at
    ``prescreen.cache_path`` (if set) so sweep cells sharing the SFT
    checkpoint skip the sampling pass.

    Args:
        cache_identity: Stable identifier of the policy being screened
            (the persistent SFT checkpoint path) — ``model_dir`` is a
            per-job scratch copy whose path would defeat cross-cell caching.
        composite_config_path: Original model zoo path. Qwen3.5 merged
            checkpoints can carry a text-only config; vLLM needs the
            composite config (with vision_config) to load them — same
            workaround as the TRL vLLM-init patch in grpo_training.py.

    Returns the filtered dataset (or the original if screening is disabled
    or fails safe).
    """
    ps_cfg = grpo_cfg.get("prescreen") or {}
    if not ps_cfg.get("enabled", False):
        return dataset

    num_samples = int(ps_cfg.get("num_samples", 8))
    reward_std_min = float(ps_cfg.get("reward_std_min", 0.05))
    min_keep = int(ps_cfg.get("min_keep", 8))
    temperature = float(ps_cfg.get("temperature", 1.0))
    cache_path = ps_cfg.get("cache_path") or ""
    max_tokens = int(grpo_cfg.get("max_completion_length", 3072))
    # Option 3 (2026-06-12): also drop prompts whose SFT samples unanimously
    # abstain (no flow-decision variance). Applied at selection time from
    # cached extract fractions — a post-hoc knob like reward_std_min/min_keep,
    # so it is NOT part of the cache key.
    require_flow_variance = bool(ps_cfg.get("require_flow_variance", False))

    prompt_keys = [row["prompt"] for row in dataset]
    # Reward signature: anything that changes scores invalidates the cache.
    reward_signature = _reward_signature(reward_fn, temperature, max_tokens)
    key = _cache_key(
        cache_identity or model_dir,
        prompt_keys, num_samples, reward_signature,
    )

    stds: Optional[Dict[str, float]] = None
    extract_fracs: Optional[Dict[str, float]] = None
    no_flow_rate: Optional[float] = None
    cache_hit = False
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("cache_key") == key:
                _cached_fracs = cached.get("extract_fracs")
                if require_flow_variance and _cached_fracs is None:
                    # Cache predates option 3: the samples are valid but we
                    # have no per-prompt extract fractions to screen on.
                    print(f"[prompt_screening] Cache at {cache_path} predates "
                          f"flow-variance screening (no extract_fracs) "
                          f"— re-screening")
                else:
                    stds = cached.get("stds", {})
                    extract_fracs = _cached_fracs
                    no_flow_rate = cached.get("no_flow_rate")
                    cache_hit = True
                    print(f"[prompt_screening] Loaded cached screen "
                          f"({len(stds)} prompts) from {cache_path}")
            else:
                print(f"[prompt_screening] Cache key mismatch at {cache_path} "
                      f"— re-screening")
        except Exception as e:
            print(f"[prompt_screening] Failed to read cache ({e}) — re-screening")

    if stds is None:
        rewards_by_prompt, no_flow_rate, extract_fracs = _sample_and_score(
            prompt_keys, reward_fn, model_dir,
            num_samples=num_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            gpu_memory_utilization=float(
                grpo_cfg.get("vllm_gpu_memory_utilization", 0.45)
            ),
            max_model_len=grpo_cfg.get("vllm_max_model_length"),
            composite_config_path=composite_config_path,
        )
        stds = {k: _std(v) for k, v in rewards_by_prompt.items()}
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "cache_key": key,
                        "num_samples": num_samples,
                        "no_flow_rate": no_flow_rate,
                        "stds": stds,
                        "extract_fracs": extract_fracs,
                    }, f)
                print(f"[prompt_screening] Wrote screen cache to {cache_path}")
            except Exception as e:
                print(f"[prompt_screening] Failed to write cache: {e}")

    # Only CI-extraction prompts may be flow-variance-dropped; judgment
    # vignettes never emit an extraction array (extract_frac structurally 0)
    # and must be exempt or the screen wipes out the Phase 4 vignette mix.
    ci_prompt_keys = {
        row["prompt"] for row in dataset
        if row.get("task_type", "ci_extraction") == "ci_extraction"
    }
    eligible_stds, n_flow_dropped = _apply_flow_variance_filter(
        stds, extract_fracs, require_flow_variance, ci_prompt_keys)
    kept_keys, _ = _select_from_stds(eligible_stds, reward_std_min, min_keep)

    kept_set = set(kept_keys)
    filtered = dataset.filter(lambda row: row["prompt"] in kept_set)

    report = {
        "cache_key": key,
        "cache_hit": cache_hit,
        "num_samples": num_samples,
        "reward_std_min": reward_std_min,
        "n_prompts_in": len(dataset),
        "n_prompts_kept": len(filtered),
        "n_prompts_dropped": len(dataset) - len(filtered),
        "require_flow_variance": require_flow_variance,
        "n_dropped_flow_variance": n_flow_dropped,
        "sft_no_flow_rate": no_flow_rate,
        "std_quantiles": _quantiles(sorted(stds.values())),
    }
    report_path = os.path.join(output_dir, "prescreen_report.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _flow_note = (
        f"; dropped {n_flow_dropped} unanimous-abstain"
        if require_flow_variance else ""
    )
    print(f"[prompt_screening] Kept {len(filtered)}/{len(dataset)} prompts "
          f"(std>={reward_std_min}{_flow_note}); SFT no-flow rate={no_flow_rate}; "
          f"report → {report_path}")
    _log_report_to_wandb(report)
    return filtered


def _log_report_to_wandb(report: Dict[str, Any]) -> None:
    """Surface the screen result on the active W&B run.

    The full report goes into ``config.prescreen`` (run-level fact, not a
    time series); the headline numbers also land in the summary under
    ``prescreen/*`` so sweep tables can sort on them directly.
    """
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.run.config.update({"prescreen": report}, allow_val_change=True)
        for k in ("n_prompts_in", "n_prompts_kept", "n_prompts_dropped",
                  "sft_no_flow_rate", "cache_hit"):
            if report.get(k) is not None:
                wandb.run.summary[f"prescreen/{k}"] = report[k]
    except Exception:
        pass


def _apply_flow_variance_filter(
    stds: Dict[str, float],
    extract_fracs: Optional[Dict[str, float]],
    require_flow_variance: bool,
    ci_prompt_keys: Optional[set] = None,
) -> Tuple[Dict[str, float], int]:
    """Drop prompts whose SFT samples unanimously abstain from extraction.

    Such a group has no flow-decision variance: every sample declared no
    flow, so the extract-vs-abstain advantage GRPO learns from is absent,
    and under scale_rewards="none" a flat abstention penalty cancels out
    under group-mean centering — the group can never teach "extract here".
    Pure-extract groups are KEPT: R_ground still ranks their extraction
    quality, a different and valid learning signal.

    Only CI-extraction prompts are eligible to be dropped. Judgment
    vignettes (task_type ``norm_judgment``) structurally never emit an
    ``extraction`` array, so their extract fraction is always 0 — without
    this guard the filter drops every vignette and silently disables the
    Phase 4 vignette mix (observed 2026-06-14: 0/413 vignettes survived a
    G=8 screen). ``ci_prompt_keys`` is the set of prompts the filter may
    consider; a prompt outside it is always kept. When None (no task-type
    information), every prompt is eligible (legacy behavior).

    A prompt is dropped only when its sampled extract fraction is exactly 0.
    Missing entries fail safe to "keep" (treated as extract_frac 1.0) so a
    partial/corrupt cache never silently empties the dataset.

    Returns (eligible stds, n_dropped). A no-op (returns a copy of stds,
    0) when disabled or when extract fractions are unavailable.
    """
    if not require_flow_variance or not extract_fracs:
        return dict(stds), 0

    def _dropped(k: str) -> bool:
        if ci_prompt_keys is not None and k not in ci_prompt_keys:
            return False  # non-CI prompt (e.g. vignette) — never dropped here
        return extract_fracs.get(k, 1.0) <= 0.0

    eligible = {k: s for k, s in stds.items() if not _dropped(k)}
    return eligible, len(stds) - len(eligible)


def _select_from_stds(
    stds: Dict[str, float],
    reward_std_min: float,
    min_keep: int,
) -> Tuple[List[str], Dict[str, float]]:
    """Threshold pre-computed stds (cache path skips re-sampling)."""
    kept = [k for k, s in stds.items() if s >= reward_std_min]
    if len(kept) < min_keep:
        ranked = sorted(stds, key=stds.get, reverse=True)[:min_keep]
        ranked_set = set(ranked)
        kept = [k for k in stds if k in ranked_set]
        print(f"[prompt_screening] WARNING: threshold left "
              f"{sum(1 for s in stds.values() if s >= reward_std_min)} prompts; "
              f"keeping top {len(kept)} by std instead")
    return kept, stds


def _quantiles(sorted_vals: List[float]) -> Dict[str, float]:
    """Distribution summary for the screening report."""
    if not sorted_vals:
        return {}
    n = len(sorted_vals)
    return {
        q: round(sorted_vals[min(n - 1, int(n * p))], 4)
        for q, p in (("p10", 0.10), ("p25", 0.25), ("p50", 0.50),
                     ("p75", 0.75), ("p90", 0.90))
    }


def _sample_and_score(
    prompt_keys: List[str],
    reward_fn,
    model_dir: str,
    num_samples: int,
    temperature: float,
    max_tokens: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    composite_config_path: str = "",
) -> Tuple[Dict[str, List[float]], float]:
    """Generate G samples per prompt from the SFT policy and score them.

    The vLLM engine is constructed and destroyed here so the GPUs are free
    again before TRL's colocated engine initializes.
    """
    import gc

    import torch
    from vllm import LLM, SamplingParams

    print(f"[prompt_screening] Sampling {num_samples} completions × "
          f"{len(prompt_keys)} prompts from {model_dir}")

    llm_kwargs: Dict[str, Any] = dict(
        model=model_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
    )
    if max_model_len:
        llm_kwargs["max_model_len"] = int(max_model_len)
    if composite_config_path:
        # Qwen3.5 merged checkpoints may carry the text-only config; give
        # vLLM the composite config (with vision_config) from the zoo path.
        def _ensure_composite(config):
            if hasattr(config, "vision_config") and config.vision_config is not None:
                return config
            from transformers import AutoConfig
            try:
                return AutoConfig.from_pretrained(
                    composite_config_path, trust_remote_code=True,
                )
            except Exception:
                return config
        llm_kwargs["hf_overrides"] = _ensure_composite
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        n=num_samples,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
    )
    try:
        outputs = llm.generate(prompt_keys, sampling)
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    # Score with the training reward, suppressing trace logging so the
    # screening pass doesn't pollute reward_traces.jsonl call numbering.
    flat_prompts: List[str] = []
    flat_completions: List[str] = []
    for prompt_key, out in zip(prompt_keys, outputs):
        for sample in out.outputs:
            flat_prompts.append(prompt_key)
            flat_completions.append(sample.text)

    # Score in chunks of whole groups: a single call covering every prompt
    # embeds and judges tens of thousands of texts in one shot — the
    # 2026-06-10 launch died on an embedding read-timeout at 1103 prompts
    # × 8 samples. Chunks must not split a group (ranked R_ground scores
    # same-prompt completions jointly).
    group_bounds: List[int] = [0]
    for out in outputs:
        group_bounds.append(group_bounds[-1] + len(out.outputs))
    groups_per_chunk = 64

    _saved_trace_path = reward_fn._trace_path
    reward_fn._trace_path = None
    flat_rewards: List[float] = []
    try:
        for g0 in range(0, len(outputs), groups_per_chunk):
            g1 = min(g0 + groups_per_chunk, len(outputs))
            lo, hi = group_bounds[g0], group_bounds[g1]
            flat_rewards.extend(reward_fn(
                prompts=flat_prompts[lo:hi],
                completions=flat_completions[lo:hi],
            ))
            print(f"[prompt_screening] Scored {hi}/{len(flat_prompts)} "
                  f"completions ({g1}/{len(outputs)} prompts)")
    finally:
        reward_fn._trace_path = _saved_trace_path

    rewards_by_prompt: Dict[str, List[float]] = {}
    pos = 0
    for prompt_key, out in zip(prompt_keys, outputs):
        n_out = len(out.outputs)
        rewards_by_prompt[prompt_key] = list(flat_rewards[pos:pos + n_out])
        pos += n_out

    # Per-completion flow decision, bucketed per prompt. Two products:
    #  - no_flow_rate: global rate the SFT policy takes the lazy path (the
    #    no-flow-collapse gate's baseline).
    #  - extract_frac_by_prompt: fraction of a prompt's G samples that
    #    extracted >=1 flow — the flow-decision-variance signal option 3
    #    screens on (a prompt with extract_frac 0 unanimously abstained).
    from .rewards import _parse_completion
    n_no_flow = 0
    extract_frac_by_prompt: Dict[str, float] = {}
    for prompt_key, out in zip(prompt_keys, outputs):
        n_extracted = 0
        for sample in out.outputs:
            parsed = _parse_completion(reward_fn._extract_text(sample.text))
            flows = (parsed.get("extraction") or []) if parsed is not None else []
            if flows:
                n_extracted += 1
                continue
            # No extracted flow: count an explicit no-exchange declaration
            # toward the no-flow rate (unparseable completions are neither
            # a clean abstention nor an extraction, matching prior semantics).
            if parsed is not None:
                reasoning = parsed.get("reasoning", {})
                has_ex = reasoning.get("has_information_exchange") \
                    if isinstance(reasoning, dict) else None
                if has_ex is False:
                    n_no_flow += 1
        n = len(out.outputs)
        extract_frac_by_prompt[prompt_key] = (n_extracted / n) if n else 0.0
    no_flow_rate = round(n_no_flow / max(len(flat_completions), 1), 4)

    return rewards_by_prompt, no_flow_rate, extract_frac_by_prompt
