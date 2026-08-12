"""GRPO training stage: Group Relative Policy Optimization via TRL.

Fine-tunes an SFT-adapted model using GRPO with composite reward signals.
Uses TRL's GRPOTrainer with vLLM for generation.

Two vLLM modes supported (configured via training.grpo.vllm_mode):

  "colocate" (default for 4-GPU setups):
    vLLM runs in the same process as training, sharing GPUs. Lower vLLM memory
    utilization (0.3). Optional sleep mode offloads vLLM weights during optimizer
    steps. Works with 4 GPUs.

  "server" (for 8+ GPU setups):
    Separate `trl vllm-serve` process on dedicated GPUs. Training on remaining
    GPUs via `accelerate launch`. Weight sync via NCCL after each training step.
    The runner is responsible for launching the server subprocess.
"""

import json
import os
import random
from typing import Any

import pandas as pd
from omegaconf import OmegaConf


def _generate_vignettes(
    norm_universes: dict[str, list],
    prompt_template: str,
) -> list[dict[str, Any]]:
    """Generate judgment vignettes from info-flow-governing norms.

    Each norm with ``governs_info_flow=true`` and a clear normative force
    (obligatory/prohibited/recommended/discouraged) becomes a vignette.
    The scenario is built from norm fields; the norm_articulation is NOT
    included (that would leak the answer).

    Returns list of dicts with: prompt_text, source_id, gold_judgment,
    source_norm (full norm dict), normative_force.
    """
    # Map normative_force → gold judgment (single source of truth in deontic.py,
    # shared with the reranker judge's appropriateness-consistency term).
    from .deontic import FORCE_TO_GOLD as _FORCE_TO_GOLD

    vignettes = []
    for source_id, norms in norm_universes.items():
        for norm in norms:
            if norm.get("governs_info_flow") is not True:
                continue
            force = norm.get("normative_force", "")
            gold = _FORCE_TO_GOLD.get(force)
            if gold is None:
                continue  # skip "permitted" and unknowns

            subject = norm.get("norm_subject", "a person")
            act = norm.get("norm_act", "share this information")
            condition = norm.get("condition_of_application", "")
            context = norm.get("context", "")

            # Build scenario (deliberately omits norm_articulation)
            scenario_parts = []
            if context:
                scenario_parts.append(
                    f"In a setting involving {context},"
                )
            scenario_parts.append(f"{subject}")
            if condition:
                scenario_parts.append(f"is in a situation where {condition}.")
            else:
                scenario_parts.append(f"is considering whether to {act}.")
            scenario = " ".join(scenario_parts)

            # Substitute into prompt template
            prompt_text = (
                prompt_template
                .replace("{{scenario}}", scenario)
                .replace("{{subject}}", subject)
                .replace("{{act}}", act)
            )

            vignettes.append({
                "prompt_text": prompt_text,
                "source_id": str(source_id),
                "gold_judgment": gold,
                "normative_force": force,
                "source_norm": norm,
            })

    return vignettes


def _resolve_vignette_universes(
    norm_universes: dict[str, list],
    vignette_norm_universes_path: str,
) -> dict[str, list]:
    """Pick the universe that judgment vignettes are generated from.

    Defaults to the grounding ``norm_universes`` — vignettes and R_ground share a
    corpus (the historical behaviour). If ``vignette_norm_universes_path`` points
    at a JSON file, vignettes are drawn from THAT corpus instead, while grounding
    and the CI-extraction prompt set stay untouched. This is a single-variable
    knob for rebalancing the judgment-vignette force mix (e.g. the more
    force-balanced top100 universe: CI-relevant app:inapp 1.72:1 vs fiction10's
    3.07:1) without re-running flow extraction.

    Uses ``isfile`` (not ``exists``) deliberately: an unset Hydra
    ``${oc.env:VIGNETTE_NORM_UNIVERSES_PATH,""}`` source resolves through
    ``os.path.abspath("")`` to the CWD (a *directory*), which must NOT be treated
    as a universe file — so anything that isn't a real file falls back to the
    grounding universe.
    """
    if vignette_norm_universes_path and os.path.isfile(vignette_norm_universes_path):
        with open(vignette_norm_universes_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # A truthy-but-non-file path (typo, stale/unmounted mount, wrong CWD under
    # submitit) must fail loud: silently falling back to the grounding universe
    # makes a "balanced-vignette" arm a byte-identical copy of its control. The
    # legit unset case resolves through os.path.abspath("") to the CWD
    # *directory* (a sentinel), which stays a silent fallback.
    if vignette_norm_universes_path and not os.path.isdir(vignette_norm_universes_path):
        raise FileNotFoundError(
            f"vignette_norm_universes_path is set but is not a file: "
            f"{vignette_norm_universes_path!r}. Fix the path or unset "
            "VIGNETTE_NORM_UNIVERSES_PATH to use the grounding universe."
        )
    return norm_universes


def _build_grpo_dataset(
    chunks_df: pd.DataFrame,
    tokenizer,
    prompt_template: str,
    enable_thinking: bool = True,
    contrastive_ratio: float = 0.0,
    all_source_ids: list[str] | None = None,
    vignettes: list[dict[str, Any]] | None = None,
    vignette_ratio: float = 0.0,
    vignette_system_prompt: str = "You are an expert in privacy norms and appropriate information sharing.",
) -> "Dataset":
    """Build GRPO training dataset from text chunks.

    Constructs prompts by applying the CI extraction instruction template
    to each chunk, then pre-applies the chat template so TRL routes
    through vLLM's ``llm.generate()`` instead of ``llm.chat()``.

    When ``contrastive_ratio > 0``, an additional ``ceil(N * ratio)`` rows
    are sampled from the original rows and appended with a randomly-chosen
    wrong ``source_id``.  A short system message is added to the contrastive
    copies so their chat-templated key is unique in the ``prompt_metadata``
    dict, while the clean ``chunk_text`` stored in ``raw_prompts`` remains
    identical.

    Args:
        chunks_df: DataFrame with columns: chunk_text, source_id.
        tokenizer: Model tokenizer for chat template formatting.
        prompt_template: CI extraction prompt with ``{{chunk_text}}`` placeholder.
        enable_thinking: Allow ``<think>`` blocks during GRPO generation.
        contrastive_ratio: Fraction of original rows to duplicate as contrastive.
        all_source_ids: List of valid source IDs for contrastive pairing.

    Returns:
        (dataset, raw_prompts) where raw_prompts maps formatted prompt → raw
        user prompt (for passing clean text to the judge).
    """
    import hashlib
    import math

    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    # Maps formatted_prompt → raw user_prompt so OnlineRGround can pass
    # clean text to the judge instead of chat-templated text.
    raw_prompts: dict[str, str] = {}

    for _, row in chunks_df.iterrows():
        chunk_text = row.get("chunk_text", "")
        if not chunk_text or (isinstance(chunk_text, float) and pd.isna(chunk_text)):
            continue

        source_id = str(row.get("source_id", ""))

        # Gold label: does this chunk actually contain information flows?
        # Used to penalize false no-flow declarations during GRPO.
        gold_has_exchange = row.get("has_information_exchange")
        if gold_has_exchange is None:
            # Fall back to flow count if available
            flow_count = row.get("ci_flow_count")
            if flow_count is not None:
                gold_has_exchange = int(flow_count) > 0

        # Build user prompt from template
        user_prompt = prompt_template.replace("{{chunk_text}}", str(chunk_text)).strip()

        # Pre-apply chat template so TRL routes through vLLM's llm.generate()
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        prompt_id = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()[:16]
        raw_prompts[formatted_prompt] = user_prompt

        rows.append({
            "prompt": formatted_prompt,
            "source_id": source_id,
            "prompt_id": prompt_id,
            "is_contrastive": False,
            "contrastive_source": None,
            "gold_has_exchange": bool(gold_has_exchange) if gold_has_exchange is not None else None,
            "task_type": "ci_extraction",
            "gold_judgment": None,
            "source_norm_articulation": None,
        })

    # --- Downsample no-flow chunks to match flow-containing chunks ---
    # The source data is heavily imbalanced (~87% no-flow). Without
    # balancing, GRPO trains mostly on no-flow chunks where there's no
    # extraction signal.  Downsample no-flow to at most N * flow count
    # so the model gets meaningful extraction practice.
    _NO_FLOW_RATIO = 1.0  # max no-flow : flow ratio (1.0 = balanced)
    flow_rows = [r for r in rows if r.get("gold_has_exchange") is True]
    no_flow_rows = [r for r in rows if r.get("gold_has_exchange") is not True]
    max_no_flow = max(int(len(flow_rows) * _NO_FLOW_RATIO), 1)
    if len(no_flow_rows) > max_no_flow and flow_rows:
        no_flow_rows = random.sample(no_flow_rows, max_no_flow)
        rows = flow_rows + no_flow_rows
        random.shuffle(rows)
        print(f"[grpo_training] Downsampled no-flow chunks: "
              f"{len(flow_rows)} flow + {max_no_flow} no-flow = {len(rows)} total")
    else:
        print(f"[grpo_training] No downsampling needed: "
              f"{len(flow_rows)} flow, {len(no_flow_rows)} no-flow")

    n_original = len(rows)

    # --- Contrastive copies: sample n% of rows with wrong source_ids ---
    if contrastive_ratio > 0.0 and all_source_ids and len(all_source_ids) > 1:
        n_contrastive = math.ceil(n_original * contrastive_ratio)
        sampled_indices = random.choices(range(n_original), k=n_contrastive)

        for idx in sampled_indices:
            orig = rows[idx]
            real_source = orig["source_id"]
            candidates = [s for s in all_source_ids if s != real_source]
            if not candidates:
                continue
            wrong_source = random.choice(candidates)

            # Add a system message to the contrastive copy so the
            # chat-templated string is distinct from the original.
            # The model sees one extra short system turn; the clean
            # chunk_text in raw_prompts stores the unmodified text.
            orig_user_prompt = raw_prompts[orig["prompt"]]

            formatted_contrastive = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": orig_user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

            # Store clean (un-modified) text for the judge
            raw_prompts[formatted_contrastive] = orig_user_prompt

            rows.append({
                "prompt": formatted_contrastive,
                "source_id": real_source,
                "prompt_id": orig["prompt_id"],
                "is_contrastive": True,
                "contrastive_source": wrong_source,
                "gold_has_exchange": orig.get("gold_has_exchange"),
                "task_type": "ci_extraction",
                "gold_judgment": None,
                "source_norm_articulation": None,
            })

        n_added = len(rows) - n_original
        print(f"[grpo_training] Contrastive copies: {n_added} rows added "
              f"(ratio={contrastive_ratio}, from {n_original} originals)")

    # --- Judgment vignettes: mix in norm-derived privacy judgment tasks ---
    n_vignettes_added = 0
    if vignette_ratio > 0.0 and vignettes:
        n_ci = len(rows)
        # vignette_ratio is the target fraction of the final dataset.
        # ratio=0.5 → equal parts CI and vignettes (n_vignettes = n_ci).
        # ratio=1.0 → capped at len(vignettes) available candidates.
        if vignette_ratio >= 1.0:
            n_vignettes = len(vignettes)
        else:
            n_vignettes = math.ceil(n_ci * vignette_ratio / (1.0 - vignette_ratio))
        sampled = random.sample(vignettes, k=n_vignettes) if n_vignettes <= len(vignettes) \
            else random.choices(vignettes, k=n_vignettes)

        for vig_idx, vig in enumerate(sampled):
            # Append a unique index to the user content so duplicate
            # vignettes (same norm fields) get distinct formatted keys
            # in the prompt_metadata dict.
            user_content = vig["prompt_text"]
            if vig_idx > 0:
                user_content = user_content.rstrip() + f"\n<!-- vig-{vig_idx} -->"

            formatted_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": vignette_system_prompt},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

            prompt_id = hashlib.sha256(user_content.encode("utf-8")).hexdigest()[:16]
            raw_prompts[formatted_prompt] = vig["prompt_text"]

            rows.append({
                "prompt": formatted_prompt,
                "source_id": vig["source_id"],
                "prompt_id": prompt_id,
                "is_contrastive": False,
                "contrastive_source": None,
                "gold_has_exchange": None,
                "task_type": "norm_judgment",
                "gold_judgment": vig["gold_judgment"],
                "source_norm_articulation": vig["source_norm"].get("norm_articulation", ""),
            })
            n_vignettes_added += 1

        print(f"[grpo_training] Judgment vignettes: {n_vignettes_added} added "
              f"(ratio={vignette_ratio}, from {len(vignettes)} candidates)")

    dataset = Dataset.from_list(rows)
    thinking_label = "enabled" if enable_thinking else "disabled"
    n_contrastive = len(dataset) - n_original - n_vignettes_added
    print(f"[grpo_training] Dataset: {len(dataset)} prompts "
          f"({n_original} CI extraction + {max(n_contrastive, 0)} contrastive "
          f"+ {n_vignettes_added} vignettes, thinking={thinking_label})")
    return dataset, raw_prompts


def run_grpo_training_stage(
    sft_checkpoint: str,
    chunks_path: str,
    norm_universes_path: str,
    output_dir: str,
    cfg: Any,
    embeddings_dir: str = "",
    reward_cache_path: str = "",
    vignette_norm_universes_path: str = "",
) -> None:
    """Run GRPO training with TRL + vLLM.

    Args:
        sft_checkpoint: Path to SFT LoRA checkpoint directory.
        chunks_path: Path to chunks parquet (chunk_text + source_id).
        norm_universes_path: Path to norm_universes.json (R_ground grounding +,
            by default, the judgment-vignette source).
        output_dir: Directory to save GRPO checkpoint.
        cfg: Hydra config with training.grpo section.
        embeddings_dir: Path to per-book .npy embeddings (for online R_ground).
        reward_cache_path: Path to reward_cache.parquet (legacy cached R_ground).
        vignette_norm_universes_path: Optional path to a SEPARATE norm_universes
            .json that judgment vignettes are drawn from (grounding/extraction
            keep ``norm_universes_path``). Empty/non-file ⇒ vignettes use the
            grounding universe (historical behaviour). See
            ``_resolve_vignette_universes``.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # Pick a free port for torch distributed to avoid collisions with
    # other training jobs on the same node (default 29500 is often taken).
    if "MASTER_PORT" not in os.environ:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
            _s.bind(("", 0))
            _free_port = str(_s.getsockname()[1])
        os.environ["MASTER_PORT"] = _free_port
        print(f"[grpo_training] Set MASTER_PORT={_free_port}")

    grpo_cfg = OmegaConf.to_container(
        OmegaConf.select(cfg, "training.grpo"), resolve=True
    )

    # Seed all RNGs (Python random, NumPy, torch) up front so that every source
    # of run-to-run variation downstream — the no-flow downsampling + shuffle in
    # _build_grpo_dataset, the data sampler, generation sampling, and model init —
    # is controlled by a single seed. This makes each run reproducible given its
    # seed and is the basis for the seed-variance sweep (sweep/seed_variance.yaml),
    # which holds all hyperparameters fixed and varies only training.grpo.seed.
    from transformers import set_seed as _set_seed
    seed = int(grpo_cfg.get("seed", 42))
    _set_seed(seed)
    print(f"[grpo_training] Seeded all RNGs with seed={seed}")

    # Load chunks
    chunks_df = pd.read_parquet(chunks_path)

    # Resolve text and source columns (ci_reasoning uses article_text/gutenberg_id)
    text_col = None
    for candidate in ("chunk_text", "article_text", "text"):
        if candidate in chunks_df.columns:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError(
            f"[grpo_training] No text column found in {chunks_path}. "
            f"Available: {list(chunks_df.columns)}"
        )
    if text_col != "chunk_text":
        chunks_df = chunks_df.rename(columns={text_col: "chunk_text"})

    source_col = None
    for candidate in ("source_id", "gutenberg_id", "book_id"):
        if candidate in chunks_df.columns:
            source_col = candidate
            break
    if source_col and source_col != "source_id":
        chunks_df["source_id"] = chunks_df[source_col].astype(str)
    elif "source_id" not in chunks_df.columns:
        chunks_df["source_id"] = "unknown"

    # Drop chunks with no text
    chunks_df = chunks_df[chunks_df["chunk_text"].notna()].reset_index(drop=True)

    # Book-level filter: restrict to a single book's chunks
    book_id = OmegaConf.select(cfg, "runtime.book_id", default=None)
    if book_id is not None:
        book_id_str = str(book_id)
        pre = len(chunks_df)
        chunks_df = chunks_df[chunks_df["source_id"] == book_id_str].reset_index(drop=True)
        print(f"[grpo_training] Filtered to book_id={book_id_str}: {len(chunks_df)}/{pre} chunks")

    # Optional: subsample for debug/smoke tests
    sample_n = OmegaConf.select(cfg, "runtime.sample_n", default=None)
    if sample_n is not None and int(sample_n) < len(chunks_df):
        chunks_df = chunks_df.sample(n=int(sample_n), random_state=42).reset_index(drop=True)
        print(f"[grpo_training] Sampled {sample_n} chunks for debug")

    # Load reward cache (legacy cached R_ground)
    if reward_cache_path and os.path.exists(reward_cache_path):
        reward_cache = pd.read_parquet(reward_cache_path)
    else:
        reward_cache = pd.DataFrame()

    # Load norm universes
    norm_universes = {}
    if norm_universes_path and os.path.exists(norm_universes_path):
        with open(norm_universes_path, "r", encoding="utf-8") as f:
            norm_universes = json.load(f)

    # Filter norm universes to single book
    if book_id is not None and norm_universes:
        book_id_str = str(book_id)
        if book_id_str in norm_universes:
            norm_universes = {book_id_str: norm_universes[book_id_str]}
            print(f"[grpo_training] Filtered norm universes to book_id={book_id_str}: "
                  f"{len(norm_universes[book_id_str])} norms")
        else:
            print(f"[grpo_training] WARNING: book_id={book_id_str} not in norm universes "
                  f"(available: {list(norm_universes.keys())[:10]})")

    # Vignette source: defaults to the grounding universe; can be pointed at a
    # separate, more force-balanced corpus (single-variable judgment-balance
    # probe — see _resolve_vignette_universes).
    vignette_norm_universes = _resolve_vignette_universes(
        norm_universes, vignette_norm_universes_path
    )
    _vignette_universe_is_separate = vignette_norm_universes is not norm_universes
    if _vignette_universe_is_separate:
        if book_id is not None:
            book_id_str = str(book_id)
            vignette_norm_universes = (
                {book_id_str: vignette_norm_universes[book_id_str]}
                if book_id_str in vignette_norm_universes else {}
            )
        print(f"[grpo_training] Vignettes drawn from SEPARATE universe: "
              f"{len(vignette_norm_universes)} sources "
              f"({vignette_norm_universes_path})")

    print(f"[grpo_training] Chunks: {len(chunks_df)}")
    print(f"[grpo_training] Norm universes: {len(norm_universes)} sources")

    # reward_weights survives the v9 removal only as the R_ground gate below
    # (weights[5] decides whether online R_ground is worth standing up).
    weights = grpo_cfg.get("reward_weights", [0.2, 0.15, 0.15, 0.15, 0.15, 0.2])
    # Resolve GRPO thinking mode: explicit training-config override wins,
    # else derive from the model's thinking_mode field (single source of truth).
    _etg_override = grpo_cfg.get("enable_thinking_grpo", None)
    if _etg_override is None:
        from dagspaces.common.stage_utils import resolve_thinking_mode
        model_cfg = getattr(cfg, "model", None) or {}
        enable_thinking_grpo = resolve_thinking_mode(model_cfg, default=True)
        print(f"[grpo_training] enable_thinking_grpo not set in training config — "
              f"derived from model.thinking_mode: {enable_thinking_grpo}")
    else:
        enable_thinking_grpo = bool(_etg_override)
        print(f"[grpo_training] enable_thinking_grpo from training config: {enable_thinking_grpo}")

    # Load context embedding model for r_context
    context_embedding_model = None
    context_model_name = grpo_cfg.get("context_embedding_model", "all-MiniLM-L6-v2")
    try:
        from dagspaces.common.stage_utils import ensure_importable_sentence_transformers

        ensure_importable_sentence_transformers()
        from sentence_transformers import SentenceTransformer
        context_embedding_model = SentenceTransformer(context_model_name)
        print(f"[grpo_training] Loaded context embedding model: {context_model_name}")
    except Exception as e:
        print(f"[grpo_training] Warning: could not load embedding model: {e}")

    # Build source context lookup from norm universes.
    # Each source maps to its list of unique norm-level context strings,
    # so R_context can do per-flow max-similarity matching instead of
    # comparing against one giant concatenated string.
    source_contexts: dict[str, list[str]] = {}
    for source_id, norms in norm_universes.items():
        contexts = set()
        for norm in norms:
            ctx = norm.get("context")
            if ctx:
                contexts.add(str(ctx))
        source_contexts[source_id] = sorted(contexts) if contexts else []

    # Trace logging: log detailed reward breakdowns on every call.
    # Each trace logs up to 4 completions, so file size stays manageable.
    trace_log_path = os.path.join(output_dir, "reward_traces.jsonl")
    trace_every = 1

    # Online R_ground: use embedding + judge servers instead of cached lookup
    online_rground = None
    use_online_rground = grpo_cfg.get("online_rground", False) and weights[5] > 0.0
    _contrastive = grpo_cfg.get("contrastive_ratio", 0.0)
    # Contrastive pairing works with both online and cached R_ground.
    # Contrastive rows are added as new dataset entries (with a trailing
    # newline to make the formatted prompt key unique).  OnlineRGround
    # retrieves norms from the wrong source for contrastive completions,
    # producing naturally low R_ground.
    if use_online_rground:
        from ..schemas import (
            CompletionRankingJudgment,
            FlowGovernanceJudgment,
            NoFlowCoverageJudgment,
        )
        from .clients import (
            EmbeddingClient,
            JudgeClient,
            NormRetriever,
            RerankerJudgeClient,
        )
        from .online_rground import OnlineRGround

        emb_port = grpo_cfg.get("embedding_server_port", 8001)
        judge_port = grpo_cfg.get("judge_server_port", 8002)

        # Resolve server URLs: config field → env var → localhost:port.
        # The runner sets GRPO_*_SERVER_URL for both managed and external
        # modes; the localhost fallback handles legacy/direct invocations.
        embedding_url = (
            str(grpo_cfg.get("embedding_server_url") or "")
            or os.environ.get("GRPO_EMBEDDING_SERVER_URL", "")
            or f"http://localhost:{emb_port}"
        )
        judge_url = (
            str(grpo_cfg.get("judge_server_url") or "")
            or os.environ.get("GRPO_JUDGE_SERVER_URL", "")
            or f"http://localhost:{judge_port}"
        )

        # Model names for vLLM API (must match the path used to launch servers)
        emb_model_name = str(
            OmegaConf.select(cfg, "embedding_model.model_source", default=None)
            or OmegaConf.select(cfg, "model.embedding_model_source", default=None)
            or ""
        )
        judge_model_name = str(
            OmegaConf.select(cfg, "judge_model.model_source") or ""
        )

        # Load judge prompt templates
        prompt_cfg = (
            OmegaConf.select(cfg, "prompt_reward_judge")
            or OmegaConf.select(cfg, "prompt")
        )
        system_prompt = str(OmegaConf.select(prompt_cfg, "system_prompt") or "")
        prompt_template = str(OmegaConf.select(prompt_cfg, "prompt_template") or "")

        # No-flow coverage judge prompt
        nf_prompt_cfg = OmegaConf.select(cfg, "prompt_no_flow_judge")
        nf_system_prompt = str(OmegaConf.select(nf_prompt_cfg, "system_prompt") or "") if nf_prompt_cfg else ""
        nf_prompt_template = str(OmegaConf.select(nf_prompt_cfg, "prompt_template") or "") if nf_prompt_cfg else ""

        # Listwise ranking judge prompt (rground_scoring="ranked")
        rk_prompt_cfg = OmegaConf.select(cfg, "prompt_reward_judge_ranking")
        rk_system_prompt = str(OmegaConf.select(rk_prompt_cfg, "system_prompt") or "") if rk_prompt_cfg else ""
        rk_prompt_template = str(OmegaConf.select(rk_prompt_cfg, "prompt_template") or "") if rk_prompt_cfg else ""

        embedding_client = EmbeddingClient(
            base_url=embedding_url,
            model_name=emb_model_name,
        )

        # Judge backend: the generative LLM judge (default) or a cross-encoder
        # reranker (Qwen3-Reranker) that scores grounding ~10x cheaper. The
        # reranker is duck-typed to JudgeClient, so OnlineRGround is unchanged;
        # it covers norm_match/governance but folds appropriateness into a
        # single relevance ordering (see RerankerJudgeClient docstring).
        _judge_backend = str(grpo_cfg.get("rground_judge_backend", "llm")).lower()
        _judge_workers = int(grpo_cfg.get("judge_max_workers", 16))
        if _judge_backend == "reranker":
            rr_port = grpo_cfg.get("reranker_server_port", 8003)
            reranker_url = (
                str(grpo_cfg.get("reranker_server_url") or "")
                or os.environ.get("GRPO_RERANKER_SERVER_URL", "")
                or f"http://localhost:{rr_port}"
            )
            reranker_model_name = str(
                OmegaConf.select(cfg, "reranker_model.model_source")
                or grpo_cfg.get("reranker_model_name")
                or ""
            )
            _rr_instruction = str(grpo_cfg.get("reranker_instruction") or "").strip()
            _rr_app_weight = float(grpo_cfg.get("reranker_app_weight", 0.2))
            _rr_kwargs = {
                "base_url": reranker_url,
                "model_name": reranker_model_name,
                "max_workers": _judge_workers,
                "app_weight": _rr_app_weight,
            }
            if _rr_instruction:
                _rr_kwargs["instruction"] = _rr_instruction
            judge_client = RerankerJudgeClient(**_rr_kwargs)
            print(f"[grpo_training] R_ground judge backend=reranker "
                  f"(url={reranker_url}, model={reranker_model_name or '<default>'}, "
                  f"app_weight={_rr_app_weight})")
        elif _judge_backend == "llm":
            judge_client = JudgeClient(
                base_url=judge_url,
                model_name=judge_model_name,
                system_prompt=system_prompt,
                prompt_template=prompt_template,
                json_schema=FlowGovernanceJudgment.model_json_schema(),
                # vLLM batches concurrent requests; the prescreen pass issues
                # thousands of ranking calls, so low concurrency dominates
                # wall-clock (4 workers × ~2200 calls was a >12h pass).
                max_workers=_judge_workers,
            )
        else:
            raise ValueError(
                f"[grpo_training] unknown rground_judge_backend={_judge_backend!r} "
                f"(expected 'llm' or 'reranker')"
            )

        norm_retriever = NormRetriever(
            norm_universes=norm_universes,
            embeddings_dir=embeddings_dir or "",
            embedding_client=embedding_client,
        )

        _contrastive_lambda = float(grpo_cfg.get("contrastive_lambda", 0.5))
        _rground_scoring = str(grpo_cfg.get("rground_scoring", "absolute"))
        # Deontic appropriateness-consistency blend (default 0 = legacy/grounding
        # only). For the LLM-judge ranked path this mirrors the reranker backend's
        # reranker_app_weight: blend the deterministic norm-force→appropriateness
        # check into R_ground so the reward rewards context-relative judgments,
        # not just topical grounding.
        _rground_app_weight = float(grpo_cfg.get("rground_app_weight", 0.0))
        # v9: app_mode="multiplicative" turns the appropriateness check into a
        # two-sided direction multiplier on R_ground (floored at app_floor),
        # instead of the legacy additive blend. See deontic.direction_multiplier.
        _rground_app_mode = str(grpo_cfg.get("rground_app_mode", "additive"))
        _rground_app_floor = float(grpo_cfg.get("rground_app_floor", 0.4))
        # v10: cost-sensitive floor for a false-permit (a prohibited-governed flow
        # called "appropriate"). None/absent = symmetric v9 behaviour.
        _rgafp = grpo_cfg.get("rground_app_floor_prohibit", None)
        _rground_app_floor_prohibit = float(_rgafp) if _rgafp is not None else None
        # v12a: cost-sensitive tier for a hedge on a prohibited-governed flow
        # (drops it below the neutral 0.7). None/absent = v10 behaviour.
        _rgahp = grpo_cfg.get("rground_app_hedge_prohibit", None)
        _rground_app_hedge_prohibit = float(_rgahp) if _rgahp is not None else None
        if _rground_scoring == "ranked" and not rk_prompt_template:
            raise ValueError(
                "[grpo_training] rground_scoring='ranked' requires the "
                "prompt_reward_judge_ranking config (prompt/reward_judge_ranking.yaml)"
            )
        online_rground = OnlineRGround(
            embedding_client=embedding_client,
            judge_client=judge_client,
            norm_retriever=norm_retriever,
            all_source_ids=list(norm_universes.keys()),
            contrastive_lambda=_contrastive_lambda,
            no_flow_judge_system_prompt=nf_system_prompt,
            no_flow_judge_prompt_template=nf_prompt_template,
            no_flow_judge_json_schema=NoFlowCoverageJudgment.model_json_schema(),
            scoring_mode=_rground_scoring,
            ranking_system_prompt=rk_system_prompt,
            ranking_prompt_template=rk_prompt_template,
            ranking_json_schema=CompletionRankingJudgment.model_json_schema(),
            rank_top_k=int(grpo_cfg.get("rank_top_k", 5)),
            rank_weight=float(grpo_cfg.get("rank_weight", 0.5)),
            app_weight=_rground_app_weight,
            app_mode=_rground_app_mode,
            app_floor=_rground_app_floor,
            app_floor_prohibit=_rground_app_floor_prohibit,
            app_hedge_prohibit=_rground_app_hedge_prohibit,
        )
        print(f"[grpo_training] Online R_ground enabled "
              f"(embed={embedding_url}, judge={judge_url}, "
              f"scoring={_rground_scoring}, "
              f"contrastive_lambda={_contrastive_lambda}, "
              f"app_weight={_rground_app_weight}, "
              f"app_mode={_rground_app_mode}, app_floor={_rground_app_floor}, "
              f"app_floor_prohibit={_rground_app_floor_prohibit}, "
              f"app_hedge_prohibit={_rground_app_hedge_prohibit})")
    elif not use_online_rground and weights[5] > 0.0:
        print(f"[grpo_training] R_ground using cached lookup "
              f"({len(reward_cache)} entries)")

    _nf_scoring = grpo_cfg.get("no_flow_scoring", "independent")
    _composition = str(grpo_cfg.get("reward_composition", "additive"))
    _judgment_weights = list(grpo_cfg.get("judgment_reward_weights", [0.5, 0.25, 0.25]))
    _abstention_penalty = float(grpo_cfg.get("abstention_penalty", 0.0))
    # Facet-3 confidence resolution in r_uncert. False (default) reproduces the
    # v9-ckpt100 keeper checkpoint; True is the corrected fall-through for the
    # per-component ablation. Enabling it changes the composite reward value.
    _confidence_fallthrough = bool(grpo_cfg.get("confidence_fallthrough", False))
    # The m-series ModularReward stack is the only reward path. The v9
    # "directional" composition and its "additive"/"gated" siblings were
    # removed with CompositeRewardFunction once the keeper was deprecated
    # (CLAUDE.md); anything other than "modular" is now a config error rather
    # than a silent fall-through to a retired reward.
    from .modular_reward import is_modular_composition
    if not is_modular_composition(_composition):
        raise ValueError(
            f"reward_composition={_composition!r} is no longer supported. The v9 "
            f"directional/additive/gated compositions were removed along with "
            f"CompositeRewardFunction; use reward_composition: modular."
        )
    from .modular_reward import make_modular_reward_from_cfg
    reward_fn = make_modular_reward_from_cfg(
        cfg, grpo_cfg, norm_universes,
        # Until 2026-07-25 traces were passed only to the legacy branch
        # while the print below fired for both — the m1 wave produced no
        # traces despite announcing them.
        trace_log_path=trace_log_path,
        trace_every_n_calls=trace_every,
    )
    print(f"[grpo_training] Modular reward stack: "
          f"auxiliaries={reward_fn.auxiliaries}, reward_core={reward_fn.reward_core}, "
          f"weights={reward_fn.weights}")
    reward_fn.enable_thinking_grpo = enable_thinking_grpo
    print(f"[grpo_training] Reward traces → {trace_log_path} (every {trace_every} calls)")

    # Pre-merge LoRA into the base model and save to a temp directory.
    # TRL's vLLM weight sync (sync_weights) doesn't reliably apply LoRA
    # for Qwen3 + vLLM 0.17, so we give vLLM the fully-merged checkpoint.
    # The trainer still uses LoRA for memory-efficient training.
    from dagspaces.common.model_registry import resolve_model_source
    base_model_path = resolve_model_source(
        str(OmegaConf.select(cfg, "model.model_source")),
        stage_name="grpo_training",
    )
    print(f"[grpo_training] Merging LoRA into base model for vLLM...")
    print(f"[grpo_training]   base: {base_model_path}")
    print(f"[grpo_training]   adapter: {sft_checkpoint}")

    # Load the FULL multimodal model (ConditionalGeneration) for merging,
    # not just CausalLM.  This preserves vision encoder weights in the saved
    # checkpoint so vLLM can load the complete multimodal architecture.
    # LoRA only touches language model layers; vision weights pass through.
    from transformers import AutoConfig as _MergeAC
    _merge_cfg = _MergeAC.from_pretrained(base_model_path, trust_remote_code=True)
    _is_multimodal_merge = hasattr(_merge_cfg, "vision_config") and _merge_cfg.vision_config is not None
    if _is_multimodal_merge:
        # Use the model's own ConditionalGeneration class to preserve vision weights
        _model_class = _merge_cfg.architectures[0] if _merge_cfg.architectures else None
        if _model_class:
            import transformers as _tf
            _cls = getattr(_tf, _model_class, None)
            if _cls is None:
                _cls = AutoModelForCausalLM
            _base = _cls.from_pretrained(
                base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            )
        else:
            _base = AutoModelForCausalLM.from_pretrained(
                base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            )
        print(f"[grpo_training] Loaded full multimodal model for merge: {type(_base).__name__}")
    else:
        _base = AutoModelForCausalLM.from_pretrained(
            base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        )

    # Remap LoRA adapter keys if needed: SFT adapters trained via
    # AutoModelForCausalLM have keys like model.layers.X, but VLM
    # architectures (Qwen3_5ForConditionalGeneration) expect
    # model.language_model.layers.X.  Without remapping, PeftModel
    # silently skips all adapter weights.
    if _is_multimodal_merge:
        from dagspaces.common.vllm_inference import _remap_lora_keys_for_vlm
        _adapter_path = _remap_lora_keys_for_vlm(
            sft_checkpoint, base_model_path, "grpo_training",
        )
    else:
        _adapter_path = sft_checkpoint

    _peft = PeftModel.from_pretrained(_base, _adapter_path)
    _merged = _peft.merge_and_unload()

    # Save merged model to both NFS (persistence) and /scratch (fast vLLM loads).
    # vLLM reloads weights from disk every time it wakes from sleep mode —
    # /scratch is local SSD, much faster than NFS for repeated reads.
    #
    # Rank-safety (TP>1 colocate via accelerate launch): every rank runs this
    # stage, so a SHARED merged-model path (both NFS `_merged_sft` and the
    # SLURM_JOB_ID-keyed scratch dir are identical across ranks) would have the
    # ranks concurrently write the same files → corruption. When WORLD_SIZE>1
    # each rank merges independently to its OWN rank-local path (barrier-free:
    # the torch PG isn't up yet before GRPOTrainer, so we avoid a rank-0-only +
    # barrier scheme). Costs one extra CPU merge per rank (~40s), no sync. The
    # single-process keeper (WORLD_SIZE unset ⇒ "1") uses the original paths
    # byte-identically.
    _rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    _world = int(os.environ.get("WORLD_SIZE", "1") or "1")
    _rank_suffix = f"_rank{_rank}" if _world > 1 else ""
    merged_dir_nfs = os.path.join(output_dir, f"_merged_sft{_rank_suffix}")
    os.makedirs(merged_dir_nfs, exist_ok=True)
    _merged.save_pretrained(merged_dir_nfs)
    # Copy multimodal processor files that save_pretrained doesn't include
    # (preprocessor_config.json is from the processor, not the model)
    import shutil as _shutil
    for _proc_file in ("preprocessor_config.json", "video_preprocessor_config.json",
                        "processor_config.json", "chat_template.json"):
        _src = os.path.join(base_model_path, _proc_file)
        if os.path.exists(_src) and not os.path.exists(os.path.join(merged_dir_nfs, _proc_file)):
            _shutil.copy2(_src, merged_dir_nfs)
    print(f"[grpo_training] Saved merged model to {merged_dir_nfs}")

    scratch_base = os.environ.get("TMPDIR", "/tmp")
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    merged_dir_scratch = os.path.join(
        scratch_base, f"grpo_merged_sft_{job_id}{_rank_suffix}")
    try:
        import shutil
        if os.path.exists(merged_dir_scratch):
            shutil.rmtree(merged_dir_scratch)
        shutil.copytree(merged_dir_nfs, merged_dir_scratch)
        merged_dir = merged_dir_scratch
        print(f"[grpo_training] Copied merged model to scratch: {merged_dir}")
    except Exception as e:
        merged_dir = merged_dir_nfs
        print(f"[grpo_training] Scratch copy failed ({e}), using NFS: {merged_dir}")

    # Free everything before reloading
    del _base, _peft, _merged
    import gc; gc.collect(); torch.cuda.empty_cache()

    # Reload as PeftModel (LoRA) on CPU for memory-efficient training.
    # model.name_or_path will point to merged_dir (scratch if available)
    # so vLLM loads from fast local disk on every sleep/wake cycle.
    base_model = AutoModelForCausalLM.from_pretrained(
        merged_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    from peft import LoraConfig as _LoraConfig
    from peft import get_peft_model
    # Build a fresh LoraConfig matching the SFT adapter's architecture.
    # LoraConfig.from_pretrained() marks the adapter as inference-only,
    # resulting in zero trainable params.
    _sft_cfg = _LoraConfig.from_pretrained(sft_checkpoint)
    lora_config = _LoraConfig(
        r=_sft_cfg.r,
        lora_alpha=_sft_cfg.lora_alpha,
        target_modules=list(_sft_cfg.target_modules),
        task_type=_sft_cfg.task_type,
        lora_dropout=_sft_cfg.lora_dropout,
    )
    model = get_peft_model(base_model, lora_config)
    print(f"[grpo_training] Re-wrapped with LoRA (trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,})")

    # Tokenizer — save to merged dir so vLLM finds it via model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(merged_dir)
    # Also save to NFS if we're using scratch
    if merged_dir != merged_dir_nfs:
        tokenizer.save_pretrained(merged_dir_nfs)

    # Build dataset (needs tokenizer for chat template pre-formatting)
    # (enable_thinking_grpo already set above, near reward function init)

    # Load CI extraction prompt template from config
    prompt_cfg = OmegaConf.select(cfg, "prompt_ci_extraction")
    if prompt_cfg:
        ci_instruction = str(OmegaConf.select(prompt_cfg, "instruction") or "")
        ci_prompt_template = str(OmegaConf.select(prompt_cfg, "prompt_template") or "")
        # Substitute instruction into template if it uses {{instruction}}
        ci_prompt_template = ci_prompt_template.replace("{{instruction}}", ci_instruction.strip())
    else:
        # Fallback: bare instruction + chunk
        from .sft_data_prep import _CI_INSTRUCTION
        ci_prompt_template = _CI_INSTRUCTION + "\n\n{{chunk_text}}"

    # m-series: rollout prompts must match the SFT training distribution.
    # The m1 wave sampled the policy on the config prompt above — an
    # instruction the SFT adapter never saw — and paid 34.4% vs 2.7% R-VALID
    # gate failure for it (A/B probe 2026-07-28, grpo_redesign wiki R1).
    # Byte-parity comes from importing the same builder SFT data prep uses,
    # never from a config copy that can drift. Keeper (non-modular) runs keep
    # the config prompt — their v9-lineage rollouts were trained on it.
    if _modular and bool(grpo_cfg.get("sft_aligned_extract_prompt", True)):
        from .sft_data_prep import sft_aligned_extract_template
        ci_prompt_template = sft_aligned_extract_template(cfg)
        print("[grpo_training] extract prompt: SFT-aligned template "
              "(grpo.sft_aligned_extract_prompt=true) — config "
              "prompt_ci_extraction is NOT used for rollouts")

    all_source_ids = list(norm_universes.keys())

    # Generate judgment vignettes from norm universes if ratio > 0
    _vignette_ratio = float(grpo_cfg.get("vignette_ratio", 0.0))
    vignettes = []
    if _vignette_ratio > 0.0:
        # A SEPARATE vignette universe that resolves to zero sources (missing
        # book key, empty corpus) would silently train with no norm_judgment
        # prompts while config still claims vignette_ratio > 0 — corrupting the
        # training mix. Fail loud on this config/data mismatch.
        if _vignette_universe_is_separate and not vignette_norm_universes:
            raise ValueError(
                f"Separate vignette universe ({vignette_norm_universes_path!r}) "
                f"resolved to 0 sources"
                + (f" for book_id={book_id}" if book_id is not None else "")
                + f", but vignette_ratio={_vignette_ratio} > 0. This would train "
                "with zero judgment vignettes. Fix the universe/book mapping or "
                "set vignette_ratio=0."
            )
        vig_prompt_cfg = OmegaConf.select(cfg, "prompt_norm_judgment")
        if vig_prompt_cfg:
            vig_sys = str(OmegaConf.select(vig_prompt_cfg, "system_prompt") or "")
            vig_tmpl = str(OmegaConf.select(vig_prompt_cfg, "prompt_template") or "")
        else:
            vig_sys = ""
            vig_tmpl = ""
        if vig_tmpl:
            vignettes = _generate_vignettes(vignette_norm_universes, vig_tmpl)
            print(f"[grpo_training] Generated {len(vignettes)} judgment vignettes "
                  f"from {sum(1 for n in sum(vignette_norm_universes.values(), []) if n.get('governs_info_flow'))} "
                  f"info-flow norms")

    # Contrastive signal is now per-completion dual scoring inside
    # OnlineRGround, so no additive contrastive rows are needed.
    # Legacy contrastive_ratio kept for backward compat but defaults to 0.
    if _modular:
        # m-series dataset-build hook (migration.md items 1–2, 6): T-EXTRACT
        # rows carry chunk_id-sampled probes, T-VIGNETTE rows are deontic
        # batteries; stratified prescreen + realized-mix → training_metadata.json.
        # The probe-bearing metadata is set on reward_fn in place.
        from .modular_reward import build_modular_dataset
        _embed_fn = None
        if context_embedding_model is not None:
            _embed_fn = lambda _t: context_embedding_model.encode(  # noqa: E731
                _t, normalize_embeddings=True
            )
        dataset, _modular_meta = build_modular_dataset(
            cfg, grpo_cfg, chunks_df, norm_universes, reward_fn,
            tokenizer, ci_prompt_template,
            output_dir=output_dir, seed=seed, embed_fn=_embed_fn,
            enable_thinking=enable_thinking_grpo,
        )
        # Downstream locals the shared metadata-write path expects.
        raw_prompts = {}
        n_contrastive = 0
        n_vignette_meta = sum(
            1 for m in reward_fn.prompt_metadata.values()
            if m.get("task_type") == "vignette"
        )
        _vig_pre = {"yes": 0, "no": 0}
        _n_screened_out = int(
            _modular_meta.get("prescreen_report", {}).get("n_dropped", 0)
        )
        n_gold_pos = sum(1 for m in reward_fn.prompt_metadata.values() if m.get("gold_has_exchange") is True)
        n_gold_neg = sum(1 for m in reward_fn.prompt_metadata.values() if m.get("gold_has_exchange") is False)
        print(f"[grpo_training] Modular dataset: {len(dataset)} rows "
              f"({n_vignette_meta} vignettes, {n_gold_pos} gold-yes / {n_gold_neg} gold-no extract)")
    else:
        dataset, raw_prompts = _build_grpo_dataset(
            chunks_df, tokenizer, ci_prompt_template, enable_thinking_grpo,
            contrastive_ratio=_contrastive,
            all_source_ids=all_source_ids,
            vignettes=vignettes,
            vignette_ratio=_vignette_ratio,
            vignette_system_prompt=vig_sys if _vignette_ratio > 0.0 and vignettes else "",
        )

        # Build prompt→metadata lookup so the reward function can access
        # source_id/prompt_id without relying on TRL forwarding dataset columns.
        # chunk_text stores the raw user prompt (pre-template) so OnlineRGround
        # can pass clean text to the judge instead of chat-templated text.
        #
        # Contrastive rows are already present in the dataset with distinct
        # formatted prompts (trailing newline in user message), so each row
        # maps to a unique metadata entry.
        reward_fn.prompt_metadata = {}
        n_contrastive = 0
        for row in dataset:
            key = row["prompt"]
            if key not in reward_fn.prompt_metadata:
                is_contrastive = row.get("is_contrastive", False)
                if is_contrastive:
                    n_contrastive += 1
                reward_fn.prompt_metadata[key] = {
                    "source_id": row.get("source_id", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "is_contrastive": is_contrastive,
                    "contrastive_source": row.get("contrastive_source"),
                    "chunk_text": raw_prompts.get(key, ""),
                    "gold_has_exchange": row.get("gold_has_exchange"),
                    "task_type": row.get("task_type", "ci_extraction"),
                    "gold_judgment": row.get("gold_judgment"),
                    "source_norm_articulation": row.get("source_norm_articulation"),
                }
        n_gold_pos = sum(1 for m in reward_fn.prompt_metadata.values() if m.get("gold_has_exchange") is True)
        n_gold_neg = sum(1 for m in reward_fn.prompt_metadata.values() if m.get("gold_has_exchange") is False)
        n_gold_unk = sum(1 for m in reward_fn.prompt_metadata.values() if m.get("gold_has_exchange") is None)
        n_vignette_meta = sum(
            1 for m in reward_fn.prompt_metadata.values()
            if m.get("task_type") in ("norm_judgment", "vignette")
        )
        # Realised vignette force mix BEFORE screening. The v11 steering variable —
        # sampling from the candidate pool is uniform/force-blind, so the realised
        # ratio must be measured, not inferred from the pool's ratio.
        from .prompt_screening import _vignette_gold_counts
        _vig_pre = _vignette_gold_counts(reward_fn.prompt_metadata.values())
        print(f"[grpo_training] Reward prompt metadata: {len(reward_fn.prompt_metadata)} entries "
              f"({n_contrastive} contrastive, {n_vignette_meta} vignettes)")
        if n_vignette_meta:
            print(f"[grpo_training] Vignette gold mix (pre-screen): "
                  f"{_vig_pre['yes']} yes : {_vig_pre['no']} no "
                  f"({_vig_pre['yes'] / max(_vig_pre['no'], 1):.2f}:1)")
        print(f"[grpo_training] Gold labels: {n_gold_pos} has_exchange=True, "
              f"{n_gold_neg} has_exchange=False, {n_gold_unk} unknown")

        # --- Variance pre-screening (Phase 2) ---
        # Sample G completions per prompt from the merged SFT policy and drop
        # prompts whose group reward std is ~0 — they produce zero-advantage
        # GRPO groups and only burn generation + judge throughput. Runs before
        # TRL's colocated vLLM engine exists; its own engine is torn down inside.
        # reward_fn.prompt_metadata is already populated, so scoring matches
        # training exactly.
        from .prompt_screening import prescreen_dataset
        _n_pre_screen = len(dataset)
        dataset = prescreen_dataset(
            dataset, reward_fn, merged_dir, grpo_cfg, output_dir,
            cache_identity=sft_checkpoint,
            composite_config_path=base_model_path,
        )
        _n_screened_out = _n_pre_screen - len(dataset)

    # --- Held-out dev split (Phase 5a) ---
    # TRL evaluates reward on eval_dataset (generation + reward pass every
    # eval_steps), giving a held-out reward curve — the first-line signal
    # for "is GRPO learning anything that generalizes".
    _dev_fraction = float(grpo_cfg.get("dev_fraction", 0.0))
    eval_dataset = None
    if _dev_fraction > 0.0 and len(dataset) >= 20:
        _split = dataset.train_test_split(test_size=_dev_fraction, seed=seed)
        dataset, eval_dataset = _split["train"], _split["test"]
        print(f"[grpo_training] Dev split: {len(dataset)} train / "
              f"{len(eval_dataset)} dev prompts (dev_fraction={_dev_fraction})")
    elif _dev_fraction > 0.0:
        print(f"[grpo_training] Skipping dev split: only {len(dataset)} "
              f"prompts after screening")

    # Gold-label stats of the FINAL training set (post-screen, post-split) —
    # the promotion gates compare the policy's no-flow rate against this
    # base rate, so it must describe what the model actually trains on.
    _train_gold_pos = sum(1 for r in dataset if r.get("gold_has_exchange") is True)
    _train_gold_neg = sum(1 for r in dataset if r.get("gold_has_exchange") is False)
    # Vignette accounting (2026-06-09 review, F6): screening can strip
    # vignettes disproportionately (their judgment rewards often have
    # degenerate variance under the SFT policy), so the configured
    # vignette_ratio describes the PRE-screen mix only. Record the realized
    # post-screen count so the paper's mix claim is auditable.
    _train_vignettes = sum(
        1 for r in dataset if r.get("task_type") in ("norm_judgment", "vignette")
    )
    # Imported here (not only in the legacy dataset-build branch above) so the
    # shared post-screen accounting is reachable on the modular path too — the
    # modular branch never runs the legacy import. Idempotent duplicate import.
    from .prompt_screening import _vignette_gold_counts
    _vig_post = _vignette_gold_counts(dataset)
    if n_vignette_meta:
        print(f"[grpo_training] Vignettes in final training set: "
              f"{_train_vignettes}/{len(dataset)} "
              f"({_train_vignettes / max(len(dataset), 1):.1%}; "
              f"pre-screen {n_vignette_meta}, configured ratio {_vignette_ratio})")
        print(f"[grpo_training] Vignette gold mix (post-screen): "
              f"{_vig_post['yes']} yes : {_vig_post['no']} no "
              f"({_vig_post['yes'] / max(_vig_post['no'], 1):.2f}:1)")

    # vLLM mode configuration
    vllm_mode = grpo_cfg.get("vllm_mode", "colocate")
    use_vllm = grpo_cfg.get("use_vllm", True)

    # Prompts are pre-formatted as raw text in _build_grpo_dataset to match
    # the exact SFT training format.  No chat_template_kwargs needed — TRL
    # will route through vLLM's llm.generate() (raw text) not llm.chat().

    grpo_config_kwargs = dict(
        output_dir=output_dir,
        num_generations=grpo_cfg.get("num_generations", 8),
        per_device_train_batch_size=grpo_cfg.get("per_device_batch_size", 2),
        gradient_accumulation_steps=grpo_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=grpo_cfg.get("learning_rate", 1e-6),
        num_train_epochs=grpo_cfg.get("num_epochs", 1),
        max_completion_length=grpo_cfg.get("max_completion_length", 4096),
        gradient_checkpointing=grpo_cfg.get("gradient_checkpointing", True),
        bf16=grpo_cfg.get("bf16", True),
        logging_steps=grpo_cfg.get("logging_steps", 5),
        save_strategy=grpo_cfg.get("save_strategy", "steps"),
        save_steps=grpo_cfg.get("save_steps", 200),
        use_vllm=use_vllm,
        seed=seed,
        data_seed=seed,
        report_to="wandb" if OmegaConf.select(cfg, "wandb.enabled") else "none",
    )

    # Optimizer/objective knobs that TRL otherwise defaults silently
    # (beta=0.0 → no KL anchor, scale_rewards="group" → std-scaled
    # advantages, mask_truncated_completions=False, num_iterations=1,
    # epsilon_high=epsilon=0.2 → symmetric clip,
    # vllm_importance_sampling_mode="sequence_mask" → zeroes the WHOLE
    # completion's gradient when its summed logp-mismatch exceeds the cap,
    # which length-biases against long completions; "token_truncate" clamps
    # per-token instead). Only forwarded when set in the training config so
    # configs that omit them keep TRL defaults.
    for _knob in ("beta", "scale_rewards", "mask_truncated_completions",
                  "num_iterations", "epsilon_high",
                  "vllm_importance_sampling_mode", "vllm_importance_sampling_cap"):
        _val = grpo_cfg.get(_knob)
        if _val is not None:
            grpo_config_kwargs[_knob] = _val

    # Held-out reward evaluation on the dev split. The global eval batch
    # must be divisible by num_generations (TRL constraint), so use exactly
    # one group per eval batch.
    if eval_dataset is not None:
        grpo_config_kwargs["eval_strategy"] = "steps"
        grpo_config_kwargs["eval_steps"] = int(grpo_cfg.get("eval_steps", 50))
        grpo_config_kwargs["per_device_eval_batch_size"] = grpo_config_kwargs["num_generations"]

    # Optional overrides: max_steps / warmup_steps take precedence over ratio
    max_steps = grpo_cfg.get("max_steps")
    if max_steps is not None:
        grpo_config_kwargs["max_steps"] = int(max_steps)
    warmup_steps = grpo_cfg.get("warmup_steps")
    if warmup_steps is not None:
        grpo_config_kwargs["warmup_steps"] = int(warmup_steps)
    else:
        grpo_config_kwargs["warmup_ratio"] = grpo_cfg.get("warmup_ratio", 0.1)

    # LR schedule (v6 2026-06-19): TRL/HF default the GRPO scheduler to a cosine
    # that decays to ~0 by end-of-run — the v5 trace showed the effective lr ≈ 0
    # over the back half, so the (correct) gold-flow advantage was never followed.
    # "cosine_with_min_lr" floors the schedule at min_lr_rate * peak_lr instead of
    # zero, keeping the update alive across all epochs. Only forwarded when set.
    lr_scheduler_type = grpo_cfg.get("lr_scheduler_type")
    if lr_scheduler_type is not None:
        grpo_config_kwargs["lr_scheduler_type"] = str(lr_scheduler_type)
        min_lr_rate = grpo_cfg.get("min_lr_rate")
        if min_lr_rate is not None:
            grpo_config_kwargs["lr_scheduler_kwargs"] = {"min_lr_rate": float(min_lr_rate)}

    if use_vllm:
        grpo_config_kwargs["vllm_mode"] = vllm_mode

        if vllm_mode == "colocate":
            # Colocate: vLLM shares GPUs with training process.
            # Lower memory utilization to leave room for training.
            grpo_config_kwargs["vllm_gpu_memory_utilization"] = grpo_cfg.get(
                "vllm_gpu_memory_utilization", 0.3
            )
            # Optional: offload vLLM weights during optimizer step
            grpo_config_kwargs["vllm_enable_sleep_mode"] = grpo_cfg.get(
                "vllm_enable_sleep_mode", True
            )
            tp = grpo_cfg.get("vllm_tensor_parallel_size")
            if tp:
                grpo_config_kwargs["vllm_tensor_parallel_size"] = tp
            max_len = grpo_cfg.get("vllm_max_model_length")
            if max_len:
                grpo_config_kwargs["vllm_max_model_length"] = max_len

            print(f"[grpo_training] vLLM colocate mode: "
                  f"gpu_mem={grpo_config_kwargs['vllm_gpu_memory_utilization']}, "
                  f"sleep_mode={grpo_config_kwargs['vllm_enable_sleep_mode']}")

        elif vllm_mode == "server":
            # Server: separate vLLM process on dedicated GPUs.
            # The runner must have started `trl vllm-serve` beforehand.
            grpo_config_kwargs["vllm_server_host"] = grpo_cfg.get(
                "vllm_server_host", "0.0.0.0"
            )
            grpo_config_kwargs["vllm_server_port"] = grpo_cfg.get(
                "vllm_server_port", 8000
            )
            grpo_config_kwargs["vllm_server_timeout"] = grpo_cfg.get(
                "vllm_server_timeout", 240.0
            )
            print(f"[grpo_training] vLLM server mode: "
                  f"host={grpo_config_kwargs['vllm_server_host']}:"
                  f"{grpo_config_kwargs['vllm_server_port']}")

    training_args = GRPOConfig(**grpo_config_kwargs)

    # libcudart-stub guard (colocate vLLM init). fla (pulled in transitively when
    # transformers loads the qwen3.5 gated-delta-rule path during the LoRA merge)
    # imports tilelang, which dlopens a `libcudart_stub.so`. vLLM's cumem
    # allocator availability check calls find_loaded_library("libcudart"), whose
    # substring scan of /proc/self/maps matches the stub first (filename
    # "libcudart_stub" satisfies the startswith("libcudart") assert) and loads a
    # runtime missing cudaDeviceReset → AttributeError, killing engine init.
    # Fix (additive, colocate-only): preload the REAL cuda runtime so it is
    # resident in maps, and wrap find_loaded_library to skip any "_stub" shadow.
    # No-op for models that never import tilelang (e.g. gemma-4). FLA_TILELANG=0
    # already disables tilelang *kernels*; this closes the *import-time* dlopen.
    if bool(grpo_cfg.get("use_vllm", True)):
        try:
            import ctypes as _ctypes
            _ctypes.CDLL("libcudart.so.12", mode=_ctypes.RTLD_GLOBAL)
        except Exception as _e:
            print(f"[grpo_training] libcudart preload skipped: {_e}")
        try:
            from vllm.utils import system_utils as _vsu

            _orig_find_loaded_library = _vsu.find_loaded_library

            def _find_loaded_library_skip_stub(lib_name: str):
                try:
                    with open("/proc/self/maps") as _f:
                        for _line in _f:
                            if lib_name in _line and "_stub" not in _line:
                                _start = _line.index("/")
                                _path = _line[_start:].strip()
                                _fname = _path.split("/")[-1]
                                if _fname.rpartition(".so")[0].startswith(lib_name):
                                    return _path
                except OSError:
                    pass
                return _orig_find_loaded_library(lib_name)

            _vsu.find_loaded_library = _find_loaded_library_skip_stub
            # cuda_wrapper binds the symbol at its module top (`from ... import
            # find_loaded_library`); patch that reference too if already imported.
            import sys as _sys

            _cw = _sys.modules.get(
                "vllm.distributed.device_communicators.cuda_wrapper"
            )
            if _cw is not None and hasattr(_cw, "find_loaded_library"):
                _cw.find_loaded_library = _find_loaded_library_skip_stub
            print("[grpo_training] Installed libcudart-stub guard for colocate vLLM init")
        except Exception as _e:
            print(f"[grpo_training] Warning: libcudart-stub guard not installed: {_e}")

    # ── Multi-GPU colocate: disable vLLM custom all-reduce (PCIe A6000 lane) ──
    # TRL's colocate LLM(...) construction (VLLMGeneration._init_vllm) hardcodes
    # its engine kwargs and exposes NO passthrough — GRPOConfig has no
    # `vllm_disable_custom_all_reduce` field — so the flag can only be injected by
    # wrapping vllm.LLM.__init__ for the duration of _init_vllm. This is MANDATORY
    # at TP>1 on this cluster: P2P is disabled cluster-wide (server.env /
    # slurm_train_2x NCCL_P2P_DISABLE=1), and vLLM's custom all-reduce probe then
    # crashes at engine init with `custom_all_reduce.cuh 'invalid argument'` (the
    # 2026-07-24 probe-calibration precedent). Mirrors the shared inference util
    # (dagspaces/common/vllm_inference.py: `setdefault disable_custom_all_reduce`
    # at TP>1). Default: on when TP>1, overridable via the config knob. Additive,
    # model-agnostic, no-op at TP=1 unless explicitly requested. Installs BEFORE
    # the qwen3.5 block so the two _init_vllm wrappers compose (qwen captures this
    # wrapper as its `orig`); for gemma-4 only this wrapper runs.
    if use_vllm and vllm_mode == "colocate":
        _tp = int(grpo_cfg.get("vllm_tensor_parallel_size", 1) or 1)
        _disable_car = grpo_cfg.get("vllm_disable_custom_all_reduce")
        if _disable_car is None:
            _disable_car = _tp > 1
        if _disable_car:
            try:
                from trl.generation.vllm_generation import VLLMGeneration as _VG_car
                _orig_init_vllm_car = _VG_car._init_vllm

                def _init_vllm_disable_custom_ar(self_vllm,
                                                 __orig=_orig_init_vllm_car):
                    from vllm import LLM as _LLM_car
                    _real_llm_init = _LLM_car.__init__

                    def _llm_init_no_custom_ar(llm_self, *args, **kwargs):
                        # Only the colocate engine passes distributed_executor_
                        # backend="external_launcher"; setdefault so an explicit
                        # caller value (or the qwen composite wrapper's kwargs)
                        # still wins on other keys.
                        kwargs.setdefault("disable_custom_all_reduce", True)
                        return _real_llm_init(llm_self, *args, **kwargs)

                    _LLM_car.__init__ = _llm_init_no_custom_ar
                    try:
                        __orig(self_vllm)
                    finally:
                        _LLM_car.__init__ = _real_llm_init

                _VG_car._init_vllm = _init_vllm_disable_custom_ar
                print("[grpo_training] Colocate TP>1: injected "
                      "disable_custom_all_reduce=True into vLLM engine init "
                      f"(TP={_tp}, PCIe-A6000 P2P-disabled lane)")
            except Exception as _e:
                print("[grpo_training] Warning: could not inject "
                      f"disable_custom_all_reduce: {_e}")

    # Qwen3.5 is natively multimodal. TRL loads the CausalLM (text-only) for
    # training, but vLLM needs the composite Qwen3_5Config (with vision_config)
    # to initialize the full model from merged_dir.  Monkey-patch to reload
    # the config from the original model zoo path.
    model_family = str(OmegaConf.select(cfg, "model.model_family", default="") or "")
    _original_model_source = str(OmegaConf.select(cfg, "model.model_source", default="") or "")
    if "qwen3.5" in model_family.lower():
        try:
            from trl.generation.vllm_generation import VLLMGeneration
            _orig_init_vllm = VLLMGeneration._init_vllm

            def _patched_init_vllm(self_vllm,
                                   _zoo_path=_original_model_source):
                from vllm import LLM as _LLM
                _orig_LLM = _LLM.__init__

                def _llm_init_with_composite_config(llm_self, *args, **kwargs):
                    # Ensure vLLM gets the composite Qwen3_5Config (with
                    # vision_config) even though merged_dir was saved from
                    # CausalLM with Qwen3_5TextConfig.
                    def _ensure_composite(config):
                        if hasattr(config, "vision_config") and config.vision_config is not None:
                            return config
                        from transformers import AutoConfig as _AC
                        try:
                            return _AC.from_pretrained(_zoo_path, trust_remote_code=True)
                        except Exception:
                            return config
                    kwargs["hf_overrides"] = _ensure_composite
                    return _orig_LLM(llm_self, *args, **kwargs)

                _LLM.__init__ = _llm_init_with_composite_config
                try:
                    _orig_init_vllm(self_vllm)
                finally:
                    _LLM.__init__ = _orig_LLM

            VLLMGeneration._init_vllm = _patched_init_vllm

            # Colocate weight-sync name remap (vLLM 0.25 composite VLM arch).
            # TRL's sync_weights pushes the text training model's params (loaded
            # as Qwen3_5ForCausalLM → names `model.*` / `lm_head.*`) one-by-one
            # into the vLLM engine model, which is the COMPOSITE
            # Qwen3_5ForConditionalGeneration (the composite-config patch above).
            # That composite nests the text stack under `language_model.`
            # (`self.language_model = Qwen3_5ForCausalLM(prefix="language_model")`),
            # so a bare `model.layers.…` name has no target and load_weights
            # raises "no module or parameter named 'model'". Patch
            # _push_param_to_vllm (runs at SYNC time, engine awake — the engine
            # model is asleep/None right after _init_vllm, so patching the model
            # object there is unreliable) to prefix text names into the composite
            # namespace. Deterministic + self-validating: a wrong name still
            # raises (vLLM checks every name), so this cannot silently mis-route.
            # Vision params are never pushed (LoRA is text-only), so untouched.
            _orig_push = VLLMGeneration._push_param_to_vllm

            def _push_param_to_vllm_lm_prefix(self_vllm, name, param):
                if self_vllm.mode == "colocate" and (
                    name.startswith("model.") or name.startswith("lm_head.")
                ):
                    name = "language_model." + name
                return _orig_push(self_vllm, name, param)

            VLLMGeneration._push_param_to_vllm = _push_param_to_vllm_lm_prefix
            print(f"[grpo_training] Patched TRL vLLM init for Qwen3.5 "
                  f"(composite config from {_original_model_source}) + "
                  f"colocate weight-sync remap (model.*/lm_head.* → language_model.*)")
        except Exception as e:
            print(f"[grpo_training] Warning: failed to patch TRL vLLM init: {e}")

    # ── gemma-4: composite-arch weight sync (the analogue to the qwen3.5 block) ──
    # gemma-4 is ALSO a composite VLM (Gemma4UnifiedForConditionalGeneration), but
    # the qwen fix does NOT transfer — it would break the sync. Two arch facts
    # (verified 2026-07-24, cu129 venv) make gemma the mirror image of qwen here:
    #
    #  1. `AutoModelForCausalLM.from_pretrained(merged_dir)` returns the FULL
    #     composite for gemma-4 (Gemma4UnifiedConfig is unmapped in the CausalLM
    #     registry → AutoModel falls back to the config's own
    #     `Gemma4UnifiedForConditionalGeneration`). So the TRAINING model's params
    #     are already composite-namespaced: `model.language_model.layers.…`
    #     (embeddings are tied — there is NO separate `lm_head.*`). Qwen, by
    #     contrast, loads TEXT-ONLY (`model.*` / `lm_head.*`) and needed manual
    #     re-prefixing into `language_model.…`.
    #  2. vLLM's Gemma4UnifiedForConditionalGeneration.load_weights applies an
    #     `hf_to_vllm_mapper` that already maps `model.language_model.` →
    #     `language_model.model.`, `lm_head.` → `language_model.lm_head.`, and a
    #     `model` catch-all → `language_model.model`. Qwen's Qwen3_5 mapper has NO
    #     such prefix rule (only stacked-qkv entries), which is why qwen needed the
    #     manual `_push_param_to_vllm` prefix and gemma does NOT.
    #
    # TRL's colocate `_push_param_to_vllm` calls `model.load_weights([(name,param)])`
    # — the SAME mapper-applying path used for the initial on-disk load (which is
    # known-good: gemma-4-12b loads & generates, canonical-models.md). So the sync
    # names resolve natively; a qwen-style manual prefix would DOUBLE-prefix
    # (`language_model.model.language_model.…`) and every push would raise.
    #
    # Two additive contributions (NO text-name rewriting — the mapper does that):
    #  (a) a TEXT-ONLY sync filter. Because the composite training model also
    #      carries FROZEN vision/audio params that the vLLM encoder-free
    #      Gemma4Unified lacks (e.g. `model.embed_vision.pos_embedding`), pushing
    #      them raises "no module or parameter named …". LoRA is text-only and
    #      vision is already loaded at engine init, so we push only
    #      `model.language_model.*` (tied `lm_head.*`) and skip the rest — the
    #      qwen path got this for free (text-only training model). Live-confirmed
    #      2026-07-24: the TP=2 smoke reached sync and hit exactly this vision
    #      param; the m0 OOM had blocked the sync entirely before then.
    #  (b) a bounded diagnostic that logs the first pushed/skipped names and
    #      annotates any residual TEXT-param failure with the gemma naming context.
    if "gemma-4" in model_family.lower():
        try:
            # Dig past PEFT/LoRA wrappers to the underlying arch class for the log.
            _under = model
            for _attr in ("base_model", "model"):
                _under = getattr(_under, _attr, _under)
            print(f"[grpo_training] gemma-4 composite weight-sync: training model "
                  f"= {type(model).__name__} (arch {type(_under).__name__}); "
                  f"relying on vLLM's Gemma4Unified hf_to_vllm_mapper "
                  f"(model.language_model.* → language_model.model.*) — no manual "
                  f"prefix (unlike qwen3.5).")

            # This branch runs ONLY for gemma-4 (never qwen), so the freshly
            # imported method is the unpatched TRL original.
            from trl.generation.vllm_generation import VLLMGeneration as _VG_g
            _orig_push_g = _VG_g._push_param_to_vllm
            _g_diag = {"n": 0, "nskip": 0}

            def _push_param_to_vllm_gemma_diag(self_vllm, name, param,
                                               __orig=_orig_push_g):
                # gemma-4 colocate sync: push ONLY the text stack
                # (`model.language_model.*` / tied `lm_head.*`) and SKIP vision /
                # audio params. The merged TRAINING model is the FULL multimodal
                # Gemma4Unified, so TRL's sync_weights iterates its vision/audio
                # embedder+tower params too — but the vLLM ENCODER-FREE
                # Gemma4Unified variant lacks most of them (verified live 07-24:
                # pushing `model.embed_vision.pos_embedding` raised "no module or
                # parameter named 'embed_vision.pos_embedding' … available:
                # {'embed_vision.embedding_projection.weight'}"). LoRA is
                # text-only and every vision/audio weight is FROZEN — identical to
                # base and already loaded by vLLM at engine init — so they never
                # need re-syncing. Skipping them reproduces the qwen path's
                # behaviour, whose text-only training model simply never yielded
                # vision params. The text names still flow through vLLM's
                # hf_to_vllm_mapper (`model.language_model.`→`language_model.model.`,
                # verified: sync[0..] land) — NO manual prefixing (that is the
                # gemma-vs-qwen distinction).
                if self_vllm.mode == "colocate" and not (
                    name.startswith("model.language_model.")
                    or name.startswith("lm_head.")
                ):
                    if _g_diag["nskip"] < 5:
                        print(f"[grpo_training] gemma-4 sync: SKIP non-text param "
                              f"'{name}' (frozen vision/audio; absent from vLLM "
                              f"encoder-free Gemma4Unified)")
                        _g_diag["nskip"] += 1
                    return None
                if self_vllm.mode == "colocate" and _g_diag["n"] < 5:
                    print(f"[grpo_training] gemma-4 sync[{_g_diag['n']}]: "
                          f"push '{name}' shape={tuple(param.shape)}")
                    _g_diag["n"] += 1
                try:
                    return __orig(self_vllm, name, param)
                except Exception as _e:
                    raise RuntimeError(
                        f"gemma-4 colocate weight sync failed on TEXT param "
                        f"'{name}'. Expected vLLM's Gemma4Unified hf_to_vllm_mapper "
                        f"to map 'model.language_model.*'→'language_model.model.*'. "
                        f"If this is a name-not-found error, the mapper contract "
                        f"changed — do NOT copy qwen's prefix (it double-prefixes). "
                        f"Original error: {_e!r}"
                    ) from _e

            _VG_g._push_param_to_vllm = _push_param_to_vllm_gemma_diag
            print("[grpo_training] Installed gemma-4 colocate weight-sync filter "
                  "(text-only push; vision/audio skipped)")
        except Exception as e:
            print(f"[grpo_training] Warning: failed to install gemma-4 sync "
                  f"diagnostic: {e}")

    print(f"[grpo_training] Starting GRPO (G={training_args.num_generations}, "
          f"lr={training_args.learning_rate}, beta={training_args.beta}, "
          f"scale_rewards={training_args.scale_rewards}, "
          f"mask_truncated={training_args.mask_truncated_completions}, "
          f"vllm={use_vllm}, mode={vllm_mode if use_vllm else 'N/A'})")

    # Callback to fix base_model_name_or_path in intermediate checkpoint
    # adapter configs.  PEFT records model.name_or_path (the ephemeral scratch
    # dir) — rewrite to the persistent base model path after every save.
    import json as _json_cb

    from transformers import TrainerCallback

    class _FixAdapterBasePathCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            ac_path = os.path.join(ckpt_dir, "adapter_config.json")
            if os.path.exists(ac_path):
                with open(ac_path) as f:
                    acfg = _json_cb.load(f)
                if acfg.get("base_model_name_or_path") != base_model_path:
                    acfg["base_model_name_or_path"] = base_model_path
                    with open(ac_path, "w") as f:
                        _json_cb.dump(acfg, f, indent=2)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[_FixAdapterBasePathCallback()],
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Fix adapter_config.json: PEFT records model.name_or_path as
    # base_model_name_or_path, which points to the ephemeral scratch dir
    # used during training.  Rewrite it to the original base model path
    # so vLLM can validate architecture compatibility at inference time.
    _adapter_cfg_path = os.path.join(output_dir, "adapter_config.json")
    if os.path.exists(_adapter_cfg_path):
        import json as _json
        with open(_adapter_cfg_path) as _f:
            _acfg = _json.load(_f)
        if _acfg.get("base_model_name_or_path") != base_model_path:
            _acfg["base_model_name_or_path"] = base_model_path
            with open(_adapter_cfg_path, "w") as _f:
                _json.dump(_acfg, _f, indent=2)
            print(f"[grpo_training] Fixed adapter_config base_model_name_or_path → {base_model_path}")

    # Write training metadata sidecar so eval runs can inherit
    # GRPO hyperparameters for W&B filtering.
    _training_meta = {
        "contrastive_ratio": _contrastive,
        "contrastive_lambda": float(grpo_cfg.get("contrastive_lambda", 0.5)),
        "num_generations": training_args.num_generations,
        "learning_rate": training_args.learning_rate,
        "lr_scheduler_type": str(training_args.lr_scheduler_type),
        "min_lr_rate": grpo_cfg.get("min_lr_rate"),
        "num_epochs": grpo_config_kwargs["num_train_epochs"],
        "beta": training_args.beta,
        "scale_rewards": training_args.scale_rewards,
        "mask_truncated_completions": training_args.mask_truncated_completions,
        "num_iterations": training_args.num_iterations,
        "epsilon_high": training_args.epsilon_high,
        "vignette_ratio": _vignette_ratio,
        # Records which corpus the judgment vignettes came from. "" ⇒ same as the
        # grounding norm_universes (historical default). The *_is_separate flag
        # records what was ACTUALLY used (a set-but-non-file path fails loud in
        # _resolve_vignette_universes, so this can't silently certify the wrong
        # corpus).
        "vignette_norm_universes_path": vignette_norm_universes_path or "",
        "vignette_universe_is_separate": _vignette_universe_is_separate,
        "n_vignettes_pre_screen": n_vignette_meta,
        # Final training set (post-screen, post-split), like n_flow_chunks —
        # the realized vignette mix, vs. the configured pre-screen ratio.
        "n_vignettes_post_screen": _train_vignettes,
        # Realised vignette FORCE mix (gold yes:no) before/after screening.
        # The v11 steering variable — the screen is force-blind and has
        # historically halved "no"-vignette survival, so the ratio the policy
        # actually trains on must be auditable per run (previously required
        # mining reward_traces.jsonl; see 2026-07-01 field note).
        "n_vignettes_yes_pre_screen": _vig_pre["yes"],
        "n_vignettes_no_pre_screen": _vig_pre["no"],
        "n_vignettes_yes_post_screen": _vig_post["yes"],
        "n_vignettes_no_post_screen": _vig_post["no"],
        "judgment_reward_weights": _judgment_weights,
        "no_flow_scoring": _nf_scoring,
        "reward_composition": _composition,
        "abstention_penalty": _abstention_penalty,
        "confidence_fallthrough": _confidence_fallthrough,
        "rground_scoring": str(grpo_cfg.get("rground_scoring", "absolute")),
        "rground_judge_backend": str(grpo_cfg.get("rground_judge_backend", "llm")).lower(),
        "reranker_app_weight": float(grpo_cfg.get("reranker_app_weight", 0.2)),
        "rground_app_weight": float(grpo_cfg.get("rground_app_weight", 0.0)),
        "rground_app_mode": str(grpo_cfg.get("rground_app_mode", "additive")),
        "rground_app_floor": float(grpo_cfg.get("rground_app_floor", 0.4)),
        "rground_app_floor_prohibit": (
            float(grpo_cfg["rground_app_floor_prohibit"])
            if grpo_cfg.get("rground_app_floor_prohibit", None) is not None else None),
        "rground_app_hedge_prohibit": (
            float(grpo_cfg["rground_app_hedge_prohibit"])
            if grpo_cfg.get("rground_app_hedge_prohibit", None) is not None else None),
        "reward_weights": list(weights),
        "online_rground": use_online_rground,
        "enable_thinking_grpo": enable_thinking_grpo,
        "n_training_rows": len(dataset),
        "n_screened_out": _n_screened_out,
        "dev_fraction": _dev_fraction,
        "n_dev_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "n_flow_chunks": _train_gold_pos,
        "n_no_flow_chunks": _train_gold_neg,
        "base_model": base_model_path,
        "sft_checkpoint": sft_checkpoint,
    }
    _meta_path = os.path.join(output_dir, "training_metadata.json")
    # MERGE, never clobber (audit 2026-07-28): build_modular_dataset writes
    # this file first with the principle-6 audit trail (prescreen_report,
    # battery_compositions, realized task mix, m1_cache_signature) — the m1
    # wave destroyed all of it by overwriting here, which is why the m1
    # record shows n_vignettes 0 despite training 180 batteries.
    _existing_meta: dict = {}
    try:
        if os.path.exists(_meta_path):
            with open(_meta_path) as _mf:
                _existing_meta = json.load(_mf)
    except Exception:
        _existing_meta = {}
    _existing_meta.update(_training_meta)
    with open(_meta_path, "w") as _mf:
        json.dump(_existing_meta, _mf, indent=2, default=str)
    _training_meta = _existing_meta
    print(f"[grpo_training] Wrote training metadata to {_meta_path}")

    # Mirror the FULL training metadata into the W&B run config — same dict
    # as training_metadata.json, so the two can never drift apart (the old
    # hand-copied subset silently omitted every redesign knob: rground_scoring,
    # reward_composition, n_screened_out, vignette counts, beta, ...).
    try:
        import wandb as _wandb
        if _wandb.run is not None:
            _wandb.run.config.update({
                "grpo_runtime": {
                    **_training_meta,
                    "n_contrastive_rows": sum(
                        1 for m in reward_fn.prompt_metadata.values()
                        if m.get("is_contrastive")
                    ),
                }
            }, allow_val_change=True)
    except Exception:
        pass

    # Run the promotion gates immediately and put the verdict next to the
    # training curves — a cell that fails gates should be visible in the
    # sweep table without anyone remembering to run the checker script.
    # (scripts/check_grpo_promotion_gates.py still works for re-checks.)
    try:
        from ..gates import check_promotion_gates
        _gates_report = check_promotion_gates(output_dir)
        _gates_path = os.path.join(output_dir, "promotion_gates.json")
        with open(_gates_path, "w") as _gf:
            json.dump(_gates_report, _gf, indent=2)
        print(f"[grpo_training] Promotion gates: "
              f"promote={_gates_report.get('promote')} → {_gates_path}")
        import wandb as _wandb
        if _wandb.run is not None:
            _wandb.run.summary["gates/promote"] = bool(_gates_report.get("promote"))
            for _gname, _g in (_gates_report.get("gates") or {}).items():
                _wandb.run.summary[f"gates/{_gname}/status"] = _g.get("status")
                for _k, _v in _g.items():
                    if isinstance(_v, (int, float)) and not isinstance(_v, bool):
                        _wandb.run.summary[f"gates/{_gname}/{_k}"] = _v
    except Exception as _e:
        print(f"[grpo_training] WARNING: promotion gates did not run: {_e}")

    print(f"[grpo_training] Saved GRPO checkpoint to {output_dir}")
