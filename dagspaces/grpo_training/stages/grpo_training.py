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
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import OmegaConf


def _generate_vignettes(
    norm_universes: Dict[str, list],
    prompt_template: str,
) -> List[Dict[str, Any]]:
    """Generate judgment vignettes from info-flow-governing norms.

    Each norm with ``governs_info_flow=true`` and a clear normative force
    (obligatory/prohibited/recommended/discouraged) becomes a vignette.
    The scenario is built from norm fields; the norm_articulation is NOT
    included (that would leak the answer).

    Returns list of dicts with: prompt_text, source_id, gold_judgment,
    source_norm (full norm dict), normative_force.
    """
    # Map normative_force → gold judgment
    _FORCE_TO_GOLD = {
        "obligatory": "yes",
        "recommended": "yes",
        "prohibited": "no",
        "discouraged": "no",
    }

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


def _build_grpo_dataset(
    chunks_df: pd.DataFrame,
    tokenizer,
    prompt_template: str,
    enable_thinking: bool = True,
    contrastive_ratio: float = 0.0,
    all_source_ids: Optional[List[str]] = None,
    vignettes: Optional[List[Dict[str, Any]]] = None,
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

    rows: List[Dict[str, Any]] = []
    # Maps formatted_prompt → raw user_prompt so OnlineRGround can pass
    # clean text to the judge instead of chat-templated text.
    raw_prompts: Dict[str, str] = {}

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
) -> None:
    """Run GRPO training with TRL + vLLM.

    Args:
        sft_checkpoint: Path to SFT LoRA checkpoint directory.
        chunks_path: Path to chunks parquet (chunk_text + source_id).
        norm_universes_path: Path to norm_universes.json.
        output_dir: Directory to save GRPO checkpoint.
        cfg: Hydra config with training.grpo section.
        embeddings_dir: Path to per-book .npy embeddings (for online R_ground).
        reward_cache_path: Path to reward_cache.parquet (legacy cached R_ground).
    """
    from trl import GRPOTrainer, GRPOConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

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

    print(f"[grpo_training] Chunks: {len(chunks_df)}")
    print(f"[grpo_training] Norm universes: {len(norm_universes)} sources")

    # Build composite reward function
    from .rewards import CompositeRewardFunction

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
        from sentence_transformers import SentenceTransformer
        context_embedding_model = SentenceTransformer(context_model_name)
        print(f"[grpo_training] Loaded context embedding model: {context_model_name}")
    except Exception as e:
        print(f"[grpo_training] Warning: could not load embedding model: {e}")

    # Build source context lookup from norm universes.
    # Each source maps to its list of unique norm-level context strings,
    # so R_context can do per-flow max-similarity matching instead of
    # comparing against one giant concatenated string.
    source_contexts: Dict[str, List[str]] = {}
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
    _contrastive = grpo_cfg.get("contrastive_ratio", 0.1)
    # Contrastive pairing works with both online and cached R_ground.
    # Contrastive rows are added as new dataset entries (with a trailing
    # newline to make the formatted prompt key unique).  OnlineRGround
    # retrieves norms from the wrong source for contrastive completions,
    # producing naturally low R_ground.
    if use_online_rground:
        from .clients import EmbeddingClient, JudgeClient, NormRetriever
        from .online_rground import OnlineRGround
        from ..schemas import FlowGovernanceJudgment, NoFlowCoverageJudgment

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

        embedding_client = EmbeddingClient(
            base_url=embedding_url,
            model_name=emb_model_name,
        )
        judge_client = JudgeClient(
            base_url=judge_url,
            model_name=judge_model_name,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            json_schema=FlowGovernanceJudgment.model_json_schema(),
        )

        norm_retriever = NormRetriever(
            norm_universes=norm_universes,
            embeddings_dir=embeddings_dir or "",
            embedding_client=embedding_client,
        )

        _contrastive_lambda = float(grpo_cfg.get("contrastive_lambda", 0.5))
        online_rground = OnlineRGround(
            embedding_client=embedding_client,
            judge_client=judge_client,
            norm_retriever=norm_retriever,
            all_source_ids=list(norm_universes.keys()),
            contrastive_lambda=_contrastive_lambda,
            no_flow_judge_system_prompt=nf_system_prompt,
            no_flow_judge_prompt_template=nf_prompt_template,
            no_flow_judge_json_schema=NoFlowCoverageJudgment.model_json_schema(),
        )
        print(f"[grpo_training] Online R_ground enabled "
              f"(embed={embedding_url}, judge={judge_url}, "
              f"contrastive_lambda={_contrastive_lambda})")
    elif not use_online_rground and weights[5] > 0.0:
        print(f"[grpo_training] R_ground using cached lookup "
              f"({len(reward_cache)} entries)")

    _nf_scoring = grpo_cfg.get("no_flow_scoring", "independent")
    _judgment_weights = list(grpo_cfg.get("judgment_reward_weights", [0.5, 0.25, 0.25]))
    reward_fn = CompositeRewardFunction(
        weights=weights,
        norm_universes=norm_universes,
        reward_cache=reward_cache,
        context_embedding_model=context_embedding_model,
        source_contexts=source_contexts,
        trace_log_path=trace_log_path,
        trace_every_n_calls=trace_every,
        online_rground=online_rground,
        no_flow_scoring=_nf_scoring,
        judgment_weights=_judgment_weights,
    )
    print(f"[grpo_training] No-flow scoring mode: {_nf_scoring}")
    reward_fn.enable_thinking_grpo = enable_thinking_grpo
    print(f"[grpo_training] Reward traces → {trace_log_path} (every {trace_every} calls)")

    # Pre-merge LoRA into the base model and save to a temp directory.
    # TRL's vLLM weight sync (sync_weights) doesn't reliably apply LoRA
    # for Qwen3 + vLLM 0.17, so we give vLLM the fully-merged checkpoint.
    # The trainer still uses LoRA for memory-efficient training.
    base_model_path = str(OmegaConf.select(cfg, "model.model_source"))
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
        from transformers import AutoModel
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
    merged_dir_nfs = os.path.join(output_dir, "_merged_sft")
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
    merged_dir_scratch = os.path.join(scratch_base, f"grpo_merged_sft_{job_id}")
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
    from peft import LoraConfig as _LoraConfig, get_peft_model
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

    all_source_ids = list(norm_universes.keys())

    # Generate judgment vignettes from norm universes if ratio > 0
    _vignette_ratio = float(grpo_cfg.get("vignette_ratio", 0.0))
    vignettes = []
    if _vignette_ratio > 0.0:
        vig_prompt_cfg = OmegaConf.select(cfg, "prompt_norm_judgment")
        if vig_prompt_cfg:
            vig_sys = str(OmegaConf.select(vig_prompt_cfg, "system_prompt") or "")
            vig_tmpl = str(OmegaConf.select(vig_prompt_cfg, "prompt_template") or "")
        else:
            vig_sys = ""
            vig_tmpl = ""
        if vig_tmpl:
            vignettes = _generate_vignettes(norm_universes, vig_tmpl)
            print(f"[grpo_training] Generated {len(vignettes)} judgment vignettes "
                  f"from {sum(1 for n in sum(norm_universes.values(), []) if n.get('governs_info_flow'))} "
                  f"info-flow norms")

    # Contrastive signal is now per-completion dual scoring inside
    # OnlineRGround, so no additive contrastive rows are needed.
    # Legacy contrastive_ratio kept for backward compat but defaults to 0.
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
    n_vignette_meta = sum(1 for m in reward_fn.prompt_metadata.values() if m.get("task_type") == "norm_judgment")
    print(f"[grpo_training] Reward prompt metadata: {len(reward_fn.prompt_metadata)} entries "
          f"({n_contrastive} contrastive, {n_vignette_meta} vignettes)")
    print(f"[grpo_training] Gold labels: {n_gold_pos} has_exchange=True, "
          f"{n_gold_neg} has_exchange=False, {n_gold_unk} unknown")

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
        report_to="wandb" if OmegaConf.select(cfg, "wandb.enabled") else "none",
    )

    # Optional overrides: max_steps / warmup_steps take precedence over ratio
    max_steps = grpo_cfg.get("max_steps")
    if max_steps is not None:
        grpo_config_kwargs["max_steps"] = int(max_steps)
    warmup_steps = grpo_cfg.get("warmup_steps")
    if warmup_steps is not None:
        grpo_config_kwargs["warmup_steps"] = int(warmup_steps)
    else:
        grpo_config_kwargs["warmup_ratio"] = grpo_cfg.get("warmup_ratio", 0.1)

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
            print(f"[grpo_training] Patched TRL vLLM init for Qwen3.5 "
                  f"(composite config from {_original_model_source})")
        except Exception as e:
            print(f"[grpo_training] Warning: failed to patch TRL vLLM init: {e}")

    print(f"[grpo_training] Starting GRPO (G={training_args.num_generations}, "
          f"vllm={use_vllm}, mode={vllm_mode if use_vllm else 'N/A'})")

    # Callback to fix base_model_name_or_path in intermediate checkpoint
    # adapter configs.  PEFT records model.name_or_path (the ephemeral scratch
    # dir) — rewrite to the persistent base model path after every save.
    from transformers import TrainerCallback
    import json as _json_cb

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
        "vignette_ratio": _vignette_ratio,
        "judgment_reward_weights": _judgment_weights,
        "no_flow_scoring": _nf_scoring,
        "reward_weights": list(weights),
        "online_rground": use_online_rground,
        "enable_thinking_grpo": enable_thinking_grpo,
        "n_training_rows": len(dataset),
        "n_flow_chunks": n_gold_pos,
        "n_no_flow_chunks": n_gold_neg,
        "base_model": base_model_path,
        "sft_checkpoint": sft_checkpoint,
    }
    _meta_path = os.path.join(output_dir, "training_metadata.json")
    with open(_meta_path, "w") as _mf:
        json.dump(_training_meta, _mf, indent=2)
    print(f"[grpo_training] Wrote training metadata to {_meta_path}")

    # Update W&B config with runtime training stats (if TRL's wandb run is active)
    try:
        import wandb as _wandb
        if _wandb.run is not None:
            _wandb.run.config.update({
                "grpo_runtime": {
                    "n_total_rows": len(dataset),
                    "n_contrastive": sum(1 for m in reward_fn.prompt_metadata.values() if m.get("is_contrastive")),
                    "n_flow_chunks": n_gold_pos,
                    "n_no_flow_chunks": n_gold_neg,
                    "contrastive_ratio": _contrastive,
                    "reward_weights": list(weights),
                    "online_rground": use_online_rground,
                    "enable_thinking_grpo": enable_thinking_grpo,
                }
            }, allow_val_change=True)
    except Exception:
        pass

    print(f"[grpo_training] Saved GRPO checkpoint to {output_dir}")
