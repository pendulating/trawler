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


def _build_grpo_dataset(
    chunks_df: pd.DataFrame,
    tokenizer,
    prompt_template: str,
    enable_thinking: bool = True,
) -> "Dataset":
    """Build GRPO training dataset from text chunks.

    Constructs prompts by applying the CI extraction instruction template
    to each chunk, then pre-applies the chat template so TRL routes
    through vLLM's ``llm.generate()`` instead of ``llm.chat()``.

    Args:
        chunks_df: DataFrame with columns: chunk_text, source_id.
        tokenizer: Model tokenizer for chat template formatting.
        prompt_template: CI extraction prompt with ``{{chunk_text}}`` placeholder.
        enable_thinking: Allow ``<think>`` blocks during GRPO generation.

    Returns:
        (dataset, raw_prompts) where raw_prompts maps formatted prompt → raw
        user prompt (for passing clean text to the judge).
    """
    import hashlib
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
        })

    dataset = Dataset.from_list(rows)
    thinking_label = "enabled" if enable_thinking else "disabled"
    print(f"[grpo_training] Dataset: {len(dataset)} prompts "
          f"(thinking={thinking_label})")
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

    print(f"[grpo_training] Chunks: {len(chunks_df)}")
    print(f"[grpo_training] Norm universes: {len(norm_universes)} sources")

    # Build composite reward function
    from .rewards import CompositeRewardFunction

    weights = grpo_cfg.get("reward_weights", [0.2, 0.15, 0.15, 0.15, 0.15, 0.2])

    # Load context embedding model for r_context
    context_embedding_model = None
    context_model_name = grpo_cfg.get("context_embedding_model", "all-MiniLM-L6-v2")
    try:
        from sentence_transformers import SentenceTransformer
        context_embedding_model = SentenceTransformer(context_model_name)
        print(f"[grpo_training] Loaded context embedding model: {context_model_name}")
    except Exception as e:
        print(f"[grpo_training] Warning: could not load embedding model: {e}")

    # Build source context lookup from norm universes
    source_contexts = {}
    for source_id, norms in norm_universes.items():
        contexts = set()
        for norm in norms:
            ctx = norm.get("context")
            if ctx:
                contexts.add(str(ctx))
        source_contexts[source_id] = "; ".join(contexts) if contexts else ""

    # Trace logging: log detailed reward breakdowns ~10 times during training.
    # With 910 training steps and 2 completions per call, trace_every ≈ 91.
    trace_log_path = os.path.join(output_dir, "reward_traces.jsonl")
    total_steps = grpo_cfg.get("max_steps") or (len(sft_df) // max(grpo_cfg.get("num_generations", 2), 1))
    trace_every = max(total_steps // 10, 1)

    # Online R_ground: use embedding + judge servers instead of cached lookup
    online_rground = None
    use_online_rground = grpo_cfg.get("online_rground", False) and weights[5] > 0.0
    _contrastive = grpo_cfg.get("contrastive_ratio", 0.1)
    if use_online_rground and _contrastive > 0.0:
        print(
            "[grpo_training] WARNING: online_rground=true with contrastive_ratio="
            f"{_contrastive}. Contrastive pairing is incompatible with "
            "online R_ground (TRL generates identical completions for duplicate "
            "prompts, so contrastive rows waste training steps with identical "
            "reward signals). Forcing contrastive_ratio=0.0."
        )
        grpo_cfg["contrastive_ratio"] = 0.0
    if use_online_rground:
        from .clients import EmbeddingClient, JudgeClient, NormRetriever
        from .online_rground import OnlineRGround
        from ..schemas import FlowGovernanceJudgment

        emb_port = grpo_cfg.get("embedding_server_port", 8001)
        judge_port = grpo_cfg.get("judge_server_port", 8002)

        # Model names for vLLM API (must match the path used to launch servers)
        emb_model_name = str(
            OmegaConf.select(cfg, "model.embedding_model_source") or ""
        )
        judge_model_name = str(
            OmegaConf.select(cfg, "judge_model.model_source") or ""
        )

        # Load judge prompt template
        prompt_cfg = (
            OmegaConf.select(cfg, "prompt_reward_judge")
            or OmegaConf.select(cfg, "prompt")
        )
        system_prompt = str(OmegaConf.select(prompt_cfg, "system_prompt") or "")
        prompt_template = str(OmegaConf.select(prompt_cfg, "prompt_template") or "")

        embedding_client = EmbeddingClient(
            base_url=f"http://localhost:{emb_port}",
            model_name=emb_model_name,
        )
        judge_client = JudgeClient(
            base_url=f"http://localhost:{judge_port}",
            model_name=judge_model_name,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            json_schema=FlowGovernanceJudgment.model_json_schema(),
        )

        if not embeddings_dir:
            raise ValueError(
                "[grpo_training] online_rground=true requires embeddings_dir "
                "(pass embeddings input from norm_universe stage)"
            )
        norm_retriever = NormRetriever(
            norm_universes=norm_universes,
            embeddings_dir=embeddings_dir,
            embedding_client=embedding_client,
        )

        online_rground = OnlineRGround(
            embedding_client=embedding_client,
            judge_client=judge_client,
            norm_retriever=norm_retriever,
        )
        print(f"[grpo_training] Online R_ground enabled "
              f"(embed=:{emb_port}, judge=:{judge_port})")
    elif not use_online_rground and weights[5] > 0.0:
        print(f"[grpo_training] R_ground using cached lookup "
              f"({len(reward_cache)} entries)")

    reward_fn = CompositeRewardFunction(
        weights=weights,
        norm_universes=norm_universes,
        reward_cache=reward_cache,
        context_embedding_model=context_embedding_model,
        source_contexts=source_contexts,
        trace_log_path=trace_log_path,
        trace_every_n_calls=trace_every,
        online_rground=online_rground,
    )
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
    _base = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    _peft = PeftModel.from_pretrained(_base, sft_checkpoint)
    _merged = _peft.merge_and_unload()

    # Save merged model to both NFS (persistence) and /scratch (fast vLLM loads).
    # vLLM reloads weights from disk every time it wakes from sleep mode —
    # /scratch is local SSD, much faster than NFS for repeated reads.
    merged_dir_nfs = os.path.join(output_dir, "_merged_sft")
    os.makedirs(merged_dir_nfs, exist_ok=True)
    _merged.save_pretrained(merged_dir_nfs)
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
    enable_thinking_grpo = grpo_cfg.get("enable_thinking_grpo", True)

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

    dataset, raw_prompts = _build_grpo_dataset(
        chunks_df, tokenizer, ci_prompt_template, enable_thinking_grpo,
    )

    # Build prompt→metadata lookup so the reward function can access
    # source_id/prompt_id without relying on TRL forwarding dataset columns.
    # Standard rows are stored; contrastive rows (same prompt text) are skipped
    # since during GRPO generation TRL doesn't know about is_contrastive —
    # all generations use the same prompt and get the standard metadata.
    # chunk_text stores the raw user prompt (pre-template) so OnlineRGround
    # can pass clean text to the judge instead of chat-templated text.
    reward_fn.prompt_metadata = {}
    for row in dataset:
        key = row["prompt"]
        if key not in reward_fn.prompt_metadata:
            reward_fn.prompt_metadata[key] = {
                "source_id": row.get("source_id", ""),
                "prompt_id": row.get("prompt_id", ""),
                "is_contrastive": False,
                "chunk_text": raw_prompts.get(key, ""),
            }
    print(f"[grpo_training] Reward prompt metadata: {len(reward_fn.prompt_metadata)} entries")

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

    # Qwen3.5 is natively multimodal — vLLM tries to load vision components
    # unless told otherwise.  Monkey-patch TRL's VLLMGeneration._init_vllm to
    # inject language_model_only=True so vLLM uses text-only mode.
    model_family = str(OmegaConf.select(cfg, "model.model_family", default="") or "")
    if "qwen3.5" in model_family.lower():
        try:
            from trl.generation.vllm_generation import VLLMGeneration
            _orig_init_vllm = VLLMGeneration._init_vllm

            def _patched_init_vllm(self_vllm):
                from vllm import LLM as _LLM
                _orig_LLM = _LLM.__init__

                def _llm_init_with_text_only(llm_self, *args, **kwargs):
                    kwargs.setdefault("language_model_only", True)
                    return _orig_LLM(llm_self, *args, **kwargs)

                _LLM.__init__ = _llm_init_with_text_only
                try:
                    _orig_init_vllm(self_vllm)
                finally:
                    _LLM.__init__ = _orig_LLM

            VLLMGeneration._init_vllm = _patched_init_vllm
            print(f"[grpo_training] Patched TRL vLLM init for Qwen3.5 text-only mode")
        except Exception as e:
            print(f"[grpo_training] Warning: failed to patch TRL vLLM init: {e}")

    print(f"[grpo_training] Starting GRPO (G={training_args.num_generations}, "
          f"vllm={use_vllm}, mode={vllm_mode if use_vllm else 'N/A'})")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[grpo_training] Saved GRPO checkpoint to {output_dir}")
