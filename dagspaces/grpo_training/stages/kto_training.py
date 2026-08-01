"""k-series KTO training stage (wiki/2026-07-31_kto_plan.md §6–§7, K2).

Wraps TRL 1.8.0 ``KTOTrainer`` for the three supervision-depth arms
(K-VERDICT / K-CITATION / K-SCRUTINIZE) and ``SFTTrainer`` for the
mandatory SFT-CTRL arm (K-CITATION's desirable rows only, plain SFT
loss). All four arms consume the SAME dataset bytes: string-format
``prompt``/``completion`` rows pass through TRL untouched (verified
against the installed source, K2 review 2026-08-01):

  * no chat re-templating — the stored prompts are the byte-exact rollout
    render (chat template + empty-think sentinel), so train bytes ≡ serve
    bytes by construction;
  * TRL appends ``tokenizer.eos_token`` to completions that lack it —
    exactly once (the build's completions never end with eos, asserted
    here), so the stop token sits inside the trained span;
  * SFT-CTRL uses the prompt-completion path with completion-only loss —
    the same masking semantics as the KTO arms' completion scoring.

Carry-forwards from the K1 review (2026-08-01), encoded as invariants:
  1. only ``prompt``/``completion``/``label`` reach the trainer (extra
     dataset columns are dropped before ``Dataset`` creation);
  2. the train dataset is pre-shuffled (seeded) and the default random
     sampler is left ON — KTO's batch-KL estimate needs class-mixed
     batches, and the parquet is ordered by chunk/recipe;
  3. dataset identity is asserted strictly: recomputed composition
     fingerprint == metadata fingerprint AND realized row/class counts
     == the metadata's, per arm. A mismatched or regenerated dataset
     refuses to train.

Additive k-series code (parallel-stack rule): no sft/grpo surface edited.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

#: KTO ladder arms -> the edit depth whose rows they train on. SFT-CTRL
#: trains on the PRIMARY arm's (citation) desirable rows only (plan §7).
ARM_DEPTH = {
    "verdict": "verdict",
    "citation": "citation",
    "scrutinize": "scrutinize",
    "sft_ctrl": "citation",
}


def compute_fingerprint(rows: pd.DataFrame) -> str:
    """The build's composition fingerprint (kto_data_prep, same hash)."""
    return hashlib.sha1(
        pd.util.hash_pandas_object(
            rows[["chunk_key", "label", "recipe", "depth"]]
        ).values.tobytes()).hexdigest()[:12]


def assert_dataset_identity(rows: pd.DataFrame, metadata: dict) -> None:
    """Refuse to train on a dataset that isn't the one the metadata
    describes (K1-review carry-forward #3: the fingerprint alone covers
    composition, not text — pair it with realized counts)."""
    fp = compute_fingerprint(rows)
    if fp != metadata["fingerprint"]:
        raise ValueError(
            f"[kto_training] dataset fingerprint {fp} != metadata "
            f"{metadata['fingerprint']} — the parquet is not the dataset "
            "this metadata describes")
    stats = metadata["recipe_stats"]
    emitted = ("mine_desirable", "undesirable", "edit_verdict",
               "edit_citation", "edit_scrutinize", "abstain_undesirable",
               "abstain_desirable_sampled", "abstain_desirable_synth")
    expected = sum(stats.get(k, 0) for k in emitted)
    if len(rows) != expected:
        raise ValueError(
            f"[kto_training] {len(rows)} rows != {expected} expected from "
            "metadata recipe_stats — truncated or extended parquet")


def select_arm_rows(rows: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Rows for one arm: shared streams + the arm's edit depth; SFT-CTRL
    = K-CITATION's desirables only (plan §7)."""
    if arm not in ARM_DEPTH:
        raise ValueError(f"[kto_training] unknown arm {arm!r} "
                         f"(expected one of {sorted(ARM_DEPTH)})")
    depth = ARM_DEPTH[arm]
    out = rows[(rows["depth"] == "shared") | (rows["depth"] == depth)]
    if arm == "sft_ctrl":
        out = out[out["label"]]
    return out.reset_index(drop=True)


def assert_arm_composition(arm_rows: pd.DataFrame, arm: str,
                           metadata: dict) -> dict:
    """Check realized arm class counts against the metadata and return the
    metadata's weight entry for the arm's depth."""
    w = metadata["arm_class_weights"][ARM_DEPTH[arm]]
    n_d = int(arm_rows["label"].sum())
    n_u = int((~arm_rows["label"]).sum())
    if arm == "sft_ctrl":
        if n_u:
            raise ValueError(f"[kto_training] SFT-CTRL selected {n_u} "
                             "undesirable rows — selection regressed")
        if n_d != w["n_desirable"]:
            raise ValueError(
                f"[kto_training] SFT-CTRL {n_d} desirables != metadata "
                f"{w['n_desirable']}")
    elif (n_d, n_u) != (w["n_desirable"], w["n_undesirable"]):
        raise ValueError(
            f"[kto_training] arm {arm}: realized {n_d}D/{n_u}U != metadata "
            f"{w['n_desirable']}D/{w['n_undesirable']}U")
    return w


def compute_save_steps(n_rows: int, per_device_bs: int, grad_accum: int,
                       save_frac: float = 0.10) -> tuple[int, int]:
    """(total optimizer steps for 1 epoch, save-every) at ``save_frac``."""
    eff = max(1, per_device_bs * grad_accum)
    total = max(1, (n_rows + eff - 1) // eff)
    return total, max(1, round(total * save_frac))


def _append_trace(path: str, entry: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _make_trace_callback(trace_path: str):
    """Step traces to JSONL: loss/lr/grad + KTO's reward/KL telemetry
    (`kl` collapse or explosion is the §11 reference-point risk made
    visible). Uniform cadence — verdicts are read from full curves, never
    trailing windows."""
    from transformers import TrainerCallback

    class KTOTraceCallback(TrainerCallback):
        def __init__(self):
            self._t0 = time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None or not state.is_world_process_zero:
                return
            _append_trace(trace_path, {
                "type": "step",
                "global_step": state.global_step,
                "wall_seconds": round(time.time() - self._t0, 1),
                **{k: v for k, v in logs.items()
                   if k in ("loss", "grad_norm", "learning_rate", "kl",
                            "rewards/chosen", "rewards/rejected",
                            "rewards/margins", "logps/chosen",
                            "logps/rejected", "epoch")},
            })

        def on_train_end(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            _append_trace(trace_path, {
                "type": "final", "global_step": state.global_step,
                "total_wall_seconds": round(time.time() - self._t0, 1),
            })

    return KTOTraceCallback()


def run_kto_training_stage(
    dataset_path: str,
    base_model: str,
    output_dir: str,
    cfg: Any,
    metadata_path: str | None = None,
) -> dict[str, Any]:
    """Train one k-series arm. Returns the run metadata dict."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kto_cfg = OmegaConf.to_container(
        OmegaConf.select(cfg, "training.kto"), resolve=True) or {}
    arm = str(kto_cfg.get("arm", "citation"))
    seed = int(kto_cfg.get("seed", 42))

    # ---- dataset + identity ----------------------------------------------
    rows = pd.read_parquet(dataset_path)
    if metadata_path is None:
        metadata_path = os.path.join(
            os.path.dirname(dataset_path), "kto_metadata.json")
    metadata = json.load(open(metadata_path))
    assert_dataset_identity(rows, metadata)
    arm_rows = select_arm_rows(rows, arm)
    weights = assert_arm_composition(arm_rows, arm, metadata)
    print(f"[kto_training] arm={arm}: {len(arm_rows)} rows "
          f"({int(arm_rows['label'].sum())}D/"
          f"{int((~arm_rows['label']).sum())}U), fingerprint "
          f"{metadata['fingerprint']}")

    # Smoke-run subsampling AFTER the identity/composition asserts — a
    # smoke never weakens the checks, it only trains on less.
    sample_fraction = kto_cfg.get("sample_fraction")
    if sample_fraction and 0.0 < float(sample_fraction) < 1.0:
        arm_rows = arm_rows.sample(
            frac=float(sample_fraction), random_state=seed,
        ).reset_index(drop=True)
        print(f"[kto_training] SMOKE: subsampled to {len(arm_rows)} rows "
              f"({float(sample_fraction):.0%}) — not a production run")

    # ---- tokenizer + eos invariants --------------------------------------
    from dagspaces.common.model_registry import resolve_model_source
    base_model = resolve_model_source(base_model, stage_name="kto_training")
    tokenizer = AutoTokenizer.from_pretrained(base_model,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token
    n_pre_eos = int(arm_rows["completion"].str.endswith(eos).sum())
    if n_pre_eos:
        raise ValueError(
            f"[kto_training] {n_pre_eos} completions already end with "
            f"{eos!r} — TRL appends eos itself; double-append would "
            "corrupt the stop-token supervision")

    # ---- carry-forward #1: three columns only; #2: seeded pre-shuffle ----
    cols = (["prompt", "completion"] if arm == "sft_ctrl"
            else ["prompt", "completion", "label"])
    ds = Dataset.from_pandas(arm_rows[cols]).shuffle(seed=seed)

    # ---- model + LoRA (capacity constant across arms, plan §6) -----------
    lora = kto_cfg.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora.get("rank", 64)),
        lora_alpha=int(lora.get("alpha", 128)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=lora.get("target_modules", "all-linear"),
        task_type=TaskType.CAUSAL_LM,
    )
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    print(f"[kto_training] base={base_model} attn={attn}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation=attn, low_cpu_mem_usage=True)

    # ---- schedule (plan §6: 1 epoch, save every 10%) ---------------------
    per_bs = int(kto_cfg.get("per_device_batch_size", 4))
    accum = int(kto_cfg.get("gradient_accumulation_steps", 8))
    total_steps, save_steps = compute_save_steps(
        len(ds), per_bs, accum, float(kto_cfg.get("save_frac", 0.10)))
    print(f"[kto_training] ~{total_steps} steps, save every {save_steps}")

    common_args = dict(
        output_dir=output_dir,
        num_train_epochs=int(kto_cfg.get("num_epochs", 1)),
        per_device_train_batch_size=per_bs,
        gradient_accumulation_steps=accum,
        learning_rate=float(kto_cfg.get("learning_rate", 5e-6)),
        lr_scheduler_type=str(kto_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(kto_cfg.get("warmup_ratio", 0.1)),
        max_grad_norm=float(kto_cfg.get("max_grad_norm", 1.0)),
        gradient_checkpointing=bool(
            kto_cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        logging_steps=int(kto_cfg.get("logging_steps", 10)),
        save_strategy="steps",
        save_steps=save_steps,
        seed=seed,
        report_to=("wandb" if OmegaConf.select(cfg, "wandb.enabled")
                   else "none"),
        dataloader_num_workers=int(kto_cfg.get("dataloader_num_workers", 2)),
    )
    trace_path = os.path.join(output_dir, "kto_traces.jsonl")
    callbacks = [_make_trace_callback(trace_path)]

    if arm == "sft_ctrl":
        from trl import SFTConfig, SFTTrainer
        args = SFTConfig(
            **common_args,
            max_length=int(kto_cfg.get("max_length", 4096)),
            # Explicit (TRL would infer it for prompt-completion data):
            # loss on the completion span only, like the KTO arms.
            completion_only_loss=True,
        )
        trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                             peft_config=peft_config,
                             processing_class=tokenizer,
                             callbacks=callbacks)
    else:
        from trl import KTOConfig, KTOTrainer
        args = KTOConfig(
            **common_args,
            beta=float(kto_cfg.get("beta", 0.1)),
            desirable_weight=float(weights["desirable_weight"]),
            undesirable_weight=float(weights["undesirable_weight"]),
            max_length=int(kto_cfg.get("max_length", 4096)),
            max_prompt_length=int(kto_cfg.get("max_prompt_length", 2048)),
        )
        # ref_model=None + peft_config: TRL uses the adapter-disabled base
        # as the implicit reference — no second model in memory.
        trainer = KTOTrainer(model=model, ref_model=None, args=args,
                             train_dataset=ds, peft_config=peft_config,
                             processing_class=tokenizer,
                             callbacks=callbacks)

    _append_trace(trace_path, {
        "type": "init", "arm": arm, "base_model": base_model,
        "n_rows": len(ds),
        "n_desirable": int(arm_rows["label"].sum()),
        "n_undesirable": int((~arm_rows["label"]).sum()),
        "weights": (None if arm == "sft_ctrl" else weights),
        "beta": (None if arm == "sft_ctrl"
                 else float(kto_cfg.get("beta", 0.1))),
        "total_steps_est": total_steps, "save_steps": save_steps,
        "lora_rank": peft_config.r, "seed": seed,
        "dataset_fingerprint": metadata["fingerprint"],
        "dataset_path": str(dataset_path),
    })

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    run_meta = {
        "arm": arm, "base_model": base_model,
        "dataset_path": str(dataset_path),
        "dataset_fingerprint": metadata["fingerprint"],
        "n_rows": len(ds), "save_steps": save_steps,
        "checkpoint_dir": output_dir,
    }
    with open(os.path.join(output_dir, "kto_run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"[kto_training] saved arm={arm} adapter -> {output_dir}")
    return run_meta
