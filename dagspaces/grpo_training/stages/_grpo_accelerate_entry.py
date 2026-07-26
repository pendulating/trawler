"""Entry point for accelerate-launched GRPO training (TP>1 colocate lane).

Called by `accelerate launch --num_processes <tp>` from GRPOTrainingRunner when
the training config requests vLLM tensor-parallel > 1 in colocate mode. TRL's
colocate vLLM requires the accelerate world size to be divisible by
`vllm_tensor_parallel_size` (VLLMGeneration._init_vllm raises otherwise), so a
TP=2 colocate cell MUST run as 2 distributed processes — a single-process direct
call fails with `tensor_parallel_size (2) must divide world size (1) evenly`.

This mirrors `_sft_accelerate_entry.py` (the SFT DDP path): it deserializes the
resolved Hydra config from a temp JSON and calls run_grpo_training_stage() in
every rank. Rank-safety of the pre-trainer merge is handled inside the stage
(rank-local merged-model paths when WORLD_SIZE>1); the GRPOTrainer/accelerate
machinery handles the training loop, and HF Trainer rank-guards checkpoint saves.

The single-GPU (TP=1) path never reaches this script — the runner calls
run_grpo_training_stage directly, byte-identical to the keeper.
"""

import argparse
import json

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--chunks_path", required=True)
    parser.add_argument("--norm_universes_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cfg_path", required=True)
    parser.add_argument("--embeddings_dir", default="")
    parser.add_argument("--reward_cache_path", default="")
    parser.add_argument("--vignette_norm_universes_path", default="")
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg_dict = json.load(f)
    cfg = OmegaConf.create(cfg_dict)

    from dagspaces.grpo_training.stages.grpo_training import run_grpo_training_stage

    run_grpo_training_stage(
        sft_checkpoint=args.sft_checkpoint,
        chunks_path=args.chunks_path,
        norm_universes_path=args.norm_universes_path,
        output_dir=args.output_dir,
        cfg=cfg,
        embeddings_dir=args.embeddings_dir,
        reward_cache_path=args.reward_cache_path,
        vignette_norm_universes_path=args.vignette_norm_universes_path,
    )


if __name__ == "__main__":
    main()
