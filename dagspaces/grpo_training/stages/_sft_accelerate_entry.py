"""Entry point for accelerate-launched SFT training.

This script is called by `accelerate launch` from the SFTTrainingRunner.
It deserializes the config and calls run_sft_training_stage().
"""

import argparse
import json

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cfg_path", required=True)
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg_dict = json.load(f)
    cfg = OmegaConf.create(cfg_dict)

    from dagspaces.grpo_training.stages.sft_training import run_sft_training_stage

    run_sft_training_stage(
        dataset_path=args.dataset_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
