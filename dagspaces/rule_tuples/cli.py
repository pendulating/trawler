import os
import hydra
from omegaconf import DictConfig

from .orchestrator import run_experiment


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        if hasattr(cfg, "wandb") and hasattr(cfg.wandb, "enabled") and not bool(cfg.wandb.enabled):
            os.environ["WANDB_DISABLED"] = "true"
        if hasattr(cfg, "wandb") and getattr(cfg.wandb, "name_prefix", None):
            os.environ.setdefault("WANDB_NAME_PREFIX", str(cfg.wandb.name_prefix))
    except Exception:
        pass
    run_experiment(cfg)


if __name__ == "__main__":
    main()


