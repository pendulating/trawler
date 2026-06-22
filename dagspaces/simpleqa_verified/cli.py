import hydra
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.stage_utils import ensure_dotenv
from .orchestrator import run_experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """SimpleQA-Verified factuality evaluation CLI."""
    ensure_dotenv()
    print(OmegaConf.to_yaml(cfg))
    run_experiment(cfg)


if __name__ == "__main__":
    main()
