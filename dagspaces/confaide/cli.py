import hydra
from omegaconf import DictConfig, OmegaConf
from .orchestrator import run_experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """CONFAIDE privacy benchmark evaluation CLI."""
    print(OmegaConf.to_yaml(cfg))
    run_experiment(cfg)


if __name__ == "__main__":
    from dagspaces.common.stage_utils import ensure_dotenv
    ensure_dotenv()
    main()
