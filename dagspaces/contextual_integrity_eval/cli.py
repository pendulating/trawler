import hydra
from omegaconf import DictConfig, OmegaConf
from .orchestrator import run_experiment

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Contextual Integrity benchmark evaluation CLI."""
    print(OmegaConf.to_yaml(cfg))
    run_experiment(cfg)

if __name__ == "__main__":
    main()



