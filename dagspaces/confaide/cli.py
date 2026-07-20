import hydra
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.stage_utils import ensure_dotenv

from .orchestrator import run_experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """CONFAIDE privacy benchmark evaluation CLI."""
    # Idempotent. Must run inside main() so it fires in submitit-launched
    # workers too (those import the module rather than executing __main__,
    # so a __main__-guarded call would skip them and downstream
    # ${oc.env:...}-style interpolations would fail.)
    ensure_dotenv()
    print(OmegaConf.to_yaml(cfg))
    run_experiment(cfg)


if __name__ == "__main__":
    main()
