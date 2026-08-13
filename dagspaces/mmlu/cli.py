"""MMLU multitask knowledge evaluation CLI."""

from dagspaces.common.cli import make_cli

from .orchestrator import run_experiment

main = make_cli(run_experiment)


if __name__ == "__main__":
    main()
