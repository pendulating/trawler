from __future__ import annotations

import argparse
import json
from typing import Any

from prompt_dev.orchestrator import run_step


def run_from_cli(step_key: str, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--theory", required=True, choices=["CI", "IG"])
    parser.add_argument("--config", default="prompt_dev/conf/config.yaml")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    out: dict[str, Any] = run_step(
        step_key=step_key,
        theory=args.theory,
        config_path=args.config,
        run_id=args.run_id,
    )
    print(json.dumps(out, indent=2))

