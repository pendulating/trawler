from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm_client import health_check


@dataclass
class ServerConfig:
    mode: str
    endpoint: str
    slurm_script: str
    startup_timeout_sec: int = 180
    poll_interval_sec: int = 5
    skip_submission_if_healthy: bool = True


def _submit_slurm(script_path: str) -> str:
    completed = subprocess.run(
        ["sbatch", script_path],
        check=True,
        text=True,
        capture_output=True,
    )
    match = re.search(r"Submitted batch job (\d+)", completed.stdout)
    if not match:
        raise RuntimeError(f"Could not parse sbatch job id from output: {completed.stdout}")
    return match.group(1)


def ensure_server(cfg: ServerConfig) -> dict[str, Any]:
    mode = (cfg.mode or "external").strip().lower()
    script_exists = Path(cfg.slurm_script).exists()
    if mode not in {"external", "slurm_managed"}:
        raise ValueError(f"Unsupported server mode: {cfg.mode}")

    if mode == "external":
        if not health_check(cfg.endpoint):
            raise RuntimeError(
                f"Server is not healthy at {cfg.endpoint}. "
                "Start it first, or use server.mode=slurm_managed."
            )
        return {"mode": "external", "endpoint": cfg.endpoint, "healthy": True}

    if cfg.skip_submission_if_healthy and health_check(cfg.endpoint):
        return {
            "mode": "slurm_managed",
            "endpoint": cfg.endpoint,
            "healthy": True,
            "submitted": False,
            "note": "Endpoint already healthy; skipped sbatch submission.",
        }

    if not script_exists:
        raise FileNotFoundError(f"SLURM serve script not found: {cfg.slurm_script}")

    job_id = _submit_slurm(cfg.slurm_script)
    started = time.time()
    while time.time() - started < cfg.startup_timeout_sec:
        if health_check(cfg.endpoint):
            return {
                "mode": "slurm_managed",
                "endpoint": cfg.endpoint,
                "healthy": True,
                "submitted": True,
                "job_id": job_id,
            }
        time.sleep(cfg.poll_interval_sec)

    raise TimeoutError(
        f"Submitted job {job_id}, but server did not become healthy within "
        f"{cfg.startup_timeout_sec} seconds. No indefinite polling is performed."
    )

