"""Eval-all orchestrator: dispatches to each eval dagspace as a subprocess.

When ``server_mode.enabled=true``, a shared vLLM OpenAI-compatible server
is launched once for the whole run and each benchmark subprocess routes
its inference through it via ``VLLM_SERVER_URL`` (see
``dagspaces/eval_all/server.py``).
"""

import atexit
import os
import signal
import subprocess
import sys
import time

from omegaconf import DictConfig, OmegaConf


# Model families with VLM prompt builders in vlm_geoprivacy_bench.
# Must match keys in dagspaces/vlm_geoprivacy_bench/model_prompts.py PROMPT_BUILDERS.
_VLM_FAMILIES = {"qwen2.5-vl", "qwen3-vl", "qwen3.5", "llama-vision", "gemma-3", "internvl2.5", "deepseek-vl2", "phi-4"}


def _is_vlm_model(model_cfg: DictConfig) -> bool:
    """Check if the model has a VLM prompt builder for vlm_geoprivacy_bench."""
    family = str(OmegaConf.select(model_cfg, "model_family") or "").lower()
    return family in _VLM_FAMILIES


def run_eval_all(cfg: DictConfig) -> None:
    """Run all configured benchmarks for the current model."""
    benchmarks = OmegaConf.to_container(cfg.benchmarks, resolve=True)
    model_cfg = cfg.model
    wandb_project = str(OmegaConf.select(cfg, "wandb.project") or "eval-all")

    skip_vlm = bool(OmegaConf.select(cfg, "runtime.skip_vlm") or False)
    debug = bool(OmegaConf.select(cfg, "runtime.debug") or False)
    sample_n = OmegaConf.select(cfg, "runtime.sample_n")
    is_vlm = _is_vlm_model(model_cfg)

    # Resolve the model config name from Hydra's override list.
    # Hydra stores CLI overrides in cfg; we need the short name (e.g. "qwen3.5-9b")
    # to pass to child dagspaces. Extract from the choice made in defaults.
    model_name = _resolve_model_name(cfg)

    # Resolve the parent output directory so child logs nest under eval_all.
    try:
        from hydra.core.utils import HydraConfig
        parent_output_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        parent_output_dir = os.getcwd()

    # Shared vLLM server (optional): launch once, inject URL into child env.
    server_info = None
    child_env = os.environ.copy()
    server_cfg = OmegaConf.select(cfg, "server_mode")
    if server_cfg is not None and bool(server_cfg.get("enabled", False)):
        from .server import launch_vllm_server, shutdown_vllm_server
        print(f"\n{'='*60}\nSERVER MODE: launching shared vLLM server\n{'='*60}")
        server_info = launch_vllm_server(
            cfg=cfg,
            model_cfg=model_cfg,
            server_cfg=server_cfg,
            output_dir=parent_output_dir,
            startup_timeout_s=float(server_cfg.get("startup_timeout_s", 900)),
        )
        child_env["VLLM_SERVER_URL"] = server_info["url"]

        # Ensure the server is cancelled on any exit path.
        def _cleanup(*_args, **_kwargs):
            if server_info is not None:
                shutdown_vllm_server(server_info)
        atexit.register(_cleanup)
        for _sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(_sig, lambda s, f: (_cleanup(), sys.exit(130)))
            except Exception:
                pass

    results = {}
    try:
        for bench_name, bench_cfg in benchmarks.items():
            module = bench_cfg["module"]
            pipeline = bench_cfg["pipeline"]
            vlm_only = bench_cfg.get("vlm_only", False)

            if vlm_only and (skip_vlm or not is_vlm):
                reason = "skip_vlm=true" if skip_vlm else f"{model_name} is text-only"
                print(f"\n{'='*60}")
                print(f"SKIP {bench_name} ({reason})")
                print(f"{'='*60}")
                results[bench_name] = "skipped"
                continue

            # No -m flag: eval_all is already on a SLURM node (via its own -m).
            # Each child dagspace's run_experiment() handles GPU job submission
            # internally. Passing -m here would nest submitit → submitit, which
            # fails to collect results.
            child_output_dir = os.path.join(parent_output_dir, bench_name)
            cmd = [
                sys.executable, "-m", module,
                f"pipeline={pipeline}",
                f"model={model_name}",
                f"wandb.project={wandb_project}",
                f"hydra.run.dir={child_output_dir}",
            ]
            if debug:
                cmd.append("runtime.debug=true")
            if sample_n is not None:
                cmd.append(f"runtime.sample_n={sample_n}")

            print(f"\n{'='*60}")
            print(f"START {bench_name} | model={model_name}")
            print(f"  cmd: {' '.join(cmd)}")
            print(f"{'='*60}\n")

            t0 = time.time()
            proc = subprocess.run(cmd, env=child_env)
            elapsed = time.time() - t0

            status = "ok" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            results[bench_name] = status
            print(f"\n  {bench_name} finished in {elapsed:.0f}s — {status}")
    finally:
        if server_info is not None:
            from .server import shutdown_vllm_server
            shutdown_vllm_server(server_info)

    # Summary
    print(f"\n{'='*60}")
    print(f"EVAL SUMMARY | model={model_name}")
    print(f"{'='*60}")
    for bench, status in results.items():
        marker = "OK" if status == "ok" else "SKIP" if status == "skipped" else "FAIL"
        print(f"  [{marker:>4}] {bench}: {status}")

    failed = [b for b, s in results.items() if s not in ("ok", "skipped")]
    if failed:
        raise RuntimeError(f"Benchmarks failed: {', '.join(failed)}")


def _resolve_model_name(cfg: DictConfig) -> str:
    """Extract the model config name from Hydra choices or model_source path."""
    # Use HydraConfig (available inside a Hydra job, including SLURM)
    try:
        from hydra.core.utils import HydraConfig
        choices = HydraConfig.get().runtime.choices
        if "model" in choices:
            return str(choices["model"])
    except Exception:
        pass
    # Fall back to basename of model_source
    import os
    source = str(OmegaConf.select(cfg, "model.model_source") or "")
    return os.path.basename(source).lower().replace(" ", "-") if source else "unknown"
