"""Eval-all orchestrator: dispatches to each eval dagspace as a subprocess.

When ``server_mode.enabled=true``, a shared vLLM OpenAI-compatible server
is launched once for the whole run and each benchmark subprocess routes
its inference through it via ``VLLM_SERVER_URL`` (see
``dagspaces/eval_all/server.py``).

When ``judge_sidecar.enabled=true``, a small CPU sidecar process is
launched once per run that watches each benchmark's output dir for
``judge_*/manifest.json`` files and forwards each row's request body
to the cluster judge endpoint. Judged benchmarks (currently just
privacylens) run their async-export pipelines, then a drain phase
waits for ``done.flag`` files, then per-benchmark finalize pipelines
parse + compute_metrics. See :mod:`dagspaces.eval_all.judge_sidecar`.
"""

import atexit
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

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

    # Async-judge sidecar (optional): launch ONE small CPU subprocess that
    # watches every benchmark's output dir for judge manifests and forwards
    # them to the cluster judge endpoint. Judged benchmarks run their
    # *_async pipelines (export-only, no compute_metrics), then we drain,
    # then we run their *_async_finalize pipelines.
    sidecar_proc: Optional[subprocess.Popen] = None
    sidecar_cfg = OmegaConf.select(cfg, "judge_sidecar")
    if sidecar_cfg is not None and bool(sidecar_cfg.get("enabled", False)):
        sidecar_proc = _launch_judge_sidecar(
            sidecar_cfg=sidecar_cfg,
            watch_root=parent_output_dir,
            child_env=child_env,
        )

        def _cleanup_sidecar(*_args: Any, **_kwargs: Any) -> None:
            _terminate_judge_sidecar(sidecar_proc)
        atexit.register(_cleanup_sidecar)
        for _sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(_sig, lambda s, f: (_cleanup_sidecar(), sys.exit(130)))
            except Exception:
                pass

    results = {}
    try:
        for bench_name, bench_cfg in benchmarks.items():
            module = bench_cfg["module"]
            pipeline = bench_cfg["pipeline"]
            vlm_only = bench_cfg.get("vlm_only", False)
            extra_args = bench_cfg.get("extra_args") or []

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
            # Per-benchmark Hydra overrides (e.g. judge.mode=batch_export for
            # the all_benchmarks_batch_export variant).
            for extra in extra_args:
                cmd.append(str(extra))

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
        # Phase 2 (drain) and Phase 3 (finalize) for the async-judge flow.
        # Run BEFORE shutting down the task vLLM server so finalize stages
        # that re-use it (none today, but defensive) still see it.
        finalize_results: Dict[str, str] = {}
        if sidecar_proc is not None:
            try:
                drained = _wait_for_judge_drain(
                    parent_output_dir,
                    sidecar_cfg=sidecar_cfg,
                )
                if not drained:
                    print(
                        "[eval_all] WARNING: judge drain timed out — finalize "
                        "stages may fail or be skipped.",
                        file=sys.stderr,
                    )
                finalize_results = _run_judged_finalize(
                    benchmarks=benchmarks,
                    benchmark_results=results,
                    model_name=model_name,
                    parent_output_dir=parent_output_dir,
                    child_env=child_env,
                    cfg=cfg,
                    wandb_project=wandb_project,
                    debug=debug,
                    sample_n=sample_n,
                )
            finally:
                _terminate_judge_sidecar(sidecar_proc)

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
    if finalize_results:
        print(f"\n  Finalize phase:")
        for bench, status in finalize_results.items():
            marker = "OK" if status == "ok" else "SKIP" if status.startswith("skipped") else "FAIL"
            print(f"  [{marker:>4}] {bench} (finalize): {status}")

    failed = [b for b, s in results.items() if s not in ("ok", "skipped")]
    failed += [
        f"{b} (finalize)"
        for b, s in finalize_results.items()
        if s not in ("ok",) and not s.startswith("skipped")
    ]
    if failed:
        raise RuntimeError(f"Benchmarks failed: {', '.join(failed)}")


# ---------------------------------------------------------------------------
# Async-judge sidecar lifecycle
# ---------------------------------------------------------------------------

def _launch_judge_sidecar(
    *,
    sidecar_cfg: Any,
    watch_root: str,
    child_env: Dict[str, str],
) -> Optional[subprocess.Popen]:
    """Spawn the judge sidecar as a same-process-tree subprocess.

    Returns the Popen handle so the parent can terminate it on exit.
    Returns ``None`` if launch fails — callers must tolerate this and
    fall back to "no async judging" (judged benchmarks will fail their
    finalize phase, which we report rather than treat as a hard error).
    """
    base_url = str(sidecar_cfg.get("base_url") or "")
    if not base_url:
        print(
            "[eval_all] judge_sidecar.enabled=true but base_url is empty — "
            "skipping sidecar launch. Set judge_sidecar.base_url or the "
            "JUDGE_BASE_URL env var.",
            file=sys.stderr,
        )
        return None

    cmd: List[str] = [
        sys.executable, "-m", "dagspaces.eval_all.judge_sidecar", "run",
        "--run-dir", watch_root,
        "--base-url", base_url,
    ]
    model = str(sidecar_cfg.get("model") or "")
    if model:
        cmd += ["--model", model]
    concurrency = sidecar_cfg.get("concurrency")
    if concurrency is not None:
        cmd += ["--concurrency", str(int(concurrency))]
    parallel = sidecar_cfg.get("parallel_manifests")
    if parallel is not None:
        cmd += ["--parallel-manifests", str(int(parallel))]
    timeout = sidecar_cfg.get("timeout")
    if timeout is not None:
        cmd += ["--timeout", str(float(timeout))]
    api_key_env = sidecar_cfg.get("api_key_env")
    if api_key_env:
        cmd += ["--api-key-env", str(api_key_env)]

    log_dir = os.path.join(watch_root, ".sidecar_logs")
    os.makedirs(log_dir, exist_ok=True)
    stdout_path = os.path.join(log_dir, "sidecar.out")
    stderr_path = os.path.join(log_dir, "sidecar.err")

    print(f"\n{'='*60}\nASYNC JUDGE: launching sidecar\n{'='*60}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  logs: {stdout_path}, {stderr_path}")
    try:
        # start_new_session lets us SIGTERM the whole sidecar process
        # group (the worker pool threads live inside it).
        proc = subprocess.Popen(
            cmd,
            env=child_env,
            stdout=open(stdout_path, "w"),
            stderr=open(stderr_path, "w"),
            start_new_session=True,
        )
        print(f"  sidecar pid: {proc.pid}\n")
        return proc
    except Exception as exc:
        print(f"[eval_all] sidecar launch failed: {exc}", file=sys.stderr)
        return None


def _terminate_judge_sidecar(proc: Optional[subprocess.Popen]) -> None:
    """SIGTERM the sidecar, SIGKILL if it doesn't exit in 30s."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _wait_for_judge_drain(
    watch_root: str,
    *,
    sidecar_cfg: Any,
) -> bool:
    """Block until every emitted judge manifest has a ``done.flag``.

    Loud progress on each poll so the operator can see the queue
    draining. Returns True on clean drain, False on timeout. Reuses
    the sidecar module's wait_for_drain helper so the polling logic
    stays in one place.
    """
    from dagspaces.eval_all.judge_sidecar import wait_for_drain

    timeout_s = float(
        sidecar_cfg.get("drain_timeout_s")
        if sidecar_cfg.get("drain_timeout_s") is not None
        else 6 * 60 * 60
    )
    poll_s = float(sidecar_cfg.get("drain_poll_interval_s") or 30.0)

    print(f"\n{'='*60}\nASYNC JUDGE: draining sidecar (timeout {timeout_s/60:.0f}m)\n{'='*60}")
    last_pending = -1

    def _on_tick(pending: int, total: int) -> None:
        nonlocal last_pending
        if pending != last_pending:
            print(
                f"[eval_all][drain] pending={pending} / total_manifests={total}",
                flush=True,
            )
            last_pending = pending

    drained = wait_for_drain(
        watch_root,
        timeout_s=timeout_s,
        poll_interval_s=poll_s,
        on_tick=_on_tick,
    )
    if drained:
        print("[eval_all] judge drain complete.\n", flush=True)
    return drained


def _run_judged_finalize(
    *,
    benchmarks: Dict[str, Any],
    benchmark_results: Dict[str, str],
    model_name: str,
    parent_output_dir: str,
    child_env: Dict[str, str],
    cfg: DictConfig,
    wandb_project: str,
    debug: bool,
    sample_n: Any,
) -> Dict[str, str]:
    """Dispatch each judged benchmark's ``finalize_pipeline`` as a child
    CLI invocation against the same per-benchmark output dir as the
    export pipeline used.

    Loudly skips (with a banner) any benchmark whose export was not
    successful or whose judge manifests are missing — those become
    ``skipped:<reason>`` entries instead of hard failures, so a single
    flaky judge run doesn't kill the whole sweep summary.
    """
    finalize_results: Dict[str, str] = {}
    for bench_name, bench_cfg in benchmarks.items():
        finalize_pipeline = bench_cfg.get("finalize_pipeline")
        if not finalize_pipeline:
            continue

        export_status = benchmark_results.get(bench_name, "missing")
        if export_status != "ok":
            print(f"\n{'!'*60}")
            print(f"  SKIP finalize for {bench_name}: export status = {export_status}")
            print(f"{'!'*60}\n")
            finalize_results[bench_name] = f"skipped:export_{export_status}"
            continue

        child_output_dir = os.path.join(parent_output_dir, bench_name)
        if not _judged_run_has_manifests(child_output_dir):
            print(f"\n{'!'*60}")
            print(f"  SKIP finalize for {bench_name}: no judge manifests on disk under {child_output_dir}")
            print(f"  (export pipeline may not have written them; check sidecar logs)")
            print(f"{'!'*60}\n")
            finalize_results[bench_name] = "skipped:no_manifest"
            continue

        cmd = [
            sys.executable, "-m", bench_cfg["module"],
            f"pipeline={finalize_pipeline}",
            f"model={model_name}",
            f"wandb.project={wandb_project}",
            f"hydra.run.dir={child_output_dir}",
        ]
        if debug:
            cmd.append("runtime.debug=true")
        if sample_n is not None:
            cmd.append(f"runtime.sample_n={sample_n}")

        print(f"\n{'='*60}")
        print(f"FINALIZE {bench_name} | pipeline={finalize_pipeline}")
        print(f"  cmd: {' '.join(cmd)}")
        print(f"{'='*60}\n")

        t0 = time.time()
        proc = subprocess.run(cmd, env=child_env)
        elapsed = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        finalize_results[bench_name] = status
        print(f"\n  {bench_name} finalize finished in {elapsed:.0f}s — {status}")
    return finalize_results


def _judged_run_has_manifests(child_output_dir: str) -> bool:
    """True if at least one ``judge_*/manifest.json`` lives under this run."""
    if not os.path.isdir(child_output_dir):
        return False
    for root, dirs, _files in os.walk(child_output_dir):
        # Sidecar manifests live under outputs/judge_*/ — only walk one
        # level under outputs to keep this O(benchmarks).
        for d in list(dirs):
            if d.startswith("judge_") or d.endswith("_judge_batch"):
                if os.path.exists(os.path.join(root, d, "manifest.json")):
                    return True
    return False


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
