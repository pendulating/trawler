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
from typing import Any

from omegaconf import DictConfig, OmegaConf

# Model families with VLM prompt builders in vlm_geoprivacy_bench.
# Must match keys in dagspaces/vlm_geoprivacy_bench/model_prompts.py PROMPT_BUILDERS.
# NOTE: "phi-4" is intentionally absent. The local /share/.../zoo/models/Phi-4
# weights are the *text-only* Phi-4, not Phi-4-multimodal — running the VLM
# benchmark against it crashes vLLM's multimodal renderer
# ("'HfRenderer' object has no attribute '_mm_req_counter'"). Re-add only if
# a genuine multimodal Phi-4 checkpoint is wired up.
# NOTE: "gemma-4" added 2026-07-18. The whole gemma-4 line (31B / 12B / E2B /
# E4B) is any-to-any multimodal, but the family was missing here, so every
# gemma-4 cell in the 2026-07-1{6,7} canonical sweeps silently self-skipped
# vlm_geoprivacy as "text-only" and has no Q7 value. Verified before enabling:
# all four checkpoints render the `<|image|>` token via build_gemma4_prompt.
_VLM_FAMILIES = {"qwen2.5-vl", "qwen3-vl", "qwen3.5", "llama-vision", "gemma-3", "gemma-4", "internvl2.5", "deepseek-vl2"}


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

    # Meta benchmark whitelist/blacklist. See conf/benchmark_filter/*.yaml
    # for named reusable filters. Precedence in the dispatch loop below:
    # explicit-disable > include-whitelist > exclude-blacklist > vlm_only.
    benchmark_include = list(OmegaConf.select(cfg, "benchmark_filter.include") or [])
    benchmark_exclude = list(OmegaConf.select(cfg, "benchmark_filter.exclude") or [])

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
    # Sweep identity for W&B: when the operator didn't export WANDB_GROUP,
    # derive it from this arm's output path (multirun/<sweep>/<time>/<arm> →
    # "<sweep>/<time>") and pin it for every child benchmark and the judge
    # sidecar. This is what makes each child run carry the eval_all_run:
    # tag and a stable resumable id — i.e. what lets W&B be scoped back to
    # this multirun dir at analysis time.
    if not (child_env.get("WANDB_GROUP") or "").strip():
        try:
            from dagspaces.common.metrics_sync import derive_group_from_output_dir

            _derived_group = derive_group_from_output_dir(parent_output_dir)
        except Exception:
            _derived_group = None
        if _derived_group:
            child_env["WANDB_GROUP"] = _derived_group
            os.environ["WANDB_GROUP"] = _derived_group  # summary run + tags
            print(f"[eval_all] WANDB_GROUP not set — derived '{_derived_group}' "
                  "from the output path")
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
        # Identity guard: run_vllm_inference only routes to the server when
        # the stage's model matches this — otherwise a stage running a
        # DIFFERENT model (e.g. vlm_geoprivacy's granularity judge) would be
        # silently hijacked to the task model.
        child_env["VLLM_SERVER_MODEL"] = str(server_info["canonical_source"])
        if server_info.get("lora_name"):
            child_env["VLLM_SERVER_LORA_NAME"] = str(server_info["lora_name"])
        # Downgrade server-routable inference nodes to a CPU launcher: their
        # work is HTTP calls now, and holding a slurm_gpu_1x allocation per
        # stage while the server owns the GPU would DOUBLE the footprint.
        # Pipeline yamls opt in via
        #   launcher: ${oc.env:TRAWLER_EVAL_INFER_LAUNCHER,slurm_gpu_1x}
        # Stages that keep a local engine (vlm inference, judge stages with a
        # different model) keep a plain slurm_gpu_1x and are unaffected.
        child_env["TRAWLER_EVAL_INFER_LAUNCHER"] = str(
            server_cfg.get("infer_launcher", "slurm_cpu")
        )

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

    # eval_all_summary parent W&B run — opens before any benchmark
    # dispatch so the per-benchmark timings get logged as events into
    # one place. Stable id keyed on the WANDB_GROUP so a SLURM-requeued
    # eval_all monitor reattaches instead of forking the timeline.
    summary_run = _open_eval_all_summary_run(
        cfg=cfg,
        wandb_project=wandb_project,
        model_name=model_name,
    )

    # Async-judge sidecar (optional): launch ONE small CPU subprocess that
    # watches every benchmark's output dir for judge manifests and forwards
    # them to the cluster judge endpoint. Judged benchmarks run their
    # *_async pipelines (export-only, no compute_metrics), then we drain,
    # then we run their *_async_finalize pipelines.
    sidecar_proc: subprocess.Popen | None = None
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
    durations: dict[str, float] = {}
    drain_seconds: float = 0.0
    try:
        for bench_name, bench_cfg in benchmarks.items():
            module = bench_cfg["module"]
            pipeline = bench_cfg["pipeline"]
            vlm_only = bench_cfg.get("vlm_only", False)
            enabled = bool(bench_cfg.get("enabled", True))
            extra_args = bench_cfg.get("extra_args") or []

            # Skip precedence (documented in conf/config.yaml):
            #   1. enabled=false on the benchmark entry           — always wins
            #   2. benchmark_filter.include non-empty whitelist
            #   3. benchmark_filter.exclude blacklist
            #   4. vlm_only constraint vs. the model family
            skip_reason: str | None = None
            if not enabled:
                skip_reason = "enabled=false"
            elif benchmark_include and bench_name not in benchmark_include:
                skip_reason = f"not in benchmark_filter.include={benchmark_include}"
            elif benchmark_exclude and bench_name in benchmark_exclude:
                skip_reason = f"in benchmark_filter.exclude={benchmark_exclude}"
            elif vlm_only and (skip_vlm or not is_vlm):
                skip_reason = "skip_vlm=true" if skip_vlm else f"{model_name} is text-only"

            if skip_reason:
                print(f"\n{'='*60}")
                print(f"SKIP {bench_name} ({skip_reason})")
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
            durations[bench_name] = elapsed
            print(f"\n  {bench_name} finished in {elapsed:.0f}s — {status}")
    finally:
        # Phase 2 (drain) and Phase 3 (finalize) for the async-judge flow.
        # Run BEFORE shutting down the task vLLM server so finalize stages
        # that re-use it (none today, but defensive) still see it.
        finalize_results: dict[str, str] = {}
        finalize_durations: dict[str, float] = {}
        if sidecar_proc is not None:
            try:
                _drain_t0 = time.time()
                drained = _wait_for_judge_drain(
                    parent_output_dir,
                    sidecar_cfg=sidecar_cfg,
                )
                drain_seconds = time.time() - _drain_t0
                if not drained:
                    print(
                        "[eval_all] WARNING: judge drain timed out — finalize "
                        "stages may fail or be skipped.",
                        file=sys.stderr,
                    )
                finalize_results, finalize_durations = _run_judged_finalize(
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

    # Close out the summary run — builds the per-benchmark table and
    # logs aggregate scalars.
    _close_eval_all_summary_run(
        run=summary_run,
        benchmarks=benchmarks,
        dispatch_results=results,
        dispatch_durations=durations,
        finalize_results=finalize_results,
        finalize_durations=locals().get("finalize_durations", {}),
        drain_seconds=drain_seconds,
        parent_output_dir=parent_output_dir,
        model_name=model_name,
    )

    failed = [b for b, s in results.items() if s not in ("ok", "skipped")]
    failed += [
        f"{b} (finalize)"
        for b, s in finalize_results.items()
        if s not in ("ok",) and not s.startswith("skipped")
    ]

    # Always emit a machine-readable status record next to the outputs. SLURM
    # reports the eval_all job as "completed successfully" regardless of
    # per-benchmark outcomes, so without this a failed benchmark is only
    # discoverable by grepping the submitit _log.out for the EVAL SUMMARY block.
    # Downstream tooling (and the next audit) can read failures.json directly.
    try:
        import json
        status_path = os.path.join(parent_output_dir, "failures.json")
        with open(status_path, "w") as fh:
            json.dump(
                {
                    "model": model_name,
                    "dispatch": results,
                    "finalize": finalize_results,
                    "failed": failed,
                    "success": not failed,
                },
                fh,
                indent=2,
            )
        print(f"\n[eval_all] status written → {status_path}")
    except Exception as exc:  # never let bookkeeping mask the real result
        print(f"[eval_all] WARNING: could not write failures.json: {exc}",
              file=sys.stderr)

    if failed:
        raise RuntimeError(f"Benchmarks failed: {', '.join(failed)}")


# ---------------------------------------------------------------------------
# Async-judge sidecar lifecycle
# ---------------------------------------------------------------------------

def _launch_judge_sidecar(
    *,
    sidecar_cfg: Any,
    watch_root: str,
    child_env: dict[str, str],
) -> subprocess.Popen | None:
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
            "JUDGE_SERVER_URL env var (JUDGE_BASE_URL also accepted).",
            file=sys.stderr,
        )
        return None

    cmd: list[str] = [
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
    max_inflight = sidecar_cfg.get("max_inflight")
    if max_inflight is not None:
        cmd += ["--max-inflight", str(int(max_inflight))]
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


def _terminate_judge_sidecar(proc: subprocess.Popen | None) -> None:
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
    benchmarks: dict[str, Any],
    benchmark_results: dict[str, str],
    model_name: str,
    parent_output_dir: str,
    child_env: dict[str, str],
    cfg: DictConfig,
    wandb_project: str,
    debug: bool,
    sample_n: Any,
) -> "tuple[dict[str, str], dict[str, float]]":
    """Dispatch each judged benchmark's ``finalize_pipeline`` as a child
    CLI invocation against the same per-benchmark output dir as the
    export pipeline used.

    Loudly skips (with a banner) any benchmark whose export was not
    successful or whose judge manifests are missing — those become
    ``skipped:<reason>`` entries instead of hard failures, so a single
    flaky judge run doesn't kill the whole sweep summary.
    """
    finalize_results: dict[str, str] = {}
    finalize_durations: dict[str, float] = {}
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
            finalize_durations[bench_name] = 0.0
            continue

        child_output_dir = os.path.join(parent_output_dir, bench_name)
        if not _judged_run_has_manifests(child_output_dir):
            print(f"\n{'!'*60}")
            print(f"  SKIP finalize for {bench_name}: no judge manifests on disk under {child_output_dir}")
            print(f"  (export pipeline may not have written them; check sidecar logs)")
            print(f"{'!'*60}\n")
            finalize_results[bench_name] = "skipped:no_manifest"
            finalize_durations[bench_name] = 0.0
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
        # Forward the same per-benchmark overrides the export used (e.g.
        # +perturb.culture=<c>) so finalize reattaches to the export's W&B
        # run (the run id is qualified by culture) instead of forking.
        for extra in (bench_cfg.get("extra_args") or []):
            cmd.append(str(extra))

        print(f"\n{'='*60}")
        print(f"FINALIZE {bench_name} | pipeline={finalize_pipeline}")
        print(f"  cmd: {' '.join(cmd)}")
        print(f"{'='*60}\n")

        t0 = time.time()
        proc = subprocess.run(cmd, env=child_env)
        elapsed = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        finalize_results[bench_name] = status
        finalize_durations[bench_name] = elapsed
        print(f"\n  {bench_name} finalize finished in {elapsed:.0f}s — {status}")
    return finalize_results, finalize_durations


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


# ---------------------------------------------------------------------------
# eval_all_summary parent W&B run
# ---------------------------------------------------------------------------

def _open_eval_all_summary_run(
    *,
    cfg: DictConfig,
    wandb_project: str,
    model_name: str,
):
    """Open the eval_all_summary parent run. Resumable by group so a
    requeued monitor reattaches; returns None if W&B is unavailable.
    """
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    if os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1"):
        return None
    if not bool(OmegaConf.select(cfg, "wandb.enabled")):
        # Respect the project's existing wandb.enabled gate so disabled
        # runs don't accidentally still open a summary.
        return None

    from dagspaces.common.wandb_logger import derive_resumable_run_id

    group = os.environ.get("WANDB_GROUP") or None
    entity = os.environ.get("WANDB_ENTITY")
    judge_mode = str(OmegaConf.select(cfg, "judge.mode") or "live")

    run_id = derive_resumable_run_id(
        group=group, dagspace="eval_all_summary",
        model=model_name, role="summary",
    )
    tags = [
        f"bench:eval_all_summary",
        f"judge_mode:{judge_mode}",
    ]
    if group:
        tags.append(f"eval_all_run:{group}")
    if model_name:
        tags.append(f"family:{model_name}")

    init_kwargs: dict[str, Any] = {
        "project": wandb_project,
        "entity": entity,
        "group": group,
        "job_type": "eval_all_summary",
        "name": f"summary-{model_name}-{group or 'standalone'}",
        "config": {
            "model": model_name,
            "judge_mode": judge_mode,
            "benchmarks": list(OmegaConf.to_container(cfg.benchmarks, resolve=True).keys()),
        },
        "tags": tags,
        "reinit": True,
    }
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"

    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"[eval_all] summary run init failed: {exc}", file=sys.stderr)
        return None


def _read_pipeline_manifest(child_output_dir: str) -> dict[str, Any]:
    """Find any pipeline_manifest.json under a child run's output and
    return it (sanity counts live in node metadata). Tolerant: returns
    {} if not found or unparseable.
    """
    if not os.path.isdir(child_output_dir):
        return {}
    for root, dirs, files in os.walk(child_output_dir):
        if "pipeline_manifest.json" in files:
            try:
                import json as _json
                with open(os.path.join(root, "pipeline_manifest.json")) as f:
                    return _json.load(f)
            except Exception:
                continue
    return {}


def _aggregate_sanity_warnings(manifest: dict[str, Any]) -> "tuple[int, str]":
    """Count sanity warnings + extract the worst-warning string from a
    pipeline_manifest.json. Used by the summary table.
    """
    nodes = manifest.get("nodes", {}) or {}
    total = 0
    worst_msg = ""
    for node_name, node_data in nodes.items():
        meta = (node_data or {}).get("metadata") or {}
        sanity = meta.get("sanity") or {}
        for stage_name, stage in sanity.items():
            if not isinstance(stage, dict):
                continue
            warnings = stage.get("warnings") or []
            total += len(warnings)
            if warnings and not worst_msg:
                worst_msg = f"{stage_name}: {warnings[0]}"
    return total, worst_msg


def _close_eval_all_summary_run(
    *,
    run,
    benchmarks: dict[str, Any],
    dispatch_results: dict[str, str],
    dispatch_durations: dict[str, float],
    finalize_results: dict[str, str],
    finalize_durations: dict[str, float],
    drain_seconds: float,
    parent_output_dir: str,
    model_name: str,
) -> None:
    """Build the per-benchmark summary table + aggregate scalars, then finish."""
    if run is None:
        return
    try:
        import wandb  # type: ignore

        from .primary_metrics import (
            PRIMARY_METRICS,
            extract_primary_metrics,
            format_primary_metrics,
        )
    except Exception as exc:
        try:
            run.finish()
        except Exception:
            pass
        print(f"[eval_all] summary close failed early: {exc}", file=sys.stderr)
        return

    cols = [
        "benchmark", "dispatch_status", "dispatch_seconds",
        "finalize_status", "finalize_seconds",
        "primary_metrics", "sanity_warnings", "worst_sanity",
    ]
    rows: list[list[Any]] = []
    aggregate_metrics: dict[str, Any] = {}

    for bench_name in benchmarks.keys():
        child_output_dir = os.path.join(parent_output_dir, bench_name)
        # Find the inner output_root the dagspace actually wrote to.
        # Convention: <child>/<inner>/outputs/... — pick the first
        # subdir that contains "outputs/".
        bench_root = child_output_dir
        if os.path.isdir(child_output_dir):
            for sub in os.listdir(child_output_dir):
                cand = os.path.join(child_output_dir, sub)
                if os.path.isdir(os.path.join(cand, "outputs")):
                    bench_root = cand
                    break

        manifest = _read_pipeline_manifest(child_output_dir)
        sanity_count, worst = _aggregate_sanity_warnings(manifest)
        primary_vals = (
            extract_primary_metrics(bench_root, bench_name)
            if bench_name in PRIMARY_METRICS else {}
        )
        primary_fmt = format_primary_metrics(primary_vals, bench_name) if primary_vals else {}
        primary_str = ", ".join(f"{k}={v}" for k, v in primary_fmt.items()) or "—"

        rows.append([
            bench_name,
            dispatch_results.get(bench_name, "missing"),
            round(dispatch_durations.get(bench_name, 0.0), 1),
            finalize_results.get(bench_name, "—"),
            round(finalize_durations.get(bench_name, 0.0), 1),
            primary_str,
            sanity_count,
            worst or "—",
        ])

        # Per-benchmark aggregate scalars (so dashboards can plot a single
        # benchmark's headline metric across many summary runs).
        for metric_name, val in primary_vals.items():
            if val is None:
                continue
            aggregate_metrics[f"summary/{bench_name}/{metric_name}"] = float(val)
        aggregate_metrics[f"summary/{bench_name}/dispatch_seconds"] = float(
            dispatch_durations.get(bench_name, 0.0)
        )
        if bench_name in finalize_durations:
            aggregate_metrics[f"summary/{bench_name}/finalize_seconds"] = float(
                finalize_durations[bench_name]
            )
        aggregate_metrics[f"summary/{bench_name}/sanity_warnings"] = int(sanity_count)

    aggregate_metrics["summary/drain_seconds"] = float(drain_seconds)
    aggregate_metrics["summary/total_dispatch_seconds"] = float(sum(dispatch_durations.values()))
    aggregate_metrics["summary/total_finalize_seconds"] = float(sum(finalize_durations.values()))
    aggregate_metrics["summary/n_benchmarks_ok"] = int(
        sum(1 for s in dispatch_results.values() if s == "ok")
    )
    aggregate_metrics["summary/n_benchmarks_failed"] = int(
        sum(1 for s in dispatch_results.values() if s != "ok" and s != "skipped")
    )

    try:
        run.log({"summary/benchmarks": wandb.Table(columns=cols, data=rows)})
        run.log(aggregate_metrics)
    except Exception as exc:
        print(f"[eval_all] summary log failed: {exc}", file=sys.stderr)

    try:
        run.finish()
    except Exception:
        pass


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
