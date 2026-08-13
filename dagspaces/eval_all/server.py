"""Shared vLLM OpenAI-compatible server lifecycle for eval_all.

Launches a single long-lived vLLM server as a SLURM job that hosts the
model once, so every benchmark subprocess can hit it over HTTP instead
of loading+unloading the model per benchmark.

The server runs ``python -m vllm.entrypoints.openai.api_server`` inside
a submitit-managed SLURM allocation. The server writes its
``host:port`` to an address file; eval_all polls that file, then
polls ``/health`` on the URL, then exports ``VLLM_SERVER_URL`` so
child benchmark subprocesses inherit it.

On normal completion, interrupt, or exception, the job is cancelled.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from typing import Any

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


def _serve_entrypoint(args: dict[str, Any]) -> None:
    """Runs inside the SLURM job — writes an address file then execs vLLM server.

    This function is pickled and shipped to the compute node by submitit.
    Keep top-level imports minimal so the pickle stays stable.
    """
    # The job's env comes from the ssh sbatch shim (fresh login env), not
    # the driver — load server.env so cache redirects (VLLM_CACHE_ROOT,
    # TRITON_CACHE_DIR) and site settings reach the exec'd vLLM server.
    from dagspaces.common.stage_utils import ensure_dotenv

    ensure_dotenv()

    hostname = socket.gethostname()
    port = int(args["port"])
    address_file = args["address_file"]

    # Port selection: a fixed port collides when two eval cells' servers land
    # on the same node (array_parallelism > 1 on one box) — the loser fails
    # to bind and burns the whole startup timeout. port=0 means "pick a free
    # ephemeral port"; a non-zero request falls back to ephemeral if taken.
    def _bind_probe(p: int) -> int | None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", p))
            return s.getsockname()[1]
        except OSError:
            return None
        finally:
            s.close()

    chosen = _bind_probe(port) if port else None
    if chosen is None:
        chosen = _bind_probe(0)
        if port:
            print(f"[vllm-server] port {port} unavailable → using {chosen}", flush=True)
    port = int(chosen)

    os.makedirs(os.path.dirname(address_file), exist_ok=True)
    with open(address_file, "w", encoding="utf-8") as f:
        f.write(f"{hostname}:{port}\n")
    print(f"[vllm-server] address={hostname}:{port} → wrote {address_file}", flush=True)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args["model_path"],
        "--served-model-name", args["served_name"],
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    if args.get("tensor_parallel_size"):
        cmd += ["--tensor-parallel-size", str(int(args["tensor_parallel_size"]))]
    if args.get("max_model_len"):
        cmd += ["--max-model-len", str(int(args["max_model_len"]))]
    if args.get("gpu_memory_utilization"):
        cmd += ["--gpu-memory-utilization", str(float(args["gpu_memory_utilization"]))]
    if args.get("max_num_seqs"):
        cmd += ["--max-num-seqs", str(int(args["max_num_seqs"]))]
    if args.get("trust_remote_code"):
        cmd += ["--trust-remote-code"]
    if args.get("enforce_eager"):
        cmd += ["--enforce-eager"]
    # LoRA: the task models are LoRA checkpoints — without these flags the
    # server silently evaluates the BASE model under the checkpoint's name.
    if args.get("lora_path"):
        cmd += [
            "--enable-lora",
            "--max-lora-rank", str(int(args.get("max_lora_rank") or 64)),
            "--lora-modules", f"{args['lora_name']}={args['lora_path']}",
        ]
    if args.get("reasoning_parser"):
        cmd += ["--reasoning-parser", str(args["reasoning_parser"])]
    if args.get("extra_args"):
        cmd += list(args["extra_args"])

    env = dict(os.environ)
    # Propagate NCCL defaults for PCIe clusters
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_SHM_DISABLE", "0")  # see get_pcie_nccl_env_vars()
    env.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")

    print(f"[vllm-server] launching: {' '.join(cmd)}", flush=True)
    # Exec so that SIGTERM from submitit propagates to vLLM cleanly.
    os.execvpe(cmd[0], cmd, env)


def _read_address_file(path: str) -> str | None:
    """Return ``host:port`` from the address file, or None if not yet written."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        return line or None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _poll_health(url: str, timeout: float) -> bool:
    """Poll ``<url>/health`` until the server responds 200 or timeout."""
    import urllib.error
    import urllib.request
    health = url.rstrip("/").removesuffix("/v1") + "/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(health, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def launch_vllm_server(
    cfg: DictConfig,
    model_cfg: DictConfig,
    server_cfg: DictConfig,
    output_dir: str,
    *,
    startup_timeout_s: float = 900.0,
) -> dict[str, Any]:
    """Launch a vLLM server as a SLURM job and wait until it is healthy.

    Args:
        model_cfg: the resolved ``cfg.model`` DictConfig.
        server_cfg: an ``eval_all.server_mode`` subtree with keys ``enabled``,
            ``launcher``, ``port``, ``gpu_memory_utilization``, etc.
        output_dir: directory under which submitit logs + address file live.
        startup_timeout_s: how long to wait for ``/health`` to return 200.

    Returns:
        A dict with keys ``job`` (submitit Job), ``url`` (base URL ending
        in ``/v1``), ``served_name``, ``address_file``.
    """
    from dagspaces.common.orchestrator import (
        create_submitit_executor,
        load_launcher_config,
        submit_slurm_job,
    )

    port = int(server_cfg.get("port", 8000))
    launcher_name = str(server_cfg.get("launcher", "slurm_gpu_1x"))
    # Launcher yamls live under common/conf/hydra/launcher/; pass the
    # eval_all local conf dir so load_launcher_config's fallback kicks in.
    conf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")

    launcher_cfg = load_launcher_config(
        cfg, launcher_name, config_dir=conf_dir,
    )
    # Orphan guard: cleanup is atexit/SIGTERM-based, so a SIGKILLed monitor
    # leaks the server job. slurm_gpu_1x's 10-day limit would burn a GPU for
    # days — cap the server's own walltime instead. 12h >> any eval cell.
    try:
        launcher_cfg["timeout_min"] = int(server_cfg.get("max_runtime_min", 720))
    except Exception:
        pass

    log_folder = os.path.join(output_dir, "vllm_server_logs")
    os.makedirs(log_folder, exist_ok=True)
    address_file = os.path.join(log_folder, "address.txt")
    # Remove any stale address file from a prior run.
    try:
        os.remove(address_file)
    except FileNotFoundError:
        pass

    # served_name stays the CANONICAL zoo path (it is what clients request and
    # what lands in exports); the load path goes through the node-local model
    # registry so weights stream from /scratch instead of NFS.
    canonical_source = str(OmegaConf.select(model_cfg, "model_source"))
    served_name = str(server_cfg.get("served_model_name", "")) or canonical_source
    from dagspaces.common.model_registry import resolve_model_source
    model_path = resolve_model_source(canonical_source, stage_name="eval_all.server")

    from dagspaces.common.vllm_inference import _is_harmony_model
    if _is_harmony_model(canonical_source):
        raise RuntimeError(
            "server_mode does not support gpt-oss/harmony models: their "
            "final-channel extraction needs the in-process harmony path "
            "(vLLM's openai_gptoss parser is not usable). Run this model "
            "with server_mode.enabled=false."
        )

    # Pull in TP / max_model_len from model engine_kwargs as defaults.
    ek = OmegaConf.select(model_cfg, "engine_kwargs") or {}
    tp = server_cfg.get("tensor_parallel_size") or ek.get("tensor_parallel_size") or 1
    max_model_len = server_cfg.get("max_model_len") or ek.get("max_model_len")

    # LoRA checkpoint: remap CausalLM-layout adapter keys if the family needs
    # it (same helper the in-process path uses), then serve the adapter under
    # a stable name. Clients address it via VLLM_SERVER_LORA_NAME.
    lora_path = str(OmegaConf.select(model_cfg, "lora_path") or "")
    lora_name = None
    if lora_path:
        from dagspaces.common.vllm_inference import _remap_lora_keys_for_vlm
        lora_path = _remap_lora_keys_for_vlm(
            lora_path, model_path, "eval_all.server",
        )
        lora_name = "sft"

    # Pick a reasoning parser based on the model family, unless disabled.
    reasoning_parser = None
    if server_cfg.get("reasoning_parser", "auto") != "none":
        explicit = server_cfg.get("reasoning_parser")
        if explicit and explicit != "auto":
            reasoning_parser = str(explicit)
        else:
            from dagspaces.common.vllm_inference import _detect_reasoning_parser
            reasoning_parser = _detect_reasoning_parser(model_path)

    args = {
        "model_path": model_path,
        "served_name": served_name,
        "port": port,
        "address_file": address_file,
        "tensor_parallel_size": tp,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": server_cfg.get("gpu_memory_utilization", 0.90),
        # None → vLLM's server default (256); the model yamls' 16 is an
        # in-process norm-extraction setting, not a serving one.
        "max_num_seqs": server_cfg.get("max_num_seqs"),
        # Parity with in-process evals (model yamls set both).
        "trust_remote_code": bool(ek.get("trust_remote_code", False)),
        "enforce_eager": bool(
            server_cfg.get("enforce_eager", ek.get("enforce_eager", False))
        ),
        "lora_path": lora_path or None,
        "lora_name": lora_name,
        "max_lora_rank": ek.get("max_lora_rank"),
        "reasoning_parser": reasoning_parser,
        "extra_args": list(server_cfg.get("extra_args", []) or []),
    }

    executor = create_submitit_executor(
        launcher_cfg=launcher_cfg,
        job_name="vllm-server",
        log_folder=log_folder,
    )
    job = submit_slurm_job(
        executor=executor,
        execute_fn=_serve_entrypoint,
        context_data=args,
        node_key="vllm_server",
        launcher_name=launcher_name,
    )
    print(f"[eval_all] vllm server submitted: job_id={job.job_id}, logs={log_folder}")

    # Wait for address file
    t0 = time.time()
    host_port = None
    while time.time() - t0 < startup_timeout_s:
        host_port = _read_address_file(address_file)
        if host_port:
            break
        # Check that the job hasn't failed
        try:
            state = job.state
        except Exception:
            state = "UNKNOWN"
        if state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED"):
            print(f"[eval_all] SERVER LAUNCH FAILED: job {job.job_id} entered "
                  f"terminal state {state}", flush=True)
            raise RuntimeError(f"vLLM server job entered terminal state: {state}")
        time.sleep(5)

    if not host_port:
        try:
            job.cancel()
        except Exception:
            pass
        # Print loudly: under the hydra-submitit launcher a raised exception
        # is captured into the job's result pickle WITHOUT being printed
        # (the driver surfaces only the first failed arm), so without this
        # line a timed-out arm looks like a clean no-op in its own log
        # (observed 2026-07-22: top-up arms 6-10 pending behind a full
        # queue for 900s, zero output).
        print(f"[eval_all] SERVER LAUNCH TIMED OUT after {startup_timeout_s}s "
              f"(job {job.job_id} never wrote {address_file} — likely stuck "
              "PENDING behind a full queue); aborting this cell.", flush=True)
        raise TimeoutError(
            f"vLLM server did not write {address_file} within {startup_timeout_s}s"
        )

    base_url = f"http://{host_port}/v1"
    print(f"[eval_all] vllm server address: {base_url}")
    print(f"[eval_all] waiting for /health ...")

    if not _poll_health(base_url, timeout=startup_timeout_s):
        try:
            job.cancel()
        except Exception:
            pass
        raise TimeoutError(
            f"vLLM server at {base_url} did not become healthy within "
            f"{startup_timeout_s}s"
        )
    print(f"[eval_all] vllm server healthy at {base_url}")

    return {
        "job": job,
        "url": base_url,
        "served_name": served_name,
        "canonical_source": canonical_source,
        "lora_name": lora_name,
        "address_file": address_file,
    }


def shutdown_vllm_server(server_info: dict[str, Any]) -> None:
    """Cancel the SLURM job hosting the server, if still running."""
    job = server_info.get("job")
    if job is None:
        return
    try:
        job.cancel()
        print(f"[eval_all] vllm server job cancelled: {job.job_id}")
    except Exception as e:
        print(f"[eval_all] failed to cancel vllm server job: {e}")
