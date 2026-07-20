"""Async-judge sidecar: small CPU process that ferries judge requests
to a separately-hosted judge LLM (typically scripts/judge_server.sub)
while eval_all advances other benchmarks.

Topology
--------
::

    eval_all (CPU monitor)              judge_server.sub (2-GPU vLLM)
        │                                       ▲
        │  spawn at startup                     │
        ▼                                       │
    judge_sidecar (THIS module, CPU only) ─HTTP─┘
        │  watches: <run_dir>/**/judge_*/manifest.json
        │  produces: output.jsonl + done.flag next to each manifest
        ▼
    privacylens_async export pipeline writes manifests
    privacylens_async_finalize / post_judge_metrics consume output.jsonl

Lifecycle modes
---------------

The CLI exposes three:

- ``oneshot``: process every manifest under ``--run-dir`` once, in
  parallel across manifests, then exit. Used for Phase 1 single-machine
  smoke tests where the operator wants explicit start/end semantics.

- ``run``: continuous watcher loop. Polls the watch root for new
  manifests, dispatches them to a worker pool, exits on SIGTERM /
  reaching the eval_all monitor's drain signal. Used by eval_all
  Phase 2 — launched as a slurm sidecar job alongside the parent
  monitor.

- ``drain``: block until every manifest under ``--run-dir`` has a
  ``done.flag``, then exit. Called by eval_all between its dispatch
  loop and its post_judge_metrics phase.

Resume safety
-------------

- Each manifest's progress is content-addressed via the OpenAI Batch
  ``custom_id`` already written by the export stage. On restart, we
  scan ``output.jsonl.partial`` and skip any row whose custom_id is
  already there.
- The W&B sidecar run uses a stable id derived from the eval_all
  group (``derive_resumable_run_id(group, "judge_sidecar",
  role="sidecar")``) with ``resume="allow"``, so a SLURM-requeued
  sidecar reattaches to the same throughput chart instead of forking.
- ``done.flag`` is written **only** after ``output.jsonl`` is atomic-
  renamed from ``output.jsonl.partial`` — a dead sidecar leaves either
  a partial file (resumable) or nothing committed (re-run-safe).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_POLL_INTERVAL_S = 5.0
DEFAULT_TICK_INTERVAL_S = 30.0
DEFAULT_PER_MANIFEST_CONCURRENCY = 8
#: Hard cap on simultaneous HTTP requests to the judge across ALL
#: manifests in this sidecar process. Without it, effective concurrency
#: is parallel_manifests x concurrency (4x8=32) PER CELL, and a 3-cell
#: sweep put 96 potential / 70 observed concurrent requests on the
#: judge server (55 running + 15 capacity-queued, 2026-07-19). Queued
#: requests burn toward the 240s client timeout -> timeout->retry storm.
#: 16/cell keeps a 3-cell sweep under the server's observed capacity.
DEFAULT_MAX_INFLIGHT_REQUESTS = 16
DEFAULT_PER_REQUEST_TIMEOUT_S = 240.0
DEFAULT_HTTP_RETRIES = 3
DEFAULT_HTTP_BACKOFF_S = 1.5
DEFAULT_DRAIN_TIMEOUT_S = 6 * 60 * 60  # 6h — generous default; eval_all overrides.

# Discover any manifest.json under the watch root that has an adjacent
# requests.jsonl — that's the only reliable cross-dagspace signal that
# this directory holds a judge batch we should fan out. Naming
# conventions vary: privacylens uses ``leakage_judge_batch`` /
# ``helpfulness_judge_batch``; future judged dagspaces may differ.
MANIFEST_GLOB = "**/manifest.json"
PARTIAL_SUFFIX = ".partial"
DONE_FLAG = "done.flag"
ERRORS_LOG = "errors.jsonl"

#: Capped at 1000 rows on the W&B failure table to mirror the parse-side cap
#: in dagspaces/common/eval_sanity.py. Same schema (FAILURE_ROW_COLUMNS) so
#: a unified "all failures across this sweep" filter works.
SIDECAR_FAILURE_CAP = 1000


# ---------------------------------------------------------------------------
# Per-manifest state
# ---------------------------------------------------------------------------

@dataclass
class ManifestStats:
    """Cumulative counters for one manifest's lifecycle."""

    manifest_path: str
    dagspace: str = ""
    stage: str = ""
    n_requests: int = 0
    n_completed: int = 0  # successful responses written
    n_failed: int = 0     # rows where every retry failed
    n_skipped: int = 0    # already-completed via partial-file resume
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    latencies_ms: List[float] = field(default_factory=list)

    def is_done(self) -> bool:
        return (self.n_completed + self.n_failed + self.n_skipped) >= self.n_requests

    def wall_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return float(end - self.started_at)

    def latency_p(self, q: float) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        i = max(0, min(len(s) - 1, int(q * (len(s) - 1))))
        return float(s[i])

    def throughput_rpm(self) -> float:
        w = self.wall_seconds()
        if w <= 0:
            return 0.0
        return float(self.n_completed) / (w / 60.0)


@dataclass
class FailureRecord:
    """One per-row failure ready to log to the sidecar's failure table."""

    custom_id: str
    manifest_path: str
    dagspace: str
    stage: str
    failure_type: str  # network / timeout / schema / content_filter / other
    attempt_count: int
    error_preview: str
    last_attempt_at: str


# ---------------------------------------------------------------------------
# Manifest discovery + parsing
# ---------------------------------------------------------------------------

def _discover_manifests(watch_root: str) -> List[str]:
    """Return absolute paths to every judge ``manifest.json`` under root.

    A manifest is recognised by having an adjacent ``requests.jsonl`` —
    that's the export-stage's contract regardless of dagspace naming
    convention. Hydra's ``.hydra/`` and submitit's ``.submitit/``
    directories also contain ``manifest.json`` files, so the
    requests.jsonl check is what makes this filter judge-specific.
    """
    pattern = os.path.join(watch_root, MANIFEST_GLOB)
    out: List[str] = []
    for p in sorted(glob.glob(pattern, recursive=True)):
        if os.path.exists(os.path.join(os.path.dirname(p), "requests.jsonl")):
            out.append(p)
    return out


def _read_manifest(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _output_paths(manifest_path: str) -> Tuple[str, str, str, str]:
    """Return (requests.jsonl, output.jsonl.partial, output.jsonl, done.flag)."""
    base = os.path.dirname(manifest_path)
    return (
        os.path.join(base, "requests.jsonl"),
        os.path.join(base, "output.jsonl" + PARTIAL_SUFFIX),
        os.path.join(base, "output.jsonl"),
        os.path.join(base, DONE_FLAG),
    )


def _load_requests(requests_path: str) -> List[Dict[str, Any]]:
    out = []
    with open(requests_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_completed_ids(partial_path: str) -> Set[str]:
    """Return custom_ids already committed to output.jsonl.partial."""
    seen: Set[str] = set()
    if not os.path.exists(partial_path):
        return seen
    try:
        with open(partial_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = rec.get("custom_id")
                if cid:
                    seen.add(str(cid))
    except OSError:
        pass
    return seen


# ---------------------------------------------------------------------------
# HTTP forwarder
# ---------------------------------------------------------------------------

def _classify_failure(exc: BaseException, status_code: Optional[int]) -> str:
    """Map an exception or non-2xx HTTP code to one of the canonical
    failure_type values shared with the parse-side sanity layer.
    """
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "network"
    if status_code is not None:
        if status_code in (400, 422):
            return "schema_violation"
        if status_code in (403, 451):
            return "content_filter"
        if status_code >= 500:
            return "network"
    return "other"


def _normalize_endpoint(base_url: str, request_url: str) -> str:
    base = base_url.rstrip("/")
    if request_url.startswith("http://") or request_url.startswith("https://"):
        return request_url
    if not request_url.startswith("/"):
        request_url = "/" + request_url
    if base.endswith("/v1") and request_url.startswith("/v1/"):
        request_url = request_url[len("/v1"):]
    return base + request_url


def _post_one(
    session: requests.Session,
    base_url: str,
    request: Dict[str, Any],
    *,
    timeout: float,
    retries: int,
    backoff: float,
    api_key: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], int]:
    """Forward one request and return (response_json, error, attempts).

    On success: ``(response_json, None, attempt_count)``. On exhaustion:
    ``(None, error_message, attempt_count)``. ``attempt_count`` always
    counts the final attempt.
    """
    url = _normalize_endpoint(base_url, request.get("url", "/v1/chat/completions"))
    body = request.get("body", {})
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_err: Optional[str] = None
    last_status: Optional[int] = None
    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            resp = session.post(url, json=body, headers=headers, timeout=timeout)
            last_status = resp.status_code
            if resp.status_code >= 400:
                last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                if resp.status_code in (400, 422, 403, 451):
                    # client-error / content-filter — no point retrying.
                    break
            else:
                try:
                    return resp.json(), None, attempt
                except ValueError as e:
                    last_err = f"non-JSON response: {e}"
                    break  # malformed body won't get better with retries
        except requests.exceptions.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            last_status = None
        if attempt < retries:
            time.sleep(backoff * (2 ** (attempt - 1)))

    failure_type = _classify_failure(
        Exception(last_err or "unknown"),
        last_status,
    )
    return None, f"[{failure_type}] {last_err}", attempt


def _emit_response_line(
    partial_path: str,
    *,
    custom_id: str,
    response_json: Optional[Dict[str, Any]],
    error: Optional[str],
    lock: threading.Lock,
) -> None:
    """Append one line to output.jsonl.partial (success or error)."""
    if response_json is not None:
        # Match OpenAI Batch API output.jsonl shape so downstream finalize
        # logic (dagspaces/common/batch_api.extract_content) Just Works.
        rec = {
            "custom_id": custom_id,
            "response": {"status_code": 200, "body": response_json},
        }
    else:
        rec = {
            "custom_id": custom_id,
            "error": error,
        }
    line = json.dumps(rec, ensure_ascii=False)
    with lock:
        with open(partial_path, "a") as f:
            f.write(line + "\n")


def _finalize_manifest(partial_path: str, output_path: str, done_path: str) -> None:
    """Atomic rename + done.flag write, in that order."""
    if not os.path.exists(partial_path):
        # Nothing was emitted — write an empty output.jsonl so finalize
        # logic finds something rather than raising FileNotFoundError.
        open(output_path, "w").close()
    else:
        os.replace(partial_path, output_path)
    with open(done_path, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat() + "\n")


# ---------------------------------------------------------------------------
# One-manifest fan-out
# ---------------------------------------------------------------------------

def process_manifest(
    manifest_path: str,
    *,
    base_url: str,
    api_key: Optional[str] = None,
    concurrency: int = DEFAULT_PER_MANIFEST_CONCURRENCY,
    timeout: float = DEFAULT_PER_REQUEST_TIMEOUT_S,
    retries: int = DEFAULT_HTTP_RETRIES,
    backoff: float = DEFAULT_HTTP_BACKOFF_S,
    on_progress: Optional[Any] = None,
    on_failure: Optional[Any] = None,
    on_start: Optional[Any] = None,
    stop_event: Optional[threading.Event] = None,
    global_sem: Optional[threading.Semaphore] = None,
) -> ManifestStats:
    """Fan-out one manifest's requests; resume-safe; emit output.jsonl + done.flag.

    Args:
        manifest_path: Absolute path to ``manifest.json``.
        base_url: Judge endpoint base URL (e.g. ``http://klara:8002`` or
            ``http://klara:8002/v1``). Trailing ``/v1`` is normalized.
        api_key: Optional bearer token for commercial endpoints.
        concurrency: Per-manifest worker pool size.
        timeout, retries, backoff: HTTP knobs.
        on_progress: Optional callback ``(stats: ManifestStats) -> None``
            invoked after every completed row (used by run_sidecar to
            emit per-tick W&B updates).
        on_failure: Optional callback ``(rec: FailureRecord) -> None``
            invoked once per row whose retries are exhausted.
        on_start: Optional callback ``(stats: ManifestStats) -> None``
            invoked once, right after the stats record is created —
            lets run_sidecar register the live object so the tick line
            can report in-flight row progress (not just finished
            manifests).
        stop_event: Optional shutdown signal — when set, the worker pool
            stops accepting new rows; in-flight rows still complete.
        global_sem: Optional semaphore shared across ALL concurrent
            process_manifest calls in this process — bounds total
            simultaneous HTTP requests to the judge regardless of how
            many manifests are in flight. Held for the full retry loop
            of one row (retries are judge load too).

    Returns the final :class:`ManifestStats`. Raises if the manifest
    itself can't be read (i.e. corrupt JSON), but never on per-row
    failures — those are recorded and execution continues.
    """
    requests_path, partial_path, output_path, done_path = _output_paths(manifest_path)

    # Idempotency: if done.flag is already present, skip everything.
    if os.path.exists(done_path):
        # Synthesize a "done with N=N" stats record so the caller can
        # update tables / counters consistently.
        n = sum(1 for _ in open(requests_path)) if os.path.exists(requests_path) else 0
        return ManifestStats(
            manifest_path=manifest_path,
            n_requests=n,
            n_completed=n,
            n_skipped=n,
            started_at=time.time(),
            finished_at=time.time(),
        )

    manifest = _read_manifest(manifest_path)
    requests_list = _load_requests(requests_path)
    completed_already = _load_completed_ids(partial_path)
    pending = [r for r in requests_list if str(r.get("custom_id")) not in completed_already]

    stats = ManifestStats(
        manifest_path=manifest_path,
        dagspace=str(manifest.get("dagspace", "")),
        stage=str(manifest.get("stage", "")),
        n_requests=len(requests_list),
        n_skipped=len(completed_already),
        started_at=time.time(),
    )
    if on_start is not None:
        on_start(stats)

    print(
        f"[sidecar] {stats.dagspace}.{stats.stage}: "
        f"{len(pending)} pending of {len(requests_list)} "
        f"(skipping {len(completed_already)} already in partial)",
        flush=True,
    )

    # No-op pass: just promote partial → final + done.flag.
    if not pending:
        _finalize_manifest(partial_path, output_path, done_path)
        stats.finished_at = time.time()
        if on_progress is not None:
            on_progress(stats)
        return stats

    write_lock = threading.Lock()
    session = requests.Session()

    def _work(req: Dict[str, Any]) -> Optional[FailureRecord]:
        if stop_event is not None and stop_event.is_set():
            return None
        cid = str(req.get("custom_id") or "")
        if global_sem is not None:
            global_sem.acquire()
        try:
            # t0 after acquire: latency measures the judge, not our queue.
            t0 = time.time()
            resp_json, err, attempts = _post_one(
                session, base_url, req,
                timeout=timeout, retries=retries, backoff=backoff,
                api_key=api_key,
            )
            latency_ms = (time.time() - t0) * 1000.0
        finally:
            if global_sem is not None:
                global_sem.release()
        stats.latencies_ms.append(latency_ms)
        _emit_response_line(
            partial_path,
            custom_id=cid,
            response_json=resp_json,
            error=err,
            lock=write_lock,
        )
        if err is None:
            stats.n_completed += 1
            return None
        stats.n_failed += 1
        # Failure type sits inside the bracketed prefix in err.
        ftype = "other"
        if err.startswith("[") and "]" in err:
            ftype = err[1:err.index("]")]
        return FailureRecord(
            custom_id=cid,
            manifest_path=manifest_path,
            dagspace=stats.dagspace,
            stage=stats.stage,
            failure_type=ftype,
            attempt_count=attempts,
            error_preview=err[:500],
            last_attempt_at=datetime.now(timezone.utc).isoformat(),
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futures = [ex.submit(_work, r) for r in pending]
        for fut in as_completed(futures):
            try:
                failure = fut.result()
            except Exception as exc:
                # Never let one row's bug kill the whole manifest.
                stats.n_failed += 1
                failure = FailureRecord(
                    custom_id="",
                    manifest_path=manifest_path,
                    dagspace=stats.dagspace,
                    stage=stats.stage,
                    failure_type="other",
                    attempt_count=0,
                    error_preview=f"worker_exception: {exc}",
                    last_attempt_at=datetime.now(timezone.utc).isoformat(),
                )
            if failure is not None and on_failure is not None:
                on_failure(failure)
            if on_progress is not None:
                on_progress(stats)

    _finalize_manifest(partial_path, output_path, done_path)
    stats.finished_at = time.time()
    if on_progress is not None:
        on_progress(stats)
    return stats


# ---------------------------------------------------------------------------
# Watcher loop + drain helper
# ---------------------------------------------------------------------------

class _SidecarRuntime:
    """Mutable runtime state for run_sidecar / oneshot.

    Held outside :func:`run_sidecar` so a SIGTERM handler and the
    W&B tick thread can reach it.
    """

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.in_flight_manifests: Set[str] = set()
        # Live ManifestStats for in-flight manifests, registered via
        # process_manifest's on_start hook — this is what lets the tick
        # line report row progress BEFORE a manifest finishes.
        self.active_stats: Dict[str, ManifestStats] = {}
        self.completed_manifests: Dict[str, ManifestStats] = {}
        self.cumulative_failures: int = 0
        self.failure_buffer: List[FailureRecord] = []
        self.lock = threading.Lock()
        self.start_time = time.time()


def _aggregate_pending(watch_root: str, runtime: _SidecarRuntime) -> List[str]:
    """Manifests not yet started and not yet done."""
    manifests = _discover_manifests(watch_root)
    pending = []
    for m in manifests:
        _, _, _, done_path = _output_paths(m)
        if os.path.exists(done_path):
            continue
        if m in runtime.in_flight_manifests:
            continue
        pending.append(m)
    return pending


def _all_done(watch_root: str) -> bool:
    manifests = _discover_manifests(watch_root)
    if not manifests:
        return False  # nothing to wait for yet
    for m in manifests:
        _, _, _, done_path = _output_paths(m)
        if not os.path.exists(done_path):
            return False
    return True


def wait_for_drain(
    watch_root: str,
    *,
    timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    on_tick: Optional[Any] = None,
) -> bool:
    """Block until every manifest under ``watch_root`` has ``done.flag``.

    Returns ``True`` if drained cleanly, ``False`` on timeout. Calls
    ``on_tick(pending_manifests, total_manifests)`` every poll if
    provided (used by eval_all to surface drain progress).
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        manifests = _discover_manifests(watch_root)
        pending = [m for m in manifests if not os.path.exists(_output_paths(m)[3])]
        if on_tick is not None:
            try:
                on_tick(len(pending), len(manifests))
            except Exception:
                pass
        if manifests and not pending:
            return True
        time.sleep(poll_interval_s)
    return False


# ---------------------------------------------------------------------------
# W&B integration (online with retries)
# ---------------------------------------------------------------------------

def _start_sidecar_wandb_run(
    *,
    project: str,
    entity: Optional[str],
    group: Optional[str],
    judge_base_url: str,
    judge_model: str,
    concurrency: int,
):
    """Open a resumable W&B run for the sidecar. Returns the run, or
    None if W&B is unavailable / disabled. Stays online with retries —
    every per-tick log() is wrapped at the call site.
    """
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    if os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1"):
        return None

    from dagspaces.common.wandb_logger import derive_resumable_run_id

    run_id = derive_resumable_run_id(group, dagspace="judge_sidecar", model=None, role="sidecar") if group else None
    tags = ["service:judge_sidecar", "judge_mode:async"]
    try:
        from urllib.parse import urlparse
        host = urlparse(judge_base_url).hostname
        if host:
            tags.append(f"judge_endpoint:{host}")
    except Exception:
        pass
    if group:
        tags.append(f"eval_all_run:{group}")

    init_kwargs: Dict[str, Any] = {
        "project": project,
        "entity": entity,
        "group": group,
        "job_type": "judge_sidecar",
        "name": f"judge_sidecar-{group or 'standalone'}",
        "config": {
            "judge_base_url": judge_base_url,
            "judge_model": judge_model,
            "concurrency": concurrency,
            "started_at": datetime.now(timezone.utc).isoformat(),
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
        print(f"[sidecar] W&B init failed (continuing without): {exc}", flush=True)
        return None


def _wandb_log_with_retries(run, payload: Dict[str, Any], *, attempts: int = 3, backoff: float = 1.0) -> None:
    if run is None:
        return
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        try:
            run.log(payload)
            return
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (2 ** i))
    print(f"[sidecar] W&B log failed after {attempts} attempts: {last_exc}", flush=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_sidecar(
    watch_root: str,
    *,
    base_url: str,
    api_key: Optional[str] = None,
    concurrency: int = DEFAULT_PER_MANIFEST_CONCURRENCY,
    parallel_manifests: int = 4,
    max_inflight_requests: int = DEFAULT_MAX_INFLIGHT_REQUESTS,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    tick_interval_s: float = DEFAULT_TICK_INTERVAL_S,
    timeout: float = DEFAULT_PER_REQUEST_TIMEOUT_S,
    retries: int = DEFAULT_HTTP_RETRIES,
    backoff: float = DEFAULT_HTTP_BACKOFF_S,
    judge_model: str = "",
    wandb_project: str = "eval-all",
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    health_check: bool = True,
    drain_after: bool = False,
) -> Dict[str, ManifestStats]:
    """Continuous sidecar loop: watches ``watch_root`` for new manifests,
    fans them out concurrently, exits on SIGTERM or when ``drain_after``
    is set and no new manifests appear for one full poll.

    Returns the final ``{manifest_path: ManifestStats}`` map.
    """
    runtime = _SidecarRuntime()
    # Global request-level backpressure: bounds simultaneous judge HTTP
    # requests across all manifests (see DEFAULT_MAX_INFLIGHT_REQUESTS).
    # <=0 disables the cap (old behavior: parallel_manifests x concurrency).
    global_sem: Optional[threading.Semaphore] = (
        threading.BoundedSemaphore(max_inflight_requests)
        if max_inflight_requests and max_inflight_requests > 0
        else None
    )

    # Health check before doing anything destructive.
    if health_check:
        try:
            r = requests.get(_normalize_endpoint(base_url, "/v1/models"), timeout=10.0)
            if r.status_code >= 400:
                print(
                    f"[sidecar] WARNING: judge endpoint health probe returned "
                    f"HTTP {r.status_code} on /v1/models — proceeding anyway",
                    file=sys.stderr,
                )
            else:
                print(f"[sidecar] judge endpoint OK: {base_url}", flush=True)
        except Exception as e:
            print(
                f"[sidecar] WARNING: judge endpoint unreachable ({e}) — "
                "is scripts/judge_server.sub up?",
                file=sys.stderr,
            )

    wandb_run = _start_sidecar_wandb_run(
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        judge_base_url=base_url,
        judge_model=judge_model,
        concurrency=concurrency,
    )

    def _on_start(stats: ManifestStats) -> None:
        # Register the live stats object so the tick line can report
        # in-flight row progress. (The previous cumulative_completed_rows
        # counter computed here was never surfaced AND was racy across
        # concurrent manifests — replaced by this registry.)
        with runtime.lock:
            runtime.active_stats[stats.manifest_path] = stats

    def _on_failure(rec: FailureRecord) -> None:
        with runtime.lock:
            runtime.cumulative_failures += 1
            if len(runtime.failure_buffer) < SIDECAR_FAILURE_CAP:
                runtime.failure_buffer.append(rec)

    # Tick thread — periodic W&B + log line. ``done=`` counts done.flag
    # files on disk (ground truth), not just manifests this process
    # finished; ``rows=`` includes live in-flight progress, so it moves
    # every tick instead of jumping only when a manifest finalizes.
    def _tick_loop() -> None:
        while not runtime.stop_event.is_set():
            time.sleep(tick_interval_s)
            # NFS glob outside the lock — it can stall for seconds.
            manifests = _discover_manifests(watch_root)
            done_disk = sum(
                1 for m in manifests if os.path.exists(_output_paths(m)[3])
            )
            with runtime.lock:
                inflight_set = set(runtime.in_flight_manifests)
                active = list(runtime.active_stats.values())
                finalized_rows = sum(
                    s.n_completed for s in runtime.completed_manifests.values()
                )
                err_rows = runtime.cumulative_failures
                wall = time.time() - runtime.start_time
            n_pending = sum(
                1 for m in manifests
                if m not in inflight_set
                and not os.path.exists(_output_paths(m)[3])
            )
            live_rows = sum(
                s.n_completed + s.n_failed + s.n_skipped for s in active
            )
            progress = " ".join(
                f"{s.dagspace}.{s.stage}:"
                f"{s.n_completed + s.n_failed + s.n_skipped}/{s.n_requests}"
                for s in active
            )
            payload = {
                "sidecar/pending_manifests": n_pending,
                "sidecar/inflight_manifests": len(inflight_set),
                "sidecar/done_manifests": done_disk,
                "sidecar/done_rows": finalized_rows,
                "sidecar/live_rows": live_rows,
                "sidecar/error_rows": err_rows,
                "sidecar/wall_seconds": wall,
                "sidecar/throughput_rpm": (
                    (finalized_rows + live_rows) / max(wall, 1.0)
                ) * 60.0,
            }
            print(
                f"[sidecar][tick] manifests pending={n_pending} "
                f"inflight={len(inflight_set)} done={done_disk}/{len(manifests)} | "
                f"rows finalized={finalized_rows} live={live_rows} "
                f"errors={err_rows}"
                + (f" | {progress}" if progress else ""),
                flush=True,
            )
            _wandb_log_with_retries(wandb_run, payload)

    tick_thread = threading.Thread(target=_tick_loop, daemon=True)
    tick_thread.start()

    # Signal handlers — graceful drain.
    def _shutdown(*_args: Any) -> None:
        print("[sidecar] received shutdown signal — stopping intake.", flush=True)
        runtime.stop_event.set()

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _shutdown)
        except Exception:
            pass

    # Worker pool across manifests.
    idle_polls_after_drain = 0
    with ThreadPoolExecutor(max_workers=max(1, parallel_manifests)) as pool:
        running_futures: Dict[Any, str] = {}

        while not runtime.stop_event.is_set():
            with runtime.lock:
                pending = _aggregate_pending(watch_root, runtime)

            for manifest_path in pending:
                if runtime.stop_event.is_set():
                    break
                with runtime.lock:
                    if manifest_path in runtime.in_flight_manifests:
                        continue
                    runtime.in_flight_manifests.add(manifest_path)

                fut = pool.submit(
                    process_manifest,
                    manifest_path,
                    base_url=base_url,
                    api_key=api_key,
                    concurrency=concurrency,
                    timeout=timeout,
                    retries=retries,
                    backoff=backoff,
                    on_failure=_on_failure,
                    on_start=_on_start,
                    stop_event=runtime.stop_event,
                    global_sem=global_sem,
                )
                running_futures[fut] = manifest_path

            # Reap finished manifests.
            done_now = [f for f in list(running_futures) if f.done()]
            for fut in done_now:
                mp = running_futures.pop(fut)
                try:
                    stats = fut.result()
                except Exception as exc:
                    print(f"[sidecar] manifest {mp} crashed: {exc}", flush=True)
                    stats = ManifestStats(manifest_path=mp)
                with runtime.lock:
                    runtime.in_flight_manifests.discard(mp)
                    runtime.active_stats.pop(mp, None)
                    runtime.completed_manifests[mp] = stats
                _wandb_log_manifest_summary(wandb_run, stats)

            if drain_after and not running_futures:
                with runtime.lock:
                    if not _aggregate_pending(watch_root, runtime):
                        idle_polls_after_drain += 1
                        if idle_polls_after_drain >= 2:
                            break
                    else:
                        idle_polls_after_drain = 0

            time.sleep(poll_interval_s)

        # Drain in-flight on shutdown.
        for fut in list(running_futures):
            try:
                stats = fut.result(timeout=300)
            except Exception:
                stats = ManifestStats(manifest_path=running_futures[fut])
            mp = running_futures[fut]
            with runtime.lock:
                runtime.in_flight_manifests.discard(mp)
                runtime.active_stats.pop(mp, None)
                runtime.completed_manifests[mp] = stats
            _wandb_log_manifest_summary(wandb_run, stats)

    runtime.stop_event.set()

    # Final failure-table dump and summary log.
    _wandb_log_failures(wandb_run, runtime.failure_buffer, dropped=max(0, runtime.cumulative_failures - len(runtime.failure_buffer)))
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    return dict(runtime.completed_manifests)


def _wandb_log_manifest_summary(run, stats: ManifestStats) -> None:
    if run is None:
        return
    payload = {
        "sidecar/manifest/last_n_requests": stats.n_requests,
        "sidecar/manifest/last_completed": stats.n_completed,
        "sidecar/manifest/last_failed": stats.n_failed,
        "sidecar/manifest/last_skipped": stats.n_skipped,
        "sidecar/manifest/last_wall_s": stats.wall_seconds(),
        "sidecar/manifest/last_p50_ms": stats.latency_p(0.5),
        "sidecar/manifest/last_p99_ms": stats.latency_p(0.99),
        "sidecar/manifest/last_throughput_rpm": stats.throughput_rpm(),
    }
    _wandb_log_with_retries(run, payload)


def _wandb_log_failures(run, failures: List[FailureRecord], *, dropped: int) -> None:
    if run is None or not failures:
        return
    try:
        import wandb  # type: ignore
        # Mirror dagspaces/common/eval_sanity.FAILURE_ROW_COLUMNS so a
        # unified W&B filter combines parse-side and sidecar-side failures.
        cols = [
            "custom_id", "dagspace", "stage", "failure_type",
            "raw_response_preview", "parse_error", "model",
            "manifest_path", "attempt_count", "error_preview", "last_attempt_at",
        ]
        rows = [
            [
                rec.custom_id, rec.dagspace, rec.stage, rec.failure_type,
                "", "", "",
                rec.manifest_path, rec.attempt_count, rec.error_preview,
                rec.last_attempt_at,
            ]
            for rec in failures
        ]
        table = wandb.Table(columns=cols, data=rows)
        run.log({"sidecar/failures": table})
        if dropped > 0:
            run.log({"sidecar/failures_dropped": dropped})
    except Exception as e:
        print(f"[sidecar] W&B failure-table log failed: {e}", flush=True)


def oneshot(
    run_dir: str,
    *,
    base_url: str,
    api_key: Optional[str] = None,
    concurrency: int = DEFAULT_PER_MANIFEST_CONCURRENCY,
    parallel_manifests: int = 4,
    max_inflight_requests: int = DEFAULT_MAX_INFLIGHT_REQUESTS,
    timeout: float = DEFAULT_PER_REQUEST_TIMEOUT_S,
    retries: int = DEFAULT_HTTP_RETRIES,
    backoff: float = DEFAULT_HTTP_BACKOFF_S,
    judge_model: str = "",
    wandb_project: str = "eval-all",
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
) -> Dict[str, ManifestStats]:
    """Process every manifest under ``run_dir`` once, then exit.

    Equivalent to ``run_sidecar(..., drain_after=True)`` but skips the
    poll-for-new-manifests loop — assumes the export pipeline has
    already finished writing all manifests.
    """
    return run_sidecar(
        run_dir,
        base_url=base_url,
        api_key=api_key,
        concurrency=concurrency,
        parallel_manifests=parallel_manifests,
        max_inflight_requests=max_inflight_requests,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
        judge_model=judge_model,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        drain_after=True,
        health_check=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-dir", required=True, help="Eval root to scan for manifests")
    common.add_argument(
        "--base-url",
        default=os.environ.get("JUDGE_SERVER_URL") or os.environ.get("JUDGE_BASE_URL", "http://localhost:8002"),
    )
    common.add_argument("--api-key-env", default=None, help="Env var holding bearer token (commercial endpoints)")
    common.add_argument("--model", default="", help="Judge model name (telemetry only)")
    common.add_argument("--concurrency", type=int, default=DEFAULT_PER_MANIFEST_CONCURRENCY)
    common.add_argument("--parallel-manifests", type=int, default=4)
    common.add_argument(
        "--max-inflight", type=int, default=DEFAULT_MAX_INFLIGHT_REQUESTS,
        help="Global cap on simultaneous judge HTTP requests across all "
             "manifests (0 disables; default %(default)s)",
    )
    common.add_argument("--timeout", type=float, default=DEFAULT_PER_REQUEST_TIMEOUT_S)
    common.add_argument("--retries", type=int, default=DEFAULT_HTTP_RETRIES)
    common.add_argument("--backoff", type=float, default=DEFAULT_HTTP_BACKOFF_S)
    common.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "eval-all"))
    common.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    common.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP"))

    sub.add_parser("oneshot", parents=[common], help="Drain all current manifests and exit")
    sub.add_parser("run", parents=[common], help="Continuous watcher loop")

    drain = sub.add_parser("drain", help="Block until all manifests have done.flag")
    drain.add_argument("--run-dir", required=True)
    drain.add_argument("--timeout", type=float, default=DEFAULT_DRAIN_TIMEOUT_S)
    drain.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S)

    return p


def _resolve_api_key(env_name: Optional[str]) -> Optional[str]:
    if not env_name:
        return None
    val = os.environ.get(env_name)
    if not val:
        print(f"[sidecar] WARNING: --api-key-env={env_name} is unset", file=sys.stderr)
    return val


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd in ("oneshot", "run"):
        api_key = _resolve_api_key(args.api_key_env)
        runner = oneshot if args.cmd == "oneshot" else run_sidecar
        runner(
            args.run_dir,
            base_url=args.base_url,
            api_key=api_key,
            concurrency=args.concurrency,
            parallel_manifests=args.parallel_manifests,
            max_inflight_requests=args.max_inflight,
            timeout=args.timeout,
            retries=args.retries,
            backoff=args.backoff,
            judge_model=args.model,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
        )
        return 0
    if args.cmd == "drain":
        ok = wait_for_drain(
            args.run_dir,
            timeout_s=args.timeout,
            poll_interval_s=args.poll_interval,
        )
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
