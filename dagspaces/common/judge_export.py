"""Mode-aware judge-export client resolution.

Shared by every dagspace that emits ``requests.jsonl`` for judging:
:mod:`dagspaces.privacylens.stages.llm_inference` and
:mod:`dagspaces.cirl_vignettes.stages.judge_leakage`.

Three judge modes, two export targets
-------------------------------------

The ``judge.mode`` knob picks between three runtime topologies; this
helper handles the two that actually emit JSONL:

- ``async``  — the export stage writes ``requests.jsonl`` to be drained
  by :mod:`dagspaces.eval_all.judge_sidecar`, which forwards each line
  to a vLLM judge server (typically ``scripts/judge_server.sub`` on
  klara). The model name written into ``body.model`` MUST match a
  model the live judge server is actually serving — otherwise vLLM
  responds ``404 The model 'X' does not exist.`` for every row and the
  finalize stage silently defaults each row to non-leaking / score-0.
  We resolve the model by probing ``<base_url>/v1/models`` at export
  time.

- ``batch_export`` — the export stage writes ``requests.jsonl`` for
  manual upload to OpenAI's Batch API. Here ``body.model`` is whatever
  OpenAI model the operator wants the batch run to use (``gpt-4o``,
  ``gpt-5``, …). The live judge endpoint is irrelevant; ``judge.batch.target_model``
  is the only knob.

The third mode, ``live``, doesn't emit JSONL at all — it calls the
judge synchronously via :class:`dagspaces.common.judge_client.JudgeClient`
and is not handled here.

Why this isn't inlined in each dagspace
---------------------------------------

Before this module existed, both privacylens and cirl_vignettes had
near-identical ``_get_batch_export_client`` helpers that read
``judge.batch.target_model`` *unconditionally* — including in async
mode. The result was that every async run wrote ``body.model="gpt-5.2"``
to ``requests.jsonl`` regardless of what the live klara judge server
was actually serving. The sidecar dutifully forwarded the requests,
vLLM 404'd every one, and the finalize stage swallowed the errors.
This module exists so that bug can only be fixed in one place.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from dagspaces.common.judge_client import JudgeClient


__all__ = [
    "JudgeExportConfigError",
    "resolve_judge_mode",
    "resolve_export_client",
    "resolve_export_endpoint",
]


class JudgeExportConfigError(RuntimeError):
    """Raised when the export client cannot be built from cfg.

    Distinct from generic ValueError so callers (and tests) can assert
    on the canonical "your config is wrong, the run will produce
    garbage" failure mode.
    """


# ---------------------------------------------------------------------------
# Mode + endpoint
# ---------------------------------------------------------------------------

def resolve_judge_mode(cfg: DictConfig) -> str:
    """Return ``cfg.judge.mode`` or 'live' if unset (lowercase, stripped)."""
    return str(OmegaConf.select(cfg, "judge.mode", default="live") or "live").lower()


def resolve_export_endpoint(cfg: DictConfig) -> str:
    """Return the URL written into each JSONL line's ``url`` field.

    For both async and batch_export this is just the chat-completions
    path — the caller (sidecar or OpenAI Batch) supplies the host.
    """
    return str(
        OmegaConf.select(cfg, "judge.batch.target_endpoint", default=None)
        or "/v1/chat/completions"
    )


# ---------------------------------------------------------------------------
# Live-server URL resolution (async / live modes)
# ---------------------------------------------------------------------------

def _resolve_live_base_url(cfg: DictConfig) -> str:
    """Resolve the live judge server URL, in priority order.

    1. ``cfg.judge.base_url`` if set and not the literal interpolation
       string (Hydra leaves unresolved ``${...}`` references as text
       when the variable isn't defined; we treat those as unset).
    2. ``cfg.judge_server_url`` (legacy alias).
    3. ``$JUDGE_SERVER_URL`` environment variable.
    """
    raw = OmegaConf.select(cfg, "judge.base_url", default=None)
    url = str(raw or "").strip()
    # Hydra leaves unresolved ${var} interpolations as the literal text
    # when the referenced var is undefined. Treat that as unset rather
    # than handing it to JudgeClient and watching the requests fail.
    if url.startswith("${") and url.endswith("}"):
        url = ""
    if not url:
        url = str(OmegaConf.select(cfg, "judge_server_url", default="") or "").strip()
    if url.startswith("${") and url.endswith("}"):
        url = ""
    if not url:
        url = os.environ.get("JUDGE_SERVER_URL", "").strip()
    return url


def _list_served_models(client: JudgeClient) -> Tuple[Optional[list], Optional[str]]:
    """Probe ``/v1/models`` on the judge endpoint.

    Returns ``(model_ids, error)``. On success, ``model_ids`` is a list
    of strings; on failure, ``error`` is a short human-readable string.
    """
    try:
        # JudgeClient builds an OpenAI SDK client lazily; reuse it so
        # the auth header / base_url normalization match what we'd send
        # in live mode.
        if client._client is None:
            return None, "judge client constructed in offline mode (no SDK client)"
        models = list(client._client.models.list())
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    ids = [getattr(m, "id", None) for m in models]
    return [i for i in ids if i], None


def _resolve_live_model_name(
    cfg: DictConfig,
    *,
    base_url: str,
    dagspace: str,
    default_max_tokens: int,
) -> Tuple[str, JudgeClient]:
    """Probe the live judge server and resolve the body.model name.

    Returns ``(model_name, online_client)`` where ``online_client`` is
    a non-offline JudgeClient against the live endpoint — the caller
    can reuse it for the actual export (after switching to offline) or
    discard it.

    Raises :class:`JudgeExportConfigError` if the endpoint is
    unreachable, has no models loaded, or ``judge.model_name`` is set
    explicitly to a value the server doesn't serve.
    """
    requested = str(
        OmegaConf.select(cfg, "judge.model_name", default="default") or "default"
    )

    # Build an online client to probe /v1/models. Fail loudly here
    # rather than during the per-row sidecar fan-out. The probe uses
    # provider="vllm" so JudgeClient skips its commercial-API key
    # requirement; we never actually POST a chat completion through
    # this client.
    try:
        probe = JudgeClient(
            base_url=base_url,
            model_name=requested,
            provider="vllm",
            max_workers=1,
            max_tokens=default_max_tokens,
            offline=False,
        )
    except Exception as exc:
        raise JudgeExportConfigError(
            f"[{dagspace}] failed to construct judge probe client for "
            f"{base_url!r}: {type(exc).__name__}: {exc}"
        ) from exc

    served, err = _list_served_models(probe)
    if served is None:
        raise JudgeExportConfigError(
            f"[{dagspace}] judge server probe failed at {base_url}/v1/models "
            f"({err}). The async-mode export pipeline refuses to write "
            f"requests.jsonl when the judge endpoint is unreachable — "
            f"fix the URL (server.env: JUDGE_SERVER_URL) or start the "
            f"judge_server.sub job before re-running."
        )
    if not served:
        raise JudgeExportConfigError(
            f"[{dagspace}] judge server at {base_url} reports zero served "
            f"models. Is scripts/judge_server.sub still loading? Re-run "
            f"once /v1/models returns a populated list."
        )

    if requested in ("", "default"):
        resolved = served[0]
        print(
            f"[{dagspace}] judge.model_name=default → resolved to "
            f"{resolved!r} via {base_url}/v1/models "
            f"(served: {served})",
            flush=True,
        )
        return resolved, probe

    if requested not in served:
        raise JudgeExportConfigError(
            f"[{dagspace}] judge.model_name={requested!r} is not served by "
            f"{base_url}. /v1/models returned: {served}. Either set "
            f"judge.model_name=default to auto-pick, or load the requested "
            f"model into the judge server."
        )
    return requested, probe


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def resolve_export_client(
    cfg: DictConfig,
    *,
    dagspace: str,
    default_max_tokens: int = 1024,
) -> Tuple[JudgeClient, Dict[str, Any]]:
    """Build a :class:`JudgeClient` for writing ``requests.jsonl``.

    Branches on ``cfg.judge.mode``:

    - ``async`` — probe ``cfg.judge.base_url`` / ``$JUDGE_SERVER_URL``,
      resolve ``judge.model_name`` (auto-detect ``default`` from
      ``/v1/models``), build an offline vLLM-flavored JudgeClient
      stamped with the resolved model name. The sidecar will forward
      each line back to the same endpoint, so the model name lines up
      with what's actually served.
    - ``batch_export`` — read ``judge.batch.target_model`` (no fallback
      default — operators must spell out the model so typos like
      ``gpt-5.2`` fail fast). Construct an offline OpenAI-flavored
      client; the JSONL is destined for OpenAI's Batch API.

    Returns ``(client, info)`` where ``info`` carries telemetry the
    caller logs alongside the manifest:

    - ``mode`` — resolved judge mode
    - ``base_url`` — async only; the probed live endpoint
    - ``served_models`` — async only; what /v1/models returned
    - ``target_model`` — batch_export only

    Raises :class:`JudgeExportConfigError` for any unrecoverable
    config problem (unreachable async judge, missing batch target,
    unsupported mode).
    """
    mode = resolve_judge_mode(cfg)
    temperature = float(
        OmegaConf.select(cfg, "judge.temperature", default=0.0) or 0.0
    )
    max_tokens = int(
        OmegaConf.select(cfg, "judge.max_tokens", default=default_max_tokens)
        or default_max_tokens
    )

    if mode == "async":
        base_url = _resolve_live_base_url(cfg)
        if not base_url:
            raise JudgeExportConfigError(
                f"[{dagspace}] judge.mode=async but no live judge endpoint "
                f"is configured. Set JUDGE_SERVER_URL in server.env (or "
                f"judge.base_url at the CLI) before running an async-mode "
                f"export. The previous behavior — silently using "
                f"judge.batch.target_model — produced 100%-defaulted metrics."
            )
        resolved_model, _probe = _resolve_live_model_name(
            cfg,
            base_url=base_url,
            dagspace=dagspace,
            default_max_tokens=max_tokens,
        )
        client = JudgeClient(
            base_url=base_url,
            model_name=resolved_model,
            provider="vllm",
            temperature=temperature,
            max_tokens=max_tokens,
            offline=True,
        )
        info = {
            "mode": "async",
            "base_url": base_url,
            "model_name": resolved_model,
        }
        print(
            f"[{dagspace}] judge export (async): base_url={base_url} "
            f"model={resolved_model} (will be written into body.model)",
            flush=True,
        )
        return client, info

    if mode == "batch_export":
        target_model = OmegaConf.select(cfg, "judge.batch.target_model", default=None)
        target_model = str(target_model or "").strip()
        if not target_model:
            raise JudgeExportConfigError(
                f"[{dagspace}] judge.mode=batch_export but "
                f"judge.batch.target_model is unset. Set it explicitly "
                f"to the OpenAI model you want to bill against (e.g. "
                f"judge.batch.target_model=gpt-4o-mini). No default — "
                f"silently choosing one would re-introduce the 'gpt-5.2' "
                f"footgun."
            )
        client = JudgeClient(
            base_url="https://api.openai.com/v1",
            model_name=target_model,
            provider="openai",
            temperature=temperature,
            max_tokens=max_tokens,
            offline=True,
        )
        info = {"mode": "batch_export", "target_model": target_model}
        print(
            f"[{dagspace}] judge export (batch_export, offline): "
            f"target_model={target_model}",
            flush=True,
        )
        return client, info

    if mode == "live":
        raise JudgeExportConfigError(
            f"[{dagspace}] judge.mode=live does not write requests.jsonl. "
            f"Use the live runner (judge_leakage / judge_helpfulness) "
            f"instead of *_judge_batch_export for live judging."
        )
    raise JudgeExportConfigError(
        f"[{dagspace}] unknown judge.mode={mode!r}. "
        f"Expected one of: live, async, batch_export."
    )
