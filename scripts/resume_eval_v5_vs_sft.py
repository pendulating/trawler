#!/usr/bin/env python
"""Resume the v5-vs-SFT privacylens eval from the (now-fixed) judge stage.

The original multirun (multirun/2026-06-19_eval_v5_vs_sft_priv/17-48-24)
completed the GPU inference stages (qa_probe_inference, agent_action_inference)
for both arms, then crashed in the *synchronous* leakage judge on a
``NameError: name 'skipped' is not defined`` (fixed in
dagspaces/privacylens/stages/llm_inference.py). Re-running the whole pipeline
would needlessly re-do GPU inference and queue behind the live v6 GRPO run on
klara. Instead we resume from exactly where it failed: reuse the existing
agent_action / qa_probe parquets and run the CPU-only judge + metrics stages
against the already-running judge server (klara:8002).

This is the same code path privacylens_clean uses (the real stage functions),
just driven directly over the cached inputs.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

# Load server.env so judge.base_url's ${oc.env:JUDGE_SERVER_URL} resolves.
from dagspaces.common.stage_utils import ensure_dotenv
ensure_dotenv()

from dagspaces.privacylens.stages.llm_inference import (
    run_leakage_judge_inference,
    run_helpfulness_judge_inference,
)
from dagspaces.privacylens.stages.compute_metrics import (
    compute_metrics,
    metrics_to_dataframe,
)

RUN = Path("multirun/2026-06-19_eval_v5_vs_sft_priv/17-48-24")
ARMS = {"0": "sft-ci", "1": "grpo-redesign-v5"}


def _served_model_name(base_url: str) -> str | None:
    """Ask the judge server what model id it actually serves.

    The privacylens config hard-codes ``judge.model_name="default"``, which
    only works if the vLLM server was launched with
    ``--served-model-name default``. When the judge server restarts without
    that flag it serves under its filesystem path and ``default`` 404s. Query
    ``/v1/models`` and use the real id so the eval is robust to restarts.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=10) as r:
            data = json.load(r)
        ids = [m["id"] for m in data.get("data", [])]
        return ids[0] if ids else None
    except Exception as e:
        print(f"[warn] could not query /v1/models: {e}", flush=True)
        return None


def resume_arm(arm: str, label: str) -> dict:
    out = RUN / arm / "privacylens_eval" / "outputs"
    cfg = OmegaConf.load(RUN / arm / ".hydra" / "config.yaml")

    # Pin the judge model to whatever the server actually serves (the
    # config's "default" alias is dropped on a server restart -> 404s).
    base_url = str(OmegaConf.select(cfg, "judge.base_url", default="") or "")
    served = _served_model_name(base_url)
    if served:
        OmegaConf.update(cfg, "judge.model_name", served, force_add=True)
        print(f"[arm {arm}] judge model -> {served}", flush=True)

    agent_df = pd.read_parquet(out / "agent_action_inference" / "results.parquet")
    qa_df = pd.read_parquet(out / "qa_probe_inference" / "results.parquet")
    print(f"\n=== arm {arm} ({label}): {len(agent_df)} agent rows, {len(qa_df)} qa rows ===",
          flush=True)

    leakage_df = run_leakage_judge_inference(agent_df, cfg)
    helpfulness_df = run_helpfulness_judge_inference(agent_df, cfg)

    # Persist judge outputs so the run dir matches a clean pipeline execution.
    for name, df in (("leakage_judge_inference", leakage_df),
                     ("helpfulness_judge_inference", helpfulness_df)):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / "results.parquet", index=False)

    metrics = compute_metrics(qa_df, leakage_df, helpfulness_df)
    cm = out / "compute_metrics"
    cm.mkdir(parents=True, exist_ok=True)
    metrics_to_dataframe(metrics).to_parquet(cm / "metrics.parquet", index=False)
    with open(cm / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[arm {arm}] wrote {cm/'metrics.json'}", flush=True)
    return metrics


def headline(label: str, m: dict) -> dict:
    qa = m.get("qa_probing", {}) or {}
    leak = m.get("leakage", {}) or {}
    helpf = m.get("helpfulness", {}) or {}
    adj = m.get("adjusted_leakage", {}) or {}
    return {
        "arm": label,
        "qa_accuracy": round(qa.get("accuracy", 0.0), 4),
        "leak_rate_parseable": round(leak.get("leakage_rate_among_parseable", 0.0), 4),
        "action_format_rate": round(leak.get("agent_action_format_rate", 0.0), 4),
        "helpful_mean_parseable": round(helpf.get("mean_score_among_parseable", 0.0), 4),
        "adjusted_leak_rate": round(adj.get("adjusted_leakage_rate", 0.0), 4),
    }


def main():
    rows = []
    for arm, label in ARMS.items():
        m = resume_arm(arm, label)
        rows.append(headline(label, m))
    print("\n================ v5-vs-SFT ground truth ================", flush=True)
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False), flush=True)
    summary.to_csv(RUN / "v5_vs_sft_summary.csv", index=False)
    print(f"\nwrote {RUN/'v5_vs_sft_summary.csv'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
