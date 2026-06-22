#!/bin/bash
# Smoke test for the June 2026 GRPO redesign (Phases 1-5).
#
# Runs the full grpo_only_online_external pipeline with
# training/grpo=online_rground_external_smoke (6 optimizer steps,
# prescreen at 4 samples/prompt, ranked judge, gated composition,
# vignette mix, dev split) on a ~200-chunk sample, then asserts every
# new artifact exists and runs the promotion-gates checker.
#
# This validates PLUMBING (all new code paths execute and produce their
# artifacts), not learning — 6 steps say nothing about reward trends, so
# a HOLD verdict from the gates checker is expected and non-fatal here.
#
# Prerequisites: server.env (auto-sourced) provides EMBEDDING_SERVER_URL,
# JUDGE_SERVER_URL, NORM_UNIVERSES_PATH, CI_REASONING_PATH; the embedding
# and judge servers must be up (scripts/launch_auxiliary_servers.sh).
#
# Usage:
#   ./scripts/smoke_grpo_redesign.sh            # submit via SLURM (slurm_train_1x)
#   LOCAL=1 ./scripts/smoke_grpo_redesign.sh    # run in-process (needs a GPU on this node)
#   SAMPLE_N=400 ./scripts/smoke_grpo_redesign.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/pierson/matt/UAIR}"
cd "$PROJECT_ROOT"

SAMPLE_N="${SAMPLE_N:-200}"
SMOKE_DIR="${SMOKE_DIR:-$PROJECT_ROOT/multirun/$(date +%Y-%m-%d)_grpo_redesign_smoke/$(date +%H-%M-%S)}"

# -------- source server.env so pre-flight sees the same values Hydra will --------
if [ -f "$PROJECT_ROOT/server.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$PROJECT_ROOT/server.env"
    set +a
fi

# -------- environment sanity --------
required_vars=(
    EMBEDDING_SERVER_URL
    JUDGE_SERVER_URL
    NORM_UNIVERSES_PATH
    CI_REASONING_PATH
)
missing=0
for v in "${required_vars[@]}"; do
    if [ -z "${!v:-}" ]; then
        echo "ERROR: env var $v is not set (expected in server.env or shell)." >&2
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "Set the missing variables and retry." >&2
    exit 1
fi

# -------- aux server health checks --------
for url_label in "${EMBEDDING_SERVER_URL}:embed" "${JUDGE_SERVER_URL}:judge"; do
    url="${url_label%:*}"
    label="${url_label##*:}"
    if ! curl -sf "${url}/health" > /dev/null 2>&1; then
        echo "ERROR: ${label} server unhealthy at ${url}/health." >&2
        echo "  Launch with: sbatch scripts/judge_server.sub  (or scripts/launch_auxiliary_servers.sh)" >&2
        exit 1
    fi
    echo "  ✓ ${label} server reachable at ${url}"
done

# -------- run the pipeline --------
overrides=(
    pipeline=grpo_only_online_external
    training/grpo=online_rground_external_smoke
    model=qwen3.5-9b/sft-ci
    "runtime.sample_n=${SAMPLE_N}"
    wandb.enabled=false
    "hydra.run.dir=${SMOKE_DIR}"
)
if [ "${LOCAL:-0}" = "1" ]; then
    # Run the training node in this process instead of submitting to SLURM.
    overrides+=("pipeline.graph.nodes.grpo_training.launcher=null")
    echo "Running LOCALLY (needs 1 free GPU on this node)"
fi

echo "Smoke output dir: ${SMOKE_DIR}"
python -m dagspaces.grpo_training.cli "${overrides[@]}"

# -------- artifact assertions --------
CKPT="${SMOKE_DIR}/grpo_only_online_external/outputs/grpo/checkpoint"
echo ""
echo "=== Artifact checks (${CKPT}) ==="

fail=0
check_file() {
    if [ -e "$1" ]; then
        echo "  ✓ $2"
    else
        echo "  ✗ MISSING: $2 ($1)" >&2
        fail=1
    fi
}

check_file "${CKPT}/adapter_model.safetensors"     "GRPO LoRA adapter"
check_file "${CKPT}/prescreen_report.json"         "Phase 2: prescreen report"
check_file "${CKPT}/training_metadata.json"        "training metadata sidecar"
check_file "${CKPT}/reward_traces.jsonl"           "reward traces"
check_file "${CKPT}/checkpoint-3/trainer_state.json" "mid-run checkpoint (gates input)"

if [ -f "${CKPT}/training_metadata.json" ]; then
    python - "$CKPT" <<'EOF' || fail=1
import glob
import json
import os
import sys

ckpt = sys.argv[1]
meta = json.load(open(f"{ckpt}/training_metadata.json"))
problems = []
for key, expect in [("rground_scoring", "ranked"),
                    ("reward_composition", "gated"),
                    ("scale_rewards", "none")]:
    if meta.get(key) != expect:
        problems.append(f"{key}={meta.get(key)!r} (expected {expect!r})")
if meta.get("beta", 0) <= 0:
    problems.append(f"beta={meta.get('beta')} (expected > 0)")
if meta.get("n_dev_rows", 0) <= 0:
    problems.append("n_dev_rows=0 (dev split did not trigger)")

print(f"  prescreen: {json.load(open(f'{ckpt}/prescreen_report.json'))}")
print(f"  metadata: G={meta.get('num_generations')} beta={meta.get('beta')} "
      f"scoring={meta.get('rground_scoring')} composition={meta.get('reward_composition')} "
      f"train_rows={meta.get('n_training_rows')} dev_rows={meta.get('n_dev_rows')} "
      f"screened_out={meta.get('n_screened_out')}")

ranked_traced = False
with open(f"{ckpt}/reward_traces.jsonl") as f:
    for line in f:
        if '"type": "ranked"' in line:
            ranked_traced = True
            break
if ranked_traced:
    print("  ✓ ranked-judge diagnostics in traces")
else:
    problems.append("no ranked-mode diagnostics in reward_traces.jsonl")

# Held-out eval should appear in the highest checkpoint's log history
step_dirs = sorted(
    glob.glob(f"{ckpt}/checkpoint-*"),
    key=lambda p: int(p.rsplit("-", 1)[1]),
)
if step_dirs:
    state = json.load(open(os.path.join(step_dirs[-1], "trainer_state.json")))
    if any("eval_reward" in e for e in state.get("log_history", [])):
        print(f"  ✓ held-out eval_reward logged ({os.path.basename(step_dirs[-1])})")
    else:
        problems.append(f"eval_reward missing from {step_dirs[-1]}/trainer_state.json")
else:
    problems.append("no checkpoint-* directories found")

if problems:
    print("\n  ✗ FAILED checks:")
    for p in problems:
        print(f"    - {p}")
    sys.exit(1)
EOF
fi

# -------- promotion gates (informative on a 6-step run) --------
echo ""
echo "=== Promotion gates (informative — HOLD is expected at 6 steps) ==="
python scripts/check_grpo_promotion_gates.py "${CKPT}" || true

echo ""
if [ "$fail" -ne 0 ]; then
    echo "SMOKE TEST FAILED — see ✗ items above."
    exit 1
fi
echo "SMOKE TEST PASSED — all redesign code paths executed and produced artifacts."
echo "Next: full single-cell run at λ=1.0 (training/grpo=online_rground_external),"
echo "then gates → eval only on PROMOTE."
