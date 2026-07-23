#!/usr/bin/env bash
# Launch the GoldCoin variance TOP-UP sweep: extra seed reps (104-108) for
# the model configs whose GoldCoin rep range in the N=3 judge-free variance
# sweep exceeded a threshold. See the sweep yaml header for the
# one-server-boot-per-model design:
#   dagspaces/eval_all/conf/sweep/eval_judgefree_variance_topup_goldcoin_2026_07_21.yaml
#
# The roster is computed HERE, from the finished N=3 sweep on disk — the
# yaml deliberately carries no model list.
#
# Env knobs:
#   WAIT_FOR_JOB=<slurm id>   gate on that job/array draining first
#   GC_RANGE_THRESHOLD=2.0    display-unit (pct) range above which a config
#                             is topped up (checked on Appl. AND Comp.)
#   FORCE_INCOMPLETE=1        allow roster selection before the N=3 sweep
#                             has 3 goldcoin reps everywhere (ranges from
#                             <3 reps UNDERESTIMATE noise — top-up may
#                             miss configs; refuse by default)
#
# Usage:
#   WAIT_FOR_JOB=150351 nohup scripts/run_eval_judgefree_variance_topup.sh \
#       > logs/eval_judgefree_variance_topup.log 2>&1 &

set -uo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# Absolute driver python (submitit bakes sys.executable — never activate).
PYTHON="${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/python"

# ── Phase 0: wait for a prior job/array to drain ──────────────────────────
# squeue is an ssh shim; transient failure looks like a drained queue, so
# require 3 consecutive empty polls.
if [ -n "${WAIT_FOR_JOB:-}" ]; then
  echo "[topup] waiting for job ${WAIT_FOR_JOB} to drain..."
  empty=0
  while [ "${empty}" -lt 3 ]; do
    if squeue -h -j "${WAIT_FOR_JOB}" 2>/dev/null | grep -q .; then
      empty=0
    else
      empty=$((empty + 1))
      echo "[topup] queue empty for ${WAIT_FOR_JOB} (${empty}/3) at $(date)"
    fi
    sleep 300
  done
  echo "[topup] job ${WAIT_FOR_JOB} drained at $(date)"
fi

# ── Phase 1: roster selection from the N=3 sweep on disk ─────────────────
ROSTER=$("$PYTHON" - <<'EOF'
import json
import os
import sys
from pathlib import Path

THRESHOLD = float(os.environ.get("GC_RANGE_THRESHOLD", "2.0"))
FORCE = os.environ.get("FORCE_INCOMPLETE", "") == "1"

MULTIRUN = Path("/share/pierson/matt/UAIR/multirun")
GC_METRICS = [
    ("appl", "compute_metrics_applicability"),
    ("comp", "compute_metrics_compliance"),
]

# (model override token) -> {seed: {appl: v, comp: v}}
by_config: dict[str, dict[int, dict[str, float]]] = {}
for pat in ("*_eval_judgefree_variance/*", "*_eval_judgefree_variance_gptoss/*"):
    for vdir in sorted(MULTIRUN.glob(pat)):
        for sub in sorted(p for p in vdir.iterdir()
                          if p.is_dir() and p.name.isdigit()):
            ov = sub / ".hydra" / "overrides.yaml"
            if not ov.exists():
                continue
            model = seed = None
            for line in ov.read_text(errors="ignore").splitlines():
                line = line.strip().lstrip("- ").strip()
                if line.startswith("model="):
                    model = line.split("=", 1)[1]
                elif line.startswith("variance_seed="):
                    seed = int(line.split("=", 1)[1])
            if model is None or seed is None:
                continue
            vals = {}
            for short, subdir in GC_METRICS:
                mp = (sub / "goldcoin" / "goldcoin_hipaa" / "outputs"
                      / subdir / "metrics.json")
                if mp.exists():
                    try:
                        vals[short] = float(
                            json.loads(mp.read_text())["macro_f1"])
                    except (ValueError, OSError, KeyError):
                        pass
            if vals:
                by_config.setdefault(model, {})[seed] = vals

if not by_config:
    print("no goldcoin variance data found on disk", file=sys.stderr)
    sys.exit(1)

incomplete = sorted(m for m, reps in by_config.items() if len(reps) < 3)
if incomplete and not FORCE:
    print(f"{len(incomplete)} configs have <3 goldcoin reps (sweep "
          f"incomplete?): {incomplete[:6]}{'...' if len(incomplete) > 6 else ''}\n"
          "Ranges from <3 reps underestimate noise. Re-run when the N=3 "
          "sweep is done, or set FORCE_INCOMPLETE=1.", file=sys.stderr)
    sys.exit(1)

roster, dropped_gptoss = [], []
for model, reps in sorted(by_config.items()):
    if len(reps) < 2:
        continue
    worst = 0.0
    for short, _ in GC_METRICS:
        vals = [v[short] for v in reps.values() if short in v]
        if len(vals) >= 2:
            worst = max(worst, (max(vals) - min(vals)) * 100.0)
    if worst > THRESHOLD:
        if model.startswith("gpt-oss"):
            dropped_gptoss.append((model, round(worst, 1)))
        else:
            roster.append((model, round(worst, 1)))

for model, rng in roster:
    print(f"select {model} (worst GC range {rng}pt)", file=sys.stderr)
for model, rng in dropped_gptoss:
    print(f"EXCLUDED {model} (range {rng}pt) — server_mode raises on "
          "harmony; top up in-process separately", file=sys.stderr)
if not roster:
    print(f"no config exceeded GC_RANGE_THRESHOLD={THRESHOLD}pt — "
          "nothing to top up", file=sys.stderr)
    sys.exit(2)
print(",".join(m for m, _ in roster))
EOF
)
rc=$?
if [ $rc -ne 0 ] || [ -z "$ROSTER" ]; then
  echo "[topup] roster selection failed or empty (rc=${rc}) — NOT launching."
  exit $rc
fi
n_models=$(echo "$ROSTER" | tr ',' '\n' | wc -l)
echo "[topup] roster (${n_models} configs): ${ROSTER}"

# ── Phase 2: launch (one arm per model; 5 seed reps inside each cell) ────
"$PYTHON" -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_judgefree_variance_topup_goldcoin_2026_07_21 \
  "model=${ROSTER}"
main_rc=$?
echo "[topup] sweep finished rc=${main_rc} at $(date)"
exit $main_rc
