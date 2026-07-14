#!/bin/bash
# Job 2 of the top100-flows GRPO plan: GRPO on FRESH top100 extraction prompts.
#
# The data lever. Three reward iterations (v9 symmetric floor, v10 false-permit
# floor, v11 rebalanced vignettes) all pinned at GoldCoin Forbid recall 0.55;
# the diagnosed non-reward constraints are (a) 704-prompt smallness, (b) the
# ~5:1 permissive skew of fiction10 governing norms, (c) the epoch-2 verdict
# freeze. This run attacks all three with a new DATA REGIME as one bundle:
#   - extraction prompts from the top100 corpus (fresh chunks, ~5x pool,
#     richer prohibited share per the top100 scaling analysis),
#   - grounding + contrastive + vignette universes ALL from the top100 build
#     (97 sources; wrong-universe draws come from 96 candidates instead of 9),
#   - 1 epoch over a capped, seeded chunk sample (fresh data at every step —
#     the anti-freeze design), instead of 3 epochs over a small pool.
# The REWARD is held at the best-validated config (v11 = v10 floor 0.1 +
# balanced vignettes, hedge tier OFF). Set HEDGE_PROHIBIT=0.5 to compose the
# v12a hedge tier instead — do this only if the v12a run's mid-run traces
# showed the tier moving commit share (decision gate in the plan note).
#
# PREREQUISITES (user's call — this script does NOT start them):
#   1. Aux servers up: embedding @ :8001, judge @ :8002
#      (scripts/launch_auxiliary_servers.sh on klara).
#   2. Job 1 done: scripts/run_extract_top100_flows.sh
#      (auto-discovered below; override TOP100_CI_REASONING_PATH to pin).
#   3. The top100 norm_universe build (auto-discovered, as in the v11 probe).
#
# Knobs (env): TOP100_CI_REASONING_PATH, TARGET_FLOW_CHUNKS (default 1400,
#   0 = no cap), EXCLUDE_BOOKS (default "35,215,6133" — the three books the
#   has_norms gate zeroed: no normative universe -> empty grounding retrieval),
#   HEDGE_PROHIBIT (default null = v11 reward), NUM_EPOCHS (default 1).
#
# Wall-time model: v11 probe = 528 steps / ~37h (~4.2 min/step, 4 prompts/step).
# TARGET_FLOW_CHUNKS=1400 -> ~2800 extraction prompts + 30% vignettes
# ~= 4000 pre-screen -> ~2500 post-screen (v11 keep-rate 64%) -> ~625 steps
# ~= ~44h. Checkpoints every 50 steps; kill-at-peak discipline applies.
#
# Plan: wiki/grpo_training_field_notes/2026-07-08_top100_flows_plan.md

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# ── 1. Resolve the top100 flows-reasoning parquet (job 1's output) ──────────
if [[ -z "${TOP100_CI_REASONING_PATH:-}" ]]; then
  TOP100_CI_REASONING_PATH=$(ls -t \
    "$PROJECT_ROOT"/outputs/*/*/COLM_flows_reasoning_qwen36/outputs/ci_reasoning/reasoning.parquet \
    "$PROJECT_ROOT"/outputs/*/*/*/COLM_flows_reasoning_qwen36/outputs/ci_reasoning/reasoning.parquet \
    "$PROJECT_ROOT"/multirun/*/*/COLM_flows_reasoning_qwen36/outputs/ci_reasoning/reasoning.parquet \
    "$PROJECT_ROOT"/multirun/*/*/*/COLM_flows_reasoning_qwen36/outputs/ci_reasoning/reasoning.parquet \
    2>/dev/null | head -1 || true)
fi
if [[ -z "${TOP100_CI_REASONING_PATH:-}" || ! -f "$TOP100_CI_REASONING_PATH" ]]; then
  echo "ERROR: no top100 flows reasoning.parquet found. Run job 1 first:" >&2
  echo "  scripts/run_extract_top100_flows.sh" >&2
  exit 1
fi
echo "[top100-flows] raw reasoning : $TOP100_CI_REASONING_PATH"

# ── 2. Curate: drop universe-less books, apply the seeded chunk cap ─────────
CURATED_PATH="${CURATED_PATH:-/share/pierson/matt/n2s4cir/data/top100flows/ci_reasoning.parquet}"
EXCLUDE_BOOKS="${EXCLUDE_BOOKS:-35,215,6133}"
TARGET_FLOW_CHUNKS="${TARGET_FLOW_CHUNKS:-1400}"

RAW_PATH="$TOP100_CI_REASONING_PATH" CURATED_PATH="$CURATED_PATH" \
EXCLUDE_BOOKS="$EXCLUDE_BOOKS" TARGET_FLOW_CHUNKS="$TARGET_FLOW_CHUNKS" \
python - <<'EOF'
import os
import pandas as pd

raw = os.environ["RAW_PATH"]
out = os.environ["CURATED_PATH"]
exclude = {b.strip() for b in os.environ["EXCLUDE_BOOKS"].split(",") if b.strip()}
target_flow = int(os.environ["TARGET_FLOW_CHUNKS"])

df = pd.read_parquet(raw)
n0, b0 = len(df), df["gutenberg_id"].astype(str).nunique()

# Drop books with no normative universe (all chunks gated has_norms=False in
# the norms extraction): their flows would retrieve an empty norm set and
# feed the grounding judge junk.
df = df[~df["gutenberg_id"].astype(str).isin(exclude)].reset_index(drop=True)

flow_rate = float(df["has_information_exchange"].mean())
n_flow = int(df["has_information_exchange"].sum())
print(f"[curate] raw: {n0} chunks / {b0} books")
print(f"[curate] after excluding {sorted(exclude)}: {len(df)} chunks / "
      f"{df['gutenberg_id'].astype(str).nunique()} books, "
      f"flow rate {flow_rate:.3f} ({n_flow} flow chunks)")

# Seeded uniform chunk cap so the run lands near the wall-time budget
# (1 epoch over ~TARGET_FLOW_CHUNKS*2 extraction prompts + 30% vignettes).
# The cap is baked into the curated artifact — what trains is what's on disk,
# and the prescreen cache key hashes exactly this prompt set.
if target_flow > 0 and n_flow > target_flow:
    n_sample = min(len(df), round(target_flow / flow_rate))
    df = df.sample(n=n_sample, random_state=42).reset_index(drop=True)
    print(f"[curate] capped to {n_sample} chunks (seed 42) -> "
          f"{int(df['has_information_exchange'].sum())} flow chunks "
          f"(target {target_flow})")
else:
    print(f"[curate] no cap applied (flow chunks {n_flow} <= target {target_flow} "
          f"or cap disabled)")

os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_parquet(out, index=False)
print(f"[curate] wrote {out} ({len(df)} rows)")
EOF

export CI_REASONING_PATH="$CURATED_PATH"

# ── 3. Point grounding + contrastive + vignettes at the top100 universes ────
if [[ -z "${TOP100_UNIVERSES_PATH:-}" ]]; then
  TOP100_UNIVERSES_PATH=$(ls -t \
    "$PROJECT_ROOT"/outputs/*/*/norm_universe_only/outputs/norm_universe/norm_universes.json \
    "$PROJECT_ROOT"/multirun/*/*/norm_universe_only/outputs/norm_universe/norm_universes.json \
    2>/dev/null | head -1 || true)
fi
if [[ -z "${TOP100_UNIVERSES_PATH:-}" || ! -f "$TOP100_UNIVERSES_PATH" ]]; then
  echo "ERROR: no top100 norm_universes.json found. Build it first:" >&2
  echo "  ABSTRACTED_NORMS_PATH=.../role_abstraction/abstracted_norms.parquet \\" >&2
  echo "    python -m dagspaces.grpo_training.cli pipeline=norm_universe_only model=qwen3.5-9b/sft-contentless-v6" >&2
  exit 1
fi
EMB_DIR="$(dirname "$TOP100_UNIVERSES_PATH")/embeddings"
if [[ ! -d "$EMB_DIR" ]] || ! ls "$EMB_DIR"/*.npy >/dev/null 2>&1; then
  echo "ERROR: embeddings dir missing/empty next to the universe build: $EMB_DIR" >&2
  exit 1
fi
# Exported vars beat server.env (ensure_dotenv uses override=False).
export NORM_UNIVERSES_PATH="$TOP100_UNIVERSES_PATH"
export NORM_EMBEDDINGS_PATH="$EMB_DIR"
export VIGNETTE_NORM_UNIVERSES_PATH="$TOP100_UNIVERSES_PATH"

# ── 4. Launch ────────────────────────────────────────────────────────────────
HEDGE_PROHIBIT="${HEDGE_PROHIBIT:-null}"   # null = v11 reward; 0.5 = compose v12a
NUM_EPOCHS="${NUM_EPOCHS:-1}"

echo "[top100-flows] training data  : $CI_REASONING_PATH"
echo "[top100-flows] universes      : $NORM_UNIVERSES_PATH (grounding+contrastive+vignettes)"
echo "[top100-flows] embeddings     : $NORM_EMBEDDINGS_PATH"
echo "[top100-flows] reward         : floor_prohibit=0.1, hedge_prohibit=$HEDGE_PROHIBIT"
echo "[top100-flows] epochs         : $NUM_EPOCHS (anti-freeze: fresh data every step)"

# Cell discipline: every knob that defines this cell is pinned explicitly,
# never riding on a yaml default (the yaml default is currently v12a!).
python -m dagspaces.grpo_training.cli \
  pipeline=grpo_only_online_external \
  training/grpo=online_rground_external \
  model=qwen3.5-9b/sft-contentless-v6 \
  training.grpo.rground_app_floor_prohibit=0.1 \
  training.grpo.rground_app_hedge_prohibit="$HEDGE_PROHIBIT" \
  training.grpo.num_epochs="$NUM_EPOCHS" \
  training.grpo.prescreen.cache_path=/share/pierson/matt/UAIR/cache/grpo_prescreen_top100flows.json \
  experiment.name=grpo_top100_flows
