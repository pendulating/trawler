"""k-series held-out probe + promotion gates (plan §5, §8; K2).

Pure logic for the two-tier held-out probe and the four pre-registered
promotion gates. The GPU driver lives in ``scripts/kto_heldout_probe.py``;
the CLI verdict wrapper in ``scripts/check_kto_promotion.py``. Everything
here is deterministic and unit-tested — the probe *measurements* go
through the production labeler (``kto_data_prep.label_completion``), never
a reimplementation.

Two-tier design (K2 plan, sign-off 2026-08-01): a full probe per save is
unaffordable (~48 GPU-h across 40 checkpoint-arms), so tier "screen" uses
a fixed seeded ~150-chunk subset (ALL held-out gold-NO + stratified
gold-YES) at every save, and tier "full" (all held-out chunks) confirms
only gate-passing checkpoints. The noise floor (0.011) was measured on 94
chunks at n=8, so the screening subset reads stronger than the floor
assumes.

Uniform-bin discipline: gates quote per-save full-subset aggregates —
never trailing windows.
"""
from __future__ import annotations

import random
from typing import Any

import pandas as pd

#: Measured constants (plan §5/§12: K0 two-seed probe on 94 chunks, n=8).
NOISE_FLOOR = 0.011
PROMOTION_BAR = 0.022  # 2x the noise floor, pre-registered
GATE_FAIL_MAX = 0.08   # plan §3/§8


# ---------------------------------------------------------------------------
# Tier-1 subset selection (deterministic)
# ---------------------------------------------------------------------------
def select_probe_chunks(
    heldout_keys: list[str],
    gold_yes: dict[str, bool],
    tier: str,
    n_screen: int = 150,
    seed: int = 42,
) -> list[str]:
    """The probe chunk set for a tier.

    ``tier="full"``: every held-out chunk. ``tier="screen"``: ALL held-out
    gold-NO chunks (the abstention gate needs them all — there are only
    ~25) + a per-book round-robin sample of gold-YES chunks up to
    ``n_screen`` total. Deterministic under ``seed``; sorted input order.
    """
    keys = sorted(heldout_keys)
    if tier == "full":
        return keys
    if tier != "screen":
        raise ValueError(f"unknown probe tier {tier!r}")
    no = [k for k in keys if not gold_yes.get(k, False)]
    yes = [k for k in keys if gold_yes.get(k, False)]
    n_yes = max(0, n_screen - len(no))
    by_book: dict[str, list[str]] = {}
    for k in yes:
        by_book.setdefault(k.split("|", 1)[0], []).append(k)
    for book, ks in by_book.items():
        random.Random(f"{seed}|{book}").shuffle(ks)
    picked: list[str] = []
    # Round-robin over books so every held-out universe is represented.
    while len(picked) < min(n_yes, len(yes)):
        for book in sorted(by_book):
            if by_book[book] and len(picked) < n_yes:
                picked.append(by_book[book].pop())
    return sorted(no + picked)


# ---------------------------------------------------------------------------
# Per-completion scoring row (driver calls the production labeler)
# ---------------------------------------------------------------------------
def probe_row(
    chunk_key: str,
    is_gold_yes: bool,
    label_result: dict[str, Any],
    no_flow: bool,
    n_teacher_flows: int,
) -> dict[str, Any]:
    """Flatten one production-labeler result into a probe record.

    ``label_result`` is ``kto_data_prep.label_completion`` output (or
    ``{"status": "gate_fail"}``); per-flow gold/correct pairs are reduced
    to violation/appropriate counts — the §8 gate inputs.
    """
    row = {
        "chunk_key": chunk_key,
        "gold_yes": bool(is_gold_yes),
        "status": label_result["status"],
        "no_flow": bool(no_flow),
        "n_teacher_flows": int(n_teacher_flows),
        "n_matched": 0, "n_viol": 0, "n_viol_correct": 0,
        "n_appr": 0, "n_appr_correct": 0,
    }
    for gold, correct in label_result.get("per_flow", []):
        if gold == "inappropriate":
            row["n_viol"] += 1
            row["n_viol_correct"] += int(correct)
        else:
            row["n_appr"] += 1
            row["n_appr_correct"] += int(correct)
    row["n_matched"] = row["n_viol"] + row["n_appr"]
    return row


def summarize_checkpoint(df: pd.DataFrame) -> dict[str, float]:
    """Pooled §8 gate metrics over one (arm, checkpoint) probe slice."""
    yes = df[df["gold_yes"]]
    no = df[~df["gold_yes"]]
    gated = df[df["status"] != "gate_fail"]
    viol = int(yes["n_viol"].sum())
    appr = int(yes["n_appr"].sum())
    teacher = int(yes["n_teacher_flows"].sum())
    return {
        "n_completions": len(df),
        "minority_acc": (yes["n_viol_correct"].sum() / viol) if viol else float("nan"),
        "majority_acc": (yes["n_appr_correct"].sum() / appr) if appr else float("nan"),
        "gate_fail_rate": 1.0 - len(gated) / len(df) if len(df) else float("nan"),
        "abstain_rate_gold_no": (no["no_flow"].mean() if len(no) else float("nan")),
        "miss_rate": (1.0 - yes["n_matched"].sum() / teacher) if teacher else float("nan"),
    }


# ---------------------------------------------------------------------------
# The four §8 gates
# ---------------------------------------------------------------------------
def evaluate_promotion_gates(
    curve: list[dict[str, Any]],
    baseline: dict[str, float],
    noise_floor: float = NOISE_FLOOR,
    promotion_bar: float = PROMOTION_BAR,
    gate_fail_max: float = GATE_FAIL_MAX,
) -> list[dict[str, Any]]:
    """Apply the pre-registered §8 gates to one arm's checkpoint curve.

    ``curve`` is ordered by training step: dicts with ``checkpoint`` plus
    the ``summarize_checkpoint`` metrics. ``baseline`` is the epoch-0 SFT
    policy's metrics on the SAME probe subset. Returns per-checkpoint
    verdict rows; ``promoted`` requires ALL of:

      1. minority gain > promotion_bar at this save AND the previous one
         (sustained — a single-save spike is noise);
      2. majority_acc >= baseline - noise_floor (no seesaw into paranoia);
      3. gate_fail_rate <= gate_fail_max AND gold-NO abstention >=
         baseline - noise_floor (R-ABSTAIN regression check);
      4. miss_rate <= baseline + noise_floor (no winning by extracting
         fewer, safer flows).
    """
    out = []
    prev_g1 = False
    for point in curve:
        gain = point["minority_acc"] - baseline["minority_acc"]
        g1_now = gain > promotion_bar
        g2 = point["majority_acc"] >= baseline["majority_acc"] - noise_floor
        g3 = (point["gate_fail_rate"] <= gate_fail_max
              and point["abstain_rate_gold_no"]
              >= baseline["abstain_rate_gold_no"] - noise_floor)
        g4 = point["miss_rate"] <= baseline["miss_rate"] + noise_floor
        out.append({
            "checkpoint": point["checkpoint"],
            "minority_gain": round(gain, 4),
            "g1_sustained_gain": g1_now and prev_g1,
            "g2_majority_holds": g2,
            "g3_format_abstain": g3,
            "g4_no_recall_dodge": g4,
            "promoted": (g1_now and prev_g1) and g2 and g3 and g4,
        })
        prev_g1 = g1_now
    return out
