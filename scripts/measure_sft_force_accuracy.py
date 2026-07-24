#!/usr/bin/env python
"""Per-force-class deontic accuracy of a canonical SFT base on fiction10-gemma4 vignettes.

Standalone measurement for the m-series GRPO redesign (task-vignettes.md build
item; wiki/grpo_redesign/data.md job 3). Templates single-scenario deontic
vignettes from the built fiction10-gemma4 norm universe, asks an SFT base model
to classify the deontic force on the full 5-way scale (greedy / temperature 0),
and reports per-force-class accuracy, per-polarity accuracy, hedge rate,
antithesis rate, and mean deontic-distance score.

Scenario templating mimics the legacy builder ``_generate_vignettes``
(dagspaces/grpo_training/stages/grpo_training.py:28). ANTI-LEAK CONTRACT: the
scenario text derived from a norm must never contain the norm's
``norm_articulation`` nor any deontic force word — asserted programmatically for
every prompt before inference.

Usage (one model per invocation, GPU required):

    python -m scripts.measure_sft_force_accuracy \
        --model dagspaces/common/conf/model/gemma-4-12b/sft-canonical.yaml \
        --model-name gemma-4-12b_sft-canonical

    # after both model runs, aggregate:
    python -m scripts.measure_sft_force_accuracy --aggregate

This file lives under scripts/ and writes only under outputs/; it imports (but
does not modify) dagspaces/ utilities.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Constants — the canonical inputs and scoring axis.
# ---------------------------------------------------------------------------

REPO = "/share/pierson/matt/UAIR"
UNIVERSE_PATH = (
    "/share/pierson/matt/UAIR/multirun/2026-07-23_universe_fiction10_gemma4/"
    "15-43-41/norm_universe_only/outputs/norm_universe/norm_universes.json"
)
OUTPUT_DIR = (
    "/share/pierson/matt/UAIR/outputs/2026-07-23_mseries_premeasure/"
    "sft_force_accuracy"
)

# Full 5-way answer scale (what the MODEL chooses among).
FORCE_OPTIONS = ["obligatory", "recommended", "permitted", "discouraged", "prohibited"]

# Deontic axis (task-vignettes.md).
AXIS = {
    "obligatory": 2,
    "recommended": 1,
    "permitted": 0,
    "discouraged": -1,
    "prohibited": -2,
}

# Gold force classes eligible for measurement (permitted excluded per
# FORCE_TO_GOLD in dagspaces/grpo_training/stages/deontic.py; the model still
# answers on the full 5-way scale).
GOLD_FORCES = ["obligatory", "recommended", "discouraged", "prohibited"]

# Gold polarity: yes = should flow, no = should not.
POLARITY = {
    "obligatory": "yes",
    "recommended": "yes",
    "permitted": None,
    "discouraged": "no",
    "prohibited": "no",
}

# Anti-leak force-word matcher: the five force words and adjective/adverb/noun
# variants (roots). Applied only to the SCENARIO text derived from a norm —
# NOT the fixed instruction, whose five force OPTIONS must appear.
FORCE_WORD_RE = re.compile(
    r"\b(oblig\w*|recommend\w*|permit\w*|permiss\w*|discourag\w*|prohibit\w*)\b",
    re.IGNORECASE,
)

SEED_CAP_PER_CLASS = 150

INSTRUCTION = (
    "You are given a scenario describing a potential flow of information. "
    "Decide the deontic force that governs whether the information described "
    "should flow in that situation. Choose exactly one label from this scale:\n"
    "  - obligatory: the flow is required\n"
    "  - recommended: the flow is favored but not required\n"
    "  - permitted: the flow is neither favored nor disfavored\n"
    "  - discouraged: the flow is disfavored but not forbidden\n"
    "  - prohibited: the flow is forbidden\n\n"
    "Scenario:\n{scenario}\n\n"
    "Respond with a single JSON object and nothing else, in this form:\n"
    '{{"force": "<one of: obligatory, recommended, permitted, discouraged, '
    'prohibited>", "reasoning": "<one or two sentences>", '
    '"governing_norm": "<the rule you believe governs this scenario>"}}'
)


# ---------------------------------------------------------------------------
# Scenario templating — mirrors _generate_vignettes (grpo_training.py:28).
# ---------------------------------------------------------------------------


def build_scenario(norm: dict) -> str:
    """Build the scenario text from norm fields (omitting norm_articulation)."""
    subject = norm.get("norm_subject") or "a person"
    act = norm.get("norm_act") or "share this information"
    condition = norm.get("condition_of_application") or ""
    context = norm.get("context") or ""

    parts: list[str] = []
    if context:
        parts.append(f"In a setting involving {context},")
    parts.append(f"{subject}")
    if condition:
        parts.append(f"is in a situation where {condition}.")
    else:
        parts.append(f"is considering whether to {act}.")
    return " ".join(parts)


def scenario_leaks(scenario: str, norm: dict) -> str | None:
    """Return a leak reason if the scenario violates the anti-leak contract, else None."""
    m = FORCE_WORD_RE.search(scenario)
    if m:
        return f"force-word:{m.group(0)!r}"
    articulation = (norm.get("norm_articulation") or "").strip()
    if articulation and articulation.lower() in scenario.lower():
        return "articulation-substring"
    return None


# ---------------------------------------------------------------------------
# Deterministic stratified sampling.
# ---------------------------------------------------------------------------


def build_eligible(universe: dict) -> tuple[dict[str, dict[str, list]], dict]:
    """Group anti-leak-clean eligible norms per (force, book).

    Returns (per_force[force][book] -> list of (idx, norm, scenario), report).
    """
    per_force: dict[str, dict[str, list]] = {f: defaultdict(list) for f in GOLD_FORCES}
    dropped_leak: dict[str, int] = defaultdict(int)
    eligible_raw: dict[str, int] = defaultdict(int)

    for book_id in sorted(universe.keys()):
        norms = universe[book_id]
        for idx, norm in enumerate(norms):
            if norm.get("governs_info_flow") is not True:
                continue
            force = (norm.get("normative_force") or "").strip().lower()
            if force not in GOLD_FORCES:
                continue
            context = (norm.get("context") or "").strip()
            if not context:
                continue
            eligible_raw[force] += 1
            scenario = build_scenario(norm)
            if scenario_leaks(scenario, norm):
                dropped_leak[force] += 1
                continue
            per_force[force][book_id].append((idx, norm, scenario))

    report = {
        "eligible_raw_per_class": dict(eligible_raw),
        "dropped_anti_leak_per_class": dict(dropped_leak),
    }
    return per_force, report


def stratified_sample(per_force: dict[str, dict[str, list]]) -> list[dict]:
    """Round-robin across books per force class, up to SEED_CAP_PER_CLASS each.

    Fully deterministic: books sorted lexically, norms kept in universe order,
    interleaved round-robin so the sample spreads across books. No RNG / no
    wall-clock randomness.
    """
    samples: list[dict] = []
    for force in GOLD_FORCES:
        by_book = per_force[force]
        books = sorted(by_book.keys())
        # Round-robin cursor per book.
        pools = {b: list(by_book[b]) for b in books}
        collected: list[tuple[str, int, dict, str]] = []
        depth = 0
        max_depth = max((len(pools[b]) for b in books), default=0)
        while len(collected) < SEED_CAP_PER_CLASS and depth < max_depth:
            for b in books:
                if depth < len(pools[b]):
                    idx, norm, scenario = pools[b][depth]
                    collected.append((b, idx, norm, scenario))
                    if len(collected) >= SEED_CAP_PER_CLASS:
                        break
            depth += 1
        for b, idx, norm, scenario in collected:
            scenario = build_scenario(norm)
            leak = scenario_leaks(scenario, norm)
            # Hard guard — the pool is already anti-leak-clean.
            assert leak is None, f"anti-leak violated for {force} {b}:{idx}: {leak}"
            samples.append(
                {
                    "id": f"{b}:{idx}",
                    "book_id": b,
                    "norm_index": idx,
                    "gold_force": force,
                    "scenario": scenario,
                    "prompt_text": INSTRUCTION.format(scenario=scenario),
                    "norm_articulation": norm.get("norm_articulation") or "",
                }
            )
    return samples


# ---------------------------------------------------------------------------
# Inference (via dagspaces.common.vllm_inference.run_vllm_inference).
# ---------------------------------------------------------------------------


def run_inference(samples: list[dict], model_yaml: str):
    import pandas as pd
    from omegaconf import OmegaConf

    from dagspaces.common.vllm_inference import run_vllm_inference

    cfg = OmegaConf.load(model_yaml)
    if "model" not in cfg:
        raise ValueError(
            f"{model_yaml} has no top-level 'model:' block; expected a "
            "@package _global_ model config"
        )

    df = pd.DataFrame(samples)

    def preprocess(row: dict) -> dict:
        row["messages"] = [{"role": "user", "content": row["prompt_text"]}]
        row["sampling_params"] = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "seed": 0,
        }
        return row

    def postprocess(row: dict) -> dict:
        return row

    out = run_vllm_inference(
        df,
        cfg,
        preprocess=preprocess,
        postprocess=postprocess,
        stage_name="sft_force_accuracy",
    )
    return out


# ---------------------------------------------------------------------------
# Parsing + scoring.
# ---------------------------------------------------------------------------


def parse_force(generated_text: str) -> tuple[str, bool]:
    """Return (force, parsed_ok). Unparseable / non-force → ('permitted', False)."""
    from dagspaces.common.json_extraction import extract_json_from_text

    obj, _err = extract_json_from_text(generated_text or "", repair=True)
    if not isinstance(obj, dict):
        return "permitted", False
    raw = obj.get("force")
    if raw is None:
        return "permitted", False
    val = str(raw).strip().lower()
    for opt in FORCE_OPTIONS:
        if opt in val:
            # exact or contained (e.g. "prohibited." / "the force is prohibited")
            if val == opt or re.search(rf"\b{opt}\b", val):
                return opt, True
    return "permitted", False


def deontic_distance(model_force: str, gold_force: str) -> float:
    return 1.0 - abs(AXIS[model_force] - AXIS[gold_force]) / 2.0


def score(results: list[dict]) -> dict:
    """Compute per-class and aggregate metrics."""
    per_class: dict[str, dict] = {}
    by_class: dict[str, list] = defaultdict(list)
    for r in results:
        by_class[r["gold_force"]].append(r)

    for force in GOLD_FORCES:
        rows = by_class.get(force, [])
        n = len(rows)
        if n == 0:
            per_class[force] = {"n": 0}
            continue
        exact = sum(1 for r in rows if r["model_force"] == force)
        gold_pol = POLARITY[force]
        polarity_correct = sum(
            1 for r in rows if POLARITY[r["model_force"]] == gold_pol
        )
        hedge = sum(
            1 for r in rows if r["model_force"] == "permitted" or not r["parsed_ok"]
        )
        antithesis = sum(
            1
            for r in rows
            if POLARITY[r["model_force"]] is not None
            and POLARITY[r["model_force"]] != gold_pol
        )
        mean_s = sum(r["s"] for r in rows) / n
        parse_fail = sum(1 for r in rows if not r["parsed_ok"])
        per_class[force] = {
            "n": n,
            "force_accuracy": exact / n,
            "polarity_accuracy": polarity_correct / n,
            "hedge_rate": hedge / n,
            "antithesis_rate": antithesis / n,
            "mean_deontic_distance": mean_s,
            "parse_fail_rate": parse_fail / n,
        }

    # Aggregate over all decisive-gold rows.
    all_rows = [r for rows in by_class.values() for r in rows]
    n_all = len(all_rows)
    agg = {"n": n_all}
    if n_all:
        agg.update(
            {
                "force_accuracy": sum(
                    1 for r in all_rows if r["model_force"] == r["gold_force"]
                )
                / n_all,
                "polarity_accuracy": sum(
                    1
                    for r in all_rows
                    if POLARITY[r["model_force"]] == POLARITY[r["gold_force"]]
                )
                / n_all,
                "hedge_rate": sum(
                    1
                    for r in all_rows
                    if r["model_force"] == "permitted" or not r["parsed_ok"]
                )
                / n_all,
                "antithesis_rate": sum(
                    1
                    for r in all_rows
                    if POLARITY[r["model_force"]] is not None
                    and POLARITY[r["model_force"]] != POLARITY[r["gold_force"]]
                )
                / n_all,
                "mean_deontic_distance": sum(r["s"] for r in all_rows) / n_all,
                "parse_fail_rate": sum(1 for r in all_rows if not r["parsed_ok"])
                / n_all,
            }
        )

    # Per-polarity accuracy (grouped by gold polarity).
    per_polarity: dict[str, dict] = {}
    for pol in ("yes", "no"):
        rows = [r for r in all_rows if POLARITY[r["gold_force"]] == pol]
        n = len(rows)
        if n == 0:
            per_polarity[pol] = {"n": 0}
            continue
        per_polarity[pol] = {
            "n": n,
            "polarity_accuracy": sum(
                1 for r in rows if POLARITY[r["model_force"]] == pol
            )
            / n,
        }

    return {"per_class": per_class, "per_polarity": per_polarity, "aggregate": agg}


# ---------------------------------------------------------------------------
# Model driver.
# ---------------------------------------------------------------------------


def run_model(model_yaml: str, model_name: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(UNIVERSE_PATH, encoding="utf-8") as f:
        universe = json.load(f)

    per_force, elig_report = build_eligible(universe)
    samples = stratified_sample(per_force)

    realized_n = {f: sum(1 for s in samples if s["gold_force"] == f) for f in GOLD_FORCES}
    print(f"[{model_name}] realized sample n per class: {realized_n}", flush=True)
    print(f"[{model_name}] eligibility report: {elig_report}", flush=True)

    out_df = run_inference(samples, model_yaml)
    out_records = out_df.to_dict("records")

    results: list[dict] = []
    for rec in out_records:
        gen = rec.get("generated_text", "") or ""
        model_force, parsed_ok = parse_force(gen)
        s = deontic_distance(model_force, rec["gold_force"])
        results.append(
            {
                "id": rec["id"],
                "book_id": rec["book_id"],
                "norm_index": rec["norm_index"],
                "gold_force": rec["gold_force"],
                "model_force": model_force,
                "parsed_ok": bool(parsed_ok),
                "s": s,
                "scenario": rec["scenario"],
                "generated_text": gen,
            }
        )

    metrics = score(results)

    payload = {
        "model_name": model_name,
        "model_yaml": model_yaml,
        "universe_path": UNIVERSE_PATH,
        "seed_cap_per_class": SEED_CAP_PER_CLASS,
        "realized_n_per_class": realized_n,
        "eligibility_report": elig_report,
        "axis": AXIS,
        "gold_forces": GOLD_FORCES,
        "metrics": metrics,
        "records": results,
    }
    out_path = os.path.join(OUTPUT_DIR, f"{model_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[{model_name}] wrote {out_path}", flush=True)
    print(f"[{model_name}] aggregate: {json.dumps(metrics['aggregate'], indent=2)}",
          flush=True)


# ---------------------------------------------------------------------------
# Aggregation across models → summary.json + table.md.
# ---------------------------------------------------------------------------


def aggregate() -> None:
    model_files = sorted(
        f
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".json") and f not in ("summary.json",)
    )
    models: dict[str, dict] = {}
    for fn in model_files:
        with open(os.path.join(OUTPUT_DIR, fn), encoding="utf-8") as f:
            payload = json.load(f)
        models[payload["model_name"]] = {
            "model_yaml": payload["model_yaml"],
            "realized_n_per_class": payload["realized_n_per_class"],
            "eligibility_report": payload["eligibility_report"],
            "metrics": payload["metrics"],
        }

    summary = {
        "universe_path": UNIVERSE_PATH,
        "seed_cap_per_class": SEED_CAP_PER_CLASS,
        "models": models,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Markdown table.
    lines: list[str] = []
    lines.append("# SFT-base per-force-class deontic accuracy (fiction10-gemma4)\n")
    lines.append(f"Universe: `{UNIVERSE_PATH}`\n")
    lines.append(f"Cap: up to {SEED_CAP_PER_CLASS} norms/force class, "
                 "deterministic round-robin over books. Model answers on the "
                 "full 5-way scale; gold excludes `permitted`.\n")

    for model_name, m in models.items():
        met = m["metrics"]
        lines.append(f"\n## {model_name}\n")
        lines.append(f"Config: `{m['model_yaml']}`\n")
        agg = met["aggregate"]
        lines.append(
            "**Aggregate** (n={n}): force-acc {fa:.3f} · polarity-acc {pa:.3f} · "
            "hedge {h:.3f} · antithesis {a:.3f} · mean-s {s:.3f} · "
            "parse-fail {pf:.3f}\n".format(
                n=agg.get("n", 0),
                fa=agg.get("force_accuracy", float("nan")),
                pa=agg.get("polarity_accuracy", float("nan")),
                h=agg.get("hedge_rate", float("nan")),
                a=agg.get("antithesis_rate", float("nan")),
                s=agg.get("mean_deontic_distance", float("nan")),
                pf=agg.get("parse_fail_rate", float("nan")),
            )
        )
        pp = met["per_polarity"]
        yes = pp.get("yes", {})
        no = pp.get("no", {})
        lines.append(
            "Per-polarity: gold-yes acc {ya:.3f} (n={yn}) · "
            "gold-no acc {na:.3f} (n={nn})\n".format(
                ya=yes.get("polarity_accuracy", float("nan")),
                yn=yes.get("n", 0),
                na=no.get("polarity_accuracy", float("nan")),
                nn=no.get("n", 0),
            )
        )
        lines.append(
            "\n| gold force | n | force-acc | polarity-acc | hedge | "
            "antithesis | mean-s | parse-fail |\n"
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|\n"
        )
        for force in GOLD_FORCES:
            c = met["per_class"].get(force, {"n": 0})
            if c["n"] == 0:
                lines.append(f"| {force} | 0 | — | — | — | — | — | — |\n")
                continue
            lines.append(
                "| {f} | {n} | {fa:.3f} | {pa:.3f} | {h:.3f} | {a:.3f} | "
                "{s:.3f} | {pf:.3f} |\n".format(
                    f=force,
                    n=c["n"],
                    fa=c["force_accuracy"],
                    pa=c["polarity_accuracy"],
                    h=c["hedge_rate"],
                    a=c["antithesis_rate"],
                    s=c["mean_deontic_distance"],
                    pf=c["parse_fail_rate"],
                )
            )

    with open(os.path.join(OUTPUT_DIR, "table.md"), "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Wrote {os.path.join(OUTPUT_DIR, 'summary.json')} and table.md", flush=True)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="path to model yaml (@package _global_)")
    ap.add_argument("--model-name", help="output filename stem")
    ap.add_argument("--aggregate", action="store_true", help="combine model jsons")
    args = ap.parse_args()

    # Load server.env (SLURM/NCCL/paths) — most is handled downstream but be safe.
    try:
        from dagspaces.common.stage_utils import ensure_dotenv

        ensure_dotenv()
    except Exception as exc:
        print(f"WARNING: ensure_dotenv failed: {exc}", file=sys.stderr)

    if args.aggregate:
        aggregate()
        return

    if not args.model or not args.model_name:
        ap.error("--model and --model-name are required unless --aggregate")

    model_yaml = args.model
    if not os.path.isabs(model_yaml):
        model_yaml = os.path.join(REPO, model_yaml)
    run_model(model_yaml, args.model_name)


if __name__ == "__main__":
    main()
