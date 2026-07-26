#!/usr/bin/env python3
"""One-time backfill of `act_polarity` over an already-built norm universe.

Why: `FORCE_TO_APPROPRIATENESS` (and `FORCE_TO_GOLD`) assume `norm_act` names
the action affirmatively. When the act is phrased as an abstention ("refrain
from disclosing…"), the force applies to the *refraining*, so the derived
judgment of the underlying flow inverts — a norm that OBLIGES refraining makes
the disclosure INappropriate, yet the mapping returns "appropriate". Found
2026-07-25; a regex bounded the affected share at <=26.4% but over-flags
compound acts ("ensure accuracy AND avoid misleading"), so the true rate was
unknown.

`act_polarity` is now a required field on `RazNormTuple` for new extractions.
This script populates it for the ALREADY-BUILT universes without re-running
norm extraction: the judgment needs only `norm_act` + `norm_articulation`, not
the source chunk. Deterministic (temperature 0), resumable, and writes a
versioned artifact so gold stays inspectable.

Outputs
-------
act_polarity.json   {gutenberg_id: {norm_index: "performing"|"refraining"}}
report.md           realized rate, inversion count, corrected class balance
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests

sys.path.insert(0, "/share/pierson/matt/UAIR")
from dagspaces.common.json_extraction import extract_json_from_text  # noqa: E402
from dagspaces.common.stage_utils import ensure_dotenv  # noqa: E402
from dagspaces.grpo_training.stages.deontic import (  # noqa: E402
    FORCE_TO_APPROPRIATENESS,
    flow_appropriateness,
)

UNIV = ("multirun/2026-07-23_universe_fiction10_gemma4/15-43-41/"
        "norm_universe_only/outputs/norm_universe/norm_universes.json")

# Mirrors the ci_schema.RazNormTuple.act_polarity description verbatim in
# substance, so the backfill is faithful to the canonical extraction ask.
SYSTEM = (
    "You label how a social norm's ACT is phrased. You are given the act and "
    "the norm as the society would state it.\n\n"
    "Answer 'performing' if norm_act names the regulated action itself.\n"
    "Answer 'refraining' if norm_act names the WITHHOLDING of that action.\n\n"
    "This is a question about HOW THE ACT IS PHRASED, and is INDEPENDENT of "
    "whether the norm permits or forbids. The same expectation can be written "
    "either way, and they are tagged differently:\n"
    "  act: 'discuss a family's private finances in company'            -> performing\n"
    "  act: 'refrain from discussing a family's private finances'       -> refraining\n"
    "  act: 'call on new neighbours within a fortnight'                 -> performing\n"
    "  act: 'avoid mentioning a guest's misfortune'                     -> refraining\n"
    "  act: 'withhold news of a death from an invalid'                  -> refraining\n\n"
    "If the act is compound and its PRIMARY verb phrase is affirmative "
    "(e.g. 'ensure the accuracy of claims and avoid misleading the audience'), "
    "answer 'performing'.\n\n"
    'Reply as JSON: {"labels": ["performing"|"refraining", ...]} with one '
    "entry per numbered item, in order."
)


def ask(url: str, model: str, items: list[dict], timeout: float = 120.0) -> list[str] | None:
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. act: {it['act']}")
        if it.get("articulation"):
            lines.append(f"   norm as stated: {it['articulation']}")
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": "\n".join(lines)}],
        "temperature": 0.0,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = requests.post(f"{url.rstrip('/')}/v1/chat/completions",
                          json=body, timeout=timeout)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"] or ""
    except Exception as exc:
        print(f"  [warn] request failed: {exc}")
        return None
    obj, _ = extract_json_from_text(raw, repair=True)
    if not isinstance(obj, dict):
        return None
    labels = obj.get("labels")
    if not isinstance(labels, list) or len(labels) != len(items):
        return None
    out = []
    for x in labels:
        v = str(x).strip().lower()
        out.append(v if v in ("performing", "refraining") else "performing")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--out", default="outputs/2026-07-25_act_polarity_backfill")
    ap.add_argument("--limit", type=int, default=0, help="0 = all eligible")
    args = ap.parse_args()

    ensure_dotenv()
    url = os.environ.get("JUDGE_SERVER_URL") or os.environ.get("VLLM_SERVER_URL")
    model = requests.get(f"{url.rstrip('/')}/v1/models", timeout=15).json()["data"][0]["id"]
    print(f"[backfill] labeller: {os.path.basename(model)} @ {url}")

    universe = json.load(open(UNIV))
    todo = []
    for gid, norms in universe.items():
        for idx, n in enumerate(norms):
            force = str(n.get("normative_force") or "").strip().lower()
            if (n.get("governs_info_flow") is True
                    and force in FORCE_TO_APPROPRIATENESS
                    and str(n.get("context") or "").strip()):
                todo.append({
                    "gid": gid, "idx": idx, "force": force,
                    "act": str(n.get("norm_act") or ""),
                    "articulation": str(n.get("norm_articulation") or "")[:220],
                })
    if args.limit:
        todo = todo[: args.limit]
    print(f"[backfill] eligible norms to label: {len(todo)}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "act_polarity.json")
    labels: dict[str, dict[str, str]] = {}
    if os.path.exists(out_path):                      # resumable
        labels = json.load(open(out_path))
        done = sum(len(v) for v in labels.values())
        print(f"[backfill] resuming — {done} already labelled")

    def is_done(it):
        return labels.get(it["gid"], {}).get(str(it["idx"])) is not None

    pending = [it for it in todo if not is_done(it)]
    for start in range(0, len(pending), args.batch):
        chunk = pending[start : start + args.batch]
        got = ask(url, model, chunk)
        if got is None:                               # per-item fallback
            got = []
            for it in chunk:
                one = ask(url, model, [it])
                got.append(one[0] if one else "performing")
        for it, lab in zip(chunk, got):
            labels.setdefault(it["gid"], {})[str(it["idx"])] = lab
        if (start // args.batch) % 20 == 0:
            json.dump(labels, open(out_path, "w"), indent=2)
            n = sum(len(v) for v in labels.values())
            print(f"  … {n}/{len(todo)} labelled", flush=True)
    json.dump(labels, open(out_path, "w"), indent=2)

    # ---- analysis --------------------------------------------------------
    n_ref = n_tot = inverted = 0
    old_mix = {"appropriate": 0, "inappropriate": 0}
    new_mix = {"appropriate": 0, "inappropriate": 0}
    for it in todo:
        pol = labels.get(it["gid"], {}).get(str(it["idx"]), "performing")
        n_tot += 1
        n_ref += pol == "refraining"
        old = FORCE_TO_APPROPRIATENESS[it["force"]]
        new = flow_appropriateness(it["force"], pol)
        old_mix[old] += 1
        new_mix[new] += 1
        inverted += old != new

    lines = [
        "# act_polarity backfill — fiction10-gemma4\n",
        f"- labelled: **{n_tot}** eligible norms",
        f"- phrased as `refraining`: **{n_ref} ({n_ref/max(1,n_tot):.1%})**",
        f"- **gold labels inverted by the fix: {inverted} ({inverted/max(1,n_tot):.1%})**\n",
        "## Flow-appropriateness class balance\n",
        "| | appropriate | inappropriate |",
        "|---|---|---|",
        f"| before (force only) | {old_mix['appropriate']} "
        f"({old_mix['appropriate']/max(1,n_tot):.1%}) | {old_mix['inappropriate']} "
        f"({old_mix['inappropriate']/max(1,n_tot):.1%}) |",
        f"| **after (force x polarity)** | {new_mix['appropriate']} "
        f"({new_mix['appropriate']/max(1,n_tot):.1%}) | {new_mix['inappropriate']} "
        f"({new_mix['inappropriate']/max(1,n_tot):.1%}) |",
    ]
    open(os.path.join(args.out, "report.md"), "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[backfill] wrote {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
