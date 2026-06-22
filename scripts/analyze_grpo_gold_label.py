#!/usr/bin/env python3
"""GRPO v2-v5 gold-label + reward-subcomponent analysis.

Reads the four grpo_redesign_full reward_traces.jsonl files and computes the
metrics we steer on:
  (1) gold-conditional abstention (the over-abstention smoking gun),
  (2) reward-subcomponent levels over training (growth check),
  (3) the R_ground=0-on-extraction audit (why extraction is under-rewarded:
      genuine ungrounding vs contrastive-clamp vs judge failure),
  (4) within-group extract-vs-abstain advantage over training.

gold_has_exchange is stored directly in every trace (100% populated), so the
conditional metrics are exact — no fingerprint reconstruction needed.

Usage: python scripts/analyze_grpo_gold_label.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
from collections import defaultdict

RUNS = ["v2", "v3", "v4", "v5"]
LAMBDA = 1.0  # contrastive_lambda used in all four runs


def load(v):
    paths = glob.glob(
        f"multirun/*grpo_redesign_full_{v}*/**/reward_traces.jsonl", recursive=True
    )
    if not paths:
        return [], []
    rows = []
    with open(paths[0]) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    ci = [r for r in rows if r.get("task_type") == "ci_extraction"]
    nj = [r for r in rows if r.get("task_type") == "norm_judgment"]
    return ci, nj


def rg_diag(r):
    """Return the single ranked rground_flows diagnostic dict, or None."""
    flows = r.get("rground_flows") or []
    if flows and isinstance(flows, list) and isinstance(flows[0], dict):
        return flows[0]
    return None


def pct(x, n):
    return f"{100*x/n:.1f}%" if n else "  -  "


print("=" * 96)
print("TABLE 1 — gold-label behaviour across runs (exact, from stored gold_has_exchange)")
print("=" * 96)
hdr = ("run", "n_ci", "goldYES%", "abstain%", "abst|YES", "abst|NO", "ext-gap",
       "Rg|ext", "Rg=0|ext", "comp|abs", "comp|ext")
print(("{:<4}" + "{:>9}" * 10).format(*hdr))
RUN_CI = {}
for v in RUNS:
    ci, nj = load(v)
    RUN_CI[v] = ci
    n = len(ci)
    gold_yes = [r for r in ci if r.get("gold_has_exchange") is True]
    gold_no = [r for r in ci if r.get("gold_has_exchange") is False]
    abst = [r for r in ci if r.get("is_no_flow")]
    ext = [r for r in ci if not r.get("is_no_flow")]
    abst_y = [r for r in gold_yes if r.get("is_no_flow")]
    abst_n = [r for r in gold_no if r.get("is_no_flow")]
    ext_y = [r for r in gold_yes if not r.get("is_no_flow")]
    ext_n = [r for r in gold_no if not r.get("is_no_flow")]
    rg_ext = [r["components"]["r_ground"] for r in ext if r.get("components")]
    rg0 = [x for x in rg_ext if x == 0.0]
    comp_abs = [r["composite"] for r in abst]
    comp_ext = [r["composite"] for r in ext]
    ext_rate_y = len(ext_y) / max(len(gold_yes), 1)
    ext_rate_n = len(ext_n) / max(len(gold_no), 1)
    print(("{:<4}" + "{:>9}" * 10).format(
        v, n,
        pct(len(gold_yes), n),
        pct(len(abst), n),
        pct(len(abst_y), len(gold_yes)),
        pct(len(abst_n), len(gold_no)),
        f"{100*(ext_rate_y-ext_rate_n):.1f}pp",
        f"{st.mean(rg_ext):.3f}" if rg_ext else "-",
        pct(len(rg0), len(rg_ext)),
        f"{st.mean(comp_abs):+.3f}" if comp_abs else "-",
        f"{st.mean(comp_ext):+.3f}" if comp_ext else "-",
    ))

print()
print("=" * 96)
print("TABLE 2 — R_ground=0-on-extraction AUDIT (why is extraction under-rewarded?)")
print("  Decompose each zero via rground_flows: correct_score vs λ·wrong_grounding")
print("  classes: JUDGE_FAIL | CLAMPED (correct_score>0.2 but wrong cancels) | GENUINE (correct_score<=0.2)")
print("=" * 96)
hdr2 = ("run", "n_ext", "Rg=0", "judge_fail", "clamped", "genuine",
        "mean_corr0", "mean_wrong0", "ext_meanW", "Rg|λ=0", "Rg|λ=.5")
print(("{:<4}" + "{:>10}" * 10).format(*hdr2))
for v in RUNS:
    ext = [r for r in RUN_CI[v] if not r.get("is_no_flow")]
    n_ext = len(ext)
    zeros = []
    all_corr, all_wrong = [], []
    rg_l0, rg_lhalf = [], []  # counterfactual R_ground at lambda 0 / 0.5
    for r in ext:
        d = rg_diag(r)
        if d is None:
            continue
        cs = d.get("correct_score")
        wg = d.get("wrong_grounding")
        jf = d.get("judge_failed", False)
        if cs is None:
            continue
        wg = wg or 0.0
        all_corr.append(cs)
        all_wrong.append(wg)
        rg_l0.append(max(0.0, min(1.0, cs)))
        rg_lhalf.append(max(0.0, min(1.0, cs - 0.5 * wg)))
        if (r.get("components") or {}).get("r_ground", None) == 0.0:
            zeros.append((cs, wg, jf))
    nz = len(zeros)
    judge_fail = sum(1 for cs, wg, jf in zeros if jf)
    clamped = sum(1 for cs, wg, jf in zeros if not jf and cs > 0.2)
    genuine = sum(1 for cs, wg, jf in zeros if not jf and cs <= 0.2)
    mc0 = st.mean([cs for cs, _, _ in zeros]) if zeros else 0.0
    mw0 = st.mean([wg for _, wg, _ in zeros]) if zeros else 0.0
    print(("{:<4}" + "{:>10}" * 10).format(
        v, n_ext, nz,
        pct(judge_fail, nz), pct(clamped, nz), pct(genuine, nz),
        f"{mc0:.3f}", f"{mw0:.3f}",
        f"{st.mean(all_wrong):.3f}" if all_wrong else "-",
        f"{st.mean(rg_l0):.3f}" if rg_l0 else "-",
        f"{st.mean(rg_lhalf):.3f}" if rg_lhalf else "-",
    ))
print("  (Rg|λ=0 / Rg|λ=.5 = counterfactual mean R_ground on extractors if the contrastive")
print("   penalty were removed / halved — gap vs Table-1 Rg|ext shows how much λ=1.0 costs.)")

print()
print("=" * 96)
print("TABLE 3 — reward-subcomponent levels over training (v5), extractors only, 8 windows")
print("  Does any discriminative component GROW? (flat => no learning on reward shape)")
print("=" * 96)
ci = RUN_CI["v5"]
ext = [r for r in ci if not r.get("is_no_flow") and r.get("components")]
ext.sort(key=lambda r: r.get("call", 0))
comps = ["r_uncert", "r_complete", "r_consist", "r_context", "r_cohere", "r_ground"]
nbin = 8
if ext:
    sz = max(1, len(ext) // nbin)
    print(("{:<8}" + "{:>9}" * 7).format("window", *[c.replace("r_", "") for c in comps], "compos"))
    for b in range(nbin):
        chunk = ext[b * sz:(b + 1) * sz] if b < nbin - 1 else ext[b * sz:]
        if not chunk:
            continue
        steps = [r.get("call", 0) for r in chunk]
        means = [st.mean([r["components"][c] for r in chunk]) for c in comps]
        cm = st.mean([r["composite"] for r in chunk])
        print(("{:<8}" + "{:>9}" * 7).format(
            f"{min(steps)}-{max(steps)}", *[f"{m:.3f}" for m in means], f"{cm:+.3f}"))

print()
print("=" * 96)
print("TABLE 4 — within-group extract-vs-abstain advantage over training (v5, gold=YES MIXED groups)")
print("  group = (call); MIXED = has >=1 abstain AND >=1 extract; advantage = mean(ext comp) - mean(abst comp)")
print("=" * 96)
groups = defaultdict(list)
for r in RUN_CI["v5"]:
    groups[r.get("call", -1)].append(r)
mixed_adv = []  # (call, adv, n_ext, n_abs)
allabs = 0
allext = 0
for call, members in groups.items():
    gy = [r for r in members if r.get("gold_has_exchange") is True]
    if not gy:
        continue
    a = [r["composite"] for r in gy if r.get("is_no_flow")]
    e = [r["composite"] for r in gy if not r.get("is_no_flow")]
    if a and e:
        mixed_adv.append((call, st.mean(e) - st.mean(a), len(e), len(a)))
    elif a and not e:
        allabs += 1
    elif e and not a:
        allext += 1
mixed_adv.sort()
print(f"  gold=YES groups: {len(mixed_adv)} mixed | {allabs} all-abstain (zero grad) | {allext} all-extract")
if mixed_adv:
    print(f"  overall mean advantage (ext-abst) in mixed groups: {st.mean([a for _,a,_,_ in mixed_adv]):+.3f}")
    nb = 6
    sz = max(1, len(mixed_adv) // nb)
    print("  " + ("{:<12}" + "{:>12}").format("call-window", "mean_adv"))
    for b in range(nb):
        chunk = mixed_adv[b * sz:(b + 1) * sz] if b < nb - 1 else mixed_adv[b * sz:]
        if not chunk:
            continue
        print("  " + ("{:<12}" + "{:>12}").format(
            f"{chunk[0][0]}-{chunk[-1][0]}",
            f"{st.mean([a for _,a,_,_ in chunk]):+.3f}"))

print()
print("=" * 96)
print("TABLE 5 — norm_judgment accuracy across runs (conservative-prior check)")
print("=" * 96)
print(("{:<4}" + "{:>10}" * 6).format("run", "n_nj", "gold-match", "recall(yes)", "recall(no)", "maj-base", "say-yes%"))
for v in RUNS:
    _, nj = load(v)
    import re
    n = len(nj)
    correct = yes_tot = yes_ok = no_tot = no_ok = say_yes = 0
    for r in nj:
        m = re.search(r'"judgment"\s*:\s*"([^"]+)"', r.get("completion", ""))
        pred = m.group(1).lower().strip() if m else None
        gold = str(r.get("gold_judgment", "")).lower().strip()
        if pred == "yes":
            say_yes += 1
        if pred == gold:
            correct += 1
        if gold == "yes":
            yes_tot += 1
            yes_ok += pred == "yes"
        elif gold == "no":
            no_tot += 1
            no_ok += pred == "no"
    maj = max(yes_tot, no_tot)
    print(("{:<4}" + "{:>10}" * 6).format(
        v, n, pct(correct, n), pct(yes_ok, yes_tot), pct(no_ok, no_tot),
        pct(maj, n), pct(say_yes, n)))
