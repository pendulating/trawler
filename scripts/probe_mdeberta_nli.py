#!/usr/bin/env python
"""Probe an NLI cross-encoder (mDeBERTa-v3 XNLI) as a fast grounding judge.

A different model *class* than the reranker: instead of relevance, score
entailment between the governing norms (premise) and the flow (hypothesis).
Entailment ≈ "the norms support this flow" is arguably closer to normative
grounding/consistency than topical relevance. In-process (~280M), CPU-friendly,
no server. Compares Spearman vs the 27B-judge grounding on the same 432
flow-bearing pairs used for the reranker probe.
"""
from __future__ import annotations

import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.probe_reranker_framings import load_pairs, render_flow_nl
from scripts.validate_reranker_judge import _spearman, _pearson

MODEL = "/share/pierson/matt/zoo/models/mdeberta_v3_base_xnli_multilingual_nli_2mil7"
# config id2label: 0=entailment, 1=neutral, 2=contradiction
E, N, C = 0, 1, 2


def main():
    pairs = load_pairs()
    print(f"flow-bearing pairs: {len(pairs)}")
    teacher = [p["teacher"] for p in pairs]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()

    @torch.no_grad()
    def nli_probs(premises, hypotheses, bs=32):
        out = []
        for i in range(0, len(premises), bs):
            enc = tok(premises[i:i + bs], hypotheses[i:i + bs],
                      return_tensors="pt", truncation=True, max_length=512,
                      padding=True)
            logits = model(**enc).logits
            out.append(torch.softmax(logits, dim=-1))
        return torch.cat(out, 0)

    def norms_text(p):
        return " ".join(str(n.get("norm_articulation", "")) for n in p["norms"])

    def flow_text(p):
        return " ; ".join(render_flow_nl(f) for f in p["flows"])

    framings = {}

    # N1: premise = norms, hypothesis = flow (descriptive)
    pr = [norms_text(p) for p in pairs]
    hy = [flow_text(p) for p in pairs]
    probs = nli_probs(pr, hy)
    framings["N1 entail: P(norms ⊨ flow)"] = probs[:, E].tolist()
    framings["N1 entail-contra"] = (probs[:, E] - probs[:, C]).tolist()

    # N2: premise = norms, hypothesis = "It is appropriate that <flow>."
    hy2 = [f"It is appropriate that {flow_text(p)}." for p in pairs]
    probs2 = nli_probs(pr, hy2)
    framings["N2 entail: P(norms ⊨ 'appropriate that flow')"] = probs2[:, E].tolist()
    framings["N2 entail-contra"] = (probs2[:, E] - probs2[:, C]).tolist()

    # N3: swap — premise = flow, hypothesis = norms
    probs3 = nli_probs(hy, pr)
    framings["N3 entail (swap): P(flow ⊨ norms)"] = probs3[:, E].tolist()

    print("\n=== Spearman / Pearson vs teacher grounding (n=%d) ===" % len(pairs))
    rows = []
    for name, scores in framings.items():
        rho = _spearman(scores, teacher)
        r = _pearson(scores, teacher)
        rows.append((rho if rho is not None else -9, name, rho, r))
    for _, name, rho, r in sorted(rows, reverse=True):
        rho_s = f"{rho:.4f}" if rho is not None else "n/a"
        r_s = f"{r:.4f}" if r is not None else "n/a"
        print(f"  rho={rho_s}  pearson={r_s}   {name}")
    print("\nViability bar ~0.6. Reranker best was ~0.19.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
