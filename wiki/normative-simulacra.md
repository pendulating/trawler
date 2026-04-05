# Normative Simulacra

Structured representations extracted from fiction, serving dual roles:
1. **Training signal** for GRPO's `R_ground` reward (per-book normative universe).
2. **Standalone scholarly artifact** — a machine-readable model of each text's normative landscape.

## Two structured types

### CI information flow tuple (IFT)
Following Benthall et al. 2024: $(s, r, u, a, t)$
- `s` — sender
- `r` — recipient
- `u` — subject (information subject / data subject)
- `a` — attribute (information type)
- `t` — transmission principle

Annotated with: societal context, appropriateness (`appropriate` / `inappropriate` / `ambiguous`), invoked norms, extraction confidence.

### Raz norm
Following Raz 1999 (**prescriptive** interpretation of norms per Nissenbaum 2020): $(d, s, a, c)$
- `d` — deontic element (the prescriptive "ought")
- `s` — norm subject
- `a` — norm act
- `c` — condition of application

Annotated with: normative force (`obligatory` / `prohibited` / `permitted` / `recommended` / `discouraged`), societal context, informational-vs-conduct flag.

> CI uses the **prescriptive** interpretation; NormBank et al. use the descriptive (observed regularity) interpretation.

## Extraction pipeline (two-stage CoT)

Chunking: 6000-char chunks, 1000-char overlap, processed sequentially.

1. **Reasoning** (`prompt:norm-reasoning-fiction`) — identify normative content in the chunk; free-form reasoning.
2. **Structured extraction** (`prompt:norm-extraction-fiction`) — formalize reasoning into typed tuples via guided decoding.

Additional stage: **Role abstraction** — replace character names with social roles (the *role* matters more than the *person* for CI reasoning).

**The norm/flow distinction** is a critical design constraint:
- Norm track → what society *expects* (the "ought")
- Flow track → how information *moves* (descriptive exchange)

This distinction is verifiable and becomes a reward signal during GRPO.

### Robustness / fallback notes

- **Silent `has_information_exchange` fallback**: if guided decoding omits this field, `ci_reasoning` infers it from `len(flows) > 0`. Decoding failures can go unnoticed.
- **Conservative row-expansion defaults**: the flow-expansion step applies empty-string defaults on malformed flow entries so the pipeline doesn't crash mid-run. Schema validity is enforced upstream by guided decoding, but check parse traces if numbers look off.
- **Extraction receives a per-flow snippet**, not the full chunk, with fallback to `article_text` if the snippet is absent. Reasoning sees the whole chunk; extraction sees the snippet the reasoning pointed to. Affects context-dependent flows that need wider lookaround.

## Schemas

- `dagspaces/historical_norms/schema.py` — base schemas
- `dagspaces/historical_norms/ci_schema.py` — CI IFT schema
- `dagspaces/historical_norms/schema_builders.py` — per-stage guided-decoding schema builders

## Normative universe

For each source text $b$: aggregate abstracted norms into $\mathcal{N}_b$, embed with sentence-transformers, and cache embeddings to disk.

Built in `dagspaces/grpo_training/stages/norm_universe.py` (runner: `runners/norm_universe.py`). Output:
- `outputs/norm_universe/norm_universes.json` — per-book norm lists + metadata
- `outputs/norm_universe/embeddings/*.npy` — per-book embeddings

At GRPO time, each extracted flow retrieves top-k norms from $\hat{\mathcal{N}}_b$ via cosine similarity, and the judge scores against those.

## Extraction model

Default: **Qwen2.5-72B-Instruct-AWQ** on 2 GPUs with guided decoding for schema validity. Config: `dagspaces/common/conf/model/qwen2.5-72b/awq.yaml`.

Judge model for `R_ground`: Qwen2.5-32B-Instruct (configurable in judge server).
