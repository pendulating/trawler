# Project Overview

## What the paper does

Teaches LLMs Contextual Integrity (CI) privacy reasoning by:

1. **Extract** — mine `(sender, recipient, subject, attribute, transmission-principle)` information flow tuples and `(deontic, subject, act, condition)` Raz norms from 10 public-domain fiction novels using Qwen2.5-72B-AWQ as the extractor.
2. **Fine-tune** — SFT (LoRA) on `(chunk → reasoning + structured IFTs)` pairs, then GRPO with a 6-component composite reward anchored by a *normative grounding* judge that scores completions against the per-book normative universe. Per-completion contrastive scoring penalizes alignment with a wrong book's norms.
3. **Evaluate** — five CI benchmarks: GoldCoin-HIPAA, VLM-GeoPrivacy, PrivacyLens, ConfAIde, CI-RL Vignettes.

## Source corpus (10 novels, Project Gutenberg)

*1984*, *Pride and Prejudice*, *Anna Karenina*, *Bleak House*, *Les Misérables*, *Middlemarch*, *The Count of Monte Cristo*, *The Age of Innocence*, *The Picture of Dorian Gray*, *Alice's Adventures in Wonderland*. Religious texts are **out of scope** for COLM 2026.

## End-to-end pipeline (high level)

```
 fiction novels (Gutenberg)
      │
      ▼  dagspaces/historical_norms
 chunks.parquet → reasoning → extraction → role_abstraction
      │                                      │
      │                           abstracted_norms.parquet
      │                                      │
      │                                      ▼  dagspaces/grpo_training
      │                              norm_universe + embeddings
      │                                      │
      └──► sft_data_prep ──► sft_training ──► reward_prep
                                                  │
                                                  ▼
                                          grpo_training (policy update)
                                                  │
                                                  ▼
                                   dagspaces/{goldcoin_hipaa,
                                               privacylens,
                                               vlm_geoprivacy_bench,
                                               confaide,
                                               cirl_vignettes}
```

## Research questions (paper)

- **RQ1**: Can fiction-derived normative simulacra teach privacy reasoning that transfers to real-world CI benchmarks?
- **RQ2**: What does GRPO add over SFT alone?
- **RQ3**: What does the normative-grounding judge reward add over purely programmatic rewards?

## Key results framing

- SFT instills a conservative prior (restrict flow) — improves recognition but not judgment correctness.
- SFT + GRPO with `R_ground` wins on GoldCoin-HIPAA compliance and correlates best with human ConfAIde annotations.
- Contrastive scoring (λ=1.0 default) prevents memorizing source-specific norms.

## Stack

Python 3.12 only. vLLM ≥0.17 for inference (multi-GPU via mp backend), TRL + PEFT for SFT/GRPO, Hydra 1.3 for config, hydra-submitit for SLURM, Weights & Biases for tracking, sentence-transformers for `R_ground` retrieval.
