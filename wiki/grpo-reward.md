# GRPO Composite Reward

GRPO (Shao et al. 2024) samples `G` completions per prompt, scores each via `R`, and updates policy from within-group relative rankings — no preference pairs, no separate reward model. CI's structured outputs admit programmatic verification, so most reward components are deterministic.

## Components

$R = \sum_i w_i R_i$ over six components. Gating signals saturate after SFT (low weight); discriminative signals carry the GRPO learning signal (high weight).

| # | Component | Weight | Type | What it measures |
|---|---|---|---|---|
| 1 | `R_uncert` (task clarity) | 0.10 | gating, programmatic | Schema validity + construct discrimination + extraction confidence |
| 2 | `R_complete` (structural completeness) | 0.05 | gating, programmatic | Presence of all required IFT fields |
| 3 | `R_consist` (internal consistency) | 0.05 | gating, programmatic | Reasoning ↔ extraction non-contradiction |
| 4 | `R_context` (context identification) | 0.20 | discriminative, programmatic | Stated context matches a prominent context in `N_b` |
| 5 | `R_cohere` (reasoning-to-extraction coherence) | 0.10 | discriminative, programmatic | Reasoning trace supports the extracted tuples |
| 6 | **`R_ground` (normative grounding)** | **0.50** | **discriminative, LLM judge** | **Judge evaluates flow against retrieved norms from `N_b`** |

Rationale for `R_ground=0.5`: pilots showed it has the **highest inter-completion variance**, making it the primary driver of GRPO's advantage estimates.

## `R_ground` details

For each extracted flow in a completion:
1. Retrieve `k=3` most similar norms from $\hat{\mathcal{N}}_b$ via semantic similarity (sentence-transformers embedding + cosine).
2. LLM judge (Qwen2.5-32B-Instruct by default) scores three criteria:
   - **Norm awareness**: invoked norms match retrieved norms
   - **Flow governance**: this flow is governed by the retrieved norms
   - **Appropriateness consistency**: appropriateness judgment agrees with the governing norm

Implementation lives under `dagspaces/grpo_training/runners/grpo_training.py` and the reward modules it imports; judge calls go through `dagspaces/common/judge_client.py` (HTTP) when `use_external_judge=true`.

## Per-completion contrastive scoring

Every `R_ground` call scores the completion against **both** the correct universe $\hat{\mathcal{N}}_b$ and a randomly chosen wrong universe $\hat{\mathcal{N}}_{b'}$:

$$R_{\text{ground}} = \text{clamp}\!\left(\bar{r}_{\text{correct}} - \lambda \cdot \bar{r}_{\text{wrong}},\ 0,\ 1\right)$$

This teaches the policy to **condition on context**, not memorize source-specific norms. Default λ = 1.0. Ablations in `app:grpo-ablation-viz`.

## Ablation configs (Qwen3.5-9B)

Model yamls under `dagspaces/common/conf/model/qwen3.5-9b/`:

| Config | Variant |
|---|---|
| `grpo-full-c10` | Full reward, contrastive λ=1.0 (primary) |
| `grpo-full-c00 / c05 / c15 / c20 / c50` | λ sweep |
| `grpo-ctx-c10` | Context-only ablation |
| `grpo-cohere-c10` | Coherence-only ablation |
| `grpo-structural-c10` | Structural-only ablation |
| `grpo-v2-lambda*`, `grpo-v3-*` | Reward formulation variants |
| `sft-and-progonly-grpo` | Phase1+Phase2 with `R_ground` zeroed |
| `sft-and-grounded-grpo` | Phase1+Phase2 with full reward |

## Running reward ablations

```bash
# Full reward
python -m dagspaces.grpo_training.cli -m pipeline=grpo_only \
  model=qwen3.5-9b/base training.grpo.use_vllm=false

# Programmatic-only (R_ground=0, weights redistributed)
python -m dagspaces.grpo_training.cli -m pipeline=grpo_programmatic_only \
  model=qwen3.5-9b/base training.grpo.use_vllm=false

# Different λ
python -m dagspaces.grpo_training.cli -m pipeline=grpo_only \
  model=qwen3.5-9b/grpo-full-c05 training.grpo.use_vllm=false
```
