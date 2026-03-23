# GRPO Training Dagspace: Technical Review

## Pipeline Overview

The `full_training` pipeline is a 5-node DAG with two parallel branches that converge at `grpo_training`:

```
                    ┌─ sft_data_prep ──→ sft_training ─────────────────┐
ci_reasoning ──────→│                                                   ├──→ grpo_training
ci_extraction ─────→│                                                   │
                    └─ norm_universe ──→ reward_prep ───────────────────┘
abstracted_norms ──→──┘
```

Three external sources feed the pipeline: `ci_reasoning.parquet` and `ci_extraction.parquet` (from the `historical_norms` dagspace, containing per-chunk reasoning and per-flow CI extractions from fiction novels), and `abstracted_norms.parquet` (raw Raz-format norms extracted per book).

---

## 1. SFT Process

### 1.1 Data Preparation (`sft_data_prep`)

**Input**: Per-flow exploded CI extraction rows + per-chunk CI reasoning rows from the `historical_norms` pipeline.

**Reconstruction**: The extraction data arrives exploded (one row per flow per chunk). The stage re-groups rows by `(gutenberg_id, chunk_id)` to reconstruct chunk-level training examples. For each chunk:

1. **Reasoning trace** — Per-flow reasoning entries are sorted by `ci_flow_index` and concatenated into a single narrative. Multi-flow chunks get numbered prefixes ("Flow 1:", "Flow 2:").

2. **Flow reconstruction** — Per-flow CI tuple fields (`ci_sender`, `ci_recipient`, `ci_subject`, `ci_information_type`, `ci_transmission_principle`) plus metadata (`ci_context`, `ci_appropriateness`, `ci_norms_invoked`, `ci_norm_source`, `ci_is_new_flow`, `ci_confidence_qual`) are assembled into flat flow dicts.

3. **Completion format** — A flat JSON object:
   ```json
   {
     "reasoning": "<narrative trace>",
     "has_information_exchange": true,
     "flows": [{ "sender": "...", "recipient": "...", ... }]
   }
   ```

4. **Chat messages** — Each example becomes a 2-turn conversation:
   - `user`: A fixed CI analysis instruction + the source chunk text
   - `assistant`: The flat JSON completion

**Negative examples**: Chunks where `has_information_exchange=False` (no CI flows detected) are included as negative training pairs at up to a 1:1 ratio with positive examples. These have empty `flows: []` and either the original reasoning text or a canned "no exchange" explanation. This teaches the model to correctly output empty flows when no information exchange is present — exercising the `r_consist` invariant. Negatives can be disabled via `training.sft.include_negative_examples: false`.

### 1.2 SFT Training (`sft_training`)

**Method**: LoRA fine-tuning via TRL's `SFTTrainer`.

**Model loading**:
- Base model loaded with bf16 precision, flash_attention_2 (with sdpa fallback)
- Optional QLoRA (4-bit NF4 quantization via BitsAndBytes) for larger models (e.g., 27B)
- Multi-GPU DDP via `accelerate launch` (the runner spawns a subprocess)

**LoRA configuration** (default):
- Rank: 64, Alpha: 128, Dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (all 7 linear projection layers)
- Task type: CAUSAL_LM

**Training hyperparameters** (default):
- 3 epochs, lr=2e-5, batch_size=2, gradient_accumulation=8 (effective batch=16)
- Warmup ratio: 0.1, weight decay: 0.01
- Max sequence length: 8192, gradient checkpointing enabled

**Chat template handling**: The tokenizer's native chat template is replaced with a TRL-compatible version that wraps assistant content in `{% generation %}...{% endgeneration %}` blocks. This tells `SFTTrainer` which tokens are completions (contribute to loss) vs. prompts (masked with label=-100). Without this, all labels become -100, producing zero loss. Templates exist for Qwen (ChatML), Phi-4, Gemma, Llama, and GPT-OSS families; family is auto-detected from the model path or tokenizer markers.

**Key detail for Qwen**: The Qwen SFT template injects an empty `<think>\n\n</think>\n` prefix before the assistant content inside the generation block. This trains the model to produce think blocks as part of its output format.

**Post-training**: The original tokenizer chat template is restored before saving the checkpoint — the `{% generation %}` template is training-only. The final checkpoint contains the LoRA adapter weights, adapter config, and tokenizer with the original template.

---

## 2. GRPO Process

### 2.1 Norm Universe Construction (`norm_universe`)

Before GRPO can train, each source book's normative universe (N̂_b) must be embedded for retrieval.

**Input**: `abstracted_norms.parquet` with `raz_*` columns (Raz-format norm decomposition: articulation, subject, prescriptive element, act, condition of application, normative force, context).

**Process**:
1. Filter to norms with valid `raz_norm_articulation`
2. Build embedding-friendly text per norm by concatenating: articulation | subject+prescriptive_element+act [when condition] | [context] | [force]
3. Load Qwen3-Embedding-8B (4096-dim, instruction-aware)
4. Embed all norms with the instruction prefix: `"Instruct: Given a prescriptive social norm from a literary text, represent it for semantic matching with information flows.\nQuery: "`
5. Group by `gutenberg_id` — each book becomes one normative universe

**Output**:
- `norm_universes.json`: Per-book lists of norm dicts with cleaned field names (raz_ prefix stripped)
- `embeddings/`: Per-book `.npy` matrices + `all_embeddings.npy` + `embedding_index.parquet`

### 2.2 Reward Preparation (`reward_prep`)

Pre-computes the R_ground (normative grounding) reward component using a Qwen2.5-72B-Instruct-AWQ judge, so GRPO training doesn't need to run expensive LLM inference at each step.

**Input**: SFT pairs parquet, norm universes JSON, pre-computed embeddings.

**Process**:
1. **Retrieval setup**: Loads per-book Qwen3-Embedding-8B embeddings. For each flow, builds a retrieval query from CI tuple fields (sender, recipient, information_type, context, transmission_principle, subject) + invoked norms.
2. **Per-flow eval rows**: For each SFT training prompt that has extracted flows:
   - Retrieves top-3 most semantically similar norms from the prompt's source book universe
   - Creates an eval row with: chunk_text, flow_json (the single flow), norm_universe_json (the 3 retrieved norms)
3. **Contrastive pairing**: With probability `contrastive_ratio` (default 0.1), duplicate the eval row but retrieve norms from a *different* book's universe. These contrastive pairs should receive lower scores since the wrong norms don't govern the flow.
4. **No-flow prompts**: Prompts with no extracted flows (negative examples) are scored 1.0 directly — no judge call needed (trivially correct, no flows to ground).
5. **Judge inference**: Calls Qwen2.5-72B via `run_vllm_inference` with guided JSON decoding (constrained to `FlowGovernanceJudgment` schema). Temperature 0.0, max 512 tokens.
6. **Judge scoring**: `judge_score = 0.5 * norm_match_score + 0.5 * governance_score`
7. **Aggregation**: Per-flow scores are averaged back to prompt level, grouped by `(prompt_id, source_id, is_contrastive)`.

**Output**: `reward_cache.parquet` with columns: `prompt_id`, `source_id`, `is_contrastive`, `judge_score`, `norm_match_score`, `governance_score`, `judge_response`.

### 2.3 GRPO Training (`grpo_training`)

**Input**: SFT LoRA checkpoint, SFT pairs parquet, reward cache parquet, norm universes JSON.

**Model setup** (LoRA merge + re-wrap):
1. Load base model + SFT LoRA → merge LoRA weights into base via `merge_and_unload()`
2. Save merged model to NFS (`_merged_sft/`) + copy to local scratch (`/tmp/`) for fast vLLM reloads
3. Free memory, reload merged model on CPU
4. Re-wrap with a fresh LoRA adapter (matching SFT's architecture: same rank, alpha, target modules) — this new LoRA is what GRPO actually trains
5. Save tokenizer to the merged directory so vLLM can find it

**Why merge-then-re-wrap?** TRL's `sync_weights` (which syncs training weights to vLLM) doesn't reliably handle LoRA for Qwen3 + vLLM 0.17. By giving vLLM a fully-merged checkpoint, weight sync works on the full model. The fresh LoRA keeps training memory-efficient.

**Dataset building**:
- Pre-applies the chat template to each user prompt with `enable_thinking=True` (or False for ablation), producing raw text strings
- Pre-formatting as raw text means TRL routes through vLLM's `llm.generate()` (not `llm.chat()`), ensuring the exact same tokenization as SFT
- Contrastive pairing: 10% of prompts duplicated with `is_contrastive=True` (same prompt text, different metadata)

**Prompt metadata workaround**: TRL doesn't forward extra dataset columns (source_id, prompt_id) to reward functions. The stage builds a `Dict[prompt_text → metadata]` lookup and attaches it to the reward function, so it can resolve source_id and prompt_id from the raw prompt string.

**vLLM modes**:
- **Colocate** (default, 4-GPU): vLLM runs in-process, shares GPUs with training. Low memory utilization (0.3). Sleep mode offloads vLLM weights during optimizer steps, freeing GPU memory for gradient computation.
- **Server** (8+ GPU): Separate `trl vllm-serve` process on dedicated GPUs. Training on remaining GPUs. Weight sync via NCCL after each step.

**Training hyperparameters** (default):
- G=8 generations per prompt, lr=1e-6, batch_size=2, gradient_accumulation=8
- 1 epoch, warmup ratio 0.1, max completion length 6144
- Gradient checkpointing, bf16

**Qwen3.5 patch**: For Qwen3.5 (natively multimodal), TRL's vLLM initialization is monkey-patched to inject `language_model_only=True`, preventing vLLM from loading vision components.

**Output**: GRPO LoRA checkpoint (adapter files + tokenizer) in `outputs/grpo/checkpoint/`.

---

## 3. Reward Function

The composite reward is a weighted sum of 6 components: **R = Σ(w_i × R_i)**

Default weights: `[0.20, 0.15, 0.15, 0.15, 0.15, 0.20]`

All components operate on the model's completion text after stripping `<think>...</think>` blocks (so reward components don't choke on JSON-like braces in reasoning traces). The completion is parsed as JSON, with automatic normalization from the flat SFT format to a nested canonical format.

### 3.1 R_uncert (Task Clarity) — weight 0.20

**Purpose**: Gates on schema validity and rewards task-appropriate metadata.

Three facets, scored additively:
1. **Schema validity (gating)**: Does the output parse as a valid `CICompletionResult` via Pydantic? → +0.6 if valid, **else return 0.0 immediately** (hard gate)
2. **Flow discrimination**: Is `has_information_exchange` present in reasoning? → +0.2
3. **Confidence calibration**: Does any extraction include `confidence_quant`? → +0.2

**Range**: 0.0 (invalid schema) or 0.6–1.0 (valid schema with varying metadata)

**Key role**: This is the **gating reward** — if the model's output doesn't parse as valid CI JSON, the entire composite reward is near-zero (only r_uncert contributes 0.0, but the other components also fail to parse and return 0.0). This creates strong gradient signal to produce structurally valid output.

### 3.2 R_complete (Structural Completeness) — weight 0.15

**Purpose**: Rewards filling in all CI tuple fields and metadata.

Checks 10 fields per extraction:
- **5-tuple**: sender, recipient, information_type, transmission_principle, subject
- **Metadata**: context, appropriateness, norm_source, confidence_qual, confidence_quant

Score = (number of non-null, non-empty fields) / 10, averaged across all extractions.

**Range**: 0.0–1.0

### 3.3 R_consist (Internal Consistency) — weight 0.15

**Purpose**: Enforces logical invariants between reasoning and extraction sections.

Boolean checks:
1. `has_information_exchange=False` → flows list AND extraction list must both be empty
2. `has_information_exchange=True` → at least one flow or extraction must be present
3. For each extraction where `is_new_flow=True` → `appropriateness` must be "inappropriate" or "ambiguous" (new flows can't be "appropriate" by definition — they lack established norms)

Score = (checks passed) / (total checks). Denominator varies per completion.

**Range**: 0.0–1.0

### 3.4 R_context (Context Identification) — weight 0.15

**Purpose**: Rewards accurate identification of the societal context in which information flows occur.

Compares the model's stated `context` field in each extraction against the source book's aggregated context metadata (built from the norm universe's context fields, semicolon-joined).

**Primary method**: Semantic similarity via sentence-transformers (`all-MiniLM-L6-v2`). Encodes source context and model contexts, computes cosine similarity, averages across extractions.

**Fallback**: Token-overlap Jaccard similarity (if embedding model unavailable).

**Range**: 0.0–1.0

### 3.5 R_cohere (Reasoning-to-Extraction Coherence) — weight 0.15

**Purpose**: Ensures the extracted CI tuples are grounded in the reasoning trace — the model shouldn't hallucinate flow participants or information types not mentioned in its own reasoning.

For each extraction, checks 4 fields (sender, recipient, information_type, subject):
- Tokenizes the field value, filters words >3 chars
- Checks if any word appears in the concatenated reasoning text (all flow entries' `reasoning` + `original_text_snippet` fields)
- Score per extraction = (matched fields) / (checked fields)

Averaged across all extractions.

**Range**: 0.0–1.0

### 3.6 R_ground (Normative Grounding) — weight 0.20

**Purpose**: Assesses whether the model's actual CI extraction is grounded in the source text's normative universe. This is the only component that requires external LLM judgment.

**Implementation (online, default)**: Evaluated live on the model's GRPO completions via two auxiliary vLLM servers:

1. **Parse completion** → extract flows via `_parse_completion()` (shared with other reward components)
2. **Embed flow queries** → send CI tuple fields to the embedding server (Qwen3-Embedding-8B, 1 GPU, port 8001) to encode into the same space as pre-computed norm universe embeddings
3. **Retrieve top-3 norms** → cosine similarity against per-book `.npy` matrices (in-memory `NormRetriever`)
4. **Judge evaluation** → send `(chunk_text, flow_json, retrieved_norms)` to the judge server (Qwen2.5-72B-Instruct-AWQ, TP=2, port 8002)
5. **Score** → `judge_score = 0.5 × norm_match_score + 0.5 × governance_score`, averaged across flows

All embedding and judge calls are batched across completions within a single reward invocation for efficiency. Completions with no extractable flows receive 0.0 (penalizes failure to extract).

**Judge criteria** (from `reward_judge.yaml`):
- **Norm awareness** (`norm_match_score`, 0–1): Do the model's `norms_invoked` semantically match the top-3 retrieved norms from the universe?
- **Flow governance** (`governance_score`, 0–1): Is this flow actually regulated by any retrieved norm, regardless of what the model invoked?

**GPU allocation** (5 GPUs total for online R_ground):
| GPU(s) | Server | Model |
|--------|--------|-------|
| 0 | Embedding server | Qwen3-Embedding-8B (TP=1) |
| 1–2 | Judge server | Qwen2.5-72B-Instruct-AWQ (TP=2) |
| 3–4 | GRPO training + policy vLLM | Policy model (colocate) |

**Contrastive pairs**: Prompts paired with the wrong book's normative universe retrieve norms from that wrong source, which should produce lower R_ground scores since mismatched norms won't govern the flow.

**Cached fallback**: When `online_rground: false`, falls back to pre-computed lookup from `reward_cache.parquet` (keyed by `prompt_id`). This legacy mode evaluates gold SFT completions, not the model's online generations.

**Range**: 0.0–1.0

### Composite Score

```
R = 0.20 × R_uncert + 0.15 × R_complete + 0.15 × R_consist
  + 0.15 × R_context + 0.15 × R_cohere + 0.20 × R_ground
```

The two highest-weighted components (R_uncert and R_ground) represent the two ends of the quality spectrum: R_uncert ensures structural validity (can the output be parsed at all?), while R_ground ensures semantic grounding (is the extraction faithful to the normative universe?). The four middle components (0.15 each) cover completeness, internal logic, context accuracy, and reasoning coherence.

### Ablation Configurations

| Config | Effect |
|---|---|
| `grpo_programmatic_only` | Sets R_ground weight to 0.0 — GRPO with only the 5 programmatic rewards |
| `no_thinking` | `enable_thinking_grpo: false` — suppresses `<think>` blocks during generation |
| `no_negatives` (SFT) | `include_negative_examples: false` — SFT without no-exchange chunks |

### Trace Logging

The `CompositeRewardFunction` logs detailed reward breakdowns to `reward_traces.jsonl` approximately 10 times during training (every ~91 steps for a typical run). Each trace includes per-component raw scores, weighted scores, the composite score, the completion text, and metadata.
