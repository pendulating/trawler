# CI Information Flow Pipeline: Design Audit

Technical documentation of the `ci_reasoning` and `ci_extraction` stages in the `historical_norms` dagspace, with a focus on identifying misaligned design decisions.

---

## 1. Pipeline Overview

The CI flow extraction pipeline is a two-stage DAG:

```
fetch_gutenberg → ci_reasoning → ci_extraction
```

- **ci_reasoning**: Given a text chunk, identify information flows through a Contextual Integrity lens. Produces structured reasoning traces with per-flow metadata.
- **ci_extraction**: Given a reasoning trace + source text, extract formal 5-component CI tuples (Subject, Sender, Recipient, Information Type, Transmission Principle) with contextual metadata.

Both stages use vLLM with guided (constrained) JSON decoding against Pydantic schemas.

---

## 2. Stage 1: CI Reasoning

### 2.1 Entry Point

**Runner**: `dagspaces/historical_norms/runners/ci_reasoning.py` → `CIReasoningRunner`
**Stage function**: `dagspaces/historical_norms/stages/ci_reasoning.py` → `run_ci_reasoning_stage(df, cfg)`

The runner loads the input parquet via `prepare_stage_input()`, calls the stage function, and saves the output parquet.

### 2.2 Prompt Resolution

The stage resolves prompts in this priority order:
1. `cfg.prompt_ci_reasoning` (set by pipeline YAML override, e.g. `prompt_ci_reasoning: ${prompt_ci_reasoning_fiction}`)
2. `cfg.prompt` (generic fallback)

Each prompt config has two fields: `system_prompt` (str) and `prompt_template` (str with `{{article_text}}` placeholder).

### 2.3 Input Schema

The stage expects a DataFrame with an `article_text` column (the text chunk to analyze).

### 2.4 Prompt Construction

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": prompt_template.replace("{{article_text}}", article_text)},
]
```

### 2.5 Guided Decoding Schema: `CIReasoningList`

```python
class CIReasoningList(BaseModel):
    reasoning: str          # Overall passage assessment (chain-of-thought, generated FIRST)
    flows: List[CIReasoningEntry]  # Up to 10 flows (max_length=10)
    has_information_exchange: bool  # Whether any flows exist
```

```python
class CIReasoningEntry(BaseModel):
    original_text_snippet: str                                        # Exact quote
    reasoning: str                                                     # Per-flow explanation
    context_identified: str                                            # Societal sphere
    flow_direction: str                                                # "who tells what to whom"
    potential_appropriateness: Literal["appropriate", "inappropriate", "ambiguous"]
    is_new_flow: bool = False                                          # Novel flow with no norms
```

### 2.6 Sampling Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `temperature` | 0.0 | Deterministic |
| `max_tokens` | 4096 | Overridden to 4096 in COLM pipeline |
| `seed` | 777 | From base config |
| `guided_decoding` | `{"json": CIReasoningList.model_json_schema()}` | Constrained output |

### 2.7 Postprocessing

1. Extract JSON from generated text (find first `{` to last `}`)
2. Parse with `json.loads`; fallback to `json_repair` if available
3. Extract fields:
   - `ci_reasoning_json`: Full JSON string of the parsed object
   - `has_information_exchange`: From `obj["has_information_exchange"]`, **but falls back to `len(flows) > 0`** if the field is missing
   - `ci_flow_count`: `len(obj["flows"])`
   - `ci_reasoning_text`: `obj["reasoning"]` (top-level reasoning)
   - `ci_reasoning_parse_error`: Error string if JSON parsing failed
4. Clean for Parquet serialization

### 2.8 Output Schema (DataFrame columns added)

| Column | Type | Source |
|--------|------|--------|
| `ci_reasoning_json` | str (JSON) | Full parsed reasoning object |
| `has_information_exchange` | bool | Whether flows were found |
| `ci_flow_count` | int | Number of flows identified |
| `ci_reasoning_text` | str | Top-level reasoning text |
| `ci_reasoning_parse_error` | str or None | Parse error if any |
| *(all original columns)* | — | Passed through |

---

## 3. Stage 2: CI Extraction

### 3.1 Entry Point

**Runner**: `dagspaces/historical_norms/runners/ci_extraction.py` → `CIExtractionRunner`
**Stage function**: `dagspaces/historical_norms/stages/ci_extraction.py` → `run_ci_extraction_stage(df, cfg)`

### 3.2 Input Filtering

The stage applies three sequential filters to the reasoning output:

1. `ci_reasoning_json` is not null/empty
2. `has_information_exchange == True`
3. `ci_flow_count > 0`

Rows that fail any filter are silently dropped.

### 3.3 Row Expansion (1:N)

**Critical design**: Each input row (one text chunk) is expanded into N rows, one per flow entry in the reasoning JSON's `flows` array.

For each flow entry `i` in `reasoning_obj["flows"]`:

| New Column | Source |
|------------|--------|
| `ci_flow_index` | `i` (0-based index) |
| `ci_reasoning_trace` | `flow_entry["reasoning"]` |
| `ci_flow_snippet` | `flow_entry["original_text_snippet"]` |
| `ci_flow_context` | `flow_entry["context_identified"]` |
| `ci_flow_direction` | `flow_entry["flow_direction"]` |
| `ci_is_new_flow_reasoning` | `flow_entry["is_new_flow"]` |

All original columns (including `article_text`, `ci_reasoning_json`, etc.) are carried forward into each expanded row.

### 3.4 Prompt Resolution

Same pattern as reasoning: `cfg.prompt_ci_extraction` → `cfg.prompt` fallback.

### 3.5 Prompt Construction

```python
def _format_prompt(row):
    text = row["ci_flow_snippet"] or row["article_text"]  # snippet preferred, chunk as fallback
    reasoning = row["ci_reasoning_trace"]
    return prompt_template
        .replace("{{article_text}}", text)
        .replace("{{reasoning_trace}}", reasoning)
```

The prompt template (fiction variant) is:
```
Source Text:
{{article_text}}

Reasoning Trace:
{{reasoning_trace}}

Extract the contextual integrity information flow tuple(s) in JSON format:
```

### 3.6 Guided Decoding Schema: `CIExtractionResult`

```python
class CIExtractionResult(BaseModel):
    information_flows: List[ContextualIntegrityFlow]
```

```python
class ContextualIntegrityFlow(BaseModel):
    flow: InformationFlowTuple
    context: str
    appropriateness: Literal["appropriate", "inappropriate", "ambiguous"]
    norms_invoked: List[str] = []
    norm_source: Literal["explicit", "implicit", "both"]
    is_new_flow: bool = False
    confidence_qual: Literal["very_uncertain", "uncertain", "somewhat_certain", "certain", "very_certain"]
    confidence_quant: int  # 0-10
```

```python
class InformationFlowTuple(BaseModel):
    subject: Optional[str] = None
    sender: str
    recipient: str
    information_type: str
    transmission_principle: str
```

### 3.7 Sampling Parameters

| Parameter | Default | Pipeline Override |
|-----------|---------|-------------------|
| `temperature` | 0.0 | — |
| `max_tokens` | 2048 | 8192 in COLM pipeline |
| `guided_decoding` | `{"json": CIExtractionResult.model_json_schema()}` | — |

### 3.8 Postprocessing

1. Parse JSON from generated text (same `{}`-bracket extraction + `json_repair` fallback)
2. Extract `information_flows` list from parsed object
3. **Only the FIRST flow is extracted into flat columns** (see issue below)
4. For the first flow, flatten:

| Column | Source |
|--------|--------|
| `ci_subject` | `flows[0].flow.subject` |
| `ci_sender` | `flows[0].flow.sender` |
| `ci_recipient` | `flows[0].flow.recipient` |
| `ci_information_type` | `flows[0].flow.information_type` |
| `ci_transmission_principle` | `flows[0].flow.transmission_principle` |
| `ci_context` | `flows[0].context` |
| `ci_appropriateness` | `flows[0].appropriateness` |
| `ci_norms_invoked` | `flows[0].norms_invoked` (list) |
| `ci_norm_source` | `flows[0].norm_source` |
| `ci_is_new_flow` | `flows[0].is_new_flow` |
| `ci_confidence_qual` | `flows[0].confidence_qual` |
| `ci_confidence_quant` | `flows[0].confidence_quant` |
| `ci_flows_raw` | Full `information_flows` list (all flows, as nested dicts) |
| `ci_flow_count` | `len(information_flows)` — **overwrites** the reasoning stage's `ci_flow_count` |

5. List values are coerced to `"; "`-joined strings via `_to_str()` for Parquet safety.

---

## 4. Joint Pipeline Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  fetch_gutenberg                                                  │
│  Output: chunks.parquet (article_text, metadata per chunk)        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  ci_reasoning                                                     │
│  Input:  1 row per text chunk                                     │
│  LLM:    Identify flows → CIReasoningList (up to 10 flows/chunk) │
│  Output: 1 row per chunk with ci_reasoning_json, flow count, etc. │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                     filter: has_information_exchange == True
                     expand: 1 row per flow (ci_flow_index)
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  ci_extraction                                                    │
│  Input:  1 row per individual flow (snippet + reasoning trace)    │
│  LLM:    Extract CI tuple → CIExtractionResult (List of flows)   │
│  Output: 1 row per flow with flattened CI tuple columns           │
│          (only first element of information_flows list used)       │
└──────────────────────────────────────────────────────────────────┘
```

### Cardinality

- Fetch produces ~N chunks per novel (depends on chunking params)
- Reasoning produces 1 row per chunk (same N), with 0-10 flows embedded in JSON
- Extraction expands to M rows where M = sum of all `ci_flow_count` across chunks
- Each expanded row gets one LLM call for structured extraction

---

## 5. Identified Misalignments and Design Issues

### ISSUE 1: Extraction schema allows List but code only uses first element (CRITICAL)

**Location**: `ci_extraction.py:208-235`

The `CIExtractionResult` schema wraps `information_flows: List[ContextualIntegrityFlow]` — the LLM is free to return multiple flows per call. But the postprocessing **only extracts `ci_flows[0]`** into flat columns:

```python
if ci_flows:
    first = ci_flows[0]
    # ... extracts only from first
```

Meanwhile, each extraction call already corresponds to a **single flow** from the reasoning stage (rows were expanded 1:1 per flow). So the schema says "give me a list of flows" but the input is a single flow's snippet + trace, and the code discards everything beyond index 0.

**Risk**: The LLM may split a single reasoning flow into sub-flows during extraction, or hallucinate additional flows. These are silently dropped. The `ci_flows_raw` column preserves the full list but the flat `ci_*` columns only reflect the first.

**Recommendation**: Either (a) constrain the schema to a single flow (not a list), or (b) handle multiple extraction flows by further expanding rows.

### ISSUE 2: `ci_flow_count` is overwritten with different semantics (MODERATE)

**Location**: `ci_extraction.py:210`

The reasoning stage sets `ci_flow_count` = number of flows identified in the chunk. After expansion, each row carries this count. Then the extraction postprocessor overwrites it:

```python
result_row["ci_flow_count"] = len(ci_flows)  # now = number of extraction flows (usually 1)
```

This means `ci_flow_count` changes meaning mid-pipeline: from "flows found by reasoning in the chunk" to "flows returned by extraction for this specific input." Downstream consumers of `ci_flow_count` cannot distinguish between these two semantics.

**Recommendation**: Use a distinct name, e.g. `ci_extraction_flow_count`, to avoid shadowing.

### ISSUE 3: Extraction input text uses snippet, but prompt says "Source Text" implying full chunk (MINOR)

**Location**: `ci_extraction.py:146-151`, prompt template

The `_format_prompt` function substitutes `{{article_text}}` with the per-flow snippet (`ci_flow_snippet`) when available, falling back to `article_text` (the full chunk). But the prompt template labels this "Source Text:" and the system prompt says "analyze the provided reasoning trace and source text."

The model receives a short snippet (a single quote) labeled as "Source Text" alongside a reasoning trace that was generated from the full chunk. If the reasoning trace references context beyond the snippet, the extraction model has no access to it.

**Risk**: The model may hallucinate context details or produce less accurate tuples because the snippet alone may lack sufficient context for full CI tuple extraction. The reasoning trace references the broader chunk context but the extraction model can't verify those claims.

**Recommendation**: Consider passing the full `article_text` alongside the snippet, or passing both explicitly (snippet + full context).

### ISSUE 4: `has_information_exchange` fallback logic masks schema violations (MINOR)

**Location**: `ci_reasoning.py:143`

```python
result_row["has_information_exchange"] = bool(obj.get("has_information_exchange", len(flows) > 0))
```

If the LLM omits `has_information_exchange` from the JSON (which shouldn't happen under guided decoding but could with parse repair), the code infers it from `len(flows) > 0`. This masks a constraint that the schema imposes: `has_information_exchange` is a required field in `CIReasoningList`. If it's missing, something went wrong with guided decoding, and silently inferring it hides the failure.

### ISSUE 5: Extraction prompt asks for "information flow tuple(s)" (plural) despite single-flow input (MINOR)

**Location**: Prompt template in both `ci_extraction_fiction.yaml` and `ci_extraction_prescriptive.yaml`

```
Extract the contextual integrity information flow tuple(s) in JSON format:
```

The input is a single flow's snippet + reasoning trace (post-expansion), but the prompt uses plural language. Combined with the `List[ContextualIntegrityFlow]` schema, this actively encourages the model to produce multiple flows — which are then discarded (see Issue 1).

### ISSUE 6: Redundant `_clean_for_parquet()` implementations (LOW)

Both `ci_reasoning.py` and `ci_extraction.py` define their own `_clean_for_parquet()` function with slightly different column lists. The common orchestrator also has `_clean_df_for_parquet()`. This creates maintenance risk if column-cleaning logic needs to change.

### ISSUE 7: `_extract_json()` is duplicated with subtle differences (LOW)

Both stages define local `_extract_json()` functions. The extraction stage's version uses `repair_json(json_text, return_objects=True)` (returns dict directly), while the reasoning stage's version uses `repair_json(json_text)` then `json.loads()`. Under guided decoding, JSON should always be valid, so these fallbacks rarely fire — but the inconsistency could cause different failure modes.

### ISSUE 8: No validation that reasoning `flows` entries contain required fields (LOW)

**Location**: `ci_extraction.py:108-116`

During row expansion, the code does `.get()` with empty-string defaults for `reasoning`, `original_text_snippet`, `context_identified`, and `flow_direction`. If guided decoding produced a malformed flow entry (e.g., missing `original_text_snippet`), the extraction stage would silently proceed with an empty snippet, leading to a vacuous extraction prompt.

### ISSUE 9: Fiction vs. prescriptive prompt default mismatch (CONFIGURATION)

**Location**: `config.yaml:17-18`

The base config defaults `prompt_ci_reasoning` and `prompt_ci_extraction` to **prescriptive** variants. The fiction pipeline YAMLs (`COLM_flows_fiction.yaml`) override these to fiction variants. If a user runs the CI extraction pipeline without the fiction pipeline override (e.g., `pipeline=ci_extraction` which uses `ci_extraction.yaml`), the default prescriptive prompts apply even though `ci_extraction.yaml` fetches from Gutenberg (fiction).

Checking `ci_extraction.yaml`: it does **not** override prompt keys. This means running `pipeline=ci_extraction` will use prescriptive prompts on fiction text from Pride and Prejudice.

### ISSUE 10: Appropriateness field naming inconsistency between stages (LOW)

- Reasoning schema: `potential_appropriateness` (in `CIReasoningEntry`)
- Extraction schema: `appropriateness` (in `ContextualIntegrityFlow`)

The name change between stages is intentional (reasoning = preliminary judgment, extraction = final), but may confuse downstream analysis that tries to compare or correlate the two assessments.

---

## 6. Schema Cross-Reference

### Fields that flow from reasoning → extraction (via row expansion)

| Reasoning Output | Extraction Input Column | Used In Extraction Prompt |
|-----------------|------------------------|--------------------------|
| `flows[i].original_text_snippet` | `ci_flow_snippet` | Yes (as `{{article_text}}`) |
| `flows[i].reasoning` | `ci_reasoning_trace` | Yes (as `{{reasoning_trace}}`) |
| `flows[i].context_identified` | `ci_flow_context` | No (not in template) |
| `flows[i].flow_direction` | `ci_flow_direction` | No (not in template) |
| `flows[i].is_new_flow` | `ci_is_new_flow_reasoning` | No (not in template) |
| `flows[i].potential_appropriateness` | *(not extracted as column)* | No |

Note: `context_identified`, `flow_direction`, `potential_appropriateness`, and `is_new_flow` from the reasoning stage are **not passed to the extraction prompt**. The extraction model re-derives `context`, `appropriateness`, and `is_new_flow` independently. This is by design (the extraction model should extract from evidence, not copy the reasoning model's preliminary labels), but it means:
- The two stages can produce contradictory judgments on the same flow
- The reasoning stage's `context_identified` is preserved as a column but never used

---

## 7. GPU and Parallelism Configuration (COLM Pipeline)

| Stage | GPUs | Tensor Parallel | Data Parallel | max_tokens |
|-------|------|-----------------|---------------|------------|
| ci_reasoning | 4 | 2 | 2 | 4096 |
| ci_extraction | 2 | *(default from model)* | *(none)* | 8192 |

The reasoning stage gets 4x GPUs with 2-way data parallelism (2 independent vLLM workers, each with 2 GPUs for tensor parallelism). The extraction stage gets 2x GPUs with no data parallelism. The extraction stage has higher max_tokens (8192 vs 4096) despite the extraction output being structurally simpler than reasoning — this may be over-provisioned.

---

## 8. Summary of Recommendations

| Priority | Issue | Recommendation |
|----------|-------|----------------|
| CRITICAL | #1: Only first extraction flow used | Constrain schema to single flow, or expand rows for multi-flow results |
| MODERATE | #2: `ci_flow_count` overwritten | Rename to `ci_extraction_flow_count` in extraction stage |
| MODERATE | #9: Default prompts are prescriptive for fiction pipeline | Add prompt overrides to `ci_extraction.yaml` |
| MINOR | #3: Snippet vs full text in extraction | Pass both snippet and full chunk to extraction prompt |
| MINOR | #5: Plural prompt language for single-flow input | Change to singular "Extract the information flow tuple" |
| LOW | #6, #7: Duplicated utilities | Consolidate into common module |
| LOW | #4, #8, #10: Silent fallbacks, no validation, naming | Address as part of broader cleanup |
