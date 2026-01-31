# Keyword Tracking in UAIR Classification

## Overview

The UAIR classification stage now tracks detailed keyword filtering information for better observability and debugging. This includes:

1. **Binary keyword flag** (`relevant_keyword`): Whether article contains any AI-related keywords
2. **Matched keywords list** (`matched_keywords`): Specific keywords found in the article
3. **Match count** (`keyword_match_count`): Total number of keyword matches (including duplicates)

## Wandb Logging

### Metrics

The following metrics are logged to Weights & Biases for each classify run:

- `classify/keyword_filtering/total_articles`: Total articles processed
- `classify/keyword_filtering/articles_with_keywords`: Count of articles with keywords
- `classify/keyword_filtering/articles_without_keywords`: Count of articles without keywords
- `classify/keyword_filtering/keyword_presence_rate`: Percentage of articles with keywords
- `classify/keyword_filtering/avg_matches_per_article`: Average matches per article (for articles with matches)
- `classify/keyword_filtering/max_matches_in_article`: Maximum matches in any single article

### Summary Fields

Top 10 most frequent keywords are logged as summary fields:

- `classify/top_keywords/01_<keyword>`: Count for most frequent keyword
- `classify/top_keywords/02_<keyword>`: Count for 2nd most frequent keyword
- ... (up to 10)

### Tables

The `classify/results` table in the `inspect_results` panel group now includes:

- `article_id`: Unique article identifier
- `is_relevant`: LLM classification result (True/False)
- `classification_mode`: How the article was classified:
  - `"llm_relevance"`: Processed by LLM
  - `"filtered_by_keyword"`: Filtered out (no keywords found)
  - `"heuristic"`: Simple heuristic classification (when LLM disabled)
- `relevant_keyword`: Binary flag (True/False)
- `matched_keywords`: JSON list of matched keywords (e.g., `["ai", "machine learning", "neural network"]`)
- `keyword_match_count`: Total matches in article
- Plus metadata columns: `article_text`, `article_path`, `country`, `year` (when available)

## Configuration

Keyword tracking is controlled by existing classification configuration:

```yaml
runtime:
  keyword_buffering: true  # Enable keyword extraction (default: true)
  prefilter_mode: pre_gating  # Filter before LLM classification
  # Options: pre_gating (filter before), post_gating (compute flag only), off
```

### Important: Result Merging in `pre_gating` Mode

When `prefilter_mode: pre_gating` is used (the default), the classify stage:

1. **Extracts keywords** from ALL input articles
2. **Materializes** the dataset with keyword flags (using `.materialize()` to cache in Ray object store)
3. **Filters out** articles without keywords before LLM processing (to save costs)
4. **Processes** only keyword-containing articles through the LLM
5. **Merges** filtered articles back into results with:
   - `is_relevant: False`
   - `classification_mode: "filtered_by_keyword"`

This ensures the W&B `classify/results` table and output parquet files contain **ALL** input articles, not just those sent to the LLM. Filtered articles are clearly marked so you can distinguish between "LLM said no" vs "filtered by keyword prefilter".

**Technical Note**: We use `.materialize()` on the unfiltered dataset before applying the filter because Ray Datasets use lazy evaluation. Without materialization, both the "all articles" and "filtered articles" references would point to the same lazy execution plan, resulting in data loss.

## Keyword Regex

The keyword extraction uses the comprehensive regex defined in `_build_relevant_regex()` which includes:

**Note**: The implementation uses `regex.finditer()` with `.group()` instead of `.findall()` to avoid capturing empty strings from internal regex groups (e.g., optional suffixes like `prompt(ing)?`).

- Core AI technologies (ai, ml, nlp, llm, agi, etc.)
- Major AI companies (OpenAI, Anthropic, Google, etc.)
- AI applications (autonomous, chatbot, deepfake, etc.)
- Risk and safety terms (bias, fairness, transparency, etc.)
- Domain applications (healthcare, financial, educational, etc.)

See `dagspaces/uair/stages/classify.py:440-858` for the full keyword list.

## Output Schema

### Classify All Output (`classify_all.parquet`)

All classified articles include:

- `article_id`: Unique article identifier
- `article_text`: Full article text
- `is_relevant`: LLM classification result
- `relevant_keyword`: Keyword presence flag
- `matched_keywords`: JSON-serialized list of matched keywords
- `keyword_match_count`: Total keyword matches
- `classification_mode`: "llm_relevance" or "heuristic"

### Classify Relevant Output (`classify_relevant.parquet`)

Subset of articles where `is_relevant == True`, with same schema as above.

## Performance Impact

Keyword extraction adds minimal overhead:

- **Streaming path**: Single Ray `.map()` operation per row
- **Pandas path**: Single Ray `.map()` operation per row
- **Regex matching**: Compiled regex, O(n) in article length
- **Deduplication**: Python set operation, O(k) in number of matches

Typical overhead: <5ms per article for keyword extraction.

## Use Cases

### 1. Debugging False Negatives

Check if articles were filtered out by keyword prefilter:

```python
import pandas as pd

df = pd.read_parquet("outputs/classify/classify_all.parquet")

# Articles without keywords that were classified as relevant
false_negatives = df[~df["relevant_keyword"] & df["is_relevant"]]
print(f"Found {len(false_negatives)} articles without keywords but relevant by LLM")
```

### 2. Analyzing Keyword Distribution

```python
import json
from collections import Counter

# Load matched keywords
all_keywords = []
for kws_json in df["matched_keywords"].dropna():
    kws = json.loads(kws_json) if isinstance(kws_json, str) else kws_json
    all_keywords.extend(kws)

# Count frequency
keyword_freq = Counter(all_keywords)
top_20 = keyword_freq.most_common(20)
print("Top 20 keywords:", top_20)
```

### 3. Optimizing Keyword List

Identify keywords that don't correlate with relevance:

```python
# Keywords that appear in non-relevant articles
non_relevant = df[~df["is_relevant"]]
non_relevant_keywords = []
for kws_json in non_relevant["matched_keywords"].dropna():
    kws = json.loads(kws_json) if isinstance(kws_json, str) else kws_json
    non_relevant_keywords.extend(kws)

# Find keywords with high false positive rate
fp_keywords = Counter(non_relevant_keywords).most_common(10)
print("Keywords in non-relevant articles:", fp_keywords)
```

## Future Enhancements

Potential improvements to keyword tracking:

1. **Contextual extraction**: Extract sentence/paragraph containing keyword
2. **Position tracking**: Track where in article keywords appear
3. **Keyword co-occurrence**: Track which keywords appear together
4. **Semantic grouping**: Group related keywords (e.g., "AI", "artificial intelligence")
5. **Temporal analysis**: Track keyword trends over time

## References

- Implementation: `dagspaces/uair/stages/classify.py`
- Wandb logging: `dagspaces/uair/orchestrator.py:316-373`
- Schema: `dagspaces/uair/config_schema.py`
