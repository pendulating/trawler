# Synthesis Stage: Dual-Input Architecture

## Overview

The synthesis stage uses a **dual-input architecture** to avoid redundantly storing article text in topic modeling outputs. This keeps intermediate data files smaller and more efficient.

## Architecture

### Input Sources

**Input 1: Cluster Assignments** (`topic.docs`)
- Source: Topic modeling stage output
- Contains: `topic_id`, `topic_prob`, `unit_id`, `topic_top_terms`, metadata
- **Does NOT contain**: `article_text` (saves ~80% storage)

**Input 2: Original Articles** (`classify.relevant`)
- Source: Classification stage output  
- Contains: `article_id`, `article_text`, `is_relevant`, metadata
- Full article content preserved

### Join Strategy

```python
# 1. Load both datasets
df_clusters = load_parquet("topic/docs_topics.parquet")     # Cluster assignments
df_articles = load_parquet("classify/classify_relevant.parquet")  # Article text

# 2. Extract article_id from cluster data (if needed)
if "unit_id" in df_clusters:
    df_clusters["article_id"] = df_clusters["unit_id"].str.split("__").str[0]

# 3. Join on article_id
df_merged = df_clusters.merge(
    df_articles[["article_id", "article_text"]],
    on="article_id",
    how="inner"  # Only keep articles in both datasets
)

# 4. Use merged data for synthesis
# Now have: topic_id, topic_prob, article_text
```

## Pipeline Configuration

```yaml
# dagspaces/uair/conf/pipeline/topic_with_synthesis.yaml

synthesis:
  stage: synthesis
  depends_on:
    - topic
  inputs:
    clusters: topic.docs          # Cluster assignments (no text)
    articles: classify.relevant   # Original articles (with text)
  outputs:
    clusters: outputs/synthesis/cluster_synthesis.parquet
```

## Data Flow Diagram

```
┌─────────────┐
│  Articles   │
│  (raw data) │
└──────┬──────┘
       │
       ↓
┌─────────────┐    outputs:              ┌──────────────┐
│  Classify   │───────────────────────→  │ relevant.pq  │
│   Stage     │    - article_id          │ - article_id │
└─────────────┘    - article_text        │ - article_text│
                   - is_relevant         │ - metadata   │
                                         └──────┬───────┘
       ┌───────────────────────────────────────┘
       │                                        │
       ↓                                        │
┌─────────────┐    outputs:              ┌──────────────┐
│    Topic    │───────────────────────→  │  docs.pq     │
│   Stage     │    - unit_id             │ - topic_id   │
└─────────────┘    - topic_id            │ - topic_prob │
                   - topic_prob           │ - NO TEXT!   │
                   - topic_terms          │ (save space) │
                   - NO article_text      └──────┬───────┘
                                                 │
       ┌─────────────────────────────────────────┤
       │                                         │
       ↓                                         ↓
┌─────────────────────────────────────────────────────┐
│              Synthesis Stage (Dual Input)           │
│                                                     │
│  1. Load clusters: topic_id, topic_prob, unit_id   │
│  2. Load articles: article_id, article_text        │
│  3. Join on article_id                             │
│  4. Synthesize: cluster → insights                 │
└──────────────────────────┬──────────────────────────┘
                           │
                           ↓
                    ┌──────────────┐
                    │ synthesis.pq │
                    │ - cluster_id │
                    │ - summary    │
                    │ - risk_type  │
                    └──────────────┘
```

## Storage Efficiency

### Without Dual Input (Option A)
```
classify.parquet:  1.2 GB  (article_text included)
topic.parquet:     1.0 GB  (article_text duplicated!)
synthesis.parquet: 0.01 GB
─────────────────
Total:             2.21 GB
```

### With Dual Input (Option B - Current)
```
classify.parquet:  1.2 GB  (article_text included)
topic.parquet:     0.2 GB  (NO article_text!)
synthesis.parquet: 0.01 GB
─────────────────
Total:             1.41 GB  (36% savings!)
```

## Implementation Details

### Orchestrator Changes (`orchestrator.py`)

```python
class SynthesisRunner(StageRunner):
    def run(self, context: StageExecutionContext) -> StageResult:
        # Get both input paths
        clusters_path = context.inputs.get("clusters")
        articles_path = context.inputs.get("articles")
        
        # Load both datasets
        df_clusters = pd.read_parquet(clusters_path)
        df_articles = pd.read_parquet(articles_path)
        
        # Pass both to synthesis
        out = run_synthesis_stage(
            df_clusters, 
            cfg, 
            logger=context.logger,
            articles_df=df_articles  # New parameter!
        )
```

### Synthesis Stage Changes (`synthesis.py`)

```python
def run_synthesis_stage(df, cfg, logger=None, articles_df=None):
    """
    Args:
        df: Cluster assignments from topic stage
        articles_df: Original articles with text (NEW!)
    """
    
    # Join clusters with articles
    if articles_df is not None:
        # Extract article_id from unit_id if needed
        if "article_id" not in df.columns:
            df["article_id"] = df["unit_id"].str.split("__").str[0]
        
        # Join to get text
        df = df.merge(
            articles_df[["article_id", "article_text"]],
            on="article_id",
            how="inner"
        )
        
        # Use article_text for synthesis
        df["chunk_text"] = df["article_text"]
    
    # Continue with normal synthesis...
```

## Join Logic Details

### Handling `unit_id` vs `article_id`

Topic stage uses `unit_id` which can be:
- For article-level clustering: `unit_id == article_id`
- For chunk-level clustering: `unit_id == "article_id__chunk_id"`

Extract article_id:
```python
# Pattern: "abc123__0" → "abc123"
df["article_id"] = df["unit_id"].str.split("__").str[0]
```

### Inner Join Strategy

We use `how="inner"` to ensure:
- Only articles that passed classification are synthesized
- Only articles with valid cluster assignments are included
- Automatic filtering of noise/outliers (topic_id == -1)

### Memory Optimization

Only join necessary columns:
```python
# DON'T: df.merge(articles_df, on="article_id")  # Too many columns!
# DO:
articles_subset = articles_df[["article_id", "article_text"]]
df = df.merge(articles_subset, on="article_id")
```

## Benefits

1. ✅ **Storage Efficiency**: 36% smaller intermediate files
2. ✅ **Clean Separation**: Each stage outputs minimal needed data
3. ✅ **Flexibility**: Can synthesize with different article sets
4. ✅ **Scalability**: Less data movement between stages
5. ✅ **Maintainability**: Clear data lineage

## Edge Cases Handled

### 1. Missing article_id in Clusters
```python
if "article_id" not in df_clusters:
    if "unit_id" in df_clusters:
        df_clusters["article_id"] = extract_from_unit_id(df_clusters["unit_id"])
    else:
        raise ValueError("Need article_id or unit_id")
```

### 2. Different Text Column Names
```python
# Handle both article_text and chunk_text
text_col = "article_text" if "article_text" in articles_df else "chunk_text"
df_merged["chunk_text"] = df_merged[text_col]
```

### 3. Mismatched Article Sets
```python
# Inner join ensures only valid matches
# Log if significant mismatch
before = len(df_clusters)
after = len(df_merged)
if after < before * 0.9:
    print(f"Warning: Lost {before - after} rows in join")
```

### 4. No Articles Provided (Fallback)
```python
if articles_df is None:
    print("Warning: Working with cluster metadata only")
    df["chunk_text"] = df["topic_top_terms"]  # Use terms as fallback
```

## Testing

### Unit Test Example
```python
def test_synthesis_dual_input():
    # Mock cluster data
    df_clusters = pd.DataFrame({
        "unit_id": ["art1__0", "art2__0"],
        "topic_id": [0, 1],
        "topic_prob": [0.9, 0.8],
    })
    
    # Mock article data
    df_articles = pd.DataFrame({
        "article_id": ["art1", "art2"],
        "article_text": ["Text 1", "Text 2"],
    })
    
    # Run synthesis
    result = run_synthesis_stage(df_clusters, cfg, articles_df=df_articles)
    
    # Verify join worked
    assert len(result) == 2
    assert "chunk_text" in result.columns
```

### Integration Test
```bash
# Run full pipeline
python -m dagspaces.uair.cli \
    +pipeline=topic_with_synthesis \
    runtime.debug=true \
    runtime.sample_n=100

# Check outputs exist
ls outputs/classify/classify_relevant.parquet  # Articles
ls outputs/topic/docs_topics.parquet           # Clusters  
ls outputs/synthesis/cluster_synthesis.parquet # Synthesis

# Verify storage efficiency
du -h outputs/classify/classify_relevant.parquet
du -h outputs/topic/docs_topics.parquet  # Should be much smaller!
```

## Troubleshooting

### Issue: "ValueError: articles_df must have 'article_id'"
**Cause**: Articles dataset missing article_id column
**Fix**: Check classify stage output has article_id

### Issue: "After join: 0 rows with text"
**Cause**: No matching article_ids between datasets
**Fix**: Check article_id format consistency
```python
# Debug
print(df_clusters["article_id"].head())
print(df_articles["article_id"].head())
```

### Issue: "MemoryError during merge"
**Cause**: Too large article dataset
**Fix**: Only load necessary columns
```python
df_articles = pd.read_parquet(path, columns=["article_id", "article_text"])
```

## Migration from Single Input

If you have existing code expecting single input:

**Old**:
```yaml
synthesis:
  inputs:
    dataset: topic.docs
```

**New**:
```yaml
synthesis:
  inputs:
    clusters: topic.docs
    articles: classify.relevant
```

## Future Enhancements

1. **Streaming Join**: Use Ray Data for large-scale joins
2. **Lazy Loading**: Only load articles for selected clusters
3. **Caching**: Cache joined data for re-runs
4. **Chunk-Level**: Support chunk-level text retrieval

## Summary

The dual-input architecture provides:
- 📦 **Efficient storage** (36% smaller files)
- 🔗 **Clean joins** on article_id
- 🚀 **Scalable design** for large datasets
- ✅ **Full text access** for high-quality synthesis

This design follows the **separation of concerns** principle: each stage outputs only what it computes, not redundant copies of input data.

