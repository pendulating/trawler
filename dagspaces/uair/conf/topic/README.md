# Topic Stage Configurations

This directory contains preset configurations for the topic modeling stage, optimized for different use cases.

## Available Configurations

### `event_detection.yaml` — Narrative Event Clustering

**Use Case**: Detect discrete news events from article corpora (e.g., "Senate AI Bill", "ChatGPT Launch", "Local AI Ban")

**Philosophy**:
- Events = 2+ articles covering the same narrative
- Singleton articles (noise) = truly unique stories
- Aggressive epsilon merging for narrative coherence
- Optimized for 100k-1M article datasets

**Key Parameters**:
```yaml
min_cluster_size: 2           # Event = at least 2 articles
min_samples: 1                # Permissive (include peripheral coverage)
cluster_selection_epsilon: 0.03  # Merge sub-events into narratives
n_neighbors: 15               # Local event structure
n_components: 25              # Separate many small events
matryoshka_dim: 384           # Rich semantics for event distinction
```

**Expected Results** (386k articles):
- **15k-40k event clusters** (mostly small: 2-10 articles)
- **60-75% clustered** (articles in events)
- **25-40% noise** (unique stories)

**Usage**:
```bash
# Method 1: Use the dedicated pipeline (recommended)
python -m dagspaces.uair.run_pipeline \
  pipeline=event_detection_from_articles \
  ARTICLES_PATH=/path/to/articles.parquet

# Method 2: Copy parameters into your pipeline's node.overrides section
# See dagspaces/uair/conf/pipeline/event_detection_from_articles.yaml for example
```

**Note**: Due to Hydra's config composition rules, you cannot use `override /topic: event_detection` in pipeline configs. Instead, either use the dedicated pipeline or copy the parameters into your pipeline's `overrides:` section.

---

### Default Configuration — Thematic Topic Modeling

**Use Case**: Discover broad thematic topics (e.g., "Healthcare AI", "Autonomous Vehicles", "AI Ethics")

**Philosophy**:
- Topics = large semantic clusters (50+ articles)
- Focus on thematic coherence over temporal events
- Fewer, more stable clusters

**Key Parameters**:
```yaml
min_cluster_size: 10-50       # Larger clusters for themes
cluster_selection_epsilon: 0.0  # No automatic merging
n_neighbors: 50               # Global structure
n_components: 15              # Fewer dimensions for broad topics
```

**Expected Results** (386k articles):
- **500-2,000 topic clusters**
- **50-70% clustered**
- **30-50% noise**

---

## When to Use Which Config?

| Goal | Config | Cluster Count | Cluster Size | Use Case |
|------|--------|---------------|--------------|----------|
| **Detect discrete news events** | `event_detection` | 15k-40k | 2-10 articles | Event timelines, narrative tracking |
| **Find broad themes** | default | 500-2k | 50+ articles | Thematic analysis, topic trends |
| **Hierarchical exploration** | custom | varies | varies | Start broad, drill down |

---

## Tuning Guide

### For Event Detection (`event_detection.yaml`)

**Problem: Too many tiny clusters**
```yaml
# Solution 1: More aggressive merging
topic.hdbscan.cluster_selection_epsilon: 0.05  # was 0.03

# Solution 2: Slightly larger events
topic.hdbscan.min_cluster_size: 3  # was 2
```

**Problem: Events too merged (lost granularity)**
```yaml
# Solution 1: Less merging
topic.hdbscan.cluster_selection_epsilon: 0.01  # was 0.03

# Solution 2: More dimensions
topic.reduce.n_components: 30  # was 25
```

**Problem: Too much noise (>40%)**
```yaml
# Solution 1: More permissive clustering
topic.hdbscan.min_samples: 1  # Already at minimum

# Solution 2: More aggressive merging (pulls in singletons)
topic.hdbscan.cluster_selection_epsilon: 0.05  # was 0.03
```

**Problem: Events not separating well**
```yaml
# Solution: Increase representation capacity
topic.reduce.n_components: 30  # was 25
topic.embed.matryoshka_dim: 512  # was 384
topic.reduce.n_neighbors: 10  # was 15 (more local focus)
```

---

## Creating Custom Configs

1. **Copy a base config**:
   ```bash
   cp dagspaces/uair/conf/topic/event_detection.yaml \
      dagspaces/uair/conf/topic/my_custom.yaml
   ```

2. **Edit parameters** for your use case

3. **Use in pipeline**:
   ```yaml
   defaults:
     - override /topic: my_custom
   ```

---

## Advanced: Parameter Deep Dive

### HDBSCAN Parameters

**`min_cluster_size`**
- **Meaning**: Minimum articles in a cluster
- **Range**: 2-100+
- **Event detection**: 2-3 (capture small events)
- **Topic modeling**: 10-50 (stable themes)

**`min_samples`**
- **Meaning**: Core point density threshold
- **Range**: 1 to min_cluster_size
- **Lower values**: More permissive, fewer noise points
- **Higher values**: More conservative, more noise points
- **Recommended**: Set lower than `min_cluster_size` (e.g., 1 when min_cluster_size=2)

**`cluster_selection_epsilon`**
- **Meaning**: Distance threshold for merging nested clusters
- **Range**: 0.0-0.10
- **0.0**: No merging (separate all sub-events)
- **0.01-0.03**: Moderate merging (related sub-events)
- **0.05-0.08**: Aggressive merging (full narratives)
- **0.10+**: Very aggressive (thematic grouping)

**Key insight**: For news events, `cluster_selection_epsilon` is the **primary control** for event granularity!

### UMAP Parameters

**`n_neighbors`**
- **Meaning**: Local neighborhood size for manifold learning
- **Range**: 2-200
- **Low (10-20)**: Emphasizes local structure (events)
- **High (50-100)**: Emphasizes global structure (topics)

**`min_dist`**
- **Meaning**: Minimum spacing between points in embedding
- **Range**: 0.0-1.0
- **0.0**: Tightest packing (best for clustering)
- **0.1-0.3**: Moderate spacing (visualization)
- **Recommendation**: Always use 0.0 when clustering (not just visualizing)

**`n_components`**
- **Meaning**: Dimensionality of reduced space
- **Range**: 2-50
- **Low (5-15)**: Fewer clusters, faster
- **Medium (15-25)**: Balanced
- **High (25-50)**: More separation, handles complexity

### Embedding Parameters

**`matryoshka_dim`**
- **Meaning**: Embedding dimension (Nomic supports 64-768)
- **Range**: 64-768
- **256**: Fast, good for most tasks
- **384**: Better semantics, recommended for events
- **512-768**: Maximum quality, slower

**`max_tokens_for_embed`**
- **Meaning**: Max tokens per article to embed
- **Range**: 512-8192
- **4096**: First ~2 pages (most articles)
- **8192**: Full article context (events need this!)

---

## Monitoring & Evaluation

### W&B Metrics to Track

```python
# Key metrics for event detection quality:
summary = {
    "topic/clusters/total": num_clusters,
    "topic/clusters/size_p50": median_cluster_size,
    "topic/clusters/size_p95": p95_cluster_size,
    "topic/noise_ratio": pct_noise,
    "topic/clustered_ratio": pct_clustered,
    
    # Size distribution
    "topic/clusters/size_2": count_size_2,
    "topic/clusters/size_3_5": count_size_3_5,
    "topic/clusters/size_6_10": count_size_6_10,
    "topic/clusters/size_11_20": count_size_11_20,
    "topic/clusters/size_20plus": count_size_20plus,
}
```

### Qualitative Evaluation

```python
# Sample clusters to review
import pandas as pd

clusters = pd.read_parquet("outputs/topic/event_clusters.parquet")

# Small events (2-5 articles)
small = clusters[clusters.groupby('topic_id').transform('size').between(2, 5)]

# Check: Are these coherent events?
for topic_id in small['topic_id'].unique()[:10]:
    event_articles = small[small['topic_id'] == topic_id]
    print(f"\n=== Event {topic_id} ({len(event_articles)} articles) ===")
    print(event_articles['article_path'].tolist())
    print(event_articles['topic_top_terms'].iloc[0])
```

---

## Performance

### Expected Runtimes (2x RTX A6000)

| Dataset Size | Embedding | UMAP+HDBSCAN | Total | Memory |
|--------------|-----------|--------------|-------|--------|
| 10k articles | ~30s | ~10s | ~1 min | 8 GB |
| 100k articles | ~5 min | ~1 min | ~6 min | 16 GB |
| 386k articles | ~20 min | ~5 min | ~25 min | 32 GB |
| 1M articles | ~50 min | ~15 min | ~65 min | 64 GB |

**Bottlenecks**:
1. **Embedding**: GPU batch size (increase for speed, decrease for memory)
2. **UMAP**: `n_components` and `n_neighbors` (higher = slower)
3. **HDBSCAN**: Dataset size (O(n log n) with RAPIDS)

---

## Examples

### Example 1: Quick Event Detection on Sample

```bash
python -m dagspaces.uair.run_pipeline \
  pipeline=event_detection_from_articles \
  ARTICLES_PATH=/path/to/articles.parquet \
  runtime.sample_n=10000  # Test on 10k sample first
```

### Example 2: Full Event Detection + Synthesis

```bash
python -m dagspaces.uair.run_pipeline \
  pipeline=event_synthesis_from_classify \
  ARTICLES_PATH=/path/to/classify_relevant.parquet
```

### Example 3: Custom Epsilon Tuning

```bash
python -m dagspaces.uair.run_pipeline \
  pipeline=event_detection_from_articles \
  topic.hdbscan.cluster_selection_epsilon=0.05  # More aggressive merging
```

### Example 4: Local Testing (No SLURM)

```yaml
# Create: conf/pipeline/event_detection_local.yaml
defaults:
  - event_detection_from_articles
  - override /hydra/launcher: local  # No SLURM

pipeline:
  graph:
    nodes:
      topic:
        launcher: local  # Run locally
```

```bash
python -m dagspaces.uair.run_pipeline \
  pipeline=event_detection_local \
  runtime.sample_n=1000  # Small sample
```

---

## Troubleshooting

### Issue: GPU OOM during embedding
```yaml
# Solution: Reduce batch size
topic.embed.batch_size: 64  # was 128
```

### Issue: RAPIDS not available
```yaml
# Solution: Fall back to CPU (slower)
topic.gpu.use_rapids: false
```

### Issue: All articles marked as noise
```yaml
# Diagnosis: Clusters too strict
# Solution 1: More permissive
topic.hdbscan.min_samples: 1  # was 5

# Solution 2: More merging
topic.hdbscan.cluster_selection_epsilon: 0.05  # was 0.03
```

### Issue: Only 1-2 giant clusters
```yaml
# Diagnosis: Too much merging or insufficient dimensions
# Solution 1: Less merging
topic.hdbscan.cluster_selection_epsilon: 0.01  # was 0.03

# Solution 2: More dimensions
topic.reduce.n_components: 30  # was 25
```

---

## References

- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Nomic Embeddings](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/)

---

## Contributing

To add a new preset configuration:

1. Create `dagspaces/uair/conf/topic/your_config.yaml`
2. Document use case and parameters in this README
3. Add example pipeline in `conf/pipeline/`
4. Test on sample data
5. Submit PR with results
