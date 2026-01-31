# Cluster Synthesis Stage - Quick Start Guide

## What is the Synthesis Stage?

The synthesis stage **aggregates articles within topic clusters** and generates **structured insights** about shared AI risk patterns using LLM-powered analysis with guided decoding.

**Think of it as**: "Given a cluster of 50 news articles about facial recognition, what AI risk does this cluster represent? What are the common themes? Which companies are involved?"

## Quick Start: 5 Minutes to Running

### 1. Prerequisites

You need:
- Topic modeling results (`outputs/topic/docs_topics.parquet`)
- vLLM-compatible model (e.g., Qwen-30B)
- GPU resources (4x GPUs recommended)

### 2. Run the Pipeline

```bash
# Activate environment
cd /share/pierson/matt/UAIR
source .venv/bin/activate

# Run pipeline: classify → topic → synthesis
python -m dagspaces.uair.cli \
    +pipeline=topic_with_synthesis \
    runtime.debug=true \
    runtime.sample_n=1000
```

This will:
1. Classify 1000 sample articles
2. Cluster them by topic
3. Synthesize insights for each cluster

### 3. Check Results

**W&B Dashboard**:
- Go to your W&B project: https://wandb.ai/your-entity/UAIR
- Look for run: `slurm-XXXXX-UAIR-synthesis-YYYYMMDD-HHMMSS-synthesis`
- Check `inspect_results/synthesis/results` table

**Output Files**:
```bash
# View synthesis results
python -c "
import pandas as pd
df = pd.read_parquet('multirun/.../outputs/synthesis/cluster_synthesis.parquet')
print(df[['cluster_id', 'num_articles', 'primary_risk_type', 'risk_confidence']].head())
print('\nSample synthesis:')
print(df.iloc[0]['synthesis_summary'])
"
```

## Key Configuration Options

### Quick Adjustments

**Speed up for testing**:
```yaml
synthesis:
  max_articles_per_cluster: 10  # Limit cluster size
  top_k_articles: 5              # Use fewer articles
```

**Improve quality**:
```yaml
synthesis:
  min_topic_prob: 0.5            # Higher confidence threshold
  articles_per_map_batch: 5      # Smaller batches = more detailed
```

**More deterministic**:
```yaml
sampling_params_synthesis:
  temperature: 0.0               # Fully deterministic
```

### Common Configurations

**Debug Mode** (fast, small sample):
```bash
python -m dagspaces.uair.cli \
    +pipeline=topic_with_synthesis \
    runtime.debug=true \
    runtime.sample_n=100 \
    synthesis.max_articles_per_cluster=10
```

**Production Mode** (full dataset):
```bash
python -m dagspaces.uair.cli \
    +pipeline=topic_with_synthesis \
    runtime.debug=false \
    runtime.sample_n=null
```

**Custom Taxonomy**:
```bash
export TAXONOMY_JSON=/path/to/custom_taxonomy.yaml
python -m dagspaces.uair.cli +pipeline=topic_with_synthesis
```

## Understanding the Output

### Cluster Synthesis Schema

Each row represents one cluster:

```python
{
    "cluster_id": 5,                          # Topic/cluster ID from topic stage
    "num_articles": 42,                       # Articles in this cluster
    "primary_risk_type": "Privacy Violations", # Main AI risk identified
    "risk_confidence": 0.85,                  # How confident (0-1)
    "synthesis_summary": "This cluster...",   # 2-3 paragraph narrative
    "common_themes": ["facial recognition", "surveillance", ...],
    "key_entities": ["Clearview AI", "Meta", ...],
    "technology_types": ["Computer Vision", "Deep Learning"],
    "impact_areas": ["security", "justice"],
    ...
}
```

### Interpreting Results

**High Confidence** (>0.8):
- Strong pattern across articles
- Clear risk type
- Consistent themes

**Medium Confidence** (0.5-0.8):
- Mixed signals
- Multiple risk types present
- Needs human review

**Low Confidence** (<0.5):
- Diverse cluster
- No clear pattern
- May need re-clustering

## Customization

### Change Synthesis Strategy

Edit `/share/pierson/matt/UAIR/dagspaces/uair/conf/pipeline/topic_with_synthesis.yaml`:

```yaml
synthesis:
  strategy: hierarchical_map_reduce  # Options:
                                      # - hierarchical_map_reduce (default)
                                      # - direct (fast, small clusters)
                                      # - longcite (with citations)
```

### Modify Prompts

Edit `/share/pierson/matt/UAIR/dagspaces/uair/conf/prompt/synthesis.yaml`:

```yaml
synthesis_prompt: |
  Generate synthesis for cluster {cluster_id}.
  
  Focus on:
  1. What specific AI technology is discussed?
  2. What harm or risk is highlighted?
  3. Who are the key actors (companies, regulators)?
  
  Be concise and evidence-based.
```

### Adjust Output Schema

Edit `/share/pierson/matt/UAIR/dagspaces/uair/conf/synthesis/schema.json`:

```json
{
  "properties": {
    "primary_risk_type": {
      "enum": ["Add", "Your", "Custom", "Categories"]
    },
    "custom_field": {
      "type": "string",
      "description": "Your custom field"
    }
  }
}
```

## Troubleshooting

### Issue: "No clusters generated"

**Cause**: All articles filtered out by `min_topic_prob`

**Fix**: Lower threshold
```yaml
synthesis:
  min_topic_prob: 0.1  # More permissive
```

### Issue: "CUDA out of memory"

**Cause**: Cluster too large for GPU

**Fix**: Reduce batch size or max articles
```yaml
synthesis:
  max_articles_per_cluster: 50
  articles_per_map_batch: 5
model:
  engine_kwargs:
    gpu_memory_utilization: 0.7  # More conservative
```

### Issue: "Schema validation error"

**Cause**: JSON schema syntax error

**Fix**: Validate schema
```bash
python -c "
import json
with open('dagspaces/uair/conf/synthesis/schema.json') as f:
    schema = json.load(f)
print('Schema valid!')
"
```

### Issue: "Ray object store full"

**Cause**: Not enough memory

**Fix**: Increase job memory
```yaml
# In launcher config
mem_gb: 128  # Increase from 64
```

## Advanced Usage

### Run Only Synthesis Stage

If you already have topic results:

```bash
python -c "
from dagspaces.uair.stages.synthesis import run_synthesis_stage
from omegaconf import OmegaConf
import pandas as pd

# Load topic results
df = pd.read_parquet('outputs/topic/docs_topics.parquet')

# Load config
cfg = OmegaConf.load('dagspaces/uair/conf/config.yaml')

# Run synthesis
result = run_synthesis_stage(df, cfg)

# Save
result.to_parquet('synthesis_results.parquet', index=False)
"
```

### Parallel Processing

Process multiple clusters in parallel:

```python
import ray
from dagspaces.uair.stages.synthesis import run_synthesis_stage

ray.init()

# Split clusters
clusters = df.groupby('topic_id')

# Process in parallel
@ray.remote
def process_cluster(cluster_df, cfg):
    return run_synthesis_stage(cluster_df, cfg)

futures = [
    process_cluster.remote(cluster, cfg)
    for _, cluster in clusters
]

results = ray.get(futures)
```

### Custom Post-Processing

```python
import pandas as pd

# Load synthesis
df = pd.read_parquet('outputs/synthesis/cluster_synthesis.parquet')

# Filter high-confidence privacy violations
privacy_clusters = df[
    (df['primary_risk_type'] == 'Privacy Violations') &
    (df['risk_confidence'] > 0.8)
]

# Aggregate statistics
print(f"Found {len(privacy_clusters)} high-confidence privacy clusters")
print(f"Total articles: {privacy_clusters['num_articles'].sum()}")
print(f"Top entities: {privacy_clusters['key_entities'].value_counts().head()}")
```

## Performance Tips

### For Speed
- Reduce `max_articles_per_cluster`
- Use `strategy: direct` for small clusters
- Lower `sampling_params_synthesis.max_tokens`
- Increase `model.batch_size` (if memory allows)

### For Quality
- Increase `top_k_articles` (use more articles)
- Lower `temperature` (more deterministic)
- Add `max_reduce_depth` (more thorough aggregation)
- Use higher `min_topic_prob` (only confident members)

### For Scale
- Use `launcher: g2_slurm_gpu_4x` (4 GPUs)
- Enable `runtime.streaming_io: true`
- Increase `runtime.job_memory_gb`
- Use Ray Data for distributed processing

## Next Steps

1. **Explore Results**: Look at W&B tables, filter by risk type
2. **Tune Prompts**: Adjust for your specific use case
3. **Add Metrics**: Compute custom quality scores
4. **Iterate**: Re-run with adjusted configs

## Getting Help

- **Design Doc**: `docs/cluster_synthesis_design.md` (detailed architecture)
- **Summary**: `docs/synthesis_stage_summary.md` (implementation overview)
- **Code**: `dagspaces/uair/stages/synthesis.py` (stage implementation)

## Example Workflow

```bash
# 1. Start with small sample
python -m dagspaces.uair.cli \
    +pipeline=topic_with_synthesis \
    runtime.debug=true \
    runtime.sample_n=100

# 2. Check results in W&B
# Look at synthesis/results table

# 3. Tune configuration
# Edit conf/pipeline/topic_with_synthesis.yaml

# 4. Re-run with larger sample
python -m dagspaces.uair.cli \
    +pipeline=topic_with_synthesis \
    runtime.debug=true \
    runtime.sample_n=1000

# 5. If satisfied, run full pipeline
python -m dagspaces.uair.cli \
    +pipeline=topic_with_synthesis \
    runtime.debug=false
```

That's it! You now have cluster-level insights about AI risks. 🎉

