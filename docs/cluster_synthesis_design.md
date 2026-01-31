# Cluster Synthesis Stage Design

## Overview

A new UAIR stage that synthesizes articles within each cluster/topic to identify shared patterns, common AI risk types, and generate structured insights using LLM-powered analysis with guided decoding.

## Architecture

### Input
- **Source**: Output from topic stage (`outputs/topic/docs_topics.parquet`)
- **Schema**: 
  - `article_id`, `article_text`, `chunk_text`
  - `topic_id`, `topic_prob`, `topic_top_terms`
  - `article_keywords`, `plot_x`, `plot_y`

### Output
- **Primary**: `outputs/synthesis/cluster_synthesis.parquet`
  - One row per cluster with synthesized insights
- **Secondary**: `outputs/synthesis/cluster_evidence.parquet`
  - Supporting evidence and article mappings

### Schema Design

```python
# cluster_synthesis.parquet
{
    "cluster_id": int,
    "num_articles": int,
    "primary_risk_type": str,  # From taxonomy
    "risk_confidence": float,
    "risk_evidence": str,  # JSON list of supporting quotes
    "common_themes": str,  # JSON list
    "temporal_patterns": str,  # JSON object
    "geographic_distribution": str,  # JSON object
    "synthesis_summary": str,  # LLM-generated narrative
    "key_entities": str,  # JSON list of companies/orgs
    "technology_types": str,  # JSON list
    "impact_areas": str,  # JSON list
    "cluster_top_terms": str,  # From topic model
}

# cluster_evidence.parquet (for inspection)
{
    "cluster_id": int,
    "article_id": str,
    "article_rank": int,  # By topic_prob
    "excerpt": str,
    "relevance_score": float,
}
```

## Implementation Strategy

### Approach 1: Hierarchical Map-Reduce (Recommended)

**Advantages:**
- Handles arbitrarily large clusters
- Preserves nuance from individual articles
- Scales with Ray Data distributed processing
- Token-efficient via progressive summarization

**Architecture:**

```
Cluster Articles (N documents)
    ↓
┌─────────────── MAP PHASE ────────────────┐
│ Parallel processing per article          │
│ - Extract key claims                     │
│ - Identify risk indicators               │
│ - Generate article summary               │
└─────────────────────────────────────────┘
    ↓
┌────────── HIERARCHICAL REDUCE ──────────┐
│ Progressive aggregation in batches       │
│ Level 1: Group 5-10 summaries           │
│ Level 2: Group L1 outputs               │
│ Level N: Final synthesis                │
└─────────────────────────────────────────┘
    ↓
┌─────────── SYNTHESIS PHASE ─────────────┐
│ Final cluster-level analysis with        │
│ guided decoding for structured output    │
└─────────────────────────────────────────┘
```

**Implementation Details:**

1. **Map Phase** (Per Article):
```python
def map_article_to_summary(article: Dict) -> Dict:
    """Extract structured insights from single article."""
    prompt = f"""Analyze this AI-related news article and extract:
    1. Main AI risk or harm discussed (use taxonomy)
    2. Key entities (companies, organizations)
    3. Technology type
    4. 2-3 most relevant quotes
    
    Article: {article['chunk_text']}
    """
    
    # Use guided decoding with schema
    schema = {
        "risk_type": str,
        "confidence": float,
        "entities": List[str],
        "technology": str,
        "key_quotes": List[str],
    }
    
    return llm_call_with_guided_decoding(prompt, schema)
```

2. **Hierarchical Reduce** (Batch Aggregation):
```python
def reduce_summaries_batch(summaries: List[Dict], level: int) -> Dict:
    """Aggregate batch of article summaries."""
    if len(summaries) <= MAX_CONTEXT_ARTICLES:
        # Can fit in single prompt
        return synthesize_directly(summaries)
    
    # Recursive batching
    batch_size = 10
    batches = chunk_list(summaries, batch_size)
    intermediate = [reduce_summaries_batch(batch, level+1) 
                   for batch in batches]
    return reduce_summaries_batch(intermediate, level+1)

def synthesize_directly(summaries: List[Dict]) -> Dict:
    """Synthesize summaries that fit in context window."""
    prompt = f"""Synthesize these {len(summaries)} article analyses:
    
    {format_summaries(summaries)}
    
    Identify:
    1. Dominant AI risk pattern across articles
    2. Common themes and entities
    3. Geographic/temporal patterns
    4. Overall narrative
    """
    
    schema = {
        "primary_risk": str,
        "confidence": float,
        "themes": List[str],
        "entities": List[str],
        "narrative": str,
    }
    
    return llm_call_with_guided_decoding(prompt, schema)
```

3. **Final Synthesis Phase**:
```python
def final_cluster_synthesis(
    cluster_id: int,
    aggregated_insights: Dict,
    top_articles: List[Dict],  # Highest topic_prob
    cluster_metadata: Dict
) -> Dict:
    """Generate final structured synthesis with guided decoding."""
    
    # Use taxonomy categories for guided decoding
    taxonomy_categories = load_taxonomy_categories()
    
    prompt = f"""Synthesize insights for AI news cluster {cluster_id}:

Cluster Statistics:
- Articles: {cluster_metadata['num_articles']}
- Top terms: {cluster_metadata['top_terms']}
- Date range: {cluster_metadata['date_range']}

Aggregated Insights:
{format_insights(aggregated_insights)}

Representative Articles:
{format_articles(top_articles[:5])}

Task: Generate comprehensive cluster synthesis."""

    # Guided decoding schema
    schema = {
        "type": "object",
        "properties": {
            "primary_risk_type": {
                "type": "string",
                "enum": taxonomy_categories
            },
            "risk_confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "common_themes": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5
            },
            "key_entities": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 10
            },
            "synthesis_summary": {
                "type": "string",
                "maxLength": 1000
            },
            "geographic_focus": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["primary_risk_type", "synthesis_summary"]
    }
    
    return vllm_guided_call(prompt, schema)
```

### Approach 2: LongCite-Inspired Attribution

**Advantages:**
- Explicit citation tracking
- Supports fact-checking
- Better for research/validation

**When to Use:**
- Clusters with high controversy
- Need for audit trails
- Research/academic use cases

**Implementation:**
```python
def synthesize_with_citations(cluster_articles: List[Dict]) -> Dict:
    """Generate synthesis with explicit article citations."""
    
    # Prepare articles with citation IDs
    articles_with_ids = [
        f"[{i}] {art['chunk_text'][:500]}..."
        for i, art in enumerate(cluster_articles)
    ]
    
    prompt = f"""Synthesize the AI risk pattern across these articles.
    
Articles:
{format_with_citations(articles_with_ids)}

Requirements:
1. Identify primary AI risk type
2. Support each claim with citations [N]
3. Note conflicting information
4. Generate concise synthesis

Format: Use [N] after each factual claim."""

    synthesis = llm_call(prompt)
    
    # Parse and structure citations
    return parse_synthesis_with_citations(synthesis, cluster_articles)
```

### Approach 3: Hybrid Agentic Workflow

**For Complex Multi-Step Analysis:**

```python
@ray.remote
class SynthesisAgent:
    """Agentic workflow for complex cluster analysis."""
    
    def __init__(self, llm_config, taxonomy):
        self.llm = initialize_vllm(llm_config)
        self.taxonomy = taxonomy
        
    def analyze_cluster(self, cluster_articles: List[Dict]) -> Dict:
        """Multi-step agentic analysis."""
        
        # Step 1: Risk identification
        risk_analysis = self.identify_primary_risk(cluster_articles)
        
        # Step 2: Entity extraction
        entities = self.extract_entities(cluster_articles)
        
        # Step 3: Temporal analysis
        temporal = self.analyze_temporal_patterns(cluster_articles)
        
        # Step 4: Synthesis with constraints
        synthesis = self.synthesize_with_context(
            risk_analysis, entities, temporal, cluster_articles
        )
        
        return synthesis
```

## Ray Data Integration

### Processing Pipeline

```python
def run_synthesis_stage(df, cfg, logger=None):
    """Cluster synthesis stage implementation."""
    
    # Ensure Ray initialized
    _ensure_ray_init(cfg)
    
    is_ray_ds = hasattr(df, "map_batches")
    
    if not is_ray_ds:
        df_in = pd.read_parquet(df)
        ds = ray.data.from_pandas(df_in)
    else:
        ds = df
    
    # Group by cluster
    def group_by_cluster(batch: pd.DataFrame) -> pd.DataFrame:
        """Group articles by cluster/topic_id."""
        return batch.groupby('topic_id').apply(
            lambda g: pd.Series({
                'cluster_id': g.name,
                'articles': g.to_dict('records'),
                'num_articles': len(g),
                'cluster_top_terms': g['topic_top_terms'].iloc[0],
            })
        ).reset_index(drop=True)
    
    # Map phase: Extract article insights
    ds_mapped = ds.map(extract_article_insights)
    
    # Group by cluster
    ds_grouped = ds_mapped.groupby('topic_id')
    
    # Reduce phase: Hierarchical synthesis per cluster
    ds_synthesized = ds_grouped.map_groups(
        synthesize_cluster_hierarchical,
        batch_format="pandas"
    )
    
    # Final synthesis with guided decoding
    engine_config = vLLMEngineProcessorConfig(
        model_source=cfg.model.model_source,
        engine_kwargs=prepare_engine_kwargs(cfg),
        batch_size=1,  # Process one cluster at a time
    )
    
    processor = build_llm_processor(
        engine_config,
        preprocess=prepare_synthesis_prompt,
        postprocess=parse_synthesis_output
    )
    
    ds_final = processor(ds_synthesized)
    
    if logger:
        # Log results
        df_out = ds_final.to_pandas()
        logger.log_table(
            df_out, 
            "synthesis/clusters",
            panel_group="inspect_results"
        )
    
    return ds_final if is_ray_ds else ds_final.to_pandas()
```

## Token Budget Management

### Strategy
```python
def calculate_token_budget(cfg) -> Dict[str, int]:
    """Calculate token allocations per phase."""
    
    max_model_len = cfg.model.engine_kwargs.max_model_len
    max_output = cfg.sampling_params_synthesis.max_tokens
    system_prompt = len(tokenizer.encode(get_system_prompt(cfg)))
    
    return {
        "map_article_input": 2048,      # Per article in map phase
        "map_article_output": 256,      # Structured extraction
        "reduce_batch_input": 4096,     # Per batch in reduce
        "reduce_batch_output": 512,     # Intermediate summary
        "synthesis_input": max_model_len - max_output - system_prompt - 512,
        "synthesis_output": max_output,  # Final output
    }

def adaptive_batching(articles: List[Dict], token_budget: int) -> List[List[Dict]]:
    """Create batches that fit token budget."""
    
    batches = []
    current_batch = []
    current_tokens = 0
    
    for article in sorted(articles, key=lambda x: x.get('topic_prob', 0), reverse=True):
        article_tokens = estimate_tokens(article['chunk_text'])
        
        if current_tokens + article_tokens > token_budget:
            if current_batch:
                batches.append(current_batch)
            current_batch = [article]
            current_tokens = article_tokens
        else:
            current_batch.append(article)
            current_tokens += article_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    return batches
```

## Configuration

### Config File: `dagspaces/uair/conf/pipeline/full_pipeline_with_synthesis.yaml`

```yaml
defaults:
  - override /hydra/launcher: slurm_monitor

pipeline:
  output_root: ${runtime.output_root}
  sources:
    articles:
      path: ${data.parquet_path}
      type: parquet
  graph:
    nodes:
      classify:
        stage: classify
        depends_on: []
        inputs:
          dataset: articles
        outputs:
          all: outputs/classify/classify_all.parquet
          relevant: outputs/classify/classify_relevant.parquet
        launcher: g2_slurm_pierson
        
      topic:
        stage: topic
        depends_on: [classify]
        inputs:
          dataset: classify.relevant
        outputs:
          docs: outputs/topic/docs_topics.parquet
        launcher: g2_slurm_gpu_1x
        
      synthesis:
        stage: synthesis
        depends_on: [topic]
        inputs:
          dataset: topic.docs
        outputs:
          clusters: outputs/synthesis/cluster_synthesis.parquet
          evidence: outputs/synthesis/cluster_evidence.parquet
        overrides:
          runtime.streaming_io: ${runtime.streaming_io}
          model.engine_kwargs.max_model_len: 8192
          model.engine_kwargs.max_num_seqs: 2
          model.batch_size: 1
          sampling_params_synthesis.max_tokens: 2048
          sampling_params_synthesis.guided_decoding.json: ${synthesis.schema_path}
        launcher: g2_slurm_gpu_4x
        wandb_suffix: synthesis

synthesis:
  # Processing strategy
  strategy: hierarchical_map_reduce  # hierarchical_map_reduce | direct | longcite
  
  # Token budgets
  map_article_tokens: 2048
  reduce_batch_tokens: 4096
  synthesis_tokens: 6144
  
  # Batching
  articles_per_map_batch: 10
  max_reduce_depth: 3
  
  # Filtering
  min_topic_prob: 0.3  # Only use high-confidence cluster members
  max_articles_per_cluster: 100
  top_k_articles: 50  # Use top K by topic_prob for synthesis
  
  # Schema
  schema_path: ${oc.env:SYNTHESIS_SCHEMA,/share/pierson/matt/UAIR/dagspaces/uair/conf/synthesis/schema.json}
  
  # Citation tracking
  enable_citations: true
  citation_threshold: 0.7

sampling_params_synthesis:
  seed: 777
  temperature: 0.1  # More deterministic
  top_p: 0.95
  max_tokens: 2048
  guided_decoding:
    json: null  # Will be set from schema_path
```

### Prompt Templates: `dagspaces/uair/conf/prompt/synthesis.yaml`

```yaml
# Map phase: Extract article insights
map_article_prompt: |
  Analyze this AI news article and extract structured information.
  
  Article ID: {article_id}
  Text: {chunk_text}
  
  Extract:
  1. Primary AI risk/harm type (use taxonomy categories)
  2. Confidence level (0-1)
  3. Key entities mentioned (companies, organizations)
  4. Technology type
  5. Most relevant quote (verbatim)
  
  Be precise and evidence-based.

# Reduce phase: Aggregate insights
reduce_batch_prompt: |
  Synthesize these {num_articles} article analyses into a coherent summary.
  
  Article Insights:
  {formatted_insights}
  
  Generate:
  1. Dominant AI risk pattern
  2. Common themes
  3. Recurring entities
  4. Key findings
  
  Keep factual and concise.

# Final synthesis: Cluster-level
synthesis_prompt: |
  Generate comprehensive synthesis for AI news cluster {cluster_id}.
  
  Cluster Metadata:
  - Total articles: {num_articles}
  - Top terms: {cluster_top_terms}
  - Date range: {date_range}
  - Geographic coverage: {countries}
  
  Aggregated Insights:
  {aggregated_insights}
  
  Top Representative Articles:
  {top_articles}
  
  Generate structured synthesis including:
  1. Primary AI risk type (from taxonomy)
  2. Risk confidence score
  3. Common themes (3-5)
  4. Key entities (companies, orgs)
  5. Technology types involved
  6. Geographic/temporal patterns
  7. Concise narrative summary (2-3 paragraphs)
  
  Use evidence from articles to support claims.

system_prompt: |
  You are an expert AI safety researcher analyzing news articles about AI risks and harms.
  Your task is to identify patterns, synthesize information, and categorize risks according to 
  the provided taxonomy. Be precise, evidence-based, and cite specific articles when possible.
```

### JSON Schema: `dagspaces/uair/conf/synthesis/schema.json`

```json
{
  "type": "object",
  "properties": {
    "primary_risk_type": {
      "type": "string",
      "enum": [
        "Bias and Discrimination",
        "Privacy Violations",
        "Misinformation and Manipulation",
        "Labor Displacement",
        "Surveillance and Control",
        "Safety and Reliability",
        "Dual Use and Weaponization",
        "Environmental Impact",
        "Concentration of Power",
        "Existential Risk",
        "Other"
      ]
    },
    "risk_confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "common_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "maxItems": 5
    },
    "key_entities": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "maxItems": 10
    },
    "technology_types": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "maxItems": 5
    },
    "geographic_distribution": {
      "type": "object",
      "properties": {
        "primary_countries": {
          "type": "array",
          "items": {"type": "string"}
        },
        "global_scope": {"type": "boolean"}
      }
    },
    "temporal_patterns": {
      "type": "object",
      "properties": {
        "trend": {
          "type": "string",
          "enum": ["increasing", "stable", "decreasing", "emerging"]
        },
        "peak_period": {"type": "string"}
      }
    },
    "synthesis_summary": {
      "type": "string",
      "minLength": 100,
      "maxLength": 2000
    },
    "impact_areas": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": [
          "healthcare", "finance", "education", "employment", 
          "justice", "governance", "media", "environment", "security"
        ]
      }
    },
    "supporting_evidence": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "article_id": {"type": "string"},
          "quote": {"type": "string"},
          "relevance": {"type": "number"}
        }
      },
      "maxItems": 5
    }
  },
  "required": [
    "primary_risk_type",
    "risk_confidence",
    "common_themes",
    "synthesis_summary"
  ]
}
```

## Optimizations

### 1. Caching Strategy
```python
# Cache article-level extractions across runs
@ray.remote
class ArticleInsightCache:
    def __init__(self):
        self.cache = {}
    
    def get(self, article_id: str) -> Optional[Dict]:
        return self.cache.get(article_id)
    
    def put(self, article_id: str, insights: Dict):
        self.cache[article_id] = insights
```

### 2. Parallel Cluster Processing
```python
# Process multiple clusters in parallel
def process_clusters_parallel(clusters: List[pd.DataFrame], cfg):
    """Process clusters in parallel with Ray."""
    
    @ray.remote(num_gpus=0.25)  # Share GPUs across clusters
    def process_single_cluster(cluster_df):
        return synthesize_cluster(cluster_df, cfg)
    
    futures = [
        process_single_cluster.remote(cluster)
        for cluster in clusters
    ]
    
    return ray.get(futures)
```

### 3. Incremental Updates
```python
def incremental_synthesis(
    existing_synthesis: pd.DataFrame,
    new_articles: pd.DataFrame,
    cfg
) -> pd.DataFrame:
    """Update synthesis when new articles are added to clusters."""
    
    # Identify affected clusters
    affected_clusters = new_articles['topic_id'].unique()
    
    # Re-synthesize only affected clusters
    updated = []
    for cluster_id in affected_clusters:
        cluster_articles = get_cluster_articles(cluster_id)
        updated.append(synthesize_cluster(cluster_articles, cfg))
    
    # Merge with existing
    return merge_synthesis_results(existing_synthesis, updated)
```

## Evaluation & Validation

### Quality Metrics

1. **Coherence Score**: How well synthesis matches cluster
2. **Citation Accuracy**: Validate extracted quotes
3. **Risk Classification Accuracy**: Compare to ground truth
4. **Coverage**: % of cluster articles referenced
5. **Novelty**: Emergent insights vs article-level

### Validation Pipeline

```python
def validate_synthesis(
    synthesis: pd.DataFrame,
    cluster_articles: pd.DataFrame,
    ground_truth: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """Compute synthesis quality metrics."""
    
    metrics = {}
    
    # 1. Coherence: Embedding similarity
    synthesis_emb = embed_text(synthesis['synthesis_summary'])
    articles_emb = embed_texts(cluster_articles['chunk_text'])
    metrics['coherence'] = cosine_similarity(synthesis_emb, articles_emb.mean(axis=0))
    
    # 2. Citation accuracy
    metrics['citation_accuracy'] = validate_citations(
        synthesis['supporting_evidence'],
        cluster_articles
    )
    
    # 3. Coverage
    referenced_articles = extract_referenced_article_ids(synthesis)
    metrics['coverage'] = len(referenced_articles) / len(cluster_articles)
    
    # 4. Risk classification accuracy (if ground truth available)
    if ground_truth is not None:
        metrics['risk_accuracy'] = compute_classification_accuracy(
            synthesis['primary_risk_type'],
            ground_truth['true_risk_type']
        )
    
    return metrics
```

## Testing Strategy

### Unit Tests
- Token budget calculations
- Batching logic
- Schema validation
- Citation parsing

### Integration Tests
- End-to-end pipeline
- Ray Data integration
- vLLM guided decoding
- Config loading

### Performance Tests
- Scalability: 10, 100, 1000 clusters
- Token efficiency
- GPU utilization
- Memory usage

## Monitoring & Debugging

### W&B Integration

```python
# Log intermediate results
logger.log_table(map_results, "synthesis/map_results")
logger.log_table(reduce_results, "synthesis/reduce_results")
logger.log_table(final_synthesis, "synthesis/final_clusters")

# Log metrics
logger.log_metrics({
    "synthesis/avg_cluster_size": avg_size,
    "synthesis/total_clusters": num_clusters,
    "synthesis/avg_synthesis_tokens": avg_tokens,
    "synthesis/processing_time_s": duration,
})

# Log distributions
logger.log_plot("synthesis/risk_distribution", 
                plt.figure(risk_type_counts))
```

## Future Enhancements

1. **Cross-Cluster Analysis**: Identify patterns across clusters
2. **Temporal Evolution**: Track how cluster themes evolve over time
3. **Multi-Modal Synthesis**: Incorporate images, videos from articles
4. **Interactive Refinement**: Allow user feedback to refine synthesis
5. **Explainability**: Generate attention maps showing which articles influenced synthesis

## References

1. LangChain Map-Reduce: https://python.langchain.com/docs/how_to/summarize_map_reduce
2. vLLM Guided Decoding: https://docs.vllm.ai/en/latest/features/structured_outputs
3. LongCite (attribution): https://arxiv.org/abs/2309.XXXXX
4. Ray Data: https://docs.ray.io/en/latest/data/data.html

