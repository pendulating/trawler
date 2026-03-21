# TF-IDF Stopwords Configuration

Custom stopwords lists for filtering non-topical terms during topic modeling TF-IDF feature extraction.

## Available Stopwords Files

### `minimal.yaml`
Just the most essential news boilerplate terms (~11 words).

**Use when**: You want light filtering with maximum topic vocabulary retention.

```yaml
stopwords:
  - newsletter
  - subscribe
  - article
  - story
  - breaking
  - update
  - click
  - read
  - more
  - share
  - comment
```

### `news_extended.yaml`
News-specific terms plus temporal/navigation terms (~90 words).

**Use when**: You want to filter news boilerplate while keeping standard English words.

Categories:
- News industry boilerplate (newsletter, subscribe, breaking, etc.)
- Temporal/metadata (monday-sunday, january-december, etc.)
- Navigation/website (click, read, share, comment, etc.)

### `comprehensive.yaml`
Combines news terms + standard English stopwords (~300+ words).

**Use when**: You want aggressive filtering similar to scikit-learn's "english" + news terms.

Categories:
- All terms from `news_extended.yaml`
- Standard English stopwords (articles, prepositions, pronouns)
- Common verbs (said, make, get, go, etc.)
- Quantifiers and adverbs

## Usage

### Method 1: Via `tfidf_stopwords_path` (Recommended)

Set the global config parameter:

```bash
# Environment variable
export TFIDF_STOPWORDS_PATH=/path/to/UAIR/dagspaces/uair/conf/stopwords/news_extended.yaml

# Or command line
python -m dagspaces.uair.cli \
  tfidf_stopwords_path=dagspaces/uair/conf/stopwords/comprehensive.yaml
```

### Method 2: Via Hydra defaults

In your config file:

```yaml
defaults:
  - _self_
  - override tfidf_stopwords_path: dagspaces/uair/conf/stopwords/minimal.yaml
```

### Method 3: Pipeline override

```yaml
nodes:
  topic:
    overrides:
      tfidf_stopwords_path: dagspaces/uair/conf/stopwords/news_extended.yaml
```

### Method 4: Direct list (fallback to old behavior)

```yaml
topic:
  tfidf_stop_words:
    - newsletter
    - subscribe
    - article
```

## File Format

Stopwords files support three formats:

### YAML with dict (recommended)
```yaml
stopwords:
  - word1
  - word2
  - word3
```

### YAML list
```yaml
- word1
- word2
- word3
```

### JSON with dict
```json
{
  "stopwords": ["word1", "word2", "word3"]
}
```

### JSON list
```json
["word1", "word2", "word3"]
```

### Plain text (.txt)
```
word1
word2
word3
# Lines starting with # are ignored
```

## Precedence

The stopwords loading follows this precedence:

1. **`tfidf_stopwords_path`** (if set and file exists)
2. **`topic.tfidf_stop_words`** (config parameter)
3. **"english"** (scikit-learn default)

## Creating Custom Stopwords

Create your own stopwords file:

```yaml
# conf/stopwords/my_domain.yaml
stopwords:
  # Your domain-specific terms
  - ai
  - artificial
  - intelligence
  - algorithm
  
  # Add news boilerplate
  - newsletter
  - subscribe
  - breaking
```

Then use it:

```bash
python -m dagspaces.uair.cli \
  tfidf_stopwords_path=dagspaces/uair/conf/stopwords/my_domain.yaml
```

## Combining with scikit-learn's English Stopwords

To combine your custom list with scikit-learn's built-in English stopwords:

```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import yaml

# Load your custom stopwords
with open('dagspaces/uair/conf/stopwords/news_extended.yaml') as f:
    custom = yaml.safe_load(f)['stopwords']

# Combine
combined = list(ENGLISH_STOP_WORDS) + custom

# Create a new YAML file
with open('dagspaces/uair/conf/stopwords/combined.yaml', 'w') as f:
    yaml.dump({'stopwords': combined}, f)
```

## Recommendations

### For news articles:
- Start with `news_extended.yaml` (90 words)
- Add domain-specific terms as needed
- Monitor topic quality and adjust

### For general text:
- Use scikit-learn's "english" (318 words) via `topic.tfidf_stop_words: english`
- Or create domain-specific lists

### For maximum vocabulary retention:
- Use `minimal.yaml` (11 words)
- Or set `tfidf_stopwords_path: null` and `topic.tfidf_stop_words: null`

## Debugging

Check which stopwords were loaded:

```python
import yaml

# Read the stopwords file
with open('dagspaces/uair/conf/stopwords/news_extended.yaml') as f:
    stopwords = yaml.safe_load(f)['stopwords']

print(f"Loaded {len(stopwords)} stopwords")
print(f"First 10: {stopwords[:10]}")

# Check if specific words are in the list
for word in ['newsletter', 'subscribe', 'ai', 'technology']:
    print(f"'{word}': {'✓ filtered' if word in stopwords else '✗ not filtered'}")
```

## Performance Impact

| Stopwords Count | Feature Space | Clustering Speed | Topic Quality |
|----------------|---------------|------------------|---------------|
| 0 (none) | Largest | Slowest | Lower (noise) |
| ~50-100 | Large | Moderate | Good |
| ~200-400 | Moderate | Fast | Best |
| ~500+ | Small | Fastest | Risk of over-filtering |

## Examples

### Example 1: News analysis with minimal filtering

```bash
python -m dagspaces.uair.cli \
  pipeline=topic_modeling_of_relevant_classifications \
  tfidf_stopwords_path=dagspaces/uair/conf/stopwords/minimal.yaml
```

### Example 2: Comprehensive news filtering

```bash
python -m dagspaces.uair.cli \
  pipeline=topic_modeling_of_relevant_classifications \
  tfidf_stopwords_path=dagspaces/uair/conf/stopwords/comprehensive.yaml
```

### Example 3: Environment variable

```bash
export TFIDF_STOPWORDS_PATH=$PWD/dagspaces/uair/conf/stopwords/news_extended.yaml
python -m dagspaces.uair.cli pipeline=topic_modeling_of_relevant_classifications
```

### Example 4: Override in pipeline config

```yaml
# conf/pipeline/my_topic_pipeline.yaml
nodes:
  topic:
    stage: topic
    overrides:
      tfidf_stopwords_path: ${oc.env:PWD}/dagspaces/uair/conf/stopwords/comprehensive.yaml
      topic.embed.batch_size: 128
      topic.hdbscan.min_cluster_size: 15
```

## See Also

- **scikit-learn TfidfVectorizer**: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- **UAIR Topic Modeling Guide**: `/docs/USER_GUIDE.md`
- **Configuration Guide**: `/docs/CONFIGURATION_GUIDE.md`
