# Technical Design Document: Feature Discovery

**Document ID:** 004_FTDD|Feature_Discovery
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Implemented
**Related PRD:** [004_FPRD|Feature_Discovery](../prds/004_FPRD|Feature_Discovery.md)

---

## 1. System Architecture

### 1.1 Extraction Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│                     Feature Extraction Flow                      │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Dataset  │───→│  Model   │───→│   SAE    │───→│ Features │  │
│  │(samples) │    │(forward) │    │(encode)  │    │(top-k)   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                        │              │               │          │
│                        ▼              ▼               ▼          │
│              ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│              │ Activations  │ │   Latent     │ │  Database    │ │
│              │ (hook)       │ │   Features   │ │  + Files     │ │
│              └──────────────┘ └──────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend                                  │
│  ┌───────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │ FeaturesPanel │  │FeatureDetailModal│ │StartExtraction │ │
│  └───────┬───────┘  └────────┬────────┘  └───────┬───────┘ │
│          │                   │                    │         │
│  ┌───────┴───────────────────┴────────────────────┴───────┐ │
│  │                 featuresStore                           │ │
│  └────────────────────────────┬────────────────────────────┘ │
└───────────────────────────────┼─────────────────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────┐
│                    Backend                                   │
│  ┌───────────────────┐  ┌───────────────────┐               │
│  │ ExtractionService │  │  FeatureService   │               │
│  └─────────┬─────────┘  └─────────┬─────────┘               │
│            │                      │                         │
│  ┌─────────┴──────────────────────┴─────────┐               │
│  │           Celery Workers                  │               │
│  │  ┌────────────────┐  ┌────────────────┐  │               │
│  │  │extract_features│  │auto_label_task │  │               │
│  │  │_task           │  │                │  │               │
│  │  └────────────────┘  └────────────────┘  │               │
│  └───────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Database Schema

### 2.1 Feature Table
```sql
CREATE TABLE features (
    id UUID PRIMARY KEY,
    training_id UUID REFERENCES trainings(id) ON DELETE CASCADE,
    external_sae_id UUID REFERENCES external_saes(id) ON DELETE CASCADE,
    neuron_index INTEGER NOT NULL,
    label VARCHAR(500),           -- Semantic label
    category VARCHAR(255),        -- Category label
    label_confidence FLOAT,
    label_source VARCHAR(50),     -- manual, auto, imported
    statistics JSONB,             -- {frequency, max, mean, interpretability}
    top_tokens JSONB,             -- [{token, count, mean_activation}]
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Either training_id or external_sae_id must be set
    CONSTRAINT features_sae_check CHECK (
        (training_id IS NOT NULL AND external_sae_id IS NULL) OR
        (training_id IS NULL AND external_sae_id IS NOT NULL)
    )
);

CREATE INDEX idx_features_training ON features(training_id);
CREATE INDEX idx_features_external_sae ON features(external_sae_id);
CREATE INDEX idx_features_label ON features USING gin(to_tsvector('english', label));
```

### 2.2 Feature Activation Table
```sql
CREATE TABLE feature_activations (
    id UUID PRIMARY KEY,
    feature_id UUID REFERENCES features(id) ON DELETE CASCADE,
    activation_value FLOAT NOT NULL,
    token VARCHAR(255),
    token_id INTEGER,
    context_before TEXT,
    context_after TEXT,
    position INTEGER,
    sample_index INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_activations_feature ON feature_activations(feature_id);
CREATE INDEX idx_activations_value ON feature_activations(activation_value DESC);
```

### 2.3 Extraction Job Table
```sql
CREATE TABLE extraction_jobs (
    id UUID PRIMARY KEY,
    training_id UUID REFERENCES trainings(id),
    external_sae_id UUID REFERENCES external_saes(id),
    dataset_id UUID REFERENCES datasets(id),
    tokenization_id UUID,
    config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    progress FLOAT DEFAULT 0,
    features_found INTEGER DEFAULT 0,
    samples_processed INTEGER DEFAULT 0,
    celery_task_id VARCHAR(255),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

---

## 3. Extraction Algorithm

### 3.1 Vectorized Extraction
```python
def extract_features_vectorized(
    model: nn.Module,
    sae: SparseAutoencoder,
    dataset: Dataset,
    config: ExtractionConfig
) -> Dict[int, List[Activation]]:
    """
    Extract top activations for all features in parallel.

    Memory-efficient approach:
    1. Process samples in batches
    2. Track top-k per feature using heap
    3. Store context windows
    """
    n_features = sae.d_sae
    top_k = config.top_k_per_feature

    # Min-heap for each feature (keeps top-k largest)
    feature_heaps = [[] for _ in range(n_features)]

    for batch_idx, batch in enumerate(dataset.iter(batch_size=config.batch_size)):
        # Get activations from model
        with torch.no_grad():
            activations = get_layer_activations(model, batch, config.layer)

        # Encode through SAE
        z = sae.encode(activations)  # [batch, seq, n_features]

        # Find activations above threshold
        mask = z > config.activation_threshold
        batch_indices, seq_indices, feature_indices = torch.where(mask)

        for b, s, f in zip(batch_indices, seq_indices, feature_indices):
            value = z[b, s, f].item()
            f_idx = f.item()

            # Maintain top-k heap
            if len(feature_heaps[f_idx]) < top_k:
                heapq.heappush(feature_heaps[f_idx], (value, {
                    'token': batch['tokens'][b][s],
                    'context_before': get_context(batch, b, s, -config.context_window),
                    'context_after': get_context(batch, b, s, config.context_window),
                    'position': s,
                    'sample_index': batch_idx * config.batch_size + b
                }))
            elif value > feature_heaps[f_idx][0][0]:
                heapq.heapreplace(feature_heaps[f_idx], (value, {...}))

    return {i: [h[1] for h in heap] for i, heap in enumerate(feature_heaps)}
```

### 3.2 Statistics Computation
```python
def compute_feature_statistics(z: Tensor, feature_idx: int) -> dict:
    """Compute statistics for a single feature."""
    feature_activations = z[:, :, feature_idx]

    return {
        "frequency": (feature_activations > 0).float().mean().item(),
        "max_activation": feature_activations.max().item(),
        "mean_activation": feature_activations[feature_activations > 0].mean().item(),
        "interpretability_score": compute_interpretability(feature_activations)
    }

def compute_interpretability(activations: Tensor) -> float:
    """
    Score based on activation distribution.
    Higher score = more interpretable (sparse, focused activations).
    """
    if activations.max() == 0:
        return 0.0

    # Sparsity component
    sparsity = 1 - (activations > 0).float().mean()

    # Concentration component (how peaked is the distribution)
    normalized = activations / activations.max()
    concentration = (normalized ** 2).mean() / normalized.mean()

    return (sparsity * 0.6 + concentration * 0.4).item()
```

---

## 4. Auto-Labeling System

### 4.1 Labeling Flow
```
┌───────────────────────────────────────────────────────────────┐
│                    Auto-Labeling Pipeline                      │
│                                                                │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐            │
│  │  Features  │──→│  Format    │──→│  OpenAI    │            │
│  │  (batch)   │   │  Examples  │   │  API Call  │            │
│  └────────────┘   └────────────┘   └────────────┘            │
│                                          │                    │
│                                          ▼                    │
│                                   ┌────────────┐             │
│                                   │   Parse    │             │
│                                   │  Response  │             │
│                                   └────────────┘             │
│                                          │                    │
│                         ┌────────────────┼────────────────┐  │
│                         ▼                ▼                ▼  │
│                   ┌──────────┐    ┌──────────┐    ┌────────┐│
│                   │ Semantic │    │ Category │    │Confidence│
│                   │  Label   │    │  Label   │    │ Score   ││
│                   └──────────┘    └──────────┘    └────────┘│
└───────────────────────────────────────────────────────────────┘
```

### 4.2 OpenAI Integration
```python
class OpenAILabelingService:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    async def label_feature(
        self,
        feature_examples: List[ActivationExample],
        prompt_template: LabelingPromptTemplate
    ) -> FeatureLabel:
        """Label a single feature using GPT-4o."""

        # Format examples
        formatted = self._format_examples(feature_examples)

        # Build prompt
        messages = [
            {"role": "system", "content": prompt_template.system_prompt},
            {"role": "user", "content": prompt_template.user_prompt.format(
                examples=formatted
            )}
        ]

        # Call API
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )

        # Parse response
        return self._parse_label_response(response.choices[0].message.content)

    def _format_examples(self, examples: List[ActivationExample]) -> str:
        """Format activation examples for prompt."""
        lines = []
        for ex in examples[:20]:  # Top 20 examples
            lines.append(f"Token: [{ex.token}]")
            lines.append(f"Context: ...{ex.context_before}[{ex.token}]{ex.context_after}...")
            lines.append(f"Activation: {ex.value:.2f}")
            lines.append("")
        return "\n".join(lines)
```

### 4.3 Labeling Prompt Templates
```sql
CREATE TABLE labeling_prompt_templates (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    system_prompt TEXT NOT NULL,
    user_prompt TEXT NOT NULL,  -- Contains {examples} placeholder
    is_default BOOLEAN DEFAULT FALSE,
    is_favorite BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 5. API Design

### 5.1 Feature Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/features` | List with filtering |
| GET | `/features/{id}` | Get feature detail |
| PATCH | `/features/{id}` | Update labels |
| POST | `/features/extraction` | Start extraction job |
| GET | `/features/extraction/{id}` | Get extraction status |
| POST | `/features/labeling` | Start auto-labeling |
| GET | `/features/labeling/{id}` | Get labeling status |
| POST | `/features/export` | Export to JSON |

### 5.2 Query Parameters
```python
class FeatureQuery(BaseModel):
    page: int = 1
    limit: int = 50
    training_id: Optional[UUID] = None
    external_sae_id: Optional[UUID] = None
    search: Optional[str] = None  # Full-text search on label
    has_label: Optional[bool] = None
    min_frequency: Optional[float] = None
    max_frequency: Optional[float] = None
    min_activation: Optional[float] = None
    sort_by: Literal['frequency', 'max', 'mean', 'interpretability'] = 'frequency'
    sort_order: Literal['asc', 'desc'] = 'desc'
```

---

## 6. WebSocket Events

### 6.1 Extraction Channel: `extraction/{job_id}`
| Event | Payload |
|-------|---------|
| `progress` | `{progress, features_found, samples_processed}` |
| `completed` | `{total_features, extraction_time}` |
| `failed` | `{error}` |

### 6.2 Labeling Channel: `labeling/{job_id}`
| Event | Payload |
|-------|---------|
| `progress` | `{progress, labeled_count, total}` |
| `result` | `{feature_id, label, category, confidence}` |
| `completed` | `{total_labeled}` |
| `failed` | `{error, last_feature}` |

---

## 7. Frontend State

### 7.1 featuresStore
```typescript
interface FeaturesState {
  features: Feature[];
  selectedFeature: Feature | null;
  extractionJobs: ExtractionJob[];
  labelingJobs: LabelingJob[];
  filters: FeatureFilters;
  pagination: { page: number; limit: number; total: number };

  // Actions
  fetchFeatures: (query: FeatureQuery) => Promise<void>;
  updateLabel: (id: string, label: string, category: string) => Promise<void>;
  startExtraction: (config: ExtractionConfig) => Promise<void>;
  startLabeling: (config: LabelingConfig) => Promise<void>;

  // WebSocket handlers
  updateExtractionProgress: (jobId: string, data: ProgressData) => void;
  updateLabelingProgress: (jobId: string, data: ProgressData) => void;
}
```

---

## 8. Token Highlighting

### 8.1 TokenHighlight Component
```typescript
interface TokenHighlightProps {
  contextBefore: string;
  token: string;
  contextAfter: string;
  activationValue: number;
}

// Renders:
// "...context before [TOKEN] context after..."
//                    ^^^^^^^
//              highlighted with intensity based on activation
```

---

## 9. Templates

### 9.1 Extraction Templates
```json
{
  "name": "Standard Extraction",
  "config": {
    "activation_threshold": 0.5,
    "top_k_per_feature": 100,
    "context_window": 10,
    "batch_size": 32,
    "filter_stop_words": true,
    "min_token_length": 3
  }
}
```

---

*Related: [PRD](../prds/004_FPRD|Feature_Discovery.md) | [TID](../tids/004_FTID|Feature_Discovery.md) | [FTASKS](../tasks/004_FTASKS|Feature_Discovery.md)*
