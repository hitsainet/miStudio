# Product Requirements Document: Native Neuronpedia Integration

## 1. Executive Summary

### 1.1 Overview
This PRD defines requirements for native Neuronpedia integration in miStudio, enabling users to:
1. Export trained SAEs with full feature dashboard data to Neuronpedia
2. Generate Neuronpedia-compatible feature visualizations natively (without SAEDashboard dependency)
3. Produce community-standard outputs that integrate with the mechanistic interpretability ecosystem
4. Share SAEs and feature analyses with the broader research community

### 1.2 Value Proposition
- **For Researchers**: Share findings with the global mech interp community via Neuronpedia
- **For the Community**: More SAEs and feature data available on Neuronpedia
- **For miStudio**: Position as a first-class citizen in the interpretability ecosystem
- **For Collaboration**: Enable cross-tool workflows (train in miStudio → explore in Neuronpedia → steer in miStudio)

### 1.3 Success Metrics
- Time to export SAE to Neuronpedia format: < 30 minutes for 16k features
- Export success rate: > 99%
- Feature data completeness: 100% of Neuronpedia required fields
- Community adoption: > 10 miStudio SAEs uploaded to Neuronpedia within 6 months

### 1.4 Strategic Context
miStudio already supports:
- ✅ Downloading SAEs from HuggingFace (including Gemma-Scope)
- ✅ Training custom SAEs
- ✅ Feature extraction and labeling
- ✅ Feature steering with comparison to Neuronpedia
- ✅ Uploading SAEs to HuggingFace

This PRD completes the ecosystem loop by enabling **export TO Neuronpedia**, making miStudio a full participant in the interpretability community rather than just a consumer.

---

## 2. Background & Context

### 2.1 What is Neuronpedia?

Neuronpedia is the leading open-source platform for mechanistic interpretability research. It provides:
- **Feature Dashboards**: Interactive visualizations of SAE features
- **Live Inference**: Test features on custom text
- **Steering**: Modify model behavior using features
- **Search**: Find features by semantic description or activation pattern
- **Collaboration**: Share features, lists, and explanations

### 2.2 Why Native Integration?

**Option A: SAEDashboard Dependency** (Rejected)
- Requires external library for data generation
- Less control over computation
- Potential version conflicts
- Additional dependencies to maintain

**Option B: Native Implementation** (Selected)
- Full control over data generation pipeline
- Optimize for miStudio's existing data structures
- No external dependencies
- Can extend/customize as needed
- Better long-term maintainability

### 2.3 Neuronpedia Data Requirements

Each feature on Neuronpedia requires:

| Data Type | Description | miStudio Status |
|-----------|-------------|-----------------|
| Top Activating Examples | 20+ text samples with token-level activations | ✅ Have (100 examples) |
| Activation Histogram | Distribution of activation values | ❌ Need to add |
| Top Positive Logits | Tokens most predicted by feature | ⚠️ Computed on-demand |
| Top Negative Logits | Tokens least predicted by feature | ⚠️ Computed on-demand |
| Feature Statistics | Frequency, max activation, L0 | ✅ Have |
| Explanations | Auto-generated feature descriptions | ✅ Have (LLM labels) |

### 2.4 Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         miStudio                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  SAE Training │───▶│   Feature    │───▶│  Neuronpedia Export  │  │
│  │    Module     │    │  Extraction  │    │      Service         │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                   │                  │
│                                                   ▼                  │
│                                          ┌──────────────────┐       │
│                                          │  Export Formats  │       │
│                                          ├──────────────────┤       │
│                                          │ • Neuronpedia JSON│       │
│                                          │ • SAELens Format  │       │
│                                          │ • HuggingFace     │       │
│                                          └──────────────────┘       │
│                                                   │                  │
└───────────────────────────────────────────────────┼──────────────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────┐
                    │                               ▼               │
                    │  ┌─────────────┐    ┌─────────────────────┐  │
                    │  │ HuggingFace │    │    Neuronpedia      │  │
                    │  │     Hub     │    │    Platform         │  │
                    │  └─────────────┘    └─────────────────────┘  │
                    │                      External Ecosystem       │
                    └───────────────────────────────────────────────┘
```

---

## 3. User Stories & Personas

### 3.1 Primary Personas

**Research Scientist (Sarah)**
- Trained a novel SAE architecture on Llama-3
- Wants to share findings with the community
- Needs features visible on Neuronpedia for paper citation
- Values scientific credibility and reproducibility

**Open Source Contributor (Marcus)**
- Trained SAEs on multiple model families
- Wants to contribute to the public interpretability commons
- Needs batch export capabilities
- Values efficiency and automation

**Safety Researcher (Jordan)**
- Identified important safety-relevant features
- Wants to enable other researchers to study them
- Needs to export specific feature subsets
- Values selective, curated sharing

### 3.2 User Stories

**Story 1: Full SAE Export**
```
As Sarah (researcher),
I want to export my trained SAE with all feature data to Neuronpedia format,
So that the community can explore and build on my research.

Acceptance Criteria:
- Can initiate export from SAE detail page
- Export includes all required Neuronpedia data
- Progress indicator shows export status
- Download as ZIP with all required files
- Clear instructions for Neuronpedia upload
```

**Story 2: Selective Feature Export**
```
As Jordan (safety researcher),
I want to export only specific features I've curated,
So that I can share safety-relevant features without noise.

Acceptance Criteria:
- Can select specific features for export
- Can export features by label/category
- Can export features above activation threshold
- Maintains all dashboard data for selected features
```

**Story 3: Incremental Export**
```
As Marcus (contributor),
I want to add new features to an existing Neuronpedia SAE,
So that I can update my contribution without re-exporting everything.

Acceptance Criteria:
- Can export delta (new features only)
- Maintains compatibility with existing Neuronpedia data
- Can merge with previous export
```

**Story 4: Export Validation**
```
As Sarah (researcher),
I want to validate my export before uploading to Neuronpedia,
So that I catch any issues before public release.

Acceptance Criteria:
- Validation checks all required fields
- Reports missing or malformed data
- Suggests fixes for common issues
- Can preview feature dashboards locally
```

---

## 4. Functional Requirements

### 4.1 Feature Dashboard Data Generation

**FR-1.1: Top Activating Examples**
- MUST generate 20+ activation examples per feature (Neuronpedia minimum)
- MUST include for each example:
  - Token strings (properly decoded)
  - Token-level activation values
  - Context window (surrounding tokens)
  - Max activation position
  - Source dataset reference
- SHOULD support configurable example count (20-100)
- SHOULD deduplicate near-identical examples
- MUST handle special tokens (BOS, EOS, PAD) correctly

**FR-1.2: Logit Lens Data**
- MUST compute top-K positive logits per feature (K=10 minimum)
- MUST compute top-K negative logits per feature (K=10 minimum)
- MUST store:
  - Token string
  - Logit value
  - Rank
- SHOULD support configurable K (10-50)
- MUST handle tied logit values correctly
- MUST use feature's decoder vector for computation

**FR-1.3: Activation Histograms**
- MUST compute activation value distribution per feature
- MUST include:
  - Histogram bin edges
  - Histogram counts
  - Total sample count
  - Non-zero activation count
- SHOULD use logarithmic bins for better visualization
- MUST handle features with very sparse activations
- SHOULD support configurable bin count (20-100)

**FR-1.4: Feature Statistics**
- MUST compute and include:
  - `activation_frequency`: Fraction of tokens where feature activates
  - `max_activation`: Maximum observed activation value
  - `mean_activation`: Mean activation (when active)
  - `std_activation`: Standard deviation of activations
  - `l0_contribution`: Feature's contribution to average L0
- SHOULD include:
  - `dead_feature`: Boolean indicating if feature never activates
  - `quantiles`: [25th, 50th, 75th, 95th, 99th] percentiles

**FR-1.5: Token Aggregation**
- MUST compute top activating tokens across all examples
- MUST include:
  - Token string
  - Total activation sum
  - Occurrence count
  - Average activation when present
- SHOULD rank by configurable metric (sum, count, average)
- MUST handle subword tokens appropriately

### 4.2 Export Formats

**FR-2.1: Neuronpedia JSON Format**
- MUST generate JSON files matching Neuronpedia's import schema
- MUST organize as:
  ```
  export/
  ├── metadata.json           # SAE-level metadata
  ├── features/
  │   ├── 0.json             # Feature 0 data
  │   ├── 1.json             # Feature 1 data
  │   └── ...
  └── explanations/
      └── explanations.json   # All feature explanations
  ```
- MUST include all required fields per Neuronpedia schema
- SHOULD compress large exports (gzip)

**FR-2.2: Feature JSON Schema**
```json
{
  "feature_index": 0,
  "activations": [
    {
      "tokens": ["The", " quick", " brown", " fox"],
      "values": [0.0, 2.3, 0.5, 1.2],
      "max_value": 2.3,
      "max_token_index": 1
    }
  ],
  "logits": {
    "top_positive": [
      {"token": "fast", "value": 3.45},
      {"token": "swift", "value": 3.21}
    ],
    "top_negative": [
      {"token": "slow", "value": -2.89},
      {"token": "lazy", "value": -2.45}
    ]
  },
  "histogram": {
    "bin_edges": [0.0, 0.5, 1.0, 1.5, 2.0],
    "counts": [10000, 500, 100, 20, 5]
  },
  "statistics": {
    "activation_frequency": 0.023,
    "max_activation": 12.5,
    "mean_activation": 1.8,
    "std_activation": 0.9
  },
  "top_tokens": [
    {"token": "quick", "total_activation": 45.6, "count": 23}
  ]
}
```

**FR-2.3: Metadata JSON Schema**
```json
{
  "model_id": "gemma-2-2b",
  "sae_id": "layer_12_res_16k",
  "neuronpedia_id": "gemma-2-2b/12-mistudio-res-16k",
  "hook_point": "blocks.12.hook_resid_post",
  "d_sae": 16384,
  "d_model": 2048,
  "architecture": "standard",
  "training_dataset": "monology/pile-uncopyrighted",
  "n_training_tokens": 1000000000,
  "l0": 45.2,
  "explained_variance": 0.92,
  "created_at": "2025-01-15T10:30:00Z",
  "miStudio_version": "1.2.0",
  "export_version": "1.0"
}
```

**FR-2.4: SAELens Compatibility**
- MUST also export in SAELens format for HuggingFace
- MUST include:
  - `cfg.json` with all required fields
  - `sae_weights.safetensors`
  - `sparsity.safetensors` (optional)
- MUST generate `pretrained_saes.yaml` entry
- SHOULD support automatic PR creation to SAELens repo

**FR-2.5: TransformerLens Hook Mapping**
- MUST map miStudio layer indices to TransformerLens hook names
- MUST support all standard hook points:
  - `blocks.{layer}.hook_resid_pre`
  - `blocks.{layer}.hook_resid_post`
  - `blocks.{layer}.hook_mlp_out`
  - `blocks.{layer}.attn.hook_z`
- MUST validate hook names against model architecture
- SHOULD auto-detect appropriate hook point from SAE config

### 4.3 Export Pipeline

**FR-3.1: Export Initiation**
- MUST provide export button on SAE detail page
- MUST show export configuration dialog:
  - Output format selection (Neuronpedia, SAELens, Both)
  - Feature selection (All, Range, Filtered)
  - Data completeness options
  - Compression settings
- MUST validate SAE has required data before export
- SHOULD estimate export time and size

**FR-3.2: Background Processing**
- MUST run export as background job
- MUST show progress indicator with stages:
  - Computing logit lens data
  - Generating histograms
  - Aggregating tokens
  - Writing JSON files
  - Compressing output
- MUST support job cancellation
- MUST handle failures gracefully with partial recovery
- SHOULD support pause/resume for large exports

**FR-3.3: Data Computation Strategy**
- MUST compute missing data on-demand during export
- MUST cache computed data for future exports
- SHOULD parallelize computation across features
- SHOULD batch GPU operations for efficiency
- MUST handle OOM gracefully with automatic batching

**FR-3.4: Export Validation**
- MUST validate output against Neuronpedia schema
- MUST check for:
  - Missing required fields
  - Invalid data types
  - Empty arrays
  - NaN/Inf values
  - Encoding issues in tokens
- MUST generate validation report
- SHOULD offer auto-fix for common issues

**FR-3.5: Output Delivery**
- MUST provide download as ZIP archive
- MUST include README with upload instructions
- SHOULD support direct upload to cloud storage
- SHOULD generate shareable link for large exports

### 4.4 Integration Points

**FR-4.1: Extraction Pipeline Integration**
- MUST integrate logit lens computation into extraction pipeline
- MUST integrate histogram computation into extraction pipeline
- SHOULD make dashboard data computation optional (config flag)
- MUST not slow down extraction when dashboard data not needed

**FR-4.2: Database Schema Extensions**
- MUST add tables/columns for:
  - Feature logit lens data
  - Feature histograms
  - Token aggregations
  - Export job tracking
- MUST support incremental updates
- SHOULD support data versioning

**FR-4.3: API Endpoints**
- MUST provide REST endpoints:
  ```
  POST /api/v1/saes/{sae_id}/export/neuronpedia
  GET  /api/v1/saes/{sae_id}/export/neuronpedia/status/{job_id}
  GET  /api/v1/saes/{sae_id}/export/neuronpedia/download/{job_id}
  POST /api/v1/saes/{sae_id}/export/validate
  GET  /api/v1/saes/{sae_id}/features/{feature_id}/dashboard
  ```
- MUST support pagination for large feature sets
- MUST include proper error responses

**FR-4.4: Neuronpedia Coordination**
- MUST generate upload form pre-fill data
- SHOULD provide direct link to Neuronpedia upload form
- SHOULD track upload status (when API available)
- MUST document manual upload process clearly

### 4.5 Explanation Export

**FR-5.1: Label Export**
- MUST export existing feature labels as explanations
- MUST format as:
  ```json
  {
    "feature_index": 0,
    "explanations": [
      {
        "description": "words related to speed and motion",
        "method": "miStudio_llm_labeling",
        "model": "gpt-4o-mini",
        "score": null,
        "created_at": "2025-01-15T10:30:00Z"
      }
    ]
  }
  ```
- MUST support multiple explanations per feature
- SHOULD include confidence scores when available

**FR-5.2: Explanation Scoring Compatibility**
- SHOULD format explanations for Neuronpedia scoring API
- SHOULD include activation data needed for scoring
- MAY integrate with Neuronpedia scorer for pre-scoring

---

## 5. Non-Functional Requirements

### 5.1 Performance

**NFR-1.1: Export Speed**
- MUST complete export of 16k features in < 30 minutes (GPU)
- MUST complete export of 16k features in < 2 hours (CPU)
- SHOULD support incremental export (only new/changed features)
- MUST show accurate progress estimates

**NFR-1.2: Memory Efficiency**
- MUST not exceed 16GB RAM for 16k feature export
- MUST support streaming export for very large SAEs (>100k features)
- SHOULD release memory promptly after computation
- MUST handle OOM gracefully

**NFR-1.3: Storage Efficiency**
- MUST compress exports to < 500MB for 16k features
- SHOULD deduplicate common token strings
- SHOULD use efficient numeric encoding

### 5.2 Reliability

**NFR-2.1: Data Integrity**
- MUST validate all exported data
- MUST handle Unicode correctly
- MUST preserve numerical precision
- MUST not corrupt data on partial failure

**NFR-2.2: Idempotency**
- MUST produce identical output for identical input
- MUST handle re-export gracefully
- SHOULD detect and skip unchanged features

**NFR-2.3: Error Handling**
- MUST provide clear error messages
- MUST support partial export on failure
- MUST log errors for debugging
- SHOULD suggest remediation steps

### 5.3 Compatibility

**NFR-3.1: Neuronpedia Compatibility**
- MUST produce output compatible with current Neuronpedia
- MUST validate against Neuronpedia schema
- SHOULD track Neuronpedia schema changes
- MUST document any compatibility limitations

**NFR-3.2: SAELens Compatibility**
- MUST produce valid SAELens format
- MUST work with SAELens `from_pretrained`
- SHOULD pass SAELens validation checks

**NFR-3.3: Model Compatibility**
- MUST support all models supported by miStudio
- MUST handle model-specific tokenizers
- MUST validate TransformerLens compatibility

### 5.4 Usability

**NFR-4.1: User Experience**
- MUST provide one-click export for common cases
- MUST show clear progress and status
- MUST provide helpful error messages
- SHOULD offer export presets

**NFR-4.2: Documentation**
- MUST document export process end-to-end
- MUST include Neuronpedia upload guide
- MUST document all configuration options
- SHOULD provide troubleshooting guide

---

## 6. Technical Architecture

### 6.1 Component Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Neuronpedia Export Service                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Export Orchestrator                         │   │
│  │  • Manages export jobs                                           │   │
│  │  • Coordinates data computation                                  │   │
│  │  • Handles progress tracking                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│           ┌──────────────────┼──────────────────┐                       │
│           ▼                  ▼                  ▼                        │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐           │
│  │  Logit Lens     │ │   Histogram     │ │    Token        │           │
│  │  Computer       │ │   Generator     │ │   Aggregator    │           │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘           │
│           │                  │                  │                        │
│           └──────────────────┼──────────────────┘                       │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Data Serializer                             │   │
│  │  • JSON generation                                               │   │
│  │  • Schema validation                                             │   │
│  │  • Compression                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Export Packager                             │   │
│  │  • File organization                                             │   │
│  │  • ZIP creation                                                  │   │
│  │  • README generation                                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Data Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   SAE        │     │   Feature    │     │  Activation  │
│   Weights    │     │   Cache      │     │    Store     │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│                 Export Data Aggregator                   │
│  • Loads SAE decoder vectors                            │
│  • Retrieves cached activation examples                 │
│  • Fetches stored statistics                            │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                 Missing Data Computer                    │
│  • Identifies missing dashboard data                    │
│  • Schedules computation jobs                           │
│  • Caches results for future use                        │
└─────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │  Logit    │    │ Histogram │    │   Token   │
    │  Lens     │    │  Builder  │    │Aggregator │
    └───────────┘    └───────────┘    └───────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  JSON Serializer                         │
│  • Formats data per Neuronpedia schema                  │
│  • Validates completeness                               │
│  • Handles encoding edge cases                          │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   File Packager                          │
│  • Organizes directory structure                        │
│  • Compresses output                                    │
│  • Generates metadata                                   │
└─────────────────────────────────────────────────────────┘
```

### 6.3 Key Algorithms

**Algorithm 1: Logit Lens Computation**
```python
def compute_logit_lens(
    sae: SAE,
    model: HookedTransformer,
    feature_indices: List[int],
    k: int = 20
) -> Dict[int, LogitLensData]:
    """
    Compute top positive and negative logits for each feature.
    
    The logit lens for a feature is computed by:
    1. Taking the feature's decoder vector (W_dec[feature_idx])
    2. Projecting through the model's unembedding matrix (W_U)
    3. Finding tokens with highest/lowest logit contributions
    """
    results = {}
    
    # Get decoder vectors for all features at once
    decoder_vectors = sae.W_dec[feature_indices]  # (n_features, d_model)
    
    # Get unembedding matrix
    W_U = model.W_U  # (d_model, vocab_size)
    
    # Compute logits for all features: (n_features, vocab_size)
    feature_logits = decoder_vectors @ W_U
    
    for i, feature_idx in enumerate(feature_indices):
        logits = feature_logits[i]
        
        # Top positive
        top_pos_indices = torch.topk(logits, k).indices
        top_pos_values = logits[top_pos_indices]
        
        # Top negative
        top_neg_indices = torch.topk(-logits, k).indices
        top_neg_values = logits[top_neg_indices]
        
        results[feature_idx] = LogitLensData(
            top_positive=[(model.tokenizer.decode([idx]), val.item()) 
                         for idx, val in zip(top_pos_indices, top_pos_values)],
            top_negative=[(model.tokenizer.decode([idx]), val.item())
                         for idx, val in zip(top_neg_indices, top_neg_values)]
        )
    
    return results
```

**Algorithm 2: Activation Histogram**
```python
def compute_activation_histogram(
    activations: np.ndarray,
    n_bins: int = 50,
    log_scale: bool = True
) -> HistogramData:
    """
    Compute histogram of feature activation values.
    
    Uses logarithmic bins for better visualization of heavy-tailed
    activation distributions.
    """
    # Filter to non-zero activations
    nonzero_acts = activations[activations > 0]
    
    if len(nonzero_acts) == 0:
        return HistogramData(
            bin_edges=[0.0],
            counts=[0],
            total_count=len(activations),
            nonzero_count=0
        )
    
    if log_scale:
        # Logarithmic bins
        log_min = np.log10(nonzero_acts.min())
        log_max = np.log10(nonzero_acts.max())
        bin_edges = np.logspace(log_min, log_max, n_bins + 1)
    else:
        # Linear bins
        bin_edges = np.linspace(0, nonzero_acts.max(), n_bins + 1)
    
    counts, _ = np.histogram(nonzero_acts, bins=bin_edges)
    
    return HistogramData(
        bin_edges=bin_edges.tolist(),
        counts=counts.tolist(),
        total_count=len(activations),
        nonzero_count=len(nonzero_acts)
    )
```

**Algorithm 3: Token Aggregation**
```python
def aggregate_top_tokens(
    activation_examples: List[ActivationExample],
    k: int = 50
) -> List[TokenAggregation]:
    """
    Aggregate tokens across all activation examples to find
    tokens that consistently cause high activations.
    """
    token_stats = defaultdict(lambda: {
        'total_activation': 0.0,
        'count': 0,
        'max_activation': 0.0
    })
    
    for example in activation_examples:
        for token, activation in zip(example.tokens, example.activations):
            if activation > 0:
                stats = token_stats[token]
                stats['total_activation'] += activation
                stats['count'] += 1
                stats['max_activation'] = max(stats['max_activation'], activation)
    
    # Sort by total activation
    sorted_tokens = sorted(
        token_stats.items(),
        key=lambda x: x[1]['total_activation'],
        reverse=True
    )[:k]
    
    return [
        TokenAggregation(
            token=token,
            total_activation=stats['total_activation'],
            count=stats['count'],
            mean_activation=stats['total_activation'] / stats['count'],
            max_activation=stats['max_activation']
        )
        for token, stats in sorted_tokens
    ]
```

### 6.4 Database Schema Extensions

```sql
-- Feature Dashboard Data
CREATE TABLE feature_dashboard_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sae_id UUID NOT NULL REFERENCES saes(id),
    feature_index INTEGER NOT NULL,
    
    -- Logit lens data (stored as JSONB)
    logit_lens_data JSONB,
    
    -- Histogram data
    histogram_data JSONB,
    
    -- Aggregated tokens
    top_tokens JSONB,
    
    -- Computation metadata
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    computation_version VARCHAR(50),
    
    UNIQUE(sae_id, feature_index)
);

CREATE INDEX idx_feature_dashboard_sae ON feature_dashboard_data(sae_id);

-- Export Jobs
CREATE TABLE neuronpedia_export_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sae_id UUID NOT NULL REFERENCES saes(id),
    user_id UUID NOT NULL REFERENCES users(id),
    
    -- Job configuration
    config JSONB NOT NULL,
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'pending',
    progress FLOAT DEFAULT 0.0,
    current_stage VARCHAR(100),
    
    -- Results
    output_path TEXT,
    file_size_bytes BIGINT,
    feature_count INTEGER,
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Error handling
    error_message TEXT,
    error_details JSONB
);

CREATE INDEX idx_export_jobs_sae ON neuronpedia_export_jobs(sae_id);
CREATE INDEX idx_export_jobs_user ON neuronpedia_export_jobs(user_id);
CREATE INDEX idx_export_jobs_status ON neuronpedia_export_jobs(status);
```

---

## 7. User Interface Specifications

### 7.1 Export Dialog

```
┌─────────────────────────────────────────────────────────────────┐
│  Export to Neuronpedia                                      [X] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SAE: gemma-2-2b-layer12-res-16k                               │
│  Features: 16,384 | Labeled: 12,847 (78%)                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Export Format                                            │   │
│  │ ○ Neuronpedia JSON only                                  │   │
│  │ ● Neuronpedia JSON + SAELens (recommended)               │   │
│  │ ○ SAELens format only                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Feature Selection                                        │   │
│  │ ● All features (16,384)                                  │   │
│  │ ○ Labeled features only (12,847)                         │   │
│  │ ○ Custom selection...                                    │   │
│  │ ○ By activation frequency: [____] to [____]              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Dashboard Data                                           │   │
│  │ ☑ Top activating examples (20 per feature)              │   │
│  │ ☑ Logit lens data (top 20 positive/negative)            │   │
│  │ ☑ Activation histograms                                  │   │
│  │ ☑ Token aggregations                                     │   │
│  │ ☑ Feature explanations                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Advanced Options                                    [▼]  │   │
│  │ • Examples per feature: [20  ▼]                          │   │
│  │ • Logit lens K: [20  ▼]                                  │   │
│  │ • Histogram bins: [50  ▼]                                │   │
│  │ • Compression: [gzip ▼]                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Estimated size: ~350 MB | Time: ~25 minutes                   │
│                                                                 │
│  ⚠️ Missing data will be computed during export.               │
│     Logit lens: 16,384 features | Histograms: 16,384 features  │
│                                                                 │
│                              [Cancel]  [Validate]  [Export]    │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Export Progress

```
┌─────────────────────────────────────────────────────────────────┐
│  Exporting to Neuronpedia                                   [X] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SAE: gemma-2-2b-layer12-res-16k                               │
│                                                                 │
│  Overall Progress                                               │
│  [████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░] 42%             │
│                                                                 │
│  Current Stage: Computing logit lens data                       │
│  [████████████████████████░░░░░░░░░░░░░░░░░░░] 58%             │
│  Features: 9,503 / 16,384                                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Stage                          Status        Time       │   │
│  │ ─────────────────────────────────────────────────────── │   │
│  │ ✓ Validating SAE data          Complete      0:12       │   │
│  │ ✓ Loading activation examples   Complete      1:45       │   │
│  │ ● Computing logit lens data     In Progress   8:23       │   │
│  │ ○ Computing histograms          Pending       ~5:00      │   │
│  │ ○ Aggregating tokens            Pending       ~2:00      │   │
│  │ ○ Generating JSON files         Pending       ~3:00      │   │
│  │ ○ Creating archive              Pending       ~1:00      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Elapsed: 10:20 | Remaining: ~11:00                            │
│                                                                 │
│                         [Cancel Export]  [Run in Background]    │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Export Complete

```
┌─────────────────────────────────────────────────────────────────┐
│  ✓ Export Complete                                          [X] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SAE: gemma-2-2b-layer12-res-16k                               │
│                                                                 │
│  Export Summary                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Features exported:     16,384                            │   │
│  │ Archive size:          347 MB                            │   │
│  │ Export time:           21:45                             │   │
│  │ Format:                Neuronpedia JSON + SAELens        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Files included:                                                │
│  • metadata.json                                                │
│  • features/ (16,384 JSON files)                               │
│  • explanations/explanations.json                              │
│  • saelens/cfg.json                                            │
│  • saelens/sae_weights.safetensors                             │
│  • README.md (upload instructions)                             │
│                                                                 │
│  Next Steps:                                                    │
│  1. Download the export archive                                 │
│  2. Fill out Neuronpedia upload form                           │
│  3. Coordinate with Neuronpedia team for import                │
│                                                                 │
│  [Download Archive]  [Copy HuggingFace Upload Command]          │
│                                                                 │
│  [Open Neuronpedia Upload Form]  [View Upload Instructions]     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementation Plan

### Phase 1: Core Data Computation (2-3 weeks)
**Goal**: Implement missing dashboard data computations

**Tasks**:
1. **Logit Lens Computer**
   - Implement batch logit lens computation
   - Add to extraction pipeline as optional step
   - Create database storage for results
   - Add caching layer

2. **Histogram Generator**
   - Implement activation histogram computation
   - Support configurable binning strategies
   - Handle edge cases (sparse features, dead features)
   - Store in database

3. **Token Aggregator**
   - Implement cross-example token aggregation
   - Support multiple ranking metrics
   - Handle subword token edge cases
   - Store top-K aggregations

**Deliverables**:
- `LogitLensComputer` class with GPU optimization
- `HistogramGenerator` class
- `TokenAggregator` class
- Database migrations for new tables
- Unit tests for all components

### Phase 2: Export Service (2-3 weeks)
**Goal**: Build export orchestration and serialization

**Tasks**:
1. **Export Orchestrator**
   - Job queue management
   - Progress tracking
   - Error handling and recovery
   - Background processing

2. **JSON Serializer**
   - Neuronpedia schema implementation
   - Validation against schema
   - Efficient serialization
   - Encoding edge cases

3. **File Packager**
   - Directory structure generation
   - Compression (gzip/zip)
   - README generation
   - Checksum verification

**Deliverables**:
- `NeuronpediaExportService` class
- `NeuronpediaSerializer` class
- `ExportPackager` class
- API endpoints for export initiation/status/download
- Background job infrastructure

### Phase 3: User Interface (1-2 weeks)
**Goal**: Build export UI in frontend

**Tasks**:
1. **Export Dialog**
   - Configuration form
   - Validation feedback
   - Estimate display
   - Format selection

2. **Progress Tracking**
   - Real-time progress updates
   - Stage breakdown
   - Time estimates
   - Cancel/background options

3. **Completion Flow**
   - Download functionality
   - Next steps guidance
   - Neuronpedia form link
   - Copy commands

**Deliverables**:
- `ExportToNeuronpedia` React component
- `ExportProgress` component
- `ExportComplete` component
- API client functions

### Phase 4: Integration & Polish (1-2 weeks)
**Goal**: End-to-end testing and refinement

**Tasks**:
1. **Integration Testing**
   - Full export workflow tests
   - Neuronpedia import validation
   - Performance benchmarking
   - Edge case handling

2. **Documentation**
   - User guide for export
   - Neuronpedia upload guide
   - Troubleshooting guide
   - API documentation

3. **Polish**
   - Error message refinement
   - Progress accuracy improvement
   - Performance optimization
   - UX refinement

**Deliverables**:
- Integration test suite
- User documentation
- Performance benchmarks
- Bug fixes and polish

---

## 9. Success Metrics

### 9.1 Technical Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Export speed (16k features, GPU) | < 30 min | Automated benchmark |
| Export success rate | > 99% | Export job completion rate |
| Data validation pass rate | 100% | Schema validation results |
| Memory usage (16k features) | < 16 GB | Peak memory monitoring |

### 9.2 Quality Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Neuronpedia import success | 100% | Manual verification |
| Feature dashboard completeness | 100% | Required field coverage |
| Explanation preservation | 100% | Label export accuracy |

### 9.3 Adoption Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Export feature usage | > 30% of SAE creators | Analytics |
| Successful Neuronpedia uploads | > 10 in 6 months | Community tracking |
| User satisfaction | > 4/5 rating | User surveys |

---

## 10. Risks & Mitigations

### 10.1 Technical Risks

**Risk 1: Neuronpedia Schema Changes**
- **Impact**: High - exports may become incompatible
- **Likelihood**: Medium - Neuronpedia is actively developed
- **Mitigation**: 
  - Version export format
  - Monitor Neuronpedia releases
  - Maintain schema validation
  - Design for extensibility

**Risk 2: Performance at Scale**
- **Impact**: Medium - large SAEs may be slow to export
- **Likelihood**: Medium - 100k+ feature SAEs exist
- **Mitigation**:
  - Streaming export for large SAEs
  - Parallel computation
  - Incremental export support
  - Progress estimates

**Risk 3: Data Quality Issues**
- **Impact**: High - poor exports reflect badly on miStudio
- **Likelihood**: Low - validation should catch issues
- **Mitigation**:
  - Comprehensive validation
  - Preview functionality
  - Automated testing
  - User feedback loop

### 10.2 Process Risks

**Risk 4: Manual Neuronpedia Upload**
- **Impact**: Medium - friction in user workflow
- **Likelihood**: High - no automated upload API exists
- **Mitigation**:
  - Clear documentation
  - Pre-fill upload forms
  - Track Neuronpedia API development
  - Provide excellent README

**Risk 5: Community Coordination**
- **Impact**: Low - may slow adoption
- **Likelihood**: Medium - requires Neuronpedia team cooperation
- **Mitigation**:
  - Early engagement with Neuronpedia team
  - Join OSMI Slack
  - Contribute to ecosystem
  - Build relationship

---

## 11. Dependencies

### 11.1 Internal Dependencies
- SAE training module (complete)
- Feature extraction pipeline (complete)
- Labeling service (complete)
- HuggingFace upload (complete)
- Background job infrastructure (may need enhancement)

### 11.2 External Dependencies
- Neuronpedia schema documentation (available)
- TransformerLens model support (available)
- Neuronpedia upload process (manual, documented)
- Community SAELens format (stable)

### 11.3 New Dependencies
```python
# No new major dependencies required
# Existing dependencies sufficient:
# - torch (GPU computation)
# - safetensors (weight serialization)
# - numpy (numerical operations)
# - gzip/zipfile (compression - standard library)
```

---

## 12. Appendices

### Appendix A: Neuronpedia Feature Schema (Full)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["feature_index", "activations", "statistics"],
  "properties": {
    "feature_index": {
      "type": "integer",
      "minimum": 0
    },
    "activations": {
      "type": "array",
      "minItems": 10,
      "items": {
        "type": "object",
        "required": ["tokens", "values"],
        "properties": {
          "tokens": {
            "type": "array",
            "items": {"type": "string"}
          },
          "values": {
            "type": "array",
            "items": {"type": "number"}
          },
          "max_value": {"type": "number"},
          "max_token_index": {"type": "integer"}
        }
      }
    },
    "logits": {
      "type": "object",
      "properties": {
        "top_positive": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "token": {"type": "string"},
              "value": {"type": "number"}
            }
          }
        },
        "top_negative": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "token": {"type": "string"},
              "value": {"type": "number"}
            }
          }
        }
      }
    },
    "histogram": {
      "type": "object",
      "properties": {
        "bin_edges": {
          "type": "array",
          "items": {"type": "number"}
        },
        "counts": {
          "type": "array",
          "items": {"type": "integer"}
        }
      }
    },
    "statistics": {
      "type": "object",
      "required": ["activation_frequency"],
      "properties": {
        "activation_frequency": {"type": "number"},
        "max_activation": {"type": "number"},
        "mean_activation": {"type": "number"},
        "std_activation": {"type": "number"}
      }
    },
    "top_tokens": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "token": {"type": "string"},
          "total_activation": {"type": "number"},
          "count": {"type": "integer"}
        }
      }
    }
  }
}
```

### Appendix B: SAELens cfg.json Schema

```json
{
  "architecture": "standard",
  "d_in": 2048,
  "d_sae": 16384,
  "dtype": "float32",
  "device": "cuda",
  "model_name": "gemma-2-2b",
  "hook_name": "blocks.12.hook_resid_post",
  "hook_layer": 12,
  "hook_head_index": null,
  "activation_fn_str": "relu",
  "activation_fn_kwargs": {},
  "apply_b_dec_to_input": true,
  "finetuning_scaling_factor": false,
  "sae_lens_training_version": null,
  "prepend_bos": true,
  "dataset_path": "monology/pile-uncopyrighted",
  "dataset_trust_remote_code": true,
  "context_size": 128,
  "normalize_activations": "none",
  "model_from_pretrained_kwargs": {}
}
```

### Appendix C: TransformerLens Hook Point Reference

| Hook Type | Format | Example |
|-----------|--------|---------|
| Residual Pre | `blocks.{layer}.hook_resid_pre` | `blocks.12.hook_resid_pre` |
| Residual Post | `blocks.{layer}.hook_resid_post` | `blocks.12.hook_resid_post` |
| MLP Out | `blocks.{layer}.hook_mlp_out` | `blocks.12.hook_mlp_out` |
| Attention Z | `blocks.{layer}.attn.hook_z` | `blocks.12.attn.hook_z` |
| Attention Q | `blocks.{layer}.attn.hook_q` | `blocks.12.attn.hook_q` |
| Attention K | `blocks.{layer}.attn.hook_k` | `blocks.12.attn.hook_k` |
| Attention V | `blocks.{layer}.attn.hook_v` | `blocks.12.attn.hook_v` |

### Appendix D: Model ID Mapping

| HuggingFace Model | TransformerLens ID |
|-------------------|-------------------|
| `openai-community/gpt2` | `gpt2-small` |
| `openai-community/gpt2-medium` | `gpt2-medium` |
| `openai-community/gpt2-large` | `gpt2-large` |
| `openai-community/gpt2-xl` | `gpt2-xl` |
| `google/gemma-2b` | `gemma-2b` |
| `google/gemma-2-2b` | `gemma-2-2b` |
| `google/gemma-2-2b-it` | `gemma-2-2b-it` |
| `google/gemma-2-9b` | `gemma-2-9b` |
| `google/gemma-2-9b-it` | `gemma-2-9b-it` |
| `meta-llama/Llama-3.1-8B` | `llama-3.1-8b` |
| `EleutherAI/pythia-70m` | `pythia-70m` |
| `EleutherAI/pythia-160m` | `pythia-160m` |
| `EleutherAI/pythia-410m` | `pythia-410m` |

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Author**: Product Team  
**Status**: Ready for Implementation
