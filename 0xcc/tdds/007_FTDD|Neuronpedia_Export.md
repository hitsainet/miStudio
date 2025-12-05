# Technical Design Document: Neuronpedia Export

**Document ID:** 007_FTDD|Neuronpedia_Export
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Implemented
**Related PRD:** [007_FPRD|Neuronpedia_Export](../prds/007_FPRD|Neuronpedia_Export.md)

---

## 1. System Architecture

### 1.1 Export Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│                     Export Pipeline                              │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │  SAE     │──→│  Model   │──→│ Features │──→│   Package    │ │
│  │  Config  │   │ (logit   │   │ (data)   │   │   Builder    │ │
│  │          │   │  lens)   │   │          │   │              │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘ │
│                                                      │          │
│                                                      ▼          │
│                                            ┌──────────────────┐ │
│                                            │   ZIP Archive    │ │
│                                            │   export.zip     │ │
│                                            └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                    Backend                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                NeuronpediaExportService                     ││
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐││
│  │  │ start_export() │  │ compute_data() │  │ build_package()│││
│  │  └────────────────┘  └────────────────┘  └────────────────┘││
│  └─────────────────────────────┬───────────────────────────────┘│
│                                │                                │
│  ┌─────────────────────────────┼───────────────────────────────┐│
│  │                    Sub-Services                             ││
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐││
│  │  │LogitLensService│  │HistogramService│  │TokenAggregator │││
│  │  └────────────────┘  └────────────────┘  └────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Celery Worker: export_task                     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Database Schema

### 2.1 Export Job Table
```sql
CREATE TABLE neuronpedia_export_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_id UUID REFERENCES trainings(id),
    external_sae_id UUID REFERENCES external_saes(id),
    config JSONB NOT NULL,           -- Export configuration
    status VARCHAR(50) DEFAULT 'pending',
    progress FLOAT DEFAULT 0,
    current_stage VARCHAR(100),
    feature_count INTEGER,
    output_path VARCHAR(500),
    file_size_bytes BIGINT,
    error_message TEXT,
    celery_task_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    CONSTRAINT export_jobs_sae_check CHECK (
        (training_id IS NOT NULL) OR (external_sae_id IS NOT NULL)
    )
);
```

### 2.2 Feature Dashboard Data Table
```sql
CREATE TABLE feature_dashboard_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_id UUID REFERENCES features(id) ON DELETE CASCADE,
    logit_lens_data JSONB,          -- {top_positive, top_negative}
    histogram_data JSONB,            -- {bin_edges, counts}
    top_tokens JSONB,                -- [{token, count, mean_activation}]
    computed_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(feature_id)
);
```

---

## 3. Export Package Structure

### 3.1 Complete Package
```
export.zip/
├── metadata.json                    # SAE configuration
├── README.md                        # Documentation
├── features/                        # Individual feature files
│   ├── 0.json
│   ├── 1.json
│   └── ...
├── explanations/
│   └── explanations.json           # All feature labels
└── saelens/
    ├── cfg.json                    # SAELens-compatible config
    └── sae_weights.safetensors     # Model weights
```

### 3.2 metadata.json Schema
```json
{
  "sae_id": "uuid-string",
  "sae_name": "gemma-2b-layer12-8x",
  "model_name": "google/gemma-2-2b",
  "layer": 12,
  "hook_name": "blocks.12.hook_resid_post",
  "d_in": 2304,
  "d_sae": 18432,
  "architecture": "standard",
  "export_date": "2025-12-05T00:00:00Z",
  "export_version": "1.0",
  "feature_count": 993,
  "includes": {
    "logit_lens": true,
    "histograms": true,
    "top_tokens": true,
    "explanations": true,
    "saelens_weights": true
  },
  "generation_info": {
    "tool": "miStudio",
    "version": "1.0.0"
  }
}
```

### 3.3 Feature JSON Schema
```json
{
  "feature_index": 42,
  "label": "expressions of romantic love",
  "category": "Emotion > Positive > Love",
  "statistics": {
    "activation_frequency": 0.023,
    "max_activation": 8.4,
    "mean_activation": 2.1,
    "interpretability_score": 0.85
  },
  "logit_lens": {
    "top_positive": [
      {"token": "love", "token_id": 1234, "value": 0.85},
      {"token": "heart", "token_id": 5678, "value": 0.72}
    ],
    "top_negative": [
      {"token": "hate", "token_id": 9012, "value": -0.45}
    ]
  },
  "histogram": {
    "bin_edges": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "counts": [1000, 500, 250, 125, 62, 31]
  },
  "top_tokens": [
    {"token": "love", "count": 45, "mean_activation": 3.2},
    {"token": "heart", "count": 32, "mean_activation": 2.8}
  ],
  "activations": [
    {
      "token": "love",
      "token_id": 1234,
      "activation_value": 4.52,
      "context_before": "I really ",
      "context_after": " you so much",
      "position": 5,
      "sample_index": 123
    }
  ]
}
```

---

## 4. Service Implementation

### 4.1 NeuronpediaExportService
```python
class NeuronpediaExportService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.logit_lens = LogitLensService()
        self.histogram = HistogramService()
        self.token_aggregator = TokenAggregatorService()

    async def start_export(
        self,
        sae_id: UUID,
        config: ExportConfig
    ) -> NeuronpediaExportJob:
        """Start export job."""
        job = NeuronpediaExportJob(
            external_sae_id=sae_id,
            config=config.dict(),
            status='pending'
        )
        self.db.add(job)
        await self.db.commit()

        # Queue Celery task
        task = export_neuronpedia_task.delay(str(job.id))
        job.celery_task_id = task.id
        await self.db.commit()

        return job

    async def build_export_package(
        self,
        job: NeuronpediaExportJob
    ) -> Path:
        """Build complete export package."""
        sae = await self._get_sae(job)
        features = await self._get_features(sae)
        config = ExportConfig(**job.config)

        # Create temp directory
        export_dir = Path(tempfile.mkdtemp())

        # Build metadata
        await self._write_metadata(export_dir, sae, config, len(features))

        # Build feature files
        total = len(features)
        for i, feature in enumerate(features):
            feature_data = await self._build_feature_data(feature, config)
            await self._write_feature_json(export_dir / 'features', feature.neuron_index, feature_data)

            # Update progress
            progress = (i + 1) / total * 0.8  # 80% for features
            emit_export_progress(job.id, progress, f"Processing feature {i+1}/{total}")

        # Build explanations
        if config.include_explanations:
            await self._write_explanations(export_dir, features)

        # Copy SAELens weights
        if config.include_saelens_weights:
            await self._copy_saelens_weights(export_dir, sae)

        # Create ZIP archive
        archive_path = await self._create_zip(export_dir, job.id)

        return archive_path

    async def _build_feature_data(
        self,
        feature: Feature,
        config: ExportConfig
    ) -> dict:
        """Build data for single feature."""
        data = {
            "feature_index": feature.neuron_index,
            "label": feature.label,
            "category": feature.category,
            "statistics": feature.statistics
        }

        # Add logit lens if configured
        if config.include_logit_lens:
            data["logit_lens"] = await self.logit_lens.compute(
                feature,
                top_k=config.logit_lens_top_k
            )

        # Add histogram if configured
        if config.include_histograms:
            data["histogram"] = await self.histogram.compute(
                feature,
                num_bins=config.histogram_bins
            )

        # Add top tokens if configured
        if config.include_top_tokens:
            data["top_tokens"] = await self.token_aggregator.aggregate(
                feature,
                top_k=config.top_tokens_k
            )

        # Add activation examples
        data["activations"] = await self._get_activation_examples(feature)

        return data
```

### 4.2 LogitLensService
```python
class LogitLensService:
    async def compute(
        self,
        feature: Feature,
        top_k: int = 20
    ) -> dict:
        """Compute promoted/suppressed tokens for feature."""
        # Load SAE and model
        sae = await self._get_sae(feature)
        model = await self._get_model(sae)

        # Get feature decoder direction
        feature_direction = sae.W_dec[:, feature.neuron_index]

        # Project through unembedding
        # shape: [hidden_dim] @ [hidden_dim, vocab_size] = [vocab_size]
        logits = feature_direction @ model.lm_head.weight.T

        # Get top positive (promoted)
        top_positive_indices = torch.topk(logits, top_k).indices
        top_positive = [
            {
                "token": tokenizer.decode([idx]),
                "token_id": idx.item(),
                "value": logits[idx].item()
            }
            for idx in top_positive_indices
        ]

        # Get top negative (suppressed)
        top_negative_indices = torch.topk(-logits, top_k).indices
        top_negative = [
            {
                "token": tokenizer.decode([idx]),
                "token_id": idx.item(),
                "value": logits[idx].item()
            }
            for idx in top_negative_indices
        ]

        return {
            "top_positive": top_positive,
            "top_negative": top_negative
        }
```

### 4.3 HistogramService
```python
class HistogramService:
    async def compute(
        self,
        feature: Feature,
        num_bins: int = 50
    ) -> dict:
        """Compute activation histogram for feature."""
        # Get all activations for this feature
        activations = await self._get_all_activations(feature)

        if len(activations) == 0:
            return {"bin_edges": [], "counts": []}

        # Compute histogram
        counts, bin_edges = np.histogram(
            activations,
            bins=num_bins,
            range=(0, activations.max())
        )

        return {
            "bin_edges": bin_edges.tolist(),
            "counts": counts.tolist()
        }
```

---

## 5. API Design

### 5.1 Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/neuronpedia/export` | Start export job |
| GET | `/neuronpedia/export/{id}` | Get job status |
| GET | `/neuronpedia/export/{id}/download` | Download archive |
| POST | `/neuronpedia/export/{id}/cancel` | Cancel export |
| GET | `/neuronpedia/exports` | List export history |

### 5.2 Export Config
```python
class ExportConfig(BaseModel):
    feature_selection: Literal['all', 'extracted', 'custom'] = 'all'
    feature_indices: Optional[List[int]] = None  # For 'custom'

    include_logit_lens: bool = True
    logit_lens_top_k: int = 20

    include_histograms: bool = True
    histogram_bins: int = 50

    include_top_tokens: bool = True
    top_tokens_k: int = 50

    include_explanations: bool = True
    include_saelens_weights: bool = True
```

---

## 6. WebSocket Events

### 6.1 Channel: `export/{job_id}`
| Event | Payload |
|-------|---------|
| `progress` | `{progress: float, stage: string, features_processed: int}` |
| `completed` | `{output_path: string, file_size: int}` |
| `failed` | `{error: string}` |

---

## 7. Celery Task

```python
@celery_app.task(bind=True, queue='processing')
def export_neuronpedia_task(self, job_id: str):
    """Export SAE to Neuronpedia format."""
    job = get_export_job(job_id)
    job.status = 'running'
    job.started_at = datetime.now()
    save_job(job)

    try:
        service = NeuronpediaExportService(db)

        # Build package
        archive_path = await service.build_export_package(job)

        # Update job
        job.status = 'completed'
        job.output_path = str(archive_path)
        job.file_size_bytes = archive_path.stat().st_size
        job.completed_at = datetime.now()

        emit_export_completed(job_id, str(archive_path), job.file_size_bytes)

    except Exception as e:
        job.status = 'failed'
        job.error_message = str(e)
        emit_export_failed(job_id, str(e))
        raise

    finally:
        save_job(job)
```

---

## 8. File Storage

```
DATA_DIR/
└── exports/
    └── neuronpedia/
        └── {job_id}/
            ├── export.zip              # Final archive
            └── temp/                   # Working directory (cleaned up)
                ├── metadata.json
                ├── features/
                └── ...
```

---

*Related: [PRD](../prds/007_FPRD|Neuronpedia_Export.md) | [TID](../tids/007_FTID|Neuronpedia_Export.md) | [FTASKS](../tasks/007_FTASKS|Neuronpedia_Export.md)*
