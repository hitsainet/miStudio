# MechInterp Studio (miStudio) - PostgreSQL Database Specification

**Document Type**: Technical Specification
**Last Updated**: 2025-10-05
**Purpose**: Complete PostgreSQL database architecture, schema, and implementation guidance

---

## Table of Contents

1. [PostgreSQL Role in the System](#postgresql-role-in-the-system)
2. [Why PostgreSQL](#why-postgresql)
3. [Database Schema](#database-schema)
4. [Relationships and Foreign Keys](#relationships-and-foreign-keys)
5. [Indexes and Performance Optimization](#indexes-and-performance-optimization)
6. [Table Partitioning](#table-partitioning)
7. [Database Migrations with Alembic](#database-migrations-with-alembic)
8. [Common Queries and Patterns](#common-queries-and-patterns)
9. [Connection Pooling](#connection-pooling)
10. [Backup and Restore](#backup-and-restore)
11. [Configuration for Edge Devices](#configuration-for-edge-devices)
12. [Data Retention Policies](#data-retention-policies)
13. [Security Considerations](#security-considerations)

---

## PostgreSQL Role in the System

PostgreSQL serves as the **primary metadata store** for MechInterp Studio. While large binary data (models, datasets, activations) is stored on local filesystem, PostgreSQL manages:

### 1. **Metadata Management**
- Dataset metadata (name, source, size, status, file paths)
- Model metadata (architecture, parameters, quantization, file paths)
- Training job configurations and state
- Feature metadata and interpretability scores
- Checkpoint metadata and file references

### 2. **Relational Integrity**
- Foreign key relationships between trainings → models → datasets
- Feature → training relationships
- Feature activations → features → dataset samples
- Checkpoint → training relationships

### 3. **Transactional Operations**
- Atomic status updates for training jobs
- Consistent state management across distributed workers
- ACID compliance for critical operations (checkpoint saves, status transitions)

### 4. **Time-Series Data**
- Training metrics stored as time-series (step, loss, sparsity, timestamp)
- Efficient queries for metric charts and progress tracking
- Historical analysis of training runs

### 5. **Search and Filtering**
- Full-text search for datasets and models
- Complex filtering queries (by status, date range, model architecture)
- Feature search by activation frequency or interpretability score

### 6. **Application State**
- User preferences (UI state, steering presets)
- System configuration (rate limits, resource allocations)
- Job queue state coordination

---

## Why PostgreSQL

### Technical Justification

1. **JSONB Support**
   - Store hyperparameters as structured JSON
   - Feature activation data (tokens, activations arrays)
   - Flexible schema evolution without migrations
   - Native indexing and querying of JSON fields

2. **Full-Text Search**
   - Built-in `tsvector` and GIN indexes
   - Search datasets and models without external search engine
   - Low overhead for edge devices

3. **Reliability**
   - ACID compliance ensures data consistency
   - Write-ahead logging (WAL) for crash recovery
   - Battle-tested in production environments

4. **Performance**
   - Excellent read performance for metadata queries
   - Efficient B-tree and GIN indexes
   - Table partitioning for large tables (feature_activations)

5. **Edge-Friendly**
   - Small footprint (~30MB RAM minimum)
   - Docker image: postgres:15-alpine (~80MB compressed)
   - Efficient on ARM64 (Jetson Orin Nano)

6. **Developer Experience**
   - Rich ecosystem (SQLAlchemy ORM, Alembic migrations)
   - Excellent monitoring tools (pg_stat_statements, pg_top)
   - Standard SQL with powerful extensions

### Alternatives Considered

| Database | Why Not Selected |
|----------|------------------|
| SQLite | Limited concurrency, no full JSONB support, challenging for multi-worker scenarios |
| MySQL | Less mature JSONB support, inferior JSON indexing performance |
| MongoDB | Overkill for structured metadata, no ACID guarantees, larger memory footprint |

---

## Database Schema

### Overview

The database consists of **10 core tables**:
- `datasets` - Dataset metadata
- `models` - Model metadata
- `trainings` - Training job configurations and state
- `training_metrics` - Time-series metrics during training
- `checkpoints` - Training checkpoint metadata
- `features` - Discovered interpretable features
- `feature_activations` - Per-feature activation examples (large table)
- `training_templates` - Reusable training configuration presets
- `extraction_templates` - Reusable feature extraction configuration presets
- `steering_presets` - Saved steering configurations

**Note**: Single-user system - no `users` table or user authentication.

---

### Table: `datasets`

Stores metadata for HuggingFace and local datasets.

```sql
CREATE TABLE datasets (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'ds_abc123' or HF repo ID
    name VARCHAR(500) NOT NULL,
    source VARCHAR(50) NOT NULL,  -- 'HuggingFace', 'Local', 'Custom'
    description TEXT,
    size_bytes BIGINT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'downloading',
        -- Values: 'downloading', 'ingesting', 'tokenizing', 'ready', 'error'
    progress FLOAT,  -- 0-100, for UI progress bars
    error_message TEXT,
    file_path VARCHAR(1000),  -- Local path: /data/datasets/raw/{id}/
    tokenized_path VARCHAR(1000),  -- Local path: /data/datasets/tokenized/{id}/

    -- Statistics (populated after ingestion)
    num_samples INTEGER,
    num_tokens BIGINT,
    vocab_size INTEGER,
    avg_sequence_length FLOAT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    CONSTRAINT datasets_status_check CHECK (status IN
        ('downloading', 'ingesting', 'tokenizing', 'ready', 'error'))
);

CREATE INDEX idx_datasets_status ON datasets(status);
CREATE INDEX idx_datasets_created_at ON datasets(created_at DESC);
CREATE INDEX idx_datasets_source ON datasets(source);

-- Full-text search index
CREATE INDEX idx_datasets_search ON datasets
    USING GIN(to_tsvector('english', name || ' ' || COALESCE(description, '')));
```

**Storage Estimate**: ~1KB per dataset, 100 datasets = 100KB

---

### Table: `models`

Stores metadata for downloaded and quantized models.

```sql
CREATE TABLE models (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'm_abc123' or HF repo ID
    name VARCHAR(500) NOT NULL,
    architecture VARCHAR(100) NOT NULL,  -- 'llama', 'gpt2', 'pythia', 'phi', etc.
    params_count BIGINT NOT NULL,  -- Total parameter count
    quantization VARCHAR(50) NOT NULL,  -- 'Q4', 'Q8', 'FP16', 'FP32', 'INT8'
    status VARCHAR(50) NOT NULL DEFAULT 'downloading',
        -- Values: 'downloading', 'loading', 'quantizing', 'ready', 'error'
    progress FLOAT,  -- 0-100
    error_message TEXT,
    file_path VARCHAR(1000),  -- Local path: /data/models/raw/{id}/
    quantized_path VARCHAR(1000),  -- Local path: /data/models/quantized/{id}/

    -- Architecture details (JSONB for flexibility)
    architecture_config JSONB,  -- { "num_layers": 12, "hidden_size": 768, ... }

    -- Resource requirements
    memory_required_bytes BIGINT,
    disk_size_bytes BIGINT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT models_status_check CHECK (status IN
        ('downloading', 'loading', 'quantizing', 'ready', 'error'))
);

CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_models_architecture ON models(architecture);
CREATE INDEX idx_models_created_at ON models(created_at DESC);

-- Full-text search index
CREATE INDEX idx_models_search ON models
    USING GIN(to_tsvector('english', name || ' ' || architecture));
```

**Storage Estimate**: ~2KB per model, 50 models = 100KB

---

### Table: `trainings`

Stores training job configurations, state, and progress.

```sql
CREATE TABLE trainings (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'tr_abc123'

    -- Foreign keys
    model_id VARCHAR(255) NOT NULL REFERENCES models(id) ON DELETE RESTRICT,
    dataset_id VARCHAR(255) NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,

    -- Configuration
    encoder_type VARCHAR(50) NOT NULL,  -- 'sparse', 'skip', 'transcoder'
    hyperparameters JSONB NOT NULL,  -- See Hyperparameters interface

    -- State
    status VARCHAR(50) NOT NULL DEFAULT 'queued',
        -- Values: 'queued', 'initializing', 'training', 'paused',
        --         'stopped', 'completed', 'failed', 'error'

    -- Progress tracking
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER NOT NULL,
    progress FLOAT GENERATED ALWAYS AS (
        CASE WHEN total_steps > 0
        THEN (current_step::FLOAT / total_steps * 100)
        ELSE 0 END
    ) STORED,

    -- Latest metrics (denormalized for quick access)
    latest_loss FLOAT,
    latest_sparsity FLOAT,
    latest_dead_neurons INTEGER,

    -- Error handling
    error_message TEXT,
    error_code VARCHAR(100),
    retry_count INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    paused_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Resource allocation
    gpu_id INTEGER,
    worker_id VARCHAR(100),

    CONSTRAINT trainings_status_check CHECK (status IN
        ('queued', 'initializing', 'training', 'paused', 'stopped',
         'completed', 'failed', 'error'))
);

CREATE INDEX idx_trainings_status ON trainings(status);
CREATE INDEX idx_trainings_model_id ON trainings(model_id);
CREATE INDEX idx_trainings_dataset_id ON trainings(dataset_id);
CREATE INDEX idx_trainings_created_at ON trainings(created_at DESC);
CREATE INDEX idx_trainings_completed_at ON trainings(completed_at DESC NULLS LAST);

-- Composite index for filtering active trainings
CREATE INDEX idx_trainings_active ON trainings(status, updated_at)
    WHERE status IN ('queued', 'initializing', 'training', 'paused');
```

**Storage Estimate**: ~3KB per training, 1000 trainings = 3MB

**Important JSONB Field**: `hyperparameters`

```json
{
  "learningRate": 0.001,
  "batchSize": 256,
  "l1Coefficient": 0.0001,
  "expansionFactor": 8,
  "trainingSteps": 10000,
  "optimizer": "AdamW",
  "lrSchedule": "cosine",
  "ghostGradPenalty": true
}
```

---

### Table: `training_metrics`

Time-series metrics collected during training.

```sql
CREATE TABLE training_metrics (
    id BIGSERIAL PRIMARY KEY,
    training_id VARCHAR(255) NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,

    -- Metrics
    step INTEGER NOT NULL,
    loss FLOAT NOT NULL,
    sparsity FLOAT,  -- L0 sparsity (average active features)
    reconstruction_error FLOAT,
    dead_neurons INTEGER,
    explained_variance FLOAT,

    -- Learning rate (for debugging)
    learning_rate FLOAT,

    -- Timestamp
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Composite unique constraint (one metric per step per training)
    CONSTRAINT training_metrics_unique_step UNIQUE (training_id, step)
);

-- Critical index for time-series queries
CREATE INDEX idx_training_metrics_training_step ON training_metrics(training_id, step);
CREATE INDEX idx_training_metrics_timestamp ON training_metrics(training_id, timestamp);
```

**Storage Estimate**: ~50 bytes per metric
- 10,000 training steps × 50 bytes = 500KB per training
- 1,000 trainings = 500MB

**Optimization**: Consider retention policy - delete metrics older than 90 days for completed trainings.

---

### Table: `checkpoints`

Metadata for training checkpoints (actual weights stored on filesystem).

```sql
CREATE TABLE checkpoints (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'cp_abc123'
    training_id VARCHAR(255) NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,

    -- Checkpoint details
    step INTEGER NOT NULL,
    loss FLOAT,
    sparsity FLOAT,

    -- File storage
    storage_path VARCHAR(1000) NOT NULL,  -- /data/trainings/{training_id}/checkpoints/{id}.pt
    file_size_bytes BIGINT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Composite unique constraint
    CONSTRAINT checkpoints_unique_step UNIQUE (training_id, step)
);

CREATE INDEX idx_checkpoints_training_id ON checkpoints(training_id);
CREATE INDEX idx_checkpoints_step ON checkpoints(training_id, step DESC);
CREATE INDEX idx_checkpoints_created_at ON checkpoints(created_at DESC);
```

**Storage Estimate**: ~500 bytes per checkpoint
- 10 checkpoints per training × 500 bytes = 5KB per training
- 1,000 trainings = 5MB

---

### Table: `features`

Discovered interpretable features from trained SAEs.

```sql
CREATE TABLE features (
    id BIGSERIAL PRIMARY KEY,
    training_id VARCHAR(255) NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,

    -- Feature identification
    neuron_index INTEGER NOT NULL,  -- Index in SAE layer
    layer INTEGER,  -- Transformer layer this feature comes from

    -- Metadata
    name VARCHAR(500),  -- User-assigned or auto-generated name
    description TEXT,  -- User-provided or LLM-generated description

    -- Statistics
    activation_frequency FLOAT NOT NULL,  -- 0-1, how often this feature activates
    interpretability_score FLOAT,  -- 0-1, automated interpretability score
    max_activation_value FLOAT,  -- Maximum observed activation

    -- User flags
    is_favorite BOOLEAN DEFAULT FALSE,
    is_hidden BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Composite unique constraint
    CONSTRAINT features_unique_neuron UNIQUE (training_id, neuron_index, layer)
);

CREATE INDEX idx_features_training_id ON features(training_id);
CREATE INDEX idx_features_activation_freq ON features(training_id, activation_frequency DESC);
CREATE INDEX idx_features_interpretability ON features(training_id, interpretability_score DESC NULLS LAST);
CREATE INDEX idx_features_favorite ON features(is_favorite) WHERE is_favorite = TRUE;
CREATE INDEX idx_features_layer ON features(layer);

-- Full-text search on feature names and descriptions
CREATE INDEX idx_features_search ON features
    USING GIN(to_tsvector('english',
        COALESCE(name, '') || ' ' || COALESCE(description, '')));
```

**Storage Estimate**: ~1KB per feature
- 16,384 features (typical SAE size) × 1KB = 16MB per training
- 100 trainings = 1.6GB

---

### Table: `feature_activations`

**Large table** storing max-activating examples for each feature.

```sql
CREATE TABLE feature_activations (
    id BIGSERIAL PRIMARY KEY,
    feature_id BIGINT NOT NULL REFERENCES features(id) ON DELETE CASCADE,

    -- Dataset sample reference
    dataset_sample_id VARCHAR(255),  -- Reference to specific sample in dataset
    sample_index INTEGER,  -- Index in dataset

    -- Token-level data (stored as JSONB)
    tokens JSONB NOT NULL,  -- ["The", "quick", "brown", "fox", ...]
    activations JSONB NOT NULL,  -- [0.0, 0.2, 0.8, 0.3, ...]

    -- Max activation in this example
    max_activation FLOAT NOT NULL,

    -- Context
    context_before TEXT,  -- Text before the max-activating token
    context_after TEXT,   -- Text after the max-activating token

    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Critical: Partition this table by feature_id ranges (see Partitioning section)

-- Indexes (on each partition)
CREATE INDEX idx_feature_activations_feature ON feature_activations(feature_id);
CREATE INDEX idx_feature_activations_max_activation
    ON feature_activations(feature_id, max_activation DESC);
```

**Storage Estimate**: ~2KB per activation example
- 100 examples per feature × 2KB = 200KB per feature
- 16,384 features × 200KB = 3.2GB per training
- **This is the largest table** - requires partitioning

---

### Table: `training_templates`

Reusable training configuration templates (presets) for quick setup.

```sql
CREATE TABLE training_templates (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'tmpl_abc123'

    -- Template details
    name VARCHAR(500) NOT NULL,
    description TEXT,

    -- Optional model/dataset references (null = generic template)
    model_id VARCHAR(255) REFERENCES models(id) ON DELETE SET NULL,
    dataset_id VARCHAR(255) REFERENCES datasets(id) ON DELETE SET NULL,

    -- Training configuration
    encoder_type VARCHAR(50) NOT NULL,  -- 'sparse', 'skip', 'transcoder'
    hyperparameters JSONB NOT NULL,  -- Complete hyperparameter set

    -- User preferences
    is_favorite BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT training_templates_encoder_type_check CHECK (encoder_type IN
        ('sparse', 'skip', 'transcoder'))
);

CREATE INDEX idx_training_templates_is_favorite ON training_templates(is_favorite)
    WHERE is_favorite = TRUE;
CREATE INDEX idx_training_templates_created_at ON training_templates(created_at DESC);
CREATE INDEX idx_training_templates_model_id ON training_templates(model_id)
    WHERE model_id IS NOT NULL;
CREATE INDEX idx_training_templates_dataset_id ON training_templates(dataset_id)
    WHERE dataset_id IS NOT NULL;
```

**Storage Estimate**: ~2KB per template, 50 templates = 100KB

**Important JSONB Field**: `hyperparameters`

```json
{
  "learningRate": 0.001,
  "batchSize": 256,
  "l1Coefficient": 0.001,
  "expansionFactor": 8,
  "trainingSteps": 10000,
  "optimizer": "AdamW",
  "lrSchedule": "cosine",
  "ghostGradPenalty": true
}
```

**Use Cases**:
- **Generic Templates**: model_id = NULL, dataset_id = NULL (works with any model/dataset)
- **Model-Specific**: model_id set, dataset_id = NULL (tuned for specific model architecture)
- **Complete Templates**: Both set (tested configuration for specific combination)

---

### Table: `extraction_templates`

Reusable feature extraction configuration templates.

```sql
CREATE TABLE extraction_templates (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'ext_tmpl_abc123'

    -- Template details
    name VARCHAR(500) NOT NULL,
    description TEXT,

    -- Extraction configuration
    layers INTEGER[] NOT NULL,  -- Array of layer indices: [0, 4, 8, 12]
    hook_types VARCHAR(50)[] NOT NULL,  -- Array of hook types: ['residual', 'mlp', 'attention']
    max_samples INTEGER,  -- NULL = use all samples
    top_k_examples INTEGER NOT NULL DEFAULT 100,  -- Max-activating examples per feature

    -- User preferences
    is_favorite BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_extraction_templates_is_favorite ON extraction_templates(is_favorite)
    WHERE is_favorite = TRUE;
CREATE INDEX idx_extraction_templates_created_at ON extraction_templates(created_at DESC);
```

**Storage Estimate**: ~500 bytes per template, 20 templates = 10KB

**Use Cases**:
- **Quick Scan**: Few layers (e.g., [0, 6, 11]), only residual stream
- **Full Analysis**: All layers, all hook types (residual, mlp, attention)
- **Mid-Layer Focus**: Target specific layer range for detailed analysis

---

### Table: `steering_presets`

Saved steering configurations for model behavior modification.

```sql
CREATE TABLE steering_presets (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'preset_abc123'
    training_id VARCHAR(255) NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,

    -- Preset details
    name VARCHAR(500) NOT NULL,
    description TEXT,

    -- Steering configuration (JSONB)
    features JSONB NOT NULL,  -- [{ "feature_id": 42, "coefficient": 2.5 }, ...]
    intervention_layer INTEGER NOT NULL,  -- Which layer to intervene

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_steering_presets_training_id ON steering_presets(training_id);
CREATE INDEX idx_steering_presets_created_at ON steering_presets(created_at DESC);
```

**Storage Estimate**: ~1KB per preset, 100 presets = 100KB

**Important JSONB Field**: `features`

```json
[
  { "feature_id": 42, "coefficient": 2.5 },
  { "feature_id": 128, "coefficient": -1.8 },
  { "feature_id": 1024, "coefficient": 3.2 }
]
```

---

## Relationships and Foreign Keys

### Entity Relationship Diagram

```
datasets ────┐
             ├─→ trainings ─→ checkpoints
models ──────┘       │
    ↑                ├─→ training_metrics
    |                │
    |                └─→ features ─→ feature_activations
    |                        │
    |                        └─→ steering_presets
    |
training_templates (optional refs to models/datasets)

extraction_templates (standalone, no foreign keys)
```

### Foreign Key Policies

1. **ON DELETE RESTRICT** (datasets, models → trainings)
   - Prevent deletion of datasets or models if trainings reference them
   - User must manually clean up trainings first
   - Prevents accidental data loss

2. **ON DELETE CASCADE** (trainings → children)
   - When training deleted, automatically delete:
     - `training_metrics`
     - `checkpoints`
     - `features`
     - `feature_activations` (via features)
     - `steering_presets`
   - Simplifies cleanup of completed trainings

3. **ON DELETE SET NULL** (models/datasets → training_templates)
   - Optional references in `training_templates` become NULL when model/dataset deleted
   - Template remains usable as generic template
   - Prevents template loss when deleting models/datasets

4. **No Foreign Keys** (extraction_templates)
   - Standalone configuration templates
   - No dependencies on other tables

---

## Indexes and Performance Optimization

### Index Strategy

1. **Primary Keys**: B-tree indexes (default)
2. **Foreign Keys**: Explicit indexes for JOIN performance
3. **Status Fields**: Filtered indexes for common queries
4. **Timestamps**: Descending indexes for "recent items" queries
5. **Full-Text Search**: GIN indexes on tsvector
6. **JSONB Fields**: GIN indexes for containment queries (if needed)

### Query Performance Tips

1. **Use Composite Indexes**
   ```sql
   -- Good: Single index covers both filters
   CREATE INDEX idx_trainings_active ON trainings(status, updated_at);

   -- Query can use this index efficiently
   SELECT * FROM trainings
   WHERE status = 'training'
   ORDER BY updated_at DESC;
   ```

2. **Avoid N+1 Queries**
   ```python
   # Bad: N+1 query problem
   trainings = session.query(Training).all()
   for training in trainings:
       model = training.model  # Separate query per training

   # Good: Use eager loading
   trainings = session.query(Training).options(
       joinedload(Training.model),
       joinedload(Training.dataset)
   ).all()
   ```

3. **Paginate Large Result Sets**
   ```python
   # Use LIMIT/OFFSET or keyset pagination
   features = session.query(Feature).filter(
       Feature.training_id == training_id
   ).order_by(
       Feature.activation_frequency.desc()
   ).limit(100).offset(page * 100).all()
   ```

4. **Use Partial Indexes**
   ```sql
   -- Index only active trainings (saves space)
   CREATE INDEX idx_trainings_active ON trainings(updated_at)
       WHERE status IN ('training', 'queued');
   ```

---

## Table Partitioning

### Why Partition `feature_activations`?

- **Size**: This table can grow to 3GB+ per training
- **Query Pattern**: Always filtered by `feature_id`
- **Performance**: Partition pruning reduces scan time by 10-100x

### Partitioning Strategy

**List Partitioning by Feature ID Ranges**

```sql
-- Drop existing table and recreate as partitioned
DROP TABLE IF EXISTS feature_activations CASCADE;

CREATE TABLE feature_activations (
    id BIGSERIAL,
    feature_id BIGINT NOT NULL,
    dataset_sample_id VARCHAR(255),
    sample_index INTEGER,
    tokens JSONB NOT NULL,
    activations JSONB NOT NULL,
    max_activation FLOAT NOT NULL,
    context_before TEXT,
    context_after TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, feature_id)  -- Include partition key in PK
) PARTITION BY RANGE (feature_id);

-- Create partitions (example for 16K features, 1K per partition)
CREATE TABLE feature_activations_0000 PARTITION OF feature_activations
    FOR VALUES FROM (0) TO (1000);

CREATE TABLE feature_activations_1000 PARTITION OF feature_activations
    FOR VALUES FROM (1000) TO (2000);

CREATE TABLE feature_activations_2000 PARTITION OF feature_activations
    FOR VALUES FROM (2000) TO (3000);

-- ... repeat for all ranges ...

CREATE TABLE feature_activations_15000 PARTITION OF feature_activations
    FOR VALUES FROM (15000) TO (16384);

-- Create indexes on each partition
CREATE INDEX idx_fa_0000_feature ON feature_activations_0000(feature_id);
CREATE INDEX idx_fa_0000_max_activation ON feature_activations_0000(max_activation DESC);

-- ... repeat for all partitions ...
```

### Automated Partition Management

```python
# Python script to create partitions dynamically
def create_feature_activation_partitions(num_features: int, partition_size: int = 1000):
    """Create partitions for feature_activations table"""
    num_partitions = (num_features + partition_size - 1) // partition_size

    for i in range(num_partitions):
        start_id = i * partition_size
        end_id = min((i + 1) * partition_size, num_features)

        partition_name = f"feature_activations_{start_id:05d}"

        sql = f"""
        CREATE TABLE IF NOT EXISTS {partition_name}
        PARTITION OF feature_activations
        FOR VALUES FROM ({start_id}) TO ({end_id});

        CREATE INDEX IF NOT EXISTS idx_{partition_name}_feature
            ON {partition_name}(feature_id);
        CREATE INDEX IF NOT EXISTS idx_{partition_name}_max_activation
            ON {partition_name}(max_activation DESC);
        """

        engine.execute(sql)
```

---

## Database Migrations with Alembic

### Setup

```bash
# Install Alembic
pip install alembic

# Initialize Alembic
cd backend
alembic init alembic
```

### Configuration: `alembic/env.py`

```python
from app.db.base import Base  # Import all models
from app.config import settings

# Configure database URL
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Import all models for autogenerate
from app.db.models import (
    dataset, model, training, training_metrics,
    checkpoint, feature, feature_activation, steering_preset
)

target_metadata = Base.metadata
```

### Creating Migrations

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Add features table"

# Create empty migration for manual SQL
alembic revision -m "Add partitioning to feature_activations"
```

### Example Migration: Initial Schema

```python
# alembic/versions/001_initial_schema.py
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001'
down_revision = None

def upgrade():
    # Create datasets table
    op.create_table(
        'datasets',
        sa.Column('id', sa.String(255), primary_key=True),
        sa.Column('name', sa.String(500), nullable=False),
        sa.Column('source', sa.String(50), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('size_bytes', sa.BigInteger, nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('progress', sa.Float),
        sa.Column('error_message', sa.Text),
        sa.Column('file_path', sa.String(1000)),
        sa.Column('tokenized_path', sa.String(1000)),
        sa.Column('num_samples', sa.Integer),
        sa.Column('num_tokens', sa.BigInteger),
        sa.Column('vocab_size', sa.Integer),
        sa.Column('avg_sequence_length', sa.Float),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True),
                  server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True),
                  server_default=sa.func.now()),
    )

    # Create indexes
    op.create_index('idx_datasets_status', 'datasets', ['status'])
    op.create_index('idx_datasets_created_at', 'datasets', ['created_at'])

    # Full-text search index
    op.execute("""
        CREATE INDEX idx_datasets_search ON datasets
        USING GIN(to_tsvector('english', name || ' ' || COALESCE(description, '')))
    """)

    # ... repeat for all tables ...

def downgrade():
    op.drop_table('datasets')
    # ... drop all tables in reverse order ...
```

### Running Migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Show current migration status
alembic current

# Show migration history
alembic history
```

---

## Common Queries and Patterns

### 1. Get Active Trainings with Model/Dataset Info

```python
from sqlalchemy.orm import joinedload

trainings = session.query(Training).options(
    joinedload(Training.model),
    joinedload(Training.dataset)
).filter(
    Training.status.in_(['training', 'queued', 'initializing'])
).order_by(Training.created_at.desc()).all()
```

**Generated SQL:**
```sql
SELECT trainings.*, models.*, datasets.*
FROM trainings
JOIN models ON trainings.model_id = models.id
JOIN datasets ON trainings.dataset_id = datasets.id
WHERE trainings.status IN ('training', 'queued', 'initializing')
ORDER BY trainings.created_at DESC;
```

---

### 2. Get Training Metrics for Chart

```python
metrics = session.query(TrainingMetric).filter(
    TrainingMetric.training_id == training_id
).order_by(TrainingMetric.step).all()
```

**Generated SQL:**
```sql
SELECT * FROM training_metrics
WHERE training_id = 'tr_abc123'
ORDER BY step;
```

**Optimization**: Use index `idx_training_metrics_training_step`

---

### 3. Get Top Features by Activation Frequency

```python
features = session.query(Feature).filter(
    Feature.training_id == training_id
).order_by(
    Feature.activation_frequency.desc()
).limit(100).all()
```

**Generated SQL:**
```sql
SELECT * FROM features
WHERE training_id = 'tr_abc123'
ORDER BY activation_frequency DESC
LIMIT 100;
```

**Optimization**: Use index `idx_features_activation_freq`

---

### 4. Get Max-Activating Examples for Feature

```python
activations = session.query(FeatureActivation).filter(
    FeatureActivation.feature_id == feature_id
).order_by(
    FeatureActivation.max_activation.desc()
).limit(10).all()
```

**Generated SQL:**
```sql
SELECT * FROM feature_activations
WHERE feature_id = 42
ORDER BY max_activation DESC
LIMIT 10;
```

**Optimization**: Partition pruning + index `idx_feature_activations_max_activation`

---

### 5. Full-Text Search Datasets

```python
from sqlalchemy import func

search_query = "machine learning"
datasets = session.query(Dataset).filter(
    func.to_tsvector('english',
        Dataset.name + ' ' + func.coalesce(Dataset.description, '')
    ).op('@@')(func.plainto_tsquery('english', search_query))
).all()
```

**Generated SQL:**
```sql
SELECT * FROM datasets
WHERE to_tsvector('english', name || ' ' || COALESCE(description, ''))
      @@ plainto_tsquery('english', 'machine learning');
```

**Optimization**: Use GIN index `idx_datasets_search`

---

### 6. Update Training Status (Atomic)

```python
from sqlalchemy import update

# Atomic status update with timestamp
session.execute(
    update(Training).where(
        Training.id == training_id
    ).values(
        status='completed',
        completed_at=func.now(),
        current_step=total_steps
    )
)
session.commit()
```

---

### 7. Get Training Statistics (Aggregations)

```python
from sqlalchemy import func

stats = session.query(
    func.count(Training.id).label('total'),
    func.sum(
        func.case([(Training.status == 'completed', 1)], else_=0)
    ).label('completed'),
    func.avg(Training.current_step / Training.total_steps * 100).label('avg_progress')
).first()
```

**Generated SQL:**
```sql
SELECT
    COUNT(id) AS total,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
    AVG(current_step::FLOAT / total_steps * 100) AS avg_progress
FROM trainings;
```

---

### 8. Delete Old Training Metrics (Retention Policy)

```python
from datetime import datetime, timedelta

# Delete metrics older than 90 days for completed trainings
cutoff_date = datetime.utcnow() - timedelta(days=90)

session.query(TrainingMetric).filter(
    TrainingMetric.timestamp < cutoff_date,
    TrainingMetric.training_id.in_(
        session.query(Training.id).filter(Training.status == 'completed')
    )
).delete(synchronize_session=False)

session.commit()
```

---

## Connection Pooling

### Why Connection Pooling?

- **Performance**: Avoid connection overhead (50-100ms per connection)
- **Resource Limits**: PostgreSQL has max connections (default 100)
- **Concurrency**: Multiple API requests share connection pool

### SQLAlchemy Configuration

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

DATABASE_URL = "postgresql://mistudio:password@postgres:5432/mistudio"

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,          # Core pool size
    max_overflow=20,       # Additional connections under load
    pool_timeout=30,       # Wait time for connection (seconds)
    pool_recycle=3600,     # Recycle connections after 1 hour
    pool_pre_ping=True,    # Verify connection before using
    echo=False,            # Set to True for SQL logging
)
```

### Edge Device Optimization

For Jetson Orin Nano (limited resources):

```python
engine = create_engine(
    DATABASE_URL,
    pool_size=5,           # Smaller pool
    max_overflow=10,
    pool_recycle=1800,     # Recycle more frequently
    pool_pre_ping=True,
)
```

### Monitoring Pool Usage

```python
# Check pool status
print(f"Pool size: {engine.pool.size()}")
print(f"Checked out: {engine.pool.checkedout()}")
print(f"Overflow: {engine.pool.overflow()}")
print(f"Checked in: {engine.pool.checkedin()}")
```

---

## Backup and Restore

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh - Automated PostgreSQL backup

BACKUP_DIR="/data/backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.sql.gz"

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup with compression
docker exec mistudio_postgres pg_dump -U mistudio -d mistudio | gzip > $BACKUP_FILE

# Verify backup
if [ $? -eq 0 ]; then
    echo "Backup successful: $BACKUP_FILE"
    # Delete backups older than 30 days
    find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete
else
    echo "Backup failed!"
    exit 1
fi
```

### Restore from Backup

```bash
#!/bin/bash
# restore.sh - Restore from backup

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: ./restore.sh <backup_file>"
    exit 1
fi

# Stop backend services
docker-compose stop backend worker

# Drop and recreate database
docker exec mistudio_postgres psql -U mistudio -c "DROP DATABASE IF EXISTS mistudio;"
docker exec mistudio_postgres psql -U mistudio -c "CREATE DATABASE mistudio;"

# Restore from backup
gunzip -c $BACKUP_FILE | docker exec -i mistudio_postgres psql -U mistudio -d mistudio

# Restart services
docker-compose start backend worker

echo "Restore complete!"
```

### Backup Schedule (cron)

```bash
# Add to crontab: backup daily at 2 AM
0 2 * * * /home/user/mistudio/scripts/backup.sh >> /var/log/mistudio_backup.log 2>&1
```

---

## Configuration for Edge Devices

### PostgreSQL Configuration: `postgresql.conf`

```ini
# Memory settings for Jetson Orin Nano (8GB RAM)
shared_buffers = 512MB              # 25% of dedicated DB RAM
effective_cache_size = 1GB          # Estimated OS cache
work_mem = 16MB                     # Per-query memory
maintenance_work_mem = 128MB        # For VACUUM, CREATE INDEX

# Connection settings
max_connections = 50                # Lower for edge device

# Write-ahead log
wal_buffers = 16MB
checkpoint_completion_target = 0.9

# Query planner
random_page_cost = 1.1              # SSDs have low random access cost
effective_io_concurrency = 200      # SSD parallelism

# Logging
log_min_duration_statement = 1000   # Log slow queries (>1s)
log_line_prefix = '%t [%p]: '
```

### Docker Compose Configuration

```yaml
services:
  postgres:
    image: postgres:15-alpine
    container_name: mistudio_postgres
    environment:
      POSTGRES_USER: mistudio
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: mistudio
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgresql.conf:/etc/postgresql/postgresql.conf
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  postgres_data:
    driver: local
```

---

## Data Retention Policies

### Policy Recommendations

1. **Training Metrics**
   - Keep all metrics for active trainings (status != 'completed')
   - Keep last 30 days for completed trainings
   - Delete metrics older than 90 days for completed trainings

2. **Feature Activations**
   - Keep top 100 examples per feature
   - Delete examples with `max_activation < threshold` (e.g., 0.1)

3. **Checkpoints**
   - Keep last 5 checkpoints per training
   - Keep best checkpoint (lowest loss)
   - Delete checkpoints for trainings older than 6 months

### Automated Cleanup Script

```python
# cleanup.py - Data retention policy enforcement

from datetime import datetime, timedelta
from sqlalchemy import func

def cleanup_training_metrics(session, days=90):
    """Delete old training metrics for completed trainings"""
    cutoff_date = datetime.utcnow() - timedelta(days=days)

    deleted = session.query(TrainingMetric).filter(
        TrainingMetric.timestamp < cutoff_date,
        TrainingMetric.training_id.in_(
            session.query(Training.id).filter(Training.status == 'completed')
        )
    ).delete(synchronize_session=False)

    session.commit()
    print(f"Deleted {deleted} old training metrics")

def cleanup_checkpoints(session, keep_count=5):
    """Keep only recent checkpoints per training"""
    trainings = session.query(Training).all()

    for training in trainings:
        checkpoints = session.query(Checkpoint).filter(
            Checkpoint.training_id == training.id
        ).order_by(Checkpoint.step.desc()).all()

        if len(checkpoints) > keep_count:
            to_delete = checkpoints[keep_count:]
            for cp in to_delete:
                # Delete file
                if os.path.exists(cp.storage_path):
                    os.remove(cp.storage_path)
                # Delete record
                session.delete(cp)

    session.commit()

def cleanup_feature_activations(session, min_activation=0.1):
    """Delete low-activation examples"""
    deleted = session.query(FeatureActivation).filter(
        FeatureActivation.max_activation < min_activation
    ).delete(synchronize_session=False)

    session.commit()
    print(f"Deleted {deleted} low-activation examples")
```

### Schedule Cleanup (cron)

```bash
# Run cleanup weekly on Sunday at 3 AM
0 3 * * 0 cd /home/user/mistudio/backend && python cleanup.py >> /var/log/mistudio_cleanup.log 2>&1
```

---

## Security Considerations

### 1. Connection Security

```python
# Use SSL for production
DATABASE_URL = "postgresql://user:pass@host:5432/db?sslmode=require"
```

### 2. Password Security

```bash
# Never hardcode passwords
# Use environment variables
export POSTGRES_PASSWORD=$(openssl rand -base64 32)
```

### 3. SQL Injection Prevention

```python
# Bad: String concatenation (vulnerable to SQL injection)
query = f"SELECT * FROM datasets WHERE name = '{user_input}'"

# Good: Use parameterized queries
query = session.query(Dataset).filter(Dataset.name == user_input)
```

### 4. Least Privilege

```sql
-- Create read-only user for monitoring tools
CREATE ROLE mistudio_readonly;
GRANT CONNECT ON DATABASE mistudio TO mistudio_readonly;
GRANT USAGE ON SCHEMA public TO mistudio_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mistudio_readonly;

CREATE USER monitoring_user WITH PASSWORD 'secure_password';
GRANT mistudio_readonly TO monitoring_user;
```

### 5. Regular VACUUM and ANALYZE

```bash
# Add to cron: weekly maintenance
0 4 * * 0 docker exec mistudio_postgres vacuumdb -U mistudio -d mistudio --analyze
```

---

## Monitoring and Troubleshooting

### Key Metrics to Monitor

1. **Connection Count**
   ```sql
   SELECT count(*) FROM pg_stat_activity;
   ```

2. **Database Size**
   ```sql
   SELECT pg_size_pretty(pg_database_size('mistudio'));
   ```

3. **Table Sizes**
   ```sql
   SELECT
       schemaname,
       tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
   FROM pg_tables
   WHERE schemaname = 'public'
   ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
   ```

4. **Slow Queries**
   ```sql
   SELECT
       query,
       calls,
       total_time,
       mean_time
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;
   ```

5. **Index Usage**
   ```sql
   SELECT
       schemaname,
       tablename,
       indexname,
       idx_scan,
       idx_tup_read,
       idx_tup_fetch
   FROM pg_stat_user_indexes
   WHERE idx_scan = 0
   ORDER BY pg_relation_size(indexrelid) DESC;
   ```

### Common Issues and Solutions

1. **Too Many Connections**
   - **Symptom**: "FATAL: remaining connection slots are reserved"
   - **Solution**: Increase `max_connections` or reduce pool size

2. **Slow Queries**
   - **Symptom**: API requests timeout
   - **Solution**: Add indexes, optimize queries, increase `work_mem`

3. **Disk Full**
   - **Symptom**: "No space left on device"
   - **Solution**: Run VACUUM, delete old data, expand storage

4. **High CPU Usage**
   - **Symptom**: System sluggish
   - **Solution**: Check for missing indexes, long-running queries

---

## Best Practices Summary

1. **Schema Design**
   - ✅ Use JSONB for flexible, nested data
   - ✅ Add indexes on all foreign keys
   - ✅ Use CHECK constraints for status fields
   - ✅ Partition large tables (>1GB)

2. **Query Optimization**
   - ✅ Use eager loading (joinedload) for relationships
   - ✅ Paginate large result sets
   - ✅ Use composite indexes for multi-column filters
   - ✅ Monitor slow queries with pg_stat_statements

3. **Data Management**
   - ✅ Implement retention policies
   - ✅ Regular VACUUM and ANALYZE
   - ✅ Automated backups (daily)
   - ✅ Test restore procedures

4. **Connection Management**
   - ✅ Use connection pooling (SQLAlchemy)
   - ✅ Set appropriate pool sizes for edge devices
   - ✅ Enable pool_pre_ping for reliability

5. **Security**
   - ✅ Use environment variables for passwords
   - ✅ Parameterized queries (ORM) to prevent SQL injection
   - ✅ Regular security updates (postgres:15-alpine)
   - ✅ Least-privilege database users

---

## Related Documentation

- [Folder Structure Specification](./001_SPEC|Folder_File_Details.md)
- [Redis Usage Guide](./000_SPEC|REDIS_GUIDANCE_USECASE.md)
- [Technical Specification](./miStudio_Specification.md)
- [OpenAPI Specification](./openapi.yaml)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-05
**Maintained By**: MechInterp Studio Team
