# MechInterp Studio (miStudio) - Complete Folder & File Structure Specification

**Document Type**: Technical Specification
**Last Updated**: 2025-10-05
**Purpose**: Comprehensive directory layout for production application with /backend and /frontend separation

---

## Table of Contents

1. [Project Root Structure](#project-root-structure)
2. [Frontend Structure](#frontend-structure)
3. [Backend Structure](#backend-structure)
4. [Data Storage Structure](#data-storage-structure)
5. [Configuration Files](#configuration-files)
6. [Docker & Deployment](#docker--deployment)
7. [Documentation](#documentation)
8. [Development Tools](#development-tools)

---

## Project Root Structure

```
mistudio/
├── frontend/                   # React TypeScript frontend application
├── backend/                    # Python FastAPI backend application
├── data/                       # Persistent data storage (not in git)
├── docker/                     # Docker configurations and scripts
├── docs/                       # Project documentation
├── scripts/                    # Utility scripts for development/deployment
├── .github/                    # GitHub Actions CI/CD workflows
├── docker-compose.yml          # Development environment orchestration
├── docker-compose.prod.yml     # Production environment orchestration
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore rules
├── README.md                   # Project overview and quick start
└── LICENSE                     # Project license

Total: ~500-1000 files (excluding node_modules, __pycache__, data)
```

---

## Frontend Structure

### Overview
```
frontend/
├── public/                     # Static assets (served directly)
├── src/                        # Source code
├── tests/                      # Test files
├── .env.development            # Development environment variables
├── .env.production             # Production environment variables
├── index.html                  # HTML entry point
├── main.tsx                    # TypeScript entry point
├── package.json                # Node dependencies and scripts
├── tsconfig.json               # TypeScript configuration
├── tsconfig.node.json          # TypeScript config for Node tools
├── vite.config.ts              # Vite build configuration
├── tailwind.config.js          # Tailwind CSS configuration
├── postcss.config.js           # PostCSS configuration
└── vitest.config.ts            # Vitest testing configuration
```

### Detailed Source Structure

```
frontend/src/
│
├── main.tsx                    # Application entry point
├── App.tsx                     # Root component
├── index.css                   # Global styles
│
├── assets/                     # Static assets imported in code
│   ├── images/
│   │   ├── logo.svg
│   │   └── icons/
│   └── fonts/
│
├── components/                 # React components
│   │
│   ├── common/                 # Shared/reusable components
│   │   ├── Button/
│   │   │   ├── Button.tsx
│   │   │   ├── Button.test.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── Modal/
│   │   │   ├── Modal.tsx
│   │   │   ├── Modal.test.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── ProgressBar/
│   │   │   ├── ProgressBar.tsx
│   │   │   ├── ProgressBar.test.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── StatusBadge/
│   │   │   ├── StatusBadge.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── LoadingState/
│   │   │   ├── LoadingState.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── ErrorState/
│   │   │   ├── ErrorState.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── EmptyState/
│   │   │   ├── EmptyState.tsx
│   │   │   └── index.ts
│   │   │
│   │   └── index.ts            # Re-export all common components
│   │
│   ├── layout/                 # Layout components
│   │   ├── Header/
│   │   │   ├── Header.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── Sidebar/
│   │   │   ├── Sidebar.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── MainLayout/
│   │   │   ├── MainLayout.tsx
│   │   │   └── index.ts
│   │   │
│   │   └── index.ts
│   │
│   ├── datasets/               # Dataset management components
│   │   ├── DatasetsPanel/
│   │   │   ├── DatasetsPanel.tsx
│   │   │   ├── DatasetsPanel.test.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── DatasetCard/
│   │   │   ├── DatasetCard.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── DatasetDetailModal/
│   │   │   ├── DatasetDetailModal.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── DatasetStatistics/
│   │   │   ├── DatasetStatistics.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── DatasetSamplesBrowser/
│   │   │   ├── DatasetSamplesBrowser.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── TokenizationSettings/
│   │   │   ├── TokenizationSettings.tsx
│   │   │   └── index.ts
│   │   │
│   │   └── index.ts
│   │
│   ├── models/                 # Model management components
│   │   ├── ModelsPanel/
│   │   │   ├── ModelsPanel.tsx
│   │   │   ├── ModelsPanel.test.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── ModelCard/
│   │   │   ├── ModelCard.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── ModelDetailModal/
│   │   │   ├── ModelDetailModal.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── ActivationExtractionPanel/
│   │   │   ├── ActivationExtractionPanel.tsx
│   │   │   └── index.ts
│   │   │
│   │   └── index.ts
│   │
│   ├── training/               # Training components
│   │   ├── TrainingPanel/
│   │   │   ├── TrainingPanel.tsx
│   │   │   ├── TrainingPanel.test.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── TrainingJobCard/
│   │   │   ├── TrainingJobCard.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── TrainingMetricsChart/
│   │   │   ├── TrainingMetricsChart.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── TrainingLogsViewer/
│   │   │   ├── TrainingLogsViewer.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── CheckpointManagement/
│   │   │   ├── CheckpointManagement.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── GPUMonitor/
│   │   │   ├── GPUMonitor.tsx
│   │   │   └── index.ts
│   │   │
│   │   └── index.ts
│   │
│   ├── features/               # Feature discovery components
│   │   ├── FeaturesPanel/
│   │   │   ├── FeaturesPanel.tsx
│   │   │   ├── FeaturesPanel.test.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── FeatureTable/
│   │   │   ├── FeatureTable.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── FeatureDetailModal/
│   │   │   ├── FeatureDetailModal.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── MaxActivatingExamples/
│   │   │   ├── MaxActivatingExamples.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── LogitLensView/
│   │   │   ├── LogitLensView.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── FeatureCorrelations/
│   │   │   ├── FeatureCorrelations.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── AblationAnalysis/
│   │   │   ├── AblationAnalysis.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── ActivationHeatmap/
│   │   │   ├── ActivationHeatmap.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── UMAPProjection/
│   │   │   ├── UMAPProjection.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── FeatureCorrelationMatrix/
│   │   │   ├── FeatureCorrelationMatrix.tsx
│   │   │   └── index.ts
│   │   │
│   │   └── index.ts
│   │
│   ├── steering/               # Feature steering components
│   │   ├── SteeringPanel/
│   │   │   ├── SteeringPanel.tsx
│   │   │   ├── SteeringPanel.test.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── FeatureSelector/
│   │   │   ├── FeatureSelector.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── SteeringCoefficientSlider/
│   │   │   ├── SteeringCoefficientSlider.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── ComparativeOutput/
│   │   │   ├── ComparativeOutput.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── SteeringMetrics/
│   │   │   ├── SteeringMetrics.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── SteeringPresets/
│   │   │   ├── SteeringPresets.tsx
│   │   │   └── index.ts
│   │   │
│   │   └── index.ts
│   │
│   ├── templates/              # Configuration template components
│   │   ├── TrainingTemplates/
│   │   │   ├── TrainingTemplates.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── ExtractionTemplates/
│   │   │   ├── ExtractionTemplates.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── TemplateCard/
│   │   │   ├── TemplateCard.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── TemplateEditor/
│   │   │   ├── TemplateEditor.tsx
│   │   │   └── index.ts
│   │   │
│   │   └── index.ts
│   │
│   └── index.ts                # Re-export all components
│
├── contexts/                   # React Context providers
│   ├── AuthContext.tsx         # Authentication state
│   ├── TrainingContext.tsx     # Training state management
│   ├── ThemeContext.tsx        # Theme/dark mode
│   └── index.ts
│
├── hooks/                      # Custom React hooks
│   ├── api/                    # API data fetching hooks
│   │   ├── useDatasets.ts
│   │   ├── useDatasets.test.ts
│   │   ├── useModels.ts
│   │   ├── useTrainings.ts
│   │   ├── useTrainingProgress.ts
│   │   ├── useFeatures.ts
│   │   ├── useFeatureExtraction.ts
│   │   ├── useTemplates.ts
│   │   └── index.ts
│   │
│   ├── useWebSocket.ts         # WebSocket connection hook
│   ├── useLocalStorage.ts      # Persistent local state
│   ├── useDebounce.ts          # Debounced values
│   ├── useIntersectionObserver.ts  # Lazy loading
│   └── index.ts
│
├── services/                   # API and service layer
│   ├── api/                    # Backend API clients
│   │   ├── client.ts           # Base API client with error handling
│   │   ├── datasets.api.ts     # Dataset endpoints
│   │   ├── models.api.ts       # Model endpoints
│   │   ├── training.api.ts     # Training endpoints
│   │   ├── features.api.ts     # Feature endpoints
│   │   ├── steering.api.ts     # Steering endpoints
│   │   ├── templates.api.ts    # Template endpoints
│   │   └── index.ts
│   │
│   ├── websocket/              # WebSocket services
│   │   ├── trainingMetrics.ws.ts
│   │   ├── systemStatus.ws.ts
│   │   └── index.ts
│   │
│   ├── mock/                   # Mock API for development
│   │   ├── mockDatasets.ts
│   │   ├── mockModels.ts
│   │   ├── mockTrainings.ts
│   │   ├── mockFeatures.ts
│   │   └── index.ts
│   │
│   └── index.ts
│
├── types/                      # TypeScript type definitions
│   ├── api.types.ts            # API request/response types
│   ├── dataset.types.ts        # Dataset-related types
│   ├── model.types.ts          # Model-related types
│   ├── training.types.ts       # Training-related types
│   ├── feature.types.ts        # Feature-related types
│   ├── steering.types.ts       # Steering-related types
│   ├── template.types.ts       # Template-related types
│   ├── errors.types.ts         # Error types
│   └── index.ts
│
├── constants/                  # Constants and configuration
│   ├── apiEndpoints.ts         # API endpoint paths
│   ├── polling.ts              # Polling intervals
│   ├── validation.ts           # Validation rules
│   ├── defaults.ts             # Default values
│   └── index.ts
│
├── utils/                      # Utility functions
│   ├── formatting.ts           # Format numbers, dates, etc.
│   ├── validation.ts           # Input validation
│   ├── errors.ts               # Error handling utilities
│   ├── storage.ts              # LocalStorage helpers
│   └── index.ts
│
└── lib/                        # Third-party library configurations
    ├── reactQuery.ts           # React Query setup
    ├── axios.ts                # Axios configuration (if used)
    └── index.ts
```

### Frontend File Count Estimate
- Components: ~100 files (.tsx + .test.tsx + index.ts)
- Hooks: ~20 files
- Services: ~30 files
- Types: ~10 files
- Utils: ~15 files
- **Total: ~175 source files**

---

## Backend Structure

### Overview
```
backend/
├── app/                        # Main application package
├── tests/                      # Test files
├── scripts/                    # Utility scripts
├── alembic/                    # Database migrations
├── requirements.txt           Ple # Python dependencies
├── requirements-dev.txt        # Development dependencies
├── pyproject.toml              # Python project configuration
├── pytest.ini                  # Pytest configuration
├── .env.example                # Example environment variables
├── .env.development            # Development environment
├── .env.production             # Production environment
├── Dockerfile                  # Docker image definition
└── README.md                   # Backend-specific documentation
```

### Detailed App Structure

```
backend/app/
│
├── main.py                     # FastAPI application entry point
├── config.py                   # Configuration management
├── dependencies.py             # Dependency injection
│
├── api/                        # API routes
│   ├── __init__.py
│   │
│   ├── v1/                     # API version 1
│   │   ├── __init__.py
│   │   ├── router.py           # Main router aggregation
│   │   │
│   │   ├── endpoints/          # API endpoint modules
│   │   │   ├── __init__.py
│   │   │   ├── datasets.py     # Dataset endpoints
│   │   │   ├── models.py       # Model endpoints
│   │   │   ├── training.py     # Training endpoints
│   │   │   ├── features.py     # Feature endpoints
│   │   │   ├── steering.py     # Steering endpoints
│   │   │   ├── templates.py    # Template endpoints
│   │   │   ├── system.py       # System/health endpoints
│   │   │   └── websocket.py    # WebSocket endpoints
│   │   │
│   │   └── dependencies.py     # Route-level dependencies
│   │
│   └── middleware/             # API middleware
│       ├── __init__.py
│       ├── cors.py             # CORS configuration
│       ├── auth.py             # Authentication middleware
│       ├── rate_limit.py       # Rate limiting
│       ├── error_handler.py    # Global error handling
│       └── logging.py          # Request logging
│
├── core/                       # Core functionality
│   ├── __init__.py
│   ├── config.py               # Core configuration
│   ├── security.py             # Security utilities (hashing, tokens)
│   ├── logging.py              # Logging configuration
│   └── exceptions.py           # Custom exception classes
│
├── db/                         # Database
│   ├── __init__.py
│   ├── base.py                 # SQLAlchemy base
│   ├── session.py              # Database session management
│   │
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset model
│   │   ├── model.py            # Model model
│   │   ├── training.py         # Training model
│   │   ├── training_metrics.py # TrainingMetrics model
│   │   ├── checkpoint.py       # Checkpoint model
│   │   ├── feature.py          # Feature model
│   │   ├── feature_activation.py  # FeatureActivation model
│   │   ├── steering_preset.py  # SteeringPreset model
│   │   ├── training_template.py    # TrainingTemplate model
│   │   ├── extraction_template.py  # ExtractionTemplate model
│   │   └── base.py             # Base model class
│   │
│   └── repositories/           # Data access layer
│       ├── __init__.py
│       ├── base.py             # Base repository with CRUD
│       ├── dataset_repo.py
│       ├── model_repo.py
│       ├── training_repo.py
│       ├── feature_repo.py
│       ├── steering_repo.py
│       └── template_repo.py
│
├── schemas/                    # Pydantic schemas (validation & serialization)
│   ├── __init__.py
│   ├── common.py               # Common schemas (pagination, errors)
│   ├── dataset.py              # Dataset schemas
│   ├── model.py                # Model schemas
│   ├── training.py             # Training schemas
│   ├── feature.py              # Feature schemas
│   ├── steering.py             # Steering schemas
│   ├── template.py             # Template schemas
│   └── system.py               # System/health schemas
│
├── services/                   # Business logic layer
│   ├── __init__.py
│   │
│   ├── dataset_service.py      # Dataset business logic
│   ├── model_service.py        # Model business logic
│   ├── training_service.py     # Training orchestration
│   ├── feature_service.py      # Feature extraction & analysis
│   ├── steering_service.py     # Steering generation
│   ├── template_service.py     # Template management
│   ├── storage_service.py      # File storage operations
│   └── cache_service.py        # Redis caching operations
│
├── workers/                    # Background job workers
│   ├── __init__.py
│   ├── celery_app.py           # Celery configuration
│   │
│   ├── tasks/                  # Celery tasks
│   │   ├── __init__.py
│   │   ├── dataset_tasks.py    # Dataset download/processing
│   │   ├── model_tasks.py      # Model download/quantization
│   │   ├── activation_tasks.py # Activation extraction
│   │   ├── training_tasks.py   # SAE training
│   │   ├── feature_tasks.py    # Feature extraction
│   │   └── steering_tasks.py   # Steering generation
│   │
│   └── utils/                  # Worker utilities
│       ├── __init__.py
│       ├── progress.py         # Progress reporting
│       └── gpu.py              # GPU management
│
├── ml/                         # Machine learning code
│   ├── __init__.py
│   │
│   ├── models/                 # ML model implementations
│   │   ├── __init__.py
│   │   ├── sparse_autoencoder.py  # SAE implementation
│   │   ├── skip_autoencoder.py    # Skip SAE
│   │   ├── transcoder.py          # Transcoder
│   │   └── base.py                # Base autoencoder class
│   │
│   ├── training/               # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   ├── optimizers.py       # Optimizer configurations
│   │   ├── schedulers.py       # LR schedulers
│   │   └── losses.py           # Custom loss functions
│   │
│   ├── inference/              # Inference utilities
│   │   ├── __init__.py
│   │   ├── activations.py      # Activation extraction
│   │   ├── steering.py         # Steering implementation
│   │   └── hooks.py            # PyTorch forward hooks
│   │
│   ├── analysis/               # Feature analysis
│   │   ├── __init__.py
│   │   ├── logit_lens.py       # Logit lens analysis
│   │   ├── correlations.py     # Feature correlations
│   │   ├── ablation.py         # Ablation studies
│   │   └── projections.py      # UMAP/t-SNE
│   │
│   └── utils/                  # ML utilities
│       ├── __init__.py
│       ├── quantization.py     # Model quantization
│       ├── tokenization.py     # Tokenization utilities
│       └── gpu_utils.py        # GPU memory management
│
├── storage/                    # File storage management
│   ├── __init__.py
│   ├── local.py                # Local filesystem storage
│   └── base.py                 # Storage interface
│
├── websockets/                 # WebSocket handlers
│   ├── __init__.py
│   ├── manager.py              # Connection manager
│   ├── training.py             # Training progress WS
│   └── system.py               # System status WS
│
└── utils/                      # Utility modules
    ├── __init__.py
    ├── hashing.py              # Hashing utilities
    ├── time.py                 # Time/date utilities
    ├── validators.py           # Custom validators
    └── file_utils.py           # File manipulation utilities
```

### Alembic (Database Migrations)

```
backend/alembic/
├── versions/                   # Migration files
│   ├── 001_initial_schema.py
│   ├── 002_add_checkpoints.py
│   ├── 003_add_feature_activations.py
│   └── ...
├── env.py                      # Alembic environment
├── script.py.mako              # Migration template
└── alembic.ini                 # Alembic configuration
```

### Backend Tests

```
backend/tests/
├── __init__.py
├── conftest.py                 # Pytest fixtures
│
├── unit/                       # Unit tests
│   ├── __init__.py
│   ├── test_services/
│   ├── test_repositories/
│   ├── test_ml/
│   └── test_utils/
│
├── integration/                # Integration tests
│   ├── __init__.py
│   ├── test_api/
│   ├── test_workers/
│   └── test_database/
│
└── e2e/                        # End-to-end tests
    ├── __init__.py
    └── test_workflows/
```

### Backend File Count Estimate
- API routes: ~15 files
- Models: ~10 files
- Schemas: ~10 files
- Services: ~10 files
- Workers/Tasks: ~15 files
- ML code: ~20 files
- Tests: ~50 files
- **Total: ~150 source files**

---

## Data Storage Structure

### Overview

The `/data` directory stores all persistent application data. This directory should be:
- **Excluded from git** (add to .gitignore)
- **Backed up regularly** in production
- **Mounted as a volume** in Docker deployments
- **Expandable** to external storage (NFS, USB drives)

### Complete Data Directory Structure

```
data/                           # Root data directory (not in git)
│
├── datasets/                   # HuggingFace datasets
│   ├── raw/                    # Original downloaded datasets
│   │   ├── dataset_ds_abc123/
│   │   │   ├── train.parquet
│   │   │   ├── validation.parquet
│   │   │   ├── test.parquet
│   │   │   └── dataset_info.json
│   │   │
│   │   └── dataset_ds_xyz456/
│   │       └── ...
│   │
│   ├── tokenized/              # Tokenized datasets (Arrow format)
│   │   ├── dataset_ds_abc123_tokenized/
│   │   │   ├── train.arrow
│   │   │   ├── validation.arrow
│   │   │   └── metadata.json
│   │   │
│   │   └── dataset_ds_xyz456_tokenized/
│   │       └── ...
│   │
│   └── metadata/               # Dataset metadata and statistics
│       ├── ds_abc123_stats.json
│       └── ds_xyz456_stats.json
│
├── models/                     # Downloaded and quantized models
│   ├── raw/                    # Original model weights
│   │   ├── model_m_abc123/
│   │   │   ├── pytorch_model.bin
│   │   │   ├── config.json
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── special_tokens_map.json
│   │   │
│   │   └── model_m_xyz456/
│   │       └── ...
│   │
│   ├── quantized/              # Quantized model variants
│   │   ├── model_m_abc123_q4/
│   │   │   ├── model.safetensors
│   │   │   ├── config.json
│   │   │   └── quantization_config.json
│   │   │
│   │   └── model_m_xyz456_fp16/
│   │       └── ...
│   │
│   └── metadata/               # Model metadata
│       ├── m_abc123_architecture.json
│       └── m_xyz456_architecture.json
│
├── activations/                # Extracted activation tensors
│   ├── extraction_ext_abc123/  # Per extraction job
│   │   ├── model_info.json     # Model and dataset references
│   │   ├── layer_0_residual.npy  # shape: [n_samples, seq_len, hidden_dim]
│   │   ├── layer_0_mlp.npy
│   │   ├── layer_0_attention.npy
│   │   ├── layer_4_residual.npy
│   │   ├── layer_4_mlp.npy
│   │   ├── layer_4_attention.npy
│   │   ├── layer_8_residual.npy
│   │   ├── ...
│   │   ├── metadata.json       # Extraction parameters
│   │   └── statistics.json     # Activation statistics
│   │
│   └── extraction_ext_xyz456/
│       └── ...
│
├── trainings/                  # Training outputs
│   ├── training_tr_abc123/     # Per training job
│   │   ├── config.json         # Hyperparameters and configuration
│   │   ├── encoder.pt          # Trained SAE encoder weights
│   │   ├── decoder.pt          # Trained SAE decoder weights
│   │   ├── optimizer_state.pt  # Optimizer state (for resuming)
│   │   │
│   │   ├── checkpoints/        # Training checkpoints
│   │   │   ├── checkpoint_step_1000.pt
│   │   │   ├── checkpoint_step_2000.pt
│   │   │   ├── checkpoint_step_5000.pt
│   │   │   └── checkpoint_final.pt
│   │   │
│   │   ├── logs/               # Training logs
│   │   │   ├── training.log
│   │   │   └── metrics.jsonl   # Line-delimited JSON metrics
│   │   │
│   │   └── tensorboard/        # TensorBoard logs (optional)
│   │       └── events.out.tfevents...
│   │
│   └── training_tr_xyz456/
│       └── ...
│
├── features/                   # Extracted features
│   ├── training_tr_abc123/     # Features from specific training
│   │   ├── features.npy        # Feature activation matrix
│   │   ├── feature_metadata.json  # Feature names, interpretability scores
│   │   │
│   │   ├── activations/        # Per-feature activation examples
│   │   │   ├── feature_0_activations.jsonl
│   │   │   ├── feature_1_activations.jsonl
│   │   │   └── ...
│   │   │
│   │   ├── analysis/           # Feature analysis results
│   │   │   ├── logit_lens.json
│   │   │   ├── correlations.npy
│   │   │   ├── ablation_results.json
│   │   │   └── umap_projection.json
│   │   │
│   │   └── visualizations/     # Generated visualizations
│   │       ├── heatmap_feature_0.png
│   │       └── ...
│   │
│   └── training_tr_xyz456/
│       └── ...
│
├── steering/                   # Steering presets and results
│   ├── presets/
│   │   ├── preset_positive_sentiment.json
│   │   ├── preset_formal_tone.json
│   │   └── ...
│   │
│   └── generations/            # Saved steered generations
│       ├── generation_gen_abc123.json
│       └── ...
│
├── templates/                  # Configuration templates for reusable settings
│   ├── training/               # Training configuration templates
│   │   ├── template_tt_abc123.json
│   │   │   # {
│   │   │   #   "id": "tt_abc123",
│   │   │   #   "name": "Fast Prototyping",
│   │   │   #   "description": "Quick SAE training for testing",
│   │   │   #   "model_id": null,
│   │   │   #   "dataset_id": null,
│   │   │   #   "encoder_type": "sparse",
│   │   │   #   "hyperparameters": { ... },
│   │   │   #   "is_favorite": true,
│   │   │   #   "created_at": "2025-10-05T10:00:00Z",
│   │   │   #   "updated_at": "2025-10-05T10:00:00Z"
│   │   │   # }
│   │   │
│   │   ├── template_tt_xyz456.json
│   │   └── ...
│   │
│   ├── extraction/             # Feature extraction configuration templates
│   │   ├── template_et_abc123.json
│   │   │   # {
│   │   │   #   "id": "et_abc123",
│   │   │   #   "name": "Quick Scan",
│   │   │   #   "description": "Fast feature extraction for initial exploration",
│   │   │   #   "layers": [0, 4, 8],
│   │   │   #   "hook_types": ["residual"],
│   │   │   #   "max_samples": 1000,
│   │   │   #   "top_k_examples": 10,
│   │   │   #   "is_favorite": false,
│   │   │   #   "created_at": "2025-10-05T10:00:00Z",
│   │   │   #   "updated_at": "2025-10-05T10:00:00Z"
│   │   │   # }
│   │   │
│   │   ├── template_et_xyz456.json
│   │   └── ...
│   │
│   └── metadata/               # Template metadata and usage statistics
│       ├── usage_stats.json     # Template usage tracking
│       └── last_used.json       # Recently used templates cache
│
├── cache/                      # Application cache (can be cleared)
│   ├── dataset_statistics/
│   ├── model_architectures/
│   └── feature_correlations/
│
├── temp/                       # Temporary files (can be cleared)
│   ├── uploads/                # Uploaded files before processing
│   └── processing/             # Files being processed
│
└── backups/                    # Database and critical data backups
    ├── postgres/
    │   ├── backup_20251005_120000.sql.gz
    │   └── ...
    │
    └── redis/
        ├── dump_20251005_120000.rdb
        └── ...
```

### Storage Size Estimates

| Directory | Typical Size | Max Expected | Growth Rate |
|-----------|-------------|--------------|-------------|
| `datasets/raw/` | 10-100 GB | 500 GB | Per dataset download |
| `datasets/tokenized/` | 5-50 GB | 250 GB | Per dataset processing |
| `models/raw/` | 5-50 GB | 200 GB | Per model download |
| `models/quantized/` | 2-20 GB | 100 GB | Per quantization |
| `activations/` | 20-200 GB | 1 TB | Per extraction job |
| `trainings/` | 10-100 GB | 500 GB | Per training job |
| `features/` | 5-50 GB | 200 GB | Per feature extraction |
| `steering/` | 100 MB | 10 GB | Per steering session |
| `templates/` | 10-50 MB | 500 MB | Per template saved |
| `cache/` | 1-10 GB | 50 GB | Automatic LRU eviction |
| `temp/` | 1-5 GB | 20 GB | Cleaned daily |
| **TOTAL** | **60-585 GB** | **2.84 TB** | Continuous |

### Storage Cleanup Strategy

```python
# Example cleanup script
cleanup_policy = {
    "temp/": "delete older than 1 day",
    "cache/": "LRU eviction when > 50GB",
    "activations/": "keep last 5 extractions",
    "trainings/checkpoints/": "keep checkpoints: 1000, 2000, 5000, final",
    "features/visualizations/": "delete when features deleted",
}
```

---

## Configuration Files

### Project Root Configuration Files

```
mistudio/
├── .gitignore                  # Git ignore patterns
├── .dockerignore               # Docker ignore patterns
├── .editorconfig               # Editor configuration
├── .prettierrc                 # Code formatting (frontend)
├── .eslintrc.json              # ESLint config (frontend)
├── docker-compose.yml          # Development environment
├── docker-compose.prod.yml     # Production environment
├── Makefile                    # Common commands
└── .env.example                # Environment variables template
```

### Example .env.example

```bash
# Application
NODE_ENV=development
PYTHON_ENV=development

# Frontend
VITE_API_BASE_URL=http://miStudio.mcslab.io:8000
VITE_WS_BASE_URL=ws://miStudio.mcslab.io:8000
VITE_USE_MOCK_API=false

# Backend
DATABASE_URL=postgresql://mistudio:password@postgres:5432/mistudio
REDIS_URL=redis://redis:6379/0
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Storage
DATA_DIR=/data

# ML/GPU
CUDA_VISIBLE_DEVICES=0
HF_TOKEN=your-huggingface-token  # Optional, for gated models
MAX_CONCURRENT_TRAININGS=1

# Workers
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1
CELERY_WORKER_CONCURRENCY=2

# Monitoring (optional)
SENTRY_DSN=
LOG_LEVEL=INFO
```

---

## Docker & Deployment

### Docker Structure

```
docker/
├── frontend/
│   └── Dockerfile              # Frontend production build
│
├── backend/
│   ├── Dockerfile              # Backend application
│   └── Dockerfile.worker       # Celery worker
│
├── nginx/
│   ├── nginx.conf              # Nginx configuration
│   └── ssl/                    # SSL certificates (if applicable)
│       ├── cert.pem
│       └── key.pem
│
└── scripts/
    ├── entrypoint.sh           # Container entrypoint scripts
    ├── wait-for-it.sh          # Wait for services to be ready
    └── backup.sh               # Backup script
```

### Docker Compose Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - VITE_API_BASE_URL=http://localhost:8000

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - ./data:/data
    environment:
      - DATABASE_URL=postgresql://mistudio:password@postgres:5432/mistudio
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis

  worker:
    build:
      context: ./backend
      dockerfile: Dockerfile.worker
    volumes:
      - ./backend:/app
      - ./data:/data
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis
      - postgres

  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=mistudio
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mistudio

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  postgres_data:
  redis_data:
```

---

## Documentation

### Documentation Structure

```
docs/
├── README.md                   # Documentation index
│
├── architecture/               # Architecture documentation
│   ├── overview.md
│   ├── frontend.md
│   ├── backend.md
│   ├── data-flow.md
│   └── deployment.md
│
├── api/                        # API documentation
│   ├── rest-api.md             # REST API endpoints
│   ├── websocket-api.md        # WebSocket protocol
│   ├── authentication.md       # Auth flow
│   └── rate-limiting.md        # Rate limiting
│
├── ml/                         # ML documentation
│   ├── sparse-autoencoders.md
│   ├── feature-extraction.md
│   ├── feature-steering.md
│   └── training-guide.md
│
├── development/                # Development guides
│   ├── setup.md                # Local setup
│   ├── contributing.md         # Contribution guidelines
│   ├── testing.md              # Testing guide
│   └── debugging.md            # Debugging tips
│
├── deployment/                 # Deployment guides
│   ├── docker.md               # Docker deployment
│   ├── kubernetes.md           # Kubernetes (future)
│   ├── jetson.md               # Jetson-specific setup
│   └── monitoring.md           # Monitoring and logging
│
└── user-guide/                 # End-user documentation
    ├── quickstart.md
    ├── datasets.md
    ├── training.md
    ├── features.md
    └── steering.md
```

---

## Development Tools

### Scripts Directory

```
scripts/
├── dev/                        # Development scripts
│   ├── setup.sh                # Initial project setup
│   ├── db-reset.sh             # Reset database
│   ├── seed-data.sh            # Seed test data
│   └── lint.sh                 # Run linters
│
├── build/                      # Build scripts
│   ├── build-frontend.sh
│   ├── build-backend.sh
│   └── build-all.sh
│
├── deploy/                     # Deployment scripts
│   ├── deploy-dev.sh
│   ├── deploy-staging.sh
│   └── deploy-prod.sh
│
├── maintenance/                # Maintenance scripts
│   ├── backup.sh               # Backup data and database
│   ├── restore.sh              # Restore from backup
│   ├── cleanup.sh              # Clean up old data
│   └── monitor.sh              # System monitoring
│
└── test/                       # Test scripts
    ├── run-unit-tests.sh
    ├── run-integration-tests.sh
    └── run-e2e-tests.sh
```

### GitHub Actions Workflows

```
.github/
├── workflows/
│   ├── frontend-ci.yml         # Frontend CI pipeline
│   ├── backend-ci.yml          # Backend CI pipeline
│   ├── deploy-staging.yml      # Deploy to staging
│   ├── deploy-prod.yml         # Deploy to production
│   └── security-scan.yml       # Security scanning
│
├── ISSUE_TEMPLATE/
│   ├── bug_report.md
│   └── feature_request.md
│
└── PULL_REQUEST_TEMPLATE.md
```

---

## File Count Summary

| Section | File Count | Notes |
|---------|-----------|-------|
| Frontend Source | ~175 | .tsx, .ts, .css files |
| Backend Source | ~150 | .py files |
| Tests (Frontend) | ~50 | .test.tsx, .test.ts |
| Tests (Backend) | ~50 | test_*.py |
| Configuration | ~30 | .json, .yml, .toml, .ini |
| Documentation | ~20 | .md files |
| Scripts | ~20 | .sh, .py |
| Docker | ~10 | Dockerfile, docker-compose |
| **TOTAL SOURCE** | **~505 files** | Excluding generated, node_modules, __pycache__ |

### Excluded from Count

- `node_modules/` (~40,000 files)
- `__pycache__/` (~500 files)
- `.venv/` or `venv/` (~10,000 files)
- `data/` (varies greatly)
- Build outputs (`dist/`, `build/`)
- Generated files (coverage reports, etc.)

---

## Quick Reference: Key Paths

### Frontend Key Files
```
frontend/src/main.tsx                      # Entry point
frontend/src/services/api/client.ts        # API client
frontend/src/components/training/TrainingPanel/  # Main training UI
frontend/src/hooks/api/useTrainingProgress.ts    # Real-time updates
```

### Backend Key Files
```
backend/app/main.py                        # Entry point
backend/app/api/v1/endpoints/training.py   # Training API
backend/app/workers/tasks/training_tasks.py  # Training worker
backend/app/ml/models/sparse_autoencoder.py  # SAE implementation
```

### Data Key Paths
```
data/models/quantized/                     # Quantized models
data/activations/                          # Activation tensors
data/trainings/{training_id}/encoder.pt    # Trained SAE
data/features/{training_id}/               # Extracted features
```

---

## Related Documentation

- [Backend Implementation Guide](./Mock-embedded-interp-ui.tsx) (lines 4-313)
- [Redis Usage Guide](./000_SPEC|REDIS_GUIDANCE_USECASE.md)
- [Refactoring Roadmap](./REFACTORING_ROADMAP.md)
- [Technical Specification](./miStudio_Specification.md)
- [OpenAPI Specification](./openapi.yaml)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-05
**Maintained By**: MechInterp Studio Team
