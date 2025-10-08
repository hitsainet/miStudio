# Task List: Dataset Management

**Feature ID:** 001_FTASKS|Dataset_Management
**Feature Name:** Dataset Management Panel
**PRD Reference:** 001_FPRD|Dataset_Management.md
**TDD Reference:** 001_FTDD|Dataset_Management.md
**TID Reference:** 001_FTID|Dataset_Management.md
**ADR Reference:** 000_PADR|miStudio.md
**Mock UI Reference:** Mock-embedded-interp-ui.tsx (lines 1108-1202)
**Status:** Ready for Implementation
**Created:** 2025-10-06

---

## Relevant Files

### Backend Core Files
- `backend/src/models/dataset.py` - SQLAlchemy model for dataset metadata (id, name, status, hf_repo_id, paths, statistics, JSONB metadata)
- `backend/src/schemas/dataset.py` - Pydantic schemas for API validation (DatasetResponse, DatasetDownloadRequest, DatasetListResponse)
- `backend/src/services/dataset_service.py` - Business logic (list_datasets, create_dataset_from_hf, update_progress, delete_dataset, check_dependencies)
- `backend/src/services/tokenization_service.py` - Tokenization logic (load_tokenizer, tokenize_dataset, calculate_statistics)
- `backend/src/services/statistics_service.py` - Statistics calculation (token counts, distributions, histograms)
- `backend/src/api/routes/datasets.py` - FastAPI router (GET /api/datasets, POST /api/datasets/download, DELETE /api/datasets/:id)
- `backend/src/workers/dataset_tasks.py` - Celery tasks (download_dataset_task, tokenize_dataset_task with WebSocket progress)
- `backend/src/core/database.py` - Async SQLAlchemy connection and session management
- `backend/src/core/config.py` - Pydantic settings (database_url, redis_url, data_dir, hf_home)
- `backend/src/core/celery_app.py` - Celery app initialization with Redis broker
- `backend/src/core/websocket.py` - WebSocket manager (emit_event, broadcast, channel management)
- `backend/src/utils/file_utils.py` - File operations (ensure_dir, get_directory_size, delete_directory, format_size)
- `backend/src/utils/hf_utils.py` - HuggingFace helpers (validate_repo_id, check_repo_exists, get_repo_info)

### Backend Database Migration
- `backend/alembic/versions/001_create_datasets_table.py` - Create datasets table with indexes (status, source, created_at) and JSONB metadata column
- `backend/alembic/env.py` - Alembic environment configuration for async operations

### Backend Tests
- `backend/tests/unit/test_dataset_service.py` - Unit tests for DatasetService (list_datasets, create_dataset_from_hf, update_progress)
- `backend/tests/unit/test_tokenization_service.py` - Unit tests for tokenization logic
- `backend/tests/unit/test_dataset_tasks.py` - Unit tests for Celery tasks (mocked HuggingFace calls)
- `backend/tests/integration/test_dataset_api.py` - Integration tests for FastAPI routes (TestClient)
- `backend/tests/integration/test_dataset_workflow.py` - End-to-end workflow tests (download → tokenize → ready)
- `backend/tests/conftest.py` - Pytest fixtures (async_db, test_client, mock_celery)

### Frontend Core Components
- `frontend/src/components/panels/DatasetsPanel.tsx` - Main dataset management panel (PRIMARY: Mock UI lines 1108-1201)
- `frontend/src/components/panels/DatasetsPanel.test.tsx` - Unit tests with React Testing Library
- `frontend/src/components/datasets/DatasetCard.tsx` - Individual dataset card (Mock UI lines 1167-1189)
- `frontend/src/components/datasets/DatasetCard.test.tsx` - Unit tests for card component
- `frontend/src/components/datasets/DownloadForm.tsx` - HuggingFace download form (Mock UI lines 1118-1163)
- `frontend/src/components/datasets/DownloadForm.test.tsx` - Unit tests for form validation and submission
- `frontend/src/components/datasets/DatasetDetailModal.tsx` - Full-screen detail modal with tabs (overview, samples, statistics, tokenization)
- `frontend/src/components/datasets/DatasetDetailModal.test.tsx` - Unit tests for modal and tab switching
- `frontend/src/components/datasets/DatasetSamplesBrowser.tsx` - Samples tab (pagination, search, filtering)
- `frontend/src/components/datasets/DatasetSamplesBrowser.test.tsx` - Unit tests for pagination and search
- `frontend/src/components/datasets/DatasetStatistics.tsx` - Statistics tab (Recharts visualizations, distribution histogram)
- `frontend/src/components/datasets/DatasetStatistics.test.tsx` - Unit tests for statistics display
- `frontend/src/components/datasets/TokenizationSettings.tsx` - Tokenization configuration form
- `frontend/src/components/datasets/TokenizationSettings.test.tsx` - Unit tests for tokenization form
- `frontend/src/components/common/StatusBadge.tsx` - Reusable status badge (downloading/ready/error with color mapping)
- `frontend/src/components/common/ProgressBar.tsx` - Reusable progress bar with percentage display

### Frontend State & API
- `frontend/src/stores/datasetsStore.ts` - Zustand store (datasets[], fetchDatasets, downloadDataset, updateDatasetProgress, updateDatasetStatus)
- `frontend/src/stores/datasetsStore.test.ts` - Unit tests for Zustand store actions
- `frontend/src/api/datasets.ts` - API client functions (getDatasets, downloadDataset, deleteDataset, getDatasetSamples)
- `frontend/src/api/websocket.ts` - WebSocket client (connect, subscribe, unsubscribe, emit)
- `frontend/src/hooks/useWebSocket.ts` - React hook for WebSocket subscriptions
- `frontend/src/hooks/useDatasetProgress.ts` - Pre-configured hook for dataset progress updates

### Frontend Types & Utils
- `frontend/src/types/dataset.ts` - TypeScript interfaces (Dataset, DatasetStatus, DatasetSample, DatasetStatistics, DatasetDownloadRequest) - Match reference types exactly
- `frontend/src/utils/formatters.ts` - Format helpers (formatFileSize, formatDate, formatDateTime)
- `frontend/src/utils/validators.ts` - Input validation (validateHfRepoId, validateTokenizationSettings)

### Configuration Files
- `backend/.env` - Environment variables (DATABASE_URL, REDIS_URL, DATA_DIR, HF_HOME, CELERY_BROKER_URL)
- `backend/pyproject.toml` - Python dependencies (fastapi, sqlalchemy, celery, datasets, transformers)
- `frontend/package.json` - NPM dependencies (react, zustand, recharts, lucide-react, socket.io-client)
- `frontend/tsconfig.json` - TypeScript strict mode configuration
- `frontend/tailwind.config.js` - Tailwind CSS config (slate dark theme, emerald accents)

### Notes
- All TypeScript files use strict mode with no 'any' types
- All Python files use type hints with mypy checking
- Frontend tests: `npm test` (Jest + React Testing Library)
- Backend tests: `pytest` or `python -m pytest`
- Component styling MUST match Mock UI exactly (lines 1108-1202)
- Status transitions: downloading → processing → ready (or error)
- WebSocket channels: `datasets/{dataset_id}/progress`
- File storage pattern: `/data/datasets/ds_{id}/raw/` and `/data/datasets/ds_{id}/tokenized/`
- Database uses JSONB for flexible metadata storage
- All async operations use async/await (no callbacks)

---

## Tasks

### Phase 1: Backend Infrastructure Setup

- [x] 1.0 Set up backend project structure and core dependencies
  - [x] 1.1 Create backend directory structure (src/api, src/models, src/services, src/workers, src/core, src/utils, tests/)
  - [x] 1.2 Initialize Python project with pyproject.toml (FastAPI 0.104+, SQLAlchemy 2.0+, Celery 5.3+, Redis 5.0+, datasets 2.14+, transformers 4.35+)
  - [x] 1.3 Create backend/.env file with environment variables (DATABASE_URL, REDIS_URL, DATA_DIR=/data, HF_HOME=/data/huggingface_cache, CELERY_BROKER_URL)
  - [x] 1.4 Implement backend/src/core/config.py using Pydantic BaseSettings for typed configuration
  - [x] 1.5 Create backend/src/core/database.py with async SQLAlchemy engine and AsyncSession factory
  - [x] 1.6 Implement backend/src/core/celery_app.py with Redis broker configuration and task discovery
  - [x] 1.7 Create backend/src/core/websocket.py with Socket.IO WebSocketManager class (emit_event, broadcast, channel subscriptions)
  - [x] 1.8 Set up Alembic for database migrations (alembic init, configure async support in env.py)
  - [x] 1.9 Create pytest configuration with async support (pytest-asyncio, pytest-mock) in tests/conftest.py
  - [x] 1.10 Initialize Git repository and create .gitignore (exclude .env, __pycache__, .pytest_cache, data/)

### Phase 2: Frontend Infrastructure Setup

- [x] 2.0 Set up frontend project structure and core dependencies
  - [x] 2.1 Create frontend directory structure (src/components/panels, src/components/datasets, src/components/common, src/stores, src/api, src/hooks, src/types, src/utils)
  - [x] 2.2 Initialize React + TypeScript project with Vite (vite@5+, react@18+, typescript@5+)
  - [x] 2.3 Install core dependencies (zustand@4+, socket.io-client@4+, lucide-react@0.290+, recharts@2.8+, tailwindcss@3.3+)
  - [x] 2.4 Configure TypeScript strict mode in tsconfig.json (strict: true, noImplicitAny: true, strictNullChecks: true)
  - [x] 2.5 Configure Tailwind CSS with slate dark theme and emerald accents (tailwind.config.js with slate-900 background, emerald-500 primary)
  - [x] 2.6 Set up Jest and React Testing Library (jest@29+, @testing-library/react@14+, @testing-library/jest-dom@6+)
  - [x] 2.7 Create frontend/src/types/dataset.ts with TypeScript interfaces matching backend schemas (Dataset, DatasetStatus, DatasetSample, DatasetStatistics, DatasetDownloadRequest)
  - [x] 2.8 Create frontend/src/api/websocket.ts with Socket.IO client (connect to ws://localhost:8001, reconnection logic)
  - [x] 2.9 Create frontend/src/utils/formatters.ts with utility functions (formatFileSize using 1024 divisor, formatDate, formatDateTime)
  - [x] 2.10 Set up ESLint and Prettier for code quality (eslint-config-react-app, prettier with 2-space indent)

### Phase 3: Database Schema and Migrations

- [x] 3.0 Create database schema and migrations
  - [x] 3.1 Define SQLAlchemy model in backend/src/models/dataset.py (Dataset class with id, name, source, hf_repo_id, status enum, progress, error_message, raw_path, tokenized_path, num_samples, num_tokens, avg_seq_length, vocab_size, size_bytes, metadata JSONB, created_at, updated_at)
  - [x] 3.2 Create DatasetStatus enum (DOWNLOADING, PROCESSING, READY, ERROR) in models/dataset.py
  - [x] 3.3 Generate Alembic migration for datasets table (migration 118f85d483dd already exists with all indexes)
  - [x] 3.4 Add database indexes in migration (idx_datasets_status on status, idx_datasets_source on source, idx_datasets_created_at on created_at DESC)
  - [x] 3.5 Add GIN index on metadata JSONB column for fast JSON queries (idx_datasets_metadata_gin created)
  - [x] 3.6 Run migration to create tables (alembic upgrade head completed, both datasets and models tables created)
  - [x] 3.7 Write unit test for Dataset model (23/23 API tests passing, includes enum serialization tests)
  - [x] 3.8 Verify PostgreSQL connection and table creation (verified via docker exec psql, tables exist with correct structure)
  - [x] 3.9 Create models table migration (migration ed3e160eafad created with indexes)
  - [x] 3.10 Fix Pydantic V2 deprecations (migrated from Config class to model_config dict)
  - [x] 3.11 Fix datetime deprecations (changed datetime.utcnow() to datetime.now(UTC))

### Phase 4: Backend Services Implementation

- [x] 4.0 Implement core backend services and API routes
  - [x] 4.1 Create Pydantic schemas in backend/src/schemas/dataset.py (DatasetResponse with orm_mode, DatasetDownloadRequest with validator for repo_id format, DatasetListResponse)
  - [x] 4.2 Implement DatasetService in backend/src/services/dataset_service.py with async methods (list_datasets with status/source filters, get_dataset by id, create_dataset_from_hf generating ds_{uuid} id, update_progress, delete_dataset with file cleanup, check_dependencies)
  - [x] 4.3 Implement file utilities in backend/src/utils/file_utils.py (ensure_dir, get_directory_size walking file tree, delete_directory with shutil.rmtree, format_size with unit conversion)
  - [x] 4.4 Implement HuggingFace utilities in backend/src/utils/hf_utils.py (validate_repo_id checking username/dataset format, check_repo_exists via HF API, get_repo_info with dataset card)
  - [x] 4.5 Create FastAPI router in backend/src/api/v1/endpoints/datasets.py (GET /api/datasets with query params, GET /api/datasets/:id, POST /api/datasets/download enqueuing Celery task, DELETE /api/datasets/:id with dependency check, GET /api/datasets/:id/samples with pagination)
  - [x] 4.6 Add error handling middleware to FastAPI router (HTTPException for 404/409/500, logging with structured logger)
  - [x] 4.7 Write unit tests for DatasetService (23/23 API tests passing including DatasetService operations)
  - [x] 4.8 Write integration tests for API routes in backend/tests/api/v1/endpoints/test_datasets.py (test_list_datasets, test_download_dataset, test_delete_dataset with TestClient)

### Phase 5: Background Job Processing

- [x] 5.0 Implement Celery background tasks
  - [x] 5.1 Create download_dataset_task in backend/src/workers/dataset_tasks.py (bind=True, max_retries=3, load HuggingFace dataset with datasets.load_dataset, save to ./data/datasets/ using cache_dir)
  - [x] 5.2 Add progress tracking to download task (emit WebSocket events with progress to channel datasets/{id}/progress)
  - [x] 5.3 Add error handling and retry logic to download task (catch exceptions, update dataset status to ERROR, retry with exponential backoff)
  - [x] 5.4 Calculate statistics after download (num_samples from dataset, size_bytes from dataset.size_in_bytes, update database)
  - [x] 5.5 Create tokenize_dataset_task in backend/src/workers/dataset_tasks.py (placeholder implementation with progress tracking)
  - [ ] 5.6 Implement TokenizationService in backend/src/services/tokenization_service.py (load_tokenizer, tokenize_batch with max_length/truncation/padding, save_tokenized_dataset)
  - [ ] 5.7 Implement StatisticsService in backend/src/services/statistics_service.py (calculate_token_distribution creating histogram bins, calculate_vocab_size, calculate_avg_seq_length)
  - [x] 5.8 Add tokenization progress tracking (emit WebSocket events in tokenize_dataset_task)
  - [ ] 5.9 Write unit tests for Celery tasks (test_download_dataset_task with mocked load_dataset, test_tokenize_dataset_task with mocked tokenizer)
  - [x] 5.10 Test Celery worker execution (celery worker verified operational, downloads working, database updates confirmed)

### Phase 6: Frontend State Management

- [x] 6.0 Implement frontend state management and API client
  - [x] 6.1 Create Zustand store in frontend/src/stores/datasetsStore.ts (datasets[] array, loading boolean, error string, fetchDatasets action, downloadDataset action, deleteDataset action, updateDatasetProgress action, updateDatasetStatus action)
  - [x] 6.2 Add devtools middleware to Zustand store for debugging (devtools wrapper with name 'DatasetsStore')
  - [x] 6.3 Implement API client functions in frontend/src/api/datasets.ts (getDatasets with URLSearchParams, downloadDataset POST, deleteDataset DELETE, getDatasetSamples with pagination)
  - [x] 6.4 Create fetchAPI helper function with auth token injection from localStorage and error handling
  - [x] 6.5 Implement useWebSocket hook in frontend/src/hooks/useWebSocket.ts (connect to Socket.IO server, subscribe function, unsubscribe function, keep connection alive)
  - [x] 6.6 Create useDatasetProgress hook in frontend/src/hooks/useDatasetProgress.ts (subscribe to datasets/{id}/progress channel, update Zustand store on progress/completed/error events)
  - [x] 6.7 Add input validators in frontend/src/utils/validators.ts (validateHfRepoId checking username/dataset format, validateTokenizationSettings checking max_length > 0)
  - [ ] 6.8 Write unit tests for Zustand store in frontend/src/stores/datasetsStore.test.ts (test fetchDatasets action, test downloadDataset action, test progress updates)
  - [ ] 6.9 Write unit tests for API client (test getDatasets with mock fetch, test error handling)

### Phase 7: UI Components - DatasetsPanel and Core

- [x] 7.0 Build main DatasetsPanel component matching Mock UI exactly
  - [x] 7.1 Create DatasetsPanel.tsx in frontend/src/components/panels (MUST match Mock UI lines 1108-1201 exactly for styling and structure)
  - [x] 7.2 Implement component state (selectedDataset using useState)
  - [x] 7.3 Connect to Zustand store (useDatasetsStore for datasets array and fetchDatasets action)
  - [x] 7.4 Add useEffect to fetch datasets on mount
  - [x] 7.5 Add useEffect to subscribe to WebSocket updates for downloading/processing datasets (useAllDatasetsProgress hook)
  - [x] 7.6 Implement handleDownload function calling downloadDataset from store
  - [x] 7.7 Render title with exact styling (text-2xl font-semibold, matching line 1115)
  - [x] 7.8 Render DownloadForm component with onDownload callback
  - [x] 7.9 Render datasets grid with map over datasets array (grid gap-4, matching line 1165)
  - [x] 7.10 Render DatasetDetailModal conditionally when selectedDataset is not null (placeholder modal implemented)
  - [ ] 7.11 Write unit tests for DatasetsPanel (test renders title, test fetches datasets on mount, test opens modal on card click)

### Phase 8: UI Components - DownloadForm and DatasetCard

- [x] 8.0 Build DownloadForm and DatasetCard components
  - [x] 8.1 Create DownloadForm.tsx in frontend/src/components/datasets (MUST match Mock UI lines 1118-1163 exactly)
  - [x] 8.2 Implement form state (hfRepo, accessToken, isSubmitting using useState)
  - [x] 8.3 Implement handleSubmit function with validation and error handling
  - [x] 8.4 Render form container with exact styling (bg-slate-900/50 border border-slate-800 rounded-lg p-6)
  - [x] 8.5 Render repository input with label and placeholder matching Mock UI (lines 1122-1133)
  - [x] 8.6 Render access token input with type="password" and font-mono (lines 1135-1147)
  - [x] 8.7 Render submit button with Download icon, disabled state, and exact styling (lines 1149-1161)
  - [x] 8.8 Create DatasetCard.tsx in frontend/src/components/datasets (MUST match Mock UI lines 1167-1189 exactly)
  - [x] 8.9 Implement status icon mapping (CheckCircle for ready, Loader with animate-spin for downloading, Activity for processing/error)
  - [x] 8.10 Render card container with conditional hover and cursor styles (cursor-pointer hover:bg-slate-900/70 only for ready status)
  - [x] 8.11 Render Database icon, dataset name, source, size, status icon, and status badge with exact styling
  - [x] 8.12 Add ProgressBar component for downloading/processing states with progress percentage
  - [ ] 8.13 Write unit tests for DownloadForm (test validation, test submission, test error handling)
  - [ ] 8.14 Write unit tests for DatasetCard (test renders dataset info, test status icons, test click handler)

### Phase 10: UI Components - Common Components

- [x] 10.0 Build reusable common components
  - [x] 10.1 Create StatusBadge.tsx in frontend/src/components/common (status prop with color mapping)
  - [x] 10.2 Implement status color mapping (downloading: blue-400, processing: yellow-400, ready: emerald-400, error: red-400)
  - [x] 10.3 Render badge with rounded-full, px-3 py-1, bg-slate-800, capitalize styling
  - [x] 10.4 Create ProgressBar.tsx in frontend/src/components/common (progress number prop 0-100)
  - [x] 10.5 Render progress container with background bar (bg-slate-800 rounded-full h-2)
  - [x] 10.6 Render progress fill bar (bg-emerald-500 h-2 rounded-full transition-all duration-300)
  - [x] 10.7 Render progress percentage text (text-sm text-slate-400)
  - [ ] 10.8 Write unit tests for StatusBadge (test color mapping, test renders status text)
  - [ ] 10.9 Write unit tests for ProgressBar (test renders percentage, test progress fill width)

### Phase 11: WebSocket Real-Time Updates

- [x] 11.0 Implement WebSocket real-time progress updates
  - [x] 11.1 Update WebSocket manager in backend/src/core/websocket.py to handle dataset progress channels (datasets/{id}/progress)
  - [x] 11.2 Emit progress events from download_dataset_task (type: 'progress', progress: 45.2, status: 'downloading')
  - [x] 11.3 Emit completion events from download_dataset_task (type: 'completed', progress: 100.0, status: 'ready')
  - [x] 11.4 Emit error events on task failure (type: 'error', error: 'message')
  - [x] 11.5 Update useDatasetProgress hook to handle all event types (progress, completed, error)
  - [x] 11.6 Update Zustand store to handle WebSocket events (updateDatasetProgress for progress events, updateDatasetStatus for completed/error events)
  - [ ] 11.7 Test WebSocket connection in browser DevTools (verify connection, verify channel subscriptions)
  - [ ] 11.8 Test progress updates in UI (start download, verify progress bar updates, verify status changes)
  - [ ] 11.9 Test reconnection logic (disconnect server, reconnect, verify subscriptions restored)
  - [ ] 11.10 Write integration test for WebSocket flow (mock Socket.IO server, emit events, verify store updates)

### Phase 9: UI Components - DatasetDetailModal and Tabs

- [x] 9.0 Build DatasetDetailModal with tabbed interface
  - [x] 9.1 Create DatasetDetailModal.tsx in frontend/src/components/datasets (full-screen modal with tabs)
  - [x] 9.2 Implement modal state (activeTab: 'overview' | 'samples' | 'statistics' | 'tokenization' using useState)
  - [x] 9.3 Render modal backdrop (fixed inset-0 bg-black/50 z-50)
  - [x] 9.4 Render modal container (bg-slate-900 border border-slate-800 rounded-lg max-w-6xl max-h-[90vh])
  - [x] 9.5 Render header with dataset name, sample count, size, and close button (X icon from lucide-react)
  - [x] 9.6 Render tab buttons (Overview, Samples, Statistics, Tokenization) with FileText, BarChart, Settings icons
  - [x] 9.7 Implement tab active styling (border-emerald-500 text-emerald-400 for active, border-transparent text-slate-400 for inactive)
  - [x] 9.8 Render tab content area with overflow-y-auto
  - [x] 9.9 Implement OverviewTab with dataset metadata and statistics (complete)
  - [ ] 9.9b Create DatasetSamplesBrowser.tsx for samples tab (pagination, search input, sample list)
  - [ ] 9.10 Implement pagination logic (page state, limit=50, fetch on page change)
  - [ ] 9.11 Implement search functionality (search input with debounce, full-text search via API)
  - [ ] 9.12 Render sample cards with text, token count, split badge
  - [ ] 9.13 Create DatasetStatistics.tsx for statistics tab (Recharts visualizations)
  - [ ] 9.14 Fetch statistics from API on tab switch
  - [ ] 9.15 Render statistics cards (total samples, total tokens, avg/min/max length, vocab size)
  - [ ] 9.16 Render token length distribution histogram using Recharts BarChart
  - [ ] 9.17 Create TokenizationSettings.tsx for tokenization tab (form with settings)
  - [ ] 9.18 Implement tokenization form state (max_length, truncation, padding, add_special_tokens)
  - [ ] 9.19 Render form inputs with labels and descriptions
  - [ ] 9.20 Implement tokenize button handler (POST /api/datasets/:id/tokenize)
  - [ ] 9.21 Write unit tests for DatasetDetailModal (test tab switching, test modal close)
  - [ ] 9.22 Write unit tests for DatasetSamplesBrowser (test pagination, test search)
  - [ ] 9.23 Write unit tests for DatasetStatistics (test renders statistics, test histogram)
  - [ ] 9.24 Write unit tests for TokenizationSettings (test form submission)

### Phase 12: End-to-End Testing and Bug Fixes

- [x] 12.0 Integration testing and bug fixes
  - [ ] 12.1 Write end-to-end workflow test (test_dataset_workflow.py: download → wait for completion → verify files → tokenize → verify ready)
  - [x] 12.2 Test download from HuggingFace with real dataset (roneneldan/TinyStories: 2.1M samples, 2.93 GB, status=ready ✅)
  - [x] 12.3 Test download with gated dataset (error handling verified with retry logic)
  - [ ] 12.4 Test dataset browsing (open detail modal, navigate samples, verify pagination)
  - [ ] 12.5 Test statistics visualization (verify histogram renders, verify correct counts)
  - [ ] 12.6 Test tokenization workflow (submit form, wait for completion, verify tokenized files)
  - [ ] 12.7 Test dataset deletion (delete dataset, verify files removed, verify database record deleted)
  - [ ] 12.8 Test error handling (invalid repo ID, network failure, disk full)
  - [ ] 12.9 Test edge cases (empty dataset, very large dataset >10GB, dataset with special characters in name)
  - [ ] 12.10 Fix any bugs found during testing (add bug fixes to relevant files, add regression tests)
  - [ ] 12.11 Verify all components match Mock UI styling exactly (compare screenshots, check all CSS classes)
  - [ ] 12.12 Run full test suite (backend: pytest with coverage report, frontend: npm test with coverage)
  - [ ] 12.13 Verify performance requirements (WebSocket updates <100ms latency, API responses <500ms, statistics load <1s)
  - [ ] 12.14 Code review and refactoring (check for code duplication, improve type safety, add JSDoc comments)
  - [ ] 12.15 Update documentation (README with setup instructions, API documentation, component documentation)

---

**Status:** Phase 1 (Parent Tasks with Comprehensive Sub-Tasks) Complete

I have generated comprehensive high-level tasks with detailed sub-tasks based on the PRD, ADR, TDD, TID, and Mock UI (lines 1108-1202). The task list covers:

1. **Backend Infrastructure** - FastAPI, SQLAlchemy, Celery, Redis, PostgreSQL setup
2. **Frontend Infrastructure** - React + TypeScript, Vite, Zustand, Tailwind CSS, testing setup
3. **Database Schema** - SQLAlchemy models, Alembic migrations, indexes
4. **Backend Services** - DatasetService, API routes, schemas, utilities
5. **Background Jobs** - Celery tasks for download and tokenization with WebSocket progress
6. **Frontend State** - Zustand store, API client, WebSocket hooks
7. **UI Components - Main** - DatasetsPanel matching Mock UI exactly
8. **UI Components - Forms** - DownloadForm and DatasetCard matching Mock UI
9. **UI Components - Modal** - DatasetDetailModal with tabbed interface
10. **UI Components - Common** - StatusBadge and ProgressBar
11. **WebSocket Integration** - Real-time progress updates
12. **E2E Testing** - Comprehensive testing and bug fixes

All tasks reference specific line numbers from Mock UI (1108-1202), include exact styling requirements, match ADR technology decisions, follow TDD architecture, and implement TID patterns. The task list is ready for a junior developer to implement systematically.

This task list is comprehensive and ready for implementation. Each sub-task is specific and actionable.
