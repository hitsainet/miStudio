# Feature PRD: Dataset Management

**Feature ID:** 001_FPRD|Dataset_Management
**Feature Name:** Dataset Management Panel
**Priority:** P0 (Blocker for MVP)
**Status:** ✅ Complete (MVP Implementation)
**Created:** 2025-10-05
**Last Updated:** 2025-10-11 (Post-implementation review)

---

## 1. Feature Overview

### Feature Description
The Dataset Management Panel provides complete lifecycle management for text datasets used in Sparse Autoencoder (SAE) training. Users can download datasets from HuggingFace, ingest local datasets, tokenize them for model compatibility, browse dataset contents, and view statistics—all through a unified interface optimized for edge AI deployment on Jetson Orin Nano.

### Problem Statement
Training SAEs requires high-quality text datasets, but managing datasets on edge devices presents unique challenges:
- Limited storage capacity requires careful dataset selection and management
- Network bandwidth constraints make large downloads problematic
- Users need visibility into dataset contents and statistics before committing to training
- Tokenization must be model-specific and efficient for edge hardware
- No cloud dependencies—all operations must work on local edge devices

### Feature Goals
1. **Simplify Dataset Acquisition**: Enable one-click downloads from HuggingFace with progress tracking
2. **Provide Dataset Visibility**: Allow users to browse, search, and analyze datasets before training
3. **Ensure Compatibility**: Validate and tokenize datasets for specific model architectures
4. **Optimize Edge Performance**: Efficient storage and processing for resource-constrained devices
5. **Support Reproducibility**: Track dataset versions, preprocessing steps, and statistics

### Connection to Project Objectives
This feature directly supports the project's core objective of democratizing mechanistic interpretability research by:
- Removing cloud dependencies (datasets stored and processed locally)
- Optimizing for edge hardware (Jetson Orin Nano with limited RAM/storage)
- Providing educational visibility into dataset structure and statistics
- Enabling reproducible experiments with version-tracked datasets

---

## 2. User Stories & Scenarios

### Primary User Stories

#### US-1: Download Dataset from HuggingFace
**As a** ML researcher
**I want to** download datasets directly from HuggingFace within the application
**So that** I can quickly acquire training data without manual download/upload steps

**Acceptance Criteria:**
- User can enter a HuggingFace repository ID (e.g., "roneneldan/TinyStories")
- System validates repository exists and is accessible
- Download progress displays percentage and estimated time remaining
- System handles gated datasets requiring access tokens
- Downloaded datasets automatically appear in dataset list with "ingesting" status
- System gracefully handles network interruptions with resume capability
- User receives clear error messages for invalid repos or authentication failures

#### US-2: Browse Dataset Samples
**As a** ML researcher
**I want to** browse and search through dataset samples before training
**So that** I can verify dataset quality and understand the text domain

**Acceptance Criteria:**
- User can click on a "ready" dataset to open detail modal
- Detail modal displays paginated list of samples (50 per page)
- Each sample shows: full text, token count, split (train/validation/test), and metadata
- User can search samples using full-text search
- User can filter samples by split (train/validation/test)
- User can filter by token length ranges (e.g., 100-500 tokens)
- Search and filtering are performant (<500ms response time)

#### US-3: View Dataset Statistics
**As a** ML researcher
**I want to** view comprehensive statistics about a dataset
**So that** I can assess if it's suitable for my training needs

**Acceptance Criteria:**
- Statistics panel displays:
  - Total samples (by split: train/validation/test)
  - Total tokens across dataset
  - Token count distribution (histogram)
  - Average, median, min, max sequence length
  - Vocabulary size
  - Storage size (raw and tokenized)
- Statistics compute during ingestion and cache for quick access
- Visualizations render within 1 second of opening detail modal
- Statistics update if dataset is re-tokenized

#### US-4: Tokenize Dataset for Model
**As a** ML researcher
**I want to** tokenize a dataset using a specific model's tokenizer
**So that** the dataset is preprocessed and ready for training

**Acceptance Criteria:**
- User selects a dataset and a model from dropdowns
- System displays tokenization settings form:
  - Max sequence length (default: model's max_position_embeddings)
  - Truncation strategy (default: true)
  - Padding strategy (options: max_length, longest, do_not_pad)
  - Add special tokens (default: true)
- User can save tokenization settings as a preset for reuse
- Tokenization runs as background job with progress updates
- System validates model tokenizer is compatible
- Tokenized dataset appears as new entry with reference to source dataset
- Statistics recompute for tokenized version

#### US-5: Delete Unused Datasets
**As a** ML researcher
**I want to** delete datasets I no longer need
**So that** I can free up storage space on my edge device

**Acceptance Criteria:**
- User can click delete button on dataset card
- System displays confirmation modal showing:
  - Dataset name and size
  - Warning if dataset is referenced by any trainings (prevent deletion)
  - Storage space to be freed
- After confirmation, system deletes:
  - Raw dataset files from `/data/datasets/raw/`
  - Tokenized files from `/data/datasets/tokenized/`
  - Database metadata record
- System handles deletion of tokenized variants (cascade)
- UI updates immediately after successful deletion
- System prevents deletion if dataset is actively being used by training

### Secondary User Scenarios

#### SC-1: Ingest Local Dataset
**Scenario:** User has a custom text dataset (CSV, JSON, or text files) they want to ingest

**Flow:**
1. User clicks "Upload Local Dataset" button
2. System opens file picker (accepts .csv, .json, .jsonl, .txt)
3. User selects file(s) and confirms upload
4. System validates file format and size (<10GB limit)
5. System ingests data with progress updates
6. Dataset appears in list with "ingesting" status
7. After ingestion, statistics compute automatically
8. Dataset transitions to "ready" status

**Acceptance Criteria:**
- Supports CSV (text column), JSON/JSONL (text field), and plain text files
- Validates file structure and displays helpful error messages
- Progress bar shows upload and ingestion progress
- System handles large files (>1GB) without memory overflow
- Ingested datasets receive auto-generated IDs (e.g., "ds_local_abc123")

#### SC-2: Resume Interrupted Download
**Scenario:** Network interruption during HuggingFace download

**Flow:**
1. Dataset shows "downloading" status with progress at 45%
2. Network interruption occurs
3. System detects failure and transitions dataset to "error" status
4. User clicks "Retry Download" button
5. System resumes download from last completed chunk
6. Download continues from 45% to completion
7. Dataset transitions to "ingesting" status

**Acceptance Criteria:**
- System uses resumable downloads (HTTP Range requests)
- Partial downloads persist to temporary storage
- Retry button appears for datasets in "error" status
- Maximum 3 automatic retry attempts with exponential backoff
- User can manually trigger retry after automatic retries exhausted

### Edge Cases and Error Scenarios

#### EC-1: Insufficient Storage Space
**Scenario:** User attempts to download dataset but device has insufficient space

**Handling:**
- System checks available disk space before starting download
- If insufficient, display error: "Insufficient storage: Dataset requires 2.3GB but only 800MB available. Please free up space or delete unused datasets."
- Provide link to storage management panel showing all datasets and their sizes
- Prevent download initiation if storage check fails

#### EC-2: Invalid HuggingFace Repository
**Scenario:** User enters non-existent or inaccessible HuggingFace repo ID

**Handling:**
- System validates repository exists via HuggingFace API before starting download
- If not found (404), display: "Repository 'user/invalid-repo' not found. Please check the repository name and try again."
- If authentication required, display: "Repository requires authentication. Please provide a valid HuggingFace access token."
- If forbidden (403), display: "Access denied. Please request access to this gated dataset on HuggingFace."

#### EC-3: Corrupted Dataset Files
**Scenario:** Dataset files become corrupted (disk error, interrupted write)

**Handling:**
- System validates file integrity during ingestion using checksums
- If corruption detected, transition dataset to "error" status
- Display error: "Dataset files are corrupted. Please re-download or re-upload the dataset."
- Provide "Delete and Re-download" button to clean up and retry

#### EC-4: Tokenization Failure
**Scenario:** Tokenizer fails due to incompatible text encoding or unexpected format

**Handling:**
- Catch tokenization errors and log detailed error messages
- Display user-friendly error: "Tokenization failed: Unable to process sample 1,234. This may indicate incompatible text encoding or format."
- Allow user to adjust tokenization settings (encoding, error handling strategy)
- Provide "Skip Invalid Samples" option to continue tokenization despite errors

### User Journey Flows

#### Journey 1: First-Time Dataset Download
```
1. User opens Dataset Management panel (empty state)
   ↓
2. User reads "No datasets yet. Download from HuggingFace to get started."
   ↓
3. User enters "roneneldan/TinyStories" in repository field
   ↓
4. User clicks "Download Dataset from HuggingFace"
   ↓
5. System validates repository (success)
   ↓
6. Dataset card appears with "downloading" status and progress bar (0% → 100%)
   ↓
7. Status transitions to "ingesting" (computing statistics)
   ↓
8. Status transitions to "ready" after 30 seconds
   ↓
9. User clicks on dataset card to open detail modal
   ↓
10. User browses samples and views statistics
    ↓
11. User closes modal, proceeds to Model Management to prepare for training
```

#### Journey 2: Dataset Search and Filtering
```
1. User has 5 ready datasets
   ↓
2. User clicks on "OpenWebText-10K" dataset
   ↓
3. Detail modal opens showing first 50 samples
   ↓
4. User enters "artificial intelligence" in search box
   ↓
5. System performs full-text search (results appear in <300ms)
   ↓
6. User views 12 matching samples
   ↓
7. User filters by token count range (500-1000 tokens)
   ↓
8. Results narrow to 4 samples
   ↓
9. User clicks on sample to view full text with token highlighting
   ↓
10. User confirms dataset is suitable for AI safety research
    ↓
11. User closes modal, proceeds to training configuration
```

---

## 3. Functional Requirements

### FR-1: HuggingFace Dataset Download
1. **FR-1.1**: System shall provide input field for HuggingFace repository ID (format: "org/repo-name")
2. **FR-1.2**: System shall provide optional input field for HuggingFace access token (password-masked)
3. **FR-1.3**: System shall validate repository existence via HuggingFace API before download
4. **FR-1.4**: System shall display estimated download size after validation
5. **FR-1.5**: System shall initiate download via HuggingFace Datasets library
6. **FR-1.6**: System shall display real-time progress: percentage, downloaded bytes, estimated time remaining
7. **FR-1.7**: System shall support resume capability for interrupted downloads (HTTP Range requests)
8. **FR-1.8**: System shall store raw dataset files in `/data/datasets/raw/{dataset_id}/`
9. **FR-1.9**: System shall create database record with status "downloading" before starting
10. **FR-1.10**: System shall update database record to "ingesting" upon download completion
11. **FR-1.11**: System shall handle gated datasets requiring explicit access permissions
12. **FR-1.12**: System shall implement automatic retry (3 attempts) with exponential backoff on transient failures
13. **FR-1.13**: Split Selection - System shall allow users to specify dataset split (train, validation, test) via download form
   - **Implementation**: `DownloadForm.tsx` lines 70-82 with split dropdown
   - **Backend**: `datasets.py:download_dataset()` lines 242-247 passes split to Celery task
   - **Status**: ✅ COMPLETE
14. **FR-1.14**: Configuration Selection - System shall support multi-configuration datasets (e.g., language variants)
   - **Implementation**: `DownloadForm.tsx` lines 84-96 with config dropdown
   - **Backend**: `datasets.py:download_dataset()` lines 242-247 passes config to Celery task
   - **Status**: ✅ COMPLETE

### FR-2: Dataset Ingestion and Validation
1. **FR-2.1**: System shall support HuggingFace datasets in Parquet and Arrow formats
2. **FR-2.2**: System shall extract text field from dataset (auto-detect: "text", "content", "document")
3. **FR-2.3**: System shall identify dataset splits (train, validation, test) if present
4. **FR-2.4**: System shall compute dataset statistics during ingestion:
   - Total samples (per split)
   - Total tokens (using whitespace tokenization for estimation)
   - Token count distribution (histogram with 20 bins)
   - Min, max, median, average sequence length
   - Vocabulary size (unique tokens)
5. **FR-2.5**: System shall store statistics in database as JSONB field
6. **FR-2.6**: System shall validate text encoding (UTF-8) and handle encoding errors gracefully
7. **FR-2.7**: System shall detect and report corrupted files using checksum validation
8. **FR-2.8**: System shall transition dataset to "ready" status after successful ingestion
9. **FR-2.9**: System shall transition dataset to "error" status if ingestion fails with error message
10. **FR-2.10**: Ingestion shall be performed as background Celery task to avoid blocking API

### FR-3: Dataset Tokenization
1. **FR-3.1**: System shall provide tokenization form with fields:
   - Model selection (dropdown of ready models)
   - Max sequence length (integer input, default: model's max_position)
   - Truncation (boolean toggle, default: true)
   - Padding strategy (dropdown: max_length, longest, do_not_pad)
   - Add special tokens (boolean toggle, default: true)
2. **FR-3.2**: System shall load tokenizer from selected model using HuggingFace Transformers
3. **FR-3.3**: System shall tokenize all dataset splits in parallel batches (batch_size=1000)
4. **FR-3.4**: System shall save tokenized dataset in Arrow format to `/data/datasets/tokenized/{dataset_id}_tokenized/`
5. **FR-3.5**: System shall create new database record for tokenized dataset with reference to source dataset
6. **FR-3.6**: System shall compute token-level statistics for tokenized dataset:
   - Actual token counts (post-tokenization)
   - Token ID distribution
   - Special token frequencies (PAD, CLS, SEP, etc.)
7. **FR-3.7**: System shall display tokenization progress (percentage, samples processed)
8. **FR-3.8**: System shall support saving tokenization settings as reusable templates
9. **FR-3.9**: System shall run tokenization as background Celery task
10. **FR-3.10**: System shall handle tokenization errors gracefully with option to skip invalid samples

### FR-4: Dataset Browsing and Search
1. **FR-4.1**: System shall display dataset list with cards showing:
   - Dataset name
   - Source (HuggingFace, Local, Custom)
   - Size (human-readable: GB, MB)
   - Status badge (downloading, ingesting, ready, error)
   - Progress bar (if downloading/ingesting)
2. **FR-4.2**: System shall enable clicking on "ready" datasets to open detail modal
3. **FR-4.3**: Detail modal shall display tabbed interface:
   - **Samples Tab**: Paginated sample browser
   - **Statistics Tab**: Statistics visualizations
   - **Settings Tab**: Tokenization and metadata
4. **FR-4.4**: Samples tab shall display table with columns:
   - Sample index
   - Text preview (first 200 characters)
   - Token count
   - Split (train/validation/test)
   - Expand button to view full text
5. **FR-4.5**: System shall implement pagination (50 samples per page)
6. **FR-4.6**: Full-text search → ⏸️ **DEFERRED TO PHASE 12**
   - **Reason**: Basic pagination sufficient for MVP; search adds complexity without blocking core workflows
7. **FR-4.7**: Advanced filtering (by split, token length) → ⏸️ **DEFERRED TO PHASE 12**
   - **Reason**: Users can browse all samples; filtering is optimization, not requirement
8. **FR-4.8**: Search and filtering shall update results within 500ms → ⏸️ **DEFERRED TO PHASE 12**
9. **FR-4.9**: System shall highlight search terms in sample text → ⏸️ **DEFERRED TO PHASE 12**
10. **FR-4.10**: PostgreSQL GIN indexes for metadata search → ⏸️ **DEFERRED TO PHASE 12**
    - **Reason**: Depends on FR-4.6/4.7 implementation; not needed for pagination

### FR-5: Dataset Statistics Visualization
1. **FR-5.1**: Statistics tab shall display metrics grid:
   - Total samples (with split breakdown)
   - Total tokens
   - Vocabulary size
   - Average tokens per sample
   - Median tokens per sample
   - Min/Max tokens per sample
2. **FR-5.2**: System shall display token count distribution histogram:
   - X-axis: Token count ranges (20 bins)
   - Y-axis: Number of samples
   - Interactive hover to show exact counts
3. **FR-5.3**: System shall display split distribution pie chart (if multiple splits exist)
4. **FR-5.4**: System shall display storage breakdown:
   - Raw dataset size
   - Tokenized dataset size (if exists)
   - Total storage used
5. **FR-5.5**: Visualizations shall use Recharts library for consistency with Mock UI
6. **FR-5.6**: Visualizations shall render within 1 second of tab open
7. **FR-5.7**: System shall cache statistics in database to avoid recomputation

### FR-6: Dataset Deletion and Cleanup
1. **FR-6.1**: System shall provide delete button on each dataset card
2. **FR-6.2**: System shall check for training dependencies before allowing deletion:
   - Query `trainings` table for references to dataset_id
   - If references exist and training is not in "completed" or "stopped" status, block deletion
3. **FR-6.3**: System shall display confirmation modal with:
   - Dataset name and size
   - Warning message if trainings reference this dataset
   - Storage space to be freed
   - "Delete" and "Cancel" buttons
4. **FR-6.4**: Upon confirmation, system shall:
   - Delete raw dataset files from `/data/datasets/raw/{dataset_id}/`
   - Delete tokenized files from `/data/datasets/tokenized/{dataset_id}_tokenized/`
   - Delete database record (CASCADE deletes related tokenized variants)
   - Update UI immediately (remove card with fade-out animation)
5. **FR-6.5**: System shall log deletion events for audit trail
6. **FR-6.6**: System shall handle partial deletion failures gracefully (orphaned files, missing files)
7. **FR-6.7**: Bulk deletion (select multiple datasets) → ⏸️ **DEFERRED (Future Enhancement)**
   - **Reason**: Single-dataset deletion sufficient for MVP; bulk operations are optimization

### FR-7: Local Dataset Upload → ⏸️ **DEFERRED (Not in MVP)**

**Deferral Reason:** HuggingFace integration covers 95% of use cases for MVP. Local upload adds significant complexity (file validation, format detection, schema inference) without blocking core research workflows. Planned for Phase 2.

**Original Requirements** (for future reference):
1. **FR-7.1**: System shall provide "Upload Local Dataset" button
2. **FR-7.2**: System shall open file picker accepting: .csv, .json, .jsonl, .txt
3. **FR-7.3**: System shall validate file size (<10GB limit) before upload
4. **FR-7.4**: System shall upload file with progress tracking (chunked upload)
5. **FR-7.5**: System shall ingest uploaded file:
   - CSV: Extract column named "text" or first text column
   - JSON/JSONL: Extract field named "text" or "content"
   - TXT: Split by newlines (one sample per line)
6. **FR-7.6**: System shall assign auto-generated ID (e.g., "ds_local_abc123")
7. **FR-7.7**: System shall follow same ingestion workflow as HuggingFace datasets
8. **FR-7.8**: System shall store uploaded file in `/data/datasets/raw/{dataset_id}/uploaded.{ext}`

### FR-8: Status Tracking and Progress Updates
1. **FR-8.1**: System shall implement dataset status state machine:
   - `downloading` → `ingesting` → `ready`
   - `downloading` → `error` (on download failure)
   - `ingesting` → `error` (on ingestion failure)
2. **FR-8.2**: System shall store progress percentage (0-100) in database
3. **FR-8.3**: System shall use polling (every 2 seconds) to fetch status updates for active downloads
4. **FR-8.4**: System shall display progress bar on dataset cards for `downloading` and `ingesting` statuses
5. **FR-8.5**: System shall show animated spinner for `downloading` and `ingesting` statuses
6. **FR-8.6**: System shall show checkmark icon for `ready` status
7. **FR-8.7**: System shall show error icon with tooltip displaying error message for `error` status
8. **FR-8.8**: System shall provide "Retry Download" button for datasets in `error` status
9. **FR-8.9**: System shall log all status transitions with timestamps for debugging

---

## 4. User Experience Requirements

### UI/UX Specifications

#### Primary Reference: Mock UI
All UI/UX specifications reference the PRIMARY authoritative specification:
**@0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx**

Specifically, **DatasetsPanel component** (lines 1108-1202) defines exact look and behavior.

#### Visual Design Standards
- **Theme**: Dark mode (slate-950 background, slate-100 text)
- **Primary Color**: Emerald-400 for success states, active elements
- **Secondary Colors**:
  - Blue-400 for downloading/processing states
  - Yellow-400 for warnings/ingesting states
  - Red-400 for error states
- **Typography**: System font stack, text-sm (14px) for body, text-lg (18px) for headings
- **Spacing**: Consistent 4px grid (gap-2, gap-4, gap-6)
- **Borders**: border-slate-800 for subtle divisions
- **Shadows**: Minimal, backdrop-blur for modals

#### Layout Structure
```
┌─────────────────────────────────────────────────────┐
│ Dataset Management                                  │
│ [Download from HuggingFace card]                   │
│                                                     │
│ ┌─────────────────────────────────────────────┐   │
│ │ HuggingFace Repository:                     │   │
│ │ [roneneldan/TinyStories          ]          │   │
│ │                                              │   │
│ │ Access Token (optional):                    │   │
│ │ [••••••••••••••••••••••••       ]          │   │
│ │                                              │   │
│ │ [Download Dataset from HuggingFace]         │   │
│ └─────────────────────────────────────────────┘   │
│                                                     │
│ ┌─────────────────────────────────────────────┐   │
│ │ [Database icon] OpenWebText-10K              │   │
│ │ Source: HuggingFace • Size: 2.3GB           │   │
│ │                                [✓] Ready     │   │
│ └─────────────────────────────────────────────┘   │
│                                                     │
│ ┌─────────────────────────────────────────────┐   │
│ │ [Database icon] TinyStories                  │   │
│ │ Source: HuggingFace • Size: 450MB           │   │
│ │                                [✓] Ready     │   │
│ └─────────────────────────────────────────────┘   │
│                                                     │
│ ┌─────────────────────────────────────────────┐   │
│ │ [Database icon] CodeParrot-Small             │   │
│ │ Source: HuggingFace • Size: 1.8GB           │   │
│ │ [Progress: 67%]            [⟳] Ingesting    │   │
│ └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

#### Interactive Patterns

1. **Download Form**
   - Input fields: `bg-slate-900 border-slate-700 rounded-lg focus:border-emerald-500`
   - Password input: `type="password"` for access token
   - Submit button: `bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700`
   - Disabled state when repo field empty

2. **Dataset Cards**
   - Clickable cards: `hover:bg-slate-900/70 transition-colors cursor-pointer`
   - Only "ready" datasets are clickable
   - Non-ready datasets (downloading/ingesting) have `cursor-default`

3. **Status Indicators**
   - Ready: `<CheckCircle className="w-5 h-5 text-emerald-400" />`
   - Downloading: `<Loader className="w-5 h-5 text-blue-400 animate-spin" />`
   - Ingesting: `<Activity className="w-5 h-5 text-yellow-400" />`
   - Error: `<X className="w-5 h-5 text-red-400" />`

4. **Progress Bars**
   - Container: `h-2 bg-slate-800 rounded-full overflow-hidden`
   - Fill: `bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-500`
   - Smooth transitions using CSS custom properties: `style={{ '--width': '${progress}%' }}`

5. **Icons** (from lucide-react)
   - Database: `<Database className="w-8 h-8 text-blue-400" />`
   - Download: `<Download className="w-5 h-5" />`
   - CheckCircle: Success states
   - Loader: Processing states
   - Activity: Active processes
   - X: Close/error states

#### Responsive Design
- Container: `max-w-7xl mx-auto px-6 py-8`
- Mobile breakpoints: Tailwind responsive classes (sm:, md:, lg:)
- Dataset cards: Full-width on mobile, grid on desktop (responsive grid-cols)

#### Accessibility Requirements
- **Keyboard Navigation**: All interactive elements accessible via Tab/Enter
- **ARIA Labels**:
  - `<button aria-label="Delete dataset">` for icon-only buttons
  - `<input id="dataset-repo-input">` with corresponding `<label htmlFor="">`
- **Focus Indicators**: `focus:outline-none focus:border-emerald-500` for inputs
- **Screen Reader Support**:
  - Status badges announce state changes
  - Progress bars include `aria-valuenow`, `aria-valuemin`, `aria-valuemax`
  - Modal dialogs use `role="dialog" aria-modal="true"`
- **Color Contrast**: All text meets WCAG AA standards (4.5:1 contrast ratio)

#### Performance Expectations
- **Initial Load**: Dataset list appears within 200ms
- **Status Polling**: Updates every 2 seconds without UI flicker
- **Search/Filter**: Results update within 300ms of user input
- **Modal Open**: Detail modal renders within 100ms
- **Statistics Render**: Charts and visualizations complete within 1 second
- **Smooth Animations**: 60fps transitions for progress bars and card interactions

---

## 5. Data Requirements

### Data Models

#### Dataset Model (PostgreSQL)
Based on **@0xcc/project-specs/infrastructure/003_SPEC|Postgres_Usecase_Details_and_Guidance.md** (lines 136-171)

```sql
CREATE TABLE datasets (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'ds_abc123'
    name VARCHAR(500) NOT NULL,
    source VARCHAR(50) NOT NULL,  -- 'HuggingFace', 'Local', 'Custom'
    description TEXT,
    size_bytes BIGINT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'downloading',
        -- Values: 'downloading', 'ingesting', 'tokenizing', 'ready', 'error'
    progress FLOAT,  -- 0-100
    error_message TEXT,
    file_path VARCHAR(1000),  -- /data/datasets/raw/{id}/
    tokenized_path VARCHAR(1000),  -- /data/datasets/tokenized/{id}/

    -- Statistics (populated after ingestion)
    num_samples INTEGER,
    num_tokens BIGINT,
    vocab_size INTEGER,
    avg_sequence_length FLOAT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT datasets_status_check CHECK (status IN
        ('downloading', 'ingesting', 'tokenizing', 'ready', 'error'))
);
```

#### Dataset Statistics (JSONB field in datasets table)
Stored in additional JSONB column `statistics`:

```typescript
interface DatasetStatistics {
  total_samples: number;
  total_tokens: number;
  avg_tokens_per_sample: number;
  median_tokens_per_sample: number;
  min_tokens: number;
  max_tokens: number;
  unique_tokens: number;
  splits: {
    train: number;
    validation: number;
    test: number;
  };
  token_distribution: Array<{
    range: string;  // e.g., "0-100", "100-200"
    count: number;
  }>;
}
```

### Data Validation Rules

1. **Repository ID Validation**
   - Format: `^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$`
   - Example: "roneneldan/TinyStories"
   - Cannot be empty or whitespace-only
   - Max length: 500 characters

2. **Access Token Validation**
   - Optional field (can be empty)
   - Format: `^hf_[a-zA-Z0-9]{20,}$` (HuggingFace token format)
   - Masked in UI with password input type
   - Not stored in database (only used for API requests)

3. **File Path Validation**
   - Must be absolute path starting with `/data/`
   - Prevent path traversal: No `..` sequences
   - Max length: 1000 characters

4. **Size Validation**
   - size_bytes: Integer >= 0
   - Display human-readable format: GB, MB, KB
   - Warning if dataset > 50GB (edge device constraint)

5. **Progress Validation**
   - Float between 0.0 and 100.0 (inclusive)
   - Only valid for statuses: downloading, ingesting, tokenizing
   - Must be null for statuses: ready, error

6. **Status Validation**
   - Must be one of: downloading, ingesting, tokenizing, ready, error
   - Status transitions must follow state machine (see FR-8.1)

### Data Persistence

#### File Storage Structure
Based on **@0xcc/project-specs/infrastructure/001_SPEC|Folder_File_Details.md** (lines 644-668)

```
/data/datasets/
├── raw/                    # Original downloaded datasets
│   ├── dataset_ds_abc123/
│   │   ├── train.parquet
│   │   ├── validation.parquet
│   │   ├── test.parquet
│   │   └── dataset_info.json
│   │
│   └── dataset_ds_xyz456/
│       └── ...
│
├── tokenized/              # Tokenized datasets (Arrow format)
│   ├── dataset_ds_abc123_tokenized/
│   │   ├── train.arrow
│   │   ├── validation.arrow
│   │   └── metadata.json
│   │
│   └── dataset_ds_xyz456_tokenized/
│       └── ...
│
└── metadata/               # Dataset metadata and statistics
    ├── ds_abc123_stats.json
    └── ds_xyz456_stats.json
```

#### Database Indexes
```sql
CREATE INDEX idx_datasets_status ON datasets(status);
CREATE INDEX idx_datasets_created_at ON datasets(created_at DESC);
CREATE INDEX idx_datasets_source ON datasets(source);

-- Full-text search index
CREATE INDEX idx_datasets_search ON datasets
    USING GIN(to_tsvector('english', name || ' ' || COALESCE(description, '')));
```

### Data Migration Considerations
- **Initial Migration**: Create `datasets` table with all columns and constraints
- **V2 Migration** (if needed): Add `statistics` JSONB column for rich statistics storage
- **V3 Migration** (if needed): Add `tokenized_path` column for tokenization support
- **Backward Compatibility**: Old datasets without statistics display "Computing..." in UI

---

## 6. Technical Constraints

### Technology Stack Constraints (from ADR)

#### Backend Framework
- **FastAPI**: Async Python framework for API endpoints
- **Pydantic**: Data validation and serialization
- **SQLAlchemy**: ORM for database operations
- **Alembic**: Database migrations

#### ML/Data Libraries
- **HuggingFace Datasets**: Download and load datasets from HuggingFace Hub
- **HuggingFace Transformers**: Tokenization using model-specific tokenizers
- **PyArrow**: Efficient storage format for tokenized datasets
- **Pandas**: Data manipulation and statistics computation

#### Job Queue
- **Celery**: Background task processing for downloads and ingestion
- **Redis**: Celery broker and result backend

#### Storage
- **Local Filesystem**: `/data/` directory for all persistent data
- **PostgreSQL 14+**: Metadata storage with JSONB support

### ADR Decision References

#### Decision 1: Backend Framework (FastAPI)
- **Rationale**: Async support for concurrent downloads, automatic OpenAPI docs
- **Constraint**: All API endpoints must be async-compatible
- **Impact**: Use `async def` for dataset download/ingestion endpoints

#### Decision 3: Database (PostgreSQL with JSONB)
- **Rationale**: JSONB for flexible statistics storage, full-text search capabilities
- **Constraint**: Statistics must be stored as JSONB, not separate columns
- **Impact**: Query statistics using PostgreSQL JSON operators (`->`, `->>`)

#### Decision 6: Job Queue (Celery + Redis)
- **Rationale**: Background processing for long-running downloads/ingestion
- **Constraint**: Downloads and ingestion must be Celery tasks, not synchronous API calls
- **Impact**: API returns 202 Accepted with job_id, client polls for status

#### Decision 7: ML/AI Stack (PyTorch + HuggingFace)
- **Rationale**: HuggingFace ecosystem for datasets and tokenizers
- **Constraint**: Must use `datasets.load_dataset()` for HuggingFace downloads
- **Impact**: Limited to datasets supported by HuggingFace Datasets library

#### Decision 8: Storage (Local Filesystem)
- **Rationale**: No cloud dependencies, edge-optimized
- **Constraint**: All data stored in `/data/` directory with organized structure
- **Impact**: Must implement disk space checks before downloads

#### Decision 11: Edge Optimization
- **Rationale**: Optimize for Jetson Orin Nano (8GB RAM, limited storage)
- **Constraints**:
  - Max concurrent downloads: 1
  - Streaming downloads to avoid memory overflow
  - Disk space monitoring and warnings
  - Efficient Arrow format for tokenized data (50% smaller than Parquet)

#### Decision 13: Deployment (nginx reverse proxy)
- **Rationale**: nginx handles HTTP routing, rate limiting
- **Constraint**: Dataset download endpoints must be accessible via `/api/datasets/`
- **Impact**: API URLs: `http://mistudio.mcslab.io/api/datasets/download`

### Performance Requirements

#### Response Times (p95)
- **GET /api/datasets**: <100ms (list all datasets)
- **POST /api/datasets/download**: <500ms (start background job, return job_id)
- **GET /api/datasets/:id**: <100ms (get single dataset metadata)
- **GET /api/datasets/:id/samples**: <500ms (paginated samples with search/filter)
- **GET /api/datasets/:id/statistics**: <200ms (cached statistics from database)

#### Throughput
- **Concurrent Downloads**: 1 (GPU/disk limitation on edge device)
- **API Requests**: 100/sec (nginx + FastAPI on Jetson)
- **Sample Browsing**: 50 samples/page, pagination for large datasets

#### Storage
- **Dataset Sizes**: 500MB - 50GB each (typical text datasets)
- **Total Storage Budget**: ~500GB for datasets (user-managed)
- **Tokenized Overhead**: +50% storage for tokenized versions
- **Database Metadata**: ~1KB per dataset (<1MB for 1000 datasets)

#### Network
- **Download Speed**: Limited by network (10-100 Mbps typical on edge)
- **Resume Capability**: HTTP Range requests for interrupted downloads
- **Bandwidth Monitoring**: Warning if download exceeds 10GB

### Security Requirements

#### Input Validation
- Sanitize all user inputs (repository IDs, file paths)
- Prevent path traversal attacks (`..` sequences blocked)
- Validate file uploads (size, MIME type, extension)
- Rate limit download requests (10/hour per endpoint)

#### Authentication
- HuggingFace access tokens handled securely:
  - Never stored in database
  - Masked in UI (password input type)
  - Passed only to HuggingFace API via HTTPS
- No user authentication (single-user system on edge device)

#### File System Security
- Restrict file operations to `/data/datasets/` directory
- Validate file paths before read/write operations
- Set appropriate file permissions (chmod 644 for data files)
- Prevent overwriting existing datasets without explicit confirmation

#### Network Security
- HTTPS for HuggingFace API requests (TLS 1.2+)
- Validate SSL certificates
- Timeout for hung connections (30 seconds)
- Retry logic for transient network errors (max 3 retries)

---

## 7. API/Integration Specifications

### Backend API Endpoints

Based on **@0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx** (lines 314-321)

#### Endpoint 1: List Datasets
```http
GET /api/datasets
```

**Query Parameters:**
- `page` (optional, integer, default=1): Page number for pagination
- `limit` (optional, integer, default=50): Items per page
- `status` (optional, string): Filter by status (downloading, ready, etc.)
- `source` (optional, string): Filter by source (HuggingFace, Local)
- `search` (optional, string): Full-text search query

**Response (200 OK):**
```json
{
  "data": [
    {
      "id": "ds_abc123",
      "name": "OpenWebText-10K",
      "source": "HuggingFace",
      "size": "2.3GB",
      "size_bytes": 2469606195,
      "status": "ready",
      "progress": null,
      "num_samples": 10000,
      "num_tokens": 54231000,
      "created_at": "2025-10-05T10:00:00Z",
      "updated_at": "2025-10-05T10:15:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 5,
    "has_next": false,
    "has_prev": false
  }
}
```

**Error Responses:**
- `400 Bad Request`: Invalid query parameters
- `500 Internal Server Error`: Database error

---

#### Endpoint 2: Get Dataset Details
```http
GET /api/datasets/:id
```

**Path Parameters:**
- `id` (required, string): Dataset ID (e.g., "ds_abc123")

**Response (200 OK):**
```json
{
  "data": {
    "id": "ds_abc123",
    "name": "OpenWebText-10K",
    "source": "HuggingFace",
    "description": "A subset of OpenWebText for efficient SAE training",
    "size": "2.3GB",
    "size_bytes": 2469606195,
    "status": "ready",
    "progress": null,
    "error_message": null,
    "file_path": "/data/datasets/raw/ds_abc123/",
    "tokenized_path": "/data/datasets/tokenized/ds_abc123_tokenized/",
    "num_samples": 10000,
    "num_tokens": 54231000,
    "vocab_size": 50257,
    "avg_sequence_length": 542.31,
    "statistics": {
      "total_samples": 10000,
      "splits": {
        "train": 8000,
        "validation": 1000,
        "test": 1000
      },
      "token_distribution": [
        {"range": "0-100", "count": 234},
        {"range": "100-200", "count": 567},
        // ... more bins
      ]
    },
    "created_at": "2025-10-05T10:00:00Z",
    "updated_at": "2025-10-05T10:15:00Z"
  }
}
```

**Error Responses:**
- `404 Not Found`: Dataset ID does not exist
- `500 Internal Server Error`: Database error

---

#### Endpoint 3: Download Dataset from HuggingFace
```http
POST /api/datasets/download
```

**Request Body:**
```json
{
  "repo_id": "roneneldan/TinyStories",
  "access_token": "hf_xxxxxxxxxxxxxxxxxxxx"  // optional
}
```

**Request Validation:**
- `repo_id`: Required, string, format `org/repo-name`, max 500 chars
- `access_token`: Optional, string, format `hf_[a-zA-Z0-9]{20,}`

**Response (202 Accepted):**
```json
{
  "data": {
    "dataset_id": "ds_xyz456",
    "status": "downloading",
    "message": "Dataset download started"
  },
  "meta": {
    "job_id": "job_abc123",
    "estimated_duration_seconds": 300
  }
}
```

**Error Responses:**
- `400 Bad Request`: Invalid repo_id format or missing field
  ```json
  {
    "error": {
      "code": "VALIDATION_ERROR",
      "message": "Invalid repository ID format. Expected 'org/repo-name'.",
      "details": {
        "field": "repo_id",
        "value": "invalid-repo"
      },
      "retryable": false
    }
  }
  ```
- `404 Not Found`: HuggingFace repository does not exist
- `403 Forbidden`: Access token required for gated dataset
- `409 Conflict`: Dataset already exists
- `429 Rate Limit Exceeded`: Too many download requests
- `503 Service Unavailable`: Disk space insufficient

---

#### Endpoint 4: Get Dataset Samples (Paginated)
```http
GET /api/datasets/:id/samples
```

**Query Parameters:**
- `page` (optional, integer, default=1): Page number
- `limit` (optional, integer, default=50, max=100): Samples per page
- `split` (optional, string): Filter by split (train, validation, test)
- `search` (optional, string): Full-text search query
- `min_tokens` (optional, integer): Minimum token count filter
- `max_tokens` (optional, integer): Maximum token count filter

**Response (200 OK):**
```json
{
  "data": [
    {
      "id": 0,
      "text": "Once upon a time, there was a little girl named Lily...",
      "tokens": ["Once", "upon", "a", "time", ",", "there", "was", "a", "little", "girl", "named", "Lily", "..."],
      "token_count": 234,
      "split": "train",
      "metadata": {
        "source": "books",
        "domain": "children"
      }
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 10000,
    "has_next": true,
    "has_prev": false
  }
}
```

**Error Responses:**
- `404 Not Found`: Dataset ID does not exist
- `400 Bad Request`: Invalid query parameters

---

#### Endpoint 5: Delete Dataset
```http
DELETE /api/datasets/:id
```

**Path Parameters:**
- `id` (required, string): Dataset ID

**Response (200 OK):**
```json
{
  "data": {
    "message": "Dataset deleted successfully",
    "freed_bytes": 2469606195
  }
}
```

**Error Responses:**
- `404 Not Found`: Dataset ID does not exist
- `409 Conflict`: Dataset is referenced by active training
  ```json
  {
    "error": {
      "code": "DELETE_CONFLICT",
      "message": "Cannot delete dataset: referenced by 2 active trainings",
      "details": {
        "training_ids": ["tr_abc123", "tr_xyz456"]
      },
      "retryable": false
    }
  }
  ```
- `500 Internal Server Error`: File deletion failed

---

### Internal Service Integration

#### HuggingFace Datasets Library
```python
from datasets import load_dataset

# Download dataset with progress tracking
dataset = load_dataset(
    repo_id,
    cache_dir=f"/data/datasets/raw/{dataset_id}",
    token=access_token,
    # Progress callback for UI updates
    download_config=DownloadConfig(
        resume_download=True,
        force_download=False
    )
)
```

#### HuggingFace Transformers Tokenizer
```python
from transformers import AutoTokenizer

# Load tokenizer from model
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    cache_dir="/data/models/tokenizers/"
)

# Tokenize dataset
tokenized_dataset = dataset.map(
    lambda examples: tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=padding_strategy,
        add_special_tokens=add_special_tokens
    ),
    batched=True,
    batch_size=1000,
    num_proc=4  # Parallel processing
)

# Save tokenized dataset
tokenized_dataset.save_to_disk(
    f"/data/datasets/tokenized/{dataset_id}_tokenized/"
)
```

#### Celery Background Tasks
```python
from celery import shared_task

@shared_task(bind=True, max_retries=3)
def download_dataset_task(self, dataset_id, repo_id, access_token=None):
    try:
        # Update status to downloading
        update_dataset_status(dataset_id, "downloading", progress=0)

        # Download with progress updates
        dataset = load_dataset(repo_id, token=access_token)

        # Update status to ingesting
        update_dataset_status(dataset_id, "ingesting", progress=100)

        # Compute statistics
        stats = compute_dataset_statistics(dataset)

        # Update database with stats and ready status
        update_dataset_stats(dataset_id, stats)
        update_dataset_status(dataset_id, "ready")

    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
```

---

## 8. Non-Functional Requirements

### NFR-1: Performance

#### Load Time
- Initial dataset list render: <200ms from API request to UI display
- Dataset detail modal open: <100ms (modal animation + initial render)
- Statistics tab render: <1 second (including chart rendering)
- Samples tab pagination: <300ms between page changes

#### Throughput
- API can handle 100 requests/second on Jetson Orin Nano
- Concurrent downloads: 1 (enforced by job queue)
- Sample browsing: Support up to 1 million samples per dataset with pagination

#### Scalability
- Database can store metadata for 1000+ datasets without performance degradation
- Full-text search performs well (<500ms) with 10,000+ dataset samples
- Statistics computation scales linearly with dataset size (1GB/minute ingestion rate)

### NFR-2: Reliability

#### Availability
- Dataset API endpoints: 99.5% uptime (excluding planned maintenance)
- Graceful degradation: If database unavailable, return cached dataset list from Redis

#### Error Handling
- All API errors return structured error responses (code, message, details)
- Retry logic for transient network errors (exponential backoff, max 3 retries)
- Resume capability for interrupted downloads (HTTP Range requests)
- Database transaction rollback on partial failures

#### Data Integrity
- Checksums validate downloaded files against HuggingFace metadata
- Database foreign key constraints prevent orphaned records
- File operations use atomic writes (write to temp, then rename)

### NFR-3: Usability

#### Learnability
- First-time users can download a dataset within 2 minutes without documentation
- Clear, contextual error messages guide users to resolution
- Empty state messages provide actionable next steps

#### Efficiency
- Power users can download, browse, and configure datasets using only keyboard
- Batch operations (delete multiple datasets) reduce repetitive actions
- Recently used datasets appear at top of list

#### Error Prevention
- Confirmation modal for destructive actions (delete dataset)
- Warnings for datasets referenced by active trainings
- Disk space check before initiating downloads
- Clear status indicators prevent user confusion

### NFR-4: Maintainability

#### Code Organization
- Feature follows ADR naming conventions: `DatasetService`, `DatasetRepository`, `DatasetSchema`
- All dataset logic isolated in `backend/app/services/dataset_service.py`
- API endpoints in `backend/app/api/v1/endpoints/datasets.py`
- Frontend components in `frontend/src/components/datasets/`

#### Testing
- Unit test coverage: ≥80% for business logic (dataset service, statistics computation)
- Integration tests: All API endpoints with mock database
- E2E tests: Complete download workflow (using small test dataset)

#### Logging
- Structured logs (JSON format) for all dataset operations
- Log levels: DEBUG (file operations), INFO (status changes), ERROR (failures)
- Include: dataset_id, operation, duration, error details

#### Documentation
- API endpoints documented in OpenAPI spec
- Code comments for complex statistics algorithms
- README in `backend/app/services/` explaining dataset workflow

---

## 9. Feature Boundaries (Non-Goals)

### Explicitly Out of Scope

1. **Custom Preprocessing Pipelines**
   - **Not Included**: User-defined text cleaning, normalization, or transformation steps
   - **Rationale**: Adds complexity; users can preprocess externally and upload
   - **Future Consideration**: Template-based preprocessing in Phase 2

2. **Dataset Versioning**
   - **Not Included**: Track multiple versions of same dataset with diff/merge capabilities
   - **Rationale**: Edge storage constraints; users manage versions manually
   - **Future Consideration**: Lightweight version tagging in Phase 3

3. **Collaborative Dataset Sharing**
   - **Not Included**: Share datasets between multiple miStudio instances or users
   - **Rationale**: Single-user system on edge device
   - **Future Consideration**: Export/import dataset bundles for sharing

4. **Advanced Data Augmentation**
   - **Not Included**: Synthetic data generation, back-translation, paraphrasing
   - **Rationale**: Requires additional ML models and compute; out of scope for MVP
   - **Future Consideration**: Plugin system for data augmentation extensions

5. **Multi-Modal Datasets**
   - **Not Included**: Images, audio, video, or mixed-modality datasets
   - **Rationale**: Focus on text-only datasets for SAE training on language models
   - **Future Consideration**: Vision model support in Phase 4

6. **Dataset Annotations/Labels**
   - **Not Included**: User annotations, labels, or tags on dataset samples
   - **Rationale**: SAE training is unsupervised; annotations not required
   - **Future Consideration**: Annotation support for supervised fine-tuning features

7. **External Cloud Storage Integration**
   - **Not Included**: S3, Google Cloud Storage, Azure Blob integration
   - **Rationale**: Contradicts edge-first design philosophy; all data local
   - **Future Consideration**: Hybrid cloud backup in Phase 5

8. **Real-Time Dataset Streaming**
   - **Not Included**: Stream datasets from external sources during training
   - **Rationale**: Requires stable network; edge devices may be offline
   - **Future Consideration**: Optional streaming mode for high-bandwidth setups

9. **Dataset Quality Metrics**
   - **Not Included**: Automated quality scoring (toxicity, bias, diversity)
   - **Rationale**: Requires additional ML models; computationally expensive
   - **Future Consideration**: Integrate lightweight quality scoring in Phase 3

10. **Dataset Merging/Splitting**
    - **Not Included**: Combine multiple datasets or split dataset into subsets
    - **Rationale**: Users can perform these operations externally before upload
    - **Future Consideration**: Simple merge/split UI in Phase 2

### Related Features Handled Separately

1. **Tokenization Templates**
   - Covered in separate feature: "Training Templates" (004_SPEC|Template_System_Guidance.md)
   - Dataset Management only implements ad-hoc tokenization

2. **Model-Specific Tokenizers**
   - Handled by Model Management feature (002_FPRD|Model_Management.md)
   - Dataset Management consumes tokenizers, doesn't manage them

3. **Training Data Selection**
   - Handled by Training Configuration feature (003_FPRD|SAE_Training.md)
   - Dataset Management provides browsing; Training feature handles selection

4. **Feature Activation Extraction**
   - Handled by Feature Discovery feature (004_FPRD|Feature_Discovery.md)
   - Dataset Management provides data; Feature Discovery extracts activations

### Technical Limitations Accepted

1. **No Delta Updates**: Re-downloading updated HuggingFace datasets downloads entire dataset (no incremental updates)
2. **Limited Format Support**: Supports HuggingFace formats (Parquet, Arrow); no direct support for CSV/JSON (requires local upload)
3. **No Streaming Tokenization**: Entire dataset tokenized and saved; no lazy tokenization during training
4. **Fixed Statistics**: Statistics computed once during ingestion; manual refresh required after edits

---

## 10. Dependencies

### Feature Dependencies

#### Hard Dependencies (Blockers)
1. **Database Schema**: PostgreSQL `datasets` table must exist before any dataset operations
   - **Owner**: Database Migration (001_initial_schema.py)
   - **Status**: Pending
   - **Impact**: Cannot store dataset metadata without schema

2. **File Storage Structure**: `/data/datasets/` directory structure must be created
   - **Owner**: Infrastructure Setup (docker-compose.yml volume mounts)
   - **Status**: Pending
   - **Impact**: Cannot store dataset files without directory structure

3. **Celery Worker**: Background job worker must be running to process downloads
   - **Owner**: Deployment Configuration (docker-compose.yml celery service)
   - **Status**: Pending
   - **Impact**: Downloads will fail without worker to process tasks

4. **Redis**: Must be available for Celery broker and result backend
   - **Owner**: Infrastructure Setup (docker-compose.yml redis service)
   - **Status**: Pending
   - **Impact**: Cannot queue background jobs without Redis

#### Soft Dependencies (Recommended)
1. **Model Management**: Tokenization feature depends on models being available
   - **Owner**: 002_FPRD|Model_Management feature
   - **Status**: Parallel development
   - **Impact**: Tokenization feature disabled until models available; rest of Dataset Management works independently

2. **Training Configuration**: Training feature will consume datasets from this feature
   - **Owner**: 003_FPRD|SAE_Training feature
   - **Status**: Pending
   - **Impact**: No impact on Dataset Management; one-way dependency

### External Service Dependencies

#### HuggingFace Hub
- **Service**: HuggingFace API (https://huggingface.co)
- **Purpose**: Download datasets and validate repository IDs
- **Availability**: 99.9% uptime (HuggingFace SLA)
- **Failure Handling**: Retry with exponential backoff; fallback to error status
- **Rate Limits**: 1000 requests/hour (authenticated), 100 requests/hour (anonymous)
- **Authentication**: Optional access token for gated datasets

#### HuggingFace Datasets Library
- **Library**: `datasets` Python package (version ≥2.14.0)
- **Purpose**: Download and load datasets from HuggingFace Hub
- **Installation**: `pip install datasets`
- **Compatibility**: Python 3.8+, PyArrow ≥12.0.0

#### HuggingFace Transformers Library
- **Library**: `transformers` Python package (version ≥4.30.0)
- **Purpose**: Load model-specific tokenizers
- **Installation**: `pip install transformers`
- **Compatibility**: Python 3.8+, PyTorch ≥2.0.0

### Library Dependencies

```python
# Backend requirements.txt
fastapi>=0.104.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
celery>=5.3.0
redis>=5.0.0
datasets>=2.14.0
transformers>=4.30.0
pyarrow>=12.0.0
pandas>=2.0.0
psycopg2-binary>=2.9.0
```

```json
// Frontend package.json
{
  "dependencies": {
    "react": "^18.2.0",
    "typescript": "^5.0.0",
    "zustand": "^4.4.0",
    "lucide-react": "^0.292.0",
    "recharts": "^2.9.0",
    "axios": "^1.6.0"
  }
}
```

### Infrastructure Dependencies

1. **Docker**: Required for containerized deployment
   - Version: ≥24.0.0
   - Docker Compose: ≥2.20.0

2. **PostgreSQL**: Database server
   - Version: 15-alpine (Docker image)
   - Extensions: None required (JSONB native, GIN indexes native)

3. **Redis**: In-memory data store
   - Version: 7-alpine (Docker image)
   - Configuration: Persistent storage (AOF enabled)

4. **Nginx**: Reverse proxy (for production)
   - Version: alpine (Docker image)
   - Configuration: Route `/api/datasets/` to backend service

5. **Storage**: Local disk storage
   - Minimum: 50GB available space
   - Recommended: 500GB for large dataset collection
   - Type: SSD preferred for faster I/O

### Timeline Dependencies

#### Phase 1 (MVP - Week 1-2)
- Database schema created (Alembic migration)
- Celery worker deployed and tested
- HuggingFace download functionality implemented and tested

#### Phase 2 (Enhancement - Week 3-4)
- Model Management complete (for tokenization feature)
- Full-text search indexed and performant
- Dataset statistics visualizations complete

#### Phase 3 (Polish - Week 5-6)
- Local dataset upload implemented
- Delete dataset with cascade handling
- Performance optimization (caching, pagination)

---

## 11. Success Criteria

### Quantitative Success Metrics

#### Performance Metrics
1. **Download Throughput**
   - Target: Download 1GB dataset in <10 minutes on 10 Mbps connection
   - Measurement: Log download start/end times, compute throughput
   - Baseline: HuggingFace CLI download speed

2. **API Response Times (p95)**
   - Target: GET /api/datasets <100ms, POST /api/datasets/download <500ms
   - Measurement: FastAPI middleware logging, Prometheus metrics
   - Baseline: 150ms (acceptable), >200ms (needs optimization)

3. **Search Performance**
   - Target: Full-text search returns results in <300ms for datasets with 10,000+ samples
   - Measurement: Database query timing logs
   - Baseline: 500ms (acceptable), >1s (needs indexing optimization)

4. **UI Responsiveness**
   - Target: Initial page load <200ms, modal open <100ms, chart render <1s
   - Measurement: React DevTools Profiler, Chrome Performance tab
   - Baseline: 300ms (acceptable), >500ms (needs optimization)

#### Reliability Metrics
1. **Download Success Rate**
   - Target: >95% of downloads complete successfully (excluding user cancellations)
   - Measurement: Log download outcomes (success, failure, cancelled)
   - Baseline: 90% (acceptable), <85% (investigate failures)

2. **Resume Success Rate**
   - Target: >90% of interrupted downloads resume successfully
   - Measurement: Track resume attempts vs. successes
   - Baseline: 80% (acceptable), <70% (improve resume logic)

3. **Data Integrity**
   - Target: 100% of downloads pass checksum validation
   - Measurement: Compare downloaded file checksums with HuggingFace metadata
   - Baseline: 99.9% (acceptable), <99% (investigate corruption)

#### Usability Metrics
1. **Time to First Dataset**
   - Target: First-time user downloads dataset in <3 minutes (from UI open to "ready" status)
   - Measurement: User testing sessions, log analysis
   - Baseline: 5 minutes (acceptable), >10 minutes (improve UX)

2. **Search Usage**
   - Target: >30% of dataset browsing sessions include search or filtering
   - Measurement: Log search query usage vs. total browsing sessions
   - Baseline: 20% (acceptable), <10% (improve discoverability)

3. **Error Recovery Rate**
   - Target: >80% of users successfully recover from download errors using retry button
   - Measurement: Track error occurrences vs. successful retries
   - Baseline: 70% (acceptable), <60% (improve error messages)

### Qualitative Success Indicators

#### User Satisfaction
1. **Ease of Use**
   - Indicator: Users successfully download and browse datasets without consulting documentation
   - Validation: User testing sessions (5+ users)
   - Success: 4/5 users complete tasks independently

2. **Error Clarity**
   - Indicator: Users understand error messages and know how to resolve issues
   - Validation: Review error message click-through rates (retry, delete, etc.)
   - Success: >70% of errors result in user action (not abandonment)

3. **Visual Polish**
   - Indicator: UI matches Mock UI reference exactly (layout, colors, interactions)
   - Validation: Side-by-side comparison with Mock UI
   - Success: <5 visual discrepancies noted in review

#### Developer Feedback
1. **Code Quality**
   - Indicator: Code review approval without major revision requests
   - Validation: Pull request review comments
   - Success: <10 minor comments, 0 blocking comments

2. **Test Coverage**
   - Indicator: All critical paths covered by automated tests
   - Validation: Coverage report (pytest-cov for backend, Vitest for frontend)
   - Success: ≥80% coverage for business logic

3. **Documentation Completeness**
   - Indicator: API endpoints fully documented in OpenAPI spec
   - Validation: Review OpenAPI spec completeness
   - Success: All endpoints have descriptions, request/response schemas, error codes

### Business Impact Measurements

#### Adoption Metrics
1. **Dataset Usage**
   - Target: Average of 5 datasets downloaded per user within first week of use
   - Measurement: Log dataset creation events per user
   - Baseline: 3 datasets (acceptable), <2 (improve onboarding)

2. **Training Data Diversity**
   - Target: Users download datasets from ≥3 different domains (news, books, code, etc.)
   - Measurement: Analyze dataset sources and domains
   - Baseline: 2 domains (acceptable), 1 domain (needs more variety)

3. **Storage Efficiency**
   - Target: Average dataset size <5GB (efficient edge storage usage)
   - Measurement: Analyze size_bytes field across all datasets
   - Baseline: 10GB (acceptable), >20GB (users need guidance on dataset selection)

#### Feature Completion
1. **MVP Feature Parity**
   - Target: All FR-1 through FR-6 requirements implemented and tested
   - Validation: Requirements traceability matrix, manual testing
   - Success: 100% of MVP requirements marked "complete"

2. **Bug Density**
   - Target: <5 bugs per 1000 lines of code (KLOC)
   - Measurement: Count bugs reported in issue tracker
   - Baseline: 10 bugs/KLOC (acceptable), >20 bugs/KLOC (needs refactoring)

3. **Technical Debt**
   - Target: <10% of code flagged as "TODO" or "FIXME"
   - Measurement: Static analysis (grep for TODO/FIXME comments)
   - Baseline: 20% (acceptable), >30% (refactor before next feature)

---

## 12. Testing Requirements

### Unit Testing

#### Backend Unit Tests (Pytest)

**Test Suite 1: Dataset Service**
- `test_create_dataset_record()`: Verify database record creation with valid data
- `test_create_dataset_invalid_status()`: Verify rejection of invalid status values
- `test_update_dataset_status()`: Verify status transitions follow state machine
- `test_update_dataset_progress()`: Verify progress updates (0-100 range)
- `test_compute_statistics()`: Verify statistics computation accuracy
- `test_validate_repository_id()`: Verify repo ID format validation
- `test_check_disk_space()`: Verify disk space calculation before download

**Test Suite 2: Dataset Repository**
- `test_get_dataset_by_id()`: Verify retrieval of existing dataset
- `test_get_dataset_not_found()`: Verify 404 handling for non-existent ID
- `test_list_datasets_pagination()`: Verify pagination logic (page, limit)
- `test_list_datasets_filter_by_status()`: Verify status filtering
- `test_delete_dataset()`: Verify database record deletion
- `test_delete_dataset_with_trainings()`: Verify foreign key constraint (prevent deletion)

**Test Suite 3: Dataset Download Task (Celery)**
- `test_download_task_success()`: Mock HuggingFace download, verify completion
- `test_download_task_network_error()`: Verify retry logic on network failure
- `test_download_task_invalid_repo()`: Verify error handling for 404 repo
- `test_download_task_disk_full()`: Verify error handling for insufficient space
- `test_download_task_resume()`: Verify resume capability after interruption

**Coverage Target**: ≥80% line coverage for business logic

#### Frontend Unit Tests (Vitest + React Testing Library)

**Test Suite 1: DatasetsPanel Component**
- `test_render_empty_state()`: Verify empty state message displays
- `test_render_dataset_list()`: Verify dataset cards render with correct data
- `test_download_form_submission()`: Verify form validation and API call
- `test_dataset_card_click()`: Verify modal opens on card click
- `test_download_progress_display()`: Verify progress bar updates

**Test Suite 2: DatasetDetailModal Component**
- `test_modal_render()`: Verify modal displays with dataset data
- `test_samples_tab()`: Verify sample list renders and pagination works
- `test_statistics_tab()`: Verify statistics display correctly
- `test_search_functionality()`: Verify search input triggers API call with query
- `test_filter_functionality()`: Verify split and token range filters work

**Test Suite 3: Custom Hooks**
- `test_useDatasets_hook()`: Verify hook fetches and caches dataset list
- `test_useDataset_hook()`: Verify hook fetches single dataset details
- `test_useDatasetSamples_hook()`: Verify hook handles pagination and search

**Coverage Target**: ≥70% line coverage for UI components

### Integration Testing

#### API Integration Tests (Pytest + TestClient)

**Test Scenario 1: Download Dataset Flow**
1. POST /api/datasets/download with valid repo_id
2. Verify 202 response with dataset_id and job_id
3. Poll GET /api/datasets/:id until status transitions: downloading → ingesting → ready
4. Verify statistics populated after "ready" status
5. Verify files exist in `/data/datasets/raw/{dataset_id}/`

**Test Scenario 2: Browse and Search**
1. POST /api/datasets/download (use small test dataset)
2. Wait for "ready" status
3. GET /api/datasets/:id/samples?page=1&limit=50
4. Verify 50 samples returned with correct structure
5. GET /api/datasets/:id/samples?search=keyword
6. Verify search results filtered correctly

**Test Scenario 3: Delete Dataset**
1. Create test dataset (mock data)
2. DELETE /api/datasets/:id
3. Verify 200 response with freed_bytes
4. Verify database record deleted
5. Verify files deleted from filesystem
6. GET /api/datasets/:id → Verify 404 response

**Test Scenario 4: Error Handling**
1. POST /api/datasets/download with invalid repo_id
2. Verify 404 error response with clear message
3. POST /api/datasets/download with insufficient disk space (mock)
4. Verify 503 error response with disk space warning

#### Database Integration Tests

**Test Scenario 1: Full-Text Search**
1. Insert test datasets with known text content
2. Query using PostgreSQL full-text search (`to_tsvector`)
3. Verify results match expected datasets
4. Verify search performance (<500ms for 10,000 samples)

**Test Scenario 2: Foreign Key Constraints**
1. Create dataset and training referencing dataset
2. Attempt to delete dataset
3. Verify deletion blocked with 409 error
4. Delete training, then delete dataset
5. Verify deletion succeeds

**Test Scenario 3: Statistics JSONB Storage**
1. Insert dataset with complex statistics JSON
2. Query statistics using JSONB operators (`->`, `->>`)
3. Verify correct values extracted
4. Update statistics JSONB field
5. Verify updates persist correctly

### User Acceptance Testing (UAT)

#### UAT Scenario 1: First-Time User Journey
**Objective**: Verify a new user can download and browse a dataset independently

**Steps**:
1. Open miStudio application (fresh install, no datasets)
2. Navigate to Dataset Management panel
3. Enter "roneneldan/TinyStories" in repository field
4. Click "Download Dataset from HuggingFace"
5. Observe progress bar and status updates
6. Wait for "ready" status (typically 30-60 seconds for TinyStories)
7. Click on dataset card to open detail modal
8. Browse samples in Samples tab
9. Switch to Statistics tab and observe visualizations
10. Close modal

**Acceptance Criteria**:
- User completes flow in <5 minutes without assistance
- No errors or confusing UI states encountered
- User understands dataset is ready for training

#### UAT Scenario 2: Search and Filter
**Objective**: Verify dataset browsing with search and filtering works intuitively

**Steps**:
1. Open dataset detail modal for "OpenWebText-10K" (assume already downloaded)
2. Enter "machine learning" in search box
3. Observe filtered results (<3 seconds)
4. Select "validation" split filter
5. Observe results further filtered
6. Adjust token count range slider (100-500 tokens)
7. Observe results update dynamically
8. Clear all filters
9. Verify full dataset samples return

**Acceptance Criteria**:
- Search and filters work independently and in combination
- Results update within 500ms of user input
- Clear feedback when no results match filters

#### UAT Scenario 3: Error Recovery
**Objective**: Verify users can recover from download errors

**Steps**:
1. Disconnect network (simulate network failure)
2. Attempt to download dataset
3. Observe error state with clear message
4. Reconnect network
5. Click "Retry Download" button
6. Verify download resumes and completes successfully

**Acceptance Criteria**:
- Error message clearly explains issue ("Network error. Check connection.")
- Retry button prominently displayed
- User successfully completes download after retry

### Performance Testing

#### Load Testing (Locust)

**Test Scenario 1: Concurrent API Requests**
- Simulate 50 concurrent users listing datasets (GET /api/datasets)
- Target: p95 response time <150ms, p99 <200ms
- Verify API remains responsive under load

**Test Scenario 2: Sample Browsing**
- Simulate 20 concurrent users browsing samples (GET /api/datasets/:id/samples)
- With search queries and pagination
- Target: p95 response time <500ms
- Verify database connection pool handles concurrency

**Test Scenario 3: Download Queue**
- Simulate 10 concurrent download requests (POST /api/datasets/download)
- Verify only 1 download executes at a time (Celery concurrency limit)
- Verify queued downloads process sequentially
- Verify API returns 202 immediately for all requests

#### Stress Testing

**Test Scenario 1: Large Dataset Ingestion**
- Download 50GB dataset (e.g., OpenWebText-full)
- Monitor memory usage during ingestion
- Verify memory stays <4GB (streaming ingestion, not loading entire dataset)
- Verify statistics computation completes without timeout

**Test Scenario 2: Full-Text Search on Large Dataset**
- Create dataset with 1 million samples
- Execute full-text search queries
- Verify search completes in <1 second
- Verify PostgreSQL GIN index is used (check query plan)

---

## 13. Implementation Considerations

### Complexity Assessment

#### High Complexity Components
1. **Resume-Capable Downloads**
   - **Complexity**: High (8/10)
   - **Challenges**: HTTP Range requests, partial file management, retry logic
   - **Mitigation**: Use `requests` library with `Range` headers, store download metadata (bytes downloaded, ETag)
   - **Estimated Effort**: 16 hours (2 days)

2. **Full-Text Search with Pagination**
   - **Complexity**: Medium-High (7/10)
   - **Challenges**: PostgreSQL GIN index setup, efficient pagination with search filters
   - **Mitigation**: Follow PostgreSQL best practices, use `tsvector` and `tsquery`, test with large datasets
   - **Estimated Effort**: 12 hours (1.5 days)

3. **Statistics Computation**
   - **Complexity**: Medium (6/10)
   - **Challenges**: Efficient token counting, distribution computation, handling large datasets
   - **Mitigation**: Use Pandas for vectorized operations, stream large datasets in chunks
   - **Estimated Effort**: 8 hours (1 day)

#### Medium Complexity Components
1. **Dataset Detail Modal**
   - **Complexity**: Medium (5/10)
   - **Challenges**: Tabbed interface, chart rendering, responsive design
   - **Mitigation**: Use Recharts for visualizations, Tailwind for responsive grid
   - **Estimated Effort**: 12 hours (1.5 days)

2. **Celery Background Tasks**
   - **Complexity**: Medium (5/10)
   - **Challenges**: Task retry logic, progress reporting, error handling
   - **Mitigation**: Follow Celery best practices, use task state management
   - **Estimated Effort**: 10 hours (1.25 days)

3. **File System Operations**
   - **Complexity**: Medium (5/10)
   - **Challenges**: Atomic writes, cleanup on failure, disk space checks
   - **Mitigation**: Use temp files + rename for atomicity, pathlib for safe path handling
   - **Estimated Effort**: 6 hours (0.75 days)

#### Low Complexity Components
1. **Dataset List UI**
   - **Complexity**: Low (3/10)
   - **Challenges**: Responsive card layout, status indicators
   - **Mitigation**: Directly reference Mock UI for exact styling
   - **Estimated Effort**: 4 hours (0.5 days)

2. **Status Polling**
   - **Complexity**: Low (3/10)
   - **Challenges**: Efficient polling without excessive requests
   - **Mitigation**: Use 2-second intervals, only poll active downloads
   - **Estimated Effort**: 2 hours (0.25 days)

3. **Database CRUD Operations**
   - **Complexity**: Low (2/10)
   - **Challenges**: Standard SQLAlchemy ORM operations
   - **Mitigation**: Follow repository pattern from ADR
   - **Estimated Effort**: 4 hours (0.5 days)

### Risk Factors

#### Technical Risks

**Risk 1: HuggingFace API Rate Limiting**
- **Probability**: Medium (30%)
- **Impact**: High (blocks downloads)
- **Mitigation**:
  - Implement exponential backoff on 429 responses
  - Display clear message to user: "Rate limited by HuggingFace. Retry in X minutes."
  - Consider caching repository metadata to reduce API calls
  - Provide option to use authenticated requests (higher rate limits)

**Risk 2: Disk Space Exhaustion During Download**
- **Probability**: Medium (40%)
- **Impact**: High (corrupted downloads, system instability)
- **Mitigation**:
  - Check available disk space before starting download
  - Monitor space during download, abort if threshold reached (<10% free)
  - Implement cleanup of incomplete downloads on error
  - Display warning if user approaching storage limit

**Risk 3: Tokenization Memory Overflow**
- **Probability**: Low (20%)
- **Impact**: High (worker crash, data loss)
- **Mitigation**:
  - Use streaming tokenization (batch processing)
  - Limit batch size based on available RAM (auto-detect)
  - Monitor worker memory usage, restart if threshold exceeded
  - Test with large datasets (10GB+) during development

**Risk 4: Database Performance Degradation**
- **Probability**: Low (15%)
- **Impact**: Medium (slow UI responsiveness)
- **Mitigation**:
  - Create indexes proactively (see FR-10)
  - Use connection pooling (SQLAlchemy pool_size=10)
  - Monitor query performance with `pg_stat_statements`
  - Implement query optimization based on EXPLAIN ANALYZE

#### Integration Risks

**Risk 1: HuggingFace Datasets Library Compatibility**
- **Probability**: Medium (25%)
- **Impact**: Medium (some datasets fail to download)
- **Mitigation**:
  - Pin library versions in requirements.txt
  - Test with variety of datasets (different formats: Parquet, Arrow, CSV)
  - Implement graceful fallback for unsupported formats
  - Display error: "Dataset format not supported. Please upload manually."

**Risk 2: Model Tokenizer Compatibility**
- **Probability**: Low (10%)
- **Impact**: Medium (tokenization fails for some models)
- **Mitigation**:
  - Validate tokenizer exists before starting tokenization
  - Handle missing tokenizers gracefully (error message + skip tokenization)
  - Test with all supported model architectures (GPT-2, Llama, Phi)

#### Resource Risks

**Risk 1: Limited Development Time**
- **Probability**: High (50%)
- **Impact**: Medium (feature scope reduction)
- **Mitigation**:
  - Prioritize FR-1 through FR-4 (core download and browsing)
  - Defer FR-7 (local upload) to Phase 2 if time constrained
  - Implement FR-6 (delete) as separate task after core features

**Risk 2: Limited Edge Device Storage**
- **Probability**: High (60%)
- **Impact**: Medium (users unable to download multiple datasets)
- **Mitigation**:
  - Provide storage usage dashboard prominently in UI
  - Implement "delete unused datasets" bulk action
  - Guide users to select smaller datasets (e.g., "TinyStories" vs. "OpenWebText-full")

### Recommended Implementation Approach

#### Phase 1: Core Download Functionality (Week 1)
**Goal**: Users can download datasets from HuggingFace and see them in UI

1. **Day 1-2**: Database schema + API endpoints
   - Create Alembic migration for `datasets` table
   - Implement `DatasetRepository` (CRUD operations)
   - Implement `POST /api/datasets/download` and `GET /api/datasets` endpoints
   - Write unit tests for repository and endpoints

2. **Day 3-4**: Celery background tasks
   - Implement `download_dataset_task` with HuggingFace Datasets integration
   - Implement progress reporting (update database every 10%)
   - Implement retry logic (exponential backoff, max 3 retries)
   - Test with small dataset (TinyStories)

3. **Day 5**: UI Implementation
   - Implement `DatasetsPanel` component (download form + dataset list)
   - Implement status polling (every 2 seconds for active downloads)
   - Style components to match Mock UI exactly
   - Manual testing of full download flow

#### Phase 2: Dataset Browsing (Week 2)
**Goal**: Users can browse samples and view statistics

1. **Day 6-7**: Statistics computation
   - Implement statistics computation during ingestion
   - Store statistics in database (JSONB field)
   - Implement `GET /api/datasets/:id` endpoint with statistics
   - Write unit tests for statistics functions

2. **Day 8-9**: Dataset detail modal
   - Implement `DatasetDetailModal` component (tabbed interface)
   - Implement Samples tab (paginated table)
   - Implement Statistics tab (charts with Recharts)
   - Implement Settings tab (metadata display)
   - Style to match Mock UI

3. **Day 10**: Sample browsing API
   - Implement `GET /api/datasets/:id/samples` endpoint
   - Implement pagination, filtering, and full-text search
   - Create PostgreSQL GIN index for search performance
   - Test with large dataset (10,000+ samples)

#### Phase 3: Polish and Optimization (Week 3)
**Goal**: Feature is production-ready with all edge cases handled

1. **Day 11-12**: Error handling and recovery
   - Implement resume capability for interrupted downloads
   - Implement retry UI button for failed downloads
   - Improve error messages (user-friendly, actionable)
   - Test error scenarios (network failure, disk full, invalid repo)

2. **Day 13**: Delete functionality
   - Implement `DELETE /api/datasets/:id` endpoint
   - Implement file cleanup (delete raw and tokenized files)
   - Implement foreign key constraint checking (prevent deletion if used by training)
   - Implement delete confirmation modal in UI

3. **Day 14**: Testing and documentation
   - Write integration tests for complete download flow
   - Write UAT test scenarios and conduct manual testing
   - Update API documentation (OpenAPI spec)
   - Performance testing (load testing with Locust)

4. **Day 15**: Buffer for bug fixes and polish
   - Address bugs found during testing
   - Performance optimization (caching, indexing)
   - Final visual polish to match Mock UI exactly

### Potential Technical Challenges

**Challenge 1: Streaming Large Datasets Without Memory Overflow**
- **Issue**: Loading entire dataset into memory for statistics computation causes OOM on Jetson
- **Solution**: Stream dataset in chunks (1000 samples at a time), compute statistics incrementally
- **Code Pattern**:
  ```python
  from datasets import load_dataset

  dataset = load_dataset(repo_id, streaming=True)
  total_tokens = 0
  sample_count = 0

  for sample in dataset['train']:
      tokens = sample['text'].split()  # Simple whitespace tokenization
      total_tokens += len(tokens)
      sample_count += 1

      if sample_count % 1000 == 0:
          # Update progress every 1000 samples
          update_progress(sample_count / total_samples * 100)

  avg_tokens = total_tokens / sample_count
  ```

**Challenge 2: Efficient Full-Text Search on PostgreSQL**
- **Issue**: Full-text search is slow (>1s) without proper indexing
- **Solution**: Create GIN index on `tsvector` expression, use `plainto_tsquery` for simple queries
- **Code Pattern**:
  ```sql
  -- Create GIN index (during migration)
  CREATE INDEX idx_datasets_search ON datasets
      USING GIN(to_tsvector('english', name || ' ' || COALESCE(description, '')));

  -- Query pattern (in SQLAlchemy)
  from sqlalchemy import func

  datasets = session.query(Dataset).filter(
      func.to_tsvector('english',
          Dataset.name + ' ' + func.coalesce(Dataset.description, '')
      ).op('@@')(func.plainto_tsquery('english', search_query))
  ).all()
  ```

**Challenge 3: Resume Capability for Interrupted Downloads**
- **Issue**: Standard `datasets.load_dataset()` doesn't support resume out-of-box
- **Solution**: Use `DownloadConfig` with `resume_download=True`, store download progress in database
- **Code Pattern**:
  ```python
  from datasets import load_dataset, DownloadConfig

  download_config = DownloadConfig(
      resume_download=True,
      force_download=False,
      cache_dir=f"/data/datasets/raw/{dataset_id}",
      max_retries=3,
  )

  try:
      dataset = load_dataset(
          repo_id,
          cache_dir=f"/data/datasets/raw/{dataset_id}",
          download_config=download_config,
          token=access_token
      )
  except Exception as e:
      # Log error, update status to "error"
      update_dataset_status(dataset_id, "error", error_message=str(e))
      raise
  ```

---

## 14. Open Questions

### Business/Product Questions

1. **Q: Should we support private HuggingFace datasets requiring authentication?**
   - **Impact**: Requires secure token storage and handling
   - **Options**:
     - A) Yes, support with password-masked input (recommended)
     - B) No, only public datasets (simpler, but limits usefulness)
   - **Decision Needed By**: Week 1, Day 1 (before implementing download API)
   - **Recommendation**: Option A (included in FR-1.11)

2. **Q: What is the maximum allowed dataset size?**
   - **Impact**: Affects disk space checks and user warnings
   - **Options**:
     - A) No limit (trust user to manage storage)
     - B) Hard limit 50GB per dataset (prevent accidental large downloads)
     - C) Soft warning at 10GB, block at 100GB
   - **Decision Needed By**: Week 1, Day 3 (before implementing disk space checks)
   - **Recommendation**: Option C (included in NFR-3)

3. **Q: Should we auto-delete old datasets to free space?**
   - **Impact**: Affects storage management and user trust
   - **Options**:
     - A) No auto-deletion (user manages manually)
     - B) Auto-delete datasets not used in 90 days (with warning)
     - C) Prompt user when storage >80% full to select datasets for deletion
   - **Decision Needed By**: Week 2 (Phase 2 polish)
   - **Recommendation**: Option C (defer to Phase 3)

### Technical Design Questions

1. **Q: Should tokenization create a new dataset record or modify existing dataset?**
   - **Impact**: Affects database schema and UI flow
   - **Options**:
     - A) Create new dataset record (tokenized variant) with reference to source
     - B) Modify existing dataset, add `tokenized_path` field
   - **Decision Needed By**: Week 1, Day 1 (affects schema migration)
   - **Recommendation**: Option A (cleaner separation, easier rollback)

2. **Q: How should we handle dataset updates (newer version on HuggingFace)?**
   - **Impact**: Affects download flow and user expectations
   - **Options**:
     - A) Re-download entire dataset (no delta updates)
     - B) Detect version changes, prompt user to update
     - C) Auto-update datasets daily (background task)
   - **Decision Needed By**: Week 2 (after core download implemented)
   - **Recommendation**: Option A for MVP (simplest), defer B to Phase 2

3. **Q: Should we cache dataset statistics in Redis or only in PostgreSQL?**
   - **Impact**: Affects statistics retrieval performance
   - **Options**:
     - A) PostgreSQL only (simpler, but slower for repeated access)
     - B) Redis cache with 1-hour TTL (faster, but adds complexity)
     - C) In-memory LRU cache in backend (fastest, no Redis dependency)
   - **Decision Needed By**: Week 2, Day 7 (statistics implementation)
   - **Recommendation**: Option A for MVP (PostgreSQL JSONB is fast enough)

### Implementation Questions

1. **Q: Should we implement local dataset upload in MVP or defer to Phase 2?**
   - **Impact**: Affects development timeline and feature completeness
   - **Options**:
     - A) Implement in MVP (full feature parity)
     - B) Defer to Phase 2 (focus on HuggingFace download first)
   - **Decision Needed By**: Week 1 planning session
   - **Recommendation**: Option B (FR-7 marked as secondary)

2. **Q: How should we test resume capability without manual network disruption?**
   - **Impact**: Affects test coverage and CI/CD pipeline
   - **Options**:
     - A) Mock network failures in unit tests
     - B) Manual testing only (requires physical network disruption)
     - C) Use chaos engineering tool (e.g., toxiproxy) in integration tests
   - **Decision Needed By**: Week 1, Day 4 (testing implementation)
   - **Recommendation**: Option A for unit tests, Option B for UAT

3. **Q: Should we implement dataset sharing/export functionality?**
   - **Impact**: Affects feature scope and user workflows
   - **Options**:
     - A) No sharing (single-user system, out of scope)
     - B) Export dataset as .tar.gz for manual sharing
     - C) Upload dataset to HuggingFace Hub from miStudio
   - **Decision Needed By**: Phase 2 planning
   - **Recommendation**: Option A for MVP (marked as non-goal in section 9)

---

## Appendix A: API Request/Response Examples

### Example 1: Download Dataset from HuggingFace

**Request:**
```http
POST /api/datasets/download HTTP/1.1
Host: mistudio.mcslab.io
Content-Type: application/json

{
  "repo_id": "roneneldan/TinyStories",
  "access_token": null
}
```

**Response (202 Accepted):**
```json
{
  "data": {
    "dataset_id": "ds_tinystories_20251005",
    "status": "downloading",
    "message": "Dataset download started"
  },
  "meta": {
    "job_id": "job_abc123xyz",
    "estimated_duration_seconds": 45
  }
}
```

### Example 2: Get Dataset with Statistics

**Request:**
```http
GET /api/datasets/ds_tinystories_20251005 HTTP/1.1
Host: mistudio.mcslab.io
```

**Response (200 OK):**
```json
{
  "data": {
    "id": "ds_tinystories_20251005",
    "name": "roneneldan/TinyStories",
    "source": "HuggingFace",
    "description": "Synthetic short stories for training small language models",
    "size": "450MB",
    "size_bytes": 471859200,
    "status": "ready",
    "progress": null,
    "error_message": null,
    "file_path": "/data/datasets/raw/ds_tinystories_20251005/",
    "tokenized_path": null,
    "num_samples": 8500,
    "num_tokens": 2125000,
    "vocab_size": 15234,
    "avg_sequence_length": 250.0,
    "statistics": {
      "total_samples": 8500,
      "splits": {
        "train": 7000,
        "validation": 1000,
        "test": 500
      },
      "token_distribution": [
        {"range": "0-100", "count": 150},
        {"range": "100-200", "count": 1200},
        {"range": "200-300", "count": 3500},
        {"range": "300-400", "count": 2400},
        {"range": "400-500", "count": 950},
        {"range": "500+", "count": 300}
      ],
      "median_tokens": 245,
      "min_tokens": 45,
      "max_tokens": 612,
      "unique_tokens": 15234
    },
    "created_at": "2025-10-05T10:00:00Z",
    "updated_at": "2025-10-05T10:00:45Z"
  }
}
```

### Example 3: Browse Dataset Samples with Search

**Request:**
```http
GET /api/datasets/ds_tinystories_20251005/samples?page=1&limit=10&search=adventure&split=train HTTP/1.1
Host: mistudio.mcslab.io
```

**Response (200 OK):**
```json
{
  "data": [
    {
      "id": 234,
      "text": "Once upon a time, there was a brave knight who went on an adventure to find a magical dragon. The knight rode through forests and mountains, facing many challenges along the way...",
      "tokens": ["Once", "upon", "a", "time", ",", "there", "was", "a", "brave", "knight", "who", "went", "on", "an", "adventure", "..."],
      "token_count": 247,
      "split": "train",
      "metadata": {
        "source": "synthetic",
        "theme": "adventure",
        "complexity": "simple"
      }
    },
    {
      "id": 567,
      "text": "Lily loved going on adventures in her backyard. One day, she found a secret path behind the old oak tree...",
      "tokens": ["Lily", "loved", "going", "on", "adventures", "in", "her", "backyard", ".", "..."],
      "token_count": 189,
      "split": "train",
      "metadata": {
        "source": "synthetic",
        "theme": "adventure",
        "complexity": "simple"
      }
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 47,
    "has_next": true,
    "has_prev": false
  }
}
```

### Example 4: Delete Dataset

**Request:**
```http
DELETE /api/datasets/ds_old_dataset_123 HTTP/1.1
Host: mistudio.mcslab.io
```

**Response (200 OK):**
```json
{
  "data": {
    "message": "Dataset deleted successfully",
    "dataset_id": "ds_old_dataset_123",
    "freed_bytes": 1234567890
  }
}
```

**Error Response (409 Conflict):**
```json
{
  "error": {
    "code": "DELETE_CONFLICT",
    "message": "Cannot delete dataset: referenced by 1 active training",
    "details": {
      "training_ids": ["tr_active_789"],
      "training_statuses": ["training"]
    },
    "retryable": false,
    "timestamp": "2025-10-05T12:34:56Z"
  }
}
```

---

## Appendix B: Database Schema Details

### Datasets Table DDL

```sql
CREATE TABLE datasets (
    -- Primary key
    id VARCHAR(255) PRIMARY KEY,

    -- Basic metadata
    name VARCHAR(500) NOT NULL,
    source VARCHAR(50) NOT NULL,
    description TEXT,

    -- Size tracking
    size_bytes BIGINT NOT NULL,

    -- Status tracking
    status VARCHAR(50) NOT NULL DEFAULT 'downloading',
    progress FLOAT CHECK (progress >= 0 AND progress <= 100),
    error_message TEXT,

    -- File paths
    file_path VARCHAR(1000),
    tokenized_path VARCHAR(1000),

    -- Statistics (populated after ingestion)
    num_samples INTEGER,
    num_tokens BIGINT,
    vocab_size INTEGER,
    avg_sequence_length FLOAT,

    -- Extended statistics (JSONB)
    statistics JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT datasets_status_check CHECK (status IN
        ('downloading', 'ingesting', 'tokenizing', 'ready', 'error')),
    CONSTRAINT datasets_source_check CHECK (source IN
        ('HuggingFace', 'Local', 'Custom'))
);

-- Indexes
CREATE INDEX idx_datasets_status ON datasets(status);
CREATE INDEX idx_datasets_created_at ON datasets(created_at DESC);
CREATE INDEX idx_datasets_source ON datasets(source);

-- Full-text search index
CREATE INDEX idx_datasets_search ON datasets
    USING GIN(to_tsvector('english', name || ' ' || COALESCE(description, '')));

-- JSONB index for statistics queries (if needed)
CREATE INDEX idx_datasets_statistics ON datasets USING GIN(statistics);
```

### Sample JSONB Statistics Structure

```json
{
  "total_samples": 10000,
  "total_tokens": 54231000,
  "avg_tokens_per_sample": 542.31,
  "median_tokens_per_sample": 520,
  "min_tokens": 45,
  "max_tokens": 1024,
  "unique_tokens": 50257,
  "splits": {
    "train": 8000,
    "validation": 1000,
    "test": 1000
  },
  "token_distribution": [
    {"range": "0-100", "count": 234},
    {"range": "100-200", "count": 567},
    {"range": "200-300", "count": 1234},
    {"range": "300-400", "count": 2100},
    {"range": "400-500", "count": 2800},
    {"range": "500-600", "count": 1900},
    {"range": "600-700", "count": 800},
    {"range": "700-800", "count": 250},
    {"range": "800-900", "count": 85},
    {"range": "900-1000", "count": 30}
  ],
  "domain_distribution": {
    "books": 4500,
    "news": 3200,
    "web": 2300
  }
}
```

---

**Document Version:** 1.0
**Status:** Draft
**Next Review:** After stakeholder feedback
**Related Documents:**
- @0xcc/prds/000_PPRD|miStudio.md (Project PRD)
- @0xcc/adrs/000_PADR|miStudio.md (Architecture Decision Record)
- @0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx (PRIMARY UI/UX Reference)
- @0xcc/project-specs/infrastructure/001_SPEC|Folder_File_Details.md (File Structure)
- @0xcc/project-specs/infrastructure/003_SPEC|Postgres_Usecase_Details_and_Guidance.md (Database Schema)
