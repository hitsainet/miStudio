# Feature PRD: Feature Discovery

**Document ID:** 004_FPRD|Feature_Discovery
**Feature:** Interpretable Feature Extraction and Analysis from Trained SAEs
**Status:** Draft
**Created:** 2025-10-06
**Last Updated:** 2025-10-06
**Owner:** miStudio Development Team

---

## 1. Feature Overview

### Purpose
Enable users to extract, browse, analyze, and interpret learned features from trained Sparse Autoencoder (SAE) models. This feature transforms trained SAE representations into human-interpretable insights by identifying activation patterns, providing max-activating examples, and offering analysis tools for understanding what concepts each SAE neuron has learned to detect.

### Business Value
- **Primary Value:** Makes trained SAE models useful by extracting interpretable features that explain model behavior
- **Research Value:** Core capability for mechanistic interpretability research - understanding what models learn
- **Educational Value:** Provides intuitive visualization of abstract neural network representations
- **Practical Value:** Enables identification of specific features for downstream steering tasks

### Success Criteria
- Users can successfully extract features from completed training jobs
- Feature browser displays searchable, sortable list of discovered features
- Feature details show clear max-activating examples with token-level highlighting
- Feature analysis tools (logit lens, correlations, ablation) provide actionable insights
- System extracts and indexes 10,000+ evaluation samples within 5 minutes

### Target Users
- **ML Researchers:** Investigating what language models learn and how they represent knowledge
- **Interpretability Scientists:** Building catalogs of interpretable features for mechanistic understanding
- **AI Safety Practitioners:** Identifying concerning features or failure modes in models

---

## 2. Feature Context

### Relationship to Project PRD
This feature implements **Core Feature #4: Feature Discovery** from the Project PRD (000_PPRD|miStudio.md), specifically:
- Priority: **P0 (Critical MVP feature)**
- Dependencies: SAE Training (requires completed training jobs with trained SAE models)
- Enables: Model Steering (requires discovered features for intervention)

### Integration with Existing Features

**Upstream Dependencies:**
- **SAE Training (003_FPRD):** Requires completed training jobs with trained SAE checkpoints
- **Dataset Management (001_FPRD):** Uses tokenized datasets for evaluation sample extraction
- **Model Management (002_FPRD):** Uses models for activation extraction during feature evaluation

**Downstream Consumers:**
- **Model Steering (005_FPRD - Pending):** Uses discovered features for model behavior modification

### UI/UX Reference
**PRIMARY REFERENCE:** `@0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`
- **FeaturesPanel Component:** Lines 2159-2584 (training selector, extraction, feature browser)
- **Feature Extraction UI:** Lines 2359-2414 (extraction configuration and progress)
- **Feature Browser Table:** Lines 2434-2567 (search, sort, feature list with token highlighting)
- **FeatureDetailModal Component:** Lines 2587-2725 (feature details, tabs, statistics)
- **MaxActivatingExamples:** Lines 2728-2800+ (token sequence highlighting with activation intensity)
- **Additional Analysis Tabs:** Logit lens, correlations, ablation (lines 2684-2720)

---

## 3. User Stories & Scenarios

### US-1: Select Training for Feature Extraction (Priority: P0)
**As a** ML researcher
**I want to** select a completed training job to analyze
**So that** I can extract interpretable features from the trained SAE

**Acceptance Criteria:**
- Given I'm on the Feature Discovery panel, I see a dropdown listing all completed training jobs
- When I select a training, I see summary information: Model name, Dataset name, Encoder type, Training start date
- When no completed trainings exist, I see message: "No completed trainings yet. Complete a training job to discover features."
- When I change training selection, the feature extraction status updates accordingly
- Given multiple completed trainings, the dropdown shows format: "{encoder_type} SAE • {model_name} • {dataset_name} • Started {date}"

**UI Reference:** Mock-embedded-interp-ui.tsx FeaturesPanel training selector (lines 2310-2351)

---

### US-2: Extract Features from Trained SAE (Priority: P0)
**As a** user with a completed training
**I want to** extract interpretable features by running the SAE on evaluation samples
**So that** I can analyze what the SAE has learned

**Acceptance Criteria:**
- Given a selected training has not been extracted, I see "Feature Extraction & Analysis" panel
- When extraction not started, I can configure: Evaluation Samples count (default 10,000), Top-K Examples per Feature (default 100)
- When I click "Extract Features", extraction begins and shows progress bar (0-100%)
- When extraction is running, I see: "Extracting features..." with real-time progress percentage, progress bar animation, status message "Processing activation patterns..."
- When extraction completes, progress reaches 100% and UI transitions to feature browser
- When I navigate away and return, extraction status persists (extraction doesn't restart)

**UI Reference:** Mock-embedded-interp-ui.tsx extraction interface (lines 2359-2414)

---

### US-3: Browse and Search Features (Priority: P0)
**As a** user with extracted features
**I want to** search and filter the feature list
**So that** I can find specific features of interest

**Acceptance Criteria:**
- Given extraction is complete, I see feature statistics: "Features Found" (count), "Interpretable" (percentage), "Activation Rate" (percentage)
- When I view the feature browser, I see a searchable feature table with columns: ID, Label, Example Context, Activation Freq, Interpretability, Actions
- When I type in the search box, features are filtered by label (case-insensitive partial match)
- When I click "Sort by" dropdown, I can select: "Activation Freq", "Interpretability", "Feature ID"
- When I click sort order toggle, sort direction switches between ascending/descending (icon rotates 180°)
- When search returns no results, I see message: "No features match your search"
- Given feature list is paginated, I see: "Showing {count} of {total} features" with Previous/Next buttons

**UI Reference:** Mock-embedded-interp-ui.tsx feature browser (lines 2434-2567)

---

### US-4: View Feature Details (Priority: P0)
**As a** researcher analyzing features
**I want to** click on a feature to see detailed analysis
**So that** I can understand what the feature detects

**Acceptance Criteria:**
- When I click a feature row in the table, a full-screen modal opens with feature details
- Given the modal is open, I see: Feature ID (e.g., "Feature #42"), editable label input, close button (X icon)
- When I edit the feature label and blur input, the label is saved
- Given feature statistics are displayed, I see 4 metrics: Activation Frequency (emerald), Interpretability (blue), Max Activation (purple), Active Samples count (yellow)
- When I view the modal, I see 4 tabs: "Examples", "Logit Lens", "Correlations", "Ablation"
- When I click a tab, the content area updates (tab gets emerald bottom border and text color)
- When I click close or press Escape, the modal dismisses

**UI Reference:** Mock-embedded-interp-ui.tsx FeatureDetailModal (lines 2587-2725)

---

### US-5: Analyze Max-Activating Examples (Priority: P0)
**As a** researcher viewing feature details
**I want to** see token sequences where the feature activates strongly
**So that** I can understand the semantic concept the feature represents

**Acceptance Criteria:**
- Given I'm on the "Examples" tab, I see a list of top-K activating contexts (default: top 100, showing 3-10 in UI)
- When I view an example, I see: Example number, Max activation value (3 decimal places), Token sequence with per-token activation highlighting
- When a token has high activation, it displays with: emerald background (opacity proportional to activation strength), brighter text color, border (if activation > 0.7)
- When I hover over a token, I see tooltip showing exact activation value (3 decimal places)
- When I view example context, tokens are displayed inline with monospace font, wrapped as needed
- Given multiple examples, they are ordered by max activation value (descending)

**UI Reference:** Mock-embedded-interp-ui.tsx MaxActivatingExamples (lines 2728-2800+), token highlighting (lines 2498-2518)

---

### US-6: Favorite Important Features (Priority: P1)
**As a** researcher tracking interesting features
**I want to** mark features as favorites
**So that** I can quickly return to features of interest

**Acceptance Criteria:**
- When I view the feature table, each row has a star icon in the Actions column
- When I click the star icon, it toggles between: unfavorited (hollow star, slate-500 color) and favorited (filled star, yellow-400 color)
- When I favorite a feature, the action persists across page refreshes
- When I click the star, the row click event (open modal) does not trigger (event propagation stops)
- Given I have favorited features, I can filter to show only favorites (future enhancement: add favorites filter)

**UI Reference:** Mock-embedded-interp-ui.tsx favorite toggle (lines 2531-2539)

---

### US-7: Analyze Feature with Logit Lens (Priority: P1)
**As an** interpretability researcher
**I want to** see what tokens the feature predicts when activated
**So that** I can understand the feature's role in next-token prediction

**Acceptance Criteria:**
- When I switch to "Logit Lens" tab, I see analysis of feature's effect on output logits
- Given logit lens analysis is available, I see: Top predicted tokens (list of 10), Probability distribution (bar chart or list), Semantic interpretation (text summary)
- When I view the token list, I see tokens ordered by probability (descending)
- Given the analysis suggests a pattern, I see interpretation text (e.g., "determiners and articles")

**UI Reference:** Mock-embedded-interp-ui.tsx LogitLensView (lines 2707-2709, implementation details in component)

---

### US-8: View Correlated Features (Priority: P1)
**As a** researcher understanding feature relationships
**I want to** see which other features activate in similar contexts
**So that** I can understand feature interactions and circuits

**Acceptance Criteria:**
- When I switch to "Correlations" tab, I see features that co-activate with the current feature
- Given correlations are computed, I see list of: Feature ID, Feature label, Correlation coefficient (0-1)
- When I view correlations, they are ordered by correlation strength (descending)
- When I click a correlated feature, I can navigate to that feature's detail view (future enhancement)

**UI Reference:** Mock-embedded-interp-ui.tsx FeatureCorrelations (lines 2711-2713)

---

### US-9: Understand Feature Impact via Ablation (Priority: P2)
**As a** researcher assessing feature importance
**I want to** see how removing the feature affects model performance
**So that** I can understand the feature's causal role in model behavior

**Acceptance Criteria:**
- When I switch to "Ablation" tab, I see analysis of feature's impact when ablated (set to zero)
- Given ablation analysis exists, I see: Perplexity delta (change in perplexity when feature ablated), Impact score (normalized importance metric 0-1)
- When perplexity increases significantly, I understand the feature is important for model performance
- Given the analysis, I can decide whether to use this feature for steering

**UI Reference:** Mock-embedded-interp-ui.tsx AblationAnalysis (lines 2715-2720)

---

### SC-1: Automated Feature Labeling (Priority: P2)
**As a** researcher with thousands of features
**I want to** automatically generate human-readable labels for features
**So that** I don't have to manually label every feature

**Acceptance Criteria:**
- When feature extraction completes, system generates automatic labels for top features based on max-activating examples
- When I view a feature, I can see the auto-generated label and edit if needed
- Given labels are generated, they use semantic patterns (e.g., "Sentiment Positive", "Code Syntax", "Question Pattern")

---

## 4. Functional Requirements

### FR-1: Training Selection and Status (8 requirements)

**FR-1.1:** System SHALL provide API endpoint `GET /api/trainings?status=completed` to retrieve completed training jobs
**FR-1.2:** System SHALL display training dropdown filtered to show only trainings with status='completed'
**FR-1.3:** System SHALL format training dropdown options as: `"{encoder_type} SAE • {model_name} • {dataset_name} • Started {date}"`
**FR-1.4:** System SHALL auto-select first completed training when none is selected
**FR-1.5:** System SHALL display training summary panel when training is selected showing: Model name, Dataset name, Encoder type
**FR-1.6:** System SHALL persist selected training ID in browser session storage
**FR-1.7:** System SHALL track extraction status per training in database `extraction_jobs` table
**FR-1.8:** System SHALL display appropriate UI state based on extraction status: 'not_started', 'extracting', 'completed', 'failed'

---

### FR-2: Feature Extraction Execution (14 requirements)

**FR-2.1:** System SHALL provide feature extraction configuration UI with fields:
- Evaluation Samples (integer, default 10000, range 1000-100000)
- Top-K Examples per Feature (integer, default 100, range 10-1000)

**FR-2.2:** System SHALL provide API endpoint `POST /api/trainings/:id/extract-features` to initiate extraction
**FR-2.3:** System SHALL create extraction job record in database with fields:
- id (format: `ext_{uuid}`)
- training_id
- status ('queued', 'extracting', 'completed', 'failed')
- progress (float 0-100)
- config (JSONB with evaluation_samples, top_k_examples)
- created_at, started_at, completed_at

**FR-2.4:** System SHALL execute feature extraction as Celery background task
**FR-2.5:** System SHALL perform the following extraction steps:
1. Load trained SAE checkpoint from training
2. Load dataset samples (evaluation_samples count)
3. For each sample: tokenize, extract model activations, pass through SAE encoder
4. Calculate per-feature statistics: activation_frequency, max_activation_value
5. Identify top-K max-activating examples for each feature
6. Store feature records in `features` table
7. Store activation examples in `feature_activations` table

**FR-2.6:** System SHALL calculate activation_frequency as: `count(activations > threshold) / total_samples` where threshold = 0.01
**FR-2.7:** System SHALL calculate interpretability_score using automated heuristic:
- Consistency of max-activating contexts (high consistency = high interpretability)
- Sparsity of activation (too sparse = low interpretability, too dense = low interpretability)
- Range: 0.0 to 1.0

**FR-2.8:** System SHALL store top-K max-activating examples per feature in `feature_activations` table with fields:
- feature_id, dataset_sample_id, tokens (JSONB array), activations (JSONB array), max_activation (float), context_before (text), context_after (text)

**FR-2.9:** System SHALL update extraction progress every 5% completion (emit WebSocket event `extraction:progress`)
**FR-2.10:** System SHALL limit parallel extractions to 1 per system (queue additional requests)
**FR-2.11:** System SHALL transition extraction status to 'completed' when all features extracted and stored
**FR-2.12:** System SHALL handle extraction errors gracefully: log error, set status='failed', store error_message
**FR-2.13:** System SHALL clean up partial extraction data if extraction fails
**FR-2.14:** System SHALL emit WebSocket event `extraction:completed` with feature statistics: total_features, avg_interpretability, avg_activation_freq

---

### FR-3: Feature Browsing and Search (12 requirements)

**FR-3.1:** System SHALL provide API endpoint `GET /api/trainings/:id/features` to retrieve features for a training
**FR-3.2:** System SHALL support query parameters for feature filtering/sorting:
- search (string): Filter by feature name (case-insensitive)
- sort_by (enum): 'activation_freq', 'interpretability', 'feature_id'
- sort_order (enum): 'asc', 'desc'
- limit (int, default 50): Results per page
- offset (int, default 0): Pagination offset
- is_favorite (boolean): Filter to favorites only

**FR-3.3:** System SHALL display feature statistics panel showing:
- "Features Found": Total count of features
- "Interpretable": Percentage with interpretability_score > 0.8
- "Activation Rate": Average activation_frequency across all features

**FR-3.4:** System SHALL provide feature browser table with columns:
- ID (feature neuron_index, monospace, slate-400)
- Label (feature name, editable)
- Example Context (token sequence with activation highlighting)
- Activation Freq (percentage, emerald-400, 2 decimal places)
- Interpretability (percentage, blue-400, 1 decimal place)
- Actions (favorite star icon)

**FR-3.5:** System SHALL implement client-side search filtering as user types (debounced 300ms)
**FR-3.6:** System SHALL highlight tokens in "Example Context" column using activation intensity:
- Background color: `rgba(16, 185, 129, ${intensity * 0.4})` (emerald with opacity)
- Text color: white if intensity > 0.6, else slate-300
- Border: 1px solid emerald if intensity > 0.7
- Tooltip: Show exact activation value on hover

**FR-3.7:** System SHALL display "No features match your search" message when filtered list is empty
**FR-3.8:** System SHALL implement pagination with "Previous" and "Next" buttons
**FR-3.9:** System SHALL display pagination info: "Showing {count} of {total} features"
**FR-3.10:** System SHALL make feature rows clickable to open feature detail modal
**FR-3.11:** System SHALL use full-text search index on `features` table for efficient search (GIN index)
**FR-3.12:** System SHALL persist search/sort preferences in browser session storage

---

### FR-4: Feature Detail Modal (10 requirements)

**FR-4.1:** System SHALL provide API endpoint `GET /api/features/:id` to retrieve feature details
**FR-4.2:** System SHALL display feature detail modal as full-screen overlay (fixed inset-0, black/50 background)
**FR-4.3:** System SHALL render modal header with:
- Feature ID (large font, e.g., "Feature #42")
- Editable label input (saves on blur)
- Close button (X icon, top-right)

**FR-4.4:** System SHALL display feature statistics in 4-column grid:
- Activation Frequency (emerald-400)
- Interpretability (blue-400)
- Max Activation (purple-400)
- Active Samples count (yellow-400)

**FR-4.5:** System SHALL provide 4 tabs: "Examples", "Logit Lens", "Correlations", "Ablation"
**FR-4.6:** System SHALL highlight active tab with: emerald-400 text color, emerald-400 bottom border (2px)
**FR-4.7:** System SHALL support modal dismiss via: close button click, Escape key press
**FR-4.8:** System SHALL save feature label edits via API `PATCH /api/features/:id` with payload: `{name: string}`
**FR-4.9:** System SHALL load tab content lazily (only fetch data when tab activated)
**FR-4.10:** System SHALL use modal backdrop click to close modal (click outside modal content)

---

### FR-5: Max-Activating Examples Display (9 requirements)

**FR-5.1:** System SHALL provide API endpoint `GET /api/features/:id/examples` to retrieve top-K max-activating examples
**FR-5.2:** System SHALL display examples in "Examples" tab sorted by max_activation descending
**FR-5.3:** System SHALL render each example in card layout (bg-slate-800/30 rounded-lg p-4) with:
- Example number header
- Max activation value (emerald-400, 3 decimal places)
- Token sequence with per-token highlighting

**FR-5.4:** System SHALL calculate token highlight intensity as: `intensity = activation / max_activation` (normalized 0-1)
**FR-5.5:** System SHALL apply token styling based on intensity:
- Background: `rgba(16, 185, 129, ${intensity * 0.4})`
- Text color: white if intensity > 0.6, else slate-300
- Border: 1px solid emerald-500 if intensity > 0.7

**FR-5.6:** System SHALL display tooltip on token hover showing: "Activation: {value}" (3 decimal places)
**FR-5.7:** System SHALL use monospace font for token sequences
**FR-5.8:** System SHALL wrap token sequences if they exceed container width
**FR-5.9:** System SHALL display "Showing {count} examples" header in Examples tab

---

### FR-6: Favorite Feature Management (6 requirements)

**FR-6.1:** System SHALL provide API endpoint `POST /api/features/:id/favorite` to mark feature as favorite
**FR-6.2:** System SHALL provide API endpoint `DELETE /api/features/:id/favorite` to unfavorite feature
**FR-6.3:** System SHALL update `features.is_favorite` boolean field in database on favorite toggle
**FR-6.4:** System SHALL display star icon in feature table Actions column:
- Unfavorited: hollow star, slate-500 color
- Favorited: filled star, yellow-400 color

**FR-6.5:** System SHALL prevent row click event when star icon clicked (event.stopPropagation())
**FR-6.6:** System SHALL support filtering feature list to show only favorites via `is_favorite=true` query param

---

### FR-7: Advanced Feature Analysis (Logit Lens, Correlations, Ablation) (11 requirements)

**FR-7.1:** System SHALL provide API endpoint `GET /api/features/:id/logit-lens` to retrieve logit lens analysis
**FR-7.2:** System SHALL calculate logit lens by:
1. Activate feature in SAE (set to high value, zero others)
2. Pass through SAE decoder to get reconstructed activation
3. Feed reconstructed activation through model to get output logits
4. Apply softmax to get token probabilities
5. Return top 10 tokens and probabilities

**FR-7.3:** System SHALL display Logit Lens tab content with:
- Top predicted tokens (list of 10)
- Probability per token (bar chart or percentage)
- Semantic interpretation text (auto-generated summary)

**FR-7.4:** System SHALL provide API endpoint `GET /api/features/:id/correlations` to retrieve correlated features
**FR-7.5:** System SHALL calculate feature correlations by:
- Compute activation vectors for all features across evaluation set
- Calculate Pearson correlation coefficient between current feature and all others
- Return top 10 correlated features (threshold: correlation > 0.5)

**FR-7.6:** System SHALL display Correlations tab with table:
- Feature ID (link to feature detail)
- Feature label
- Correlation coefficient (2 decimal places)

**FR-7.7:** System SHALL provide API endpoint `GET /api/features/:id/ablation` to retrieve ablation analysis
**FR-7.8:** System SHALL calculate ablation analysis by:
1. Run model on evaluation set with feature active (baseline perplexity)
2. Run model on same set with feature ablated (set to zero)
3. Calculate perplexity delta: `ablated_perplexity - baseline_perplexity`
4. Calculate impact score: normalized importance (0-1)

**FR-7.9:** System SHALL display Ablation tab with metrics:
- Perplexity Delta (float, 1 decimal place, higher = more important)
- Impact Score (percentage, 1 decimal place)

**FR-7.10:** System SHALL cache analysis results in database for performance (recalculate only if feature changes)
**FR-7.11:** System SHALL display loading spinner while analysis is computing

---

### FR-8: Automated Feature Labeling (6 requirements)

**FR-8.1:** System SHALL auto-generate feature labels during extraction using heuristic:
- Extract top 5 max-activating examples
- Identify common patterns (e.g., parts of speech, semantic concepts)
- Generate descriptive label (e.g., "Sentiment Positive", "Code Syntax")

**FR-8.2:** System SHALL use simple pattern matching for label generation:
- High activation on punctuation → "Punctuation"
- High activation on question words → "Question Pattern"
- High activation on code tokens → "Code Syntax"
- High activation on sentiment words → "Sentiment Positive/Negative"
- Fallback: "Feature {neuron_index}"

**FR-8.3:** System SHALL store auto-generated label in `features.name` field
**FR-8.4:** System SHALL allow users to override auto-generated labels via edit input
**FR-8.5:** System SHALL mark features as "auto-labeled" vs "user-labeled" via `features.label_source` field
**FR-8.6:** System SHALL re-generate labels if user clicks "Auto-label" button in feature detail (future enhancement)

---

## 5. Non-Functional Requirements

### Performance Requirements
- **Extraction Speed:** System SHALL complete feature extraction for 10,000 samples within 5 minutes on Jetson Orin Nano
- **Feature Browser Load Time:** Feature list SHALL load within 1 second for 2,000 features
- **Search Response Time:** Search filtering SHALL update UI within 300ms of user input
- **Modal Open Time:** Feature detail modal SHALL open within 500ms of row click

### Scalability Requirements
- **Feature Count:** System SHALL efficiently handle 16,384 features (typical SAE hidden dimension)
- **Example Storage:** System SHALL store 100 examples per feature (1.6M examples for 16K features)
- **Concurrent Extractions:** System SHALL queue extraction requests if already extracting

### Reliability Requirements
- **Extraction Fault Tolerance:** System SHALL resume extraction from last checkpoint if interrupted
- **Data Integrity:** System SHALL ensure atomic feature extraction (all features or none)

### Usability Requirements
- **Visual Clarity:** Token activation highlighting SHALL clearly show activation intensity
- **Intuitive Navigation:** Users SHALL understand how to navigate from feature list to details within 30 seconds
- **Search Responsiveness:** Search SHALL feel instant (debounced but responsive)

---

## 6. Technical Requirements

### Architecture Constraints (from ADR)
- **Backend Framework:** FastAPI with Celery for background extraction jobs
- **Database:** PostgreSQL 14+ with JSONB for tokens/activations storage, GIN indexes for full-text search
- **ML Framework:** PyTorch 2.0+ for SAE inference and activation extraction
- **Frontend:** React 18+ with Zustand state management, WebSocket for real-time progress

### Database Schema

#### extraction_jobs Table
```sql
CREATE TABLE extraction_jobs (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'ext_abc123'
    training_id VARCHAR(255) NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,

    -- Configuration
    config JSONB NOT NULL,  -- {evaluation_samples, top_k_examples}

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'queued',
        -- Values: 'queued', 'extracting', 'completed', 'failed'
    progress FLOAT DEFAULT 0,
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT extraction_jobs_status_check CHECK (status IN
        ('queued', 'extracting', 'completed', 'failed'))
);

CREATE INDEX idx_extraction_jobs_training_id ON extraction_jobs(training_id);
CREATE INDEX idx_extraction_jobs_status ON extraction_jobs(status);
```

#### features Table (from 003_SPEC|Postgres_Usecase_Details_and_Guidance.md)
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
    label_source VARCHAR(50) DEFAULT 'auto',  -- 'auto' or 'user'

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

#### feature_activations Table
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

-- Critical: Partition this table by feature_id ranges for performance
CREATE INDEX idx_feature_activations_feature ON feature_activations(feature_id);
CREATE INDEX idx_feature_activations_max_activation
    ON feature_activations(feature_id, max_activation DESC);
```

**Storage Estimate:** ~2KB per activation example × 100 examples × 16,384 features = ~3.2GB per training

#### feature_analysis_cache Table (for caching expensive analyses)
```sql
CREATE TABLE feature_analysis_cache (
    id BIGSERIAL PRIMARY KEY,
    feature_id BIGINT NOT NULL REFERENCES features(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL,  -- 'logit_lens', 'correlations', 'ablation'

    -- Cached results (JSONB)
    results JSONB NOT NULL,

    -- Cache metadata
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT feature_analysis_cache_unique UNIQUE (feature_id, analysis_type)
);

CREATE INDEX idx_feature_analysis_cache_feature ON feature_analysis_cache(feature_id);
CREATE INDEX idx_feature_analysis_cache_expires ON feature_analysis_cache(expires_at);
```

---

### API Specifications

#### GET /api/trainings/:id/extraction-status
**Purpose:** Get extraction status for training
**Response:** 200 OK
```json
{
  "extraction_id": "ext_abc123",
  "status": "extracting",
  "progress": 67.3,
  "started_at": "2025-10-06T10:00:00Z",
  "config": {
    "evaluation_samples": 10000,
    "top_k_examples": 100
  }
}
```

#### POST /api/trainings/:id/extract-features
**Purpose:** Start feature extraction
**Request Body:**
```json
{
  "evaluation_samples": 10000,
  "top_k_examples": 100
}
```
**Response:** 201 Created
```json
{
  "extraction_id": "ext_abc123",
  "status": "queued",
  "created_at": "2025-10-06T10:00:00Z"
}
```

#### GET /api/trainings/:id/features
**Purpose:** List features for training
**Query Parameters:**
- `search` (string): Filter by name
- `sort_by` (enum): activation_freq | interpretability | feature_id
- `sort_order` (enum): asc | desc
- `limit` (int): Max results (default 50)
- `offset` (int): Pagination offset
- `is_favorite` (bool): Filter favorites

**Response:** 200 OK
```json
{
  "features": [
    {
      "id": 1234,
      "neuron_index": 42,
      "name": "Sentiment Positive",
      "activation_frequency": 0.23,
      "interpretability_score": 0.94,
      "max_activation_value": 0.89,
      "is_favorite": false,
      "example_context": {
        "tokens": ["I", "really", "love", "this"],
        "activations": [0.02, 0.05, 0.89, 0.03]
      }
    }
  ],
  "total": 2048,
  "limit": 50,
  "offset": 0
}
```

#### GET /api/features/:id
**Purpose:** Get feature details
**Response:** 200 OK
```json
{
  "id": 1234,
  "training_id": "tr_abc123",
  "neuron_index": 42,
  "layer": 12,
  "name": "Sentiment Positive",
  "description": null,
  "activation_frequency": 0.23,
  "interpretability_score": 0.94,
  "max_activation_value": 0.89,
  "is_favorite": false,
  "active_samples": 2300,
  "created_at": "2025-10-06T10:30:00Z"
}
```

#### PATCH /api/features/:id
**Purpose:** Update feature metadata
**Request Body:**
```json
{
  "name": "Updated Label",
  "description": "Custom description"
}
```
**Response:** 200 OK (updated feature object)

#### POST /api/features/:id/favorite
**Purpose:** Mark feature as favorite
**Response:** 200 OK
```json
{
  "is_favorite": true
}
```

#### DELETE /api/features/:id/favorite
**Purpose:** Unfavorite feature
**Response:** 200 OK
```json
{
  "is_favorite": false
}
```

#### GET /api/features/:id/examples
**Purpose:** Get max-activating examples
**Query Parameters:**
- `limit` (int): Max examples (default 100)

**Response:** 200 OK
```json
{
  "examples": [
    {
      "id": 5678,
      "tokens": ["The", "cat", "sat", "on", "the", "mat"],
      "activations": [0.01, 0.12, 0.08, 0.03, 0.02, 0.05],
      "max_activation": 0.12,
      "sample_index": 4521
    }
  ]
}
```

#### GET /api/features/:id/logit-lens
**Purpose:** Get logit lens analysis
**Response:** 200 OK
```json
{
  "top_tokens": ["the", "a", "an", "this", "that"],
  "probabilities": [0.23, 0.18, 0.15, 0.12, 0.09],
  "interpretation": "determiners and articles"
}
```

#### GET /api/features/:id/correlations
**Purpose:** Get correlated features
**Response:** 200 OK
```json
{
  "correlations": [
    {
      "feature_id": 89,
      "name": "Sentence Start",
      "correlation": 0.87
    }
  ]
}
```

#### GET /api/features/:id/ablation
**Purpose:** Get ablation analysis
**Response:** 200 OK
```json
{
  "perplexity_delta": 2.3,
  "impact_score": 0.76,
  "baseline_perplexity": 12.4,
  "ablated_perplexity": 14.7
}
```

---

### WebSocket Events

**Channel:** `extraction:{extraction_id}`
**Events:**
- `extraction:progress` - Progress update (every 5%)
- `extraction:completed` - Extraction finished
- `extraction:failed` - Extraction error

**Example Event Payload (extraction:progress):**
```json
{
  "extraction_id": "ext_abc123",
  "progress": 67.3,
  "features_extracted": 11000,
  "total_features": 16384
}
```

---

## 7. UI/UX Specifications

### Feature Discovery Panel Layout

**Reference:** Mock-embedded-interp-ui.tsx FeaturesPanel (lines 2159-2584)

**Structure:**

1. **Header:**
   - Title: "Feature Discovery" (text-2xl font-semibold)

2. **Training Selector Panel:** (bg-slate-900/50 border border-slate-800 rounded-lg p-6)
   - Label: "Select Training Job to Analyze"
   - Dropdown: Full width, large text (text-lg), emerald focus border
   - Training Summary Cards: 3-column grid showing Model, Dataset, Encoder (bg-slate-800/30)

3. **Feature Extraction Panel:** (shown when training selected and extraction not complete)
   - **Configuration Inputs:** (grid-cols-2)
     - Evaluation Samples (number input, default 10000)
     - Top-K Examples per Feature (number input, default 100)
   - **Extract Button:** Full width, bg-emerald-600, lightning bolt icon
   - **Progress Bar:** (when extracting) emerald gradient, animated, with percentage

4. **Feature Statistics:** (shown when extraction complete)
   - 3-column grid of stat cards (bg-slate-800/50 rounded-lg p-4)
   - Features Found (emerald-400), Interpretable % (blue-400), Activation Rate % (purple-400)

5. **Feature Browser:** (border-t border-slate-700 pt-6)
   - **Search and Sort Controls:** (flex gap-3)
     - Search input: flex-1, placeholder "Search features..."
     - Sort dropdown: "Activation Freq" | "Interpretability" | "Feature ID"
     - Sort order toggle: Arrow icon, rotates 180° on toggle

   - **Feature Table:** (overflow-x-auto)
     - Columns: ID | Label | Example Context | Activation Freq | Interpretability | Actions
     - Rows: hover:bg-slate-800/30, cursor-pointer, clickable
     - Token highlighting: emerald background with opacity based on activation intensity

   - **Pagination:** (border-t border-slate-700 pt-4)
     - Info: "Showing {count} of {total} features"
     - Buttons: Previous | Next

---

### Feature Detail Modal Layout

**Reference:** Mock-embedded-interp-ui.tsx FeatureDetailModal (lines 2587-2725)

**Structure:**

1. **Modal Overlay:** (fixed inset-0 bg-black/50 z-50, centered)

2. **Modal Content:** (max-w-6xl bg-slate-900 rounded-lg max-h-90vh flex flex-col)

3. **Header:** (border-b border-slate-800 p-6)
   - **Title Row:**
     - Feature ID (text-2xl font-bold)
     - Close button (X icon, top-right)
   - **Label Input:** (max-w-md px-3 py-1 bg-slate-800 border border-slate-700)
   - **Statistics Grid:** (4 columns, bg-slate-800/50 rounded-lg p-3)
     - Activation Frequency (emerald-400)
     - Interpretability (blue-400)
     - Max Activation (purple-400)
     - Active Samples (yellow-400)

4. **Tabs:** (border-b border-slate-800)
   - Buttons: Examples | Logit Lens | Correlations | Ablation
   - Active tab: border-b-2 border-emerald-400 text-emerald-400

5. **Content Area:** (p-6 overflow-y-auto flex-1)
   - **Examples Tab:** Max-activating example cards with token highlighting
   - **Logit Lens Tab:** Top predicted tokens and probabilities
   - **Correlations Tab:** Table of correlated features
   - **Ablation Tab:** Perplexity delta and impact score

---

### Token Highlighting Visualization

**Reference:** Mock-embedded-interp-ui.tsx token highlighting (lines 2498-2518, 2750-2770)

**Algorithm:**
```javascript
const maxActivation = Math.max(...activations);
const intensity = activation / maxActivation; // Normalize 0-1

const style = {
  backgroundColor: `rgba(16, 185, 129, ${intensity * 0.4})`, // emerald-500
  color: intensity > 0.6 ? '#fff' : '#cbd5e1', // white or slate-300
  border: intensity > 0.7 ? '1px solid rgba(16, 185, 129, 0.5)' : 'none'
};
```

**Visual Effect:**
- Low activation (0.0-0.3): Barely visible emerald tint
- Medium activation (0.3-0.6): Noticeable emerald background, slate text
- High activation (0.6-0.7): Strong emerald background, white text
- Very high activation (0.7-1.0): Strong emerald background, white text, emerald border

---

## 8. Testing Requirements

### Unit Tests
- Test feature extraction logic (activation frequency calculation, top-K selection)
- Test auto-labeling heuristics (pattern matching, label generation)
- Test token highlighting intensity calculation (normalization, color mapping)
- Test search filtering (case-insensitive, partial match)
- Test sort logic (activation_freq, interpretability, feature_id, asc/desc)

### Integration Tests
- Test end-to-end extraction flow: POST extract → progress updates → completion
- Test feature browser: load features → search → sort → click → modal open
- Test favorite toggle: click star → API call → state update → persistence
- Test analysis tabs: click tab → API call → render results

### Performance Tests
- Benchmark extraction speed: 10,000 samples on Jetson Orin Nano (target < 5 minutes)
- Measure feature list load time: 16,384 features (target < 1 second)
- Measure search responsiveness: debounce delay and update time (target < 300ms)
- Measure modal open time: feature detail load (target < 500ms)

---

## 9. Dependencies

### Upstream Dependencies (Must be complete before implementation)
- **SAE Training (003_FPRD):** Requires completed training jobs with trained SAE checkpoints
- **Dataset Management (001_FPRD):** Requires tokenized datasets for evaluation samples
- **Model Management (002_FPRD):** Requires models for activation extraction

### Downstream Dependents (Blocked until this feature is complete)
- **Model Steering (005_FPRD):** Requires discovered features to perform interventions

### External Dependencies
- **PyTorch:** For SAE inference and activation extraction
- **NumPy:** For numerical operations (correlation, statistics)
- **WebSocket:** For real-time extraction progress updates

---

## 10. Risks & Mitigations

### Technical Risks

**Risk 1: Feature Extraction Too Slow on Edge Device**
- **Likelihood:** Medium
- **Impact:** High (poor UX if extraction takes >10 minutes)
- **Mitigation:**
  - Optimize extraction loop (batch processing, GPU acceleration)
  - Implement progress updates to show work is happening
  - Allow users to configure smaller evaluation sample sizes
  - Consider pre-extracting features immediately after training completes

**Risk 2: Feature Storage Grows Too Large**
- **Likelihood:** High
- **Impact:** Medium (disk space exhaustion)
- **Mitigation:**
  - Implement retention policy (delete features from old trainings)
  - Use database partitioning for `feature_activations` table
  - Allow users to delete extracted features they don't need
  - Consider storing only top-K examples (not all activations)

**Risk 3: Search Performance Degrades with Large Feature Sets**
- **Likelihood:** Medium
- **Impact:** Medium (slow search UX)
- **Mitigation:**
  - Use PostgreSQL full-text search with GIN indexes
  - Implement client-side debouncing (300ms) to reduce query frequency
  - Paginate results (50 features per page)
  - Consider Elasticsearch for very large feature sets (post-MVP)

---

### Usability Risks

**Risk 4: Auto-Generated Labels Are Not Useful**
- **Likelihood:** High
- **Impact:** Medium (users must manually label everything)
- **Mitigation:**
  - Provide easy label editing (click to edit, auto-save)
  - Allow users to disable auto-labeling
  - Consider LLM-based labeling (GPT-4 API) for better quality (post-MVP)
  - Show auto-label source so users know which labels to trust

**Risk 5: Token Highlighting is Confusing**
- **Likelihood:** Low
- **Impact:** Medium (users misinterpret feature meaning)
- **Mitigation:**
  - Include tooltip with exact activation values
  - Provide legend explaining color intensity mapping
  - Test with users to validate clarity
  - Consider alternative visualizations (heatmap, bar chart)

---

## 11. Future Enhancements

### Post-MVP Features (Not included in initial implementation)

1. **LLM-Based Automated Feature Descriptions**
   - Use GPT-4 to generate detailed descriptions from max-activating examples
   - Priority: P1, Timeline: Q2 2026

2. **Feature Clustering and Visualization**
   - t-SNE or UMAP projection of feature space
   - Interactive cluster exploration
   - Priority: P2, Timeline: Q3 2026

3. **Feature Composition Analysis**
   - Identify which features combine to form higher-level concepts
   - Circuit discovery automation
   - Priority: P2, Timeline: Q3 2026

4. **Export Feature Catalog**
   - Export features as JSON, CSV, or interactive HTML report
   - Share feature catalogs with research community
   - Priority: P2, Timeline: Q2 2026

5. **Real-Time Feature Search with Semantic Similarity**
   - Search by example text (find features that activate on similar context)
   - Embedding-based search (use sentence transformers)
   - Priority: P2, Timeline: Q4 2026

---

## 12. Open Questions

### Questions for Stakeholders

1. **Automatic Feature Extraction:** Should feature extraction start automatically when training completes, or require explicit user trigger?
   - **Recommendation:** Explicit trigger (gives users control), but add "Auto-extract" setting

2. **Feature Retention:** Should old features be automatically deleted after a certain period?
   - **Recommendation:** Keep features until user deletes training, warn if storage >80%

3. **Analysis Caching:** How long should expensive analyses (logit lens, ablation) be cached before recomputation?
   - **Recommendation:** Cache for 7 days, recompute if feature is updated

4. **Interpretability Score:** Should we expose the interpretability score calculation algorithm to users, or keep it opaque?
   - **Recommendation:** Keep opaque in MVP, add "About Scores" tooltip with high-level explanation

---

## 13. Success Metrics

### Feature Adoption Metrics
- **Extraction Usage Rate:** Percentage of completed trainings with features extracted (target: >80%)
- **Feature Browser Engagement:** Average time spent in feature browser per session (target: >5 minutes)
- **Favorite Feature Rate:** Average number of features favorited per user (target: >10)

### Quality Metrics
- **Auto-Label Quality:** Percentage of auto-generated labels kept by users (target: >60%)
- **Interpretability Score Accuracy:** Correlation between interpretability score and user ratings (target: >0.7)

### Performance Metrics
- **Extraction Speed:** Average time to extract 10K samples on Jetson Orin Nano (target: <5 minutes)
- **Search Latency:** 95th percentile search response time (target: <300ms)
- **Modal Load Time:** Average feature detail modal open time (target: <500ms)

---

## 14. Appendices

### A. Glossary

- **Feature:** An individual neuron in the SAE that has learned to detect a specific concept or pattern
- **Activation Frequency:** How often a feature activates above threshold (0.01) across evaluation samples
- **Interpretability Score:** Automated metric (0-1) estimating how human-interpretable a feature is
- **Max-Activating Examples:** Text samples where the feature had highest activation values
- **Token Highlighting:** Visual representation of per-token activation intensity using color/opacity
- **Logit Lens:** Analysis technique showing what tokens a feature predicts when activated
- **Feature Correlation:** Measure of how often two features co-activate (Pearson correlation)
- **Ablation:** Technique of setting a feature to zero to measure its causal impact on model output

### B. References

**Research Papers:**
- Anthropic (2023): "Towards Monosemanticity" - max-activating examples methodology
- Nostalgebraist (2020): "Interpreting GPT: The Logit Lens" - logit lens technique

**UI/UX Reference:**
- Mock-embedded-interp-ui.tsx (PRIMARY REFERENCE)
- Anthropic SAE Visualizer (public tool)
- TransformerLens Library (activation visualization patterns)

---

**Document End**
**Total Sections:** 14
**Total Functional Requirements:** 76 (across FR-1 through FR-8)
**Estimated Implementation Time:** 3-4 weeks (2 developers)
**Review Status:** Pending stakeholder review
