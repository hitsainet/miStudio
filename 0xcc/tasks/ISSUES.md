# Issues Task List

This document tracks bugs, defects, and issues discovered during development and testing that require remediation.

## Status Legend
- `[ ]` = Open issue
- `[x]` = Fixed/Resolved
- `[?]` = Needs investigation

---

## Recently Resolved Issues (November 2025)

### [ISS-007] Category and Description Columns Missing from Features Table
- **Status**: `[x]` RESOLVED (2025-11-10)
- **Severity**: High (critical data not visible to users)
- **Discovered**: 2025-11-10
- **Component**: Frontend - Features Table, Backend - Feature Service
- **Description**:
  - Category and Description columns existed in database and code but were not visible in browser
  - Two-part issue affecting both frontend and backend

- **Root Causes**:
  1. **Backend Schema Issue**: Pydantic response schemas (`FeatureResponse`, `FeatureDetailResponse`) were missing the `category` field entirely
  2. **Backend Service Issue**: Even after schema fix, service layer wasn't passing `category=feature.category` when constructing response objects
  3. **Frontend Component Issue**: ExtractionJobCard.tsx was missing Category and Description columns in its features table

- **Fixes Applied**:
  1. **Backend Schema** (commit ef418cc):
     - Added `category: Optional[str] = None` to `FeatureResponse` schema (line 100)
     - Added `category: Optional[str] = None` to `FeatureDetailResponse` schema (line 153)
     - File: `backend/src/schemas/feature.py`

  2. **Backend Service** (commit ef418cc):
     - Added `category=feature.category` to 4 locations in feature_service.py (lines 170, 340, 428, 498)
     - File: `backend/src/services/feature_service.py`

  3. **Frontend Component** (commit 5e2f802):
     - Added Category column header (line 461) and cell (lines 495-503)
     - Added Description column header (line 462) and cell (lines 504-512)
     - Updated colSpan from 6 to 8 (lines 472, 478)
     - File: `frontend/src/components/features/ExtractionJobCard.tsx`

- **User Verification**: User confirmed fix working: "the page I was having trouble with is working great now. Thank you."

- **Related Commits**:
  - ef418cc: Backend schema and service fixes
  - 5e2f802: Frontend ExtractionJobCard fixes

---

### [ISS-008] "Ġ" Characters Appearing in Token Display
- **Status**: `[x]` RESOLVED (2025-11-10)
- **Severity**: Low-Medium (cosmetic, affects readability)
- **Discovered**: 2025-11-10
- **Component**: Frontend - Token Highlighting
- **Description**:
  - BPE tokenizer markers appearing in token displays
  - "Ġ" (GPT-2 space marker), "▁" (SentencePiece), "##" (BERT) visible to users
  - Makes token text harder to read

- **Fix Applied** (commit ef418cc):
  - Added `cleanToken()` function to strip BPE markers (lines 27-32)
  - Applied to both `TokenHighlight` and `TokenHighlightCompact` components
  - Markers replaced with appropriate characters (space or nothing)
  - File: `frontend/src/components/features/TokenHighlight.tsx`

- **Implementation**:
  ```typescript
  const cleanToken = (token: string): string => {
    return token
      .replace(/^Ġ/g, ' ')  // GPT-2 space marker → space
      .replace(/^▁/g, ' ')  // SentencePiece space marker → space
      .replace(/^##/g, '');  // BERT continuation marker → nothing
  };
  ```

- **Related Commits**:
  - ef418cc: Token cleaning implementation

---

### [ISS-009] Documentation Files Scattered Across Multiple Locations
- **Status**: `[x]` RESOLVED (2025-11-10)
- **Severity**: Low (organizational)
- **Discovered**: 2025-11-10
- **Component**: Project Documentation
- **Description**:
  - .md files scattered across `backend/`, `backend/docs/`, `docs/` directories
  - Difficult to find and maintain documentation
  - Needed centralization to `0xcc/docs/`

- **Fix Applied** (commit d945163):
  - Moved 15 .md files to `0xcc/docs/` using `git mv` (preserves history)
  - Removed empty `backend/docs/` and `docs/` directories
  - Now 45 total .md files centralized in `0xcc/docs/`

- **Files Moved**:
  - From `backend/` (8 files):
    - CELERY_WORKERS.md
    - HOW_TEXT_CLEANING_WORKS.md
    - MULTI_TOKENIZATION_REFACTORING_PLAN.md
    - orphan_analysis.md
    - TEXT_CLEANING.md
    - TEXT_CLEANING_IMPLEMENTATION_COMPLETE.md
    - TEXT_CLEANING_TESTING_WORKFLOW.md
    - TRAINING_DELETION_FIX.md

  - From `backend/docs/` (6 files):
    - BUG_FIX_Training_Restart_2025-11-05.md
    - prompt_specificity_comparison.md
    - SAE_Hyperparameter_Optimization_Guide.md
    - steering_prompt_comparison.md
    - steering_prompt_example.md
    - TODO_UI_Improvements_2025-11-09.md

  - From `docs/` (1 file):
    - QUEUE_ARCHITECTURE.md

- **Related Commits**:
  - d945163: Documentation consolidation

---

## UI/UX Issues

### [ISS-001] Feature Labeling Progress Bar Frozen in UI
- **Status**: `[ ]` Open
- **Severity**: Medium
- **Discovered**: 2025-11-09
- **Component**: Frontend - Feature Discovery Panel
- **Description**:
  - The labeling progress bar in the UI appears to freeze and not update in real-time
  - Backend labeling job continues processing correctly (verified via API and database)
  - Job reached 100% completion (24,576 features labeled) but UI still shows earlier progress
  - WebSocket real-time updates may not be working correctly for labeling progress

- **Expected Behavior**:
  - Progress bar should update in real-time as labeling progresses
  - Should reflect accurate progress from WebSocket events or polling fallback

- **Actual Behavior**:
  - Progress bar freezes at an earlier percentage
  - Does not reflect actual backend progress

- **Reproduction Steps**:
  1. Start a feature labeling job
  2. Observe progress bar in Feature Discovery panel
  3. Progress bar stops updating after some time
  4. Backend continues processing (verifiable via API endpoint)

- **Technical Details**:
  - Backend endpoint: `GET /api/v1/labeling/{job_id}` returns correct progress
  - Labeling job ID format: `label_extr_{timestamp}_train_{id}_{timestamp}`
  - Database shows job progressing correctly (17,720 → 24,576 features)
  - Likely issue with WebSocket channel subscription or frontend state management

- **Related Files**:
  - `frontend/src/components/panels/FeatureDiscoveryPanel.tsx`
  - `frontend/src/stores/featuresStore.ts`
  - `frontend/src/hooks/useExtractionWebSocket.ts` (may need labeling equivalent)
  - `backend/src/api/v1/endpoints/labeling.py`
  - `backend/src/workers/websocket_emitter.py` (check labeling events)

- **Potential Root Causes**:
  1. Missing WebSocket subscription for labeling progress
  2. WebSocket channel name mismatch
  3. Frontend polling fallback not working
  4. State management not updating UI on progress events
  5. Component not re-rendering on store updates

- **Priority**: Medium (functionality works, but user feedback is poor)
- **Workaround**: User can refresh page or check backend logs/API for actual progress

---

### [ISS-006] Feature Favorite Toggle and Sorting
- **Status**: `[ ]` Open
- **Severity**: Enhancement
- **Discovered**: 2025-11-09
- **Component**: Frontend - Feature Browser
- **Description**:
  - Users need ability to mark features as favorites for quick access
  - Need to toggle favorite status (star icon) on/off directly in feature table
  - Need to filter and sort by favorite status
  - Backend already supports `is_favorite` field and toggle API endpoint

- **Desired Behavior**:
  - **Star Icon in Table**: Clickable star icon in "Of Interest" column
    - Filled star (⭐) for favorited features
    - Empty star (☆) for non-favorited features
    - Click to toggle favorite status
  - **Filter by Favorites**: Checkbox or toggle to show only favorited features
  - **Sort by Favorites**: Ability to sort table by favorite status (favorites first)
  - **Visual Feedback**: Loading state during toggle, error handling if toggle fails
  - **Persist State**: Favorite status persists across sessions (database-backed)

- **Current State**:
  - Database has `is_favorite` boolean field (default: false)
  - Backend API endpoint exists: `PATCH /api/v1/features/{feature_id}/favorite`
  - Frontend has `toggleFavorite()` method in featuresStore
  - UI shows "Of Interest" column but no interactive toggle
  - No filter or sort by favorite functionality

- **Implementation Requirements**:
  1. **Table Row Star Icon**:
     - Replace static text in "Of Interest" column with clickable star icon
     - Use Lucide React `Star` icon (outline for false, filled for true)
     - Add click handler that calls `toggleFavorite(feature.id)` with event.stopPropagation()
     - Show loading spinner during API call
     - Handle errors with toast notification

  2. **Filter by Favorites**:
     - Add "Show Favorites Only" checkbox above feature table
     - Update search filters to include `is_favorite: true` when checked
     - Clear checkbox when other filters change (optional)

  3. **Sort by Favorites**:
     - Add "Favorites" option to sort dropdown
     - Backend may need to add `is_favorite` to allowed sort_by fields
     - Sort order: favorites first (desc), then non-favorites

  4. **Visual Polish**:
     - Hover effect on star icon (scale up, color change)
     - Transition animation for star fill/unfill
     - Tooltip: "Mark as favorite" / "Remove from favorites"
     - Update table styling to make star stand out

- **Related Files**:
  - `frontend/src/components/features/FeaturesPanel.tsx` (add star icon in table)
  - `frontend/src/stores/featuresStore.ts` (already has toggleFavorite method)
  - `frontend/src/types/features.ts` (FeatureSearchRequest may need is_favorite filter)
  - `backend/src/api/v1/endpoints/features.py` (check sort_by validation)
  - `backend/src/services/features_service.py` (check if is_favorite sorting supported)

- **API Endpoints** (already implemented):
  - `GET /api/v1/features?training_id={id}&is_favorite=true` - Filter favorites
  - `PATCH /api/v1/features/{feature_id}/favorite` - Toggle favorite status
  - `GET /api/v1/features?sort_by=is_favorite&sort_order=desc` - Sort by favorites (may need backend update)

- **Example UI Code**:
  ```tsx
  // In "Of Interest" column
  <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
    <button
      onClick={() => handleToggleFavorite(feature.id)}
      disabled={isTogglingFavorite}
      className="text-slate-400 hover:text-yellow-400 transition-colors"
      title={feature.is_favorite ? "Remove from favorites" : "Mark as favorite"}
    >
      {feature.is_favorite ? (
        <Star className="w-5 h-5 fill-yellow-400 text-yellow-400" />
      ) : (
        <Star className="w-5 h-5" />
      )}
    </button>
  </td>
  ```

- **Priority**: Medium (improves user workflow for feature curation)
- **Dependencies**: None (backend already supports this)

---

## Backend Issues

### [ISS-002] Labeling Job Status Shows "labeling" After Completion
- **Status**: `[?]` Needs investigation
- **Severity**: Low-Medium
- **Discovered**: 2025-11-09
- **Component**: Backend - Labeling Service
- **Description**:
  - Labeling job status remains "labeling" even after all features are labeled
  - Database shows features_labeled = total_features but status ≠ "completed"
  - May be related to batch commit delay (COMMIT_BATCH_SIZE = 2000)

- **Investigation Needed**:
  - Check if final status update is being made after last batch
  - Verify completion detection logic in `labeling_service.py`
  - Check if WebSocket completion event is emitted

- **Related Files**:
  - `backend/src/services/labeling_service.py` (line ~428)
  - `backend/src/services/openai_labeling_service.py`
  - `backend/src/workers/labeling_tasks.py`

---

## Performance Issues

_(None currently tracked)_

---

## Feature Enhancements

### [ISS-003] Implement Three-Field Feature Labeling System
- **Status**: `[ ]` Open
- **Severity**: Enhancement
- **Discovered**: 2025-11-09
- **Component**: Backend - Labeling Service
- **Description**:
  - Current labeling system only populates `category` field
  - Database schema supports three fields: `category`, `name`, `description`
  - Need to populate all three fields for complete feature interpretation

- **Desired Behavior**:
  - **category** (varchar 255): High-level grouping (e.g., "names", "political_terms", "programming_syntax")
  - **name** (varchar 500): Specific label (e.g., "elizabeth_variations", "trump_mentions", "python_keywords")
  - **description** (text): Detailed interpretation based on actual activations
    - Example: "This feature activates for different variations of the name Elizabeth, including Lizzie, Liz, Beth, Betty, and Eliza. Strong activation on personal references and biographical content."

- **Current State**:
  - `category` field: Being populated (dual-label system implemented)
  - `name` field: Being populated as "specific_label" (dual-label system implemented)
  - `description` field: Empty, needs population with interpretation

- **Implementation Requirements**:
  1. Update OpenAI labeling prompt to request three fields: category, specific, interpretation
  2. Modify `_parse_dual_label()` to `_parse_triple_label()` in `openai_labeling_service.py`
  3. Update labeling service to save interpretation to `description` field
  4. Update frontend to display description/interpretation in feature cards
  5. Ensure full-text search index includes interpretation (already configured via `idx_features_fulltext_search`)

- **Related Files**:
  - `backend/src/services/openai_labeling_service.py` (lines 156-251: prompt and parsing)
  - `backend/src/services/labeling_service.py` (lines 417-424: database save)
  - `backend/src/models/feature.py` (database model)
  - Frontend feature display components

- **Example Output**:
  ```
  category: "names"
  name: "elizabeth_variations"
  description: "Activates for different forms of 'Elizabeth' including Lizzie, Liz, Beth, Betty, and Eliza. Shows strong response to personal references, biographical content, and name mentions in various contexts."
  ```

- **Priority**: Medium (enhancement to improve feature interpretability)
- **Dependencies**: Should be implemented together with ISS-004

---

### [ISS-004] Add Support for Local Ollama OpenAI-Compatible API
- **Status**: `[ ]` Open
- **Severity**: Enhancement
- **Discovered**: 2025-11-09
- **Component**: Backend - Labeling Service Configuration
- **Description**:
  - Currently hardcoded to use OpenAI API endpoint
  - Need to support local Ollama endpoint at `ollama.mcslab.io`
  - Ollama provides OpenAI-compatible API for local LLM inference

- **Desired Behavior**:
  - Configurable base URL for OpenAI-compatible API endpoints
  - Support both OpenAI.com and local Ollama endpoints
  - Allow selection of model and endpoint in UI

- **Implementation Requirements**:
  1. Add configuration settings for API base URL and model selection
  2. Update `openai_labeling_service.py` to use configurable base URL
  3. Add UI controls for selecting labeling provider (OpenAI vs Ollama)
  4. Add model selection dropdown for chosen provider
  5. Handle different authentication methods (API key for OpenAI, none for local Ollama)

- **Configuration Options Needed**:
  - `LABELING_API_BASE_URL`: Default to OpenAI, allow override for Ollama
  - `LABELING_MODEL`: Configurable model name (gpt-4o-mini, llama3.1, etc.)
  - `LABELING_API_KEY`: Optional, only required for OpenAI

- **Related Files**:
  - `backend/src/core/config.py` (add configuration settings)
  - `backend/src/services/openai_labeling_service.py` (use configurable base_url)
  - Frontend labeling configuration UI

- **Priority**: Medium (enables local inference for cost savings and privacy)
- **Dependencies**: Should be implemented together with ISS-003

---

### [ISS-005] Clean Token Data - Remove Junk Characters and Noise
- **Status**: `[ ]` Open
- **Severity**: Enhancement
- **Discovered**: 2025-11-09
- **Component**: Backend - Feature Extraction & Token Collection
- **Description**:
  - Token statistics used for feature labeling contain junk characters, punctuation, and noise
  - These useless tokens reduce labeling quality and make patterns harder to identify
  - Need to filter/clean token data before computing statistics and sending to LLM

- **Desired Behavior**:
  - Filter out pure punctuation tokens (e.g., ".", ",", "!", "?", ";", ":")
  - Remove special characters and noise (e.g., "\n", "\t", "\r", "▁", "Ġ")
  - Filter out single-character tokens that are not letters/numbers
  - Remove tokens that are only whitespace or control characters
  - Keep meaningful tokens: words, numbers, meaningful subwords
  - Apply filtering before computing token_stats and before labeling

- **Examples of Tokens to Remove**:
  - Pure punctuation: ".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "[", "]", "{", "}"
  - Whitespace tokens: " ", "\n", "\t", "\r", "▁" (Unicode space marker)
  - BPE markers: "Ġ" (GPT-2 space marker), "##" (BERT continuation)
  - Control characters: "\x00", "\ufffd" (replacement character)
  - Single special chars: "#", "@", "$", "%", "&", "*", "+", "=", "|", "\\", "/"

- **Examples of Tokens to Keep**:
  - Words: "Elizabeth", "trump", "python", "covid"
  - Subwords: "Liz", "beth", "tion", "ing"
  - Numbers: "2024", "100", "3.14"
  - Hyphenated words: "self-attention", "end-to-end"
  - Meaningful symbols in context: "C++", "F#", ".NET"

- **Implementation Requirements**:
  1. Add token filtering function in extraction service
  2. Apply filter to token_stats before computing top_k for labeling
  3. Add configuration option to enable/disable filtering (default: enabled)
  4. Add configuration for custom filter patterns (regex-based)
  5. Log statistics: tokens before/after filtering, percentage removed

- **Related Files**:
  - `backend/src/services/extraction_service.py` (token statistics collection)
  - `backend/src/services/labeling_service.py` (prepare token stats for labeling)
  - `backend/src/services/openai_labeling_service.py` (receives token stats)
  - `backend/src/core/config.py` (add token filtering configuration)

- **Filtering Function Pseudocode**:
  ```python
  def is_meaningful_token(token: str) -> bool:
      # Remove pure whitespace/control chars
      if token.strip() == "" or all(c in string.whitespace for c in token):
          return False

      # Remove BPE/tokenizer markers
      cleaned = token.replace("Ġ", "").replace("▁", "").replace("##", "")
      if cleaned.strip() == "":
          return False

      # Remove pure punctuation (no alphanumeric content)
      if not any(c.isalnum() for c in cleaned):
          return False

      # Keep if has meaningful content
      return True
  ```

- **Priority**: Medium (improves labeling quality and LLM token efficiency)
- **Dependencies**: Should be implemented before or alongside ISS-003 for best results

---

## Security Issues

_(None currently tracked)_

---

## Memory/Resource Issues

_(None currently tracked - extraction memory issues were resolved 2025-11-09)_

---

## Notes

- Issues should be referenced in commit messages when fixed: `fix: resolve ISS-001 frozen progress bar`
- Closed issues should be moved to a separate "ISSUES_RESOLVED.md" file periodically
- Use ISS-XXX format for issue IDs (3 digits, zero-padded)
