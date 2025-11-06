# TODO: UI Improvements & Bug Fixes
**Date:** 2025-11-05
**Priority:** Medium
**Estimated Time:** 8-12 hours

## Overview
This document outlines five UI/UX improvements requested by the user, with detailed implementation steps for each.

---

## 1. Model Tile - Extraction Status Indicator (Swirl Clock)

### Requirement
**Location:** Tab > Models > ModelTile (small swirl clock right of "Extract Activations" button)

**Current Behavior:** Status indicator exists but doesn't reflect extraction completion status

**Desired Behavior:**
- Grey (muted) when NO completed extraction jobs for the model
- Bright green when ONE OR MORE completed extraction jobs exist for the model

### Implementation Steps

#### Backend Tasks
- [ ] **1.1** Verify extraction_jobs table has model relationship
  - Check if we can query extractions by model via training relationship
  - Query: `extraction_jobs -> trainings -> model_id`

- [ ] **1.2** Create API endpoint or extend existing models endpoint
  - Add field: `has_completed_extractions: boolean`
  - Query: Check if any extraction_jobs exist with:
    - `status = 'completed'`
    - `training.model_id = <model_id>`

  **File:** `backend/src/api/v1/endpoints/models.py`
  ```python
  # Add to model response:
  has_completed_extractions = db.query(ExtractionJob).join(Training).filter(
      Training.model_id == model.id,
      ExtractionJob.status == 'completed'
  ).first() is not None
  ```

#### Frontend Tasks
- [ ] **1.3** Update ModelsStore to include extraction status
  - Add `hasCompletedExtractions` field to model type
  - Update API response parsing

  **File:** `frontend/src/stores/modelsStore.ts`

- [ ] **1.4** Locate ModelTile component and swirl clock element
  - Find the Clock/Loader icon component
  - Identify current styling

  **File:** `frontend/src/components/models/ModelTile.tsx` (or similar)

- [ ] **1.5** Implement conditional styling for swirl clock
  ```tsx
  <Clock
    className={model.hasCompletedExtractions
      ? "text-emerald-500"
      : "text-gray-400"
    }
  />
  ```

- [ ] **1.6** Test with multiple models
  - Model with no extractions → grey clock
  - Model with completed extraction → green clock
  - Model with only failed/running extractions → grey clock

---

## 2. Training Jobs Filter - Category Count Bug

### Requirement
**Location:** Tab > Training > TrainingJobs > Job filter buttons

**Current Behavior:** Category totals change based on visible/paginated results

**Desired Behavior:** Show total counts for each category across ALL training jobs, regardless of pagination/visibility

### Implementation Steps

#### Analysis
- [ ] **2.1** Identify TrainingJobs filter component
  - Locate filter buttons (All, Pending, Running, Completed, Failed, etc.)
  - Find where counts are calculated

  **Likely File:** `frontend/src/components/panels/TrainingPanel.tsx` or `frontend/src/components/training/TrainingList.tsx`

- [ ] **2.2** Trace count calculation logic
  - Currently: Counts from `filteredTrainings.filter(t => t.status === 'completed').length`
  - Should be: Separate API call for counts OR store-level aggregation

#### Backend Tasks
- [ ] **2.3** Verify API endpoint returns total counts
  - Check if `/api/v1/trainings` endpoint returns counts by status
  - If not, add `counts` field to response metadata

  **File:** `backend/src/api/v1/endpoints/trainings.py`
  ```python
  # In list_trainings endpoint, add:
  status_counts = {
      "pending": db.query(Training).filter(Training.status == 'pending').count(),
      "running": db.query(Training).filter(Training.status == 'running').count(),
      "completed": db.query(Training).filter(Training.status == 'completed').count(),
      "failed": db.query(Training).filter(Training.status == 'failed').count(),
  }
  return {
      "data": trainings,
      "pagination": {...},
      "counts": status_counts  # Add this
  }
  ```

#### Frontend Tasks
- [ ] **2.4** Update trainingsStore to fetch and store counts
  - Add `statusCounts` field to store
  - Update API response parsing

  **File:** `frontend/src/stores/trainingsStore.ts`

- [ ] **2.5** Update filter buttons to use total counts
  - Change from: `filteredTrainings.filter(...).length`
  - Change to: `trainingsStore.statusCounts[status]`

  **Example:**
  ```tsx
  <FilterButton
    label="Completed"
    count={trainingsStore.statusCounts.completed}  // Not filteredTrainings.length
  />
  ```

- [ ] **2.6** Test filter count behavior
  - Verify counts remain constant when switching filters
  - Verify counts update only when new trainings are added/completed
  - Test with pagination (counts should not change)

---

## 3. Discovered Features - Star Favorites (Of Interest)

### Requirement
**Location:** Tab > Features (Discovered Features panel)

**Current Behavior:**
- Stars exist under "Actions" column
- Toggle doesn't persist

**Desired Behavior:**
- Toggle stars to bright green when "on"
- Persist toggle state (database)
- Rename "Actions" column to "Of Interest"

### Implementation Steps

#### Database Tasks
- [ ] **3.1** Add `is_favorite` column to features table
  ```sql
  ALTER TABLE features
  ADD COLUMN is_favorite BOOLEAN DEFAULT FALSE;

  CREATE INDEX idx_features_favorite ON features(is_favorite);
  ```

  **File:** Create migration: `backend/alembic/versions/xxx_add_feature_favorites.py`

- [ ] **3.2** Update Feature model
  - Add `is_favorite` field

  **File:** `backend/src/models/feature.py` (or wherever Feature model is defined)

- [ ] **3.3** Update Feature schema
  - Add `is_favorite: bool = False` to response schema

  **File:** `backend/src/schemas/feature.py`

#### Backend API Tasks
- [ ] **3.4** Create PATCH endpoint for toggling favorite
  ```python
  @router.patch("/features/{feature_id}/favorite")
  async def toggle_feature_favorite(
      feature_id: str,
      is_favorite: bool,
      db: AsyncSession = Depends(get_db)
  ):
      feature = await db.get(Feature, feature_id)
      feature.is_favorite = is_favorite
      await db.commit()
      return {"success": True}
  ```

  **File:** `backend/src/api/v1/endpoints/features.py` (create if doesn't exist)

- [ ] **3.5** Update list features endpoint to return `is_favorite`
  - Ensure feature responses include the favorite status

#### Frontend Tasks
- [ ] **3.6** Update featuresStore with toggle method
  ```typescript
  async toggleFavorite(featureId: string, isFavorite: boolean) {
      await api.patch(`/features/${featureId}/favorite`, { is_favorite: isFavorite });
      // Update local state
      this.features = this.features.map(f =>
          f.id === featureId ? { ...f, isFavorite } : f
      );
  }
  ```

  **File:** `frontend/src/stores/featuresStore.ts`

- [ ] **3.7** Locate Features table/list component
  - Find "Actions" column
  - Locate star icon component

  **Likely File:** `frontend/src/components/features/FeaturesTable.tsx`

- [ ] **3.8** Rename "Actions" column to "Of Interest"
  ```tsx
  <th>Of Interest</th>  // Previously "Actions"
  ```

- [ ] **3.9** Implement star toggle with persistence
  ```tsx
  <Star
    className={feature.isFavorite ? "text-emerald-500 fill-emerald-500" : "text-gray-400"}
    onClick={() => featuresStore.toggleFavorite(feature.id, !feature.isFavorite)}
  />
  ```

- [ ] **3.10** Add filter for favorite features
  - Optional: Add "Favorites" filter button to show only starred features

- [ ] **3.11** Test favorite persistence
  - Toggle star on → verify API call → refresh page → star stays on
  - Test multiple features
  - Test filter if implemented

---

## 4. UI Normalization - Text Size & Spacing

### Requirement
**Goal:** Normalize interface so that at 100% zoom:
- All text is comfortable to read
- Text is not too large
- Compress width where possible
- Take advantage of blank spaces
- **Focus especially on tiles** (ModelTile, TrainingCard, DatasetCard, etc.)

### Implementation Steps

#### Analysis Phase
- [ ] **4.1** Audit current text sizes at 100% zoom
  - Document current font sizes for: headings, body, labels, buttons
  - Identify oversized text
  - Identify cramped areas

  **Create audit document:** `frontend/docs/UI_Text_Audit.md`

- [ ] **4.2** Screenshot all tiles at 100% zoom
  - ModelTile
  - TrainingCard
  - DatasetCard
  - ExtractionCard (if exists)
  - FeatureCard (if exists)

- [ ] **4.3** Identify blank spaces and compression opportunities
  - Excessive padding
  - Unnecessarily wide elements
  - Wasted horizontal space

#### Global Typography Improvements
- [ ] **4.4** Review Tailwind typography scale
  - Current: Check if using consistent scale (text-xs, text-sm, text-base, etc.)
  - Target sizes at 100% zoom:
    - Panel titles: `text-lg` or `text-xl` (18-20px)
    - Card titles: `text-base` or `text-lg` (16-18px)
    - Body text: `text-sm` (14px)
    - Labels/metadata: `text-xs` (12px)

- [ ] **4.5** Create/update global typography utilities
  **File:** `frontend/src/index.css` or `frontend/tailwind.config.js`
  ```css
  .tile-title { @apply text-base font-semibold; }
  .tile-body { @apply text-sm; }
  .tile-label { @apply text-xs text-gray-400; }
  ```

#### Tile-Specific Improvements
- [ ] **4.6** ModelTile spacing optimization
  - Reduce padding: `p-6` → `p-4`
  - Compress header spacing
  - Optimize button sizes

  **File:** `frontend/src/components/models/ModelTile.tsx`

- [ ] **4.7** TrainingCard spacing optimization
  - Reduce vertical gaps between sections
  - Compress hyperparameters display
  - Optimize chart sizes if present

  **File:** `frontend/src/components/training/TrainingCard.tsx`

- [ ] **4.8** DatasetCard spacing optimization
  - Compress statistics display
  - Reduce padding
  - Optimize icon sizes

  **File:** `frontend/src/components/datasets/DatasetCard.tsx`

- [ ] **4.9** Grid spacing optimization
  - Review grid gaps: `gap-6` → `gap-4` where appropriate
  - Ensure tiles don't feel cramped but use space efficiently

#### Width Compression
- [ ] **4.10** Identify fixed-width elements that can be flexible
  - Change `w-64` to `w-auto` or appropriate responsive width
  - Use `max-w-*` instead of `w-*` where possible

- [ ] **4.11** Optimize modal widths
  - Ensure modals are not unnecessarily wide
  - Use `max-w-2xl` or `max-w-4xl` instead of `max-w-6xl`

- [ ] **4.12** Review panel layouts
  - Ensure panels use available space efficiently
  - Compress side panels if too wide

#### Testing
- [ ] **4.13** Test at multiple zoom levels
  - 90%, 100%, 110%, 125%
  - Verify text remains readable at all zoom levels
  - Ensure no overflow or layout breaks

- [ ] **4.14** Test on different screen sizes
  - 1920x1080 (Full HD)
  - 2560x1440 (2K)
  - 1366x768 (Laptop)

- [ ] **4.15** Compare before/after screenshots
  - Document improvements
  - Verify compression doesn't sacrifice usability

---

## 5. Extraction Layer(s) Label Fix

### Requirement
**Location:** Extraction details display (likely in ExtractionCard or ExtractionDetailsModal)

**Current:** "Layers: 20"

**Desired:** "Layer(s): 20"

**Reason:** Disambiguate that it's the layer number(s) being extracted, not the number of layers

### Implementation Steps

- [ ] **5.1** Find all instances of "Layers:" label in extraction displays
  ```bash
  grep -r "Layers:" frontend/src/components
  ```

- [ ] **5.2** Update ExtractionCard/ExtractionTile (if exists)
  - Change: `Layers: {extraction.layers}`
  - To: `Layer(s): {extraction.layers}`

  **Potential Files:**
  - `frontend/src/components/extraction/ExtractionCard.tsx`
  - `frontend/src/components/extraction/ExtractionTile.tsx`

- [ ] **5.3** Update ExtractionDetailsModal (if exists)
  - Same change: "Layers:" → "Layer(s):"

  **File:** `frontend/src/components/extraction/ExtractionDetailsModal.tsx`

- [ ] **5.4** Update Training details if it shows extraction layer info
  - Check TrainingCard for extraction layer display

  **File:** `frontend/src/components/training/TrainingCard.tsx`

- [ ] **5.5** Update any tooltips or help text
  - Ensure consistency: "Layer(s)" throughout the app

- [ ] **5.6** Test display with different layer values
  - Single layer: "Layer(s): 0" (reads slightly odd but unambiguous)
  - Multiple layers: "Layer(s): 0, 5, 10"
  - Alternative format if preferred: "Layer(s) [0, 5, 10]"

---

## Implementation Order (Recommended)

### Phase 1: Quick Wins (2-3 hours)
1. ✅ **Task 5** - Layer(s) label fix (15 min)
2. ✅ **Task 1** - Extraction status indicator (1 hour)
3. ✅ **Task 2** - Training filter counts fix (1 hour)

### Phase 2: Feature Enhancement (3-4 hours)
4. ✅ **Task 3** - Star favorites with persistence (3-4 hours)
   - Database migration
   - Backend API
   - Frontend implementation
   - Testing

### Phase 3: UI Polish (3-5 hours)
5. ✅ **Task 4** - UI normalization and spacing (3-5 hours)
   - Audit phase
   - Systematic improvements
   - Testing across resolutions

---

## Testing Checklist

### Functional Testing
- [ ] Extraction status indicator updates correctly
- [ ] Training filter counts remain stable during pagination
- [ ] Feature favorites persist across page reloads
- [ ] Layer(s) label displays correctly

### Visual Testing
- [ ] Text is readable at 100% zoom on 1920x1080
- [ ] No text is oversized or undersized
- [ ] Tiles are compact but not cramped
- [ ] Spacing is consistent across panels
- [ ] Colors are correct (grey vs bright green)

### Responsive Testing
- [ ] Test on 1366x768 (laptop)
- [ ] Test on 1920x1080 (full HD)
- [ ] Test on 2560x1440 (2K)
- [ ] Test at 90%, 100%, 110%, 125% zoom

### Cross-browser Testing
- [ ] Chrome
- [ ] Firefox
- [ ] Safari (if available)

---

## Files to Modify (Summary)

### Backend
- [ ] `backend/alembic/versions/xxx_add_feature_favorites.py` (new)
- [ ] `backend/src/models/feature.py`
- [ ] `backend/src/schemas/feature.py`
- [ ] `backend/src/api/v1/endpoints/features.py` (create if needed)
- [ ] `backend/src/api/v1/endpoints/models.py`
- [ ] `backend/src/api/v1/endpoints/trainings.py`

### Frontend
- [ ] `frontend/src/stores/modelsStore.ts`
- [ ] `frontend/src/stores/trainingsStore.ts`
- [ ] `frontend/src/stores/featuresStore.ts`
- [ ] `frontend/src/components/models/ModelTile.tsx`
- [ ] `frontend/src/components/training/TrainingCard.tsx`
- [ ] `frontend/src/components/training/TrainingList.tsx`
- [ ] `frontend/src/components/features/FeaturesTable.tsx`
- [ ] `frontend/src/components/datasets/DatasetCard.tsx`
- [ ] `frontend/src/components/extraction/*` (various)
- [ ] `frontend/src/index.css` (global typography)
- [ ] `frontend/tailwind.config.js` (optional custom utilities)

---

## Notes

- All changes should maintain consistency with existing design patterns
- Follow the Mock UI specification for styling guidance
- Use Tailwind's emerald color palette for green states (`text-emerald-500`)
- Use gray palette for muted states (`text-gray-400`)
- Ensure WebSocket updates work correctly with new features
- Test database migrations in development before applying to production

---

**Created:** 2025-11-05
**Estimated Total Time:** 8-12 hours
**Priority Tasks:** 1, 2, 5 (quick wins)
**Complex Tasks:** 3, 4 (require more time)
