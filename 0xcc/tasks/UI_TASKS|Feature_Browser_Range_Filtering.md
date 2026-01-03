# Task: Feature Browser Range Filtering

## Overview
Add min/max range filtering capability for "Activation Frequency" and "Max Activation" columns in the Feature Browser. Users need to filter features to an optimal combination of ranges to identify highly activating, interpretable features.

## Requirements
- Min and max range inputs for both Activation Frequency and Max Activation columns
- Filters persist until affirmatively cleared or adjusted by the user
- Navigation (pagination, search) works on filtered population
- Default values: min=0, max=100 (includes all features)

## Target Files

### Backend
- `backend/src/schemas/feature.py` - Add range filter fields to FeatureSearchRequest
- `backend/src/services/feature_service.py` - Add WHERE clauses for range filtering
- `backend/src/api/v1/endpoints/features.py` - Accept new query parameters

### Frontend
- `frontend/src/types/features.ts` - Extend FeatureSearchRequest type
- `frontend/src/components/features/ExtractionJobCard.tsx` - Add range input UI
- `frontend/src/stores/featuresStore.ts` - Ensure filter persistence via searchFilters

---

## Implementation Tasks

### 1. Backend Schema Update (feature.py)
- [ ] Add 4 new Optional[float] fields to FeatureSearchRequest:

```python
# Add to class FeatureSearchRequest in backend/src/schemas/feature.py

min_activation_freq: Optional[float] = Field(
    default=None,
    ge=0,
    le=100,
    description="Minimum activation frequency filter (0-100)"
)
max_activation_freq: Optional[float] = Field(
    default=None,
    ge=0,
    le=100,
    description="Maximum activation frequency filter (0-100)"
)
min_max_activation: Optional[float] = Field(
    default=None,
    ge=0,
    description="Minimum max activation value filter"
)
max_max_activation: Optional[float] = Field(
    default=None,
    ge=0,
    description="Maximum max activation value filter"
)
```

### 2. Backend Service Update (feature_service.py)
- [ ] Add range filter WHERE clauses in `list_features()` method (around line 95-120)
- [ ] Add same filters to count_query for accurate pagination

```python
# Add after is_favorite filter section in list_features()

# Apply activation frequency range filter
if search_params.min_activation_freq is not None:
    query = query.where(Feature.activation_frequency >= search_params.min_activation_freq)
    count_query = count_query.where(Feature.activation_frequency >= search_params.min_activation_freq)

if search_params.max_activation_freq is not None:
    query = query.where(Feature.activation_frequency <= search_params.max_activation_freq)
    count_query = count_query.where(Feature.activation_frequency <= search_params.max_activation_freq)

# Apply max activation range filter
if search_params.min_max_activation is not None:
    query = query.where(Feature.max_activation >= search_params.min_max_activation)
    count_query = count_query.where(Feature.max_activation >= search_params.min_max_activation)

if search_params.max_max_activation is not None:
    query = query.where(Feature.max_activation <= search_params.max_max_activation)
    count_query = count_query.where(Feature.max_activation <= search_params.max_max_activation)
```

### 3. Backend API Endpoint Update (features.py)
- [ ] Verify list_features endpoint passes search_params to service correctly
- [ ] Add query parameter documentation for new range filters (optional, for OpenAPI docs)

### 4. Frontend Type Update (features.ts)
- [ ] Extend FeatureSearchRequest interface with 4 new optional fields:

```typescript
// Update in frontend/src/types/features.ts

export interface FeatureSearchRequest {
  search?: string | null;
  sort_by?: 'activation_freq' | 'max_activation' | 'feature_id';
  sort_order?: 'asc' | 'desc';
  is_favorite?: boolean | null;
  limit?: number;
  offset?: number;
  // New range filters
  min_activation_freq?: number | null;
  max_activation_freq?: number | null;
  min_max_activation?: number | null;
  max_max_activation?: number | null;
}
```

### 5. Frontend Store Update (featuresStore.ts)
- [ ] Verify `searchFilters[extraction.id]` persists range filter values
- [ ] Ensure `setSearchFilters()` handles new range fields
- [ ] Reset offset to 0 when range filters change (like existing filter behavior)

### 6. Frontend UI - Range Filter Inputs (ExtractionJobCard.tsx)
- [ ] Add range filter state (or use existing `filters` from searchFilters)
- [ ] Create collapsible "Filters" row below search bar
- [ ] Add 2 pairs of min/max number inputs:

```tsx
{/* Range Filters Section */}
<div className="flex flex-wrap items-center gap-4 mb-3 p-2 bg-slate-800/50 rounded">
  {/* Activation Frequency Range */}
  <div className="flex items-center gap-2">
    <span className="text-xs text-slate-400">Act. Freq:</span>
    <input
      type="number"
      min={0}
      max={100}
      step={0.1}
      placeholder="Min"
      value={filters.min_activation_freq ?? ''}
      onChange={(e) => handleRangeFilterChange('min_activation_freq', e.target.value)}
      className="w-16 px-2 py-1 bg-slate-800 border border-slate-700 rounded text-slate-100 text-xs focus:outline-none focus:border-emerald-500"
    />
    <span className="text-slate-500">-</span>
    <input
      type="number"
      min={0}
      max={100}
      step={0.1}
      placeholder="Max"
      value={filters.max_activation_freq ?? ''}
      onChange={(e) => handleRangeFilterChange('max_activation_freq', e.target.value)}
      className="w-16 px-2 py-1 bg-slate-800 border border-slate-700 rounded text-slate-100 text-xs focus:outline-none focus:border-emerald-500"
    />
  </div>

  {/* Max Activation Range */}
  <div className="flex items-center gap-2">
    <span className="text-xs text-slate-400">Max Act:</span>
    <input
      type="number"
      min={0}
      step={0.1}
      placeholder="Min"
      value={filters.min_max_activation ?? ''}
      onChange={(e) => handleRangeFilterChange('min_max_activation', e.target.value)}
      className="w-16 px-2 py-1 bg-slate-800 border border-slate-700 rounded text-slate-100 text-xs focus:outline-none focus:border-emerald-500"
    />
    <span className="text-slate-500">-</span>
    <input
      type="number"
      min={0}
      step={0.1}
      placeholder="Max"
      value={filters.max_max_activation ?? ''}
      onChange={(e) => handleRangeFilterChange('max_max_activation', e.target.value)}
      className="w-16 px-2 py-1 bg-slate-800 border border-slate-700 rounded text-slate-100 text-xs focus:outline-none focus:border-emerald-500"
    />
  </div>

  {/* Clear Filters Button */}
  <button
    type="button"
    onClick={handleClearRangeFilters}
    className="px-2 py-1 text-xs text-slate-400 hover:text-slate-200 transition-colors"
    title="Clear range filters"
  >
    Clear
  </button>
</div>
```

### 7. Frontend Event Handlers (ExtractionJobCard.tsx)
- [ ] Add `handleRangeFilterChange()` function with debouncing (similar to search)
- [ ] Add `handleClearRangeFilters()` function to reset range values
- [ ] Ensure filter changes reset pagination offset to 0
- [ ] Trigger API refetch when filters change

```typescript
// Debounced range filter handler
const handleRangeFilterChange = (field: string, value: string) => {
  const numValue = value === '' ? null : parseFloat(value);

  // Clear existing debounce timer
  if (rangeDebounceTimer) {
    clearTimeout(rangeDebounceTimer);
  }

  // Set new timer for debounced API call
  const timer = window.setTimeout(() => {
    setSearchFilters(extraction.id, {
      ...filters,
      [field]: numValue,
      offset: 0  // Reset pagination
    });
    fetchExtractionFeatures(extraction.id);
  }, 500);  // 500ms debounce

  setRangeDebounceTimer(timer);
};

const handleClearRangeFilters = () => {
  setSearchFilters(extraction.id, {
    ...filters,
    min_activation_freq: null,
    max_activation_freq: null,
    min_max_activation: null,
    max_max_activation: null,
    offset: 0
  });
  fetchExtractionFeatures(extraction.id);
};
```

### 8. Styling
- [ ] Match existing search bar styling (slate-800 backgrounds, slate-700 borders)
- [ ] Use consistent sizing (text-xs for filter labels)
- [ ] Inputs should be compact (w-16 or w-20)
- [ ] Consider making filters collapsible to save vertical space

### 9. Testing
- [ ] Test range filtering on Activation Frequency (min only, max only, both)
- [ ] Test range filtering on Max Activation (min only, max only, both)
- [ ] Test combined range filters (both columns filtered simultaneously)
- [ ] Verify pagination shows correct total count after filtering
- [ ] Verify "Clear" button resets all range filters
- [ ] Verify filters persist when navigating between pages
- [ ] Verify filters persist when expanding/collapsing feature browser
- [ ] Test with edge cases: min > max, empty inputs, very large values

---

## Notes

### Current Filter Persistence Mechanism
The existing `searchFilters[extraction.id]` object in `featuresStore.ts` already handles per-extraction filter persistence. The new range filter fields will be stored alongside existing fields (search, sort_by, sort_order, etc.).

### Default Behavior
- Empty filter inputs (null/undefined) should NOT apply any filter (show all)
- Default display should show all features (equivalent to min=0, max=∞)
- Users can optionally apply filters by entering values

### Pagination Integration
- When any range filter changes, offset must reset to 0
- Total count in pagination should reflect filtered population
- Page navigation should work correctly on filtered dataset

### Performance Considerations
- Backend filters are applied via SQL WHERE clauses (efficient)
- Debounce range input changes to avoid excessive API calls
- Consider adding indexes on `activation_frequency` and `max_activation` columns if performance degrades with large datasets

---

## Related Files
- `backend/src/models/feature.py` - Feature model with activation_frequency, max_activation columns
- `frontend/src/api/features.ts` - API client functions for feature list endpoint
- `frontend/src/stores/featuresStore.ts` - Zustand store managing feature data and filters

## Priority
Medium - Essential for feature discovery workflow to identify optimal features

## Estimated Effort
~2-3 hours (backend: 30min, frontend: 1.5-2hrs, testing: 30min)
