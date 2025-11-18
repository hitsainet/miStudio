# Feature Token Analysis Tab - Implementation Plan

## Overview

Add a new "Token Analysis" tab to the feature detail modal that displays a filtered, sorted list of tokens appearing in the feature's activation examples. The analysis will remove junk tokens (special markers, single characters, pure punctuation, etc.) and present meaningful tokens with their occurrence counts.

## Current State

**Feature Modal Tabs (existing):**
- Examples - Shows top-K activation examples
- Logit Lens - Shows predicted tokens analysis
- Correlations - Shows feature correlations
- Ablation - Shows ablation analysis

**Data Available:**
- Feature activations stored in `feature_activations` table
- Each activation has `tokens` field (JSONB array)
- 100 examples stored per feature (top-k max activations)

## Proposed UI

### New Tab: "Token Analysis"

**Location:** Fifth tab in feature modal, after "Ablation"

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ Token Analysis                                              │
├─────────────────────────────────────────────────────────────┤
│ Summary Statistics:                                         │
│   • Original tokens: 294                                    │
│   • Filtered tokens: 245                                    │
│   • Junk removed: 49                                        │
│   • Diversity: 58.8%                                        │
├─────────────────────────────────────────────────────────────┤
│ Rank  Token                          Count      %           │
│ ────  ─────                          ─────      ─          │
│   1   The                            9          3.67%       │
│   2   A                              6          2.45%       │
│   3   ious                           5          2.04%       │
│  ...                                                         │
│                                                             │
│ [Show more...] (if > 50 entries)                           │
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- Clean, tabular layout
- Shows rank, token (cleaned), count, percentage
- Summary stats at top
- Pagination or "show more" for long lists
- Default: show top 50, expandable to all

## Backend Implementation

### 1. API Endpoint

**New Endpoint:** `GET /api/v1/features/{feature_id}/token-analysis`

**Location:** `backend/src/api/v1/endpoints/features.py`

**Response Schema:**
```python
{
  "feature_id": "feat_train_36_03609",
  "summary": {
    "total_examples": 100,
    "original_token_count": 294,
    "filtered_token_count": 245,
    "junk_removed": 49,
    "total_token_occurrences": 500,
    "diversity_percent": 58.8
  },
  "tokens": [
    {
      "rank": 1,
      "token": "The",
      "count": 9,
      "percentage": 3.67
    },
    ...
  ]
}
```

### 2. Service Layer

**New Function:** `get_feature_token_analysis(db, feature_id)`

**Location:** `backend/src/services/feature_service.py`

**Logic:**
1. Query all `feature_activations` for the feature
2. Extract all tokens from JSONB `tokens` field
3. Count token occurrences using `Counter`
4. Apply junk filtering rules
5. Sort by count descending, then by token value
6. Calculate statistics
7. Return structured data

### 3. Junk Filter Rules

**Function:** `is_junk_token(token: str) -> bool`

**Location:** `backend/src/utils/token_filters.py` (new file)

**Rules:**
```python
def is_junk_token(token: str) -> bool:
    """
    Determine if a token should be filtered as junk.

    Filters:
    - Special tokens: <s>, </s>, <pad>, <unk>, BOM
    - Single characters (letters, numbers, punctuation)
    - Pure punctuation (no alphanumeric)
    - Short subword fragments without vowels (len <= 3)
    - Pure numbers
    - Only whitespace marker (▁)
    """
    import re

    # Special tokens
    if token in ['<s>', '</s>', '<pad>', '<unk>', '\ufeff']:
        return True

    # Single characters
    if len(token) <= 1:
        return True

    # Just punctuation
    if re.match(r'^[^a-zA-Z0-9]+$', token):
        return True

    # Short fragments without vowels
    if len(token) <= 3 and not re.search(r'[aeiouAEIOU]', token):
        return True

    # Just numbers
    if re.match(r'^[0-9]+$', token):
        return True

    # Only whitespace marker
    if token == '▁':
        return True

    return False
```

### 4. Token Cleaning

**Function:** `clean_token_display(token: str) -> str`

**Purpose:** Clean up token for display (remove ▁ prefix, etc.)

```python
def clean_token_display(token: str) -> str:
    """Remove tokenizer artifacts for cleaner display."""
    # Remove leading space marker
    cleaned = token.replace('▁', '').strip()

    # If empty after cleaning, return original
    if not cleaned:
        return token

    return cleaned
```

## Frontend Implementation

### 1. New Tab Component

**File:** `frontend/src/components/features/FeatureTokenAnalysis.tsx`

**Props:**
```typescript
interface FeatureTokenAnalysisProps {
  featureId: string;
}
```

**Structure:**
```tsx
export const FeatureTokenAnalysis: React.FC<FeatureTokenAnalysisProps> = ({
  featureId
}) => {
  const [data, setData] = useState<TokenAnalysisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    fetchTokenAnalysis(featureId);
  }, [featureId]);

  // Display logic
  const displayTokens = showAll ? data.tokens : data.tokens.slice(0, 50);

  return (
    <div className="p-4">
      {/* Summary Stats */}
      <SummaryStats data={data.summary} />

      {/* Token Table */}
      <TokenTable tokens={displayTokens} />

      {/* Show More Button */}
      {!showAll && data.tokens.length > 50 && (
        <button onClick={() => setShowAll(true)}>
          Show all {data.tokens.length} tokens
        </button>
      )}
    </div>
  );
};
```

### 2. API Client

**File:** `frontend/src/api/features.ts`

**New Function:**
```typescript
export const fetchFeatureTokenAnalysis = async (
  featureId: string
): Promise<TokenAnalysisData> => {
  const response = await axios.get(
    `${API_BASE_URL}/api/v1/features/${featureId}/token-analysis`
  );
  return response.data;
};
```

### 3. Types

**File:** `frontend/src/types/features.ts`

**New Types:**
```typescript
export interface TokenAnalysisSummary {
  total_examples: number;
  original_token_count: number;
  filtered_token_count: number;
  junk_removed: number;
  total_token_occurrences: number;
  diversity_percent: number;
}

export interface TokenEntry {
  rank: number;
  token: string;
  count: number;
  percentage: number;
}

export interface TokenAnalysisData {
  feature_id: string;
  summary: TokenAnalysisSummary;
  tokens: TokenEntry[];
}
```

### 4. Update Feature Modal

**File:** `frontend/src/components/features/FeatureModal.tsx`

**Changes:**
1. Add new tab to tab list
2. Import `FeatureTokenAnalysis` component
3. Add conditional rendering for token analysis tab

```tsx
const tabs = [
  { id: 'examples', label: 'Examples', icon: Activity },
  { id: 'logit', label: 'Logit Lens', icon: Info },
  { id: 'correlations', label: 'Correlations', icon: GitBranch },
  { id: 'ablation', label: 'Ablation', icon: Zap },
  { id: 'tokens', label: 'Token Analysis', icon: List }, // NEW
];

// In render:
{activeTab === 'tokens' && (
  <FeatureTokenAnalysis featureId={feature.id} />
)}
```

## Implementation Steps

### Phase 1: Backend (Estimated: 2-3 hours)

1. **Create token filter utilities** (`backend/src/utils/token_filters.py`)
   - [ ] Implement `is_junk_token()` function
   - [ ] Implement `clean_token_display()` function
   - [ ] Add unit tests for filter rules

2. **Add service layer function** (`backend/src/services/feature_service.py`)
   - [ ] Implement `get_feature_token_analysis()`
   - [ ] Query feature_activations
   - [ ] Apply filtering and counting
   - [ ] Calculate summary statistics
   - [ ] Return structured data

3. **Create API endpoint** (`backend/src/api/v1/endpoints/features.py`)
   - [ ] Add `GET /features/{feature_id}/token-analysis` route
   - [ ] Wire up service function
   - [ ] Add response schema validation
   - [ ] Add error handling

4. **Testing**
   - [ ] Test with Feature 299 (EOS detector - should show minimal tokens)
   - [ ] Test with Feature 3609 (diverse tokens - 245 filtered)
   - [ ] Verify filtering rules work correctly
   - [ ] Test edge cases (no activations, empty tokens)

### Phase 2: Frontend (Estimated: 2-3 hours)

1. **Create types** (`frontend/src/types/features.ts`)
   - [ ] Add `TokenAnalysisSummary` interface
   - [ ] Add `TokenEntry` interface
   - [ ] Add `TokenAnalysisData` interface

2. **Create API client** (`frontend/src/api/features.ts`)
   - [ ] Add `fetchFeatureTokenAnalysis()` function
   - [ ] Handle loading states
   - [ ] Handle errors

3. **Create Token Analysis component** (`frontend/src/components/features/FeatureTokenAnalysis.tsx`)
   - [ ] Create main component structure
   - [ ] Add summary stats display
   - [ ] Create token table
   - [ ] Add show more/less functionality
   - [ ] Add loading spinner
   - [ ] Add error handling

4. **Update Feature Modal** (`frontend/src/components/features/FeatureModal.tsx`)
   - [ ] Add "Token Analysis" tab
   - [ ] Import new component
   - [ ] Add conditional rendering
   - [ ] Choose appropriate icon (List or Hash)

5. **Styling**
   - [ ] Match existing modal styling
   - [ ] Use consistent table styles
   - [ ] Ensure responsive layout
   - [ ] Add hover states

6. **Testing**
   - [ ] Test with various features
   - [ ] Verify data displays correctly
   - [ ] Test pagination/show more
   - [ ] Test loading and error states

### Phase 3: Integration & Polish (Estimated: 1 hour)

1. **Integration Testing**
   - [ ] Test full flow: click feature → click Token Analysis tab → see data
   - [ ] Verify performance with large token lists
   - [ ] Test with features that have different token patterns

2. **Polish**
   - [ ] Add tooltips if needed
   - [ ] Optimize rendering for large lists
   - [ ] Add empty state handling
   - [ ] Performance optimization (memoization if needed)

3. **Documentation**
   - [ ] Update API documentation
   - [ ] Add code comments
   - [ ] Document filter rules

## Technical Considerations

### Performance

1. **Backend:**
   - Token analysis involves iterating through 100 examples per feature
   - Filtering happens in-memory (fast)
   - Consider caching results if performance is an issue
   - Query is not heavy (100 rows, small JSONB arrays)

2. **Frontend:**
   - Default to showing 50 tokens (avoid rendering thousands)
   - Use "show more" pattern rather than pagination
   - Lazy load component (only fetch when tab is clicked)

### Caching Strategy

**Option A: No caching (simple)**
- Calculate on every request
- Fast enough for 100 examples (~10ms)
- Always shows current data

**Option B: Redis caching (if needed)**
- Cache results for 1 hour
- Invalidate on feature update
- Only implement if performance becomes an issue

**Recommendation:** Start with Option A, add caching only if needed.

### Edge Cases

1. **No activations:** Show message "No activation examples available"
2. **All tokens filtered:** Show "No meaningful tokens found after filtering"
3. **Very long token list:** Use virtual scrolling if >500 tokens (unlikely)
4. **Feature not found:** Return 404 with clear error message

## Testing Strategy

### Backend Tests

**File:** `backend/tests/unit/test_token_filters.py`

```python
def test_is_junk_token():
    # Special tokens
    assert is_junk_token('<s>') == True
    assert is_junk_token('</s>') == True

    # Single chars
    assert is_junk_token('a') == True
    assert is_junk_token('1') == True

    # Punctuation
    assert is_junk_token(',') == True
    assert is_junk_token('.,!') == True

    # Short fragments without vowels
    assert is_junk_token('th') == True
    assert is_junk_token('str') == True

    # Valid tokens
    assert is_junk_token('The') == False
    assert is_junk_token('about') == False
    assert is_junk_token('news') == False
```

**File:** `backend/tests/integration/test_feature_token_analysis.py`

```python
async def test_get_feature_token_analysis():
    # Test with Feature 3609 (diverse tokens)
    result = await get_feature_token_analysis(db, "feat_train_36_03609")

    assert result['summary']['original_token_count'] > 0
    assert result['summary']['filtered_token_count'] < result['summary']['original_token_count']
    assert len(result['tokens']) == result['summary']['filtered_token_count']

    # Verify sorting (by count desc)
    counts = [t['count'] for t in result['tokens']]
    assert counts == sorted(counts, reverse=True)
```

### Frontend Tests

**File:** `frontend/src/components/features/FeatureTokenAnalysis.test.tsx`

```typescript
describe('FeatureTokenAnalysis', () => {
  it('renders summary statistics', async () => {
    render(<FeatureTokenAnalysis featureId="feat_test" />);
    await waitFor(() => {
      expect(screen.getByText(/Original tokens:/)).toBeInTheDocument();
    });
  });

  it('renders token table', async () => {
    render(<FeatureTokenAnalysis featureId="feat_test" />);
    await waitFor(() => {
      expect(screen.getByText('Rank')).toBeInTheDocument();
      expect(screen.getByText('Token')).toBeInTheDocument();
    });
  });

  it('shows "show more" for long lists', async () => {
    // Mock data with >50 tokens
    render(<FeatureTokenAnalysis featureId="feat_test" />);
    await waitFor(() => {
      expect(screen.getByText(/Show all/)).toBeInTheDocument();
    });
  });
});
```

## Success Criteria

- [ ] New "Token Analysis" tab appears in feature modal
- [ ] Tab displays filtered token list for Feature 3609 (245 tokens)
- [ ] Tab displays minimal tokens for Feature 299 (1 token after filtering)
- [ ] Summary statistics are accurate
- [ ] Tokens are sorted correctly (count desc, then alphabetically)
- [ ] UI is responsive and matches existing design
- [ ] Performance is acceptable (<100ms backend, <50ms frontend render)
- [ ] Error states are handled gracefully
- [ ] Code is tested and documented

## Future Enhancements (Optional)

1. **Search/Filter:** Add search box to filter tokens by text
2. **Export:** Export token list to CSV/JSON
3. **Token Context:** Click token to see examples where it appears
4. **Visualization:** Add word cloud or bar chart
5. **Comparison:** Compare token distributions across multiple features
6. **Custom Filters:** Allow user to customize junk filter rules

## Files to Create/Modify

### Backend
- ✅ Create: `backend/src/utils/token_filters.py`
- ✅ Modify: `backend/src/services/feature_service.py`
- ✅ Modify: `backend/src/api/v1/endpoints/features.py`
- ✅ Create: `backend/tests/unit/test_token_filters.py`
- ✅ Create: `backend/tests/integration/test_feature_token_analysis.py`

### Frontend
- ✅ Modify: `frontend/src/types/features.ts`
- ✅ Modify: `frontend/src/api/features.ts`
- ✅ Create: `frontend/src/components/features/FeatureTokenAnalysis.tsx`
- ✅ Modify: `frontend/src/components/features/FeatureModal.tsx`
- ✅ Create: `frontend/src/components/features/FeatureTokenAnalysis.test.tsx`

### Documentation
- ✅ This document: `0xcc/docs/Feature_Token_Analysis_Tab_Implementation_Plan.md`

## Estimated Timeline

- **Backend Implementation:** 2-3 hours
- **Frontend Implementation:** 2-3 hours
- **Integration & Testing:** 1 hour
- **Total:** 5-7 hours

## Notes

- Keep filter rules consistent with the analysis we did manually
- Ensure the tab is performant (lazy load, don't block other tabs)
- Match the existing UI/UX patterns in the feature modal
- Consider adding this to the feature card hover preview (future enhancement)

---

**Ready to implement:** Yes
**Priority:** Medium
**Dependencies:** None (uses existing feature_activations data)
