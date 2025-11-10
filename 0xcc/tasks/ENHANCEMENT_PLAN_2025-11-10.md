# Enhancement Implementation Plan - November 2025

**Created:** 2025-11-10
**Status:** Planning Phase
**Total Enhancements:** 4 (ISS-003, ISS-004, ISS-005, ISS-006)
**Estimated Total Time:** 12-18 hours

---

## Executive Summary

This plan addresses 4 medium-priority enhancements that will significantly improve feature interpretability, labeling quality, and user workflow. These enhancements build on the existing dual-label system and feature discovery infrastructure completed in November 2025.

**Key Goals:**
1. Improve feature interpretability with richer labeling (ISS-003, ISS-005)
2. Enable cost-effective local LLM labeling (ISS-004)
3. Improve feature curation workflow (ISS-006)

---

## Enhancement Priority & Dependencies

```
┌─────────────────────────────────────────────────────┐
│ PHASE 1: Foundation (4-6 hours)                    │
│ ┌───────────────────────────────────────────────┐  │
│ │ ISS-005: Token Data Cleaning                  │  │
│ │ Priority: HIGH (enables better labeling)      │  │
│ │ Dependencies: None                            │  │
│ └───────────────────────────────────────────────┘  │
│                         │                           │
│                         ├──► Better token stats     │
│                         │                           │
│ ┌───────────────────────────────────────────────┐  │
│ │ ISS-006: Feature Favorites Toggle             │  │
│ │ Priority: MEDIUM (improves UX)                │  │
│ │ Dependencies: None (backend ready)            │  │
│ └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ PHASE 2: Enhanced Labeling (8-12 hours)            │
│ ┌───────────────────────────────────────────────┐  │
│ │ ISS-003: Three-Field Labeling System          │  │
│ │ Priority: MEDIUM (rich interpretations)       │  │
│ │ Dependencies: ISS-005 (cleaner tokens)        │  │
│ └───────────────────────────────────────────────┘  │
│                         │                           │
│                         ├──► 3-field system ready   │
│                         │                           │
│ ┌───────────────────────────────────────────────┐  │
│ │ ISS-004: Ollama Local LLM Support             │  │
│ │ Priority: LOW-MEDIUM (cost savings)           │  │
│ │ Dependencies: ISS-003 (shares infra)          │  │
│ └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## Enhancement 1: Token Data Cleaning [ISS-005]

**Priority:** HIGH (Foundation for better labeling)
**Effort:** 2-3 hours
**Dependencies:** None
**Status:** Ready to implement

### Problem
Token statistics sent to LLM labeling contain:
- Pure punctuation tokens (".", ",", "!", etc.)
- BPE markers ("Ġ", "▁", "##")
- Control characters ("\n", "\t", "\x00")
- Single special characters ("#", "@", "$")

These junk tokens:
- Reduce LLM token efficiency (wasted prompt space)
- Confuse the model about feature semantics
- Make patterns harder to identify

### Solution: Token Filtering Pipeline

**Implementation Steps:**

#### Step 1: Create Token Filter Utility (30 min)
**File:** `backend/src/utils/token_filter.py` (NEW)

```python
"""Token filtering utilities for feature labeling."""
import re
import string
from typing import Dict, List

class TokenFilter:
    """Filters out junk tokens from activation statistics."""

    def __init__(
        self,
        remove_punctuation: bool = True,
        remove_bpe_markers: bool = True,
        remove_single_chars: bool = True,
        min_token_length: int = 2,
        keep_patterns: List[str] = None
    ):
        self.remove_punctuation = remove_punctuation
        self.remove_bpe_markers = remove_bpe_markers
        self.remove_single_chars = remove_single_chars
        self.min_token_length = min_token_length
        self.keep_patterns = keep_patterns or [r'C\+\+', r'F#', r'\.NET']

    def is_meaningful_token(self, token: str) -> bool:
        """Check if token contains meaningful information."""
        # Check keep patterns first (e.g., "C++", "F#")
        for pattern in self.keep_patterns:
            if re.match(pattern, token):
                return True

        # Remove BPE markers for checking
        if self.remove_bpe_markers:
            cleaned = token.replace("Ġ", "").replace("▁", "").replace("##", "")
        else:
            cleaned = token

        # Filter empty or whitespace-only
        if not cleaned or cleaned.strip() == "":
            return False

        # Filter control characters
        if any(ord(c) < 32 for c in cleaned):
            return False

        # Filter single characters (unless it's alphanumeric)
        if self.remove_single_chars and len(cleaned) == 1:
            if not cleaned.isalnum():
                return False

        # Filter pure punctuation (no alphanumeric content)
        if self.remove_punctuation:
            if not any(c.isalnum() for c in cleaned):
                return False

        # Filter by minimum length
        if len(cleaned) < self.min_token_length:
            return False

        return True

    def filter_token_stats(
        self,
        token_stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Filter token statistics dictionary."""
        filtered = {}
        for token, stats in token_stats.items():
            if self.is_meaningful_token(token):
                filtered[token] = stats

        return filtered

    def get_filter_stats(
        self,
        original: Dict[str, Dict[str, float]],
        filtered: Dict[str, Dict[str, float]]
    ) -> Dict[str, int]:
        """Get statistics about filtering."""
        return {
            "original_count": len(original),
            "filtered_count": len(filtered),
            "removed_count": len(original) - len(filtered),
            "removal_percentage": ((len(original) - len(filtered)) / len(original) * 100)
                                  if original else 0
        }

# Pre-configured filters
def get_standard_filter() -> TokenFilter:
    """Standard filter for most use cases."""
    return TokenFilter(
        remove_punctuation=True,
        remove_bpe_markers=True,
        remove_single_chars=True,
        min_token_length=2
    )

def get_aggressive_filter() -> TokenFilter:
    """Aggressive filter for noisy data."""
    return TokenFilter(
        remove_punctuation=True,
        remove_bpe_markers=True,
        remove_single_chars=True,
        min_token_length=3
    )

def get_minimal_filter() -> TokenFilter:
    """Minimal filter (only control chars and whitespace)."""
    return TokenFilter(
        remove_punctuation=False,
        remove_bpe_markers=True,
        remove_single_chars=False,
        min_token_length=1
    )
```

#### Step 2: Add Configuration (15 min)
**File:** `backend/src/core/config.py`

```python
# Token filtering configuration
token_filter_enabled: bool = Field(
    default=True,
    description="Enable token filtering for feature labeling"
)
token_filter_mode: str = Field(
    default="standard",
    description="Token filter mode: standard, aggressive, minimal"
)
```

#### Step 3: Integrate into Labeling Service (1 hour)
**File:** `backend/src/services/labeling_service.py`

**Changes:**
- Import TokenFilter utilities (line ~10)
- Initialize filter in `__init__()` based on config
- Apply filter before computing top_k tokens for labeling
- Log filtering statistics

**Integration point** (around line 230-250):
```python
# Before sending to LLM labeling
if self.token_filter:
    filtered_stats = self.token_filter.filter_token_stats(token_stats)
    filter_stats = self.token_filter.get_filter_stats(token_stats, filtered_stats)

    logger.info(
        f"Token filtering: {filter_stats['original_count']} → "
        f"{filter_stats['filtered_count']} tokens "
        f"({filter_stats['removal_percentage']:.1f}% removed)"
    )

    token_stats = filtered_stats
```

#### Step 4: Add Tests (30 min)
**File:** `backend/tests/test_token_filter.py` (NEW)

Test cases:
- `test_removes_punctuation()` - Pure punctuation filtered
- `test_removes_bpe_markers()` - BPE markers filtered
- `test_keeps_meaningful_tokens()` - Words, numbers kept
- `test_keeps_special_patterns()` - "C++", "F#" kept
- `test_filter_stats()` - Statistics calculation correct

#### Step 5: Documentation (15 min)
**File:** `0xcc/docs/Token_Filtering.md` (NEW)

Document:
- What tokens are filtered and why
- Configuration options (standard/aggressive/minimal)
- Performance impact (minimal - filtering is fast)
- Example before/after statistics

**Estimated Time:** 2-3 hours
**Testing:** Backend unit tests + manual verification with labeling job
**Acceptance Criteria:**
- ✅ TokenFilter utility created and tested
- ✅ Integration into labeling service complete
- ✅ Logging shows filtering statistics
- ✅ LLM prompts contain only meaningful tokens
- ✅ Labeling quality improves (subjective, but noticeable)

---

## Enhancement 2: Feature Favorites Toggle [ISS-006]

**Priority:** MEDIUM (Improves UX)
**Effort:** 2-3 hours
**Dependencies:** None (backend already complete)
**Status:** Ready to implement

### Problem
Users need to mark features as "of interest" for quick access:
- No way to toggle favorite status in UI
- No way to filter by favorites
- No way to sort favorites first

Backend already supports this (`is_favorite` field, toggle API), just needs frontend.

### Solution: Interactive Star Icon with Filtering

**Implementation Steps:**

#### Step 1: Update FeaturesPanel Star Icon (1 hour)
**File:** `frontend/src/components/features/FeaturesPanel.tsx`

**Changes:**
1. Import Star icon from lucide-react
2. Replace static "Of Interest" column content with clickable star button
3. Add click handler with event.stopPropagation()
4. Add loading state during toggle
5. Add error handling

**Code location** (around line 480-500, in the table row mapping):
```typescript
// Replace current "Of Interest" column
<td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
  <button
    onClick={() => handleToggleFavorite(feature.id, !feature.is_favorite)}
    disabled={togglingFavorite === feature.id}
    className={`relative transition-all duration-200 ${
      feature.is_favorite
        ? 'text-yellow-400 hover:text-yellow-500'
        : 'text-slate-500 hover:text-yellow-400'
    } ${togglingFavorite === feature.id ? 'opacity-50 cursor-wait' : 'cursor-pointer hover:scale-110'}`}
    title={feature.is_favorite ? "Remove from favorites" : "Mark as favorite"}
  >
    {togglingFavorite === feature.id ? (
      <Loader2 className="w-5 h-5 animate-spin" />
    ) : feature.is_favorite ? (
      <Star className="w-5 h-5 fill-yellow-400" />
    ) : (
      <Star className="w-5 h-5" />
    )}
  </button>
</td>
```

Add handler function:
```typescript
const [togglingFavorite, setTogglingFavorite] = useState<string | null>(null);

const handleToggleFavorite = async (featureId: string, newValue: boolean) => {
  setTogglingFavorite(featureId);
  try {
    await featuresStore.toggleFavorite(featureId, newValue);
    // Optimistic update - store already handles this
  } catch (error) {
    console.error('Failed to toggle favorite:', error);
    // Show error toast (if toast system exists)
  } finally {
    setTogglingFavorite(null);
  }
};
```

#### Step 2: Add Favorites Filter (45 min)
**Location:** Above the features table, in the filters section

```typescript
<div className="flex items-center gap-2">
  <input
    type="checkbox"
    id="favorites-only"
    checked={filters.is_favorite || false}
    onChange={(e) => handleFilterChange({ is_favorite: e.target.checked || undefined })}
    className="w-4 h-4 text-emerald-500 border-slate-600 rounded focus:ring-emerald-500"
  />
  <label htmlFor="favorites-only" className="text-sm text-slate-300">
    Show favorites only
  </label>
</div>
```

Update filters state type to include `is_favorite?: boolean`

#### Step 3: Add Favorites Sort Option (30 min)
**Location:** Sort dropdown

Add to sort options:
```typescript
const sortOptions = [
  { value: 'neuron_index', label: 'Feature ID' },
  { value: 'activation_frequency', label: 'Activation Frequency' },
  { value: 'interpretability_score', label: 'Interpretability' },
  { value: 'is_favorite', label: 'Favorites First' },  // NEW
];
```

Backend should already support `sort_by=is_favorite` - verify in `feature_service.py`

#### Step 4: Update ExtractionJobCard (30 min)
**File:** `frontend/src/components/features/ExtractionJobCard.tsx`

Apply same star icon changes to the features table in ExtractionJobCard (lines 495-503 in "Of Interest" column).

**Estimated Time:** 2-3 hours
**Testing:** Manual testing with multiple features
**Acceptance Criteria:**
- ✅ Star icon clickable in both FeaturesPanel and ExtractionJobCard
- ✅ Visual feedback during toggle (loading spinner)
- ✅ Favorites persist across page reloads
- ✅ "Show favorites only" filter works
- ✅ "Favorites First" sort works
- ✅ Hover effects and tooltips work

---

## Enhancement 3: Three-Field Labeling System [ISS-003]

**Priority:** MEDIUM (Richer interpretations)
**Effort:** 4-6 hours
**Dependencies:** ISS-005 (cleaner tokens improve quality)
**Status:** Ready after ISS-005 complete

### Problem
Current dual-label system only populates `category` and `name`:
- `category`: "names"
- `name`: "elizabeth_variations"
- `description`: NULL ❌

Need to populate `description` with detailed interpretation.

### Solution: Triple-Field Labeling Prompt

**Implementation Steps:**

#### Step 1: Update OpenAI Labeling Prompt (1 hour)
**File:** `backend/src/services/openai_labeling_service.py`

**Current prompt** (lines 156-190):
```
Analyze this sparse autoencoder feature...
[Token statistics table]
Provide TWO labels:
1. CATEGORY: Broad type
2. SPECIFIC: Precise label
```

**New prompt:**
```
Analyze this sparse autoencoder feature for mechanistic interpretability.

[Token statistics table with cleaned tokens from ISS-005]

Provide THREE labels for this feature:

1. CATEGORY (1-2 words): High-level semantic grouping
   Examples: "names", "negation", "technical_terms", "temporal"

2. SPECIFIC LABEL (2-4 words): Precise pattern identification
   Examples: "elizabeth_name_variations", "negation_prefixes", "python_keywords"

3. DETAILED INTERPRETATION (2-3 sentences): Rich description based on activation patterns
   - Explain what triggers this feature (specific tokens, contexts, patterns)
   - Describe the semantic or syntactic function
   - Mention notable variations or edge cases
   - Reference actual tokens from the statistics table

Example:
CATEGORY: names
SPECIFIC: elizabeth_name_variations
INTERPRETATION: This feature activates strongly for different forms and nicknames of "Elizabeth" including Lizzie, Liz, Beth, Betty, Eliza, and Lizabeth. The feature shows consistent activation across personal references, biographical content, and direct address contexts. It appears to capture a name-identity pattern rather than just string matching, as it responds to various diminutive and formal variants.

Respond in this EXACT format:
category: [category]
specific: [specific_label]
interpretation: [detailed interpretation]
```

#### Step 2: Update Response Parser (1 hour)
**File:** `backend/src/services/openai_labeling_service.py`

**Current:** `_parse_dual_label()` (lines 225-251)
**New:** `_parse_triple_label()`

```python
def _parse_triple_label(self, response_text: str) -> dict:
    """Parse triple-label response into category, name, description."""
    lines = response_text.strip().split('\n')
    result = {
        'category': None,
        'name': None,
        'description': None
    }

    for line in lines:
        line = line.strip()
        if line.startswith('category:'):
            result['category'] = line.split(':', 1)[1].strip().lower().replace(' ', '_')
        elif line.startswith('specific:'):
            result['name'] = line.split(':', 1)[1].strip().lower().replace(' ', '_')
        elif line.startswith('interpretation:'):
            result['description'] = line.split(':', 1)[1].strip()

    # Validation
    if not result['category'] or not result['name']:
        raise ValueError(f"Missing required fields in response: {response_text}")

    # Description is optional but log warning if missing
    if not result['description']:
        logger.warning(f"Missing interpretation in response for {result['name']}")
        result['description'] = None  # Will be stored as NULL

    return result
```

#### Step 3: Update Database Save Logic (30 min)
**File:** `backend/src/services/labeling_service.py` (lines 417-424)

**Current:** Only saves `category` and `name`
**Updated:** Also save `description`

```python
feature.category = label_result['category']
feature.name = label_result['name']
feature.description = label_result.get('description')  # NEW
feature.label_source = 'openai'
feature.labeled_at = datetime.utcnow()
```

#### Step 4: Update Local Labeling Service (1 hour)
**File:** `backend/src/services/local_labeling_service.py`

Apply same changes:
- Update prompt template to request 3 fields
- Update parser to extract 3 fields
- Use same format as OpenAI service for consistency

#### Step 5: Update Frontend Display (1.5 hours)
**Files to update:**
- `frontend/src/components/features/FeatureDetailModal.tsx` - Show interpretation
- `frontend/src/components/features/FeaturesPanel.tsx` - Already shows description (verified earlier)
- `frontend/src/components/features/ExtractionJobCard.tsx` - Already shows description (verified earlier)

Add interpretation section to FeatureDetailModal:
```typescript
{feature.description && (
  <div className="mt-6">
    <h4 className="text-sm font-semibold text-slate-300 mb-2">
      Feature Interpretation
    </h4>
    <p className="text-sm text-slate-400 leading-relaxed">
      {feature.description}
    </p>
  </div>
)}
```

#### Step 6: Add Tests (1 hour)
**File:** `backend/tests/test_openai_labeling_service.py`

Test cases:
- `test_parse_triple_label()` - All 3 fields parsed correctly
- `test_triple_label_missing_interpretation()` - Handles missing interpretation
- `test_triple_label_validation()` - Validates required fields
- `test_labeling_saves_description()` - Description saved to database

**Estimated Time:** 4-6 hours
**Testing:** Label a small extraction (100 features) and verify descriptions
**Acceptance Criteria:**
- ✅ Prompt requests 3 fields with clear format
- ✅ Parser extracts category, name, description
- ✅ Description saved to database
- ✅ Frontend displays interpretation in detail modal
- ✅ Works with both OpenAI and local labeling
- ✅ Handles missing interpretation gracefully

---

## Enhancement 4: Ollama Local LLM Support [ISS-004]

**Priority:** LOW-MEDIUM (Cost savings, but optional)
**Effort:** 3-4 hours
**Dependencies:** ISS-003 (shares prompt infrastructure)
**Status:** Ready after ISS-003 complete

### Problem
- Currently hardcoded to OpenAI API (`https://api.openai.com/v1`)
- Local Ollama endpoint available at `ollama.mcslab.io`
- Ollama provides OpenAI-compatible API
- Would enable zero-cost labeling with local models

### Solution: Configurable API Base URL

**Implementation Steps:**

#### Step 1: Add Configuration (30 min)
**File:** `backend/src/core/config.py`

```python
# Labeling API configuration
labeling_api_provider: str = Field(
    default="openai",
    description="Labeling API provider: openai, ollama, local"
)
labeling_api_base_url: Optional[str] = Field(
    default=None,
    description="Base URL for labeling API (None = use provider default)"
)
labeling_api_model: str = Field(
    default="gpt-4o-mini",
    description="Model to use for labeling"
)
ollama_base_url: str = Field(
    default="http://ollama.mcslab.io:11434/v1",
    description="Ollama API base URL"
)
```

#### Step 2: Update OpenAI Service to Support Custom Base URL (1 hour)
**File:** `backend/src/services/openai_labeling_service.py`

**Changes:**
```python
from openai import OpenAI

class OpenAILabelingService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        ...
    ):
        # Determine base URL
        if base_url is None:
            base_url = "https://api.openai.com/v1"  # Default

        # Determine API key requirement
        requires_auth = not base_url.startswith("http://ollama")

        # Initialize client
        self.client = OpenAI(
            api_key=api_key if requires_auth else "not-needed",
            base_url=base_url
        )
        self.model = model
```

#### Step 3: Update Labeling Service Factory (30 min)
**File:** `backend/src/services/labeling_service.py`

Update initialization to pass base_url from config:
```python
def _create_labeling_service(self):
    provider = settings.labeling_api_provider

    if provider == "openai":
        base_url = settings.labeling_api_base_url or "https://api.openai.com/v1"
        return OpenAILabelingService(
            api_key=settings.openai_api_key,
            base_url=base_url,
            model=settings.labeling_api_model
        )
    elif provider == "ollama":
        return OpenAILabelingService(
            api_key=None,  # Not needed for Ollama
            base_url=settings.ollama_base_url,
            model=settings.labeling_api_model  # e.g., "llama3.1"
        )
    elif provider == "local":
        return LocalLabelingService(...)
```

#### Step 4: Add Frontend UI Controls (1.5 hours)
**File:** Frontend labeling configuration component (location TBD)

Add provider selector:
```typescript
<select value={provider} onChange={(e) => setProvider(e.target.value)}>
  <option value="openai">OpenAI API</option>
  <option value="ollama">Ollama (Local)</option>
  <option value="local">Local LLM (Phi-3)</option>
</select>

{provider === 'openai' && (
  <>
    <input
      type="password"
      placeholder="OpenAI API Key"
      value={apiKey}
      onChange={(e) => setApiKey(e.target.value)}
    />
    <select value={model} onChange={(e) => setModel(e.target.value)}>
      <option value="gpt-4o-mini">GPT-4o Mini ($0.0001/feature)</option>
      <option value="gpt-4">GPT-4 Turbo ($0.001/feature)</option>
    </select>
  </>
)}

{provider === 'ollama' && (
  <select value={model} onChange={(e) => setModel(e.target.value)}>
    <option value="llama3.1">Llama 3.1 8B (Free)</option>
    <option value="qwen2.5">Qwen 2.5 7B (Free)</option>
    <option value="mistral">Mistral 7B (Free)</option>
  </select>
)}
```

#### Step 5: Add Documentation (30 min)
**File:** `0xcc/docs/Ollama_Labeling_Setup.md` (NEW)

Document:
- How to set up Ollama locally or use mcslab.io endpoint
- Model recommendations (Llama 3.1 8B works well)
- Performance comparison (OpenAI vs Ollama)
- Cost savings ($0 vs ~$1.64 per 16K features)

#### Step 6: Test with Ollama (1 hour)
**Manual testing:**
1. Configure provider = "ollama"
2. Set model = "llama3.1"
3. Run small labeling job (100 features)
4. Verify:
   - Connection to Ollama successful
   - Labels generated correctly
   - Quality comparable to OpenAI

**Estimated Time:** 3-4 hours
**Testing:** Manual test with Ollama endpoint
**Acceptance Criteria:**
- ✅ Configuration supports multiple providers
- ✅ OpenAI service accepts custom base_url
- ✅ Ollama endpoint works without API key
- ✅ Frontend UI allows provider selection
- ✅ Documentation created
- ✅ Tested end-to-end with Ollama

---

## Implementation Timeline

### Week 1: Foundation (4-6 hours)

**Day 1-2:**
- [ ] ISS-005: Token Data Cleaning (2-3 hours)
  - Create TokenFilter utility
  - Integrate into labeling service
  - Add tests and documentation
- [ ] ISS-006: Feature Favorites Toggle (2-3 hours)
  - Add star icon to FeaturesPanel
  - Add favorites filter and sort
  - Update ExtractionJobCard

**Checkpoint:** Verify token filtering improves prompt quality, favorites work smoothly

### Week 2: Enhanced Labeling (8-12 hours)

**Day 3-4:**
- [ ] ISS-003: Three-Field Labeling System (4-6 hours)
  - Update OpenAI prompt and parser
  - Update Local LLM prompt and parser
  - Update database save logic
  - Update frontend display
  - Test with small extraction

**Day 5:**
- [ ] ISS-004: Ollama Support (3-4 hours)
  - Add configuration
  - Update OpenAI service for custom base_url
  - Add frontend provider selector
  - Test with Ollama endpoint

**Checkpoint:** Verify rich interpretations generated, Ollama works as alternative

---

## Testing Strategy

### Unit Tests
- `test_token_filter.py` - Token filtering logic
- `test_openai_labeling_service.py` - Triple-label parsing
- `test_labeling_service.py` - Integration tests

### Integration Tests
1. **Token Filtering**
   - Run extraction with filtering enabled
   - Check logs for filtering statistics
   - Verify LLM prompts cleaner

2. **Favorites**
   - Toggle multiple features as favorites
   - Filter by favorites
   - Sort by favorites
   - Verify persistence across reloads

3. **Triple-Field Labeling**
   - Label 100 features with new system
   - Verify all 3 fields populated
   - Check interpretation quality (manual review)

4. **Ollama**
   - Label 100 features via Ollama
   - Compare quality to OpenAI
   - Verify cost = $0

### Acceptance Testing
- [ ] Token filtering removes 20-40% of junk tokens
- [ ] Favorites toggle works smoothly (<200ms response)
- [ ] Descriptions are meaningful and specific
- [ ] Ollama labels are 80%+ as good as OpenAI

---

## Success Metrics

### Quantitative
- **Token Filtering:** 20-40% reduction in junk tokens
- **Favorites:** <200ms toggle response time
- **Triple-Field:** 95%+ features have descriptions
- **Ollama:** Labeling cost reduced from ~$1.64 to $0 per 16K features

### Qualitative
- Token statistics easier to interpret
- Favorite features workflow feels natural
- Descriptions add significant interpretability value
- Ollama quality acceptable for research use

---

## Rollback Plan

If any enhancement causes issues:

1. **Token Filtering:** Set `token_filter_enabled=False` in config
2. **Favorites:** UI-only change, no data corruption risk
3. **Triple-Field:** Falls back gracefully (description=NULL is fine)
4. **Ollama:** Falls back to OpenAI by setting provider="openai"

No database migrations needed for any enhancement (all fields already exist).

---

## Post-Implementation Tasks

After all enhancements complete:

1. **Documentation:**
   - Update Feature Discovery docs with new capabilities
   - Create user guide for favorites workflow
   - Document triple-field labeling format

2. **Performance Monitoring:**
   - Monitor token filtering impact on labeling speed
   - Track favorites usage patterns
   - Compare OpenAI vs Ollama label quality over time

3. **User Feedback:**
   - Gather feedback on description usefulness
   - Assess whether Ollama quality is acceptable
   - Identify additional filter patterns needed

---

## Open Questions

1. **Token Filtering:**
   - Should we expose filter configuration in UI?
   - Need whitelist for domain-specific tokens (e.g., chemical symbols)?

2. **Triple-Field Labeling:**
   - Should description be required or optional?
   - Max length limit for descriptions? (currently TEXT = unlimited)

3. **Ollama:**
   - Which Ollama models to recommend?
   - Should we auto-detect Ollama availability?

---

**Created:** 2025-11-10
**Last Updated:** 2025-11-10
**Next Review:** After Phase 1 complete (ISS-005, ISS-006)
