# Enhancement: Token Filtering Per-Job Configuration

**Created:** 2025-11-11
**Status:** In Progress
**Priority:** High
**Related:** Three-Stage Token Filtering Architecture

## Overview
Enhance token filtering system to provide more granular control at the dataset tokenization level, including complete punctuation removal and custom character filtering.

## Requirements

### 1. New Filter Mode: STRICT
Add a fifth filter mode that completely removes all punctuation:
- **MINIMAL**: Only control chars and nulls
- **CONSERVATIVE**: + whitespace-only tokens
- **STANDARD**: + pure punctuation + single non-alnum chars
- **AGGRESSIVE**: + short tokens (1-2 chars with <50% alnum)
- **STRICT** (NEW): + ALL punctuation tokens (even within words)

### 2. Custom Character Filtering
Allow users to specify additional characters to filter:
- Input: Text field for custom characters (e.g., "~@#$%")
- Backend: Accept `custom_filter_chars` parameter
- Implementation: Add characters to junk token detection

### 3. Complete Punctuation Removal Option
Add checkbox: "Remove ALL Punctuation"
- When enabled: Uses STRICT mode + removes all `string.punctuation` characters
- Override for existing modes
- More aggressive than AGGRESSIVE mode

## Implementation Plan

### Backend Changes

#### 1. Update `TokenFilter` class ([backend/src/utils/token_filter.py](backend/src/utils/token_filter.py))
```python
- Add STRICT mode to FilterMode enum
- Add remove_all_punctuation parameter to __init__
- Add custom_filter_chars parameter to __init__
- Update _is_junk_token() to handle STRICT mode
- Add _contains_any_punctuation() method
```

#### 2. Update Dataset Tokenization Schema ([backend/src/schemas/dataset.py](backend/src/schemas/dataset.py))
```python
class TokenizationConfigRequest(BaseModel):
    filter_enabled: bool = False
    filter_mode: FilterMode = "conservative"  # Now includes "strict"
    remove_all_punctuation: bool = False  # NEW
    custom_filter_chars: Optional[str] = None  # NEW (e.g., "~@#$%")
    junk_ratio_threshold: float = 0.7
```

#### 3. Update Database Schema
Add columns to `dataset_tokenizations` table:
- `remove_all_punctuation` (BOOLEAN, default FALSE)
- `custom_filter_chars` (VARCHAR, nullable)

Migration: `alembic revision --autogenerate -m "add token filter customization fields"`

#### 4. Update Tokenization Service ([backend/src/services/tokenization_service.py](backend/src/services/tokenization_service.py))
- Pass new parameters to TokenFilter constructor
- Log custom filtering configuration

### Frontend Changes

#### 1. Dataset Tokenization UI ([frontend/src/components/panels/DatasetsPanel.tsx](frontend/src/components/panels/DatasetsPanel.tsx))
Add new controls in tokenization configuration modal:
```typescript
const [filterMode, setFilterMode] = useState<'minimal' | 'conservative' | 'standard' | 'aggressive' | 'strict'>('conservative');
const [removeAllPunctuation, setRemoveAllPunctuation] = useState(false);
const [customFilterChars, setCustomFilterChars] = useState('');
```

UI Layout:
```
┌─ Token Filtering ────────────────────┐
│ [x] Enable Token Filtering           │
│                                       │
│ Filter Mode: [Dropdown ▼]            │
│   • Minimal                           │
│   • Conservative (recommended)        │
│   • Standard                          │
│   • Aggressive                        │
│   • Strict (NEW)                      │
│                                       │
│ [x] Remove ALL Punctuation (NEW)     │
│   ⓘ Removes every punctuation        │
│     character, even within words     │
│                                       │
│ Custom Characters to Filter: (NEW)   │
│ [________________]                    │
│   ⓘ Additional characters to         │
│     remove (e.g., ~@#$%)              │
└───────────────────────────────────────┘
```

#### 2. Update TypeScript Types ([frontend/src/types/dataset.ts](frontend/src/types/dataset.ts))
```typescript
export type TokenFilterMode = 'minimal' | 'conservative' | 'standard' | 'aggressive' | 'strict';

export interface TokenizationConfig {
  filter_enabled: boolean;
  filter_mode: TokenFilterMode;
  remove_all_punctuation?: boolean;  // NEW
  custom_filter_chars?: string;       // NEW
  junk_ratio_threshold?: number;
}
```

## Testing Plan

### Unit Tests
- [ ] Test STRICT mode filters all punctuation
- [ ] Test custom_filter_chars removes specified characters
- [ ] Test remove_all_punctuation overrides mode settings
- [ ] Test combination of STRICT + custom chars

### Integration Tests
- [ ] Test tokenization with STRICT mode
- [ ] Test tokenization with custom filter chars
- [ ] Test tokenization with remove_all_punctuation
- [ ] Verify filtered stats are correct

### UI Tests
- [ ] Test filter mode dropdown includes "strict"
- [ ] Test punctuation removal checkbox
- [ ] Test custom characters input field
- [ ] Test configuration persists across page refreshes

## Migration Strategy

### For Existing Tokenizations
- Default `remove_all_punctuation = FALSE`
- Default `custom_filter_chars = NULL`
- Existing behavior unchanged

### For New Tokenizations
- Users can optionally enable new features
- Conservative mode remains default
- Clear UI guidance on when to use STRICT mode

## Documentation Updates
- [ ] Update Dataset Management docs with new filter options
- [ ] Add examples of when to use each mode
- [ ] Document custom character filtering format
- [ ] Update API documentation

## Success Criteria
- [x] STRICT mode removes 100% of punctuation tokens
- [x] Custom characters are correctly filtered from tokens
- [x] UI clearly explains each option
- [x] Backward compatible with existing tokenizations
- [x] Performance impact < 10% vs current implementation

## Notes
- STRICT mode may significantly reduce dataset size
- Recommend testing with small sample first
- Custom chars should be validated (no alphanumeric allowed)
- Consider adding preset character sets (emojis, math symbols, etc.)
