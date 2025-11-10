# Integration Plan: Filtering Stages 1 & 2

**Created:** 2025-11-10
**Status:** Ready for Implementation
**Estimated Time:** 3-4 hours
**Priority:** Medium (Stage 3 already working, these are enhancements)

---

## Overview

This document outlines the integration work needed to complete the three-stage filtering architecture. Stage 3 (Pre-Labeling Feature Filter) is already complete and active. Stages 1 and 2 require integration into existing workers.

---

## Current Status

### ✅ Complete:
- Configuration settings for all three stages
- `TokenFilter.is_junk_sequence()` method for sample-level analysis
- `TokenFilter.is_junk_token()` method for token-level analysis
- `FeatureFilter.is_junk_feature()` method for feature-level analysis (already integrated)
- Comprehensive documentation

### ⏳ Pending:
- **Stage 1:** Integration into dataset tokenization worker
- **Stage 2:** Integration into feature extraction worker

---

## Stage 1: Dataset Tokenization Filter Integration

### Target Files:
1. **Primary:** `backend/src/services/tokenization_service.py`
   - Class: `_TokenizationMapper`
   - Method: `__call__(self, examples)`
   - Line: ~109 (after tokenization, before return)

2. **Secondary:** `backend/src/workers/dataset_tasks.py`
   - Function: `tokenize_dataset_task()` or similar
   - Purpose: Pass filter settings to TokenizationMapper

### Implementation Steps:

#### Step 1.1: Add Filter Parameters to _TokenizationMapper
**Location:** `tokenization_service.py`, `_TokenizationMapper.__init__()`

```python
def __init__(
    self,
    tokenizer_name: str,
    text_column: str,
    max_length: int,
    truncation: bool,
    padding: str,
    add_special_tokens: bool,
    return_attention_mask: bool,
    stride: int = 0,
    return_overflowing_tokens: bool = False,
    enable_cleaning: bool = True,
    enable_filtering: bool = False,  # NEW
    filter_mode: str = "conservative",  # NEW
    junk_ratio_threshold: float = 0.7,  # NEW
):
    # ... existing code ...
    self.enable_filtering = enable_filtering
    self.filter_mode = filter_mode
    self.junk_ratio_threshold = junk_ratio_threshold
    self._token_filter = None  # Lazy-loaded
```

#### Step 1.2: Add Lazy Filter Loader
**Location:** `tokenization_service.py`, after `_get_text_cleaner()`

```python
def _get_token_filter(self):
    """Lazy-load token filter in worker process."""
    if self._token_filter is None:
        from ..utils.token_filter import TokenFilter, FilterMode
        mode = FilterMode[self.filter_mode.upper()]
        self._token_filter = TokenFilter(mode=mode)
    return self._token_filter
```

#### Step 1.3: Add Filtering Logic to __call__()
**Location:** `tokenization_service.py`, `_TokenizationMapper.__call__()`, after line 112

```python
def __call__(self, examples):
    # ... existing tokenization code ...

    result = tokenizer(
        texts_to_tokenize,
        **kwargs
    )

    # NEW: Filter junk samples if enabled
    if self.enable_filtering:
        token_filter = self._get_token_filter()
        tokenizer_obj = self._get_tokenizer()

        # Filter out samples with too many junk tokens
        filtered_indices = []
        for idx, input_ids in enumerate(result['input_ids']):
            if not token_filter.is_junk_sequence(
                input_ids,
                tokenizer_obj,
                self.junk_ratio_threshold
            ):
                filtered_indices.append(idx)

        # Keep only non-junk samples
        if filtered_indices:
            result = {
                key: [value[i] for i in filtered_indices]
                for key, value in result.items()
            }
        else:
            # All samples filtered - return empty result
            result = {key: [] for key in result.keys()}

    return result
```

#### Step 1.4: Pass Settings from Config
**Location:** `dataset_tasks.py` or wherever `_TokenizationMapper` is instantiated

```python
from src.core.config import settings

mapper = _TokenizationMapper(
    tokenizer_name=tokenizer_name,
    text_column=text_column,
    # ... other params ...
    enable_filtering=settings.tokenization_filter_enabled,
    filter_mode=settings.tokenization_filter_mode,
    junk_ratio_threshold=settings.tokenization_junk_ratio_threshold,
)
```

### Testing:
1. Enable filter: `TOKENIZATION_FILTER_ENABLED=true` in `.env`
2. Tokenize small test dataset
3. Check logs for filtering statistics
4. Verify filtered sample count makes sense

---

## Stage 2: Extraction Filter Integration

### Target Files:
1. **Primary:** `backend/src/workers/extraction_tasks.py`
   - Function: Feature extraction function
   - Purpose: Filter tokens before SAE forward pass

### Implementation Steps:

#### Step 2.1: Find Extraction Code
Search for where tokens are fed into the SAE model:
```bash
grep -n "sae\(" backend/src/workers/extraction_tasks.py
grep -n "forward\(" backend/src/workers/extraction_tasks.py
grep -n "input_ids" backend/src/workers/extraction_tasks.py
```

#### Step 2.2: Add Filter Before SAE Forward Pass
**Pseudo-location:** Before SAE forward pass in extraction worker

```python
from src.utils.token_filter import TokenFilter, FilterMode
from src.core.config import settings

# Initialize filter if enabled
if settings.extraction_filter_enabled:
    mode = FilterMode[settings.extraction_filter_mode.upper()]
    token_filter = TokenFilter(mode=mode)

    # For each batch of tokens:
    filtered_input_ids = []
    filtered_attention_mask = []

    for batch_idx, input_ids in enumerate(batch['input_ids']):
        # Filter tokens
        keep_indices = [
            idx for idx, token_id in enumerate(input_ids)
            if not token_filter.is_junk_token(
                tokenizer.convert_ids_to_tokens(token_id)
            )
        ]

        # Keep only non-junk tokens
        filtered_input_ids.append([input_ids[i] for i in keep_indices])
        if 'attention_mask' in batch:
            filtered_attention_mask.append([batch['attention_mask'][batch_idx][i] for i in keep_indices])

    # Update batch
    batch['input_ids'] = filtered_input_ids
    if 'attention_mask' in batch:
        batch['attention_mask'] = filtered_attention_mask

# Now feed filtered batch into SAE
activations = sae(batch['input_ids'])
```

### Testing:
1. Enable filter: `EXTRACTION_FILTER_ENABLED=true` in `.env`
2. Run feature extraction on test model
3. Check that junk tokens are excluded
4. Verify feature quality improves

---

## Integration Checklist

### Pre-Integration:
- [ ] Read through `tokenization_service.py` to understand current flow
- [ ] Read through `extraction_tasks.py` to find SAE forward pass
- [ ] Create test dataset for validation
- [ ] Document current tokenization/extraction behavior

### Stage 1 Integration:
- [ ] Add filter parameters to `_TokenizationMapper.__init__()`
- [ ] Add `_get_token_filter()` method
- [ ] Add filtering logic to `__call__()`
- [ ] Pass settings from config
- [ ] Add logging for filtered sample count
- [ ] Test with small dataset
- [ ] Verify no performance regression

### Stage 2 Integration:
- [ ] Locate SAE forward pass in extraction worker
- [ ] Add filter initialization
- [ ] Add token filtering loop
- [ ] Update batch with filtered tokens
- [ ] Add logging for filtered token count
- [ ] Test with small model
- [ ] Verify feature quality

### Post-Integration:
- [ ] Run full integration test (dataset → training → extraction → labeling)
- [ ] Monitor all three filtering stages in logs
- [ ] Document filtering statistics
- [ ] Update Token_And_Feature_Filtering.md with "COMPLETE" status
- [ ] Commit with message: `feat: integrate three-stage token filtering system`

---

## Expected Outcomes

### Stage 1 (Tokenization):
- Log output: `INFO: Tokenization filter: 45000 samples kept, 5000 junk samples filtered (10.0% filtered)`
- Smaller dataset files
- Cleaner training data

### Stage 2 (Extraction):
- Log output: `INFO: Extraction filter: 2.5M tokens kept, 500K junk tokens filtered (16.7% filtered)`
- Cleaner feature activations
- Better semantic feature quality

### Stage 3 (Already Working):
- Log output: `INFO: Pre-labeling filter: 10245 features to label, 6139 junk features skipped (37.5% filtered)`
- Lower API costs
- Cleaner feature labels

---

## Rollback Plan

If integration causes issues:

1. **Disable filters immediately:**
   ```bash
   TOKENIZATION_FILTER_ENABLED=false
   EXTRACTION_FILTER_ENABLED=false
   # Keep Stage 3 enabled - it's working fine
   ```

2. **Revert code changes:**
   ```bash
   git revert <commit-hash>
   ```

3. **Debug:**
   - Check logs for errors
   - Test filter logic independently
   - Verify token ID → token string conversion works correctly

---

## Notes

- **Complexity:** Stage 1 is simpler (sample-level), Stage 2 is more complex (token-level with sequence integrity)
- **Performance:** Filtering adds minimal overhead (~1-2% processing time)
- **Safety:** All filters disabled by default, opt-in only
- **Reversibility:** Stage 1 is permanent (affects dataset), Stage 2 affects training, Stage 3 is fully reversible

---

**Document Version:** 1.0
**Next Session:** Implement Stage 1 first, then Stage 2, then test both together
