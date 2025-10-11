# Task List: Dataset Management Enhancement 01

**Feature ID:** 001_FTASKS|Dataset_Management_ENH_01
**Feature Name:** Dataset Management - Missing Features from Mock UI Reference
**PRD Reference:** 001_FPRD|Dataset_Management_ENH_01.md
**TDD Reference:** 001_FTDD|Dataset_Management_ENH_01.md
**TID Reference:** 001_FTID|Dataset_Management_ENH_01.md
**ADR Reference:** 000_PADR|miStudio.md
**Mock UI Reference:** Mock-embedded-interp-ui.tsx (lines 4086-4393)
**Status:** Ready for Implementation
**Created:** 2025-10-11

**Extends:** 001_FTASKS|Dataset_Management.md

---

## Relevant Files

### Backend Files (Extensions)
- `backend/src/schemas/dataset.py` - **EXTENDED**: Add `padding`, `truncation`, `add_special_tokens`, `return_attention_mask` fields to `DatasetTokenizeRequest`
- `backend/src/schemas/metadata.py` - **NEW**: Pydantic validation schemas for `TokenizationMetadata` with histogram and unique tokens fields
- `backend/src/services/tokenization_service.py` - **EXTENDED**: Add `calculate_unique_tokens()`, `calculate_histogram()`, `calculate_split_distribution()` methods
- `backend/src/workers/dataset_tasks.py` - **EXTENDED**: Update `tokenize_dataset_task` to accept new parameters and store in metadata
- `backend/src/api/v1/endpoints/datasets.py` - **NEW**: Add `POST /tokenize-preview` endpoint with `TokenizePreviewRequest` schema

### Backend Tests (New)
- `backend/tests/unit/test_tokenization_enhancements.py` - **NEW**: Unit tests for padding validation, truncation validation, unique tokens calculation, histogram calculation, split distribution
- `backend/tests/integration/test_tokenize_preview.py` - **NEW**: Integration tests for tokenize-preview endpoint (success, text too long, invalid tokenizer, special tokens, attention mask)

### Frontend Components (New)
- `frontend/src/components/common/ToggleSwitch.tsx` - **NEW**: Reusable toggle component with label, help text, ARIA attributes
- `frontend/src/components/datasets/TokenizationPreview.tsx` - **NEW**: Preview component with text area, token chips, loading/error states
- `frontend/src/components/datasets/SequenceLengthHistogram.tsx` - **NEW**: Histogram visualization with 7 buckets, gradient bars, summary statistics
- `frontend/src/components/datasets/SplitDistribution.tsx` - **NEW**: Three-card layout for train/val/test splits with color coding

### Frontend Files (Extensions)
- `frontend/src/components/datasets/DatasetDetailModal.tsx` - **EXTENDED**: Add padding/truncation dropdowns, special tokens/attention mask toggles, TokenizationPreview, SequenceLengthHistogram, SplitDistribution to TokenizationTab and StatisticsTab
- `frontend/src/types/dataset.ts` - **EXTENDED**: Add `padding`, `truncation`, `add_special_tokens`, `return_attention_mask`, `unique_tokens`, `median_seq_length`, `histogram`, `splits` to `TokenizationMetadata` and `DatasetMetadata` interfaces
- `frontend/src/api/datasets.ts` - **EXTENDED**: Add `tokenizePreview()` API client function

### Frontend Tests (New)
- `frontend/src/components/datasets/TokenizationPreview.test.tsx` - **NEW**: Component tests for preview form rendering, token display, special token styling, disabled states
- `frontend/src/components/common/ToggleSwitch.test.tsx` - **NEW**: Component tests for toggle rendering, state changes, disabled states, accessibility
- `frontend/src/components/datasets/SequenceLengthHistogram.test.tsx` - **NEW**: Component tests for histogram rendering, bar widths, summary statistics
- `frontend/src/components/datasets/SplitDistribution.test.tsx` - **NEW**: Component tests for split cards rendering, color coding, missing splits handling

---

## Tasks

### Phase 14: P1 Features (High Priority - 26-38 hours)

**Goal:** Implement critical tokenization features that block production use.

---

#### Task 14.1: Implement Padding Strategy Selection (4-6 hours)

**Purpose:** Allow users to select padding strategy (max_length, longest, do_not_pad).

**Acceptance Criteria:**
- ✅ Backend accepts `padding` parameter in tokenization request
- ✅ Tokenizer uses specified padding strategy
- ✅ Padding strategy stored in `dataset.metadata.tokenization.padding`
- ✅ Frontend dropdown has 3 options with help text
- ✅ All padding strategies produce correct outputs

**Sub-Tasks:**

- [ ] 14.1.1 Update `DatasetTokenizeRequest` schema with `padding` field (30 minutes)
  - **File:** `backend/src/schemas/dataset.py` (line ~45)
  - **Changes:**
    - Add `padding: Literal["max_length", "longest", "do_not_pad"] = "max_length"` field with description
    - Add `@field_validator("padding")` to validate allowed values
    - Add import: `from typing import Literal` and `from pydantic import field_validator`
  - **Testing:** Create `DatasetTokenizeRequest` with each padding value, verify validation
  - **Reference:** TID Section 2.1

- [ ] 14.1.2 Update `tokenize_dataset_task` to use dynamic padding (45 minutes)
  - **File:** `backend/src/workers/dataset_tasks.py` (line ~200-250)
  - **Changes:**
    - Add `padding: str = "max_length"` parameter to task definition
    - Create `padding_config` mapping dict (max_length→"max_length", longest→"longest", do_not_pad→False)
    - Update `tokenize_function` to use `padding=padding_config.get(padding, "max_length")`
    - Store padding in metadata: `metadata_update["tokenization"]["padding"] = padding`
  - **Testing:** Run task with each padding strategy, verify tokenized output lengths
  - **Reference:** TID Section 2.1

- [ ] 14.1.3 Add padding dropdown to TokenizationTab (1 hour)
  - **File:** `frontend/src/components/datasets/DatasetDetailModal.tsx` (line ~487-590)
  - **Changes:**
    - Add state: `const [paddingStrategy, setPaddingStrategy] = useState<'max_length' | 'longest' | 'do_not_pad'>('max_length')`
    - Insert dropdown after stride slider with 3 options
    - Add help text: "Controls how sequences are padded. 'Max Length' pads all to max_length for consistent memory usage."
    - Update `handleTokenize` to include `padding: paddingStrategy` in request body
  - **Styling:** `w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:border-emerald-500`
  - **Testing:** Select each option, verify state updates and API request includes correct padding
  - **Reference:** TID Section 3.1

- [ ] 14.1.4 Write unit tests for padding validation (45 minutes)
  - **File:** `backend/tests/unit/test_tokenization_enhancements.py` (NEW)
  - **Tests:**
    - `test_valid_padding_strategies()` - Test all 3 valid strategies
    - `test_default_padding_strategy()` - Test default is max_length
    - `test_invalid_padding_strategy()` - Test invalid value raises ValidationError
  - **Assertions:** Verify schema validation, error messages
  - **Reference:** TID Section 2.1

- [ ] 14.1.5 Write integration test for padding behavior (1 hour)
  - **File:** `backend/tests/integration/test_dataset_api.py`
  - **Test:** `test_tokenize_with_different_padding_strategies()`
  - **Steps:**
    1. Create test dataset
    2. Tokenize with `padding="max_length"`, verify all sequences have same length
    3. Tokenize with `padding="do_not_pad"`, verify sequences have variable lengths
  - **Assertions:** Verify output lengths match expected behavior
  - **Reference:** TID Section 2.1

**Estimated Time:** 4-6 hours
**Priority:** P1 (Critical)
**Dependencies:** None

---

#### Task 14.2: Implement Truncation Strategy Selection (4-6 hours)

**Purpose:** Allow users to select truncation strategy (longest_first, only_first, only_second, do_not_truncate).

**Acceptance Criteria:**
- ✅ Backend accepts `truncation` parameter in tokenization request
- ✅ Tokenizer uses specified truncation strategy
- ✅ Truncation strategy stored in `dataset.metadata.tokenization.truncation`
- ✅ Frontend dropdown has 4 options with help text
- ✅ All truncation strategies produce correct outputs

**Sub-Tasks:**

- [ ] 14.2.1 Update `DatasetTokenizeRequest` schema with `truncation` field (30 minutes)
  - **File:** `backend/src/schemas/dataset.py` (after padding field)
  - **Changes:**
    - Add `truncation: Literal["longest_first", "only_first", "only_second", "do_not_truncate"] = "longest_first"` field
    - Add `@field_validator("truncation")` to validate allowed values
  - **Testing:** Create request with each truncation value
  - **Reference:** TID Section 2.2

- [ ] 14.2.2 Update `tokenize_dataset_task` to use dynamic truncation (45 minutes)
  - **File:** `backend/src/workers/dataset_tasks.py` (line ~200-250)
  - **Changes:**
    - Add `truncation: str = "longest_first"` parameter to task definition
    - Create `truncation_config` mapping dict (longest_first→True, only_first→"only_first", only_second→"only_second", do_not_truncate→False)
    - Update `tokenize_function` to use `truncation=truncation_config.get(truncation, True)`
    - Store truncation in metadata: `metadata_update["tokenization"]["truncation"] = truncation`
  - **Testing:** Run task with multi-sequence inputs, verify truncation behavior
  - **Reference:** TID Section 2.2

- [ ] 14.2.3 Add truncation dropdown to TokenizationTab (1 hour)
  - **File:** `frontend/src/components/datasets/DatasetDetailModal.tsx` (after padding dropdown)
  - **Changes:**
    - Add state: `const [truncationStrategy, setTruncationStrategy] = useState<'longest_first' | 'only_first' | 'only_second' | 'do_not_truncate'>('longest_first')`
    - Insert dropdown with 4 options
    - Add help text: "Controls truncation for sequences exceeding max_length. Useful for Q&A pairs or multi-sequence inputs."
    - Update `handleTokenize` to include `truncation: truncationStrategy`
  - **Testing:** Select each option, verify API request
  - **Reference:** TID Section 3.1

- [ ] 14.2.4 Write unit tests for truncation validation (45 minutes)
  - **File:** `backend/tests/unit/test_tokenization_enhancements.py`
  - **Tests:**
    - `test_valid_truncation_strategies()` - Test all 4 valid strategies
    - `test_default_truncation_strategy()` - Test default is longest_first
    - `test_invalid_truncation_strategy()` - Test invalid value raises error
  - **Reference:** TID Section 2.2

- [ ] 14.2.5 Write integration test for truncation behavior (1 hour)
  - **File:** `backend/tests/integration/test_dataset_api.py`
  - **Test:** `test_tokenize_with_different_truncation_strategies()`
  - **Steps:**
    1. Create test dataset with long sequences
    2. Tokenize with `truncation="longest_first"`, verify sequences truncated
    3. Tokenize with `truncation="do_not_truncate"`, verify error on overflow
  - **Reference:** TID Section 2.2

**Estimated Time:** 4-6 hours
**Priority:** P1 (Critical)
**Dependencies:** None (can be done in parallel with Task 14.1)

---

#### Task 14.3: Implement Tokenization Preview (8-12 hours)

**Purpose:** Allow users to preview tokenization on sample text before processing full dataset.

**Acceptance Criteria:**
- ✅ Backend endpoint `POST /api/datasets/tokenize-preview` responds in <1 second
- ✅ Tokenizers cached in memory (LRU cache, max 10)
- ✅ Special tokens highlighted with emerald color
- ✅ Regular tokens highlighted with slate color
- ✅ Token count and special token count displayed
- ✅ Error handling for invalid tokenizers and text too long

**Sub-Tasks:**

- [ ] 14.3.1 Create `POST /tokenize-preview` endpoint (3 hours)
  - **File:** `backend/src/api/v1/endpoints/datasets.py` (after existing tokenize endpoint)
  - **Changes:**
    - Add `@lru_cache(maxsize=10)` decorated `load_tokenizer_cached()` function
    - Create `TokenizePreviewRequest` schema (tokenizer_name, text max 1000 chars, max_length, padding, truncation, add_special_tokens, return_attention_mask)
    - Create `TokenInfo` schema (id, text, type, position)
    - Create `TokenizePreviewResponse` schema (tokens, attention_mask, token_count, sequence_length, special_token_count)
    - Implement endpoint with tokenization and token type detection
    - Add error handling for invalid tokenizers (400) and tokenization errors (500)
  - **Testing:** POST with sample text, verify tokens returned
  - **Reference:** TID Section 2.3

- [ ] 14.3.2 Add tokenizer caching for performance (30 minutes)
  - **File:** `backend/src/api/v1/endpoints/datasets.py`
  - **Implementation:** Already included in 14.3.1 with `@lru_cache(maxsize=10)`
  - **Testing:** Call preview endpoint multiple times, verify cache hits (check logs)
  - **Performance Target:** <1s p95 response time
  - **Reference:** TID Section 6.1

- [ ] 14.3.3 Create `TokenizationPreview` component (3 hours)
  - **File:** `frontend/src/components/datasets/TokenizationPreview.tsx` (NEW)
  - **Structure:**
    - State: `previewText`, `tokens`, `tokenCount`, `specialTokenCount`, `loading`, `error`
    - Text area with 1000 char limit and character counter
    - "Tokenize Preview" button with loading state
    - Token chips container with flex-wrap
    - Token chip styling: emerald for special (bg-emerald-700), slate for regular (bg-slate-700)
    - Error message display (bg-red-900/20 border-red-700)
  - **Testing:** Render component, enter text, click button, verify tokens display
  - **Reference:** TID Section 3.3

- [ ] 14.3.4 Integrate `TokenizationPreview` into TokenizationTab (1 hour)
  - **File:** `frontend/src/components/datasets/DatasetDetailModal.tsx` (before tokenize button)
  - **Changes:**
    - Import `TokenizationPreview` component
    - Insert component with all current settings as props (tokenizerName, maxLength, padding, truncation, addSpecialTokens, returnAttentionMask)
  - **Testing:** Open modal, verify preview section renders, test preview functionality
  - **Reference:** TID Section 3.3

- [ ] 14.3.5 Write integration tests for tokenize-preview endpoint (2 hours)
  - **File:** `backend/tests/integration/test_tokenize_preview.py` (NEW)
  - **Tests:**
    - `test_tokenize_preview_success()` - Test with valid input
    - `test_tokenize_preview_text_too_long()` - Test with 1001 chars (should fail)
    - `test_tokenize_preview_invalid_tokenizer()` - Test with invalid tokenizer name
    - `test_tokenize_preview_no_special_tokens()` - Test with add_special_tokens=False
    - `test_tokenize_preview_no_attention_mask()` - Test with return_attention_mask=False
  - **Assertions:** Status codes, response structure, token counts
  - **Reference:** TID Section 2.3

- [ ] 14.3.6 Write component tests for `TokenizationPreview` (1.5 hours)
  - **File:** `frontend/src/components/datasets/TokenizationPreview.test.tsx` (NEW)
  - **Tests:**
    - `test_renders_preview_form()` - Test form elements present
    - `test_displays_tokens_after_success()` - Test tokens render after API call
    - `test_disables_button_when_text_empty()` - Test disabled state
    - `test_shows_special_tokens_with_different_styling()` - Test emerald vs slate colors
  - **Mocking:** Use MSW to mock `/api/datasets/tokenize-preview` endpoint
  - **Reference:** TID Section 5.2

**Estimated Time:** 8-12 hours
**Priority:** P1 (Critical)
**Dependencies:** Task 14.1, Task 14.2 (needs padding/truncation dropdowns for full functionality)

---

#### Task 14.4: Implement Sequence Length Histogram (10-14 hours)

**Purpose:** Replace simple 3-bar chart with full 7-bucket histogram showing sequence length distribution.

**Acceptance Criteria:**
- ✅ Backend calculates 7 histogram buckets (0-100, 100-200, 200-400, 400-600, 600-800, 800-1000, 1000+)
- ✅ Percentages sum to ~100% (99.9-100.1 allowed for floating point)
- ✅ Median sequence length calculated and displayed
- ✅ Frontend histogram uses emerald gradient bars
- ✅ Summary shows Min/Median/Max below chart

**Sub-Tasks:**

- [ ] 14.4.1 Add `calculate_histogram()` method to `TokenizationService` (3 hours)
  - **File:** `backend/src/services/tokenization_service.py` (after calculate_unique_tokens)
  - **Implementation:**
    - Static method signature: `calculate_histogram(seq_lengths: np.ndarray, max_length: int) -> List[Dict[str, Any]]`
    - Define bins: `[0, 100, 200, 400, 600, 800, 1000, max_length]`
    - Loop through bins, count samples in each range using `np.sum((seq_lengths >= min) & (seq_lengths < max))`
    - For last bin (1000+), count all samples >= 1000
    - Calculate percentage: `count / len(seq_lengths) * 100`
    - Return list of dicts with keys: range, min, max, count, percentage (rounded to 1 decimal)
  - **Error Handling:** Raise ValueError if seq_lengths is empty
  - **Testing:** Test with known lengths, verify bucket counts and percentages
  - **Reference:** TID Section 2.5

- [ ] 14.4.2 Update metadata schema with `histogram` field (30 minutes)
  - **File:** `backend/src/schemas/metadata.py`
  - **Changes:**
    - Create `HistogramBucket` Pydantic model (range: str, min: int, max: int, count: int, percentage: float with Field(ge=0, le=100))
    - Add `histogram: Optional[List[HistogramBucket]] = None` to `TokenizationMetadata`
    - Add `median_seq_length: Optional[float] = None` to `TokenizationMetadata`
  - **Testing:** Create metadata with histogram, verify validation
  - **Reference:** TID Section 2.5

- [ ] 14.4.3 Integrate histogram calculation into `calculate_statistics()` (1 hour)
  - **File:** `backend/src/services/tokenization_service.py` (update existing method)
  - **Changes:**
    - Add `seq_lengths_array = np.array(seq_lengths)` after sequence length collection
    - Add `stats["median_seq_length"] = float(np.median(seq_lengths_array))`
    - Add try/except for histogram calculation: `stats["histogram"] = TokenizationService.calculate_histogram(seq_lengths_array, tokenization_settings.get("max_length", 512))`
  - **Error Handling:** Catch exceptions, print warning, set histogram to None
  - **Testing:** Run tokenization, verify histogram in metadata
  - **Reference:** TID Section 2.5

- [ ] 14.4.4 Create `SequenceLengthHistogram` component (3 hours)
  - **File:** `frontend/src/components/datasets/SequenceLengthHistogram.tsx` (NEW)
  - **Structure:**
    - Props: `histogram: HistogramBucket[]`, `minLength`, `medianLength`, `maxLength`
    - Map over histogram buckets
    - Each bar: flex container with range label (w-28), bar container (flex-1 h-8 bg-slate-700), optional count/percentage text
    - Bar fill: `bg-gradient-to-r from-emerald-500 to-emerald-400` with width `${bucket.percentage}%`
    - Show count/percentage inside bar if percentage > 5%, otherwise show to the right
    - Summary line: "Min: X tokens • Median: Y tokens • Max: Z tokens" below chart
  - **Styling:** `space-y-2` for bars, `border-t border-slate-700 pt-4 mt-4` for summary
  - **Testing:** Render with sample histogram data, verify bar widths and text
  - **Reference:** TID Section 3.4

- [ ] 14.4.5 Integrate `SequenceLengthHistogram` into StatisticsTab (1 hour)
  - **File:** `frontend/src/components/datasets/DatasetDetailModal.tsx` (replace existing simple chart)
  - **Changes:**
    - Import `SequenceLengthHistogram` component
    - Replace simple 3-bar chart with: `{tokenizationStats?.histogram && <SequenceLengthHistogram ... />}`
    - Add fallback message if no histogram: "Histogram not available. Re-tokenize to generate."
  - **Testing:** View Statistics tab, verify histogram displays correctly
  - **Reference:** TID Section 3.4

- [ ] 14.4.6 Write unit tests for histogram calculation (2 hours)
  - **File:** `backend/tests/unit/test_tokenization_enhancements.py`
  - **Tests:**
    - `test_histogram_basic()` - Test with known sequence lengths, verify bucket counts
    - `test_histogram_percentages_sum_to_100()` - Test with random lengths, verify sum is ~100%
    - `test_histogram_empty_array()` - Test error handling for empty input
    - `test_histogram_all_same_length()` - Test edge case where all sequences have same length
  - **Reference:** TID Section 2.5

- [ ] 14.4.7 Write component tests for `SequenceLengthHistogram` (1.5 hours)
  - **File:** `frontend/src/components/datasets/SequenceLengthHistogram.test.tsx` (NEW)
  - **Tests:**
    - `test_renders_histogram_bars()` - Test all 7 bars render
    - `test_bar_widths_match_percentages()` - Test bar widths based on percentage
    - `test_shows_summary_statistics()` - Test min/median/max display
  - **Reference:** TID Section 5.2

**Estimated Time:** 10-14 hours
**Priority:** P1 (Critical)
**Dependencies:** None (can be done in parallel with other tasks)

---

### Phase 15: P2 Features (Medium Priority - 20-26 hours)

**Goal:** Implement nice-to-have tokenization features that improve UX.

---

#### Task 15.1: Implement Special Tokens Toggle (3-4 hours)

**Purpose:** Allow users to enable/disable automatic addition of special tokens (BOS, EOS, PAD).

**Acceptance Criteria:**
- ✅ Backend accepts `add_special_tokens` boolean parameter
- ✅ Tokenizer respects setting when tokenizing
- ✅ Setting stored in metadata
- ✅ Frontend toggle switch renders correctly
- ✅ Toggle state updates API request

**Sub-Tasks:**

- [ ] 15.1.1 Update `DatasetTokenizeRequest` schema (15 minutes)
  - **File:** `backend/src/schemas/dataset.py`
  - **Changes:** Add `add_special_tokens: bool = Field(default=True, description="Add BOS/EOS tokens")`
  - **Testing:** Create request with True/False, verify validation
  - **Reference:** TID Section 2.3

- [ ] 15.1.2 Update `tokenize_dataset_task` to use `add_special_tokens` (30 minutes)
  - **File:** `backend/src/workers/dataset_tasks.py`
  - **Changes:**
    - Add `add_special_tokens: bool = True` parameter to task
    - Pass to tokenizer: `add_special_tokens=add_special_tokens`
    - Store in metadata: `metadata_update["tokenization"]["add_special_tokens"] = add_special_tokens`
  - **Testing:** Run task with True/False, verify token counts
  - **Reference:** TID Section 2.3

- [ ] 15.1.3 Create `ToggleSwitch` component (1 hour)
  - **File:** `frontend/src/components/common/ToggleSwitch.tsx` (NEW)
  - **Structure:**
    - Props: `checked: boolean`, `onChange`, `label`, `helpText?`, `disabled?`
    - Layout: flex container with label/helpText on left, toggle button on right
    - Toggle button: `h-6 w-11 rounded-full` with emerald-600 (checked) or slate-700 (unchecked) background
    - Knob: `h-4 w-4 rounded-full bg-white` with `translate-x-6` (checked) or `translate-x-1` (unchecked)
  - **Accessibility:** `aria-checked`, `aria-label` attributes
  - **Testing:** Render, click, verify state changes
  - **Reference:** TID Section 3.2

- [ ] 15.1.4 Add special tokens toggle to TokenizationTab (30 minutes)
  - **File:** `frontend/src/components/datasets/DatasetDetailModal.tsx`
  - **Changes:**
    - Add state: `const [addSpecialTokens, setAddSpecialTokens] = useState(true)`
    - Insert `<ToggleSwitch>` component after truncation dropdown
    - Label: "Add Special Tokens", Help text: "Include BOS, EOS, PAD tokens (recommended for most models)"
    - Update `handleTokenize` to include `add_special_tokens: addSpecialTokens`
  - **Testing:** Toggle switch, verify API request
  - **Reference:** TID Section 3.2

- [ ] 15.1.5 Write unit tests for `ToggleSwitch` (45 minutes)
  - **File:** `frontend/src/components/common/ToggleSwitch.test.tsx` (NEW)
  - **Tests:**
    - `test_renders_toggle()` - Test rendering
    - `test_toggle_changes_state()` - Test clicking changes state
    - `test_disabled_state()` - Test disabled prop
  - **Reference:** TID Section 5.2

**Estimated Time:** 3-4 hours
**Priority:** P2 (Medium)
**Dependencies:** None

---

#### Task 15.2: Implement Attention Mask Toggle (3-4 hours)

**Purpose:** Allow users to enable/disable attention mask generation.

**Acceptance Criteria:**
- ✅ Backend accepts `return_attention_mask` boolean parameter
- ✅ Tokenizer respects setting when tokenizing
- ✅ Setting stored in metadata
- ✅ Frontend toggle switch renders correctly
- ✅ Toggle state updates API request

**Sub-Tasks:**

- [ ] 15.2.1 Update `DatasetTokenizeRequest` schema (15 minutes)
  - **File:** `backend/src/schemas/dataset.py`
  - **Changes:** Add `return_attention_mask: bool = Field(default=True, description="Return attention mask")`
  - **Testing:** Create request with True/False
  - **Reference:** TID Section 2.3

- [ ] 15.2.2 Update `tokenize_dataset_task` to use `return_attention_mask` (30 minutes)
  - **File:** `backend/src/workers/dataset_tasks.py`
  - **Changes:**
    - Add `return_attention_mask: bool = True` parameter
    - Pass to tokenizer: `return_attention_mask=return_attention_mask`
    - Store in metadata: `metadata_update["tokenization"]["return_attention_mask"] = return_attention_mask`
  - **Testing:** Run task with True/False, verify attention_mask field present/absent
  - **Reference:** TID Section 2.3

- [ ] 15.2.3 Add attention mask toggle to TokenizationTab (30 minutes)
  - **File:** `frontend/src/components/datasets/DatasetDetailModal.tsx`
  - **Changes:**
    - Add state: `const [returnAttentionMask, setReturnAttentionMask] = useState(true)`
    - Insert `<ToggleSwitch>` component after special tokens toggle
    - Label: "Return Attention Mask", Help text: "Generate attention masks (disable to save memory if model doesn't use them)"
    - Update `handleTokenize` to include `return_attention_mask: returnAttentionMask`
  - **Testing:** Toggle switch, verify API request
  - **Reference:** TID Section 3.2

- [ ] 15.2.4 Write integration test (1 hour)
  - **File:** `backend/tests/integration/test_dataset_api.py`
  - **Test:** `test_tokenize_with_attention_mask_disabled()`
  - **Steps:**
    1. Create dataset
    2. Tokenize with `return_attention_mask=False`
    3. Load tokenized dataset, verify no attention_mask field
  - **Reference:** TID Section 2.3

**Estimated Time:** 3-4 hours
**Priority:** P2 (Medium)
**Dependencies:** Task 15.1 (uses ToggleSwitch component)

---

#### Task 15.3: Implement Unique Tokens Metric (6-8 hours)

**Purpose:** Calculate and display the number of unique token IDs in the dataset.

**Acceptance Criteria:**
- ✅ Backend calculates unique tokens count
- ✅ Unique tokens stored in `dataset.metadata.tokenization.unique_tokens`
- ✅ Frontend displays unique tokens metric
- ✅ Handles large datasets efficiently (set-based algorithm)

**Sub-Tasks:**

- [ ] 15.3.1 Add `calculate_unique_tokens()` method to `TokenizationService` (2 hours)
  - **File:** `backend/src/services/tokenization_service.py` (after calculate_statistics)
  - **Implementation:**
    - Static method: `calculate_unique_tokens(tokenized_dataset: HFDataset) -> int`
    - Initialize `unique_tokens = set()`
    - Loop through dataset, `unique_tokens.update(example["input_ids"])`
    - Return `len(unique_tokens)`
  - **Error Handling:** Raise ValueError if dataset empty or no input_ids
  - **Performance:** O(n) time, O(u) space where u = unique tokens
  - **Testing:** Test with known token sets, verify count
  - **Reference:** TID Section 2.4

- [ ] 15.3.2 Update metadata schema with `unique_tokens` field (15 minutes)
  - **File:** `backend/src/schemas/metadata.py`
  - **Changes:** Add `unique_tokens: Optional[int] = None` to `TokenizationMetadata`
  - **Testing:** Create metadata with unique_tokens, verify validation
  - **Reference:** TID Section 2.4

- [ ] 15.3.3 Integrate into `calculate_statistics()` (30 minutes)
  - **File:** `backend/src/services/tokenization_service.py`
  - **Changes:**
    - Add try/except: `stats["unique_tokens"] = TokenizationService.calculate_unique_tokens(tokenized_dataset)`
    - On exception, print warning and set to None
  - **Testing:** Run tokenization, verify unique_tokens in metadata
  - **Reference:** TID Section 2.4

- [ ] 15.3.4 Update StatisticsTab to display unique tokens (1 hour)
  - **File:** `frontend/src/components/datasets/DatasetDetailModal.tsx`
  - **Changes:**
    - Add new `StatCard` component for unique tokens
    - Display format: "50,257 tokens (100% of GPT-2 vocab)"
    - Calculate percentage: `(unique_tokens / tokenizer_vocab_size) * 100`
    - Insert after total tokens card
  - **Testing:** View Statistics tab, verify unique tokens displayed
  - **Reference:** PRD Section 3.2 FR-4.1

- [ ] 15.3.5 Write unit tests for unique tokens calculation (1.5 hours)
  - **File:** `backend/tests/unit/test_tokenization_enhancements.py`
  - **Tests:**
    - `test_unique_tokens_basic()` - Test with known token IDs
    - `test_unique_tokens_with_duplicates()` - Test duplicates counted once
    - `test_unique_tokens_empty_dataset()` - Test error handling
    - `test_unique_tokens_missing_input_ids()` - Test error handling
  - **Reference:** TID Section 2.4

**Estimated Time:** 6-8 hours
**Priority:** P2 (Medium)
**Dependencies:** None

---

#### Task 15.4: Implement Split Distribution (8-10 hours)

**Purpose:** Display distribution of samples across train/validation/test splits.

**Acceptance Criteria:**
- ✅ Backend calculates split counts and percentages
- ✅ Split distribution stored in `dataset.metadata.splits`
- ✅ Frontend displays 3 color-coded cards (emerald/blue/purple)
- ✅ Handles missing splits gracefully (e.g., no test split)

**Sub-Tasks:**

- [ ] 15.4.1 Add `calculate_split_distribution()` method to `TokenizationService` (2 hours)
  - **File:** `backend/src/services/tokenization_service.py`
  - **Implementation:**
    - Static method: `calculate_split_distribution(dataset: Any) -> Dict[str, Dict[str, Any]]`
    - Check if dataset has splits: `if not hasattr(dataset, "keys"): return {"train": {...}}`
    - Calculate total samples: `sum(len(split) for split in dataset.values())`
    - Loop through splits, calculate count and percentage
    - Return dict mapping split_name → {count, percentage}
  - **Error Handling:** Handle empty datasets, single-split datasets
  - **Testing:** Test with DatasetDict, test with single split, test with missing splits
  - **Reference:** TID Section 2.6

- [ ] 15.4.2 Update metadata schema with `splits` field (30 minutes)
  - **File:** `backend/src/schemas/metadata.py`
  - **Changes:**
    - Create `SplitInfo` model (count: int, percentage: float)
    - Create `SplitsMetadata` model (train?: SplitInfo, validation?: SplitInfo, test?: SplitInfo)
    - Add `splits: Optional[SplitsMetadata] = None` to `DatasetMetadata`
  - **Testing:** Create metadata with splits, verify validation
  - **Reference:** TID Section 2.6

- [ ] 15.4.3 Integrate split calculation into `download_dataset_task` (1 hour)
  - **File:** `backend/src/workers/dataset_tasks.py`
  - **Changes:**
    - After `dataset.save_to_disk(raw_path)`, call `splits = TokenizationService.calculate_split_distribution(dataset)`
    - Store in metadata: `ds.extra_metadata = {"splits": splits}`
  - **Testing:** Download dataset with splits, verify splits in metadata
  - **Reference:** TID Section 2.6

- [ ] 15.4.4 Create `SplitDistribution` component (2 hours)
  - **File:** `frontend/src/components/datasets/SplitDistribution.tsx` (NEW)
  - **Structure:**
    - Props: `splits: {train?, validation?, test?}`
    - Define split configs with colors: train (emerald), validation (blue), test (purple)
    - Filter to active splits: `splitConfigs.filter(s => s.data)`
    - Grid layout: `grid grid-cols-3 gap-4`
    - Each card: bg color (emerald-900/30), border color, label, count, percentage
  - **Styling:** Match Mock UI lines 4371-4391
  - **Testing:** Render with 3 splits, render with 2 splits, render with 0 splits
  - **Reference:** TID Section 3.5

- [ ] 15.4.5 Integrate `SplitDistribution` into StatisticsTab (30 minutes)
  - **File:** `frontend/src/components/datasets/DatasetDetailModal.tsx`
  - **Changes:**
    - Import `SplitDistribution` component
    - Insert after histogram: `{dataset.metadata?.splits && <SplitDistribution splits={dataset.metadata.splits} />}`
  - **Testing:** View Statistics tab, verify split cards display
  - **Reference:** TID Section 3.5

- [ ] 15.4.6 Write unit tests for split distribution calculation (1.5 hours)
  - **File:** `backend/tests/unit/test_tokenization_enhancements.py`
  - **Tests:**
    - `test_split_distribution_balanced()` - Test with 80/10/10 split
    - `test_split_distribution_missing_split()` - Test with only train/val
    - `test_split_distribution_single_split()` - Test single-split dataset
  - **Reference:** TID Section 2.6

- [ ] 15.4.7 Write component tests for `SplitDistribution` (1 hour)
  - **File:** `frontend/src/components/datasets/SplitDistribution.test.tsx` (NEW)
  - **Tests:**
    - `test_renders_split_cards()` - Test 3 cards render
    - `test_correct_color_coding()` - Test emerald/blue/purple colors
    - `test_handles_missing_splits()` - Test graceful degradation
  - **Reference:** TID Section 5.2

**Estimated Time:** 8-10 hours
**Priority:** P2 (Medium)
**Dependencies:** None

---

### Phase 16: Integration & Testing (6-8 hours)

**Goal:** Ensure all new features work together correctly and match Mock UI exactly.

---

#### Task 16.1: End-to-End Testing (3-4 hours)

**Purpose:** Test complete workflow from tokenization to statistics display.

**Sub-Tasks:**

- [ ] 16.1.1 Test full tokenization workflow with all settings (1 hour)
  - **Steps:**
    1. Download TinyStories dataset
    2. Open Tokenization tab
    3. Select padding: "longest", truncation: "only_first"
    4. Enable special tokens, disable attention mask
    5. Preview tokenization on sample text
    6. Apply tokenization
    7. Wait for completion (WebSocket updates)
    8. Open Statistics tab
    9. Verify histogram displays 7 buckets
    10. Verify split distribution shows train/val/test
  - **Acceptance Criteria:** All steps complete without errors, metadata persists

- [ ] 16.1.2 Test edge cases (1 hour)
  - **Cases:**
    - Empty preview text (button should be disabled)
    - Very long preview text (1000+ chars, should show error)
    - Invalid tokenizer name (should show error)
    - Dataset with only one split (should handle gracefully)
    - Dataset with all sequences same length (histogram should work)
  - **Acceptance Criteria:** All edge cases handled gracefully with helpful error messages

- [ ] 16.1.3 Test backward compatibility (1 hour)
  - **Steps:**
    1. Load existing tokenized dataset (created before enhancements)
    2. Open Statistics tab
    3. Verify fallback messages for missing histogram/unique tokens
    4. Re-tokenize with new settings
    5. Verify new statistics appear
  - **Acceptance Criteria:** Old datasets don't crash, re-tokenization adds new features

**Estimated Time:** 3-4 hours

---

#### Task 16.2: Mock UI Alignment Verification (2-3 hours)

**Purpose:** Verify all components match Mock UI exactly.

**Sub-Tasks:**

- [ ] 16.2.1 Verify TokenizationTab matches Mock UI (1 hour)
  - **Reference:** Mock UI lines 4086-4288
  - **Check:**
    - Padding dropdown (lines 4154-4168): 3 options, help text
    - Truncation dropdown (lines 4170-4185): 4 options, help text
    - Special tokens toggle (lines 4187-4203): label, help text, styling
    - Attention mask toggle (lines 4205-4221): label, help text, styling
    - Tokenization preview (lines 4225-4263): text area, button, token chips
  - **Method:** Side-by-side comparison, screenshot diff
  - **Acceptance Criteria:** All styling matches exactly (colors, spacing, fonts)

- [ ] 16.2.2 Verify StatisticsTab matches Mock UI (1 hour)
  - **Reference:** Mock UI lines 4290-4393
  - **Check:**
    - Histogram (lines 4344-4369): 7 buckets, emerald gradient, summary line
    - Split distribution (lines 4371-4391): 3 cards, color-coded, percentages
  - **Method:** Side-by-side comparison, screenshot diff
  - **Acceptance Criteria:** All styling matches exactly

- [ ] 16.2.3 Update any mismatches (variable time)
  - **If mismatches found:** Update CSS classes to match Mock UI
  - **Priority:** Exact match required for MVP

**Estimated Time:** 2-3 hours

---

#### Task 16.3: Performance Testing (1 hour)

**Purpose:** Verify performance meets requirements.

**Sub-Tasks:**

- [ ] 16.3.1 Test tokenization preview performance (<1s p95)
  - **Method:**
    1. Open DevTools Network tab
    2. Call preview endpoint 10 times
    3. Measure response times
  - **Target:** <1s p95
  - **If fails:** Verify tokenizer caching is working

- [ ] 16.3.2 Test histogram calculation performance (<2s for 1M samples)
  - **Method:**
    1. Tokenize large dataset (TinyStories: 2.1M samples)
    2. Measure histogram calculation time (check backend logs)
  - **Target:** <2s
  - **If fails:** Verify NumPy vectorization is used

**Estimated Time:** 1 hour

---

#### Task 16.4: Documentation Updates (0.5-1 hour)

**Purpose:** Update documentation with new features.

**Sub-Tasks:**

- [ ] 16.4.1 Update CLAUDE.md with enhancement completion (15 minutes)
  - **File:** `CLAUDE.md`
  - **Changes:**
    - Update "Current Status" section
    - Mark Phase 14 and Phase 15 as complete
    - Update "Test Status" with new test counts

- [ ] 16.4.2 Update task list with completion status (15 minutes)
  - **File:** `0xcc/tasks/001_FTASKS|Dataset_Management_ENH_01.md`
  - **Changes:** Mark all tasks as complete `[x]`

**Estimated Time:** 0.5-1 hour

---

## Implementation Notes

### Priority Order

**Week 1 (P1 Features):**
1. Day 1: Task 14.1 (Padding) + Task 14.2 (Truncation) - Can be done in parallel
2. Day 2-3: Task 14.3 (Tokenization Preview)
3. Day 4-5: Task 14.4 (Histogram)

**Week 2 (P2 Features):**
1. Day 1: Task 15.1 (Special Tokens) + Task 15.2 (Attention Mask) - Use shared ToggleSwitch
2. Day 2-3: Task 15.3 (Unique Tokens) + Task 15.4 (Split Distribution) - Can be done in parallel

**Week 2 (Testing & Polish):**
1. Day 4-5: Task 16.1-16.4 (Integration, alignment, performance, docs)

### Parallelization Opportunities

**Backend Tasks (can be done simultaneously):**
- Task 14.1 (Padding) and Task 14.2 (Truncation) - Independent schema/service updates
- Task 15.3 (Unique Tokens) and Task 15.4 (Split Distribution) - Independent calculations

**Frontend Tasks (can be done after backend):**
- All dropdown/toggle UI components can be implemented in parallel
- Component tests can be written concurrently

### Testing Strategy

**Unit Tests First:**
- Write unit tests for each new service method before integration
- Target: >90% coverage for new code

**Integration Tests:**
- Test API endpoints after unit tests pass
- Use TestClient for FastAPI endpoints

**Component Tests:**
- Write component tests after UI implementation
- Use React Testing Library + MSW for mocking

**E2E Tests:**
- Test complete workflows in Phase 16
- Manual testing + automated Playwright tests (future)

### Code Quality Checklist

**Before Each Commit:**
- [ ] All tests passing (backend: `pytest`, frontend: `npm test`)
- [ ] No console.log/print debugging statements
- [ ] No commented-out code blocks
- [ ] Code follows project naming conventions
- [ ] Type hints on all Python functions
- [ ] No `any` types in TypeScript
- [ ] Error handling implemented
- [ ] Docstrings on all new functions

### Common Issues and Solutions

**Issue 1: Percentages don't sum to exactly 100%**
- **Cause:** Floating point rounding
- **Solution:** Allow 99.9-100.1 range in validation

**Issue 2: Tokenizer loading slow on first preview**
- **Cause:** HuggingFace downloads on first use
- **Solution:** Cache tokenizers with `@lru_cache(maxsize=10)`

**Issue 3: Frontend crashes on incomplete metadata**
- **Cause:** Missing null checks
- **Solution:** Use optional chaining: `tokenizationStats?.histogram`

**Issue 4: Histogram bars look wrong**
- **Cause:** CSS width calculation
- **Solution:** Use `Math.max(bucket.percentage, 0.5)` for minimum visibility

---

**Document Status:** Ready for Implementation
**Total Tasks:** 4 major tasks (14.1-14.4, 15.1-15.4), 16 phases, ~46 sub-tasks
**Estimated Total Effort:** 46-64 hours (6-8 working days)
**Target Completion:** 2 weeks with 1 developer

---

**Approval Required From:**
- [ ] Tech Lead (feasibility review)
- [ ] QA Lead (acceptance criteria review)

**Change Log:**
- 2025-10-11: Initial creation from TID and PRD
