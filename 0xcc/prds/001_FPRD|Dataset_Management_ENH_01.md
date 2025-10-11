# Feature Product Requirements Document: Dataset Management Enhancement 01

**Document ID:** 001_FPRD|Dataset_Management_ENH_01
**Feature:** Dataset Management - Missing Features from Mock UI Reference
**Status:** Ready for Implementation
**Created:** 2025-10-11
**Last Updated:** 2025-10-11
**Owner:** miStudio Development Team
**Priority:** P0 (Critical - Blocks Feature 2)

**Extends:** 001_FPRD|Dataset_Management.md
**Related Documents:**
- Gap Analysis: `Dataset_Management_Feature_Gap_Analysis.md`
- Original PRD: `001_FPRD|Dataset_Management.md`
- ADR: `000_PADR|miStudio.md`
- UI Reference: `Mock-embedded-interp-ui.tsx` (lines 4086-4393)

---

## 1. Executive Summary

### 1.1 Purpose

This enhancement document addresses **critical feature gaps** discovered between the production implementation and the Mock UI reference design (`Mock-embedded-interp-ui.tsx.reference`). The gap analysis revealed **9 missing or partially implemented features** representing **46-64 hours of development** that were not captured in the original MVP scope.

**Why This Matters:**
- Mock UI is the **PRIMARY REFERENCE** (CLAUDE.md mandate)
- Production should match design specification, not vice versa
- These are NOT convenience features - they are essential for production use
- Technical debt cost > upfront implementation cost

### 1.2 Gap Summary

| Component | Missing Features | Effort | Impact |
|-----------|------------------|--------|---------|
| **Tokenization Tab** | 5 features | 22-32h | Users cannot optimize tokenization or preview results |
| **Statistics Tab** | 4 features | 24-32h | Users cannot understand distribution or verify splits |
| **Total** | **9 features** | **46-64h** | **Production-blocking gaps** |

### 1.3 Recommended Approach

**Option A: Implement All Missing Features** (Selected)
- **Timeline:** 46-64 hours (~6-8 working days)
- **Rationale:** Closes gap completely, maintains design consistency, prevents technical debt
- **Phased:** P1 features (26-38h) then P2 features (20-26h)

---

## 2. Problem Statement

### 2.1 Current State

The production Dataset Management implementation has **60% feature parity** with the Mock UI reference:

**Tokenization Tab (40% Complete):**
- ✅ Tokenizer selection (8 options)
- ✅ Max length slider (128-2048)
- ✅ Stride slider
- ❌ Padding strategy dropdown
- ❌ Truncation strategy dropdown
- ❌ Add special tokens toggle
- ❌ Return attention mask toggle
- ❌ Tokenization preview with live visualization

**Statistics Tab (50% Complete):**
- ✅ Total samples, tokens, avg/min/max length
- ✅ Simple 3-bar chart
- ❌ Unique tokens metric
- ❌ Median length metric
- ❌ Full histogram (7+ buckets)
- ❌ Split distribution (train/val/test)

### 2.2 User Impact

**Without these features, users cannot:**

1. **Optimize tokenization for their models**
   - No padding/truncation strategy = one-size-fits-all approach
   - Cannot handle multi-sequence inputs (question-answer pairs)

2. **Preview tokenization results**
   - Must process entire dataset (hours) to see if settings are correct
   - No way to test before committing to full tokenization

3. **Understand token distribution**
   - Simple min/avg/max doesn't show skew, outliers, or clustering
   - Cannot identify dataset quality issues

4. **Verify dataset splits**
   - No visibility into train/val/test ratios
   - Cannot validate data integrity before training

5. **Experiment with special tokens**
   - Cannot test raw sequences or custom tokenizers
   - Cannot disable attention masks for models that don't use them

6. **Calculate vocabulary size**
   - Missing unique tokens metric prevents memory estimation

### 2.3 Why This Gap Exists

1. **Incomplete Feature Extraction**: Original PRD creation didn't break down ALL settings visible in Mock UI
2. **Design Drift**: Production evolved based on immediate needs (stride, efficiency metrics) rather than strict Mock UI adherence
3. **Scope Reduction**: MVP focused on "core functionality" without recognizing Mock UI had essential (not convenience) features
4. **Mock UI Not Fully Analyzed**: Team compared production against PRD requirements, but Mock UI had additional features not explicitly listed in PRD

---

## 3. Requirements

### 3.1 Tokenization Tab Requirements

#### FR-3.1: Padding Strategy Selection (P1 - High Priority)

**Description:**
Users must be able to select padding strategy for tokenization to optimize memory usage and performance.

**Mock UI Reference:** Lines 4154-4168

**Options:**
- `max_length` - Pad all sequences to max_length (consistent memory)
- `longest` - Pad to longest sequence in batch (variable memory)
- `do_not_pad` - No padding (minimal memory)

**Acceptance Criteria:**
- ✅ Dropdown in Tokenization Tab with 3 options
- ✅ Default: `max_length`
- ✅ Setting persists in tokenization metadata
- ✅ Backend applies correct padding strategy to HuggingFace tokenizer
- ✅ Help text explains each option

**User Story:**
> As a researcher, I want to select padding strategy so that I can optimize memory usage for my specific model architecture and batch size.

**Estimated Effort:** 4-6 hours (backend + frontend + validation + tests)

---

#### FR-3.2: Truncation Strategy Selection (P1 - High Priority)

**Description:**
Users must be able to select truncation strategy to handle sequences exceeding max_length.

**Mock UI Reference:** Lines 4170-4185

**Options:**
- `longest_first` - Truncate longest sequence first (balanced)
- `only_first` - Only truncate first sequence (preserve second)
- `only_second` - Only truncate second sequence (preserve first)
- `do_not_truncate` - No truncation (error on overflow)

**Acceptance Criteria:**
- ✅ Dropdown in Tokenization Tab with 4 options
- ✅ Default: `longest_first`
- ✅ Setting persists in tokenization metadata
- ✅ Backend applies correct truncation strategy to HuggingFace tokenizer
- ✅ Help text explains use case for each option (e.g., question-answer pairs)

**User Story:**
> As a researcher working with multi-sequence inputs (Q&A pairs, prompts), I want to control truncation strategy so that I can preserve the most important content for my task.

**Estimated Effort:** 4-6 hours (backend + frontend + validation + tests)

---

#### FR-3.3: Add Special Tokens Toggle (P2 - Medium Priority)

**Description:**
Users must be able to enable/disable automatic addition of special tokens (BOS, EOS, PAD, etc.).

**Mock UI Reference:** Lines 4187-4203

**Options:**
- `true` - Add special tokens (default, for most models)
- `false` - No special tokens (for custom tokenizers or raw sequences)

**Acceptance Criteria:**
- ✅ Toggle switch in Tokenization Tab
- ✅ Default: `true`
- ✅ Setting persists in tokenization metadata
- ✅ Backend passes `add_special_tokens` to HuggingFace tokenizer
- ✅ Help text explains when to disable (custom tokenizers, raw sequences)

**User Story:**
> As a researcher experimenting with custom tokenizers, I want to control special token addition so that I can test raw token sequences without BOS/EOS/PAD interference.

**Estimated Effort:** 3-4 hours (backend + frontend + tests)

---

#### FR-3.4: Return Attention Mask Toggle (P2 - Medium Priority)

**Description:**
Users must be able to enable/disable attention mask generation for models that don't use them.

**Mock UI Reference:** Lines 4205-4221

**Options:**
- `true` - Generate attention mask (default, for most models)
- `false` - No attention mask (for models without attention)

**Acceptance Criteria:**
- ✅ Toggle switch in Tokenization Tab
- ✅ Default: `true`
- ✅ Setting persists in tokenization metadata
- ✅ Backend passes `return_attention_mask` to HuggingFace tokenizer
- ✅ Help text explains memory savings when disabled

**User Story:**
> As a researcher working with simple models that don't use attention, I want to disable attention mask generation so that I can save memory and storage space.

**Estimated Effort:** 3-4 hours (backend + frontend + tests)

---

#### FR-3.5: Tokenization Preview (P1 - High Priority)

**Description:**
Users must be able to preview tokenization on sample text BEFORE processing the entire dataset.

**Mock UI Reference:** Lines 4225-4263

**Components:**
- Text area input for sample text (3-5 lines)
- "Tokenize Preview" button
- Visual display of tokens as colored chips
- Token count display
- Special token highlighting (different color)

**Acceptance Criteria:**
- ✅ Preview section below tokenization settings
- ✅ Text area accepts up to 1000 characters
- ✅ Preview button calls `POST /api/datasets/tokenize-preview` endpoint
- ✅ Tokens rendered as chips: emerald for special tokens, slate for regular
- ✅ Each chip shows token text and ID (on hover)
- ✅ Token count matches expected
- ✅ Works with all tokenizer configurations (padding, truncation, special tokens)
- ✅ Preview completes in <1 second
- ✅ Error handling for invalid inputs

**User Story:**
> As a researcher, I want to preview tokenization on sample text so that I can verify my settings are correct before committing hours to processing the full dataset.

**Estimated Effort:** 8-12 hours (backend endpoint + frontend + token visualization + error handling)

**Backend Endpoint:**
```python
POST /api/datasets/tokenize-preview
{
  "tokenizer_name": "gpt2",
  "text": "Sample text to tokenize...",
  "max_length": 512,
  "padding": "max_length",
  "truncation": "longest_first",
  "add_special_tokens": true,
  "return_attention_mask": true
}

Response:
{
  "tokens": [
    {"id": 50256, "text": "<|endoftext|>", "type": "special", "position": 0},
    {"id": 27565, "text": "Sample", "type": "regular", "position": 1},
    {"id": 2420, "text": " text", "type": "regular", "position": 2},
    ...
  ],
  "attention_mask": [1, 1, 1, 0, 0, ...],
  "token_count": 15,
  "sequence_length": 512
}
```

---

### 3.2 Statistics Tab Requirements

#### FR-4.1: Unique Tokens Metric (P2 - Medium Priority)

**Description:**
Display the count of unique tokens (vocabulary size) in the tokenized dataset.

**Mock UI Reference:** Line 4296 (implicit in summary cards)

**Calculation:**
```python
unique_tokens = len(set(all_token_ids_in_dataset))
```

**Acceptance Criteria:**
- ✅ Summary card in Statistics Tab
- ✅ Shows unique token count and percentage of tokenizer vocab
- ✅ Calculated during tokenization and stored in metadata
- ✅ Displayed with format: "50,257 tokens (100% of GPT-2 vocab)"

**User Story:**
> As a researcher, I want to see unique token count so that I can estimate memory requirements for embeddings and vocabulary tables.

**Estimated Effort:** 6-8 hours (backend calculation + caching + frontend display)

---

#### FR-4.2: Median Length Metric (P2 - Medium Priority)

**Description:**
Display the median sequence length (in addition to current avg/min/max).

**Mock UI Reference:** Line 4296 (implicit in summary cards), Line 4360 (histogram summary)

**Calculation:**
```python
median_seq_length = np.median(sequence_lengths)
```

**Acceptance Criteria:**
- ✅ Summary card or inline metric in Statistics Tab
- ✅ Calculated during tokenization and stored in metadata
- ✅ Displayed with format: "Median: 342 tokens"
- ✅ Included in histogram summary line (Min/Median/Max)

**User Story:**
> As a researcher, I want to see median sequence length so that I can identify skewed distributions where mean is misleading.

**Estimated Effort:** 2-3 hours (included in histogram calculation)

---

#### FR-4.3: Sequence Length Histogram (P1 - High Priority)

**Description:**
Replace simple 3-bar chart with full histogram showing distribution of sequence lengths across 7+ buckets.

**Mock UI Reference:** Lines 4344-4369

**Design:**
- **Bins:** 0-100, 100-200, 200-400, 400-600, 600-800, 800-1000, 1000+ tokens
- **Visualization:** Horizontal bar chart with gradient colors (emerald-500 to emerald-400)
- **Data:** Sample count and percentage per bucket
- **Summary:** Min/Median/Max below chart

**Acceptance Criteria:**
- ✅ Histogram with 7 length buckets
- ✅ Each bar shows count and percentage
- ✅ Gradient emerald colors (matches Mock UI)
- ✅ Percentages sum to 100%
- ✅ Min/Median/Max summary below chart
- ✅ Handles edge cases (all same length, wide spread)
- ✅ Calculates bins dynamically based on max_length

**User Story:**
> As a researcher, I want to see a detailed sequence length histogram so that I can identify distribution skew, outliers, and clustering patterns that impact training.

**Estimated Effort:** 10-14 hours (backend histogram calculation + binning strategy + frontend chart + tests)

**Backend Calculation:**
```python
def calculate_histogram(lengths: np.ndarray, max_length: int) -> List[Dict[str, Any]]:
    bins = [0, 100, 200, 400, 600, 800, 1000, max_length]
    histogram = []

    for i in range(len(bins) - 1):
        count = np.sum((lengths >= bins[i]) & (lengths < bins[i+1]))
        histogram.append({
            "range": f"{bins[i]}-{bins[i+1]}" if i < len(bins) - 2 else f"{bins[i]}+",
            "min": bins[i],
            "max": bins[i+1] if i < len(bins) - 2 else max_length,
            "count": int(count),
            "percentage": float(count / len(lengths) * 100),
        })

    return histogram
```

---

#### FR-4.4: Split Distribution (P2 - Medium Priority)

**Description:**
Display the distribution of samples across dataset splits (train/validation/test).

**Mock UI Reference:** Lines 4371-4391

**Design:**
- **Cards:** 3 cards showing train/val/test splits
- **Data:** Sample count and percentage of total per split
- **Colors:** Emerald (train), blue (val), purple (test)

**Acceptance Criteria:**
- ✅ 3 split cards in Statistics Tab
- ✅ Each card shows split name, sample count, and percentage
- ✅ Color-coded: emerald-500 (train), blue-500 (val), purple-500 (test)
- ✅ Data extracted from HuggingFace dataset splits
- ✅ Handles missing splits (e.g., no test split)
- ✅ Percentages sum to 100%

**User Story:**
> As a researcher, I want to see split distribution so that I can verify my train/val/test ratios are correct before starting SAE training.

**Estimated Effort:** 8-10 hours (backend split calculation + metadata storage + frontend display)

**Backend Calculation:**
```python
def calculate_split_distribution(dataset: HFDataset) -> Dict[str, Dict[str, Any]]:
    splits = {}
    total_samples = sum(len(split) for split in dataset.values())

    for split_name, split_data in dataset.items():
        count = len(split_data)
        splits[split_name] = {
            "count": count,
            "percentage": (count / total_samples * 100) if total_samples > 0 else 0
        }

    return splits
```

---

## 4. Implementation Phases

### Phase 1: P1 Features (High Priority) - 26-38 hours

**Priority:** Implement first - these are production-blocking gaps

**Features:**
1. FR-3.1: Padding strategy (4-6h)
2. FR-3.2: Truncation strategy (4-6h)
3. FR-3.5: Tokenization preview (8-12h)
4. FR-4.3: Sequence length histogram (10-14h)

**Dependencies:**
- Phase 1 tasks are independent and can be parallelized
- Tokenization preview requires padding/truncation settings (but can use defaults initially)

**Testing Strategy:**
- Unit tests for each backend service method
- Integration tests for API endpoints
- Frontend component tests for UI rendering
- E2E test: Preview tokenization → apply settings → tokenize → view histogram

---

### Phase 2: P2 Features (Medium Priority) - 20-26 hours

**Priority:** Implement after Phase 1 - important but not blocking

**Features:**
1. FR-3.3: Add special tokens toggle (3-4h)
2. FR-3.4: Return attention mask toggle (3-4h)
3. FR-4.1: Unique tokens metric (6-8h)
4. FR-4.2: Median length metric (2-3h)
5. FR-4.4: Split distribution (8-10h)

**Dependencies:**
- FR-4.2 (median) should be implemented with FR-4.3 (histogram) for efficiency
- All other tasks are independent

**Testing Strategy:**
- Unit tests for each feature
- Integration test: Full tokenization with all settings enabled
- Regression test: Verify existing features still work

---

## 5. Success Criteria

### 5.1 Functional Completeness

**Tokenization Tab:**
- ✅ All 8 Mock UI features implemented (5 new + 3 existing)
- ✅ Preview works with all tokenizer configurations
- ✅ Settings persist across page refreshes
- ✅ Help text guides users on when to use each option

**Statistics Tab:**
- ✅ All 5 Mock UI sections implemented (4 new + 1 existing)
- ✅ Histogram accurately represents distribution
- ✅ Split distribution handles edge cases (missing splits, unbalanced)
- ✅ Unique tokens calculated correctly

### 5.2 User Experience

- ✅ No crashes or errors when viewing incomplete tokenization data
- ✅ Graceful degradation with helpful error messages
- ✅ Loading states for all async operations
- ✅ Responsive design (works on 1920x1080 and 2560x1440)

### 5.3 Performance

- ✅ Tokenization preview completes in <1 second
- ✅ Histogram calculation completes in <2 seconds for 1M samples
- ✅ Statistics tab loads in <1 second
- ✅ WebSocket updates have <100ms latency

### 5.4 Code Quality

- ✅ All TypeScript strict mode with no `any` types
- ✅ All Python type hints with MyPy checking
- ✅ Backend test coverage >90%
- ✅ Frontend test coverage >80%
- ✅ No console warnings or errors

### 5.5 Design Consistency

- ✅ All components match Mock UI exactly (CSS classes, colors, spacing)
- ✅ Visual regression tests pass (screenshot comparison)
- ✅ Lucide icons match Mock UI
- ✅ Tailwind slate dark theme consistent throughout

---

## 6. Non-Goals

**Explicitly Out of Scope:**
- Custom binning for histogram (fixed 7 buckets per Mock UI)
- Export statistics to CSV/JSON (future enhancement)
- Advanced tokenizer configuration (e.g., wordpiece vs BPE)
- Tokenization performance optimization (batching, parallelization)
- Multi-language tokenizer support beyond HuggingFace models
- Custom tokenizer upload (only HuggingFace Hub models)

---

## 7. Risk Assessment

### 7.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Tokenization preview performance | Medium | High | Cache tokenizers, limit preview to 1000 chars |
| Histogram calculation on large datasets | High | Medium | Use NumPy vectorization, sample if >1M samples |
| Breaking changes to existing tokenization | Low | High | Extensive regression testing before merge |
| Frontend state management complexity | Medium | Medium | Use Zustand store for tokenization settings |

### 7.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| 46-64 hour estimate exceeded | Medium | High | Use phased approach (P1 first, P2 later) |
| Blocking Feature 2 progress | High | Medium | Parallel workstreams: one engineer on gap, one on Feature 2 prep |
| Testing takes longer than expected | Medium | Medium | Prioritize critical paths, defer edge case tests |

---

## 8. Dependencies

### 8.1 Internal Dependencies

**Blocked By:**
- None (all existing features are complete)

**Blocks:**
- Feature 2 (Model Management) - Should not start until Dataset Management is feature-complete

### 8.2 External Dependencies

- HuggingFace `transformers` library (tokenization preview)
- HuggingFace `datasets` library (split information)
- NumPy (histogram binning, statistics)
- Socket.IO (WebSocket progress updates)

---

## 9. Acceptance Testing

### 9.1 Test Scenarios

**Scenario 1: Tokenization Preview**
1. User opens Tokenization Tab
2. User selects "gpt2" tokenizer
3. User enters sample text: "Once upon a time..."
4. User clicks "Tokenize Preview"
5. Preview shows tokens within 1 second
6. Special tokens highlighted in emerald
7. Token count matches expected

**Scenario 2: Histogram Visualization**
1. User tokenizes dataset with 10,000 samples
2. User opens Statistics Tab
3. Histogram displays 7 length buckets
4. Each bar shows count and percentage
5. Percentages sum to 100%
6. Min/Median/Max summary below chart

**Scenario 3: Padding Strategy Impact**
1. User selects "max_length" padding
2. User tokenizes dataset
3. Statistics show 100% utilization (all sequences padded to max_length)
4. User re-tokenizes with "longest" padding
5. Statistics show variable utilization (less padding)

**Scenario 4: Split Distribution**
1. User downloads dataset with train/val/test splits
2. User opens Statistics Tab
3. Split distribution shows 3 cards: 80% train, 10% val, 10% test
4. Colors match Mock UI (emerald/blue/purple)

### 9.2 Performance Testing

**Load Tests:**
- Tokenization preview with 1000 char text: <1s
- Histogram calculation with 1M samples: <2s
- Statistics tab full render: <1s
- WebSocket progress update latency: <100ms

**Stress Tests:**
- Tokenization preview with unicode/emoji characters
- Histogram with all samples same length (edge case)
- Histogram with wide spread (0 to 8192 tokens)
- Split distribution with missing splits

---

## 10. Rollout Plan

### 10.1 Development Phases

**Week 1 (Phase 1 - P1 Features):**
- Day 1-2: Padding strategy (FR-3.1)
- Day 2-3: Truncation strategy (FR-3.2)
- Day 3-5: Tokenization preview (FR-3.5)
- Day 5-7: Sequence length histogram (FR-4.3)

**Week 2 (Phase 2 - P2 Features):**
- Day 1: Special tokens toggle (FR-3.3)
- Day 1: Attention mask toggle (FR-3.4)
- Day 2-3: Unique tokens metric (FR-4.1)
- Day 3: Median length metric (FR-4.2)
- Day 4-5: Split distribution (FR-4.4)

**Week 2 (Testing & Polish):**
- Day 5-7: Integration testing, bug fixes, visual regression tests

### 10.2 Testing Milestones

- **Milestone 1 (End of Week 1):** P1 features complete, unit tests passing
- **Milestone 2 (End of Week 2):** All features complete, integration tests passing
- **Milestone 3 (End of Week 2):** E2E tests passing, production-ready

### 10.3 Deployment Strategy

**Development:**
- Feature branch: `feature/dataset-management-enhancements`
- Merge to `develop` after each phase passes tests
- Deploy to staging environment for QA testing

**Production:**
- Deploy after full test suite passes
- Monitor WebSocket connections for 24 hours
- Rollback plan: Revert to previous version if critical bugs found

---

## 11. Monitoring and Metrics

### 11.1 Success Metrics

**Usage Metrics:**
- % of users using tokenization preview before full tokenization (target: >50%)
- Average time spent viewing Statistics tab (target: >30s)
- % of datasets with custom padding/truncation settings (target: >20%)

**Performance Metrics:**
- Tokenization preview p95 latency (target: <1s)
- Histogram calculation p95 time (target: <2s)
- WebSocket update latency p95 (target: <100ms)

**Quality Metrics:**
- Frontend error rate (target: <0.1%)
- Backend API error rate for new endpoints (target: <0.5%)
- Test coverage (target: >85% overall)

### 11.2 Alerts

- **Critical:** Tokenization preview endpoint error rate >5%
- **Critical:** WebSocket connection failures >10%
- **Warning:** Histogram calculation time >5s
- **Warning:** Statistics tab load time >2s

---

## 12. Future Enhancements

**Post-MVP Improvements (Phase 3+):**
- Custom histogram binning (user-configurable buckets)
- Export statistics to CSV/JSON
- Tokenization preview with multiple examples (batch preview)
- Tokenization settings templates (save/load presets)
- Advanced statistics: token frequency distribution, rare token count
- Vocabulary overlap analysis (compare datasets)

---

## 13. Open Questions

1. **Histogram Binning:** Should bins be user-configurable or fixed per Mock UI? → **Decision:** Fixed 7 buckets per Mock UI
2. **Unique Tokens:** Should we cache calculation or recalculate on Statistics tab open? → **Decision:** Calculate during tokenization, store in metadata
3. **Tokenization Preview:** Should we limit characters or tokens? → **Decision:** Limit to 1000 characters input
4. **Split Distribution:** How to handle datasets with only one split (e.g., no test)? → **Decision:** Show only existing splits, omit missing ones

---

## 14. Appendix

### 14.1 Mock UI Line References

**Tokenization Tab:**
- Tokenizer selection: Lines 4123-4136
- Max length: Lines 4139-4152
- Padding strategy: Lines 4154-4168
- Truncation strategy: Lines 4170-4185
- Add special tokens: Lines 4187-4203
- Attention mask: Lines 4205-4221
- Tokenization preview: Lines 4225-4263
- Apply button: Lines 4265-4285

**Statistics Tab:**
- Summary cards: Lines 4316-4342
- Sequence length histogram: Lines 4344-4369
- Split distribution: Lines 4371-4391

### 14.2 Related Gap Analysis Sections

- Section 1: Tokenization Tab Gap Analysis (Lines 26-129)
- Section 2: Statistics Tab Gap Analysis (Lines 131-210)
- Section 3: Combined Gap Summary (Lines 212-228)
- Section 6: Detailed Implementation Plan (Lines 330-648)

---

**Document Status:** Ready for Technical Design
**Next Step:** Create 001_FTDD|Dataset_Management_ENH_01.md with architecture and design
**Estimated Total Effort:** 46-64 hours (6-8 working days)
**Priority:** P0 (Critical - Must complete before Feature 2)

---

**Approval Required From:**
- [ ] Product Owner (feature scope confirmation)
- [ ] Tech Lead (feasibility review)
- [ ] UI/UX Designer (Mock UI alignment verification)
- [ ] QA Lead (acceptance criteria review)

**Change Log:**
- 2025-10-11: Initial creation from gap analysis findings
