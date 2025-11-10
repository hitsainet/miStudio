# Enhancement: Per-Job Token Filtering Configuration

**Created:** 2025-11-10
**Status:** Planned
**Priority:** High
**Estimated Effort:** 8-12 hours
**Related:** [Token_And_Feature_Filtering.md](../docs/Token_And_Feature_Filtering.md), [INTEGRATION_PLAN_Filtering_Stages_1_and_2.md](INTEGRATION_PLAN_Filtering_Stages_1_and_2.md)

---

## Problem Statement

Currently, token filtering is configured via environment variables (global settings). This approach has several limitations:

1. **Inflexible**: All jobs use the same filtering settings
2. **Not User-Friendly**: Requires editing `.env` file and restarting services
3. **No Experimentation**: Can't A/B test different filtering strategies
4. **No Template Support**: Can't save filtering preferences in templates
5. **No Per-Job Control**: Can't adjust filtering based on dataset characteristics

---

## Proposed Solution

Make token filtering a **per-job configuration** saved in templates, similar to how we handle other hyperparameters.

### Affected Job Types

1. **Dataset Tokenization Jobs** (Stage 1: Sample-level filtering)
2. **Feature Extraction Jobs** (Stage 2: Token-level filtering)
3. ~~Labeling Jobs~~ (Stage 3 already implemented, no changes needed)

---

## Design Specifications

### 1. Dataset Tokenization Filtering Configuration

**Template Addition:**
- Add "Filtering" section to Dataset Templates
- Show when creating/editing dataset tokenization jobs

**UI Fields:**
```
┌─ Filtering Settings ────────────────────────────────────┐
│                                                          │
│  [x] Enable Sample Filtering                         ⓘ  │
│      Filter out samples with too many junk tokens       │
│                                                          │
│  Filter Mode:                                           │
│  ( ) Minimal     - Only control characters           ⓘ  │
│  (•) Conservative - + whitespace-only tokens         ⓘ  │
│                                                          │
│  Junk Ratio Threshold: [0.70] (0.0 - 1.0)           ⓘ  │
│  Skip samples if >X% of tokens are junk                 │
│                                                          │
│  ⚠️  Warning: Filtering is permanent for tokenized data │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Tooltip Specifications:**

1. **Enable Sample Filtering (ⓘ):**
   ```
   Sample-Level Filtering

   Analyzes each text sample during tokenization and permanently removes
   samples that contain too many junk tokens.

   What gets filtered:
   • Samples that are mostly punctuation (e.g., "!!! ??? ...")
   • Samples that are mostly whitespace or empty
   • Samples with >70% single-character tokens
   • Samples with control characters (null bytes, etc.)

   ⚠️ Warning: This is PERMANENT. Filtered samples will not be included
   in your tokenized dataset and cannot be recovered.

   Use when:
   ✓ You want zero-tolerance for junk data
   ✓ Your dataset is large enough to afford sample loss (>10K samples)
   ✓ You've inspected your data and know it contains junk

   Skip when:
   ✗ This is your first time using filtering
   ✗ You have a small dataset (<1K samples)
   ✗ You're unsure about data quality
   ```

2. **Minimal Mode (ⓘ):**
   ```
   Minimal Filtering Mode

   Only removes samples with control characters and null bytes.
   This is the safest option.

   Filters:
   • Null bytes (\0)
   • Control characters (ASCII codes 0-31, except newline/tab)

   Does NOT filter:
   • Whitespace (spaces, tabs, newlines)
   • Punctuation (periods, commas, quotes, etc.)
   • Single characters (a, b, c, etc.)
   • Any normal text

   Example: "Hello\0World" → FILTERED
   Example: "!!! ??? ..." → KEPT
   Example: "   " (just spaces) → KEPT

   Typical filtering rate: <1% of samples
   ```

3. **Conservative Mode (ⓘ):**
   ```
   Conservative Filtering Mode (Recommended)

   Removes samples with control characters AND samples that are
   entirely whitespace or empty.

   Filters everything from Minimal, plus:
   • Whitespace-only samples ("   ", "\n\n\n", "\t\t")
   • Empty samples after trimming

   Does NOT filter:
   • Punctuation (periods, commas, quotes, etc.)
   • Single characters (a, b, c, etc.)
   • Normal text with spaces

   Example: "Hello World" → KEPT
   Example: "!!! ??? ..." → KEPT
   Example: "   " (just spaces) → FILTERED
   Example: "\n\n\n" (just newlines) → FILTERED

   Typical filtering rate: 1-5% of samples
   ```

4. **Junk Ratio Threshold (ⓘ):**
   ```
   Junk Ratio Threshold

   A sample is filtered if MORE than this percentage of its tokens
   are classified as junk according to the selected filter mode.

   How it works:
   1. Tokenize the sample into tokens
   2. Count how many tokens are "junk" (based on filter mode)
   3. Calculate junk_ratio = junk_tokens / total_tokens
   4. If junk_ratio > threshold → filter the sample

   Examples (Conservative mode, threshold = 0.7):

   Sample: "Hello world"
   → Tokens: ["Hello", " ", "world"]
   → Junk: [" "] (1 out of 3 = 33%)
   → Result: KEPT (33% < 70%)

   Sample: "   !!! ??? ..."
   → Tokens: [" ", " ", " ", "!", "!", "!", " ", "?", "?", "?", " ", ".", ".", "."]
   → Junk: all whitespace (50%)
   → Result: KEPT (50% < 70%)

   Sample: "\n\n\n\n\n\n\n\n\n\n\n"
   → Tokens: all whitespace (100%)
   → Result: FILTERED (100% > 70%)

   Recommended values:
   • 0.5 (50%) - Aggressive filtering
   • 0.7 (70%) - Balanced (default)
   • 0.9 (90%) - Very conservative
   ```

**Database Schema Changes:**
```sql
-- Add to datasets table or tokenization_configs table
ALTER TABLE datasets ADD COLUMN tokenization_filter_enabled BOOLEAN DEFAULT false;
ALTER TABLE datasets ADD COLUMN tokenization_filter_mode VARCHAR(20) DEFAULT 'conservative';
ALTER TABLE datasets ADD COLUMN tokenization_junk_ratio_threshold FLOAT DEFAULT 0.7;
```

**Backend Changes:**
- Add fields to `DatasetCreate` Pydantic schema
- Pass configuration to `tokenize_dataset_task()` from request payload (not from settings)
- Store configuration in database for reference

**Frontend Changes:**
- Add filtering section to Dataset panel's tokenization form
- Add filtering fields to Dataset Templates
- Add "Filtering Stats" display in completed tokenization jobs (samples filtered count)

---

### 2. Feature Extraction Filtering Configuration

**Template Addition:**
- Add "Token Filtering" section to Extraction Templates
- Show when creating/editing extraction jobs

**UI Fields:**
```
┌─ Token Filtering Settings ──────────────────────────────┐
│                                                          │
│  [x] Enable Token Filtering                          ⓘ  │
│      Filter junk tokens before SAE feature extraction   │
│                                                          │
│  Filter Mode:                                           │
│  ( ) Minimal     - Only control characters           ⓘ  │
│  ( ) Conservative - + whitespace-only                ⓘ  │
│  (•) Standard    - + punctuation, single chars       ⓘ  │
│  ( ) Aggressive  - + short tokens, low-entropy       ⓘ  │
│                                                          │
│  ℹ️  Helps SAE focus on semantic features               │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Tooltip Specifications:**

1. **Enable Token Filtering (ⓘ):**
   ```
   Token-Level Filtering During Feature Extraction

   Filters junk tokens BEFORE they are fed into the SAE model during
   feature extraction. This helps the SAE focus on learning semantic
   features rather than punctuation or whitespace patterns.

   What gets filtered:
   • Punctuation-only tokens (periods, commas, quotes, etc.)
   • Whitespace-only tokens (spaces, tabs, newlines)
   • Single non-alphanumeric characters
   • Control characters
   • Optionally: short tokens, low-entropy tokens (aggressive mode)

   ⚠️ Important: This affects SAE training quality. Filtered tokens
   do not contribute to feature activations.

   Use when:
   ✓ You want SAE to focus on semantic/syntactic features
   ✓ Junk tokens are creating noisy feature activations
   ✓ You're analyzing natural language (not code/structured data)

   Skip when:
   ✗ You're analyzing code (punctuation matters for syntax)
   ✗ You want features for ALL token types
   ✗ This is your first extraction (try without filtering first)
   ```

2. **Minimal Mode (ⓘ):**
   ```
   Minimal Filtering Mode

   Only filters control characters and null bytes.
   Almost all normal tokens are kept.

   Filters:
   • Null bytes (\0)
   • Control characters (ASCII codes 0-31, except newline/tab)

   Does NOT filter:
   • Whitespace (spaces, tabs, newlines)
   • Punctuation (periods, commas, quotes, etc.)
   • Single characters (a, b, c, etc.)
   • Any normal text

   Example tokens:
   "Hello" → KEPT
   "," → KEPT
   " " (space) → KEPT
   "a" → KEPT
   "\0" (null) → FILTERED

   Typical filtering rate: <1% of tokens
   Use when: You want almost all tokens included
   ```

3. **Conservative Mode (ⓘ):**
   ```
   Conservative Filtering Mode

   Filters control characters and whitespace-only tokens.
   Good balance for most use cases.

   Filters everything from Minimal, plus:
   • Whitespace-only tokens (" ", "\n", "\t")

   Does NOT filter:
   • Punctuation (periods, commas, quotes, etc.)
   • Single characters (a, b, c, etc.)
   • Normal text tokens

   Example tokens:
   "Hello" → KEPT
   "," → KEPT
   " " (space) → FILTERED
   "a" → KEPT
   "world" → KEPT

   Typical filtering rate: 5-15% of tokens
   Use when: You want SAE to learn word/punctuation patterns
   ```

4. **Standard Mode (ⓘ - Recommended):**
   ```
   Standard Filtering Mode (Recommended)

   Filters whitespace, punctuation, and single non-alphanumeric
   characters. Good for semantic feature extraction.

   Filters everything from Conservative, plus:
   • Pure punctuation tokens (".", ",", "!", "?", etc.)
   • Single non-alphanumeric characters ("@", "#", "$", etc.)

   Does NOT filter:
   • Multi-character tokens ("Hello", "world", etc.)
   • Single alphanumeric characters ("a", "1", etc.)
   • Words with punctuation ("don't", "3.14")

   Example tokens:
   "Hello" → KEPT
   "," → FILTERED
   " " (space) → FILTERED
   "a" → KEPT
   "." → FILTERED
   "3.14" → KEPT (multi-char)
   "don't" → KEPT (contains letters)

   Typical filtering rate: 15-30% of tokens
   Use when: You want SAE to focus on words and meaning
   ```

5. **Aggressive Mode (ⓘ):**
   ```
   Aggressive Filtering Mode

   Filters aggressively to keep only substantial semantic tokens.
   Use with caution - may filter too much.

   Filters everything from Standard, plus:
   • Very short tokens (1-2 characters) that are mostly non-alphanumeric
   • Low-entropy tokens (repeated characters)

   Example tokens:
   "Hello" → KEPT
   "," → FILTERED
   " " (space) → FILTERED
   "a" → KEPT (alphanumeric)
   "." → FILTERED
   ".." → FILTERED (short, non-alnum)
   "!!!" → FILTERED (low entropy)
   "OK" → KEPT (2 chars, alphanumeric)
   "@#" → FILTERED (short, non-alnum)

   Typical filtering rate: 30-50% of tokens
   Use when: You want only substantial words for semantic analysis
   Warning: May be too aggressive for most use cases
   ```

**Database Schema Changes:**
```sql
-- Add to extractions table or extraction_configs table
ALTER TABLE extractions ADD COLUMN extraction_filter_enabled BOOLEAN DEFAULT false;
ALTER TABLE extractions ADD COLUMN extraction_filter_mode VARCHAR(20) DEFAULT 'standard';
```

**Backend Changes:**
- Add fields to `ExtractionCreate` Pydantic schema
- Pass configuration to extraction worker from request payload (not from settings)
- Store configuration in database for reference

**Frontend Changes:**
- Add filtering section to Feature Discovery panel's extraction form
- Add filtering fields to Extraction Templates
- Add "Filtering Stats" display in completed extractions (tokens filtered count)

---

### 3. Stage 3 (Pre-Labeling Filter) - No Changes Needed

Stage 3 already works at the feature level and is enabled by default. It doesn't need per-job configuration because:
- It's reversible (just marks features, doesn't delete data)
- It's cost-optimization focused (not quality-focused)
- Global settings are appropriate for this use case

**Keep existing behavior:**
- Controlled by environment variables (`PRE_LABELING_FILTER_ENABLED`, etc.)
- Applied automatically during labeling
- No UI configuration needed

---

## Implementation Plan

### Phase 1: Database Schema & Backend API (3-4 hours)

#### Step 1.1: Add Database Columns
- [ ] Create migration for datasets table (tokenization filter fields)
- [ ] Create migration for extractions table (extraction filter fields)
- [ ] Run migrations
- [ ] Verify columns exist

#### Step 1.2: Update Pydantic Schemas
- [ ] Add filter fields to `DatasetCreate` schema
- [ ] Add filter fields to `DatasetResponse` schema
- [ ] Add filter fields to `ExtractionCreate` schema
- [ ] Add filter fields to `ExtractionResponse` schema

#### Step 1.3: Update Service Layer
- [ ] Modify `tokenize_dataset_task()` to accept filter config from payload
- [ ] Remove reliance on `settings.tokenization_filter_*` in worker
- [ ] Modify extraction worker to accept filter config from payload
- [ ] Remove reliance on `settings.extraction_filter_*` in worker
- [ ] Store filter config in database after job completion

---

### Phase 2: Frontend UI (4-5 hours)

#### Step 2.1: Dataset Tokenization Form
- [ ] Add "Filtering Settings" collapsible section to DatasetPanel
- [ ] Add enable/disable checkbox
- [ ] Add filter mode radio buttons (minimal/conservative)
- [ ] Add junk ratio threshold slider (0.0 - 1.0)
- [ ] Add warning message about permanence
- [ ] Add tooltip explanations for each option

#### Step 2.2: Extraction Form
- [ ] Add "Token Filtering Settings" collapsible section to Feature Discovery panel
- [ ] Add enable/disable checkbox
- [ ] Add filter mode radio buttons (minimal/conservative/standard/aggressive)
- [ ] Add tooltip explanations for each mode

#### Step 2.3: Template Support
- [ ] Add filter fields to Dataset Template form
- [ ] Add filter fields to Dataset Template card display
- [ ] Add filter fields to Extraction Template form
- [ ] Add filter fields to Extraction Template card display
- [ ] Test save/load/apply template functionality

#### Step 2.4: Statistics Display
- [ ] Add "Filtering Stats" section to completed tokenization job display
  - Show: "X samples kept, Y filtered (Z%)"
- [ ] Add "Filtering Stats" section to completed extraction job display
  - Show: "X tokens kept, Y filtered (Z%)"

---

### Phase 3: Testing & Documentation (1-2 hours)

#### Step 3.1: Functional Testing
- [ ] Test tokenization with filtering enabled/disabled
- [ ] Test extraction with each filter mode
- [ ] Test template save/load with filter settings
- [ ] Verify statistics display correctly
- [ ] Verify database stores configuration

#### Step 3.2: Documentation Updates
- [ ] Update `Token_And_Feature_Filtering.md` with per-job configuration instructions
- [ ] Update user-facing documentation
- [ ] Add configuration examples
- [ ] Update integration plan document

---

## Database Migration Templates

### Migration 1: Dataset Tokenization Filter Fields

```python
"""add_tokenization_filter_fields

Revision ID: xxx
Revises: yyy
Create Date: 2025-11-10

"""
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column('datasets', sa.Column('tokenization_filter_enabled', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('datasets', sa.Column('tokenization_filter_mode', sa.String(20), nullable=False, server_default='conservative'))
    op.add_column('datasets', sa.Column('tokenization_junk_ratio_threshold', sa.Float(), nullable=False, server_default='0.7'))

def downgrade():
    op.drop_column('datasets', 'tokenization_junk_ratio_threshold')
    op.drop_column('datasets', 'tokenization_filter_mode')
    op.drop_column('datasets', 'tokenization_filter_enabled')
```

### Migration 2: Extraction Filter Fields

```python
"""add_extraction_filter_fields

Revision ID: xxx
Revises: yyy
Create Date: 2025-11-10

"""
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column('extractions', sa.Column('extraction_filter_enabled', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('extractions', sa.Column('extraction_filter_mode', sa.String(20), nullable=False, server_default='standard'))

def downgrade():
    op.drop_column('extractions', 'extraction_filter_mode')
    op.drop_column('extractions', 'extraction_filter_enabled')
```

---

## UI Mockup Examples

### Dataset Tokenization Form - Filtering Section

```
┌─ Tokenization Settings ─────────────────────────────────┐
│                                                          │
│  Model: [GPT-2                                    ▼]    │
│  Max Length: [512                                  ]    │
│  Truncation: [x] Enabled                               │
│                                                          │
│  ▼ Advanced Settings                                    │
│    ...                                                  │
│                                                          │
│  ▼ Filtering Settings                                   │
│                                                          │
│    [x] Enable Sample Filtering                          │
│        Remove samples with mostly junk tokens           │
│                                                          │
│    Filter Mode:                                         │
│    ( ) Minimal     - Only control characters   ⓘ       │
│    (•) Conservative - + whitespace tokens      ⓘ       │
│                                                          │
│    Junk Ratio Threshold: [━━━━━●━━━] 0.70              │
│    Skip samples if >70% of tokens are junk              │
│                                                          │
│    ⚠️  Filtering is permanent - filtered samples        │
│       will not be included in tokenized dataset         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Extraction Form - Token Filtering Section

```
┌─ Extraction Settings ────────────────────────────────────┐
│                                                          │
│  Training: [train_20251109_183840                ▼]    │
│  Top-K Features: [100                              ]    │
│                                                          │
│  ▼ Token Filtering Settings                             │
│                                                          │
│    [x] Enable Token Filtering                           │
│        Filter junk tokens during extraction             │
│                                                          │
│    Filter Mode:                                         │
│    ( ) Minimal     - Control chars only        ⓘ       │
│    ( ) Conservative - + Whitespace             ⓘ       │
│    (•) Standard    - + Punctuation, singles    ⓘ       │
│    ( ) Aggressive  - + Short, low-entropy      ⓘ       │
│                                                          │
│    ℹ️  Helps SAE focus on semantic features             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Benefits

### User Experience
- **Intuitive Configuration**: All settings in one place in UI
- **Experimentation Friendly**: Easy to try different filtering strategies
- **Template Support**: Save proven configurations for reuse
- **Visual Feedback**: See filtering statistics in job results

### Technical Benefits
- **No Service Restarts**: Changes take effect immediately
- **Per-Job Flexibility**: Different datasets/models may need different filtering
- **Audit Trail**: Database stores what filtering was used for each job
- **A/B Testing**: Easy to compare filtered vs unfiltered results

### Consistency
- **Matches Existing Patterns**: Uses same template/configuration approach as other features
- **Follows UX Standards**: Similar to how we handle training hyperparameters

---

## Open Questions

1. **Default Values**: What should be the default for new users?
   - Suggestion: Disabled by default with prominent "Enable Filtering" checkbox
   - Users opt-in after understanding the trade-offs

2. **Template Migration**: Should we add default filter settings to existing templates?
   - Suggestion: Add with sensible defaults (disabled) to avoid breaking changes

3. **Statistics Storage**: Should we store filtering statistics in database or just show in logs?
   - Suggestion: Store basic stats (samples_filtered_count, tokens_filtered_count) in database

4. **Validation**: Should we validate that filter settings are compatible with dataset characteristics?
   - Suggestion: Add warnings if threshold is too aggressive (>0.9 or <0.3)

---

## Success Criteria

- [ ] Users can configure filtering per-job without editing `.env`
- [ ] Filtering settings are saved in templates
- [ ] UI provides clear explanations of each filter mode
- [ ] Statistics show filtering effectiveness
- [ ] No service restarts required for configuration changes
- [ ] Existing jobs continue to work (backward compatible)
- [ ] Documentation clearly explains filtering options

---

## Related Files

**Backend:**
- `backend/src/schemas/dataset.py` - Add filter fields to DatasetCreate/Response
- `backend/src/schemas/extraction.py` - Add filter fields to ExtractionCreate/Response
- `backend/src/services/tokenization_service.py` - Accept filter config from payload
- `backend/src/workers/extraction_tasks.py` - Accept filter config from payload
- `backend/src/models/dataset.py` - Add filter columns to Dataset model
- `backend/src/models/extraction.py` - Add filter columns to Extraction model

**Frontend:**
- `frontend/src/components/panels/DatasetPanel.tsx` - Add filtering UI
- `frontend/src/components/panels/FeatureDiscoveryPanel.tsx` - Add filtering UI
- `frontend/src/components/templates/DatasetTemplateForm.tsx` - Add filter fields
- `frontend/src/components/templates/ExtractionTemplateForm.tsx` - Add filter fields
- `frontend/src/types/dataset.ts` - Add filter types
- `frontend/src/types/extraction.ts` - Add filter types

**Database:**
- Create migrations for datasets and extractions tables

**Documentation:**
- `0xcc/docs/Token_And_Feature_Filtering.md` - Update configuration section

---

**Document Version:** 1.0
**Next Steps:** Review and approve design, then proceed with Phase 1 implementation
