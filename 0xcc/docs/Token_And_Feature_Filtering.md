# Token and Feature Filtering System

**Created:** 2025-11-10
**Updated:** 2025-11-10
**Status:** Three-Stage Architecture (Pre-Labeling Active, Tokenization & Extraction Pending Integration)
**Purpose:** Zero-tolerance junk filtering at dataset, extraction, and labeling stages

---

## Overview

The filtering system provides **three-stage filtering** to prevent junk tokens and features from entering the SAE training and labeling pipeline:

1. **Stage 1: Dataset Tokenization Filter** (Optional, Sample-Level, Permanent)
   - Filters entire samples if >70% tokens are junk during tokenization
   - **Status:** Configuration ready, integration pending
   - **Impact:** Prevents junk data from being stored in dataset
   - Use for zero-tolerance filtering at the source

2. **Stage 2: Extraction Filter** (Optional, Token-Level, Training-Impact)
   - Filters junk tokens before SAE feature extraction
   - **Status:** Configuration ready, integration pending
   - **Impact:** Prevents junk tokens from affecting SAE training
   - Use to ensure clean data for feature learning

3. **Stage 3: Pre-Labeling Feature Filter** (Enabled by default, Feature-Level, Reversible)
   - Filters features before sending to LLM
   - **Status:** ✅ IMPLEMENTED and ENABLED by default
   - **Impact:** Saves API costs by skipping junk features (~$0.50-1.00 per extraction)
   - Already active and working

---

## Stage 1: Dataset Tokenization Filter (CONFIGURATION READY)

### Purpose
Remove entire samples that contain too many junk tokens during dataset tokenization. This is the most aggressive filtering option - it permanently prevents junk data from entering your datasets.

### Status
**Configuration:** ✅ Complete
**Integration:** ⏳ Pending (next session)
**Default:** Disabled (opt-in for zero-tolerance mode)

### How It Works

**Sample-Level Analysis:**
```python
# For each sample during tokenization:
1. Tokenize the text into token IDs
2. Analyze the token sequence for junk ratio
3. If >70% of tokens are junk → SKIP entire sample
4. Otherwise → Keep sample in dataset
```

**What Gets Filtered:**
- Samples that are mostly punctuation (e.g., "!!! ??? ...")
- Samples that are mostly whitespace
- Samples with >70% single-character tokens
- Samples with >70% control characters

### Configuration

**Environment Variables** (`.env`):
```bash
# Enable tokenization-time filtering (USE WITH CAUTION - permanent!)
TOKENIZATION_FILTER_ENABLED=false  # Default: false

# Filter mode
TOKENIZATION_FILTER_MODE=conservative  # Options: minimal, conservative

# Junk ratio threshold (skip if >X% tokens are junk)
TOKENIZATION_JUNK_RATIO_THRESHOLD=0.7  # Default: 0.7 (70%)
```

### When to Use

✅ **Enable when:**
- You want zero-tolerance for junk data in training
- You're okay with losing some samples permanently
- Your dataset is large enough to afford sample loss
- You want smaller, cleaner datasets

❌ **Keep disabled when:**
- This is your first time using filtering (try Stage 3 first)
- You have a small dataset
- You're unsure about token quality
- You want reversible filtering

---

## Stage 2: Extraction Filter (CONFIGURATION READY)

### Purpose
Filter junk tokens before they're fed into SAE feature extraction. This prevents junk tokens from affecting SAE training and feature learning.

### Status
**Configuration:** ✅ Complete
**Integration:** ⏳ Pending (next session)
**Default:** Disabled (opt-in for extraction-time filtering)

### How It Works

**Token-Level Filtering:**
```python
# During feature extraction:
1. Load tokenized samples from dataset
2. For each token in sequence:
   - Analyze if token is junk (punctuation, whitespace, single-char, etc.)
   - Skip junk tokens during SAE forward pass
3. Only meaningful tokens contribute to feature activations
```

**What Gets Filtered:**
- All punctuation-only tokens
- All whitespace-only tokens
- Single non-alphanumeric characters
- Control characters
- Optionally: short tokens, low-entropy tokens (aggressive mode)

### Configuration

**Environment Variables** (`.env`):
```bash
# Enable extraction-time filtering
EXTRACTION_FILTER_ENABLED=false  # Default: false

# Filter mode (how aggressive to be)
EXTRACTION_FILTER_MODE=standard  # Options: minimal, conservative, standard, aggressive
```

**Filter Modes:**
- `minimal`: Only control characters
- `conservative`: + whitespace-only tokens
- `standard`: + punctuation, single non-alphanumeric chars
- `aggressive`: + short tokens (1-2 chars), low-entropy tokens

### When to Use

✅ **Enable when:**
- You want SAE to focus on semantic features
- Junk tokens are affecting feature quality
- You want cleaner feature activations
- You've already tokenized datasets and don't want to re-process

❌ **Keep disabled when:**
- You want features to learn punctuation patterns (rare)
- You're analyzing code/syntax (punctuation matters)
- First time using filtering (try Stage 3 first)

---

## Stage 3: Pre-Labeling Feature Filter (ACTIVE)

### Purpose
Skip features that activate primarily on punctuation, whitespace, or single characters - these waste API calls and clutter results with low-value labels.

### How It Works

**Filter Logic:**
```python
# Feature is marked as junk if:
- 100% of top-10 tokens are whitespace
- >80% of top-10 tokens are punctuation/whitespace
- >70% of top-10 tokens are single characters
```

**What Happens to Filtered Features:**
- Marked with `name="unlabeled_junk"`, `category="system"`, `label_source="filter"`
- NOT sent to LLM (saves API costs)
- Still visible in database (can be re-labeled later if needed)

### Configuration

**Environment Variables** (`.env`):
```bash
# Enable/disable pre-labeling filter
PRE_LABELING_FILTER_ENABLED=true  # Default: true

# Thresholds (0.0-1.0)
PRE_LABELING_JUNK_RATIO_THRESHOLD=0.8      # Skip if >80% tokens are junk
PRE_LABELING_SINGLE_CHAR_THRESHOLD=0.7     # Skip if >70% tokens are single char
```

**Example Usage:**
```python
# In labeling_service.py (automatic)
if settings.pre_labeling_filter_enabled:
    feature_filter = FeatureFilter(
        junk_ratio_threshold=0.8,
        single_char_ratio_threshold=0.7
    )

    # Features like these get filtered:
    # - Feature #42: top tokens = [",", ".", ";", "!", "?", ...]
    # - Feature #103: top tokens = [" ", "\n", "\t", ...]
    # - Feature #215: top tokens = ["a", "b", "c", "d", "e", ...]
```

### Expected Results

**Filtering Statistics:**
- Typical: 20-40% of features filtered as junk
- Example: 16,384 features → ~10,000 sent to LLM, ~6,000 skipped
- **Cost Savings:** ~$0.60 per extraction (at $0.0001/feature with GPT-4o-mini)

**Logs:**
```
INFO: Pre-labeling filter enabled - analyzing 16384 features
INFO: Pre-labeling filter: 10245 features to label, 6139 junk features skipped (37.5% filtered)
INFO: Labeling 10245 features for extraction extr_20251109_183840_train_55
```

---

## Integration Status Summary

| Stage | Configuration | Implementation | Integration | Status |
|-------|--------------|----------------|-------------|--------|
| **Stage 1: Tokenization** | ✅ Complete | ✅ `is_junk_sequence()` added | ⏳ Pending | Disabled by default |
| **Stage 2: Extraction** | ✅ Complete | ⏳ Pending | ⏳ Pending | Disabled by default |
| **Stage 3: Pre-Labeling** | ✅ Complete | ✅ Complete | ✅ Complete | **ACTIVE** ✅ |

---

## Recommended Usage Strategy

### For First-Time Users:
1. **Start with Stage 3 only** (already enabled)
   - Monitor filtering statistics in logs
   - Check filtered features in database
   - Adjust thresholds if needed

2. **Add Stage 2 if needed** (next session)
   - Enable if junk features persist
   - Use `standard` mode initially
   - Monitor SAE training quality

3. **Add Stage 1 cautiously** (future)
   - Only if you need permanent filtering
   - Test on small dataset first
   - Verify sample quality before full deployment

### For Advanced Users:
- Enable all three stages for maximum junk elimination
- Use `aggressive` mode for Stage 2 if analyzing natural language only
- Adjust thresholds based on your specific dataset characteristics

---

## Legacy Documentation: Original Two-Stage Plan (SUPERSEDED)

_This section documents the original two-stage architecture for reference._

### ~~Stage 1: Tokenization Filter (OPTIONAL)~~
**Note:** This has been redesigned as the three-stage architecture described above.

### Purpose
Remove junk tokens during dataset tokenization (permanent filtering).

### Status
**Implemented but DISABLED by default** - too aggressive for most use cases.

### Configuration

```bash
# Enable tokenization-time filtering (USE WITH CAUTION)
TOKENIZATION_FILTER_ENABLED=false  # Default: false

# Filter mode
TOKENIZATION_FILTER_MODE=conservative  # Options: minimal, conservative
```

**Filter Modes:**
- `minimal`: Only control characters (null bytes, etc.)
- `conservative`: + whitespace-only tokens

### When to Use
- You're 100% certain some tokens are never useful
- You want smaller dataset files
- You don't mind permanent data loss

### When NOT to Use
- Default case - keep tokenization filter OFF
- Even "junk" tokens can represent meaningful patterns
- Pre-labeling filter is safer (reversible)

---

## Filter Classes

### TokenFilter
**Location:** `backend/src/utils/token_filter.py`

**Modes:**
- `minimal`: Control chars only
- `conservative`: + whitespace (for tokenization)
- `standard`: + punctuation, single chars (for labeling)
- `aggressive`: + short tokens, low entropy

**Usage:**
```python
from src.utils.token_filter import TokenFilter, FilterMode

# For tokenization (conservative)
filter = TokenFilter(mode=FilterMode.CONSERVATIVE)
filtered_tokens = filter.filter_token_list(tokens)

# For labeling (standard)
filter = TokenFilter(mode=FilterMode.STANDARD)
clean_stats = filter.filter_token_stats(token_stats)
```

### FeatureFilter
**Location:** `backend/src/utils/token_filter.py`

**Purpose:** Identify features likely to be labeled as junk

**Usage:**
```python
from src.utils.token_filter import FeatureFilter

filter = FeatureFilter(
    junk_ratio_threshold=0.8,
    single_char_ratio_threshold=0.7
)

is_junk = filter.is_junk_feature(token_stats)
```

---

## Integration Points

### Pre-Labeling Filter Integration
**File:** `backend/src/services/labeling_service.py` (lines 265-315)

```python
# Automatic integration (no code changes needed)
# 1. Fetch all features
# 2. If pre_labeling_filter_enabled:
#    - Aggregate token stats
#    - Filter junk features
#    - Mark as "unlabeled_junk"
# 3. Label remaining features
```

### Tokenization Filter Integration
**Status:** Not yet integrated (feature implemented, integration pending)

**Planned Location:** `backend/src/services/dataset_service.py`

Will filter tokens during:
- HuggingFace dataset downloads
- Local dataset ingestion
- Dataset tokenization

---

## Monitoring and Debugging

### Check Filter Status
```bash
# View logs during labeling
tail -f backend/celery_start.log | grep -i filter

# Expected output:
# INFO: Pre-labeling filter enabled - analyzing 16384 features
# INFO: Pre-labeling filter: 10245 features to label, 6139 junk features skipped (37.5% filtered)
```

### Query Filtered Features
```sql
-- Count filtered features
SELECT COUNT(*) FROM features
WHERE label_source = 'filter' AND name = 'unlabeled_junk';

-- View filtered features
SELECT neuron_index, name, category, label_source
FROM features
WHERE label_source = 'filter'
ORDER BY neuron_index
LIMIT 10;
```

### Disable Filter (If Needed)
```bash
# In .env file
PRE_LABELING_FILTER_ENABLED=false

# Restart backend
./stop-mistudio.sh
./start-mistudio.sh
```

---

## Future Enhancements

### Short-Term (Next Labeling Job)
- ✅ Pre-labeling filter (DONE - enabled by default)
- Monitor filtering statistics to tune thresholds
- Possibly adjust thresholds based on results

### Medium-Term
- Tokenization filter integration into dataset service
- UI toggle for enabling/disabling pre-labeling filter
- Frontend display of filtering statistics

### Long-Term
- Machine learning-based junk detection
- Custom filter rules per model/dataset
- Re-labeling interface for filtered features

---

## FAQ

**Q: Will filtered features be completely deleted?**
A: No. They're marked as `unlabeled_junk` but remain in database. You can re-label them later if needed.

**Q: What if the filter is too aggressive?**
A: Adjust thresholds in `.env`:
```bash
PRE_LABELING_JUNK_RATIO_THRESHOLD=0.9    # More conservative (filter less)
PRE_LABELING_SINGLE_CHAR_THRESHOLD=0.8
```

**Q: Can I disable the filter entirely?**
A: Yes, set `PRE_LABELING_FILTER_ENABLED=false` in `.env`

**Q: How much does this save?**
A: Typically 20-40% of features filtered = ~$0.50-1.00 saved per extraction with OpenAI

**Q: Does this affect SAE training?**
A: No. Filtering only affects labeling, not training or feature extraction.

**Q: When should I use tokenization-time filtering?**
A: Rarely. Only if you're certain tokens are never useful AND you want smaller datasets. Pre-labeling filter is safer.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-10
**Related Files:**
- `backend/src/utils/token_filter.py` - Filter implementation
- `backend/src/services/labeling_service.py` - Pre-labeling integration
- `backend/src/core/config.py` - Configuration settings
