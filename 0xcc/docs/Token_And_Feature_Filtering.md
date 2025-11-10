# Token and Feature Filtering System

**Created:** 2025-11-10
**Status:** Implemented (Pre-Labeling Filter Active)
**Purpose:** Reduce API costs and focus on semantic features by filtering junk tokens and features

---

## Overview

The filtering system provides two-stage filtering to improve feature labeling quality and reduce costs:

1. **Stage 1: Tokenization Filter** (Optional, Conservative, Permanent)
   - Filters tokens during dataset creation
   - **Status:** Implemented but disabled by default
   - Use when you know certain tokens are never useful

2. **Stage 2: Pre-Labeling Feature Filter** (Enabled by default, Aggressive, Reversible)
   - Filters features before sending to LLM
   - **Status:** Implemented and ENABLED by default
   - Saves API costs by skipping junk features

---

## Stage 2: Pre-Labeling Feature Filter (ACTIVE)

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

## Stage 1: Tokenization Filter (OPTIONAL)

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
