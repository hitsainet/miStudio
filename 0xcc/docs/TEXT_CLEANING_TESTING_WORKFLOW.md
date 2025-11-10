# Text Cleaning Testing Workflow

**Purpose:** Validate that text cleaning is working correctly at each stage of the pipeline.

**Validation Levels:** 5 checkpoints from earliest (unit test) to latest (feature quality)

---

## Overview of Validation Timeline

```
1. Unit Test (30 seconds)
   ↓
2. Tokenization Logs (during tokenization)
   ↓
3. Tokenized Sample Inspection (after tokenization completes)
   ↓
4. SAE Training (hours)
   ↓
5. Feature Correlation Analysis (after training)
```

---

## Validation 1: Unit Test (EARLIEST - 30 seconds)

**When:** Before any dataset work, validates cleaning logic in isolation

**Command:**
```bash
cd /home/x-sean/app/miStudio/backend
PYTHONPATH=. venv/bin/python3 test_text_cleaning.py
```

**Expected Output:**
```
================================================================================
TEXT CLEANING DEMONSTRATION
================================================================================

Standard Cleaner:
--------------------------------------------------------------------------------
1. Input:  'This is normal text.'
   Output: 'This is normal text.'

2. Input:  '<html><body>HTML text with tags</body></html>'
   Output: 'HTML text with tags'

3. Input:  'Text with &nbsp; HTML &lt;entities&gt;'
   Output: 'Text with HTML entities'

4. Input:  'Text with URL https://example.com/path'
   Output: 'Text with URL https://example.com/path'

5. Input:  'Email: user@example.com in text'
   Output: 'Email: user@example.com in text'

6. Input:  'Control\x00chars\x01\x02\x03\x04'
   Output: 'Control chars'

7. Input:  'Excessive!!!!!!! punctuation??????'
   Output: 'Excessive!!! punctuation???'

8. Input:  '    Too    much    whitespace    '
   Output: 'Too much whitespace'

9. Input:  "<?xml version='1.0'?><doc>XML content</doc>"
   Output: 'XML content'

10. Input:  'Short'
    Output: FILTERED OUT

11. Input:  ''
    Output: FILTERED OUT

Aggressive Cleaner:
--------------------------------------------------------------------------------
... (shows more aggressive cleaning with URLs/emails removed)

Minimal Cleaner:
--------------------------------------------------------------------------------
... (shows minimal cleaning with only control chars removed)
```

**✅ Validation Criteria:**
- [ ] HTML tags removed (input #2)
- [ ] HTML entities removed (input #3)
- [ ] Control characters removed (input #6)
- [ ] Excessive punctuation limited to 3 (input #7)
- [ ] Whitespace normalized (input #8)
- [ ] Short texts filtered out (input #10)
- [ ] Empty texts filtered out (input #11)

**Time:** ~30 seconds
**Result:** Confirms TextCleaner class works correctly in isolation

---

## Validation 2: Tokenization Logs (REAL-TIME)

**When:** During dataset tokenization, confirms cleaning is active in production

**Steps:**

### 2.1 Start Monitoring Celery Logs
```bash
# Terminal 1: Monitor Celery worker logs
tail -f /tmp/celery_worker.log | grep -E "(text cleaner|tokeniz|progress)"
```

### 2.2 Trigger Tokenization

**Option A: Via UI**
1. Open miStudio: http://mistudio.mcslab.io
2. Go to "Datasets" panel
3. Find a downloaded dataset (or download one first)
4. Click "Tokenize" button
5. Select tokenizer (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
6. Click "Start Tokenization"

**Option B: Via API**
```bash
# Get dataset ID
curl -s http://localhost:8000/api/v1/datasets | jq '.data[0].id'

# Trigger tokenization (replace DATASET_ID)
curl -X POST http://localhost:8000/api/v1/datasets/DATASET_ID/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "tokenizer_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "max_length": 512,
    "stride": 0,
    "padding": "max_length",
    "truncation": "longest_first",
    "add_special_tokens": true,
    "return_attention_mask": true
  }'
```

### 2.3 Watch for Log Message

**Expected Log Output:**
```
[2025-11-02 12:34:56,789: INFO/ForkPoolWorker-1] Using standard text cleaner for preprocessing
[2025-11-02 12:34:56,790: INFO/ForkPoolWorker-1] Dataset schema analysis:
[2025-11-02 12:34:56,790: INFO/ForkPoolWorker-1]   Available text columns: ['text']
[2025-11-02 12:34:56,790: INFO/ForkPoolWorker-1]   Selected column: text
[2025-11-02 12:34:56,791: INFO/ForkPoolWorker-1] Tokenizing dataset with multiprocessing...
```

**✅ Validation Criteria:**
- [ ] Log message appears: **"Using standard text cleaner for preprocessing"**
- [ ] Message appears BEFORE "Tokenizing dataset with multiprocessing..."
- [ ] No errors in logs

**Time:** Real-time during tokenization (appears within first 30 seconds)
**Result:** Confirms cleaning is enabled in production tokenization pipeline

---

## Validation 3: Tokenized Sample Inspection (MANUAL VERIFICATION)

**When:** After tokenization completes, validates that junk was actually removed from data

**Steps:**

### 3.1 Get Tokenized Dataset Path

```bash
# Get dataset details (replace DATASET_ID)
curl -s http://localhost:8000/api/v1/datasets/DATASET_ID | jq '.tokenized_path'

# Example output: "/home/x-sean/app/miStudio/backend/data/datasets/Skylion007_openwebtext_tokenized"
```

### 3.2 Inspect Tokenized Data

**Option A: Quick Sample Check (Python)**
```bash
cd /home/x-sean/app/miStudio/backend

PYTHONPATH=. venv/bin/python3 << 'EOF'
from datasets import load_from_disk
import sys

# Load tokenized dataset
dataset_path = "data/datasets/Skylion007_openwebtext_tokenized"
dataset = load_from_disk(dataset_path)

print(f"Dataset size: {len(dataset)} samples")
print(f"Features: {dataset.features}")
print("\n" + "="*80)
print("SAMPLE INSPECTION (first 10 samples)")
print("="*80)

# Check first 10 samples
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

for i in range(min(10, len(dataset))):
    sample = dataset[i]
    input_ids = sample['input_ids']

    # Decode tokens to see original text
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)

    print(f"\nSample {i}:")
    print(f"  Token count: {len(input_ids)}")
    print(f"  Decoded text (first 200 chars): {decoded[:200]}")

    # Check for junk indicators
    has_html = '<' in decoded or '>' in decoded
    has_entities = '&nbsp;' in decoded or '&lt;' in decoded or '&gt;' in decoded
    has_control = any(ord(c) < 32 and c not in ['\n', '\t', '\r'] for c in decoded)

    if has_html:
        print(f"  ⚠️  WARNING: HTML tags found!")
    if has_entities:
        print(f"  ⚠️  WARNING: HTML entities found!")
    if has_control:
        print(f"  ⚠️  WARNING: Control characters found!")
    if not (has_html or has_entities or has_control):
        print(f"  ✅ Clean (no HTML/entities/control chars detected)")

EOF
```

**Option B: Deep Analysis Script**
```bash
cd /home/x-sean/app/miStudio/backend

cat > inspect_tokenized_cleaning.py << 'EOF'
#!/usr/bin/env python3
"""
Inspect tokenized dataset to verify text cleaning worked.
"""
from datasets import load_from_disk
from transformers import AutoTokenizer
import re

dataset_path = "data/datasets/Skylion007_openwebtext_tokenized"
tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading dataset and tokenizer...")
dataset = load_from_disk(dataset_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

print(f"\nDataset: {len(dataset)} samples")

# Patterns to detect junk
html_tag_pattern = re.compile(r'<[^>]+>')
html_entity_pattern = re.compile(r'&[a-zA-Z]+;|&#\d+;')
excessive_punct_pattern = re.compile(r'[!?.,;:\-]{4,}')

# Statistics
total_samples = min(1000, len(dataset))  # Sample first 1000
samples_with_html = 0
samples_with_entities = 0
samples_with_excessive_punct = 0
samples_with_control_chars = 0

print(f"\nAnalyzing first {total_samples} samples...")

for i in range(total_samples):
    sample = dataset[i]
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)

    # Check for junk
    if html_tag_pattern.search(decoded):
        samples_with_html += 1
    if html_entity_pattern.search(decoded):
        samples_with_entities += 1
    if excessive_punct_pattern.search(decoded):
        samples_with_excessive_punct += 1
    if any(ord(c) < 32 and c not in ['\n', '\t', '\r'] for c in decoded):
        samples_with_control_chars += 1

print("\n" + "="*80)
print("CLEANING VALIDATION RESULTS")
print("="*80)
print(f"Samples analyzed: {total_samples}")
print(f"Samples with HTML tags: {samples_with_html} ({samples_with_html/total_samples*100:.2f}%)")
print(f"Samples with HTML entities: {samples_with_entities} ({samples_with_entities/total_samples*100:.2f}%)")
print(f"Samples with excessive punctuation (4+): {samples_with_excessive_punct} ({samples_with_excessive_punct/total_samples*100:.2f}%)")
print(f"Samples with control characters: {samples_with_control_chars} ({samples_with_control_chars/total_samples*100:.2f}%)")
print("="*80)

if samples_with_html == 0 and samples_with_entities == 0 and samples_with_excessive_punct == 0 and samples_with_control_chars == 0:
    print("\n✅ SUCCESS: No junk detected! Text cleaning is working.")
else:
    print("\n⚠️  WARNING: Junk detected! Text cleaning may not be working correctly.")
    print("   Expected all counts to be 0 (or very close to 0).")
EOF

chmod +x inspect_tokenized_cleaning.py
PYTHONPATH=. venv/bin/python3 inspect_tokenized_cleaning.py
```

**✅ Validation Criteria:**
- [ ] Samples contain NO HTML tags (< 0.1% false positives acceptable)
- [ ] Samples contain NO HTML entities (< 0.1%)
- [ ] Samples contain NO excessive punctuation (4+ repeats)
- [ ] Samples contain NO control characters (except \n, \t, \r)
- [ ] Decoded text looks clean and readable

**Time:** 1-2 minutes (depends on dataset size)
**Result:** Confirms junk was actually removed from tokenized data

---

## Validation 4: Training Monitoring (HOURS)

**When:** During SAE training, monitor for any issues caused by cleaning

**Steps:**

### 4.1 Start Training with Cleaned Dataset

**Via UI:**
1. Go to "Training" panel
2. Click "New Training"
3. Select:
   - Dataset: Your tokenized dataset (with cleaning)
   - Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
   - Layer: 14
   - Architecture:
     - Hidden dim: 2048
     - Latent dim: 8192
     - L1 alpha: **0.001** (NOT 0.05!)
4. Click "Start Training"

### 4.2 Monitor Training Progress

**Check Logs:**
```bash
# Watch training progress
tail -f /tmp/celery_worker.log | grep -E "(Training|loss|step|feature)"
```

**Watch UI:**
- Loss should decrease smoothly
- No sudden spikes or NaN values
- Training completes successfully

**✅ Validation Criteria:**
- [ ] Training starts successfully
- [ ] Loss decreases over time (not flat or increasing)
- [ ] No NaN or Inf values in loss
- [ ] Training completes without errors
- [ ] Final loss is reasonable (< 200 for TinyLlama)

**Time:** Hours (depends on training steps)
**Result:** Confirms cleaned data doesn't break training

---

## Validation 5: Feature Quality Analysis (FINAL)

**When:** After training completes, validates that features are cleaner and more interpretable

**Steps:**

### 5.1 Extract Features

1. Go to "Features" panel
2. Click "New Extraction"
3. Select your new training (with cleaned data)
4. Set parameters:
   - Dataset: Same dataset used for training
   - Top K: 100
   - Batch size: 32
5. Click "Start Extraction"

### 5.2 Analyze Feature Correlations

**Via UI:**
1. Go to "Features" panel
2. Click on a feature to open detail modal
3. Go to "Correlation" tab
4. Click "Calculate Correlation"
5. Review top tokens

**Via API:**
```bash
# Get extraction ID
EXTRACTION_ID=$(curl -s http://localhost:8000/api/v1/extractions | jq -r '.data[0].id')

# Get feature list
curl -s http://localhost:8000/api/v1/extractions/$EXTRACTION_ID/features | jq '.data[0:10]'

# Get correlation for a feature (replace FEATURE_ID)
FEATURE_ID=$(curl -s http://localhost:8000/api/v1/extractions/$EXTRACTION_ID/features | jq -r '.data[0].id')
curl -s http://localhost:8000/api/v1/features/$FEATURE_ID/correlation | jq
```

**Compare with Old Training (without cleaning):**

**OLD Training (train_05555a4b) - WITHOUT cleaning:**
```json
{
  "top_tokens": ["\u0004", "\u0003", "\u0001", "\u0002", "<s>", ...],
  "top_tokens_full": ["<NULL>", "<NULL>", "<NULL>", ...]
}
```
- ⚠️ Control characters
- ⚠️ Special tokens only
- ⚠️ No interpretable words

**NEW Training - WITH cleaning:**
```json
{
  "top_tokens": ["the", "and", "of", "to", "in", "a", "is", ...],
  "top_tokens_full": ["the", "and", "of", "to", "in", ...]
}
```
- ✅ Real words
- ✅ Interpretable patterns
- ✅ No junk tokens

**✅ Validation Criteria:**
- [ ] Top correlated tokens are WORDS, not control characters
- [ ] No `\u0000`, `\u0001`, `\u0002`, etc. in top tokens
- [ ] No `<NULL>` in decoded tokens
- [ ] No HTML tags in top correlations
- [ ] No HTML entities in top correlations
- [ ] Features show interpretable patterns

**Time:** 10-30 minutes (depends on extraction speed)
**Result:** Final confirmation that text cleaning improved feature quality

---

## Validation 6: Side-by-Side Comparison (OPTIONAL)

**When:** To quantitatively measure cleaning impact

**Steps:**

### 6.1 Train Two SAEs

**Training A: WITHOUT Cleaning (Baseline)**
```bash
# Temporarily disable cleaning for comparison
cd /home/x-sean/app/miStudio/backend

# Create test script
cat > test_without_cleaning.py << 'EOF'
from src.services.tokenization_service import TokenizationService
from datasets import load_from_disk

# Load raw dataset
dataset = load_from_disk("data/datasets/Skylion007_openwebtext")
tokenizer = TokenizationService.load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Tokenize WITHOUT cleaning
tokenized = TokenizationService.tokenize_dataset(
    dataset=dataset,
    tokenizer=tokenizer,
    text_column="text",
    enable_cleaning=False,  # DISABLE for baseline
    max_length=512,
)

# Save
tokenized.save_to_disk("data/datasets/test_without_cleaning")
print("Saved dataset WITHOUT cleaning")
EOF

PYTHONPATH=. venv/bin/python3 test_without_cleaning.py
```

**Training B: WITH Cleaning (Current Default)**
- Use normal tokenization (cleaning enabled by default)

### 6.2 Train Both SAEs

Train two identical SAEs:
- Same model (TinyLlama)
- Same hyperparameters (L1 alpha=0.001, latent_dim=8192)
- Same dataset source
- Only difference: Cleaning enabled vs disabled

### 6.3 Compare Results

**Metrics to Compare:**
- Number of "junk" features (control chars, HTML, etc.)
- Average correlation interpretability score
- Top-10 token quality (manual inspection)
- Feature activation sparsity
- Final training loss

**✅ Expected Results:**
- Cleaned training has FEWER junk features
- Cleaned training has MORE interpretable correlations
- Cleaned training has SIMILAR or BETTER sparsity
- Cleaned training has SIMILAR final loss

**Time:** Several hours (two full training runs)
**Result:** Quantitative proof that cleaning improves feature quality

---

## Quick Test Checklist

**For rapid validation (under 5 minutes):**

1. ✅ **Unit Test** (30 sec)
   ```bash
   cd /home/x-sean/app/miStudio/backend
   PYTHONPATH=. venv/bin/python3 test_text_cleaning.py
   ```

2. ✅ **Trigger Tokenization** (1 min)
   - Via UI: Click "Tokenize" on any dataset

3. ✅ **Check Logs** (real-time)
   ```bash
   tail -f /tmp/celery_worker.log | grep "text cleaner"
   # Should see: "Using standard text cleaner for preprocessing"
   ```

4. ✅ **Inspect Sample** (2 min)
   ```bash
   cd /home/x-sean/app/miStudio/backend
   PYTHONPATH=. venv/bin/python3 -c "
   from datasets import load_from_disk
   from transformers import AutoTokenizer
   ds = load_from_disk('data/datasets/Skylion007_openwebtext_tokenized')
   tok = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
   sample = tok.decode(ds[0]['input_ids'][:200])
   print('Sample:', sample)
   print('Has HTML tags:', '<' in sample and '>' in sample)
   print('Has entities:', '&nbsp;' in sample or '&lt;' in sample)
   "
   ```

**Total Time:** ~5 minutes
**Confidence Level:** High (validates cleaning is active and working)

---

## Troubleshooting

### Problem: Log message "Using standard text cleaner" not appearing

**Possible Causes:**
1. Old code still running (backend not restarted)
2. Celery worker not restarted
3. Logs going to different file

**Solution:**
```bash
# Restart everything
cd /home/x-sean/app/miStudio
./stop-mistudio.sh
./start-mistudio.sh

# Check correct log file
ps aux | grep celery  # Get worker PID
# Check /tmp/celery_worker.log or console output
```

### Problem: Junk still appearing in tokenized samples

**Possible Causes:**
1. Dataset was tokenized BEFORE cleaning implementation
2. Cleaning is disabled somewhere in code
3. Tokenizer adds special tokens that look like junk

**Solution:**
```bash
# Check when dataset was tokenized
curl -s http://localhost:8000/api/v1/datasets/DATASET_ID | jq '.updated_at, .extra_metadata.tokenization'

# If before 2025-11-02, re-tokenize:
# 1. Clear existing tokenization (UI: "Clear Tokenization" button)
# 2. Re-tokenize dataset (UI: "Tokenize" button)
```

### Problem: Training fails with cleaned data

**Possible Causes:**
1. Dataset too small after filtering (cleaning removed too many samples)
2. L1 alpha still too high (should be 0.001, not 0.05)
3. Memory issues

**Solution:**
```bash
# Check dataset size after tokenization
curl -s http://localhost:8000/api/v1/datasets/DATASET_ID | jq '.num_samples, .extra_metadata.tokenization'

# If dataset too small:
# - Use aggressive cleaner with lower min_length
# - Or use minimal cleaner
# - Or use different dataset source
```

---

## Summary

**Earliest Validation Point:** Unit test (30 seconds)
**Fastest Production Validation:** Tokenization logs (real-time, < 1 minute)
**Most Reliable Validation:** Tokenized sample inspection (2-5 minutes)
**Final Confirmation:** Feature correlation analysis (after training, hours)

**Recommended Quick Test:**
1. Run unit test (30 sec) ✅
2. Tokenize dataset (trigger via UI)
3. Check logs for "Using standard text cleaner" (< 1 min) ✅
4. Inspect tokenized samples (2 min) ✅

**Total Time:** ~5 minutes to validate cleaning is working correctly.
