# Technical Design Document: Dataset Management Enhancement 01

**Document ID:** 001_FTDD|Dataset_Management_ENH_01
**Feature:** Dataset Management - Missing Features from Mock UI Reference
**Status:** Ready for Implementation
**Created:** 2025-10-11
**Last Updated:** 2025-10-11
**Owner:** miStudio Development Team

**Extends:** 001_FTDD|Dataset_Management.md
**Related Documents:**
- PRD: `001_FPRD|Dataset_Management_ENH_01.md`
- Gap Analysis: `Dataset_Management_Feature_Gap_Analysis.md`
- Original TDD: `001_FTDD|Dataset_Management.md`
- ADR: `000_PADR|miStudio.md`

---

## 1. Executive Summary

This Technical Design Document outlines the architecture and implementation approach for 9 missing Dataset Management features identified in the gap analysis. The design extends the existing Dataset Management system with **minimal breaking changes** to maintain backward compatibility while closing the feature gap with the Mock UI reference.

**Key Technical Decisions:**
- **Schema Evolution:** Extend metadata JSONB structure (no new columns needed)
- **API Versioning:** Add new endpoints, keep existing ones unchanged
- **Frontend State:** Extend Zustand store with new tokenization settings
- **Backend Services:** Add new methods to existing services (no new services needed)

**Architecture Alignment:** All enhancements follow the existing Component-based architecture with clear separation between presentation, business logic, and data layers.

---

## 2. System Architecture

### 2.1 High-Level Architecture Changes

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend Layer (NEW)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  TokenizationTab (Extended)                               │  │
│  │  - Padding/Truncation dropdowns (NEW)                    │  │
│  │  - Special tokens/attention mask toggles (NEW)           │  │
│  │  - TokenizationPreview component (NEW)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  StatisticsTab (Extended)                                 │  │
│  │  - SequenceLengthHistogram component (NEW)               │  │
│  │  - SplitDistribution component (NEW)                     │  │
│  │  - UniqueTokensMetric component (NEW)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
            │                             │
            │ HTTP/REST (NEW)             │ Existing
            │                             │
┌───────────▼─────────────────────────────▼────────────────────────┐
│                  Backend Layer (Extended)                         │
│  ┌──────────────────┐   ┌──────────────────┐                    │
│  │   API Routes     │   │  Existing Routes │                    │
│  │  /tokenize-      │   │  /tokenize       │                    │
│  │   preview (NEW)  │   │  /samples        │                    │
│  └────────┬─────────┘   └──────┬───────────┘                    │
│           │                     │                                │
│  ┌────────▼─────────────────────▼───────────┐                   │
│  │  TokenizationService (Extended)          │                   │
│  │  - tokenize_preview() (NEW)              │                   │
│  │  - calculate_histogram() (NEW)           │                   │
│  │  - calculate_unique_tokens() (NEW)       │                   │
│  │  - calculate_split_distribution() (NEW)  │                   │
│  │  - Existing methods (unchanged)          │                   │
│  └──────────────────────────────────────────┘                   │
└───────────────────────────────────────────────────────────────────┘
            │
┌───────────▼───────────────────────────────────────────────────────┐
│                 Data Layer (Schema Evolution)                     │
│  ┌───────────────┐  ┌────────────────────────────────────┐      │
│  │  PostgreSQL   │  │  datasets.metadata (JSONB)          │      │
│  │  - datasets   │  │  {                                  │      │
│  │    table      │  │    "tokenization": {                │      │
│  │               │  │      "padding": "max_length",       │      │
│  │               │  │      "truncation": "longest_first", │      │
│  │               │  │      "add_special_tokens": true,    │      │
│  │               │  │      "return_attention_mask": true, │      │
│  │               │  │      "unique_tokens": 50257,        │      │
│  │               │  │      "median_seq_length": 342.5,    │      │
│  │               │  │      "histogram": [...]             │      │
│  │               │  │    },                               │      │
│  │               │  │    "splits": {...}                  │      │
│  │               │  │  }                                  │      │
│  └───────────────┘  └────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Relationships

**New Components:**
```
TokenizationTab (Extended)
├── PaddingStrategyDropdown (NEW)
├── TruncationStrategyDropdown (NEW)
├── SpecialTokensToggle (NEW)
├── AttentionMaskToggle (NEW)
└── TokenizationPreview (NEW)
    ├── PreviewTextArea
    ├── PreviewButton
    └── TokenChipList
        └── TokenChip[]

StatisticsTab (Extended)
├── UniqueTokensMetric (NEW)
├── MedianLengthMetric (NEW)
├── SequenceLengthHistogram (NEW)
│   ├── HistogramBar[]
│   └── HistogramSummary
└── SplitDistribution (NEW)
    └── SplitCard[]
```

**Modified Services:**
```
TokenizationService (Extended)
├── tokenize_preview() (NEW)
├── calculate_histogram() (NEW)
├── calculate_unique_tokens() (NEW)
├── calculate_split_distribution() (NEW)
└── calculate_statistics() (MODIFIED - add new metrics)
```

---

## 3. Data Design

### 3.1 Database Schema Changes

**No new columns required!** All new data stored in existing `metadata` JSONB column.

**Extended Metadata Structure:**
```json
{
  "schema": {
    "text_columns": ["text"],
    "column_info": {...},
    "all_columns": [...],
    "is_multi_column": false
  },
  "tokenization": {
    // Existing fields
    "tokenizer_name": "gpt2",
    "text_column_used": "text",
    "max_length": 512,
    "stride": 0,
    "num_tokens": 1000000,
    "avg_seq_length": 342.5,
    "min_seq_length": 10,
    "max_seq_length": 512,

    // NEW: Tokenization settings
    "padding": "max_length",                    // NEW
    "truncation": "longest_first",              // NEW
    "add_special_tokens": true,                 // NEW
    "return_attention_mask": true,              // NEW

    // NEW: Additional statistics
    "unique_tokens": 50257,                     // NEW
    "median_seq_length": 342.5,                 // NEW
    "histogram": [                              // NEW
      {"range": "0-100", "min": 0, "max": 100, "count": 150, "percentage": 1.5},
      {"range": "100-200", "min": 100, "max": 200, "count": 2300, "percentage": 23.0},
      {"range": "200-400", "min": 200, "max": 400, "count": 5000, "percentage": 50.0},
      {"range": "400-600", "min": 400, "max": 600, "count": 2000, "percentage": 20.0},
      {"range": "600-800", "min": 600, "max": 800, "count": 400, "percentage": 4.0},
      {"range": "800-1000", "min": 800, "max": 1000, "count": 100, "percentage": 1.0},
      {"range": "1000+", "min": 1000, "max": 512, "count": 50, "percentage": 0.5}
    ]
  },
  "splits": {                                   // NEW
    "train": {"count": 8000, "percentage": 80.0},
    "validation": {"count": 1000, "percentage": 10.0},
    "test": {"count": 1000, "percentage": 10.0}
  },
  "download": {
    "split": "train",
    "config": "default"
  }
}
```

**Migration Required:** None (JSONB schema evolution is backward-compatible)

**Validation Strategy:**
```python
# backend/src/schemas/metadata.py (EXTENDED)

class TokenizationMetadata(BaseModel):
    """Tokenization statistics and configuration."""
    # Existing fields
    tokenizer_name: str
    text_column_used: str
    max_length: int = Field(ge=1, le=8192)
    stride: int = Field(ge=0)
    num_tokens: int = Field(ge=0)
    avg_seq_length: float = Field(ge=0)
    min_seq_length: int = Field(ge=0)
    max_seq_length: int = Field(ge=0)

    # NEW: Tokenization settings
    padding: Literal["max_length", "longest", "do_not_pad"] = "max_length"
    truncation: Literal["longest_first", "only_first", "only_second", "do_not_truncate"] = "longest_first"
    add_special_tokens: bool = True
    return_attention_mask: bool = True

    # NEW: Additional statistics
    unique_tokens: Optional[int] = None
    median_seq_length: Optional[float] = None
    histogram: Optional[List[HistogramBucket]] = None


class HistogramBucket(BaseModel):
    """Single bucket in sequence length histogram."""
    range: str  # "0-100", "100-200", etc.
    min: int
    max: int
    count: int
    percentage: float = Field(ge=0, le=100)


class SplitInfo(BaseModel):
    """Information about a single dataset split."""
    count: int = Field(ge=0)
    percentage: float = Field(ge=0, le=100)


class SplitsMetadata(BaseModel):
    """Dataset split distribution."""
    train: Optional[SplitInfo] = None
    validation: Optional[SplitInfo] = None
    test: Optional[SplitInfo] = None


class DatasetMetadata(BaseModel):
    """Complete dataset metadata container (EXTENDED)."""
    schema: Optional[SchemaMetadata] = None
    tokenization: Optional[TokenizationMetadata] = None
    splits: Optional[SplitsMetadata] = None  # NEW
    download: Optional[DownloadMetadata] = None
```

---

## 4. API Design

### 4.1 New Endpoints

#### Endpoint: `POST /api/datasets/tokenize-preview`

**Purpose:** Preview tokenization on sample text without processing full dataset

**Request:**
```json
{
  "tokenizer_name": "gpt2",
  "text": "Once upon a time in a land far away...",
  "max_length": 512,
  "padding": "max_length",
  "truncation": "longest_first",
  "add_special_tokens": true,
  "return_attention_mask": true
}
```

**Response (200 OK):**
```json
{
  "tokens": [
    {
      "id": 50256,
      "text": "<|endoftext|>",
      "type": "special",
      "position": 0
    },
    {
      "id": 7454,
      "text": "Once",
      "type": "regular",
      "position": 1
    },
    {
      "id": 2402,
      "text": " upon",
      "type": "regular",
      "position": 2
    },
    // ... more tokens
  ],
  "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...],
  "token_count": 8,
  "sequence_length": 512,
  "special_token_count": 1
}
```

**Response (400 Bad Request):**
```json
{
  "detail": "Text exceeds maximum length of 1000 characters"
}
```

**Performance Requirements:**
- Response time: <1s p95
- Max text length: 1000 characters
- Tokenizer caching: In-memory LRU cache (up to 10 tokenizers)

**Error Handling:**
- 400: Invalid tokenizer name, text too long, invalid settings
- 500: Tokenizer loading failure, tokenization error

**Implementation Notes:**
```python
# backend/src/api/v1/endpoints/datasets.py

from functools import lru_cache
from transformers import AutoTokenizer

@lru_cache(maxsize=10)
def load_tokenizer_cached(tokenizer_name: str):
    """Cache tokenizers in memory for fast preview."""
    return AutoTokenizer.from_pretrained(tokenizer_name)

@router.post("/tokenize-preview")
async def tokenize_preview(request: TokenizePreviewRequest):
    """Preview tokenization on sample text."""
    # Validate input length
    if len(request.text) > 1000:
        raise HTTPException(400, "Text exceeds maximum length")

    # Load tokenizer (cached)
    tokenizer = load_tokenizer_cached(request.tokenizer_name)

    # Tokenize with settings
    result = tokenizer(
        request.text,
        max_length=request.max_length,
        padding=request.padding,
        truncation=request.truncation != "do_not_truncate",
        add_special_tokens=request.add_special_tokens,
        return_attention_mask=request.return_attention_mask,
    )

    # Convert token IDs to token strings
    tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])

    # Build response
    return {
        "tokens": [
            {
                "id": token_id,
                "text": token_text,
                "type": "special" if token_id in tokenizer.all_special_ids else "regular",
                "position": idx,
            }
            for idx, (token_id, token_text) in enumerate(zip(result["input_ids"], tokens))
        ],
        "attention_mask": result.get("attention_mask", []),
        "token_count": len(result["input_ids"]),
        "sequence_length": request.max_length,
        "special_token_count": sum(1 for tid in result["input_ids"] if tid in tokenizer.all_special_ids),
    }
```

---

### 4.2 Modified Endpoints

#### Endpoint: `POST /api/datasets/{dataset_id}/tokenize` (EXTENDED)

**Purpose:** Extend tokenization endpoint to accept new settings

**Request (EXTENDED):**
```json
{
  "tokenizer_name": "gpt2",
  "max_length": 512,
  "stride": 0,
  // NEW fields:
  "padding": "max_length",
  "truncation": "longest_first",
  "add_special_tokens": true,
  "return_attention_mask": true
}
```

**Backward Compatibility:**
- All new fields are optional with sensible defaults
- Existing requests without new fields continue to work
- Response format unchanged (new data only in metadata)

**Implementation:**
```python
# backend/src/schemas/dataset.py (EXTENDED)

class DatasetTokenizeRequest(BaseModel):
    tokenizer_name: str = "gpt2"
    max_length: int = Field(default=512, ge=1, le=8192)
    stride: int = Field(default=0, ge=0)

    # NEW: Tokenization settings with defaults
    padding: Literal["max_length", "longest", "do_not_pad"] = "max_length"
    truncation: Literal["longest_first", "only_first", "only_second", "do_not_truncate"] = "longest_first"
    add_special_tokens: bool = True
    return_attention_mask: bool = True
```

---

## 5. Service Layer Design

### 5.1 TokenizationService Extensions

**New Methods:**

```python
# backend/src/services/tokenization_service.py (EXTENDED)

class TokenizationService:
    """Service for dataset tokenization (EXTENDED)."""

    @staticmethod
    def calculate_histogram(
        seq_lengths: np.ndarray,
        max_length: int
    ) -> List[Dict[str, Any]]:
        """
        Calculate sequence length histogram with 7 buckets.

        Args:
            seq_lengths: Array of sequence lengths
            max_length: Maximum sequence length (defines last bucket)

        Returns:
            List of histogram buckets with count and percentage

        Raises:
            ValueError: If seq_lengths is empty
        """
        if len(seq_lengths) == 0:
            raise ValueError("Cannot calculate histogram for empty array")

        # Define bin edges
        bins = [0, 100, 200, 400, 600, 800, 1000, max_length]
        histogram = []

        for i in range(len(bins) - 1):
            # Count samples in this bin
            if i < len(bins) - 2:
                count = np.sum((seq_lengths >= bins[i]) & (seq_lengths < bins[i+1]))
            else:
                # Last bin: 1000+
                count = np.sum(seq_lengths >= bins[i])

            histogram.append({
                "range": f"{bins[i]}-{bins[i+1]}" if i < len(bins) - 2 else f"{bins[i]}+",
                "min": bins[i],
                "max": bins[i+1] if i < len(bins) - 2 else max_length,
                "count": int(count),
                "percentage": float(count / len(seq_lengths) * 100),
            })

        return histogram


    @staticmethod
    def calculate_unique_tokens(
        tokenized_dataset: HFDataset
    ) -> int:
        """
        Calculate number of unique tokens in dataset.

        Args:
            tokenized_dataset: HuggingFace dataset with input_ids

        Returns:
            Count of unique token IDs

        Performance:
            - Uses set for O(1) lookups
            - Memory: ~4 bytes per unique token
            - Time: O(n) where n = total tokens
        """
        unique_tokens = set()

        for example in tokenized_dataset:
            if "input_ids" in example:
                unique_tokens.update(example["input_ids"])

        return len(unique_tokens)


    @staticmethod
    def calculate_split_distribution(
        dataset: HFDataset
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate distribution of samples across splits.

        Args:
            dataset: HuggingFace DatasetDict with multiple splits

        Returns:
            Dictionary mapping split name to count and percentage

        Example:
            {
                "train": {"count": 8000, "percentage": 80.0},
                "validation": {"count": 1000, "percentage": 10.0},
                "test": {"count": 1000, "percentage": 10.0}
            }
        """
        splits = {}
        total_samples = sum(len(split) for split in dataset.values())

        for split_name, split_data in dataset.items():
            count = len(split_data)
            splits[split_name] = {
                "count": count,
                "percentage": (count / total_samples * 100) if total_samples > 0 else 0
            }

        return splits


    @staticmethod
    def calculate_statistics(
        tokenized_dataset: HFDataset,
        original_dataset: HFDataset,  # NEW: for split distribution
        tokenization_settings: Dict[str, Any]  # NEW: for storing settings
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive tokenization statistics (EXTENDED).

        Args:
            tokenized_dataset: Tokenized HuggingFace dataset
            original_dataset: Original dataset (for split info)
            tokenization_settings: Tokenization configuration

        Returns:
            Dictionary with all statistics including new metrics
        """
        # Existing statistics calculation
        if len(tokenized_dataset) == 0:
            raise ValueError("Cannot calculate statistics for empty dataset")

        seq_lengths = []
        total_tokens = 0
        samples_without_input_ids = 0

        for example in tokenized_dataset:
            if "input_ids" in example:
                seq_len = len(example["input_ids"])
                seq_lengths.append(seq_len)
                total_tokens += seq_len
            else:
                samples_without_input_ids += 1

        if not seq_lengths:
            raise ValueError(
                f"No valid tokenized samples found. "
                f"{samples_without_input_ids}/{len(tokenized_dataset)} samples missing input_ids"
            )

        seq_lengths_array = np.array(seq_lengths)

        # Existing metrics
        stats = {
            "num_tokens": total_tokens,
            "num_samples": len(tokenized_dataset),
            "avg_seq_length": float(seq_lengths_array.mean()),
            "min_seq_length": int(seq_lengths_array.min()),
            "max_seq_length": int(seq_lengths_array.max()),
        }

        # NEW: Tokenization settings
        stats.update(tokenization_settings)

        # NEW: Additional metrics
        stats["unique_tokens"] = TokenizationService.calculate_unique_tokens(tokenized_dataset)
        stats["median_seq_length"] = float(np.median(seq_lengths_array))
        stats["histogram"] = TokenizationService.calculate_histogram(
            seq_lengths_array,
            tokenization_settings.get("max_length", 512)
        )

        # NEW: Split distribution (if multiple splits exist)
        if hasattr(original_dataset, "keys"):  # DatasetDict
            stats["splits"] = TokenizationService.calculate_split_distribution(original_dataset)

        return stats
```

---

## 6. Frontend Component Architecture

### 6.1 TokenizationTab Component Extensions

**File:** `frontend/src/components/datasets/DatasetDetailModal.tsx`

**New State:**
```typescript
// Existing state
const [tokenizerName, setTokenizerName] = useState('gpt2');
const [maxLength, setMaxLength] = useState(512);
const [stride, setStride] = useState(0);

// NEW: Tokenization settings state
const [paddingStrategy, setPaddingStrategy] = useState<'max_length' | 'longest' | 'do_not_pad'>('max_length');
const [truncationStrategy, setTruncationStrategy] = useState<'longest_first' | 'only_first' | 'only_second' | 'do_not_truncate'>('longest_first');
const [addSpecialTokens, setAddSpecialTokens] = useState(true);
const [returnAttentionMask, setReturnAttentionMask] = useState(true);

// NEW: Preview state
const [previewText, setPreviewText] = useState('');
const [previewTokens, setPreviewTokens] = useState<TokenPreview[]>([]);
const [previewLoading, setPreviewLoading] = useState(false);
```

**New UI Components:**

```tsx
{/* NEW: Padding Strategy Dropdown */}
<div className="space-y-2">
  <label htmlFor="padding-strategy" className="block text-sm font-medium text-slate-300">
    Padding Strategy
  </label>
  <select
    id="padding-strategy"
    value={paddingStrategy}
    onChange={(e) => setPaddingStrategy(e.target.value as any)}
    className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
  >
    <option value="max_length">Max Length (pad to max_length)</option>
    <option value="longest">Longest (pad to longest in batch)</option>
    <option value="do_not_pad">Do Not Pad</option>
  </select>
  <p className="text-xs text-slate-500">
    Controls how sequences are padded. "Max Length" pads all to max_length for consistent memory usage.
  </p>
</div>

{/* NEW: Truncation Strategy Dropdown */}
<div className="space-y-2">
  <label htmlFor="truncation-strategy" className="block text-sm font-medium text-slate-300">
    Truncation Strategy
  </label>
  <select
    id="truncation-strategy"
    value={truncationStrategy}
    onChange={(e) => setTruncationStrategy(e.target.value as any)}
    className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
  >
    <option value="longest_first">Longest First (truncate longest sequence)</option>
    <option value="only_first">Only First (preserve second sequence)</option>
    <option value="only_second">Only Second (preserve first sequence)</option>
    <option value="do_not_truncate">Do Not Truncate (error on overflow)</option>
  </select>
  <p className="text-xs text-slate-500">
    Controls truncation for sequences exceeding max_length. Useful for Q&A pairs or multi-sequence inputs.
  </p>
</div>

{/* NEW: Special Tokens Toggle */}
<div className="flex items-center justify-between">
  <div>
    <label htmlFor="special-tokens" className="text-sm font-medium text-slate-300">
      Add Special Tokens
    </label>
    <p className="text-xs text-slate-500 mt-1">
      Include BOS, EOS, PAD tokens (recommended for most models)
    </p>
  </div>
  <button
    type="button"
    onClick={() => setAddSpecialTokens(!addSpecialTokens)}
    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
      addSpecialTokens ? 'bg-emerald-600' : 'bg-slate-700'
    }`}
  >
    <span
      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
        addSpecialTokens ? 'translate-x-6' : 'translate-x-1'
      }`}
    />
  </button>
</div>

{/* NEW: Attention Mask Toggle */}
<div className="flex items-center justify-between">
  <div>
    <label htmlFor="attention-mask" className="text-sm font-medium text-slate-300">
      Return Attention Mask
    </label>
    <p className="text-xs text-slate-500 mt-1">
      Generate attention masks (disable to save memory if model doesn't use them)
    </p>
  </div>
  <button
    type="button"
    onClick={() => setReturnAttentionMask(!returnAttentionMask)}
    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
      returnAttentionMask ? 'bg-emerald-600' : 'bg-slate-700'
    }`}
  >
    <span
      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
        returnAttentionMask ? 'translate-x-6' : 'translate-x-1'
      }`}
    />
  </button>
</div>
```

---

### 6.2 TokenizationPreview Component

**File:** `frontend/src/components/datasets/TokenizationPreview.tsx` (NEW)

```typescript
import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';

interface Token {
  id: number;
  text: string;
  type: 'special' | 'regular';
  position: number;
}

interface TokenizationPreviewProps {
  tokenizerName: string;
  maxLength: number;
  padding: string;
  truncation: string;
  addSpecialTokens: boolean;
  returnAttentionMask: boolean;
}

export const TokenizationPreview: React.FC<TokenizationPreviewProps> = ({
  tokenizerName,
  maxLength,
  padding,
  truncation,
  addSpecialTokens,
  returnAttentionMask,
}) => {
  const [previewText, setPreviewText] = useState('');
  const [tokens, setTokens] = useState<Token[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePreview = async () => {
    if (!previewText.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/datasets/tokenize-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tokenizer_name: tokenizerName,
          text: previewText,
          max_length: maxLength,
          padding,
          truncation,
          add_special_tokens: addSpecialTokens,
          return_attention_mask: returnAttentionMask,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Preview failed');
      }

      const data = await response.json();
      setTokens(data.tokens);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="border-t border-slate-700 pt-6 mt-6">
      <h4 className="text-sm font-semibold text-slate-300 mb-3">Preview Tokenization</h4>

      {/* Preview Text Input */}
      <textarea
        value={previewText}
        onChange={(e) => setPreviewText(e.target.value)}
        placeholder="Enter text to preview tokenization... (max 1000 characters)"
        maxLength={1000}
        rows={3}
        className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-sm font-mono"
      />

      {/* Character Count */}
      <div className="flex justify-between items-center mt-2">
        <span className="text-xs text-slate-500">
          {previewText.length} / 1000 characters
        </span>

        {/* Preview Button */}
        <button
          onClick={handlePreview}
          disabled={!previewText.trim() || loading}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center gap-2 text-sm transition-colors"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Tokenizing...
            </>
          ) : (
            'Tokenize Preview'
          )}
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mt-3 p-3 bg-red-900/20 border border-red-700 rounded-lg text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Token Chips */}
      {tokens.length > 0 && (
        <div className="mt-4 bg-slate-800/30 rounded-lg p-4">
          <div className="flex flex-wrap gap-1 mb-3">
            {tokens.map((token) => (
              <span
                key={token.position}
                title={`Token ID: ${token.id}`}
                className={`px-2 py-1 rounded text-xs font-mono ${
                  token.type === 'special'
                    ? 'bg-emerald-700 text-emerald-200 border border-emerald-600'
                    : 'bg-slate-700 text-slate-200 border border-slate-600'
                }`}
              >
                {token.text}
              </span>
            ))}
          </div>
          <div className="flex justify-between text-xs text-slate-400">
            <span>{tokens.length} tokens</span>
            <span>
              {tokens.filter((t) => t.type === 'special').length} special tokens
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
```

---

### 6.3 SequenceLengthHistogram Component

**File:** `frontend/src/components/datasets/SequenceLengthHistogram.tsx` (NEW)

```typescript
import React from 'react';

interface HistogramBucket {
  range: string;
  min: number;
  max: number;
  count: number;
  percentage: number;
}

interface SequenceLengthHistogramProps {
  histogram: HistogramBucket[];
  minLength: number;
  medianLength: number;
  maxLength: number;
}

export const SequenceLengthHistogram: React.FC<SequenceLengthHistogramProps> = ({
  histogram,
  minLength,
  medianLength,
  maxLength,
}) => {
  return (
    <div className="bg-slate-800/50 rounded-lg p-6">
      <h3 className="text-lg font-semibold text-slate-100 mb-4">
        Sequence Length Distribution
      </h3>

      {/* Histogram Bars */}
      <div className="space-y-2">
        {histogram.map((bucket, idx) => (
          <div key={idx} className="flex items-center gap-3">
            {/* Bucket Range Label */}
            <div className="w-28 text-sm text-slate-400 font-mono">
              {bucket.range}
            </div>

            {/* Bar Container */}
            <div className="flex-1 h-8 bg-slate-700 rounded overflow-hidden">
              {/* Bar Fill with Gradient */}
              <div
                className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 flex items-center justify-end pr-2 transition-all duration-300"
                style={{ width: `${Math.max(bucket.percentage, 0.5)}%` }}
              >
                {bucket.percentage > 5 && (
                  <span className="text-xs text-white font-medium">
                    {bucket.count.toLocaleString()} ({bucket.percentage.toFixed(1)}%)
                  </span>
                )}
              </div>
            </div>

            {/* Count/Percentage (if bar too small) */}
            {bucket.percentage <= 5 && (
              <div className="w-32 text-xs text-slate-400">
                {bucket.count.toLocaleString()} ({bucket.percentage.toFixed(1)}%)
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Summary Statistics */}
      <div className="mt-4 pt-4 border-t border-slate-700 text-sm text-slate-400">
        <span className="font-medium">Min:</span> {minLength} tokens •{' '}
        <span className="font-medium">Median:</span> {medianLength.toFixed(1)} tokens •{' '}
        <span className="font-medium">Max:</span> {maxLength} tokens
      </div>
    </div>
  );
};
```

---

### 6.4 SplitDistribution Component

**File:** `frontend/src/components/datasets/SplitDistribution.tsx` (NEW)

```typescript
import React from 'react';

interface SplitInfo {
  count: number;
  percentage: number;
}

interface SplitDistributionProps {
  splits: {
    train?: SplitInfo;
    validation?: SplitInfo;
    test?: SplitInfo;
  };
}

export const SplitDistribution: React.FC<SplitDistributionProps> = ({ splits }) => {
  const splitConfigs = [
    { name: 'train', label: 'Training', color: 'emerald', data: splits.train },
    { name: 'validation', label: 'Validation', color: 'blue', data: splits.validation },
    { name: 'test', label: 'Test', color: 'purple', data: splits.test },
  ];

  const activeSplits = splitConfigs.filter((s) => s.data);

  if (activeSplits.length === 0) {
    return null;
  }

  return (
    <div className="bg-slate-800/50 rounded-lg p-6">
      <h3 className="text-lg font-semibold text-slate-100 mb-4">Split Distribution</h3>

      <div className="grid grid-cols-3 gap-4">
        {activeSplits.map((split) => (
          <div
            key={split.name}
            className={`bg-${split.color}-900/30 border border-${split.color}-700 rounded-lg p-4`}
          >
            <div className="text-xs text-slate-400 mb-1">{split.label}</div>
            <div className="text-2xl font-bold text-slate-100">
              {split.data!.count.toLocaleString()}
            </div>
            <div className={`text-sm text-${split.color}-400 mt-1`}>
              {split.data!.percentage.toFixed(1)}% of total
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
```

---

## 7. Testing Strategy

### 7.1 Backend Unit Tests

**New Test File:** `backend/tests/unit/test_tokenization_enhancements.py`

```python
import pytest
import numpy as np
from src.services.tokenization_service import TokenizationService


class TestHistogramCalculation:
    """Tests for histogram calculation."""

    def test_histogram_basic(self):
        """Test basic histogram calculation."""
        seq_lengths = np.array([50, 150, 250, 450, 650, 850, 1200])
        histogram = TokenizationService.calculate_histogram(seq_lengths, 2048)

        assert len(histogram) == 7
        assert histogram[0]["range"] == "0-100"
        assert histogram[0]["count"] == 1
        assert histogram[6]["range"] == "1000+"

    def test_histogram_percentages_sum_to_100(self):
        """Test that percentages sum to 100%."""
        seq_lengths = np.random.randint(0, 512, 10000)
        histogram = TokenizationService.calculate_histogram(seq_lengths, 512)

        total_percentage = sum(bucket["percentage"] for bucket in histogram)
        assert 99.9 <= total_percentage <= 100.1  # Allow floating point error

    def test_histogram_empty_array(self):
        """Test histogram with empty array raises error."""
        with pytest.raises(ValueError, match="empty array"):
            TokenizationService.calculate_histogram(np.array([]), 512)


class TestUniqueTokens:
    """Tests for unique token calculation."""

    def test_unique_tokens_basic(self):
        """Test unique token count."""
        dataset = MockDataset([
            {"input_ids": [1, 2, 3, 4]},
            {"input_ids": [2, 3, 5, 6]},
            {"input_ids": [1, 5, 7, 8]},
        ])

        unique = TokenizationService.calculate_unique_tokens(dataset)
        assert unique == 8  # {1, 2, 3, 4, 5, 6, 7, 8}

    def test_unique_tokens_duplicates(self):
        """Test that duplicates are counted once."""
        dataset = MockDataset([
            {"input_ids": [1, 1, 1, 1]},
            {"input_ids": [1, 1, 1, 1]},
        ])

        unique = TokenizationService.calculate_unique_tokens(dataset)
        assert unique == 1


class TestSplitDistribution:
    """Tests for split distribution calculation."""

    def test_split_distribution_balanced(self):
        """Test split distribution with balanced splits."""
        dataset = MockDatasetDict({
            "train": MockDataset([{"text": "a"}] * 8000),
            "validation": MockDataset([{"text": "b"}] * 1000),
            "test": MockDataset([{"text": "c"}] * 1000),
        })

        splits = TokenizationService.calculate_split_distribution(dataset)

        assert splits["train"]["count"] == 8000
        assert splits["train"]["percentage"] == 80.0
        assert splits["validation"]["percentage"] == 10.0
        assert splits["test"]["percentage"] == 10.0

    def test_split_distribution_missing_split(self):
        """Test split distribution with missing test split."""
        dataset = MockDatasetDict({
            "train": MockDataset([{"text": "a"}] * 9000),
            "validation": MockDataset([{"text": "b"}] * 1000),
        })

        splits = TokenizationService.calculate_split_distribution(dataset)

        assert "train" in splits
        assert "validation" in splits
        assert "test" not in splits
```

---

### 7.2 API Integration Tests

**New Test File:** `backend/tests/integration/test_tokenize_preview.py`

```python
import pytest
from fastapi.testclient import TestClient


def test_tokenize_preview_success(client: TestClient):
    """Test successful tokenization preview."""
    response = client.post("/api/datasets/tokenize-preview", json={
        "tokenizer_name": "gpt2",
        "text": "Hello world!",
        "max_length": 512,
        "padding": "max_length",
        "truncation": "longest_first",
        "add_special_tokens": True,
        "return_attention_mask": True,
    })

    assert response.status_code == 200
    data = response.json()

    assert "tokens" in data
    assert "token_count" in data
    assert len(data["tokens"]) > 0
    assert data["tokens"][0]["type"] in ["special", "regular"]


def test_tokenize_preview_text_too_long(client: TestClient):
    """Test preview with text exceeding max length."""
    response = client.post("/api/datasets/tokenize-preview", json={
        "tokenizer_name": "gpt2",
        "text": "a" * 1001,  # Exceeds 1000 char limit
        "max_length": 512,
    })

    assert response.status_code == 400
    assert "exceeds maximum length" in response.json()["detail"].lower()


def test_tokenize_preview_invalid_tokenizer(client: TestClient):
    """Test preview with invalid tokenizer."""
    response = client.post("/api/datasets/tokenize-preview", json={
        "tokenizer_name": "invalid-tokenizer-name",
        "text": "Hello world!",
    })

    assert response.status_code == 400 or response.status_code == 500
```

---

### 7.3 Frontend Component Tests

**New Test File:** `frontend/src/components/datasets/TokenizationPreview.test.tsx`

```typescript
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TokenizationPreview } from './TokenizationPreview';
import { rest } from 'msw';
import { setupServer } from 'msw/node';

const server = setupServer(
  rest.post('/api/datasets/tokenize-preview', (req, res, ctx) => {
    return res(
      ctx.json({
        tokens: [
          { id: 50256, text: '<|endoftext|>', type: 'special', position: 0 },
          { id: 15496, text: 'Hello', type: 'regular', position: 1 },
          { id: 995, text: ' world', type: 'regular', position: 2 },
        ],
        token_count: 3,
        sequence_length: 512,
        special_token_count: 1,
      })
    );
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('TokenizationPreview', () => {
  it('renders preview form', () => {
    render(
      <TokenizationPreview
        tokenizerName="gpt2"
        maxLength={512}
        padding="max_length"
        truncation="longest_first"
        addSpecialTokens={true}
        returnAttentionMask={true}
      />
    );

    expect(screen.getByPlaceholderText(/Enter text to preview/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Tokenize Preview/i })).toBeInTheDocument();
  });

  it('displays tokens after successful preview', async () => {
    render(
      <TokenizationPreview
        tokenizerName="gpt2"
        maxLength={512}
        padding="max_length"
        truncation="longest_first"
        addSpecialTokens={true}
        returnAttentionMask={true}
      />
    );

    const textarea = screen.getByPlaceholderText(/Enter text to preview/i);
    const button = screen.getByRole('button', { name: /Tokenize Preview/i });

    fireEvent.change(textarea, { target: { value: 'Hello world!' } });
    fireEvent.click(button);

    await waitFor(() => {
      expect(screen.getByText('<|endoftext|>')).toBeInTheDocument();
      expect(screen.getByText('Hello')).toBeInTheDocument();
      expect(screen.getByText(' world')).toBeInTheDocument();
    });

    expect(screen.getByText(/3 tokens/i)).toBeInTheDocument();
  });

  it('disables button when text is empty', () => {
    render(
      <TokenizationPreview
        tokenizerName="gpt2"
        maxLength={512}
        padding="max_length"
        truncation="longest_first"
        addSpecialTokens={true}
        returnAttentionMask={true}
      />
    );

    const button = screen.getByRole('button', { name: /Tokenize Preview/i });
    expect(button).toBeDisabled();
  });

  it('shows special tokens with different styling', async () => {
    render(
      <TokenizationPreview
        tokenizerName="gpt2"
        maxLength={512}
        padding="max_length"
        truncation="longest_first"
        addSpecialTokens={true}
        returnAttentionMask={true}
      />
    );

    const textarea = screen.getByPlaceholderText(/Enter text to preview/i);
    fireEvent.change(textarea, { target: { value: 'Hello' } });

    const button = screen.getByRole('button', { name: /Tokenize Preview/i });
    fireEvent.click(button);

    await waitFor(() => {
      const specialToken = screen.getByText('<|endoftext|>');
      expect(specialToken).toHaveClass('bg-emerald-700');
    });
  });
});
```

---

## 8. Performance Optimization

### 8.1 Backend Performance

**Tokenizer Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=10)
def load_tokenizer_cached(tokenizer_name: str):
    """Cache up to 10 tokenizers in memory."""
    return AutoTokenizer.from_pretrained(tokenizer_name)
```

**NumPy Vectorization:**
```python
# BEFORE: Python loop (slow for 1M+ samples)
unique_tokens = set()
for example in dataset:
    unique_tokens.update(example["input_ids"])

# AFTER: NumPy vectorization (10x faster)
all_tokens = np.concatenate([example["input_ids"] for example in dataset])
unique_tokens = np.unique(all_tokens)
```

**Histogram Binning:**
```python
# Use NumPy boolean indexing (vectorized)
count = np.sum((lengths >= bins[i]) & (lengths < bins[i+1]))
```

**Performance Targets:**
- Tokenization preview: <1s p95
- Histogram calculation (1M samples): <2s
- Unique tokens calculation (1M samples): <3s
- Split distribution: <500ms

---

### 8.2 Frontend Performance

**Component Memoization:**
```typescript
export const TokenChip = React.memo<TokenChipProps>(({ token }) => {
  return (
    <span className={token.type === 'special' ? 'bg-emerald-700' : 'bg-slate-700'}>
      {token.text}
    </span>
  );
});
```

**Virtualization (if >1000 tokens):**
```typescript
import { FixedSizeList } from 'react-window';

// Only render visible tokens (improves performance for very long sequences)
<FixedSizeList
  height={400}
  itemCount={tokens.length}
  itemSize={32}
  width="100%"
>
  {({ index, style }) => (
    <div style={style}>
      <TokenChip token={tokens[index]} />
    </div>
  )}
</FixedSizeList>
```

**Debounced Preview:**
```typescript
import { useDebouncedCallback } from 'use-debounce';

const debouncedPreview = useDebouncedCallback(
  (text: string) => {
    handlePreview(text);
  },
  500  // Wait 500ms after user stops typing
);
```

---

## 9. Deployment Considerations

### 9.1 Database Migration

**No migration required!** All new data stored in existing JSONB `metadata` column.

**Backward Compatibility:**
- Old datasets without new metadata fields will work fine
- Frontend displays "Not Available" for missing metrics
- Tokenization with defaults works for all datasets

---

### 9.2 API Versioning

**No API version change required:**
- New endpoint: `/api/datasets/tokenize-preview` (v1)
- Modified endpoint: `/api/datasets/{id}/tokenize` (v1, backward-compatible)
- All new fields are optional with defaults

---

### 9.3 Rollback Plan

**If critical bugs found:**
1. Revert frontend to previous version (no backend changes needed)
2. Old tokenization endpoint continues to work without new fields
3. New preview endpoint can be disabled via feature flag

**Feature Flag:**
```python
# backend/src/core/config.py
class Settings(BaseSettings):
    ENABLE_TOKENIZATION_PREVIEW: bool = Field(default=True)

# backend/src/api/v1/endpoints/datasets.py
@router.post("/tokenize-preview")
async def tokenize_preview(...):
    if not settings.ENABLE_TOKENIZATION_PREVIEW:
        raise HTTPException(503, "Feature temporarily disabled")
    ...
```

---

## 10. Risk Mitigation

### 10.1 Technical Risks

| Risk | Mitigation |
|------|-----------|
| Tokenization preview slow | Implement tokenizer caching, limit text to 1000 chars |
| Histogram calculation slow | Use NumPy vectorization, sample if >1M samples |
| Breaking existing tokenization | Extensive regression testing, all new fields optional |
| Frontend state complexity | Use Zustand store with typed actions |

### 10.2 Data Integrity

**Metadata Validation:**
- Pydantic schemas validate all new fields
- JSONB schema allows evolution without breaking old data
- Deep merge preserves existing metadata sections

**Testing Strategy:**
- Unit tests for all new service methods
- Integration tests for API endpoints
- E2E tests for full user workflows
- Visual regression tests for UI components

---

## 11. Monitoring and Observability

### 11.1 New Metrics

**Backend:**
- `tokenize_preview_requests_total` (counter)
- `tokenize_preview_duration_seconds` (histogram)
- `histogram_calculation_duration_seconds` (histogram)
- `unique_tokens_calculation_duration_seconds` (histogram)

**Frontend:**
- Preview button clicks (analytics)
- Average tokens per preview (analytics)
- Error rate for preview endpoint (monitoring)

---

## 12. Documentation Updates

### 12.1 API Documentation

**New Endpoint Documentation:**
```yaml
/api/datasets/tokenize-preview:
  post:
    summary: Preview tokenization on sample text
    description: |
      Tokenizes sample text with specified settings without processing the full dataset.
      Useful for testing tokenizer configuration before committing to full tokenization.
    parameters:
      - in: body
        schema:
          type: object
          required: [tokenizer_name, text, max_length]
          properties:
            tokenizer_name:
              type: string
              example: "gpt2"
            text:
              type: string
              maxLength: 1000
              example: "Once upon a time..."
            # ... more parameters
    responses:
      200:
        description: Tokenization preview
        schema:
          type: object
          properties:
            tokens: ...
```

---

**Document End**
**Status:** Ready for Implementation Document (TID) creation
**Next Step:** Create 001_FTID|Dataset_Management_ENH_01.md with specific coding hints
**Total Sections:** 12
**Estimated Implementation Time:** 46-64 hours (6-8 working days)
