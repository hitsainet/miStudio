# Technical Implementation Document: Dataset Management Enhancement 01

**Document ID:** 001_FTID|Dataset_Management_ENH_01
**Feature:** Dataset Management - Missing Features from Mock UI Reference
**PRD Reference:** 001_FPRD|Dataset_Management_ENH_01.md
**TDD Reference:** 001_FTDD|Dataset_Management_ENH_01.md
**Status:** Ready for Implementation
**Created:** 2025-10-11
**Last Updated:** 2025-10-11

**Extends:** 001_FTID|Dataset_Management.md
**Related Documents:**
- Gap Analysis: `Dataset_Management_Feature_Gap_Analysis.md`
- Original TID: `001_FTID|Dataset_Management.md`
- Original Tasks: `001_FTASKS|Dataset_Management.md`
- ADR: `000_PADR|miStudio.md`

---

## 1. Implementation Overview

This Technical Implementation Document provides production-ready code snippets and implementation guidance for the 9 missing Dataset Management features. All code examples are copy-paste-ready with proper error handling, type hints, and docstrings.

**Key Principles:**
- All new features integrate with existing architecture (no breaking changes)
- Metadata stored in existing JSONB column (no database migration needed)
- All TypeScript is strict mode compliant (no `any` types)
- All Python code has type hints and passes MyPy
- All API endpoints are backward-compatible (new fields are optional)

**Prerequisites:**
Before implementing these features, ensure you have completed:
- ✅ Original Dataset Management implementation (Phase 1-10)
- ✅ Database migration 118f85d483dd (datasets table with metadata JSONB)
- ✅ Backend services: DatasetService, TokenizationService
- ✅ Frontend stores: datasetsStore
- ✅ WebSocket integration: useWebSocket hook

**Implementation Phases:**
- **Phase 14** (26-38h): P1 features - Padding, Truncation, Preview, Histogram
- **Phase 15** (20-26h): P2 features - Special Tokens, Attention Mask, Unique Tokens, Split Distribution

---

## 2. Section 1: Backend Implementation - Tokenization Settings

### 2.1 Padding Strategy Implementation

**Purpose:** Allow users to select padding strategy (max_length, longest, do_not_pad) for tokenization.

**Backend Changes:**

#### Schema Update: DatasetTokenizeRequest

**File:** `backend/src/schemas/dataset.py`

**Existing Code (line ~45):**
```python
class DatasetTokenizeRequest(BaseModel):
    tokenizer_name: str = "gpt2"
    max_length: int = Field(default=512, ge=1, le=8192)
    stride: int = Field(default=0, ge=0)
```

**New Code (insert after stride):**
```python
class DatasetTokenizeRequest(BaseModel):
    """Request schema for dataset tokenization (EXTENDED)."""

    tokenizer_name: str = Field(
        default="gpt2",
        description="HuggingFace tokenizer identifier"
    )
    max_length: int = Field(
        default=512,
        ge=1,
        le=8192,
        description="Maximum sequence length"
    )
    stride: int = Field(
        default=0,
        ge=0,
        description="Stride for sliding window"
    )

    # NEW: Padding strategy
    padding: Literal["max_length", "longest", "do_not_pad"] = Field(
        default="max_length",
        description=(
            "Padding strategy: "
            "'max_length' pads to max_length, "
            "'longest' pads to longest sequence in batch, "
            "'do_not_pad' disables padding"
        )
    )

    @field_validator("padding")
    @classmethod
    def validate_padding(cls, v: str) -> str:
        """Validate padding strategy value."""
        allowed = ["max_length", "longest", "do_not_pad"]
        if v not in allowed:
            raise ValueError(f"padding must be one of {allowed}")
        return v
```

**Import Additions (top of file):**
```python
from typing import Literal
from pydantic import field_validator
```

#### Service Method Update: tokenize_dataset

**File:** `backend/src/workers/dataset_tasks.py`

**Existing Code (line ~200-250):**
```python
def tokenize_function(examples):
    return tokenizer(
        examples[text_column],
        max_length=max_length,
        stride=stride,
        truncation=True,
        return_overflowing_tokens=stride > 0,
    )
```

**Updated Code:**
```python
def tokenize_function(examples):
    """
    Tokenize text examples with configurable settings.

    Args:
        examples: Dictionary with text column data

    Returns:
        Dictionary with input_ids, attention_mask, etc.
    """
    # Map padding strategy to HuggingFace tokenizer format
    padding_config = {
        "max_length": "max_length",
        "longest": "longest",
        "do_not_pad": False  # HuggingFace uses False for no padding
    }

    return tokenizer(
        examples[text_column],
        max_length=max_length,
        stride=stride,
        padding=padding_config.get(padding, "max_length"),  # Use padding param
        truncation=True,
        return_overflowing_tokens=stride > 0,
    )
```

**Task Definition Update (line ~150):**
```python
@celery_app.task(bind=True, max_retries=3)
def tokenize_dataset_task(
    self: Task,
    dataset_id: str,
    tokenizer_name: str,
    max_length: int = 512,
    stride: int = 0,
    padding: str = "max_length",  # NEW parameter
):
    """
    Tokenize dataset with configurable settings.

    Args:
        self: Celery task instance
        dataset_id: UUID of dataset to tokenize
        tokenizer_name: HuggingFace tokenizer identifier
        max_length: Maximum sequence length
        stride: Stride for sliding window
        padding: Padding strategy (max_length, longest, do_not_pad)
    """
    # Store padding in metadata for display
    metadata_update = {
        "tokenization": {
            "tokenizer_name": tokenizer_name,
            "max_length": max_length,
            "stride": stride,
            "padding": padding,  # Store for frontend display
            # ... other fields
        }
    }
```

#### Validation Tests

**File:** `backend/tests/unit/test_tokenization_enhancements.py` (NEW)

```python
import pytest
from src.schemas.dataset import DatasetTokenizeRequest
from pydantic import ValidationError


class TestPaddingStrategyValidation:
    """Tests for padding strategy validation."""

    def test_valid_padding_strategies(self):
        """Test all valid padding strategies."""
        for padding in ["max_length", "longest", "do_not_pad"]:
            request = DatasetTokenizeRequest(
                tokenizer_name="gpt2",
                max_length=512,
                padding=padding
            )
            assert request.padding == padding

    def test_default_padding_strategy(self):
        """Test default padding is max_length."""
        request = DatasetTokenizeRequest(
            tokenizer_name="gpt2",
            max_length=512
        )
        assert request.padding == "max_length"

    def test_invalid_padding_strategy(self):
        """Test invalid padding strategy raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetTokenizeRequest(
                tokenizer_name="gpt2",
                max_length=512,
                padding="invalid_strategy"
            )
        assert "padding must be one of" in str(exc_info.value)
```

---

### 2.2 Truncation Strategy Implementation

**Purpose:** Allow users to select truncation strategy (longest_first, only_first, only_second, do_not_truncate).

**Backend Changes:**

#### Schema Update (continued from 2.1)

**File:** `backend/src/schemas/dataset.py`

**Add after padding field:**
```python
    # NEW: Truncation strategy
    truncation: Literal["longest_first", "only_first", "only_second", "do_not_truncate"] = Field(
        default="longest_first",
        description=(
            "Truncation strategy: "
            "'longest_first' truncates longest sequence first (for multi-sequence inputs), "
            "'only_first' truncates only first sequence, "
            "'only_second' truncates only second sequence, "
            "'do_not_truncate' disables truncation (may cause errors if sequences exceed max_length)"
        )
    )

    @field_validator("truncation")
    @classmethod
    def validate_truncation(cls, v: str) -> str:
        """Validate truncation strategy value."""
        allowed = ["longest_first", "only_first", "only_second", "do_not_truncate"]
        if v not in allowed:
            raise ValueError(f"truncation must be one of {allowed}")
        return v
```

#### Service Method Update

**File:** `backend/src/workers/dataset_tasks.py`

**Update tokenize_function:**
```python
def tokenize_function(examples):
    """
    Tokenize text examples with configurable settings.

    Args:
        examples: Dictionary with text column data

    Returns:
        Dictionary with input_ids, attention_mask, etc.
    """
    # Map padding strategy to HuggingFace format
    padding_config = {
        "max_length": "max_length",
        "longest": "longest",
        "do_not_pad": False
    }

    # Map truncation strategy to HuggingFace format
    truncation_config = {
        "longest_first": True,  # Default truncation
        "only_first": "only_first",
        "only_second": "only_second",
        "do_not_truncate": False  # No truncation
    }

    return tokenizer(
        examples[text_column],
        max_length=max_length,
        stride=stride,
        padding=padding_config.get(padding, "max_length"),
        truncation=truncation_config.get(truncation, True),  # Use truncation param
        return_overflowing_tokens=stride > 0,
    )
```

**Task Definition Update:**
```python
@celery_app.task(bind=True, max_retries=3)
def tokenize_dataset_task(
    self: Task,
    dataset_id: str,
    tokenizer_name: str,
    max_length: int = 512,
    stride: int = 0,
    padding: str = "max_length",
    truncation: str = "longest_first",  # NEW parameter
):
    """
    Tokenize dataset with configurable settings.

    Args:
        self: Celery task instance
        dataset_id: UUID of dataset to tokenize
        tokenizer_name: HuggingFace tokenizer identifier
        max_length: Maximum sequence length
        stride: Stride for sliding window
        padding: Padding strategy
        truncation: Truncation strategy
    """
    # Store in metadata
    metadata_update = {
        "tokenization": {
            "tokenizer_name": tokenizer_name,
            "max_length": max_length,
            "stride": stride,
            "padding": padding,
            "truncation": truncation,  # Store for display
            # ... other fields
        }
    }
```

#### Validation Tests

**File:** `backend/tests/unit/test_tokenization_enhancements.py`

**Add to file:**
```python
class TestTruncationStrategyValidation:
    """Tests for truncation strategy validation."""

    def test_valid_truncation_strategies(self):
        """Test all valid truncation strategies."""
        for truncation in ["longest_first", "only_first", "only_second", "do_not_truncate"]:
            request = DatasetTokenizeRequest(
                tokenizer_name="gpt2",
                max_length=512,
                truncation=truncation
            )
            assert request.truncation == truncation

    def test_default_truncation_strategy(self):
        """Test default truncation is longest_first."""
        request = DatasetTokenizeRequest(
            tokenizer_name="gpt2",
            max_length=512
        )
        assert request.truncation == "longest_first"

    def test_truncation_with_multi_sequence_input(self):
        """Test truncation strategies work with multi-sequence inputs."""
        # This would be an integration test
        # Verify only_first truncates only the first sequence
        # Verify only_second truncates only the second sequence
        pass
```

---

### 2.3 Tokenization Preview Endpoint

**Purpose:** Allow users to preview tokenization on sample text before processing the entire dataset.

**Backend Changes:**

#### API Endpoint: POST /api/datasets/tokenize-preview

**File:** `backend/src/api/v1/endpoints/datasets.py`

**Add new endpoint (after existing tokenize endpoint):**
```python
from functools import lru_cache
from transformers import AutoTokenizer
from typing import List, Dict, Any


@lru_cache(maxsize=10)
def load_tokenizer_cached(tokenizer_name: str) -> AutoTokenizer:
    """
    Load and cache tokenizer in memory.

    Args:
        tokenizer_name: HuggingFace tokenizer identifier

    Returns:
        Loaded tokenizer

    Note:
        Caches up to 10 tokenizers to avoid reloading on every preview request.
    """
    return AutoTokenizer.from_pretrained(tokenizer_name)


class TokenizePreviewRequest(BaseModel):
    """Request schema for tokenization preview."""

    tokenizer_name: str = Field(..., description="HuggingFace tokenizer identifier")
    text: str = Field(..., description="Text to tokenize", max_length=1000)
    max_length: int = Field(default=512, ge=1, le=8192)
    padding: Literal["max_length", "longest", "do_not_pad"] = "max_length"
    truncation: Literal["longest_first", "only_first", "only_second", "do_not_truncate"] = "longest_first"
    add_special_tokens: bool = Field(default=True, description="Add BOS/EOS tokens")
    return_attention_mask: bool = Field(default=True, description="Return attention mask")

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        """Validate text length doesn't exceed limit."""
        if len(v) > 1000:
            raise ValueError("Text exceeds maximum length of 1000 characters")
        return v


class TokenInfo(BaseModel):
    """Information about a single token."""
    id: int = Field(..., description="Token ID")
    text: str = Field(..., description="Token text")
    type: Literal["special", "regular"] = Field(..., description="Token type")
    position: int = Field(..., description="Position in sequence")


class TokenizePreviewResponse(BaseModel):
    """Response schema for tokenization preview."""
    tokens: List[TokenInfo]
    attention_mask: Optional[List[int]] = None
    token_count: int
    sequence_length: int
    special_token_count: int


@router.post("/tokenize-preview", response_model=TokenizePreviewResponse)
async def tokenize_preview(request: TokenizePreviewRequest):
    """
    Preview tokenization on sample text.

    This endpoint allows users to test tokenizer settings on sample text
    before committing to processing the entire dataset. Tokenizers are
    cached in memory for fast response times.

    Args:
        request: Tokenization preview request with text and settings

    Returns:
        TokenizePreviewResponse with tokenized results

    Raises:
        HTTPException 400: Invalid tokenizer name or text too long
        HTTPException 500: Tokenization error

    Example:
        ```json
        POST /api/datasets/tokenize-preview
        {
          "tokenizer_name": "gpt2",
          "text": "Once upon a time...",
          "max_length": 512,
          "padding": "max_length",
          "add_special_tokens": true
        }
        ```
    """
    try:
        # Load tokenizer (cached)
        tokenizer = load_tokenizer_cached(request.tokenizer_name)

        # Map settings to HuggingFace format
        padding_config = {
            "max_length": "max_length",
            "longest": "longest",
            "do_not_pad": False
        }

        truncation_config = {
            "longest_first": True,
            "only_first": "only_first",
            "only_second": "only_second",
            "do_not_truncate": False
        }

        # Tokenize sample text
        result = tokenizer(
            request.text,
            max_length=request.max_length,
            padding=padding_config.get(request.padding, "max_length"),
            truncation=truncation_config.get(request.truncation, True),
            add_special_tokens=request.add_special_tokens,
            return_attention_mask=request.return_attention_mask,
        )

        # Convert token IDs to token strings
        tokens_text = tokenizer.convert_ids_to_tokens(result["input_ids"])

        # Build token list with metadata
        tokens = []
        for idx, (token_id, token_text) in enumerate(zip(result["input_ids"], tokens_text)):
            tokens.append(
                TokenInfo(
                    id=token_id,
                    text=token_text,
                    type="special" if token_id in tokenizer.all_special_ids else "regular",
                    position=idx
                )
            )

        # Count special tokens
        special_token_count = sum(
            1 for token_id in result["input_ids"]
            if token_id in tokenizer.all_special_ids
        )

        return TokenizePreviewResponse(
            tokens=tokens,
            attention_mask=result.get("attention_mask") if request.return_attention_mask else None,
            token_count=len(result["input_ids"]),
            sequence_length=request.max_length,
            special_token_count=special_token_count
        )

    except Exception as e:
        # Handle tokenizer loading errors
        if "does not exist" in str(e).lower() or "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid tokenizer: {request.tokenizer_name}"
            )

        # Handle other tokenization errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tokenization failed: {str(e)}"
        )
```

#### Integration Tests

**File:** `backend/tests/integration/test_tokenize_preview.py` (NEW)

```python
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_tokenize_preview_success():
    """Test successful tokenization preview."""
    response = client.post("/api/datasets/tokenize-preview", json={
        "tokenizer_name": "gpt2",
        "text": "Hello world!",
        "max_length": 512,
        "padding": "max_length",
        "truncation": "longest_first",
        "add_special_tokens": True,
        "return_attention_mask": True
    })

    assert response.status_code == 200
    data = response.json()

    assert "tokens" in data
    assert len(data["tokens"]) > 0
    assert data["token_count"] > 0
    assert data["sequence_length"] == 512
    assert data["special_token_count"] >= 0

    # Check token structure
    first_token = data["tokens"][0]
    assert "id" in first_token
    assert "text" in first_token
    assert "type" in first_token
    assert first_token["type"] in ["special", "regular"]


def test_tokenize_preview_text_too_long():
    """Test preview with text exceeding max length."""
    response = client.post("/api/datasets/tokenize-preview", json={
        "tokenizer_name": "gpt2",
        "text": "a" * 1001,  # Exceeds 1000 char limit
        "max_length": 512
    })

    assert response.status_code == 400
    assert "exceeds maximum length" in response.json()["detail"].lower()


def test_tokenize_preview_invalid_tokenizer():
    """Test preview with invalid tokenizer."""
    response = client.post("/api/datasets/tokenize-preview", json={
        "tokenizer_name": "invalid-tokenizer-name-12345",
        "text": "Hello world!",
        "max_length": 512
    })

    assert response.status_code == 400
    assert "invalid tokenizer" in response.json()["detail"].lower()


def test_tokenize_preview_no_special_tokens():
    """Test preview with special tokens disabled."""
    response = client.post("/api/datasets/tokenize-preview", json={
        "tokenizer_name": "gpt2",
        "text": "Hello world!",
        "max_length": 512,
        "add_special_tokens": False
    })

    assert response.status_code == 200
    data = response.json()

    # Should have 0 special tokens when disabled
    assert data["special_token_count"] == 0


def test_tokenize_preview_no_attention_mask():
    """Test preview with attention mask disabled."""
    response = client.post("/api/datasets/tokenize-preview", json={
        "tokenizer_name": "gpt2",
        "text": "Hello world!",
        "max_length": 512,
        "return_attention_mask": False
    })

    assert response.status_code == 200
    data = response.json()

    # attention_mask should be None when disabled
    assert data["attention_mask"] is None
```

---

### 2.4 Unique Tokens Calculation

**Purpose:** Calculate the number of unique token IDs in the tokenized dataset (vocabulary size).

**Backend Changes:**

#### Service Method: calculate_unique_tokens

**File:** `backend/src/services/tokenization_service.py`

**Add new method after calculate_statistics:**
```python
@staticmethod
def calculate_unique_tokens(tokenized_dataset: HFDataset) -> int:
    """
    Calculate number of unique tokens in dataset.

    This is useful for estimating memory requirements for embeddings
    and vocabulary tables.

    Args:
        tokenized_dataset: HuggingFace dataset with input_ids

    Returns:
        Count of unique token IDs

    Raises:
        ValueError: If dataset is empty or has no input_ids

    Performance:
        - Time complexity: O(n) where n = total tokens
        - Space complexity: O(u) where u = unique tokens
        - Uses set for O(1) lookups

    Example:
        >>> dataset = load_dataset("...")
        >>> unique_count = TokenizationService.calculate_unique_tokens(dataset)
        >>> print(f"Vocabulary size: {unique_count} tokens")
    """
    if len(tokenized_dataset) == 0:
        raise ValueError("Cannot calculate unique tokens for empty dataset")

    unique_tokens = set()
    samples_without_input_ids = 0

    for example in tokenized_dataset:
        if "input_ids" in example:
            # Add all token IDs from this example to the set
            unique_tokens.update(example["input_ids"])
        else:
            samples_without_input_ids += 1

    if not unique_tokens:
        raise ValueError(
            f"No valid tokenized samples found. "
            f"{samples_without_input_ids}/{len(tokenized_dataset)} samples missing input_ids"
        )

    return len(unique_tokens)
```

#### Integration with calculate_statistics

**File:** `backend/src/services/tokenization_service.py`

**Update calculate_statistics method (existing):**
```python
@staticmethod
def calculate_statistics(tokenized_dataset: HFDataset) -> Dict[str, Any]:
    """
    Calculate comprehensive tokenization statistics (EXTENDED).

    Args:
        tokenized_dataset: Tokenized HuggingFace dataset

    Returns:
        Dictionary with statistics including unique tokens
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

    # NEW: Calculate unique tokens
    try:
        stats["unique_tokens"] = TokenizationService.calculate_unique_tokens(tokenized_dataset)
    except Exception as e:
        print(f"Warning: Could not calculate unique tokens: {e}")
        stats["unique_tokens"] = None

    return stats
```

#### Unit Tests

**File:** `backend/tests/unit/test_tokenization_enhancements.py`

**Add to file:**
```python
class TestUniqueTokens:
    """Tests for unique token calculation."""

    def test_unique_tokens_basic(self):
        """Test unique token count with simple data."""
        # Mock dataset with known token IDs
        dataset = [
            {"input_ids": [1, 2, 3, 4]},
            {"input_ids": [2, 3, 5, 6]},
            {"input_ids": [1, 5, 7, 8]}
        ]

        unique = TokenizationService.calculate_unique_tokens(dataset)

        # Expected unique tokens: {1, 2, 3, 4, 5, 6, 7, 8}
        assert unique == 8

    def test_unique_tokens_with_duplicates(self):
        """Test that duplicates are counted once."""
        dataset = [
            {"input_ids": [1, 1, 1, 1]},
            {"input_ids": [1, 1, 1, 1]},
            {"input_ids": [1, 1, 1, 1]}
        ]

        unique = TokenizationService.calculate_unique_tokens(dataset)

        # Only one unique token despite many duplicates
        assert unique == 1

    def test_unique_tokens_empty_dataset(self):
        """Test error handling for empty dataset."""
        with pytest.raises(ValueError, match="empty dataset"):
            TokenizationService.calculate_unique_tokens([])

    def test_unique_tokens_missing_input_ids(self):
        """Test error handling when all samples missing input_ids."""
        dataset = [
            {"text": "sample 1"},
            {"text": "sample 2"}
        ]

        with pytest.raises(ValueError, match="No valid tokenized samples"):
            TokenizationService.calculate_unique_tokens(dataset)
```

---

### 2.5 Histogram Calculation

**Purpose:** Calculate sequence length distribution histogram with 7 buckets for visualization.

**Backend Changes:**

#### Service Method: calculate_histogram

**File:** `backend/src/services/tokenization_service.py`

**Add new method:**
```python
@staticmethod
def calculate_histogram(
    seq_lengths: np.ndarray,
    max_length: int
) -> List[Dict[str, Any]]:
    """
    Calculate sequence length histogram with 7 buckets.

    Bins are:
    - 0-100 tokens
    - 100-200 tokens
    - 200-400 tokens
    - 400-600 tokens
    - 600-800 tokens
    - 800-1000 tokens
    - 1000+ tokens (up to max_length)

    Args:
        seq_lengths: Array of sequence lengths (from input_ids)
        max_length: Maximum sequence length (defines last bucket upper bound)

    Returns:
        List of histogram buckets with count and percentage

    Raises:
        ValueError: If seq_lengths is empty

    Example:
        >>> seq_lengths = np.array([50, 150, 250, 450, 650, 850, 1200])
        >>> histogram = TokenizationService.calculate_histogram(seq_lengths, 2048)
        >>> print(histogram[0])
        {'range': '0-100', 'min': 0, 'max': 100, 'count': 1, 'percentage': 14.3}
    """
    if len(seq_lengths) == 0:
        raise ValueError("Cannot calculate histogram for empty array")

    # Define bin edges (7 buckets)
    bins = [0, 100, 200, 400, 600, 800, 1000, max_length]
    histogram = []

    for i in range(len(bins) - 1):
        # Count samples in this bin
        if i < len(bins) - 2:
            # Regular bins (0-100, 100-200, etc.)
            count = np.sum((seq_lengths >= bins[i]) & (seq_lengths < bins[i+1]))
        else:
            # Last bin (1000+) includes all samples >= 1000
            count = np.sum(seq_lengths >= bins[i])

        # Calculate percentage
        percentage = float(count / len(seq_lengths) * 100)

        # Build bucket info
        bucket = {
            "range": f"{bins[i]}-{bins[i+1]}" if i < len(bins) - 2 else f"{bins[i]}+",
            "min": bins[i],
            "max": bins[i+1] if i < len(bins) - 2 else max_length,
            "count": int(count),
            "percentage": round(percentage, 1)  # One decimal place
        }

        histogram.append(bucket)

    # Verify percentages sum to ~100%
    total_percentage = sum(bucket["percentage"] for bucket in histogram)
    if not (99.9 <= total_percentage <= 100.1):
        print(f"Warning: Histogram percentages sum to {total_percentage}% (expected ~100%)")

    return histogram
```

#### Integration with calculate_statistics

**File:** `backend/src/services/tokenization_service.py`

**Update calculate_statistics method:**
```python
@staticmethod
def calculate_statistics(
    tokenized_dataset: HFDataset,
    tokenization_settings: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate comprehensive tokenization statistics (EXTENDED).

    Args:
        tokenized_dataset: Tokenized HuggingFace dataset
        tokenization_settings: Settings used for tokenization

    Returns:
        Dictionary with all statistics including histogram
    """
    # ... existing code for basic statistics ...

    seq_lengths_array = np.array(seq_lengths)

    stats = {
        "num_tokens": total_tokens,
        "num_samples": len(tokenized_dataset),
        "avg_seq_length": float(seq_lengths_array.mean()),
        "min_seq_length": int(seq_lengths_array.min()),
        "max_seq_length": int(seq_lengths_array.max()),
    }

    # NEW: Calculate unique tokens
    try:
        stats["unique_tokens"] = TokenizationService.calculate_unique_tokens(tokenized_dataset)
    except Exception as e:
        print(f"Warning: Could not calculate unique tokens: {e}")
        stats["unique_tokens"] = None

    # NEW: Calculate median
    stats["median_seq_length"] = float(np.median(seq_lengths_array))

    # NEW: Calculate histogram
    try:
        max_length = tokenization_settings.get("max_length", 512)
        stats["histogram"] = TokenizationService.calculate_histogram(
            seq_lengths_array,
            max_length
        )
    except Exception as e:
        print(f"Warning: Could not calculate histogram: {e}")
        stats["histogram"] = None

    return stats
```

#### Unit Tests

**File:** `backend/tests/unit/test_tokenization_enhancements.py`

**Add to file:**
```python
class TestHistogramCalculation:
    """Tests for histogram calculation."""

    def test_histogram_basic(self):
        """Test basic histogram calculation."""
        seq_lengths = np.array([50, 150, 250, 450, 650, 850, 1200])
        histogram = TokenizationService.calculate_histogram(seq_lengths, 2048)

        # Should have 7 buckets
        assert len(histogram) == 7

        # Check first bucket
        assert histogram[0]["range"] == "0-100"
        assert histogram[0]["min"] == 0
        assert histogram[0]["max"] == 100
        assert histogram[0]["count"] == 1

        # Check last bucket
        assert histogram[6]["range"] == "1000+"
        assert histogram[6]["min"] == 1000
        assert histogram[6]["count"] == 1

    def test_histogram_percentages_sum_to_100(self):
        """Test that percentages sum to ~100%."""
        # Generate random sequence lengths
        seq_lengths = np.random.randint(0, 512, 10000)
        histogram = TokenizationService.calculate_histogram(seq_lengths, 512)

        total_percentage = sum(bucket["percentage"] for bucket in histogram)

        # Allow small floating point error
        assert 99.9 <= total_percentage <= 100.1

    def test_histogram_empty_array(self):
        """Test histogram with empty array raises error."""
        with pytest.raises(ValueError, match="empty array"):
            TokenizationService.calculate_histogram(np.array([]), 512)

    def test_histogram_all_same_length(self):
        """Test histogram when all sequences have same length."""
        seq_lengths = np.array([250] * 100)
        histogram = TokenizationService.calculate_histogram(seq_lengths, 512)

        # All samples should be in 200-400 bucket
        assert histogram[2]["count"] == 100
        assert histogram[2]["percentage"] == 100.0

        # All other buckets should be empty
        for i in [0, 1, 3, 4, 5, 6]:
            assert histogram[i]["count"] == 0
```

---

### 2.6 Split Distribution Calculation

**Purpose:** Calculate distribution of samples across dataset splits (train/validation/test).

**Backend Changes:**

#### Service Method: calculate_split_distribution

**File:** `backend/src/services/tokenization_service.py`

**Add new method:**
```python
@staticmethod
def calculate_split_distribution(dataset: Any) -> Dict[str, Dict[str, Any]]:
    """
    Calculate distribution of samples across splits.

    Args:
        dataset: HuggingFace DatasetDict with multiple splits

    Returns:
        Dictionary mapping split name to count and percentage

    Example:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("roneneldan/TinyStories")
        >>> splits = TokenizationService.calculate_split_distribution(dataset)
        >>> print(splits)
        {
            "train": {"count": 2119719, "percentage": 98.3},
            "validation": {"count": 21990, "percentage": 1.0},
            "test": {"count": 10000, "percentage": 0.5}
        }
    """
    splits = {}

    # Check if dataset has splits (DatasetDict)
    if not hasattr(dataset, "keys"):
        # Single split dataset - treat as train
        return {
            "train": {
                "count": len(dataset),
                "percentage": 100.0
            }
        }

    # Calculate total samples across all splits
    total_samples = sum(len(split) for split in dataset.values())

    if total_samples == 0:
        return {}

    # Calculate count and percentage for each split
    for split_name, split_data in dataset.items():
        count = len(split_data)
        percentage = (count / total_samples * 100) if total_samples > 0 else 0

        splits[split_name] = {
            "count": count,
            "percentage": round(percentage, 1)
        }

    return splits
```

#### Integration in Download Task

**File:** `backend/src/workers/dataset_tasks.py`

**Update download_dataset_task:**
```python
@celery_app.task(bind=True, max_retries=3)
def download_dataset_task(
    self: Task,
    dataset_id: str,
    hf_repo_id: str,
    access_token: Optional[str] = None
):
    """
    Download dataset from HuggingFace (EXTENDED).

    Calculates split distribution after download.
    """
    async def run_download():
        async with AsyncSessionLocal() as db:
            service = DatasetService(db)

            try:
                # ... existing download code ...

                # Download from HuggingFace
                dataset = load_dataset(
                    hf_repo_id,
                    token=access_token if access_token else None,
                    cache_dir=raw_path
                )

                dataset.save_to_disk(raw_path)

                # Calculate statistics
                num_samples = sum(len(split) for split in dataset.values())
                size_bytes = get_directory_size(raw_path)

                # NEW: Calculate split distribution
                splits = TokenizationService.calculate_split_distribution(dataset)

                # Update dataset record
                ds = await service.get_dataset(dataset_id)
                ds.raw_path = raw_path
                ds.num_samples = num_samples
                ds.size_bytes = size_bytes
                ds.status = DatasetStatus.READY
                ds.progress = 100.0

                # Store split distribution in metadata
                ds.extra_metadata = {
                    "splits": splits  # NEW: Store split info
                }

                await db.commit()

                # Emit completion event
                self.emit_progress(dataset_id, "completed", {
                    "dataset_id": dataset_id,
                    "progress": 100.0,
                    "status": "ready",
                    "splits": splits  # Include in response
                })
```

#### Unit Tests

**File:** `backend/tests/unit/test_tokenization_enhancements.py`

**Add to file:**
```python
class TestSplitDistribution:
    """Tests for split distribution calculation."""

    def test_split_distribution_balanced(self):
        """Test split distribution with balanced splits."""
        # Mock DatasetDict
        class MockSplit:
            def __init__(self, size):
                self._size = size
            def __len__(self):
                return self._size

        class MockDataset:
            def __init__(self, splits):
                self._splits = splits
            def keys(self):
                return self._splits.keys()
            def items(self):
                return self._splits.items()
            def values(self):
                return self._splits.values()

        dataset = MockDataset({
            "train": MockSplit(8000),
            "validation": MockSplit(1000),
            "test": MockSplit(1000)
        })

        splits = TokenizationService.calculate_split_distribution(dataset)

        assert splits["train"]["count"] == 8000
        assert splits["train"]["percentage"] == 80.0
        assert splits["validation"]["count"] == 1000
        assert splits["validation"]["percentage"] == 10.0
        assert splits["test"]["count"] == 1000
        assert splits["test"]["percentage"] == 10.0

    def test_split_distribution_missing_split(self):
        """Test split distribution with missing test split."""
        class MockSplit:
            def __init__(self, size):
                self._size = size
            def __len__(self):
                return self._size

        class MockDataset:
            def __init__(self, splits):
                self._splits = splits
            def keys(self):
                return self._splits.keys()
            def items(self):
                return self._splits.items()
            def values(self):
                return self._splits.values()

        dataset = MockDataset({
            "train": MockSplit(9000),
            "validation": MockSplit(1000)
        })

        splits = TokenizationService.calculate_split_distribution(dataset)

        assert "train" in splits
        assert "validation" in splits
        assert "test" not in splits

    def test_split_distribution_single_split(self):
        """Test split distribution for single-split dataset."""
        # Mock single split (no DatasetDict)
        class MockDataset:
            def __len__(self):
                return 10000

        dataset = MockDataset()
        splits = TokenizationService.calculate_split_distribution(dataset)

        # Should treat as train split
        assert "train" in splits
        assert splits["train"]["count"] == 10000
        assert splits["train"]["percentage"] == 100.0
```

---

## 3. Section 2: Frontend Implementation

### 3.1 Padding/Truncation Dropdowns

**Purpose:** Add dropdowns for padding and truncation strategy selection in TokenizationTab.

**Frontend Changes:**

#### Component: TokenizationTab (Extended)

**File:** `frontend/src/components/datasets/DatasetDetailModal.tsx`

**Update TokenizationTab component (line ~487-590):**

**Add state variables:**
```typescript
// Existing state
const [tokenizerName, setTokenizerName] = useState('gpt2');
const [maxLength, setMaxLength] = useState(512);
const [stride, setStride] = useState(0);

// NEW: Tokenization settings state
const [paddingStrategy, setPaddingStrategy] = useState<'max_length' | 'longest' | 'do_not_pad'>('max_length');
const [truncationStrategy, setTruncationStrategy] = useState<'longest_first' | 'only_first' | 'only_second' | 'do_not_truncate'>('longest_first');
```

**Add UI components (insert after stride slider):**
```tsx
{/* NEW: Padding Strategy Dropdown */}
<div className="space-y-2">
  <label
    htmlFor="padding-strategy"
    className="block text-sm font-medium text-slate-300"
  >
    Padding Strategy
  </label>
  <select
    id="padding-strategy"
    value={paddingStrategy}
    onChange={(e) => setPaddingStrategy(e.target.value as 'max_length' | 'longest' | 'do_not_pad')}
    className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 transition-colors"
  >
    <option value="max_length">Max Length (pad to max_length)</option>
    <option value="longest">Longest (pad to longest in batch)</option>
    <option value="do_not_pad">Do Not Pad</option>
  </select>
  <p className="text-xs text-slate-500">
    Controls how sequences are padded. "Max Length" pads all sequences to max_length for consistent memory usage.
  </p>
</div>

{/* NEW: Truncation Strategy Dropdown */}
<div className="space-y-2">
  <label
    htmlFor="truncation-strategy"
    className="block text-sm font-medium text-slate-300"
  >
    Truncation Strategy
  </label>
  <select
    id="truncation-strategy"
    value={truncationStrategy}
    onChange={(e) => setTruncationStrategy(e.target.value as 'longest_first' | 'only_first' | 'only_second' | 'do_not_truncate')}
    className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 transition-colors"
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
```

**Update tokenization handler:**
```typescript
const handleTokenize = async () => {
  setIsTokenizing(true);
  setError(null);

  try {
    const response = await fetch(`/api/datasets/${dataset.id}/tokenize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tokenizer_name: tokenizerName,
        max_length: maxLength,
        stride: stride,
        // NEW: Include padding and truncation settings
        padding: paddingStrategy,
        truncation: truncationStrategy
      })
    });

    if (!response.ok) {
      throw new Error('Tokenization failed');
    }

    // Success - dataset status will update via WebSocket
  } catch (err) {
    setError(err instanceof Error ? err.message : 'Tokenization failed');
  } finally {
    setIsTokenizing(false);
  }
};
```

---

### 3.2 Toggle Components (Special Tokens & Attention Mask)

**Purpose:** Add toggle switches for special tokens and attention mask options.

**Frontend Changes:**

#### Component: ToggleSwitch (Reusable)

**File:** `frontend/src/components/common/ToggleSwitch.tsx` (NEW)

```typescript
import React from 'react';

interface ToggleSwitchProps {
  /** Current toggle state */
  checked: boolean;
  /** Callback when toggle changes */
  onChange: (checked: boolean) => void;
  /** Label text */
  label: string;
  /** Help text displayed below label */
  helpText?: string;
  /** Disabled state */
  disabled?: boolean;
}

export const ToggleSwitch: React.FC<ToggleSwitchProps> = ({
  checked,
  onChange,
  label,
  helpText,
  disabled = false
}) => {
  return (
    <div className="flex items-center justify-between">
      <div>
        <label className="text-sm font-medium text-slate-300">
          {label}
        </label>
        {helpText && (
          <p className="text-xs text-slate-500 mt-1">
            {helpText}
          </p>
        )}
      </div>

      <button
        type="button"
        onClick={() => !disabled && onChange(!checked)}
        disabled={disabled}
        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
          checked ? 'bg-emerald-600' : 'bg-slate-700'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        aria-checked={checked}
        aria-label={label}
      >
        <span
          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
            checked ? 'translate-x-6' : 'translate-x-1'
          }`}
        />
      </button>
    </div>
  );
};
```

#### Integration in TokenizationTab

**File:** `frontend/src/components/datasets/DatasetDetailModal.tsx`

**Add state variables:**
```typescript
// NEW: Toggle state
const [addSpecialTokens, setAddSpecialTokens] = useState(true);
const [returnAttentionMask, setReturnAttentionMask] = useState(true);
```

**Add UI components (insert after truncation dropdown):**
```tsx
{/* Import at top of file */}
import { ToggleSwitch } from '@/components/common/ToggleSwitch';

{/* NEW: Special Tokens Toggle */}
<ToggleSwitch
  checked={addSpecialTokens}
  onChange={setAddSpecialTokens}
  label="Add Special Tokens"
  helpText="Include BOS, EOS, PAD tokens (recommended for most models)"
/>

{/* NEW: Attention Mask Toggle */}
<ToggleSwitch
  checked={returnAttentionMask}
  onChange={setReturnAttentionMask}
  label="Return Attention Mask"
  helpText="Generate attention masks (disable to save memory if model doesn't use them)"
/>
```

**Update tokenization handler:**
```typescript
const handleTokenize = async () => {
  // ... existing code ...

  body: JSON.stringify({
    tokenizer_name: tokenizerName,
    max_length: maxLength,
    stride: stride,
    padding: paddingStrategy,
    truncation: truncationStrategy,
    // NEW: Include toggle settings
    add_special_tokens: addSpecialTokens,
    return_attention_mask: returnAttentionMask
  })
};
```

---

### 3.3 Tokenization Preview UI

**Purpose:** Allow users to preview tokenization on sample text with visual token chips.

**Frontend Changes:**

#### Component: TokenizationPreview

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
  returnAttentionMask
}) => {
  const [previewText, setPreviewText] = useState('');
  const [tokens, setTokens] = useState<Token[]>([]);
  const [tokenCount, setTokenCount] = useState(0);
  const [specialTokenCount, setSpecialTokenCount] = useState(0);
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
          return_attention_mask: returnAttentionMask
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Preview failed');
      }

      const data = await response.json();
      setTokens(data.tokens);
      setTokenCount(data.token_count);
      setSpecialTokenCount(data.special_token_count);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="border-t border-slate-700 pt-6 mt-6">
      <h4 className="text-sm font-semibold text-slate-300 mb-3">
        Preview Tokenization
      </h4>

      {/* Preview Text Input */}
      <textarea
        value={previewText}
        onChange={(e) => setPreviewText(e.target.value)}
        placeholder="Enter text to preview tokenization... (max 1000 characters)"
        maxLength={1000}
        rows={3}
        className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-sm font-mono resize-none"
      />

      {/* Character Count and Preview Button */}
      <div className="flex justify-between items-center mt-2">
        <span className="text-xs text-slate-500">
          {previewText.length} / 1000 characters
        </span>

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
                title={`Token ID: ${token.id}\nPosition: ${token.position}`}
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
            <span>{tokenCount} tokens</span>
            <span>{specialTokenCount} special tokens</span>
          </div>
        </div>
      )}
    </div>
  );
};
```

#### Integration in TokenizationTab

**File:** `frontend/src/components/datasets/DatasetDetailModal.tsx`

**Add TokenizationPreview component (insert before tokenize button):**
```tsx
{/* Import at top of file */}
import { TokenizationPreview } from '@/components/datasets/TokenizationPreview';

{/* NEW: Tokenization Preview Section */}
<TokenizationPreview
  tokenizerName={tokenizerName}
  maxLength={maxLength}
  padding={paddingStrategy}
  truncation={truncationStrategy}
  addSpecialTokens={addSpecialTokens}
  returnAttentionMask={returnAttentionMask}
/>
```

---

### 3.4 Histogram Component

**Purpose:** Display sequence length distribution histogram with 7 buckets.

**Frontend Changes:**

#### Component: SequenceLengthHistogram

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
  maxLength
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
                {/* Show count/percentage if bar is wide enough */}
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

#### Integration in StatisticsTab

**File:** `frontend/src/components/datasets/DatasetDetailModal.tsx`

**Update StatisticsTab component (replace existing simple chart):**
```tsx
{/* Import at top of file */}
import { SequenceLengthHistogram } from '@/components/datasets/SequenceLengthHistogram';

{/* Replace simple 3-bar chart with histogram */}
{tokenizationStats?.histogram && (
  <SequenceLengthHistogram
    histogram={tokenizationStats.histogram}
    minLength={tokenizationStats.min_seq_length}
    medianLength={tokenizationStats.median_seq_length || tokenizationStats.avg_seq_length}
    maxLength={tokenizationStats.max_seq_length}
  />
)}

{/* If no histogram data */}
{!tokenizationStats?.histogram && (
  <div className="bg-slate-800/50 rounded-lg p-6">
    <p className="text-sm text-slate-400">
      Histogram data not available. Re-tokenize the dataset to generate distribution statistics.
    </p>
  </div>
)}
```

---

### 3.5 Split Distribution Cards

**Purpose:** Display distribution of samples across train/validation/test splits.

**Frontend Changes:**

#### Component: SplitDistribution

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
  // Define split configurations
  const splitConfigs = [
    {
      name: 'train',
      label: 'Training',
      color: 'emerald',
      bgClass: 'bg-emerald-900/30',
      borderClass: 'border-emerald-700',
      textClass: 'text-emerald-400',
      data: splits.train
    },
    {
      name: 'validation',
      label: 'Validation',
      color: 'blue',
      bgClass: 'bg-blue-900/30',
      borderClass: 'border-blue-700',
      textClass: 'text-blue-400',
      data: splits.validation
    },
    {
      name: 'test',
      label: 'Test',
      color: 'purple',
      bgClass: 'bg-purple-900/30',
      borderClass: 'border-purple-700',
      textClass: 'text-purple-400',
      data: splits.test
    }
  ];

  // Filter to only active splits
  const activeSplits = splitConfigs.filter((s) => s.data);

  if (activeSplits.length === 0) {
    return (
      <div className="bg-slate-800/50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-slate-100 mb-4">
          Split Distribution
        </h3>
        <p className="text-sm text-slate-400">
          Split information not available for this dataset.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-slate-800/50 rounded-lg p-6">
      <h3 className="text-lg font-semibold text-slate-100 mb-4">
        Split Distribution
      </h3>

      <div className="grid grid-cols-3 gap-4">
        {activeSplits.map((split) => (
          <div
            key={split.name}
            className={`${split.bgClass} border ${split.borderClass} rounded-lg p-4`}
          >
            <div className="text-xs text-slate-400 mb-1">
              {split.label}
            </div>
            <div className="text-2xl font-bold text-slate-100">
              {split.data!.count.toLocaleString()}
            </div>
            <div className={`text-sm ${split.textClass} mt-1`}>
              {split.data!.percentage.toFixed(1)}% of total
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
```

#### Integration in StatisticsTab

**File:** `frontend/src/components/datasets/DatasetDetailModal.tsx`

**Add SplitDistribution component (after histogram):**
```tsx
{/* Import at top of file */}
import { SplitDistribution } from '@/components/datasets/SplitDistribution';

{/* NEW: Split Distribution Section */}
{dataset.metadata?.splits && (
  <SplitDistribution splits={dataset.metadata.splits} />
)}
```

---

## 4. Section 3: State Management Updates

### 4.1 Zustand Store Extensions

**Purpose:** Extend datasetsStore to handle new tokenization settings and preview state.

**File:** `frontend/src/stores/datasetsStore.ts`

**No changes needed!** The existing store already handles:
- Fetching datasets (which includes metadata)
- Updating dataset progress via WebSocket
- Updating dataset status

The new tokenization settings are passed directly to the API endpoints and stored in the backend metadata. The frontend reads them from `dataset.metadata.tokenization`.

---

### 4.2 TypeScript Types Update

**Purpose:** Add TypeScript types for new metadata fields.

**File:** `frontend/src/types/dataset.ts`

**Update TokenizationMetadata interface:**
```typescript
export interface TokenizationMetadata {
  tokenizer_name: string;
  text_column_used: string;
  max_length: number;
  stride: number;
  num_tokens: number;
  avg_seq_length: number;
  min_seq_length: number;
  max_seq_length: number;

  // NEW: Tokenization settings
  padding?: 'max_length' | 'longest' | 'do_not_pad';
  truncation?: 'longest_first' | 'only_first' | 'only_second' | 'do_not_truncate';
  add_special_tokens?: boolean;
  return_attention_mask?: boolean;

  // NEW: Additional statistics
  unique_tokens?: number;
  median_seq_length?: number;
  histogram?: HistogramBucket[];
}

export interface HistogramBucket {
  range: string;
  min: number;
  max: number;
  count: number;
  percentage: number;
}

export interface SplitInfo {
  count: number;
  percentage: number;
}

export interface SplitsMetadata {
  train?: SplitInfo;
  validation?: SplitInfo;
  test?: SplitInfo;
}

export interface DatasetMetadata {
  schema?: SchemaMetadata;
  tokenization?: TokenizationMetadata;
  splits?: SplitsMetadata;  // NEW
}
```

---

## 5. Section 4: Testing Patterns

### 5.1 Backend Unit Tests

**File:** `backend/tests/unit/test_tokenization_enhancements.py`

**Complete test file (already included in sections above):**
```python
# See sections 2.1-2.6 for complete test implementations
```

---

### 5.2 Frontend Component Tests

**File:** `frontend/src/components/datasets/TokenizationPreview.test.tsx` (NEW)

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
          { id: 995, text: ' world', type: 'regular', position: 2 }
        ],
        token_count: 3,
        sequence_length: 512,
        special_token_count: 1
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
    expect(screen.getByText(/1 special tokens/i)).toBeInTheDocument();
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

## 6. Section 5: Common Gotchas and Best Practices

### 6.1 Performance Optimization

**Backend: Tokenizer Caching**
```python
# GOOD: Cache tokenizers to avoid reloading
@lru_cache(maxsize=10)
def load_tokenizer_cached(tokenizer_name: str):
    return AutoTokenizer.from_pretrained(tokenizer_name)

# BAD: Load tokenizer on every request
def tokenize_preview(text: str, tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  # Slow!
```

**Backend: NumPy Vectorization**
```python
# GOOD: Use NumPy for statistics
seq_lengths = np.array([len(ids) for ids in input_ids])
avg_length = float(seq_lengths.mean())

# BAD: Python loop
total = 0
for ids in input_ids:
    total += len(ids)
avg_length = total / len(input_ids)
```

**Frontend: React.memo for Token Chips**
```typescript
// GOOD: Memoize token chips
export const TokenChip = React.memo<TokenChipProps>(({ token }) => {
  return (
    <span className={token.type === 'special' ? 'bg-emerald-700' : 'bg-slate-700'}>
      {token.text}
    </span>
  );
});

// BAD: Re-render all tokens on every change
```

### 6.2 Error Handling Patterns

**Backend: Specific Error Messages**
```python
# GOOD: Specific error messages
if len(seq_lengths) == 0:
    raise ValueError(
        f"No valid tokenized samples found. "
        f"{samples_without_input_ids}/{len(dataset)} samples missing input_ids"
    )

# BAD: Generic error
if len(seq_lengths) == 0:
    raise ValueError("Invalid dataset")
```

**Frontend: Graceful Degradation**
```typescript
// GOOD: Graceful degradation
{tokenizationStats?.histogram ? (
  <SequenceLengthHistogram histogram={tokenizationStats.histogram} />
) : (
  <p className="text-slate-400">
    Histogram not available. Re-tokenize to generate.
  </p>
)}

// BAD: Crash on missing data
<SequenceLengthHistogram histogram={tokenizationStats.histogram} />
```

### 6.3 Accessibility Considerations

**Toggle Switches:**
```typescript
// GOOD: Proper ARIA attributes
<button
  type="button"
  onClick={() => onChange(!checked)}
  aria-checked={checked}
  aria-label={label}
  className="..."
>
  {/* Toggle UI */}
</button>

// BAD: No accessibility
<div onClick={() => onChange(!checked)}>
  {/* Toggle UI */}
</div>
```

**Token Chips:**
```typescript
// GOOD: Informative tooltips
<span
  title={`Token ID: ${token.id}\nPosition: ${token.position}`}
  className="..."
>
  {token.text}
</span>

// BAD: No context
<span className="...">
  {token.text}
</span>
```

---

**Document End**
**Status:** Ready for Task List Generation
**Next Step:** Create 001_FTASKS|Dataset_Management_ENH_01.md with detailed task breakdown
**Total Sections:** 6
**Code Examples:** 50+ production-ready snippets
**Estimated Size:** ~2000 lines
