# HP-2 Phase 3 Implementation Plan
## Worker Task Testing to Reach 60% Backend Coverage

**Status:** 📋 **READY FOR FUTURE EXECUTION**
**Prerequisites:** HP-2 Phase 2 Complete ✅ (52.56% coverage achieved)
**Estimated Effort:** 20-30 hours
**Target:** 60-65% backend coverage

---

## Executive Summary

This document provides a detailed implementation plan for HP-2 Phase 3, which focuses on testing worker task modules to close the remaining 7.44% gap to reach the 60% backend coverage target. Phase 3 is **planned but not yet executed**, waiting for dedicated time allocation.

**Why Phase 3 is Separate:**
- Worker tasks are significantly more complex than service layer code
- Require extensive mocking of GPU operations, file I/O, and external APIs
- Need careful fixture design for reproducible tests
- Estimated 20-30 hours of focused development time

**Current State After Phase 2:**
- Backend Coverage: 52.56%
- Service Layer: Excellent (5 services >90%)
- Worker Tasks: Poor (6-22% coverage) ← **Phase 3 Target**

---

## Phase 3 Goals

### Primary Objective
Increase backend coverage from **52.56% to 60%** through targeted worker task testing.

### Secondary Objectives
1. Establish mocking patterns for GPU operations
2. Create reusable test fixtures for worker tasks
3. Test critical error handling paths
4. Document worker testing best practices

---

## Target Files & Expected Impact

### 1. Training Tasks (Highest Priority)
**File:** `backend/src/workers/training_tasks.py`
- **Lines:** 521 total
- **Current Coverage:** 6.53% (34 lines)
- **Target Coverage:** 40-50% (~208-260 lines)
- **Expected Impact:** +2.5-3.0% overall backend coverage
- **Estimated Effort:** 8-10 hours

**Complexity Factors:**
- GPU operations (torch.cuda)
- Training loop with gradient accumulation
- Checkpoint save/restore
- Progress tracking and metrics logging
- Hook management for activation capture
- Mixed precision training (AMP)

### 2. Model Tasks (High Priority)
**File:** `backend/src/workers/model_tasks.py`
- **Lines:** 417 total
- **Current Coverage:** 8.63% (36 lines)
- **Target Coverage:** 40-50% (~167-209 lines)
- **Expected Impact:** +2.0-2.5% overall backend coverage
- **Estimated Effort:** 6-8 hours

**Complexity Factors:**
- HuggingFace API integration
- Model download and caching
- Quantization (bitsandbytes)
- Architecture extraction
- Memory estimation

### 3. Dataset Tasks (Medium Priority)
**File:** `backend/src/workers/dataset_tasks.py`
- **Lines:** 246 total
- **Current Coverage:** 9.35% (23 lines)
- **Target Coverage:** 40-50% (~98-123 lines)
- **Expected Impact:** +1.5-2.0% overall backend coverage
- **Estimated Effort:** 4-6 hours

**Complexity Factors:**
- HuggingFace dataset loading
- Local file ingestion
- Tokenization operations
- Data validation and streaming

### 4. Extraction Tasks (Lower Priority)
**File:** `backend/src/workers/extraction_tasks.py`
- **Lines:** 35 total
- **Current Coverage:** 22.86% (8 lines)
- **Target Coverage:** 60-70% (~21-25 lines)
- **Expected Impact:** +0.5% overall backend coverage
- **Estimated Effort:** 2-3 hours

**Complexity Factors:**
- Feature extraction coordination
- Database update patterns

---

## Implementation Strategy

### Phase 3A: Setup & Infrastructure (4-6 hours)

#### Task 3A.1: Create Mock Fixtures Directory
**File:** `backend/tests/fixtures/`

Create centralized mock fixtures:

```python
# tests/fixtures/mock_gpu.py
"""Mock fixtures for GPU operations."""
import pytest
from unittest.mock import Mock, MagicMock

@pytest.fixture
def mock_cuda_available():
    """Mock torch.cuda.is_available() returning True."""
    with patch('torch.cuda.is_available', return_value=True):
        yield

@pytest.fixture
def mock_cuda_device():
    """Mock GPU device properties."""
    device_props = Mock()
    device_props.total_memory = 16 * 1024**3  # 16GB
    device_props.name = 'NVIDIA RTX 4090'
    with patch('torch.cuda.get_device_properties', return_value=device_props):
        yield

@pytest.fixture
def mock_cuda_memory():
    """Mock CUDA memory allocation."""
    with patch('torch.cuda.memory_allocated', return_value=8 * 1024**3):
        with patch('torch.cuda.max_memory_allocated', return_value=12 * 1024**3):
            yield
```

```python
# tests/fixtures/mock_models.py
"""Mock fixtures for model operations."""
import pytest
from unittest.mock import Mock, MagicMock

@pytest.fixture
def mock_hf_model():
    """Mock HuggingFace model."""
    model = MagicMock()
    model.config.hidden_size = 768
    model.config.num_hidden_layers = 12
    model.config.num_attention_heads = 12
    return model

@pytest.fixture
def mock_model_download():
    """Mock model download from HuggingFace."""
    with patch('transformers.AutoModel.from_pretrained') as mock:
        mock.return_value = mock_hf_model()
        yield mock
```

```python
# tests/fixtures/mock_datasets.py
"""Mock fixtures for dataset operations."""
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_hf_dataset():
    """Mock HuggingFace dataset."""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=1000)
    dataset.__getitem__ = Mock(return_value={
        'text': 'Sample text',
        'input_ids': [101, 102, 103]
    })
    return dataset

@pytest.fixture
def mock_dataset_load():
    """Mock dataset loading from HuggingFace."""
    with patch('datasets.load_dataset') as mock:
        mock.return_value = mock_hf_dataset()
        yield mock
```

#### Task 3A.2: Create Conftest with Shared Fixtures
**File:** `backend/tests/unit/conftest.py` (update)

Add worker-specific fixtures to existing conftest:

```python
# Import all mock fixtures
pytest_plugins = [
    'tests.fixtures.mock_gpu',
    'tests.fixtures.mock_models',
    'tests.fixtures.mock_datasets',
]

@pytest.fixture
def mock_training_data():
    """Sample training configuration."""
    return {
        'id': 't_test_training_123',
        'model_id': 'm_test_model',
        'dataset_id': 'd_test_dataset',
        'hyperparameters': {
            'hidden_dim': 768,
            'latent_dim': 16384,
            'l1_alpha': 0.001,
            'learning_rate': 0.0001,
            'batch_size': 64,
            'total_steps': 10000
        }
    }
```

---

### Phase 3B: Training Tasks Tests (8-10 hours)

#### Task 3B.1: Test Training Task Initialization
**File:** `backend/tests/unit/test_training_tasks.py`

**Test Coverage Areas:**
1. Training initialization with mock GPU
2. Model and dataset loading
3. SAE creation with correct dimensions
4. Optimizer and scheduler setup
5. Error handling for missing resources

**Sample Test Structure:**

```python
"""
Unit tests for training_tasks.py worker functions.

Tests focus on business logic with mocked GPU/file operations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.workers.training_tasks import TrainingTask
from src.models.training import TrainingStatus


class TestTrainingTaskProgressUpdate:
    """Test TrainingTask.update_training_progress()."""

    def test_update_progress_success(self, async_session, mock_training_data):
        """Test successful progress update."""
        task = TrainingTask()

        # Create training record
        training = Training(
            id=mock_training_data['id'],
            status=TrainingStatus.RUNNING.value,
            progress=0.0
        )
        async_session.add(training)
        async_session.commit()

        # Update progress
        task.update_training_progress(
            training_id=mock_training_data['id'],
            step=100,
            total_steps=1000,
            loss=0.5,
            l0_sparsity=0.05
        )

        # Verify update
        async_session.refresh(training)
        assert training.progress == 10.0
        assert training.current_step == 100
        assert training.current_loss == 0.5
        assert training.current_l0_sparsity == 0.05


class TestTrainingTaskMetricLogging:
    """Test TrainingTask.log_metric()."""

    def test_log_metric_success(self, async_session, mock_training_data):
        """Test successful metric logging."""
        # Test implementation
        pass


@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.get_device_properties')
class TestTrainSAETaskExecution:
    """Test train_sae_task Celery task."""

    def test_training_initialization(
        self,
        mock_device_props,
        mock_cuda,
        async_session,
        mock_training_data
    ):
        """Test training task initializes correctly."""
        # Mock GPU properties
        mock_device_props.return_value.total_memory = 16 * 1024**3

        # Test implementation
        pass

    def test_training_oom_handling(
        self,
        mock_device_props,
        mock_cuda,
        async_session,
        mock_training_data
    ):
        """Test OOM error handling and recovery."""
        # Test implementation
        pass
```

**Test Checklist:**
- [ ] update_training_progress() - success case
- [ ] update_training_progress() - training not found
- [ ] log_metric() - success case
- [ ] train_sae_task - initialization
- [ ] train_sae_task - model loading with quantization
- [ ] train_sae_task - dataset loading
- [ ] train_sae_task - SAE creation
- [ ] train_sae_task - training loop (mocked iterations)
- [ ] train_sae_task - checkpoint save
- [ ] train_sae_task - OOM error handling
- [ ] train_sae_task - data corruption error
- [ ] train_sae_task - GPU memory error
- [ ] train_sae_task - completion and cleanup

---

### Phase 3C: Model Tasks Tests (6-8 hours)

#### Task 3C.1: Test Model Download Task
**File:** `backend/tests/unit/test_model_tasks.py`

**Test Coverage Areas:**
1. Model download initiation
2. Progress tracking during download
3. Model caching behavior
4. Quantization operations
5. Architecture extraction
6. Error handling (network, disk, memory)

**Sample Test Structure:**

```python
"""
Unit tests for model_tasks.py worker functions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.workers.model_tasks import download_model_task
from src.models.model import ModelStatus


@patch('transformers.AutoModel.from_pretrained')
@patch('transformers.AutoTokenizer.from_pretrained')
class TestDownloadModelTask:
    """Test download_model_task Celery task."""

    def test_download_success(
        self,
        mock_tokenizer,
        mock_model,
        async_session
    ):
        """Test successful model download."""
        # Mock model and tokenizer
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()

        # Test implementation
        pass

    def test_download_network_error(
        self,
        mock_tokenizer,
        mock_model,
        async_session
    ):
        """Test handling of network errors during download."""
        mock_model.side_effect = ConnectionError("Network unavailable")

        # Test implementation
        pass

    def test_download_with_quantization(
        self,
        mock_tokenizer,
        mock_model,
        async_session
    ):
        """Test model download with quantization."""
        # Test implementation
        pass


class TestModelArchitectureExtraction:
    """Test extract_model_architecture_task."""

    @patch('transformers.AutoConfig.from_pretrained')
    def test_architecture_extraction_success(
        self,
        mock_config,
        async_session
    ):
        """Test successful architecture extraction."""
        # Mock config
        mock_config.return_value.to_dict.return_value = {
            'hidden_size': 768,
            'num_hidden_layers': 12
        }

        # Test implementation
        pass
```

**Test Checklist:**
- [ ] download_model_task - success with FP16
- [ ] download_model_task - success with INT8 quantization
- [ ] download_model_task - success with INT4 quantization
- [ ] download_model_task - progress tracking
- [ ] download_model_task - network error
- [ ] download_model_task - disk space error
- [ ] download_model_task - invalid model ID
- [ ] extract_model_architecture_task - success
- [ ] extract_model_architecture_task - missing config
- [ ] cleanup_model_task - file deletion
- [ ] cleanup_model_task - error handling

---

### Phase 3D: Dataset Tasks Tests (4-6 hours)

#### Task 3D.1: Test Dataset Loading Task
**File:** `backend/tests/unit/test_dataset_tasks.py`

**Test Coverage Areas:**
1. HuggingFace dataset loading
2. Local file ingestion
3. Tokenization operations
4. Progress tracking
5. Data validation
6. Error handling

**Sample Test Structure:**

```python
"""
Unit tests for dataset_tasks.py worker functions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.workers.dataset_tasks import load_dataset_task


@patch('datasets.load_dataset')
class TestLoadDatasetTask:
    """Test load_dataset_task Celery task."""

    def test_load_hf_dataset_success(
        self,
        mock_load,
        async_session
    ):
        """Test successful HuggingFace dataset load."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_load.return_value = mock_dataset

        # Test implementation
        pass

    def test_load_dataset_streaming(
        self,
        mock_load,
        async_session
    ):
        """Test dataset loading with streaming."""
        # Test implementation
        pass


class TestDatasetTokenization:
    """Test tokenize_dataset_task."""

    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_tokenization_success(
        self,
        mock_tokenizer,
        async_session
    ):
        """Test successful dataset tokenization."""
        # Test implementation
        pass
```

**Test Checklist:**
- [ ] load_dataset_task - HF dataset success
- [ ] load_dataset_task - local file success
- [ ] load_dataset_task - streaming mode
- [ ] load_dataset_task - progress tracking
- [ ] load_dataset_task - network error
- [ ] load_dataset_task - invalid dataset ID
- [ ] tokenize_dataset_task - success
- [ ] tokenize_dataset_task - tokenization error
- [ ] process_dataset_task - validation success
- [ ] process_dataset_task - validation failure

---

### Phase 3E: Extraction Tasks Tests (2-3 hours)

#### Task 3E.1: Test Feature Extraction Task
**File:** `backend/tests/unit/test_extraction_tasks.py` (expand existing)

**Test Coverage Areas:**
1. Extraction task coordination
2. Database update patterns
3. Error handling
4. Status transitions

---

### Phase 3F: Verification & Documentation (2-3 hours)

#### Task 3F.1: Run Final Coverage Analysis
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

**Verify Targets:**
- [ ] Overall backend coverage ≥60%
- [ ] training_tasks.py ≥40%
- [ ] model_tasks.py ≥40%
- [ ] dataset_tasks.py ≥40%
- [ ] Service layer maintained >90% (5+ services)

#### Task 3F.2: Create Phase 3 Completion Summary
**File:** `0xcc/docs/HP2_Phase3_Completion_Summary.md`

Document:
- Final coverage metrics
- Tests created (count, lines, pass rate)
- Mocking patterns established
- Challenges encountered
- Recommendations for maintenance

#### Task 3F.3: Update Task Lists
Update:
- `SUPP_TASKS|Progress_Architecture_Improvements.md`
- Mark HP-2 as fully complete
- Document final backend coverage

---

## Mocking Patterns Reference

### Pattern 1: Mock GPU Operations

```python
@pytest.fixture
def mock_gpu_environment():
    """Complete mock GPU environment."""
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.device_count', return_value=1):
            with patch('torch.cuda.get_device_name', return_value='Mock GPU'):
                with patch('torch.cuda.memory_allocated', return_value=0):
                    yield
```

### Pattern 2: Mock Model Loading

```python
@pytest.fixture
def mock_model_loader():
    """Mock model loading from HuggingFace."""
    with patch('src.ml.model_loader.load_model_from_hf') as mock:
        model = MagicMock()
        model.config.hidden_size = 768
        model.to = Mock(return_value=model)
        mock.return_value = model
        yield mock
```

### Pattern 3: Mock File Operations

```python
@pytest.fixture
def mock_checkpoint_save():
    """Mock checkpoint save operations."""
    with patch('torch.save') as mock_save:
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.return_value = None
            yield mock_save
```

### Pattern 4: Mock Dataset Operations

```python
@pytest.fixture
def mock_dataset_iterator():
    """Mock dataset iterator for training."""
    def batch_generator():
        for i in range(10):
            yield {
                'input_ids': torch.randint(0, 1000, (64, 128)),
                'attention_mask': torch.ones(64, 128)
            }
    return batch_generator()
```

---

## Success Criteria

### Coverage Targets
- ✅ Overall Backend: ≥60% (from 52.56%)
- ✅ Worker Tasks Average: ≥40% (from ~9%)
- ✅ Service Layer: Maintain >90% for critical services
- ✅ Test Pass Rate: >85%

### Quality Metrics
- ✅ All critical error paths tested
- ✅ Reusable mock fixtures created
- ✅ Tests run in <2 minutes
- ✅ Clear documentation of mocking patterns

### Deliverables
- ✅ 3 new test files (training, model, dataset tasks)
- ✅ Mock fixtures directory with reusable mocks
- ✅ Phase 3 completion summary
- ✅ Updated task documentation

---

## Risk Assessment

### High Risk Items
1. **Complex Mocking Required**
   - Mitigation: Start with simple tests, iterate
   - Fallback: Accept lower coverage for complex paths

2. **GPU-Specific Code Hard to Test**
   - Mitigation: Mock all torch.cuda operations
   - Fallback: Focus on CPU code paths

3. **Long Test Execution Time**
   - Mitigation: Use minimal test data
   - Fallback: Mark slow tests with pytest markers

### Medium Risk Items
1. **Fixture Complexity**
   - Mitigation: Document fixtures thoroughly
   - Fallback: Use simpler, inline mocks

2. **Test Maintenance Burden**
   - Mitigation: Keep tests focused on logic
   - Fallback: Reduce coverage target to 55%

---

## Timeline Estimate

### Optimistic (20 hours - 1 week full-time)
- Phase 3A: 3 hours
- Phase 3B: 7 hours
- Phase 3C: 5 hours
- Phase 3D: 3 hours
- Phase 3E: 1 hour
- Phase 3F: 1 hour

### Realistic (25 hours - 1.5 weeks)
- Phase 3A: 5 hours
- Phase 3B: 8 hours
- Phase 3C: 6 hours
- Phase 3D: 4 hours
- Phase 3E: 2 hours
- Phase 3F: 2 hours (includes debugging)

### Conservative (30 hours - 2 weeks)
- Phase 3A: 6 hours (complex fixture setup)
- Phase 3B: 10 hours (debugging mocks)
- Phase 3C: 7 hours (quantization edge cases)
- Phase 3D: 5 hours (streaming complications)
- Phase 3E: 3 hours
- Phase 3F: 3 hours (documentation)

---

## Execution Prerequisites

### Required Before Starting
1. ✅ HP-2 Phase 2 complete (52.56% coverage)
2. ✅ Service layer tests passing (>85% rate)
3. ✅ Test patterns established and documented
4. ⏳ **Dedicated 20-30 hour time block allocated**
5. ⏳ **Team agreement on coverage targets**

### Nice to Have
- Integration test environment with actual GPU
- CI/CD pipeline for automated testing
- Code review process for test quality

---

## Alternative Approaches

### Alternative 1: Integration-Heavy Testing
**Approach:** Test with real GPU, models, datasets
- **Pros:** More realistic, catches real issues
- **Cons:** Slow (10-20 min), requires GPU, brittle
- **Coverage:** Could reach 65-70%
- **Time:** 30-40 hours

### Alternative 2: Minimal Critical Path Testing
**Approach:** Test only critical error paths, skip happy paths
- **Pros:** Faster (12-16 hours), focuses on risk
- **Cons:** Lower coverage (55-58%), less comprehensive
- **Coverage:** 55-58%
- **Time:** 12-16 hours

### Alternative 3: Recommended Balanced Approach
**Approach:** Mock unit tests + select integration tests
- **Pros:** Good coverage, maintainable, fast tests
- **Cons:** Requires careful mock design
- **Coverage:** 60-65%
- **Time:** 20-30 hours (as planned above)

---

## Conclusion

HP-2 Phase 3 is a well-defined path to reaching 60% backend coverage through systematic worker task testing. The plan is **ready for execution** when dedicated time is allocated.

**Current Recommendation:** Proceed with Phase 3 when project priorities allow for a dedicated 20-30 hour effort block (approximately 1-2 weeks).

**Status:** 📋 **PLANNED - AWAITING EXECUTION APPROVAL**

---

## Appendix: Quick Start Checklist

When ready to start Phase 3:

- [ ] Review this implementation plan
- [ ] Allocate 20-30 hour time block
- [ ] Create feature branch: `feature/hp2-phase3-worker-tests`
- [ ] Start with Phase 3A (mock fixtures setup)
- [ ] Follow test implementation order: Training → Model → Dataset
- [ ] Run coverage checks after each major test file
- [ ] Document learnings and patterns as you go
- [ ] Create Phase 3 completion summary
- [ ] Merge to main with coverage verification

**Ready to Execute:** When prerequisites met ✓
