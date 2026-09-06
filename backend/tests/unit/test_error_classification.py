"""
Unit tests for error classification logic.

Tests the classify_extraction_error function that classifies errors
and suggests retry parameters.
"""

import pytest
from src.workers.model_tasks import classify_extraction_error
from src.ml.model_loader import OutOfMemoryError
from src.services.activation_service import ActivationExtractionError


class TestOOMErrorClassification:
    """Test OOM (Out of Memory) error classification."""

    def test_classify_oom_error_via_exception_type(self):
        """Test OOM classification via OutOfMemoryError exception type."""
        error = OutOfMemoryError("CUDA out of memory")
        batch_size = 64

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "OOM"
        assert suggested_params == {"batch_size": 32}  # Half of 64

    def test_classify_oom_error_via_string_match_out_of_memory(self):
        """Test OOM classification via 'out of memory' string."""
        error = RuntimeError("out of memory: tried to allocate 4GB")
        batch_size = 32

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "OOM"
        assert suggested_params == {"batch_size": 16}  # Half of 32

    def test_classify_oom_error_via_string_match_cuda_oom(self):
        """Test OOM classification via 'cuda oom' string."""
        error = RuntimeError("CUDA OOM: Failed to allocate memory")
        batch_size = 128

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "OOM"
        assert suggested_params == {"batch_size": 64}  # Half of 128

    def test_suggest_retry_params_for_oom_minimum_batch_size(self):
        """Test OOM retry params respects minimum batch size of 1."""
        error = OutOfMemoryError("CUDA out of memory")
        batch_size = 1  # Already at minimum

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "OOM"
        assert suggested_params == {"batch_size": 1}  # Cannot go below 1

    def test_suggest_retry_params_for_oom_batch_size_2(self):
        """Test OOM retry params with batch_size=2."""
        error = OutOfMemoryError("CUDA out of memory")
        batch_size = 2

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "OOM"
        assert suggested_params == {"batch_size": 1}  # 2 // 2 = 1


class TestValidationErrorClassification:
    """Test VALIDATION error classification."""

    def test_classify_validation_error_not_found(self):
        """Test VALIDATION classification for 'not found' errors."""
        error = ActivationExtractionError("Model not found")
        batch_size = 64

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "VALIDATION"
        assert suggested_params == {}  # No retry params for validation errors

    def test_classify_validation_error_not_ready(self):
        """Test VALIDATION classification for 'not ready' errors."""
        error = ActivationExtractionError("Dataset not ready")
        batch_size = 64

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "VALIDATION"
        assert suggested_params == {}

    def test_classify_validation_error_case_insensitive(self):
        """Test VALIDATION classification is case-insensitive."""
        error = ActivationExtractionError("Model NOT FOUND")
        batch_size = 64

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "VALIDATION"


class TestExtractionErrorClassification:
    """Test EXTRACTION error classification."""

    def test_classify_extraction_error_generic(self):
        """Test EXTRACTION classification for generic ActivationExtractionError."""
        error = ActivationExtractionError("Failed to extract activations")
        batch_size = 64

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "EXTRACTION"
        assert suggested_params == {}

    def test_classify_extraction_error_hook_failure(self):
        """Test EXTRACTION classification for hook-related errors."""
        error = ActivationExtractionError("Hook registration failed")
        batch_size = 64

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "EXTRACTION"
        assert suggested_params == {}


class TestTimeoutErrorClassification:
    """Test TIMEOUT error classification."""

    def test_classify_timeout_error_via_string_timeout(self):
        """Test TIMEOUT classification via 'timeout' string."""
        error = TimeoutError("Operation timeout after 300 seconds")
        batch_size = 64

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "TIMEOUT"
        assert suggested_params == {"batch_size": 32}  # Half of 64

    def test_classify_timeout_error_via_string_timed_out(self):
        """Test TIMEOUT classification via 'timed out' string."""
        error = RuntimeError("Request timed out")
        batch_size = 128

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "TIMEOUT"
        assert suggested_params == {"batch_size": 64}  # Half of 128

    def test_classify_timeout_error_minimum_batch_size(self):
        """Test TIMEOUT retry params respects minimum batch size of 1."""
        error = TimeoutError("Operation timeout")
        batch_size = 1

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "TIMEOUT"
        assert suggested_params == {"batch_size": 1}  # Cannot go below 1


class TestUnknownErrorClassification:
    """Test UNKNOWN error classification."""

    def test_classify_unknown_error_generic_exception(self):
        """Test UNKNOWN classification for generic exceptions."""
        error = Exception("Something went wrong")
        batch_size = 64

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "UNKNOWN"
        assert suggested_params == {}

    def test_classify_unknown_error_value_error(self):
        """Test UNKNOWN classification for ValueError."""
        error = ValueError("Invalid parameter")
        batch_size = 64

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "UNKNOWN"
        assert suggested_params == {}

    def test_classify_unknown_error_type_error(self):
        """Test UNKNOWN classification for TypeError."""
        error = TypeError("Expected int, got str")
        batch_size = 64

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "UNKNOWN"
        assert suggested_params == {}


class TestErrorClassificationRetryParams:
    """Test retry parameter suggestions."""

    def test_retry_params_batch_size_reduction_powers_of_two(self):
        """Test batch size reduction maintains powers of two when possible."""
        # Test common batch sizes
        batch_sizes = [128, 64, 32, 16, 8, 4, 2, 1]
        error = OutOfMemoryError("OOM")

        for batch_size in batch_sizes:
            error_type, suggested_params = classify_extraction_error(error, batch_size)

            expected_batch_size = max(1, batch_size // 2)
            assert suggested_params["batch_size"] == expected_batch_size

    def test_retry_params_batch_size_reduction_odd_numbers(self):
        """Test batch size reduction with odd numbers."""
        # Test odd batch sizes
        test_cases = [
            (33, 16),   # 33 // 2 = 16
            (17, 8),    # 17 // 2 = 8
            (9, 4),     # 9 // 2 = 4
            (5, 2),     # 5 // 2 = 2
            (3, 1),     # 3 // 2 = 1
        ]

        error = OutOfMemoryError("OOM")

        for batch_size, expected in test_cases:
            error_type, suggested_params = classify_extraction_error(error, batch_size)
            assert suggested_params["batch_size"] == expected

    def test_retry_params_empty_for_non_resource_errors(self):
        """Test retry params empty for errors that don't need resource adjustment."""
        error = ActivationExtractionError("Generic error")
        batch_size = 64

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert suggested_params == {}


class TestErrorClassificationCaseInsensitivity:
    """Test error classification is case-insensitive."""

    def test_oom_error_case_insensitive(self):
        """Test OOM classification is case-insensitive."""
        test_messages = [
            "out of memory",
            "Out Of Memory",
            "OUT OF MEMORY",
            "cuda oom",
            "CUDA OOM",
            "Cuda Oom",
        ]

        for message in test_messages:
            error = RuntimeError(message)
            error_type, _ = classify_extraction_error(error, 64)
            assert error_type == "OOM", f"Failed for message: {message}"

    def test_timeout_error_case_insensitive(self):
        """Test TIMEOUT classification is case-insensitive."""
        test_messages = [
            "timeout",
            "Timeout",
            "TIMEOUT",
            "timed out",
            "Timed Out",
            "TIMED OUT",
        ]

        for message in test_messages:
            error = RuntimeError(message)
            error_type, _ = classify_extraction_error(error, 64)
            assert error_type == "TIMEOUT", f"Failed for message: {message}"

    def test_validation_error_case_insensitive(self):
        """Test VALIDATION classification is case-insensitive."""
        test_messages = [
            "not found",
            "Not Found",
            "NOT FOUND",
            "not ready",
            "Not Ready",
            "NOT READY",
        ]

        for message in test_messages:
            error = ActivationExtractionError(message)
            error_type, _ = classify_extraction_error(error, 64)
            assert error_type == "VALIDATION", f"Failed for message: {message}"


class TestErrorClassificationEdgeCases:
    """Test error classification edge cases."""

    def test_classify_error_with_empty_message(self):
        """Test classification with empty error message."""
        error = Exception("")
        batch_size = 64

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "UNKNOWN"
        assert suggested_params == {}

    def test_classify_error_with_multiline_message(self):
        """Test classification with multiline error message."""
        error = RuntimeError("CUDA error\nout of memory\nAllocated: 8GB")
        batch_size = 64

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        # Should still detect OOM in multiline message
        assert error_type == "OOM"
        assert suggested_params == {"batch_size": 32}

    def test_classify_error_with_very_large_batch_size(self):
        """Test classification with very large batch size."""
        error = OutOfMemoryError("OOM")
        batch_size = 10000

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        assert error_type == "OOM"
        assert suggested_params == {"batch_size": 5000}  # Half of 10000

    def test_classify_multiple_error_indicators(self):
        """Test classification when multiple error indicators present."""
        # Message contains both OOM and timeout, OOM should take precedence
        error = RuntimeError("Operation timed out due to out of memory")
        batch_size = 64

        error_type, suggested_params = classify_extraction_error(error, batch_size)

        # OOM check comes first in the if-elif chain, so it takes precedence
        assert error_type == "OOM"
        assert suggested_params == {"batch_size": 32}
