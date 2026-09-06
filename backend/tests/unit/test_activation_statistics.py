"""
Activation statistics must be measured, or reported as absent — never zero.

Reported 2026-08-27. A 73 GB extraction of gemma-4 layers 44 and 46 stored:

    max_activation: 29.859375   mean_magnitude: 0.0
    min_activation: 0.0         std_activation: 0.0    sparsity_percent: 7.11

Three of five statistics were exactly zero beside a valid maximum. The cause
was float16: activations are stored fp16, one chunk is 100 x 512 x 3840 =
196,608,000 elements, and `abs_chunk.sum()` reduces in the input dtype, whose
maximum is 65,504. It returned `inf` on the first chunk. The handler then did

    if np.isinf(mean_magnitude): mean_magnitude = 0.0

turning an overflow into a plausible-looking measurement, and
`max(0, nan) -> 0` did the same to the standard deviation.

Two further defects in the same block:
  * `variance = E[x^2] - E[|x|]^2` is not a variance.
  * min/max were taken over |x|, so "Min Activation" was 0 for any real tensor.
"""

import numpy as np
import pytest

from src.services.activation_service import ActivationService


def _svc():
    return ActivationService.__new__(ActivationService)


class TestFloat16DoesNotOverflow:
    def test_a_large_float16_array_does_not_report_zero(self):
        """The production shape, scaled down but still past fp16's range."""
        svc = _svc()
        # 4,000,000 elements of 1.5 sums to 6e6 — far beyond fp16's 65,504.
        array = np.full((100, 200, 200), 1.5, dtype=np.float16)

        stats = svc._chunked_statistics(array, chunk_size=10)

        assert stats["mean_magnitude"] == pytest.approx(1.5, rel=1e-6), (
            "the float16 accumulator overflowed and the failure was reported "
            "as a number"
        )
        assert stats["mean_magnitude"] != 0.0
        assert stats["std_activation"] == pytest.approx(0.0, abs=1e-6)

    def test_naive_float16_summation_really_does_overflow(self):
        """Negative control for the premise: prove the hazard is real."""
        array = np.full((100, 200, 200), 1.5, dtype=np.float16)
        assert np.isinf(np.abs(array).sum()), "premise no longer holds"
        assert not np.isinf(np.abs(array).sum(dtype=np.float64))


class TestTheStatisticsAreCorrect:
    def _array(self):
        rng = np.random.default_rng(0)
        return rng.normal(0, 3, size=(40, 50, 60)).astype(np.float16)

    def test_chunked_agrees_with_direct(self):
        """Two paths, one answer. They diverged silently before."""
        svc = _svc()
        array = self._array()

        chunked = svc._chunked_statistics(array, chunk_size=7)
        direct = svc._direct_statistics(array)

        for key in chunked:
            assert chunked[key] == pytest.approx(direct[key], rel=1e-6), key

    def test_they_agree_with_numpy_in_float64(self):
        svc = _svc()
        array = self._array()
        truth = array.astype(np.float64)

        stats = svc._chunked_statistics(array, chunk_size=7)

        assert stats["mean_magnitude"] == pytest.approx(np.abs(truth).mean(), rel=1e-6)
        assert stats["std_activation"] == pytest.approx(truth.std(), rel=1e-4)
        assert stats["min_activation"] == pytest.approx(truth.min(), rel=1e-6)
        assert stats["max_activation"] == pytest.approx(truth.max(), rel=1e-6)

    def test_min_is_the_real_minimum_not_the_minimum_magnitude(self):
        """min(|x|) is 0 for any real activation tensor and says nothing."""
        svc = _svc()
        array = np.array([[[-4.0, 0.0, 2.0]]], dtype=np.float16)

        stats = svc._direct_statistics(array)

        assert stats["min_activation"] == pytest.approx(-4.0)
        assert stats["min_activation"] != 0.0

    def test_variance_is_of_the_raw_values(self):
        """E[x^2] - E[|x|]^2 understates a zero-centred distribution."""
        svc = _svc()
        array = np.array([[[-2.0, 2.0, -2.0, 2.0]]], dtype=np.float32)

        stats = svc._direct_statistics(array)

        assert stats["std_activation"] == pytest.approx(2.0, rel=1e-6)
        # the old formula gave sqrt(4 - 4) = 0 for this input
        assert stats["std_activation"] != 0.0


class TestFailureIsReportedAsAbsence:
    def test_an_empty_array_yields_none_not_zero(self):
        svc = _svc()
        stats = svc._direct_statistics(np.zeros((0, 4, 4), dtype=np.float16))
        assert all(v is None for v in stats.values()), stats

    def test_non_finite_input_yields_none_not_zero(self):
        svc = _svc()
        array = np.array([[[np.inf, 1.0]]], dtype=np.float32)
        stats = svc._direct_statistics(array)
        assert stats["mean_magnitude"] is None
        assert stats["mean_magnitude"] != 0.0

    def test_an_all_zero_tensor_still_reports_zero(self):
        """A genuine zero must survive — absence and zero are different."""
        svc = _svc()
        stats = svc._direct_statistics(np.zeros((2, 4, 4), dtype=np.float16))
        assert stats["mean_magnitude"] == 0.0
        assert stats["sparsity_percent"] == pytest.approx(100.0)
