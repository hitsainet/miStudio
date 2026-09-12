"""The mixture the operator asked for is the mixture the run reads.

WHY THIS FILE EXISTS. Sources were pooled by concatenation and sampled
uniformly, so the mixture was proportional to PADDED ROW COUNT. Measured:
Bloomberg supplied 19.1% of sampled activations and 1.2% of real tokens, because
its documents are 17-token headlines in a 512-token window. A corpus chosen for
financial features contributed almost nothing but padding.
"""

import pytest

from src.services.dataset_mixture import (
    allocate_tokens,
    describe_mixture,
    normalise_weights,
)


class TestNormalisation:

    def test_weights_are_scaled_to_sum_to_one(self):
        assert normalise_weights([1, 3], 2) == [0.25, 0.75]

    def test_none_means_no_preference(self):
        assert normalise_weights(None, 3) is None

    def test_a_length_mismatch_is_refused(self):
        """Silently zipping would pair weights with the wrong sources."""
        with pytest.raises(ValueError, match="one-to-one"):
            normalise_weights([1.0, 1.0], 3)

    def test_negative_and_all_zero_weights_are_refused(self):
        with pytest.raises(ValueError):
            normalise_weights([1.0, -1.0], 2)
        with pytest.raises(ValueError):
            normalise_weights([0.0, 0.0], 2)


class TestAllocation:

    def test_no_weights_is_proportional_to_availability(self):
        """The historical behaviour — correct once padding is excluded."""
        alloc = allocate_tokens([800, 200], 500)
        assert alloc == [400, 100]

    def test_weights_override_availability(self):
        """The Bloomberg case: a small source can be asked for a real share."""
        alloc = allocate_tokens([1000, 1000], 200, weights=[0.9, 0.1])
        assert alloc == [180, 20]

    def test_the_whole_budget_is_allocated(self):
        for weights in (None, [0.5, 0.3, 0.2], [1, 1, 1]):
            alloc = allocate_tokens([500, 500, 500], 900, weights=weights)
            assert sum(alloc) == 900, f"budget not met for weights={weights}"

    def test_a_source_is_never_over_drawn(self):
        """Asking for more than a corpus holds must not invent tokens."""
        alloc = allocate_tokens([50, 1000], 600, weights=[0.5, 0.5])
        assert alloc[0] <= 50
        assert sum(alloc) == 600, "the shortfall must be taken up by the others"

    def test_shortfall_is_redistributed_not_silently_dropped(self):
        alloc = allocate_tokens([10, 1000], 500, weights=[0.5, 0.5])
        assert alloc[0] == 10
        assert alloc[1] == 490

    def test_it_cannot_exceed_what_exists_in_total(self):
        alloc = allocate_tokens([100, 100], 10_000)
        assert alloc == [100, 100]

    def test_a_zero_weight_source_is_excluded(self):
        alloc = allocate_tokens([500, 500], 300, weights=[0.0, 1.0])
        assert alloc[0] == 0
        assert alloc[1] == 300

    def test_degenerate_inputs(self):
        assert allocate_tokens([], 100) == []
        assert allocate_tokens([100, 100], 0) == [0, 0]
        assert allocate_tokens([0, 0], 50) == [0, 0]


class TestDescription:

    def test_it_states_the_realised_mixture(self):
        """The operator must be able to check intent against reality."""
        text = describe_mixture(["web", "chat"], [750, 250])
        assert "75.0%" in text and "25.0%" in text
        assert "web" in text and "chat" in text
