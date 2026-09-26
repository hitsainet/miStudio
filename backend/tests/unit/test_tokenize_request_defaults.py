"""The tokenization request defaults to the block length the corpus uses.

Every tokenization in the rebuilt corpus is packed to 2048-token blocks, and
the circuit-study extraction template reads 2048-token rows. A request that
omits `max_length` used to get 512, which builds a second, shorter tokenization
of the same dataset under a different directory -- one extraction's selector
would then have two candidates of different shapes to choose between.
"""

import pytest
from pydantic import ValidationError

from src.schemas.dataset import DatasetTokenizeRequest


def test_max_length_defaults_to_2048():
    assert DatasetTokenizeRequest(model_id="m_x").max_length == 2048


def test_an_explicit_max_length_still_wins():
    assert DatasetTokenizeRequest(model_id="m_x", max_length=512).max_length == 512


def test_stride_is_bounded_by_the_default_length():
    assert DatasetTokenizeRequest(model_id="m_x", stride=2048).stride == 2048
    with pytest.raises(ValidationError):
        DatasetTokenizeRequest(model_id="m_x", stride=2049)
