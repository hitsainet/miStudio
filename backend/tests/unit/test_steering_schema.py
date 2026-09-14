"""
Unit tests for steering request schemas (Feature 011).

Covers the Feature 011 changes:
- selected_features now accepts up to 20 features (was 4) on both the
  combined (blended) and comparison paths.
- The unique-color validator was dropped: duplicate colors are allowed.
- The color palette was widened from 4 to 20 named colors.
"""

import pytest
from pydantic import ValidationError

from src.schemas.steering import (
    CombinedSteeringRequest,
    SelectedFeature,
    SteeringComparisonRequest,
)

# The 20-name palette (original 4 first, for continuity) — must stay in
# lock-step with the frontend FeatureColor union.
PALETTE = [
    "teal", "blue", "purple", "amber", "rose", "cyan", "lime", "orange",
    "fuchsia", "sky", "emerald", "violet", "pink", "indigo", "yellow",
    "red", "green", "sapphire", "magenta", "gold",
]


def _feature(idx: int, color: str = "teal") -> SelectedFeature:
    return SelectedFeature(feature_idx=idx, layer=6, strength=2.0, color=color)


@pytest.mark.parametrize(
    "request_cls", [CombinedSteeringRequest, SteeringComparisonRequest]
)
def test_accepts_twenty_features(request_cls):
    """Both steering paths accept the full 20-feature selection."""
    feats = [_feature(i, PALETTE[i]) for i in range(20)]
    req = request_cls(sae_id="s", prompt="hi", selected_features=feats)
    assert len(req.selected_features) == 20


@pytest.mark.parametrize(
    "request_cls", [CombinedSteeringRequest, SteeringComparisonRequest]
)
def test_rejects_twenty_one_features(request_cls):
    """A 21st feature is rejected by the max_length=20 constraint."""
    feats = [_feature(i, PALETTE[i]) for i in range(20)] + [_feature(99)]
    with pytest.raises(ValidationError):
        request_cls(sae_id="s", prompt="hi", selected_features=feats)


@pytest.mark.parametrize(
    "request_cls", [CombinedSteeringRequest, SteeringComparisonRequest]
)
def test_requires_at_least_one_feature(request_cls):
    """An empty selection is still rejected (min_length=1)."""
    with pytest.raises(ValidationError):
        request_cls(sae_id="s", prompt="hi", selected_features=[])


def test_duplicate_colors_allowed():
    """Feature 011 dropped the unique-color validator — dupes are fine now."""
    feats = [_feature(1, "teal"), _feature(2, "teal")]
    req = SteeringComparisonRequest(sae_id="s", prompt="hi", selected_features=feats)
    assert [f.color for f in req.selected_features] == ["teal", "teal"]


def test_new_palette_colors_accepted():
    """The 16 new palette colors are valid."""
    for color in PALETTE[4:]:
        assert _feature(1, color).color == color


def test_unknown_color_rejected():
    """A color outside the 20-name palette is still rejected."""
    with pytest.raises(ValidationError):
        _feature(1, "chartreuse")
