"""The template schema and the endpoint schema must agree on shared fields.

An extraction template is a SAVED extraction config, so any value a template
accepts must be one the endpoint accepts. They drifted:

    context_prefix/suffix_tokens   template le=100   endpoint le=50
    top_k_examples                 template ge=1     endpoint ge=10

Starting an extraction with 100 tokens of context — a value the template schema
advertised as valid — returned a bare `422 Unprocessable Entity` with no
indication which field was wrong:

    POST /api/v1/saes/{id}/extract-features ... 422 Unprocessable Entity

This test compares the two models field by field so the next divergence fails
here instead of at the point of use.
"""

import pytest

from src.schemas.extraction import ExtractionConfigRequest
from src.schemas.extraction_template import (
    ExtractionTemplateCreate,
    ExtractionTemplateUpdate,
)


def _bounds(model, name):
    """Extract (ge, le) for a field, or None when it is unconstrained."""
    f = model.model_fields.get(name)
    if f is None:
        return None
    ge = le = None
    for m in f.metadata:
        ge = getattr(m, "ge", ge)
        le = getattr(m, "le", le)
    return (ge, le)


SHARED = [
    "top_k_examples",
    "context_prefix_tokens",
    "context_suffix_tokens",
    "evaluation_samples",
    "min_activation_frequency",
]


class TestBoundsAgree:
    @pytest.mark.parametrize("field", SHARED)
    def test_template_create_matches_the_endpoint(self, field):
        endpoint = _bounds(ExtractionConfigRequest, field)
        template = _bounds(ExtractionTemplateCreate, field)
        if endpoint is None or template is None:
            pytest.skip(f"{field} is not on both models")
        assert template == endpoint, (
            f"{field}: template allows {template}, endpoint allows {endpoint}. "
            f"A template saved at the template's limit would be refused by the "
            f"endpoint with a bare 422."
        )

    @pytest.mark.parametrize("field", SHARED)
    def test_template_update_matches_the_endpoint(self, field):
        endpoint = _bounds(ExtractionConfigRequest, field)
        template = _bounds(ExtractionTemplateUpdate, field)
        if endpoint is None or template is None:
            pytest.skip(f"{field} is not on both models")
        assert template == endpoint, f"{field}: {template} vs {endpoint}"


class TestTheValuesThatBrokeIt:
    def test_100_tokens_of_context_is_accepted(self):
        """The exact request that returned 422."""
        cfg = ExtractionConfigRequest(
            context_prefix_tokens=100, context_suffix_tokens=100,
            top_k_examples=15,
        )
        assert cfg.context_prefix_tokens == 100
        assert cfg.context_suffix_tokens == 100
        assert cfg.top_k_examples == 15

    def test_the_same_values_are_storable_as_a_template(self):
        t = ExtractionTemplateCreate(
            name="100+100 ctx, 15 examples",
            layer_indices=[45], hook_types=["residual"],
            context_prefix_tokens=100, context_suffix_tokens=100,
            top_k_examples=15,
        )
        assert t.context_prefix_tokens == 100

    def test_beyond_the_ceiling_is_still_refused_by_both(self):
        with pytest.raises(Exception):
            ExtractionConfigRequest(context_prefix_tokens=101)
        with pytest.raises(Exception):
            ExtractionTemplateCreate(
                name="x", layer_indices=[45], hook_types=["residual"],
                context_prefix_tokens=101,
            )
