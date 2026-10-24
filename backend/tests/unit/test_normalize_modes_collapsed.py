"""MIS-E2E-085 — `anthropic_rescale` was a second name for the same arithmetic.

`sqrt(dim / ‖x‖²)` and `sqrt(dim) / ‖x‖` are the same number. Measured, not
assumed: max |difference| 7.2e-7 on float32, i.e. epsilon. So the PPRD's "six
paper-grounded frameworks" were five plus an alias, and a user selecting
`standard_anthropic` to reproduce Templeton et al. got SAELens' method while any
comparison between the two measured float noise.

The docstring claimed `E[‖x‖²] = dim` — a DATASET expectation. The code computes
a PER-SAMPLE rescale. Those are genuinely different: a dataset-wide scalar
preserves relative magnitude between tokens, a per-sample one discards it. The
paper's method is not implemented at all.

Collapsed rather than reimplemented, and these tests pin BOTH halves of why:

  * one code path, so two copies of the same arithmetic cannot drift into being
    differently wrong; and
  * BEHAVIOUR UNCHANGED, because every SAE ever trained under
    `normalize_activations='anthropic_rescale'` was trained with these
    semantics. Changing what the string means would silently alter the meaning
    of existing artifacts rather than fix them.
"""

import inspect
import math

import pytest
import torch

from src.ml.sparse_autoencoder import normalize_activations, denormalize_activations


@pytest.mark.parametrize("shape", [(64, 768), (8, 16, 512), (1, 4)])
def test_the_two_names_produce_bit_identical_output(shape):
    """Not "close" — identical. They share one branch now."""
    torch.manual_seed(0)
    x = torch.randn(*shape) * 3.7
    dim = shape[-1]

    a, ca = normalize_activations(x, "constant_norm_rescale", dim)
    b, cb = normalize_activations(x, "anthropic_rescale", dim)

    assert torch.equal(a, b), "the alias must not drift from what it aliases"
    assert torch.equal(ca, cb)


def test_the_behaviour_is_the_pre_collapse_behaviour():
    """Existing checkpoints must keep meaning what they meant.

    The collapse is a de-duplication, not a semantic change. If this fails,
    every SAE trained under either name has been silently reinterpreted.
    """
    torch.manual_seed(1)
    x = torch.randn(32, 256) * 2.0
    dim = 256

    out, coeff = normalize_activations(x, "anthropic_rescale", dim)

    expected_coeff = math.sqrt(dim) / torch.clamp(
        x.norm(dim=-1, keepdim=True), min=1e-6
    )
    assert torch.allclose(coeff, expected_coeff, atol=0, rtol=0)
    assert torch.allclose(out, x * expected_coeff, atol=0, rtol=0)


def test_normalisation_still_round_trips_under_both_names():
    torch.manual_seed(2)
    x = torch.randn(16, 128) * 5.0
    for mode in ("constant_norm_rescale", "anthropic_rescale", "none"):
        out, coeff = normalize_activations(x, mode, 128)
        back = denormalize_activations(out, coeff, mode)
        assert torch.allclose(back, x, atol=1e-4), mode


def test_none_is_still_a_no_op_and_unknown_is_still_rejected():
    """The collapse must not have swallowed the other branches."""
    x = torch.randn(4, 8)
    out, coeff = normalize_activations(x, "none", 8)
    assert torch.equal(out, x)
    assert torch.equal(coeff, torch.ones_like(x[..., :1]))

    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize_activations(x, "definitely_not_a_mode", 8)


def test_the_docstring_no_longer_claims_a_property_the_code_lacks():
    """A docstring asserting `E[‖x‖²] = dim` is what stopped anyone looking.

    The finding's real damage was not the duplicate branch — it was that the
    documentation described a second method convincingly enough that nobody
    checked whether it existed.
    """
    doc = inspect.getdoc(normalize_activations) or ""
    assert "alias" in doc.lower(), "the docstring must say the two are the same"
    assert "7.2e-7" in doc, "record the measurement, not the impression"
    # It may still MENTION the claim in order to correct it; what it must not
    # do is present it as a mode's behaviour.
    assert "- 'anthropic_rescale':     scale so E[‖x‖²] = dim" not in doc


def test_there_is_exactly_one_rescale_implementation():
    """Two copies of the same arithmetic can only drift into being different."""
    src = inspect.getsource(normalize_activations)
    assert src.count("math.sqrt(dim)") == 1, (
        "the rescale is implemented more than once — collapse it to one branch"
    )
