"""JumpReLU's reconstruction loss is measured where its sparsity coefficient is defined.

`JumpReLUSAE.forward` compared the reconstruction with the RAW activations while
the L0 penalty is a fixed λ per active latent. The balance between the two then
depended on the hook point's activation scale. Norm-module captures (per-element
scale near 1) trained well at λ=1e-3; true resid_post on LFM2.5-1.2B (per-element
std ~0.03) made the whole reconstruction term ~0.001, and train_24e5e7d3 collapsed
to under one active latent per token. The standard SAE was moved to normalized
space by MIS-E2E-086; JumpReLU was missed.

These tests use inputs at a resid_post-like scale, so a raw-space loss cannot
agree with them by construction.

MUTATION CONTROLS (2026-09-14; each applied alone, this module run red, restored
byte-identically and checked by sha256):
  J1  reconstruction loss back on raw `x_hat, x`
        -> test_the_reconstruction_loss_does_not_depend_on_the_activation_scale,
           test_the_penalty_is_weighed_against_a_reconstruction_of_order_one,
           test_the_zero_ablation_loss_is_in_the_same_space_as_the_reconstruction
  J2  zero-ablation loss back on raw `x`
        -> test_the_zero_ablation_loss_is_in_the_same_space_as_the_reconstruction
"""

import torch

from src.ml.sparse_autoencoder import JumpReLUSAE

D_MODEL, D_SAE = 64, 256


def _model(normalize="constant_norm_rescale", sparsity_coeff=1e-3):
    torch.manual_seed(0)
    return JumpReLUSAE(
        d_model=D_MODEL, d_sae=D_SAE, sparsity_coeff=sparsity_coeff, initial_threshold=0.01,
        normalize_activations=normalize,
    )


def _resid_post_like(n=512, std=0.03):
    torch.manual_seed(1)
    return torch.randn(n, D_MODEL) * std + 0.01


def test_the_reconstruction_loss_does_not_depend_on_the_activation_scale():
    model = _model()
    x = _resid_post_like()

    _, _, at_small_scale = model(x)
    _, _, at_unit_scale = model(x / 0.03)

    assert torch.allclose(at_small_scale["loss_reconstruction"], at_unit_scale["loss_reconstruction"], rtol=1e-4)
    assert torch.allclose(at_small_scale["loss_l0"], at_unit_scale["loss_l0"], rtol=1e-4)


def test_the_penalty_is_weighed_against_a_reconstruction_of_order_one():
    """With per-sample norm rescaled to sqrt(d), a poor reconstruction costs O(1) per element.

    In raw space the same reconstruction costs ~std² ≈ 1e-3, below a single active
    latent's λ — the collapse."""
    model = _model()
    with torch.no_grad():
        model.W_dec.zero_()  # the worst reconstruction: the decoder bias alone
    _, _, losses = model(_resid_post_like())

    assert losses["loss_reconstruction"].item() > 0.1
    assert losses["loss_reconstruction"].item() > 100 * model.sparsity_coeff


def test_the_zero_ablation_loss_is_in_the_same_space_as_the_reconstruction():
    model = _model()
    with torch.no_grad():
        model.W_dec.zero_()  # the reconstruction IS the decoder bias, so the two losses coincide
    _, _, losses = model(_resid_post_like())

    assert torch.allclose(losses["loss_zero"], losses["loss_reconstruction"], rtol=1e-5)


def test_the_fvu_is_still_reported_in_raw_space():
    model = _model()
    x = _resid_post_like()
    x_hat, _, losses = model(x)

    assert torch.allclose(losses["fvu"], (x - x_hat).var() / (x.var() + 1e-8), rtol=1e-5)


def test_without_normalization_the_loss_is_the_raw_mse():
    model = _model(normalize="none")
    x = _resid_post_like()
    x_hat, _, losses = model(x)

    assert torch.allclose(losses["loss_reconstruction"], torch.nn.functional.mse_loss(x_hat, x), rtol=1e-6)
