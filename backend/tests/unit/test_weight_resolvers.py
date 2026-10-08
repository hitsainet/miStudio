"""Orientation pins for resolve_decoder_weight / resolve_encoder_weight
(016 Task 0.1, from 018 review R2-A4).

These GATE the IDL-32 cross-layer weight prior cos(W_dec(Li)[:,i], W_enc(Lj)[j,:])
used by 016's discovery ranking and 015's hazard detection: if either resolver
flips orientation for any SAE format family, the prior is silently garbage.
Covers all three families: nn.Linear encoders (Standard/TopK), tied weights,
and JumpReLU raw-parameter SAEs (compat-property access).
"""

import pytest

torch = pytest.importorskip("torch")

from src.ml.sparse_autoencoder import JumpReLUSAE, SparseAutoencoder, TopKSAE
from src.services.steering_service import (
    resolve_decoder_weight,
    resolve_encoder_weight,
)

D_MODEL, D_SAE = 8, 16


class TestOrientationByFormat:
    """Decoder resolves [d_model, d_sae]; encoder resolves [d_sae, d_model]."""

    def _pin(self, sae):
        dec = resolve_decoder_weight(sae)
        enc = resolve_encoder_weight(sae)
        assert dec is not None and enc is not None
        assert tuple(dec.shape) == (D_MODEL, D_SAE), f"decoder {tuple(dec.shape)}"
        assert tuple(enc.shape) == (D_SAE, D_MODEL), f"encoder {tuple(enc.shape)}"
        return dec, enc

    def test_standard_untied(self):
        self._pin(SparseAutoencoder(hidden_dim=D_MODEL, latent_dim=D_SAE))

    def test_standard_tied(self):
        dec, enc = self._pin(
            SparseAutoencoder(hidden_dim=D_MODEL, latent_dim=D_SAE, tied_weights=True))
        # Tied: encoder row i IS decoder column i — exact, not approximate.
        assert torch.allclose(dec[:, 3], enc[3, :])

    def test_topk(self):
        self._pin(TopKSAE(hidden_dim=D_MODEL, latent_dim=D_SAE, k=4))

    def test_jumprelu_untied(self):
        self._pin(JumpReLUSAE(d_model=D_MODEL, d_sae=D_SAE))

    def test_jumprelu_tied(self):
        dec, enc = self._pin(
            JumpReLUSAE(d_model=D_MODEL, d_sae=D_SAE, tied_weights=True))
        assert torch.allclose(dec[:, 5], enc[5, :])


class TestPriorSelfIdentity:
    """The IDL-32 prior of a tied feature with ITSELF must be exactly 1 —
    a flipped orientation in either resolver breaks this identity."""

    @pytest.mark.parametrize("make", [
        lambda: SparseAutoencoder(hidden_dim=D_MODEL, latent_dim=D_SAE, tied_weights=True),
        lambda: JumpReLUSAE(d_model=D_MODEL, d_sae=D_SAE, tied_weights=True),
    ])
    def test_self_cosine_is_one(self, make):
        sae = make()
        dec = resolve_decoder_weight(sae)
        enc = resolve_encoder_weight(sae)
        for i in (0, 7, D_SAE - 1):
            cos = torch.nn.functional.cosine_similarity(
                dec[:, i], enc[i, :], dim=0)
            assert torch.isclose(cos, torch.tensor(1.0), atol=1e-6)
