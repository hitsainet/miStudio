"""Every architecture reports the standard, per-dimension-centred FVU — and the legacy one beside it.

SAE TRAINING REMEDIATION, ITEM 5 (2026-09-15). The only FVU miStudio computed was
JumpReLU's ``var(x - x_hat) / var(x)`` over every element with ONE global mean.
On LFM2.5-1.2B's residual stream — large, near-constant outlier dimensions — that
reads about 0.05 low (held-out L11 of train_6247e768: 0.26 legacy vs 0.32 centred),
and the other four architectures reported no FVU at all.

THE FIXTURE MUST MAKE THE FORMULAS DISAGREE. On data whose every dimension has
zero mean the two definitions coincide, and a test would pass whichever value was
stored under which key. ``_offset_batch`` gives one dimension a large constant
offset; ``test_the_fixture_separates_the_formulas`` is the guard that it still does.
The references here are written out independently in float64 — never through
``sae_metrics`` — so a wrong helper cannot agree with itself.

MUTATION CONTROLS (2026-09-15; each applied alone by the WS-EVAL runner, this file run,
source restored and verified by sha256). All went red:
  F1 centred FVU uses one global mean (x.mean())          -> fvu_centred_is_the_per_dimension_centred_value[all 6],
                                                             the_two_are_not_the_same_number_here[all 6], sequence_shaped, transcoder target
  F2 losses dict swaps legacy and centred                 -> both value tests [all 6], not_the_same_number [all 6], transcoder target
  F3 Standard SAE drops reconstruction_quality            -> [standard] and [anthropic]: centred, legacy, no_gradient, not_the_same
  F4 Skip SAE drops reconstruction_quality                -> [skip]: centred, legacy, no_gradient, not_the_same
  F5 Transcoder scored against its input                  -> [transcoder] centred, test_a_transcoder_is_scored_against_its_target
  F6 TopK drops reconstruction_quality                    -> [topk]: centred, legacy, no_gradient, not_the_same
  F7 JumpReLU FVU computed in normalised space            -> fvu_centred_is_the_per_dimension_centred_value[jumprelu]
  F8 FvuSums accumulates in float32                       -> TestChunkedSumsEqualOnePass::test_centred[1000, 333, 7, 1]
"""

import pytest
import torch

from src.ml.sae_metrics import FvuSums, fvu_pair
from src.ml.sparse_autoencoder import create_sae

D, LATENT, N = 16, 64, 512


def _offset_batch(seed: int = 0, n: int = N) -> torch.Tensor:
    """Small per-element noise, plus one dimension sitting far from zero and barely moving."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, D, generator=g) * 0.1
    x[:, 0] += 50.0
    x[:, 3] += -20.0
    return x


def _reference_legacy(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    x, r = x.double().reshape(-1), (x.double() - x_hat.double()).reshape(-1)
    return float(r.var() / (x.var() + 1e-8))


def _reference_centred(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    x2, xh = x.double().reshape(-1, D), x_hat.double().reshape(-1, D)
    return float(((x2 - xh) ** 2).sum() / ((x2 - x2.mean(0)) ** 2).sum())


ARCHITECTURES = [
    pytest.param("standard_saelens", {}, id="standard"),
    pytest.param("standard_anthropic", {}, id="anthropic"),
    pytest.param("skip", {}, id="skip"),
    pytest.param("transcoder", {}, id="transcoder"),
    pytest.param("topk", {"top_k": 4}, id="topk"),
    pytest.param("jumprelu", {}, id="jumprelu"),
]


def _forward(arch: str, kwargs: dict, x: torch.Tensor, target: torch.Tensor):
    torch.manual_seed(0)
    sae = create_sae(architecture_type=arch, hidden_dim=D, latent_dim=LATENT, **kwargs)
    # A reconstruction that is neither perfect nor useless: the SAE's bias learns
    # nothing at init, so give every architecture its data mean, as training does.
    with torch.no_grad():
        for name in ("b_dec", "b_pre", "decoder_bias"):
            param = getattr(sae, name, None)
            if isinstance(param, torch.nn.Parameter):
                normed, _ = sae.normalize(x) if hasattr(sae, "normalize") else (x, None)
                param.copy_(normed.mean(0))
                break
    if arch == "transcoder":
        x_hat, _, losses = sae(x, target, return_loss=True)
    else:
        x_hat, _, losses = sae(x, return_loss=True)
    return x_hat, losses


def test_the_fixture_separates_the_formulas():
    """NEGATIVE CONTROL on the fixture itself. Without the offsets the two agree."""
    x = _offset_batch()
    x_hat = x + torch.randn_like(x) * 0.05
    assert _reference_centred(x, x_hat) > 5 * _reference_legacy(x, x_hat)

    centred_data = x - x.mean(0)
    assert _reference_centred(centred_data, centred_data + (x_hat - x)) == pytest.approx(
        _reference_legacy(centred_data, centred_data + (x_hat - x)), rel=0.01
    ), "zero-mean data should make the formulas coincide — the offsets are what separate them"


@pytest.mark.parametrize("arch, kwargs", ARCHITECTURES)
class TestEveryArchitectureReportsBoth:
    def test_fvu_centred_is_the_per_dimension_centred_value(self, arch, kwargs):
        x = _offset_batch(1)
        target = _offset_batch(2) if arch == "transcoder" else x
        x_hat, losses = _forward(arch, kwargs, x, target)

        assert "fvu_centred" in losses, f"{arch} reports no centred FVU"
        assert float(losses["fvu_centred"]) == pytest.approx(
            _reference_centred(target, x_hat), rel=1e-4
        )

    def test_fvu_keeps_the_legacy_global_mean_meaning(self, arch, kwargs):
        x = _offset_batch(1)
        target = _offset_batch(2) if arch == "transcoder" else x
        x_hat, losses = _forward(arch, kwargs, x, target)

        assert "fvu" in losses, f"{arch} reports no legacy FVU"
        assert float(losses["fvu"]) == pytest.approx(_reference_legacy(target, x_hat), rel=1e-4)

    def test_the_two_are_not_the_same_number_here(self, arch, kwargs):
        """If they were, the two tests above could not tell which key holds which value."""
        x = _offset_batch(1)
        target = _offset_batch(2) if arch == "transcoder" else x
        _x_hat, losses = _forward(arch, kwargs, x, target)
        assert float(losses["fvu_centred"]) > 1.5 * float(losses["fvu"])

    def test_no_gradient_flows_through_either(self, arch, kwargs):
        x = _offset_batch(1)
        target = _offset_batch(2) if arch == "transcoder" else x
        _x_hat, losses = _forward(arch, kwargs, x, target)
        assert not losses["fvu"].requires_grad and not losses["fvu_centred"].requires_grad


def test_a_transcoder_is_scored_against_its_target_not_its_input():
    """Training passes (x, x), where input and target coincide; this fixture does not."""
    x = _offset_batch(1)
    target = _offset_batch(2) * 3.0
    x_hat, losses = _forward("transcoder", {}, x, target)
    assert float(losses["fvu_centred"]) == pytest.approx(_reference_centred(target, x_hat), rel=1e-4)
    assert float(losses["fvu_centred"]) != pytest.approx(_reference_centred(x, x_hat), rel=1e-2)


def test_a_sequence_shaped_batch_is_centred_over_tokens():
    x = _offset_batch(3).reshape(8, N // 8, D)
    x_hat = x + torch.randn_like(x) * 0.05
    legacy, centred = fvu_pair(x, x_hat)
    assert float(centred) == pytest.approx(_reference_centred(x, x_hat), rel=1e-4)
    assert float(legacy) == pytest.approx(_reference_legacy(x, x_hat), rel=1e-4)


class TestChunkedSumsEqualOnePass:
    """The held-out and post-run evaluations aggregate over chunks; the answer must not depend on it."""

    def _pair(self):
        x = _offset_batch(4, n=1000)
        return x, x + torch.randn_like(x) * 0.07

    @pytest.mark.parametrize("chunk", [1000, 333, 7, 1])
    def test_centred(self, chunk):
        x, x_hat = self._pair()
        sums = FvuSums()
        for i in range(0, x.shape[0], chunk):
            sums.update(x[i:i + chunk], x_hat[i:i + chunk])
        assert sums.centred() == pytest.approx(_reference_centred(x, x_hat), rel=1e-9)

    @pytest.mark.parametrize("chunk", [1000, 333, 7])
    def test_legacy(self, chunk):
        x, x_hat = self._pair()
        sums = FvuSums()
        for i in range(0, x.shape[0], chunk):
            sums.update(x[i:i + chunk], x_hat[i:i + chunk])
        assert sums.legacy() == pytest.approx(_reference_legacy(x, x_hat), rel=1e-6)

    def test_nothing_accumulated_is_none_not_zero(self):
        """0.0 means a perfect reconstruction; an empty evaluation has no FVU."""
        assert FvuSums().centred() is None and FvuSums().legacy() is None

    def test_a_width_change_between_chunks_is_refused(self):
        sums = FvuSums().update(torch.zeros(2, D), torch.zeros(2, D))
        with pytest.raises(ValueError):
            sums.update(torch.zeros(2, D + 1), torch.zeros(2, D + 1))
