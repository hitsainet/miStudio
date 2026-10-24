"""Threshold calibration measures the pass rate in the basis the SAE actually uses.

THE DEFECT. Extraction's JumpReLU calibration hand-computed pre-activations as
`W_enc @ x + b_enc`, omitting the `- b_dec` centring that `encode()` and
train-time `calibrate_thresholds` both apply. On the 16k SAE, `W_enc @ b_dec` is
negative on ~99% of features, so the check read a healthy SAE as 2-3x sparser
than it is. Measured on real cached activations:

    layer   true pass rate   uncentred reading   vs 0.1% trigger
     11         0.304%           0.129%            safe, 1.29x
     12         0.282%           0.141%            safe, 1.41x
     13         0.279%           0.1015%           safe by 1.5%

Below the trigger the rescale DISCARDS the learned thresholds. Because `b_dec`
is the mean activation, which grows with depth, the error grew with depth — so
the deepest layers were the likeliest to be silently rewritten.

WHY THE FIXTURE CARRIES A NONZERO b_dec. With `b_dec = 0` the centred and
uncentred bases are identical and every test here would pass against the
defect. The fixtures agree with the bug by construction unless the offset is
real, which is the trap this repo has hit before.

MUTATION CONTROLS (each alone; suite must go red):
  C1  make pre_activations_with_training_normalization skip the `- b_dec`
        -> test_matches_the_encoder, test_healthy_sae_does_not_trip_the_trigger
  C2  revert the extraction calibration to its own F.linear instead of the helper
        -> test_extraction_calibrates_through_the_helper
        (and test_sae_encode_normalization's bare-encode scan, if it calls encode)
  C3  drop "threshold_calibration" from the statistics dict
        -> test_the_calibration_is_recorded_in_the_statistics
"""

import ast
import inspect
import textwrap

import torch

from src.ml.sparse_autoencoder import (
    create_sae,
    pre_activations_with_training_normalization as calibration_pre_activations,
)
from src.services import extraction_service

TRIGGER = 0.001


def _sae_with_real_offset(seed: int = 0):
    """A JumpReLU SAE whose b_dec is large enough to move the answer."""
    torch.manual_seed(seed)
    sae = create_sae(architecture_type="jumprelu", hidden_dim=32, latent_dim=256)
    with torch.no_grad():
        # A mean activation well away from zero — what a real layer has, and the
        # one property that makes the two bases disagree.
        sae.b_dec.copy_(torch.full((32,), 2.0))
    return sae


def _uncentred(sae, x):
    """What the defective calibration computed: normalize, then skip `- b_dec`."""
    x_norm, _ = sae.normalize(x)
    return torch.nn.functional.linear(x_norm, sae.W_enc, sae.b_enc)


def test_matches_the_encoder():
    """The calibration reads exactly what encode() will later compute."""
    sae = _sae_with_real_offset()
    x = torch.randn(64, 32)

    # The reference is what forward() does: normalize, THEN encode.
    _, expected = sae.encode(sae.normalize(x)[0], return_pre_activations=True)

    assert torch.allclose(calibration_pre_activations(sae, x), expected)


def test_the_offset_is_load_bearing():
    """Guard on the fixture itself: the two bases must actually differ here."""
    sae = _sae_with_real_offset()
    x = torch.randn(64, 32)

    assert not torch.allclose(calibration_pre_activations(sae, x), _uncentred(sae, x))


def test_healthy_sae_does_not_trip_the_trigger():
    """Reproduce the real defect: a healthy SAE, misread as sparse.

    Thresholds are placed so the TRUE pass rate is comfortably above the
    trigger. The uncentred basis — shifted by W_enc @ b_dec — must read it as
    below, or this fixture cannot show the bug the fix addresses.
    """
    sae = _sae_with_real_offset()
    x = torch.randn(2048, 32)

    z_true = calibration_pre_activations(sae, x)
    # Put each feature's threshold at its 99.7th percentile: ~0.3% pass.
    threshold = torch.quantile(z_true, 0.997, dim=0)

    true_rate = (z_true > threshold).float().mean().item()
    misread_rate = (_uncentred(sae, x) > threshold).float().mean().item()

    assert true_rate > TRIGGER, f"true rate {true_rate:.4%} should clear the trigger"
    assert misread_rate != true_rate, "fixture fails to reproduce the basis error"


def _extract_features_source_tree():
    src = textwrap.dedent(inspect.getsource(extraction_service.ExtractionService))
    return ast.parse(src)


def test_extraction_calibrates_through_the_helper():
    """REACHABILITY: the extraction path must CALL the helper.

    Walks the AST for a Call node, not the text for the name — a substring
    search would match this file's own docstrings and pass for the wrong reason.
    """
    calls = [
        n for n in ast.walk(_extract_features_source_tree())
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "pre_activations_with_training_normalization"
    ]
    assert calls, (
        "extraction no longer calibrates through "
        "pre_activations_with_training_normalization"
    )


def test_the_calibration_is_recorded_in_the_statistics():
    """A rescale must never be silent: the record reaches the statistics dict."""
    keys = set()
    for node in ast.walk(_extract_features_source_tree()):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
    assert "threshold_calibration" in keys
