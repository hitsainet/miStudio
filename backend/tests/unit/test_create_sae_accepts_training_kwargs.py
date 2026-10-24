"""``create_sae`` accepts the keyword set the training task sends, for every architecture.

FOUND 2026-09-15 while building the post-run evaluation's SAE loader. The training
task calls ``create_sae`` with one keyword set for all architectures, including
``ste_bandwidth`` (a JumpReLU knob added later). ``create_sae`` filters JumpReLU-only
keywords out before constructing a Standard, Skip or Transcoder SAE — and the filter
set did not name ``ste_bandwidth``, so each of those raised TypeError from
``__init__``. No test built one through the training task's keywords: the placement
harness replaces ``create_sae`` with a stub.

MUTATION CONTROLS (2026-09-15; each applied alone, this file run, source restored and
verified by sha256). Both went red:
  K1 'ste_bandwidth' removed from the JumpReLU-only filter  -> builds[standard, standard_saelens, standard_anthropic, skip, transcoder]
  K2 Transcoder no longer filters normalize_activations    -> builds[transcoder]
"""

import pytest

from src.ml.sparse_autoencoder import create_sae

#: Exactly the keywords train_sae_task passes (workers/training_tasks.py), with the
#: values a default hyperparameter set produces.
TRAINING_TASK_KWARGS = dict(
    ghost_gradient_penalty=0.0,
    normalize_activations="constant_norm_rescale",
    top_k_sparsity=None,
    top_k=None,
    aux_k=None,
    aux_loss_alpha=None,
    initial_threshold=0.5,
    bandwidth=0.01,
    ste_bandwidth=0.5,
    sparsity_coeff=None,
    normalize_decoder=True,
)


@pytest.mark.parametrize(
    "architecture",
    ["standard", "standard_saelens", "standard_anthropic", "skip", "transcoder", "topk", "jumprelu"],
)
def test_every_architecture_builds_from_the_training_tasks_keywords(architecture):
    sae = create_sae(architecture_type=architecture, hidden_dim=8, latent_dim=16, l1_alpha=1e-3,
                     **TRAINING_TASK_KWARGS)
    assert sum(p.numel() for p in sae.parameters()) > 0


def test_the_jumprelu_ste_bandwidth_still_reaches_jumprelu():
    """NEGATIVE CONTROL: filtering it for the others must not drop it for JumpReLU."""
    sae = create_sae(architecture_type="jumprelu", hidden_dim=8, latent_dim=16,
                     **{**TRAINING_TASK_KWARGS, "ste_bandwidth": 0.123})
    assert sae.activation.ste_bandwidth == pytest.approx(0.123)
