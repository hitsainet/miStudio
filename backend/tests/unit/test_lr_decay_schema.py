"""`lr_decay_steps` in the training schema (tracker item 7).

Validation: `warmup_steps + lr_decay_steps <= total_steps`. Templates validate
through the same model, so a template cannot store a schedule that no run can
follow.

MUTATION CONTROL (2026-09-15, WS-LOOP): B12, the validator's condition replaced by
`if False:` -> test_warmup_plus_decay_beyond_the_run_is_refused and
test_a_warmup_longer_than_the_run_is_refused_too went red; bytes restored, sha256
verified.
"""

import pytest
from pydantic import ValidationError

from src.schemas.training import TrainingHyperparameters

BASE = dict(hidden_dim=8, latent_dim=16, learning_rate=1e-3, batch_size=64, total_steps=1000)


def test_decay_is_off_by_default():
    assert TrainingHyperparameters(**BASE).lr_decay_steps == 0


def test_warmup_plus_decay_may_fill_the_run_exactly():
    hp = TrainingHyperparameters(**BASE, warmup_steps=200, lr_decay_steps=800)
    assert (hp.warmup_steps, hp.lr_decay_steps) == (200, 800)


def test_warmup_plus_decay_beyond_the_run_is_refused():
    with pytest.raises(ValidationError, match="must not exceed total_steps"):
        TrainingHyperparameters(**BASE, warmup_steps=200, lr_decay_steps=801)


def test_a_warmup_longer_than_the_run_is_refused_too():
    with pytest.raises(ValidationError, match="must not exceed total_steps"):
        TrainingHyperparameters(**BASE, warmup_steps=1001)


def test_a_negative_decay_is_refused():
    with pytest.raises(ValidationError):
        TrainingHyperparameters(**BASE, lr_decay_steps=-1)


def test_it_survives_serialisation_for_the_worker():
    dumped = TrainingHyperparameters(**BASE, warmup_steps=100, lr_decay_steps=300).model_dump()
    assert dumped["lr_decay_steps"] == 300
