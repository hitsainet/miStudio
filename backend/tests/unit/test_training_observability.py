"""A training records what it actually trained on, and reports held-out numbers by default.

TWO GAPS, both about auditing a run after the fact.

1. The realised mixture was never stored. A training kept only the REQUESTED
   `dataset_weights` — a bare positional array with nothing naming its
   positions — and reported the realised split only in a log line. When a
   request could not be honoured (every source fits, so the mixture silently
   follows availability) nothing outside stdout said so. Extraction already
   records its realised split per run; training now does too, per source, in
   BOTH the cached and the on-the-fly path.

2. `holdout_fraction` defaulted to 0, so every number a run reported was
   in-sample. That is how a per-layer FVU gradient on train_b14d263e went
   unverified until a separate post-run evaluation. It now defaults to 0.02.
   An explicit 0 still reproduces a historical run exactly.

MUTATION CONTROLS (each alone; suite must go red):
  O1  drop the cached-path record
        -> test_both_training_paths_record_the_mixture
  O2  drop the on-the-fly record
        -> test_both_training_paths_record_the_mixture
  O3  restore the schema default to 0.0
        -> test_holdout_defaults_to_two_percent
  O4  make realised_mixture_record drop the source label
        -> test_each_entry_names_its_source
  O5  change the worker's resume fallback away from 0.0
        -> test_old_runs_still_resume_without_a_holdout
"""

import ast
import inspect

import pytest

from src.schemas.training import TrainingHyperparameters
from src.services.dataset_mixture import realised_mixture_record
from src.workers import training_tasks


# ── the record ─────────────────────────────────────────────────────────────

def test_each_entry_names_its_source():
    """Named, so it cannot be mis-paired the way a positional array can."""
    record = realised_mixture_record(["ext_a", "ext_b"], [600, 400], [0.5, 0.5])

    assert [r["source"] for r in record] == ["ext_a", "ext_b"]


def test_realised_and_requested_sit_side_by_side():
    record = realised_mixture_record(["a", "b", "c"], [600, 300, 100], [0.5, 0.3, 0.2])

    assert [round(r["realised_fraction"], 3) for r in record] == [0.6, 0.3, 0.1]
    assert [round(r["requested_weight"], 3) for r in record] == [0.5, 0.3, 0.2]
    assert sum(r["realised_fraction"] for r in record) == pytest.approx(1.0)


def test_no_request_is_recorded_as_no_request():
    """Availability-proportional is a split too, and must not claim a request."""
    record = realised_mixture_record(["a", "b"], [3, 1])

    assert [r["requested_weight"] for r in record] == [None, None]
    assert [r["realised_fraction"] for r in record] == [0.75, 0.25]


def test_labels_and_allocations_must_pair():
    with pytest.raises(ValueError):
        realised_mixture_record(["a", "b", "c"], [1, 2])


# ── the wiring ─────────────────────────────────────────────────────────────

def _calls_to(fn_name):
    tree = ast.parse(inspect.getsource(training_tasks))
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == fn_name
    ]


def test_both_training_paths_record_the_mixture():
    """REACHABILITY: once for the cached path, once for on-the-fly.

    An AST walk for Call nodes — a text search would match the comments that
    describe the record, and pass for the wrong reason.
    """
    assert len(_calls_to("realised_mixture_record")) == 2


# ── the holdout default ────────────────────────────────────────────────────

def _hyperparameters(**overrides):
    body = {"hidden_dim": 8, "latent_dim": 16, "learning_rate": 1e-4,
            "batch_size": 64, "total_steps": 100}
    body.update(overrides)
    return TrainingHyperparameters(**body)


def test_holdout_defaults_to_two_percent():
    assert _hyperparameters().holdout_fraction == 0.02


def test_an_explicit_zero_is_still_honoured():
    """Reproducing a historical run exactly must remain possible."""
    assert _hyperparameters(holdout_fraction=0.0).holdout_fraction == 0.0


def test_old_runs_still_resume_without_a_holdout():
    """The worker's fallback for hyperparameters LACKING the key stays 0.0.

    Stored hyperparameters of an old run predate the field. Defaulting them to
    0.02 on resume would change what a resumed run holds out mid-training.

    AN AST CHECK OF THE CONSTANT, NOT A SUBSTRING. The first version asserted
    `"hp.get('holdout_fraction') or 0.0" in source` — and control O5, changing
    the fallback to `or 0.02`, SURVIVED, because "0.0" is a prefix of "0.02"
    and the substring still matched. Reading the numeric value is what bites.
    """
    tree = ast.parse(inspect.getsource(training_tasks))
    fallbacks = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)
                and len(node.values) == 2):
            continue
        call, default = node.values
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "get" and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == "holdout_fraction"
                and isinstance(default, ast.Constant)):
            fallbacks.append(default.value)
    assert fallbacks, "the resume fallback for holdout_fraction was not found"
    assert all(v == 0.0 for v in fallbacks), f"resume fallback changed: {fallbacks}"
