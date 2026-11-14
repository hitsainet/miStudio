"""The definition builder draws its test vectors from the sets as they were SCORED (ENH-006 G16, 2026-11-03).

The first export of `pm_c4cbed67349d` was refused in seconds: "too few positive examples: 0 positive and
1000000 negative". The builder built every evaluation set as `eval`, so the everyday-text view — negatives
only by construction — failed the positives floor, and it would next have rendered all 1,000,000 rows on
the CPU, because the builder ignored the seeded sample the evaluation scored. The probe also held two
evaluations of six sets (the run's own and a re-evaluation), each of which the builder prepared twice.

Pinned here: the role decision, the recorded sample reaching `_prepare_examples`, one evaluation per set,
and the two call sites that must use them.
"""
from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

from src.services import probe_definition_builder as builder
from src.services import probe_monitor_run as run_module
from src.services.probe_monitor_run import preparation_role, sample_seed_for


def _view(role, view_id="pmd_everyday"):
    return SimpleNamespace(id=view_id, role=role)


def _evaluation(dataset_id, metrics=None, tag=""):
    return SimpleNamespace(dataset_id=dataset_id, metrics=metrics if metrics is not None else {}, tag=tag)


# ── the role ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("role", ["calibration", "everyday_text", "nonsense_text"])
def test_a_negatives_only_view_is_prepared_as_calibration(role):
    assert preparation_role(_view(role)) == "calibration"


@pytest.mark.parametrize("role", ["eval", "train"])
def test_every_other_view_is_prepared_as_eval(role):
    assert preparation_role(_view(role)) == "eval"


# ── the sample ────────────────────────────────────────────────────────────────────────────────


def test_a_sampled_evaluation_gives_its_rows_and_the_view_and_size_seed():
    view = _view("everyday_text")
    evaluation = _evaluation(view.id, {"sample": {"rows": 5000, "seed_basis": "view+rows",
                                                  "source_rows": 1_000_000}})
    assert builder.vector_source_preparation(evaluation, view) == {
        "role": "calibration", "max_rows": 5000, "sample_seed": sample_seed_for(view.id, 5000)}


def test_an_evaluation_without_a_sample_scored_its_set_whole():
    view = _view("eval", "pmd_eval")
    for metrics in ({}, None, {"sample": None}):
        assert builder.vector_source_preparation(_evaluation(view.id, metrics), view) == {
            "role": "eval", "max_rows": None, "sample_seed": None}


# ── one evaluation per set ────────────────────────────────────────────────────────────────────


def test_the_newest_evaluation_of_each_set_is_kept_in_first_seen_order():
    rows = [_evaluation("a", tag="a1"), _evaluation("b", tag="b1"), _evaluation("a", tag="a2"),
            _evaluation("c", tag="c1")]
    assert [e.tag for e in builder.newest_evaluation_per_set(rows)] == ["a2", "b1", "c1"]


# ── the call sites ────────────────────────────────────────────────────────────────────────────


def _calls(function, name, tree=None):
    tree = tree or ast.parse(inspect.getsource(function).lstrip())
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == name]


def test_the_builder_prepares_each_newest_set_as_it_was_scored():
    """By AST: `build` prepares sets in one place, from `newest_evaluation_per_set`, with the keywords
    `vector_source_preparation` returns and no role of its own."""
    tree = ast.parse(inspect.getsource(builder.build).lstrip())
    calls = _calls(builder.build, "_prepare_examples", tree)
    assert len(calls) == 1, "build prepares evaluation sets in more than one place"
    keywords = calls[0].keywords
    assert not any(k.arg == "role" for k in keywords), "build names a role of its own"
    spread = [ast.unparse(k.value) for k in keywords if k.arg is None]
    assert spread == ["vector_source_preparation(evaluation, view)"]

    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and any(child is calls[0] for child in ast.walk(node))]
    assert loops and ast.unparse(loops[-1].iter) == "newest_evaluation_per_set(evaluations)"


def test_the_scoring_loop_takes_its_role_from_the_same_decision():
    calls = _calls(run_module._evaluate_probes_on_sets, "_prepare_examples")
    assert len(calls) == 1
    role = {k.arg: k.value for k in calls[0].keywords}.get("role")
    assert role is not None and ast.unparse(role) == "preparation_role(view)"
