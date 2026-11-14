"""Steering at temperature 0 is greedy decoding, not a refusal.

Found by the Python 3.13 GPU smoke (2026-10-21): a comparison at temperature 0
failed with transformers 5's "`temperature` (=0.0) has to be a strictly positive
float", because sampling was always on.
"""

import ast
import inspect
import textwrap

from src.services import steering_service
from src.services.steering_service import resolve_sampling


def _kwargs(**over):
    base = {"max_new_tokens": 8, "do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 50}
    base.update(over)
    return base


def test_temperature_zero_turns_sampling_off_and_drops_the_knobs():
    out = resolve_sampling(_kwargs(temperature=0.0))
    assert out["do_sample"] is False
    assert "temperature" not in out and "top_p" not in out and "top_k" not in out
    assert out["max_new_tokens"] == 8


def test_a_positive_temperature_is_left_alone():
    kw = _kwargs(temperature=0.7)
    assert resolve_sampling(kw) == kw


def test_greedy_already_requested_is_left_alone():
    kw = _kwargs(do_sample=False, temperature=0.0)
    assert resolve_sampling(kw) == kw


def test_the_generation_path_calls_it_before_generate():
    tree = ast.parse(textwrap.dedent(inspect.getsource(steering_service.SteeringService._generate_text)))
    calls = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "resolve_sampling"
    ]
    generate = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "generate"
    ]
    assert len(calls) == 1, "resolve_sampling must be called exactly once in _generate_text"
    assert generate and calls[0] < min(generate), "it must run before model.generate"
