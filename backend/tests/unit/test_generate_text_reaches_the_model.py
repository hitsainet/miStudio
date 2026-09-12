"""`_generate_text` must actually run — the suite never called it.

MIS-E2E-068(2)'s fix referenced `self._model`, which does not exist on
`SteeringService` (models live in the `_loaded_models` cache). Every generation
raised `'SteeringService' object has no attribute '_model'` and steering failed
outright — while 3,438 tests passed, because nothing in the suite invoked this
method. That is a coverage hole with a live blast radius, so it gets a test
that drives the real code path with a stub model rather than one that reads the
source.
"""

import asyncio
from types import SimpleNamespace

import pytest
import torch


class _StubTokenizer:
    """Minimal tokenizer: enough for the length arithmetic under test."""

    model_max_length = 4096
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, n_tokens: int = 12):
        self._n = n_tokens

    def __call__(self, prompt, return_tensors=None, **kwargs):
        ids = torch.ones((1, self._n), dtype=torch.long)
        # A real tokenizer returns a BatchEncoding, which supports `.to(device)`
        # and subscripting. A plain dict does not, and the production code
        # calls `.to(...)` — using the real type keeps the stub honest.
        from transformers import BatchEncoding

        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, ids, skip_special_tokens=True):
        return "generated text"


class _StubModel:
    def __init__(self, window: int = 4096):
        self.config = SimpleNamespace(max_position_embeddings=window,
                                      model_type="stub", architectures=["StubForCausalLM"])
        self.device = torch.device("cpu")

    def generate(self, **kwargs):
        return torch.ones((1, 20), dtype=torch.long)

    def eval(self):
        return self

    def parameters(self):
        return iter([torch.zeros(1)])


@pytest.fixture
def service():
    from src.services.steering_service import SteeringService

    svc = SteeringService()
    svc._reset_model_state = lambda model: None  # no real state to reset
    return svc


def _params(max_new_tokens=32):
    from src.schemas.steering import GenerationParams

    return GenerationParams(max_new_tokens=max_new_tokens, temperature=0.7,
                            top_p=0.9, top_k=50)


class TestItRuns:
    def test_generate_text_completes_against_a_stub_model(self, service):
        """The regression: this raised AttributeError for every prompt."""
        text, tokens, ms = asyncio.run(
            service._generate_text(_StubModel(), _StubTokenizer(), "hello", _params())
        )
        assert isinstance(text, str)
        assert tokens >= 0

    def test_it_does_not_touch_a_self_dot_model_attribute(self, service):
        """`SteeringService` has no `_model`; reaching for one is the bug."""
        assert not hasattr(service, "_model"), (
            "a `_model` attribute now exists, so this test would no longer "
            "catch the mistake it was written for — re-check the fix"
        )


class TestTheWindowComesFromTheModel:
    def test_a_larger_window_admits_a_longer_prompt(self, service):
        """With the old hardcoded 2048 this prompt would have been truncated."""
        tok = _StubTokenizer(n_tokens=3000)
        text, _, _ = asyncio.run(
            service._generate_text(_StubModel(window=8192), tok, "x", _params(64))
        )
        assert isinstance(text, str)

    def test_a_prompt_that_cannot_fit_is_refused(self, service):
        tok = _StubTokenizer(n_tokens=4000)
        with pytest.raises(ValueError, match="Shorten the prompt"):
            asyncio.run(
                service._generate_text(_StubModel(window=2048), tok, "x", _params(64))
            )

    def test_max_new_tokens_larger_than_the_window_is_refused(self, service):
        """The old arithmetic produced a negative budget and truncated to nothing."""
        with pytest.raises(ValueError, match="no room for a prompt"):
            asyncio.run(
                service._generate_text(_StubModel(window=512), _StubTokenizer(),
                                       "x", _params(2048))
            )

    def test_a_sentinel_model_max_length_does_not_win(self, service):
        """Some tokenizers report a huge sentinel for "no limit"."""
        tok = _StubTokenizer(n_tokens=10)
        tok.model_max_length = 1_000_000_000_000
        text, _, _ = asyncio.run(
            service._generate_text(_StubModel(window=2048), tok, "x", _params(32))
        )
        assert isinstance(text, str)
