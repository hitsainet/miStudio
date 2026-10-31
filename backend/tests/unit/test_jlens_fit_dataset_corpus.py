"""Naming a dataset as the fitting corpus, instead of inlining 1200 documents.

BR-007 makes the corpus part of the recipe and the caller's choice, so there is
no server-chosen default: exactly one of `prompts` or `dataset_id` is required.
Naming a dataset is BETTER provenance than inlining its text -- the artifact
records `dataset + n_prompts + max_chars + seed`, which re-derives the corpus,
where 1200 opaque strings cannot be re-derived at all.

It is also the only route to a reference-sized corpus: 1200 documents at the
reference recipe's 2000-char cap is 2.2 MB against a 1024 KB body cap, so the
inline path 413s on the very recipe it exists to reproduce (observed
2026-09-05 fitting gemma-4-12B).
"""

import pytest
from pydantic import ValidationError

from src.api.v1.endpoints.jlens import FitRequest

PROBE = {"prompt": "The capital of France is", "expected_intermediate": " Paris"}


class TestExactlyOneCorpusSource:
    def test_inline_prompts_alone_is_valid(self):
        r = FitRequest(model_id="m_x", prompts=["a"] * 100, semantic_probe=PROBE)
        assert r.dataset_id is None

    def test_dataset_id_alone_is_valid(self):
        r = FitRequest(model_id="m_x", dataset_id="ds_1", semantic_probe=PROBE)
        assert r.prompts is None
        assert r.n_prompts == 1200

    def test_neither_is_refused(self):
        """Otherwise the SERVER picks a corpus, which BR-007 forbids."""
        with pytest.raises(ValidationError, match="exactly one"):
            FitRequest(model_id="m_x", semantic_probe=PROBE)

    def test_both_is_refused(self):
        """An ambiguous recipe: the artifact could not say which it used."""
        with pytest.raises(ValidationError, match="exactly one"):
            FitRequest(model_id="m_x", prompts=["a"], dataset_id="ds_1")

    def test_empty_prompt_list_counts_as_absent(self):
        """`prompts=[]` must not sneak past as 'supplied'."""
        with pytest.raises(ValidationError, match="exactly one"):
            FitRequest(model_id="m_x", prompts=[])


class TestSamplingDefaults:
    def test_max_chars_matches_the_reference_effective_input(self):
        """Reference passes max_chars 2000 but max_seq_len 128; ~550 chars is
        that in characters, so the sample is honest about what the model read."""
        assert FitRequest(model_id="m", dataset_id="d").max_chars == 550

    def test_min_chars_floor_exists(self):
        """A near-empty prompt linearises around no context and drags the mean
        J toward the null input."""
        assert FitRequest(model_id="m", dataset_id="d").min_chars == 400

    def test_seed_is_recorded_and_defaults_deterministically(self):
        assert FitRequest(model_id="m", dataset_id="d").sample_seed == 0

    @pytest.mark.parametrize("n", [99, 20001])
    def test_n_prompts_is_bounded(self, n):
        """Below the fitter's own floor of 100 the request is dead on arrival."""
        with pytest.raises(ValidationError):
            FitRequest(model_id="m", dataset_id="d", n_prompts=n)
