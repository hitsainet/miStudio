"""Finding the unembedding tensor across flat and nested weight layouts.

The logit lens looked for exactly two flat key names. gemma-4-12B-it is
multimodal and nests its text tower, so its unembedding lives at
`model.language_model.embed_tokens.weight` — neither name matched, and every
logit-lens request on a gemma-4 SAE returned HTTP 500 with:

    No unembedding tensor in .../model.safetensors;
    tried ['lm_head.weight', 'model.embed_tokens.weight']

The real key list is taken from the actual checkpoint (677 tensors).
"""

import pytest
import torch

from src.services.analysis_service import _find_unembedding_key, _validated

# Verbatim from the gemma-4-12B-it checkpoint on disk.
GEMMA4_KEYS = [
    "model.embed_audio.embedding_projection.weight",
    "model.embed_vision.embedding_projection.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
]


class TestNestedLayouts:
    def test_gemma4_multimodal_is_found(self):
        assert _find_unembedding_key(GEMMA4_KEYS) == \
            "model.language_model.embed_tokens.weight"

    def test_the_vision_and_audio_projections_are_never_chosen(self):
        """A looser 'embed' search would return these. They are not unembeddings."""
        got = _find_unembedding_key(GEMMA4_KEYS)
        assert "vision" not in got and "audio" not in got

    def test_a_multimodal_model_with_an_untied_head_prefers_lm_head(self):
        keys = GEMMA4_KEYS + ["model.language_model.lm_head.weight"]
        assert _find_unembedding_key(keys).endswith("lm_head.weight")

    def test_the_shallowest_match_wins(self):
        """A top-level tensor is the model's own; a deeper one is a submodule's.

        NEITHER key here is in the exact-name list — the first version used
        "model.embed_tokens.weight", which short-circuits on the exact match and
        never reaches the depth sort, so reversing that sort SURVIVED the test.
        """
        keys = ["a.b.c.d.embed_tokens.weight", "outer.embed_tokens.weight"]
        assert _find_unembedding_key(keys) == "outer.embed_tokens.weight"


class TestFlatLayoutsStillWork:
    def test_tied_embeddings(self):
        """granite-4.1-8b sets tie_word_embeddings=true."""
        assert _find_unembedding_key(
            ["model.embed_tokens.weight", "model.layers.0.mlp.up_proj.weight"]
        ) == "model.embed_tokens.weight"

    def test_untied_prefers_lm_head_over_input_embeddings(self):
        assert _find_unembedding_key(
            ["lm_head.weight", "model.embed_tokens.weight"]) == "lm_head.weight"

    def test_nothing_matching_returns_none(self):
        assert _find_unembedding_key(["model.layers.0.mlp.up_proj.weight"]) is None

    def test_empty(self):
        assert _find_unembedding_key([]) is None


class TestTheShapeGuard:
    """Suffix matching is permissive, so the tensor is checked before use."""

    def test_a_plausible_unembedding_passes(self):
        t = torch.zeros(262144, 3840)
        assert _validated(t, "k") is t

    def test_a_one_dimensional_tensor_is_refused(self):
        with pytest.raises(ValueError, match="1-D"):
            _validated(torch.zeros(4096), "some.bias")

    def test_a_wrong_way_round_matrix_is_refused(self):
        """vocab must exceed d_model; otherwise it is not an unembedding."""
        with pytest.raises(ValueError, match="vocab"):
            _validated(torch.zeros(3840, 262144), "k")

    def test_the_error_names_the_key_that_was_chosen(self):
        """Otherwise the failure surfaces as a matmul error far from here."""
        with pytest.raises(ValueError, match="model.embed_vision"):
            _validated(torch.zeros(8), "model.embed_vision.weight")
