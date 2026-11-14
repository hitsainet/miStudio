"""A loaded SAE is built in the basis it TRAINED in.

Two halves of one silent-wrong-basis defect:

* Extraction called ``create_sae`` without ``normalize_activations``, so every
  SAE got the constructor default ``constant_norm_rescale`` whatever its
  cfg.json recorded. An SAE trained with ``none`` would be normalised anyway.
* The cfg.json writer fell back to ``"none"`` when the training hyperparameters
  omitted the key — while training itself resolves an omitted key to the
  FRAMEWORK default. So the file could describe a basis the SAE never used.

Harmless on this estate only because every SAE here trained with the default.
"""
import ast
import inspect
from types import SimpleNamespace

import pytest

from src.ml.community_format import CommunityStandardConfig
from src.ml.sparse_autoencoder import (
    SUPPORTED_NORMALIZATIONS,
    create_sae,
    resolve_loaded_sae_normalization,
)
from src.services import extraction_service


class TestResolver:

    def test_native_weights_with_no_config_keep_the_training_default(self):
        assert resolve_loaded_sae_normalization(None) == "constant_norm_rescale"

    @pytest.mark.parametrize("mode", SUPPORTED_NORMALIZATIONS)
    def test_a_recorded_mode_is_used_as_recorded(self, mode):
        assert resolve_loaded_sae_normalization(
            SimpleNamespace(normalize_activations=mode)) == mode

    def test_none_recorded_is_not_overridden(self):
        """The case the defect broke: the default must not win over "none"."""
        assert resolve_loaded_sae_normalization(
            SimpleNamespace(normalize_activations="none")) == "none"

    def test_an_unimplemented_mode_is_refused_by_name(self):
        with pytest.raises(ValueError, match="expected_average_only_in"):
            resolve_loaded_sae_normalization(
                SimpleNamespace(normalize_activations="expected_average_only_in"))

    def test_the_resolved_mode_reaches_the_built_sae(self):
        cfg = SimpleNamespace(normalize_activations="none")
        sae = create_sae("jumprelu", hidden_dim=8, latent_dim=16,
                         normalize_activations=resolve_loaded_sae_normalization(cfg))
        assert sae.normalize_activations == "none"


class TestExtractionPassesIt:

    def _create_sae_calls(self):
        tree = ast.parse(inspect.getsource(extraction_service))
        return [n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "create_sae"]

    def test_every_create_sae_call_passes_the_resolved_normalization(self):
        calls = self._create_sae_calls()
        assert calls, "extraction_service no longer builds an SAE via create_sae"
        for call in calls:
            kw = {k.arg: k.value for k in call.keywords}
            assert "normalize_activations" in kw, (
                f"create_sae at line {call.lineno} omits normalize_activations; "
                "every SAE would get the constructor default")
            value = kw["normalize_activations"]
            assert isinstance(value, ast.Call) and getattr(
                value.func, "id", None) == "resolve_loaded_sae_normalization", (
                f"line {call.lineno} must pass resolve_loaded_sae_normalization(...)")


class TestConfigWriterRecordsWhatTrainingUsed:

    def _cfg(self, **hp):
        base = {"architecture_type": "jumprelu", "hidden_dim": 8, "latent_dim": 16}
        base.update(hp)
        return CommunityStandardConfig.from_training_hyperparams(
            base, model_name="m", layer=3)

    def test_an_omitted_key_records_the_framework_default_not_none(self):
        assert self._cfg().normalize_activations == "constant_norm_rescale"

    def test_a_recorded_key_is_written_as_recorded(self):
        assert self._cfg(normalize_activations="none").normalize_activations == "none"

    def test_the_anthropic_framework_default_is_its_own(self):
        cfg = self._cfg(architecture_type="standard_anthropic")
        assert cfg.normalize_activations == "anthropic_rescale"
