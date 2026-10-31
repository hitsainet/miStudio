"""Each exported SAE's cfg.json names its OWN hook (review R1-A, A5).

A multi-hook training exports layer_L_residual/ and layer_L_mlp/ side by side, and
save_multilayer_community_checkpoint hands each SAE's hook type to the config builder.
The builder ignored it and wrote blocks.L.hook_resid_post into every cfg.json, so an MLP
or attention SAE was labelled a residual SAE for everything that reads the export
(sae_manager_service infers a hook from it, Neuronpedia and SAELens take it as the hook).
The finalize service writes through the same function, so it inherits both.

MUTATION CONTROLS (one line broken at a time, this file run, bytes restored, sha256 and the
working tree verified clean; the full table is in the A5 record):
  X1 from_training_hyperparams back to "resid_post"             -> both export tests
  X2 save_multilayer_community_checkpoint stops passing the hook -> the export test
"""

import json

import pytest

from src.ml.community_format import CommunityStandardConfig, training_hook_point
from src.ml.sparse_autoencoder import create_sae
from src.services.checkpoint_service import CheckpointService


def test_a_multi_hook_export_names_each_saes_hook(tmp_path):
    combos = [(3, "residual"), (3, "mlp"), (4, "attention")]
    models = dict((key, create_sae("standard_saelens", hidden_dim=8, latent_dim=16)) for key in combos)
    CheckpointService.save_multilayer_community_checkpoint(
        models=models,
        base_output_dir=str(tmp_path),
        model_name="org/tiny",
        layer_hook_combinations=combos,
        hyperparams=dict(hidden_dim=8, latent_dim=16, architecture_type="standard_saelens",
                         hook_types=["residual", "mlp", "attention"]),
        training_id="train_hooks",
        checkpoint_step=10,
    )
    named = []
    for layer, hook in combos:
        cfg = json.loads((tmp_path / ("layer_" + str(layer) + "_" + hook) / "cfg.json").read_text())
        named.append((layer, hook, cfg["hook_point"], cfg["hook_name"]))
    assert named == [
        (3, "residual", "blocks.3.hook_resid_post", "blocks.3.hook_resid_post"),
        (3, "mlp", "blocks.3.hook_mlp_out", "blocks.3.hook_mlp_out"),
        (4, "attention", "blocks.4.hook_attn_out", "blocks.4.hook_attn_out"),
    ]


def test_a_config_without_a_hook_type_is_a_residual_sae():
    config = CommunityStandardConfig.from_training_hyperparams(
        hyperparams=dict(hidden_dim=8, latent_dim=16), model_name="org/tiny", layer=5,
    )
    assert config.hook_point == config.hook_name == "blocks.5.hook_resid_post"


def test_an_unknown_training_hook_type_is_refused_rather_than_labelled_residual():
    with pytest.raises(ValueError, match="Unknown training hook type"):
        training_hook_point(2, "resid_pre")
