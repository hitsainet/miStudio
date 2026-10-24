"""Hook and layer records for SAEs as HuggingFace actually publishes them (review R3-B).

R2-E made every ExternalSAE writer record the SAE's hook from its cfg.json or a Gemma Scope set
name, because a NULL hook reads as residual and every refusal passes it. Round 3 fetched real
configs from HuggingFace and found whole published families still recorded NULL:

* Gemma Scope 2 (google/gemma-scope-2-*) writes config.json (``type``, ``hf_hook_point_out``) and
  names its kind in the FOLDER (mlp_out/), not the repository;
* Llama Scope writes hyperparams.json (``hook_point_in`` / ``hook_point_out``); Qwen Scope and
  dictionary_learning write config.json (``hook_point``, ``trainer.submodule_name``);
* EleutherAI sparsify writes no hook at all: the directory name is the hookpoint;
* Gemma Scope 1 transcoders recorded NULL, and its -res sets' embedding/ SAEs recorded ``residual``.

And the LAYER: the download task took it only from the old SAELens ``hook_point_layer``, so every
SAELens 5 (``hook_layer``) and SAELens 6 (no layer key) download recorded NULL. A NULL layer turns
off the own-layer checks in capture, steering and the cluster allocation, and feature extraction
hooks ``external_sae.layer or 0``.

Every config below is copied from the file on HuggingFace (repository and path beside it), trimmed
to the keys that matter. The SAELens 6 shape is the one CONSTRUCTED case (SAELens 6 nests the hook
under ``metadata`` and writes no layer); no fetched release was saved by SAELens 6.

MUTATION CONTROLS (one line broken at a time, the listed files run, bytes restored, sha256 and
``git diff`` verified; the table is in review_sae_remediation_R3_B_2026-09-15.md):
  L1 download task: ``sae.layer = recorded_layer ...`` back to ``metadata.get("layer")`` inside ``if metadata``
        -> the SAELens 5, SAELens 6, sparsify and Gemma Scope 2 task tests
  L2 resolve_sae_layer ignores the hook name -> the SAELens 6 rows, the Gemma Scope 2 rows, the precedence test
  L3 resolve_sae_layer ignores ``hook_layer`` -> test_a_config_layer_is_used_when_the_hook_names_none
  L4 import_from_file: ``layer=request.layer`` again -> the local-import layer test
  G1 SAE_CONFIG_FILENAMES without config.json -> the Gemma Scope 2, Qwen Scope, dictionary_learning rows
  G2 SAE_CONFIG_FILENAMES without hyperparams.json -> the Llama Scope row
  G3 the transcoder rule removed -> the Gemma Scope 2 transcoder/clt rows, the SAELens transcoder test
  G4 sparsify_hookpoint not consulted -> the sparsify rows and the sparsify task test
  G5 Gemma Scope 2 folder kinds not consulted -> the name-only Gemma Scope 2 row
  G6 resolve_sae_hook reads each origin alone again (not joined) -> the name-only Gemma Scope 2 row
  G7 the embedding folder rule removed -> the Gemma Scope 1 embedding row
  M1 ``feedforward`` / ``ffn`` / feed+forward removed from the MLP rule -> the Gemma Scope 2 mlp rows, LFM2 names
  M2 the embedding tokens removed -> the embedding rows and names
"""

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pytest

from src.core.config import settings
from src.models.external_sae import ExternalSAE
from src.services.sae_hook_support import classify_hook, non_residual_hook_reason
from src.services.sae_manager_service import (
    HOOK_SOURCE_GEMMA_SCOPE_NAME,
    HOOK_SOURCE_SPARSIFY_DIRECTORY,
    SAEManagerService,
    resolve_sae_hook,
    resolve_sae_layer,
)

# ── real configs, as fetched from HuggingFace on 2026-09-15 ─────────────────

# jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8 blocks.8.hook_resid_pre_768/cfg.json (old SAELens)
GPT2_OLD_SAELENS = {"model_name": "gpt2-small", "hook_point": "blocks.8.hook_resid_pre", "hook_point_layer": 8, "hook_point_head_index": None}
# tommmcgrath/gpt2-small-mlp-out-saes sae_group_gpt2_blocks.0.hook_mlp_out_24576:v1/cfg.json
GPT2_MLP_TM = {"model_name": "gpt2", "model_class_name": "HookedTransformer", "hook_point": "blocks.0.hook_mlp_out",
               "hook_point_eval": "blocks.{layer}.attn.pattern", "hook_point_layer": 0, "hook_point_head_index": None}
# Juliushanhanhan/llama-3-8b-it-res blocks.25.hook_resid_post/cfg.json (SAELens 5)
LLAMA3_RES_SAELENS5 = {"model_name": "meta-llama/Meta-Llama-3-8B-Instruct", "model_class_name": "HookedTransformer",
                       "hook_name": "blocks.25.hook_resid_post", "hook_eval": "NOT_IN_USE", "hook_layer": 25, "hook_head_index": None}
# JoshEngels/Mistral-7B-Residual-Stream-SAEs mistral_7b_layer_8/cfg.json (SAELens 5)
MISTRAL_PRE_SAELENS5 = {"d_in": 4096, "d_sae": 65536, "model_name": "mistral-7b", "hook_name": "blocks.8.hook_resid_pre", "hook_layer": 8, "hook_head_index": None}
# jbloom/GPT2-Small-OAI-v5-32k-resid-mid-SAEs v5_32k_layer_0/cfg.json (SAELens 5)
OAI_MID_SAELENS5 = {"architecture": "standard", "d_in": 768, "d_sae": 32768, "model_name": "gpt2-small", "hook_name": "blocks.0.hook_resid_mid", "hook_layer": 0, "hook_head_index": None}
# ctigges/pythia-70m-deduped__att-sm_processed 0-att-sm/cfg.json (SAELens 5)
PYTHIA_ATT_SAELENS5 = {"architecture": "standard", "d_in": 512, "d_sae": 32768, "model_name": "pythia-70m-deduped", "hook_name": "blocks.0.hook_attn_out", "hook_layer": 0, "hook_head_index": None}
# CONSTRUCTED: SAELens 6 nests the hook under metadata and writes no layer key
SAELENS6 = {"d_in": 2304, "d_sae": 16384, "architecture": "jumprelu", "metadata": {"model_name": "gemma-2-2b", "hook_name": "blocks.20.hook_resid_post", "hook_head_index": None}}
# google/gemma-scope-2-4b-pt {resid_post,mlp_out,attn_out,transcoder}/layer_17_width_16k_l0_big/config.json
_GS2 = {"width": 16384, "model_name": "google/gemma-3-4b-pt", "architecture": "jump_relu", "l0": 150, "affine_connection": False}
GS2_RES_17 = {**_GS2, "hf_hook_point_in": "model.layers.17.output", "hf_hook_point_out": "model.layers.17.output", "type": "sae"}
GS2_MLP_17 = {**_GS2, "hf_hook_point_in": "model.layers.17.post_feedforward_layernorm.output", "hf_hook_point_out": "model.layers.17.post_feedforward_layernorm.output", "type": "sae"}
GS2_ATT_17 = {**_GS2, "hf_hook_point_in": "model.layers.17.self_attn.o_proj.input", "hf_hook_point_out": "model.layers.17.self_attn.o_proj.input", "type": "sae"}
GS2_TRANSCODER_17 = {**_GS2, "hf_hook_point_in": "model.layers.17.pre_feedforward_layernorm.output", "hf_hook_point_out": "model.layers.17.post_feedforward_layernorm.output", "type": "transcoder"}
# google/gemma-scope-2-270m-pt clt/width_262k_l0_big/config.json and crosscoder/layer_5_9_12_15_width_1m_l0_big/config.json
GS2_CLT = {"hf_hook_point_in": "model.layers.{all}.pre_feedforward_layernorm.output", "hf_hook_point_out": "model.layers.{all}.post_feedforward_layernorm.output",
           "width": 262080, "model_name": "google/gemma-3-270m-pt", "architecture": "jump_relu", "l0": 150, "affine_connection": False, "type": "clt"}
GS2_CROSSCODER = {"hf_hook_point_in": "model.layers.{5,9,12,15}.output", "hf_hook_point_out": "model.layers.{5,9,12,15}.output",
                  "width": 1048576, "model_name": "google/gemma-3-270m-pt", "architecture": "jump_relu", "l0": 150, "affine_connection": False, "type": "crosscoder"}
# fnlp/Llama3_1-8B-Base-LXM-8x Llama3_1-8B-Base-L15M-8x/hyperparams.json (Llama Scope)
LLAMA_SCOPE_L15M = {"device": "cuda:0", "seed": 42, "dtype": "torch.bfloat16", "hook_point_in": "blocks.15.hook_mlp_out", "hook_point_out": "blocks.15.hook_mlp_out",
                    "expansion_factor": 8, "d_model": 4096, "d_sae": 32768}
# Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 config.json (Qwen Scope; one file per layer: layer5.sae.pt)
QWEN_SCOPE = {"model_type": "topk_sae", "base_model": "Qwen/Qwen3-1.7B-Base", "d_model": 2048, "d_sae": 32768, "k": 50, "num_layers": 28, "hook_point": "resid_post", "dtype": "float32"}
# andyrdt/saes-llama-3.1-8b-instruct resid_post_layer_3/trainer_1/config.json (dictionary_learning)
DICTIONARY_LEARNING = {"trainer": {"trainer_class": "BatchTopKTrainer", "dict_class": "BatchTopKSAE", "activation_dim": 4096, "dict_size": 131072, "k": 64,
                                   "layer": 3, "lm_name": "meta-llama/Llama-3.1-8B-Instruct", "submodule_name": "resid_post_layer_3"}}
# EleutherAI/sae-pythia-70m-32k {layers.3,layers.0.mlp,layers.0.attention}/cfg.json (sparsify)
SPARSIFY_PYTHIA = {"expansion_factor": 32, "normalize_decoder": True, "num_latents": 32768, "k": 16, "signed": False, "d_in": 512}
# EleutherAI/sae-llama-3-8b-32x embed_tokens/cfg.json (sparsify, no num_latents)
SPARSIFY_LLAMA3 = {"expansion_factor": 32, "normalize_decoder": True, "k": 192, "signed": False, "d_in": 4096}
# EleutherAI/skip-transcoder-Llama-3.2-1B-131k layers.5.mlp/cfg.json
SPARSIFY_SKIP_TRANSCODER = {"expansion_factor": 32, "normalize_decoder": True, "num_latents": 131072, "k": 32, "multi_topk": False, "skip_connection": True, "d_in": 2048}


def _lay_out(root: Path, relative: str, config_name: Optional[str], cfg: Optional[dict], weights: str) -> Path:
    """An SAE directory as it arrives: the config (if any) and a weights file."""
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    if config_name:
        (directory / config_name).write_text(json.dumps(cfg))
    if weights == "params.npz":
        np.savez(directory / "params.npz", W_enc=np.zeros((8, 16), dtype=np.float32), W_dec=np.zeros((16, 8), dtype=np.float32),
                 b_enc=np.zeros(16, dtype=np.float32), b_dec=np.zeros(8, dtype=np.float32), threshold=np.zeros(16, dtype=np.float32))
    else:
        (directory / weights).write_bytes(b"")
    return directory


# (id, directory, config file, config, weights, origins, hook, source, layer, kind)
REAL = [
    ("saelens-old-resid-pre", "blocks.8.hook_resid_pre_768", "cfg.json", GPT2_OLD_SAELENS, "sae_weights.safetensors",
     ("jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8", "blocks.8.hook_resid_pre_768"), "blocks.8.hook_resid_pre", "cfg.json", 8, "resid_pre"),
    ("saelens-old-mlp-with-attn-eval", "sae_group_gpt2_blocks.0.hook_mlp_out_24576:v1", "cfg.json", GPT2_MLP_TM, "sae_weights.safetensors",
     ("tommmcgrath/gpt2-small-mlp-out-saes",), "blocks.0.hook_mlp_out", "cfg.json", 0, "mlp"),
    ("saelens5-resid-post-25", "blocks.25.hook_resid_post", "cfg.json", LLAMA3_RES_SAELENS5, "sae_weights.safetensors",
     ("Juliushanhanhan/llama-3-8b-it-res", "blocks.25.hook_resid_post"), "blocks.25.hook_resid_post", "cfg.json", 25, "residual"),
    ("saelens5-resid-pre-8", "mistral_7b_layer_8", "cfg.json", MISTRAL_PRE_SAELENS5, "sae_weights.safetensors",
     ("JoshEngels/Mistral-7B-Residual-Stream-SAEs", "mistral_7b_layer_8"), "blocks.8.hook_resid_pre", "cfg.json", 8, "resid_pre"),
    ("saelens5-resid-mid", "v5_32k_layer_0", "cfg.json", OAI_MID_SAELENS5, "sae_weights.safetensors",
     ("jbloom/GPT2-Small-OAI-v5-32k-resid-mid-SAEs", "v5_32k_layer_0"), "blocks.0.hook_resid_mid", "cfg.json", 0, "resid_mid"),
    ("saelens5-attn-out", "0-att-sm", "cfg.json", PYTHIA_ATT_SAELENS5, "sae_weights.safetensors",
     ("ctigges/pythia-70m-deduped__att-sm_processed", "0-att-sm"), "blocks.0.hook_attn_out", "cfg.json", 0, "attention"),
    ("saelens6-metadata", "blocks.20.hook_resid_post", "cfg.json", SAELENS6, "sae_weights.safetensors",
     ("someone/gemma-2-2b-saes", "blocks.20.hook_resid_post"), "blocks.20.hook_resid_post", "cfg.json", 20, "residual"),
    ("gemma-scope-2-resid-post", "resid_post/layer_17_width_16k_l0_big", "config.json", GS2_RES_17, "params.safetensors",
     ("google/gemma-scope-2-4b-pt", "resid_post/layer_17_width_16k_l0_big"), "model.layers.17.output", "config.json", 17, "residual"),
    ("gemma-scope-2-mlp-out", "mlp_out/layer_17_width_16k_l0_big", "config.json", GS2_MLP_17, "params.safetensors",
     ("google/gemma-scope-2-4b-pt", "mlp_out/layer_17_width_16k_l0_big"), "model.layers.17.post_feedforward_layernorm.output", "config.json", 17, "mlp"),
    ("gemma-scope-2-attn-out", "attn_out/layer_17_width_16k_l0_big", "config.json", GS2_ATT_17, "params.safetensors",
     ("google/gemma-scope-2-4b-pt", "attn_out/layer_17_width_16k_l0_big"), "model.layers.17.self_attn.o_proj.input", "config.json", 17, "attention"),
    ("gemma-scope-2-transcoder", "transcoder/layer_17_width_16k_l0_big", "config.json", GS2_TRANSCODER_17, "params.safetensors",
     ("google/gemma-scope-2-4b-pt", "transcoder/layer_17_width_16k_l0_big"), "transcoder", "config.json", 17, "mlp"),
    ("gemma-scope-2-clt", "clt/width_262k_l0_big", "config.json", GS2_CLT, "params.safetensors",
     ("google/gemma-scope-2-270m-pt", "clt/width_262k_l0_big"), "transcoder", "config.json", None, "mlp"),
    ("gemma-scope-2-crosscoder", "crosscoder/layer_5_9_12_15_width_1m_l0_big", "config.json", GS2_CROSSCODER, "params.safetensors",
     ("google/gemma-scope-2-270m-pt", "crosscoder/layer_5_9_12_15_width_1m_l0_big"), "crosscoder", "config.json", None, None),
    ("llama-scope-hyperparams", "Llama3_1-8B-Base-L15M-8x", "hyperparams.json", LLAMA_SCOPE_L15M, "final.safetensors",
     ("fnlp/Llama3_1-8B-Base-LXM-8x", "Llama3_1-8B-Base-L15M-8x"), "blocks.15.hook_mlp_out", "hyperparams.json", 15, "mlp"),
    ("qwen-scope-root-config", "", "config.json", QWEN_SCOPE, "layer5.sae.pt",
     ("Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50", "layer5.sae.pt"), "resid_post", "config.json", 5, "residual"),
    ("dictionary-learning", "resid_post_layer_3/trainer_1", "config.json", DICTIONARY_LEARNING, "ae.pt",
     ("andyrdt/saes-llama-3.1-8b-instruct", "resid_post_layer_3/trainer_1"), "resid_post_layer_3", "config.json", 3, "residual"),
    ("sparsify-layer", "layers.3", "cfg.json", SPARSIFY_PYTHIA, "sae.safetensors",
     ("EleutherAI/sae-pythia-70m-32k", "layers.3"), "layers.3", HOOK_SOURCE_SPARSIFY_DIRECTORY, 3, "residual"),
    ("sparsify-mlp", "layers.0.mlp", "cfg.json", SPARSIFY_PYTHIA, "sae.safetensors",
     ("EleutherAI/sae-pythia-70m-32k", "layers.0.mlp"), "layers.0.mlp", HOOK_SOURCE_SPARSIFY_DIRECTORY, 0, "mlp"),
    ("sparsify-attention", "layers.0.attention", "cfg.json", SPARSIFY_PYTHIA, "sae.safetensors",
     ("EleutherAI/sae-pythia-70m-32k", "layers.0.attention"), "layers.0.attention", HOOK_SOURCE_SPARSIFY_DIRECTORY, 0, "attention"),
    ("sparsify-embed-tokens", "embed_tokens", "cfg.json", SPARSIFY_LLAMA3, "sae.safetensors",
     ("EleutherAI/sae-llama-3-8b-32x", "embed_tokens"), "embed_tokens", HOOK_SOURCE_SPARSIFY_DIRECTORY, None, "embedding"),
    ("sparsify-skip-transcoder", "layers.5.mlp", "cfg.json", SPARSIFY_SKIP_TRANSCODER, "sae.safetensors",
     ("EleutherAI/skip-transcoder-Llama-3.2-1B-131k", "layers.5.mlp"), "layers.5.mlp", HOOK_SOURCE_SPARSIFY_DIRECTORY, 5, "mlp"),
    ("gemma-scope-1-mlp", "layer_3/width_16k/average_l0_71", None, None, "params.npz",
     ("google/gemma-scope-2b-pt-mlp", "layer_3/width_16k/average_l0_71"), "mlp", HOOK_SOURCE_GEMMA_SCOPE_NAME, 3, "mlp"),
    ("gemma-scope-1-embedding", "embedding/width_4k/average_l0_6", None, None, "params.npz",
     ("google/gemma-scope-2b-pt-res", "embedding/width_4k/average_l0_6"), "embedding", HOOK_SOURCE_GEMMA_SCOPE_NAME, None, "embedding"),
    ("gemma-scope-1-transcoders", "layer_3/width_16k/average_l0_76", None, None, "params.npz",
     ("google/gemma-scope-2b-pt-transcoders", "layer_3/width_16k/average_l0_76"), "transcoder", HOOK_SOURCE_GEMMA_SCOPE_NAME, 3, "mlp"),
    ("gemma-scope-2-name-only", "mlp_out/layer_17_width_16k_l0_big", None, None, "params.safetensors",
     ("google/gemma-scope-2-4b-pt", "mlp_out/layer_17_width_16k_l0_big/params.safetensors"), "mlp", HOOK_SOURCE_GEMMA_SCOPE_NAME, 17, "mlp"),
]


@pytest.mark.parametrize("row", REAL, ids=[r[0] for r in REAL])
def test_a_published_sae_records_its_hook_layer_and_the_refusals_read_it(tmp_path, row):
    _, relative, config_name, cfg, weights, origins, hook, source, layer, kind = row
    directory = _lay_out(tmp_path, relative, config_name, cfg, weights)
    recorded = resolve_sae_hook(directory, *origins)
    assert (recorded.hook_type, recorded.source) == (hook, source)
    assert resolve_sae_layer(directory, recorded.hook_type, *origins) == layer
    if kind is not None:
        assert classify_hook(recorded.hook_type) == kind
        refused = non_residual_hook_reason(recorded.hook_type, "Feature extraction") is not None
        assert refused == (kind != "residual"), (recorded.hook_type, kind)


def test_a_hf_cache_path_names_its_gemma_scope_set(tmp_path):
    """A local import from HuggingFace's cache: ``models--google--gemma-scope-2b-pt-att/snapshots/<sha>/...``."""
    relative = "hub/models--google--gemma-scope-2b-pt-att/snapshots/0123abcd/layer_5/width_16k/average_l0_71"
    directory = _lay_out(tmp_path, relative, None, None, "params.npz")
    recorded = resolve_sae_hook(directory, str(directory))
    assert (recorded.hook_type, recorded.source) == ("attention", HOOK_SOURCE_GEMMA_SCOPE_NAME)
    assert resolve_sae_layer(directory, recorded.hook_type, str(directory)) == 5


# ── precedence ──────────────────────────────────────────────────────────────


def test_cfg_json_outranks_config_json_and_both_outrank_a_directory_name(tmp_path):
    directory = _lay_out(tmp_path, "layers.3", "cfg.json", {**SPARSIFY_PYTHIA, "hook_name": "blocks.3.hook_mlp_out"}, "sae.safetensors")
    (directory / "config.json").write_text(json.dumps(QWEN_SCOPE))
    assert resolve_sae_hook(directory).hook_type == "blocks.3.hook_mlp_out"
    (directory / "cfg.json").write_text(json.dumps(SPARSIFY_PYTHIA))
    assert resolve_sae_hook(directory) == ("resid_post", "config.json")


def test_a_directory_is_a_sparsify_hookpoint_only_beside_a_sparsify_config(tmp_path):
    assert resolve_sae_hook(_lay_out(tmp_path / "a", "layers.3", "cfg.json", {"d_in": 8}, "sae.safetensors")).hook_type is None
    assert resolve_sae_hook(_lay_out(tmp_path / "b", "layers.3", None, None, "sae.safetensors")).hook_type is None
    assert resolve_sae_hook(_lay_out(tmp_path / "c", "my_sae", "cfg.json", SPARSIFY_PYTHIA, "sae.safetensors")).hook_type is None


def test_a_saelens_transcoder_is_recorded_as_a_transcoder_not_by_its_input_name(tmp_path):
    cfg = {"hook_name": "blocks.3.hook_resid_mid", "hook_name_out": "blocks.3.hook_mlp_out", "hook_layer": 3}
    directory = _lay_out(tmp_path, "t", "cfg.json", cfg, "sae_weights.safetensors")
    assert resolve_sae_hook(directory).hook_type == "transcoder"
    assert non_residual_hook_reason("transcoder", "X") is not None


def test_the_hook_names_layer_outranks_a_disagreeing_config_integer_and_is_logged(tmp_path, caplog):
    directory = _lay_out(tmp_path, "d", "cfg.json", {"hook_name": "blocks.12.hook_resid_post", "hook_layer": 11}, "sae_weights.safetensors")
    with caplog.at_level(logging.WARNING):
        assert resolve_sae_layer(directory, "blocks.12.hook_resid_post") == 12
    assert "gives layer 11" in caplog.text


def test_a_config_layer_is_used_when_the_hook_names_none(tmp_path):
    for key in ("hook_point_layer", "hook_layer"):
        directory = _lay_out(tmp_path / key, "d", "cfg.json", {key: 7}, "sae_weights.safetensors")
        assert resolve_sae_layer(directory, "mlp") == 7, key
    nested = _lay_out(tmp_path / "metadata", "d", "cfg.json", {"metadata": {"hook_layer": 9}}, "sae_weights.safetensors")
    assert resolve_sae_layer(nested, None) == 9
    boolean = _lay_out(tmp_path / "bool", "d", "cfg.json", {"hook_layer": True}, "sae_weights.safetensors")
    assert resolve_sae_layer(boolean, None) is None
    assert resolve_sae_layer(None, None, "someone/saes", "width_16k") is None


# ── the matcher on real names ───────────────────────────────────────────────

# Each name is recorded by some writer above or appears in a published release / the model
# code this project hooks; the kind is what the refusals must decide.
REAL_NAMES = [
    ("residual", "residual"), ("blocks.3.hook_resid_post", "residual"), ("model.layers.17.output", "residual"),
    ("layers.10", "residual"), ("resid_post", "residual"), ("resid_post_layer_3", "residual"),
    ("blocks.0.hook_resid_pre", "resid_pre"), ("blocks.0.hook_resid_mid", "resid_mid"),
    ("embed_tokens", "embedding"), ("hook_embed", "embedding"), ("embedding", "embedding"),
    ("mlp", "mlp"), ("blocks.0.hook_mlp_out", "mlp"), ("layers.0.mlp", "mlp"), ("transcoder", "mlp"),
    ("blocks.3.ln2.hook_normalized", "mlp"), ("model.layers.17.post_feedforward_layernorm.output", "mlp"),
    ("model.layers.3.feed_forward", "mlp"), ("model.layers.3.ffn_norm", "mlp"),
    ("attention", "attention"), ("att", "attention"), ("blocks.0.hook_attn_out", "attention"),
    ("blocks.3.attn.hook_z", "attention"), ("layers.0.attention", "attention"),
    ("model.layers.17.self_attn.o_proj.input", "attention"),
]


@pytest.mark.parametrize("name, kind", REAL_NAMES)
def test_a_real_hook_name_is_classified_as_the_point_it_names(name, kind):
    assert classify_hook(name) == kind
    assert (non_residual_hook_reason(name, "X") is None) == (kind == "residual")


def test_the_resid_pre_and_resid_mid_refusals_are_the_round_2_text_unchanged():
    """Review R3-B widened the rule (feed-forward names, embeddings) and left these as R2-B wrote them."""
    assert non_residual_hook_reason("blocks.8.hook_resid_pre", "Circuit capture") == (
        "Circuit capture reads every SAE at its layer's output (resid_post), and this SAE (hook "
        "'blocks.8.hook_resid_pre') reads the residual stream before the layer (resid_pre, the previous "
        "layer's output): the result would describe another point in the model."
    )
    assert non_residual_hook_reason("blocks.0.hook_resid_mid", "Steering") == (
        "Steering reads every SAE at its layer's output (resid_post), and this SAE (hook "
        "'blocks.0.hook_resid_mid') reads the residual stream halfway through the layer (resid_mid, after "
        "attention and before the MLP): the result would describe another point in the model."
    )


# ── the writers use it: the download task and local import ─────────────────


class _Query:
    def __init__(self, row):
        self.row = row

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.row


class _Session:
    def __init__(self, row):
        self.row = row
        self.commits = 0

    def query(self, model):
        assert model is ExternalSAE
        return _Query(self.row)

    def commit(self):
        self.commits += 1


def _run_download(monkeypatch, tmp_path, repo_id: str, filepath: str, lay_out: Callable[[Path], None]) -> ExternalSAE:
    """The real download task body (copied harness of test_sae_hook_recorded_at_import)."""
    from src.services.huggingface_sae_service import HuggingFaceSAEService
    from src.workers import sae_tasks

    row = ExternalSAE(id="sae_dl", name="download", source="huggingface", status="pending", progress=0.0, sae_metadata={})
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    session = _Session(row)

    @contextmanager
    def get_sync_db():
        yield session

    calls = []

    async def download_sae(repo_id, filepath, local_dir, **kwargs):
        calls.append((repo_id, filepath))
        target = Path(local_dir) / filepath
        lay_out(target)
        return {"local_path": str(target)}

    monkeypatch.setattr(sae_tasks, "get_sync_db", get_sync_db)
    monkeypatch.setattr(sae_tasks, "emit_sae_download_progress", lambda **kwargs: None)
    monkeypatch.setattr(HuggingFaceSAEService, "download_sae", staticmethod(download_sae))
    result = sae_tasks.download_sae_task.run(sae_id=row.id, repo_id=repo_id, filepath=filepath)
    assert result["status"] == "success" and calls == [(repo_id, filepath)]
    assert row.status == "ready" and session.commits > 0
    return row


def _writer(config_name, cfg, weights):
    def lay_out(target: Path):
        target.mkdir(parents=True, exist_ok=True)
        if config_name:
            (target / config_name).write_text(json.dumps(cfg))
        (target / weights).write_bytes(b"")
    return lay_out


@pytest.mark.parametrize("repo_id, filepath, config_name, cfg, weights, hook, layer", [
    # community_standard format: get_sae_info reads only hook_point_layer, so these were NULL
    ("Juliushanhanhan/llama-3-8b-it-res", "blocks.25.hook_resid_post", "cfg.json", LLAMA3_RES_SAELENS5, "sae_weights.safetensors", "blocks.25.hook_resid_post", 25),
    ("someone/gemma-2-2b-saes", "blocks.20.hook_resid_post", "cfg.json", SAELENS6, "sae_weights.safetensors", "blocks.20.hook_resid_post", 20),
    # format "unknown": the task had no metadata at all, so it wrote no layer
    ("EleutherAI/sae-pythia-70m-32k", "layers.0.mlp", "cfg.json", SPARSIFY_PYTHIA, "sae.safetensors", "layers.0.mlp", 0),
    ("google/gemma-scope-2-4b-pt", "mlp_out/layer_17_width_16k_l0_big", "config.json", GS2_MLP_17, "params.safetensors",
     "model.layers.17.post_feedforward_layernorm.output", 17),
], ids=["saelens5", "saelens6", "sparsify-mlp", "gemma-scope-2-mlp"])
def test_the_download_task_records_the_layer_and_hook(monkeypatch, tmp_path, repo_id, filepath, config_name, cfg, weights, hook, layer):
    row = _run_download(monkeypatch, tmp_path, repo_id, filepath, _writer(config_name, cfg, weights))
    assert (row.hook_type, row.layer) == (hook, layer)


def test_the_download_task_still_takes_a_gemma_scope_1_layer_from_its_folder(monkeypatch, tmp_path):
    def lay_out(target: Path):
        _lay_out(target.parent, target.name, None, None, "params.npz")
    row = _run_download(monkeypatch, tmp_path, "google/gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_82", lay_out)
    assert (row.hook_type, row.layer) == ("residual", 12)


@pytest.mark.asyncio
async def test_local_import_fills_a_missing_layer_and_refuses_one_that_disagrees(
    async_session, monkeypatch, tmp_path
):
    """R3B-14, decided here: a requested layer contradicting the SAE's own record is REFUSED.

    It used to be kept, with a ``logger.warning`` nobody reads. The layer decides where every
    consumer reads -- extraction hooks it, the export and the push publish and key by it,
    steering steers at it -- so when the request and the files disagree one of them is wrong,
    and this repo refuses rather than picks.

    Also pinned: the refusal happens BEFORE the copy. Resolution ran after ``copytree``, so a
    refused import would have left its bytes in SAE storage with no row pointing at them.
    """
    from src.schemas.sae import SAEImportFromFileRequest

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    _lay_out(tmp_path / "data" / "imports", "llama25", "cfg.json", LLAMA3_RES_SAELENS5, "sae_weights.safetensors")

    filled = await SAEManagerService.import_from_file(
        async_session, SAEImportFromFileRequest(file_path="imports/llama25", name="a")
    )
    assert (filled.hook_type, filled.layer) == ("blocks.25.hook_resid_post", 25)

    # A request that AGREES with the record is still accepted.
    agreeing = await SAEManagerService.import_from_file(
        async_session, SAEImportFromFileRequest(file_path="imports/llama25", name="b", layer=25)
    )
    assert agreeing.layer == 25

    storage = tmp_path / "data" / "saes"
    before = sorted(p.name for p in storage.iterdir()) if storage.exists() else []

    with pytest.raises(ValueError, match="record layer 25"):
        await SAEManagerService.import_from_file(
            async_session, SAEImportFromFileRequest(file_path="imports/llama25", name="c", layer=7)
        )

    after = sorted(p.name for p in storage.iterdir()) if storage.exists() else []
    assert after == before, "a refused import copied the SAE into storage with no row to find it by"
