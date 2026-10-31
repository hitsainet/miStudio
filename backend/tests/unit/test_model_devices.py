"""On a split model, "the model's device" is not one device.

`model.device` is the first parameter's device. Written for one card, code moved
inputs, SAEs and steering vectors there; on a model split across cards that is
the wrong card for every layer on another one. These tests use `meta` and `cpu`
tensors to stand in for two cards, so they run anywhere.
"""

import torch

from src.ml import model_devices as md


class _Split(torch.nn.Module):
    """A decoder whose first registered parameter is NOT the embedding."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, 4, device="meta")       # "the other card"
        self.embed = torch.nn.Embedding(10, 4, device="cpu")   # the input card

    def get_input_embeddings(self):
        return self.embed


def test_inputs_go_to_the_embedding_not_the_first_parameter():
    model = _Split()
    assert next(model.parameters()).device.type == "meta"
    assert md.input_device(model).type == "cpu"


def test_a_model_without_the_accessor_uses_its_first_tensor():
    model = torch.nn.Linear(2, 2, device="cpu")
    assert md.input_device(model).type == "cpu"


def test_every_device_the_model_holds_tensors_on_is_listed():
    assert {d.type for d in md.model_devices(_Split())} == {"cpu", "meta"}


def test_off_gpu_modules_are_read_from_the_device_map():
    model = _Split()
    model.hf_device_map = {"embed": 0, "layers.0": 1, "layers.1": "disk", "head": "cpu"}
    assert md.off_gpu_modules(model) == {"layers.1": "disk", "head": "cpu"}


def test_a_model_loaded_onto_one_device_has_no_map_and_no_offload():
    assert md.off_gpu_modules(torch.nn.Linear(2, 2)) == {}


def test_hooks_are_removed_from_a_dispatched_model():
    from accelerate.hooks import ModelHook, add_hook_to_module

    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    add_hook_to_module(model[0], ModelHook())
    assert hasattr(model[0], "_hf_hook")

    md.detach_dispatch_hooks(model)

    assert not hasattr(model[0], "_hf_hook")
