"""
Phase 8.2/8.3: the two guards that keep J-space inside its envelope.

BR-006 / PADR IDL-42 say `W_U J` is never formed. That is a statement about
what the CODE DOES, not about how big a file is, so it needs a guard that
watches allocations rather than a size check — a materialisation that happens
transiently in memory and is then discarded never shows up on disk at all.

BR-032 says structure comes from discovery, never a whitelist. This workbench
deleted `SUPPORTED_ARCHITECTURES` once already, and the way that whitelist
survived as long as it did was a suite that only ever exercised one
architecture.

MUTATION CONTROLS (each must turn this file red):
  * materialise W_U @ J anywhere on the readout path -> "no vocab allocation" fails
  * branch on a model name in a J-space module       -> "no architecture name" fails
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

MODULES = [
    "src.services.jlens_readout_service",
    "src.services.jlens_artifact_service",
    "src.services.jlens_validation",
    "src.services.jlens_band_report",
    "src.services.jlens_band_service",
    "src.services.jlens_model_registry",
    "src.ml.jlens_fitter",
    "src.ml.jlens_metrics",
]

ARCH_NAMES = ("lfm2", "gemma", "llama", "granite", "qwen", "mistral", "phi", "gpt2")


def _module(name: str):
    import importlib

    return importlib.import_module(name)


@pytest.mark.parametrize("module_name", MODULES)
def test_no_architecture_name_in_an_executable_path(module_name: str):
    """Docstrings may NAME a model when explaining why; code may not branch on one.

    The distinction matters: the reason a rule exists is often a specific
    model's behaviour, and forbidding that in prose would push the explanation
    out of the code. What is forbidden is a decision that depends on which
    architecture is loaded.
    """
    tree = ast.parse(inspect.getsource(_module(module_name)))

    for node in ast.walk(tree):
        # Identifiers: names, attributes, function/class definitions.
        labels = []
        if isinstance(node, ast.Name):
            labels.append(node.id)
        elif isinstance(node, ast.Attribute):
            labels.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            labels.append(node.name)
        for label in labels:
            lowered = label.lower()
            assert not any(a in lowered for a in ARCH_NAMES), (
                f"{module_name} has an architecture name in an identifier: "
                f"{label!r}. Structure comes from discovery (BR-032)."
            )

    # String COMPARISONS against a model name are the whitelist shape.
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for operand in [node.left, *node.comparators]:
                if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                    lowered = operand.value.lower()
                    assert not any(a in lowered for a in ARCH_NAMES), (
                        f"{module_name} compares against {operand.value!r} — "
                        "that is the whitelist this project already deleted once."
                    )


class AllocationWatcher(torch.overrides.TorchFunctionMode):
    """Records the shape of every tensor a torch op returns.

    A guard on PEAK MEMORY would be flaky and would also pass a
    materialisation that fits. Watching shapes catches the specific thing
    BR-006 forbids: a tensor with a VOCABULARY-sized dimension paired with
    d_model, which is `W_U J` however it was spelled.
    """

    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor):
            self.shapes.append(tuple(out.shape))
        return out


N_VOCAB = 997  # prime, so it cannot coincide with a d_model or layer count
D_MODEL = 8


def _readout_service():
    from src.services.jlens_readout_service import ReadoutService

    class Block(torch.nn.Module):
        def forward(self, hidden, **_):
            return (hidden * 1.01,)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block() for _ in range(3)])
            self.embed = torch.nn.Embedding(64, D_MODEL)
            self.config = None

        def forward(self, input_ids=None, **_):
            hidden = self.embed(input_ids)
            for b in self.blocks:
                hidden = b(hidden)[0]
            return hidden

    class Structure:
        def __init__(self, model):
            self.layers_module = model.blocks
            self.num_layers = 3
            self.attention_module = None
            self.residual_norm_module = None

    class Tok:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, ids, **_):
            return "tok"

        def convert_ids_to_tokens(self, ids):
            return ["tok"] * len(ids) if isinstance(ids, list) else "tok"

    model = Model()
    return ReadoutService(
        model=model,
        tokenizer=Tok(),
        structure=Structure(model),
        unembedding=torch.randn(N_VOCAB, D_MODEL),
        model_name="test",
    )


def test_the_readout_path_never_allocates_a_vocab_by_d_model_tensor():
    """`W_U J` is never formed — asserted on allocations, not on file size.

    A transient materialisation is discarded before anything is written, so a
    size check on the artifact cannot see it. At the reference model this
    tensor would be 268 MB per layer and 4.3 GB across the stack.
    """
    from src.services.jlens_readout_service import JacobianTransport

    service = _readout_service()
    transport = JacobianTransport({0: torch.eye(D_MODEL), 1: torch.eye(D_MODEL), 2: torch.eye(D_MODEL)})

    watcher = AllocationWatcher()
    with watcher:
        list(service.stream("abc", [transport], top_n=3))

    forbidden = [
        shape
        for shape in watcher.shapes
        if len(shape) == 2 and N_VOCAB in shape and D_MODEL in shape
    ]
    # W_U itself is [n_vocab, d_model] and is a legitimate single tensor; what
    # is forbidden is PRODUCING another one, which is what W_U @ J does.
    assert len(forbidden) <= 1, (
        f"a [{N_VOCAB}, {D_MODEL}] tensor was produced on the readout path: "
        f"{forbidden}. That shape is the materialised dictionary (BR-006)."
    )


def test_the_watcher_would_catch_a_materialisation():
    """Negative control for the guard itself.

    Without this, a watcher that silently recorded nothing would pass the test
    above forever — the guard has to be shown to bite.
    """
    W_U = torch.randn(N_VOCAB, D_MODEL)
    J = torch.eye(D_MODEL)

    watcher = AllocationWatcher()
    with watcher:
        _ = W_U @ J  # the prohibited operation, performed deliberately

    forbidden = [
        shape
        for shape in watcher.shapes
        if len(shape) == 2 and N_VOCAB in shape and D_MODEL in shape
    ]
    assert len(forbidden) >= 1, "the allocation watcher does not see the shape it guards"
