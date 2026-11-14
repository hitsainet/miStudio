"""A unified/multimodal config has no top-level `vocab_size`.

Reported live, 2026-08-23: two extraction jobs on `gemma-4-12B-it` failed with

    Extraction failed: 'Gemma4UnifiedConfig' object has no attribute 'vocab_size'

after 18s and 2m48s, both producing 0 samples.

`model.config.vocab_size` names one architecture's layout. Unified and
multimodal configs keep the text-model fields on a SUB-CONFIG — `text_config`,
`llm_config`, `decoder`, by family — and the top level carries only the
composition. That is exactly the assumption `layer_discovery` exists to remove
everywhere else, and this one line still made it.
"""

import pytest
import torch
from torch import nn

from src.ml.layer_discovery import resolve_vocab_size


class _TextConfig:
    vocab_size = 262_144


class _Gemma4UnifiedConfig:
    """The reported shape: composition at the top, text fields nested."""

    model_type = "gemma4_unified"
    text_config = _TextConfig()
    vision_config = object()


class _PlainConfig:
    vocab_size = 32_000


class _EmptyConfig:
    model_type = "mystery"


def _model(config, vocab_rows=None):
    m = nn.Module()
    m.config = config
    if vocab_rows is not None:
        emb = nn.Embedding(vocab_rows, 8)
        m.get_input_embeddings = lambda: emb
    else:
        m.get_input_embeddings = lambda: None
    return m


def test_the_reported_failure_now_resolves():
    """The exact config class name from the error message."""
    assert resolve_vocab_size(_model(_Gemma4UnifiedConfig())) == 262_144


def test_a_plain_decoder_config_is_unaffected():
    """Negative control for the direction of the fix: the common case must not
    change, and must not silently start reading the embedding table instead."""
    assert resolve_vocab_size(_model(_PlainConfig(), vocab_rows=99)) == 32_000


def test_the_embedding_table_is_the_fallback():
    """Ground truth. A token id is usable iff the table has a row for it,
    whatever the config says — and it is the bound the caller actually needs."""
    assert resolve_vocab_size(_model(_EmptyConfig(), vocab_rows=50_257)) == 50_257


def test_none_when_nothing_can_answer():
    """Not a default. Substituting a guess would put a made-up bound on a real
    token-id validation, which is worse than the crash it replaced."""
    assert resolve_vocab_size(_model(_EmptyConfig())) is None


def test_a_nested_config_under_an_unexpected_name_still_resolves():
    """Discovered, not named. A family this shop has not run yet is covered
    without an edit — the same rule the three narrow-scope guards in this audit
    each broke."""

    class _Odd:
        pass

    class _Wrapper:
        llm = _Odd()

    _Wrapper.llm.vocab_size = 12_345
    assert resolve_vocab_size(_model(_Wrapper())) == 12_345


def test_extraction_refuses_rather_than_guessing():
    """The caller must fail loudly when the size is unknowable."""
    import inspect

    from src.services.activation_service import ActivationService

    import ast
    import textwrap

    src = inspect.getsource(ActivationService)
    assert "resolve_vocab_size(model)" in src

    # Parse the ATTRIBUTE ACCESS. The replacement comment quotes
    # `model.config.vocab_size` to explain why it was removed, and a substring
    # check cannot tell the explanation from the code. (Fifth time this trap has
    # appeared in this remediation — the rule is: parse, or strip the prose.)
    tree = ast.parse(textwrap.dedent(src))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "vocab_size"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "config"
    ]
    assert not offenders, (
        f"direct `.config.vocab_size` reads remain at lines {offenders} — that "
        f"is what failed on gemma-4-12B-it"
    )


def test_a_zero_or_negative_config_value_is_not_trusted():
    """A malformed config must fall through to ground truth, not return 0 and
    make every token id look out of range."""

    class _Zero:
        vocab_size = 0

    assert resolve_vocab_size(_model(_Zero(), vocab_rows=1000)) == 1000
