"""G38 review round 3: lines no earlier test caught (mutation survivors), pinned.

R3-S2: `RenderedExample.scored_mask` gates the skip on the ROW's flag (`skip_template_prefix`). Every
test so far reached it only through `render_messages`, which sets the flag and `template_prefix_ids`
together, so dropping the flag check left the suite green: the second gate (no prefix ids) hid it.
A row that KNOWS its template prefix but was not rendered to skip it must keep every position; that
is the absent-means-false guarantee at the one place the mask is cut.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.probe_monitor_render import (
    TEMPLATE_PREFIX_SCOPES,
    render_messages,
    row_template_prefix,
)

REPO = Path(__file__).resolve().parents[3]
CASES = json.loads((REPO / "docs" / "schemas" / "template-prefix-cases.json").read_text(encoding="utf-8"))
STRUCTURAL = CASES["structural"]


def _tokenizer(case):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = case.get("vocab") or STRUCTURAL["vocab"]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if case["tokenizer_adds_bos"]:
        backend.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", vocab.index("<s>"))]
        )
    fast = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", unk_token="[UNK]")
    fast.chat_template = STRUCTURAL["templates"][case["template"]]
    return fast


@pytest.mark.parametrize(
    "case",
    [c for c in STRUCTURAL["cases"] if c["expected"]["template_prefix"] > 0 and "stale_cached_prefix" not in c],
    ids=lambda c: c["name"],
)
def test_a_row_that_knows_its_prefix_but_does_not_skip_keeps_every_position(case):
    tokenizer = _tokenizer(case)
    row = render_messages(tokenizer, case["messages"])          # no flag: a pre-G38 run's row
    row.template_prefix_ids = row_template_prefix(tokenizer, row.input_ids)
    # The fixture must make the two behaviours differ, or the test is decorative.
    assert row.template_prefix_tokens == case["expected"]["template_prefix"] > 0
    assert row.skip_template_prefix is False
    for scope in sorted(TEMPLATE_PREFIX_SCOPES):
        assert row.scored_mask(scope) == row._scope_mask(scope), scope
