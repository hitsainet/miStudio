"""G38: pooling skips the chat template's fixed prefix, pinned against miLLM by a shared case file.

`docs/schemas/template-prefix-cases.json` is byte-identical in miStudio and miLLM. miStudio computes
the prefix in `probe_monitor_render` (`template_prefix_ids`, `row_template_prefix`) and applies it in
`RenderedExample.scored_mask`; miLLM computes it in `probe_turns` and applies it in
`probe_scope.scored_mask`. Two implementations mirrored by hand drift silently unless each is held
to the same answers, so each repo runs its OWN code over every case and asserts P and the mask of
every window, and asserts the two copies are byte-identical.

* `structural` — WordLevel tokenizers built from the file itself, so they ALWAYS run;
* `real` — named real tokenizers, identified by the sha256 of their chat template. A missing one
  SKIPS LOUDLY (the reason names it); it never passes vacuously.

Regenerating the file is a contract change: the expected values were derived by hand for the
structural set (`0xcc/reviews/g38_template_prefix_2026-11-08.md`) and must be re-read, not
re-blessed.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

import pytest

from src.services.probe_monitor_render import (
    CONTRACT_WINDOW_SCOPES,
    render_messages,
    template_prefix_ids,
    template_prefix_length,
)
from tests.support.sibling_repos import miserver_repo
from tests.support.cross_repo import require_cross_repo, test_env

REPO = Path(__file__).resolve().parents[3]
CASES_PATH = REPO / "docs" / "schemas" / "template-prefix-cases.json"
MILLM_CASES = miserver_repo() / "docs" / "schemas" / "template-prefix-cases.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))
STRUCTURAL = CASES["structural"]
REAL = CASES["real"]


def _structural_tokenizer(case):
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
    if "stale_cached_prefix" in case:
        # The state a date-stamped preamble leaves after midnight: yesterday's prefix in the cache.
        from src.services import probe_monitor_render as render

        render.template_prefix_ids(fast)
        render._header_cache[fast][f"template_prefix:{render.template_hash(fast)}"] = (
            fast.convert_tokens_to_ids(case["stale_cached_prefix"])
        )
    return fast


def _bits(mask):
    return "".join("1" if keep else "0" for keep in mask)


def _masks(row, windows):
    out = {}
    for window in windows:
        scope = CONTRACT_WINDOW_SCOPES[window]
        if scope == "last_user" and not row.role_mask_reliable:
            out[window] = _bits([False] * len(row.input_ids))   # `_row_mask`'s rule
        else:
            out[window] = _bits(row.scored_mask(scope))
    return out


def _check(tokenizer, case, ids_key):
    expected = case["expected"]
    row = render_messages(tokenizer, case["messages"], skip_template_prefix=True)
    if ids_key == "tokens":
        assert tokenizer.convert_ids_to_tokens(row.input_ids) == expected["tokens"]
    else:
        assert row.input_ids == expected["ids"]
    assert row.prompt_tokens == expected["prompt_tokens"]
    assert row.template_prefix_tokens == expected["template_prefix"]
    # The function miLLM mirrors, on the served ids alone.
    assert template_prefix_length(tokenizer, row.input_ids) == expected["template_prefix"]
    assert _masks(row, expected["masks"]) == expected["masks"]


@pytest.mark.parametrize("case", STRUCTURAL["cases"], ids=lambda c: c["name"])
def test_structural_cases(case):
    _check(_structural_tokenizer(case), case, "tokens")


@pytest.mark.parametrize("case", STRUCTURAL["cases"], ids=lambda c: c["name"])
def test_without_the_skip_every_window_is_what_it_was(case):
    """THE OLD-PROBE GUARANTEE, per case: a row rendered without the flag is masked exactly as before
    G38 — `all` every position, `prompt` `[0, prompt_tokens)`, `response` the rest, `last_user` the
    span the file pins — stated from the counts, not from the code under test."""
    tokenizer = _structural_tokenizer(case)
    row = render_messages(tokenizer, case["messages"])
    n, boundary = len(row.input_ids), case["expected"]["prompt_tokens"]
    prefix = case["expected"]["template_prefix"]
    assert row.skip_template_prefix is False and row.template_prefix_ids is None
    masks = _masks(row, ("all", "prompt", "response", "last_user"))
    assert {k: masks[k] for k in ("all", "prompt", "response")} == {
        "all": "1" * n,
        "prompt": "1" * boundary + "0" * (n - boundary),
        "response": "0" * boundary + "1" * (n - boundary),
    }
    # `last_user` WITHOUT the skip is one contiguous span; WITH it, that span less [0, P).
    on = [i for i, bit in enumerate(masks["last_user"]) if bit == "1"]
    assert on == list(range(on[0], on[-1] + 1)) if on else True
    skipped = [i for i, bit in enumerate(case["expected"]["masks"]["last_user"]) if bit == "1"]
    assert skipped == [i for i in on if i >= prefix]


def test_the_structural_set_covers_every_shape():
    """The set is only a guard if it reaches the shapes the operator decided on: a full prefix, a row
    with its own system prompt that shares only part of it (G38-3), a P of 0, multi-turn, an
    assistant-ended row, and a tokenizer-added BOS."""
    cases = STRUCTURAL["cases"]
    prefixes = {c["expected"]["template_prefix"] for c in cases}
    assert 0 in prefixes and len(prefixes) >= 3
    assert any(c["messages"][0]["role"] == "system" and c["expected"]["template_prefix"] > 0 for c in cases)
    assert any(len([m for m in c["messages"] if m["role"] == "user"]) > 1 for c in cases)
    assert any(c["messages"][-1]["role"] == "assistant" for c in cases)
    assert any(
        c["tokenizer_adds_bos"] and not STRUCTURAL["templates"][c["template"]].startswith("<s>")
        for c in cases
    )
    # `last_user` skips the prefix too (operator, 2026-11-08): no case scores [0, P) in it, and a
    # single-turn case (span starting INSIDE the prefix) and a multi-turn one (span after it) exist.
    assert not any(
        "1" in c["expected"]["masks"]["last_user"][: c["expected"]["template_prefix"]] for c in cases
    )
    assert any(len(c["messages"]) == 1 and c["expected"]["template_prefix"] > 0
               and "1" in c["expected"]["masks"]["last_user"] for c in cases)
    assert any(len([m for m in c["messages"] if m["role"] == "user"]) > 1
               and "1" in c["expected"]["masks"]["last_user"] for c in cases)
    # And the two cases that are not a template alone.
    assert any("stale_cached_prefix" in c for c in cases)
    assert any("vocab" in c and c["expected"]["template_prefix"] == 0 for c in cases)


def test_the_prefix_is_the_start_tokens_and_first_user_header():
    """`template_prefix_ids` is `first_user_header`'s prefix placed in the served ids, on every
    structural template that has a header — one machinery, as the design note requires."""
    from src.services.probe_monitor_render import first_user_header

    checked = 0
    for case in STRUCTURAL["cases"]:
        tokenizer = _structural_tokenizer(case)
        if "vocab" in case:
            continue        # probes it cannot tell apart: the prefix is deliberately empty
        found = first_user_header(tokenizer, refresh=True)
        served = template_prefix_ids(tokenizer, refresh=True)
        if found is None:
            continue
        assert served[len(served) - len(found[0]):] == found[0]
        checked += 1
    assert checked >= 5


# ── the named real tokenizers: skip LOUDLY when absent ──────────────────────────────────────


def _real_tokenizer(hf_id):
    sha = REAL["tokenizers"][hf_id]["chat_template_sha256"]
    dirs = []
    configured = test_env("MIWORKBENCH_REAL_TOKENIZERS", "MISTUDIO_REAL_TOKENIZERS")
    if configured:
        dirs += [Path(p) for p in configured.split(os.pathsep) if p]
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    dirs += [Path(p) for p in sorted(glob.glob(str(hub / f"models--{hf_id.replace('/', '--')}" / "snapshots" / "*")))]
    from transformers import AutoTokenizer

    for path in dirs:
        if not (path / "tokenizer_config.json").exists():
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(path))
        except Exception:  # noqa: BLE001 - a snapshot this transformers cannot read
            continue
        template = getattr(tokenizer, "chat_template", None) or ""
        if hashlib.sha256(template.encode("utf-8")).hexdigest() == sha:
            return tokenizer
    pytest.skip(
        f"NO {hf_id} TOKENIZER WITH CHAT TEMPLATE {sha[:12]} ON THIS MACHINE — the real-tokenizer "
        "template-prefix cases DID NOT RUN. Set MIWORKBENCH_REAL_TOKENIZERS (or MISTUDIO_REAL_TOKENIZERS) (os.pathsep-separated "
        "tokenizer directories) to check them; the structural cases ran."
    )


_REAL_TOKENIZERS = {}


@pytest.mark.parametrize("case", REAL["cases"], ids=lambda c: c["name"])
def test_real_tokenizer_cases(case):
    if case["tokenizer"] not in _REAL_TOKENIZERS:
        _REAL_TOKENIZERS[case["tokenizer"]] = _real_tokenizer(case["tokenizer"])
    _check(_REAL_TOKENIZERS[case["tokenizer"]], case, "ids")


# ── one rule, two repos ─────────────────────────────────────────────────────────────────────


def test_the_case_file_is_identical_in_millm():
    """⚠ TWO IMPLEMENTATIONS, ONE PREFIX. If miLLM's copy differs, the bar is cut over one set of
    positions and served over another — the defect class this file exists for."""
    if not MILLM_CASES.exists():
        if require_cross_repo():
            pytest.fail(f"miLLM's copy of the template-prefix cases is missing at {MILLM_CASES}")
        pytest.skip("miLLM checkout not present")
    assert MILLM_CASES.read_bytes() == CASES_PATH.read_bytes()
