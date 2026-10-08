"""The probe served-render rule, pinned against miLLM by a shared case file (2026-10-08).

miStudio's `probe_monitor_render.served_render` and miLLM's `probe_scoring.served_render` are
mirrored BY HAND. Every probe this repo trains, calibrates and exports is rendered by the first;
every input miLLM scores is rendered by the second. Until this file nothing tied them together,
and the defects of the week this was written were exactly that class — two paths drifting
silently while each repo's own suite stayed green.

`docs/schemas/served-render-cases.json` is byte-identical in both repos. Each repo runs its OWN
renderer over every case and asserts the ids, the generation-prompt branch, the prompt/response
boundary and the `last_user` span. Two sets:

* `structural` — WordLevel tokenizers built from the file itself, so they ALWAYS run;
* `real` — the Llama-3.1-8B-Instruct tokenizer, identified by its chat-template hash. When no
  tokenizer with that hash is on this machine the real cases SKIP LOUDLY (the reason names the
  hash and the variable to set); they never pass vacuously.

A case may carry `known_divergence.mistudio`: a place where this repo is KNOWN not to meet the
rule miLLM serves. It runs as a STRICT xfail, so it goes red the moment either side changes: the
divergence is recorded, never pinned as correct. `expected` stays the rule. TODAY THERE IS NONE.
The one there was — F1, a template that writes no BOS while its tokenizer adds one, where miLLM
served one BOS and this trained on none — closed on 2026-10-08 when miStudio adopted miLLM's
start-of-text rule (`prompt_encoding`), and `test_no_case_is_a_known_divergence_any_more` keeps
the marker from creeping back without a decision.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from src.services.probe_monitor_render import render_messages, served_render

REPO = Path(__file__).resolve().parents[3]
CASES_PATH = REPO / "docs" / "schemas" / "served-render-cases.json"
MILLM_CASES = (
    Path(os.environ.get("MILLM_REPO", "/home/x-sean/app/miLLM"))
    / "docs" / "schemas" / "served-render-cases.json"
)
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))
STRUCTURAL = CASES["structural"]
REAL = CASES["real"]


def _params(cases):
    out = []
    for case in cases:
        divergence = (case.get("known_divergence") or {}).get("mistudio")
        marks = [pytest.mark.xfail(strict=True, reason=f"KNOWN DIVERGENCE: {divergence}")] if divergence else []
        out.append(pytest.param(case, marks=marks, id=case["name"]))
    return out


# ── the tokenizer-free set: always runs ─────────────────────────────────────────────────────


def _structural_tokenizer(case):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = STRUCTURAL["vocab"]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if case["tokenizer_adds_bos"]:
        backend.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", vocab.index("<s>"))]
        )
    bos = case.get("bos_token", "<s>")
    kwargs = {"bos_token": bos} if bos else {}
    fast = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", **kwargs)
    fast.chat_template = STRUCTURAL["templates"][case["template"]]
    return fast


def _last_user_span(row):
    mask = row.scored_mask("last_user")
    on = [i for i, keep in enumerate(mask) if keep]
    if not on:
        return None
    assert on == list(range(on[0], on[-1] + 1)), "the last_user window must be one contiguous span"
    return [on[0], on[-1] + 1]


def _check(tokenizer, case, ids_key):
    expected = case["expected"]
    messages = case["messages"]
    text, ids, prompt_tokens = served_render(tokenizer, messages)
    # WHICH BRANCH: the text is the template's render with exactly the expected generation flag.
    assert text == tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=expected["generation_prompt"]
    )
    if ids_key == "tokens":
        assert tokenizer.convert_ids_to_tokens(ids) == expected["tokens"]
    else:
        assert ids == expected["ids"]
    assert prompt_tokens == expected["prompt_tokens"]

    # THE WIRING every probe input goes through: render_messages, not served_render alone.
    row = render_messages(tokenizer, messages)
    assert row.input_ids == ids and row.served_form is True
    assert row.prompt_tokens == expected["prompt_tokens"]
    assert _last_user_span(row) == expected["last_user_span"]

    if "max_length" in case:
        cut = render_messages(tokenizer, messages, max_length=case["max_length"])
        assert cut.truncated is True
        assert cut.input_ids == expected["truncated_ids"]


@pytest.mark.parametrize("case", _params(STRUCTURAL["cases"]))
def test_structural_cases(case):
    _check(_structural_tokenizer(case), case, "tokens")


def test_the_structural_set_covers_every_branch():
    """The set is only a guard if it reaches both branches, a null boundary and a doubled BOS."""
    expected = [c["expected"] for c in STRUCTURAL["cases"]]
    assert {e["generation_prompt"] for e in expected} == {True, False}
    assert any(e["prompt_tokens"] is None for e in expected)
    assert any(e["last_user_span"] is None for e in expected)
    assert any(
        c["tokenizer_adds_bos"] and STRUCTURAL["templates"][c["template"]].startswith("<s>")
        for c in STRUCTURAL["cases"]
    ), "no case where the template AND the tokenizer would each supply a BOS"
    assert any(c["messages"][-1]["role"] == "tool" for c in STRUCTURAL["cases"])


def test_no_case_is_a_known_divergence_any_more():
    """F1 closed (operator, 2026-10-08): miStudio uses miLLM's start-of-text rule. A new divergence
    is a DECISION — re-adding the marker must fail here and be argued in a review record."""
    marked = [c["name"] for c in STRUCTURAL["cases"] + REAL["cases"] if c.get("known_divergence")]
    assert marked == []


def test_the_f1_form_is_in_the_set():
    """The case that WAS the divergence stays: a template with no BOS and a BOS-adding tokenizer,
    expected to serve exactly one BOS (from the tokenizer). Removing it would un-pin the rule."""
    f1 = [
        c for c in STRUCTURAL["cases"]
        if c["tokenizer_adds_bos"] and not STRUCTURAL["templates"][c["template"]].startswith("<s>")
    ]
    assert f1 and all(c["expected"]["tokens"][:2] == ["<s>", "<|user|>"] for c in f1)


# ── the named real tokenizer: skips LOUDLY when absent ──────────────────────────────────────


def _real_tokenizer():
    sha = REAL["tokenizer"]["chat_template_sha256"]
    dirs = []
    configured = os.environ.get("MISTUDIO_REAL_TOKENIZERS")
    if configured:
        dirs += [Path(p) for p in configured.split(os.pathsep) if p]
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    for repo in ("models--meta-llama--Llama-3.1-8B-Instruct", "models--unsloth--Llama-3.1-8B-Instruct"):
        dirs += sorted((hub / repo / "snapshots").glob("*"))
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
        f"NO {REAL['tokenizer']['hf_id']} TOKENIZER WITH CHAT TEMPLATE {sha[:12]} ON THIS MACHINE "
        "— the real-tokenizer served-render cases DID NOT RUN. Set MISTUDIO_REAL_TOKENIZERS "
        "(os.pathsep-separated tokenizer directories) to check them; the structural cases ran."
    )


@pytest.fixture(scope="module")
def llama():
    return _real_tokenizer()


@pytest.mark.parametrize("case", _params(REAL["cases"]))
def test_real_tokenizer_cases(case, llama):
    _check(llama, case, "ids")


# ── one rule, two repos ─────────────────────────────────────────────────────────────────────


def test_the_case_file_is_identical_in_millm():
    """⚠ TWO RENDERERS, ONE RULE. If miLLM's copy differs, the two sides are pinned to different
    rules and each suite stays green while they drift — the defect class this file exists for."""
    if not MILLM_CASES.exists():
        if os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(f"miLLM's copy of the served-render cases is missing at {MILLM_CASES}")
        pytest.skip("miLLM checkout not present")
    assert MILLM_CASES.read_bytes() == CASES_PATH.read_bytes()
