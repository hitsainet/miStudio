"""miStudio uses miLLM's start-of-text (BOS) rule for probe renders (operator decision on F1, 2026-10-08).

Until this date `served_render` always passed `add_special_tokens=False`. On a template that writes
no BOS while its tokenizer adds one (TinyLlama-1.1B-Chat, in miLLM's dev model list) miLLM serves
one BOS and miStudio trained on none. Now both use one rule (`src/services/prompt_encoding.py`,
mirroring miLLM `millm/services/prompt_encoding.py`), and a probe's render form records what the
rule did (`bos_handling`). Review record: `0xcc/reviews/render_bos_rule_2026-10-08.md`.

Four things are pinned here:

1. **The encoder IS miLLM's**: loaded from the miLLM checkout by file and compared decision for
   decision and id for id on every tokenizer shape (skips when miLLM is absent; REQUIRED under
   `MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1`).
2. **Positions survive the tokenizer's BOS**: on the F1 form every message, the `last_user` span
   and the content mask move by the one token the tokenizer put in front; a trailing special token
   (an `<s> $A </s>` tokenizer) is scaffolding, and leaves `last_user` and the assistant-ended
   boundary unresolved exactly as miLLM does.
3. **Production probes do not change**: on the real Llama-3.1 tokenizer the rule's ids equal the
   always-False ids for every real served-render case (skips LOUDLY without the tokenizer).
4. **TinyLlama does**: on the real TinyLlama-Chat tokenizer the rule adds exactly one BOS, and an
   always-False probe on it is refused (skips LOUDLY without the tokenizer).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from src.services import prompt_encoding as ours
from src.services.probe_monitor_render import (
    always_false_rule_is_the_rule,
    render_form_record,
    render_messages,
    served_render,
)

pytest.importorskip("tokenizers")

REPO = Path(__file__).resolve().parents[3]
MILLM_ENCODING = (
    Path(os.environ.get("MILLM_REPO", "/home/x-sean/app/miLLM"))
    / "millm" / "services" / "prompt_encoding.py"
)
CASES = json.loads((REPO / "docs" / "schemas" / "served-render-cases.json").read_text("utf-8"))

WORDS = ["<s>", "</s>", "<unk>", "<|user|>", "<|assistant|>", "<|end|>", "virus", "spreads",
         "fast", "an", "answer", "first", "question", "a", "b", "x", "y"]
BOS_TEMPLATE = (
    "<s> {% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)
NO_BOS_TEMPLATE = BOS_TEMPLATE[len("<s> "):]


def _tokenizer(template=NO_BOS_TEMPLATE, *, adds=None, bos="<s>"):
    """`adds`: None (adds nothing), "bos" (`<s> $A`, TinyLlama) or "bos_eos" (`<s> $A </s>`)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(WORDS)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if adds == "bos":
        backend.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", vocab["<s>"])]
        )
    elif adds == "bos_eos":
        backend.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>", special_tokens=[("<s>", vocab["<s>"]), ("</s>", vocab["</s>"])]
        )
    kwargs = {"bos_token": bos} if bos else {}
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="</s>", **kwargs)
    tok.chat_template = template
    return tok


SHAPES = {
    "template writes the BOS, tokenizer would add one (Llama 3)": dict(template=BOS_TEMPLATE, adds="bos"),
    "template writes the BOS, tokenizer adds nothing": dict(template=BOS_TEMPLATE),
    "F1: no BOS in the template, tokenizer adds one (TinyLlama)": dict(adds="bos"),
    "no BOS in the template, tokenizer adds BOS and EOS": dict(adds="bos_eos"),
    "a bos_token nobody writes or adds (granite)": dict(),
    "no bos_token at all (Qwen2.5)": dict(bos=None),
}
TURNS = [
    [{"role": "user", "content": "virus spreads fast"}],
    [{"role": "user", "content": "first question"}, {"role": "assistant", "content": "an answer"},
     {"role": "user", "content": "virus spreads fast"}],
    [{"role": "user", "content": "virus spreads fast"}, {"role": "assistant", "content": "an answer"}],
]


# ── 1. the encoder is miLLM's ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def millm_encoding():
    if not MILLM_ENCODING.exists():
        if os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(f"miLLM's prompt_encoding is missing at {MILLM_ENCODING}")
        pytest.skip("miLLM checkout not present")
    spec = importlib.util.spec_from_file_location("_millm_prompt_encoding", MILLM_ENCODING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # standard library imports only
    return module


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("generation_prompt", [True, False])
def test_the_rule_is_millm_s_decision_for_decision(millm_encoding, shape, generation_prompt):
    tok = _tokenizer(**SHAPES[shape])
    for messages in TURNS:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
        assert ours.bos_text(tok) == millm_encoding.bos_text(tok)
        assert ours.rendered_chat_adds_special_tokens(tok, text) == (
            millm_encoding.rendered_chat_adds_special_tokens(tok, text)
        )
        assert ours.rendered_chat_ids(tok, text) == millm_encoding.rendered_chat_ids(tok, text)


def test_bos_text_reads_only_a_real_string(millm_encoding):
    """A double's attribute is not a string; both repos refuse to treat it as one."""
    class Double:
        bos_token = object()

    for value in (Double(), type("Empty", (), {"bos_token": ""})(), object()):
        assert ours.bos_text(value) is None and millm_encoding.bos_text(value) is None


# ── 2. the record, and positions on the F1 form ─────────────────────────────────────────────


@pytest.mark.parametrize("shape,expected", [
    ("template writes the BOS, tokenizer would add one (Llama 3)", (False, True, False, 1)),
    ("template writes the BOS, tokenizer adds nothing", (False, True, False, 1)),
    ("F1: no BOS in the template, tokenizer adds one (TinyLlama)", (True, False, True, 1)),
    ("no BOS in the template, tokenizer adds BOS and EOS", (True, False, True, 1)),
    ("a bos_token nobody writes or adds (granite)", (True, False, False, 0)),
    ("no bos_token at all (Qwen2.5)", (True, False, False, 0)),
])
def test_the_record_says_what_the_rule_did(shape, expected):
    record = render_form_record(_tokenizer(**SHAPES[shape]))
    add, wrote, added, count = expected
    assert record == {
        "generation_prompt": True,
        "add_special_tokens": add,
        "bos_handling": {"template_wrote_bos": wrote, "tokenizer_added_bos": added, "bos_count": count},
    }


@pytest.mark.parametrize("shape,same", [
    ("template writes the BOS, tokenizer would add one (Llama 3)", True),
    ("template writes the BOS, tokenizer adds nothing", True),
    ("F1: no BOS in the template, tokenizer adds one (TinyLlama)", False),
    ("no BOS in the template, tokenizer adds BOS and EOS", False),
    ("a bos_token nobody writes or adds (granite)", True),
    ("no bos_token at all (Qwen2.5)", True),
])
def test_the_old_rule_is_the_rule_exactly_where_the_ids_agree(shape, same):
    """The old-probe policy's ground truth: identical ids → the old record is the served form."""
    tok = _tokenizer(**SHAPES[shape])
    assert always_false_rule_is_the_rule(tok) is same
    for messages in TURNS:
        text = served_render(tok, messages)[0]
        assert (ours.rendered_chat_ids(tok, text) == ours.template_ids(tok, text)) is same


class TestTheF1FormPlacesEveryPositionAfterTheTokenizersBos:
    def test_one_bos_and_every_message_shifted_by_one(self):
        f1, written = _tokenizer(adds="bos"), _tokenizer(template=BOS_TEMPLATE, adds="bos")
        messages = TURNS[1]
        a, b = render_messages(f1, messages), render_messages(written, messages)
        # Same tokens, one place apart in ownership: the F1 BOS belongs to no message, the
        # template's BOS belongs to message 0's prefix render.
        assert a.input_ids == b.input_ids and a.input_ids[:2] == [0, WORDS.index("<|user|>")]
        assert a.role_mask_reliable and a.token_message[0] == -1 and a.token_roles[0] == ""
        assert a.token_message[1:] == b.token_message[1:]
        assert a.last_user_start == b.last_user_start == 9
        assert a.prompt_tokens == len(a.input_ids)

    def test_the_content_mask_is_placed_too(self):
        row = render_messages(_tokenizer(adds="bos"), TURNS[0])
        tokens = _tokenizer(adds="bos").convert_ids_to_tokens(row.input_ids)
        kept = [t for t, keep in zip(tokens, row.content_mask) if keep]
        assert len(row.content_mask) == len(row.input_ids)
        assert kept == ["virus", "spreads", "fast"]

    def test_the_assistant_ended_boundary_is_placed(self):
        row = render_messages(_tokenizer(adds="bos"), TURNS[2])
        tokens = _tokenizer(adds="bos").convert_ids_to_tokens(row.input_ids)
        assert tokens[: row.prompt_tokens] == ["<s>", "<|user|>", "virus", "spreads", "fast", "<|end|>", "<|assistant|>"]
        assert tokens[row.prompt_tokens:] == ["an", "answer", "<|end|>"]

    def test_a_trailing_special_token_is_scaffolding_and_unresolves_what_millm_cannot_place(self):
        """`<s> $A </s>`: the `</s>` belongs to no message. miLLM places `last_user` only when the
        served ids END with the render, and its assistant-ended head (which ends `</s>` too) is not
        a prefix — so both are unresolved there and here, never guessed."""
        tok = _tokenizer(adds="bos_eos")
        row = render_messages(tok, TURNS[1])
        assert tok.convert_ids_to_tokens(row.input_ids)[-1] == "</s>"
        assert row.token_message[-1] == -1 and row.content_mask[-1] is False
        assert row.role_mask_reliable and row.last_user_start is None
        assert render_messages(tok, TURNS[2]).prompt_tokens is None


# ── 3 and 4. the real tokenizers ─────────────────────────────────────────────────────────────


def _find_tokenizer(sha=None, repos=(), env="MISTUDIO_REAL_TOKENIZERS", label=""):
    from transformers import AutoTokenizer

    dirs = [Path(p) for p in os.environ.get(env, "").split(os.pathsep) if p]
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    for repo in repos:
        dirs += sorted((hub / repo / "snapshots").glob("*"))
    for path in dirs:
        if not (path / "tokenizer_config.json").exists():
            continue
        try:
            tok = AutoTokenizer.from_pretrained(str(path))
        except Exception:  # noqa: BLE001 - a snapshot this transformers cannot read
            continue
        template = getattr(tok, "chat_template", None) or ""
        if sha is None or hashlib.sha256(template.encode()).hexdigest() == sha:
            return tok
    pytest.skip(f"NO {label} TOKENIZER ON THIS MACHINE — these real-tokenizer BOS checks DID NOT RUN. Set {env}.")


@pytest.fixture(scope="module")
def llama():
    real = CASES["real"]["tokenizer"]
    return _find_tokenizer(
        real["chat_template_sha256"],
        ("models--meta-llama--Llama-3.1-8B-Instruct", "models--unsloth--Llama-3.1-8B-Instruct"),
        label=real["hf_id"],
    )


@pytest.fixture(scope="module")
def tinyllama():
    return _find_tokenizer(
        None, ("models--TinyLlama--TinyLlama-1.1B-Chat-v1.0",), env="MISTUDIO_TINYLLAMA_TOKENIZER",
        label="TinyLlama-1.1B-Chat-v1.0",
    )


class TestTheProductionModelIsUnchanged:
    def test_every_real_case_s_ids_are_the_always_false_ids(self, llama):
        """Both production probes are Llama-3.1: the rule changes none of their ids."""
        for case in CASES["real"]["cases"]:
            text, ids, _ = served_render(llama, case["messages"])
            assert ids == ours.template_ids(llama, text) == case["expected"]["ids"]
            assert ids.count(llama.bos_token_id) == 1 and ids[0] == llama.bos_token_id

    def test_its_record_and_the_old_probe_policy(self, llama):
        assert render_form_record(llama) == {
            "generation_prompt": True,
            "add_special_tokens": False,
            "bos_handling": {"template_wrote_bos": True, "tokenizer_added_bos": False, "bos_count": 1},
        }
        assert always_false_rule_is_the_rule(llama) is True


class TestTinyLlamaGainsExactlyOneBos:
    def test_one_bos_from_the_tokenizer(self, tinyllama):
        messages = [{"role": "user", "content": "hello there"}]
        text, ids, prompt_tokens = served_render(tinyllama, messages)
        assert not text.startswith(tinyllama.bos_token)
        assert ids[0] == tinyllama.bos_token_id and ids[1:] == ours.template_ids(tinyllama, text)
        assert ids.count(tinyllama.bos_token_id) == 1 and prompt_tokens == len(ids)
        row = render_messages(tinyllama, messages)
        assert row.input_ids == ids and row.role_mask_reliable and row.token_message[0] == -1

    def test_its_record_and_the_old_probe_policy(self, tinyllama):
        assert render_form_record(tinyllama)["bos_handling"] == {
            "template_wrote_bos": False, "tokenizer_added_bos": True, "bos_count": 1,
        }
        assert always_false_rule_is_the_rule(tinyllama) is False
