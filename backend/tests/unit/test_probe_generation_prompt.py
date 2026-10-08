"""Probes render conversations the way miLLM SERVES them (operator decision, 2026-10-08).

miLLM scores every live chat with the model's generation prompt after the last user turn; miStudio
rendered its corpus without it, so every probe was calibrated on a form it never sees live
(production: `pm_dcc6b7e0a850` AUROC 0.9558 served against 0.9420 without the prompt; miStudio's
own figure 0.9417). The rule now lives in ONE function, `probe_monitor_render.served_render`, and
these tests pin four things:

  1. the rule itself, on a REAL tokenizer with a Llama-3-style template that emits its own BOS
     and a post-processor that would add a second one;
  2. that it is miLLM's rule, by running miLLM's own `ProbeInputPreparer` in miLLM's own venv
     over the same tokenizer (skipped without a miLLM checkout, required under
     `MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1`);
  3. that every capture path reaches it — by the AST, asserting the CALL, not the name;
  4. that the render form is recorded and that a probe without it is refused for evaluate, the GPU
     re-cut and export, before any model loads.

MUTATION CONTROLS: recorded in `0xcc/reviews/probe_generation_prompt_2026-10-08.md`.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.services import probe_monitor_render as render_module
from src.services.probe_monitor_render import (
    is_served_render_form,
    render_all,
    render_form_record,
    render_messages,
)

pytest.importorskip("tokenizers")

#: What a run on the Llama-shaped fixture below records: its template writes the BOS, so the
#: start-of-text rule asks for no special tokens and the ids carry exactly that one BOS.
SERVED = {
    "generation_prompt": True,
    "add_special_tokens": False,
    "bos_handling": {"template_wrote_bos": True, "tokenizer_added_bos": False, "bos_count": 1},
}
#: The form recorded between the served render and the start-of-text rule (both 2026-10-08):
#: `add_special_tokens` false for every model, and nothing said about the BOS.
ALWAYS_FALSE = {"generation_prompt": True, "add_special_tokens": False}

BOS = "<|begin_of_text|>"
HEADER = ["<|start_header_id|>", "assistant", "<|end_header_id|>"]

#: Llama-3-shaped: the TEMPLATE emits the BOS, every turn is header + content + <|eot_id|>, and the
#: generation prompt is the assistant header with no content.
LLAMA_STYLE = (
    "{{ bos_token }}{% for m in messages %}<|start_header_id|> {{ m['role'] }} <|end_header_id|> "
    "{{ m['content'] }} <|eot_id|> {% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|> assistant <|end_header_id|> {% endif %}"
)

#: A template whose generation prompt is NOT the header an assistant turn is rendered with (a
#: "thinking" opener), so the head render WITH it is not a prefix of the full render.
DIVERGENT_GENERATION_PROMPT = LLAMA_STYLE.replace(
    "{% if add_generation_prompt %}<|start_header_id|> assistant <|end_header_id|> {% endif %}",
    "{% if add_generation_prompt %}<|start_header_id|> assistant <|end_header_id|> think {% endif %}",
)

WORDS = [
    BOS, "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<unk>",
    "user", "assistant", "system", "think",
    "is", "this", "risky", "yes", "no", "be", "brief", "transfer", "the", "funds", "now",
    # The role-header probes both repos render (`_HEADER_PROBES`, `_FIRST_PROBES`) must tokenize
    # to DIFFERENT ids, or the header cannot be isolated and `last_user` resolves on nothing.
    "a", "b", "x", "y",
]


def _tokenizer(template: str = LLAMA_STYLE, *, prepend_bos: bool = True):
    """A real fast tokenizer. With `prepend_bos` its post-processor adds a BOS on
    `add_special_tokens=True`, exactly as Llama 3's does — so a renderer that asks for special
    tokens produces TWO, which is the duplicate the served form forbids."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {word: index for index, word in enumerate(WORDS)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if prepend_bos:
        backend.post_processor = processors.TemplateProcessing(
            single=f"{BOS} $A", special_tokens=[(BOS, vocab[BOS])]
        )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token=BOS)
    tokenizer.chat_template = template
    return tokenizer


def _tokens(tokenizer, ids):
    return tokenizer.convert_ids_to_tokens(list(ids))


USER_ONLY = [{"role": "user", "content": "is this risky"}]
MULTI_USER_ENDED = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "transfer the funds"},
    {"role": "assistant", "content": "no"},
    {"role": "user", "content": "is this risky"},
]
ASSISTANT_ENDED = [
    {"role": "user", "content": "is this risky"},
    {"role": "assistant", "content": "yes"},
]


# ── 1. the rule, on a real tokenizer ──────────────────────────────────────────────────────────


class TestTheFixtureCanSeeTheDefects:
    def test_the_tokenizer_WOULD_add_a_second_bos(self):
        """Without this the add-special-tokens mutation is invisible: a fixture with no
        post-processor makes the flag a no-op and agrees with the defect by construction."""
        tok = _tokenizer()
        text = tok.apply_chat_template(USER_ONLY, tokenize=False, add_generation_prompt=True)
        assert len(tok(text, add_special_tokens=True)["input_ids"]) == (
            len(tok(text, add_special_tokens=False)["input_ids"]) + 1
        )

    def test_the_template_really_renders_a_generation_prompt(self):
        tok = _tokenizer()
        with_prompt = tok.apply_chat_template(USER_ONLY, tokenize=False, add_generation_prompt=True)
        without = tok.apply_chat_template(USER_ONLY, tokenize=False, add_generation_prompt=False)
        assert with_prompt != without and with_prompt.startswith(without)


class TestAUserEndedRowIsRenderedAsServed:
    def test_one_user_turn_ENDS_WITH_THE_GENERATION_PROMPT(self):
        tok = _tokenizer()
        row = render_messages(tok, USER_ONLY)
        assert _tokens(tok, row.input_ids)[-3:] == HEADER

    def test_and_carries_EXACTLY_ONE_bos(self):
        tok = _tokenizer()
        row = render_messages(tok, USER_ONLY)
        tokens = _tokens(tok, row.input_ids)
        assert tokens[0] == BOS
        assert tokens.count(BOS) == 1, tokens

    def test_the_ids_are_the_template_with_the_prompt_and_no_added_specials(self):
        """Exactly the sequence miLLM serves once its duplicate-BOS fix lands."""
        tok = _tokenizer()
        row = render_messages(tok, MULTI_USER_ENDED)
        text = tok.apply_chat_template(MULTI_USER_ENDED, tokenize=False, add_generation_prompt=True)
        assert row.input_ids == tok(text, add_special_tokens=False)["input_ids"]
        assert row.text == text

    def test_plain_prose_is_one_user_turn_WITH_the_prompt(self):
        """The training corpus is prose; `parse_input` makes it one user turn, and that turn is
        what miLLM's `text` input renders too (T-49)."""
        from src.services.probe_monitor_inputs import parse_input

        tok = _tokenizer()
        parsed = parse_input("is this risky")
        rows, summary = render_all(tok, [parsed.messages])
        assert summary.rendered == 1
        assert _tokens(tok, rows[0].input_ids)[-3:] == HEADER

    def test_every_position_is_prompt(self):
        tok = _tokenizer()
        row = render_messages(tok, USER_ONLY)
        assert row.served_form is True
        assert row.prompt_tokens == len(row.input_ids)


class TestWhichPositionsEachWindowScores:
    def test_all_scores_EVERY_position_including_the_generation_prompt(self):
        """miLLM's `all` is every served position; a mean over fewer is another distribution."""
        tok = _tokenizer()
        row = render_messages(tok, USER_ONLY)
        assert row.scored_mask("all") == [True] * len(row.input_ids)

    def test_input_is_every_position_on_a_user_ended_row(self):
        tok = _tokenizer()
        row = render_messages(tok, MULTI_USER_ENDED)
        assert row.scored_mask("input") == [True] * len(row.input_ids)

    def test_last_assistant_is_EMPTY_on_a_user_ended_row(self):
        """No response exists. The earlier assistant turn is history, not this reply — miLLM's
        `response` window is empty here, and so is this."""
        tok = _tokenizer()
        row = render_messages(tok, MULTI_USER_ENDED)
        assert not any(row.scored_mask("last_assistant"))

    def test_last_user_ENDS_BEFORE_the_generation_prompt(self):
        """miLLM's span ends at the render through the newest user turn WITHOUT the prompt."""
        tok = _tokenizer()
        row = render_messages(tok, MULTI_USER_ENDED)
        span = [t for t, keep in zip(_tokens(tok, row.input_ids), row.scored_mask("last_user")) if keep]
        assert span == ["<|start_header_id|>", "user", "<|end_header_id|>", "is", "this", "risky", "<|eot_id|>"]


class TestAnAssistantEndedRowHasMiLLMsBoundary:
    def test_it_is_rendered_WITHOUT_the_generation_prompt(self):
        tok = _tokenizer()
        row = render_messages(tok, ASSISTANT_ENDED)
        text = tok.apply_chat_template(ASSISTANT_ENDED, tokenize=False, add_generation_prompt=False)
        assert row.input_ids == tok(text, add_special_tokens=False)["input_ids"]
        assert _tokens(tok, row.input_ids)[-2:] == ["yes", "<|eot_id|>"]

    def test_the_boundary_is_the_head_rendered_WITH_the_generation_prompt(self):
        tok = _tokenizer()
        row = render_messages(tok, ASSISTANT_ENDED)
        head_text = tok.apply_chat_template(ASSISTANT_ENDED[:-1], tokenize=False, add_generation_prompt=True)
        head = tok(head_text, add_special_tokens=False)["input_ids"]
        assert row.prompt_tokens == len(head)
        assert row.input_ids[: len(head)] == head

    def test_the_response_starts_AFTER_the_assistant_header(self):
        """The header belongs to the prompt, as miLLM splits it: `response` is the reply alone."""
        tok = _tokenizer()
        row = render_messages(tok, ASSISTANT_ENDED)
        response = [t for t, k in zip(_tokens(tok, row.input_ids), row.scored_mask("last_assistant")) if k]
        assert response == ["yes", "<|eot_id|>"]
        prompt = row.scored_mask("input")
        assert prompt == [not k for k in row.scored_mask("last_assistant")]

    def test_a_boundary_that_is_not_a_prefix_is_NOT_GUESSED(self):
        tok = _tokenizer(DIVERGENT_GENERATION_PROMPT)
        row = render_messages(tok, ASSISTANT_ENDED)
        assert row.prompt_tokens is None
        assert not any(row.scored_mask("input")) and not any(row.scored_mask("last_assistant"))
        assert all(row.scored_mask("all")), "the row still scores under `all`"

    def test_the_positional_windows_SURVIVE_an_unreliable_role_mask(self):
        """miLLM's windows need no roles; a row with guessed roles keeps its boundary here too
        rather than collapsing `input` to `all`."""
        from src.services.probe_monitor_capture import _row_mask

        tok = _tokenizer()
        row = render_messages(tok, ASSISTANT_ENDED, roles_known=False)
        assert row.role_mask_reliable is False
        assert _row_mask(row, "input") == [i < row.prompt_tokens for i in range(len(row.input_ids))]


class TestTruncationMovesEveryPositionalField:
    def test_the_boundary_moves_with_the_ids(self):
        tok = _tokenizer()
        full = render_messages(tok, ASSISTANT_ENDED)
        cut = render_messages(tok, ASSISTANT_ENDED, max_length=len(full.input_ids) - 2)
        assert cut.truncated and cut.input_ids == full.input_ids[2:]
        assert cut.prompt_tokens == full.prompt_tokens - 2
        assert cut.scored_mask("last_assistant") == full.scored_mask("last_assistant")[2:]

    def test_keep_tail_cuts_the_content_mask_too(self):
        """The builder once cut three fields by hand and left the content mask full length, so
        `zip` paired every flag with the wrong position."""
        tok = _tokenizer()
        row = render_messages(tok, MULTI_USER_ENDED)
        assert row.content_mask is not None
        row.keep_tail(5)
        assert len(row.content_mask) == len(row.token_roles) == len(row.token_message) == 5
        assert len(row.scored_mask("content")) == 5


# ── 2. it is miLLM's rule — run miLLM's own code, in miLLM's own venv ─────────────────────────

MILLM = Path(os.environ.get("MILLM_REPO", "/home/x-sean/app/miLLM"))
# MILLM_PYTHON for a miLLM WORKTREE, which has no venv of its own: the script puts MILLM on
# sys.path first, so the interpreter's site-packages supply the libraries and MILLM the code.
MILLM_PY = Path(os.environ.get("MILLM_PYTHON", str(MILLM / "venv" / "bin" / "python")))
REQUIRED = os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1"

_MILLM_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from types import SimpleNamespace
    sys.path.insert(0, sys.argv[1])
    from transformers import PreTrainedTokenizerFast
    from millm.services.probe_scoring import ProbeInputPreparer

    tok = PreTrainedTokenizerFast.from_pretrained(sys.argv[2])
    def render(messages, generation_prompt):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
    preparer = ProbeInputPreparer(tok, render)
    out = []
    for conversation in json.loads(sys.argv[3]):
        item = SimpleNamespace(
            kinds=lambda: ["messages"],
            messages=[SimpleNamespace(role=m["role"], content=m["content"]) for m in conversation],
        )
        p = preparer.prepare(0, item)
        out.append({"ids": p.ids, "prompt_tokens": p.prompt_tokens,
                    "span": list(p.last_user_span) if p.last_user_span else None, "error": p.error})
    print(json.dumps(out))
    """
)


def _millm_prepare(tokenizer, conversations, tmp_path):
    if not (MILLM.exists() and MILLM_PY.exists()):
        if REQUIRED:
            pytest.fail(f"MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1 but miLLM is not at {MILLM}")
        pytest.skip(f"miLLM checkout (with venv) not present at {MILLM}")
    directory = tmp_path / "tok"
    tokenizer.save_pretrained(str(directory))
    script = tmp_path / "prepare.py"
    script.write_text(_MILLM_SCRIPT)
    done = subprocess.run(
        [str(MILLM_PY), "-I", str(script), str(MILLM), str(directory), json.dumps(conversations)],
        capture_output=True, text=True, timeout=300, cwd=str(tmp_path),
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


CONVERSATIONS = [USER_ONLY, MULTI_USER_ENDED, ASSISTANT_ENDED,
                 [{"role": "system", "content": "be brief"}, {"role": "user", "content": "is this risky"}]]


def test_the_ids_and_the_boundary_are_EXACTLY_millms(tmp_path):
    """⚠ A TOKENIZER WITH NO BOS POST-PROCESSOR, deliberately: miLLM's preparer encodes with
    `tokenizer(text)` and its duplicate-BOS fix (landing in parallel) changes that to
    `add_special_tokens=False`. On a tokenizer that adds nothing the two agree, so this compares
    the RENDER RULE and the boundary without depending on which side of that fix miLLM is on."""
    tok = _tokenizer(prepend_bos=False)
    theirs = _millm_prepare(tok, CONVERSATIONS, tmp_path)
    for conversation, other in zip(CONVERSATIONS, theirs):
        assert other["error"] is None, other
        ours = render_messages(tok, conversation)
        assert ours.input_ids == other["ids"], conversation
        assert ours.prompt_tokens == other["prompt_tokens"], conversation


def test_the_last_user_span_is_millms_where_millm_resolves_one(tmp_path):
    """User-ended rows always resolve in miLLM. ⚠ An assistant-ended `messages` input does NOT
    (miLLM `last_user_token_span` compares the served ids against a render WITH the generation
    prompt, which an assistant-ended render never has) — a miLLM defect reported, not copied."""
    tok = _tokenizer(prepend_bos=False)
    theirs = _millm_prepare(tok, CONVERSATIONS, tmp_path)
    compared = 0
    for conversation, other in zip(CONVERSATIONS, theirs):
        if other["span"] is None:
            assert conversation[-1]["role"] == "assistant", (conversation, other)
            continue
        ours = render_messages(tok, conversation).scored_mask("last_user")
        start, end = other["span"]
        assert ours == [start <= i < end for i in range(len(ours))], conversation
        compared += 1
    assert compared >= 3


# ── 3. every capture path reaches the one function — asserted by CALL, on the AST ────────────


def _calls(fn) -> set:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            names.add(getattr(func, "id", None) or getattr(func, "attr", None))
    return names


def test_the_one_function_owns_the_full_render():
    """`served_render` is the ONLY caller of `_apply_template` that renders a whole conversation
    with anything but the prefix form, and `render_messages` is its only caller."""
    tree = ast.parse(inspect.getsource(render_module))
    owners = {}
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            owners.setdefault(name, set()).add(fn.name)
            if name == "_apply_template":
                prompt = [k for k in node.keywords if k.arg == "generation_prompt"]
                if prompt and not (isinstance(prompt[0].value, ast.Constant) and prompt[0].value.value is False):
                    assert fn.name == "served_render", (
                        f"{fn.name} renders with a generation prompt outside the one function"
                    )
    assert owners["apply_chat_template"] == {"_apply_template"}
    assert owners["served_render"] == {"render_messages"}


def test_render_messages_takes_its_ids_from_the_one_function():
    tree = ast.parse(inspect.getsource(render_module.render_messages))
    assigned = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "served_render"
    ]
    assert len(assigned) == 1
    targets = ast.unparse(assigned[0].targets[0])
    assert "full_ids" in targets and "prompt_tokens" in targets


@pytest.mark.parametrize("module_name,function,callee", [
    # training capture, evaluation, calibration and the GPU re-cut
    ("src.services.probe_monitor_run", "execute_probe_run", "_prepare_examples"),
    ("src.services.probe_monitor_run", "_evaluate_probes_on_sets", "_prepare_examples"),
    ("src.services.probe_monitor_run", "_prepare_calibration_rows", "_prepare_examples"),
    ("src.services.probe_monitor_run", "recut_probe_windows_on_gpu", "_prepare_calibration_rows"),
    ("src.services.probe_monitor_run", "_prepare_examples", "render_all"),
    # the offline "try it on your own text" score
    ("src.services.probe_monitor_run", "score_one", "render_messages"),
    # the definition's test vectors and their round-trip check
    ("src.services.probe_definition_builder", "build", "_prepare_examples"),
    ("src.services.probe_definition_builder", "_messages_round_trip", "render_messages"),
    # and the chain into the one function
    ("src.services.probe_monitor_render", "render_all", "render_messages"),
    ("src.services.probe_monitor_render", "render_messages", "served_render"),
])
def test_every_capture_path_CALLS_its_way_to_the_one_function(module_name, function, callee):
    import importlib

    fn = getattr(importlib.import_module(module_name), function)
    assert callee in _calls(fn), f"{module_name}.{function} no longer calls {callee}"


def test_no_probe_module_renders_or_tokenizes_on_its_own():
    """A second `apply_chat_template` or a bare `tokenizer(...)` in a probe module is a second
    render rule, free to disagree with the served one."""
    services = Path(render_module.__file__).parent
    for path in sorted(services.glob("probe_*.py")):
        if path.name == "probe_monitor_render.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                attr = getattr(node.func, "attr", None)
                name = getattr(node.func, "id", None)
                assert attr != "apply_chat_template", f"{path.name}:{node.lineno} renders on its own"
                assert name != "tokenizer", f"{path.name}:{node.lineno} tokenizes on its own"


# ── 4. the render form is recorded, and a probe without it is refused ──────────────────────────


class TestTheRecordedForm:
    def test_the_record_is_the_served_form(self):
        assert render_form_record(_tokenizer()) == SERVED

    def test_the_record_is_measured_PER_MODEL(self):
        """The F1 form (no BOS in the template, a BOS-adding tokenizer) and a model with no BOS
        at all record what the rule did to THEM — the record is the tokenizer's, not a constant."""
        no_bos_template = LLAMA_STYLE.replace("{{ bos_token }}", "")
        assert render_form_record(_tokenizer(no_bos_template)) == {
            "generation_prompt": True,
            "add_special_tokens": True,
            "bos_handling": {"template_wrote_bos": False, "tokenizer_added_bos": True, "bos_count": 1},
        }
        assert render_form_record(_tokenizer(no_bos_template, prepend_bos=False)) == {
            "generation_prompt": True,
            "add_special_tokens": True,
            "bos_handling": {"template_wrote_bos": False, "tokenizer_added_bos": False, "bos_count": 0},
        }

    def test_every_record_the_renderer_writes_is_a_valid_contract_block(self):
        from src.schemas.probe_definition import RenderForm

        no_bos_template = LLAMA_STYLE.replace("{{ bos_token }}", "")
        for tok in (_tokenizer(), _tokenizer(no_bos_template), _tokenizer(no_bos_template, prepend_bos=False)):
            record = render_form_record(tok)
            assert RenderForm.model_validate(record).model_dump() == record

    @pytest.mark.parametrize("recorded,served", [
        (SERVED, True),
        # On its face; `probe_render` checks the tokenizer before any reuse.
        (ALWAYS_FALSE, True),
        # `add_special_tokens: true` with nothing saying what it added is not a recorded rule.
        ({"generation_prompt": True, "add_special_tokens": True}, False),
        # A `bos_handling` the rule could not have produced is not read.
        ({**SERVED, "add_special_tokens": True}, False),
        ({**SERVED, "bos_handling": {"template_wrote_bos": True, "tokenizer_added_bos": True, "bos_count": 1}}, False),
        ({**SERVED, "bos_handling": {"template_wrote_bos": True, "tokenizer_added_bos": False, "bos_count": 2}}, False),
        ({**SERVED, "bos_handling": {"template_wrote_bos": True, "tokenizer_added_bos": False, "bos_count": 0}}, False),
        ({**SERVED, "bos_handling": {"template_wrote_bos": "yes", "tokenizer_added_bos": False, "bos_count": 1}}, False),
        ({**SERVED, "x": 1}, False),
        (None, False),                                          # not recorded ≠ served
        ({}, False),
        ({"generation_prompt": False, "add_special_tokens": False}, False),
        ({"generation_prompt": True}, False),
        # The flag ABSENT, or not a real boolean, is not "true" (mutation M6b survived without
        # these: a gate reading `is not False` admitted both).
        ({"add_special_tokens": False}, False),
        ({"generation_prompt": "true", "add_special_tokens": False}, False),
        ({"generation_prompt": 1, "add_special_tokens": False}, False),
    ])
    def test_only_the_exact_served_form_reads_as_served(self, recorded, served):
        assert is_served_render_form(recorded) is served

    def test_execute_probe_run_records_the_form_in_its_environment(self):
        from src.services import probe_monitor_run

        tree = ast.parse(inspect.getsource(probe_monitor_run.execute_probe_run))
        recorded = [
            value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "update"
            and ast.unparse(node.func.value) == "environment"
            for arg in node.args if isinstance(arg, ast.Dict)
            for key, value in zip(arg.keys, arg.values)
            if isinstance(key, ast.Constant) and key.value == "render_form"
        ]
        assert len(recorded) == 1 and ast.unparse(recorded[0]) == "render_form_record(tokenizer)"


class TestProbeRender:
    def _render(self, run_env=None, probe_form="unset", **kwargs):
        from src.services.probe_monitor_run import probe_render

        def unreachable(_run):
            raise AssertionError("the tokenizer was loaded for a form that does not need it")

        kwargs.setdefault("tokenizer_loader", unreachable)
        run = SimpleNamespace(id="pmr_x", environment=run_env or {})
        probe = SimpleNamespace(id="pm_x") if probe_form == "unset" else SimpleNamespace(id="pm_x", render_form=probe_form)
        return probe_render(run, probe, **kwargs)

    def test_a_probe_trained_today_is_allowed(self):
        out = self._render({"render_form": SERVED}, SERVED)
        assert out == {
            "recorded": SERVED, "effective": SERVED, "served": None, "bos_note": None, "refusal": None,
        }

    def test_NOT_RECORDED_reads_as_without_the_generation_prompt_and_is_refused(self):
        out = self._render({"seed": 1}, None)
        assert out["recorded"] is None
        assert "WITHOUT the generation prompt" in out["refusal"] and "Retrain" in out["refusal"]

    def test_the_probe_row_is_read_before_the_run(self):
        assert self._render({}, SERVED)["refusal"] is None
        assert self._render({"render_form": SERVED}, "unset")["refusal"] is None

    def test_a_recorded_non_served_form_is_refused(self):
        other = {"generation_prompt": False, "add_special_tokens": False}
        assert "not the form miLLM serves" in self._render({"render_form": other}, other)["refusal"]

    def test_a_probe_and_run_that_disagree_are_refused_not_reconciled(self):
        refusal = self._render({"render_form": SERVED}, ALWAYS_FALSE)
        assert "but its run records" in refusal["refusal"]
        assert refusal["effective"] is None

    def test_an_unknown_key_is_refused(self):
        odd = {**SERVED, "x": 1}
        assert "not the form miLLM serves" in self._render({"render_form": odd}, odd)["refusal"]


class TestTheAlwaysFalseRuleOnOldProbes:
    """F1 (operator, 2026-10-08): a probe recorded under the always-False rule is the served form
    exactly when its model's template writes the BOS, or its tokenizer adds nothing — and is
    refused, like an old-render probe, where the tokenizer adds a BOS the template does not."""

    NO_BOS_TEMPLATE = LLAMA_STYLE.replace("{{ bos_token }}", "")

    def _render(self, tokenizer=None, loader=None):
        from src.services.probe_monitor_run import probe_render

        run = SimpleNamespace(id="pmr_x", environment={"render_form": ALWAYS_FALSE})
        probe = SimpleNamespace(id="pm_x", render_form=ALWAYS_FALSE)
        return probe_render(run, probe, tokenizer=tokenizer, tokenizer_loader=loader)

    def test_a_template_that_writes_the_bos_is_the_served_form(self):
        """Both production probes (Llama-3.1). Their ids do not change, so nothing is refused, and
        the definition built now states the rule's record — which is true of their ids too."""
        out = self._render(_tokenizer())
        assert out["refusal"] is None
        assert out["effective"] == SERVED and out["served"] == SERVED
        assert "template writes the BOS" in out["bos_note"]

    def test_a_tokenizer_that_adds_nothing_is_the_served_form(self):
        out = self._render(_tokenizer(self.NO_BOS_TEMPLATE, prepend_bos=False))
        assert out["refusal"] is None
        assert out["effective"]["bos_handling"] == {
            "template_wrote_bos": False, "tokenizer_added_bos": False, "bos_count": 0,
        }
        assert "tokenizer adds nothing" in out["bos_note"]

    def test_the_F1_form_is_REFUSED(self):
        """Trained on zero BOS, served with one: the same mixing an old-render probe is refused
        for, refused the same way."""
        out = self._render(_tokenizer(self.NO_BOS_TEMPLATE))
        assert out["effective"] is None
        assert "trained on no BOS where miLLM serves one" in out["refusal"]
        assert "Retrain" in out["refusal"]

    def test_the_run_s_tokenizer_is_loaded_when_none_is_in_hand(self):
        calls = []

        def loader(run):
            calls.append(run.id)
            return _tokenizer(self.NO_BOS_TEMPLATE)

        out = self._render(loader=loader)
        assert calls == ["pmr_x"]
        assert out["refusal"] is not None

    def test_a_tokenizer_that_cannot_be_loaded_REFUSES_rather_than_passes(self):
        def loader(_run):
            raise FileNotFoundError("weights gone")

        out = self._render(loader=loader)
        assert "could not be loaded" in out["refusal"] and "FileNotFoundError" in out["refusal"]
        assert out["effective"] is None


def _old_probe_db(env=None):
    probe = SimpleNamespace(id="pm_old", run_id="pmr_old", render_form=None, length_bands=None,
                            calibration_lengths_path=None, calibration_dataset_id="pmd_cal",
                            calibration_scores_path=None, window_decisions={})
    run = SimpleNamespace(id="pmr_old", model_id="m", calibration_dataset_id="pmd_cal",
                          environment=env if env is not None else {"model_dtype": "bfloat16"})

    class _Q:
        def __init__(self, row): self.row = row
        def filter(self, *a, **k): return self
        def first(self): return self.row
        def one_or_none(self): return self.row
        def all(self): return []

    db = SimpleNamespace(query=lambda model: _Q(probe if model.__name__ == "ProbeMonitor" else run))
    return db, probe, run


def _never_load(*_a, **_k):
    raise AssertionError("the model was loaded for a probe the render gate should have refused")


class TestAnOldRenderProbeIsRefusedBeforeAnyModelLoads:
    def test_evaluate(self, monkeypatch):
        from src.services import probe_monitor_run as pmr

        db, _probe, _run = _old_probe_db()
        monkeypatch.setattr(pmr, "probe_precision", lambda db, run: {"refusal": None})
        monkeypatch.setattr(pmr, "_load_model_for_run", _never_load)
        with pytest.raises(pmr.ProbeRenderRefused, match="WITHOUT the generation prompt"):
            pmr.evaluate_probe(db, "pm_old", ["pmd_1"])

    def test_the_gpu_recut(self, monkeypatch):
        from src.services import probe_monitor_run as pmr
        from src.services.probe_recalibration import RecalibrationRefused

        db, _probe, _run = _old_probe_db()
        monkeypatch.setattr(pmr, "probe_precision", lambda db, run: {"refusal": None})
        monkeypatch.setattr(pmr, "context_from_row", lambda row: SimpleNamespace(scope="all", target_fpr=0.01))
        monkeypatch.setattr(pmr, "_load_model_for_run", _never_load)
        with pytest.raises(RecalibrationRefused) as caught:
            pmr.recut_probe_windows_on_gpu(db, "pm_old", target_fpr=0.01)
        assert caught.value.code == "render_mismatch"

    def test_export(self, monkeypatch):
        from src.services import probe_definition_builder as builder
        from src.services import probe_monitor_run as pmr
        from src.services.probe_definition_builder import ProbeExportRefused

        monkeypatch.setattr(pmr, "probe_precision", lambda db, run: {"refusal": None})
        monkeypatch.setattr(pmr, "recompute_rung", lambda db, probe_id: 2)
        monkeypatch.setattr(builder, "check_export_gate", lambda *a, **k: None)
        monkeypatch.setattr(pmr, "load_probe", lambda db, pid: (None, None))
        monkeypatch.setattr(pmr, "context_from_row", lambda row: SimpleNamespace(scope="all"))
        db, probe, _run = _old_probe_db()
        probe.sae_id = None
        db.refresh = lambda row: None
        try:
            builder.build(db, "pm_old", model_loader=_never_load,
                          acknowledge_below_rung2={"reason": "unit test of the render gate"})
        except ProbeExportRefused as refused:
            assert "WITHOUT the generation prompt" in str(refused), str(refused)
            assert refused.status == 409
        else:
            pytest.fail("an old-render probe was exported")

    @pytest.mark.parametrize("fn,gate", [
        ("evaluate_probe", "require_served_render"),
        ("recut_probe_windows_on_gpu", "probe_render"),
    ])
    def test_the_gate_precedes_the_load_in_source_order(self, fn, gate):
        from src.services import probe_monitor_run as pmr

        tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(pmr, fn))))
        def first(name):
            return min(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                       and (getattr(n.func, "id", None) == name or getattr(n.func, "attr", None) == name))
        assert first(gate) < first("_load_model_for_run")


def test_the_three_gpu_endpoints_409_on_an_old_render_and_queue_nothing(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from fastapi import HTTPException

    import src.core.database as database
    from src.api.v1.endpoints import probe_monitors as ep
    from src.services import probe_monitor_run as pmr

    _db, probe, run = _old_probe_db()

    class _Session:
        def query(self, model):
            class _Q:
                def filter(self, *a, **k): return self
                def first(self): return probe if model.__name__ == "ProbeMonitor" else run
            return _Q()
        def close(self): pass

    monkeypatch.setattr(database, "SyncSessionLocal", lambda: _Session())
    monkeypatch.setattr(pmr, "probe_precision", lambda db, r: {"refusal": None})
    queued = MagicMock()
    monkeypatch.setattr(ep, "gpu_delay", lambda task, gpu: queued)
    db = MagicMock()
    db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: probe))

    calls = [
        lambda: ep.evaluate_probe_endpoint("pm_old", ["pmd_1"], db=db),
        lambda: ep.recalibrate_probe_endpoint(
            "pm_old", SimpleNamespace(target_fpr=0.01, reason=None), recut_windows=True
        ),
        lambda: ep.build_definition(
            "pm_old", SimpleNamespace(acknowledge_below_rung2=None, vector_count=16, seed=1), db=db
        ),
    ]
    for call in calls:
        with pytest.raises(HTTPException) as caught:
            asyncio.run(call())
        assert caught.value.status_code == 409
        assert caught.value.detail["code"] == "render_mismatch"
    queued.assert_not_called()


# ── 4b. the always-False rule on a model where it differs: refused on every path (F1) ─────────

_NO_BOS_TEMPLATE = LLAMA_STYLE.replace("{{ bos_token }}", "")


def _always_false_probe_db(monkeypatch, tokenizer):
    """An old-rule probe (served form, no `bos_handling`) whose model's tokenizer is `tokenizer`,
    reached through the gate's REAL tokenizer loader seam — nothing is handed to the gate."""
    from src.services import probe_monitor_run as pmr

    db, probe, run = _old_probe_db({"model_dtype": "bfloat16", "render_form": ALWAYS_FALSE})
    probe.render_form = dict(ALWAYS_FALSE)
    loads = []

    def load_tokenizer(row):
        loads.append(row.id)
        return tokenizer

    monkeypatch.setattr(pmr, "_load_tokenizer_for_run", load_tokenizer)
    return db, probe, run, loads


class TestAnAlwaysFalseProbeOnTheF1FormIsRefusedBeforeAnyModelLoads:
    def test_evaluate(self, monkeypatch):
        from src.services import probe_monitor_run as pmr

        db, _probe, _run, loads = _always_false_probe_db(monkeypatch, _tokenizer(_NO_BOS_TEMPLATE))
        monkeypatch.setattr(pmr, "probe_precision", lambda db, run: {"refusal": None})
        monkeypatch.setattr(pmr, "_load_model_for_run", _never_load)
        with pytest.raises(pmr.ProbeRenderRefused, match="trained on no BOS"):
            pmr.evaluate_probe(db, "pm_old", ["pmd_1"])
        assert loads == ["pmr_old"]

    def test_evaluate_PASSES_the_gate_where_the_template_writes_the_bos(self, monkeypatch):
        """The positive control: on a Llama-shaped model the same record is the served form, so
        the gate lets it through to the model load (which this test then stops)."""
        from src.services import probe_monitor_run as pmr

        db, _probe, _run, loads = _always_false_probe_db(monkeypatch, _tokenizer())
        monkeypatch.setattr(pmr, "probe_precision", lambda db, run: {"refusal": None})
        monkeypatch.setattr(pmr, "context_from_row", lambda row: SimpleNamespace(scope="all"))
        monkeypatch.setattr(pmr, "_load_model_for_run", _never_load)
        with pytest.raises(AssertionError, match="the model was loaded"):
            pmr.evaluate_probe(db, "pm_old", ["pmd_1"])
        assert loads == ["pmr_old"]

    def test_the_gpu_recut(self, monkeypatch):
        from src.services import probe_monitor_run as pmr
        from src.services.probe_recalibration import RecalibrationRefused

        db, _probe, _run, _loads = _always_false_probe_db(monkeypatch, _tokenizer(_NO_BOS_TEMPLATE))
        monkeypatch.setattr(pmr, "probe_precision", lambda db, run: {"refusal": None})
        monkeypatch.setattr(
            pmr, "context_from_row", lambda row: SimpleNamespace(scope="all", target_fpr=0.01)
        )
        monkeypatch.setattr(pmr, "_load_model_for_run", _never_load)
        with pytest.raises(RecalibrationRefused) as caught:
            pmr.recut_probe_windows_on_gpu(db, "pm_old", target_fpr=0.01)
        assert caught.value.code == "render_mismatch" and "trained on no BOS" in str(caught.value)

    def test_export(self, monkeypatch):
        from src.services import probe_definition_builder as builder
        from src.services import probe_monitor_run as pmr
        from src.services.probe_definition_builder import ProbeExportRefused

        monkeypatch.setattr(pmr, "probe_precision", lambda db, run: {"refusal": None})
        monkeypatch.setattr(pmr, "recompute_rung", lambda db, probe_id: 2)
        monkeypatch.setattr(builder, "check_export_gate", lambda *a, **k: None)
        monkeypatch.setattr(pmr, "load_probe", lambda db, pid: (None, None))
        monkeypatch.setattr(pmr, "context_from_row", lambda row: SimpleNamespace(scope="all"))
        db, probe, _run, _loads = _always_false_probe_db(monkeypatch, _tokenizer(_NO_BOS_TEMPLATE))
        probe.sae_id = None
        db.refresh = lambda row: None
        with pytest.raises(ProbeExportRefused, match="trained on no BOS") as caught:
            builder.build(db, "pm_old", model_loader=_never_load,
                          acknowledge_below_rung2={"reason": "unit test of the BOS gate"})
        assert caught.value.status == 409

    def test_the_api_409s_before_queueing(self, monkeypatch):
        import asyncio

        from fastapi import HTTPException

        import src.core.database as database
        from src.api.v1.endpoints import probe_monitors as ep

        _db, probe, run, _loads = _always_false_probe_db(monkeypatch, _tokenizer(_NO_BOS_TEMPLATE))

        class _Session:
            def query(self, model):
                class _Q:
                    def filter(self, *a, **k): return self
                    def first(self): return probe if model.__name__ == "ProbeMonitor" else run
                return _Q()
            def close(self): pass

        monkeypatch.setattr(database, "SyncSessionLocal", lambda: _Session())
        with pytest.raises(HTTPException) as caught:
            asyncio.run(ep.refuse_render_mismatch("pm_old"))
        assert caught.value.status_code == 409 and caught.value.detail["code"] == "render_mismatch"
        assert "trained on no BOS" in caught.value.detail["message"]


def test_the_definition_publishes_the_EFFECTIVE_form():
    """An always-False probe proven equivalent is published WITH its `bos_handling`, so the
    document says exactly how its ids were built; reading `recorded` would publish the old,
    unrecorded form."""
    from src.services import probe_definition_builder

    tree = ast.parse(textwrap.dedent(inspect.getsource(probe_definition_builder.build)))
    renders = [
        ast.unparse(kw.value) for n in ast.walk(tree) if isinstance(n, ast.Call)
        for kw in n.keywords if kw.arg == "render"
    ]
    assert renders == ["RenderForm(**render['effective'])"]


def test_the_offline_score_reports_both_renders():
    from src.services import probe_monitor_run

    tree = ast.parse(textwrap.dedent(inspect.getsource(probe_monitor_run.score_one)))
    keys = {k.value for n in ast.walk(tree) if isinstance(n, ast.Dict) for k in n.keys
            if isinstance(k, ast.Constant)}
    assert {"render", "trained_with", "scored_with"} <= keys
    assert "probe_render" in _calls(probe_monitor_run.score_one)


class TestTheContract:
    def test_the_block_is_optional_and_absent_means_not_recorded(self):
        from src.schemas.probe_definition import ProbeDefinitionV1

        field = ProbeDefinitionV1.model_fields["render"]
        assert not field.is_required() and field.default is None

    def test_it_refuses_an_unknown_key(self):
        from pydantic import ValidationError

        from src.schemas.probe_definition import RenderForm

        with pytest.raises(ValidationError):
            RenderForm(generation_prompt=True, add_special_tokens=False, bos="twice")

    def test_the_summary_decides_served_by_the_gates_rule(self):
        from src.schemas.probe_monitor import ProbeMonitorSummary

        fields = {
            "id": "pm", "run_id": "pmr", "layer": 1, "rule": "mean", "rule_params": {},
            "variant": "dense", "sae_id": None, "sae_feature_indices": None, "val_metrics": {},
            "selected": False, "threshold": None, "target_fpr": None, "realised_fpr": None,
            "threshold_source": None, "streamable": True, "rung": 0, "rung_reasons": [],
            "created_at": "2026-10-08T00:00:00Z",
        }
        assert ProbeMonitorSummary(**fields, render_form=SERVED).model_dump()["render_served"] is True
        assert ProbeMonitorSummary(**fields).model_dump()["render_served"] is False
        dumped = ProbeMonitorSummary(**fields, render_form=SERVED).model_dump()
        assert dumped["render_bos_recorded"] is True
        dumped = ProbeMonitorSummary(**fields, render_form=ALWAYS_FALSE).model_dump()
        assert dumped["render_served"] is True and dumped["render_bos_recorded"] is False
        assert ProbeMonitorSummary(**fields).model_dump()["render_bos_recorded"] is False

    def test_an_always_false_document_still_validates_and_still_means_it(self):
        """ADDITIVE: a document written before `bos_handling` existed parses unchanged, and the
        absent field stays absent — never filled in as "the tokenizer added one"."""
        from src.schemas.probe_definition import RenderForm

        parsed = RenderForm.model_validate(ALWAYS_FALSE)
        assert parsed.bos_handling is None and parsed.add_special_tokens is False
        assert parsed.model_dump(exclude_none=True) == ALWAYS_FALSE

    @pytest.mark.parametrize("bad", [
        {"template_wrote_bos": True, "tokenizer_added_bos": True, "bos_count": 1},
        {"template_wrote_bos": True, "tokenizer_added_bos": False, "bos_count": 2},
        {"template_wrote_bos": False, "tokenizer_added_bos": True, "bos_count": 0},
    ])
    def test_a_bos_record_the_rule_could_not_produce_is_refused(self, bad):
        from pydantic import ValidationError

        from src.schemas.probe_definition import RenderForm

        with pytest.raises(ValidationError):
            RenderForm(generation_prompt=True, add_special_tokens=not bad["template_wrote_bos"], bos_handling=bad)

    def test_add_special_tokens_must_follow_the_template(self):
        from pydantic import ValidationError

        from src.schemas.probe_definition import RenderForm

        with pytest.raises(ValidationError):
            RenderForm(generation_prompt=True, add_special_tokens=True, bos_handling=SERVED["bos_handling"])
