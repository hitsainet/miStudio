"""The role mask, against a REAL tokenizer with a real Jinja chat template.

MUTATION CONTROLS (each verified to fail the suite; recorded in the review):
  M96   the prefix-property check is removed        → the rewriting-template test fails
  M97   an empty render is returned instead of raising → the empty-template test fails
  M98   `_encode` passes `add_special_tokens=True`  → the alignment test fails
  M99   truncation keeps the HEAD instead of the tail → the truncation test fails
  M100  scaffolding is scored under scope `all`      → the scaffolding test fails
  M101  `roles_known=False` no longer forces unreliable → the guessed-roles test fails
  M102  `scored_mask` allows a role scope on an unreliable row → the refusal test fails
  M103  `last_assistant` returns every assistant turn → the last-turn test fails

⚠ WHY A REAL TOKENIZER AND NOT A STUB. The property under test is that each prefix's
TOKEN IDS are a prefix of the next, and that is a claim about tokenization, not about
string concatenation. A stub that returns `"".join(...)` and splits on spaces makes
the property true by construction — the fixtures would agree with the code, which is
this repo's most-recorded reason for a green suite over a real bug. These tests build
a `PreTrainedTokenizerFast` over a tiny word-level vocabulary, so merges, unknown
tokens and whitespace behave as they really do.
"""
import pytest

from src.services.probe_monitor_render import (
    RenderedExample,
    render_all,
    render_messages,
    template_hash,
)

pytest.importorskip("tokenizers")


def _tokenizer(template: str):
    """A real fast tokenizer with a real Jinja chat template. No download."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    words = [
        "<unk>", "<s>", "</s>", "<turn>", "</turn>", "<sys>",
        "user", "assistant", "system",
        "hello", "world", "risk", "safe", "transfer", "funds", "no",
        "a", "b", "c", "d", "e", "f", "g", "h",
    ]
    vocab = {word: index for index, word in enumerate(words)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    # WhitespaceSplit, not Whitespace: the latter also splits on punctuation, which
    # would shatter `<turn>` into three unknown tokens and make every id `<unk>`.
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    # ⚠ A POST-PROCESSOR, NOT JUST bos_token/eos_token. Naming the tokens on the
    # wrapper is not enough — `add_special_tokens=True` consults the BACKEND's
    # post-processor, and with none set the flag is a no-op. That is how the first
    # version of this fixture agreed with the defect by construction and let a
    # mutation flipping the flag SURVIVE the whole suite: the trap this repo has
    # recorded more often than any other. With this processor the flag shifts every
    # position by one, which is exactly the misalignment it exists to prevent.
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", vocab["<s>"]), ("</s>", vocab["</s>"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.chat_template = template
    return tokenizer


#: Well-behaved: each turn is appended and nothing earlier is rewritten.
APPEND_ONLY = (
    "{% for m in messages %}<turn> {{ m['role'] }} {{ m['content'] }} </turn> {% endfor %}"
)

#: ⚠ REWRITES EARLIER TURNS. Once a system message is present it emits a `<sys>`
#: block at the FRONT, so `tok(P_1)` is not a prefix of `tok(P_2)`. Real templates do
#: this (some hoist a system block; some close a tool call retroactively).
REWRITES = (
    "{% if messages | selectattr('role', 'equalto', 'system') | list %}<sys> {% endif %}"
    "{% for m in messages %}<turn> {{ m['role'] }} {{ m['content'] }} </turn> {% endfor %}"
)

#: ⚠ AN if/elif WITH NO else — TinyLlama's shape. An unexpected role renders to
#: nothing at all, WITHOUT raising.
NO_ELSE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'user' %}<turn> user {{ m['content'] }} </turn> "
    "{% elif m['role'] == 'assistant' %}<turn> assistant {{ m['content'] }} </turn> "
    "{% endif %}{% endfor %}"
)

CONVERSATION = [
    {"role": "user", "content": "transfer funds"},
    {"role": "assistant", "content": "no"},
]


class TestTheRoleMaskIsBuiltFromRealTokenization:
    def test_every_content_token_gets_its_message_role(self):
        tokenizer = _tokenizer(APPEND_ONLY)
        rendered = render_messages(tokenizer, CONVERSATION)
        assert rendered.role_mask_reliable
        assert len(rendered.token_roles) == len(rendered.input_ids)
        assert len(rendered.token_message) == len(rendered.input_ids)
        assert set(rendered.token_roles) == {"user", "assistant"}

    def test_the_fixture_tokenizer_HAS_special_tokens_to_add(self):
        """Proves the next test can fail. A tokenizer with no BOS makes
        `add_special_tokens=True` a no-op, and the alignment test then passes
        against the bug it was written to catch."""
        tokenizer = _tokenizer(APPEND_ONLY)
        text = tokenizer.apply_chat_template(
            CONVERSATION, tokenize=False, add_generation_prompt=False
        )
        with_specials = tokenizer(text, add_special_tokens=True)["input_ids"]
        without = tokenizer(text, add_special_tokens=False)["input_ids"]
        assert len(with_specials) > len(without), (
            "this fixture adds no special tokens, so it cannot detect a renderer that "
            "asks for them — the fixture would agree with the defect by construction"
        )

    def test_the_render_does_NOT_add_special_tokens(self):
        """The template already emits whatever the model expects. A second BOS shifts
        every position by one and silently misaligns the role mask against the ids."""
        tokenizer = _tokenizer(APPEND_ONLY)
        rendered = render_messages(tokenizer, CONVERSATION)
        text = tokenizer.apply_chat_template(
            CONVERSATION, tokenize=False, add_generation_prompt=False
        )
        assert rendered.input_ids == tokenizer(text, add_special_tokens=False)["input_ids"]
        assert rendered.input_ids[0] != tokenizer.bos_token_id, (
            "a BOS was prepended by the tokenizer on top of the template's own output"
        )

    def test_the_mask_and_the_ids_stay_the_same_length_under_every_path(self):
        """A mask one token shorter than the ids scores the wrong positions, and the
        shift is silent — the classic off-by-one from an extra BOS."""
        tokenizer = _tokenizer(APPEND_ONLY)
        for messages in (
            CONVERSATION,
            [{"role": "user", "content": "hello"}],
            [{"role": "system", "content": "safe"}, *CONVERSATION],
        ):
            rendered = render_messages(tokenizer, messages)
            assert len(rendered.token_roles) == len(rendered.input_ids)
            assert len(rendered.token_message) == len(rendered.input_ids)

    def test_roles_are_assigned_in_message_ORDER(self):
        tokenizer = _tokenizer(APPEND_ONLY)
        rendered = render_messages(tokenizer, CONVERSATION)
        first_user = rendered.token_roles.index("user")
        first_assistant = rendered.token_roles.index("assistant")
        assert first_user < first_assistant

    def test_a_three_turn_conversation_gets_three_message_indices(self):
        tokenizer = _tokenizer(APPEND_ONLY)
        rendered = render_messages(
            tokenizer,
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ],
        )
        assert set(rendered.token_message) == {0, 1, 2}


class TestThePrefixPropertyIsCHECKEDNotAssumed:
    def test_a_template_that_rewrites_earlier_turns_is_DETECTED(self):
        """The whole method rests on this. A template that hoists a system block
        breaks the prefix relation, and using the mask anyway would score the wrong
        tokens under the right-sounding name."""
        tokenizer = _tokenizer(REWRITES)
        rendered = render_messages(
            tokenizer,
            [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "safe"},
            ],
        )
        assert rendered.role_mask_reliable is False

    def test_and_an_append_only_template_is_NOT_flagged(self):
        """Otherwise the detection is useless: flagging everything is the same as
        flagging nothing."""
        tokenizer = _tokenizer(APPEND_ONLY)
        assert render_messages(tokenizer, CONVERSATION).role_mask_reliable is True

    def test_an_unreliable_row_REFUSES_a_role_scope(self):
        tokenizer = _tokenizer(REWRITES)
        rendered = render_messages(
            tokenizer,
            [{"role": "user", "content": "hello"}, {"role": "system", "content": "safe"}],
        )
        with pytest.raises(ValueError, match="unreliable"):
            rendered.scored_mask("assistant")

    def test_but_scope_all_still_works_on_it(self):
        """Falling back to `all` keeps the row usable; refusing it outright would
        discard data for a template quirk."""
        tokenizer = _tokenizer(REWRITES)
        rendered = render_messages(
            tokenizer,
            [{"role": "user", "content": "hello"}, {"role": "system", "content": "safe"}],
        )
        assert any(rendered.scored_mask("all"))

    def test_a_template_that_RAISES_on_a_partial_conversation_falls_back(self):
        """Several real templates require alternating roles and reject a prefix."""

        class Strict:
            chat_template = "strict"

            def __init__(self):
                self._inner = _tokenizer(APPEND_ONLY)

            def apply_chat_template(self, messages, **kwargs):
                if len(messages) == 1:
                    raise ValueError("conversation must have at least two turns")
                return self._inner.apply_chat_template(messages, **kwargs)

            def __call__(self, *args, **kwargs):
                return self._inner(*args, **kwargs)

        rendered = render_messages(Strict(), CONVERSATION)
        assert rendered.role_mask_reliable is False
        assert rendered.input_ids, "the row is still rendered, just without roles"


class TestGuessedRolesAreNotVerifiableByPrefixRendering:
    def test_roles_known_False_forces_the_mask_unreliable(self):
        """⚠ NOT REDUNDANT WITH THE PREFIX CHECK. A `SIMPLE_LIST` input has roles
        assigned BY POSITION upstream, so the template renders exactly what it was
        handed and the prefix property holds perfectly — while the roles are a guess.
        Prefix rendering verifies the MASK against the roles it was given; it cannot
        know they were invented."""
        tokenizer = _tokenizer(APPEND_ONLY)
        honest = render_messages(tokenizer, CONVERSATION, roles_known=True)
        guessed = render_messages(tokenizer, CONVERSATION, roles_known=False)
        assert honest.role_mask_reliable is True
        assert guessed.role_mask_reliable is False
        assert honest.input_ids == guessed.input_ids, (
            "the flag must change only the mask's trustworthiness, never the ids"
        )

    def test_render_all_refuses_a_misaligned_flag_list(self):
        """Flags landing on the wrong rows would mark the wrong rows unreliable."""
        tokenizer = _tokenizer(APPEND_ONLY)
        with pytest.raises(ValueError, match="parallel"):
            render_all(tokenizer, [CONVERSATION, CONVERSATION], roles_known=[True])


class TestAnEmptyRenderIsRefused:
    def test_a_template_with_no_else_branch_raises_rather_than_returning_nothing(self):
        """TinyLlama's template is an `if/elif` with no `else`, so an unexpected role
        renders to an empty string WITHOUT raising — which became an all-pad row here
        once already. A zero-token example trains on nothing and reports as fine."""
        tokenizer = _tokenizer(NO_ELSE)
        with pytest.raises(ValueError, match="empty string"):
            render_messages(tokenizer, [{"role": "tool", "content": "hello"}])

    def test_the_same_template_works_for_a_role_it_handles(self):
        tokenizer = _tokenizer(NO_ELSE)
        assert render_messages(tokenizer, [{"role": "user", "content": "hello"}]).input_ids

    def test_an_empty_conversation_raises(self):
        with pytest.raises(ValueError, match="empty conversation"):
            render_messages(_tokenizer(APPEND_ONLY), [])


class TestScopes:
    def _rendered(self):
        return render_messages(
            _tokenizer(APPEND_ONLY),
            [
                {"role": "user", "content": "transfer funds"},
                {"role": "assistant", "content": "no"},
                {"role": "user", "content": "risk"},
                {"role": "assistant", "content": "safe"},
            ],
        )

    def test_assistant_scope_scores_only_assistant_tokens(self):
        rendered = self._rendered()
        mask = rendered.scored_mask("assistant")
        roles = {r for r, keep in zip(rendered.token_roles, mask) if keep}
        assert roles == {"assistant"}

    def test_user_scope_scores_only_user_tokens(self):
        rendered = self._rendered()
        mask = rendered.scored_mask("user")
        assert {r for r, keep in zip(rendered.token_roles, mask) if keep} == {"user"}

    def test_last_assistant_is_NOT_every_assistant_turn(self):
        """An earlier assistant turn is context the model was CONDITIONED on, not text
        it produced this time — a different thing to detect."""
        rendered = self._rendered()
        every = sum(rendered.scored_mask("assistant"))
        last = sum(rendered.scored_mask("last_assistant"))
        assert 0 < last < every

    def test_last_assistant_picks_the_HIGHEST_message_index(self):
        rendered = self._rendered()
        mask = rendered.scored_mask("last_assistant")
        indices = {m for m, keep in zip(rendered.token_message, mask) if keep}
        assert indices == {3}

    def test_last_assistant_with_no_assistant_turn_scores_nothing(self):
        rendered = render_messages(
            _tokenizer(APPEND_ONLY), [{"role": "user", "content": "hello"}]
        )
        assert not any(rendered.scored_mask("last_assistant"))

    def test_scaffolding_is_never_scored_even_under_scope_all(self):
        """A BOS token and a generation prompt carry no content, and including them
        makes a `mean` rule's denominator depend on the template rather than the text."""
        rendered = RenderedExample(
            input_ids=[1, 10, 11, 2],
            token_roles=["", "user", "user", ""],
            token_message=[-1, 0, 0, -1],
            text="",
        )
        assert rendered.scored_mask("all") == [False, True, True, False]

    def test_an_unknown_scope_raises_and_names_the_options(self):
        rendered = self._rendered()
        with pytest.raises(ValueError, match="unknown scope"):
            rendered.scored_mask("assistant_only")


class TestTruncationKeepsTheEnd:
    def test_it_drops_the_FRONT(self):
        """The assistant's reply is at the END of a conversation, so head-truncation
        removes exactly the tokens an `assistant`-scoped probe exists to read."""
        tokenizer = _tokenizer(APPEND_ONLY)
        long_conversation = [
            {"role": "user", "content": " ".join(["a"] * 40)},
            {"role": "assistant", "content": "safe"},
        ]
        full = render_messages(tokenizer, long_conversation)
        clipped = render_messages(tokenizer, long_conversation, max_length=12)
        assert clipped.truncated is True
        assert len(clipped.input_ids) == 12
        assert clipped.input_ids == full.input_ids[-12:]
        assert "assistant" in clipped.token_roles, (
            "the assistant turn was truncated away, which is the one thing that must "
            "survive"
        )

    def test_the_mask_is_truncated_WITH_the_ids(self):
        tokenizer = _tokenizer(APPEND_ONLY)
        clipped = render_messages(
            tokenizer,
            [{"role": "user", "content": " ".join(["a"] * 40)},
             {"role": "assistant", "content": "safe"}],
            max_length=8,
        )
        assert len(clipped.token_roles) == len(clipped.input_ids) == 8
        assert len(clipped.token_message) == 8

    def test_a_short_row_is_not_marked_truncated(self):
        rendered = render_messages(
            _tokenizer(APPEND_ONLY), CONVERSATION, max_length=4096
        )
        assert rendered.truncated is False


class TestTheTemplateHashIsRecorded:
    def test_two_templates_hash_differently(self):
        """The template is part of the function a probe learned: the same weights over
        a different template read different tokens at different positions. 033
        publishes this so a runtime can refuse a mismatch."""
        assert template_hash(_tokenizer(APPEND_ONLY)) != template_hash(_tokenizer(REWRITES))

    def test_the_same_template_hashes_stably(self):
        assert template_hash(_tokenizer(APPEND_ONLY)) == template_hash(_tokenizer(APPEND_ONLY))

    def test_a_tokenizer_with_no_template_hashes_the_empty_string(self):
        """Not an error: a base model has no chat template, and a plain-text probe
        dataset does not need one. It must still produce a recorded value."""
        import hashlib

        tokenizer = _tokenizer(APPEND_ONLY)
        tokenizer.chat_template = None
        assert template_hash(tokenizer) == hashlib.sha256(b"").hexdigest()


class TestRenderAllAccountsForEveryRow:
    def test_the_result_is_PARALLEL_to_the_input_with_None_for_failures(self):
        """Returning only the successes is how labels and examples drift apart."""
        tokenizer = _tokenizer(NO_ELSE)
        conversations = [
            [{"role": "user", "content": "hello"}],
            [{"role": "tool", "content": "hello"}],     # renders empty → fails
            [{"role": "user", "content": "world"}],
        ]
        results, summary = render_all(tokenizer, conversations)
        assert len(results) == len(conversations)
        assert results[1] is None
        assert summary.rendered == 2 and summary.failed == 1

    def test_it_counts_scored_tokens_for_the_memmap_presize(self):
        """The token-capture memmap is pre-sized from this, so counting during render
        is the only pass that sees every row before the GPU work starts."""
        tokenizer = _tokenizer(APPEND_ONLY)
        _, summary = render_all(tokenizer, [CONVERSATION, CONVERSATION], scope="assistant")
        assert summary.scored_tokens > 0
        _, all_scope = render_all(tokenizer, [CONVERSATION, CONVERSATION], scope="all")
        assert all_scope.scored_tokens > summary.scored_tokens

    def test_an_unreliable_row_is_counted_AND_falls_back_to_all(self):
        tokenizer = _tokenizer(REWRITES)
        _, summary = render_all(
            tokenizer,
            [[{"role": "user", "content": "hello"}, {"role": "system", "content": "safe"}]],
            scope="assistant",
        )
        assert summary.role_mask_unreliable == 1
        assert summary.scored_tokens > 0, (
            "the fallback to scope 'all' did not happen, so the row contributes "
            "nothing and the memmap would be under-sized"
        )

    def test_failures_are_sampled_not_swallowed(self):
        tokenizer = _tokenizer(NO_ELSE)
        _, summary = render_all(tokenizer, [[{"role": "tool", "content": "x"}]] * 3)
        assert summary.failed == 3
        assert summary.failures, "no failure reason was kept, so the cause is invisible"

    def test_the_summary_dict_carries_the_log_keys(self):
        _, summary = render_all(_tokenizer(APPEND_ONLY), [CONVERSATION])
        assert set(summary.as_dict()) == {
            "rendered", "failed", "role_mask_unreliable", "truncated", "scored_tokens"
        }
