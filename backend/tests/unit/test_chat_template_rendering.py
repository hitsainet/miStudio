"""Chat data must carry the model's REAL turn delimiters.

WHY THIS FILE EXISTS. Conversation columns were flattened with a hardcoded
`"<|{role}|>\\n{content}\\n"`. Those markers are in no vocabulary here, so they
tokenize as ordinary characters — an SAE trained on chat data learned features
for the literal text `<`, `|`, `user` rather than for turn structure. Every
model trained on in this project is an INSTRUCT model, so the chat scaffolding
is the part of the deployment distribution that was missing entirely.

The sequencing hazard this guards: `ultrachat_200k` and `lmsys-chat-1m` both
expose a `messages` column, so ingesting them without this fix would train on
`<|user|>` literals.
"""

import inspect

import pytest

from src.services.tokenization_service import TokenizationService
from src.utils.conversation_formats import (
    ConversationFormat,
    extract_messages_from_conversation,
)

render = TokenizationService.render_conversation

OPENAI_CONV = [
    {"role": "user", "content": "what is a sparse autoencoder"},
    {"role": "assistant", "content": "a dictionary learner"},
]
SHAREGPT_CONV = [
    {"from": "human", "value": "what is a sparse autoencoder"},
    {"from": "gpt", "value": "a dictionary learner"},
]


class _Tokenizer:
    """Stands in for a real tokenizer's template contract."""

    def __init__(self, chat_template="{% for m in messages %}...{% endfor %}"):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False, "the caller must ask for text, not ids"
        assert add_generation_prompt is False, (
            "the SAE reads completed turns; it is not continuing one"
        )
        return "".join(
            f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>" for m in messages
        )


class TestMessageExtraction:

    def test_openai_turns_become_role_content_pairs(self):
        assert extract_messages_from_conversation(
            OPENAI_CONV, ConversationFormat.OPENAI
        ) == [
            {"role": "user", "content": "what is a sparse autoencoder"},
            {"role": "assistant", "content": "a dictionary learner"},
        ]

    def test_sharegpt_roles_are_normalised(self):
        """Templates dispatch on canonical names and mis-render anything else."""
        assert [m["role"] for m in extract_messages_from_conversation(
            SHAREGPT_CONV, ConversationFormat.SHAREGPT
        )] == ["user", "assistant"]

    def test_a_bare_list_alternates_speakers(self):
        msgs = extract_messages_from_conversation(
            ["hello", "hi", "bye"], ConversationFormat.SIMPLE_LIST
        )
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]

    def test_an_empty_conversation_yields_nothing(self):
        assert extract_messages_from_conversation([], ConversationFormat.OPENAI) == []


class TestRendering:

    def test_the_models_real_delimiters_are_used(self):
        out = render(OPENAI_CONV, ConversationFormat.OPENAI, _Tokenizer())
        assert "<start_of_turn>" in out
        assert "<|user|>" not in out, (
            "the invented pseudo-marker is back; it is in no model vocabulary "
            "and tokenizes as literal characters"
        )

    def test_plain_never_invents_markers(self):
        out = render(OPENAI_CONV, ConversationFormat.OPENAI, _Tokenizer(), "plain")
        assert "<|user|>" not in out and "<start_of_turn>" not in out
        assert "what is a sparse autoencoder" in out
        assert "a dictionary learner" in out

    def test_auto_falls_back_to_plain_without_a_template(self):
        """A tokenizer with no template must not resurrect the pseudo-markers."""
        out = render(OPENAI_CONV, ConversationFormat.OPENAI, _Tokenizer(None), "auto")
        assert "<|user|>" not in out
        assert "what is a sparse autoencoder" in out

    def test_requiring_a_template_refuses_when_there_is_none(self):
        """Silent degradation is what produced the original defect."""
        with pytest.raises(ValueError, match="chat_template"):
            render(OPENAI_CONV, ConversationFormat.OPENAI, _Tokenizer(None), "chat_template")

    def test_none_is_treated_as_plain(self):
        out = render(OPENAI_CONV, ConversationFormat.OPENAI, _Tokenizer(), "none")
        assert "<start_of_turn>" not in out

    def test_an_empty_conversation_renders_empty(self):
        assert render([], ConversationFormat.OPENAI, _Tokenizer()) == ""

    def test_sharegpt_reaches_the_template_with_canonical_roles(self):
        out = render(SHAREGPT_CONV, ConversationFormat.SHAREGPT, _Tokenizer())
        assert "<start_of_turn>user" in out and "<start_of_turn>assistant" in out
        assert "human" not in out and "gpt" not in out


class TestATemplateThatRejectsTheConversation:
    """Real templates refuse shapes they were not written for.

    `lmsys-chat-1m` contains consecutive user turns, trailing system turns and
    empty content. Losing the whole document over that is worse than losing its
    scaffolding — but the fallback must be LOUD, because a silent one is how the
    original defect survived.
    """

    class _StrictTokenizer:
        chat_template = "strict"

        def apply_chat_template(self, messages, **kwargs):
            if any(
                a["role"] == b["role"] for a, b in zip(messages, messages[1:])
            ):
                raise ValueError("conversation roles must alternate")
            return "".join(f"<t>{m['role']}:{m['content']}</t>" for m in messages)

    def test_a_rejected_conversation_falls_back_to_content(self, caplog):
        conv = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        with caplog.at_level("WARNING"):
            out = render(conv, ConversationFormat.OPENAI, self._StrictTokenizer())

        assert "first" in out and "second" in out
        assert "<t>" not in out
        assert any("rejected" in r.message for r in caplog.records), (
            "the fallback must be reported; a silent one hides a whole corpus "
            "being rendered differently than intended"
        )

    def test_an_acceptable_conversation_still_uses_the_template(self):
        """NEGATIVE CONTROL: the fallback must not swallow the normal path."""
        out = render(OPENAI_CONV, ConversationFormat.OPENAI, self._StrictTokenizer())
        assert "<t>" in out


class TestItIsReachableFromPreprocessing:
    """A renderer nothing calls is not shipped.

    `preprocess_conversation_dataset` is the production path; it must use the
    same function these tests cover, not a second copy of the logic.
    """

    def test_preprocessing_delegates_to_render_conversation(self, monkeypatch):
        from datasets import Dataset

        calls = []
        real = TokenizationService.render_conversation

        def _spy(conv, fmt, tokenizer, chat_format="auto"):
            calls.append(chat_format)
            return real(conv, fmt, tokenizer, chat_format)

        monkeypatch.setattr(TokenizationService, "render_conversation", _spy)

        ds = Dataset.from_dict({"messages": [OPENAI_CONV, OPENAI_CONV]})
        out = TokenizationService.preprocess_conversation_dataset(
            dataset=ds,
            preprocessing_config={
                "source_column": "messages",
                "format": "openai",
                "output_column": "text",
            },
            num_proc=1,
            tokenizer=_Tokenizer(),
            chat_format="chat_template",
        )

        assert calls, (
            "preprocessing does not call render_conversation — the production "
            "path and the tested path are two different implementations"
        )
        assert calls == ["chat_template"] * 2, "chat_format did not reach the renderer"
        assert "<start_of_turn>" in out["text"][0]
        assert "<|user|>" not in out["text"][0]


class TestATemplateThatSwallowsTheConversation:
    """A template can return EMPTY without raising, and the document vanishes.

    Round 2 demonstrated this on the real TinyLlama-1.1B-Chat template: it is an
    if/elif over user|system|assistant with no else, so a role the mapping did
    not canonicalise renders to '' and raises nothing. The document then
    tokenizes as an all-pad row and disappears from the corpus. This is the
    "a silent fallback is worse than a crash" class recorded in CLAUDE.md.
    """

    class _SelectiveTokenizer:
        chat_template = "selective"

        def apply_chat_template(self, messages, **kwargs):
            # Renders only roles it knows — exactly like the Zephyr family.
            return "".join(
                f"<|{m['role']}|>{m['content']}"
                for m in messages
                if m["role"] in ("user", "assistant", "system")
            )

    def test_an_empty_render_falls_back_instead_of_losing_the_document(self, caplog):
        conv = [{"role": "tool", "content": "payload that must survive"}]
        with caplog.at_level("WARNING"):
            out = render(conv, ConversationFormat.OPENAI, self._SelectiveTokenizer())

        assert "payload that must survive" in out, (
            "the conversation rendered to empty and was silently dropped"
        )
        assert any("empty" in r.message for r in caplog.records)

    def test_a_normal_conversation_still_uses_the_template(self):
        """NEGATIVE CONTROL: the guard must not fire on the happy path."""
        out = render(OPENAI_CONV, ConversationFormat.OPENAI, self._SelectiveTokenizer())
        assert "<|user|>" in out and "what is a sparse autoencoder" in out


class TestRoleAndContentNormalisation:

    def test_capitalised_roles_are_canonicalised(self):
        """`mapping.get(k.lower(), original)` kept the ORIGINAL casing, so
        'User' passed through and a template dispatching on 'user' rendered
        nothing for it."""
        msgs = extract_messages_from_conversation(
            [{"role": "User", "content": "a"}, {"role": "ASSISTANT", "content": "b"}],
            ConversationFormat.OPENAI,
        )
        assert [m["role"] for m in msgs] == ["user", "assistant"]

    def test_null_content_is_empty_not_the_string_None(self):
        """`turn.get("content", "")` returns None when the key is present and
        null — routine for OpenAI rows carrying tool_calls. `str(None)` put the
        literal 'None' into training text, and the downstream `if m["content"]`
        filter does not help because 'None' is truthy."""
        msgs = extract_messages_from_conversation(
            [{"role": "assistant", "content": None}], ConversationFormat.OPENAI
        )
        assert msgs[0]["content"] == ""
        assert msgs[0]["content"] != "None"

    def test_null_sharegpt_value_is_empty_too(self):
        msgs = extract_messages_from_conversation(
            [{"from": "gpt", "value": None}], ConversationFormat.SHAREGPT
        )
        assert msgs[0]["content"] == ""


class TestDoubleBos:
    """Llama-2/Mistral/Gemma templates emit BOS in the STRING.

    Tokenizing that with add_special_tokens=True prepends a second one —
    tokenizers do not de-duplicate — putting a maximally-correlated
    high-magnitude artefact at position 0 of every chat document.
    """

    class _BosTokenizer:
        chat_template = "t"
        bos_token_id = 1

        def apply_chat_template(self, messages, **kwargs):
            return "<s>" + "".join(m["content"] for m in messages)

        def __call__(self, text, add_special_tokens=True):
            ids = [1] if text.startswith("<s>") else []
            ids += [99] * 3
            if add_special_tokens:
                ids = [1] + ids
            return {"input_ids": ids}

    class _NoBosTokenizer:
        chat_template = "t"
        bos_token_id = 1

        def apply_chat_template(self, messages, **kwargs):
            return "".join(m["content"] for m in messages)

        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [99] * 3}

    def test_a_template_that_emits_bos_is_detected(self):
        assert TokenizationService.chat_template_emits_bos(self._BosTokenizer())

    def test_a_template_that_does_not_is_not_flagged(self):
        """NEGATIVE CONTROL — flagging everything would disable BOS wrongly."""
        assert not TokenizationService.chat_template_emits_bos(self._NoBosTokenizer())

    def test_a_tokenizer_without_a_template_is_not_flagged(self):
        class _Plain:
            chat_template = None
            bos_token_id = 1

        assert not TokenizationService.chat_template_emits_bos(_Plain())

    def test_detection_is_empirical_not_by_model_name(self):
        """Templates vary within a family; measuring is cheap."""
        src = inspect.getsource(TokenizationService.chat_template_emits_bos)
        assert "apply_chat_template" in src, "detection does not actually render"
