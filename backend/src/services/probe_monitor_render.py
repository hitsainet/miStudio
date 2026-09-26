"""Rendering messages with the model's own chat template, and the per-token role mask.

WHY PREFIX RENDERING RATHER THAN OFFSETS (FTID I1). A probe's `scope` says which
tokens it may score — `assistant` only, `all`, and so on — so every token needs a
role. The obvious route is `return_offsets_mapping` against the rendered string, but
offsets are per-tokenizer, absent on some slow tokenizers, and say nothing about
which MESSAGE a character came from once a template interleaves delimiters. Prefix
rendering asks the template itself: render `m[:1]`, `m[:2]`, … and the tokens each
render adds belong to that message.

⚠ AND IT IS VERIFIED, NOT ASSUMED. The method holds only if each prefix's tokens are
a true prefix of the next — which is false for any template that rewrites earlier
turns (some add a system block once a system message appears; some close a tool call
retroactively). When the property fails the row is marked `role_mask_unreliable` and
is allowed ONLY scope `all`, because a wrong role mask silently scores the wrong
tokens: a probe told "assistant" that actually reads the user's text is a different
detector reporting under the wrong name.

⚠ A CHAT TEMPLATE CAN RETURN EMPTY WITHOUT RAISING. TinyLlama's is an `if/elif`
chain with no `else`, so an unexpected role renders to nothing at all — which became
an all-pad row here once before. `render_messages` refuses an empty render rather
than producing a zero-token example that trains on nothing.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: `scope` values and the roles each admits. `last_assistant` is separate from
#: `assistant` because a multi-turn conversation's earlier assistant turns are
#: context the model was CONDITIONED on, not text it produced this time.
SCOPE_ROLES: Dict[str, Optional[frozenset]] = {
    "all": None,                               # every real token
    "assistant": frozenset({"assistant"}),
    "user": frozenset({"user"}),
    "last_assistant": frozenset({"assistant"}),  # plus the last-turn restriction
}


@dataclass
class RenderedExample:
    """One rendered row: ids, the role of every token, and what may be scored."""

    input_ids: List[int]
    #: Parallel to `input_ids`. "" for a token no message claims — template
    #: scaffolding such as a leading BOS or a trailing generation prompt.
    token_roles: List[str]
    #: Which message index each token came from; -1 for scaffolding. Needed by
    #: `last_assistant`, which cannot be expressed by role alone.
    token_message: List[int]
    text: str
    role_mask_reliable: bool = True
    truncated: bool = False

    def scored_mask(self, scope: str) -> List[bool]:
        """Which positions a probe with this `scope` may score.

        Scaffolding is never scored under any scope, including `all`: a BOS token and
        a generation prompt carry no content, and including them makes a `mean` rule's
        denominator depend on the template rather than on the text.
        """
        if scope not in SCOPE_ROLES:
            raise ValueError(f"unknown scope {scope!r}; use one of {sorted(SCOPE_ROLES)}")
        if not self.role_mask_reliable and scope != "all":
            raise ValueError(
                f"this row's role mask is unreliable, so scope {scope!r} cannot be "
                f"honoured; only 'all' is available (see role_mask_unreliable)"
            )
        allowed = SCOPE_ROLES[scope]
        if scope == "last_assistant":
            last = self._last_message_index_with_role("assistant")
            if last is None:
                return [False] * len(self.input_ids)
            return [m == last for m in self.token_message]
        if allowed is None:
            return [m >= 0 for m in self.token_message]
        return [role in allowed for role in self.token_roles]

    def _last_message_index_with_role(self, role: str) -> Optional[int]:
        best = None
        for index, token_role in zip(self.token_message, self.token_roles):
            if token_role == role and index >= 0:
                best = index if best is None else max(best, index)
        return best


def template_hash(tokenizer: Any) -> str:
    """sha256 of the chat template, recorded on the run (FR-15).

    The template is part of the function a probe learned: the same weights over a
    different template read different tokens at different positions. 033 publishes
    this hash so a serving runtime can refuse a mismatch instead of silently scoring
    a differently-delimited conversation.
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _apply_template(tokenizer: Any, messages: Sequence[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=False
    )


def _encode(tokenizer: Any, text: str) -> List[int]:
    # `add_special_tokens=False`: the template already emits whatever the model
    # expects, and adding a second BOS shifts every position by one — which would
    # silently misalign the role mask against the ids.
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def render_messages(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    *,
    max_length: Optional[int] = None,
    roles_known: bool = True,
) -> RenderedExample:
    """Render one conversation and label every token with its message.

    `roles_known=False` forces `role_mask_reliable` off REGARDLESS of the prefix
    property, and that is not redundant. A `SIMPLE_LIST` input (a bare list of
    strings) has its roles assigned BY POSITION upstream, so the prefix check passes
    perfectly — the template renders exactly what it was given — while the roles
    themselves are a guess. Prefix rendering can only verify that the MASK matches
    the roles it was handed; it cannot know those roles were invented.

    Raises `ValueError` when the template renders to nothing — see the module
    docstring on TinyLlama's `if/elif` with no `else`.
    """
    if not messages:
        raise ValueError("cannot render an empty conversation")

    reliable_override = bool(roles_known)
    full_text = _apply_template(tokenizer, messages)
    if not full_text or not full_text.strip():
        raise ValueError(
            "the chat template rendered this conversation to an empty string. Some "
            "templates are an if/elif chain with no else, so an unexpected role "
            "produces nothing and the row would train on no tokens at all"
        )
    full_ids = _encode(tokenizer, full_text)
    if not full_ids:
        raise ValueError("the rendered conversation tokenized to zero tokens")

    # Prefix lengths: render m[:1], m[:2], … and record where each ends.
    boundaries: List[int] = []
    reliable = True
    previous_ids: List[int] = []
    for i in range(1, len(messages) + 1):
        try:
            prefix_text = _apply_template(tokenizer, messages[:i])
        except Exception as exc:  # noqa: BLE001 - a template may reject a partial turn
            logger.warning(
                "probe_monitor.role_mask_unreliable: the template refused prefix %d of "
                "%d (%s); falling back to scope 'all' for this row",
                i, len(messages), exc,
            )
            reliable = False
            break
        prefix_ids = _encode(tokenizer, prefix_text)
        # THE PREFIX PROPERTY, CHECKED — not assumed.
        if prefix_ids[: len(previous_ids)] != previous_ids:
            logger.warning(
                "probe_monitor.role_mask_unreliable: prefix %d is not a token-prefix of "
                "prefix %d, so this template rewrites earlier turns; falling back to "
                "scope 'all' for this row",
                i - 1, i,
            )
            reliable = False
            break
        boundaries.append(len(prefix_ids))
        previous_ids = prefix_ids

    # The final prefix must also be a prefix of the FULL render. It usually is
    # identical, but a template that appends a trailing newline only for the complete
    # conversation would break the alignment silently.
    if reliable and previous_ids and full_ids[: len(previous_ids)] != previous_ids:
        logger.warning(
            "probe_monitor.role_mask_unreliable: the complete render is not an "
            "extension of its own last prefix; falling back to scope 'all'"
        )
        reliable = False

    if not reliable_override:
        reliable = False

    token_roles = [""] * len(full_ids)
    token_message = [-1] * len(full_ids)
    if reliable:
        start = 0
        for message_index, end in enumerate(boundaries):
            role = str(messages[message_index].get("role", ""))
            for position in range(start, min(end, len(full_ids))):
                token_roles[position] = role
                token_message[position] = message_index
            start = end
    else:
        # Every real token belongs to "the conversation" and to no message. Scope
        # `all` still works; any role-scoped request raises rather than guessing.
        for position in range(len(full_ids)):
            token_message[position] = 0
            token_roles[position] = ""

    truncated = False
    if max_length is not None and len(full_ids) > max_length:
        # ⚠ TRUNCATE THE FRONT, KEEPING THE END. The assistant's reply is at the end
        # of a conversation, so head-truncation would remove exactly the tokens an
        # `assistant`-scoped probe exists to read. Recorded on the row so the run can
        # report how many examples lost context rather than discovering it later.
        keep = max_length
        full_ids = full_ids[-keep:]
        token_roles = token_roles[-keep:]
        token_message = token_message[-keep:]
        truncated = True

    return RenderedExample(
        input_ids=full_ids,
        token_roles=token_roles,
        token_message=token_message,
        text=full_text,
        role_mask_reliable=reliable,
        truncated=truncated,
    )


@dataclass
class RenderSummary:
    """What a render pass has to report (FR-15 and the §12 log keys)."""

    rendered: int = 0
    failed: int = 0
    role_mask_unreliable: int = 0
    truncated: int = 0
    scored_tokens: int = 0
    failures: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rendered": self.rendered,
            "failed": self.failed,
            "role_mask_unreliable": self.role_mask_unreliable,
            "truncated": self.truncated,
            "scored_tokens": self.scored_tokens,
        }


def render_all(
    tokenizer: Any,
    conversations: Sequence[Sequence[Dict[str, str]]],
    *,
    scope: str = "all",
    max_length: Optional[int] = None,
    roles_known: Optional[Sequence[bool]] = None,
) -> Tuple[List[Optional[RenderedExample]], RenderSummary]:
    """Render many conversations, counting every failure rather than dropping it.

    The returned list is PARALLEL to the input, with None where rendering failed, so
    a caller keeps the correspondence between rows and labels. Returning only the
    successes is how labels and examples drift apart by a handful of rows.

    `scored_tokens` is summed here because the token-capture memmap is pre-sized from
    it (FTID §13) — counting during render is the only pass that sees every row before
    the GPU work starts.
    """
    if roles_known is not None and len(roles_known) != len(conversations):
        raise ValueError(
            f"{len(roles_known)} roles_known flags against {len(conversations)} "
            f"conversations — they must be parallel or the flags land on wrong rows"
        )
    results: List[Optional[RenderedExample]] = []
    summary = RenderSummary()
    for position, messages in enumerate(conversations):
        try:
            rendered = render_messages(
                tokenizer,
                messages,
                max_length=max_length,
                roles_known=True if roles_known is None else bool(roles_known[position]),
            )
        except Exception as exc:  # noqa: BLE001 - a bad row must not stop the pass
            summary.failed += 1
            if len(summary.failures) < 20:
                summary.failures.append(str(exc)[:200])
            results.append(None)
            continue
        summary.rendered += 1
        if not rendered.role_mask_reliable:
            summary.role_mask_unreliable += 1
        if rendered.truncated:
            summary.truncated += 1
        effective_scope = "all" if not rendered.role_mask_reliable else scope
        summary.scored_tokens += sum(rendered.scored_mask(effective_scope))
        results.append(rendered)
    return results, summary
