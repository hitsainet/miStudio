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
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .prompt_encoding import rendered_chat_ids, template_ids

logger = logging.getLogger(__name__)

#: `scope` values and the roles each admits. `last_assistant` is separate from
#: `assistant` because a multi-turn conversation's earlier assistant turns are
#: context the model was CONDITIONED on, not text it produced this time.
SCOPE_ROLES: Dict[str, Optional[frozenset]] = {
    "all": None,                               # every real token
    "assistant": frozenset({"assistant"}),
    "user": frozenset({"user"}),
    "last_assistant": frozenset({"assistant"}),  # plus the last-turn restriction
    #: THE MESSAGES' OWN TEXT, NOTHING THE TEMPLATE ADDED (2026-10-05). `all` and `user` both score
    #: the template's scaffolding inside a message's span — role headers, end-of-turn tokens and,
    #: on Llama-3.1, the ~25-token system block it injects into message 0 even when no system
    #: message was sent. For a 20-token joke that block is MORE than half of what a `mean` averages,
    #: and its share moves with the text's length. `content` is located by offsets, verified, and
    #: falls back to `all` (counted) where it cannot be — see `_content_mask`.
    "content": None,
    #: EVERYTHING THE MODEL WAS GIVEN — the exact complement of `last_assistant`.
    #:
    #: ⚠ THIS EXISTS TO SETTLE A DISAGREEMENT BETWEEN THE TWO REPOS, and it is not the same as
    #: `user`. The contract's `prompt` is defined positionally, `[0, n_prompt)`, which is what
    #: miLLM scores. `user` excludes the system preamble AND every earlier assistant turn — both
    #: of which sit inside `[0, n_prompt)` on a real request. So calibrating a `prompt` threshold
    #: under `user` would set a bar for a narrower window than the runtime reads, which is the
    #: precise failure this whole feature exists to remove: a quantile of one distribution
    #: applied to another.
    #:
    #: `input` is the split a served request actually has — history plus the new user message is
    #: the input, the final assistant turn is the output. Handled specially below, like
    #: `last_assistant`, because it cannot be expressed by role alone.
    "input": None,
    #: THE NEWEST USER MESSAGE ONLY — its whole span as the renderer attributes it (header,
    #: content and end-of-turn), the same span miLLM computes by rendering the same prefixes.
    #:
    #: ⚠ WHY IT EXISTS (operator, 2026-10-04). `input` is everything before the reply, so on a
    #: client that resends the conversation (Open WebUI, LibreChat, most agent frameworks) a
    #: high-stakes earlier turn kept firing on every later turn, and a long system prompt or a
    #: retrieved document diluted or triggered it. `last_user` asks the question operators mean:
    #: is THIS message high-stakes?
    "last_user": frozenset({"user"}),
}



#: WHICH INTERNAL SCOPE EACH CONTRACT WINDOW MEANS. The one definition of it, so the calibration
#: stage and the exporter cannot drift into two answers.
#:
#: ⚠ `prompt` IS `input`, NOT `user`. The contract defines `prompt` positionally — `[0, n_prompt)`
#: — which is what miLLM scores, and that span holds the system preamble and every earlier
#: assistant turn. `user` excludes both. Calibrating a `prompt` threshold under `user` would cut
#: a bar for a narrower window than the runtime reads, which is a quantile of one distribution
#: applied to another: exactly the failure per-window thresholds exist to remove.
CONTRACT_WINDOW_SCOPES: Dict[str, str] = {
    "all": "all",
    "prompt": "input",
    "response": "last_assistant",
    "last_user": "last_user",
}


#: ⚠ THE RENDER FORM IS PART OF A PROBE'S IDENTITY (operator decision, 2026-10-08).
#:
#: miLLM scores every live chat with the model's GENERATION PROMPT at the end — on Llama 3,
#: `<|start_header_id|>assistant<|end_header_id|>\n\n` after the last user turn — and its
#: `/api/probes/score` renders a user-ended conversation the same way. miStudio rendered its
#: corpus WITHOUT it until this date, so every probe was trained, calibrated and evaluated on a
#: form it never sees live. Measured on production: `pm_dcc6b7e0a850` scored AUROC 0.9558 in
#: miLLM's served form against 0.9420 with the prompt removed, while miStudio recorded 0.9417.
#: A threshold cut under one render and served under another is a quantile of one distribution
#: applied to another.
#:
#: So `render_messages` renders exactly what miLLM serves (`ProbeInputPreparer.prepare`):
#:   * a conversation ending on any turn but an assistant one — which includes plain prose
#:     rendered as one user turn — WITH the generation prompt; every position is prompt;
#:   * an assistant-ended conversation WITHOUT it, its prompt/response boundary at the render of
#:     the preceding turns WITH it — accepted only when that is a token prefix of the full render;
#:   * tokenized by miLLM's start-of-text rule (`prompt_encoding.rendered_chat_ids`): no
#:     special tokens when the render already begins with the BOS text, the tokenizer's own
#:     otherwise — exactly one BOS when the model uses one, never a duplicate (2026-10-08, F1).
#:
#: What a run, a probe row and a definition RECORD is `render_form_record(tokenizer)`. An absent
#: record means "not recorded", which for this estate means rendered WITHOUT the generation
#: prompt — never assumed to be the served form (`probe_monitor_run.probe_render`).
SERVED_GENERATION_PROMPT = True

#: The one-turn conversation the per-model BOS record is measured on. Any user turn would do: the
#: rule turns on whether the render BEGINS with the BOS text, which is the template's preamble.
_BOS_PROBE = ({"role": "user", "content": "x"},)


def render_form_record(tokenizer: Any) -> Dict[str, Any]:
    """The render form `render_messages` produces for THIS model, as a run, a probe row and a
    definition record it.

    `add_special_tokens` is what the rule passed to the tokenizer and `bos_handling` what came of
    it (`prompt_encoding.bos_handling`), measured on the served render of a one-turn conversation
    by the same functions `served_render` uses — so the record cannot describe a different rule.
    """
    from .prompt_encoding import bos_handling, rendered_chat_adds_special_tokens

    text = _bos_probe_text(tokenizer)
    return {
        "generation_prompt": SERVED_GENERATION_PROMPT,
        "add_special_tokens": rendered_chat_adds_special_tokens(tokenizer, text),
        "bos_handling": bos_handling(tokenizer, text),
    }


def _bos_probe_text(tokenizer: Any) -> str:
    """The SERVED render of the one-turn probe conversation — taken from `render_messages`, the
    one path every probe input takes, so the record measures that render and no other."""
    return render_messages(tokenizer, [dict(m) for m in _BOS_PROBE]).text


def always_false_rule_is_the_rule(tokenizer: Any) -> bool:
    """Whether `add_special_tokens=False` — the rule before 2026-10-08's start-of-text rule — gives
    THIS model exactly the rule's ids: true when its template writes the BOS or its tokenizer adds
    nothing. Judged on the render `render_form_record` measures (read by
    `probe_monitor_run.probe_render`)."""
    from .prompt_encoding import tokenizer_adds_nothing

    return tokenizer_adds_nothing(tokenizer, _bos_probe_text(tokenizer))


#: `render_form_status` answers. Only `SERVED` and `SERVED_BOS_UNRECORDED` are the served form, and
#: the second only once its model's tokenizer has shown the old rule changes nothing
#: (`probe_monitor_run.probe_render`).
RENDER_SERVED = "served"
RENDER_SERVED_BOS_UNRECORDED = "served_bos_unrecorded"
RENDER_NOT_SERVED = "not_served"
RENDER_NOT_RECORDED = "not_recorded"


def render_form_status(recorded: Any) -> str:
    """What a RECORDED render form says, without loading anything.

    * `not_recorded` — `None`: trained before render recording, WITHOUT the generation prompt;
    * `served` — generation prompt on, and a `bos_handling` record that is internally consistent
      under the start-of-text rule (validated by the contract's own `RenderForm`);
    * `served_bos_unrecorded` — generation prompt on, `add_special_tokens: false`, no
      `bos_handling`: the ALWAYS-False rule miStudio used from the served render until the BOS
      rule (both 2026-10-08). Its ids equal the rule's exactly when the model's template writes
      the BOS or its tokenizer adds nothing — a property of the TOKENIZER, so this cannot decide
      it, and `probe_render` checks it before any reuse;
    * `not_served` — anything else (no generation prompt, `add_special_tokens: true` with nothing
      to say what it added, an inconsistent `bos_handling`, a non-bool, an unknown key).
    """
    if recorded is None:
        return RENDER_NOT_RECORDED
    if not isinstance(recorded, dict) or recorded.get("generation_prompt") is not True:
        return RENDER_NOT_SERVED
    if recorded.get("bos_handling") is None:
        if set(recorded) <= {"generation_prompt", "add_special_tokens", "bos_handling"} and (
            recorded.get("add_special_tokens") is False
        ):
            return RENDER_SERVED_BOS_UNRECORDED
        return RENDER_NOT_SERVED
    from pydantic import ValidationError

    from ..schemas.probe_definition import RenderForm

    try:
        RenderForm.model_validate(recorded, strict=True)
    except ValidationError:
        return RENDER_NOT_SERVED
    return RENDER_SERVED


def is_served_render_form(recorded: Any) -> bool:
    """Whether a RECORDED render form is the served form on its face: `served`, or
    `served_bos_unrecorded` (whose BOS equivalence `probe_render` verifies against the tokenizer
    before any reuse). `None` (not recorded) is not."""
    return render_form_status(recorded) in (RENDER_SERVED, RENDER_SERVED_BOS_UNRECORDED)

#: The scopes a served-form row answers POSITIONALLY, from `prompt_tokens`, as miLLM does — never
#: from the role mask, so an unreliable role mask does not move them to `all`.
POSITIONAL_SCOPES = frozenset({"all", "input", "last_assistant"})


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
    #: WHERE THE NEWEST USER MESSAGE'S OWN ROLE HEADER STARTS, or `None` when it cannot be found
    #: (no user turn, no recognisable header, or truncation cut into the message) — `last_user`
    #: then scores nothing on this row.
    #:
    #: ⚠ WHY THE MESSAGE SPAN IS NOT ENOUGH (review round 1, H1). Message 0 owns everything its
    #: prefix render adds: the BOS and, on Llama-3.1, a ~25-token system block the template injects
    #: even when no system message was sent. Calibration rows are mostly single-turn, so a bar cut
    #: over the message span would be a quantile of means that are ~60% template boilerplate —
    #: applied at serve time to multi-turn requests, where the newest message is never message 0
    #: and carries none of it. Starting at the role header makes the span the same shape in both.
    #:
    #: `0` for a hand-built row (no preamble known); `render_messages` always sets it explicitly.
    last_user_start: Optional[int] = 0
    #: Parallel to `input_ids`: True where the token is part of some message's own text.
    #: `None` when it could not be located (see `_content_mask`); `effective_scope` then scores
    #: the row under `all`, and `render_all` counts it.
    content_mask: Optional[List[bool]] = None
    #: True when this row was rendered in the SERVED form (`served_render`): generation
    #: prompt on a row that does not end with an assistant turn. `render_messages` always sets
    #: it; the False default exists for hand-built rows and keeps the pre-2026-10-08 semantics.
    served_form: bool = False
    #: WHERE THE PROMPT ENDS, positionally, exactly as miLLM computes it: `len(input_ids)` for a
    #: row with no response, the generation-prompt boundary for an assistant-ended one, and
    #: `None` when that boundary is not a token prefix of the full render (miLLM then scores no
    #: prompt or response window, and neither does this). Only read when `served_form`.
    prompt_tokens: Optional[int] = None

    def keep_tail(self, limit: int) -> bool:
        """Truncate to the last `limit` tokens, moving EVERY positional field with the ids.

        ⚠ ONE PLACE FOR IT. The test-vector builder once cut `input_ids`, `token_roles` and
        `token_message` by hand and left `content_mask`, `last_user_start` and the prompt
        boundary describing the uncut row — a mask longer than the ids pairs each flag with the
        wrong position under `zip`, silently. Returns True when anything was cut.
        """
        if limit < 1 or len(self.input_ids) <= limit:
            return False
        cut = len(self.input_ids) - limit
        if self.last_user_start is not None:
            # A newest message cut at the front is a PARTIAL turn — left out of `last_user`
            # rather than calibrated as though it were whole (review round 1, L1).
            self.last_user_start = None if self.last_user_start < cut else self.last_user_start - cut
        if self.prompt_tokens is not None:
            self.prompt_tokens = max(0, self.prompt_tokens - cut)
        self.input_ids = list(self.input_ids)[-limit:]
        self.content_mask = None if self.content_mask is None else list(self.content_mask)[-limit:]
        self.token_roles = list(self.token_roles)[-limit:]
        self.token_message = list(self.token_message)[-limit:]
        self.truncated = True
        return True

    def _served_mask(self, scope: str) -> Optional[List[bool]]:
        """The positional windows of a served-form row, as miLLM's `scored_mask` defines them.

        `all` is EVERY position — the generation prompt included, because miLLM scores it, and a
        mean over the served tokens is not a mean over the message tokens. `input` is
        `[0, prompt_tokens)` and `last_assistant` is `[prompt_tokens, n)`; neither needs the role
        mask, which is why they survive an unreliable one here exactly as they do in miLLM.
        Other scopes are role- or span-based and return None (handled by the caller).
        """
        n = len(self.input_ids)
        if scope == "all":
            return [True] * n
        if scope in ("input", "last_assistant"):
            if self.prompt_tokens is None:
                return [False] * n
            boundary = min(self.prompt_tokens, n)
            if scope == "input":
                return [i < boundary for i in range(n)]
            return [i >= boundary for i in range(n)]
        return None

    def scored_mask(self, scope: str) -> List[bool]:
        """Which positions a probe with this `scope` may score.

        ⚠ ON A SERVED-FORM ROW (`served_form`, everything `render_messages` produces since
        2026-10-08) `all`, `input` and `last_assistant` are POSITIONAL and match miLLM's
        windows exactly: `all` scores every position INCLUDING the generation prompt, because
        miLLM does, and a probe calibrated over fewer positions than it is served over is
        calibrated on another distribution. Before that date the rule here was "scaffolding is
        never scored", which made a `mean`'s denominator the message tokens alone — exactly the
        train/serve gap the served form closes. Role- and span-based scopes are unchanged.
        """
        if scope not in SCOPE_ROLES:
            raise ValueError(f"unknown scope {scope!r}; use one of {sorted(SCOPE_ROLES)}")
        if self.served_form and scope in POSITIONAL_SCOPES:
            return self._served_mask(scope)  # type: ignore[return-value]
        if not self.role_mask_reliable and scope != "all":
            raise ValueError(
                f"this row's role mask is unreliable, so scope {scope!r} cannot be "
                f"honoured; only 'all' is available (see role_mask_unreliable)"
            )
        allowed = SCOPE_ROLES[scope]
        if scope == "input":
            last = self._last_message_index_with_role("assistant")
            if last is None:
                # Nothing was generated, so the whole row is input. This is the ordinary case
                # for the prose training corpus, where a row is a single user turn.
                return [m >= 0 for m in self.token_message]
            return [m >= 0 and m != last for m in self.token_message]
        if scope == "content":
            if self.content_mask is None:
                raise ValueError(
                    "this row has no content mask, so scope 'content' cannot be honoured; "
                    "only 'all' is available (see effective_scope)"
                )
            return [bool(c) and m >= 0 for c, m in zip(self.content_mask, self.token_message)]
        if scope == "last_assistant":
            last = self._last_message_index_with_role("assistant")
            if last is None:
                return [False] * len(self.input_ids)
            return [m == last for m in self.token_message]
        if scope == "last_user":
            # The newest user message, from its own role header (`last_user_start`). A row with
            # none scores nothing in this window, and the scorer leaves such a row out of the
            # window's negatives rather than inventing one.
            last = self._last_message_index_with_role("user")
            if last is None or self.last_user_start is None:
                return [False] * len(self.input_ids)
            start = self.last_user_start
            return [m == last and i >= start for i, m in enumerate(self.token_message)]
        if allowed is None:
            return [m >= 0 for m in self.token_message]
        return [role in allowed for role in self.token_roles]

    def _last_message_index_with_role(self, role: str) -> Optional[int]:
        best = None
        for index, token_role in zip(self.token_message, self.token_roles):
            if token_role == role and index >= 0:
                best = index if best is None else max(best, index)
        return best


def effective_scope(example: "RenderedExample", scope: str) -> str:
    """The scope a row is actually scored under: `all` where its mask for `scope` is unavailable.

    ONE PLACE, used by the render's token count (which pre-sizes the capture memmap) and by the
    capture's mask, so the two cannot disagree about a row. `last_user` is not handled here — it
    scores nothing on an unreliable row, and `_row_mask` owns that exception.
    """
    if example.served_form and scope in POSITIONAL_SCOPES:
        return scope
    if not example.role_mask_reliable:
        return "all"
    if scope == "content" and example.content_mask is None:
        return "all"
    return scope


def _content_mask(
    tokenizer: Any,
    full_text: str,
    full_ids: Sequence[int],
    messages: Sequence[Dict[str, str]],
    boundaries: Sequence[int],
) -> Optional[List[bool]]:
    """Which tokens of the render fall inside some message's own text, or None.

    ⚠ OFFSETS ARE USED ONLY TO FIND TEXT, AND ONLY WHEN VERIFIED. The module header explains why
    roles come from prefix rendering, not offsets; that still holds — message ownership is
    unchanged, and this mask is intersected with it. Offsets answer a narrower question (where in
    the render is this message's text?) and are trusted only when (a) the tokenizer provides them,
    (b) the offsets call reproduces the exact ids of the real encoding, and (c) every non-empty
    message's content is found inside THAT MESSAGE'S OWN span (a template that escapes or rewrites
    content fails this). Any failure returns None, and the row is scored under `all`.

    ⚠ SEARCHED WITHIN THE MESSAGE'S SPAN, LAST MATCH (review, 2026-10-05). A search of the whole
    render from the start matched a short message against the PREAMBLE — "Today" against Llama's
    "Today Date: …" — and would have scored the template under the text's name. Message 0's span
    holds the preamble too, so the LAST occurrence within the span is taken: headers precede the
    text, so the text is the final copy. `boundaries` are the prefix-render ends, one per message.
    """
    try:
        encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
    except (NotImplementedError, ValueError, TypeError, KeyError):
        return None
    if list(encoded["input_ids"]) != list(full_ids) or len(offsets) != len(full_ids):
        return None
    if len(boundaries) != len(messages):
        return None

    def char_at(token: int) -> int:
        return len(full_text) if token >= len(offsets) else int(offsets[token][0])

    spans: List[Tuple[int, int]] = []
    token_start = 0
    for message, token_end in zip(messages, boundaries):
        content = str(message.get("content") or "").strip()
        span_start, span_end = char_at(token_start), char_at(token_end)
        token_start = token_end
        if not content:
            continue
        at = full_text.rfind(content, span_start, span_end)
        if at < 0:
            return None
        spans.append((at, at + len(content)))
    if not spans:
        return None
    return [
        end > start and any(start < span_end and span_start < end for span_start, span_end in spans)
        for start, end in offsets
    ]


def template_hash(tokenizer: Any) -> str:
    """sha256 of the chat template, recorded on the run (FR-15).

    The template is part of the function a probe learned: the same weights over a
    different template read different tokens at different positions. 033 publishes
    this hash so a serving runtime can refuse a mismatch instead of silently scoring
    a differently-delimited conversation.
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _apply_template(
    tokenizer: Any, messages: Sequence[Dict[str, str]], *, generation_prompt: bool = False
) -> str:
    """The template, verbatim. `generation_prompt` defaults OFF because every caller but
    `served_render` is rendering a PREFIX — the message spans and the role-header probes, which
    miLLM computes without it too (`probe_turns`)."""
    return tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=generation_prompt
    )


def served_render(
    tokenizer: Any, messages: Sequence[Dict[str, str]]
) -> Tuple[str, List[int], Optional[int]]:
    """`(text, ids, prompt_tokens)` — a conversation rendered EXACTLY as miLLM serves it.

    ⚠ THE ONE PLACE THE SERVED RENDER RULE LIVES (see `render_form_record`). Every probe input
    — training capture, calibration, evaluation, the GPU re-cut, the offline score and the
    definition's test vectors — reaches it through `render_messages`, and an AST guard
    (`test_probe_generation_prompt.py`) pins that this is the only full-conversation render.

    Mirrors miLLM `ProbeInputPreparer.prepare` branch for branch:
      * last turn is `assistant` → render WITHOUT the generation prompt; the prompt ends at the
        render of `messages[:-1]` WITH it, accepted only when those ids are a prefix of the full
        render (`None` otherwise, never guessed);
      * anything else → render WITH the generation prompt; the whole row is prompt.

      * ids by miLLM's start-of-text rule (`prompt_encoding.rendered_chat_ids`), for the full
        render AND the boundary render, as miLLM tokenizes both.

    Pinned against miLLM by `docs/schemas/served-render-cases.json` (`test_served_render_cases`).
    ⚠ UNTIL 2026-10-08 THIS ALWAYS PASSED `add_special_tokens=False` (finding F1): on a template
    that writes no BOS while its tokenizer adds one (TinyLlama-Chat / Zephyr-style) miLLM served
    one BOS and this trained on none. It now uses miLLM's rule, and the case runs as a pass.
    """
    if str(messages[-1].get("role", "")) == "assistant":
        text = _apply_template(tokenizer, messages, generation_prompt=False)
        ids = rendered_chat_ids(tokenizer, text)
        try:
            head = (
                rendered_chat_ids(
                    tokenizer, _apply_template(tokenizer, messages[:-1], generation_prompt=True)
                )
                if len(messages) > 1
                else []
            )
        except Exception:  # noqa: BLE001 - a template may refuse a partial conversation
            # miLLM fails such an INPUT outright; here the row keeps its `all` positions (which
            # are the full render either way) and simply has no prompt/response boundary — the
            # same "never guessed" outcome as a boundary that is not a prefix.
            head = []
        prompt_tokens = len(head) if head and ids[: len(head)] == head else None
        return text, ids, prompt_tokens
    text = _apply_template(tokenizer, messages, generation_prompt=SERVED_GENERATION_PROMPT)
    ids = rendered_chat_ids(tokenizer, text)
    return text, ids, len(ids)


def _encode(tokenizer: Any, text: str) -> List[int]:
    """TEMPLATE SPACE: the render's own ids, nothing added (`prompt_encoding.template_ids`).

    ⚠ NOT THE SERVED IDS. Prefix renders, role headers and offsets are positions INSIDE the
    render; computed here and then placed in the served ids by `render_messages`, as miLLM's
    `probe_turns` does. Asking for special tokens here would put the tokenizer's BOS (and on some
    tokenizers an EOS) inside every prefix and break the prefix property for nothing.
    """
    return template_ids(tokenizer, text)


#: Two tiny conversations whose final user turns differ only in content. The common start of those
#: turns' token spans is the template's user header — template-agnostic, and the SAME construction
#: miLLM uses (`probe_turns.user_header_ids`), pinned by a shared case file both repos test.
_HEADER_PROBES = (
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "x"}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "y"}],
)
#: Single-message renders whose contents differ in one character: their common prefix is the
#: template's preamble plus the first user header, and holds NO content (review round 2, H-A).
_FIRST_PROBES = ([{"role": "user", "content": "x"}], [{"role": "user", "content": "y"}])
#: Keyed WEAKLY by the tokenizer object, then by template: an `id()` can be reused after GC by a
#: tokenizer with the same template and a different vocabulary (review round 2, L-A).
_header_cache: "weakref.WeakKeyDictionary[Any, Dict[str, Any]]" = weakref.WeakKeyDictionary()


def _cached(tokenizer: Any, name: str, compute: Any, *, refresh: bool = False) -> Any:
    key = f"{name}:{template_hash(tokenizer)}"
    try:
        per_tokenizer = _header_cache.setdefault(tokenizer, {})
    except TypeError:  # an object that cannot be weakly referenced is simply not cached
        return compute()
    if refresh or key not in per_tokenizer:
        per_tokenizer[key] = compute()
    return per_tokenizer[key]


def _normalised_text(tokenizer: Any, ids: Sequence[int]) -> str:
    text = tokenizer.decode(list(ids), skip_special_tokens=False)
    return text.replace("\u2581", " ").strip()


def first_user_header(tokenizer: Any, *, refresh: bool = False) -> Optional[Tuple[List[int], int]]:
    """`(prefix, start)` for a conversation whose message 0 is a user turn, or `None`.

    `prefix` is what every such render begins with — preamble and header, never content — and
    `start` is where the header begins inside it. ⚠ NEVER LOCATED IN THE CONTENT: the first
    version searched message 0's span for the header's last occurrence, so a user who typed a
    role header into a single-turn message moved the window past everything before it (review
    round 2, H-A). And the header is matched by its TEXT at its own position, because a
    SentencePiece tokenizer encodes a header at the start of a string with a leading `▁` that the
    same header after a turn does not have — the first version therefore found no header on any
    single-turn row of such a template (M-A). Identical in miLLM (`probe_turns.first_user_header`).

    ⚠ THE PREFIX HOLDS THE PREAMBLE, AND A PREAMBLE CAN CHANGE WITH THE CLOCK (review round 3,
    M1): templates that call `strftime_now` stamp today's date into it, so a prefix cached
    yesterday matches nothing today. Callers pass `refresh=True` once on a mismatch.
    """
    return _cached(tokenizer, "first", lambda: _first_user_header(tokenizer), refresh=refresh)


def _first_user_header(tokenizer: Any) -> Optional[Tuple[List[int], int]]:
    header = user_header_ids(tokenizer)
    if not header:
        return None
    try:
        a, b = (_encode(tokenizer, _apply_template(tokenizer, c)) for c in _FIRST_PROBES)
    except Exception:  # noqa: BLE001 - a template that refuses a lone user turn has no answer here
        return None
    prefix: List[int] = []
    for x, y in zip(a, b):
        if x != y:
            break
        prefix.append(x)
    # The header ends the prefix. Its tokens may differ at the front only (a merged `▁`), so take
    # the longest common suffix, plus one merged token when it is shorter than the header.
    shared = 0
    while shared < min(len(header), len(prefix)) and prefix[-1 - shared] == header[-1 - shared]:
        shared += 1
    start = len(prefix) - shared - (0 if shared == len(header) else 1)
    if start < 0 or _normalised_text(tokenizer, prefix[start:]) != _normalised_text(tokenizer, header):
        return None
    return prefix, start


def user_header_ids(tokenizer: Any) -> Optional[List[int]]:
    """The token ids of a user turn's role header, or `None` when they cannot be isolated."""
    return _cached(tokenizer, "header", lambda: _user_header_ids(tokenizer))


def _user_header_ids(tokenizer: Any) -> Optional[List[int]]:
    spans: List[List[int]] = []
    try:
        for conversation in _HEADER_PROBES:
            before = _encode(tokenizer, _apply_template(tokenizer, conversation[:2]))
            through = _encode(tokenizer, _apply_template(tokenizer, conversation))
            if through[: len(before)] != before:
                spans = []
                break
            spans.append(through[len(before):])
    except Exception:  # noqa: BLE001 - a template that refuses the probe has no isolable header
        spans = []
    header: Optional[List[int]] = None
    if len(spans) == 2:
        common: List[int] = []
        for a, b in zip(*spans):
            if a != b:
                break
            common.append(a)
        header = common or None
    return header


def last_user_header_start(
    tokenizer: Any, ids: Sequence[int], boundaries: Sequence[int], last: int
) -> Optional[int]:
    """Where message `last`'s own role header starts in `ids`.

    A later message starts at the previous message's boundary — its span holds nothing but its
    own turn. Message 0 also holds the BOS and any preamble the template injects; its header
    start comes from content-free renders (`first_user_header`), and the row must begin with
    exactly that prefix or it is left out.
    """
    if last > 0:
        return int(boundaries[last - 1])
    def matches(found: Optional[Tuple[List[int], int]]) -> bool:
        return found is not None and list(ids[: len(found[0])]) == found[0] and (
            len(found[0]) <= int(boundaries[0])
        )

    found = first_user_header(tokenizer)
    if not matches(found):
        # Recompute once before refusing: the cached preamble may be stale (a date).
        found = first_user_header(tokenizer, refresh=True)
        if not matches(found):
            return None
    return found[1]


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
    full_text, full_ids, prompt_tokens = served_render(tokenizer, messages)
    if not full_text or not full_text.strip():
        raise ValueError(
            "the chat template rendered this conversation to an empty string. Some "
            "templates are an if/elif chain with no else, so an unexpected role "
            "produces nothing and the row would train on no tokens at all"
        )
    if not full_ids:
        raise ValueError("the rendered conversation tokenized to zero tokens")

    # ⚠ POSITIONS ARE FOUND IN TEMPLATE SPACE AND PLACED IN THE SERVED IDS (2026-10-08, F1).
    # `full_ids` follow the start-of-text rule, so on a template that writes no BOS the tokenizer
    # has put its own in front (and some tokenizers append an EOS). The prefix renders below are
    # template space, nothing added, so every message boundary is shifted by the `lead` tokens the
    # tokenizer added in front; whatever it added behind is scaffolding no message owns. This is
    # miLLM's placement (`probe_turns.last_user_token_span`), not a second rule.
    core_ids = _encode(tokenizer, full_text)
    lead = _placement(full_ids, core_ids)
    trailing = 0 if lead is None else len(full_ids) - lead - len(core_ids)

    # Prefix lengths: render m[:1], m[:2], … and record where each ends.
    boundaries: List[int] = []
    reliable = lead is not None
    if not reliable:
        logger.warning(
            "probe_monitor.role_mask_unreliable: the served ids do not contain the render's own "
            "ids as one run, so no message can be placed; falling back to scope 'all'"
        )
    previous_ids: List[int] = []
    for i in range(1, len(messages) + 1 if reliable else 1):
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
    if reliable and previous_ids and core_ids[: len(previous_ids)] != previous_ids:
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
            for position in range(lead + start, min(lead + end, lead + len(core_ids))):
                token_roles[position] = role
                token_message[position] = message_index
            start = end
    else:
        # Every real token belongs to "the conversation" and to no message. Scope
        # `all` still works; any role-scoped request raises rather than guessing.
        for position in range(len(full_ids)):
            token_message[position] = 0
            token_roles[position] = ""

    last_user_start: Optional[int] = None
    if reliable:
        last_user = max(
            (i for i, m in enumerate(messages) if str(m.get("role", "")) == "user"), default=None
        )
        # miLLM places the span only when the served ids END with the render's own ids
        # (`served[offset:] == full`), so a trailing special token leaves it unresolved there,
        # and it is left out here rather than calibrated over a span nothing scores.
        if last_user is not None and trailing == 0:
            found = last_user_header_start(tokenizer, core_ids, boundaries, last_user)
            last_user_start = None if found is None else lead + found

    content_mask: Optional[List[bool]] = None
    if reliable:
        located = _content_mask(tokenizer, full_text, core_ids, messages, boundaries)
        if located is not None:
            content_mask = [False] * lead + located + [False] * trailing

    example = RenderedExample(
        input_ids=full_ids,
        token_roles=token_roles,
        token_message=token_message,
        text=full_text,
        role_mask_reliable=reliable,
        truncated=False,
        last_user_start=last_user_start,
        content_mask=content_mask,
        served_form=True,
        prompt_tokens=prompt_tokens,
    )
    if max_length is not None:
        # ⚠ TRUNCATE THE FRONT, KEEPING THE END. The assistant's reply is at the end
        # of a conversation, so head-truncation would remove exactly the tokens an
        # `assistant`-scoped probe exists to read. Recorded on the row so the run can
        # report how many examples lost context rather than discovering it later.
        example.keep_tail(max_length)
    return example


def _placement(served: Sequence[int], core: Sequence[int]) -> Optional[int]:
    """How many tokens the tokenizer put IN FRONT of the render's own ids, or None when the served
    ids do not contain them as one run (nothing then can be placed). The first, i.e. smallest,
    offset: anything added goes before or after the render, never inside it."""
    served, core = list(served), list(core)
    for lead in range(0, len(served) - len(core) + 1):
        if served[lead: lead + len(core)] == core:
            return lead
    return None


@dataclass
class RenderSummary:
    """What a render pass has to report (FR-15 and the §12 log keys)."""

    rendered: int = 0
    failed: int = 0
    role_mask_unreliable: int = 0
    #: Rows whose `content` mask could not be located, scored under `all` instead. Counted for
    #: every run, reported whatever the scope, so a run scoped `content` shows its fallback.
    content_mask_unavailable: int = 0
    truncated: int = 0
    scored_tokens: int = 0
    failures: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rendered": self.rendered,
            "failed": self.failed,
            "role_mask_unreliable": self.role_mask_unreliable,
            "content_mask_unavailable": self.content_mask_unavailable,
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
        if rendered.content_mask is None:
            summary.content_mask_unavailable += 1
        if rendered.truncated:
            summary.truncated += 1
        summary.scored_tokens += sum(rendered.scored_mask(effective_scope(rendered, scope)))
        results.append(rendered)
    return results, summary
