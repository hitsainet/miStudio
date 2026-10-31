"""The text a rule reads: a conversation reduced to the span the probe scores (Feature 040 FR-33; FTDD §6.2; 040-D16).

Rules read the SAME span the probe scores for a view, reduced without a tokenizer: the messages a scope
selects, their ``content`` joined by one newline (U+000A). A message whose content is not a string
contributes nothing (no empty line); an empty string is kept. Nothing is normalised and no template markup
is ever added. Spans map back to ``(message_index, offset)``.

Keyed on miStudio's INTERNAL scope names (FTID D-4, TI-13), the keys of
``services/probe_monitor_render.SCOPE_ROLES`` (`test_rule_signals_reduce.py` pins the two equal):

| scope | messages |
|---|---|
| ``all``, ``content`` | every message |
| ``assistant`` / ``user`` | every message of that role |
| ``last_assistant`` | the final message, when it is an assistant's (the reply); else none (``no_scored_tokens``) |
| ``input`` | every message before that final assistant message — all of them when there is none |
| ``last_user`` | the newest user message (``no_user_turn`` when there is none) |

The contract windows map through `CONTRACT_WINDOW_SCOPES` (pinned equal to the render module's). The
serving windows are miLLM 039's (``rule-text-reduction-cases.json``, OWNED by miLLM and vendored here
byte-identically); `window_texts` answers them through this module's own reducer, so every vendored
case runs the production code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCOPES = ("all", "assistant", "user", "last_assistant", "content", "input", "last_user")
#: Contract window → internal scope (equal to `probe_monitor_render.CONTRACT_WINDOW_SCOPES`).
CONTRACT_WINDOW_SCOPES: Dict[str, str] = {
    "all": "all",
    "prompt": "input",
    "response": "last_assistant",
    "last_user": "last_user",
}
JOIN = "\n"


@dataclass(frozen=True)
class Reduced:
    text: str
    #: `(message_index, start in text, length)` for each message joined, in order.
    segments: Tuple[Tuple[int, int, int], ...]
    #: Why the scope has no text (`no_scored_tokens`, `no_user_turn`), else None.
    missing: Optional[str] = None

    def locate(self, start: int, end: int) -> List[Dict[str, int]]:
        """A span of `text` as message-relative pieces `{message_index, start, end}` (split at joins)."""
        out: List[Dict[str, int]] = []
        for index, s, length in self.segments:
            lo, hi = max(start, s), min(end, s + length)
            if lo < hi or (start == end and s <= start <= s + length):
                out.append({"message_index": index, "start": lo - s, "end": max(lo, hi) - s})
                if start == end:
                    break
        return out


def _role(message: Any) -> Optional[str]:
    return message.get("role") if isinstance(message, Mapping) else getattr(message, "role", None)


def _content(message: Any) -> Optional[str]:
    value = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    return value if isinstance(value, str) else None


def _final_assistant(messages: Sequence[Any]) -> Optional[int]:
    if messages and _role(messages[-1]) == "assistant":
        return len(messages) - 1
    return None


def selected(messages: Sequence[Any], scope: str) -> Tuple[List[int], Optional[str]]:
    """The indices a scope reads, and why there are none when that is a named absence."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; one of {list(SCOPES)}")
    n = len(messages)
    if scope in ("all", "content"):
        return list(range(n)), None
    if scope in ("assistant", "user"):
        return [i for i in range(n) if _role(messages[i]) == scope], None
    final = _final_assistant(messages)
    if scope == "last_assistant":
        return ([final], None) if final is not None else ([], "no_scored_tokens")
    if scope == "input":
        return [i for i in range(n) if i != final], None
    users = [i for i in range(n) if _role(messages[i]) == "user" and _content(messages[i]) is not None]
    return ([users[-1]], None) if users else ([], "no_user_turn")


def reduce(messages: Sequence[Any], scope: str) -> Reduced:
    indices, missing = selected(messages, scope)
    parts: List[str] = []
    segments: List[Tuple[int, int, int]] = []
    at = 0
    for i in indices:
        content = _content(messages[i])
        if content is None:
            continue
        if parts:
            at += len(JOIN)
        segments.append((i, at, len(content)))
        parts.append(content)
        at += len(content)
    if missing is None and not segments and scope == "last_assistant":
        missing = "no_scored_tokens"
    return Reduced(JOIN.join(parts), tuple(segments), missing)


def as_messages(row: Any) -> List[Dict[str, Any]]:
    """A row's input as messages: plain prose is ONE user message (as `build_examples` makes it)."""
    if isinstance(row, str):
        return [{"role": "user", "content": row}]
    if isinstance(row, (list, tuple)):
        return [dict(m) if isinstance(m, Mapping) else {"role": _role(m), "content": _content(m)} for m in row]
    if row is None:
        return []
    return [{"role": "user", "content": str(row)}]


def window_texts(*, messages: Optional[Sequence[Any]] = None, prompt: Any = None,
                 reply_text: Optional[str] = None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """miLLM 039's four serving windows, `(texts, missing)`, from this module's reducer.

    A chat's reply is the final assistant message; a text completion (`prompt`) has no user turn."""
    texts: Dict[str, str] = {}
    missing: Dict[str, str] = {}
    if messages is not None:
        conversation = list(messages) + ([{"role": "assistant", "content": reply_text}]
                                          if isinstance(reply_text, str) else [])
        for window, scope in CONTRACT_WINDOW_SCOPES.items():
            reduced = reduce(conversation, scope)
            if reduced.missing is not None:
                missing[window] = reduced.missing
            else:
                texts[window] = reduced.text
        if not isinstance(reply_text, str) and _final_assistant(conversation) is not None:
            # A history that ends on an assistant turn but carries no reply: that turn is prompt, not reply.
            texts.pop("response", None)
            missing["response"] = "no_scored_tokens"
            texts["prompt"] = reduce(conversation, "all").text
        return texts, missing
    if isinstance(prompt, str):
        texts["prompt"] = prompt
    elif isinstance(prompt, (list, tuple)) and prompt and isinstance(prompt[0], str):
        texts["prompt"] = prompt[0]
    missing["last_user"] = "text_completion_has_no_user_turn"
    if isinstance(reply_text, str):
        texts["response"] = reply_text
    else:
        missing["response"] = "no_scored_tokens"
    if "prompt" in texts and "response" in texts:
        texts["all"] = texts["prompt"] + JOIN + texts["response"]
    elif "prompt" in texts or "response" in texts:
        texts["all"] = texts.get("prompt", texts.get("response", ""))
    return texts, missing
