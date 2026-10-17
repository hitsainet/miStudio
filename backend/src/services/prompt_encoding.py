"""How a chat-template render becomes token ids: miLLM's start-of-text (BOS) rule, mirrored.

⚠ THE OPERATOR'S DECISION ON F1 (2026-10-08). Until this date miStudio tokenized every probe render
with `add_special_tokens=False`, on the reasoning that the template emits the BOS itself. That is
true of Llama 3, gemma and LFM2.5 and false of a template that writes NO BOS while its tokenizer
adds one (TinyLlama-1.1B-Chat / Zephyr-style Llama 2 templates). On that form miLLM serves one BOS
and miStudio trained on none: a probe calibrated on sequences its server never produces
(`0xcc/reviews/served_render_cases_2026-10-08.md` §F1). miStudio now uses miLLM's rule.

THE RULE, BRANCH FOR BRANCH WITH miLLM `millm/services/prompt_encoding.py`
(`bos_text`, `rendered_chat_adds_special_tokens`, `rendered_chat_ids`):

* the render already begins with the tokenizer's `bos_token` TEXT → `add_special_tokens=False`
  (the template carries the one BOS);
* otherwise → `add_special_tokens=True`: the tokenizer adds exactly what it adds by default — one
  BOS on TinyLlama, nothing at all on a tokenizer that adds nothing (Qwen2.5 has no `bos_token`;
  granite 4.x has one it never adds).

The result is never a duplicate, and exactly one BOS when the model uses one. Pinned against miLLM
by `docs/schemas/served-render-cases.json` (`test_served_render_cases.py`), which runs both
repos' renderers over the same cases.

`bos_handling` records what the rule DID for one render, so a probe's definition can say exactly
how its ids were built (`RenderForm.bos_handling` in the contract).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def bos_text(tokenizer: Any) -> Optional[str]:
    """The tokenizer's BOS token as text, or None when it has none or it is unreadable.

    `isinstance` rather than truthiness, exactly as miLLM: a test double's attribute is not a
    string, and treating it as one would make the rule depend on whatever the double returns.
    """
    value = getattr(tokenizer, "bos_token", None)
    return value if isinstance(value, str) and value else None


def rendered_chat_adds_special_tokens(tokenizer: Any, text: str) -> bool:
    """`add_special_tokens` for one template render: False exactly when the render already begins
    with the BOS text — the one decision this module exists to make."""
    bos = bos_text(tokenizer)
    return not (bos is not None and text.startswith(bos))


def rendered_chat_ids(tokenizer: Any, text: str) -> List[int]:
    """The served ids of one template render, under the rule above."""
    return list(
        tokenizer(text, add_special_tokens=rendered_chat_adds_special_tokens(tokenizer, text))[
            "input_ids"
        ]
    )


def template_ids(tokenizer: Any, text: str) -> List[int]:
    """The render's ids with NOTHING added by the tokenizer — "template space".

    Not a served form. Positions inside a render (message boundaries, role headers, the
    `last_user` span, offsets) are computed here and then PLACED inside the served ids, exactly as
    miLLM's `probe_turns` places its spans (`offset = len(served) - len(full)`).
    """
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _leading_bos(ids: Sequence[int], bos_id: Optional[int]) -> int:
    if bos_id is None:
        return 0
    count = 0
    for token in ids:
        if token != bos_id:
            break
        count += 1
    return count


def bos_handling(tokenizer: Any, text: str) -> Dict[str, Any]:
    """What the rule did to ONE render: `{template_wrote_bos, tokenizer_added_bos, bos_count}`.

    * `template_wrote_bos` — the render begins with the BOS text (so nothing was added);
    * `tokenizer_added_bos` — the tokenizer was asked for its special tokens and put a BOS in
      front of the render's own ids;
    * `bos_count` — the BOS tokens the served ids BEGIN with. 1 when the model uses one, 0 when
      it does not; 2 would be the duplicate the rule exists to prevent.

    Measured from the ids, never inferred from the flags.
    """
    bos_id = getattr(tokenizer, "bos_token_id", None)
    bos_id = bos_id if isinstance(bos_id, int) else None
    served = rendered_chat_ids(tokenizer, text)
    core = template_ids(tokenizer, text)
    adds = rendered_chat_adds_special_tokens(tokenizer, text)
    return {
        "template_wrote_bos": not adds,
        "tokenizer_added_bos": adds and _leading_bos(served, bos_id) > _leading_bos(core, bos_id),
        "bos_count": _leading_bos(served, bos_id),
    }


def tokenizer_adds_nothing(tokenizer: Any, text: str) -> bool:
    """Whether the rule's ids for this render equal the render's own ids — i.e. whether the
    pre-2026-10-08 always-False rule produced EXACTLY what the rule produces now."""
    return rendered_chat_ids(tokenizer, text) == template_ids(tokenizer, text)
