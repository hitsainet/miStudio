"""
miLLM inference (category: millm_inference) — score and generate against the model miLLM SERVES
(Feature 034, FR-16, FR-19, FR-21, FR-23).

Built for miDataworks' loop: label rows with a classifier's scores, generate candidate text,
and run either as a batch (the batch tools, `millm_batches.py`, share this category).

Four rules hold for every tool here:

1. **They never swap the served model.** Every request says `X-miLLM-Load-Policy: refuse`, so a
   request naming a model that is not resident is REFUSED with
   `{"refused": "model_not_resident", "next": "millm_load_model", …}` instead of evicting
   whatever is serving. Loading is a separate, operator-approved step.
2. **Strict.** Every request says `X-miLLM-Strict: true`: a field miLLM would ignore is a 400
   naming it, never a silent drop.
3. **Provenance is reported, never assumed.** Results come back as
   `{"result": <miLLM's body>, "millm_headers": {steering, seed, constrained, ignored_fields}}`
   with each header verbatim when miLLM sent it.
4. **Bounded.** Synchronous tools cap their inputs; over a cap they refuse before anything is
   sent and name the batch tools.

Four result shapes an agent must tell apart: `{"error": …}`-style argument refusals (fix the
call), `{"unavailable": "millm", …}` (miLLM is down — report it, don't loop),
`{"refused": <code>, …}` (miLLM declined — follow `next`, or wait `retry_after_s`), and a raised
tool error (miLLM rejected the request; its message says why).

All tools are health-gated; arguments are validated BEFORE the gate, so an agent debugging its
payload is never told "miLLM is down".
"""

import inspect
from typing import Annotated, Any, Literal, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from ..context import ExternalContext
from ..health_gate import HealthGate
from ..millm_client import MiLLMClient
from ..millm_send import send, sent_headers, steering_state

#: Synchronous caps (FTDD §9). Interpolated into the descriptions so the text cannot drift from
#: the check. Revised by measurement at acceptance.
MAX_SCORE_CONVERSATIONS = 32
MAX_SCORE_PROMPTS = 32
MAX_GENERATE_TOKENS = 1024

BATCH_TOOLS = "millm_upload_batch_file and millm_submit_batch"


class ChatMessage(BaseModel):
    """One chat turn."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


#: The four result shapes, stated in every description (list_tools() does not carry the module
#: docstring, so an agent learns them here or nowhere).
SHAPES = (
    'Four result shapes: `{"result", "millm_headers"}` (success, with miLLM\'s provenance '
    'headers verbatim); `{"refused": <code>, "next", "retry_after_s"}` (miLLM declined — follow '
    '`next` or wait); `{"unavailable": "millm", "reason"}` (miLLM is down — report it, do not '
    'loop); `{"error": …}` (your arguments — fix and retry).'
)

# Descriptions are module constants so they can interpolate the caps (an f-string is not a
# docstring); `@mcp.tool(description=…)` is what `list_tools()` hands an agent.
SCORE_CHAT_DESCRIPTION = inspect.cleandoc(f"""Score the next token of one or more conversations on
    the served model — UNSTEERED by the request (`X-miLLM-Steering` in the result says what
    actually applied).

    Sends `logprobs: true`, `max_tokens: 1`, `temperature: 0`: one scored position per
    conversation, deterministic. Use `allowed_token_ids` for a classifier's label tokens.

    At most {MAX_SCORE_CONVERSATIONS} conversations per call (`messages` plus `extra_messages`);
    for more, use the batch tools ({BATCH_TOOLS}).

    Never loads a model: naming one that is not resident returns
    `{{"refused": "model_not_resident", "next": "millm_load_model"}}`; a model leased by someone
    else returns `refused: "model_leased"`. A `503` returns `retry_after_s`. A read timeout means
    the request may already have been applied — check before retrying.

{SHAPES}""")

SCORE_COMPLETION_DESCRIPTION = inspect.cleandoc(f"""Score the next token of raw prompts on the
    served model (`/v1/completions` scoring mode) — UNSTEERED by the request; `X-miLLM-Steering`
    in the result says what actually applied. No chat template is applied: use
    millm_score_chat for conversations.

    Sends `max_tokens: 1`, `temperature: 0` and `logprobs`. At most {MAX_SCORE_PROMPTS} prompts
    per call; for more, use the batch tools ({BATCH_TOOLS}).

    Never loads a model: a model that is not resident returns
    `{{"refused": "model_not_resident", "next": "millm_load_model"}}`; one leased by someone
    else returns `refused: "model_leased"`. A `503` returns `retry_after_s`. A read timeout means
    the request may already have been applied — check before retrying.

{SHAPES}""")


GENERATE_DESCRIPTION = inspect.cleandoc(f"""Generate a reply on the served model, with
    optional inline steering, a seed, structured output, or the SAE activations it produced.

    `steering` is passed to miLLM VERBATIM: `{{"sae_id"?, "features": [{{"index", "strength"}}]}}`;
    `{{"features": []}}` means explicitly unsteered for this request. The result's
    `millm_headers.steering` is miLLM's `X-miLLM-Steering` header verbatim, and `steering_state`
    reads it: `unsteered` ONLY when the header is exactly `none`; `unknown` is not unsteered; a
    missing header is `not_reported` — a globally active profile steers chat unless the request
    opts out, so never assume. `seed` is echoed in `millm_headers.seed` with the scope miLLM
    promised; `response_format` is constrained only where miLLM says so (`constrained`).

    `max_tokens` is capped at {MAX_GENERATE_TOKENS}; for more, or many prompts, use the batch
    tools ({BATCH_TOOLS}). Never loads a model: a non-resident model returns
    `{{"refused": "model_not_resident", "next": "millm_load_model"}}`. A read timeout means the
    request may already have been applied — check before retrying.

    {SHAPES}""")


def _over_cap(what: str, count: int, cap: int) -> dict[str, Any]:
    return {
        "error": "over_synchronous_cap",
        "message": (
            f"{count} {what} is over this tool's synchronous cap of {cap}. Split the work, or "
            f"run it as a batch with {BATCH_TOOLS} — a batch has no such cap and never blocks "
            "the served model."
        ),
    }


def register(
    mcp: FastMCP, millm: MiLLMClient, gate: HealthGate, ctx: ExternalContext
) -> None:
    @mcp.tool(description=SCORE_CHAT_DESCRIPTION)
    async def millm_score_chat(
        model: Annotated[str, Field(description="The served model's name, as millm_list_models reports it. Must be RESIDENT: this tool never loads one")],
        messages: Annotated[list[ChatMessage], Field(description="The conversation to score; the model's next token is scored", min_length=1)],
        extra_messages: Annotated[Optional[list[list[ChatMessage]]], Field(description="More conversations scored in the same forward pass; result index i+1 is extra_messages[i]")] = None,
        top_logprobs: Annotated[Optional[int], Field(description="How many alternative tokens to return per position (0-20)", ge=0, le=20)] = None,
        allowed_token_ids: Annotated[Optional[list[int]], Field(description="Restrict the next token to these ids, renormalised over the set (e.g. the ids of 'yes' and 'no')", min_length=1, max_length=1024)] = None,
        lease_id: Annotated[Optional[str], Field(description="Your lease id, if you hold miLLM's lease; sent as X-miLLM-Lease")] = None,
    ) -> Any:
        """Score the next token of one or more conversations on the served model."""
        conversations = 1 + len(extra_messages or [])
        if conversations > MAX_SCORE_CONVERSATIONS:
            return _over_cap("conversations", conversations, MAX_SCORE_CONVERSATIONS)
        body: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "logprobs": True,
            "max_tokens": 1,
            "temperature": 0,
        }
        if extra_messages:
            body["extra_messages"] = [[m.model_dump() for m in c] for c in extra_messages]
        if top_logprobs is not None:
            body["top_logprobs"] = top_logprobs
        if allowed_token_ids is not None:
            body["allowed_token_ids"] = list(allowed_token_ids)
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        return await send(gate, millm.post(
            "/v1/chat/completions", json_body=body, headers=sent_headers(lease_id), full=True,
        ))


    @mcp.tool(description=SCORE_COMPLETION_DESCRIPTION)
    async def millm_score_completion(
        model: Annotated[str, Field(description="The served model's name. Must be RESIDENT: this tool never loads one")],
        prompt: Annotated[list[str], Field(description="Raw prompts (no chat template applied); the next token of each is scored", min_length=1)],
        logprobs: Annotated[int, Field(description="How many alternative tokens to return per position (0-20)", ge=0, le=20)] = 5,
        allowed_token_ids: Annotated[Optional[list[int]], Field(description="Restrict the next token to these ids, renormalised over the set", min_length=1, max_length=1024)] = None,
        add_special_tokens: Annotated[bool, Field(description="Prepend the tokenizer's special tokens (BOS). Leave true unless the prompt already carries them")] = True,
        lease_id: Annotated[Optional[str], Field(description="Your lease id, if you hold miLLM's lease; sent as X-miLLM-Lease")] = None,
    ) -> Any:
        """Score the next token of raw prompts on the served model (`/v1/completions` scoring)."""
        if len(prompt) > MAX_SCORE_PROMPTS:
            return _over_cap("prompts", len(prompt), MAX_SCORE_PROMPTS)
        body: dict[str, Any] = {
            "model": model,
            "prompt": list(prompt),
            "logprobs": logprobs,
            "max_tokens": 1,
            "temperature": 0,
            "add_special_tokens": add_special_tokens,
        }
        if allowed_token_ids is not None:
            body["allowed_token_ids"] = list(allowed_token_ids)
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        return await send(gate, millm.post(
            "/v1/completions", json_body=body, headers=sent_headers(lease_id), full=True,
        ))

    @mcp.tool(description=GENERATE_DESCRIPTION)
    async def millm_generate(
        model: Annotated[str, Field(description="The served model's name. Must be RESIDENT: this tool never loads one")],
        messages: Annotated[list[ChatMessage], Field(description="The conversation to continue", min_length=1)],
        max_tokens: Annotated[int, Field(description=f"Tokens to generate, 1-{MAX_GENERATE_TOKENS}", ge=1)],
        seed: Annotated[Optional[int], Field(description="Reproducible sampling; echoed in millm_headers.seed with its scope", ge=0)] = None,
        temperature: Annotated[Optional[float], Field(description="Sampling temperature (0-2); omit for miLLM's default", ge=0, le=2)] = None,
        response_format: Annotated[Optional[dict[str, Any]], Field(description="OpenAI response_format ({'type': 'json_object'} or a json_schema), passed verbatim")] = None,
        steering: Annotated[Optional[dict[str, Any]], Field(description="Inline steering, passed verbatim: {'sae_id'?, 'features': [{'index', 'strength'}]}; {'features': []} = explicitly unsteered")] = None,
        return_sae_activations: Annotated[Optional[dict[str, Any]], Field(description="miLLM's per-request activations request, passed verbatim")] = None,
        lease_id: Annotated[Optional[str], Field(description="Your lease id, if you hold miLLM's lease; sent as X-miLLM-Lease")] = None,
    ) -> Any:
        """Generate a reply on the served model, reporting steering exactly as miLLM does."""
        if max_tokens > MAX_GENERATE_TOKENS:
            return _over_cap("max_tokens", max_tokens, MAX_GENERATE_TOKENS)
        body: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
        }
        # Passed through VERBATIM (FR-16, FR-28.1): this tool adds no `profile` beside
        # `steering` (miLLM refuses the pair) and interprets none of these.
        for key, value in (("seed", seed), ("temperature", temperature),
                           ("response_format", response_format), ("steering", steering),
                           ("return_sae_activations", return_sae_activations)):
            if value is not None:
                body[key] = value
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        result = await send(gate, millm.post(
            "/v1/chat/completions", json_body=body, headers=sent_headers(lease_id), full=True,
        ))
        if "millm_headers" in result:
            # Read from miLLM's header and nothing else: a tool never writes "unsteered".
            result["steering_state"] = steering_state(result["millm_headers"].get("steering"))
        return result
