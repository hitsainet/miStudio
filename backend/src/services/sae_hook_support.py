"""Which consumers can take an SAE of a given hook type (review R1-A, A5; widened in review R2-B).

A training over several hook types trains an MLP-output and an attention-output SAE beside
each residual one, and all of them import as ExternalSAE rows. Feature extraction hooks each
layer's decoder output for every SAE, and the Neuronpedia export and push label every SAE
resid_post and key it by layer alone. Given an MLP or attention SAE, each did the wrong thing
without a word: features computed from the wrong activations, a residual label on an MLP
SAE, and a push that overwrote the residual SAE of the same layer. Those consumers refuse
such SAEs, with the reason, through this one rule.

REVIEW R2-B. Circuit capture, the steering resolver and core, and the cluster allocation read
or write every SAE at the decoder layer's output too, and now refuse through the same rule.
The rule matches NAME TOKENS, not substrings, and knows the names imported SAEs carry:
Gemma Scope's ``att``, SAELens' ``hook_z`` / ``hook_q`` / ``hook_pattern``, transcoders and
their ``ln2.hook_normalized`` input. A residual hook recorded BEFORE the layer's output
(``resid_pre``, ``resid_mid``) is refused as well: every consumer here reads layer L's output,
which is ``resid_pre`` of layer L+1, so such an SAE would be read a layer (or half a layer)
late. A residual or unrecorded hook is never refused.

REVIEW R3-B. Real names from HuggingFace (Gemma Scope 2's config.json, sparsify's hookpoint
directories, LFM2's modules) showed three more: an MLP named ``feed_forward`` / ``ffn`` /
``post_feedforward_layernorm`` (Gemma Scope 2's mlp_out) passed as residual, and so did an
embedding SAE (Gemma Scope 1's ``embedding/`` sets, sparsify's ``embed_tokens``,
``hook_embed``), which reads the residual stream before the first layer and is refused like
``resid_pre``. ``classify_hook`` exposes the kind the rule decides, so a consumer that must
COMPARE two hooks (the cluster import binding) uses the same rule as the refusals.

REVIEW R3-B (R3B-11). The same consumers read an SAE AT ITS LAYER, and a NULL
``external_saes.layer`` was turned into ``layer or 0`` by every one of them: feature
extraction hooked layer 0, the Neuronpedia export and push published and keyed layer 0, the
feature browser reported every feature at layer 0 (so steering routed them there), and the
steering loader steered layer 0. None of them said so. ``refuse_unrecorded_layer`` refuses
instead, naming the SAE and how to record the layer -- the hook rule's two sentences and this
one are the only reasons an SAE is refused, and both raise ``UnsupportedSae`` so an endpoint
answers either with 422. The refusals came AFTER ``scripts/backfill_sae_hooks.py``, which
records the layer of rows written before the writers did: refusing first would have failed
every one of those rows at once.
"""

import re
from typing import Optional, Type

#: Tokens that name an MLP-side hook: its output, its input, or a transcoder across it.
_MLP_TOKENS = frozenset({"mlp", "transcoder", "transcoders", "ffn", "feedforward"})
#: Tokens that name an attention hook: its output, or a per-head quantity inside it.
_ATTENTION_TOKENS = frozenset({"attn", "attention", "att", "z", "q", "k", "v", "result", "pattern"})
#: Tokens that name the embedding output: the residual stream before the first layer.
_EMBEDDING_TOKENS = frozenset({"embed", "embedding", "embeddings"})
#: Tokens that name the residual stream.
_RESIDUAL_TOKENS = frozenset({"resid", "residual", "res"})
#: Where a residual hook other than the layer's output reads, by its position token.
_BEFORE_OUTPUT = {
    "pre": "before the layer (resid_pre, the previous layer's output)",
    "mid": "halfway through the layer (resid_mid, after attention and before the MLP)",
}
_BEFORE_FIRST_LAYER = "before the first layer (the embedding output)"

#: The kinds ``classify_hook`` returns.
KIND_RESIDUAL = "residual"
KIND_MLP = "mlp"
KIND_ATTENTION = "attention"
KIND_EMBEDDING = "embedding"
KIND_RESID_PRE = "resid_pre"
KIND_RESID_MID = "resid_mid"


class UnsupportedSae(ValueError):
    """An SAE a consumer cannot use, with the reason.

    A ValueError, so callers that already handle a service's refusals still do; endpoints
    answer it with 422. The two reasons are subclasses, so a handler that wants either
    catches this and one that wants a specific reason still can.
    """


class UnsupportedSaeHook(UnsupportedSae):
    """A consumer that reads every SAE as the residual stream was given another hook's SAE."""


class UnrecordedSaeLayer(UnsupportedSae):
    """A consumer that reads an SAE AT ITS LAYER was given an SAE recording no layer.

    Seven consumers took that NULL as ``layer or 0`` and read, exported or steered the
    wrong layer without a word (review R3-B, R3B-11).
    """


def _tokens(hook_type: str) -> list:
    return [token for token in re.split(r"[^a-z0-9]+", str(hook_type).lower()) if token]


def refuse_non_residual(
    hook_type: Optional[str], consumer: str, error: Type[Exception] = UnsupportedSaeHook
) -> None:
    """Raise ``error`` (UnsupportedSaeHook by default) when ``consumer`` cannot take an SAE at ``hook_type``."""
    reason = non_residual_hook_reason(hook_type, consumer)
    if reason:
        raise error(reason)


#: How an operator records a layer the SAE never recorded. Named in every layer refusal:
#: a refusal that does not say what to do next is a dead end.
BACKFILL_HINT = (
    "Record it by running backend/scripts/backfill_sae_hooks.py (a dry run by default), "
    "which resolves the layer from the SAE's own files and names, or set the layer on the "
    "SAE itself."
)


def unrecorded_layer_reason(
    layer: Optional[int], consumer: str, sae_id: Optional[str] = None
) -> Optional[str]:
    """Why ``consumer`` cannot use an SAE recording no layer, or None when it can.

    Only a NULL is refused. Layer 0 as a RECORDED value is a real layer and is accepted;
    what is refused is a NULL silently becoming 0.
    """
    if layer is not None:
        return None
    named = f" {sae_id}" if sae_id else ""
    return (
        consumer + " reads an SAE at the layer it was trained on, and SAE" + named
        + " records no layer. Using it would silently read LAYER 0 -- almost certainly not "
        "where this SAE was trained, and the result would describe the wrong activations "
        "without a word. " + BACKFILL_HINT
    )


def refuse_unrecorded_layer(
    layer: Optional[int],
    consumer: str,
    sae_id: Optional[str] = None,
    error: Type[Exception] = UnrecordedSaeLayer,
) -> None:
    """Raise ``error`` when ``consumer`` cannot take an SAE that records no layer."""
    reason = unrecorded_layer_reason(layer, consumer, sae_id)
    if reason:
        raise error(reason)


def non_residual_hook_reason(hook_type: Optional[str], consumer: str) -> Optional[str]:
    """Why ``consumer`` cannot take an SAE recorded at ``hook_type``, or None when it can.

    Accepts miStudio's training names (residual, mlp, attention) and the TransformerLens,
    SAELens and Gemma Scope names imported SAEs carry (hook_mlp_out, blocks.3.hook_attn_out,
    att, hook_z). A row with no hook recorded is taken as residual, as every consumer always
    took it.
    """
    kind = classify_hook(hook_type)
    if kind in (KIND_MLP, KIND_ATTENTION):
        label = "an MLP-side" if kind == KIND_MLP else "an attention-output"
        return (
            consumer + " supports residual-stream SAEs only, and this is " + label + " SAE (hook "
            + repr(hook_type) + "): it reads or labels every SAE as the layer's residual output, "
            "so the result would describe the wrong activations."
        )
    where = _BEFORE_FIRST_LAYER if kind == KIND_EMBEDDING else _BEFORE_OUTPUT.get(
        kind[len("resid_"):] if kind in (KIND_RESID_PRE, KIND_RESID_MID) else ""
    )
    if where:
        return (
            consumer + " reads every SAE at its layer's output (resid_post), and this SAE "
            "(hook " + repr(hook_type) + ") reads the residual stream " + where
            + ": the result would describe another point in the model."
        )
    return None


def classify_hook(hook_type: Optional[str]) -> Optional[str]:
    """The kind of point ``hook_type`` names, by the rule the refusals apply; None when unrecorded.

    One of ``residual`` (the layer's output, and any name the rule does not recognise, as
    every consumer always took it), ``mlp``, ``attention``, ``embedding``, ``resid_pre``,
    ``resid_mid``.
    """
    if not hook_type:
        return None
    tokens = _tokens(hook_type)
    feed_forward = any(a == "feed" and b == "forward" for a, b in zip(tokens, tokens[1:]))
    if _MLP_TOKENS.intersection(tokens) or ("ln2" in tokens and "normalized" in tokens) or feed_forward:
        return KIND_MLP
    if _ATTENTION_TOKENS.intersection(tokens):
        return KIND_ATTENTION
    if _EMBEDDING_TOKENS.intersection(tokens):
        return KIND_EMBEDDING
    if _RESIDUAL_TOKENS.intersection(tokens):
        for position in _BEFORE_OUTPUT:
            if position in tokens:
                return "resid_" + position
    return KIND_RESIDUAL
