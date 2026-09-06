"""
J-space readout: what a model is poised to say, per layer and per position.

MODEL-AGNOSTIC BY CONSTRUCTION (BR-032, PADR IDL-41). Structure is resolved
through `discover_transformer_structure`, never an architecture whitelist, a
model-name branch, or an upstream fitter's layout detection. miStudio already
deleted its SUPPORTED_ARCHITECTURES whitelist once; this module does not
reintroduce one. There is deliberately no architecture name anywhere in the
executable path.

THE HOOK POINT IS THE DECODER-LAYER OUTPUT, NEVER A NORM MODULE.
`TransformerStructure.residual_norm_module` sounds like the residual stream and
is not. On a hybrid model it resolves to a post-attention RMSNorm, and this
project has already paid for that confusion once: in steering, a vector applied
there was renormalised away and steered output was byte-identical to unsteered
at every dial (see steering_core.py:230, PADR IDL-38). A readout taken at a norm
fails the same way and is HARDER to notice — plausibly-shaped numbers with the
signal scaled out. See `_capture_residuals`.

CPU BY DESIGN. A readout is one matvec per (layer, position) — fractions of a
GFLOP. It gains nothing from CUDA and everything it touches is contended: the
previous logit-lens implementation ran on GPU and failed outright the moment
serving occupied the card. Residual capture is the one GPU-touching step and its
device is chosen explicitly rather than inherited.

NEVER MATERIALISE `W_U J` (BR-006, PADR IDL-42). Token directions are
synthesised on demand. The envelope bound is derived from the loaded model's own
dimensions, never a constant — the required-vs-materialised ratio scales with
vocabulary (~32x at 65k, ~111x at 256k), so a hardcoded bound passes on one
model while missing a real materialisation on another.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from ..schemas.jlens import (
    LayerApplicability,
    LensDoneMessage,
    LensMetaMessage,
    LensTokenMessage,
    LensTypeSlice,
)

logger = logging.getLogger(__name__)

# Readouts run here. Not configurable by accident: see the module docstring.
READOUT_DEVICE = "cpu"

# Config keys that describe per-layer kind on hybrid architectures. Read
# generically — this is a property lookup, not an architecture branch.
_LAYER_TYPE_KEYS = ("layer_types", "layer_type_list", "block_types")

# Substrings that mark a layer as performing attention. Kept as data so a new
# architecture is a list entry rather than a code change.
_ATTENTION_MARKERS = ("attention", "attn")

# Final-norm attribute names across families. Data, not a branch: a new family
# is an entry here, never an `if architecture == ...`.
_FINAL_NORM_NAMES = (
    "norm",
    "ln_f",
    "final_layernorm",
    "final_layer_norm",
    "embedding_norm",
)


def _module_dtype(module: Any) -> Optional[torch.dtype]:
    """The dtype a module's own parameters are stored in, if it has any.

    Needed because a real checkpoint is NOT one dtype: a model can carry a
    bfloat16 norm alongside fp16 activations, and feeding one to the other
    raises rather than promoting.
    """
    try:
        return next(module.parameters()).dtype
    except (StopIteration, AttributeError):
        return None


def _module_device(module: Any) -> Optional[torch.device]:
    """The device a module's own parameters live on, if it has any."""
    try:
        return next(module.parameters()).device
    except (StopIteration, AttributeError):
        return None


def _model_device(model: Any) -> Optional[str]:
    """Where the model's parameters actually are.

    Read off a parameter rather than from `model.device` or a remembered
    argument: a device-mapped model has no single `.device`, and a remembered
    argument is a second claim that can disagree with the truth — which is
    exactly how a CUDA-resident model came to be fed CPU input ids.
    """
    try:
        return str(next(model.parameters()).device)
    except (StopIteration, AttributeError):
        return None


class LensTransport(ABC):
    """Maps a residual-stream activation into readout space.

    THE SINGLE SUBSTITUTION POINT (PADR IDL-40). The logit lens is the
    degenerate case J = I; the Jacobian lens is one extra d_model^2 matvec.
    Consumers never branch on lens type — they hold a transport.
    """

    #: Wire-format lens type this transport produces.
    lens_type: str

    @abstractmethod
    def apply(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Transport one activation. `h` is [d_model]; returns [d_model]."""

    @abstractmethod
    def requires_artifact(self) -> bool:
        ...

    def covers(self, layers: Sequence[int]) -> List[int]:
        """Which of `layers` this transport can actually serve.

        A transport that cannot answer for a layer must say so HERE, before any
        work happens, rather than raising partway through a stream. The logit
        lens covers everything; a Jacobian artifact covers what was fitted, and
        a partial fit is a shape the product offers.
        """
        return list(layers)


class IdentityTransport(LensTransport):
    """Logit lens: J = I. Requires no artifact at all (BR-005).

    This is why the substrate ships before any fitting exists.
    """

    lens_type = "LOGIT_LENS"

    def apply(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        return h

    def requires_artifact(self) -> bool:
        return False


class JacobianTransport(LensTransport):
    """Jacobian lens: one d_model x d_model matrix per analysed layer.

    Holds per-layer matrices on CPU and never forms `W_U J`. A layer absent from
    the artifact raises rather than silently falling back to identity — serving
    logit data under a Jacobian label is prohibited (BR-019 rung discipline).
    """

    lens_type = "JACOBIAN_LENS"

    def __init__(
        self,
        jacobians: Dict[int, torch.Tensor],
        compute_dtype: torch.dtype = torch.float32,
        scales: Optional[Dict[int, float]] = None,
    ):
        """`scales` undoes the fp16 storage rescale (see `_to_storage_dtype`).

        UNSCALING IS NOT COSMETIC. The fitter divides each matrix down so the
        fp16 cast cannot saturate, and the stored matrix is therefore J/alpha.
        Ranked readouts are blind to this — the model's final norm divides a
        positive scalar straight back out — but everything that does not
        normalise is not: probe scores and intervention magnitudes came out
        scaled by an unrecorded per-layer alpha, which made them incomparable
        across layers.

        Absent or 1.0 means no rescale was applied. A missing entry is treated
        as 1.0 rather than refused, because artifacts fitted before the scale
        was recorded are still readable — they simply had no rescale to undo in
        the common case, and a hard refusal would strand them.
        """
        if not jacobians:
            raise ValueError("JacobianTransport requires at least one layer matrix")
        for layer, j in jacobians.items():
            if j.ndim != 2 or j.shape[0] != j.shape[1]:
                raise ValueError(
                    f"J[{layer}] has shape {tuple(j.shape)}; expected a square "
                    "[d_model, d_model] matrix"
                )
        # Cast ONCE at construction, not per call. Artifacts are serialised
        # fp16 and the readout computes in fp32; casting inside apply() would
        # copy a d_model^2 matrix on every (layer, position) — 8 MB per call at
        # d_model 2048, thousands of times per readout.
        # Unscale ONCE here, with the same reasoning as the dtype cast: doing
        # it inside apply() would multiply a d_model^2 matrix on every
        # (layer, position).
        factors = scales or {}
        self._j = {}
        for layer, j in jacobians.items():
            matrix = j.to(compute_dtype)
            alpha = float(factors.get(layer, 1.0) or 1.0)
            if alpha != 1.0:
                matrix = matrix * alpha
            self._j[layer] = matrix
        self._scales = {l: float(factors.get(l, 1.0) or 1.0) for l in jacobians}
        self._compute_dtype = compute_dtype

    def apply(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        j = self._j.get(layer)
        if j is None:
            raise KeyError(
                f"No Jacobian for layer {layer}. Refusing to fall back to "
                "identity: that would serve logit-lens data under a "
                "JACOBIAN_LENS label."
            )
        return j @ h.to(self._compute_dtype)

    def requires_artifact(self) -> bool:
        return True

    def covers(self, layers: Sequence[int]) -> List[int]:
        return [l for l in layers if l in self._j]


@dataclass
class CapturedResiduals:
    """Residual stream per analysed layer: {layer: [positions, d_model]}."""

    by_layer: Dict[int, torch.Tensor]
    hook_target: str  # recorded so a wrong-hook run is diagnosable after the fact


def build_layer_applicability(
    structure: Any, model_config: Any
) -> List[LayerApplicability]:
    """Classify each layer by what is computable there (BR-032).

    Layer kind is per-layer state, not a model property: a hybrid model
    interleaves convolutional and attention layers, so "freeze Q/K" is undefined
    on some of them and attention-broadcast metrics are computable on others.

    Inapplicable is expressed as None, never False. A False would be averaged by
    a downstream consumer and would silently understate; None forces the
    consumer to decide.
    """
    n_layers = int(getattr(structure, "num_layers", 0) or 0)

    layer_types: Optional[Sequence[Any]] = None
    for key in _LAYER_TYPE_KEYS:
        value = getattr(model_config, key, None)
        if isinstance(value, (list, tuple)) and value:
            layer_types = value
            break

    out: List[LayerApplicability] = []
    for i in range(n_layers):
        if layer_types is not None and i < len(layer_types):
            kind = str(layer_types[i]).lower()
            has_attention = any(m in kind for m in _ATTENTION_MARKERS)
        else:
            # Homogeneous model: no per-layer kind published. Every layer
            # attends if the structure discovered an attention module at all.
            has_attention = bool(getattr(structure, "attention_module", None))

        out.append(
            LayerApplicability(
                layer=i,
                has_attention=has_attention,
                # None (absent) when the concept does not apply here.
                frozen_qk_applicable=True if has_attention else None,
                broadcast_metrics_applicable=True if has_attention else None,
            )
        )
    return out


class ReadoutService:
    """Produces wire-format readout streams for any loadable model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        structure: Any,
        unembedding: torch.Tensor,
        model_name: str,
        capture_device: Optional[str] = None,
    ):
        """
        Args:
            unembedding: W_U as [n_vocab, d_model]. Loaded WITHOUT instantiating
                a second copy of the model — see
                analysis_service.load_unembedding_matrix.
            capture_device: device for the forward pass only. The readout itself
                always runs on CPU (READOUT_DEVICE). Defaults to WHERE THE MODEL
                ACTUALLY IS, not to CPU — see below.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.structure = structure
        self.model_name = model_name
        # ASK THE MODEL WHERE IT LIVES. Defaulting to "cpu" is a second,
        # independent claim about the model's device, and the two silently
        # disagreed: the worker's model cache is keyed by model id alone, so a
        # fit that loaded gemma onto CUDA left a CUDA-resident model that the
        # next readout received while still placing input_ids on CPU —
        # "index is on cpu, different from other tensors on cuda:0", raised
        # inside the embedding lookup before any readout math ran.
        #
        # This does NOT put readouts on the GPU: nothing here loads a model onto
        # an accelerator, it only follows one already resident. The readout math
        # stays on READOUT_DEVICE regardless, which is what keeps it off the
        # serving card.
        self.capture_device = capture_device or _model_device(model) or "cpu"

        # W_U stays on CPU with the readout. [n_vocab, d_model].
        #
        # CAST ONCE, here. A real checkpoint mixes dtypes and the matvec needs
        # one family, but casting inside the per-cell hot path allocates a whole
        # new [n_vocab, d_model] tensor EVERY (layer, position) — 2.4 GB per
        # call at a 256k vocabulary. Caught by the envelope guard, which exists
        # for exactly this shape. Same lesson as JacobianTransport casting J
        # once in its constructor.
        self.W_U = unembedding.to(READOUT_DEVICE).to(torch.float32)
        self.n_vocab, self.d_model = self.W_U.shape

        # The model's own final norm, resolved once. Looked up by attribute
        # shape rather than architecture name (BR-032): different families call
        # it norm / ln_f / final_layernorm.
        self._final_norm = self._resolve_final_norm(model, structure)
        if self._final_norm is None:
            # LOUD, because the fallback is invisible in the output. The
            # docstring on `_normalize` has always claimed this was "recorded";
            # nothing recorded it, and a whole gemma-4-12B fit was rejected by
            # the semantic check while the artifact itself was fine. Plain RMS
            # is a scalar divide, so it cannot change a ranking — a readout
            # taken without the learned per-channel gain is not obviously
            # wrong, it is just wrong.
            logger.warning(
                "No final norm found on %s; falling back to plain RMS, which "
                "drops the learned per-channel gain and leaves token rankings "
                "identical to applying no norm at all. Readouts from this "
                "model are NOT trustworthy — add its final-norm attribute name "
                "to _FINAL_NORM_NAMES, or check that the decoder-layer list "
                "was discovered.",
                model_name,
            )

        # Decoded-token cache. A readout decodes the same high-frequency ids at
        # every (layer, position); without this the tokenizer dominates runtime.
        self._decode_cache: Dict[int, str] = {}

    @staticmethod
    def _resolve_final_norm(model: Any, structure: Any = None) -> Optional[Any]:
        """Find the model's final normalisation module, if it exposes one.

        Attribute-name search over a data list, not an architecture branch —
        adding a family is a list entry.

        SEARCH THE DECODER LAYERS' OWN PARENT FIRST. Looking only at `model`
        and `model.model` is an assumption about DEPTH, and a unified
        (text+vision+audio) checkpoint breaks it: gemma-4-12B keeps its stack
        at `model.model.language_model.layers` and its final norm beside them
        at `model.model.language_model.norm`, one level below where this
        looked. It found nothing, and `_normalize` fell back to plain RMS.

        THE FALLBACK CANNOT BE DETECTED BY EYE, which is why this went
        unnoticed through a whole fit. Plain RMS is a SCALAR divide, so it
        leaves the ranking bit-identical to applying no norm at all — the
        readout is not obviously broken, it is merely missing the learned
        per-channel gain that carries most of the signal. Observed on
        gemma-4-12B: layers 42-45 read `𒅘 𒉼 𒈿 𒉘 ...` (rare cuneiform, the
        same tokens for every prompt) without the norm and
        `' Paris', 'Paris', '巴黎', ' París'` with it, from the identical
        residuals and the identical W_U. A 53-minute fit was rejected by the
        semantic check over this, and the artifact was fine.

        The norm is a SIBLING of the decoder-layer list in every family here,
        so the layer list's parent is the principled place to look and needs no
        knowledge of how deeply a tower is nested. The old owners are still
        searched after it, so nothing that resolved before resolves differently.
        """
        owners: list = []
        layers_module = getattr(structure, "layers_module", None)
        if layers_module is not None:
            for module in model.modules():
                if any(child is layers_module for child in module.children()):
                    owners.append(module)
                    break
        owners.extend([getattr(model, "model", None), model])
        for owner in owners:
            if owner is None:
                continue
            for name in _FINAL_NORM_NAMES:
                candidate = getattr(owner, name, None)
                # isinstance(nn.Module), not callable(). `callable` is far too
                # loose here: a BOUND METHOD named `norm` passes it, and so
                # does any other callable attribute that happens to share a
                # name. Silently normalising with the wrong object produces a
                # readout that looks fine and ranks tokens wrongly — the same
                # class of silent failure as hooking the wrong module.
                if isinstance(candidate, torch.nn.Module):
                    return candidate
        return None

    # ---------------------------------------------------------------- capture

    def _capture_residuals(self, input_ids: torch.Tensor, layers: Sequence[int]) -> CapturedResiduals:
        """Capture resid_post at each requested layer.

        HOOK TARGET IS `structure.layers_module[L]` — the decoder layer itself,
        whose output IS the residual stream after that block. NOT
        `structure.residual_norm_module`, which on a hybrid model is a
        post-attention RMSNorm that renormalises the signal away (module
        docstring; PADR IDL-38).
        """
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                # Decoder blocks commonly return a tuple whose first element is
                # the hidden state. Handle both shapes without knowing the
                # architecture.
                hidden = output[0] if isinstance(output, tuple) else output
                # [batch, positions, d_model] -> [positions, d_model], on CPU
                # so the readout never holds device memory.
                captured[layer_idx] = hidden[0].detach().to(READOUT_DEVICE)

            return hook

        try:
            for layer_idx in layers:
                module = self.structure.layers_module[layer_idx]
                handles.append(module.register_forward_hook(make_hook(layer_idx)))

            with torch.no_grad():
                self.model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()

        return CapturedResiduals(
            by_layer=captured,
            hook_target="layers_module[L]",
        )

    # ---------------------------------------------------------------- readout

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the model's OWN final norm before unembedding.

        A plain L2 normalisation is NOT what a transformer does: RMSNorm divides
        by sqrt(mean(x^2)) and then applies a learned per-dimension weight, and
        LayerNorm additionally re-centres. Substituting L2 rescales every logit
        by a per-position constant and drops the learned weighting entirely —
        the readout still looks plausible (it is a monotone rescale for a single
        position) but token RANKINGS shift wherever the learned weight is not
        uniform, which is the whole output of this service.

        Falls back to RMS only when the model exposes no final norm, and records
        that it did so rather than pretending equivalence.
        """
        if self._final_norm is not None:
            with torch.no_grad():
                # CAST TO THE NORM'S OWN DTYPE FIRST.
                #
                # Found on the cluster, not in any fixture: gemma-2-2b-it loads
                # with a bfloat16 final norm while the captured residual and the
                # unembedding arrive as fp16, and the call died with
                # "expected scalar type BFloat16 but found Half". Mixed
                # precision is ordinary on a real checkpoint and never appears
                # in a single-dtype test stack.
                #
                # AND TO THE NORM'S OWN DEVICE, for the same class of reason.
                # The norm is a live module belonging to the model, so it sits
                # wherever the model sits — which is NOT where the readout runs.
                # The readout deliberately works on READOUT_DEVICE while the
                # model may be resident on an accelerator, and calling a CUDA
                # module with a CPU tensor dies with "found at least two
                # devices, cuda:0 and cpu". Borrow the module, then come
                # straight back: the result must land on the readout's device,
                # not the model's, or every downstream matvec inherits the
                # mismatch instead.
                norm_dtype = _module_dtype(self._final_norm) or x.dtype
                norm_device = _module_device(self._final_norm) or x.device
                out = self._final_norm(x.to(device=norm_device, dtype=norm_dtype))
                return out.to(device=x.device, dtype=x.dtype)
        # Documented fallback: RMS without learned weights.
        rms = x.pow(2).mean().sqrt().clamp_min(1e-6)
        return x / rms

    def _rank_at(
        self, h: torch.Tensor, layer: int, transport: LensTransport, top_n: int
    ) -> Tuple[List[str], List[float]]:
        """Full ranked readout at one (layer, position).

        `softmax(W_U . norm(J . h))`, then top-n. One matvec over the
        unembedding call; no n_vocab x d_model array is ever formed.
        """
        transported = transport.apply(h.to(READOUT_DEVICE), layer)
        normed = self._normalize(transported)

        # ONE COMPUTE DTYPE for the matvec, for the same reason: a real
        # checkpoint mixes families and torch raises rather than promoting.
        # fp32 also keeps ranking stable — the logit gaps that decide order are
        # small relative to fp16 precision near the top of the distribution.
        logits = self.W_U @ normed.to(self.W_U.dtype)  # [n_vocab]
        probs = torch.softmax(logits, dim=-1)

        k = min(top_n, probs.numel())
        top = torch.topk(probs, k)
        # Batch-decode: one call instead of top_n calls per (layer, position).
        # At 16 layers x 20 positions x 8 that is 2,560 decode calls collapsed
        # to 320.
        tokens = self._decode_batch([int(i) for i in top.indices])
        return tokens, [float(p) for p in top.values]

    def _decode_batch(self, ids: Sequence[int]) -> List[str]:
        """Decode ids individually but through one cached path.

        Each id must decode on its own — decoding the list as a sequence would
        merge sub-word pieces into a single string and lose the per-rank
        alignment the wire format requires.
        """
        out: List[str] = []
        for i in ids:
            cached = self._decode_cache.get(i)
            if cached is None:
                cached = self.tokenizer.decode([i])
                self._decode_cache[i] = cached
            out.append(cached)
        return out

    def stream(
        self,
        prompt: str,
        transports: Sequence[LensTransport],
        layers: Optional[Sequence[int]] = None,
        top_n: int = 8,
    ) -> Iterator[Any]:
        """Yield meta -> token* -> done in the upstream wire format (BR-029)."""
        if not transports:
            raise ValueError("at least one transport is required")

        n_layers = int(self.structure.num_layers)
        selected = list(layers) if layers is not None else list(range(n_layers))
        out_of_range = [l for l in selected if l < 0 or l >= n_layers]
        if out_of_range:
            raise ValueError(
                f"layers {out_of_range} outside range 0..{n_layers - 1}"
            )

        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.capture_device)
        ids = [int(i) for i in input_ids[0]]

        # Bound the PRODUCT before doing any work. Per-field limits do not:
        # see check_readout_budget.
        check_readout_budget(len(ids), len(selected), self.d_model)

        # EACH TYPE GETS ITS OWN LAYER LIST. `layers_by_type` is per lens type
        # in the wire format precisely because the types can differ, and giving
        # both the same list made the Jacobian lens unusable on any PARTIAL
        # artifact: the panel sends no explicit layers, the server defaulted to
        # every layer, and the transport refused the first one it lacked —
        # "No Jacobian for layer 0" on a 9-of-16-layer fit. The refusal was
        # right; asking was not.
        #
        # A type that covers NOTHING in the requested range is dropped entirely
        # rather than emitted with an empty axis: an empty slice renders as a
        # lens that found nothing, which is the confusion this whole feature
        # exists to prevent.
        covered = [(t, t.covers(selected)) for t in transports]
        servable = [(t, ls) for t, ls in covered if ls]
        if not servable:
            missing = ", ".join(t.lens_type for t, _ in covered)
            raise ValueError(
                f"none of the requested lenses ({missing}) covers any of layers "
                f"{selected}. A Jacobian lens covers the layers it was fitted "
                "for; fit the missing layers or narrow the request."
            )

        # Capture only what some transport will actually read.
        needed = sorted({l for _, ls in servable for l in ls})
        residuals = self._capture_residuals(input_ids, needed)

        applicability = build_layer_applicability(
            self.structure, getattr(self.model, "config", None)
        )

        yield LensMetaMessage(
            model=self.model_name,
            types=[t.lens_type for t, _ in servable],
            layers_by_type={t.lens_type: list(ls) for t, ls in servable},
            top_n=top_n,
            prompt_len=len(ids),
            layer_applicability=applicability,
        )

        for position, token_id in enumerate(ids):
            slices: List[LensTypeSlice] = []
            for transport, transport_layers in servable:
                per_layer_tokens: List[List[str]] = []
                per_layer_probs: List[List[float]] = []
                for layer in transport_layers:
                    h = residuals.by_layer[layer][position]
                    toks, probs = self._rank_at(h, layer, transport, top_n)
                    per_layer_tokens.append(toks)
                    per_layer_probs.append(probs)
                slices.append(
                    LensTypeSlice(
                        type=transport.lens_type,
                        top_tokens=per_layer_tokens,
                        top_probs=per_layer_probs,
                    )
                )

            yield LensTokenMessage(
                position=position,
                token=self.tokenizer.decode([token_id]),
                id=token_id,
                is_generated=False,
                results=slices,
            )

        yield LensDoneMessage()

    # ------------------------------------------------------------------ probe

    def probe(
        self,
        h: torch.Tensor,
        layer: int,
        token_strings: Sequence[str],
        transport: LensTransport,
    ) -> Dict[str, float]:
        """Score an activation against named directions without ranking (BR-008).

        A token's lens direction is row t of `W_U J` — synthesised on demand as
        `W_U[t,:] @ J`, one vector-matrix product. The dictionary is never
        formed.

        NOTE the probe score and the full-ranking position can disagree, because
        the ranking applies a data-dependent normalisation this does not. Which
        mode is canonical must be recorded per analysis (BR-008).
        """
        transported = transport.apply(h.to(READOUT_DEVICE), layer)
        scores: Dict[str, float] = {}
        for s in token_strings:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if not ids:
                continue
            direction = self.W_U[ids[0]]          # [d_model]
            scores[s] = float(direction @ transported)
        return scores


class ReadoutTooLarge(ValueError):
    """A request whose COST exceeds the envelope (BR-028).

    Distinct from a malformed request: the shape is valid, the work is not.
    BR-028 requires operations that cannot fit the envelope to fail with a
    stated reason rather than degrade silently.
    """


# Cost ceilings for a single readout. These bound the PRODUCT, which per-field
# limits do not: prompt<=8000 chars, layers<=512 and top_n<=100 are each
# reasonable and together permit ~102 million ranked readouts holding ~8.4 GB
# of residuals — found in review round 2, after round 1 had bounded the fields
# individually and considered the matter closed.
MAX_READOUT_CELLS = 200_000        # positions x layers
MAX_RESIDUAL_BYTES = 512 * 1024 * 1024


def check_readout_budget(
    n_positions: int, n_layers: int, d_model: int, dtype_bytes: int = 4
) -> None:
    """Refuse a request whose cost exceeds the envelope, with the numbers.

    Called BEFORE capture, so an oversized request costs nothing rather than
    OOMing partway through a forward pass.
    """
    cells = n_positions * n_layers
    if cells > MAX_READOUT_CELLS:
        raise ReadoutTooLarge(
            f"readout would compute {cells:,} (position x layer) cells, over "
            f"the {MAX_READOUT_CELLS:,} limit. Narrow the layer selection or "
            "shorten the prompt."
        )

    resid = n_positions * n_layers * d_model * dtype_bytes
    if resid > MAX_RESIDUAL_BYTES:
        raise ReadoutTooLarge(
            f"readout would hold {resid / 1e9:.2f} GB of residuals, over the "
            f"{MAX_RESIDUAL_BYTES / 1e9:.2f} GB limit. Narrow the layer "
            "selection or shorten the prompt."
        )


@contextmanager
def jlens_budget_override(
    max_cells: Optional[int] = None, max_residual_bytes: Optional[int] = None
):
    """Temporarily tighten (or relax) the readout budget.

    Exists so a test can exercise the refusal path without allocating the
    gigabytes the real ceiling permits. Restores the previous values on exit,
    including when the body raises.
    """
    global MAX_READOUT_CELLS, MAX_RESIDUAL_BYTES
    prev_cells, prev_bytes = MAX_READOUT_CELLS, MAX_RESIDUAL_BYTES
    if max_cells is not None:
        MAX_READOUT_CELLS = max_cells
    if max_residual_bytes is not None:
        MAX_RESIDUAL_BYTES = max_residual_bytes
    try:
        yield
    finally:
        MAX_READOUT_CELLS, MAX_RESIDUAL_BYTES = prev_cells, prev_bytes


def envelope_bound_bytes(
    d_model: int, n_layers: int, dtype_bytes: int = 2, tolerance: float = 1.5
) -> int:
    """Maximum acceptable artifact size for THIS model (BR-006).

    Derived from the model's own dimensions, never a constant. The
    required-vs-materialised ratio scales with vocabulary — about 32x at a 65k
    vocabulary, about 111x at 256k — so a bound hardcoded for one model passes
    on another while missing a real materialisation.
    """
    return int(d_model * d_model * dtype_bytes * n_layers * tolerance)
