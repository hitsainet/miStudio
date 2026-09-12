"""
Cluster strength allocation (Feature 013, IDL-29).

Computes a principled starting strength allocation for steering a CLUSTER of
SAE features, replacing guessed uniform strengths. Design (pressure-tested; see
0xcc/tdds/013_FTDD|Cluster_Strength_Model.md for the derivation):

    weights      w_i   = s~_i / Σ s~_j          (similarity-normalized; equal sims ⇒ equal shares)
    frequency    f_eff = Σ w_i f_i / Σ w_i      (over members with known f)
    dir. budget  B_dir = clamp(a − b·f_eff, m, M)   (the empirically-fit solo law at cluster level)
    gain         G     = ‖Σ σ_i w_i d_i‖₂       (exact resultant norm of the injected direction)
    total budget B     = min( B_dir / max(G, G_FLOOR)**γ, Σ b*(f_i) )   (fitted γ=0 ⇒ B=B_dir)
    strengths    s_i   = σ_i · B · w_i          (rounded to 0.1; residual → largest member)

The hook injects v = Σ strength_i · d_i, so ‖v‖ = B·G; setting B = B_dir/G makes
the *injected vector* match the validated solo magnitude. Identical members ⇒
G=1 ⇒ B=B_dir (the constant-budget heuristic MCP experiments discovered);
orthogonal members ⇒ B = B_dir·√N. Small G means members nearly cancel — that
degrades direction QUALITY, so guards are coherence flags, not magnitude caps.

The math core is pure (no I/O) for unit testing; decoder access goes through
steering_service.resolve_decoder_weight so the gain sees exactly the directions
the hook will inject.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FORMULA_ID = "freq-budget/sim-alloc@1"
G_FLOOR = 0.05
ROUND_GRAIN = 0.1

# Defaults from the IDL-27 solo-law fit (experiment c4a273f1). Overridable per
# SAE via settings.steering_cluster_constants_json so MCP calibration can write
# results without a code change.
DEFAULT_CONSTANTS: Dict[str, float] = {
    "a": 2.9,       # intercept
    "b": 2.6,       # slope on effective frequency
    "m": 1.0,       # budget floor
    "M": 3.0,       # budget ceiling
    "cohesion_gate": 0.5,  # below this group cohesion → recommend solo baselines
    # Gain exponent γ in B = B_dir / max(G, G_FLOOR)**γ. Empirical validation
    # (2026-07-16, runbook 0xcc/docs/Archive/cluster-strength-validation.md, C1-C3 on
    # sae_eb8374929894): the γ=1 magnitude compensation overdrives ~2× — member
    # features saturate on Σ|s| regardless of decoder-direction cancellation,
    # so the fitted default is γ=0 (B = B_dir). G is still measured, reported,
    # and used for the cancellation flag; γ stays overridable per-SAE.
    "gamma": 0.0,
}

# Cluster-level default when NO member has a known frequency: derived as the
# clamp midpoint of the RESOLVED constants (see _default_b_dir) so per-SAE
# overrides move it. Deliberately NOT the solo path's DEFAULT_STRENGTH=10 —
# that is a per-feature legacy fallback; a cluster of unknowns starts
# conservative. Exception: N=1 with unknown frequency returns the solo default
# (10.0) so the MCP path matches the UI's solo baseline exactly.
DEFAULT_B_DIR = 2.0  # kept for external references/tests; runtime uses _default_b_dir
SOLO_DEFAULT_STRENGTH = 10.0


@dataclass
class AllocationMember:
    feature_idx: int
    layer: int
    similarity: Optional[float] = None
    activation_frequency: Optional[float] = None
    sign: int = 1  # +1 boost, -1 suppress


@dataclass
class MultiLayerMember(AllocationMember):
    """A cluster member carrying the SAE for its own layer (Feature 015).

    Extends the 013 AllocationMember with ``sae_id`` so a multi-layer cluster
    partitions cleanly and each layer's response can report which SAE defined
    its directions.
    """

    sae_id: Optional[str] = None


@dataclass
class AllocationResult:
    B: float
    B_dir: float
    G: float
    f_eff: Optional[float]
    weights: List[float]
    strengths: List[float]
    flags: List[str] = field(default_factory=list)
    cancellation_pair: Optional[Tuple[int, int]] = None  # feature idxs of worst pair
    constants_used: Dict[str, float] = field(default_factory=dict)
    formula_id: str = FORMULA_ID
    approximate: bool = False  # True when decoder unavailable (G := 1)


def resolve_constants(settings_json: Optional[str], sae_id: Optional[str]) -> Dict[str, float]:
    """Merge DEFAULT_CONSTANTS ← config default ← per-SAE override."""
    constants = dict(DEFAULT_CONSTANTS)
    if settings_json:
        try:
            cfg = json.loads(settings_json)
            constants.update(cfg.get("default", {}) or {})
            if sae_id:
                constants.update((cfg.get("per_sae", {}) or {}).get(sae_id, {}) or {})
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"[ClusterAllocation] Bad steering_cluster_constants_json ({e}); using defaults")
    return constants


def _default_b_dir(c: Dict[str, float]) -> float:
    """Default direction budget: the clamp midpoint of the RESOLVED constants.

    Args:
        c: Resolved constants (after per-SAE overrides).

    Returns:
        (m + M) / 2 — moves with per-SAE overrides, unlike a hardcoded value.
    """
    return (c["m"] + c["M"]) / 2.0


def _solo_cap(f: Optional[float], c: Dict[str, float]) -> float:
    """b*(f): the member's solo-optimal contribution to the budget cap.

    Args:
        f: Activation frequency in [0, 1], or None/NaN/out-of-range.
        c: Resolved constants.

    Returns:
        clamp(a − b·f, m, M); the constants midpoint when f is unusable.
    """
    if f is None or not (0.0 <= f <= 1.0) or math.isnan(f):
        return _default_b_dir(c)
    return max(c["m"], min(c["M"], c["a"] - c["b"] * f))


def compute_allocation(
    members: List[AllocationMember],
    decoder=None,  # Optional[torch.Tensor] [d_model, d_sae]; None ⇒ approximate G=1
    constants: Optional[Dict[str, float]] = None,
    group_cohesion: Optional[float] = None,
) -> AllocationResult:
    """
    Pure-math allocation (FTDD §2 steps 1–7 + §3 edge table). No I/O.

    Raises ValueError on an empty member list or mixed layers (the caller maps
    mixed-layer to a refusal + solo-baseline fallback; N=1 is the caller's
    solo-path short-circuit but is still computed correctly here).
    """
    c = dict(constants or DEFAULT_CONSTANTS)
    if not members:
        raise ValueError("members must be non-empty")
    layers = {m.layer for m in members}
    if len(layers) > 1:
        raise ValueError(f"mixed-layer member set {sorted(layers)}: cluster budget model requires a single layer")

    n = len(members)
    flags: List[str] = []

    # N=1 with unknown frequency: return the SOLO default (10.0) so the MCP
    # path matches the UI's Feature-011 baseline exactly (IDL-29 step 10). With
    # a known frequency the general math below already reduces to the solo law.
    if n == 1:
        f0 = members[0].activation_frequency
        if f0 is None or (isinstance(f0, float) and math.isnan(f0)) or not (0.0 <= f0 <= 1.0):
            return AllocationResult(
                B=SOLO_DEFAULT_STRENGTH,
                B_dir=SOLO_DEFAULT_STRENGTH,
                G=1.0,
                f_eff=None,
                weights=[1.0],
                strengths=[members[0].sign * SOLO_DEFAULT_STRENGTH],
                flags=["default_budget", "solo"],
                constants_used={k: (constants or DEFAULT_CONSTANTS).get(k, DEFAULT_CONSTANTS.get(k)) for k in ("a", "b", "m", "M", "cohesion_gate", "gamma")},
                approximate=decoder is None,
            )

    # -- Step 1: similarity-normalized weights -------------------------------
    valid_sims = [m.similarity for m in members
                  if m.similarity is not None and not math.isnan(m.similarity)]
    mean_sim = (sum(valid_sims) / len(valid_sims)) if valid_sims else 1.0
    s_tilde: List[float] = []
    for m in members:
        s = m.similarity
        if s is None or math.isnan(s):
            s = mean_sim  # neutral imputation
        s_tilde.append(max(0.0, s))
    total_s = sum(s_tilde)
    if total_s <= 0:
        weights = [1.0 / n] * n  # all missing/zero → uniform
        flags.append("uniform_weights")
    else:
        weights = [s / total_s for s in s_tilde]
        if any(w == 0 for w in weights):
            flags.append("inactive_member")  # zero-sim member gets zero strength

    # -- Step 2: effective frequency ------------------------------------------
    known = [(w, m.activation_frequency) for w, m in zip(weights, members)
             if m.activation_frequency is not None
             and not math.isnan(m.activation_frequency)
             and 0.0 <= m.activation_frequency <= 1.0]
    if known:
        wsum = sum(w for w, _ in known)
        if wsum > 0:
            f_eff: Optional[float] = sum(w * f for w, f in known) / wsum
        else:
            # Known frequencies exist but all on zero-weight members — use the
            # plain mean rather than falsely claiming no frequencies were known.
            f_eff = sum(f for _, f in known) / len(known)
    else:
        f_eff = None

    # -- Step 3: direction budget ---------------------------------------------
    if f_eff is None:
        b_dir = _default_b_dir(c)  # clamp midpoint of the RESOLVED constants
        flags.append("default_budget")  # no frequencies known
    else:
        b_dir = _solo_cap(f_eff, c)

    # -- Step 4: gain (exact resultant norm, signed weights) -------------------
    approximate = False
    cancellation_pair: Optional[Tuple[int, int]] = None
    if decoder is not None:
        import torch  # local import: math core stays importable without torch

        idxs = [m.feature_idx for m in members]
        d_sae = decoder.shape[1]
        bad = [i for i in idxs if i < 0 or i >= d_sae]
        if bad:
            raise ValueError(f"feature indices out of bounds for SAE with {d_sae} features: {bad}")

        with torch.no_grad():
            # RAW columns, exactly as the steering hook injects them
            # (steering_service hook: `decoder_weight[:, feat_idx]`, no
            # normalization). The solo law was fit THROUGH the hook, so b(f) is
            # the target *injected magnitude*; computing G on normalized
            # directions would mispredict on any non-unit-norm decoder and
            # silently reweight members by their column norms.
            cols = decoder[:, idxs].to(torch.float32)          # [d_model, n]
            norms = torch.linalg.vector_norm(cols, dim=0)
            # Transparency flag when decoder columns deviate from unit norm —
            # the law is extrapolating there (validation protocol covers).
            if bool(((norms - 1.0).abs() > 0.1).any()):
                flags.append("nonunit_decoder")
            sw = torch.tensor(
                [m.sign * w for m, w in zip(members, weights)],
                dtype=torch.float32,
                device=cols.device,
            )
            resultant = cols @ sw                              # [d_model]
            g = float(torch.linalg.vector_norm(resultant))

            # Cancellation check over the POSITIVE-sign subset (explicit
            # suppressors are exempt from triggering it, but must not disable
            # the check for everyone else — FTDD edge row).
            pos = [k for k, m in enumerate(members) if m.sign == 1]
            if len(pos) > 1:
                pw = torch.tensor([weights[k] for k in pos], dtype=torch.float32, device=cols.device)
                pcols = cols[:, pos]
                pnorms = torch.clamp(torch.linalg.vector_norm(pcols, dim=0), min=1e-8)
                punit = pcols / pnorms
                g_pos = float(torch.linalg.vector_norm(punit @ (pw / pw.sum())))
                if g_pos < 1.0 / math.sqrt(len(pos)):
                    flags.append("cancellation")
                    cos = punit.t() @ punit
                    cos.fill_diagonal_(1.0)
                    flat = torch.argmin(cos)
                    i, j = int(flat) // len(pos), int(flat) % len(pos)
                    cancellation_pair = (members[pos[i]].feature_idx, members[pos[j]].feature_idx)
    else:
        g = 1.0  # constant-budget conservative fallback (errs weak)
        approximate = True
        flags.append("approximate")

    # -- Step 5: total budget ---------------------------------------------------
    cap = sum(_solo_cap(m.activation_frequency, c) for m in members)
    gamma = c.get("gamma", DEFAULT_CONSTANTS["gamma"])
    b = min(b_dir / (max(g, G_FLOOR) ** gamma) if gamma else b_dir, cap)
    if b == cap and n > 1:
        flags.append("cap_bound")

    # -- Cohesion gate ------------------------------------------------------------
    if group_cohesion is not None and group_cohesion < c.get("cohesion_gate", 0.5):
        flags.append("low_cohesion")

    # -- Step 6: allocation with SAFE rounding-residual distribution --------------
    # Distribute the residual one grain at a time across unzeroed members in
    # weight order, never crossing zero (a fold that flips a boost member into
    # a suppressor — or dumps the whole budget on one member — is worse than a
    # small Σ|s| deviation; the previous single-member fold did exactly that at
    # n≈20). Grain-limited cases keep their honest rounding and are flagged.
    raw = [m.sign * b * w for m, w in zip(members, weights)]
    strengths = [round(x / ROUND_GRAIN) * ROUND_GRAIN for x in raw]
    residual = b - sum(abs(s) for s in strengths)
    order = sorted(range(n), key=lambda i: -weights[i])
    guard = 0
    while abs(residual) >= ROUND_GRAIN / 2 and guard < 10 * n:
        adjusted = False
        for k in order:
            if residual > 0:
                # grow |s_k| by one grain in its own sign direction
                strengths[k] = round(strengths[k] + members[k].sign * ROUND_GRAIN, 1)
                residual -= ROUND_GRAIN
                adjusted = True
            else:
                # shrink |s_k| by one grain, but never past zero
                if abs(strengths[k]) >= ROUND_GRAIN:
                    strengths[k] = round(strengths[k] - members[k].sign * ROUND_GRAIN, 1)
                    residual += ROUND_GRAIN
                    adjusted = True
            if abs(residual) < ROUND_GRAIN / 2:
                break
            guard += 1
        if not adjusted:
            flags.append("grain_limited")
            break
    strengths = [round(s, 1) for s in strengths]

    return AllocationResult(
        B=round(b, 4),
        B_dir=round(b_dir, 4),
        G=round(g, 4),
        f_eff=round(f_eff, 4) if f_eff is not None else None,
        weights=[round(w, 4) for w in weights],
        strengths=strengths,
        flags=flags,
        cancellation_pair=cancellation_pair,
        constants_used={k: c[k] for k in ("a", "b", "m", "M", "cohesion_gate", "gamma") if k in c},
        approximate=approximate,
    )


MULTI_LAYER_FORMULA_ID = "freq-budget/sim-alloc/per-layer@1"


@dataclass
class MultiLayerAllocationResult:
    """Per-layer allocation for a cluster that spans multiple layers (015).

    ``layers`` maps each layer to its own ``AllocationResult`` (the UNCHANGED
    013 output) plus the ``sae_id`` that layer allocated against. ``strengths``
    is the per-member allocation flattened back into the ORIGINAL request-member
    order (client convenience). ``hazards`` is filled by the endpoint (needs the
    encoder/decoder tensors + optional circuit edges — steering_hazards).
    """

    layers: Dict[int, "AllocationResult"]
    layer_sae_ids: Dict[int, Optional[str]]
    strengths: List[float]
    formula_id: str = MULTI_LAYER_FORMULA_ID


def compute_multi_layer_allocation(
    members: List["MultiLayerMember"],
    decoders: Optional[Dict[int, Any]] = None,
    constants: Optional[Dict[str, float]] = None,
    group_cohesion: Optional[float] = None,
    constants_by_layer: Optional[Dict[int, Dict[str, float]]] = None,
) -> MultiLayerAllocationResult:
    """Allocate strengths across a MULTI-LAYER cluster (Feature 015).

    Partitions ``members`` by layer and runs the EXISTING single-layer
    ``compute_allocation`` on each partition against THAT layer's decoder — the
    IDL-29 math is reused verbatim, never forked. The per-partition results are
    reassembled with each member's strength returned to its original position so
    ``strengths`` mirrors the request order.

    Notes:
      * ``decoders`` maps layer -> decoder tensor ([d_model, d_sae]); a missing
        layer degrades that partition to the approximate (G=1) allocation, same
        as the single-layer endpoint.
      * ``group_cohesion`` (a whole-cluster property) is applied to every
        partition's gate — matching how the single-layer path uses it.
      * A SINGLE-layer member set is still valid here: the caller (endpoint)
        must route single-layer requests to ``compute_allocation`` directly so
        the 013 response is byte-identical; this function is for the >1-layer
        case, but does not itself error on one layer.

    Args:
        members: cluster members carrying ``layer`` and (015) ``sae_id``.
        decoders: {layer -> decoder tensor}; None ⇒ all approximate.
        constants: resolved IDL-29 constants (shared across layers).
        group_cohesion: source cluster cohesion for each partition's gate.

    Returns:
        MultiLayerAllocationResult with per-layer results + flattened strengths.
    """
    if not members:
        raise ValueError("members must be non-empty")

    decoders = decoders or {}

    # Stable partition by layer preserving first-seen layer order, and remember
    # each member's original index so strengths return to request order.
    partitions: Dict[int, List[Tuple[int, "MultiLayerMember"]]] = {}
    layer_order: List[int] = []
    for orig_idx, m in enumerate(members):
        if m.layer not in partitions:
            partitions[m.layer] = []
            layer_order.append(m.layer)
        partitions[m.layer].append((orig_idx, m))

    layers_out: Dict[int, AllocationResult] = {}
    layer_sae_ids: Dict[int, Optional[str]] = {}
    flat_strengths: List[Optional[float]] = [None] * len(members)

    for layer in layer_order:
        entries = partitions[layer]
        part_members = [
            AllocationMember(
                feature_idx=m.feature_idx,
                layer=m.layer,
                similarity=m.similarity,
                activation_frequency=m.activation_frequency,
                sign=m.sign,
            )
            for _, m in entries
        ]
        # The layer's SAE id — first member's (all members of a layer share it
        # by construction: the endpoint validates one SAE per layer).
        layer_sae_ids[layer] = entries[0][1].sae_id

        # Per-layer constants when a layer's SAE has its own override (015 R1
        # ARCH-4 — the endpoint resolved constants_by_layer but passed only the
        # request-level set, so a per-SAE override was silently ignored).
        layer_constants = (constants_by_layer or {}).get(layer, constants)
        result = compute_allocation(
            part_members,
            decoder=decoders.get(layer),
            constants=layer_constants,
            group_cohesion=group_cohesion,
        )
        layers_out[layer] = result

        # Scatter this partition's strengths back to original member positions.
        for (orig_idx, _), s in zip(entries, result.strengths):
            flat_strengths[orig_idx] = s

    # Every slot filled (each member belongs to exactly one partition).
    strengths = [s if s is not None else 0.0 for s in flat_strengths]

    return MultiLayerAllocationResult(
        layers=layers_out,
        layer_sae_ids=layer_sae_ids,
        strengths=strengths,
    )


def rebalance(
    strengths: List[float],
    weights: List[float],
    signs: List[int],
    pinned: List[bool],
    total_budget: float,
) -> Tuple[List[float], List[str]]:
    """
    Budget-preserving rebalance (FTDD step 8) — reference implementation kept
    server-side for parity tests; the frontend mirrors this arithmetic.

    Pinned members keep their values; the remaining budget R = B − Σ|pinned|
    redistributes across unpinned members by renormalized weights. R < 0 ⇒
    warn + unpinned to 0. Never silently rescales pins.
    """
    flags: List[str] = []
    n = len(strengths)
    pinned_total = sum(abs(s) for s, p in zip(strengths, pinned) if p)
    r = total_budget - pinned_total
    out = list(strengths)
    unpinned = [i for i in range(n) if not pinned[i]]
    if not unpinned:
        return out, flags
    if r < 0:
        flags.append("over_budget")
        for i in unpinned:
            out[i] = 0.0
        return out, flags
    wsum = sum(weights[i] for i in unpinned)
    for i in unpinned:
        share = (weights[i] / wsum) if wsum > 0 else 1.0 / len(unpinned)
        out[i] = round(signs[i] * r * share, 1)
    return out, flags
