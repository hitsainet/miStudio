"""
Cross-layer steering hazards (Feature 015, BR-004/BR-024 — IDL-32/IDL-35).

When a multi-layer circuit steers an UPSTREAM feature that drives a
DOWNSTREAM steered feature, their influences COMPOUND (or, with opposite
signs, CANCEL). We surface this — WARNED, never silently corrected (BR-004).

Two evidence sources, in priority order:
  PRIMARY   — a stored circuit edge at rung >= 2 (017-validated): the warning
              is QUANTIFIED from the edge's measured effect size
              ("validated edge, ES=X — combined influence ≈ Y× the naive sum").
  FALLBACK  — the IDL-32 weight prior cos(W_dec(Lᵢ)[:,i], W_enc(Lⱼ)[j,:])
              above a threshold; EVERY such warning is labeled `heuristic`
              per the evidence-ladder language rules (IDL-35) — never causal.

Pure functions over weight tensors + edge dicts — no GPU orchestration, no
DB — so the whole hazard matrix is exhaustively unit-testable.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PRIOR_THRESHOLD = 0.5  # config steering_hazard_prior_threshold


@dataclass
class Hazard:
    type: str            # "compounding" | "cancellation"
    up: Dict[str, int]   # {layer, feature_idx}
    down: Dict[str, int]
    evidence: str        # source label — "validated:ES=…" | "heuristic:weight_prior=…"
    rung: int            # the edge's rung (0 for a pure heuristic pair)
    quantified_effect: Optional[float] = None  # ES for validated edges
    #: The effect size was measured on a CLUSTER-level edge and inherited by
    #: this feature pair, not measured on this pair.
    #:
    #: A supernode's activation is `A_C(t) = max_k a_{l,i_k}(t)` (Appendix A.4),
    #: so a cluster edge's ES was measured on a signal that at any token is ONE
    #: member's activation — whichever was carrying the cluster. Resolving that
    #: edge to feature membership at steering time is what A.4 prescribes, and
    #: it is what `expand_cluster_edges` does, but the resulting number belongs
    #: to the cluster pair and not to any particular member pair.
    #:
    #: Without this flag such a hazard rendered as `validated:ES=0.800`, to
    #: three decimals, indistinguishable from an effect size measured on that
    #: exact pair. This module already separates measured from heuristic; the
    #: third case is measured-HERE from inherited. A.4's own answer for "which
    #: member pairs carry it" is to MEASURE them — the A.3 drill-down restricted
    #: to the two memberships — never to apportion the cluster number.
    inherited_from_cluster_edge: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "up": self.up, "down": self.down,
                "evidence": self.evidence, "rung": self.rung,
                "quantified_effect": self.quantified_effect,
                "inherited_from_cluster_edge": self.inherited_from_cluster_edge}


def weight_prior(up_decoder, up_idx: int, down_encoder, down_idx: int) -> float:
    """cos(W_dec(Lᵢ)[:, i], W_enc(Lⱼ)[j, :]) — the IDL-32 prior. Orientation:
    decoder is [d_model, d_sae] (column i), encoder is [d_sae, d_model] (row j)
    — both project the same d_model space, so the cosine is well-defined.
    Reuses the resolve_*_weight conventions (add resolve_encoder_weight beside
    resolve_decoder_weight). Out-of-range indices ⇒ 0.0 (no hazard), never an
    IndexError — detect_hazards is a public pure function (R1 #5)."""
    import torch

    if not (0 <= up_idx < up_decoder.shape[1]) or \
            not (0 <= down_idx < down_encoder.shape[0]):
        return 0.0
    d_i = up_decoder[:, up_idx]        # [d_model]
    e_j = down_encoder[down_idx, :]    # [d_model]
    return float(torch.nn.functional.cosine_similarity(
        d_i.float(), e_j.float(), dim=0))


def _edge_key(e: Dict[str, Any]) -> Tuple:
    up, down = e.get("up", {}), e.get("down", {})
    return (up.get("layer"), up.get("feature_idx"),
            down.get("layer"), down.get("feature_idx"))


def expand_cluster_edges(
    circuit_edges: Optional[List[Dict[str, Any]]],
    resolve_cluster,
    keep: Optional[Set[Tuple[int, int]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Cluster-level edges → the feature-level edges they stand for.

    Returns ``(edges, unresolved)``.

    BR-016 makes a cluster a first-class circuit member (`member_kind:
    cluster_ref`), so a circuit's edges can be cluster-level — and those are the
    edges most worth having, because an edge that reached rung 2 is a MEASURED
    effect size rather than the weight-prior heuristic. `detect_hazards` dropped
    every one of them: its keys are `(layer, feature_idx, …)` and a cluster-ref
    endpoint has no `feature_idx`, so the key could never match and the edge was
    skipped. The result was silent and backwards — steering a cluster-membered
    circuit discarded its best evidence and fell back to the heuristic, or to
    nothing, while reporting an empty hazard list either way.

    A cluster edge asserts that the upstream cluster drives the downstream one,
    so it stands for every (upstream feature → downstream feature) pair across
    the two memberships, and each inherits the edge's rung and effect size. That
    inheritance is the honest reading of a supernode edge and it is also the
    conservative one: it can only ADD warnings.

    BOUNDED BY `keep`, which is the set of `(layer, feature_idx)` actually being
    steered. Two twenty-feature clusters would otherwise expand to four hundred
    edges of which at most a handful are reachable, and `detect_hazards` only
    ever looks up pairs drawn from the steered list. Filtering to that set is
    exactly equivalent and keeps the product small; passing `keep=None` expands
    everything, which is what the tests do.

    UNRESOLVABLE PROFILES ARE RETURNED, NOT DROPPED. A deleted or empty cluster
    profile means an edge that cannot be checked, and the caller has to be able
    to say so — "no hazards" and "not analysed" are different claims, and this
    project has already been bitten by a UI where an empty list read as safety.
    """
    edges = list(circuit_edges or [])
    if not edges:
        return [], []

    out: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    cache: Dict[Any, List[int]] = {}

    def _members(ref: Dict[str, Any]) -> Optional[List[int]]:
        pid = ref.get("cluster_profile_id")
        if pid is None:
            return None
        if pid not in cache:
            try:
                cache[pid] = [int(i) for i in (resolve_cluster(pid) or [])]
            except Exception:  # noqa: BLE001 - an edge must not fail allocation
                cache[pid] = []
        return cache[pid]

    def _side(ref: Dict[str, Any]) -> Tuple[Optional[List[int]], bool]:
        """(feature indices, was_a_cluster)."""
        if ref.get("feature_idx") is not None:
            return [int(ref["feature_idx"])], False
        return _members(ref), True

    for e in edges:
        up, down = e.get("up") or {}, e.get("down") or {}
        up_idx, up_was_cluster = _side(up)
        down_idx, down_was_cluster = _side(down)

        if not up_was_cluster and not down_was_cluster:
            out.append(e)          # already feature-level; untouched
            continue

        if not up_idx or not down_idx:
            unresolved.append({
                "up": dict(up),
                "down": dict(down),
                "reason": (
                    "a cluster endpoint resolved to no features — the profile is "
                    "missing or empty, so this edge could not be checked"
                ),
            })
            continue

        up_layer, down_layer = up.get("layer"), down.get("layer")
        for ui in up_idx:
            if keep is not None and (up_layer, ui) not in keep:
                continue
            for di in down_idx:
                if keep is not None and (down_layer, di) not in keep:
                    continue
                expanded = dict(e)
                expanded["up"] = {**up, "feature_idx": ui}
                expanded["down"] = {**down, "feature_idx": di}
                # Kept so a consumer can tell an inherited effect size from one
                # measured on this feature pair directly.
                expanded["expanded_from_cluster_edge"] = True
                out.append(expanded)

    return out, unresolved


def detect_hazards(
    steered: List[Dict[str, int]],
    *,
    circuit_edges: Optional[List[Dict[str, Any]]] = None,
    decoders: Optional[Dict[int, Any]] = None,
    encoders: Optional[Dict[int, Any]] = None,
    prior_threshold: float = DEFAULT_PRIOR_THRESHOLD,
) -> List[Hazard]:
    """Surface compounding/cancellation across co-steered members.

    `steered` = [{layer, feature_idx, strength}] — the members being steered.
    For each upstream→downstream pair (up.layer < down.layer):
      • if a stored edge (up→down) at rung >= 2 exists, QUANTIFY from its ES;
      • else, if the weight prior >= threshold, warn labeled `heuristic`.
    Sign of the pair's steering strengths decides compounding vs cancellation.
    """
    edges_by_key = {}
    for e in (circuit_edges or []):
        k = _edge_key(e)
        # A cluster-ref endpoint still has no `feature_idx`, so its key can
        # never match the feature-level steered list. Callers run
        # `expand_cluster_edges` first, which turns those into the feature pairs
        # they stand for; anything still unexpanded here is genuinely unusable.
        if None in k:
            continue
        edges_by_key[k] = e

    # Feature-level members only (a cluster-ref member has no feature_idx).
    feats = [m for m in steered if m.get("feature_idx") is not None]

    hazards: List[Hazard] = []
    seen_pairs = set()  # dedup (R1 #6): duplicate members must not double-warn
    for up in feats:
        for down in feats:
            if up["layer"] >= down["layer"]:
                continue
            pair = (up["layer"], up["feature_idx"], down["layer"], down["feature_idx"])
            if pair in seen_pairs:
                continue
            up_ref = {"layer": up["layer"], "feature_idx": up["feature_idx"]}
            down_ref = {"layer": down["layer"], "feature_idx": down["feature_idx"]}
            # co-steered sign: same sign ⇒ compounding, opposite ⇒ cancellation
            same_sign = (up.get("strength", 0) >= 0) == (down.get("strength", 0) >= 0)
            key = pair

            edge = edges_by_key.get(key)
            if edge is not None and int(edge.get("rung", 0)) >= 2 \
                    and edge.get("effect_size") is not None:
                es = float(edge["effect_size"])
                # a validated NEGATIVE edge flips the compounding/cancellation
                edge_positive = es >= 0
                compounding = same_sign == edge_positive
                inherited = bool(edge.get("expanded_from_cluster_edge"))
                # SAY WHICH PAIR THE NUMBER BELONGS TO. `expand_cluster_edges`
                # marks the edges it resolved out of a cluster edge, and that
                # mark used to die here: the Hazard was built without it, so an
                # ES measured on a supernode pair reached the caller wearing the
                # same label as one measured on this feature pair.
                label = (
                    f"validated:cluster-ES={es:.3f} (inherited from a "
                    "cluster-level edge; not measured on this feature pair)"
                    if inherited
                    else f"validated:ES={es:.3f}"
                )
                hazards.append(Hazard(
                    type="compounding" if compounding else "cancellation",
                    up=up_ref, down=down_ref,
                    evidence=label,
                    rung=int(edge.get("rung", 2)),
                    quantified_effect=es,
                    inherited_from_cluster_edge=inherited))
                seen_pairs.add(pair)
                continue

            # heuristic fallback — weight prior (labeled, never causal)
            if decoders is not None and encoders is not None:
                dec = decoders.get(up["layer"])
                enc = encoders.get(down["layer"])
                if dec is not None and enc is not None:
                    prior = weight_prior(dec, up["feature_idx"], enc,
                                         down["feature_idx"])
                    if abs(prior) >= prior_threshold:
                        # prior sign + steering sign → compounding/cancellation
                        prior_positive = prior >= 0
                        compounding = same_sign == prior_positive
                        hazards.append(Hazard(
                            type="compounding" if compounding else "cancellation",
                            up=up_ref, down=down_ref,
                            evidence=f"heuristic:weight_prior={prior:.3f}",
                            rung=0, quantified_effect=None))
                        seen_pairs.add(pair)
    return hazards
