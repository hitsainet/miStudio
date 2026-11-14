"""
Edge-type classifier (Feature 018, BR-020/BR-021, CIRCUITS-002 A.9).

Types every cross-layer edge as COMPUTED, PERSISTENCE, or (when Tier-2.5
evidence exists) ATTENTION_MEDIATED, with the classification signals
disclosed per edge. The direct residual weight prior lives HERE — as an
echo-detector input — and must never act as a standalone ranking booster
(BR-020); ranking consumes only the derived `distinctness` value.

Persistence rule (any 2 of 3):
  - weight prior >= P_HI            (direct residual-path alignment)
  - token-identity overlap >= O_HI  (top-activating contexts share tokens)
  - label similarity >= S_HI        (labels describe the same concept)

Low prior + high association is the EXPECTED signature of MLP-mediated
computation — computed edges are never penalized for a low prior.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

P_HI_DEFAULT = 0.9
O_HI_DEFAULT = 0.5  # calibrated on the audit fixture: top-token sets sharing half their union is strong echo evidence
S_HI_DEFAULT = 0.85


def token_identity_overlap(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Jaccard overlap of normalized top-token sets (empty ⇒ 0.0)."""
    norm = lambda ts: {t.strip().lower() for t in ts if t and t.strip()}
    a, b = norm(tokens_a), norm(tokens_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def measured_token_overlap(
    tokens_a: List[str], tokens_b: List[str]
) -> Tuple[Optional[float], str]:
    """Token-identity overlap AND whether it was measured at all.

    `token_identity_overlap` answers 0.0 when either side has no top tokens,
    which is indistinguishable from "both sides were read and share nothing".
    Live 2026-10-20 (tracker 3.4 A1/A2): every endpoint of 34 mined edges was
    unlabelled with no `example_tokens_summary`, and every edge recorded
    `token_overlap: 0.0` — a measured-looking zero on evidence nobody had.
    `(None, "absent")` says the signal was not measured; `"jaccard"` that it was.
    """
    norm = lambda ts: {t.strip().lower() for t in ts if t and t.strip()}
    if not norm(tokens_a) or not norm(tokens_b):
        return None, "absent"
    return token_identity_overlap(tokens_a, tokens_b), "jaccard"


def label_similarity(label_a: Optional[str], label_b: Optional[str]) -> Tuple[float, str]:
    """Label-concept similarity with an honest degradation ladder.

    Preferred: an embedding stack (not wired in v1 — recorded). Fallback:
    normalized token-set overlap of the labels themselves, marked as the
    weaker 'token_set' method so the disclosure stays honest (018 FTID §2.5).
    """
    if not label_a or not label_b:
        return 0.0, "absent"
    tok = lambda s: set(re.split(r"[_\W]+", s.lower())) - {""}
    a, b = tok(label_a), tok(label_b)
    if not a or not b:
        return 0.0, "absent"
    return len(a & b) / len(a | b), "token_set"


def classify_edge(
    *,
    weight_prior: Optional[float],
    up_top_tokens: List[str],
    down_top_tokens: List[str],
    up_label: Optional[str],
    down_label: Optional[str],
    mediating_heads: Optional[List[int]] = None,
    p_hi: float = P_HI_DEFAULT,
    o_hi: float = O_HI_DEFAULT,
    s_hi: float = S_HI_DEFAULT,
) -> Dict:
    """Classify one edge; returns {type, signals} with full disclosure.

    When the label-similarity method degrades to token_set (no embedding
    stack), the persistence rule requires BOTH remaining strong signals
    (prior + token overlap) unless the token_set label signal also fires —
    i.e. the 2-of-3 rule counts the degraded signal only when it is positive
    evidence, never as an abstention.
    """
    # Unmeasured (no top tokens on a side) is disclosed as None with
    # token_method "absent", and cannot vote — the same rule as the prior below.
    overlap, token_method = measured_token_overlap(up_top_tokens, down_top_tokens)
    label_sim, label_method = label_similarity(up_label, down_label)
    # An unmeasured prior cannot vote, and it is disclosed as unmeasured (None), not
    # as a measured 0.0: "the weights say these are unrelated" and "nobody looked"
    # are different claims and a reader of the signals must be able to tell them apart.
    prior = weight_prior if weight_prior is not None else 0.0

    votes = {
        "weight_prior": prior >= p_hi,
        "token_overlap": overlap is not None and overlap >= o_hi,
        "label_similarity": label_sim >= s_hi,
    }
    signals = {
        "weight_prior": round(prior, 4) if weight_prior is not None else None,
        "token_overlap": round(overlap, 4) if overlap is not None else None,
        "token_method": token_method,
        "label_similarity": round(label_sim, 4),
        "label_method": label_method,
        "thresholds": {"p_hi": p_hi, "o_hi": o_hi, "s_hi": s_hi},
        "votes": {k: bool(v) for k, v in votes.items()},
    }

    if mediating_heads:
        edge_type = "attention_mediated"
    elif sum(votes.values()) >= 2:
        edge_type = "persistence"
    else:
        edge_type = "computed"

    # Ranking never consumes the prior directly (BR-020): distinctness is
    # the only ranking-facing output. COMPUTED edges are never penalized —
    # distinctness is 1.0 by definition (a lone strong signal, including a
    # high prior, must not de-rank a computed edge; review R1 finding #3).
    # PERSISTENCE edges grade by vote count (2-of-3 → 1/3 left, 3-of-3 → 0),
    # so stronger echoes de-rank harder and 2- vs 3-vote echoes are
    # distinguishable.
    if edge_type == "persistence":
        echo_confidence = sum(votes.values()) / 3.0
    else:
        echo_confidence = 0.0
    signals["echo_confidence"] = round(min(1.0, echo_confidence), 4)
    signals["distinctness"] = round(1.0 - signals["echo_confidence"], 4)

    return {"type": edge_type, "signals": signals}


# ── Wiring: typing a built circuit's edges (018 6.1, 2026-10-20) ──────────────
#
# Until 2026-10-20 nothing in production called `classify_edge`. Every edge
# `CircuitService.from_candidates` built took the schema default `type: "computed"`
# with `type_signals: null`, so the EdgeTable's type column, the persistence
# filter and the list's edge-type filter all read a value no classifier had
# produced. The helpers below are what `from_candidates` now calls.


def parse_top_tokens(summary) -> List[str]:
    """`Feature.example_tokens_summary` as a token list.

    Labeling writes it as one comma-joined string ("▁worth, ▁entries"); older rows
    hold a list. SentencePiece/BPE word markers are dropped so "▁fear" and "fear"
    compare equal, which is what the audit fixture's plain tokens assume.
    """
    if not summary:
        return []
    items = summary if isinstance(summary, list) else str(summary).split(",")
    out = []
    for t in items:
        tok = str(t).strip().lstrip("\u2581\u0120").strip()
        if tok:
            out.append(tok)
    return out


def type_circuit_edges(
    edges: List[Dict],
    *,
    features: Dict[Tuple[int, int], Dict],
    priors: Dict[Tuple[int, int, int, int], float],
) -> Dict[str, int]:
    """Classify every feature-to-feature edge in place; return the type counts.

    `features[(layer, idx)]` = {"label": str|None, "top_tokens": [str]} (absent ⇒
    no label, no tokens). `priors[(up_layer, up_idx, down_layer, down_idx)]` = the
    measured weight prior (absent ⇒ unmeasured, disclosed as None). An edge with a
    cluster endpoint is left at its type and says why in `type_signals`: the
    classifier's three signals are defined on single features.
    """
    counts: Dict[str, int] = {}
    for edge in edges:
        up, down = edge.get("up") or {}, edge.get("down") or {}
        ui, di = up.get("feature_idx"), down.get("feature_idx")
        if ui is None or di is None:
            edge["type_signals"] = {
                "classified": False,
                "reason": "cluster endpoint: the edge-type classifier types feature pairs",
            }
            counts["unclassified"] = counts.get("unclassified", 0) + 1
            continue
        uf = features.get((up["layer"], ui)) or {}
        df = features.get((down["layer"], di)) or {}
        prior = priors.get((up["layer"], ui, down["layer"], di))
        result = classify_edge(
            weight_prior=prior,
            up_top_tokens=uf.get("top_tokens") or [],
            down_top_tokens=df.get("top_tokens") or [],
            up_label=uf.get("label"),
            down_label=df.get("label"),
            mediating_heads=(edge.get("position") or {}).get("mediating_heads"),
        )
        edge["type"] = result["type"]
        edge["type_signals"] = result["signals"]
        if prior is not None:
            edge["weight_prior"] = round(float(prior), 4)
        counts[result["type"]] = counts.get(result["type"], 0) + 1
    return counts
