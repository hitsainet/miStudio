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
    overlap = token_identity_overlap(up_top_tokens, down_top_tokens)
    label_sim, label_method = label_similarity(up_label, down_label)
    prior = weight_prior if weight_prior is not None else 0.0

    votes = {
        "weight_prior": prior >= p_hi,
        "token_overlap": overlap >= o_hi,
        "label_similarity": label_sim >= s_hi,
    }
    signals = {
        "weight_prior": round(prior, 4),
        "token_overlap": round(overlap, 4),
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
