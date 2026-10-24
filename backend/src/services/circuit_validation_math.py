"""
Circuit validation math (Feature 017, IDL-34 — A.5/A.7 normative).

Pure functions: effect size, shuffled-non-edge null, sign-consistency gate,
survival/uplift, necessity/sufficiency. No GPU, no DB — the GPU services
(intervention/faithfulness) supply the measured Δ arrays and these decide
the verdicts, so every statistical rule is unit-pinned on synthetic inputs.

Rung semantics: an edge reaches rung 2 (causally_validated) iff |ES| exceeds
the shuffled-non-edge null percentile AND sign-consistency ≥ threshold; else
it records tested_and_failed (history, never a demotion — 018's ladder rule).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_NULL_PERCENTILE = 95.0
DEFAULT_SIGN_FRAC = 0.8   # 8/10 prompts must agree in sign
MIN_NULL_SAMPLES = 10     # below this the null is underpowered → cannot validate


@dataclass
class EdgeVerdict:
    effect_size: float
    sign_consistency: float
    null_percentile_value: float
    passed: bool
    reason: str


def effect_size(delta_p: Sequence[float], sigma_d: float) -> float:
    """ES = mean_p(Δ_p) / σ_d — Δ_p is per-prompt (clean − intervened) a_d over
    clean-fire tokens; σ_d is the downstream feature's activation SD from the
    SAME capture store (never a fresh corpus — that silently rescales ES)."""
    dp = np.asarray(delta_p, dtype=np.float64)
    if len(dp) == 0 or sigma_d == 0:
        return 0.0
    return float(dp.mean() / sigma_d)


def sign_consistency(delta_p: Sequence[float]) -> float:
    """Fraction of prompts whose Δ agrees with the mean sign (a genuine causal
    edge pushes the same direction across prompts; noise doesn't)."""
    dp = np.asarray(delta_p, dtype=np.float64)
    if len(dp) == 0:
        return 0.0
    m = np.sign(dp.mean())
    if m == 0:
        return 0.0
    return float((np.sign(dp) == m).mean())


def edge_verdict(delta_p: Sequence[float], sigma_d: float,
                 null_effect_sizes: Sequence[float], *,
                 percentile: float = DEFAULT_NULL_PERCENTILE,
                 sign_frac: float = DEFAULT_SIGN_FRAC) -> EdgeVerdict:
    """rung-2 iff |ES| > null percentile AND sign-consistent ≥ sign_frac."""
    es = effect_size(delta_p, sigma_d)
    sc = sign_consistency(delta_p)
    null = np.asarray(null_effect_sizes, dtype=np.float64)
    # Missing OR underpowered null ⇒ CANNOT validate (R1 #1 / Q1): a missing
    # null used to give thresh=0.0, so any nonzero ES passed on ZERO evidence
    # — a false-positive rung-2 (the worst possible failure in front of AI
    # scientists). Too few null samples ⇒ an unreliable threshold that also
    # lets real edges pass too easily. An unvalidatable edge FAILS.
    if len(null) < MIN_NULL_SAMPLES:
        return EdgeVerdict(
            es, sc, 0.0, False,
            f"null underpowered ({len(null)} < {MIN_NULL_SAMPLES} samples) — "
            f"cannot validate this edge")
    thresh = float(np.percentile(np.abs(null), percentile))
    if abs(es) <= thresh:
        return EdgeVerdict(es, sc, thresh, False,
                           f"|ES|={abs(es):.3f} ≤ null p{percentile}={thresh:.3f}")
    if sc < sign_frac:
        return EdgeVerdict(es, sc, thresh, False,
                           f"sign-consistency {sc:.2f} < {sign_frac}")
    return EdgeVerdict(es, sc, thresh, True,
                       f"|ES|={abs(es):.3f} > null p{percentile}={thresh:.3f}, "
                       f"sign {sc:.2f} ≥ {sign_frac}")


def survival_rate(passed_flags: Sequence[bool]) -> Optional[float]:
    flags = list(passed_flags)
    return (sum(flags) / len(flags)) if flags else None


def uplift(attr_survival: Optional[float],
           coact_survival: Optional[float]) -> Optional[float]:
    """Survival-rate uplift = survival(attr order) − survival(coact order) at
    EQUAL K. None if either tier had nothing to test (016's whole point:
    does attribution re-ranking raise the ablation survival rate?)."""
    if attr_survival is None or coact_survival is None:
        return None
    return round(attr_survival - coact_survival, 4)


# ── faithfulness (A.7) ───────────────────────────────────────────────────

def necessity(b_clean: float, b_ablate_m: float, b_ablate_all: float) -> Optional[float]:
    """(B_clean − B_ablate_M) / (B_clean − B_ablate_all) — how much of the
    total ablatable behavior the circuit's members account for."""
    denom = b_clean - b_ablate_all
    # denom <= 0 (ablating everything didn't lower behavior) ⇒ the ratio
    # inverts sign / explodes — a nonsense "necessity = -3.2" (R1 #13). Not a
    # score: return None.
    if denom <= 0:
        return None
    return round((b_clean - b_ablate_m) / denom, 4)


def sufficiency(b_ablate_nonmembers: float, b_ablate_all: float,
                b_clean: float) -> Optional[float]:
    """(B_ablate_topk_nonmembers − B_ablate_all) / (B_clean − B_ablate_all) —
    how much behavior survives when ONLY non-members are ablated."""
    denom = b_clean - b_ablate_all
    if denom <= 0:  # sign-inversion guard (R1 #13)
        return None
    return round((b_ablate_nonmembers - b_ablate_all) / denom, 4)
