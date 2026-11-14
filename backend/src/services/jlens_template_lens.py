"""Template lens: directions for MULTI-TOKEN concepts (BR-023, 027 task 4.3).

A lens direction is a vocabulary row, so a concept whose name is several tokens has none.
The template lens builds one from FORWARD PASSES ONLY: for each phrase, take contexts in
which the phrase is the natural continuation, average the residual at the FINAL position,
mean-centre against a baseline set, and whiten by the REGULARISED inverse covariance of
the baseline:

    d = (Σ_baseline + λ I)^{-1} (μ_phrase − μ_baseline)

The contexts are the caller's. BR-023 says "generate contexts"; generating them needs a
language model, and a template built from contexts nobody can inspect is not reproducible,
so they are supplied and recorded (their count and a digest) rather than invented here.

THE THREE FAILURE MODES THE SOURCE PAPER DOCUMENTS travel with every result
(`LIMITATIONS`), first-class rather than buried: early layers tend to skip ahead to the
eventual answer; final-layer next-word agreement is unreliable; and a small set of words
appears spuriously across many transcripts — mitigated by the caller's `filter_words`,
which are refused as phrases rather than silently built.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

LIMITATIONS = (
    "early layers tend to skip ahead to the eventual answer, so an early-layer template "
    "direction can encode where the context is GOING rather than the phrase itself",
    "final-layer next-word agreement is unreliable: the last layer's direction can disagree "
    "with the model's own next-token choice",
    "a small set of words appears spuriously across many transcripts; filter them "
    "(`filter_words`) rather than trusting their templates",
)

#: Default ridge, RELATIVE to the mean diagonal of the baseline covariance, so it means
#: the same on every model scale. With fewer baseline contexts than d_model the covariance
#: is singular and the ridge is what makes the inverse exist at all.
DEFAULT_RIDGE = 0.1

#: The fewest baseline contexts a whitening is computed from. Below this the covariance
#: estimate is noise and the "whitened" direction mostly amplifies it.
MIN_BASELINE = 8


@dataclass
class TemplateDirection:
    phrase: str
    layer: int
    direction: torch.Tensor  # [d_model], unit norm
    n_contexts: int
    raw_norm: float          # ||(Σ+λI)^-1 Δμ|| before normalising, for comparison
    separation: float        # Δμ projected on the direction, in baseline std units


def contexts_digest(contexts: Sequence[str]) -> str:
    return hashlib.sha256("\u0000".join(contexts).encode("utf-8")).hexdigest()


def whitening_operator(baseline: torch.Tensor, ridge: float = DEFAULT_RIDGE):
    """(μ_baseline, (Σ + λI)^{-1}) from baseline final-position states [m, d]."""
    if baseline.ndim != 2:
        raise ValueError("baseline states must be [n_contexts, d_model]")
    m, d = baseline.shape
    if m < MIN_BASELINE:
        raise ValueError(
            f"{m} baseline contexts; the whitening needs at least {MIN_BASELINE} — below "
            "that the covariance is noise and the inverse amplifies it"
        )
    x = baseline.to(torch.float64)
    mu = x.mean(dim=0)
    centred = x - mu
    cov = centred.T @ centred / max(1, m - 1)
    lam = ridge * float(torch.diagonal(cov).mean().clamp_min(1e-12))
    inv = torch.linalg.inv(cov + lam * torch.eye(d, dtype=torch.float64))
    return mu, inv, cov, lam


def template_direction(
    phrase: str,
    layer: int,
    phrase_states: torch.Tensor,
    mu_baseline: torch.Tensor,
    inv_cov: torch.Tensor,
    cov: torch.Tensor,
) -> TemplateDirection:
    if phrase_states.ndim != 2 or phrase_states.shape[0] == 0:
        raise ValueError(f"{phrase!r}: no contexts were captured")
    delta = phrase_states.to(torch.float64).mean(dim=0) - mu_baseline
    raw = inv_cov @ delta
    norm = float(raw.norm())
    if norm == 0.0:
        raise ValueError(f"{phrase!r}: its contexts are indistinguishable from the baseline")
    unit = raw / norm
    spread = float((unit @ cov @ unit).clamp_min(1e-24).sqrt())
    return TemplateDirection(
        phrase=phrase,
        layer=layer,
        direction=unit.to(torch.float32),
        n_contexts=int(phrase_states.shape[0]),
        raw_norm=norm,
        separation=float(unit @ delta) / spread,
    )


def build_templates(
    phrase_states: Dict[str, Dict[int, torch.Tensor]],
    baseline_states: Dict[int, torch.Tensor],
    *,
    ridge: float = DEFAULT_RIDGE,
    filter_words: Optional[Sequence[str]] = None,
) -> List[TemplateDirection]:
    """Every (phrase, layer) direction. Filtered words are refused, not built."""
    blocked = {w.strip().lower() for w in (filter_words or []) if w.strip()}
    refused = [p for p in phrase_states if p.strip().lower() in blocked]
    if refused:
        raise ValueError(
            f"{refused} are on the spurious-word filter list (BR-023's third failure "
            "mode); remove them from the vocabulary rather than building their templates"
        )
    out: List[TemplateDirection] = []
    operators = {L: whitening_operator(states, ridge) for L, states in baseline_states.items()}
    for phrase, by_layer in phrase_states.items():
        for layer, states in sorted(by_layer.items()):
            mu, inv, cov, _lam = operators[layer]
            out.append(template_direction(phrase, layer, states, mu, inv, cov))
    return out
