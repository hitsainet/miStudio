"""A no-LLM prior on whether an SAE feature is interpretable.

Reads signals already computed and persisted by the NLP worker into
`features.nlp_analysis` (see `nlp_analysis_service.analyze_feature`). Nothing
here recomputes anything, calls a model, or touches the GPU — the whole point is
to rank 30k features for free before spending a model on any of them.

WHY NOT `features.interpretability_score`: that column measures something else.
Its formula (`extraction_service.calculate_interpretability_score`) is
`consistency * 0.7 + sparsity * 0.3` over BINARISED activation patterns — it
never inspects which tokens fire, so a feature on "the" scores like a feature on
"Trump". Most features clear its sparsity band and collect a flat +0.3, which is
why the extraction statistic reads 97.3% interpretable on an extraction where an
LLM refused 42%. It is a liveness check wearing an interpretability name, it is
NOT NULL, and it has live consumers. Left alone.

EVERY SIGNAL IS REPORTED SEPARATELY. A single blended number would hide which
input carries the information, and on the first run nobody knows that yet —
`score_signals` exists so the validation harness can report per-signal
separation and the weights can be set from evidence instead of taste.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Signal weights for the combined prior. Deliberately flat-ish and deliberately
# provisional: they are a starting point to be replaced by whatever the
# validation harness shows actually separates refused from labeled features.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "concentration": 0.30,
    "token_purity": 0.25,
    "cluster_dominance": 0.20,
    "content_ratio": 0.15,
    "whole_word_ratio": 0.10,
}


def _ratio(numerator: Any, denominator: Any) -> Optional[float]:
    """Guarded division returning None rather than a misleading 0.0."""
    try:
        n = float(numerator)
        d = float(denominator)
    except (TypeError, ValueError):
        return None
    if d <= 0:
        return None
    return max(0.0, min(1.0, n / d))


def extract_signals(nlp_analysis: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Pull every available signal out of one `nlp_analysis` blob.

    Each value is in [0, 1] where HIGHER MEANS MORE LIKELY INTERPRETABLE, or
    None when the underlying field is absent. None is not zero: a missing field
    means "unknown", and scoring it as the worst possible value would rank
    features with incomplete analysis as junk.
    """
    out: Dict[str, Optional[float]] = {
        "concentration": None,
        "token_purity": None,
        "cluster_dominance": None,
        "content_ratio": None,
        "whole_word_ratio": None,
        "activation_cv": None,
    }
    if not isinstance(nlp_analysis, dict):
        return out

    pta = nlp_analysis.get("prime_token_analysis") or {}
    stats = nlp_analysis.get("activation_stats") or {}
    clusters = nlp_analysis.get("semantic_clusters") or []
    n_examples = nlp_analysis.get("num_examples_analyzed")

    # How often the modal prime token is THE prime token. The most direct
    # "does this feature have one thing to say" measure available without a model.
    conc = pta.get("concentration_ratio")
    if isinstance(conc, (int, float)):
        out["concentration"] = max(0.0, min(1.0, float(conc)))

    # Inverted diversity: 94 unique tokens across 100 examples is a feature with
    # nothing in common; 3 unique across 100 is a feature with one thing to say.
    diversity = _ratio(pta.get("unique_count"), pta.get("total_count"))
    if diversity is not None:
        out["token_purity"] = 1.0 - diversity

    # Share of examples in the largest semantic cluster.
    if isinstance(clusters, list) and clusters and isinstance(clusters[0], dict):
        out["cluster_dominance"] = _ratio(clusters[0].get("size"), n_examples)

    # token_types keys are SPARSE — a blob may carry only some of them, so every
    # lookup defaults to 0 and the denominator comes from total_count rather
    # than from summing the dict (which would make the ratio always 1.0).
    token_types = pta.get("token_types")
    if isinstance(token_types, dict):
        total = pta.get("total_count")
        content = token_types.get("content_words", 0) or 0
        out["content_ratio"] = _ratio(content, total)

    # fragment_percentage is 0-100, NOT 0-1. Mid-word BPE debris is the single
    # clearest sign that a feature's prime tokens are not words at all.
    frag = pta.get("fragment_percentage")
    if isinstance(frag, (int, float)):
        out["whole_word_ratio"] = max(0.0, min(1.0, 1.0 - float(frag) / 100.0))

    # Carried for the harness to score, NOT in DEFAULT_WEIGHTS. Whether a low
    # coefficient of variation indicates a crisp feature or merely a flat one is
    # genuinely unknown here, and guessing a direction would bake a guess into
    # every score. The harness reports its separation; if it earns a weight it
    # gets one.
    cv = stats.get("coefficient_of_variation")
    if isinstance(cv, (int, float)) and cv >= 0:
        out["activation_cv"] = float(cv)

    return out


def triage_prior(
    nlp_analysis: Optional[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Combine the available signals into one [0, 1] prior, or None.

    Returns None when NO weighted signal is available — an honest "unknown"
    rather than a 0.0 that would sort indistinguishably from a genuinely junk
    feature. Renormalises over the signals that ARE present, so a feature with
    partial analysis is scored on what it has rather than penalised for what it
    lacks.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    signals = extract_signals(nlp_analysis)

    total_weight = 0.0
    acc = 0.0
    for name, weight in w.items():
        value = signals.get(name)
        if value is None:
            continue
        acc += weight * value
        total_weight += weight

    if total_weight <= 0:
        return None
    return max(0.0, min(1.0, acc / total_weight))


def score_signals(nlp_analysis: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Every signal plus the combined prior, for per-signal evaluation."""
    out = dict(extract_signals(nlp_analysis))
    out["prior"] = triage_prior(nlp_analysis)
    return out
