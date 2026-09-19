"""Does the no-LLM prior separate features an LLM refused from ones it labeled?

This is the gate on the whole triage strategy, and it costs nothing: it scores
features that are ALREADY labeled and compares against a verdict that already
exists. No GPU, no model, no new inference.

Ground truth is `features.category == 'uninterpretable'`, which is the enforced
LLM refusal marker (`openai_labeling_service._enforce_refusal`). On
extr_20260828_080834_sae_sae_39cc_002 that is ~2,207 refused against ~3,073
labeled, from gemma-4-12B-it.

Reports AUC per SIGNAL as well as for the combined prior. Per-signal matters
more than the total on a first run: if one signal carries everything and the
blend dilutes it, the blend is the wrong answer and the weights should follow
the evidence.

AUC reading:
    0.50  the signal is noise
    0.65  the plan's gate — below this, stop and reconsider
    0.70+ genuinely useful for ranking

Usage (inside the backend pod, where the data lives):
    kubectl exec -n mistudio deploy/mistudio-backend -c backend -- \
        env PYTHONPATH=/app python3 /tmp/validate_triage_prior.py [extraction_id]
"""

import json
import os
import sys
from collections import defaultdict

from sqlalchemy import text

from src.core.database import SyncSessionLocal
from src.services.feature_triage_prior import score_signals

DEFAULT_EXTRACTION = "extr_20260828_080834_sae_sae_39cc_002"
GATE = 0.65
# Point at a consensus set to validate against SOUND ground truth instead of the
# stored verdict. Measured 2026-08-30/09-01: gemma-4-12B-it flips its refuse/
# label verdict on 47% of features when the same passages are merely reordered
# and reproduces its own stored label 30% of the time, so `features.category`
# is a coin flip. Noisy labels ATTENUATE AUC toward 0.5, so a prior that clears
# the gate against stored labels is genuinely good, while one that fails there
# may still be fine against consensus — which is why both are worth running.
CONSENSUS_IN = os.environ.get("CONSENSUS_IN", "")


def auc(positives, negatives):
    """Rank-based AUC (Mann-Whitney U), ties counted as half.

    Written out rather than pulled from sklearn so the tie handling is visible:
    a signal that returns the same value for many features (concentration_ratio
    is heavily tied near 0) would otherwise be flattered or punished silently
    depending on the implementation.
    """
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def describe(values):
    if not values:
        return "n=0"
    s = sorted(values)
    n = len(s)
    return (f"n={n} min={s[0]:.3f} p25={s[n // 4]:.3f} med={s[n // 2]:.3f} "
            f"p75={s[3 * n // 4]:.3f} max={s[-1]:.3f}")


def main():
    extraction = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXTRACTION
    db = SyncSessionLocal()

    truth = None
    if CONSENSUS_IN:
        c = json.load(open(CONSENSUS_IN))
        if c.get("extraction") != extraction:
            print(f"REFUSING: consensus set is for {c.get('extraction')}, not "
                  f"{extraction}. Scoring one extraction's features against "
                  f"another's verdicts is not a result.")
            return 1
        truth = {f["feature_id"]: f["consensus"]
                 for f in c["features"] if f.get("unanimous")}
        rows = db.execute(text("""
            SELECT id, category, name, nlp_analysis FROM features
            WHERE extraction_job_id = :e AND nlp_analysis IS NOT NULL
              AND id = ANY(:ids)
        """), {"e": extraction, "ids": list(truth)}).fetchall()
        rows = [(r[1], r[2], r[3], truth[r[0]]) for r in rows]
        print(f"ground truth: CONSENSUS ({CONSENSUS_IN})")
    else:
        rows = db.execute(text("""
            SELECT category, name, nlp_analysis
            FROM features
            WHERE extraction_job_id = :e
              AND labeled_at IS NOT NULL
              AND nlp_analysis IS NOT NULL
        """), {"e": extraction}).fetchall()
        rows = [(r[0], r[1], r[2], None) for r in rows]
        print("ground truth: STORED VERDICT (unstable — 47% flip rate; treat "
              "the AUC as a floor)")

    print(f"extraction : {extraction}")
    print(f"features with nlp_analysis : {len(rows):,}")
    if not rows:
        print("\nNOTHING TO SCORE. Either no labels yet, or the NLP worker has "
              "not run for this extraction — check nlp_status on the extraction.")
        return 1

    # The refusal marker, matching _enforce_refusal / _REFUSAL_LABELS.
    REFUSAL = {"uninterpretable", "noise", "none", "unknown", ""}

    refused = defaultdict(list)   # signal -> [values]  (LLM said uninterpretable)
    labeled = defaultdict(list)   # signal -> [values]  (LLM gave a real label)
    n_ref = n_lab = 0
    missing = defaultdict(int)

    for category, name, nlp, consensus in rows:
        if consensus is not None:
            is_refusal = bool(consensus)
        else:
            is_refusal = (category or "").lower() == "uninterpretable" or \
                         (name or "").lower() in REFUSAL
        bucket = refused if is_refusal else labeled
        n_ref, n_lab = (n_ref + 1, n_lab) if is_refusal else (n_ref, n_lab + 1)
        for signal, value in score_signals(nlp).items():
            if value is None:
                missing[signal] += 1
                continue
            bucket[signal].append(value)

    print(f"  refused : {n_ref:,}")
    print(f"  labeled : {n_lab:,}")
    if n_ref == 0 or n_lab == 0:
        print("\nONLY ONE CLASS PRESENT — separation is undefined. Not a result.")
        return 1

    print(f"\n{'signal':<20} {'AUC':>6}  {'verdict':<12} coverage")
    print("-" * 78)

    results = {}
    for signal in sorted(set(list(refused) + list(labeled))):
        # A signal predicts INTERPRETABILITY, so the labeled class should score
        # HIGHER. AUC is computed with labeled as the positive class.
        a = auc(labeled[signal], refused[signal])
        results[signal] = a
        if a is None:
            verdict = "no data"
        elif a >= 0.70:
            verdict = "USEFUL"
        elif a >= GATE:
            verdict = "marginal"
        elif a <= 0.35:
            verdict = "INVERTED"      # informative, just pointing the other way
        else:
            verdict = "noise"
        cov = len(labeled[signal]) + len(refused[signal])
        pct = 100.0 * cov / len(rows)
        print(f"{signal:<20} {a if a is None else round(a, 3)!s:>6}  "
              f"{verdict:<12} {cov:,} ({pct:.0f}%)")

    print("\nper-class distributions (combined prior):")
    print(f"  labeled : {describe(labeled['prior'])}")
    print(f"  refused : {describe(refused['prior'])}")

    if missing:
        print("\nmissing values by signal:")
        for s, c in sorted(missing.items(), key=lambda kv: -kv[1]):
            if c:
                print(f"  {s:<20} {c:,} of {len(rows):,}")

    prior_auc = results.get("prior")
    best = max(((s, a) for s, a in results.items() if a is not None and s != "prior"),
               key=lambda kv: max(kv[1], 1 - kv[1]), default=(None, None))

    print("\n" + "=" * 78)
    if prior_auc is None:
        print("VERDICT: no prior could be computed. Gate NOT passed.")
        return 1
    print(f"combined prior AUC = {prior_auc:.3f}   (gate {GATE})")
    if best[0]:
        print(f"strongest single signal: {best[0]} at {best[1]:.3f}")
        if prior_auc < max(best[1], 1 - best[1]) - 0.02:
            print("  NOTE: the blend is WORSE than its best component. Reweight "
                  "toward that signal rather than shipping the average.")
    print("GATE:", "PASS" if prior_auc >= GATE else "FAIL — reconsider before Phase 2")
    return 0 if prior_auc >= GATE else 2


if __name__ == "__main__":
    raise SystemExit(main())
