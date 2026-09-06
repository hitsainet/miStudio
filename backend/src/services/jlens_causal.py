"""
Paper-consistent intervention scoring: perturb, CONTINUE THE FORWARD PASS, read
the model's own output.

WHY THIS FILE EXISTS. The first implementation applied a primitive to a captured
activation, pushed the result through the Jacobian transport, and reported the
mean absolute displacement in lens space. That is not what the source paper
measures, and the deviation was not cosmetic:

  * The paper perturbs and then *"allow[s] the forward pass to continue"*, for
    both steering and coordinate patching. It reads the effect from the model's
    real output distribution.
  * Its headline figure is the *"fraction of trials in which the swap places the
    target-appropriate answer at the top of the model's output distribution"*,
    and *"the fraction of swap targets reaching top-5"*, with *"Wilson 95%
    CIs"*. A raw activation-space norm appears nowhere as an effect size.
  * Its results run over many prompts — 50 two-hop prompts, 192 swap trials —
    not one.

MEASURING IN LENS SPACE INVENTED A BUG THE PAPER CANNOT HAVE. The transport is
linear and `apply_additive` is `h + s*v`, so

    J(h + s*v) - J(h) = s*J(v)

and `h` cancels. The reported number was therefore independent of the prompt,
the position, and the entire forward pass that produced the activation.
Confirmed on hardware: two unrelated prompts returned 0.01739214 to seven
significant figures. A rank in the model's real output distribution cannot have
that property, because softmax is not linear.

WHAT THIS MODULE IS AND IS NOT. It scores; it does not run the model. The
forward pass lives in the worker, which owns the model, the hooks and the GPU.
Keeping the statistics here means they can be tested without a model, which is
the only reason the Wilson interval below has tests at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

#: 95% two-sided. Named rather than inlined so a report can state the interval
#: it used; a CI whose confidence level is unrecorded cannot be compared with
#: another one.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple:
    """Wilson score interval for a binomial proportion.

    WILSON, NOT NORMAL-APPROXIMATION. The counts here are small (tens of trials)
    and the proportions are often near 0 or 1 — exactly where the textbook
    `p ± z*sqrt(p(1-p)/n)` produces intervals that extend past 0 or 1 and are
    badly mis-covered. At 0 successes of 20 the normal approximation reports
    `0.000 ± 0.000`, which claims certainty from the one observation that
    carries least.

    Returns `(0.0, 1.0)` for `n == 0` — no trials is maximal uncertainty, not
    zero effect.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


#: Fewest trials at which disjoint Wilson intervals are arithmetically possible.
#:
#: DERIVED, NOT CHOSEN: at n=3 a perfect 3/3 intervened arm gives [0.4385, 1.0]
#: and a perfect 0/3 control gives [0.0, 0.5615], which overlap. At n=4 they are
#: [0.5101, 1.0] and [0.0, 0.4899], which do not. Both UI paths sent a single
#: prompt — one trial — so every Steer and every Swap reported "no effect was
#: demonstrated" regardless of what the model did.
MIN_TRIALS_FOR_SEPARATION = 4


@dataclass(frozen=True)
class Trial:
    """One prompt, scored three ways.

    All three ranks come from the SAME prompt through the SAME procedure, so
    the only difference between them is what was added to the residual. A
    control scored on a different prompt is not a control.

    A rank of `None` means the target token was not in the top-k that was
    examined — kept distinct from a large rank so an unbounded search is never
    implied by a number that was actually a cutoff.
    """

    prompt: str
    baseline_rank: Optional[int]
    intervened_rank: Optional[int]
    control_rank: Optional[int]


def _hits(ranks: Sequence[Optional[int]], within: int) -> int:
    return sum(1 for r in ranks if r is not None and r < within)


@dataclass(frozen=True)
class CausalReport:
    """The paper's figures: top-1 and top-5 rates, intervened versus control.

    THE COMPARISON IS THE FINDING, not the intervened rate. A steering vector
    that puts the target at top-1 in 80% of trials has demonstrated nothing if a
    matched random direction does the same — which is precisely what the
    matched-norm control is there to reveal.

    The BASELINE rate is carried too, because an intervention that "achieves"
    top-1 on prompts where the model already answered that way has moved
    nothing. Without it, a well-chosen prompt set can manufacture any result.
    """

    trials: List[Trial]
    target_token: str
    primitive: str
    layers: List[int]
    #: None for primitives that ignore it. An ablation and a swap take no
    #: strength — recording the request's nominal value would put a number on
    #: the evidence that nothing in the run consumed.
    strength: Optional[float]

    def _rates(self, pick, within: int) -> Dict[str, float]:
        ranks = [pick(t) for t in self.trials]
        n = len(ranks)
        hits = _hits(ranks, within)
        lo, hi = wilson_interval(hits, n)
        return {
            "hits": hits,
            "n": n,
            "rate": (hits / n) if n else 0.0,
            "ci95_low": lo,
            "ci95_high": hi,
        }

    def summary(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "target_token": self.target_token,
            "primitive": self.primitive,
            "layers": list(self.layers),
            "strength": self.strength,
            "n_trials": len(self.trials),
        }
        for name, pick in (
            ("baseline", lambda t: t.baseline_rank),
            ("intervened", lambda t: t.intervened_rank),
            ("control", lambda t: t.control_rank),
        ):
            out[f"{name}_top1"] = self._rates(pick, 1)
            out[f"{name}_top5"] = self._rates(pick, 5)

        # THE HEADLINE, stated as a difference of rates with both sides visible.
        # A single "effect size" that hides which rate it came from cannot be
        # audited, and this project has already shipped one number that turned
        # out to describe something other than what its name said.
        out["excess_top1_over_control"] = (
            out["intervened_top1"]["rate"] - out["control_top1"]["rate"]
        )
        out["excess_top5_over_control"] = (
            out["intervened_top5"]["rate"] - out["control_top5"]["rate"]
        )
        out["separated_from_control"] = self.separated_from_control()

        # WHETHER THE QUESTION COULD HAVE BEEN ANSWERED AT ALL. Below four
        # trials no outcome separates — a PERFECT intervened arm against a
        # PERFECT null control still overlaps — so `separated_from_control:
        # false` at n<4 says nothing about the intervention and everything
        # about the sample size. Reported separately because the two readings
        # are opposite: one is "no effect was demonstrated", the other is
        # "nothing could have been demonstrated".
        out["separation_attainable"] = self.separation_attainable()
        out["min_trials_for_separation"] = MIN_TRIALS_FOR_SEPARATION
        if not out["separation_attainable"]:
            out["caveat"] = (
                f"{len(self.trials)} trial(s): separation is not attainable "
                f"below {MIN_TRIALS_FOR_SEPARATION}. Even a perfect intervened "
                "arm against a perfect null control produces overlapping "
                "Wilson intervals at this sample size, so the verdict "
                "describes the sample and not the intervention. Add prompts."
            )
        return out

    def separation_attainable(self) -> bool:
        """Could ANY outcome at this trial count separate the intervals?

        The best case is every intervened trial a hit and every control trial a
        miss. If those intervals still overlap, the experiment cannot produce a
        positive result, and reporting its null as a finding about the
        direction is a category error.

        Verified numerically: attainable from four trials, never below.
        """
        n = len(self.trials)
        if n == 0:
            return False
        best_low, _ = wilson_interval(n, n)
        _, best_high = wilson_interval(0, n)
        return best_low > best_high

    def separated_from_control(self) -> bool:
        """Whether the intervened and control top-1 intervals are disjoint.

        NON-OVERLAPPING WILSON INTERVALS, not a bare difference of rates. With
        twenty trials a 10-point gap is noise, and reporting it as an effect is
        how a null becomes a finding. This is deliberately conservative:
        non-overlap implies significance at roughly this level, while overlap
        does not imply its absence — so `False` means "not demonstrated here",
        never "demonstrated absent".
        """
        a = self._rates(lambda t: t.intervened_rank, 1)
        b = self._rates(lambda t: t.control_rank, 1)
        return a["ci95_low"] > b["ci95_high"]
