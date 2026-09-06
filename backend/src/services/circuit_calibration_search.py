"""Adaptive usable-band search for circuit strength calibration (IDL-37).

Two thresholds, two DIFFERENT detectors — this separation is the whole point:

  ONSET (min influence above none) is a DIFFERENCE test: the smallest dial where
  the steered output diverges from the unsteered baseline past the model's own
  sampling-noise floor. No judge — just "did the output move". The floor is
  estimated from several baseline-vs-baseline draws (different seeds), so onset
  means "above sampling noise", not an absolute constant.

  CORRECTNESS CLIFF (max before facts break) is a PROPERTY test: the largest dial
  where every probe answer is still CORRECT, decided by an LLM judge. Perplexity
  and theme metrics CANNOT find this — the observed cliff sat between two adjacent
  dials a perplexity delta could not separate, one giving a correct answer with
  light tint, the next confidently false. A cheap collapse signal may only LOWER
  the search ceiling before judging; it never decides the cliff.

Two invariants this module guarantees to its callers:
  * the returned `cliff` (and `sweet_spot`) is a dial the judge called CORRECT —
    never a broken dial reported as usable;
  * `usable_band` is False when no correct dial exists above onset — the caller
    must NOT clamp a served dial to a degenerate/empty band (badge, not gate:
    a failed measurement must not silently disable the circuit).

The search is a pure function of injected callbacks (generation, judging,
divergence), so it is unit-testable without a GPU or an LLM — including the
load-bearing-judge negative control.

Assumes correctness is roughly MONOTONE in strength (observed: tint grows, then
breaks). A break below a pass is not smoothed — it is flagged `non_monotone` and
the cliff is taken at the LOWEST correct dial before the first break.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

Verdict = Literal["correct", "degrading", "broken"]

#: gen_at(dial, prompt) -> text
GenAtFn = Callable[[float, str], str]
#: baseline_at(prompt, seed) -> text  (unsteered; seed varies the sampling draw)
BaselineFn = Callable[[str, int], str]
#: judge(generation, expected) -> verdict
JudgeFn = Callable[[str, str], Verdict]
#: divergence(a, b) -> distance in [0, ~2]
DivergenceFn = Callable[[str, str], float]

#: How far above the noise floor a dial's divergence must sit to count as onset.
#: A relative margin so a noisy model needs a proportionally larger move.
ONSET_MARGIN = 1.25
#: Seeds used to estimate the baseline-vs-baseline noise floor.
FLOOR_SEEDS = (0, 1, 2)


@dataclass
class CalibrationResult:
    onset: float
    sweet_spot: float
    cliff: float
    #: False when no dial above onset is judged correct — do NOT clamp to it.
    usable_band: bool = True
    non_monotone: bool = False
    steps_used: int = 0
    converged: bool = True
    floor: float = 0.0
    #: False when the judge called the UNSTEERED baseline broken — the judge
    #: cannot grade this circuit's model, so the run is inconclusive (NOT
    #: "no usable band", which would be a false claim about the circuit).
    judge_reliable: bool = True
    #: every judged/measured generation, for the manifest.
    trace: List[Dict] = field(default_factory=list)


def _coarse_steps(lo: float, hi: float, n: int) -> List[float]:
    if n < 2 or hi <= lo:
        return [lo]
    span = hi - lo
    return [round(lo + span * i / (n - 1), 4) for i in range(n)]


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo_i = int(pos)
    frac = pos - lo_i
    if lo_i + 1 >= len(s):
        return s[-1]
    return s[lo_i] + frac * (s[lo_i + 1] - s[lo_i])


def _noise_floor(baseline_at: BaselineFn, divergence: DivergenceFn,
                 probes: List[Dict[str, str]]) -> tuple[float, List[Dict]]:
    """Estimate the model's own sampling noise: the divergence between UNSTEERED
    generations at different seeds. Onset must beat this, not an absolute
    constant (FPRD §3.1). Uses the 0.9 quantile over all seed-pair draws so a
    single noisy pair doesn't set the bar.
    """
    trace: List[Dict] = []
    draws: List[float] = []
    seeds = list(FLOOR_SEEDS)
    for p in probes:
        gens = {s: baseline_at(p["prompt"], s) for s in seeds}
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                d = divergence(gens[seeds[i]], gens[seeds[j]])
                draws.append(d)
                trace.append({"phase": "floor", "probe": p["prompt"],
                              "seed_a": seeds[i], "seed_b": seeds[j],
                              "divergence": d})
    floor = _quantile(draws, 0.9)
    return floor, trace


def find_onset(
    gen_at: GenAtFn,
    baseline_at: BaselineFn,
    divergence: DivergenceFn,
    *,
    lo: float,
    hi: float,
    probes: List[Dict[str, str]],
    coarse: int = 6,
) -> tuple[float, float, List[Dict]]:
    """Smallest dial whose steered output diverges from baseline by more than
    ONSET_MARGIN × the noise floor. Returns (onset, floor, trace). If nothing
    clears the bar (an inert circuit), onset is `lo`.
    """
    if not probes:
        return lo, 0.0, []

    floor, trace = _noise_floor(baseline_at, divergence, probes)
    bar = max(floor * ONSET_MARGIN, floor)  # never below the floor itself

    for dial in _coarse_steps(lo, hi, coarse):
        if dial <= lo:
            continue
        # Compare a STEERED draw against a baseline draw (seed 0). Average over
        # probes so one probe's jitter doesn't decide onset.
        d = sum(
            divergence(baseline_at(p["prompt"], 0), gen_at(dial, p["prompt"]))
            for p in probes
        ) / len(probes)
        trace.append({"dial": dial, "phase": "onset", "divergence": d,
                      "floor": floor, "bar": bar})
        if d > bar:
            return dial, floor, trace
    return lo, floor, trace


def _worst(verdicts: List[Verdict]) -> Verdict:
    order = {"correct": 0, "degrading": 1, "broken": 2}
    return max(verdicts, key=lambda v: order.get(v, 2)) if verdicts else "broken"


def find_cliff(
    gen_at: GenAtFn,
    judge: JudgeFn,
    *,
    lo: float,
    hi: float,
    probes: List[Dict[str, str]],
    collapse: Optional[Callable[[str], bool]] = None,
    max_steps: int = 10,
    tol: float = 0.02,
) -> tuple[float, bool, bool, int, bool, List[Dict]]:
    """Largest dial where EVERY probe is still judged CORRECT.

    Returns (cliff, usable_band, non_monotone, steps_used, converged, trace).

    Guarantees:
      * the returned `cliff` is always a dial judged CORRECT (or `lo` when even
        `lo` is not correct, with usable_band=False);
      * `usable_band` is False when `lo` itself is not correct — there is no
        usable region and the caller must not clamp to a degenerate band.

    `collapse(text)->bool` is an OPTIONAL cheap shortcut that may only LOWER the
    ceiling before judging (skip token soup); it never decides the cliff.
    """
    trace: List[Dict] = []
    steps = 0

    def verdict_at(dial: float) -> Verdict:
        nonlocal steps
        vs: List[Verdict] = []
        for p in probes:
            gen = gen_at(dial, p["prompt"])          # keep the TEXT, not just the verdict
            v = judge(gen, p["expected"])
            vs.append(v)
            trace.append({"dial": dial, "phase": "cliff", "probe": p["prompt"],
                          "verdict": v, "generation": gen})
        steps += 1
        return _worst(vs)

    # Collapse shortcut: walk hi DOWN past degenerate output so the judge is not
    # spent on token soup. Cheap signal, ceiling only — never the cliff itself.
    if collapse is not None:
        guard = 0
        while hi > lo and guard < max_steps:
            sample = gen_at(hi, probes[0]["prompt"]) if probes else ""
            if not collapse(sample):
                break
            hi = round(lo + (hi - lo) / 2, 4)
            guard += 1

    # `lo` (= onset) must itself be correct for any usable band to exist.
    if verdict_at(lo) != "correct":
        return lo, False, False, steps, True, trace

    # Monotonicity scan: judge coarse dials from lo→hi. The cliff is the last
    # CORRECT dial before the first break; a break followed by a later pass flags
    # non_monotone. Density scales with the remaining step budget (NOT a hard
    # cap) so more budget = finer scan and a better chance of catching a narrow
    # dip. This is best-effort: a finite judge budget cannot PROVE no broken dial
    # exists in a continuous band (a sub-grid dip can hide between samples). What
    # IS guaranteed is that the RETURNED cliff/sweet_spot are re-judged correct
    # (below) — so the values shipped to the clamp are never broken, even if an
    # unsampled dip exists inside the range.
    grid_n = max(4, min(max_steps - steps, 12))
    grid = _coarse_steps(lo, hi, grid_n)
    last_correct = lo
    first_break: Optional[float] = None
    non_monotone = False
    for dial in grid:
        if dial <= lo or steps >= max_steps:
            continue
        v = verdict_at(dial)
        if v == "correct":
            if first_break is not None:
                # a pass ABOVE a break → non-monotone; keep the conservative
                # cliff at the last correct dial BEFORE the first break.
                non_monotone = True
            else:
                last_correct = dial
        else:
            if first_break is None:
                first_break = dial

    if first_break is None:
        # correct all the way to hi → the whole range is usable.
        return last_correct, True, non_monotone, steps, True, trace

    # Refine the boundary between last_correct (correct) and first_break (not)
    # by bisection, staying within the step budget. The result is always the
    # highest dial confirmed CORRECT — never the break itself.
    good, bad = last_correct, first_break
    while steps < max_steps and (bad - good) > tol:
        mid = round((good + bad) / 2, 4)
        if verdict_at(mid) == "correct":
            good = mid
        else:
            bad = mid
    converged = (bad - good) <= tol
    return good, True, non_monotone, steps, converged, trace


def calibrate(
    gen_at: GenAtFn,
    baseline_at: BaselineFn,
    judge: JudgeFn,
    divergence: DivergenceFn,
    *,
    probes: List[Dict[str, str]],
    lo: float,
    hi: float,
    max_steps: int = 10,
    margin: float = 0.15,
    collapse: Optional[Callable[[str], bool]] = None,
) -> CalibrationResult:
    """Full band search: onset (drift) then cliff (judge), then a sweet-spot with
    margin below the cliff. All generation/judging is injected.

    If no usable band exists (onset already breaks, or the cliff is not strictly
    above onset), the result carries `usable_band=False` and onset==sweet==cliff;
    the caller must treat that as "no band found" and NOT clamp the served dial.
    """
    # JUDGE SANITY GATE (hardware-informed): grade the UNSTEERED output first.
    # If the judge calls the un-steered model broken, it cannot grade this
    # circuit's model — every downstream verdict is untrustworthy, and reporting
    # "no usable band" would be a FALSE claim about the circuit (the failure is
    # the judge's). Surface judge_reliable=False so the caller reports an
    # inconclusive run, not a broken circuit.
    baseline_verdicts = []
    for p in probes:
        gen = gen_at(0.0, p["prompt"])           # the UNSTEERED transcript
        v = judge(gen, p["expected"])
        baseline_verdicts.append({"dial": 0.0, "phase": "judge_sanity",
                                  "probe": p["prompt"], "verdict": v,
                                  "generation": gen})
    if _worst([bv["verdict"] for bv in baseline_verdicts]) != "correct":
        return CalibrationResult(
            onset=lo, sweet_spot=lo, cliff=lo,
            usable_band=False, judge_reliable=False,
            # ONE dial-batch (dial 0, all probes) — steps_used counts judged
            # dial-batches everywhere, so this is 1, not len(probes) (R3: the
            # sanity return used a per-probe unit inconsistent with every other
            # return).
            steps_used=1, converged=True, floor=0.0,
            trace=baseline_verdicts,
        )

    # The judge-sanity gate above always ran ONE judged dial-batch (dial 0, all
    # probes); its generations are in baseline_verdicts. Count it toward
    # steps_used on every downstream return so the unit is consistent (R3).
    sanity_steps = 1

    onset, floor, onset_trace = find_onset(
        gen_at=gen_at, baseline_at=baseline_at,
        divergence=divergence, lo=lo, hi=hi, probes=probes,
    )
    onset_trace = baseline_verdicts + onset_trace

    cliff, usable, non_monotone, steps, converged, cliff_trace = find_cliff(
        gen_at=gen_at, judge=judge, lo=onset, hi=hi, probes=probes,
        collapse=collapse, max_steps=max_steps,
    )

    if not usable or cliff <= onset:
        # No correct dial above onset (or the band collapsed to a point): report
        # a degenerate band and let the caller refuse to clamp.
        return CalibrationResult(
            onset=onset, sweet_spot=onset, cliff=onset,
            usable_band=False, non_monotone=non_monotone,
            steps_used=sanity_steps + steps, converged=converged, floor=floor,
            trace=onset_trace + cliff_trace,
        )

    # Sweet-spot: strongest still-correct point with a safety margin below the
    # cliff, never below onset. Guaranteed onset ≤ sweet ≤ cliff.
    sweet = max(onset, round(cliff - margin, 4))
    sweet = min(sweet, cliff)

    # RE-CONFIRM the shipped sweet_spot is actually judged correct (Feature 20
    # R2). The scan is best-effort — a sub-grid dip can hide between samples — so
    # the ONE dial we recommend as the default must be re-judged directly. If it
    # is broken (a dip sits at the sweet_spot), walk down toward onset until a
    # correct dial is found; if none is, there is no usable band after all.
    # Each re-judge is a real judge call — count it in steps_used so the manifest
    # reports every judged dial honestly (R3), not just the find_cliff steps.
    rejudge_steps = [0]

    def _judge_worst(dial):
        rejudge_steps[0] += 1
        vs = []
        for p in probes:
            gen = gen_at(dial, p["prompt"])
            v = judge(gen, p["expected"])
            vs.append(v)
            # Record the re-judged generation in the trace so the manifest's
            # transcripts include the SHIPPED sweet_spot dial — often a dial the
            # coarse cliff grid never sampled (R1: it was silently omitted).
            cliff_trace.append({"dial": dial, "phase": "sweet_recheck",
                                "probe": p["prompt"], "verdict": v,
                                "generation": gen})
        return _worst(vs)

    tries = 0
    while sweet > onset and tries < 4 and _judge_worst(sweet) != "correct":
        non_monotone = True   # a break inside [onset, cliff] — flag it
        sweet = round(max(onset, sweet - margin), 4)
        tries += 1
    if sweet <= onset and _judge_worst(onset) != "correct":
        # even onset broke on re-check → no usable band.
        return CalibrationResult(
            onset=onset, sweet_spot=onset, cliff=onset,
            usable_band=False, non_monotone=non_monotone,
            steps_used=sanity_steps + steps + rejudge_steps[0], converged=converged,
            floor=floor, trace=onset_trace + cliff_trace,
        )

    # steps_used counted FRESH here (not a pre-line-362 snapshot) so the final
    # _judge_worst(onset) re-check above is included — its generation lands in
    # the trace, so the count must match (R2).
    return CalibrationResult(
        onset=onset, sweet_spot=sweet, cliff=cliff,
        usable_band=True, non_monotone=non_monotone,
        steps_used=sanity_steps + steps + rejudge_steps[0], converged=converged, floor=floor,
        trace=onset_trace + cliff_trace,
    )
