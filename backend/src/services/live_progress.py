"""Live progress for the long probe jobs: a throttle, a sliding-window rate, and an ETA.

⚠ WHY THIS EXISTS (operator, 2026-10-21: "The probe run does not show progress"). A judge run wrote
progress once per SET, so a two-set run read 0% for the whole first set — twenty minutes at about
two judge calls a second, while miServer's request log was the only place the work was visible. A
probe training run wrote a percentage within a stage, but nothing said WHAT it was counting, how
fast, or how long was left, so "token_capture · 47%" could not be told from a stall.

What a live row now carries, in ``progress_detail`` (JSONB, NULL until a job reports):
  * judge runs — rows judged and planned across every set, the current set and its rows, the rate
    over the last minute, an ETA, and the parse failures so far (``JudgeProgress``);
  * training runs — the stage, the units it counts (batches, probes, sets), the item inside it
    (an epoch, a set's batches) and a rate and ETA for the stage (``StageMeter``).

Every write goes through ``record_progress``, which refuses a TERMINAL row, so a throttled write
in flight can never overwrite an operator's cancellation. Rate and ETA are reported as null until
two samples a measurable time apart exist — never a guessed number, and the ETA carries
``eta_provisional: true`` until the stage has run `ETA_SETTLE_UNITS` reports AND `ETA_SETTLE_SECONDS`
(ENH-006 G23).
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from ..core.clock import utc_now

#: A judge run writes at most once per this many seconds, or once per `JUDGE_PROGRESS_ROWS` rows,
#: whichever comes first — plus at every set boundary and at the end. At ~2 rows/s that is a write
#: every 5 s; at 100 rows/s it is four writes a second, one cheap UPDATE each.
JUDGE_PROGRESS_SECONDS: float = 5.0
JUDGE_PROGRESS_ROWS: int = 25

#: The window the rate is measured over. Long enough to smooth one slow reply, short enough that
#: the ETA follows a judge that has sped up or slowed down.
RATE_WINDOW_SECONDS: float = 60.0

#: ⚠ ENH-006 G23: AN ETA IS PROVISIONAL UNTIL BOTH OF THESE HAVE PASSED IN THE CURRENT STAGE (or, for
#: a judge run, the run). Specialist run `pmr_192c5725e9c0` read about 30 hours left from the first
#: dozen batches of its longest-document set, with the rate still rounding to 0.0, and within minutes
#: fell to 97 minutes; the cancelled 347-hour run of 2026-11-03 read the same way on its first day.
#: An operator could not tell a real overrun from an early guess.
#:
#: SECONDS: two rate windows. The rate is a `RATE_WINDOW_SECONDS` sliding window, so before one
#: window has passed it is a rate over the first few seconds; and the first window holds the
#: stage's warm-up (first allocations, the first batches). At two windows the window no longer
#: contains anything from the first minute, so the rate is the stage's steady state.
ETA_SETTLE_SECONDS: float = 2 * RATE_WINDOW_SECONDS
#: UNITS: reports counted (one per batch, epoch or row). Twice the dozen batches behind the
#: 30-hour reading, so a stage of slow batches — where two minutes is only a handful — still needs
#: enough of them that one long batch does not set the rate.
ETA_SETTLE_UNITS: int = 25


class EtaSettling:
    """Whether an ETA is still provisional: fewer than `ETA_SETTLE_UNITS` reports, or less than
    `ETA_SETTLE_SECONDS` between the first report and the latest. Both must pass.

    Measured between REPORTS, the same samples the rate is computed from, and never by reading
    the clock again: the rate and the verdict on it then describe the same span."""

    def __init__(self, *, units: int = ETA_SETTLE_UNITS, seconds: float = ETA_SETTLE_SECONDS) -> None:
        self._units = units
        self._seconds = seconds
        self._first_at: Optional[float] = None
        self._last_at: Optional[float] = None
        self.count = 0

    def observe(self, now: float) -> None:
        if self._first_at is None:
            self._first_at = now
        self._last_at = now
        self.count += 1

    def provisional(self) -> bool:
        if self._first_at is None or self._last_at is None:
            return True
        return self.count < self._units or self._last_at - self._first_at < self._seconds


class RateWindow:
    """Units per second over the last `window_seconds`, from (time, count) samples."""

    def __init__(self, window_seconds: float = RATE_WINDOW_SECONDS) -> None:
        self.window_seconds = window_seconds
        self._samples: Deque[Tuple[float, float]] = deque()

    def add(self, now: float, count: float) -> None:
        self._samples.append((now, count))
        # Keep ONE sample at or before the window's start, so the span is the whole window rather
        # than whatever happened to fall inside it.
        while len(self._samples) > 2 and self._samples[1][0] <= now - self.window_seconds:
            self._samples.popleft()

    def reset(self) -> None:
        self._samples.clear()

    def rate(self) -> Optional[float]:
        if len(self._samples) < 2:
            return None
        (t0, c0), (t1, c1) = self._samples[0], self._samples[-1]
        if t1 - t0 <= 0 or c1 < c0:
            return None
        return (c1 - c0) / (t1 - t0)


def eta_seconds(remaining: float, rate: Optional[float]) -> Optional[float]:
    """Seconds left at `rate`, or None when there is no rate to say it with."""
    if remaining <= 0:
        return 0.0
    if not rate or rate <= 0:
        return None
    return remaining / rate


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(float(value), digits)


class JudgeProgress:
    """Rows judged out of rows PLANNED across every set, throttled, with a rate and an ETA.

    `plan` is ``[(set_id, set_name, planned_rows)]`` computed before the first call, so the bar's
    denominator never moves: `planned_rows` is ``min(max_rows_per_set, set size)``, the exact
    number `judge_rows` will be handed for that set.

    `write(progress, detail)` persists one reading and returns whether it landed;
    `execute_judge_run` passes a closure over ``record_progress``.
    """

    def __init__(
        self,
        write: Callable[[float, Dict[str, Any]], Any],
        plan: Sequence[Tuple[str, str, int]],
        *,
        clock: Callable[[], float] = time.monotonic,
        every_seconds: float = JUDGE_PROGRESS_SECONDS,
        every_rows: int = JUDGE_PROGRESS_ROWS,
        window_seconds: float = RATE_WINDOW_SECONDS,
    ) -> None:
        self._write = write
        self.plan: List[Tuple[str, str, int]] = [(a, b, int(c)) for a, b, c in plan]
        self.rows_total = sum(planned for _id, _name, planned in self.plan)
        self.rows_done = 0
        self.parse_failures = 0
        self.set_index: Optional[int] = None
        self.set_rows_done = 0
        self._clock = clock
        self._every_seconds = every_seconds
        self._every_rows = max(1, int(every_rows))
        self._rate = RateWindow(window_seconds)
        self._settling = EtaSettling()
        self._last_write_at: Optional[float] = None
        self._rows_at_last_write = 0
        self.writes = 0

    # ── what is reported ─────────────────────────────────────────────────────
    def progress(self) -> float:
        if self.rows_total <= 0:
            return 0.0
        return min(100.0, 100.0 * self.rows_done / self.rows_total)

    def snapshot(self, phase: str = "judging") -> Dict[str, Any]:
        rate = self._rate.rate()
        current = self.plan[self.set_index] if self.set_index is not None else None
        return {
            "kind": "judge",
            "phase": phase,
            "rows_done": self.rows_done,
            "rows_total": self.rows_total,
            "set_index": self.set_index,
            "set_count": len(self.plan),
            "set_id": current[0] if current else None,
            "set_name": current[1] if current else None,
            "set_rows_done": self.set_rows_done if current else None,
            "set_rows_total": current[2] if current else None,
            "rate_per_second": _round(rate),
            "eta_seconds": _round(eta_seconds(self.rows_total - self.rows_done, rate), 0),
            "eta_provisional": self._settling.provisional(),
            "parse_failures": self.parse_failures,
            "updated_at": utc_now().isoformat(),
        }

    # ── events ───────────────────────────────────────────────────────────────
    def begin(self) -> None:
        """The plan is known and nothing is judged yet: say how much there is to do."""
        now = self._clock()
        self._rate.add(now, self.rows_done)
        self._flush(now, "planned")

    def start_set(self, index: int) -> None:
        self.set_index = index
        self.set_rows_done = 0
        self._flush(self._clock())

    def row(self, answer: Any) -> None:
        """One row judged. `answer` None is a reply that could not be parsed."""
        self.rows_done += 1
        self.set_rows_done += 1
        if answer is None:
            self.parse_failures += 1
        now = self._clock()
        self._rate.add(now, self.rows_done)
        self._settling.observe(now)
        if (
            self.rows_done - self._rows_at_last_write >= self._every_rows
            or self._last_write_at is None
            or now - self._last_write_at >= self._every_seconds
        ):
            self._flush(now)

    def end_set(self) -> None:
        self._flush(self._clock())

    def _flush(self, now: float, phase: str = "judging") -> None:
        self._write(self.progress(), self.snapshot(phase))
        self.writes += 1
        self._last_write_at = now
        self._rows_at_last_write = self.rows_done


class StageMeter:
    """The rate and ETA of one training-run stage, from its own (done, total) reports.

    Kept separate from the throttle in `execute_probe_run.heartbeat`, which owns WHEN to write; this
    owns WHAT a reading says. Fed on every report so the rate covers the whole window, not only the
    instants that happened to be written.
    """

    def __init__(
        self,
        stage: str,
        unit: str,
        *,
        stage_index: int,
        stage_count: int,
        clock: Callable[[], float] = time.monotonic,
        window_seconds: float = RATE_WINDOW_SECONDS,
    ) -> None:
        self.stage = stage
        self.unit = unit
        self.stage_index = stage_index
        self.stage_count = stage_count
        self._clock = clock
        self._rate = RateWindow(window_seconds)
        self._settling = EtaSettling()

    def observe(self, position: float) -> None:
        """`position` is the stage's progress in its own units, fractional parts included."""
        now = self._clock()
        self._rate.add(now, position)
        self._settling.observe(now)

    def snapshot(
        self, done: float, total: float, current: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        rate = self._rate.rate()
        position = done + _fraction_of(current)
        return {
            "kind": "training",
            "stage": self.stage,
            "stage_index": self.stage_index,
            "stage_count": self.stage_count,
            "done": _round(done, 3),
            "total": _round(total, 3),
            "unit": self.unit,
            "current": current,
            "rate_per_second": _round(rate, 3),
            "eta_seconds": _round(eta_seconds(total - position, rate), 0),
            "eta_provisional": self._settling.provisional(),
            "updated_at": utc_now().isoformat(),
        }


def _fraction_of(current: Optional[Dict[str, Any]]) -> float:
    """How far through the current item is, 0..1 — 0 when nothing says."""
    if not current:
        return 0.0
    done, total = current.get("done"), current.get("total")
    if not total:
        return 0.0
    return max(0.0, min(1.0, float(done or 0) / float(total)))


def stage_entry_detail(stage: str, stage_index: int, stage_count: int) -> Dict[str, Any]:
    """What a stage says the moment it begins, before it has counted anything."""
    return {
        "kind": "training",
        "stage": stage,
        "stage_index": stage_index,
        "stage_count": stage_count,
        "done": None,
        "total": None,
        "unit": None,
        "current": None,
        "rate_per_second": None,
        "eta_seconds": None,
        "eta_provisional": True,
        "updated_at": utc_now().isoformat(),
    }
