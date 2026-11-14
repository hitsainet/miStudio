"""An early ETA says it is early (ENH-006 G23).

Specialist run `pmr_192c5725e9c0` began scoring a 5,000-row everyday-text sample (737 batches) and
read about 30 hours left from the first dozen batches of its longest-document set, with the rate
still rounding to 0.0; within minutes it read 97 minutes. The cancelled 347-hour run of 2026-11-03
read the same way on its first day, so an operator could not tell a real overrun from an early
guess. Every reading now carries `eta_provisional`: true until the stage (or judge run) has made
`ETA_SETTLE_UNITS` reports AND `ETA_SETTLE_SECONDS` lie between its first report and its latest.
Both are required.

A fake clock drives every case; no test sleeps.
"""
from __future__ import annotations

import pytest

from src.services import live_progress as lp
from src.services.live_progress import (
    ETA_SETTLE_SECONDS,
    ETA_SETTLE_UNITS,
    RATE_WINDOW_SECONDS,
    JudgeProgress,
    StageMeter,
    stage_entry_detail,
)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_the_thresholds_are_the_recorded_ones():
    """Recorded in 0xcc/reviews/mcp_gaps_g17_g41_2026-11-08.md: 25 reports, two rate windows."""
    assert ETA_SETTLE_UNITS == 25
    assert ETA_SETTLE_SECONDS == 2 * RATE_WINDOW_SECONDS == 120.0


def _meter(clock):
    return StageMeter("evaluating", "sets", stage_index=6, stage_count=8, clock=clock)


class TestTheStageMeter:
    def test_a_stage_that_has_just_begun_is_provisional(self):
        assert stage_entry_detail("evaluating", 6, 8)["eta_provisional"] is True
        clock = Clock()
        meter = _meter(clock)
        assert meter.snapshot(0, 3)["eta_provisional"] is True

    def test_the_g23_reading_twelve_slow_batches_is_provisional(self):
        """A dozen batches, ten seconds apart, give a rate and an ETA — and the ETA says it is early."""
        clock = Clock()
        meter = _meter(clock)
        for batch in range(12):
            meter.observe(batch * 0.001)
            clock.now += 10.0
        reading = meter.snapshot(0, 3, {"name": "everyday", "done": 12, "total": 737})
        assert reading["eta_seconds"] is not None and reading["eta_seconds"] > 0
        assert reading["eta_provisional"] is True

    def test_many_batches_in_a_few_seconds_are_still_provisional(self):
        """Units without time: 100 fast reports in 10 s. The window has not filled."""
        clock = Clock()
        meter = _meter(clock)
        for batch in range(100):
            meter.observe(batch * 0.01)
            clock.now += 0.1
        assert meter.snapshot(1, 3)["eta_provisional"] is True

    def test_minutes_with_few_batches_are_still_provisional(self):
        """Time without units: ten reports over five minutes."""
        clock = Clock()
        meter = _meter(clock)
        for batch in range(10):
            meter.observe(batch * 0.01)
            clock.now += 30.0
        assert clock.now - 1000.0 > ETA_SETTLE_SECONDS
        assert meter.snapshot(0, 3)["eta_provisional"] is True

    def test_settled_once_both_have_passed(self):
        clock = Clock()
        meter = _meter(clock)
        for batch in range(ETA_SETTLE_UNITS):
            meter.observe(batch * 0.01)
            clock.now += ETA_SETTLE_SECONDS / (ETA_SETTLE_UNITS - 1)
        clock.now -= ETA_SETTLE_SECONDS / (ETA_SETTLE_UNITS - 1)       # the last report's time
        assert clock.now - 1000.0 == pytest.approx(ETA_SETTLE_SECONDS)
        reading = meter.snapshot(0, 3)
        assert reading["eta_provisional"] is False and reading["eta_seconds"] is not None

    def test_one_report_short_is_provisional(self):
        clock = Clock()
        meter = _meter(clock)
        for batch in range(ETA_SETTLE_UNITS - 1):
            meter.observe(batch * 0.01)
            clock.now += 10.0
        assert meter.snapshot(0, 3)["eta_provisional"] is True

    def test_one_second_short_is_provisional(self):
        clock = Clock()
        meter = _meter(clock)
        for batch in range(ETA_SETTLE_UNITS):
            meter.observe(batch * 0.01)
        clock.now += ETA_SETTLE_SECONDS - 1
        meter.observe(1.0)
        assert meter.snapshot(0, 3)["eta_provisional"] is True
        clock.now += 1
        meter.observe(1.1)
        assert meter.snapshot(0, 3)["eta_provisional"] is False

    def test_reading_it_does_not_move_the_clock(self):
        """The judge's tests drive a clock that steps on every read; the verdict must not read it."""
        reads = []

        def clock():
            reads.append(1)
            return 1000.0 + len(reads)

        meter = _meter(clock)
        meter.observe(0.0)
        before = len(reads)
        meter.snapshot(0, 3)
        assert len(reads) == before

    def test_a_new_stage_starts_provisional_again(self):
        """The heartbeat builds a meter per stage; a settled stage does not settle the next."""
        clock = Clock()
        first = _meter(clock)
        for batch in range(50):
            first.observe(batch * 0.01)
            clock.now += 5.0
        assert first.snapshot(0, 3)["eta_provisional"] is False
        assert _meter(clock).snapshot(0, 3)["eta_provisional"] is True


class TestTheJudge:
    def _tracker(self, clock):
        return JudgeProgress(lambda progress, detail: True, [("a", "A", 1000)], clock=clock)

    def test_early_rows_are_provisional_and_settle(self):
        clock = Clock()
        tracker = self._tracker(clock)
        tracker.begin()
        tracker.start_set(0)
        for _ in range(30):
            tracker.row(object())
            clock.now += 1.0
        reading = tracker.snapshot()
        assert reading["eta_seconds"] is not None and reading["eta_provisional"] is True
        for _ in range(100):
            tracker.row(object())
            clock.now += 1.0
        assert tracker.snapshot()["eta_provisional"] is False


def test_the_run_summary_carries_it():
    from types import SimpleNamespace

    from src.schemas.probe_monitor import ProbeRunSummary

    run = SimpleNamespace(
        id="pmr_1", model_id="m", train_dataset_id="v", eval_dataset_ids=[], calibration_dataset_id=None,
        behaviour=None, behaviour_label=None, sub_question=None, behaviour_version=None, status="running",
        stage="evaluating", progress=80.0,
        progress_detail={"eta_seconds": 108000.0, "eta_provisional": True},
        error_message=None, gpu_uuid=None, config={}, created_at=lp.utc_now(), updated_at=lp.utc_now(),
        completed_at=None)
    summary = ProbeRunSummary.of(run, None)
    assert summary.eta_seconds == 108000.0 and summary.eta_provisional is True
