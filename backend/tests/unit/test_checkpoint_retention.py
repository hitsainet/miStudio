"""Tests for the checkpoint retention policy.

Pruning deletes files permanently, so these tests exist to pin the GUARDS, not
just the happy path. The policy core is deliberately session-free, so almost all
of this is table-driven against plain objects.

MUTATION CONTROLS (each must turn a test red):
  * drop the ``keep_best`` union in select_prunable_steps -> best-kept tests fail
  * drop the ``ordered[-keep_last:]`` slice -> newest-kept tests fail
  * drop the min-age filter -> young-checkpoint test fails
  * remove ACTIVE_TRAINING_STATUSES check -> running-training test fails
  * flip DEFAULT_ENABLED/DEFAULT_DRY_RUN -> defaults test fails
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from src.services.checkpoint_retention import (
    ACTIVE_TRAINING_STATUSES,
    DEFAULT_DRY_RUN,
    DEFAULT_ENABLED,
    DEFAULT_KEEP_BEST,
    DEFAULT_KEEP_LAST,
    RetentionPolicy,
    _parse_bool,
    _parse_int,
    plan_from_checkpoints,
    select_prunable_steps,
)


@dataclass
class FakeCheckpoint:
    """Stand-in for the ORM row — the policy only reads these attributes."""

    id: str
    step: int
    is_best: bool = False
    created_at: Optional[datetime] = None
    file_size_bytes: Optional[int] = None
    storage_path: str = "/nonexistent/checkpoint.safetensors"


OLD = datetime(2020, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2020, 1, 10, tzinfo=timezone.utc)


def _ckpts(*specs) -> list:
    """Build checkpoints from step or (step, is_best), all comfortably old."""
    out = []
    for i, spec in enumerate(specs):
        step, is_best = spec if isinstance(spec, tuple) else (spec, False)
        out.append(
            FakeCheckpoint(id=f"ckpt_{i}", step=step, is_best=is_best, created_at=OLD)
        )
    return out


class TestDefaultsAreInert:
    """Shipping defaults must not delete anything on their own."""

    def test_disabled_by_default(self):
        assert DEFAULT_ENABLED is False, (
            "checkpoint pruning must ship disabled — enabling it by default would "
            "delete user data on first upgrade"
        )

    def test_dry_run_by_default(self):
        assert DEFAULT_DRY_RUN is True

    def test_keeps_best_by_default(self):
        assert DEFAULT_KEEP_BEST is True

    def test_policy_dataclass_matches_module_defaults(self):
        policy = RetentionPolicy()
        assert policy.enabled is DEFAULT_ENABLED
        assert policy.dry_run is DEFAULT_DRY_RUN
        assert policy.keep_last == DEFAULT_KEEP_LAST
        assert policy.keep_best is DEFAULT_KEEP_BEST


class TestSelectPrunableSteps:
    POLICY = RetentionPolicy(
        enabled=True, dry_run=False, keep_last=2, keep_best=True, min_age_hours=24
    )

    def test_keeps_last_two_and_prunes_the_rest(self):
        prunable, kept = select_prunable_steps(
            _ckpts(2000, 4000, 6000, 8000, 10000), self.POLICY, now=NOW
        )
        assert kept == [8000, 10000]
        assert prunable == [2000, 4000, 6000]

    def test_best_step_is_never_prunable(self):
        """THE load-bearing guard: best != last for SAE training."""
        prunable, kept = select_prunable_steps(
            _ckpts(2000, 4000, (6000, True), 8000, 10000), self.POLICY, now=NOW
        )
        assert 6000 in kept
        assert 6000 not in prunable
        assert prunable == [2000, 4000]

    def test_newest_step_always_kept_even_with_keep_last_one(self):
        policy = RetentionPolicy(keep_last=1, keep_best=False, min_age_hours=24)
        prunable, kept = select_prunable_steps(_ckpts(1000, 2000, 3000), policy, now=NOW)
        assert kept == [3000]
        assert prunable == [1000, 2000]

    def test_young_checkpoints_are_kept(self):
        """A step is only prunable when every row in it is older than the cutoff."""
        young = FakeCheckpoint(id="young", step=2000, created_at=NOW - timedelta(hours=1))
        old = FakeCheckpoint(id="old", step=1000, created_at=OLD)
        newest = FakeCheckpoint(id="newest", step=9000, created_at=OLD)
        prunable, kept = select_prunable_steps([old, young, newest], self.POLICY, now=NOW)
        assert 2000 in kept, "a checkpoint younger than min_age_hours must be kept"
        assert 1000 in prunable

    def test_multilayer_step_is_all_or_nothing(self):
        """One row per (step, layer) — pruning must select whole steps.

        Deleting a subset of a step's layers leaves an unloadable checkpoint.
        """
        rows = [
            FakeCheckpoint(id="s1000-l34", step=1000, created_at=OLD),
            FakeCheckpoint(id="s1000-l35", step=1000, created_at=OLD),
            FakeCheckpoint(id="s1000-l36", step=1000, created_at=OLD),
            FakeCheckpoint(id="s2000-l34", step=2000, created_at=OLD),
            FakeCheckpoint(id="s3000-l34", step=3000, created_at=OLD),
        ]
        prunable, kept = select_prunable_steps(rows, self.POLICY, now=NOW)
        assert prunable == [1000]

        plan = plan_from_checkpoints("t1", "completed", rows, self.POLICY, now=NOW)
        assert sorted(plan.checkpoint_ids) == ["s1000-l34", "s1000-l35", "s1000-l36"]

    def test_best_row_excluded_even_if_its_step_were_selected(self):
        """Defence in depth inside plan_from_checkpoints."""
        rows = [
            FakeCheckpoint(id="a", step=1000, is_best=True, created_at=OLD),
            FakeCheckpoint(id="b", step=2000, created_at=OLD),
            FakeCheckpoint(id="c", step=3000, created_at=OLD),
            FakeCheckpoint(id="d", step=4000, created_at=OLD),
        ]
        plan = plan_from_checkpoints("t1", "completed", rows, self.POLICY, now=NOW)
        assert "a" not in plan.checkpoint_ids

    def test_empty_input(self):
        assert select_prunable_steps([], self.POLICY, now=NOW) == ([], [])

    def test_fewer_steps_than_keep_last_prunes_nothing(self):
        prunable, kept = select_prunable_steps(_ckpts(1000, 2000), self.POLICY, now=NOW)
        assert prunable == []
        assert kept == [1000, 2000]

    def test_null_created_at_is_treated_as_new(self):
        rows = [
            FakeCheckpoint(id="unknown", step=1000, created_at=None),
            FakeCheckpoint(id="x", step=2000, created_at=OLD),
            FakeCheckpoint(id="y", step=3000, created_at=OLD),
        ]
        prunable, _ = select_prunable_steps(rows, self.POLICY, now=NOW)
        assert 1000 not in prunable, "unknown age must fail safe (keep)"

    def test_naive_created_at_does_not_crash(self):
        """created_at can be naive; comparing to an aware cutoff would raise."""
        rows = [
            FakeCheckpoint(id="naive", step=1000, created_at=datetime(2020, 1, 1)),
            FakeCheckpoint(id="b", step=2000, created_at=OLD),
            FakeCheckpoint(id="c", step=3000, created_at=OLD),
        ]
        prunable, _ = select_prunable_steps(rows, self.POLICY, now=NOW)
        assert prunable == [1000]


class TestPlanGuards:
    POLICY = RetentionPolicy(
        enabled=True, dry_run=False, keep_last=2, keep_best=True, min_age_hours=24
    )

    @pytest.mark.parametrize("status", sorted(ACTIVE_TRAINING_STATUSES))
    def test_active_training_is_never_planned(self, status):
        """A live run may still write to or resume from its checkpoints."""
        plan = plan_from_checkpoints(
            "t1", status, _ckpts(1000, 2000, 3000, 4000), self.POLICY, now=NOW
        )
        assert plan.checkpoint_ids == []
        assert plan.skipped_reason is not None
        assert not plan.is_actionable

    @pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
    def test_terminal_training_is_planned(self, status):
        plan = plan_from_checkpoints(
            "t1", status, _ckpts(1000, 2000, 3000, 4000), self.POLICY, now=NOW
        )
        assert plan.is_actionable
        assert plan.skipped_reason is None

    def test_no_checkpoints_is_not_actionable(self):
        plan = plan_from_checkpoints("t1", "completed", [], self.POLICY, now=NOW)
        assert plan.skipped_reason == "no checkpoints"

    def test_estimated_bytes_uses_recorded_size(self):
        rows = [
            FakeCheckpoint(id="a", step=1000, created_at=OLD, file_size_bytes=1_000),
            FakeCheckpoint(id="b", step=2000, created_at=OLD, file_size_bytes=2_000),
            FakeCheckpoint(id="c", step=3000, created_at=OLD, file_size_bytes=4_000),
            FakeCheckpoint(id="d", step=4000, created_at=OLD, file_size_bytes=8_000),
        ]
        plan = plan_from_checkpoints("t1", "completed", rows, self.POLICY, now=NOW)
        assert plan.estimated_bytes == 3_000

    def test_missing_size_does_not_break_planning(self):
        """file_size_bytes is NULL for every row the training loop writes."""
        plan = plan_from_checkpoints(
            "t1", "completed", _ckpts(1000, 2000, 3000, 4000), self.POLICY, now=NOW
        )
        assert plan.is_actionable
        assert plan.estimated_bytes == 0


class TestSettingParsers:
    @pytest.mark.parametrize("raw", ["true", "TRUE", "1", "yes", "on", " true "])
    def test_truthy(self, raw):
        assert _parse_bool(raw, False) is True

    @pytest.mark.parametrize("raw", ["false", "0", "no", "off", "FALSE", " off "])
    def test_explicit_falsy(self, raw):
        assert _parse_bool(raw, True) is False

    @pytest.mark.parametrize("raw", ["", "garbage", "t", "Y", "enabled"])
    def test_unrecognised_value_falls_back_to_default(self, raw):
        """Unparseable MUST fail safe, not fail to False.

        For checkpoint_prune_dry_run, False means DELETE FILES — so a typo
        written through the settings API must never arm deletion.
        """
        assert _parse_bool(raw, True) is True
        assert _parse_bool(raw, False) is False

    def test_bool_missing_uses_default(self):
        assert _parse_bool(None, True) is True
        assert _parse_bool(None, False) is False

    def test_int_parses(self):
        assert _parse_int("5", 2, minimum=1) == 5

    def test_int_garbage_falls_back_to_default(self):
        assert _parse_int("not-a-number", 2, minimum=1) == 2

    def test_int_below_minimum_is_clamped(self):
        """keep_last=0 would delete every step including the newest."""
        assert _parse_int("0", 2, minimum=1) == 1

    def test_int_missing_uses_default(self):
        assert _parse_int(None, 7, minimum=1) == 7
