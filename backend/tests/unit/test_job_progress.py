"""The progress marker must answer "did it move?" — and admit when it cannot.

Reported 2026-08-28: the third member of a three-SAE extraction batch was failed
at 186 minutes with "no progress ... may indicate a crashed worker". Nothing had
crashed; it had never started, because only the first batch member is dispatched
and one job alone takes ~169 minutes.

The helper's contract is deliberately asymmetric. It supplies evidence for a
DESTRUCTIVE decision, so "I don't know" must never read as "it is stalled":
None means no evidence and the caller must not reap on it.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.workers import job_progress


class FakeRedis:
    """Enough Redis to exercise the marker, with call visibility."""

    def __init__(self, initial=None, fail=False):
        self.store = dict(initial or {})
        self.expires = {}
        self.fail = fail

    def get(self, key):
        if self.fail:
            raise RuntimeError("redis down")
        return self.store.get(key)

    def set(self, key, value, ex=None):
        if self.fail:
            raise RuntimeError("redis down")
        self.store[key] = value
        self.expires[key] = ex

    def expire(self, key, ttl):
        self.expires[key] = ttl

    def delete(self, key):
        self.store.pop(key, None)


NOW = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)


def _marker(value, age_seconds):
    return json.dumps(
        {
            "value": str(value),
            "first_seen_at": (NOW - timedelta(seconds=age_seconds)).isoformat(),
        }
    )


class TestItReportsAStall:
    def test_an_unchanged_counter_reports_how_long_it_has_been_still(self):
        r = FakeRedis({"janitor:progress:extraction:j1": _marker(0.128, 900)})
        stalled = job_progress.progress_stalled_seconds(
            "extraction", "j1", 0.128, now=NOW, client=r
        )
        assert stalled == pytest.approx(900, abs=1)

    def test_a_counter_that_moved_reports_no_stall(self):
        r = FakeRedis({"janitor:progress:extraction:j1": _marker(0.128, 900)})
        stalled = job_progress.progress_stalled_seconds(
            "extraction", "j1", 0.140, now=NOW, client=r
        )
        assert stalled == 0.0

    def test_a_moved_counter_re_arms_the_clock(self):
        """Otherwise a job that advances once every 10 minutes accumulates a
        stall it never actually had."""
        r = FakeRedis({"janitor:progress:extraction:j1": _marker(0.128, 900)})
        job_progress.progress_stalled_seconds("extraction", "j1", 0.140, now=NOW, client=r)

        stored = json.loads(r.store["janitor:progress:extraction:j1"])
        assert stored["value"] == "0.14"
        assert datetime.fromisoformat(stored["first_seen_at"]) == NOW

    def test_a_stall_does_not_reset_its_own_start_time(self):
        """The TTL is refreshed; first_seen_at is not. A long stall on a quiet
        system must not expire the evidence of itself."""
        r = FakeRedis({"janitor:progress:extraction:j1": _marker(0.128, 900)})
        job_progress.progress_stalled_seconds("extraction", "j1", 0.128, now=NOW, client=r)

        stored = json.loads(r.store["janitor:progress:extraction:j1"])
        assert datetime.fromisoformat(stored["first_seen_at"]) == NOW - timedelta(seconds=900)
        assert r.expires["janitor:progress:extraction:j1"] == job_progress.MARKER_TTL_SECONDS


class TestItAdmitsWhenItCannotAnswer:
    """Each of these must be None, not 0 and not a large number."""

    def test_the_first_sighting_claims_nothing(self):
        r = FakeRedis()
        assert (
            job_progress.progress_stalled_seconds("extraction", "j1", 0.1, now=NOW, client=r)
            is None
        )
        # ...but it starts the clock for next time.
        assert "janitor:progress:extraction:j1" in r.store

    def test_a_missing_counter_claims_nothing(self):
        r = FakeRedis()
        assert (
            job_progress.progress_stalled_seconds("extraction", "j1", None, now=NOW, client=r)
            is None
        )

    def test_a_redis_failure_claims_nothing(self):
        """Evidence for a destructive decision must fail to 'unknown'."""
        r = FakeRedis({"janitor:progress:extraction:j1": _marker(0.1, 9999)}, fail=True)
        assert (
            job_progress.progress_stalled_seconds("extraction", "j1", 0.1, now=NOW, client=r)
            is None
        )

    def test_a_corrupt_marker_claims_nothing(self):
        r = FakeRedis({"janitor:progress:extraction:j1": "not json"})
        assert (
            job_progress.progress_stalled_seconds("extraction", "j1", 0.1, now=NOW, client=r)
            is None
        )


class TestCountersOfAnyShape:
    """Each lifecycle advances a differently-named value."""

    @pytest.mark.parametrize(
        "old,new,moved",
        [
            (0.128, 0.129, True),          # extraction progress, float
            (10300, 10400, True),          # training current_step, int
            (4336, 4336, False),           # activation samples_processed, still
            (12, 13, True),                # labeling examples_completed
            (0.0, 0.0, False),             # never started
        ],
    )
    def test_it_compares_whatever_it_is_given(self, old, new, moved):
        r = FakeRedis({"janitor:progress:k:1": _marker(old, 600)})
        stalled = job_progress.progress_stalled_seconds("k", "1", new, now=NOW, client=r)
        assert (stalled == 0.0) is moved

    def test_zero_is_a_real_value_not_an_absence(self):
        """A job sitting at 0 for hours is the reported bug's signature; it must
        be measurable, not discarded as falsy."""
        r = FakeRedis({"janitor:progress:k:1": _marker(0.0, 11160)})
        assert job_progress.progress_stalled_seconds("k", "1", 0.0, now=NOW, client=r) > 11000
