"""The capture buffers every event in host RAM, and nothing checked memory.

`cap_cda1e1da6a0a` was confirmed against an estimate of 20,075,978,187 events,
ran to 45.6%, and the worker was OOM-killed holding ~110 GB of rows on a 131 GB
node with no cgroup limit. The existing ceiling could not have caught it: it is
five times the run's OWN estimate, so a 20-billion-event estimate authorises
100 billion. A ceiling expressed as a multiple of the number that is itself the
problem scales with the problem.

The consequence was worse than a failed run. `assert_no_active_gpu_run` counts
any non-terminal row, so the wedged "running" row refused every subsequent
capture with a 409 until it was cleared by hand.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.services import circuit_capture_service as cap


class TestTheBudgetIsReadFromWhatTheKernelKillsOn:
    def test_a_cgroup_v2_limit_wins(self, tmp_path):
        limit = tmp_path / "memory.max"
        limit.write_text("8589934592\n")          # 8 GiB
        with patch.object(cap, "Path", lambda p: limit if "memory.max" in str(p) else Path(p)):
            assert cap._memory_budget_bytes() == 8589934592

    def test_an_UNLIMITED_cgroup_falls_through_to_MemAvailable(self, tmp_path):
        """`memory.max` is literally "max" on this cluster, which is not a
        number and must not be parsed as one."""
        cg = tmp_path / "memory.max"
        cg.write_text("max\n")
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal: 131689160 kB\nMemAvailable: 111775844 kB\n")

        def fake(p):
            p = str(p)
            if "memory.max" in p:
                return cg
            if "meminfo" in p:
                return meminfo
            return Path("/nonexistent")

        with patch.object(cap, "Path", fake):
            assert cap._memory_budget_bytes() == 111775844 * 1024

    def test_an_UNREADABLE_environment_reports_no_verdict(self):
        """None, not a guess. Inventing a budget could refuse a capture that
        would have run perfectly well."""
        with patch.object(cap, "Path", lambda p: Path("/nonexistent/nope")):
            assert cap._memory_budget_bytes() is None


class TestTheRefusal:
    def test_the_production_run_would_have_been_REFUSED(self):
        """20,075,978,187 events against the 111 GB this node had free."""
        with patch.object(cap, "_memory_budget_bytes",
                          return_value=111_775_844 * 1024):
            msg = cap._exceeds_memory_budget(20_075_978_187)

        assert msg is not None
        assert "GB" in msg

    def test_a_capture_that_FITS_is_not_refused(self):
        """Control: a guard that refuses everything is not a guard."""
        with patch.object(cap, "_memory_budget_bytes",
                          return_value=111_775_844 * 1024):
            assert cap._exceeds_memory_budget(100_000_000) is None

    def test_finalize_overhead_is_counted_not_just_the_buffer(self):
        """`finalize()` concatenates, argsorts and materialises — roughly 3.7x
        the buffered bytes at peak. A guard comparing only the buffer passes
        and then dies inside the sort."""
        budget = 100 * 2**30
        rows_that_just_fit_unmultiplied = budget // cap.EVENT_BYTES

        with patch.object(cap, "_memory_budget_bytes", return_value=budget):
            assert cap._exceeds_memory_budget(rows_that_just_fit_unmultiplied) is not None
        assert cap.PEAK_MEMORY_MULTIPLIER >= 2

    def test_no_verdict_means_no_refusal(self):
        with patch.object(cap, "_memory_budget_bytes", return_value=None):
            assert cap._exceeds_memory_budget(10**12) is None


class TestTheRelativeCeilingCannotSubstitute:
    def test_the_old_ceiling_would_have_PASSED_the_run_that_OOMed(self):
        """Why an absolute bound was needed, stated as a test.

        At 45.6% of its estimate the run held ~9.15 billion events. The
        5x-estimate ceiling sits at 100 billion — it never fires, because it is
        derived from the same inflated number.
        """
        events_est = 20_075_978_187
        buffered_at_oom = int(events_est * 0.456)

        assert not cap.exceeds_size_ceiling(buffered_at_oom, events_est), (
            "precondition: the relative ceiling did not fire — that is the bug"
        )
        with patch.object(cap, "_memory_budget_bytes",
                          return_value=111_775_844 * 1024):
            assert cap._exceeds_memory_budget(buffered_at_oom) is not None, (
                "the absolute guard must catch what the relative one cannot"
            )


class TestTheRefusalIsActionable:
    """A refusal that lists knobs makes the reader redo arithmetic the service
    has already done — and, in the first version, points at a knob that does
    not work.

    Measured on the real corpus: epsilon 0.1 -> 0.9 removed only 80% of events
    (20.1B -> 4.1B, still 183 GB against a 97 GB budget). These features have
    so little dynamic range that even a 90%-of-max threshold admits most of
    them. sample_cap is the lever that is actually linear.
    """

    BUDGET = 97 * 2**30

    def test_it_computes_a_sample_cap_that_would_FIT(self):
        with patch.object(cap, "_memory_budget_bytes", return_value=self.BUDGET):
            suggested = cap._suggested_sample_cap(20_076_133_937, 2000)

            assert suggested is not None
            # the suggestion must itself survive the guard it came from
            scaled = int(20_076_133_937 * suggested / 2000)
            assert cap._exceeds_memory_budget(scaled) is None, (
                f"the suggested sample_cap {suggested} would be refused too"
            )

    def test_the_suggestion_is_PROPORTIONAL_not_a_constant(self):
        """Twice the events, half the documents."""
        with patch.object(cap, "_memory_budget_bytes", return_value=self.BUDGET):
            a = cap._suggested_sample_cap(10**10, 2000)
            b = cap._suggested_sample_cap(2 * 10**10, 2000)

        assert b == pytest.approx(a / 2, rel=0.02)

    def test_it_never_suggests_zero_documents(self):
        with patch.object(cap, "_memory_budget_bytes", return_value=1024):
            assert cap._suggested_sample_cap(10**12, 2000) >= 1

    def test_no_budget_means_no_suggestion_rather_than_a_guess(self):
        with patch.object(cap, "_memory_budget_bytes", return_value=None):
            assert cap._suggested_sample_cap(10**10, 2000) is None
