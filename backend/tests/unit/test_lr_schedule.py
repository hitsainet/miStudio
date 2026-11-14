"""The learning-rate curve: warmup, constant, and the opt-in linear decay (tracker item 7).

The curve is a pure function (`lr_multiplier`) and a builder (`build_lr_scheduler`)
that the training task calls; these tests drive the REAL LambdaLR the builder
returns, including through a save/load the way a resumed training does.

MUTATION CONTROLS (2026-09-15, WS-LOOP; applied alone, this file and the
resume-equivalence file run, bytes restored, sha256 verified). Both red:
  B10 the scheduler reads optimizer steps as training steps
        -> test_with_accumulation_the_curve_is_read_in_training_steps,
           resume equivalence [rolling-jumprelu-accum]
  B11 the decay reaches zero a step early
        -> the three boundary tests and the mid-decay resume (4 failed)
Worker wiring of this schedule (B45, B46) is in test_training_resume_equivalence.py.
"""

import io

import pytest
import torch

from src.services.lr_schedule import build_lr_scheduler, lr_multiplier


class TestTheCurveAtItsBoundaries:
    def test_warmup_starts_at_zero_and_reaches_one_at_warmup_steps(self):
        kw = dict(total_steps=100, warmup_steps=10, decay_steps=0)
        assert lr_multiplier(0, **kw) == 0.0
        assert lr_multiplier(5, **kw) == pytest.approx(0.5)
        assert lr_multiplier(9, **kw) == pytest.approx(0.9)
        assert lr_multiplier(10, **kw) == 1.0
        assert lr_multiplier(99, **kw) == 1.0

    def test_decay_off_is_constant_to_the_end(self):
        assert lr_multiplier(99, total_steps=100, warmup_steps=0, decay_steps=0) == 1.0
        assert lr_multiplier(100, total_steps=100, warmup_steps=0, decay_steps=0) == 1.0

    def test_decay_is_linear_to_zero_over_the_final_steps(self):
        kw = dict(total_steps=100, warmup_steps=10, decay_steps=20)
        assert lr_multiplier(79, **kw) == 1.0  # last constant step
        assert lr_multiplier(80, **kw) == 1.0  # the decay starts at 1: no jump
        assert lr_multiplier(81, **kw) == pytest.approx(19 / 20)
        assert lr_multiplier(90, **kw) == pytest.approx(0.5)
        assert lr_multiplier(99, **kw) == pytest.approx(1 / 20)  # the last step that runs
        assert lr_multiplier(100, **kw) == 0.0

    def test_warmup_and_decay_can_meet_exactly(self):
        kw = dict(total_steps=30, warmup_steps=10, decay_steps=20)
        assert lr_multiplier(9, **kw) == pytest.approx(0.9)
        assert lr_multiplier(10, **kw) == 1.0
        assert lr_multiplier(29, **kw) == pytest.approx(1 / 20)

    def test_an_overlapping_legacy_config_takes_the_smaller_factor(self):
        # warmup + decay > total is refused by the schema; an older row can
        # still carry it, and the curve must not jump.
        kw = dict(total_steps=20, warmup_steps=15, decay_steps=10)
        assert lr_multiplier(12, **kw) == pytest.approx(min(12 / 15, 8 / 10))


def _lrs_per_training_step(scheduler, optimizer, steps, accum):
    """The learning rate each training step's optimizer update uses, stepping as the loop does."""
    used = []
    for step in range(steps):
        if (step + 1) % accum == 0:
            used.append((step, optimizer.param_groups[0]["lr"]))
            optimizer.step()
            scheduler.step()
    return used


def _optimizer():
    param = torch.nn.Parameter(torch.zeros(1))
    param.grad = torch.zeros(1)
    return torch.optim.Adam([param], lr=1.0, betas=(0.0, 0.999))


class TestTheSchedulerTheTaskBuilds:
    def test_without_accumulation_every_step_follows_the_curve(self):
        opt = _optimizer()
        sched = build_lr_scheduler(opt, total_steps=40, warmup_steps=5, decay_steps=10)
        for step, lr in _lrs_per_training_step(sched, opt, 40, accum=1):
            assert lr == pytest.approx(lr_multiplier(step, total_steps=40, warmup_steps=5, decay_steps=10)), step

    def test_with_accumulation_the_curve_is_read_in_training_steps(self):
        """An update at training step s covers steps s-k+1..s; its factor is read at s-k+1.

        The old lambda read the scheduler's own count (optimizer steps) as training
        steps, so with k=4 warmup took 4x as many training steps as configured.
        """
        k = 4
        opt = _optimizer()
        sched = build_lr_scheduler(opt, total_steps=80, warmup_steps=16, decay_steps=16, grad_accum_steps=k)
        used = _lrs_per_training_step(sched, opt, 80, accum=k)
        assert len(used) == 20
        for step, lr in used:
            window_start = step - k + 1
            expected = lr_multiplier(window_start, total_steps=80, warmup_steps=16, decay_steps=16)
            assert lr == pytest.approx(expected), (step, lr, expected)
        # Warmup really ends at training step 16, not 64.
        assert dict(used)[19] == pytest.approx(1.0)


class TestResumeContinuesTheCurve:
    def test_a_scheduler_restored_mid_decay_continues_where_it_stopped(self):
        total, warmup, decay, stop_at = 60, 5, 30, 42  # stop inside the decay window
        kw = dict(total_steps=total, warmup_steps=warmup, decay_steps=decay)

        straight_opt = _optimizer()
        straight = build_lr_scheduler(straight_opt, **kw)
        expected = _lrs_per_training_step(straight, straight_opt, total, accum=1)

        first_opt = _optimizer()
        first = build_lr_scheduler(first_opt, **kw)
        _lrs_per_training_step(first, first_opt, stop_at, accum=1)
        buffer = io.BytesIO()
        torch.save({"optimizer": first_opt.state_dict(), "scheduler": first.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)

        # A fresh process: new objects, built exactly as the task builds them.
        resumed_opt = _optimizer()
        resumed = build_lr_scheduler(resumed_opt, **kw)
        assert resumed_opt.param_groups[0]["lr"] == 0.0  # a fresh scheduler is at warmup step 0
        resumed_opt.load_state_dict(saved["optimizer"])
        resumed.load_state_dict(saved["scheduler"])

        continued = [
            (stop_at + i, lr)
            for i, (_, lr) in enumerate(_lrs_per_training_step(resumed, resumed_opt, total - stop_at, accum=1))
        ]
        assert continued == pytest.approx(expected[stop_at:])
        assert continued[0][1] == pytest.approx((total - stop_at) / decay)


class TestResumeContinuesTheExactMultiplier:
    """A resume that lands inside the warmup, or inside the decay, with accumulation (review R2-C).

    The resume test above stops inside the decay without accumulation. A scheduler restored
    mid-warmup or mid-decay with k=4 must continue at the next WINDOW's factor, read at that
    window's first training step, exactly as the uninterrupted scheduler does.
    """

    @pytest.mark.parametrize("stop_at_window", [2, 13])  # inside the warmup; inside the decay
    def test_restored_with_accumulation_continues_at_the_next_windows_factor(self, stop_at_window):
        k, total, warmup, decay = 4, 64, 16, 16
        kw = dict(total_steps=total, warmup_steps=warmup, decay_steps=decay, grad_accum_steps=k)
        schedule = dict(total_steps=total, warmup_steps=warmup, decay_steps=decay)
        stop_step = stop_at_window * k  # training steps run before the checkpoint

        straight_opt = _optimizer()
        straight = build_lr_scheduler(straight_opt, **kw)
        expected = _lrs_per_training_step(straight, straight_opt, total, accum=k)

        first_opt = _optimizer()
        first = build_lr_scheduler(first_opt, **kw)
        _lrs_per_training_step(first, first_opt, stop_step, accum=k)
        buffer = io.BytesIO()
        torch.save({"optimizer": first_opt.state_dict(), "scheduler": first.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)

        resumed_opt = _optimizer()
        resumed = build_lr_scheduler(resumed_opt, **kw)
        resumed_opt.load_state_dict(saved["optimizer"])
        resumed.load_state_dict(saved["scheduler"])

        continued = [
            (stop_step + step, lr)
            for step, lr in _lrs_per_training_step(resumed, resumed_opt, total - stop_step, accum=k)
        ]
        assert continued == pytest.approx(expected[stop_at_window:])
        # The first resumed update is the next window's factor, read at its first step, and
        # the fixture sits where that matters: neither 0 nor 1.
        first_lr = continued[0][1]
        assert first_lr == pytest.approx(lr_multiplier(stop_step, **schedule))
        assert 0.0 < first_lr < 1.0
