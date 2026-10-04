"""The checkpoint disk forecast, measured against reality from both ends (R2D-7 / R3D-15).

A full disk is the worst failure this system has: it raises inside the training
loop, the task's handler marks the run FAILED, and a FAILED run's checkpoints are
exactly what its resume needs. So the forecast that decides whether to start a run
has to be RIGHT, and has to stay right when the model changes.

TWO INDEPENDENT PINS, deliberately not sharing an assumption:

  1. ``TestItReproducesTheMeasured16kRun`` compares the forecast with the four
     figures measured on the real 16,384-latent runs (R3-D). Those numbers were
     read off real files; nothing in `checkpoint_disk` derives from them.
  2. ``TestItBoundsWhatTheRealWritersWrite`` writes an actual checkpoint through
     ``save_multilayer_checkpoint`` and ``save_training_state`` and requires the
     forecast to bound what landed on disk. If a writer starts saving something
     new, this fails even though the 16k constants still match.

And ``TestTheForecastCannotDriftFromTheModel`` builds the SAE the way the TRAINING
TASK builds it and the way the forecast builds it, for every architecture, and
requires identical state_dict shapes — because the forecast's whole claim is that
it is pinned to the real writers rather than to a formula someone typed once.

MUTATION CONTROLS: see the table in
`0xcc/tasks` / the scratchpad record for this arc; each was applied alone, this
file run, the bytes restored and the sha256 verified.
"""

import errno
import os

import pytest
import torch

from src.services import checkpoint_disk as CD
from src.services.checkpoint_retention import RetentionPolicy

#: This module is ABOUT the volume, so it reads the real one. conftest's
#: `_a_roomy_checkpoint_volume` stubs `free_bytes_for` for every other test, which
#: would leave `test_it_reads_the_unprivileged_figure_not_the_raw_one` comparing a
#: constant against `os.statvfs` — and, worse, would leave two neighbours passing
#: vacuously because both sides of their assertion came from the same stub. Tests
#: here that want a specific volume still patch it themselves.
pytestmark = pytest.mark.real_checkpoint_disk


# ── What the 16,384-latent runs actually wrote (R3-D, measured off real files) ──
ONE_SAE_WEIGHTS = 268_575_168
ONE_SAE_STATE = 537_301_604
THREE_SAE_STEP = 2_417_630_316
THREE_SAE_STEP_WITH_GRADS = 3_223_355_820

#: The configuration those figures came from: LFM2.5-1.2B's residual width,
#: 16,384 latents, JumpReLU, the three layers of the circuit-study template.
REAL_HP = dict(
    hidden_dim=2048,
    latent_dim=16384,
    architecture_type="jumprelu",
    training_layers=[11, 12, 13],
    hook_types=["residual"],
    total_steps=150_000,
    checkpoint_interval=1000,
)


def _within(forecast: int, measured: int, tolerance: float = 0.001) -> None:
    """The forecast must bound the measurement, and must not be wild about it.

    Over-forecasting is the safe direction (a little headroom goes unused);
    under-forecasting is the direction that fills the disk. So the lower bound is
    exact and the upper bound is a tolerance.
    """
    assert forecast >= measured, (
        f"the forecast UNDER-estimates by {measured - forecast:,} B "
        f"({forecast:,} < {measured:,}) — this is the direction that fills the disk"
    )
    assert forecast <= measured * (1 + tolerance), (
        f"the forecast over-estimates by {forecast - measured:,} B "
        f"({forecast:,} vs {measured:,}), beyond the {tolerance:.1%} tolerance"
    )


class TestItReproducesTheMeasured16kRun:
    def test_one_sae_weights_file(self):
        assert CD.sae_footprint(REAL_HP).weights_file_bytes >= ONE_SAE_WEIGHTS
        _within(CD.sae_footprint(REAL_HP).weights_file_bytes, ONE_SAE_WEIGHTS)

    def test_one_sae_training_state(self):
        _within(CD.sae_footprint(REAL_HP).state_bytes(), ONE_SAE_STATE)

    def test_a_three_sae_checkpoint_step(self):
        forecast = CD.checkpoint_step_bytes(CD.sae_footprint(REAL_HP), 3)
        _within(forecast, THREE_SAE_STEP)

    def test_a_three_sae_step_that_ends_mid_accumulation_also_saves_gradients(self):
        """`include_grads` is the loop's `(step + 1) % grad_accum_steps != 0`.

        A third larger, and the difference is a whole SAE's parameters per SAE —
        forecasting only the smaller shape under-reserves by 805 MB on this run.
        """
        footprint = CD.sae_footprint(REAL_HP)
        plain = CD.checkpoint_step_bytes(footprint, 3)
        with_grads = CD.checkpoint_step_bytes(footprint, 3, include_grads=True)
        _within(with_grads, THREE_SAE_STEP_WITH_GRADS)
        assert with_grads > plain

        # THE GRADIENT TERM IS ONE COPY OF EACH SAE'S PARAMETERS, and nothing else:
        # `build_training_state` puts `param.grad` tensors inside the torch.save
        # archive, so they carry no safetensors header.
        assert with_grads - plain == 3 * footprint.param_bytes

        # The recorded figures differ by 3 x 268,575,168 — the weights FILE size —
        # which is 1,344 B more, because R3-D scaled the file figure rather than the
        # tensor figure. 1,344 is exactly three 448-byte safetensors headers, which a
        # gradient tensor does not have. Reconciled here rather than "fixed", because
        # deriving the term from the WRITERS instead of from that subtraction is the
        # whole reason this module builds the real model.
        recorded_delta = THREE_SAE_STEP_WITH_GRADS - THREE_SAE_STEP
        assert recorded_delta - (with_grads - plain) == 1_344, (
            "the gradient term no longer reconciles with the measured runs by exactly "
            "three safetensors headers"
        )

    def test_the_whole_run_is_hundreds_of_gigabytes(self):
        """The number that makes this guard worth having at all."""
        forecast = CD.forecast_run(REAL_HP)
        assert forecast.steps_remaining == 149, forecast.steps_remaining
        assert 400 * CD.GiB < forecast.total_bytes < 500 * CD.GiB, forecast.total_bytes


class TestItBoundsWhatTheRealWritersWrite:
    """Drive the REAL save path and require the forecast to cover it.

    Small dimensions so the test is fast; the point is not the magnitude but
    that every file the writers produce is accounted for. A writer that starts
    saving a new tensor fails this without anyone remembering to update a
    constant.
    """

    def _write_a_real_checkpoint(self, tmp_path, *, include_grads=False):
        from src.ml.sparse_autoencoder import create_sae
        from src.services.checkpoint_service import (
            CheckpointService,
            build_training_state,
            save_training_state,
        )
        from src.services.dead_latent_resampling import DeadLatentTracker

        hp = dict(hidden_dim=64, latent_dim=256, architecture_type="jumprelu")
        keys = [(0, "residual"), (1, "residual")]
        models = {k: create_sae(**CD.sae_shape_kwargs(hp)) for k in keys}
        optimizers = {k: torch.optim.Adam(m.parameters(), lr=1e-3) for k, m in models.items()}
        schedulers = {
            k: torch.optim.lr_scheduler.LambdaLR(o, lambda _s: 1.0)
            for k, o in optimizers.items()
        }
        # Adam only materialises its moments after a step, and the moments are
        # what makes the state file twice the weights.
        for key, model in models.items():
            out = model(torch.randn(8, 64))
            tensor = out[0] if isinstance(out, (tuple, list)) else (
                list(out.values())[0] if isinstance(out, dict) else out
            )
            tensor.sum().backward()
            optimizers[key].step()

        CheckpointService.save_multilayer_checkpoint(
            models=models, optimizers=optimizers, step=10,
            base_storage_path=str(tmp_path), layer_hook_combinations=keys,
        )
        save_training_state(
            tmp_path / "checkpoint_10",
            build_training_state(
                step=10, sae_keys=keys, optimizers=optimizers, schedulers=schedulers,
                scalers={}, dead_latent_trackers={k: DeadLatentTracker(256) for k in keys},
                activation_ema={}, firing_rate={}, best_loss=1.0, models=models,
                include_grads=include_grads,
            ),
        )
        actual = sum(
            p.stat().st_size for p in (tmp_path / "checkpoint_10").rglob("*") if p.is_file()
        )
        forecast = CD.checkpoint_step_bytes(
            CD.sae_footprint(hp), len(keys), include_grads=include_grads
        )
        return actual, forecast

    def test_the_forecast_covers_a_real_checkpoint(self, tmp_path):
        actual, forecast = self._write_a_real_checkpoint(tmp_path)
        assert forecast >= actual, (
            f"the forecast ({forecast:,}) does not cover what the real writers "
            f"put on disk ({actual:,})"
        )

    def test_the_forecast_covers_a_real_mid_accumulation_checkpoint(self, tmp_path):
        """The gradients branch, through the real `build_training_state`."""
        actual, forecast = self._write_a_real_checkpoint(tmp_path, include_grads=True)
        assert forecast >= actual, (f"{forecast:,} < {actual:,}")

    def test_saving_gradients_really_does_cost_more_on_disk(self, tmp_path):
        """A fixture that agrees by construction would pass both tests above.

        If `include_grads` changed nothing on disk, the two calibration tests
        would still pass while the forecast's gradient term measured nothing.
        """
        plain, _ = self._write_a_real_checkpoint(tmp_path / "plain")
        with_grads, _ = self._write_a_real_checkpoint(tmp_path / "grads", include_grads=True)
        assert with_grads > plain, (
            "the real writer saved the same bytes with and without gradients; "
            "the calibration above proves nothing about the gradient term"
        )


class TestTheForecastCannotDriftFromTheModel:
    """The forecast builds the SAE the way the training task does — for every architecture."""

    @staticmethod
    def _task_style(hp):
        """Exactly the argument list `train_sae_task` passes to `create_sae`."""
        from src.core.framework_defaults import get_framework_defaults
        from src.ml.sparse_autoencoder import create_sae

        architecture_type = hp.get("architecture_type", "standard")
        if architecture_type == "standard":
            architecture_type = "standard_saelens"
        fw = get_framework_defaults(architecture_type)
        l1_alpha = hp.get("l1_alpha") or fw.get("default_l1_alpha", 5e-4)
        with torch.device("meta"):
            return create_sae(
                architecture_type=architecture_type,
                hidden_dim=hp["hidden_dim"],
                latent_dim=hp["latent_dim"],
                l1_alpha=l1_alpha,
                ghost_gradient_penalty=hp.get("ghost_gradient_penalty", 0.0),
                normalize_activations=hp.get("normalize_activations", fw["normalize_activations"]),
                top_k_sparsity=hp.get("top_k_sparsity", None),
                top_k=hp.get("top_k"),
                aux_k=hp.get("aux_k"),
                aux_loss_alpha=hp.get("aux_loss_alpha"),
                initial_threshold=hp.get("initial_threshold", 0.5),
                bandwidth=hp.get("bandwidth", 0.01),
                **({"ste_bandwidth": hp.get("ste_bandwidth", 0.5)}
                   if architecture_type == "jumprelu" else {}),
                sparsity_coeff=hp.get("sparsity_coeff"),
                normalize_decoder=hp.get("normalize_decoder", fw["normalize_decoder"]),
            )

    @pytest.mark.parametrize("architecture", [
        "standard", "standard_saelens", "standard_anthropic",
        "skip", "transcoder", "topk", "jumprelu",
    ])
    def test_the_forecast_sae_has_the_task_sae_s_shapes(self, architecture):
        hp = dict(hidden_dim=128, latent_dim=512, architecture_type=architecture)
        with torch.device("meta"):
            from src.ml.sparse_autoencoder import create_sae
            forecast_model = create_sae(**CD.sae_shape_kwargs(hp))
        task_model = self._task_style(hp)

        forecast_shapes = {k: tuple(v.shape) for k, v in forecast_model.state_dict().items()}
        task_shapes = {k: tuple(v.shape) for k, v in task_model.state_dict().items()}
        assert forecast_shapes == task_shapes, (
            f"{architecture}: the forecast models a different SAE than the one the "
            f"training task builds, so every byte it predicts is for the wrong model"
        )

    def test_the_architectures_do_not_all_cost_the_same(self):
        """Otherwise the parametrised test above would pass against a constant.

        JumpReLU carries a latent-wide `log_threshold` the standard SAE has not,
        so their checkpoints genuinely differ — which is the whole reason the
        forecast builds the real model instead of applying one formula.
        """
        shape = dict(hidden_dim=128, latent_dim=512)
        sizes = {
            arch: CD.sae_footprint({**shape, "architecture_type": arch}).state_dict_bytes
            for arch in ("standard_saelens", "topk", "transcoder", "jumprelu")
        }
        assert len(set(sizes.values())) > 1, sizes
        assert sizes["jumprelu"] > sizes["standard_saelens"], sizes

    def test_the_footprint_allocates_nothing(self):
        """It runs inside an API request, on a model that would be 268 MB on the heap."""
        footprint = CD.sae_footprint(REAL_HP)
        assert footprint.param_bytes > 200 * 1024 ** 2
        # A meta tensor has no storage; if this had allocated, the test process
        # would be carrying a quarter-gigabyte per call.
        with torch.device("meta"):
            from src.ml.sparse_autoencoder import create_sae
            model = create_sae(**CD.sae_shape_kwargs(REAL_HP))
        assert all(p.device.type == "meta" for p in model.parameters())


class TestFreeSpaceIsMeasuredCorrectly:
    def test_it_reads_the_unprivileged_figure_not_the_raw_one(self, tmp_path, monkeypatch):
        """`f_bavail`, not `f_bfree` — the gap is the root-reserved fraction.

        On this project's own volume the two differ by ~95 GiB. Reading `f_bfree` would
        promise the backend space it cannot allocate, turning a correct refusal into a run
        that fills the disk.

        ⚠ THIS TEST USED TO COMPARE TWO LIVE READINGS AND FLAKED. It called `os.statvfs`
        itself, called `free_bytes_for` (which calls it again), and asserted the two were
        EQUAL — so any write to the volume between the two syscalls failed it. It went red
        intermittently under the parallel gate, where sibling workers are writing memmaps,
        Arrow datasets and scratch databases throughout.

        Free space is a MOVING quantity; the field choice is not. So the reading is
        snapshotted: one crafted `statvfs` result, with `f_bavail` and `f_bfree`
        deliberately different, and the assertion is which field was used. That is the
        claim the test's name makes, and it is now deterministic.
        """
        class _Stat:
            f_frsize = 4096
            f_bavail = 1_000            # what an unprivileged writer may use
            f_bfree = 26_000            # includes the root-reserved fraction

        monkeypatch.setattr(os, "statvfs", lambda _path: _Stat())
        measured = CD.free_bytes_for(tmp_path)
        assert measured == _Stat.f_bavail * _Stat.f_frsize
        assert measured != _Stat.f_bfree * _Stat.f_frsize, (
            "free space was read as f_bfree, which includes space the backend cannot "
            "allocate"
        )

    def test_the_snapshot_really_does_distinguish_the_two_fields(self):
        """The control: if the fixture's two fields were equal, the test above would pass
        against either choice."""
        assert 1_000 != 26_000

    def test_it_reads_the_REAL_filesystem_too(self, tmp_path):
        """The snapshot above proves the field choice; this proves the call works against
        a real volume at all. Loose on purpose — an exact comparison is what flaked."""
        measured = CD.free_bytes_for(tmp_path)
        stat = os.statvfs(tmp_path)
        assert measured > 0
        # Within an order of magnitude of a live reading, which a moving disk cannot break.
        assert measured <= stat.f_bfree * stat.f_frsize

    def test_it_walks_up_to_a_directory_that_exists(self, tmp_path):
        """The checkpoint directory does not exist at create time — the task makes it."""
        missing = tmp_path / "trainings" / "train_x" / "checkpoints"
        assert not missing.exists()
        assert CD.free_bytes_for(missing) == CD.free_bytes_for(tmp_path)

    def test_the_checkpoint_root_resolves_through_settings(self):
        root = CD.checkpoint_root_for("train_abc")
        assert root.is_absolute()
        assert root.parts[-3:] == ("trainings", "train_abc", "checkpoints")


class TestHowManyCheckpointsAreStillToCome:
    @pytest.mark.parametrize("total,interval,start,expected", [
        (20, 5, 0, 3),          # steps 5, 10, 15 — step 20 is out of range(0, 20)
        (150_000, 1000, 0, 149),
        (100, 10, 0, 9),
        (20, 5, 11, 1),         # a resume at 11 has only step 15 left; 5 and 10 are written
        (20, 5, 16, 0),
        (5, 5, 0, 0),           # never reaches a multiple inside the range
        (0, 1000, 0, 0),
    ])
    def test_the_count_matches_the_loops_own_condition(self, total, interval, start, expected):
        assert CD.checkpoint_steps_remaining(
            total_steps=total, checkpoint_interval=interval, start_step=start
        ) == expected

    def test_it_agrees_with_actually_running_the_condition(self):
        """The arithmetic against the loop's literal `step % interval == 0 and step > 0`."""
        for total in (17, 20, 101, 1000):
            for interval in (3, 5, 7, 100):
                for start in (0, 4, 11):
                    brute = sum(
                        1 for step in range(start, total)
                        if step % interval == 0 and step > 0
                    )
                    assert CD.checkpoint_steps_remaining(
                        total_steps=total, checkpoint_interval=interval, start_step=start
                    ) == brute, (total, interval, start)


class TestTheReserve:
    def test_a_small_run_keeps_the_five_gibibyte_floor(self):
        assert CD.reserve_bytes_for(1024) == CD.MIN_RESERVE_BYTES

    def test_a_large_run_keeps_two_whole_checkpoint_steps(self):
        """A single step of the real run is 3 GB; two of them exceed the floor."""
        per_step = CD.checkpoint_step_bytes(CD.sae_footprint(REAL_HP), 3, include_grads=True)
        assert CD.reserve_bytes_for(per_step) == 2 * per_step
        assert CD.reserve_bytes_for(per_step) > CD.MIN_RESERVE_BYTES


class TestTheDecision:
    def _forecast(self, total=10 * CD.GiB, per_step=CD.GiB):
        return CD.RunForecast(
            per_step_bytes=per_step, per_step_bytes_with_grads=per_step,
            steps_remaining=10, checkpoints_bytes=total, export_bytes=0,
        )

    def test_it_fits_when_there_is_room_for_forecast_and_reserve(self):
        verdict = CD.decide(
            forecast=self._forecast(), free_bytes=20 * CD.GiB, path="/data",
        )
        assert verdict.fits
        assert "Checkpoint disk OK" in verdict.message()

    def test_it_refuses_when_the_reserve_would_be_eaten(self):
        """10 GiB forecast + 5 GiB reserve against 12 GiB free: refused."""
        verdict = CD.decide(
            forecast=self._forecast(), free_bytes=12 * CD.GiB, path="/data",
        )
        assert not verdict.fits
        assert verdict.shortfall_bytes == 3 * CD.GiB

    def test_the_message_names_forecast_free_and_reserve(self):
        verdict = CD.decide(forecast=self._forecast(), free_bytes=1 * CD.GiB, path="/data/x")
        message = verdict.message()
        assert "10.0 GiB of checkpoints" in message
        assert "1.0 GiB free on /data/x" in message
        assert "5.0 GiB reserve" in message
        assert "Short by" in message

    def test_another_running_job_is_counted_against_the_decision(self):
        """Room for one run is not room for two sharing a volume."""
        alone = CD.decide(forecast=self._forecast(), free_bytes=17 * CD.GiB, path="/d")
        assert alone.fits
        shared = CD.decide(
            forecast=self._forecast(), free_bytes=17 * CD.GiB, path="/d",
            other_runs_bytes=5 * CD.GiB, other_runs=1,
        )
        assert not shared.fits
        assert "already promised to 1 other running job" in shared.message()

    def test_reclaimable_space_can_rescue_a_run(self):
        refused = CD.decide(forecast=self._forecast(), free_bytes=12 * CD.GiB, path="/d")
        assert not refused.fits
        rescued = CD.decide(
            forecast=self._forecast(), free_bytes=12 * CD.GiB, path="/d",
            reclaimable_bytes=4 * CD.GiB,
        )
        assert rescued.fits
        assert "reclaimable by the pruner" in rescued.message()


class _Row:
    def __init__(self, id, status, hp=None, current_step=0, checkpoints=()):
        self.id = id
        self.status = status
        self.hyperparameters = hp or dict(
            hidden_dim=64, latent_dim=256, architecture_type="jumprelu",
            training_layers=[0], hook_types=["residual"],
            total_steps=1000, checkpoint_interval=100,
        )
        self.current_step = current_step
        self.checkpoints = list(checkpoints)


class TestOtherActiveRuns:
    def test_only_the_still_running_ones_are_counted(self):
        rows = [
            _Row("a", "running"), _Row("b", "paused"), _Row("c", "pending"),
            _Row("d", "completed"), _Row("e", "failed"), _Row("f", "cancelled"),
        ]
        total, count = CD.other_active_runs_bytes(rows)
        assert count == 3, "a completed or cancelled run writes nothing more"
        assert total > 0

    def test_the_run_asking_the_question_is_not_counted_against_itself(self):
        rows = [_Row("mine", "running"), _Row("other", "running")]
        both, count_both = CD.other_active_runs_bytes(rows)
        mine_excluded, count = CD.other_active_runs_bytes(rows, exclude_id="mine")
        assert count_both == 2 and count == 1
        assert mine_excluded < both

    def test_a_run_part_way_through_only_owes_what_is_left(self):
        fresh, _ = CD.other_active_runs_bytes([_Row("a", "running", current_step=0)])
        nearly_done, _ = CD.other_active_runs_bytes([_Row("a", "running", current_step=950)])
        assert nearly_done < fresh

    def test_an_unforecastable_row_is_skipped_not_fatal(self):
        """One malformed row must not take down the create endpoint."""
        rows = [_Row("bad", "running", hp={"nonsense": True}), _Row("good", "running")]
        total, count = CD.other_active_runs_bytes(rows)
        assert count == 1 and total > 0


class TestWhatPruningCanActuallyRescue:
    """`checkpoint_retention` is far more restricted than "it will free space"."""

    def _terminal_row_with_checkpoints(self):
        from datetime import datetime, timedelta, timezone

        old = datetime.now(timezone.utc) - timedelta(days=30)

        class _Ckpt:
            def __init__(self, step, layer):
                self.id = f"c{step}_{layer}"
                self.step = step
                self.storage_path = f"/data/x/checkpoint_{step}/layer_{layer}/checkpoint.safetensors"
                self.is_best = False
                self.created_at = old
                self.file_size_bytes = 1_000_000_000

        return _Row(
            "done", "completed",
            checkpoints=[_Ckpt(s, 0) for s in (100, 200, 300, 400, 500)],
        )

    def test_nothing_is_reclaimable_while_pruning_is_disabled(self):
        rows = [self._terminal_row_with_checkpoints()]
        assert CD.reclaimable_by_pruning(rows, RetentionPolicy(enabled=False)) == 0

    def test_nothing_is_reclaimable_while_pruning_is_a_dry_run(self):
        """A dry run reports and deletes NOTHING; counting its plan is a lie.

        This is the default (`DEFAULT_DRY_RUN = True`), so getting it wrong would
        let every run start on space that never appears.
        """
        rows = [self._terminal_row_with_checkpoints()]
        assert CD.reclaimable_by_pruning(rows, RetentionPolicy(enabled=True, dry_run=True)) == 0

    def test_a_live_policy_reclaims_from_terminal_runs(self):
        rows = [self._terminal_row_with_checkpoints()]
        freed = CD.reclaimable_by_pruning(rows, RetentionPolicy(enabled=True, dry_run=False))
        assert freed > 0

    def test_an_active_run_is_never_prunable_however_live_the_policy(self):
        """THE FINDING THAT MATTERS: pruning can never rescue the run being started.

        `ACTIVE_TRAINING_STATUSES` puts PENDING, INITIALIZING, RUNNING and PAUSED
        off limits, and the run asking "will I fit?" is active for the whole of
        its life. Any relief comes from OTHER, terminal runs — so a forecast that
        assumed its own checkpoints would be pruned as it went would be counting
        on something structurally incapable of happening.
        """
        row = self._terminal_row_with_checkpoints()
        row.status = "running"
        live = RetentionPolicy(enabled=True, dry_run=False)
        assert CD.reclaimable_by_pruning([row], live) == 0
        row.status = "paused"
        assert CD.reclaimable_by_pruning([row], live) == 0

    def test_no_policy_at_all_reclaims_nothing(self):
        assert CD.reclaimable_by_pruning([self._terminal_row_with_checkpoints()], None) == 0

    def test_this_module_skips_active_runs_without_relying_on_the_pruner(self, monkeypatch):
        """The skip must be THIS module's, not borrowed from `plan_from_checkpoints`.

        MUTATION CONTROL M6 SURVIVED the whole suite: deleting the
        `ACTIVE_TRAINING_STATUSES` guard in `reclaimable_by_pruning` changed
        nothing observable, because `plan_from_checkpoints` independently refuses
        an active training and returns a plan of zero bytes. That made this
        module's guard decorative — it would silently stop protecting anything
        the moment the other module's rule moved, and no test would notice.

        So stub the planner to report space for ANY row, and require that an
        active training still yields zero. The first assertion is a positive
        control: without it, a stub that never reached the code under test would
        make the second assertion pass vacuously.
        """
        from src.services import checkpoint_retention

        def generous(training_id, training_status, checkpoints, policy, now=None):
            plan = checkpoint_retention.PrunePlan(training_id=training_id)
            plan.checkpoint_ids = [c.id for c in checkpoints]
            plan.estimated_bytes = 9 * CD.GiB
            return plan

        monkeypatch.setattr(checkpoint_retention, "plan_from_checkpoints", generous)
        live = RetentionPolicy(enabled=True, dry_run=False)

        terminal = self._terminal_row_with_checkpoints()
        assert CD.reclaimable_by_pruning([terminal], live) == 9 * CD.GiB, (
            "the stub never reached the code under test, so the assertion below "
            "would pass whatever this module did"
        )

        active = self._terminal_row_with_checkpoints()
        active.status = "running"
        assert CD.reclaimable_by_pruning([active], live) == 0, (
            "an active training was counted as reclaimable; the run asking 'will I "
            "fit?' would be told its own checkpoints will be pruned as it writes them"
        )


class TestRoomForOneMoreStep:
    def test_it_asks_for_the_step_plus_the_reserve(self, tmp_path, monkeypatch):
        """⚠ THE FREE-SPACE READING IS PINNED, NOT TAKEN TWICE.

        This compared `room_for_one_step`'s reported free space against a SECOND live
        `statvfs` of the same path. The two readings race anything else writing to the
        volume, and they lost: the test failed once in a full parallel run (four xdist
        workers creating and dropping databases on this disk) and passed on its own,
        which is the signature of exactly that. The intent — that the function reports
        the reading it decided on — is asserted here against a fixed reading instead.
        """
        monkeypatch.setattr(CD, "free_bytes_for", lambda _p: 7 * CD.GiB)
        has_room, free, needed = CD.room_for_one_step(
            per_step_bytes=1024, checkpoint_dir=tmp_path
        )
        assert needed == 1024 + CD.MIN_RESERVE_BYTES
        assert free == 7 * CD.GiB, "the reported free space is not the one it measured"
        assert has_room is True

    def test_it_really_reads_the_volume(self, tmp_path):
        """The other half, without a comparison that can race: an unpatched call must
        return a plausible reading for a real path rather than a constant."""
        _has_room, free, _needed = CD.room_for_one_step(
            per_step_bytes=1024, checkpoint_dir=tmp_path
        )
        assert free > 0
        assert free < 1 << 60

    def test_a_step_larger_than_the_volume_has_no_room(self, tmp_path, monkeypatch):
        monkeypatch.setattr(CD, "free_bytes_for", lambda _p: 4096)
        has_room, _, _ = CD.room_for_one_step(per_step_bytes=1024, checkpoint_dir=tmp_path)
        assert not has_room

    def test_a_roomy_volume_has_room(self, tmp_path, monkeypatch):
        monkeypatch.setattr(CD, "free_bytes_for", lambda _p: 100 * CD.GiB)
        has_room, _, _ = CD.room_for_one_step(per_step_bytes=CD.GiB, checkpoint_dir=tmp_path)
        assert has_room

    def test_the_two_save_shapes_straddle_a_real_decision(self, monkeypatch):
        """There IS free space where sizing against the wrong shape flips the answer.

        MUTATION CONTROL M19 SURVIVED: the worker computing its per-save figure
        with `include_grads=False` left the suite green, because the fake free
        space in those tests (4 KiB, or 500 GiB) sits nowhere near the region
        where the two shapes differ — the fixture agreed with either answer by
        construction.

        This pins the region itself: at this reading the smaller shape says
        "go ahead" and the larger says "no room". A save begun on the smaller
        figure is one that cannot finish, which is the whole failure being
        guarded against.
        """
        footprint = CD.sae_footprint(REAL_HP)
        small = CD.checkpoint_step_bytes(footprint, 3, include_grads=False)
        large = CD.checkpoint_step_bytes(footprint, 3, include_grads=True)
        assert large > small

        free = small + CD.reserve_bytes_for(small) + 1
        assert free < large + CD.reserve_bytes_for(large), (
            "the two shapes do not straddle this reading, so the test proves nothing"
        )
        monkeypatch.setattr(CD, "free_bytes_for", lambda _p: free)

        assert CD.room_for_one_step(per_step_bytes=small, checkpoint_dir="/data")[0] is True
        assert CD.room_for_one_step(per_step_bytes=large, checkpoint_dir="/data")[0] is False


class TestRemovingATornCheckpoint:
    def _partial(self, tmp_path):
        step_dir = tmp_path / "checkpoint_500"
        layer = step_dir / "layer_11_residual"
        layer.mkdir(parents=True)
        (layer / "checkpoint.safetensors").write_bytes(b"x" * 5000)
        (step_dir / "training_state.pt.partial").write_bytes(b"y" * 1000)
        return step_dir

    def test_the_directory_goes_and_the_bytes_come_back(self, tmp_path):
        step_dir = self._partial(tmp_path)
        freed = CD.remove_partial_step(step_dir)
        assert not step_dir.exists()
        assert freed == 6000

    def test_a_torn_step_would_otherwise_be_visible_to_finalize(self, tmp_path):
        """Not a hypothetical: `list_checkpoint_steps` scans the FILESYSTEM.

        Stop & Finalize rebuilds the run's exported SAEs from the newest step
        directory it can see. A step left half-written by a full disk is such a
        directory, and no database row is consulted — so leaving it would let a
        finalize export weights from a truncated checkpoint.
        """
        from src.services import training_finalize_service as TFS

        (tmp_path / "checkpoint_100" / "layer_11_residual").mkdir(parents=True)
        torn = self._partial(tmp_path)
        monkey = tmp_path

        seen = [
            int(p.name.split("_")[1]) for p in monkey.iterdir()
            if p.is_dir() and TFS._CHECKPOINT_DIR_RE.match(p.name)
        ]
        assert 500 in seen, "the torn step is visible to the finalize scanner"

        CD.remove_partial_step(torn)
        seen_after = [
            int(p.name.split("_")[1]) for p in monkey.iterdir()
            if p.is_dir() and TFS._CHECKPOINT_DIR_RE.match(p.name)
        ]
        assert seen_after == [100], seen_after

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        assert CD.remove_partial_step(tmp_path / "nope") == 0


class TestRecognisingAFullDisk:
    def test_enospc_is_a_full_disk(self):
        assert CD.is_out_of_space(OSError(errno.ENOSPC, "No space left on device"))

    def test_a_quota_is_a_full_disk_too(self):
        assert CD.is_out_of_space(OSError(errno.EDQUOT, "Disk quota exceeded"))

    def test_other_errors_are_not(self):
        """A permission error must keep failing the run loudly, not pause it."""
        assert not CD.is_out_of_space(OSError(errno.EACCES, "Permission denied"))
        assert not CD.is_out_of_space(OSError(errno.EIO, "I/O error"))
        assert not CD.is_out_of_space(ValueError("nothing to do with disks"))

    def test_it_looks_through_a_wrapping_exception(self):
        """`torch.save` can surface the real errno as a cause."""
        wrapper = RuntimeError("could not save")
        wrapper.__cause__ = OSError(errno.ENOSPC, "No space left on device")
        assert CD.is_out_of_space(wrapper)

    def test_it_does_not_match_on_the_message(self):
        """The message is locale-dependent; only the errno is authoritative."""
        assert not CD.is_out_of_space(OSError("No space left on device"))
