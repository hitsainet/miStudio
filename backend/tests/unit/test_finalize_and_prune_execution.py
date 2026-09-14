"""End-to-end tests for the two functions that actually MUTATE things.

The policy/discovery helpers are covered elsewhere; these cover the parts that
write files and delete data:
  * finalize_from_checkpoint  — reads checkpoints, writes community_format
  * _execute_plan             — deletes checkpoint rows and files
  * delete_checkpoint_files   — the shared sync unlink/rmdir helper

MUTATION CONTROLS (each must turn a test red):
  * pass `hp` instead of `export_hp` to the community writer -> cfg.json test fails
  * drop the completeness check -> partial-step test fails
  * commit the row BEFORE unlinking in _execute_plan -> ordering test fails
  * ignore policy.keep_best in _execute_plan -> keep_best test fails
"""

import json
from pathlib import Path

import pytest
import torch

from src.ml.sparse_autoencoder import create_sae
from src.services.checkpoint_retention import RetentionPolicy, PrunePlan
from src.services.checkpoint_service import CheckpointService
from src.services.training_finalize_service import (
    FinalizeError,
    finalize_from_checkpoint,
)
from src.workers.prune_checkpoints import _execute_plan

HIDDEN, LATENT = 32, 128


class FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter_by(self, **kw):
        rows = [
            r for r in self._rows
            if all(getattr(r, k, None) == v for k, v in kw.items())
        ]
        return FakeQuery(rows)

    def filter(self, *args):
        """Actually APPLY the filters.

        Returning self unchanged made the fake agree with any fixture by
        construction: expected_sae_keys filters by (training_id, step), so an
        ignoring filter hands back every row of every step and a multi-step test
        would assert something false.
        """
        rows = self._rows
        for expr in args:
            try:
                column = expr.left.key
                value = expr.right.value
            except AttributeError:  # not a simple Column == literal
                continue
            rows = [r for r in rows if getattr(r, column, None) == value]
        return FakeQuery(rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class FakeSession:
    """Minimal stand-in: finalize only needs query().filter_by().first()/all()."""

    def __init__(self, by_type):
        self._by_type = by_type
        self.deleted = []
        self.commits = 0

    def query(self, model):
        return FakeQuery(self._by_type.get(model.__name__, []))

    def delete(self, obj):
        self.deleted.append(obj)

    def commit(self):
        self.commits += 1


class FakeTraining:
    def __init__(self, tid="t1", hp=None, model_id="m1"):
        self.id = tid
        self.hyperparameters = hp or {}
        self.model_id = model_id


class FakeModel:
    def __init__(self, mid="m1", repo_id="ibm-granite/granite-4.1-8b"):
        self.id = mid
        self.repo_id = repo_id


class FakeCheckpointRow:
    def __init__(self, cid, step, storage_path, is_best=False, extra=None):
        self.id = cid
        self.step = step
        self.storage_path = storage_path
        self.is_best = is_best
        self.extra_metadata = extra or {}


def _write_ckpt(path: Path, layer: int, hook: str = "residual", step: int = 1000):
    model = create_sae(
        architecture_type="jumprelu",
        hidden_dim=HIDDEN,
        latent_dim=LATENT,
        l1_alpha=1e-3,
        initial_threshold=0.5,
        bandwidth=0.01,
        sparsity_coeff=1e-3,
        normalize_decoder=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    path.parent.mkdir(parents=True, exist_ok=True)
    CheckpointService.save_checkpoint(
        model=model, optimizer=opt, step=step, storage_path=str(path),
        extra_metadata={"layer_idx": layer, "hook_type": hook},
    )


def _setup_training(tmp_path, monkeypatch, layers=(34, 35), step=1000, hp=None):
    from src.services import training_finalize_service as svc

    monkeypatch.setattr(svc.settings, "data_dir", tmp_path)
    step_dir = tmp_path / "trainings" / "t1" / "checkpoints" / f"checkpoint_{step}"
    rows = []
    for layer in layers:
        f = step_dir / f"layer_{layer}_residual" / "checkpoint.safetensors"
        _write_ckpt(f, layer, step=step)
        rows.append(
            FakeCheckpointRow(
                f"ckpt_{layer}", step, str(f),
                extra={"layer_idx": layer, "hook_type": "residual"},
            )
        )
    training = FakeTraining(hp=hp or {
        "architecture_type": "jumprelu",
        "hidden_dim": HIDDEN,
        "latent_dim": LATENT,
        "training_layers": list(layers),
        "hook_types": ["residual"],
    })
    db = FakeSession({
        "Training": [training],
        "Model": [FakeModel()],
        "Checkpoint": rows,
    })
    return db, step_dir


class TestFinalizeEndToEnd:
    def test_writes_community_format_for_every_layer(self, tmp_path, monkeypatch):
        db, _ = _setup_training(tmp_path, monkeypatch, layers=(34, 35))

        result = finalize_from_checkpoint(db, "t1")

        out = Path(result["community_format_dir"])
        assert result["sae_count"] == 2
        for layer in (34, 35):
            d = out / f"layer_{layer}_residual"
            assert (d / "cfg.json").is_file(), f"missing cfg.json for layer {layer}"
            assert (d / "sae_weights.safetensors").is_file()

    def test_cfg_json_matches_the_weights_not_stale_hyperparameters(
        self, tmp_path, monkeypatch
    ):
        """THE cfg.json drift guard.

        The training task corrects hidden_dim in memory and never persists it,
        so stored hyperparameters can disagree with the actual tensors. cfg.json
        is built from hyperparameters — it must carry the CHECKPOINT's dims, or
        every downstream loader reads a config that contradicts the weights.
        """
        stale = {
            "architecture_type": "standard",   # WRONG: checkpoint is JumpReLU
            "hidden_dim": HIDDEN * 4,          # WRONG on purpose
            "latent_dim": LATENT * 4,          # WRONG on purpose
            "training_layers": [34],
            "hook_types": ["residual"],
        }
        db, _ = _setup_training(tmp_path, monkeypatch, layers=(34,), hp=stale)

        result = finalize_from_checkpoint(db, "t1")

        cfg = json.loads(
            (Path(result["community_format_dir"]) / "layer_34_residual" / "cfg.json").read_text()
        )
        assert cfg["d_in"] == HIDDEN, "cfg.json d_in must come from the checkpoint"
        assert cfg["d_sae"] == LATENT, "cfg.json d_sae must come from the checkpoint"
        assert cfg["architecture"] == "jumprelu", (
            "cfg.json architecture must reflect the real checkpoint class, else a "
            "loader applies plain ReLU to a jump-thresholded SAE"
        )

    def test_incomplete_step_is_refused(self, tmp_path, monkeypatch):
        """A SIGTERMed worker can leave a step with only some layers written.

        Exporting it would present a 1-layer subset as a complete 2-layer run.
        """
        db, step_dir = _setup_training(tmp_path, monkeypatch, layers=(34, 35))
        # Simulate the torn write: layer 35's directory never landed.
        import shutil
        shutil.rmtree(step_dir / "layer_35_residual")

        with pytest.raises(FinalizeError, match="incomplete"):
            finalize_from_checkpoint(db, "t1")

    def test_missing_training_raises(self, tmp_path, monkeypatch):
        from src.services import training_finalize_service as svc

        monkeypatch.setattr(svc.settings, "data_dir", tmp_path)
        db = FakeSession({"Training": [], "Model": [], "Checkpoint": []})
        with pytest.raises(FinalizeError, match="Training not found"):
            finalize_from_checkpoint(db, "nope")

    def test_no_checkpoints_raises(self, tmp_path, monkeypatch):
        from src.services import training_finalize_service as svc

        monkeypatch.setattr(svc.settings, "data_dir", tmp_path)
        db = FakeSession({
            "Training": [FakeTraining()], "Model": [FakeModel()], "Checkpoint": [],
        })
        with pytest.raises(FinalizeError, match="no on-disk checkpoints"):
            finalize_from_checkpoint(db, "t1")

    def test_stale_layer_dirs_from_a_previous_finalize_are_removed(
        self, tmp_path, monkeypatch
    ):
        """community_format/ must never mix SAEs from two different steps."""
        db, _ = _setup_training(tmp_path, monkeypatch, layers=(34,))
        out = tmp_path / "trainings" / "t1" / "community_format"
        stale = out / "layer_99_residual"
        stale.mkdir(parents=True)
        (stale / "cfg.json").write_text("{}")

        finalize_from_checkpoint(db, "t1")

        assert not stale.exists(), (
            "a layer dir from an earlier finalize survived; sae_manager would "
            "offer it alongside the new SAEs"
        )


class TestDeleteCheckpointFiles:
    def test_removes_file_and_empty_parents(self, tmp_path):
        f = tmp_path / "checkpoint_1000" / "layer_34_residual" / "checkpoint.safetensors"
        f.parent.mkdir(parents=True)
        f.write_bytes(b"x" * 512)

        freed = CheckpointService.delete_checkpoint_files(str(f))

        assert freed == 512
        assert not f.exists()
        assert not f.parent.exists(), "empty layer dir should be removed"
        assert not f.parent.parent.exists(), "empty step dir should be removed"

    def test_keeps_step_dir_when_a_sibling_layer_remains(self, tmp_path):
        step = tmp_path / "checkpoint_1000"
        a = step / "layer_34_residual" / "checkpoint.safetensors"
        b = step / "layer_35_residual" / "checkpoint.safetensors"
        for f in (a, b):
            f.parent.mkdir(parents=True)
            f.write_bytes(b"y" * 10)

        CheckpointService.delete_checkpoint_files(str(a))

        assert not a.parent.exists()
        assert b.exists(), "sibling layer must survive"
        assert step.exists(), "step dir must survive while a sibling remains"

    def test_missing_file_reports_zero(self, tmp_path):
        assert CheckpointService.delete_checkpoint_files(str(tmp_path / "gone")) == 0


class TestExecutePlan:
    POLICY = RetentionPolicy(enabled=True, dry_run=False, keep_best=True)

    def _row(self, tmp_path, cid, step, is_best=False, layer=34):
        f = tmp_path / f"checkpoint_{step}" / f"layer_{layer}_residual" / "checkpoint.safetensors"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"z" * 100)
        return FakeCheckpointRow(cid, step, str(f), is_best=is_best)

    def test_deletes_rows_and_files(self, tmp_path):
        rows = [self._row(tmp_path, "a", 1000), self._row(tmp_path, "b", 2000)]
        db = FakeSession({"Checkpoint": rows})
        plan = PrunePlan(training_id="t1", checkpoint_ids=["a", "b"])

        deleted, freed, failed = _execute_plan(db, plan, self.POLICY)

        assert deleted == 2
        assert freed == 200
        assert {r.id for r in db.deleted} == {"a", "b"}
        for r in rows:
            assert not Path(r.storage_path).exists()

    def test_row_is_committed_only_after_a_successful_unlink(self, tmp_path):
        """Ordering guard.

        Committing the row deletion over a FAILED unlink strands the file
        forever: with no row, no future prune can ever plan it again, and the
        run reports "0.00 GB freed" as if nothing needed doing.
        """
        rows = [self._row(tmp_path, "a", 1000)]
        db = FakeSession({"Checkpoint": rows})
        path = Path(rows[0].storage_path)

        observed = {}
        real_commit = db.commit

        def spy_commit():
            observed["file_existed_at_commit"] = path.exists()
            real_commit()

        db.commit = spy_commit
        _execute_plan(db, PrunePlan(training_id="t1", checkpoint_ids=["a"]), self.POLICY)

        # The row commit must happen only AFTER a successful unlink, so that a
        # failed unlink never leaves a committed row pointing at a live file
        # that no future prune can see.
        assert observed["file_existed_at_commit"] is False
        assert not path.exists()

    def test_respects_keep_best(self, tmp_path):
        """A row promoted to best after planning must be skipped."""
        rows = [self._row(tmp_path, "a", 1000, is_best=True)]
        db = FakeSession({"Checkpoint": rows})

        deleted, freed, failed = _execute_plan(
            db, PrunePlan(training_id="t1", checkpoint_ids=["a"]), self.POLICY
        )

        assert deleted == 0
        assert Path(rows[0].storage_path).exists(), "best checkpoint was deleted"

    def test_keep_best_false_allows_deleting_best(self, tmp_path):
        """With keep_best off, planner and executor must AGREE and delete it.

        Previously the executor skipped best rows unconditionally, so the
        setting was dead and a mixed step could be half-deleted.
        """
        rows = [self._row(tmp_path, "a", 1000, is_best=True)]
        db = FakeSession({"Checkpoint": rows})
        policy = RetentionPolicy(enabled=True, dry_run=False, keep_best=False)

        deleted, _, _ = _execute_plan(
            db, PrunePlan(training_id="t1", checkpoint_ids=["a"]), policy
        )

        assert deleted == 1

    def test_a_promoted_best_row_aborts_its_whole_step(self, tmp_path):
        """Per-STEP selection demands per-STEP execution.

        Skipping only the promoted row would delete the step's other layers and
        leave a checkpoint that cannot be loaded — exactly what step-granularity
        exists to prevent.
        """
        a = self._row(tmp_path, "s1-l34", 1000, is_best=True, layer=34)
        b = self._row(tmp_path, "s1-l35", 1000, layer=35)
        c = self._row(tmp_path, "s2-l34", 2000, layer=34)
        db = FakeSession({"Checkpoint": [a, b, c]})

        deleted, _, _ = _execute_plan(
            db,
            PrunePlan(training_id="t1", checkpoint_ids=["s1-l34", "s1-l35", "s2-l34"]),
            self.POLICY,
        )

        assert deleted == 1, "only the unaffected step should be deleted"
        assert Path(a.storage_path).exists()
        assert Path(b.storage_path).exists(), (
            "sibling layer of a promoted step was deleted -> unloadable checkpoint"
        )
        assert not Path(c.storage_path).exists()

    def test_a_step_is_all_or_nothing_when_a_file_cannot_be_deleted(
        self, tmp_path, monkeypatch
    ):
        """F1: a mid-step unlink failure must keep EVERY row of that step.

        Committing row-by-row left layer 1 deleted while its siblings survived,
        and expected_sae_keys derives the expected layer set from the SURVIVING
        rows — so a later finalize would call that torn step complete and export
        2 of 3 SAEs as a whole run.
        """
        a = self._row(tmp_path, "s1-l34", 1000, layer=34)
        b = self._row(tmp_path, "s1-l35", 1000, layer=35)
        db = FakeSession({"Checkpoint": [a, b]})

        real = CheckpointService.delete_checkpoint_files
        calls = {"n": 0}

        def flaky(path):
            calls["n"] += 1
            if calls["n"] == 2:      # second layer of the step fails
                raise OSError("ENOSPC")
            return real(path)

        monkeypatch.setattr(
            CheckpointService, "delete_checkpoint_files", staticmethod(flaky)
        )

        deleted, freed, failed = _execute_plan(
            db, PrunePlan(training_id="t1", checkpoint_ids=["s1-l34", "s1-l35"]),
            self.POLICY,
        )

        assert deleted == 0, "no row of a partially-failed step may be deleted"
        assert failed == 1
        assert db.deleted == [], (
            "a row was deleted while its sibling survived -> torn step that a "
            "later finalize would treat as complete"
        )

    def test_missing_row_is_skipped(self, tmp_path):
        db = FakeSession({"Checkpoint": []})
        deleted, freed, failed = _execute_plan(
            db, PrunePlan(training_id="t1", checkpoint_ids=["gone"]), self.POLICY
        )
        assert (deleted, freed, failed) == (0, 0, 0)

    def test_row_survives_when_the_file_cannot_be_deleted(self, tmp_path, monkeypatch):
        """A failed unlink must NOT delete the row.

        Committing the row over a failed unlink strands a multi-GB file that no
        future prune can ever plan again (planning is row-driven), while the run
        logs "0.00 GB freed" as though nothing needed doing.
        """
        rows = [self._row(tmp_path, "a", 1000)]
        db = FakeSession({"Checkpoint": rows})

        def boom(_path):
            raise OSError("read-only file system")

        monkeypatch.setattr(
            CheckpointService, "delete_checkpoint_files", staticmethod(boom)
        )

        deleted, freed, failed = _execute_plan(
            db, PrunePlan(training_id="t1", checkpoint_ids=["a"]), self.POLICY
        )

        assert (deleted, freed, failed) == (0, 0, 1)
        assert db.deleted == [], "row was deleted despite the file surviving"


class TestDeleteFilesRaisesOnRealFailure:
    def test_unlink_failure_propagates(self, tmp_path, monkeypatch):
        """An existing-but-undeletable file must raise, not report 0 freed."""
        f = tmp_path / "layer_34_residual" / "checkpoint.safetensors"
        f.parent.mkdir(parents=True)
        f.write_bytes(b"x" * 10)

        import pathlib

        def boom(self):
            raise OSError("EACCES")

        monkeypatch.setattr(pathlib.Path, "unlink", boom)

        with pytest.raises(OSError):
            CheckpointService.delete_checkpoint_files(str(f))


class TestStepFallback:
    """Pins round 1's 'fall back to the newest COMPLETE step' behaviour.

    Previously unpinned: the only incomplete-step test had a single step on
    disk, so `list_checkpoint_steps(...)[:-1]` was empty and the fallback loop
    never executed. Replacing the loop body with `for candidate in []` survived.
    """

    def test_falls_back_to_the_newest_complete_step(self, tmp_path, monkeypatch):
        import shutil

        db, _ = _setup_training(tmp_path, monkeypatch, layers=(34, 35), step=1000)
        # A second, NEWER step that was torn mid-write (only layer 34 landed).
        newer = tmp_path / "trainings" / "t1" / "checkpoints" / "checkpoint_2000"
        _write_ckpt(newer / "layer_34_residual" / "checkpoint.safetensors", 34, step=2000)

        result = finalize_from_checkpoint(db, "t1")

        assert result["checkpoint_step"] == 1000, (
            "must fall back to the older COMPLETE step rather than exporting a "
            "partial newest step"
        )
        assert result["sae_count"] == 2

    def test_explicit_step_is_never_silently_replaced(self, tmp_path, monkeypatch):
        """A caller who names a step must get that step or an error."""
        import shutil

        db, step_dir = _setup_training(tmp_path, monkeypatch, layers=(34, 35), step=1000)
        shutil.rmtree(step_dir / "layer_35_residual")

        with pytest.raises(FinalizeError, match="incomplete"):
            finalize_from_checkpoint(db, "t1", checkpoint_step=1000)


class TestStaleLayerDirPruning:
    """Pins that _prune_stale_layer_dirs SPARES the dirs it is about to write."""

    def test_wanted_dirs_survive(self, tmp_path, monkeypatch):
        db, _ = _setup_training(tmp_path, monkeypatch, layers=(34, 35))

        result = finalize_from_checkpoint(db, "t1")

        out = Path(result["community_format_dir"])
        for layer in (34, 35):
            weights = out / f"layer_{layer}_residual" / "sae_weights.safetensors"
            assert weights.is_file(), (
                f"layer {layer} was pruned away after being written"
            )

    def test_partial_export_does_not_destroy_a_previous_complete_one(
        self, tmp_path, monkeypatch
    ):
        """The data-loss guard.

        Pruning stale dirs BEFORE writing meant a torn step rmtree'd a good
        two-layer export and then wrote back one layer.
        """
        db, _ = _setup_training(tmp_path, monkeypatch, layers=(34, 35), step=1000)
        first = finalize_from_checkpoint(db, "t1")
        out = Path(first["community_format_dir"])
        assert (out / "layer_35_residual").is_dir()

        # A newer torn step appears; finalize again with no explicit step.
        newer = tmp_path / "trainings" / "t1" / "checkpoints" / "checkpoint_2000"
        _write_ckpt(newer / "layer_34_residual" / "checkpoint.safetensors", 34, step=2000)

        second = finalize_from_checkpoint(db, "t1")

        assert second["checkpoint_step"] == 1000
        assert (out / "layer_35_residual").is_dir(), (
            "a previously complete export was destroyed by a partial one"
        )


class TestPolicyClamping:
    """Pins the maximum clamp added in round 1 (previously unexercised)."""

    def test_keep_last_is_clamped_at_both_ends(self):
        from src.services.checkpoint_retention import (
            SETTING_KEEP_LAST, policy_from_values,
        )

        assert policy_from_values({SETTING_KEEP_LAST: "0"}).keep_last == 1
        assert policy_from_values({SETTING_KEEP_LAST: "999999"}).keep_last == 50

    def test_min_age_hours_is_clamped_at_both_ends(self):
        from src.services.checkpoint_retention import (
            SETTING_MIN_AGE_HOURS, policy_from_values,
        )

        assert policy_from_values({SETTING_MIN_AGE_HOURS: "-5"}).min_age_hours == 0
        assert policy_from_values({SETTING_MIN_AGE_HOURS: "999999"}).min_age_hours == 8760

    def test_absent_values_use_conservative_defaults(self):
        from src.services.checkpoint_retention import policy_from_values

        policy = policy_from_values({})
        assert policy.enabled is False
        assert policy.dry_run is True


class TestCompletenessCheckFailsClosed:
    """A DB error during the completeness check must ABORT, not skip the check.

    Returning an empty set on error disables the one guard that stops a torn
    step being exported as a whole run — in response to the single failure mode
    it cannot distinguish from "no metadata recorded". It also leaves the
    session in a failed transaction, so the export gets written and only THEN
    does the caller blow up.
    """

    def test_db_error_raises_instead_of_skipping_the_check(
        self, tmp_path, monkeypatch
    ):
        from sqlalchemy.exc import SQLAlchemyError

        db, _ = _setup_training(tmp_path, monkeypatch, layers=(34, 35))

        rolled_back = {"called": False}

        def boom(model):
            if model.__name__ == "Checkpoint":
                raise SQLAlchemyError("connection reset")
            return FakeQuery(db._by_type.get(model.__name__, []))

        monkeypatch.setattr(db, "query", boom)
        monkeypatch.setattr(
            db, "rollback", lambda: rolled_back.__setitem__("called", True),
            raising=False,
        )

        with pytest.raises(FinalizeError, match="cannot be verified as complete"):
            finalize_from_checkpoint(db, "t1")

        assert rolled_back["called"], (
            "the poisoned session was not rolled back; the caller's next query "
            "would raise PendingRollbackError only after files were written"
        )

    def test_no_community_format_is_written_when_the_check_fails(
        self, tmp_path, monkeypatch
    ):
        from sqlalchemy.exc import SQLAlchemyError

        db, _ = _setup_training(tmp_path, monkeypatch, layers=(34, 35))

        def boom(model):
            if model.__name__ == "Checkpoint":
                raise SQLAlchemyError("connection reset")
            return FakeQuery(db._by_type.get(model.__name__, []))

        monkeypatch.setattr(db, "query", boom)
        monkeypatch.setattr(db, "rollback", lambda: None, raising=False)

        with pytest.raises(FinalizeError):
            finalize_from_checkpoint(db, "t1")

        out = tmp_path / "trainings" / "t1" / "community_format"
        assert not out.exists(), "an export was written despite an unverifiable step"
