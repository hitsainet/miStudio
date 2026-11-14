"""OSD-36b — orphaned checkpoint directories, reclaimed behind the SAME dry-run gate.

Row-driven planning cannot see a step directory whose rows are gone, so those bytes
were never reclaimed. The scan is the filesystem side, and its guards are the point:
nothing outside `trainings/<id>/checkpoints/checkpoint_<step>`, nothing a row
references, nothing in an active training, nothing young.
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.services import checkpoint_retention as cr

OLD = time.time() - 7 * 24 * 3600


class FakeDB:
    def __init__(self, storage_paths, statuses):
        self.storage_paths = storage_paths
        self.statuses = statuses

    def query(self, *cols):
        outer = self

        class _Q:
            def all(self_inner):
                if cols[0] is cr.Checkpoint.storage_path:
                    return [(p,) for p in outer.storage_paths]
                return list(outer.statuses.items())

        return _Q()


def _step(root: Path, training: str, step: int, age=OLD, size=1000) -> Path:
    d = root / "trainings" / training / "checkpoints" / f"checkpoint_{step}"
    (d / "layer_11_residual").mkdir(parents=True)
    f = d / "layer_11_residual" / "checkpoint.safetensors"
    f.write_bytes(b"x" * size)
    (d / "training_state.pt").write_bytes(b"y" * size)
    for p in (f, d / "training_state.pt", d / "layer_11_residual", d):
        os.utime(p, (age, age))
    return d


@pytest.fixture
def estate(tmp_path):
    kept = _step(tmp_path, "train_a", 100)
    rowless = _step(tmp_path, "train_a", 200)
    gone = _step(tmp_path, "train_gone", 50)
    young = _step(tmp_path, "train_a", 300, age=time.time())
    active = _step(tmp_path, "train_live", 10)
    export = tmp_path / "trainings" / "train_gone" / "community_format"
    export.mkdir(parents=True)
    (export / "sae.safetensors").write_bytes(b"z")
    os.utime(export / "sae.safetensors", (OLD, OLD))
    db = FakeDB(
        storage_paths=[str(kept / "layer_11_residual" / "checkpoint.safetensors")],
        statuses={"train_a": "completed", "train_live": "running"},
    )
    return tmp_path, db, dict(kept=kept, rowless=rowless, gone=gone, young=young,
                              active=active, export=export)


def _find(root, db, **policy):
    return cr.find_orphaned_checkpoint_dirs(
        db, root, cr.RetentionPolicy(**policy), resolve=lambda p: p
    )


def test_it_finds_exactly_the_orphans(estate):
    root, db, d = estate
    found = {Path(o.path): o for o in _find(root, db)}
    assert set(found) == {d["rowless"], d["gone"]}
    assert found[d["gone"]].reason == "training row deleted"
    assert found[d["rowless"]].reason == "step has no checkpoint rows"
    assert found[d["rowless"]].size_bytes == 2000


def test_a_row_written_as_a_data_path_still_protects_its_step(estate):
    root, db, d = estate
    db.storage_paths = ["/data/trainings/train_a/checkpoints/checkpoint_200/layer_11_residual/x"]
    found = cr.find_orphaned_checkpoint_dirs(
        db, root, cr.RetentionPolicy(),
        resolve=lambda p: str(p).replace("/data", str(root), 1),
    )
    assert d["rowless"] not in {Path(o.path) for o in found}


def test_min_age_spares_a_step_being_written(estate):
    root, db, d = estate
    assert d["young"] not in {Path(o.path) for o in _find(root, db)}
    # And it is the AGE that spared it: with no minimum age the same directory is found.
    assert d["young"] in {Path(o.path) for o in _find(root, db, min_age_hours=0)}


def test_reclaim_deletes_only_step_dirs_and_never_the_export(estate):
    root, db, d = estate
    orphans = _find(root, db)
    removed, freed, failures = cr.reclaim_orphans(orphans, root)
    assert removed == 2 and failures == []
    assert not d["rowless"].exists() and not d["gone"].exists()
    assert d["kept"].exists() and d["young"].exists() and d["active"].exists()
    assert d["export"].exists(), "community_format is an exported SAE, never a checkpoint"


def test_reclaim_refuses_a_path_outside_the_checkpoint_tree(estate, tmp_path):
    root, _db, d = estate
    stray = cr.OrphanDir(path=str(d["export"]), training_id="train_gone", step=0,
                         reason="forged", size_bytes=1)
    removed, _freed, failures = cr.reclaim_orphans([stray], root)
    assert removed == 0 and failures and d["export"].exists()


class TestTheSweepHonoursDryRun:
    def _run(self, monkeypatch, root, db, dry_run):
        from src.core.config import settings
        from src.workers import prune_checkpoints as mod
        from src.workers.prune_checkpoints import prune_checkpoints_task

        class _Ctx:
            def __enter__(self):
                return db

            def __exit__(self, *a):
                return False

        policy = cr.RetentionPolicy(enabled=True, dry_run=dry_run)
        monkeypatch.setattr(mod, "load_policy", lambda _db: policy)
        monkeypatch.setattr(mod, "iter_prunable_trainings", lambda _db: [])
        monkeypatch.setattr(settings, "data_dir", root)
        monkeypatch.setattr(prune_checkpoints_task, "get_db", lambda: _Ctx(), raising=False)
        monkeypatch.setattr(
            mod, "find_orphaned_checkpoint_dirs",
            lambda db_, data_dir, pol: cr.find_orphaned_checkpoint_dirs(
                db_, data_dir, pol, resolve=lambda p: p),
        )
        return prune_checkpoints_task.run()

    def test_dry_run_reports_and_deletes_nothing(self, monkeypatch, estate):
        root, db, d = estate
        result = self._run(monkeypatch, root, db, dry_run=True)
        assert result["orphaned_dirs"] == 2 and result["orphaned_bytes"] == 4000
        assert result["orphaned_dirs_removed"] == 0
        assert d["rowless"].exists() and d["gone"].exists()

    def test_live_run_reclaims(self, monkeypatch, estate):
        root, db, d = estate
        result = self._run(monkeypatch, root, db, dry_run=False)
        assert result["orphaned_dirs_removed"] == 2 and result["orphaned_bytes_freed"] == 4000
        assert not d["rowless"].exists() and d["kept"].exists()
