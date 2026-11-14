"""A view id becomes a directory name under `data_dir`; a traversal id must be refused, never deleted
(CodeQL py/path-injection #69-#72 on the mirror: `purge_view`, `purge_snapshots`).

The purge route only reaches these with an id matching an existing archived row (server-generated
`pmd_<hex>`), so this is defence in depth: the directory helpers confine the id themselves."""

from __future__ import annotations

import os
import shutil

import pytest

from src.core.config import settings
from src.services import rules_evaluation, view_columns
from src.utils.file_utils import resolve_under

HOSTILE = ["..", ".", "", "../victim", "../../etc", "a/b", "/etc", "..\\victim", "pmd_x\x00"]


@pytest.fixture
def world(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(settings, "data_dir", data, raising=False)
    # This test must stay harmless even when the code under test is broken (a mutation, a regression):
    # rmtree may only ever delete strictly inside `data`; anything else is recorded and NOT deleted.
    real_rmtree = shutil.rmtree
    escaped: list[str] = []

    def guarded_rmtree(path, *args, **kwargs):
        target = os.path.realpath(path)
        if not target.startswith(os.path.realpath(data) + os.sep):
            escaped.append(target)
            return None
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", guarded_rmtree)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("must survive")
    # A victim INSIDE data_dir but outside both roots: `..` from either root lands on data_dir itself.
    (data / "other.txt").write_text("must survive")
    yield data, victim, escaped


@pytest.mark.parametrize("name", HOSTILE)
def test_resolve_under_refuses_traversal(tmp_path, name):
    with pytest.raises(ValueError):
        resolve_under(tmp_path, name)


def test_resolve_under_refuses_a_symlink_out_of_the_base(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, base / "pmd_link")
    with pytest.raises(ValueError):
        resolve_under(base, "pmd_link")


def test_resolve_under_accepts_a_plain_id(tmp_path):
    assert resolve_under(tmp_path, "pmd_0123456789ab") == tmp_path.resolve() / "pmd_0123456789ab"


@pytest.mark.parametrize("name", HOSTILE)
def test_purge_view_refuses_traversal_and_deletes_nothing(world, name):
    data, victim, escaped = world
    (data / rules_evaluation.CACHE_DIR).mkdir(parents=True)
    with pytest.raises(ValueError):
        rules_evaluation.purge_view(name)
    assert (victim / "keep.txt").read_text() == "must survive"
    assert (data / "other.txt").read_text() == "must survive"
    assert (data / rules_evaluation.CACHE_DIR).is_dir()
    assert escaped == []


@pytest.mark.parametrize("name", HOSTILE)
def test_purge_snapshots_refuses_traversal_and_deletes_nothing(world, name):
    data, victim, escaped = world
    (data / view_columns.SNAPSHOT_DIR).mkdir(parents=True)
    with pytest.raises(ValueError):
        view_columns.purge_snapshots(name)
    with pytest.raises(ValueError):
        view_columns.snapshot_files(name)
    assert (victim / "keep.txt").read_text() == "must survive"
    assert (data / "other.txt").read_text() == "must survive"
    assert (data / view_columns.SNAPSHOT_DIR).is_dir()
    assert escaped == []


def test_a_plain_view_id_is_still_purged(world):
    data, _victim, escaped = world
    cache = data / rules_evaluation.CACHE_DIR / "pmd_0123456789ab"
    snaps = data / view_columns.SNAPSHOT_DIR / "pmd_0123456789ab"
    cache.mkdir(parents=True)
    snaps.mkdir(parents=True)
    assert view_columns.snapshot_files("pmd_0123456789ab") == {"probe_monitor_views": str(snaps.resolve())}
    assert rules_evaluation.purge_view("pmd_0123456789ab") == str(cache.resolve())
    assert view_columns.purge_snapshots("pmd_0123456789ab") == str(snaps.resolve())
    assert not cache.exists() and not snaps.exists()
    assert escaped == []
