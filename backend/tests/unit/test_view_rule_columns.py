"""A view's text column and producer rule columns (Feature 040 S3; FR-12, FR-14, FR-34; 040-T22, T23; FTASKS 20.2, 20.3).

Through the REAL registration and `PATCH …/columns` routes over a real split:
* the manifest's `extensions` rule-set listing is RECORDED at registration (040-T23);
* producer `rule_columns` must name a set the listing carries (040-T22) and columns the split has; their
  verdict columns are snapshotted by position (040-T11), the text is never copied;
* each field is declared once (additive): a change is refused `column_not_additive`;
* the drift check refuses a split whose row count or commit moved (`split_changed`);
* a purge deletes the view's cached rule verdicts with it (20.3).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.config import settings
from src.services import rules_evaluation, view_columns
from src.services.detector_next import DetectorRefused
from tests.support.detector_fixtures import dataset_row, write_split
from tests.support.random_behaviour import rng_for
from tests.unit.test_probe_dataset_manifest import manifest

API = "/api/v1/probe-monitors"
N = 40
SET_DIGEST = "c" * 64


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


def listed_manifest(name="qv-set-x", version=2, digest=SET_DIGEST):
    out = manifest({"train": N})
    out["extensions"] = {view_columns.RULE_SET_LISTING_KEY: {"recorded": True, "rule_sets": [
        {"name": name, "version": version, "sha256": digest, "behaviour_key": "some-key", "use": "veto",
         "path": f"rule_sets/{name}@{version}.json"}]}}
    return out


async def _dataset(async_session, data_dir, rng):
    cols = {"text": [f"row {i}" for i in range(N)], "body": [f"other {i}" for i in range(N)],
            "label": [str(i % 2) for i in range(N)], "rs_hit": [i % 3 == 0 for i in range(N)],
            "rs_vetoed": [i % 5 == 0 for i in range(N)]}
    ds = await dataset_row(async_session, write_split(data_dir, cols))
    await async_session.commit()
    return ds, cols


def body(ds, **extra):
    out = {"name": f"view-{ds.id.hex[:6]}", "dataset_id": str(ds.id), "input_column": "text", "label_column": "label",
           "label_mapping": {"1": "positive", "0": "negative"}, "role": "eval", "split": "train"}
    out.update(extra)
    return out


RULES = {"rule_match": "rs_hit", "rule_veto": "rs_vetoed", "rule_set_id": "qv-set-x@2", "rule_set_digest": SET_DIGEST}


async def test_registration_records_the_listing_and_snapshots_the_verdict_columns(client, async_session, data_dir,
                                                                                   request):
    ds, cols = await _dataset(async_session, data_dir, rng_for(request.node.nodeid))
    made = await client.post(f"{API}/datasets", json=body(ds, dataset_version_manifest=listed_manifest(),
                                                          rule_columns=RULES, text_column="body"))
    assert made.status_code == 201, made.text
    view = made.json()
    assert view["rule_columns"] == RULES and view["text_column"] == "body"
    assert view["rule_set_listing"]["rule_sets"][0]["sha256"] == SET_DIGEST
    snap = view_columns.load(view["column_snapshots"]["rs_hit"], "rs_hit")
    assert [snap.values[c] for c in np.asarray(snap.codes).tolist()] == cols["rs_hit"]
    assert "body" not in view["column_snapshots"] and "text" not in view["column_snapshots"]


async def test_rule_columns_need_the_listing(client, async_session, data_dir, request):
    ds, _ = await _dataset(async_session, data_dir, rng_for(request.node.nodeid))
    response = await client.post(f"{API}/datasets", json=body(ds, rule_columns=RULES))
    assert response.status_code == 422
    assert response.json()["detail"]["rule"] == "requires_rule_set_listing"


async def test_an_unlisted_set_is_refused(client, async_session, data_dir, request):
    ds, _ = await _dataset(async_session, data_dir, rng_for(request.node.nodeid))
    other = dict(RULES, rule_set_digest="d" * 64)
    response = await client.post(f"{API}/datasets", json=body(ds, dataset_version_manifest=listed_manifest(),
                                                               rule_columns=other))
    assert response.status_code == 422 and response.json()["detail"]["code"] == "rule_set_not_listed"


@pytest.mark.parametrize("field,value", [("rule_columns", dict(RULES, rule_match="nope")), ("text_column", "nope")])
async def test_an_absent_column_is_refused(client, async_session, data_dir, request, field, value):
    ds, _ = await _dataset(async_session, data_dir, rng_for(request.node.nodeid))
    response = await client.post(f"{API}/datasets", json=body(ds, dataset_version_manifest=listed_manifest(),
                                                              **{field: value}))
    assert response.status_code == 422 and response.json()["detail"]["code"] == "column_absent"


async def test_patch_declares_once_and_refuses_a_change(client, async_session, data_dir, request):
    ds, _ = await _dataset(async_session, data_dir, rng_for(request.node.nodeid))
    view = (await client.post(f"{API}/datasets", json=body(ds, dataset_version_manifest=listed_manifest()))).json()
    first = await client.patch(f"{API}/datasets/{view['id']}/columns", json={"text_column": "body", "rule_columns": RULES})
    assert first.status_code == 200, first.text
    assert first.json()["added"] == ["text_column", "rule_columns"]
    same = await client.patch(f"{API}/datasets/{view['id']}/columns", json={"text_column": "body"})
    assert same.status_code == 200 and same.json()["added"] == []
    changed = await client.patch(f"{API}/datasets/{view['id']}/columns", json={"text_column": "text"})
    assert changed.status_code == 422 and changed.json()["detail"]["code"] == "column_not_additive"
    history = await client.get(f"{API}/datasets/{view['id']}")
    assert history.json()["rule_columns"] == RULES


async def test_the_drift_check_refuses_a_moved_split(async_session, data_dir, request):
    from types import SimpleNamespace

    ds, _ = await _dataset(async_session, data_dir, rng_for(request.node.nodeid))
    view = SimpleNamespace(id="pmd_x", split=None, counts={"positive": 20, "negative": 21}, dataset_version=None)
    with pytest.raises(DetectorRefused) as caught:
        rules_evaluation.open_view_split(ds, view)
    assert caught.value.code == "split_changed" and caught.value.details["rows_now"] == N
    view.counts = {"positive": 20, "negative": 20}
    assert len(rules_evaluation.open_view_split(ds, view)) == N
    moved = SimpleNamespace(**{**vars(view), "dataset_version": {"commit": "a" * 40}})
    ds.revision = "b" * 40
    with pytest.raises(DetectorRefused) as caught:
        rules_evaluation.open_view_split(ds, moved)
    assert caught.value.details["commit_registered"] == "a" * 40


async def test_purge_deletes_the_views_cached_verdicts(client, async_session, data_dir, request):
    ds, _ = await _dataset(async_session, data_dir, rng_for(request.node.nodeid))
    view = (await client.post(f"{API}/datasets", json=body(ds))).json()
    cached = rules_evaluation.cache_root() / view["id"] / "x"
    cached.mkdir(parents=True)
    (cached / "rule_match.npy").write_bytes(b"x")
    from src.core.clock import utc_now
    from src.models.rules_evaluation import RuleVerdictCache

    async_session.add(RuleVerdictCache(view_id=view["id"], rule_set_digest="a" * 64, engine_digest="b" * 64,
                                       reading_digest="c" * 64, n_rows=1, bytes=1, path=str(cached),
                                       created_at=utc_now(), last_used_at=utc_now()))
    await async_session.commit()
    await client.post(f"{API}/datasets/{view['id']}/archive", json={})
    purged = (await client.post(f"{API}/datasets/purge", json={"ids": [view["id"]], "reason": "x"})).json()
    result = purged["results"][0]
    assert result["outcome"] == "purged"
    assert result["rows_removed"]["rule_verdicts"] == str(rules_evaluation.cache_root() / view["id"])
    assert not (rules_evaluation.cache_root() / view["id"]).exists()
    # Review R1 (S3): and its index rows (the success report reads verdicts through them)
    from sqlalchemy import select

    from src.models.rules_evaluation import RuleVerdictCache

    left = (await async_session.execute(select(RuleVerdictCache).where(RuleVerdictCache.view_id == view["id"]))).all()
    assert left == []
