"""FR-47 (a) at fixture level (Feature 040 S2; FTASKS 14.3): the pilot diagnosis reproduced through the app.

Synthetic arrays with the RECORDED counts of `pmr_bbe9a3608802`'s selected probe (research
`2026-10-22_ssn_diagnosis.md`): threshold 10.126; `ssn-ctx-id-test` 729 positives + 547 negatives with 71
positives above the bar (9.7%), `ssn-ctx-ood-unseen-carriers` 828 + 672 with 49 (5.9%). Through the REAL
breakdown route: one group over a constant column gives the headline recall, and adding `kind` through
the column update (FR-14) gives the per-kind table of diagnosis §3. The live reading on the deployed arrays
is FTASKS 15.6 (a dated record, CPU only).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.config import settings
from tests.support.detector_fixtures import (
    Arrays,
    dataset_row,
    declare,
    evaluation_row,
    model_row,
    run_and_probe,
    view_row,
    write_split,
)

API = "/api/v1/probe-monitors"
BAR = 10.126

# (kind, label, n, flagged) per view — the diagnosis §3 table, flagged counts chosen to its percentages and
# summing to the §1 totals.
ID_TABLE = [("pos", 1, 625, 56), ("bad_check", 1, 94, 11), ("mixed_delim", 1, 10, 4),
            ("masked", 0, 20, 1), ("wrong_len", 0, 73, 5), ("mixed_delim", 0, 71, 1), ("adj_letters", 0, 76, 1),
            ("med_nophrase", 0, 95, 1), ("med_far", 0, 82, 0), ("lookalike", 0, 130, 0)]
OOD_TABLE = [("pos", 1, 683, 37), ("bad_check", 1, 133, 9), ("mixed_delim", 1, 12, 3),
             ("masked", 0, 31, 1), ("wrong_len", 0, 92, 2), ("mixed_delim", 0, 87, 1), ("adj_letters", 0, 93, 3),
             ("med_nophrase", 0, 128, 0), ("med_far", 0, 89, 0), ("lookalike", 0, 152, 0)]


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


def split_and_arrays(table):
    kinds, labels, scores = [], [], []
    for kind, label, n, flagged in table:
        for i in range(n):
            kinds.append(kind)
            labels.append(label)
            scores.append(BAR + 1.0 if i < flagged else BAR - 5.0)
    total = len(kinds)
    # every split row is evaluated, at its own position, in a shuffled order (order must not matter)
    order = np.random.default_rng(1).permutation(total)
    arrays = Arrays(scores=np.asarray(scores)[order], positions=order, n_scored=np.full(total, 40),
                    labels=np.asarray(labels, dtype=np.int8)[order])
    return {"text": [str(i) for i in range(total)], "label": [str(x) for x in labels], "kind": kinds,
            "set": ["all"] * total}, arrays


async def test_the_diagnosis_recalls_and_the_per_kind_table(client, async_session, data_dir):
    model = await model_row(async_session)
    readings = {}
    for name, table in (("id", ID_TABLE), ("ood", OOD_TABLE)):
        columns, arrays = split_and_arrays(table)
        ds = await dataset_row(async_session, write_split(data_dir, columns))
        view = await view_row(async_session, ds, role="eval",
                              distribution="in_distribution" if name == "id" else "out_of_distribution")
        await declare(async_session, view, ["set"])
        train = await view_row(async_session, ds, role="train")
        run, probe = await run_and_probe(async_session, model, train, eval_views=[view], threshold=BAR,
                                         scope="all", target_fpr=0.01, n_negatives=540)
        evaluation = await evaluation_row(async_session, probe, run, view, arrays)
        await async_session.commit()
        url = f"{API}/probes/{probe.id}/evaluations/{evaluation.id}/breakdown"
        headline = (await client.get(url, params={"by_column": "set"})).json()["groups"][0]
        readings[name] = headline
        added = await client.patch(f"{API}/datasets/{view.id}/columns", json={"group_columns": ["set", "kind"]})
        assert added.status_code == 200, added.text
        per_kind = (await client.get(url, params={"by_column": "kind", "limit": 50})).json()["groups"]
        got = {g["value"]: g for g in per_kind}
        for kind, label, n, flagged in table:
            side = got[kind]["positives" if label == 1 else "negatives"]
            assert (side["n"], side["flagged"]) == (n, flagged), (name, kind, label)
    assert round(readings["id"]["positives"]["recall"] * 100, 1) == 9.7
    assert round(readings["ood"]["positives"]["recall"] * 100, 1) == 5.9
    assert round(readings["id"]["negatives"]["fpr"] * 100, 1) == 1.6
    assert round(readings["ood"]["negatives"]["fpr"] * 100, 1) == 1.0
