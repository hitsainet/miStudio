"""Shape-A: a cancelled spot-check STOPS between turns and keeps what it judged, labelled partial (US-5).

The operator's route writes the flag (`request_cancel("probe_spot_check", …)`) while the judge is answering
the first turn; the run must judge no further turn. Control C-SPOTPOLL: remove the per-turn poll → red.
"""
from __future__ import annotations

import pytest

from src.services import probe_spot_check as spot
from tests.unit.test_behaviour_catalogue import sync_db  # noqa: F401
from tests.unit.test_probe_spot_check import FakeJudge, _started, queued, wire  # noqa: F401


@pytest.mark.asyncio
async def test_a_cancel_mid_run_stops_between_turns_and_keeps_the_judged_turn(client, sync_db, wire, queued):
    from src.core.cancellation import request_cancel
    from src.models.probe_spot_check import ProbeSpotCheck, ProbeSpotCheckItem

    check_id, *_ = await _started(client, sync_db, wire, queued)

    def answer(messages):
        request_cancel("probe_spot_check", check_id)
        return '{"answer": "yes"}'

    judge = FakeJudge(answer)
    sync_db.expire_all()
    result = spot.execute(sync_db, check_id, judge_client=judge.client())
    assert result["status"] == "cancelled"
    assert len(judge.requests) == 1, "a turn was judged after the cancel"
    row = sync_db.query(ProbeSpotCheck).filter(ProbeSpotCheck.id == check_id).one()
    assert (row.status, row.partial) == ("cancelled", True)
    assert row.metrics["partial"] is True
    assert sync_db.query(ProbeSpotCheckItem).filter(ProbeSpotCheckItem.spot_check_id == check_id).count() == 1


@pytest.mark.asyncio
async def test_the_cancel_route_writes_the_flag(client, sync_db, wire, queued):
    check_id, *_ = await _started(client, sync_db, wire, queued)
    response = await client.post(f"/api/v1/probe-monitors/spot-checks/{check_id}/cancel")
    assert response.status_code == 202 and response.json()["status"] == "cancelling"
    assert (await client.get(f"/api/v1/probe-monitors/spot-checks/{check_id}")).json()["status"] == "cancelled"
