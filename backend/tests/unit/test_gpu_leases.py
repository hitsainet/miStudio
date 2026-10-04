"""GPU leases keep two jobs off one card (multi-GPU Phase 3, decision D6).

Phase 3 runs one job per card at once, and an Auto job's card is chosen when a
worker is free (decision 5, 2026-09-14). NVML free memory cannot stop two jobs
landing on one card — a job just placed has not allocated yet — so the choice is
recorded as a lease the moment it is made.

REAL POSTGRES, deliberately. A lease is refused by a primary key and guarded by a
row lock; a fake session reproduces neither, and would pass while two jobs took
the same card. The fixture creates its own database and ONLY the lease table, and
fails — never skips — when Postgres is unreachable: tests here once skipped
silently and three "verified" controls asserted nothing.

MUTATION CONTROLS (2026-09-14; each alone on services/gpu_leases.py, restored
byte-identically and checked by sha256) — all six went red:
  L1 no refusal when another job holds a card   -> held card refused, split all-or-none
  L2 renew revives an expired lease              -> a renewal never revives
  L3 release deletes any holder's leases         -> release frees only the holder's own
  L4 expired leases not cleared before taking    -> an expired lease frees the card
  L5 a concurrent insert reported as taken       -> taken between read and insert
  L6 refusal without its rollback                -> a refused job does not keep the row locked
     (survived until that test was written: nothing else held a refused session open)
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from src.models.gpu_lease import GpuLease
from src.services import gpu_leases as L
from tests.unit.gpu_lease_db import lease_engine

TI = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
T0 = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
DB_NAME = "mistudio_test_gpu_leases"


@pytest.fixture(scope="module")
def engine():
    # ``MISTUDIO_GPU_LEASE_TEST_DB`` names the database when set, as for every other
    # lease test module (gpu_lease_db): this module used a FIXED name, so two agents
    # running it at once dropped and recreated each other's table mid-test.
    eng = lease_engine(DB_NAME)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    """A factory of independent sessions — each one is a separate job's connection."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM gpu_leases"))
    make = sessionmaker(bind=engine, expire_on_commit=False)
    opened = []

    def new():
        opened.append(make())
        return opened[-1]

    yield new
    for s in opened:
        s.close()


def test_a_free_card_is_taken(session):
    assert L.acquire(session(), [RTX], "training:a", now=T0) is True
    assert L.live_leases(session(), now=T0) == {RTX: "training:a"}


def test_a_held_card_is_refused_to_another_job(session):
    assert L.acquire(session(), [TI], "training:a", now=T0)
    assert L.acquire(session(), [TI], "extraction:b", now=T0) is False
    assert L.live_leases(session(), now=T0) == {TI: "training:a"}


def test_a_split_takes_every_card_or_none(session):
    assert L.acquire(session(), [TI], "training:a", now=T0)

    assert L.acquire(session(), [TI, RTX], "extraction:split", now=T0) is False

    leases = L.live_leases(session(), now=T0)
    assert leases == {TI: "training:a"}, "a refused split kept the card it could take"


def test_the_same_job_renews_rather_than_blocks_itself(session):
    assert L.acquire(session(), [RTX], "training:a", now=T0, ttl_seconds=60)
    assert L.acquire(session(), [RTX], "training:a", now=T0 + timedelta(seconds=50), ttl_seconds=60)
    # Renewed at +50 s: still held at +100 s, which the first lease alone would not be.
    assert L.live_leases(session(), now=T0 + timedelta(seconds=100)) == {RTX: "training:a"}


def test_an_expired_lease_frees_the_card(session):
    assert L.acquire(session(), [RTX], "training:dead-worker", now=T0, ttl_seconds=600)
    later = T0 + timedelta(seconds=601)

    assert L.live_leases(session(), now=later) == {}
    assert L.acquire(session(), [RTX], "extraction:b", now=later) is True
    assert L.live_leases(session(), now=later) == {RTX: "extraction:b"}


def test_a_renewal_never_revives_an_expired_lease(session):
    assert L.acquire(session(), [RTX], "training:a", now=T0, ttl_seconds=600)
    later = T0 + timedelta(seconds=601)

    assert L.renew(session(), "training:a", now=later) == 0
    assert L.live_leases(session(), now=later) == {}


def test_a_live_lease_is_renewed(session):
    assert L.acquire(session(), [TI, RTX], "extraction:split", now=T0, ttl_seconds=600)
    assert L.renew(session(), "extraction:split", now=T0 + timedelta(seconds=300), ttl_seconds=600) == 2
    assert L.live_leases(session(), now=T0 + timedelta(seconds=800)) == {TI: "extraction:split", RTX: "extraction:split"}


def test_a_release_frees_only_the_holders_own_cards(session):
    assert L.acquire(session(), [TI], "training:a", now=T0)
    assert L.acquire(session(), [RTX], "extraction:b", now=T0)

    assert L.release(session(), "training:a") == 1
    assert L.release(session(), "training:a", uuids=[RTX]) == 0

    assert L.live_leases(session(), now=T0) == {RTX: "extraction:b"}


def test_a_refused_job_does_not_keep_the_holders_row_locked(session):
    """A refused acquire has already read the holder's row FOR UPDATE.

    Without its rollback that lock stays in the refused worker's open session —
    a worker keeps its session for the whole task — and the job that really
    holds the card cannot renew or release it until that session closes.
    MUTATION CONTROL (2026-09-14): drop the rollback on a refusal -> this test
    fails on the lock timeout; nothing else in this file noticed.
    """
    holder, refused, other = session(), session(), session()
    assert L.acquire(holder, [TI], "training:a", now=T0)
    assert L.acquire(refused, [TI], "extraction:b", now=T0) is False   # `refused` stays open

    other.execute(text("SET lock_timeout = '2s'"))
    assert L.release(other, "training:a") == 1


def test_a_card_taken_between_the_read_and_the_insert_is_refused_not_raised(session, monkeypatch):
    """The primary key is the last word: a rival that commits the card after this
    job's read but before its insert wins, and this job gets False with nothing taken."""
    job, rival = session(), session()
    real_add = job.add

    def rival_commits_first(obj):
        rival.add(GpuLease(gpu_uuid=obj.gpu_uuid, holder="training:rival", acquired_at=T0,
                           heartbeat_at=T0, expires_at=T0 + timedelta(hours=1)))
        rival.commit()
        real_add(obj)

    monkeypatch.setattr(job, "add", rival_commits_first)

    assert L.acquire(job, [RTX], "extraction:b", now=T0) is False
    assert L.live_leases(session(), now=T0) == {RTX: "training:rival"}


def test_a_lease_needs_a_card():
    with pytest.raises(ValueError):
        L.acquire(None, [], "training:a")
