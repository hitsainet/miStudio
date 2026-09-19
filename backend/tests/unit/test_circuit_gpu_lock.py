"""One circuit GPU task per GPU, under a per-card lock key (multi-GPU Phase 3).

The circuit guard used to refuse any new circuit GPU run while one was active
anywhere, under one global advisory lock. In per-card mode
``CircuitCaptureService.assert_no_active_gpu_run`` locks the key of each card the
request could use and applies ``circuit_gpu_lock.conflict``. The advisory-lock
tests run against REAL POSTGRES (the unit suite's schema), because a lock key is
only proven by a second connection failing to take it.

MUTATION CONTROLS (2026-09-14; each alone, restored byte-identically, sha256 checked,
`git diff` clean) — all red:
  C1 a named card takes the global key               -> its own key; the other card's key stays free (Postgres)
  C2 a run on the named card is not a conflict       -> the rule; a capture on one card refuses that card (Postgres)
  C3 no capacity limit                               -> four tests, including two runs filling a two-card node
  C4 the service ignores per-card mode               -> the per-card lock and run-row tests (Postgres)
  C5 an endpoint guards without its request          -> each guard call passes the request it checks
"""

import ast
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from src.models.circuit_runs import CircuitCaptureRun
from src.models.steering_record_run import SteeringRecordRun
from src.services import circuit_gpu_lock as L
from src.services.circuit_capture_service import CaptureConflictError, CircuitCaptureService
from src.services.gpu_placement import GpuCard

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
TI = GpuCard(0, TI_UUID, "NVIDIA GeForce RTX 3080 Ti", 12_288, 11_900)
RTX = GpuCard(1, RTX_UUID, "NVIDIA GeForce RTX 3090", 24_576, 21_500)
CARDS = [TI, RTX]
SRC = Path(__file__).resolve().parents[2] / "src"


def run_on(uuid, description="Capture c1 is already running"):
    return L.ActiveRun(description, frozenset({uuid.lower()}))


def unknown(description="Attribution pass on d1 is running"):
    return L.ActiveRun(description)


class TestTheRule:
    def test_nothing_active_admits_anything(self):
        for request in ("auto", "all", TI_UUID):
            assert L.conflict(request, [], 2) is None

    def test_a_card_with_a_run_is_refused_and_the_other_card_is_not(self):
        active = [run_on(TI_UUID)]
        assert "on that GPU" in L.conflict(TI_UUID, active, 2)
        assert L.conflict(RTX_UUID, active, 2) is None

    def test_auto_is_admitted_while_a_card_is_free_and_refused_when_every_card_has_one(self):
        assert L.conflict("auto", [run_on(TI_UUID)], 2) is None
        assert "one per GPU" in L.conflict("auto", [run_on(TI_UUID), unknown()], 2)

    def test_all_waits_for_every_card(self):
        assert L.conflict("all", [unknown()], 2) is not None

    def test_a_run_across_every_card_blocks_everything(self):
        whole = [L.ActiveRun("Capture c9 is already running", whole_node=True)]
        assert L.conflict(RTX_UUID, whole, 2) is not None
        assert L.conflict("auto", whole, 2) is not None

    def test_an_unknown_card_only_counts_toward_capacity(self):
        assert L.conflict(TI_UUID, [unknown()], 2) is None
        assert L.conflict(TI_UUID, [unknown(), unknown("Validation pass on d2 is running")], 2) is not None

    def test_a_single_card_node_is_the_old_one_at_a_time(self):
        assert L.conflict("auto", [unknown()], 1) is not None

    def test_the_binding_of_a_run_row(self):
        assert L.binding("auto", None, [TI_UUID, RTX_UUID]) == (frozenset({TI_UUID.lower(), RTX_UUID.lower()}), False)
        assert L.binding(RTX_UUID, TI_UUID, None) == (frozenset({TI_UUID.lower()}), False)
        assert L.binding("all") == (frozenset(), True)
        assert L.binding(RTX_UUID) == (frozenset({RTX_UUID.lower()}), False)
        assert L.binding("auto") == (frozenset(), False)
        assert L.binding(None) == (frozenset(), False)


class TestKeys:
    def test_a_named_card_takes_only_its_own_key(self):
        assert L.keys_for(TI_UUID, CARDS) == [L.lock_key(TI_UUID)]
        assert L.lock_key(TI_UUID) == L.lock_key(TI_UUID.lower()) != L.lock_key(RTX_UUID)

    def test_auto_and_all_take_every_cards_key_in_order(self):
        expected = sorted([L.lock_key(TI_UUID), L.lock_key(RTX_UUID)])
        assert L.keys_for("auto", CARDS) == L.keys_for("all", CARDS) == expected

    @pytest.mark.parametrize("inventory", [[TI, RTX], [RTX, TI]], ids=["nvml-order", "reversed"])
    def test_the_keys_are_taken_in_one_order_whatever_order_the_cards_are_listed(self, inventory):
        """Two transactions that take overlapping keys in different orders can deadlock.

        REVIEW ROUND 1 (2026-09-14). The test above passed with `sorted()` REMOVED from
        `keys_for` (control C6 survived the whole file): its fixture lists the 3080 Ti
        first, and the 3080 Ti's key happens to be the smaller, so the unsorted list
        already was the sorted one. This test lists the cards both ways.
        Negative control C6 re-run against it: red (the `reversed` case).
        """
        keys = L.keys_for("auto", inventory)
        assert keys == sorted(keys) == sorted([L.lock_key(TI_UUID), L.lock_key(RTX_UUID)])

    def test_no_card_is_the_global_key(self):
        assert L.keys_for(TI_UUID, []) == [L.GLOBAL_LOCK_KEY]

    def test_keys_fit_a_postgres_bigint(self):
        assert 0 < L.lock_key(RTX_UUID) < 2**63


@pytest.fixture
def pg(async_engine, monkeypatch):
    """Two independent sync sessions over the unit suite's database, per-card mode on the two-card node."""
    sync_url = async_engine.url.set(
        drivername=async_engine.url.drivername.replace("+asyncpg", "").replace("+psycopg", ""))
    engine = create_engine(sync_url)
    make = sessionmaker(bind=engine)
    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
    monkeypatch.setattr(L, "list_cards", lambda: list(CARDS))
    sessions = []

    def session():
        sessions.append(make())
        return sessions[-1]

    yield session
    for s in sessions:
        s.rollback()
        s.close()
    cleanup = make()
    cleanup.query(SteeringRecordRun).delete()
    cleanup.query(CircuitCaptureRun).delete()
    cleanup.commit()
    cleanup.close()
    engine.dispose()


def _free(session, uuid) -> bool:
    return bool(session.execute(text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": L.lock_key(uuid)}).scalar())


class TestTheLockIsPerCard:
    def test_a_named_card_locks_its_key_and_not_the_other_cards(self, pg):
        holder = pg()
        CircuitCaptureService.assert_no_active_gpu_run(holder, gpu_request=TI_UUID)   # transaction left open

        other = pg()
        assert _free(other, TI_UUID) is False
        assert _free(other, RTX_UUID) is True

    def test_an_auto_request_locks_every_card(self, pg):
        holder = pg()
        CircuitCaptureService.assert_no_active_gpu_run(holder, gpu_request="auto")

        other = pg()
        assert (_free(other, TI_UUID), _free(other, RTX_UUID)) == (False, False)


class TestAgainstRunRows:
    def test_a_capture_on_one_card_refuses_that_card_and_admits_the_other(self, pg):
        db = pg()
        db.add(CircuitCaptureRun(status="running", manifest={}, gpu_request=TI_UUID, gpu_uuid=TI_UUID))
        db.commit()

        with pytest.raises(CaptureConflictError, match="on that GPU"):
            CircuitCaptureService.assert_no_active_gpu_run(db, gpu_request=TI_UUID)
        db.rollback()
        CircuitCaptureService.assert_no_active_gpu_run(db, gpu_request=RTX_UUID)
        db.rollback()
        CircuitCaptureService.assert_no_active_gpu_run(db, gpu_request="auto")
        db.rollback()

    def test_two_active_runs_fill_a_two_card_node(self, pg):
        db = pg()
        db.add(CircuitCaptureRun(status="running", manifest={}, gpu_request=TI_UUID, gpu_uuid=TI_UUID))
        db.add(SteeringRecordRun(status="pending", artifact_kind="circuit", gpu_request="auto"))
        db.commit()

        with pytest.raises(CaptureConflictError, match="one per GPU"):
            CircuitCaptureService.assert_no_active_gpu_run(db, gpu_request="auto")

    def test_single_mode_still_runs_one_circuit_task_at_a_time(self, pg, monkeypatch):
        db = pg()
        db.add(CircuitCaptureRun(status="running", manifest={}, gpu_request=TI_UUID, gpu_uuid=TI_UUID))
        db.commit()
        monkeypatch.setattr(settings, "gpu_worker_mode", "single")

        with pytest.raises(CaptureConflictError, match="already running"):
            CircuitCaptureService.assert_no_active_gpu_run(db, gpu_request=RTX_UUID)


#: Endpoint module -> how many guard calls it makes.
GUARDED_ENDPOINTS = {
    "api/v1/endpoints/circuits.py": 4,
    "api/v1/endpoints/circuit_discovery.py": 3,
    "api/v1/endpoints/circuit_validation.py": 2,
}


class TestEveryEndpointGuardsItsOwnRequest:
    @pytest.mark.parametrize("module", sorted(GUARDED_ENDPOINTS))
    def test_each_guard_call_passes_the_request_it_checks(self, module):
        tree = ast.parse((SRC / module).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", None) == "assert_no_active_gpu_run"]
        assert len(calls) == GUARDED_ENDPOINTS[module]
        for call in calls:
            passed = {kw.arg: kw.value for kw in call.keywords}
            assert isinstance(passed.get("gpu_request"), ast.Name) and passed["gpu_request"].id == "gpu_request", (
                f"{module}:{call.lineno} guards without the request it is checking")
