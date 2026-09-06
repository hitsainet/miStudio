"""calibrate_circuit_strength must be reachable AND wired to the real endpoint.

The repo's shipping gate (a capability is not shipped until a test FAILS when its
wiring is removed): assert the tool is in the LIVE registry the server builds,
that it posts to the calibration endpoint (not something that looks right and
does nothing), and that the GPU guard + cleanup actually see calibration — the
correctness requirements without which a calibration could race and OOM, or wedge
the single-GPU guard forever.
"""

import asyncio
import inspect
import os


def _tools():
    os.environ.setdefault("MILLM_API_URL", "http://millm.test")
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server
    mcp, _c = build_server(
        MCPSettings(tool_categories="circuits", allow_anonymous=True), stdio=True)
    return {t.name: t for t in asyncio.run(mcp.list_tools())}


class TestTheCalibrationToolIsReachable:
    def test_it_is_registered_on_the_built_server(self):
        assert "calibrate_circuit_strength" in _tools(), (
            "calibrate_circuit_strength is not reachable from the built server — "
            "the calibration capability does not exist for any agent"
        )

    def test_it_actually_POSTS_the_right_path_and_payload(self):
        """Behaviour, not a source grep (the repo's gate: assert payload AND
        call count). Register the tools against a fake client that records the
        call, invoke the tool, and assert what it really sent."""
        calls = []

        class _FakeClient:
            async def post(self, path, json_body=None):
                calls.append((path, json_body))
                return {"ok": True}

            async def get(self, *a, **k):
                return {}

            async def patch(self, *a, **k):
                return {}

            async def delete(self, *a, **k):
                return {}

        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.config import MCPSettings
        from src.mcp_server.tools import circuits as mod

        mcp = FastMCP("test")
        mod.register(mcp, _FakeClient(), MCPSettings(allow_anonymous=True))
        asyncio.run(mcp.call_tool("calibrate_circuit_strength", {
            "circuit_id": "crc_x", "step_budget": 12, "probe_count": 2,
            "margin": 0.2, "seed": 5,
            "judge_endpoint": "http://j/v1", "judge_model": "m"}))

        assert len(calls) == 1, "the tool must issue exactly one POST"
        path, body = calls[0]
        assert path == "/circuits/crc_x/calibration", (
            f"posted to {path}, not the calibration endpoint")
        # The payload the endpoint needs — wrong keys silently no-op the run.
        assert body["step_budget"] == 12 and body["probe_count"] == 2
        assert body["margin"] == 0.2 and body["seed"] == 5
        assert body["judge_endpoint"] == "http://j/v1"
        assert body["judge_model"] == "m"

    def test_every_parameter_is_described(self):
        """The discoverability gate: an agent sees only names + descriptions."""
        tool = _tools()["calibrate_circuit_strength"]
        props = (tool.inputSchema or {}).get("properties", {})
        assert props, "calibrate_circuit_strength exposes no parameters?"
        undescribed = [p for p, s in props.items() if not s.get("description")]
        assert not undescribed, (
            f"calibrate_circuit_strength parameters lack descriptions: {undescribed}"
        )

    def test_the_description_explains_the_two_thresholds_and_the_clamp(self):
        d = (_tools()["calibrate_circuit_strength"].description or "").lower()
        assert "onset" in d and "cliff" in d, (
            "the description does not name both thresholds an agent must reason about"
        )
        assert "clamp" in d or "intensity_range" in d, (
            "the description does not say the served dial gets clamped — the whole point"
        )

    def test_reproduce_calibration_is_registered_and_posts_correctly(self):
        assert "reproduce_calibration" in _tools(), (
            "reproduce_calibration is unreachable — FPRD §8.5 reproduction has no "
            "agent-facing surface")
        calls = []

        class _FakeClient:
            async def post(self, path, json_body=None):
                calls.append((path, json_body))
                return {"ok": True}

            async def get(self, *a, **k):
                return {}

            async def patch(self, *a, **k):
                return {}

            async def delete(self, *a, **k):
                return {}

        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.config import MCPSettings
        from src.mcp_server.tools import circuits as mod

        mcp = FastMCP("test")
        mod.register(mcp, _FakeClient(), MCPSettings(allow_anonymous=True))
        asyncio.run(mcp.call_tool("reproduce_calibration", {"manifest_id": "vman_x"}))
        assert calls == [
            ("/circuits/calibration-manifests/vman_x/reproduce", None)], (
            f"reproduce_calibration posted {calls}, not the reproduce endpoint")


class TestCalibrationSharesTheSingleGPUGuard:
    """Calibration loads a model and holds the GPU like faithfulness. If the
    guard/cleanup don't see it, a calibration can race a capture (OOM) or a
    crashed one wedges the guard forever."""

    def test_the_gpu_guard_checks_calibration_status(self):
        from src.services import circuit_capture_service as mod
        src = inspect.getsource(mod.CircuitCaptureService.assert_no_active_gpu_run)
        assert "calibration_status" in src, (
            "assert_no_active_gpu_run does not check calibration_status — a "
            "calibration can race a capture/faithfulness and OOM"
        )

    def test_cleanup_reclaims_a_stuck_calibration(self):
        from src.workers import cleanup_stuck_circuit_runs as mod
        src = inspect.getsource(mod)
        assert "calibration_status" in src and "calibration_task_id" in src, (
            "the stuck-run cleanup does not reclaim a crashed calibration — it "
            "would wedge the single-GPU guard for every circuit task"
        )


class TestTheStatusWriteSurvivesAnAbortedTransaction:
    """P3: if the run failed on a DB error, the session is aborted; the status
    write must roll back FIRST or the in-flight marker is never cleared and the
    single-GPU guard wedges for 60 minutes."""

    def test_set_status_rolls_back_then_writes(self):
        from src.workers import circuit_calibration_tasks as mod

        class _Row:
            calibration_status = "running"

        class _AbortedThenOK:
            def __init__(self):
                self.rolled_back = False
                self.committed = False
                self._row = _Row()

            def rollback(self):
                self.rolled_back = True   # clears the aborted state

            def query(self, _m):
                db = self
                if not db.rolled_back:
                    # Before rollback, any query on an aborted txn raises.
                    raise RuntimeError("current transaction is aborted")

                class _Q:
                    def filter(self, *a, **k):
                        return self

                    def first(self_inner):
                        return db._row
                return _Q()

            def commit(self):
                self.committed = True

        db = _AbortedThenOK()
        mod._set_status(db, "crc_x", "failed")
        assert db.rolled_back, "must roll back the aborted transaction first"
        assert db._row.calibration_status == "failed"
        assert db.committed, "the status write must land after rollback"


class TestTheTaskIsRegisteredWithCelery:
    """The task module must be in Celery's autodiscovery list, or the worker
    never registers run_circuit_calibration and every .delay() raises
    NotRegistered — the capability is unreachable in production while every
    unit test that imports the module directly still passes."""

    def test_the_calibration_task_module_is_autodiscovered(self):
        from src.core import celery_app as mod
        src = inspect.getsource(mod)
        assert "src.workers.circuit_calibration_tasks" in src, (
            "circuit_calibration_tasks is not in Celery autodiscovery — the "
            "worker will not register the task and every dispatch will fail"
        )

    def test_the_task_name_matches_what_the_endpoint_dispatches(self):
        """The endpoint calls run_circuit_calibration.delay; the registered name
        must be importable and stable."""
        from src.workers.circuit_calibration_tasks import run_circuit_calibration
        assert run_circuit_calibration.name == (
            "src.workers.circuit_calibration_tasks.run_circuit_calibration")
