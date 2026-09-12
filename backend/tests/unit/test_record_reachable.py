"""The steered-transcript recorder must be reachable AND wired (repo gate).

Behaviour-based: register the real tools against a fake client and assert the
exact POST path + payload; assert the GPU guard + cleanup see the record marker;
assert the Celery task is registered. A capability is not shipped until a test
FAILS when its wiring is removed.
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


class _FakeClient:
    def __init__(self):
        self.calls = []

    async def post(self, path, json_body=None):
        self.calls.append(("POST", path, json_body))
        return {"ok": True}

    async def get(self, path, **params):
        self.calls.append(("GET", path, params))
        return {"manifests": [{"kind": "steering_samples", "id": "vman_a"},
                              {"kind": "calibration", "id": "vman_b"}]}

    async def patch(self, *a, **k):
        return {}

    async def delete(self, *a, **k):
        return {}


def _register(fake):
    from mcp.server.fastmcp import FastMCP

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import circuits as mod
    mcp = FastMCP("test")
    mod.register(mcp, fake, MCPSettings(allow_anonymous=True))
    return mcp


class TestTheRecorderToolsAreReachable:
    def test_both_registered_on_the_built_server(self):
        names = _tools()
        assert "record_steering_samples" in names
        assert "get_steering_samples" in names

    def test_record_posts_the_right_path_and_payload(self):
        fake = _FakeClient()
        mcp = _register(fake)
        asyncio.run(mcp.call_tool("record_steering_samples", {
            "artifact": {"kind": "circuit", "circuit_id": "crc_x"},
            "dials": [0.4, 0.6], "prompts": ["p1"], "max_tokens": 80, "seed": 3}))
        assert len(fake.calls) == 1
        method, path, body = fake.calls[0]
        assert method == "POST" and path == "/circuits/steering-samples"
        assert body["artifact"] == {"kind": "circuit", "circuit_id": "crc_x"}
        assert body["dials"] == [0.4, 0.6] and body["prompts"] == ["p1"]
        assert body["max_tokens"] == 80 and body["seed"] == 3

    def test_get_by_circuit_id_filters_to_steering_samples_kind(self):
        fake = _FakeClient()
        mcp = _register(fake)
        asyncio.run(mcp.call_tool("get_steering_samples", {"circuit_id": "crc_x"}))
        method, path, params = fake.calls[0]
        assert method == "GET" and path == "/validation-manifests"
        assert params.get("circuit_id") == "crc_x"

    def test_get_by_manifest_id_fetches_that_manifest(self):
        # R2: the manifest_id branch is the PRIMARY path for cluster/feature
        # records (not circuit-linked). It must be wired, not just present.
        fake = _FakeClient()
        mcp = _register(fake)
        asyncio.run(mcp.call_tool("get_steering_samples", {"manifest_id": "vman_x"}))
        method, path, _params = fake.calls[0]
        assert method == "GET" and path == "/validation-manifests/vman_x"

    def test_get_with_both_args_errors_without_calling(self):
        fake = _FakeClient()
        mcp = _register(fake)
        asyncio.run(mcp.call_tool("get_steering_samples",
                                  {"circuit_id": "crc_x", "manifest_id": "vman_x"}))
        assert fake.calls == []   # "exactly one" — refuses, issues no request

    def test_every_parameter_is_described(self):
        for name in ("record_steering_samples", "get_steering_samples"):
            tool = _tools()[name]
            props = (tool.inputSchema or {}).get("properties", {})
            undescribed = [p for p, s in props.items() if not s.get("description")]
            assert not undescribed, f"{name} params lack descriptions: {undescribed}"


class TestTheRecorderSharesTheSingleGPUGuard:
    def test_guard_checks_the_record_marker(self):
        from src.services import circuit_capture_service as mod
        src = inspect.getsource(mod.CircuitCaptureService.assert_no_active_gpu_run)
        assert "SteeringRecordRun" in src, (
            "assert_no_active_gpu_run does not check steering_record_runs — a "
            "record job can race a capture/calibration and OOM")

    def test_cleanup_reclaims_a_stuck_record(self):
        from src.workers import cleanup_stuck_circuit_runs as mod
        src = inspect.getsource(mod)
        assert "SteeringRecordRun" in src, (
            "the stuck-run cleanup does not reclaim a crashed record job — it "
            "would wedge the single-GPU guard")


class TestTheTaskIsRegistered:
    def test_autodiscovered(self):
        from src.core import celery_app as mod
        assert "src.workers.circuit_record_tasks" in inspect.getsource(mod)

    def test_task_name_stable(self):
        from src.workers.circuit_record_tasks import run_circuit_record
        assert run_circuit_record.name == (
            "src.workers.circuit_record_tasks.run_circuit_record")

    def test_status_write_rolls_back_first(self):
        """A DB-error failure leaves the session aborted; _set_status must roll
        back before writing or the marker stays set and wedges the guard."""
        from src.workers import circuit_record_tasks as mod

        class _Row:
            status = "running"
            error = None

        class _AbortedThenOK:
            def __init__(self):
                self.rolled_back = False
                self.committed = False
                self._row = _Row()

            def rollback(self):
                self.rolled_back = True

            def query(self, _m):
                db = self
                if not db.rolled_back:
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
        mod._set_status(db, "srr_x", "failed", error="boom")
        assert db.rolled_back and db._row.status == "failed" and db.committed
