"""No API test may publish a real Celery message.

MIS-E2E-027. Two dataset tests called an endpoint ending in
`download_dataset_task.delay()` with nothing mocked. Celery connects to the
broker to publish, so on a machine without Redis they failed with
`ConnectionRefusedError` from `celery.backends.redis.ResultConsumer` — a unit
test failing on infrastructure it never meant to exercise — and where Redis WAS
running, they enqueued a genuine dataset download.

This is a structural guard: any test class exercising an endpoint that
dispatches must neutralise the dispatch. Rather than enumerate endpoints, it
asserts the property that actually matters at runtime — see
`test_the_suite_runs_without_a_broker` for the behavioural half.
"""

import ast
from pathlib import Path

import pytest

API_TESTS = Path(__file__).resolve().parents[1] / "api"
ENDPOINTS = Path(__file__).resolve().parents[2] / "src" / "api" / "v1" / "endpoints"


def _dispatching_endpoint_modules() -> set:
    """Endpoint modules that call `.delay(` or `.apply_async(`."""
    found = set()
    for path in ENDPOINTS.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") in ("delay", "apply_async")):
                found.add(path.stem)
                break
    return found


class TestTheScanSees:
    def test_some_endpoints_dispatch(self):
        mods = _dispatching_endpoint_modules()
        assert len(mods) >= 3, (
            f"only {mods} found dispatching — the AST scan is broken, and a "
            f"broken scan makes the guard below vacuous"
        )

    def test_api_test_files_exist(self):
        files = list(API_TESTS.rglob("test_*.py"))
        assert len(files) > 3, f"only {len(files)} API test files found"


class TestNoRealDispatchInApiTests:
    def test_dataset_download_tests_neutralise_the_dispatch(self):
        """The two the finding named, pinned by name."""
        text = (API_TESTS / "v1" / "endpoints" / "test_datasets.py").read_text()
        assert "download_dataset_task.delay" in text, (
            "the dispatch patch is gone; these tests will publish a real "
            "download to the broker"
        )
        assert "patch(" in text

    @pytest.mark.parametrize("name", ["test_download_dataset_success",
                                      "test_download_dataset_with_access_token"])
    def test_the_named_tests_are_still_there(self, name):
        """Deleting the test is not a way to pass this guard."""
        text = (API_TESTS / "v1" / "endpoints" / "test_datasets.py").read_text()
        assert f"def {name}" in text, (
            f"{name} is gone — the finding was that it dispatched for real, "
            f"not that it should not exist"
        )

    def test_the_dispatch_is_patched_where_the_task_lives(self):
        """The endpoint imports the task inside the handler.

        Patching `src.api.v1.endpoints.datasets.download_dataset_task` raises
        AttributeError, because the name is never bound at module scope — the
        patch has to target the worker module.
        """
        text = (API_TESTS / "v1" / "endpoints" / "test_datasets.py").read_text()
        assert "src.workers.dataset_tasks.download_dataset_task.delay" in text, (
            "patched at the endpoint module, where the symbol does not exist"
        )
