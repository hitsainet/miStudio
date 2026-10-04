"""The monitor never answers for a card nobody asked about.

With two GPUs (RTX 3080 Ti at index 0, RTX 3090 at index 1 since 2026-09-13)
a default of GPU 0 is a guess, and /system/all's "fall back to GPU 0 if the id
is invalid" showed one card's numbers under another card's id.
"""

import types
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.v1.endpoints import system
from src.workers import gpu_watchdog_task


def _to_dict(label):
    return types.SimpleNamespace(to_dict=lambda: {"card": label})


@pytest.fixture
def client(monkeypatch):
    gpu_service = MagicMock()
    gpu_service.is_available.return_value = True
    gpu_service.get_device_count.return_value = 2
    gpu_service.get_gpu_metrics.side_effect = lambda index: _to_dict(f"metrics-{index}")
    gpu_service.get_gpu_info.side_effect = lambda index: _to_dict(f"info-{index}")
    gpu_service.get_gpu_processes.side_effect = lambda index: []

    system_service = MagicMock()
    for name in ("get_system_metrics", "get_network_rates", "get_disk_rates"):
        getattr(system_service, name).return_value = _to_dict(name)
    system_service.get_disk_usage.return_value = []

    monkeypatch.setattr(system, "get_gpu_monitor_service", lambda: gpu_service)
    monkeypatch.setattr(system, "get_system_monitor_service", lambda: system_service)

    app = FastAPI()
    app.include_router(system.router)
    return TestClient(app)


class TestPerCardEndpointsNeedACard:
    @pytest.mark.parametrize("path", ["/system/gpu-metrics", "/system/gpu-info", "/system/gpu-processes"])
    def test_gpu_id_is_required(self, client, path):
        assert client.get(path).status_code == 422

    def test_the_named_card_is_the_one_reported(self, client):
        response = client.get("/system/gpu-metrics", params={"gpu_id": 1})

        assert response.status_code == 200
        assert response.json()["metrics"] == {"card": "metrics-1"}


class TestAllMonitoringData:
    def test_an_invalid_card_is_an_error_not_gpu_0(self, client):
        response = client.get("/system/all", params={"gpu_id": 5})

        assert response.status_code == 400
        assert "Invalid GPU ID: 5" in response.json()["detail"]

    def test_a_named_card_keeps_the_single_card_shape(self, client):
        gpu = client.get("/system/all", params={"gpu_id": 1}).json()["gpu"]

        assert gpu["selected_gpu_id"] == 1
        assert gpu["metrics"] == {"card": "metrics-1"}
        assert gpu["info"] == {"card": "info-1"}

    def test_no_card_named_reports_every_card(self, client):
        gpu = client.get("/system/all").json()["gpu"]

        assert gpu["selected_gpu_id"] is None
        assert [card["gpu_id"] for card in gpu["cards"]] == [0, 1]
        assert [card["metrics"] for card in gpu["cards"]] == [{"card": "metrics-0"}, {"card": "metrics-1"}]


class TestWatchdogAttribution:
    def test_a_process_on_an_unlisted_card_is_skipped_not_blamed_on_gpu_0(self):
        apps = "101, python, GPU-aaaa, 2048\n202, python, GPU-unknown, 512\n"
        cards = "GPU-aaaa, 1\n"

        def run(args, **kwargs):
            stdout = apps if "--query-compute-apps=pid,process_name,gpu_uuid,used_memory" in args else cards
            return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        with patch("subprocess.run", side_effect=run):
            processes = gpu_watchdog_task.get_gpu_processes()

        assert [(p.pid, p.gpu_id) for p in processes] == [(101, 1)]
