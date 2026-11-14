"""A failing endpoint logs its exception and returns a generic reason, never the exception text.

CodeQL py/stack-trace-exposure #27/#28/#54 (security alerts 2026-10-17). Each test makes the
endpoint's guarded operation raise an exception whose message carries a SECRET (the shape of
a database DSN), then requires the secret to be absent from the response and present in the
log — so the information is kept for the operator and withheld from the caller.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

SECRET = "postgresql://np_user:hunter2@10.0.0.9:5432/neuronpedia"


def _boom(*_args, **_kwargs):
    raise RuntimeError(f"connection refused: {SECRET}")


def _assert_withheld_and_logged(response, caplog):
    assert SECRET not in json.dumps(response, default=str)
    assert "hunter2" not in json.dumps(response, default=str)
    assert any(
        SECRET in (r.exc_text or "") or SECRET in r.getMessage() or (
            r.exc_info and SECRET in str(r.exc_info[1])
        )
        for r in caplog.records
    ), "the exception must still reach the backend log"


def test_analysis_cleanup(monkeypatch, caplog):
    import torch

    from src.api.v1.endpoints import features

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    with caplog.at_level(logging.ERROR):
        response = asyncio.run(features.cleanup_analysis_gpu())
    assert response["error"]
    _assert_withheld_and_logged(response, caplog)


def test_local_neuronpedia_status(monkeypatch, caplog):
    from src.api.v1.endpoints import neuronpedia

    monkeypatch.setattr(neuronpedia.settings, "neuronpedia_local_db_url", SECRET)
    monkeypatch.setattr(neuronpedia, "get_neuronpedia_local_push_service", _boom)
    with caplog.at_level(logging.ERROR):
        response = asyncio.run(neuronpedia.get_local_neuronpedia_status())
    assert response["connected"] is False and response["error"]
    # The configured URL is never part of this response either way; only the error could leak.
    assert SECRET not in str(response["error"])
    response = {k: v for k, v in response.items() if k != "public_url"}
    _assert_withheld_and_logged(response, caplog)


def test_enter_steering_mode(monkeypatch, caplog):
    from src.api.v1.endpoints import steering
    from src.services import gpu_dispatch

    monkeypatch.setattr(gpu_dispatch, "per_card_workers", lambda: False)
    monkeypatch.setattr(steering, "_is_steering_worker_running", lambda: (False, None))
    monkeypatch.setattr(steering, "_spawn_steering_worker", _boom)
    with caplog.at_level(logging.ERROR):
        response = asyncio.run(steering.enter_steering_mode())
    assert response["success"] is False and response["message"]
    _assert_withheld_and_logged(response, caplog)
