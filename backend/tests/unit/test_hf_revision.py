"""Hub ref → commit, and the error mapping a download request answers with (035 FR-2, D31, D41).

MUTATION CONTROLS (recorded in 0xcc/reviews/035_implementation_controls_2026-10-18.md):
  C1  the GatedRepoError branch moved below RepositoryNotFoundError → the gated test fails
      (huggingface-hub 1.28 makes GatedRepoError a SUBCLASS of RepositoryNotFoundError, so a
      gated dataset would be reported as nonexistent).
"""
from __future__ import annotations

import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

# The Hub's own HTTP module (httpx on hub 1.x, httpx2 on 2.x): a fake raising plain httpx errors
# kept these tests green while every real 2.x timeout escaped the mapping as a 500.
from huggingface_hub.utils import httpx

from src.services import hf_revision
from src.services.hf_revision import HUB_TIMEOUT_S, HubResolutionError, resolve_dataset_commit
from tests.support.fake_hub import COMMIT_A, COMMIT_B, FakeHub, install

TOKEN = "hf_SECRETSECRETSECRET"


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/api/x"))


def _http(cls, status):
    return lambda: cls("boom", response=_response(status))


class TestItResolves:
    def test_the_fake_is_the_thing_called(self, monkeypatch):
        """A patch at the wrong attribute would let the real Hub answer silently."""
        hub = install(monkeypatch, FakeHub(refs={"v1.0": COMMIT_B}))
        assert resolve_dataset_commit("org/ds", "v1.0", TOKEN) == COMMIT_B
        assert hub.calls == [
            {"repo_id": "org/ds", "revision": "v1.0", "timeout": HUB_TIMEOUT_S, "token": TOKEN}
        ]

    def test_no_ref_asks_for_the_default_branch(self, monkeypatch):
        hub = install(monkeypatch)
        assert resolve_dataset_commit("org/ds", None, None) == COMMIT_A
        assert hub.calls[0]["revision"] is None

    def test_the_timeout_is_30_seconds(self):
        assert HUB_TIMEOUT_S == 30.0

    @pytest.mark.parametrize("sha", [None, "", "main", "A" * 40, "a" * 39])
    def test_an_answer_without_a_commit_is_refused(self, monkeypatch, sha):
        install(monkeypatch, FakeHub(default=sha))
        with pytest.raises(HubResolutionError) as caught:
            resolve_dataset_commit("org/ds", None, None)
        assert caught.value.kind == "unreachable" and caught.value.status_code == 502


@pytest.mark.parametrize(
    "raises, kind, status",
    [
        (_http(GatedRepoError, 403), "forbidden", 403),
        (_http(RepositoryNotFoundError, 404), "not_found", 422),
        (_http(RepositoryNotFoundError, 401), "not_found", 422),
        (_http(RevisionNotFoundError, 404), "not_found", 422),
        (lambda: HFValidationError("bad id"), "not_found", 422),
        (_http(HfHubHTTPError, 403), "forbidden", 403),
        (_http(HfHubHTTPError, 401), "forbidden", 403),
        (_http(HfHubHTTPError, 400), "not_found", 422),
        (_http(HfHubHTTPError, 500), "unreachable", 502),
        (_http(HfHubHTTPError, 503), "unreachable", 502),
        (lambda: httpx.ReadTimeout("slow"), "unreachable", 502),
        (lambda: httpx.ConnectError("refused"), "unreachable", 502),
        (lambda: ConnectionError("offline"), "unreachable", 502),
    ],
)
def test_every_hub_failure_maps_to_one_answer(monkeypatch, raises, kind, status):
    install(monkeypatch, FakeHub(raises=raises))
    with pytest.raises(HubResolutionError) as caught:
        resolve_dataset_commit("org/ds", "v1", TOKEN)
    assert (caught.value.kind, caught.value.status_code) == (kind, status)
    assert "org/ds" in caught.value.message
    assert TOKEN not in caught.value.message, "the token leaked into an error message"


def test_a_gated_repo_is_forbidden_not_missing(monkeypatch):
    """C1. GatedRepoError IS-A RepositoryNotFoundError; the order of the branches decides."""
    assert issubclass(GatedRepoError, RepositoryNotFoundError)
    install(monkeypatch, FakeHub(raises=_http(GatedRepoError, 403)))
    with pytest.raises(HubResolutionError) as caught:
        resolve_dataset_commit("org/gated", None, None)
    assert caught.value.kind == "forbidden"
    assert "token cannot read org/gated" in caught.value.message


def test_an_unknown_kind_is_a_programming_error():
    with pytest.raises(ValueError):
        HubResolutionError("teapot", "x")


def test_the_request_helper_resolves_the_token_like_the_task(monkeypatch):
    """TI1: the same precedence the download task uses (request → env → stored setting)."""
    hub = install(monkeypatch)
    monkeypatch.setattr(
        "src.services.huggingface_sae_service.resolve_hf_token", lambda explicit: f"resolved:{explicit}"
    )
    assert hf_revision.resolve_for_request("org/ds", None, "from-request") == COMMIT_A
    assert hub.calls[0]["token"] == "resolved:from-request"


class TestTheRealHubClientsErrorsAreMapped:
    """The REAL `HfApi` and the Hub's REAL HTTP client, with the network replaced by a transport.

    Every test above raises an exception the test author chose, so it proves the mapping for THAT
    type and nothing about the type a real Hub call raises. huggingface-hub 2.0 replaced httpx with
    httpx2, whose exceptions are not httpx's and not OSError: the module caught
    `httpx.TimeoutException`, a 2.x timeout escaped as a 500, and the fakes (raising plain httpx)
    stayed green. Here nothing is faked above the socket, so the type is whatever the Hub raises.

    CONTROL (2026-10-18, hub 2.2.0): `except hub_httpx.…` → `except httpx.…` in hf_revision.py
    makes both cases below fail with an unmapped httpx2 exception.
    """

    @pytest.fixture
    def hub_transport(self):
        from huggingface_hub import set_client_factory
        from huggingface_hub.utils._http import default_client_factory

        def install_transport(handler):
            set_client_factory(
                lambda: httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
            )

        yield install_transport
        set_client_factory(default_client_factory)

    @pytest.mark.parametrize(
        "error", [httpx.ReadTimeout, httpx.ConnectError], ids=["timeout", "refused"]
    )
    def test_a_transport_failure_is_unreachable(self, hub_transport, error):
        calls = []

        def handler(request):
            calls.append(str(request.url))
            raise error("no answer", request=request)

        hub_transport(handler)
        with pytest.raises(HubResolutionError) as caught:
            resolve_dataset_commit("org/ds", "v1", TOKEN)
        # hub 2.x may also fetch its `/api/agent-harnesses` registry (best effort, errors
        # swallowed) before the real call, so look for the dataset request among the calls.
        assert any("/api/datasets/org/ds" in url for url in calls), calls
        assert (caught.value.kind, caught.value.status_code) == ("unreachable", 502)
        assert TOKEN not in caught.value.message

    def test_a_real_answer_resolves_through_the_real_client(self, hub_transport):
        def handler(request):
            return httpx.Response(200, json={"id": "org/ds", "sha": COMMIT_B}, request=request)

        hub_transport(handler)
        assert resolve_dataset_commit("org/ds", "v1", TOKEN) == COMMIT_B
