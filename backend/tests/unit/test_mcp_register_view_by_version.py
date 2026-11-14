"""`register_probe_dataset(dataset_version_id=...)`: name the version, miStudio fetches its manifest
(ENH-006 D116 G2).

`dataset_version_manifest` makes the caller pass the whole document through the call: 9 KB and about
150 columns for a small version. On 2026-11-02 that was skipped, so a view went live with
`dataset_version: null` and its row counts were checked by hand. A check a caller skips because it is
heavy is not a check.

**The property that matters is the one that goes quiet.** With miDataworks absent or broken, the
registration must REFUSE, never fall through to a view that looks like the others and has verified
nothing. So every failure path below counts the POSTs to miStudio's route and requires zero.

Everything real except miDataworks' socket: the tool on the server `build_server` builds, reached over
the protocol; the real `MiWorkbenchClient` into the real route and a real Arrow dataset; the real
`DataworksClient` (its stamp, its error unwrapping, its non-JSON refusal) over a recording transport.

miDataworks stays optional (suite rule). No test here imports it; without `dataset_version_id` nothing
touches it, and without `MIDATAWORKS_API_URL` the tool still registers plain views.

Controls are recorded in `0xcc/reviews/mcp_gaps_controls_2026-11-02.md`.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from unittest.mock import MagicMock, patch

import httpx
import pytest
from mcp import Client
from sqlalchemy import func, select

from src.mcp_server.config import MCPSettings
from src.mcp_server.server import build_server
from src.mcp_server.tools import howto, probe_monitors
from src.models.dataset import DatasetStatus
from src.models.probe_monitor import ProbeMonitorDataset
from tests.support.fake_hub import COMMIT_A, FakeHub, install
from tests.support.next_calls import live_input_schemas
from tests.unit.test_probe_dataset_manifest import arrow_dict, dataset_id, manifest  # noqa: F401 - fixtures
from tests.unit.test_reachability import EXPECTED_CALLS
from tests.support.cross_repo import require_cross_repo

TOOL = "register_probe_dataset"
VERSION = "c09048b2-a223-43db-a609-f8d7b357a6d3"
DW_URL = "http://midataworks.test"
MANIFEST_PATH = f"/api/v1/versions/{VERSION}/handoff-manifest"


def view_args(dataset, **extra):
    return {"name": "stakes-test", "dataset_id": dataset, "input_column": "text", "label_column": "label",
            "label_mapping": {"yes": "positive", "no": "negative"}, "split": "test", **extra}


class RecordingTransport(httpx.AsyncBaseTransport):
    """Records what reaches miStudio's backend, then lets the real app answer."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[tuple[str, str, dict]] = []

    async def handle_async_request(self, request):
        self.calls.append((request.method, request.url.path, json.loads(request.content or b"{}")))
        return await self.inner.handle_async_request(request)


class FakeDataworks:
    """miDataworks' socket. `respond` answers, or raises as a connection failure does."""

    def __init__(self, respond):
        self.respond = respond
        self.requests: list[httpx.Request] = []
        self.clients: list = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    def install(self, monkeypatch):
        real = probe_monitors._dataworks_client

        def make(url):
            built = real(url)
            built._http = httpx.AsyncClient(  # noqa: SLF001 - the production client, over a test socket
                transport=httpx.MockTransport(self.handler), base_url=url
            )
            self.clients.append(built)
            return built

        monkeypatch.setattr(probe_monitors, "_dataworks_client", make)
        return self


def serves(document):
    return FakeDataworks(lambda request: httpx.Response(200, json=document))


@pytest.fixture
def world(client, monkeypatch):
    """A tool caller on the built server, with miStudio's route real and its calls recorded."""

    def build(dataworks_url: str = DW_URL, categories: str = "probe_monitors,read"):
        monkeypatch.delenv("MIDATAWORKS_API_URL", raising=False)
        settings = MCPSettings(tool_categories=categories, allow_anonymous=True, _env_file=None,
                               midataworks_api_url_override=dataworks_url)
        mcp, studio = build_server(settings, stdio=True)
        backend = RecordingTransport(client._transport)  # noqa: SLF001
        studio._http = httpx.AsyncClient(  # noqa: SLF001
            transport=backend, base_url="http://test/api/v1",
            headers=dict(studio._http.headers),  # noqa: SLF001 - keeps the agent header
        )

        async def call(tool=TOOL, /, **arguments):
            async with Client(mcp) as session:
                return await session.call_tool(tool, arguments)

        return call, backend

    return build


def answer(result) -> dict:
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)


def text_of(result) -> str:
    return " ".join(block.text for block in result.content)


async def views_in(session) -> int:
    return (await session.execute(select(func.count()).select_from(ProbeMonitorDataset))).scalar_one()


def posts(backend) -> list:
    return [c for c in backend.calls if c[0] == "POST"]


class TestTheLiveRegistry:
    def test_the_parameter_is_on_the_tool_the_deployed_manifest_builds(self):
        from tests.unit.test_reachability import TestTheDEPLOYMENTEnablesWhatTheCodeRegisters as T

        deployed = ",".join(sorted(T()._deployed_categories()))
        mcp, _ = build_server(MCPSettings(tool_categories=deployed, allow_anonymous=True, _env_file=None),
                              stdio=True)
        tool = next(t for t in asyncio.run(mcp.list_tools()) if t.name == TOOL)
        assert {"dataset_version_id", "dataset_version_manifest"} <= set(tool.input_schema["properties"])
        assert "dataset_version_id" not in tool.input_schema["required"]

    def test_the_contract_and_howto_read_the_same_registry(self):
        assert "dataset_version_id" in live_input_schemas()[TOOL]["properties"]


class TestNamingAVersion:
    @pytest.mark.asyncio
    async def test_the_manifest_is_fetched_once_and_is_what_the_route_verifies(
        self, world, dataset_id, monkeypatch  # noqa: F811
    ):
        document = manifest()
        dw = serves(document).install(monkeypatch)
        call, backend = world()

        view = answer(await call(**view_args(dataset_id, dataset_version_id=VERSION)))

        assert [(r.method, r.url.path) for r in dw.requests] == [("GET", MANIFEST_PATH)]
        # The identity every miDataworks request carries; a request without it reads as a person's.
        assert dw.requests[0].headers["X-Dataworks-Agent"] == "agent:miworkbench-mcp"

        sent = posts(backend)
        assert [(c[0], c[1]) for c in sent] == [("POST", "/api/v1/probe-monitors/datasets")]
        assert sent[0][2]["dataset_version_manifest"] == document
        # The route does not know this name: it takes the document, which is the one contract.
        assert "dataset_version_id" not in sent[0][2]

        record = view["dataset_version"]
        assert record is not None, "the view went in unverified"
        assert (record["version_id"], record["split"], record["manifest_rows"], record["downloaded_rows"]) == (
            "dv_1", "test", 80, 80)

    @pytest.mark.asyncio
    async def test_it_reaches_the_same_route_as_the_proxy_tool(self, world, dataset_id, monkeypatch):  # noqa: F811
        """One miDataworks route for the document, whichever tool asks; the proxy's path is pinned
        by `test_reachability.py`."""
        dw = serves(manifest()).install(monkeypatch)
        call, _ = world()
        answer(await call(**view_args(dataset_id, dataset_version_id="v1")))
        assert dw.requests[0].url.path == EXPECTED_CALLS["dataworks_get_version_manifest"][1]

    @pytest.mark.asyncio
    async def test_its_client_is_closed(self, world, dataset_id, monkeypatch):  # noqa: F811
        """A client per call: one left open per registration is a socket leak in a long-lived pod."""
        dw = serves(manifest()).install(monkeypatch)
        call, _ = world()
        answer(await call(**view_args(dataset_id, dataset_version_id=VERSION)))
        assert [c._http.is_closed for c in dw.clients] == [True]  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_it_works_with_the_dataworks_CATEGORY_off(self, world, dataset_id, monkeypatch):  # noqa: F811
        """Detected at run time from the URL, not from which proxy tools are enabled."""
        serves(manifest()).install(monkeypatch)
        call, _ = world(categories="probe_monitors")
        assert answer(await call(**view_args(dataset_id, dataset_version_id=VERSION)))["dataset_version"]


class TestTheSameRefusalsApply:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("rows", [81, 79])
    async def test_a_row_count_mismatch_is_the_routes_409_naming_both(
        self, world, dataset_id, async_session, monkeypatch, rows  # noqa: F811
    ):
        serves(manifest(splits={"train": 120, "test": rows})).install(monkeypatch)
        call, _ = world()
        before = await views_in(async_session)
        refused = await call(**view_args(dataset_id, dataset_version_id=VERSION))
        assert refused.is_error
        assert text_of(refused).endswith(
            f"backend 409: split 'test': manifest states {rows} rows, downloaded split has 80"
        ), text_of(refused)
        assert await views_in(async_session) == before

    @pytest.mark.asyncio
    async def test_a_split_the_version_does_not_describe_is_refused(self, world, dataset_id, async_session, monkeypatch):  # noqa: F811
        serves(manifest(splits={"train": 120})).install(monkeypatch)
        call, _ = world()
        refused = await call(**view_args(dataset_id, dataset_version_id=VERSION))
        assert refused.is_error and "backend 409" in text_of(refused) and "'test'" in text_of(refused)
        assert await views_in(async_session) == 0

    @pytest.mark.asyncio
    async def test_a_document_that_fails_the_schema_is_refused_naming_the_path(self, world, dataset_id, async_session, monkeypatch):  # noqa: F811
        document = manifest()
        document["version"]["number"] = 0
        serves(document).install(monkeypatch)
        call, _ = world()
        refused = await call(**view_args(dataset_id, dataset_version_id=VERSION))
        assert refused.is_error and "$.version.number" in text_of(refused)
        assert await views_in(async_session) == 0


class TestMiDataworksAbsentNeverMeansUnverified:
    """Every path here ends with no view and no POST: a silent unverified view is the defect."""

    @pytest.mark.asyncio
    async def test_no_url_configured(self, world, dataset_id, async_session, monkeypatch):  # noqa: F811
        dw = FakeDataworks(lambda r: pytest.fail("no URL is configured: nothing may be called")).install(monkeypatch)
        call, backend = world(dataworks_url="")
        result = answer(await call(**view_args(dataset_id, dataset_version_id=VERSION)))
        assert result["unavailable"] == "dataworks"
        assert "MIDATAWORKS_API_URL" in result["reason"] and "NOT registered" in result["reason"]
        assert backend.calls == [] and dw.clients == []
        assert await views_in(async_session) == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")],
                             ids=["connection refused", "read timeout"])
    async def test_it_does_not_answer(self, world, dataset_id, async_session, monkeypatch, failure):  # noqa: F811
        def respond(request):
            raise failure

        FakeDataworks(respond).install(monkeypatch)
        call, backend = world()
        result = answer(await call(**view_args(dataset_id, dataset_version_id=VERSION)))
        assert result["unavailable"] == "dataworks" and "miDataworks" in result["reason"]
        assert backend.calls == []
        assert await views_in(async_session) == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [502, 503, 504])
    async def test_its_ingress_answers_for_a_dead_backend(self, world, dataset_id, async_session, monkeypatch, status):  # noqa: F811
        FakeDataworks(lambda r: httpx.Response(status, text="bad gateway")).install(monkeypatch)
        call, backend = world()
        result = answer(await call(**view_args(dataset_id, dataset_version_id=VERSION)))
        assert result["unavailable"] == "dataworks" and str(status) in result["reason"]
        assert backend.calls == []
        assert await views_in(async_session) == 0

    @pytest.mark.asyncio
    async def test_a_malformed_url_is_unavailable_and_not_a_traceback(self, world, dataset_id, async_session, monkeypatch):  # noqa: F811
        """`httpx.InvalidURL` is not an `httpx.HTTPError`, so the client does not catch it
        (MIS-E2E-116 reached agents as a traceback on every gated tool)."""
        call, backend = world(dataworks_url="http://midataworks:notaport")
        result = answer(await call(**view_args(dataset_id, dataset_version_id=VERSION)))
        assert result["unavailable"] == "dataworks" and "malformed MIDATAWORKS_API_URL" in result["reason"]
        assert backend.calls == []
        assert await views_in(async_session) == 0

    @pytest.mark.asyncio
    async def test_a_version_it_does_not_know_is_its_own_error_verbatim(self, world, dataset_id, async_session, monkeypatch):  # noqa: F811
        FakeDataworks(lambda r: httpx.Response(404, json={"error": {
            "code": "not_found", "message": "no version nope", "details": None}})).install(monkeypatch)
        call, backend = world()
        refused = await call(**view_args(dataset_id, dataset_version_id="nope"))
        assert refused.is_error and "no version nope" in text_of(refused) and "not_found" in text_of(refused)
        assert backend.calls == []
        assert await views_in(async_session) == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("response", [
        httpx.Response(200, text="<html>splash</html>"),
        httpx.Response(200),
        httpx.Response(200, json=[{"kind": "midataworks.dataset-version"}]),
        httpx.Response(200, json={}),
        httpx.Response(500, json={"error": {"code": "boom", "message": "broke", "details": None}}),
    ], ids=["html page", "empty body", "a list", "an empty object", "a 500"])
    async def test_an_answer_that_is_not_a_manifest_registers_nothing(
        self, world, dataset_id, async_session, monkeypatch, response  # noqa: F811
    ):
        FakeDataworks(lambda r: response).install(monkeypatch)
        call, backend = world()
        result = await call(**view_args(dataset_id, dataset_version_id=VERSION))
        said = text_of(result)
        assert result.is_error or "refused" in said, said
        assert posts(backend) == []
        assert await views_in(async_session) == 0


class TestArgumentsAreCheckedBeforeAnythingIsCalled:
    """A payload mistake must not read as "miDataworks is down", and costs no fetch."""

    @pytest.mark.asyncio
    async def test_a_manifest_and_an_id_together_are_refused(self, world, dataset_id, async_session, monkeypatch):  # noqa: F811
        dw = serves(manifest()).install(monkeypatch)
        call, backend = world()
        result = answer(await call(**view_args(dataset_id, dataset_version_id=VERSION,
                                               dataset_version_manifest=manifest())))
        assert result["refused"] == "manifest_and_version_id"
        assert dw.requests == [] and backend.calls == []
        assert await views_in(async_session) == 0

    @pytest.mark.asyncio
    async def test_both_are_refused_even_with_miDataworks_absent(self, world, dataset_id):  # noqa: F811
        call, _ = world(dataworks_url="")
        result = answer(await call(**view_args(dataset_id, dataset_version_id=VERSION,
                                               dataset_version_manifest=manifest())))
        assert result["refused"] == "manifest_and_version_id"

    @pytest.mark.asyncio
    async def test_an_id_without_a_split_is_refused(self, world, dataset_id, async_session, monkeypatch):  # noqa: F811
        dw = serves(manifest()).install(monkeypatch)
        call, backend = world()
        args = view_args(dataset_id, dataset_version_id=VERSION)
        del args["split"]
        result = answer(await call(**args))
        assert result["refused"] == "split_required" and "split" in result["reason"]
        assert dw.requests == [] and backend.calls == []
        assert await views_in(async_session) == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", ["../../settings", "a/b", "", "a b", "x" * 65])
    async def test_an_id_that_could_walk_the_url_is_refused(self, world, dataset_id, monkeypatch, bad):  # noqa: F811
        dw = serves(manifest()).install(monkeypatch)
        call, backend = world()
        result = await call(**view_args(dataset_id, dataset_version_id=bad))
        assert result.is_error
        assert dw.requests == [] and backend.calls == []


class TestTheExistingParameterAndTheStandaloneCase:
    @pytest.mark.asyncio
    async def test_a_manifest_passed_in_goes_through_unchanged_and_fetches_nothing(
        self, world, dataset_id, monkeypatch  # noqa: F811
    ):
        dw = FakeDataworks(lambda r: pytest.fail("the caller gave the document")).install(monkeypatch)
        call, backend = world()
        document = manifest()
        view = answer(await call(**view_args(dataset_id, dataset_version_manifest=document)))
        assert dw.clients == []
        assert posts(backend)[0][2]["dataset_version_manifest"] == document
        assert view["dataset_version"]["manifest_rows"] == 80

    @pytest.mark.asyncio
    async def test_a_manifest_passed_in_needs_no_miDataworks(self, world, dataset_id):  # noqa: F811
        """A document already in hand is verified against the download, with no sibling running."""
        call, _ = world(dataworks_url="")
        view = answer(await call(**view_args(dataset_id, dataset_version_manifest=manifest())))
        assert view["dataset_version"]["manifest_rows"] == 80

    @pytest.mark.asyncio
    async def test_a_plain_view_needs_no_miDataworks(self, world, dataset_id, monkeypatch):  # noqa: F811
        """miStudio stands alone. Unverified is what the caller asked for here, and it says so."""
        dw = FakeDataworks(lambda r: pytest.fail("nothing was asked of miDataworks")).install(monkeypatch)
        call, backend = world(dataworks_url="")
        view = answer(await call(**view_args(dataset_id)))
        assert view["dataset_version"] is None
        assert dw.clients == [] and len(posts(backend)) == 1
        assert "dataset_version_manifest" not in posts(backend)[0][2]


class TestAnAgentsRegistrationDoesNotConfirm:
    """Operator's decision, 2026-11-02. A send from miDataworks carries the operator's `hub_push` approval. An
    agent holding a manifest carries nothing, and was recorded `by: midataworks` all the same, which passed the
    check a person makes before a probe can be primary."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("argument", ["dataset_version_id", "dataset_version_manifest"])
    async def test_it_is_unconfirmed_verified_and_says_how(self, world, dataset_id, monkeypatch, argument):  # noqa: F811
        serves(manifest()).install(monkeypatch)
        call, _ = world()
        given = {"dataset_version_id": VERSION} if argument == "dataset_version_id" else {"dataset_version_manifest": manifest()}
        view = answer(await call(**view_args(dataset_id, behaviour="pii", role="train", **given)))
        assert view["dataset_version"]["manifest_rows"] == 80, "the manifest is still verified"
        assert view["positive_labels_confirmed"] is None
        assert view["positive_labels"]["confirmed"] is False
        assert "operator" in view["positive_labels"]["next"] and "'pii'" in view["positive_labels"]["next"]

    @pytest.mark.asyncio
    async def test_the_operator_then_confirms_and_an_agent_cannot(self, world, dataset_id, client, monkeypatch):  # noqa: F811
        serves(manifest()).install(monkeypatch)
        call, _ = world()
        view = answer(await call(**view_args(dataset_id, behaviour="pii", role="train", dataset_version_id=VERSION)))
        url = f"/api/v1/probe-monitors/datasets/{view['id']}/confirm-positive"
        refused = await client.post(url, json={"key": "pii"}, headers={"X-MiStudio-Agent": "mcp"})
        assert refused.status_code == 403
        confirmed = await client.post(url, json={"key": "pii"})
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["confirmed_for_this_behaviour"] is True

    @pytest.mark.asyncio
    async def test_a_send_still_confirms(self, dataset_id, client):  # noqa: F811
        """miDataworks' POST carries no agent header. Its body is the same shape as the agent's."""
        made = await client.post("/api/v1/probe-monitors/datasets", json=view_args(
            dataset_id, behaviour="pii", role="train", dataset_version_manifest=manifest()))
        assert made.status_code in (200, 201), made.text
        confirmed = made.json()["positive_labels_confirmed"]
        assert (confirmed["key"], confirmed["origin"], confirmed["by"]) == ("pii", "midataworks", "midataworks")

    @pytest.mark.asyncio
    async def test_a_calibration_view_has_nothing_to_confirm(self, world, dataset_id, monkeypatch):  # noqa: F811
        serves(manifest()).install(monkeypatch)
        call, _ = world()
        args = view_args(dataset_id, behaviour="pii", role="calibration", dataset_version_id=VERSION)
        args["label_mapping"] = {"yes": "excluded", "no": "negative"}
        view = answer(await call(**args))
        assert "positive_labels" not in view

    def test_miDataworks_send_does_not_stamp_the_agent_header(self):
        """The fix tells the two apart by that header. Were the send to stamp it, it would stop confirming."""
        from tests.support.sibling_repos import midataworks_repo

        # miDataworks renamed the module in its P4 (020): the new name first, the old for a checkout before it.
        clients = midataworks_repo() / "backend" / "src" / "clients"
        client_source = next((p for p in (clients / "miworkbench_client.py", clients / "mistudio_client.py")
                              if p.is_file()), clients / "miworkbench_client.py")
        if not client_source.is_file():
            message = (f"miDataworks is not at {client_source}; set MIDATAWORKS_REPO. Skipping is expected in CI and on "
                       "the public mirror; set MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1 to make this a failure.")
            if require_cross_repo():
                pytest.fail(message)
            pytest.skip(message)
        assert "/probe-monitors/datasets" in client_source.read_text(), "the send's POST moved; re-read this"
        assert "X-MiStudio-Agent" not in client_source.read_text()


class TestTheWholeStepWithNoScript:
    """D116's complaint, as one test: the two steps the dry run took around MCP."""

    @pytest.mark.asyncio
    async def test_download_poll_and_register_verified_all_over_mcp(
        self, world, async_session, arrow_dict, monkeypatch  # noqa: F811
    ):
        from src.models.dataset import Dataset

        install(monkeypatch, FakeHub(default=COMMIT_A))
        serves(manifest()).install(monkeypatch)
        call, _ = world()
        with patch("src.workers.dataset_tasks.download_dataset_task.delay") as delay:
            delay.return_value = MagicMock(id="task-1")
            row = answer(await call("download_dataset", repo_id="org/stakes", split="test"))
        assert row["status"] == "downloading"

        # The worker finishes: the one part this test cannot run.
        stored = await async_session.get(Dataset, uuid.UUID(str(row["id"])))
        stored.status = DatasetStatus.READY
        stored.raw_path = str(arrow_dict)
        await async_session.commit()

        polled = answer(await call("get_dataset", dataset_id=str(row["id"])))
        assert polled["status"] == "ready"

        view = answer(await call(**view_args(str(row["id"]), dataset_version_id=VERSION)))
        assert view["dataset_id"] == str(row["id"])
        assert view["dataset_version"]["manifest_rows"] == view["dataset_version"]["downloaded_rows"] == 80


class TestTheHowtoIsNowTrue:
    """It said "No step needs the UI, a script or a file download" while two steps needed curl or
    went unverified (ENH-006 D116)."""

    def test_it_names_both_and_in_the_order_they_run(self):
        text = howto.TOPICS["behaviour_detectors"]
        assert "No step needs the UI, a script or a file download." in text
        for name in ("download_dataset", "get_dataset", "dataset_version_id"):
            assert f"`{name}`" in text, name
        assert text.index("`download_dataset`") < text.index("`register_probe_dataset`")
        assert text.index("`register_probe_dataset`") < text.index("`dataset_version_id`")
        assert text.index("`dataset_version_id`") < text.index("`start_probe_training_run`")

    def test_what_it_names_is_live(self):
        schemas = live_input_schemas()
        assert "download_dataset" in schemas and "get_dataset" in schemas
        assert "dataset_version_id" in schemas[TOOL]["properties"]
