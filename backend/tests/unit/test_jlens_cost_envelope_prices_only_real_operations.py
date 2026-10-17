"""The cost envelope prices only operations the product can RUN (suite D13, 2026-10-16).

`/jlens/cost-estimate` quoted prices for `decomposition`, `annotation_sweep`
and `template_lens_build` while no route ran any of them. (`decomposition`
and `annotation_sweep` came back the same day WITH their routes,
`POST /jlens/decompose` and `POST /jlens/annotation-sweeps` — the pairing below
is what allowed them back, as did `template_lens_build` with
`POST /jlens/template-lens`.) A price is a promise:
an agent reading the known-operations list in a 400, or the tool's own
description, plans a sweep nobody can start. These tests pair every priced
class with a route in the app's SERVED schema (`app.openapi()["paths"]`, not
`app.routes` — included routers carry no `.path` since FastAPI 0.141), and read
the refusal the endpoint actually returns.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.jlens_watchlist import OPERATION_ROUTES, OperationClass


@pytest.fixture(scope="module")
def served_paths():
    return app.openapi()["paths"]


def test_every_priced_operation_names_a_route(served_paths):
    assert set(OPERATION_ROUTES) == set(OperationClass), (
        "a priced class with no route is a quote for an operation nobody can start"
    )


@pytest.mark.parametrize("operation", list(OperationClass))
def test_every_priced_operation_is_SERVED(operation, served_paths):
    method, path = OPERATION_ROUTES[operation]
    assert path in served_paths, f"{operation.value} is priced but {path} is not served"
    assert method in served_paths[path], f"{path} does not serve {method.upper()}"


@pytest.mark.parametrize("dead", ["not_an_operation", "sae_training"])
def test_an_unknown_operation_is_refused_by_the_live_endpoint(dead):
    """The endpoint must not price them, and its 400 must not list them."""
    client = TestClient(app)
    resp = client.get(
        "/api/v1/jlens/cost-estimate",
        params={"operation": dead, "d_model": 2048, "n_layers": 16},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "unknown operation" in detail
    assert f"'{dead}'" not in detail.split("Known:", 1)[1]


def test_a_real_operation_is_still_priced_through_the_live_endpoint():
    client = TestClient(app)
    resp = client.get(
        "/api/v1/jlens/cost-estimate",
        params={"operation": "readout", "d_model": 2048, "n_layers": 16, "n_positions": 8},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["operation"] == "readout"
    assert body["basis"]


def test_the_unused_readout_mode_literal_is_gone():
    """`ReadoutMode` declared a 'decomposition' mode nothing produced or read."""
    import src.schemas.jlens as schemas

    assert not hasattr(schemas, "ReadoutMode")


def test_the_mcp_tool_description_lists_only_priced_operations():
    """The tool description is the only documentation an agent reads."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from mcp.server.fastmcp import FastMCP

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import jlens as jlens_tools

    mcp = FastMCP("t")
    jlens_tools.register(
        mcp, MagicMock(get=AsyncMock(return_value={})), MCPSettings(allow_anonymous=True)
    )
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    schema = tools["jlens_cost_estimate"].inputSchema
    described = schema["properties"]["operation"]["description"]
    for op in OperationClass:
        assert op.value in described
    # All three once-dead quotes are back, each WITH its route.
    for back in ("decomposition", "annotation_sweep", "template_lens_build"):
        assert back in described
    # n_features returned with the sweep it sizes (it was dropped while no sweep existed).
    assert "n_features" in schema["properties"]


def test_a_decomposition_estimate_needs_the_vocabulary_and_says_so():
    """Every pursuit iteration is a pass over the vocabulary: no n_vocab, no number."""
    client = TestClient(app)
    refused = client.get(
        "/api/v1/jlens/cost-estimate",
        params={"operation": "decomposition", "d_model": 2048, "n_layers": 2},
    )
    assert refused.status_code == 400
    assert "n_vocab" in refused.json()["detail"]
    priced = client.get(
        "/api/v1/jlens/cost-estimate",
        params={"operation": "decomposition", "d_model": 2048, "n_layers": 2,
                "n_positions": 4, "n_vocab": 256000, "k": 8},
    )
    assert priced.status_code == 200, priced.text
    small = client.get(
        "/api/v1/jlens/cost-estimate",
        params={"operation": "decomposition", "d_model": 2048, "n_layers": 2,
                "n_positions": 4, "n_vocab": 32000, "k": 8},
    )
    # The vocabulary is a real factor, not decoration.
    assert priced.json()["order_of_magnitude_seconds"] > small.json()["order_of_magnitude_seconds"]
    assert "control" in priced.json()["basis"]
