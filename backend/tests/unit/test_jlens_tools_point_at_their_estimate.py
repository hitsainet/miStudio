"""Every priced J-space operation's tool names its estimate (028 task 4.2, BR-028).

The envelope existed and only an agent that already knew to call
`jlens_cost_estimate` would. The run tool's own description is the one text an
agent reads before committing, so each tool that RUNS a priced operation must name
the estimate with that operation's value. The tool↔operation map is checked
against `OperationClass` so a newly priced operation without a tool here fails.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.mcpserver import MCPServer

from src.mcp_server.config import MCPSettings
from src.mcp_server.tools import jlens
from src.services.jlens_watchlist import OperationClass

RUN_TOOL = {
    OperationClass.ARTIFACT_CONSTRUCTION: "fit_jlens_artifact",
    OperationClass.READOUT: "jlens_readout",
    OperationClass.INTERVENTION_RUN: "run_jlens_intervention",
    OperationClass.DECOMPOSITION: "jlens_decompose",
    OperationClass.ANNOTATION_SWEEP: "start_jlens_annotation_sweep",
    OperationClass.TEMPLATE_LENS_BUILD: "build_jlens_template_lens",
}


@pytest.fixture(scope="module")
def descriptions():
    mcp = MCPServer("t")
    jlens.register(mcp, MagicMock(get=AsyncMock(), post=AsyncMock()), MCPSettings(allow_anonymous=True))
    return {t.name: (t.description or "") for t in asyncio.run(mcp.list_tools())}


def test_every_priced_operation_has_a_run_tool():
    assert set(RUN_TOOL) == set(OperationClass)


@pytest.mark.parametrize("operation", list(OperationClass))
def test_the_run_tool_names_its_estimate(operation, descriptions):
    text = descriptions[RUN_TOOL[operation]]
    assert "jlens_cost_estimate" in text
    assert f'operation="{operation.value}"' in text
