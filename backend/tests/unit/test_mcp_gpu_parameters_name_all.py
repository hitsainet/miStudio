"""Every MCP tool with a `gpu` parameter tells an agent whether "all" works.

Multi-GPU Phase 2 made "all" a real request: split the model across every card.
The REST schema says so (`schemas/gpu.py`), and the J-lens tools said so. The
circuit, steering, model-download and labeling tools — every one of which runs
split — still described only "auto" and a GPU UUID, so an agent reading the
registered schema could not learn the choice existed. A J-lens fit, which never
splits, says "all" is refused.

Read from the LIVE registry (`build_server`), never from the modules'
annotations: what an agent reads is what the registered tool advertises. The
table is exhaustive — a new tool with a `gpu` parameter fails here until it is
classified — so the next GPU tool cannot ship silent about "all".

MUTATION CONTROLS (review round 1, 2026-09-14; each applied alone, this module
run red, restored byte-identically and checked by sha256):
  D1  circuits `GpuParam` without "all"            -> the seven circuit tools that take it
  D2  circuits `ReproduceGpuParam` without "all"   -> reproduce_calibration, reproduce_validation
  D3  steering GPU_PARAM_DESCRIPTION without "all" -> steer_combined, steer_compare, steer_sweep
  D4  download_model's gpu text without "all"      -> download_model
  D5  resume_labeling's gpu text without "all"      -> resume_labeling
  D5b the sweep's gpu text without "all"            -> start_labeling_resume_sweep
  D6  the fit's text stops saying "all" is refused  -> fit_jlens_artifact
All 7 went red. Before the text fix, 14 of the 19 split-capable tools failed here.
"""

from __future__ import annotations

import asyncio
import re

import pytest

#: Every registered tool that takes `gpu`: True when its job honours "all".
SPLITS = {
    # circuits — capture, attribution, validation, faithfulness, calibration,
    # the recorder, and both reproductions all place with allow_shard=True.
    "start_circuit_capture": True,
    "run_attribution_pass": True,
    "validate_circuit_edges": True,
    "run_circuit_faithfulness": True,
    "calibrate_circuit_strength": True,
    "reproduce_calibration": True,
    "reproduce_validation": True,
    "record_steering_samples": True,
    # steering (steering_service.place_job(..., allow_shard=True))
    "steer_combined": True,
    "steer_compare": True,
    "steer_sweep": True,
    # model download (model_tasks, allow_shard=True)
    "download_model": True,
    # local labeling (labeling_service, allow_shard=True)
    "resume_labeling": True,
    "start_labeling_resume_sweep": True,
    # J-lens
    "acquire_jlens_artifact": True,
    "compute_jlens_band_report": True,
    "jlens_readout": True,
    "run_jlens_intervention": True,
    "fit_jlens_artifact": False,
}

ALL = re.compile(r"""["']all["']""")


@pytest.fixture(scope="module")
def gpu_descriptions():
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    mp = pytest.MonkeyPatch()
    mp.setenv("MILLM_API_URL", "http://millm.test")
    try:
        settings = MCPSettings(
            tool_categories="read,groups,steering,labeling,experiments,profiles,circuits,jlens,jobs,models,admin",
            allow_anonymous=True,
        )
        mcp, _client = build_server(settings, stdio=True)
        tools = asyncio.run(mcp.list_tools())
    finally:
        mp.undo()
    return {
        tool.name: tool.inputSchema["properties"]["gpu"].get("description") or ""
        for tool in tools
        if "gpu" in tool.inputSchema.get("properties", {})
    }


def test_every_tool_with_a_gpu_parameter_is_classified(gpu_descriptions):
    assert set(gpu_descriptions) == set(SPLITS)


@pytest.mark.parametrize("name", sorted(n for n, splits in SPLITS.items() if splits))
def test_a_tool_whose_job_splits_offers_all(gpu_descriptions, name):
    text = gpu_descriptions[name]
    assert ALL.search(text), f"{name} never tells an agent that \"all\" splits the model: {text!r}"
    assert "split" in text


def test_the_fit_says_all_is_refused(gpu_descriptions):
    text = gpu_descriptions["fit_jlens_artifact"]
    assert ALL.search(text) and "refused" in text, text
