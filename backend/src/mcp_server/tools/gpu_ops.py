"""Is the GPU free? (category: jobs) — MCP gap G9, 2026-10-21.

Before a GPU run, the end-to-end probe run of 2026-10-20 needed `ssh` + `nvidia-smi` on the node, `curl` to
miLLM and miForge, and six miStudio MCP calls over five tools — and `list_gpus` answered only
static card facts. Three tools replace that:

* `get_gpu_status` — every card by UUID (or one): memory used/total, utilisation, compute
  processes, miStudio's lease on the card and the miStudio jobs that recorded it;
* `list_active_jobs` — every queued or running miStudio job, of EVERY kind, in one read;
* `gpu_preflight(gpu)` — both, plus miLLM's resident model, lease and per-card memory, and
  miForge's active runs, with a verdict and the reasons for it.

STANDALONE FIRST. miLLM and miForge are optional siblings: with no URL configured the preflight
says `configured: false` for that product and carries on — never an error. A configured product
that does not answer is `reachable: false` with the reason, and makes the verdict `uncertain`,
because "I could not ask" is not "it is idle".

THE PREFLIGHT IS NATIVE, NOT A `millm_*` TOOL, deliberately: it must work on a server with no miLLM
categories enabled. It builds its own short-timeout miLLM client from `MILLM_API_URL` and reads
two read-only routes (`/api/health/detailed`, `/api/health/gpus`) that `millm_status` already
relies on; nothing here changes any product's state.

GPUs ARE NAMED BY UUID. Indices follow PCI bus order and moved on this estate in 2026-09; a tool
that accepted one could report on a different card than the caller meant.
"""

from typing import Annotated, Any, Optional

import httpx
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import BackendError, MiStudioClient
from ..config import MCPSettings
from ..millm_client import MiLLMClient

#: Sibling reads must not hold the agent up: a product that cannot answer in this long is
#: reported unreachable, and the verdict says so.
SIBLING_TIMEOUT_S = 8.0

GPU_UUID_PARAM = (
    "The card's UUID (GPU-…), from list_gpus or get_gpu_status. Indices are refused: they follow "
    "PCI bus order and can name a different card after a reboot"
)


def _millm_client(url: str) -> MiLLMClient:
    """The miLLM client the preflight reads with. A function so a test can hand in a transport."""
    return MiLLMClient(url, timeout=SIBLING_TIMEOUT_S)


def _forge_client(url: str) -> httpx.AsyncClient:
    """The miForge HTTP client. miForge has no client class here; two read-only GETs need none."""
    return httpx.AsyncClient(base_url=url, timeout=SIBLING_TIMEOUT_S)


def _norm(uuid: Optional[str]) -> str:
    text = (uuid or "").strip().lower()
    return text if text.startswith("gpu-") else f"gpu-{text}"


def _refusal(exc: BackendError) -> dict[str, Any]:
    return {"refused": True, "status_code": exc.status_code, "detail": exc.detail}


async def millm_block(settings: MCPSettings, gpu: str) -> dict[str, Any]:
    """What miLLM is doing with this card: resident model, lease, its memory on the card."""
    url = settings.millm_api_url
    if not url:
        return {"configured": False,
                "detail": "MILLM_API_URL is not set on this MCP server; miLLM was not consulted"}
    millm = _millm_client(url)
    try:
        detailed = await millm.get("/api/health/detailed")
        memory = await millm.get("/api/health/gpus")
    except BackendError as exc:
        return {"configured": True, "reachable": False, "url": url, "reason": exc.detail}
    finally:
        await millm.close()
    card = next((c for c in (memory or {}).get("cards") or []
                 if _norm(c.get("uuid")) == _norm(gpu)), None)
    resident_here = bool(card) and (
        bool(card.get("millm_allocated_mb")) or card.get("engine_memory") is not None
    )
    return {
        "configured": True,
        "reachable": True,
        "model_loaded": (detailed or {}).get("model_loaded"),
        "model_name": (detailed or {}).get("model_name"),
        "model_placement": (detailed or {}).get("model_placement"),
        "lease": (detailed or {}).get("lease"),
        "weights": (detailed or {}).get("weights"),
        "card": card,
        "resident_on_this_card": resident_here,
    }


async def miforge_block(settings: MCPSettings) -> dict[str, Any]:
    """miForge's pending and running training runs (miForge's run list names no card)."""
    url = settings.miforge_api_url
    if not url:
        return {"configured": False,
                "detail": "MIFORGE_API_URL is not set on this MCP server; miForge was not consulted"}
    forge = _forge_client(url)
    runs: list[dict[str, Any]] = []
    try:
        for status in ("running", "pending"):
            response = await forge.get("/api/v1/runs", params={"status": status, "limit": 200})
            if response.status_code >= 400:
                return {"configured": True, "reachable": False, "url": url,
                        "reason": f"miForge answered {response.status_code} on /api/v1/runs"}
            try:
                body = response.json()
            except ValueError:
                return {"configured": True, "reachable": False, "url": url,
                        "reason": "miForge returned non-JSON from /api/v1/runs (a misrouted "
                                  "ingress or proxy page) — NOT an empty run list"}
            listed = body.get("runs") if isinstance(body, dict) else None
            if not isinstance(listed, list):
                # miForge's GET /api/v1/runs answers `{runs: [...], total, ...}`. Any other shape
                # is not "no runs": say so rather than read it as empty.
                return {"configured": True, "reachable": False, "url": url,
                        "reason": "miForge's /api/v1/runs answer has no `runs` list"}
            runs += [r for r in listed if isinstance(r, dict)]
    except httpx.HTTPError as exc:
        return {"configured": True, "reachable": False, "url": url,
                "reason": f"miForge unreachable: {type(exc).__name__}: {exc}"}
    finally:
        await forge.aclose()
    return {"configured": True, "reachable": True, "active_runs": runs,
            "active_run_count": len(runs)}


def verdict(studio_card: Optional[dict[str, Any]], studio_reason: Optional[str],
            queued_auto: list[dict[str, Any]], millm_state: dict[str, Any],
            forge_state: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """`(verdict, busy_reasons, uncertain_reasons)`. `clear` only when nothing says otherwise."""
    busy: list[str] = []
    uncertain: list[str] = []
    if studio_card is None:
        uncertain.append(studio_reason or "miStudio cannot see this card")
    else:
        busy += studio_card.get("in_use_by") or []
    if queued_auto:
        uncertain.append(
            f"{len(queued_auto)} queued miStudio job(s) have no card yet and may take this one "
            "when they start: " + ", ".join(f"{j['kind']} {j['id']}" for j in queued_auto[:5]))
    if millm_state.get("configured"):
        if not millm_state.get("reachable"):
            uncertain.append(f"miLLM is configured but did not answer: {millm_state.get('reason')}")
        elif millm_state.get("resident_on_this_card"):
            busy.append(f"miLLM's resident model {millm_state.get('model_name')!r} is on this card")
    if forge_state.get("configured"):
        if not forge_state.get("reachable"):
            uncertain.append(f"miForge is configured but did not answer: {forge_state.get('reason')}")
        elif forge_state.get("active_run_count"):
            uncertain.append(
                f"miForge has {forge_state['active_run_count']} pending/running run(s); its run list "
                "does not name a card, so check them before assuming this one is free")
    if busy:
        return "busy", busy, uncertain
    return ("uncertain" if uncertain else "clear"), busy, uncertain


def register(mcp: MCPServer, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def get_gpu_status(
        gpu: Annotated[Optional[str], Field(description=f"Only this card. {GPU_UUID_PARAM}. Omit for every card")] = None,
    ) -> Any:
        """Live state of every GPU miStudio can see (one or many), by UUID: memory used / free /
        total (MiB), utilisation, compute processes, miStudio's GPU lease on the card and the
        miStudio jobs that recorded it, plus `in_use` and the reasons.

        A process is attributed only when it is a miStudio steering worker; miLLM, miForge and
        miStudio's own worker run in other pods, so their PIDs show `owner: null`. For "can I
        start a GPU job on this card?" use gpu_preflight, which also asks miLLM and miForge."""
        return await client.get("/operations/gpu-status", gpu=gpu)

    @mcp.tool()
    async def list_active_jobs(
        kind: Annotated[Optional[str], Field(description="Only this kind, e.g. probe_training, probe_judge, sae_training, feature_extraction, circuit_capture, task_queue, steering_worker. An unknown kind is refused with the list")] = None,
        gpu: Annotated[Optional[str], Field(description=f"Only jobs that recorded this card. {GPU_UUID_PARAM}")] = None,
    ) -> Any:
        """Every miStudio job that is queued or running, of EVERY kind, in one call: SAE training,
        extraction, activation extraction, labeling (all three), tokenization, downloads, probe
        training / evaluation / judge, circuit capture / discovery / attribution / validation /
        faithfulness / calibration, steering recordings and workers, Neuronpedia, grouping, and
        the generic task queue.

        `errors` names any source that could not be read — then the list may be incomplete, which
        is not the same as nothing running."""
        return await client.get("/operations/active-jobs", kind=kind, gpu=gpu)

    @mcp.tool()
    async def gpu_preflight(
        gpu: Annotated[str, Field(description=GPU_UUID_PARAM, min_length=4)],
    ) -> Any:
        """"Is this card free for a GPU job?" in one call, across the suite.

        Combines miStudio's live card state and every active miStudio job with miLLM's resident
        model, lease and per-card memory (when MILLM_API_URL is set) and miForge's pending and
        running runs (when MIFORGE_API_URL is set). Returns `verdict`:

        * `clear` — nothing holds the card and every configured product answered;
        * `busy` — `busy_reasons` names what holds it (a lease, a running job, miLLM's model,
          an unattributed process using memory);
        * `uncertain` — nothing is known to hold it, but `uncertain_reasons` says what could not
          be ruled out (a product that did not answer, queued jobs with no card yet, miForge runs).

        A sibling that is not configured is reported `configured: false` and does not affect the
        verdict: miStudio works without miLLM and miForge. Read-only; starts nothing."""
        if not gpu.strip().lower().startswith("gpu-"):
            return {"refused": True, "detail": f"{gpu!r} is not a GPU UUID. {GPU_UUID_PARAM}."}
        try:
            studio = await client.get("/operations/gpu-status", gpu=gpu)
            jobs = await client.get("/operations/active-jobs")
        except BackendError as exc:
            return {"gpu": gpu, **_refusal(exc)}
        cards = studio.get("cards") or []
        card = cards[0] if cards else None
        queued_auto = [
            j for j in jobs.get("jobs") or []
            if j.get("status") in ("queued", "pending") and not j.get("gpu_uuids")
        ]
        millm_state = await millm_block(settings, gpu)
        forge_state = await miforge_block(settings)
        result, busy, uncertain = verdict(card, studio.get("reason"), queued_auto, millm_state,
                                          forge_state)
        return {
            "gpu": gpu,
            "verdict": result,
            "busy_reasons": busy,
            "uncertain_reasons": uncertain,
            "mistudio": {
                "card": card,
                "reason": studio.get("reason"),
                "active_jobs": jobs.get("jobs") or [],
                "job_read_errors": jobs.get("errors") or [],
            },
            "millm": millm_state,
            "miforge": forge_state,
            "read_at": jobs.get("read_at"),
        }
