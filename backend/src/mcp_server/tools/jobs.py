"""Job-tracking tools (category: jobs) — the async-202 polling half (BR-1.5)."""

from typing import Annotated, Any, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import MiWorkbenchClient
from ..config import MCPSettings


#: The tool that reads a job row whose Celery id a tool handed out (MCP gap, 2026-11-10): table → (tool, the
#: tool's id argument). Each is checked against the LIVE registry by test_task_id_lookup.py. A table not here is
#: still found by the lookup and answered with its table, row id and status.
TASK_ROW_READERS: dict[str, tuple[str, str]] = {
    "probe_monitor_runs": ("get_probe_training_run", "run_id"),
    "probe_spot_checks": ("get_probe_spot_check", "spot_check_id"),
    "probe_blend_reports": ("get_blend_report", "report_id"),
    "circuit_discovery_runs": ("get_discovery_results", "run_id"),
    "trainings": ("get_training", "training_id"),
    "models": ("get_model", "model_id"),
}

_NOT_FOUND_NEXT = ("Nothing records this id. A `task_id` from start_probe_training_run, score or judge tools "
                   "is a Celery id; read the run with its own reader (e.g. get_probe_training_run), or call "
                   "list_active_jobs to see what is running.")


async def _answer_lookup(client: MiWorkbenchClient, task_id: str, body: Any) -> Any:
    """`GET /task-queue/lookup/{task_id}`'s answer → the task-queue record, a pointer to the job row's reader,
    or `not_found` (the tool body issues the lookup itself, so the contract and route guards see the call)."""
    from ..client import BackendError

    found = body.get("found") if isinstance(body, dict) else None
    if not found:
        return {"task_id": task_id, "status": "not_found", "next": _NOT_FOUND_NEXT}
    queued = next((f for f in found if f.get("table") == "task_queue"), None)
    if queued is not None:
        try:
            return await client.get(f"/task-queue/{queued['row_id']}")
        except BackendError as exc:
            if exc.status_code != 404:
                raise
    row = next((f for f in found if f.get("table") != "task_queue"), found[0])
    out: dict[str, Any] = {"task_id": task_id, "status": row.get("status"), "found_in": row["table"],
                           "row_id": row["row_id"]}
    reader = TASK_ROW_READERS.get(row["table"])
    if reader is not None:
        tool, arg = reader
        out["next"] = [{"tool": tool, "arguments": {arg: row["row_id"]},
                        "why": f"{row['table']} row of this task: read it with {tool}"}]
    else:
        out["next"] = (f"This id belongs to a {row['table']} row ({row['row_id']}); list_active_jobs shows it "
                       "while it runs.")
    if len(found) > 1:
        out["also_found"] = [f for f in found if f is not row]
    return out


def register(mcp: MCPServer, client: MiWorkbenchClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def get_task_status(
        task_queue_id: Annotated[Optional[str], Field(description="Task-queue id (tq_xxxx), e.g. the `task_queue_id` build_probe_definition returns. Circuit GPU tools do NOT create these — poll their run rows instead")] = None,
        list_active: Annotated[bool, Field(description="Return every running/queued job instead of one record")] = False,
        task_id: Annotated[Optional[str], Field(description="A Celery task id a tool returned as `task_id` (e.g. start_probe_training_run's). Answers the task-queue record when one exists, else WHERE it is recorded — the job row, its status and the tool that reads it — or status not_found; never the task list")] = None,
    ) -> Any:
        """Poll background jobs. With task_queue_id: that job's record (status,
        progress, error). With task_id (a Celery id a tool returned): its record, or a
        pointer to the row and tool that track it. With list_active=true: all currently
        running/queued jobs. With nothing: the 25 most recent task-queue records. Task
        ids are durable — they survive reconnects."""
        from ..client import BackendError

        if task_id:
            return await _answer_lookup(client, task_id, await client.get(f"/task-queue/lookup/{task_id}"))
        if task_queue_id:
            try:
                return await client.get(f"/task-queue/{task_queue_id}")
            except BackendError as exc:
                if exc.status_code != 404:
                    raise
                # Agents pass a Celery id here too: resolve it before saying not found.
                resolved = await _answer_lookup(client, task_queue_id,
                                                await client.get(f"/task-queue/lookup/{task_queue_id}"))
                if isinstance(resolved, dict) and resolved.get("status") == "not_found":
                    return {"task_queue_id": task_queue_id, "status": "not_found", "next": _NOT_FOUND_NEXT}
                return resolved
        if list_active:
            return await client.get("/task-queue/active")
        return await client.get("/task-queue", limit=25)

    @mcp.tool()
    async def list_gpus() -> Any:
        """The GPUs miWorkbench can place jobs on — the UUIDs and indices every `gpu` parameter
        accepts (2026-10-19: those descriptions named `/api/v1/system/gpu-list` and no tool
        read it). Pin work by UUID: indices follow PCI bus order and have moved before."""
        return await client.get("/system/gpu-list")
