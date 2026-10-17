"""
miLLM Batch API tools (category: millm_inference) — upload a JSON Lines file, run it as a batch,
poll it, cancel it, page through the results (Feature 034, FR-16, FR-22; miLLM contract §4f).

For work too large for the synchronous tools: thousands of rows scored or generated through the
same admission slot and service code as the synchronous endpoints. A batch NEVER loads, unloads or
swaps a model; it waits for the model its rows name.

Each tool issues ONE HTTP call (the upload is its own tool so the caller assertion can hold), and
every call sends `X-miLLM-Load-Policy: refuse` and `X-miLLM-Strict: true` — inert on these routes,
sent anyway so one rule covers every inference tool (FR-19).

The module name is the one miLLM's own route guard expects for this surface; its calls are
literal paths so that guard can check each against miLLM's routers.
"""

import inspect
import json
from typing import Annotated, Any, Literal, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from ..context import ExternalContext
from ..health_gate import HealthGate
from ..millm_client import MiLLMClient
from ..millm_send import send, sent_headers

#: Upload caps, checked BEFORE the gate (FTDD §9). miLLM's own limits are higher (50,000 lines,
#: 200 MiB); these keep one tool call's argument a size an agent transcript can carry.
MAX_BATCH_ROWS = 10_000
MAX_BATCH_BYTES = 16 * 1024 * 1024
MAX_CUSTOM_ID = 512
#: Rows per `millm_batch_results` page.
RESULTS_PAGE = 100

#: The endpoints a batch line may name (miLLM contract §4f).
BatchEndpoint = Literal[
    "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/api/probes/score"
]


class BatchRow(BaseModel):
    """One request in a batch: your id for it, and the body the synchronous endpoint takes."""

    model_config = ConfigDict(extra="forbid")

    custom_id: str = Field(min_length=1, max_length=MAX_CUSTOM_ID)
    body: dict[str, Any]


def build_lines(endpoint: str, rows: list[BatchRow]) -> bytes:
    """The JSONL file: `{custom_id, method: "POST", url: <endpoint>, body}` per line
    (miLLM FR-26.1.5). `url` IS the batch's endpoint — a line naming another is invalid."""
    return "".join(
        json.dumps({"custom_id": r.custom_id, "method": "POST", "url": endpoint,
                    "body": r.body}, separators=(",", ":"), ensure_ascii=False) + "\n"
        for r in rows
    ).encode("utf-8")


UPLOAD_DESCRIPTION = inspect.cleandoc(f"""
    Upload the rows of a batch to miLLM as a JSON Lines file (`purpose: "batch"`), the first of
    two steps: then call millm_submit_batch with the returned file `id` and the SAME endpoint.

    Each row is `{{custom_id, body}}`; `body` is exactly what the synchronous endpoint takes
    (for `/api/probes/score` it has no `model` — the batch runs on the resident model). The tool
    builds each line as `{{custom_id, method: "POST", url: <endpoint>, body}}`.

    Refused before anything is sent: duplicate `custom_id`s, more than {MAX_BATCH_ROWS:,} rows,
    or a file over {MAX_BATCH_BYTES // (1024 * 1024)} MiB. miLLM validates every line strictly
    when the batch starts; invalid lines come back in the error file, not as a failed upload.
""")

SUBMIT_DESCRIPTION = inspect.cleandoc("""
    Run an uploaded file as a batch (`completion_window: "24h"`). Returns the batch object in
    `validating`; poll millm_batch_status.

    A batch never loads or swaps a model: rows naming a model that is not resident make it wait
    (`millm.waiting_reason`) or fail, never evict. It takes its own lease on the resident model
    unless you pass yours (`lease_id`), in which case it runs under your lease and never
    releases it. `pack: true` runs scoring rows in packed forwards — 2.7x faster, and 0.57% of
    rows changed their top token when measured — so it is off unless you ask; unpacked rows equal
    the synchronous endpoint. `metadata` is stored with the batch.
""")

RESULTS_DESCRIPTION = inspect.cleandoc(f"""
    Page through a finished batch's output (or error) file: `{{rows, total, next_offset}}`,
    {RESULTS_PAGE} rows per page. Each row is `{{id, custom_id, response: {{status_code, body,
    millm: {{headers}}}}, error}}` in input order — match on `custom_id`, never on position.

    Pass the batch's `output_file_id` (or `error_file_id`) from millm_batch_status. The file is
    read as JSON Lines and REFUSED if miLLM answers with any other content type — a proxy page is
    never "zero rows".
""")


def register(
    mcp: FastMCP, millm: MiLLMClient, gate: HealthGate, ctx: ExternalContext
) -> None:
    @mcp.tool(description=UPLOAD_DESCRIPTION)
    async def millm_upload_batch_file(
        endpoint: Annotated[BatchEndpoint, Field(description="The endpoint every row is for; pass the same one to millm_submit_batch")],
        rows: Annotated[list[BatchRow], Field(description="The requests: [{custom_id, body}]", min_length=1)],
        filename: Annotated[Optional[str], Field(description="A name for the file; default batch.jsonl", min_length=1, max_length=255)] = None,
        lease_id: Annotated[Optional[str], Field(description="Your lease id, if you hold miLLM's lease; sent as X-miLLM-Lease")] = None,
    ) -> Any:
        """Upload the rows of a batch to miLLM as a JSON Lines file."""
        if len(rows) > MAX_BATCH_ROWS:
            return {"error": "too_many_rows",
                    "message": f"{len(rows):,} rows is over the {MAX_BATCH_ROWS:,}-row cap of "
                               "one upload; split the work into several batches."}
        seen: set[str] = set()
        repeated: set[str] = set()
        for row in rows:
            (repeated if row.custom_id in seen else seen).add(row.custom_id)
        duplicates = sorted(repeated)
        if duplicates:
            return {"error": "duplicate_custom_id",
                    "message": f"custom_id must be unique within a batch; repeated: "
                               f"{duplicates[:10]}"}
        content = build_lines(endpoint, rows)
        if len(content) > MAX_BATCH_BYTES:
            return {"error": "file_too_large",
                    "message": f"the file is {len(content):,} bytes, over the "
                               f"{MAX_BATCH_BYTES:,}-byte cap; split it."}
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        return await send(gate, millm.post(
            "/v1/files",
            upload={"filename": filename or "batch.jsonl", "content": content,
                    "purpose": "batch"},
            headers=sent_headers(lease_id), full=True,
        ))

    @mcp.tool(description=SUBMIT_DESCRIPTION)
    async def millm_submit_batch(
        input_file_id: Annotated[str, Field(description="The id millm_upload_batch_file returned", min_length=1)],
        endpoint: Annotated[BatchEndpoint, Field(description="The endpoint the file's rows are for")],
        pack: Annotated[Optional[bool], Field(description="Packed scoring forwards (faster, not bit-equal to the synchronous endpoint). Omit for miLLM's default (off)")] = None,
        metadata: Annotated[Optional[dict[str, str]], Field(description="Free-form labels stored with the batch")] = None,
        lease_id: Annotated[Optional[str], Field(description="Run under YOUR lease (the batch then never releases it)")] = None,
    ) -> Any:
        """Run an uploaded file as a batch."""
        body: dict[str, Any] = {"input_file_id": input_file_id, "endpoint": endpoint,
                                "completion_window": "24h"}
        if pack is not None:
            body["pack"] = pack
        if metadata is not None:
            body["metadata"] = dict(metadata)
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        return await send(gate, millm.post(
            "/v1/batches", json_body=body, headers=sent_headers(lease_id), full=True,
        ))

    @mcp.tool()
    async def millm_batch_status(
        batch_id: Annotated[str, Field(description="The batch id millm_submit_batch returned", min_length=1)],
    ) -> Any:
        """A batch's status, row counts, file ids and why it is waiting, if it is.

        Statuses are OpenAI's: validating, in_progress, finalizing, completed, failed,
        cancelling, cancelled, expired. `millm.waiting_reason` says why a batch is not running
        (model_not_resident, lease_unavailable, weights_torn, queued). Read results with
        millm_batch_results(output_file_id).
        """
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        return await send(gate, millm.get(
            f"/v1/batches/{batch_id}", headers=sent_headers(), full=True,
        ))

    @mcp.tool()
    async def millm_cancel_batch(
        batch_id: Annotated[str, Field(description="The batch to cancel", min_length=1)],
    ) -> Any:
        """Cancel a batch. Rows already run keep their results; unrun rows become
        `batch_cancelled` error lines. A terminal batch answers `batch_state_conflict`."""
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        return await send(gate, millm.post(
            f"/v1/batches/{batch_id}/cancel", headers=sent_headers(), full=True,
        ))

    @mcp.tool(description=RESULTS_DESCRIPTION)
    async def millm_batch_results(
        file_id: Annotated[str, Field(description="output_file_id or error_file_id from millm_batch_status", min_length=1)],
        offset: Annotated[int, Field(description="First row to return", ge=0)] = 0,
        limit: Annotated[int, Field(description=f"Rows per page, 1-{RESULTS_PAGE}", ge=1, le=RESULTS_PAGE)] = RESULTS_PAGE,
    ) -> Any:
        """Page through a finished batch's output or error file."""
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        result = await send(gate, millm.get(
            f"/v1/files/{file_id}/content", jsonl=True, headers=sent_headers(), full=True,
        ))
        if "result" not in result:
            return result
        rows = result["result"]
        end = offset + limit
        return {
            "rows": rows[offset:end],
            "total": len(rows),
            "next_offset": end if end < len(rows) else None,
            "millm_headers": result["millm_headers"],
        }
