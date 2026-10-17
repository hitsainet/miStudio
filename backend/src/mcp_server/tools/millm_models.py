"""
miLLM model management (category: millm_models) — acquisition, residency and
disposal of the weights miLLM SERVES.

Written because the workflow that prompted it was unreachable over MCP:
"unload the current gguf, download and load the IQ4 version, test it." Every
step of that existed as a REST route and none of it as a tool, so the whole
sequence had to be done with `curl` inside the pod.

The asymmetry with miStudio's `models` category is deliberate and load-bearing:
**miLLM holds ONE model resident at a time.** Loading evicts whatever is there.
miStudio has no residency at all — it loads per task — so it has no
load/unload tools and no eviction to warn about.

All tools are health-gated: when miLLM is unreachable they return
{"unavailable": "millm", "reason": …} rather than raising (contract §3).
"""

from typing import Annotated, Any, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..client import BackendError
from ..health_gate import HealthGate, gated
from ..millm_client import MiLLMClient
from ..context import ExternalContext

#: miLLM's lease TTL bound (miLLM FR-29.1.2, contract §4e): also its default.
MAX_LEASE_TTL_S = 7200


def register(
    mcp: FastMCP, millm: MiLLMClient, gate: HealthGate, ctx: ExternalContext
) -> None:
    async def _actual_context() -> Optional[int]:
        """The context the engine ACTUALLY got, not the one that was asked for.

        A GGUF model loads at the largest context that fits: 8192 requested and
        4096 obtained on both quantizations tried on the 24 GB card here. A
        4703-token prompt against a 4096 window is the next failure a user hits,
        and nothing else on the system explains it — so every tool that reports
        a loaded model reports this beside it.
        """
        try:
            info = await millm.get("/api/health/inference")
        except Exception:
            return None
        if not isinstance(info, dict):
            return None
        value = info.get("context_length")
        return int(value) if isinstance(value, int) and value > 0 else None

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_list_models(
        loaded_only: Annotated[bool, Field(description="Return only the resident model")] = False,
    ) -> Any:
        """What models miLLM has, and which one is serving right now.

        Start here. The answer to "why won't it work?" is usually in this
        output: a model that is `error`, a download stuck at an unknown
        percentage, or a resident model with a smaller context window than the
        prompt being sent.

        Statuses are `ready` (on disk), `loaded` (resident and serving),
        `downloading`, `loading` and `error`. Exactly one row can be `loaded`.

        Each row carries `gguf_label` (Q4_K_M, IQ4_XS) where the model is a
        GGUF quantization, and the model's `name` embeds it as `repo:QUANT` —
        two quantizations of one repository are DIFFERENT MODELS with different
        sizes and quality, and both are separately selectable.

        `context_length` on the resident model is the context the engine
        ACTUALLY obtained, which is often smaller than the file declares.
        """
        result = await millm.get("/api/models")
        rows = result.get("data") if isinstance(result, dict) else result
        if not isinstance(rows, list):
            return result

        # Read the live context ONLY if something is resident. It is a property
        # of the running engine, so with nothing loaded there is nothing to ask
        # about and the extra round trip buys a null.
        any_loaded = any(
            isinstance(r, dict) and r.get("status") == "loaded" for r in rows
        )
        context = await _actual_context() if any_loaded else None
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if loaded_only and row.get("status") != "loaded":
                continue
            row = dict(row)
            if row.get("status") == "loaded":
                # Reported ONLY for the resident model: it is a property of the
                # live engine, not of the row, and attaching it to a model that
                # is merely on disk would state a context nothing has measured.
                row["context_length"] = context
                if context is None:
                    row["context_length_note"] = (
                        "could not be read from /api/health/inference; the "
                        "model is loaded but its actual window is unknown"
                    )
            out.append(_with_progress_note(row))
        return {"models": out, "count": len(out)}

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_get_model(
        model_id: Annotated[int, Field(description="Model row id from millm_list_models")],
    ) -> Any:
        """One model's full record — status, size, quantization, and download
        progress if it is still downloading.

        Poll this after millm_download_model. `download_progress` is in-process
        and reads null when the download is running in a different worker, so a
        null percentage means "progress unknown", NOT "no progress" — this tool
        says which.
        """
        result = await millm.get(f"/api/models/{model_id}")
        row = result.get("data") if isinstance(result, dict) else result
        return _with_progress_note(row) if isinstance(row, dict) else result

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_preview_model_repo(
        repo_id: Annotated[str, Field(description="HuggingFace repo id, e.g. mradermacher/gemma-4-31b-it-GGUF")],
        hf_token: Annotated[Optional[str], Field(description="Token for a gated repo; never logged")] = None,
    ) -> Any:
        """Inspect a HuggingFace repo WITHOUT downloading it.

        **Call this before millm_download_model on any GGUF repo.** A GGUF
        repository holds many mutually exclusive quantizations, and downloading
        the repo means downloading all of them: 121 GB where the one you want
        is 2.78 GB. That is measured, not illustrative.

        `gguf_quants` lists each selectable quantization with its MEASURED
        total size. Pass a chosen entry's `label` and its files' `path` values
        straight into millm_download_model — they are the input that tool
        expects, so no filename has to be guessed.

        A quantization may be SPLIT across numbered parts (`is_split`). Its
        `files` list is then several entries and EVERY one is required;
        downloading a subset produces a directory that looks complete and a
        model that cannot load.

        `gguf_companions` are projectors and similar side files. They are NOT
        quantizations — an `mmproj-f16.gguf` is a vision projector, and on a
        31B repo it is a 1.2 GB file that would read like a bargain and serve
        nothing. Download one only alongside the quantization it accompanies.
        """
        body: dict[str, Any] = {"repo_id": repo_id}
        if hf_token:
            body["hf_token"] = hf_token
        return await millm.post("/api/models/preview", json_body=body)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_download_model(
        repo_id: Annotated[str, Field(description="HuggingFace repo id")],
        quantization: Annotated[str, Field(description="Coarse bucket: Q2, Q4, Q8, FP16, FP32. For a GGUF repo pass the bucket matching gguf_label")] = "Q4",
        gguf_label: Annotated[Optional[str], Field(description="Exact quantization label from millm_preview_model_repo, e.g. Q4_K_M or IQ4_XS. Required to pick ONE quant from a GGUF repo")] = None,
        gguf_files: Annotated[Optional[list[str]], Field(description="EVERY file path of that one quantization, from the preview. A subset yields a model that cannot load")] = None,
        custom_name: Annotated[Optional[str], Field(description="Display name. Omit to get `repo:QUANT`, which keeps quantizations distinguishable")] = None,
        revision: Annotated[Optional[str], Field(description="Git revision to pin")] = None,
        trust_remote_code: Annotated[bool, Field(description="Required by some architectures; executes repo code at load time")] = False,
        hf_token: Annotated[Optional[str], Field(description="Token for a gated repo; never logged")] = None,
    ) -> Any:
        """Start downloading a model. Returns immediately with the new row's id.

        **This is long and large.** Poll `millm_get_model(model_id)` rather
        than waiting; a quantization of a 31B model is ~16 GB and the whole of
        a GGUF repo can be 121 GB.

        For a GGUF repo, call millm_preview_model_repo first and pass the
        chosen quantization's `label` as `gguf_label` and its file `path`s as
        `gguf_files`. Omitting them downloads EVERY quantization in the repo.

        Leave `custom_name` unset unless you have a reason. The default name is
        `repo:QUANT`, which is what makes two quantizations of one repository
        separately addressable — a custom name is not tagged, so two custom
        names that collide are indistinguishable to every OpenAI client.

        Downloading does not load. Follow with millm_load_model.
        """
        body: dict[str, Any] = {
            "source": "huggingface",
            "repo_id": repo_id,
            "quantization": quantization,
            "trust_remote_code": bool(trust_remote_code),
        }
        if gguf_label:
            body["gguf_label"] = gguf_label
        if gguf_files:
            body["gguf_files"] = list(gguf_files)
        if custom_name:
            body["custom_name"] = custom_name
        if revision:
            body["revision"] = revision
        if hf_token:
            body["hf_token"] = hf_token
        # ⚠ "/api/models", NOT "/api/models/download". miLLM serves the download as
        # POST on the collection; there is no /download sub-path and never was, so this
        # tool 404'd on every call an agent ever made. Found 2026-09-27 by pointing the
        # cross-repo path guard at this module for the first time — miStudio's own caller
        # assertion could not see it, because the tool and its documented path agreed
        # with each other and both were wrong.
        return await millm.post("/api/models", json_body=body)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_cancel_download(
        model_id: Annotated[int, Field(description="Model row id of the download to stop")],
    ) -> Any:
        """Stop an in-progress download and clean up its partial files."""
        return await millm.post(f"/api/models/{model_id}/cancel")

    @mcp.tool()
    async def millm_load_model(
        model_id: Annotated[int, Field(description="Model row id from millm_list_models")],
        approval_request_id: Annotated[Optional[str], Field(description="The id returned by a first call that answered `pending_approval`. Pass it once the operator has approved, to carry out exactly that load")] = None,
        lease_id: Annotated[Optional[str], Field(description="Your lease id from millm_acquire_lease, when you hold the lease; sent as the X-miLLM-Lease header and never stored")] = None,
    ) -> Any:
        """Make a model the one miLLM serves. **This EVICTS whatever is resident** — and, when
        operator approval is on (the default), it waits for the operator first.

        Two shapes, depending on whether you pass `approval_request_id`:

        1. **First call (no id):** files an approval request in miStudio's banner and returns
           `{"approval_request_id", "status": "pending_approval", "hint"}`. Nothing is loaded.
        2. **Re-call with the id** once the operator approved: the approval is consumed (once,
           for exactly this `model_id`) and the load STARTS — poll millm_list_models until its
           status is `loaded`. A re-call before approval answers `{"status": "pending"}`;
           after a denial `denied`; after 24 hours unused `expired`; a second re-call
           `consumed`. Denied and expired need a NEW approval: call again without the id.

        miLLM serves one model at a time, so loading is also an unload of the current one, and
        any SAE attached to it goes with it. Call millm_list_models first if you need to know
        what you are replacing.

        miLLM can still REFUSE an approved load, two ways: a model LOCKED for steering will not
        be swapped out (the refusal names the locked model — steering vectors fitted against one
        model are meaningless against another), and a model whose LEASE is held by someone else
        is refused unless you pass that lease's `lease_id`. A refused load has still spent its
        approval; the result says so.

        The context the engine obtains may be smaller than the file declares — it takes the
        largest that fits in VRAM. millm_list_models reports the actual figure once the load
        completes; check it before sending long prompts.
        """
        # The approved payload is exactly this. NEVER the lease id: it is the only proof of
        # holding a lease (miLLM FR-29.1.6) and an approval row is readable by anyone on the LAN.
        payload = {"model_id": model_id}
        approval_on = ctx.settings.millm_load_approval
        if approval_on and approval_request_id is None:
            try:
                created = await ctx.mistudio.post(
                    "/mcp/approvals",
                    json_body={"tool_name": "millm_load_model", "payload": payload},
                )
            except BackendError as exc:
                if exc.status_code == 0:
                    # The approval queue is miStudio's; without it nothing can be approved.
                    return {"unavailable": "mistudio", "reason": str(exc)}
                raise
            return {
                "approval_request_id": created["id"],
                "status": "pending_approval",
                "hint": (
                    "Loading evicts the resident model, so it needs operator approval in "
                    "miStudio's banner. Re-call millm_load_model with this "
                    "approval_request_id once it is approved."
                ),
            }

        # Gate BEFORE consuming: a down miLLM must not cost an approval (FTID §3.3, TI4).
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}

        if approval_on:
            refusal = await _consume(approval_request_id, payload)
            if refusal is not None:
                return refusal

        headers = {"X-miLLM-Lease": lease_id} if lease_id else None
        try:
            response = await millm.post(
                f"/api/models/{model_id}/load", headers=headers, full=True
            )
        except BackendError as exc:
            if exc.status_code == 0:
                gate.invalidate("millm")
                failed: dict[str, Any] = {"unavailable": "millm", "reason": str(exc)}
            elif not approval_on:
                raise
            else:
                failed = {"load_started": False, "status_code": exc.status_code,
                          "error": exc.detail}
            if approval_on:
                failed["approval_consumed"] = True
                failed["hint"] = (
                    "The approval was spent on this attempt. To try again, call "
                    "millm_load_model without approval_request_id to request a new one."
                )
            return failed
        result = response.body
        if isinstance(result, dict) and result.get("error"):
            return result
        return {
            "load_started": True,
            "model": result.get("data") if isinstance(result, dict) else result,
            "next": (
                "poll millm_list_models until status is 'loaded', then read "
                "context_length before sending long prompts"
            ),
        }

    async def _consume(approval_request_id: str, payload: dict[str, Any]) -> Optional[dict]:
        """Spend the approval. `None` on success; otherwise the structured status to return."""
        try:
            await ctx.mistudio.post(
                f"/mcp/approvals/{approval_request_id}/consume",
                json_body={"tool_name": "millm_load_model", "payload": payload},
            )
            return None
        except BackendError as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            status = detail.get("status")
            if exc.status_code == 409 and status:
                hint = {
                    "pending": "The operator has not decided yet. Re-call later with the "
                               "same approval_request_id.",
                    "consumed": "This approval was already used. Request a new one by calling "
                                "without approval_request_id.",
                }.get(status, "Request a new approval by calling millm_load_model without "
                              "approval_request_id.")
                return {"approval_request_id": approval_request_id, "status": status,
                        "reason": detail.get("reason"), "hint": hint}
            if exc.status_code == 404:
                return {"approval_request_id": approval_request_id, "status": "not_found",
                        "hint": "No such approval. Call without approval_request_id to "
                                "request one."}
            if exc.status_code == 422:
                return {"approval_request_id": approval_request_id, "status": "mismatch",
                        "reason": exc.detail,
                        "hint": "That approval is for a different load. An approval covers "
                                "exactly the model it names."}
            raise

    # ── Leases (Feature 034, FR-18; miLLM Feature 29) ─────────────────────────────────────
    #
    # A lease pins the RESIDENT model for a holder: while it is live, nobody else's load evicts
    # it and nobody else's request auto-loads over it. It loads nothing, so it needs no approval
    # (P-05). The lease id is the only proof of holding: it is returned once, by the grant, and
    # travels afterwards ONLY in the `X-miLLM-Lease` header — never a path, a body, an approval
    # payload or a log line.

    def _ttl_refusal(ttl_seconds: Optional[int]) -> Optional[dict]:
        """1-7200, validated before the gate and REFUSED, never clamped (miLLM FR-29.1.2)."""
        if ttl_seconds is not None and not 1 <= ttl_seconds <= MAX_LEASE_TTL_S:
            return {"error": "invalid_ttl",
                    "message": f"ttl_seconds must be 1-{MAX_LEASE_TTL_S} (got {ttl_seconds}); "
                               "it is refused, never clamped. Omit it for the maximum."}
        return None

    @mcp.tool()
    async def millm_acquire_lease(
        model_id: Annotated[int, Field(description="Model row id of the RESIDENT model (millm_list_models: status 'loaded')")],
        reason: Annotated[str, Field(description="Why you are pinning it, shown to anyone refused because of your lease (1-512 chars)", min_length=1, max_length=512)],
        ttl_seconds: Annotated[Optional[int], Field(description=f"Seconds until the lease expires, 1-{MAX_LEASE_TTL_S}. Omit for {MAX_LEASE_TTL_S}")] = None,
    ) -> Any:
        """Pin the model miLLM is serving so nobody else's load or request swaps it out.

        Returns the lease, including `lease_id` — **the only time it is ever returned**. Keep
        it: pass it as `lease_id` to millm_load_model and every scoring, generation and batch
        tool, and to millm_renew_lease / millm_release_lease. It cannot be retrieved later.

        One lease per server, and only on the RESIDENT model (this loads nothing). A lease ends
        at its expiry, on release, when the model stops being resident, and at EVERY miLLM
        restart. Holder is this MCP path's identity. No operator approval is needed: a lease
        evicts nothing. Refusals: `MODEL_NOT_RESIDENT`, `MODEL_LEASED` (someone else holds it),
        `INVALID_LEASE_REQUEST`.
        """
        refused = _ttl_refusal(ttl_seconds)
        if refused:
            return refused
        ok, reason_unavailable = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason_unavailable}
        body: dict[str, Any] = {"holder": ctx.identity, "reason": reason}
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        lease = await millm.post(f"/api/models/{model_id}/lease", json_body=body)
        return {
            "lease": lease,
            "hint": "lease_id is shown ONCE. Pass it as lease_id to keep using this model; "
                    "renew before expires_at, release when done.",
        }

    @mcp.tool()
    async def millm_renew_lease(
        model_id: Annotated[int, Field(description="Model row id the lease is on")],
        lease_id: Annotated[str, Field(description="The lease_id millm_acquire_lease returned")],
        ttl_seconds: Annotated[Optional[int], Field(description=f"New TTL counted from NOW, 1-{MAX_LEASE_TTL_S}. Omit for {MAX_LEASE_TTL_S}")] = None,
    ) -> Any:
        """Extend your lease: the new expiry is now + ttl_seconds.

        After a miLLM restart every lease is gone and this answers `LEASE_NOT_FOUND` — acquire a
        new one. An expired lease answers `LEASE_EXPIRED`.
        """
        refused = _ttl_refusal(ttl_seconds)
        if refused:
            return refused
        ok, reason_unavailable = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason_unavailable}
        body = {"ttl_seconds": ttl_seconds} if ttl_seconds is not None else None
        return await millm.post(
            f"/api/models/{model_id}/lease/renew", json_body=body,
            headers={"X-miLLM-Lease": lease_id},
        )

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_release_lease(
        model_id: Annotated[int, Field(description="Model row id the lease is on")],
        lease_id: Annotated[str, Field(description="The lease_id millm_acquire_lease returned")],
    ) -> Any:
        """End your lease now, so others may load or swap the model again."""
        return await millm.delete(
            f"/api/models/{model_id}/lease", headers={"X-miLLM-Lease": lease_id}
        )

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_lease_status(
        model_id: Annotated[int, Field(description="Model row id")],
    ) -> Any:
        """Who holds a model's lease, why and until when — and the last lease that ended.

        Never returns a lease id: a lease you did not acquire cannot be used, only waited for.
        """
        return await millm.get(f"/api/models/{model_id}/lease")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_unload_model(
        model_id: Annotated[int, Field(description="Model row id to evict from GPU memory")],
    ) -> Any:
        """Evict a model from GPU memory, freeing the card.

        Waits for pending inference to finish first. After this, miLLM serves
        nothing until something is loaded — an OpenAI request naming a model
        auto-loads it, so this is about freeing VRAM, not about blocking use.
        """
        return await millm.post(f"/api/models/{model_id}/unload")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_delete_model(
        model_id: Annotated[int, Field(description="Model row id to delete permanently")],
        acknowledge_permanent: Annotated[bool, Field(description="Delete the row AND its files from disk. Irreversible; re-downloading a large quantization takes tens of GB and a long wait")] = False,
    ) -> Any:
        """Delete a model permanently — the database row and the files on disk.

        Refuses unless `acknowledge_permanent=True`, and the refusal names the
        model, its quantization and its size so a stale id is visible before it
        costs anything. There is no undo: re-acquiring a 16 GB quantization is
        a fresh download.

        miLLM already refuses to delete a LOADED model outright, with no
        override — unload it first. What was unprotected was everything else: a
        READY 16 GB download had nothing between it and a mistyped id, which is
        precisely the case an agent asked to "clean up the old model" hits.
        This mirrors `millm_delete_circuit`'s gate.

        The size lookup is best-effort and deliberately FAILS OPEN: if the row
        cannot be read the delete still proceeds once acknowledged, so the tool
        stays usable during an outage. It guards against the plausible mistake,
        not against a determined caller.
        """
        if not acknowledge_permanent:
            row = None
            try:
                result = await millm.get(f"/api/models/{model_id}")
                row = result.get("data") if isinstance(result, dict) else result
            except Exception:
                row = None

            described = "this model"
            if isinstance(row, dict):
                label = row.get("gguf_label")
                size = row.get("disk_size_mb")
                described = str(row.get("name") or f"model {model_id}")
                if label:
                    described += f" ({label})"
                if isinstance(size, (int, float)) and size > 0:
                    described += f", {size / 1024:.1f} GB on disk"
            return {
                "refused": "delete_not_acknowledged",
                "model_id": model_id,
                "would_delete": described,
                "status": row.get("status") if isinstance(row, dict) else None,
                "reason": (
                    f"Deleting {described} removes the row AND its files. This "
                    "cannot be undone — getting it back means downloading it "
                    "again. Confirm this is the right model, then pass "
                    "acknowledge_permanent=true."
                ),
            }
        return await millm.delete(f"/api/models/{model_id}")


def _with_progress_note(row: dict) -> dict:
    """Say whether a download has no progress or merely UNREPORTED progress.

    `download_progress` is written in-process. A download running in another
    worker — or one that has just restarted — reads null, and a client that
    renders null as 0% shows a stalled bar over a download that is proceeding
    normally. The two cases need different reactions from an agent (wait, vs
    investigate), so they must not render identically.
    """
    if row.get("status") != "downloading":
        return row
    row = dict(row)
    if row.get("download_progress") is None:
        row["progress_note"] = (
            "downloading; progress is UNKNOWN, not zero — the percentage is "
            "tracked in-process and is unreadable from another worker. Poll "
            "millm_get_model and watch `status` rather than the percentage."
        )
    return row
