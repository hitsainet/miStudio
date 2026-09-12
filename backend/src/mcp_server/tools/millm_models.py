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

from ..health_gate import HealthGate, gated
from ..millm_client import MiLLMClient


def register(mcp: FastMCP, millm: MiLLMClient, gate: HealthGate) -> None:
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
        return await millm.post("/api/models/download", json_body=body)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_cancel_download(
        model_id: Annotated[int, Field(description="Model row id of the download to stop")],
    ) -> Any:
        """Stop an in-progress download and clean up its partial files."""
        return await millm.post(f"/api/models/{model_id}/cancel")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_load_model(
        model_id: Annotated[int, Field(description="Model row id from millm_list_models")],
    ) -> Any:
        """Make a model the one miLLM serves. Returns as soon as the load
        STARTS — poll millm_list_models until its status is `loaded`.

        **This EVICTS whatever is resident.** miLLM serves one model at a time,
        so loading is also an unload of the current one, and any SAE attached
        to it goes with it. Call millm_list_models first if you need to know
        what you are replacing.

        A model LOCKED for steering will not be swapped out: the refusal names
        the locked model, because steering vectors fitted against one model are
        meaningless against another.

        The context the engine obtains may be smaller than the file declares —
        it takes the largest that fits in VRAM. millm_list_models reports the
        actual figure once the load completes; check it before sending long
        prompts.
        """
        result = await millm.post(f"/api/models/{model_id}/load")
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
