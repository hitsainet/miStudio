"""
miStudio model management (category: models) — the weights miStudio runs
extraction, training, steering and J-lens work against.

**miStudio has no residency.** It loads a model for the duration of a task and
releases it, so there is nothing to load, nothing to unload, and no eviction to
warn about. That is the one real difference from the `millm_models` category,
which manages a single resident model that is SERVING; the verbs are otherwise
deliberately the same so an agent that has learned one surface can use the
other.

Model TRAINING and the SAE lifecycle have their own tools. This is acquisition
and disposal only.
"""

from typing import Annotated, Any, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def list_models(
        search: Annotated[Optional[str], Field(description="Substring match on the model name")] = None,
        status: Annotated[Optional[str], Field(description="Filter by status: ready, downloading, loading, quantizing, error")] = None,
        architecture: Annotated[Optional[str], Field(description="Filter by architecture, e.g. llama, gemma3, granitemoehybrid")] = None,
        limit: Annotated[int, Field(description="Rows to return (max 100)", ge=1, le=100)] = 50,
    ) -> Any:
        """What models miStudio has, and whether each is usable.

        Start here before any extraction, training or J-lens fit — every one of
        those names a model id, and a model in `downloading` or `error` is not
        one of them.

        Note this is miStudio's OWN catalogue. miLLM keeps a separate one for
        what it SERVES; use millm_list_models for that. A repo present in both
        is two independent downloads on disk.
        """
        params: dict[str, Any] = {"limit": limit}
        if search:
            params["search"] = search
        if status:
            params["status"] = status
        if architecture:
            params["architecture"] = architecture
        return await client.get("/models", **params)

    @mcp.tool()
    async def get_model(
        model_id: Annotated[str, Field(description="Model id (m_xxxxxxxx) from list_models")],
    ) -> Any:
        """One model's record — status, quantization, disk size, architecture,
        and the error message if a download failed.

        Poll this after download_model: the download is a background job and
        the row's `status` is what says whether it finished.
        """
        return await client.get(f"/models/{model_id}")

    @mcp.tool()
    async def get_model_architecture(
        model_id: Annotated[str, Field(description="Model id (m_xxxxxxxx)")],
    ) -> Any:
        """The model's discovered layer structure — layer count, hidden size,
        and the module paths extraction and steering hook.

        Read this before choosing layers for an extraction or a J-lens fit
        rather than assuming a layer count. Architectures here are discovered
        dynamically, not whitelisted, so a model's real depth is a fact about
        the checkpoint and never a default: a guessed layer index silently
        captures the wrong thing.
        """
        return await client.get(f"/models/{model_id}/architecture")

    @mcp.tool()
    async def download_model(
        repo_id: Annotated[str, Field(description="HuggingFace repo id, e.g. google/gemma-2-2b")],
        quantization: Annotated[str, Field(description="FP32, FP16, Q8, Q4 or Q2. FP16 is the default and the safest for interpretability work")] = "FP16",
        trust_remote_code: Annotated[bool, Field(description="Required by some architectures; executes repo code at load time")] = False,
        access_token: Annotated[Optional[str], Field(description="Token for a gated repo; never logged")] = None,
    ) -> Any:
        """Start downloading a model into miStudio's catalogue. Returns
        immediately with the new row's id; poll `get_model(model_id)`.

        Downloads are large and slow — a 12B checkpoint at FP16 is ~24 GB.

        **Quantization is not a free choice for interpretability work.**
        bitsandbytes replaces the linear layers and leaves everything else at
        the checkpoint's dtype, so a quantized model is a different computation
        from the one being studied. Q8/Q4 exist here because a large model may
        not otherwise fit the card at all; prefer FP16 when it fits.

        miStudio does not select individual GGUF quantizations — that is a
        miLLM capability, because miStudio needs a PyTorch module tree to hook
        and a GGUF file served through llama.cpp has none. Use
        millm_download_model for GGUF.
        """
        body: dict[str, Any] = {
            "repo_id": repo_id,
            "quantization": quantization,
            "trust_remote_code": bool(trust_remote_code),
        }
        if access_token:
            body["access_token"] = access_token
        return await client.post("/models/download", json_body=body)

    @mcp.tool()
    async def cancel_model_download(
        model_id: Annotated[str, Field(description="Model id of the download to stop")],
    ) -> Any:
        """Stop an in-progress download and clean up its partial files."""
        return await client.delete(f"/models/{model_id}/cancel")

    @mcp.tool()
    async def delete_model(
        model_id: Annotated[str, Field(description="Model id (m_xxxxxxxx) to delete permanently")],
        acknowledge_permanent: Annotated[bool, Field(description="Delete the row AND queue its files for removal. Irreversible; re-downloading a large checkpoint takes tens of GB and a long wait")] = False,
    ) -> Any:
        """Delete a model permanently — the database row, and its files on disk
        via a background cleanup job.

        Refuses unless `acknowledge_permanent=True`, and the refusal names the
        model and its size so a stale id is visible before it costs anything.
        There is no undo.

        miStudio already refuses (409) when a TRAINING references the model, so
        that case is protected. What was not protected is everything else: a
        `ready` 24 GB checkpoint with no training against it yet had nothing
        between it and a mistyped id — precisely the case an agent asked to
        "clean up the old model" hits. This mirrors `millm_delete_circuit`'s
        gate.

        Extractions, SAEs and J-lens artifacts fitted from this model are NOT
        deleted with it, and they name a model that no longer exists.

        The lookup is best-effort and deliberately FAILS OPEN: if the row
        cannot be read the delete still proceeds once acknowledged, so the tool
        stays usable during an outage.
        """
        if not acknowledge_permanent:
            row = None
            try:
                result = await client.get(f"/models/{model_id}")
                row = result.get("data") if isinstance(result, dict) else result
            except Exception:
                row = None

            described = f"model {model_id}"
            status = None
            if isinstance(row, dict):
                status = row.get("status")
                described = str(row.get("name") or described)
                size = row.get("size_mb") or row.get("disk_size_mb")
                if isinstance(size, (int, float)) and size > 0:
                    described += f", {size / 1024:.1f} GB on disk"
            return {
                "refused": "delete_not_acknowledged",
                "model_id": model_id,
                "would_delete": described,
                "status": status,
                "reason": (
                    f"Deleting {described} removes the row and queues its files "
                    "for removal. This cannot be undone, and any extraction, "
                    "SAE or J-lens artifact fitted from it will name a model "
                    "that no longer exists. Confirm this is the right model, "
                    "then pass acknowledge_permanent=true."
                ),
            }
        return await client.delete(f"/models/{model_id}")
