"""
Shared utilities for interacting with the miLLM management API.
"""

import logging
import time
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_MODEL_LOAD_TIMEOUT = 300  # seconds to wait for miLLM model to reach LOADED status


def ensure_model_loaded(endpoint_url: str, model_name: str) -> None:
    """
    If ``endpoint_url`` points to a miLLM instance, ensure the named model is loaded.

    Silently no-ops when:
    - The endpoint is not a miLLM instance (non-200 from /api/models)
    - The model is already loaded
    - Any unexpected error occurs (so the main task can still attempt the call)
    """
    try:
        parsed = urlparse(endpoint_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        resp = requests.get(f"{base_url}/api/models", timeout=10)
        if resp.status_code != 200:
            return  # Not a miLLM instance or unreachable

        payload = resp.json()
        if not payload.get("success"):
            return

        models = payload.get("data") or []

        model_entry = None
        for m in models:
            if m.get("name", "").lower() == model_name.lower():
                model_entry = m
                break

        if model_entry is None:
            logger.warning(
                "miLLM model %r not found at %s — proceeding without pre-load",
                model_name,
                base_url,
            )
            return

        model_id = model_entry["id"]
        status = model_entry.get("status", "")

        if status == "loaded":
            return  # Already in GPU — nothing to do

        if status in ("downloading", "error"):
            logger.warning(
                "miLLM model %r has status=%r; cannot load — proceeding anyway",
                model_name,
                status,
            )
            return

        # status is "ready" or "loading" — trigger load if needed
        if status == "ready":
            logger.info(
                "miLLM model %r (id=%s) is not loaded; triggering load...",
                model_name,
                model_id,
            )
            load_resp = requests.post(f"{base_url}/api/models/{model_id}/load", timeout=30)
            if load_resp.status_code not in (200, 202):
                logger.warning(
                    "miLLM load request returned %d — proceeding anyway",
                    load_resp.status_code,
                )
                return

        # Poll until loaded or timeout
        deadline = time.time() + _MODEL_LOAD_TIMEOUT
        while time.time() < deadline:
            time.sleep(5)
            poll = requests.get(f"{base_url}/api/models/{model_id}", timeout=10)
            if poll.status_code != 200:
                break
            poll_payload = poll.json()
            current_status = (poll_payload.get("data") or {}).get("status", "")
            if current_status == "loaded":
                logger.info("miLLM model %r loaded successfully", model_name)
                return
            if current_status == "error":
                logger.error("miLLM model %r failed to load (status=error)", model_name)
                return

        logger.warning(
            "miLLM model %r did not reach loaded status within %ds",
            model_name,
            _MODEL_LOAD_TIMEOUT,
        )

    except Exception as exc:
        logger.warning("ensure_model_loaded: %s (proceeding anyway)", exc)
