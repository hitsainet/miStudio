"""Version endpoint."""

import logging
import os
from pathlib import Path

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/version", tags=["version"])

#: What `_read_version` returns when it genuinely cannot tell. Distinguishable
#: from a real version by a caller, and logged at ERROR so it does not sit
#: unnoticed the way `"unknown"` did.
UNKNOWN = "unknown"


def _read_version() -> str:
    """This build's version.

    MIS-E2E-028: every deployed pod reported `"unknown"`. The Docker build
    context is `backend/` while `VERSION` lives at the repo ROOT, so the file
    never entered the image and all three candidate paths missed — and the
    fallback was a plausible-looking string, so nothing surfaced the failure.
    A version check could not distinguish a stale pod from a current one.

    `MISTUDIO_VERSION` is the container's answer, baked in at build time by the
    Dockerfile's `APP_VERSION` build arg. The file probes remain for local runs
    from a source checkout, where the repo root really is above this module.
    """
    baked = os.environ.get("MISTUDIO_VERSION", "").strip()
    if baked and baked != UNKNOWN:
        return baked

    for candidate in [
        Path("/app/VERSION"),                    # container, written at build
        Path(__file__).parents[5] / "VERSION",   # repo root, source checkout
        Path(__file__).parents[4] / "VERSION",
        Path(__file__).parents[3] / "VERSION",
    ]:
        try:
            if candidate.exists():
                text = candidate.read_text().strip()
                if text:
                    return text
        except OSError:
            continue

    # LOUD. Returning a plausible string quietly is what let this sit in
    # production unnoticed — the endpoint answered, and the answer was wrong.
    logger.error(
        "Version could not be determined: MISTUDIO_VERSION is unset and no "
        "VERSION file was found. The image was probably built without the "
        "APP_VERSION build arg (see backend/Dockerfile)."
    )
    return UNKNOWN


@router.get("", summary="Get application version")
def get_version():
    return {"version": _read_version(), "app": "miStudio"}
