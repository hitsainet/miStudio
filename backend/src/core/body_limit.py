"""A request-body size ceiling that applies to every route, not a chosen few.

MIS-E2E-036. The circuit import endpoint carried this check itself:

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_IMPORT_BYTES:
        raise HTTPException(413, ...)

Two things were wrong with it, and the second is the reason this moved.

1. **It is skipped for a chunked request.** `Transfer-Encoding: chunked` sends
   no `Content-Length`, so `if content_length:` is false and the cap does not
   run. A client that wants past the cap simply omits the header.

2. **It cannot work where it was written.** The handler signature was
   `async def import_circuit(payload: Dict[str, Any], request: Request, ...)`.
   FastAPI reads and JSON-parses the whole body to build `payload` *before* the
   handler's first statement executes. By the time the check ran, the bytes it
   was meant to refuse were already read and already parsed into memory. The
   guard could reject the response, never the cost.

A sibling sweep found the same shape spread unevenly: `cluster_profiles`
measured `approx_size` after parsing (same ordering problem), and the four
template import endpoints — labeling, extraction, training, prompt — had no cap
at all. Fixing the one endpoint the finding named would have left four open.

So the ceiling lives in ASGI middleware, below the framework:

- a declared `Content-Length` over the cap is refused before the body is read;
- the receive channel is metered, so a chunked or lying-header body is refused
  at the moment it crosses the cap rather than after it is buffered.

Routes that legitimately accept large uploads are listed in `EXEMPT_PREFIXES`
and are exempt from the cap only, never from their own validation.
"""

import logging
from typing import Iterable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

#: 1 MB. A circuit definition, a cluster profile or a template bundle is a few
#: KB; anything approaching this is hostile rather than merely large.
MAX_BODY_BYTES = 1_048_576

#: Paths that carry real file uploads and must not be capped at 1 MB. Prefix
#: matched against the request path.
#:
#: Deliberately EMPTY. The first draft of this list named
#: `/api/v1/datasets/upload`, which does not exist — the same dead-path defect
#: MIS-E2E-154 catalogued 41 of. No route in this API declares `UploadFile`;
#: every body is JSON, and a JSON body over 1 MB is hostile on all of them.
#: `test_body_limit.py::test_exempt_list_names_only_real_routes` fails if an
#: entry stops matching a mounted route, and
#: `test_a_file_upload_route_must_be_exempted_deliberately` fails if a route
#: starts accepting `UploadFile` without a decision being recorded here — so
#: the list is checked against the registry rather than trusted.
EXEMPT_PREFIXES: tuple[str, ...] = ()

_METHODS_WITH_BODIES = frozenset({"POST", "PUT", "PATCH"})


class BodyTooLarge(Exception):
    """Raised through the ASGI receive channel when the cap is crossed."""


def _is_exempt(path: str, exempt: Iterable[str]) -> bool:
    return any(path.startswith(prefix) for prefix in exempt)


class BodySizeLimitMiddleware:
    """Refuse a request body over `max_bytes`, header or no header."""

    def __init__(
        self,
        app: ASGIApp,
        max_bytes: int = MAX_BODY_BYTES,
        exempt_prefixes: tuple[str, ...] = EXEMPT_PREFIXES,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.exempt_prefixes = exempt_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in _METHODS_WITH_BODIES:
            await self.app(scope, receive, send)
            return

        if _is_exempt(scope.get("path", ""), self.exempt_prefixes):
            await self.app(scope, receive, send)
            return

        # Cheap path: an honest header lets us refuse before reading anything.
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    await self._refuse(send, "Malformed Content-Length header", 400)
                    return
                if declared > self.max_bytes:
                    await self._refuse(send, self._too_large_detail())
                    return
                break

        # Expensive path: meter what actually arrives. This is what catches a
        # chunked body and a header that lies about its size.
        read = 0
        exceeded = False

        async def metered_receive() -> Message:
            nonlocal read, exceeded
            message = await receive()
            if message["type"] == "http.request":
                read += len(message.get("body", b""))
                if read > self.max_bytes:
                    exceeded = True
                    # End the stream so the framework's body read returns
                    # rather than hanging waiting for a body we will refuse.
                    return {"type": "http.disconnect"}
            return message

        sent_status: dict = {}

        async def guarded_send(message: Message) -> None:
            # If the cap was crossed mid-body the downstream app may still try
            # to answer (usually a 422 from a truncated parse). Replace the
            # first response with the honest 413 and drop the rest.
            if exceeded:
                if message["type"] == "http.response.start" and not sent_status:
                    sent_status["done"] = True
                    await self._refuse(send, self._too_large_detail())
                return
            await send(message)

        await self.app(scope, metered_receive, guarded_send)

        if exceeded and not sent_status:
            await self._refuse(send, self._too_large_detail())

    def _too_large_detail(self) -> str:
        return f"Request body exceeds the {self.max_bytes // 1024} KB cap"

    async def _refuse(self, send: Send, detail: str, status: int = 413) -> None:
        import json

        body = json.dumps({"detail": detail}).encode()
        logger.warning("Refused a request body: %s", detail)
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})
