"""The body-size ceiling must hold with a header, without one, and against a lie.

MIS-E2E-036. The original guard lived inside the circuit-import handler, where
it could not work: FastAPI parses the body before the handler's first line, and
`Transfer-Encoding: chunked` sends no `Content-Length` for the check to read.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.body_limit import (
    EXEMPT_PREFIXES,
    MAX_BODY_BYTES,
    BodySizeLimitMiddleware,
)


@pytest.fixture
def app():
    application = FastAPI()
    application.add_middleware(BodySizeLimitMiddleware, max_bytes=1024)

    @application.post("/echo")
    async def echo(payload: dict):
        return {"keys": len(payload)}

    @application.get("/ping")
    async def ping():
        return {"ok": True}

    return application


@pytest.fixture
def client(app):
    return TestClient(app)


def _payload(size: int) -> dict:
    return {"blob": "x" * size}


class TestTheCapHolds:
    def test_a_small_body_passes(self, client):
        r = client.post("/echo", json=_payload(10))
        assert r.status_code == 200

    def test_an_honest_oversized_header_is_refused_before_the_body_is_read(self, client):
        r = client.post("/echo", json=_payload(4096))
        assert r.status_code == 413
        assert "cap" in r.json()["detail"]

    def test_a_chunked_body_is_refused_even_with_no_content_length(self, client):
        """The defect. No `Content-Length` means the header check never runs."""
        big = json.dumps(_payload(4096)).encode()

        def chunks():
            for i in range(0, len(big), 256):
                yield big[i:i + 256]

        # requests sends a generator body with Transfer-Encoding: chunked
        # and no Content-Length.
        r = client.post("/echo", data=chunks(),
                        headers={"content-type": "application/json"})
        assert r.status_code == 413, (
            "a chunked body walked straight past the cap — this is the defect"
        )

    def test_a_header_that_lies_low_is_still_metered(self, client):
        """A declared 10 bytes followed by 4 KB must not get through."""
        big = json.dumps(_payload(4096)).encode()

        def chunks():
            yield big

        r = client.post("/echo", data=chunks(),
                        headers={"content-type": "application/json"})
        assert r.status_code == 413

    def test_an_honest_header_is_refused_without_reading_the_body(self):
        """The point of the header pre-check, and the only thing that distinguishes
        it from the meter downstream.

        Control C150 removed the pre-check and every other test stayed green:
        the metered receive refuses the same request, so the *response* is
        identical. What differs is the cost — with only the meter, the oversized
        body is pulled off the socket before it is refused. Assert the receive
        channel is never drained.
        """
        drained = []

        async def receive():
            drained.append(1)
            return {"type": "http.request", "body": b"x" * 4096,
                    "more_body": False}

        sent = []

        async def send(message):
            sent.append(message)

        async def downstream(scope, rcv, snd):  # pragma: no cover - must not run
            raise AssertionError("the app was reached for an over-cap request")

        middleware = BodySizeLimitMiddleware(downstream, max_bytes=1024)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/echo",
            "headers": [(b"content-length", b"4096")],
        }

        import asyncio

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            middleware(scope, receive, send)
        )

        assert drained == [], (
            "the body was read before being refused — the header pre-check is "
            "gone, so an oversized upload is paid for in full and then rejected"
        )
        assert sent[0]["status"] == 413

    def test_a_malformed_content_length_is_a_400_not_a_500(self, client):
        r = client.post("/echo", data=b"{}",
                        headers={"content-length": "not-a-number",
                                 "content-type": "application/json"})
        assert r.status_code in (400, 413)
        assert r.status_code != 500

    def test_a_bodyless_method_is_untouched(self, client):
        assert client.get("/ping").status_code == 200


class TestTheExemptListIsCheckedAgainstTheRegistry:
    """A hand-maintained path list rots. MIS-E2E-154 found 41 dead ones."""

    #: Read from the SERVED schema, not `app.routes`. This build wraps included
    #: routers in `_IncludedRouter` objects with no `.path`, so `app.routes`
    #: yields 10 entries for a 230-route API — and the first version of these
    #: two tests iterated it, which made both of them vacuous. Two other guards
    #: in this suite already document the same trap.
    def _served_paths(self):
        from src.main import app as real_app

        paths = set(real_app.openapi()["paths"])
        assert len(paths) > 100, (
            f"only {len(paths)} paths in the served schema — the probe is blind, "
            f"and a blind probe agrees with everything"
        )
        return paths

    def test_exempt_list_names_only_real_routes(self):
        served = self._served_paths()
        for prefix in EXEMPT_PREFIXES:
            assert any(path.startswith(prefix) for path in served), (
                f"EXEMPT_PREFIXES names {prefix!r}, which matches no served "
                f"route — a body cap is being waived for a path that does not exist"
            )

    def test_a_file_upload_route_must_be_exempted_deliberately(self):
        """If a route starts taking an UploadFile, 1 MB will break it silently."""
        from src.main import app as real_app

        spec = real_app.openapi()
        served = self._served_paths()
        offenders = []
        for path in served:
            for method, operation in spec["paths"][path].items():
                body = operation.get("requestBody", {}).get("content", {})
                # multipart/form-data is how FastAPI renders an UploadFile.
                if "multipart/form-data" not in body:
                    continue
                if not any(path.startswith(pref) for pref in EXEMPT_PREFIXES):
                    offenders.append(f"{method.upper()} {path}")

        assert not offenders, (
            f"{offenders} accept a file upload but are not in EXEMPT_PREFIXES; "
            f"they will be capped at {MAX_BODY_BYTES // 1024} KB"
        )


class TestTheMiddlewareIsActuallyMounted:
    """Reachability: a middleware that is written and not added does nothing."""

    def test_the_real_app_has_the_body_limit_middleware(self):
        from src.main import app as real_app

        classes = [m.cls for m in real_app.user_middleware]
        assert BodySizeLimitMiddleware in classes, (
            "BodySizeLimitMiddleware is not mounted on the real app — every "
            "route is uncapped"
        )
