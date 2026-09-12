"""Wave 7: beat expiry, the bearer comparison, and a claim that was false.

MIS-E2E-095, -117, -094.
"""

import ast
import hmac
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"


class TestEveryBeatEntryExpires:
    """MIS-E2E-095. A tick with no expiry survives the backlog that delayed it.

    Derived from the loaded config, not from the source text: a `grep` for
    `expires` counts occurrences in comments too, and the point is what Celery
    actually receives.
    """

    @pytest.fixture
    def beat_schedule(self):
        from src.core.celery_app import celery_app

        return celery_app.conf.beat_schedule

    def test_the_schedule_is_not_empty(self, beat_schedule):
        # Guards the test itself: an empty dict passes every check below.
        assert len(beat_schedule) >= 10, (
            f"only {len(beat_schedule)} beat entries — did the schedule move?"
        )

    def test_every_entry_is_a_shape_celery_accepts(self, beat_schedule):
        """Build the real object. This is the test that was missing.

        The first version of this fix put `expires` at the TOP LEVEL of each
        entry, and the first version of this test asserted `"expires" in cfg`
        — so the test and the code shared one wrong mental model and agreed
        with each other. The suite was green and `celery beat` crashlooped in
        production with:

            TypeError: ScheduleEntry.__init__() got an unexpected keyword
            argument 'expires'

        `expires` is a message option; it belongs in `options`. Nothing short
        of constructing what Celery constructs would have caught that.
        """
        from celery.beat import ScheduleEntry

        from src.core.celery_app import celery_app

        broken = {}
        for name, cfg in beat_schedule.items():
            try:
                ScheduleEntry(name=name, app=celery_app, **cfg)
            except Exception as exc:  # noqa: BLE001 - reporting any rejection
                broken[name] = f"{type(exc).__name__}: {exc}"
        assert not broken, (
            f"celery beat cannot build these entries and will crashloop on "
            f"startup: {broken}"
        )

    def test_every_entry_declares_an_expiry(self, beat_schedule):
        missing = [
            name for name, cfg in beat_schedule.items()
            if "expires" not in cfg.get("options", {})
        ]
        assert not missing, (
            f"{missing} have no `expires` in their options. A tick queued "
            f"behind a long GPU task is still delivered when the queue drains, "
            f"so an hour of blocked ticks fires at once against state that has "
            f"moved on."
        )

    def test_the_expiry_survives_into_the_built_entry(self, beat_schedule):
        """Present in the dict is not the same as reaching Celery."""
        from celery.beat import ScheduleEntry

        from src.core.celery_app import celery_app

        for name, cfg in beat_schedule.items():
            entry = ScheduleEntry(name=name, app=celery_app, **cfg)
            assert entry.options.get("expires"), (
                f"{name} builds, but its expiry did not reach the entry's "
                f"options — the message will be sent without one"
            )

    def test_each_expiry_is_shorter_than_its_own_period(self, beat_schedule):
        wrong = {
            name: (cfg["schedule"], cfg["options"]["expires"])
            for name, cfg in beat_schedule.items()
            if not isinstance(cfg["schedule"], (int, float))
            or cfg["options"]["expires"] >= cfg["schedule"]
        }
        assert not wrong, (
            f"{wrong}: an expiry at or beyond the period never discards anything "
            f"— the next tick is already due by then"
        )


class TestBearerComparisonSurvivesAnyAlphabet:
    """MIS-E2E-117. `compare_digest` on two `str` raises for non-ASCII."""

    def test_the_underlying_defect_is_real(self):
        with pytest.raises(TypeError):
            hmac.compare_digest("tökén", "secret")

    def test_the_middleware_compares_bytes(self):
        """Parsed, not grepped — the explanatory comment names the old call."""
        source = (SRC / "mcp_server" / "server.py").read_text()
        tree = ast.parse(source)

        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "compare_digest"
        ]
        assert calls, "the bearer check is gone entirely"
        for call in calls:
            for arg in call.args:
                assert (
                    isinstance(arg, ast.Call)
                    and getattr(arg.func, "attr", "") == "encode"
                ), (
                    "compare_digest is being handed a str; a non-ASCII bearer "
                    "token raises TypeError and returns 500 from an "
                    "unauthenticated, LAN-reachable port instead of 401"
                )

    @pytest.mark.asyncio
    async def test_a_non_ascii_token_is_a_401_not_a_500(self):
        from starlette.requests import Request

        from src.mcp_server.server import BearerAuthMiddleware

        middleware = BearerAuthMiddleware(app=None, token="s3cr3t")

        async def receive():  # pragma: no cover - never awaited
            return {"type": "http.request"}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [(b"authorization", "Bearer tökén".encode("utf-8"))],
            "query_string": b"",
        }
        request = Request(scope, receive)

        async def call_next(_):  # pragma: no cover - must not be reached
            raise AssertionError("a bad token reached the application")

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_the_right_token_still_gets_through(self):
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        from src.mcp_server.server import BearerAuthMiddleware

        middleware = BearerAuthMiddleware(app=None, token="s3cr3t")

        async def receive():  # pragma: no cover
            return {"type": "http.request"}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [(b"authorization", b"Bearer s3cr3t")],
            "query_string": b"",
        }

        async def call_next(_):
            return JSONResponse({"ok": True})

        response = await middleware.dispatch(Request(scope, receive), call_next)
        assert response.status_code == 200


class TestNoFalseClaimAboutVramReclaim:
    """MIS-E2E-094. The comment promised a reclaim that cannot happen."""

    def test_the_setting_is_not_described_as_restarting_the_solo_worker(self):
        source = (SRC / "core" / "celery_app.py").read_text()
        # The corrective comment must PARAPHRASE the removed claim, never quote
        # it: a substring check cannot tell a quotation from an assertion, and
        # that trap has now appeared eight times in this remediation. Stripping
        # quoted spans was tried and is not enough either — the quote wrapped
        # across a newline and slipped through the regex.
        assert "this triggers a full worker restart" not in source, (
            "the comment claims --pool=solo recycles the worker after N tasks. "
            "It does not: max_tasks_per_child recycles a prefork CHILD, and the "
            "solo pool has none. The GPU memory it promises to clean is cleaned "
            "by explicit empty_cache() calls and the gpu watchdog instead."
        )

    def test_the_file_says_plainly_that_the_setting_is_inert(self):
        """A negative check alone passes on a file that says nothing at all."""
        source = (SRC / "core" / "celery_app.py").read_text()
        assert "does NOTHING under --pool=solo" in source, (
            "the note explaining that max_tasks_per_child is inert under the "
            "solo pool is gone; the next reader will assume it recycles"
        )

    def test_the_real_reclaim_mechanism_still_exists(self):
        """If this goes, the corrected comment starts lying in the other direction."""
        callers = [
            path for path in SRC.rglob("*.py")
            if "empty_cache" in path.read_text()
        ]
        assert len(callers) >= 10, (
            f"only {len(callers)} modules call empty_cache; the comment in "
            f"celery_app.py points at them as the actual reclaim path"
        )
