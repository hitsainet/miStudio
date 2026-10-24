"""The Settings PIN must not be reachable through the settings CRUD it gates.

WHY THIS EXISTS

MIS-E2E-055, confirmed live as MIS-E2E-165. The PIN was stored as an ordinary
row under `settings_pin_hash` with `is_sensitive=False`, so the generic routes
beside it defeated the gate three separate ways:

  READ    `GET /settings` returned the PBKDF2 salt+hash unmasked — verified
          against the running deployment: 150 characters, is_sensitive=False,
          unauthenticated. A 4-digit PIN is 10,000 offline candidates.
  WRITE   `PUT /settings` validated only membership in a two-element URL set,
          so the PIN could be overwritten without knowing the current one,
          bypassing the `current_pin` check in `/pin/set`.
  DELETE  `DELETE /settings/{key}` removed it, after which `/pin/set` sees
          `existing is None` and stops requiring a current PIN at all.

None of this is covered by the accepted no-app-auth posture (MIS-E2E-002): the
PIN's entire threat model is gating the credential panel from someone who
already has network access, which is exactly the population nginx admits.

The guard lives in `AppSettingService`, not on the routes, because this audit
found five instances of a correct guard applied at one call site and not its
siblings. These tests exercise every route that reaches it.

NEGATIVE CONTROL: remove `settings_pin_hash` from `_PROTECTED_KEYS` and every
test in `TestThePinIsNotReachableThroughSettingsCrud` must fail. Verified
2026-08-23.
"""

import pytest
from httpx import AsyncClient


async def _set_pin(client: AsyncClient, pin: str = "1234"):
    r = await client.post("/api/v1/settings/pin/set", json={"pin": pin})
    assert r.status_code == 200, r.text
    return r


class TestThePinIsNotReachableThroughSettingsCrud:
    async def test_the_listing_does_not_include_it(self, client: AsyncClient):
        """The live exposure: GET /settings returned the hash in the clear."""
        await _set_pin(client)
        r = await client.get("/api/v1/settings")
        assert r.status_code == 200
        keys = [row["key"] for row in r.json()]
        assert "settings_pin_hash" not in keys, (
            "the PIN hash appears in the generic settings listing — this is the "
            "exposure confirmed live as MIS-E2E-165"
        )

    async def test_the_hash_never_appears_in_a_listing_response_body(
        self, client: AsyncClient
    ):
        """Stronger than the key check: the VALUE must not appear anywhere.

        A future refactor could rename the key and reintroduce the leak; this
        catches the material itself.
        """
        await _set_pin(client)
        r = await client.get("/api/v1/settings")
        assert "pbkdf2:sha256:" not in r.text, (
            "a PBKDF2 hash is present in the settings listing response"
        )

    async def test_fetching_it_by_key_is_a_404(self, client: AsyncClient):
        """Absent, not masked. The generic API has no business confirming the
        row exists — `/pin/status` answers that question properly."""
        await _set_pin(client)
        r = await client.get("/api/v1/settings/settings_pin_hash")
        assert r.status_code == 404, r.text

    async def test_overwriting_it_is_refused(self, client: AsyncClient):
        """The write bypass: setting the PIN without knowing the current one."""
        await _set_pin(client, "1234")
        r = await client.put(
            "/api/v1/settings",
            json={"key": "settings_pin_hash", "value": "pbkdf2:sha256:600000$x$y",
                  "is_sensitive": False, "category": "system"},
        )
        assert r.status_code == 403, r.text

        # And the original PIN still works — the refusal was real, not cosmetic.
        v = await client.post("/api/v1/settings/pin/verify", json={"pin": "1234"})
        assert v.status_code == 200 and v.json()["valid"] is True

    async def test_the_bulk_route_is_not_a_way_around_it(self, client: AsyncClient):
        """/bulk must not be the loophole.

        MIS-E2E-073 is the same shape: /bulk skipped the URL validation the
        single-key route applied. A guard that only covers the obvious route is
        the pattern this audit kept finding.
        """
        await _set_pin(client, "1234")
        r = await client.put(
            "/api/v1/settings/bulk",
            json={"settings": [{"key": "settings_pin_hash",
                                "value": "pbkdf2:sha256:600000$x$y",
                                "is_sensitive": False, "category": "system"}]},
        )
        assert r.status_code == 403, r.text
        v = await client.post("/api/v1/settings/pin/verify", json={"pin": "1234"})
        assert v.json()["valid"] is True

    async def test_deleting_it_is_refused(self, client: AsyncClient):
        """The delete bypass — with no row, /pin/set stops asking for the
        current PIN, which removes the gate entirely."""
        await _set_pin(client, "1234")
        r = await client.delete("/api/v1/settings/settings_pin_hash")
        assert r.status_code == 403, r.text

        status = await client.get("/api/v1/settings/pin/status")
        assert status.json()["configured"] is True, (
            "the PIN row was removed despite the refusal"
        )


class TestThePinItselfStillWorks:
    """The negative half. A guard that breaks the feature is not a fix."""

    async def test_set_then_verify(self, client: AsyncClient):
        await _set_pin(client, "4321")
        r = await client.post("/api/v1/settings/pin/verify", json={"pin": "4321"})
        assert r.status_code == 200 and r.json()["valid"] is True

    async def test_a_wrong_pin_is_rejected(self, client: AsyncClient):
        await _set_pin(client, "4321")
        r = await client.post("/api/v1/settings/pin/verify", json={"pin": "0000"})
        assert r.json()["valid"] is False

    async def test_changing_it_requires_the_current_one(self, client: AsyncClient):
        await _set_pin(client, "1111")
        bad = await client.post("/api/v1/settings/pin/set",
                                json={"pin": "2222", "current_pin": "9999"})
        assert bad.status_code == 401, bad.text
        good = await client.post("/api/v1/settings/pin/set",
                                 json={"pin": "2222", "current_pin": "1111"})
        assert good.status_code == 200, good.text


class TestRemovingThePin:
    """Settings → API Keys → "Remove" (found by the ENH-003 inventory, 2026-10-20).

    The button called the generic `DELETE /settings/settings_pin_hash`, which
    the guard above refuses with 403 — correctly — so it showed "Failed to
    remove PIN" every time. Removal now goes through `POST /settings/pin/remove`,
    which, like changing the PIN, asks for the current one.

    NEGATIVE CONTROLS (2026-10-20): drop the route and every test here fails
    (405/404); drop the `_verify_pin` check and the wrong-PIN and missing-PIN
    tests fail; drop `_privileged=True` from the delete and the success test
    fails with the ProtectedSettingError the generic route raises.
    """

    async def test_it_removes_the_pin_given_the_current_one(self, client: AsyncClient):
        await _set_pin(client, "1234")
        r = await client.post("/api/v1/settings/pin/remove", json={"current_pin": "1234"})
        assert r.status_code == 200, r.text
        assert r.json() == {"success": True, "removed": True}

        status = await client.get("/api/v1/settings/pin/status")
        assert status.json()["configured"] is False

        # And with no PIN the gate is open again — /verify is trivially valid.
        v = await client.post("/api/v1/settings/pin/verify", json={"pin": "anything"})
        assert v.json()["valid"] is True

    async def test_a_wrong_current_pin_is_refused(self, client: AsyncClient):
        await _set_pin(client, "1234")
        r = await client.post("/api/v1/settings/pin/remove", json={"current_pin": "9999"})
        assert r.status_code == 401, r.text
        status = await client.get("/api/v1/settings/pin/status")
        assert status.json()["configured"] is True

    async def test_a_missing_current_pin_is_refused(self, client: AsyncClient):
        """Removal without the PIN would be the MIS-E2E-055 delete bypass again."""
        await _set_pin(client, "1234")
        for body in ({}, {"current_pin": None}, {"current_pin": ""}):
            r = await client.post("/api/v1/settings/pin/remove", json=body)
            assert r.status_code == 400, (body, r.text)
        status = await client.get("/api/v1/settings/pin/status")
        assert status.json()["configured"] is True

    async def test_the_bypass_waives_the_current_pin(self, client: AsyncClient, monkeypatch):
        """The recovery path: MISTUDIO_BYPASS_PIN=true, PIN forgotten."""
        from src.core.config import settings

        await _set_pin(client, "1234")
        monkeypatch.setattr(settings, "bypass_settings_pin", True)
        r = await client.post("/api/v1/settings/pin/remove", json={})
        assert r.status_code == 200, r.text
        status = await client.get("/api/v1/settings/pin/status")
        assert status.json()["configured"] is False

    async def test_with_no_pin_it_is_a_no_op(self, client: AsyncClient):
        r = await client.post("/api/v1/settings/pin/remove", json={})
        assert r.status_code == 200, r.text
        assert r.json() == {"success": True, "removed": False}

    async def test_the_generic_delete_is_still_refused(self, client: AsyncClient):
        """The new route is the ONLY way in; the guard above must not loosen."""
        await _set_pin(client, "1234")
        r = await client.delete("/api/v1/settings/settings_pin_hash")
        assert r.status_code == 403, r.text


class TestThePinRoutesCommitBeforeResponding:
    """`/pin/set` and `/pin/remove` commit before returning, so the frontend's
    immediate `GET /pin/status` sees the change. Production's `get_db` commits
    only after the response is sent. The test client shares one session across
    requests, so a missing commit is invisible to every other test here: this
    counts the commits instead.

    NEGATIVE CONTROL (2026-10-20): deleting the `await db.commit()` in
    `remove_pin` survived the whole suite until this test; with it, it fails.
    """

    @staticmethod
    def _count_commits(async_session, monkeypatch) -> list[int]:
        calls = [0]
        real = async_session.commit

        async def counting_commit():
            calls[0] += 1
            await real()

        monkeypatch.setattr(async_session, "commit", counting_commit)
        return calls

    async def test_set_commits(self, client: AsyncClient, async_session, monkeypatch):
        calls = self._count_commits(async_session, monkeypatch)
        await _set_pin(client, "1234")
        assert calls[0] == 1

    async def test_remove_commits(self, client: AsyncClient, async_session, monkeypatch):
        await _set_pin(client, "1234")
        calls = self._count_commits(async_session, monkeypatch)
        r = await client.post("/api/v1/settings/pin/remove", json={"current_pin": "1234"})
        assert r.status_code == 200, r.text
        assert calls[0] == 1
