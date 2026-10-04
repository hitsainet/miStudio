"""The operator's stored API key must not reach a host the request chose.

WHY THIS EXISTS

MIS-E2E-069 — `POST /labeling/models/openai` took a caller-chosen
`endpoint_url`, and when the body omitted `api_key` it read the operator's
`openai_api_key` from the database, DECRYPTED it, and attached it as
`Authorization: Bearer` to that host. The absence of a credential in the request
is what triggered one being spent. A single unauthenticated POST exfiltrated it,
defeating the AES-256-GCM at rest, the masking on every read and the Settings
PIN in one call. `validate_llm_endpoint_url` existed and had two call sites,
neither of them this one.

MIS-E2E-072 — `_save_request_for_testing` writes debug artifacts. The cURL
branch was hardened with an explicit comment ("NEVER write the real bearer token
to disk"); the Postman branch sixty lines below in the same function was not,
and `export_format` defaults to `"both"`, so the DEFAULT path wrote the
operator's key to disk once per feature labelled.

NEGATIVE CONTROLS, both verified 2026-08-23:
  * make `_host_may_receive_stored_key` return True unconditionally →
    `test_an_arbitrary_host_gets_no_stored_key` fails
  * restore `f"Bearer {self.api_key}"` in either Postman writer →
    `test_no_postman_writer_embeds_the_real_key` fails
"""

import re
from pathlib import Path

import pytest

from src.api.v1.endpoints.labeling import _host_may_receive_stored_key


class _FakeSettings:
    """Stand-in for the app settings row lookup."""

    def __init__(self, configured=None):
        self._configured = configured

    async def get_decrypted_value(self, _db, key):
        return self._configured if key == "openai_compatible_endpoint" else None


@pytest.fixture
def patched_settings(monkeypatch):
    def _apply(configured=None):
        import src.api.v1.endpoints.labeling as mod
        monkeypatch.setattr(
            mod.AppSettingService, "get_decrypted_value",
            _FakeSettings(configured).get_decrypted_value,
        )
        # `settings` has no `openai_compatible_endpoint` attribute, and the
        # helper reads it with getattr(..., None) — so there is nothing to
        # patch. Asserted rather than assumed, because if the field is ever
        # added, these tests must start controlling it.
        assert not hasattr(mod.settings, "openai_compatible_endpoint"), (
            "settings now defines openai_compatible_endpoint — this fixture "
            "must patch it, or the allow-list tests stop being hermetic"
        )
    return _apply


class TestTheStoredKeyIsHostGated:
    async def test_an_arbitrary_host_gets_no_stored_key(self, patched_settings):
        """The exfiltration path itself."""
        patched_settings(configured=None)
        assert await _host_may_receive_stored_key(
            None, "https://collector.attacker.tld") is False

    async def test_openai_itself_is_allowed(self, patched_settings):
        """The negative half — a gate that blocks everything is also broken."""
        patched_settings(configured=None)
        assert await _host_may_receive_stored_key(
            None, "https://api.openai.com/v1") is True

    async def test_the_configured_endpoint_is_allowed(self, patched_settings):
        """The operator designated this host by saving it in Settings."""
        patched_settings(configured="http://ollama.internal:11434/v1")
        assert await _host_may_receive_stored_key(
            None, "http://ollama.internal:11434/v1") is True

    async def test_a_lookalike_suffix_host_is_refused(self, patched_settings):
        """`api.openai.com.evil.tld` must not pass.

        This is why the check compares the parsed HOST rather than a prefix or
        a substring of the URL.
        """
        patched_settings(configured=None)
        assert await _host_may_receive_stored_key(
            None, "https://api.openai.com.evil.tld/v1") is False

    async def test_a_path_trick_does_not_admit_a_foreign_host(self, patched_settings):
        """The allowed name appearing in the PATH must not matter."""
        patched_settings(configured=None)
        assert await _host_may_receive_stored_key(
            None, "https://evil.tld/api.openai.com/v1") is False

    async def test_a_garbage_url_is_refused(self, patched_settings):
        """Fails closed on anything unparseable."""
        patched_settings(configured=None)
        assert await _host_may_receive_stored_key(None, "not a url") is False
        assert await _host_may_receive_stored_key(None, "") is False


class TestTheEndpointValidatesBeforeSpendingAnything:
    def test_the_handler_calls_validate_llm_endpoint_url(self):
        """The validator must be ON this path, not merely present in the tree.

        Fails closed: `index()` raises if the handler is renamed or moved, so a
        refactor turns this red rather than silently disarming it — four
        source-scrape guards in this audit failed open.
        """
        src = Path("src/api/v1/endpoints/labeling.py").read_text()
        i = src.index("async def fetch_openai_models(")
        body = src[i:i + 4000]
        assert "validate_llm_endpoint_url(request.endpoint_url)" in body, (
            "the only path that attaches a stored credential does not validate "
            "the URL it sends it to"
        )


class TestNoDebugArtifactEmbedsTheKey:
    def test_no_postman_writer_embeds_the_real_key(self):
        """Both writers, not just the one that was reported.

        The cURL branch was fixed and the Postman branch sixty lines away was
        not — the 'fixed one representative' pattern this audit found five
        times. Asserting over ALL matches is what makes that recurrence
        visible.
        """
        src = Path("src/services/openai_labeling_service.py").read_text()
        offenders = re.findall(r'"Bearer \{self\.api_key\}"', src)
        assert not offenders, (
            f"{len(offenders)} artifact writer(s) still embed the real bearer "
            f"token; use a placeholder as the curl branch does"
        )

    def test_the_placeholder_is_actually_used(self):
        """Guards the guard: the assertion above passes if the header is simply
        deleted, which would be a different bug."""
        src = Path("src/services/openai_labeling_service.py").read_text()
        assert src.count('"Bearer {{OPENAI_API_KEY}}"') == 2, (
            "expected both Postman writers to emit the placeholder header"
        )
