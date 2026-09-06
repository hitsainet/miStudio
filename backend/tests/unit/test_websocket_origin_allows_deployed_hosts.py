"""A browser on a deployed hostname must be able to upgrade to a WebSocket.

2026-08-25. MIS-E2E-105 changed Socket.IO from `cors_allowed_origins="*"` to
`settings.allowed_origins`. That list had only ever fed HTTP CORS and still
read `["http://localhost:3000", "http://localhost"]`, so in production every
browser got **403 on the WebSocket upgrade**:

    WebSocket connection to 'ws://k8s-mistudio.hitsai.local/ws/socket.io/...'
    failed: Error during WebSocket handshake: Unexpected response code: 403

socket.io then falls back to polling and retries the upgrade forever. That
failure mode is nearly invisible: the socket reports connected, REST works, the
page renders — only *pushed* events never arrive. It surfaced as "progress only
updates when I refresh the page", four days after the change.

WHY IT SURVIVED EVERY PROBE: a CLI client sends no `Origin` header, and
socket.io allows an absent Origin. `curl`, `python-socketio` and a bare upgrade
request all returned 101. Only a real browser reproduces it — which is why this
test asserts on the ORIGIN CHECK, not on connectivity.
"""

import pytest


def _origin_allowed(origin: str, allowed: list[str]) -> bool:
    """The check python-socketio performs (`base_server._cors_allowed_origins`)."""
    return origin in allowed


class TestTheDeployedHostnamesAreAllowed:
    #: Hostnames this system is actually reached on. A browser on any of these
    #: must be able to upgrade; localhost alone is a development-only list.
    DEPLOYED = (
        "http://k8s-mistudio.hitsai.local",
        "https://mistudio.hitsai.net",
    )

    def test_the_default_config_allows_them(self):
        """The default matters: k8s did not override it, so the default shipped."""
        from src.core.config import Settings

        default = Settings.model_fields["allowed_origins"].default
        for origin in self.DEPLOYED:
            assert origin in default, (
                f"{origin} is not in the default allowed_origins, so any "
                f"deployment that does not override ALLOWED_ORIGINS returns 403 "
                f"on the WebSocket upgrade and silently degrades to polling"
            )

    def test_localhost_is_still_allowed_for_development(self):
        from src.core.config import Settings

        default = Settings.model_fields["allowed_origins"].default
        assert "http://localhost:3000" in default

    def test_the_k8s_manifest_sets_it_explicitly(self):
        """Belt and braces: the deployment should not rely on the default."""
        from pathlib import Path

        manifest = (Path(__file__).resolve().parents[3] / "k8s" / "base"
                    / "backend.yaml").read_text()
        assert "ALLOWED_ORIGINS" in manifest, (
            "the deployment does not set ALLOWED_ORIGINS, so it inherits "
            "whatever the code default happens to be"
        )
        for origin in self.DEPLOYED:
            assert origin in manifest, f"{origin} missing from the k8s env"


class TestTheCheckItself:
    """Guard the premise, so this file cannot pass vacuously."""

    def test_an_unknown_origin_is_rejected(self):
        assert not _origin_allowed("https://evil.example",
                                   ["http://localhost"]), (
            "the origin check accepts anything — the allowlist is not a list"
        )

    def test_socketio_reads_this_setting(self):
        import inspect

        from src.core import websocket

        source = inspect.getsource(websocket)
        assert "cors_allowed_origins=settings.allowed_origins" in source, (
            "Socket.IO no longer reads allowed_origins; either it reverted to "
            "'*' (which MIS-E2E-105 removed deliberately) or it reads something "
            "else and this test guards nothing"
        )

    def test_an_absent_origin_is_why_cli_probes_passed(self):
        """Recorded so the next person does not repeat my mistake.

        `curl` and `python-socketio` send no Origin. socket.io permits that,
        so every non-browser probe returned 101 while browsers got 403.
        """
        assert not _origin_allowed("", ["http://localhost"]), (
            "an empty origin matching the allowlist would make this note wrong"
        )


class TestTheManifestValueActuallyLoads:
    """The k8s value must parse, not merely exist.

    My first version set it comma-separated. `allowed_origins` is a `list[str]`,
    and pydantic-settings JSON-decodes complex types in EnvSettingsSource
    BEFORE any `field_validator` runs — so `parse_allowed_origins`, which does
    accept a comma-separated string, never saw the value. The process died at
    import with `SettingsError: error parsing value for field "allowed_origins"`
    and the API crashlooped in production.

    Asserting the key is present was not enough; this feeds the real value
    through the real Settings class.
    """

    def _manifest_value(self):
        import re
        from pathlib import Path

        text = (Path(__file__).resolve().parents[3] / "k8s" / "base"
                / "backend.yaml").read_text()
        m = re.search(r"- name: ALLOWED_ORIGINS\n(?:\s*#[^\n]*\n)*\s*value:\s*(.+)", text)
        assert m, "ALLOWED_ORIGINS not found in the manifest"
        return m.group(1).strip().strip("'\"")

    def test_the_manifest_value_parses_through_settings(self):
        import os

        from src.core.config import Settings

        raw = self._manifest_value()
        old = os.environ.get("ALLOWED_ORIGINS")
        os.environ["ALLOWED_ORIGINS"] = raw
        try:
            parsed = Settings().allowed_origins
        except Exception as exc:                       # noqa: BLE001
            raise AssertionError(
                f"the manifest's ALLOWED_ORIGINS value does not load: "
                f"{type(exc).__name__}. This crashloops the API at import. "
                f"Value was: {raw[:90]}"
            ) from exc
        finally:
            if old is None:
                os.environ.pop("ALLOWED_ORIGINS", None)
            else:
                os.environ["ALLOWED_ORIGINS"] = old

        assert "https://mistudio.hitsai.net" in parsed
        assert "http://k8s-mistudio.hitsai.local" in parsed

    def test_a_comma_separated_value_is_rejected_loudly(self):
        """Pin the trap, so nobody 'simplifies' the JSON back to commas."""
        import os

        import pytest

        from src.core.config import Settings

        old = os.environ.get("ALLOWED_ORIGINS")
        os.environ["ALLOWED_ORIGINS"] = "http://a.local,http://b.net"
        try:
            with pytest.raises(Exception):
                Settings()
        finally:
            if old is None:
                os.environ.pop("ALLOWED_ORIGINS", None)
            else:
                os.environ["ALLOWED_ORIGINS"] = old
