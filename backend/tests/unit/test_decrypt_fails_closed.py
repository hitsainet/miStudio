"""Decryption must distinguish "never encrypted" from "failed to authenticate".

WHY THIS EXISTS

`decrypt_value` caught every exception — including `InvalidTag` — logged a
warning, and returned the stored bytes as if they were plaintext. Three findings
root in that one swallow:

  MIS-E2E-004  AES-GCM's integrity half is discarded. An authenticated-encryption
               primitive whose authentication failure is ignored is not providing
               integrity.
  MIS-E2E-056  After a key change the raw base64 ciphertext was handed to
               `OpenAILabelingService`, which sent it to api.openai.com in an
               `Authorization: Bearer` header. Encrypted credential material left
               the network boundary because the failure was designed to be silent.
  MIS-E2E-041  Legacy plaintext rows work perfectly and forever, because nothing
               can distinguish them from decrypted ones — so no backfill is ever
               prompted.

The split now: not-an-envelope is legacy data, returned as-is AND COUNTED;
InvalidTag raises.

NEGATIVE CONTROL: change `except InvalidTag` back to swallowing, and
`test_a_tampered_envelope_raises` must fail. Verified 2026-08-23.
"""

import base64
import os

import pytest

os.environ.setdefault("SETTINGS_ENCRYPTION_KEY", "audit-test-key-" + "x" * 32)

from src.core.encryption import (  # noqa: E402
    DecryptionError,
    decrypt_value,
    encrypt_value,
    legacy_plaintext_reads,
    mask_value,
)


def _tamper(ciphertext_b64: str) -> str:
    """Flip one bit inside a valid envelope: still well-formed, tag now wrong."""
    raw = bytearray(base64.b64decode(ciphertext_b64))
    raw[-1] ^= 0x01
    return base64.b64encode(bytes(raw)).decode()


class TestAuthenticationFailureIsNotSilent:
    def test_a_round_trip_still_works(self):
        """The fix must not break the normal path."""
        assert decrypt_value(encrypt_value("sk-proj-value")) == "sk-proj-value"

    def test_a_tampered_envelope_raises(self):
        """The core of MIS-E2E-056."""
        tampered = _tamper(encrypt_value("sk-proj-REAL"))
        with pytest.raises(DecryptionError):
            decrypt_value(tampered, setting_key="openai_api_key")

    def test_the_ciphertext_is_never_returned_on_auth_failure(self):
        """Explicit, because returning it is the actual harm.

        A test that only asserts "raises" would pass against a version that
        raised *after* leaking the value somewhere. This asserts the value does
        not come back at all.
        """
        tampered = _tamper(encrypt_value("sk-proj-REAL"))
        try:
            returned = decrypt_value(tampered, setting_key="openai_api_key")
        except DecryptionError:
            return  # correct
        pytest.fail(
            f"ciphertext was returned instead of raising: {returned[:24]}… — "
            "this is what reached api.openai.com in an Authorization header"
        )

    def test_a_wrong_key_raises_rather_than_returning_ciphertext(self):
        """The realistic trigger: SETTINGS_ENCRYPTION_KEY rotated or regenerated.

        Simulated by corrupting the envelope, which is indistinguishable from a
        key change at the AES-GCM layer — both are a tag that does not verify.
        """
        with pytest.raises(DecryptionError):
            decrypt_value(_tamper(encrypt_value("hf_token_value")),
                          setting_key="hf_token")


class TestLegacyPlaintextIsStillTolerated:
    def test_a_plaintext_row_is_returned_as_is(self):
        """The case the fallback was actually built for — must keep working."""
        assert decrypt_value("plain-legacy-key", setting_key="hf_token") == "plain-legacy-key"

    def test_a_plaintext_row_is_counted(self):
        """MIS-E2E-041: counted, not merely logged.

        The exposure persisted because the path was silent. A counter makes
        "how many credentials are still unencrypted?" answerable.
        """
        before = len(legacy_plaintext_reads())
        decrypt_value("another-plaintext", setting_key="openai_api_key")
        after = legacy_plaintext_reads()
        assert len(after) == before + 1
        assert "openai_api_key" in after

    def test_a_short_value_is_treated_as_plaintext_not_as_tampering(self):
        """Too short to be an envelope is legacy data, not an attack."""
        assert decrypt_value("abc", setting_key="hf_token") == "abc"


class TestMaskingStillProtectsShortSecrets:
    def test_a_short_value_is_not_revealed_whole(self):
        """MIS-E2E-061: mask_value emitted every character of a 4-7 char value."""
        masked = mask_value("abcd")
        assert masked != "abcd", "a 4-character secret was returned unmasked"
        assert "abcd" not in masked, f"the whole value survives in {masked!r}"
