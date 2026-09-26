"""The SAE upload must resolve its token the way every other HuggingFace path here does.

⚠ WHY THIS FILE EXISTS. `HuggingFaceSAEService.upload_sae` was the one path in
`huggingface_sae_service.py` that did **not** call `resolve_hf_token`: it took a REQUIRED
`access_token` and handed it straight to `HfApi`. The preview and download paths both resolved,
so a token stored in Settings → API Keys worked for reading from HuggingFace and not for writing
to it, and the operator had to paste a write token into every upload request.

The reason it went unnoticed for so long is the more interesting half. The probe-definition
publish endpoint's docstring asserted that `resolve_hf_token` was "the same resolver the SAE
upload uses" — a claim about a NEIGHBOURING path, stated in a comment, tested by nothing. It was
simply untrue. Comments describing code one file away are exactly what this repo's guards keep
catching, so the claim now has a test under it.

The refusal matters as much as the resolution: an anonymous `HfApi` does not fail when it is
constructed. It fails at the commit, after `create_repo` has already run, with a 401 that reads
like a bad credential rather than a missing one.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.services.huggingface_sae_service import HuggingFaceSAEService, resolve_hf_token


class _RecordingApi:
    """An `HfApi` stand-in that records the token it was constructed with."""

    constructed_with: List[Optional[str]] = []

    def __init__(self, token: Optional[str] = None) -> None:
        type(self).constructed_with.append(token)
        self.commits: List[Dict[str, Any]] = []

    def create_repo(self, **kwargs: Any) -> None:
        self.commits.append({"op": "create_repo", **kwargs})

    def upload_folder(self, **kwargs: Any) -> Any:
        self.commits.append({"op": "upload_folder", **kwargs})
        return type("CommitInfo", (), {"oid": "c0ffee1234"})()

    def upload_file(self, **kwargs: Any) -> Any:
        self.commits.append({"op": "upload_file", **kwargs})
        return type("CommitInfo", (), {"oid": "deadbeef99"})()


@pytest.fixture
def recording_api(monkeypatch: pytest.MonkeyPatch):
    _RecordingApi.constructed_with = []
    monkeypatch.setattr("src.services.huggingface_sae_service.HfApi", _RecordingApi)
    return _RecordingApi


@pytest.fixture
def sae_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "sae_local"
    directory.mkdir()
    (directory / "sae_weights.safetensors").write_bytes(b"\x00" * 32)
    return directory


def _upload(**kwargs: Any) -> Dict[str, Any]:
    return asyncio.run(HuggingFaceSAEService.upload_sae(**kwargs))


class TestTheUploadResolvesItsToken:
    def test_an_absent_token_falls_back_to_the_stored_one(
        self, recording_api, sae_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`access_token=None` must reach `HfApi` as the resolved token, not as None.

        This is the whole defect: before the fix, None went to `HfApi` unchanged and the upload
        was anonymous even with a perfectly good token in Settings.
        """
        monkeypatch.setattr(
            "src.services.huggingface_sae_service.resolve_hf_token",
            lambda explicit=None: explicit or "hf_from_settings",
        )
        _upload(
            local_path=sae_dir,
            repo_id="someone/probe-sae",
            filepath="layer_11",
            access_token=None,
            create_repo=True,
        )
        assert recording_api.constructed_with == ["hf_from_settings"]

    def test_an_explicit_token_still_wins(
        self, recording_api, sae_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.services.huggingface_sae_service.resolve_hf_token",
            lambda explicit=None: explicit or "hf_from_settings",
        )
        _upload(
            local_path=sae_dir,
            repo_id="someone/probe-sae",
            filepath="layer_11",
            access_token="hf_from_the_request",
        )
        assert recording_api.constructed_with == ["hf_from_the_request"]

    def test_no_token_anywhere_is_refused_before_the_repo_is_touched(
        self, recording_api, sae_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refusal, not an anonymous upload that 401s after `create_repo`."""
        monkeypatch.setattr(
            "src.services.huggingface_sae_service.resolve_hf_token",
            lambda explicit=None: None,
        )
        with pytest.raises(ValueError) as caught:
            _upload(
                local_path=sae_dir,
                repo_id="someone/probe-sae",
                filepath="layer_11",
                access_token=None,
                create_repo=True,
            )
        message = str(caught.value)
        assert "write permissions" in message
        assert "Settings" in message
        # Nothing was constructed, so nothing was created.
        assert recording_api.constructed_with == []

    def test_the_resolver_is_the_shared_one(self) -> None:
        """The claim the probe-publish docstring makes, asserted rather than written down.

        `resolve_hf_token` must be the object the upload path reaches for — not a second copy that
        can drift from it. This is the guard the false comment needed.
        """
        import src.services.huggingface_sae_service as module

        assert module.resolve_hf_token is resolve_hf_token
        source = module.HuggingFaceSAEService.upload_sae.__doc__ or ""
        assert "resolve_hf_token" in source


class TestTheCommitHashIsReported:
    def test_a_folder_upload_reports_the_commit_it_made(
        self, recording_api, sae_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`commit_hash` was hardcoded None while the CommitInfo was in hand and discarded."""
        monkeypatch.setattr(
            "src.services.huggingface_sae_service.resolve_hf_token",
            lambda explicit=None: "hf_tok",
        )
        result = _upload(
            local_path=sae_dir,
            repo_id="someone/probe-sae",
            filepath="layer_11",
            access_token="hf_tok",
        )
        assert result["commit_hash"] == "c0ffee1234"

    def test_a_single_file_upload_reports_its_own_commit(
        self, recording_api, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.services.huggingface_sae_service.resolve_hf_token",
            lambda explicit=None: "hf_tok",
        )
        one_file = tmp_path / "sae.safetensors"
        one_file.write_bytes(b"\x00" * 16)
        result = _upload(
            local_path=one_file,
            repo_id="someone/probe-sae",
            filepath="layer_11",
            access_token="hf_tok",
        )
        assert result["commit_hash"] == "deadbeef99"

    def test_an_api_without_an_oid_is_tolerated(
        self, sae_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An older hub returning None must not break the upload; the sha is a nicety."""

        class _NoOid(_RecordingApi):
            def upload_folder(self, **kwargs: Any) -> None:  # type: ignore[override]
                return None

        monkeypatch.setattr("src.services.huggingface_sae_service.HfApi", _NoOid)
        monkeypatch.setattr(
            "src.services.huggingface_sae_service.resolve_hf_token",
            lambda explicit=None: "hf_tok",
        )
        result = _upload(
            local_path=sae_dir,
            repo_id="someone/probe-sae",
            filepath="layer_11",
            access_token="hf_tok",
        )
        assert result["commit_hash"] is None


class TestTheRequestSchemaAllowsTheStoredToken:
    def test_access_token_is_optional(self) -> None:
        """A REQUIRED field made the stored token unreachable through the API.

        The service resolving is necessary and not sufficient: while `SAEUploadRequest.access_token`
        was `str = Field(...)`, every request had to carry a token or be rejected at validation,
        so the resolver would never have been consulted.
        """
        from src.schemas.sae import SAEUploadRequest

        request = SAEUploadRequest(
            sae_id="sae_1", repo_id="someone/repo", filepath="layer_11"
        )
        assert request.access_token is None
