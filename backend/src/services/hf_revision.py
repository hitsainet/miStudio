"""Resolve a Hugging Face dataset ref to the commit it names (Feature 035, XR-3; FTDD D41).

WHY THIS EXISTS. Until 035 a dataset download took no revision: `load_dataset` fetched whatever the
default branch held on the day, and nothing recorded which commit that was. Feature 034 then let
miDataworks hand miStudio a dataset "at commit X" — and the commit miStudio recorded for that handoff
was a claim, because the download could not have been asked for X (034 FTDD §12).

So a download now names an optional ref (commit, branch or tag), and it is resolved HERE, in the
request, before any row or task exists (D9). The identity, the directory and the download all use
the resolved 40-character commit — never the ref, because a branch moves and a tag can be re-pointed,
and an identity keyed on either would call two different snapshots the same dataset.

THE ERROR MAPPING IS THE CONTRACT (D31). Every Hub failure creates nothing and maps to one of three
answers a caller can act on:

    not_found    → 422  the repo or the ref does not exist (or is private and no usable token)
    forbidden    → 403  miStudio's stored token cannot read the repo (gated, or refused)
    unreachable  → 502  the Hub did not answer usefully (timeout, connection, 5xx, a non-commit sha)

⚠ ORDER MATTERS. In huggingface-hub 1.28 `GatedRepoError` SUBCLASSES `RepositoryNotFoundError`, so a
mapping that tests "not found" first reports a gated dataset as nonexistent — telling the operator to
fix a repo id that is correct, when the fix is to accept the licence.

`HfApi` is read from this module's namespace at call time, so a test fakes it at
`src.services.hf_revision.HfApi` and asserts the fake was called; nothing else in the process is
touched.
"""
from __future__ import annotations

import re
from typing import Optional

from huggingface_hub import HfApi  # noqa: F401 - patched by tests at this module attribute
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

# THE HUB'S OWN HTTP MODULE, not `import httpx`. huggingface-hub 2.0 moved its client to httpx2,
# whose exception classes are DISTINCT from httpx's: catching httpx's TimeoutException catches
# nothing a 2.x Hub call raises, so a Hub timeout escaped as a 500 instead of a 502 "unreachable".
# `huggingface_hub.utils.httpx` is httpx on hub 1.30+ and httpx2 on 2.x: always the one the Hub uses.
from huggingface_hub.utils import httpx as hub_httpx  # type: ignore[attr-defined]  # lazy module export

#: A resolved commit. Lowercase hex only: `datasets.revision` carries a CHECK constraint with the same
#: pattern, and the directory name embeds it, so nothing else may ever reach either.
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

#: D31: a Hub call that has not answered in 30 s is reported as unreachable rather than left to hold
#: the request open.
HUB_TIMEOUT_S = 30.0

_STATUS = {"not_found": 422, "forbidden": 403, "unreachable": 502}


class HubResolutionError(Exception):
    """A ref that could not be resolved. `kind` ∈ {not_found, forbidden, unreachable}."""

    def __init__(self, kind: str, message: str):
        if kind not in _STATUS:
            raise ValueError(f"unknown resolution failure kind {kind!r}")
        super().__init__(message)
        self.kind = kind
        self.message = message

    @property
    def status_code(self) -> int:
        return _STATUS[self.kind]


def is_commit(value: Optional[str]) -> bool:
    return bool(value) and bool(COMMIT_RE.match(value))  # type: ignore[arg-type]


def _status_of(exc: HfHubHTTPError) -> Optional[int]:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def resolve_dataset_commit(repo_id: str, ref: Optional[str], token: Optional[str]) -> str:
    """The 40-character commit `ref` names in dataset repo `repo_id` (None = the default branch).

    Blocking: call it through `asyncio.to_thread` from a request. The token is passed to the Hub and
    never placed in a message.
    """
    wanted = ref or None
    shown = wanted or "the default branch"
    try:
        info = HfApi(token=token).dataset_info(repo_id, revision=wanted, timeout=HUB_TIMEOUT_S)
    except GatedRepoError as exc:  # BEFORE RepositoryNotFoundError: it is a subclass of it
        raise HubResolutionError(
            "forbidden",
            f"miStudio's stored Hugging Face token cannot read {repo_id} (gated: accept its "
            f"licence on the Hub with the account whose token miStudio holds)",
        ) from exc
    except RevisionNotFoundError as exc:
        raise HubResolutionError(
            "not_found", f"revision {shown!r} was not found in dataset {repo_id}"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise HubResolutionError(
            "not_found",
            f"dataset {repo_id} was not found on the Hugging Face Hub (or it is private and "
            f"miStudio holds no token that can read it); revision {shown!r} was not resolved",
        ) from exc
    except HFValidationError as exc:
        raise HubResolutionError(
            "not_found", f"{repo_id!r} at {shown!r} is not a valid Hub dataset reference ({exc})"
        ) from exc
    except HfHubHTTPError as exc:
        status = _status_of(exc)
        if status in (401, 403):
            raise HubResolutionError(
                "forbidden", f"miStudio's stored Hugging Face token cannot read {repo_id}"
            ) from exc
        if status is not None and 400 <= status < 500:
            raise HubResolutionError(
                "not_found",
                f"the Hub refused {repo_id} at {shown!r} (HTTP {status})",
            ) from exc
        raise HubResolutionError(
            "unreachable",
            f"the Hugging Face Hub did not answer (HTTP {status}) while resolving {repo_id} "
            f"at {shown!r}",
        ) from exc
    except hub_httpx.TimeoutException as exc:
        raise HubResolutionError(
            "unreachable",
            f"the Hugging Face Hub did not answer (timed out after {HUB_TIMEOUT_S:g} s) while "
            f"resolving {repo_id} at {shown!r}",
        ) from exc
    except (hub_httpx.TransportError, OSError) as exc:
        # OSError covers ConnectionError and OfflineModeIsEnabled. HfHubHTTPError is also an
        # OSError, which is why it is handled above this line.
        raise HubResolutionError(
            "unreachable",
            f"the Hugging Face Hub did not answer ({type(exc).__name__}) while resolving "
            f"{repo_id} at {shown!r}",
        ) from exc

    sha = getattr(info, "sha", None)
    if not is_commit(sha):
        raise HubResolutionError(
            "unreachable",
            f"the Hugging Face Hub answered for {repo_id} at {shown!r} without a commit "
            f"(got {sha!r}); nothing was created",
        )
    return sha  # type: ignore[return-value]


def resolve_for_request(repo_id: str, ref: Optional[str], access_token: Optional[str]) -> str:
    """`resolve_dataset_commit` with the token resolved the way the download task resolves it.

    `resolve_hf_token` reads the encrypted setting through a SYNC session, so it runs in the same
    worker thread as the Hub call rather than on the event loop (FTID TI1).
    """
    from .huggingface_sae_service import resolve_hf_token

    return resolve_dataset_commit(repo_id, ref, resolve_hf_token(access_token))
