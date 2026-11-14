"""A recording stand-in for `huggingface_hub.HfApi`, installed at `src.services.hf_revision.HfApi`.

Feature 035: `POST /datasets/download` resolves its ref through the Hub before anything is created.
A test that posts to that route without this fake reaches the real Hub — which is a network test
that fails offline and, online, pins whatever the real default branch holds today.

The fake RECORDS every call (repo, revision, timeout, token) so a test can assert it was used: a
patch at the wrong attribute would otherwise do nothing silently and the real Hub would answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

COMMIT_A = "a" * 40
COMMIT_B = "b" * 40


@dataclass
class FakeHub:
    #: ref (None = default branch) → commit. A ref not in the map resolves to `default`.
    refs: Dict[Optional[str], str] = field(default_factory=dict)
    default: str = COMMIT_A
    #: When set, `dataset_info` raises what this returns instead of answering.
    raises: Optional[Callable[[], BaseException]] = None
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def factory(self):
        hub = self

        class _Api:
            def __init__(self, token=None, **_):
                self.token = token

            def dataset_info(self, repo_id, *, revision=None, timeout=None, **_):
                hub.calls.append(
                    {"repo_id": repo_id, "revision": revision, "timeout": timeout, "token": self.token}
                )
                if hub.raises is not None:
                    raise hub.raises()
                return SimpleNamespace(sha=hub.refs.get(revision, hub.default))

        return _Api


def install(monkeypatch, hub: Optional[FakeHub] = None) -> FakeHub:
    hub = hub or FakeHub()
    monkeypatch.setattr("src.services.hf_revision.HfApi", hub.factory())
    return hub
