"""A downloaded dataset's identity is (repo, config, split) — not the repo (032 FR-2).

WHY THIS EXISTS. Until FR-2 the repo id alone identified a download: the raw
directory was `<datasets_dir>/<org>_<name>`, and `POST /datasets/download` returned
409 for any second request naming the same repo. That is wrong for any repo with
several configs, and it blocked feature 032 outright — the acceptance dataset
`Arrrlex/models-under-pressure` ships a `training` config to fit a probe on and six
`*_balanced` configs to evaluate it on. Under the old rule the first download won
and the other seven were refused as duplicates; worse, had the guard not refused
them, they would all have written to ONE directory and silently overwritten each
other. This repo has already paid for exactly that shape once: a 512 and a 2048
tokenization of one dataset overwrote the same directory because the path omitted
`max_length` while the row recorded it.

THE DECISION IS EXTRACTED INTO PURE FUNCTIONS ON PURPOSE. A path built inline in
the download task can only be checked by reading it, and a guard that compares
"the same fields" in the endpoint can drift from the path the worker writes. These
functions are unit-tested, and `test_dataset_multi_config.py` asserts the CALL by
walking the AST of both sites — the house answer to a guard satisfied by the wrong
occurrence.

BACKWARD COMPATIBILITY IS BY CONSTRUCTION, NOT BY A FLAG. With `config` and `split`
both None the name is byte-identical to the old one, so every row already on disk
keeps its path and nothing needs migrating.
"""
from __future__ import annotations

import re
from typing import Mapping, Optional, Tuple

#: Anything outside this becomes `_`. Config and split names come from a remote
#: repo, so they are sanitised before they reach a filesystem path — a config
#: named `../../etc` must not escape `datasets_dir`. The download path is also
#: passed through `settings.resolve_deletable_path` before any delete, so this is
#: defence in depth rather than the only guard.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")

#: Cap each component so a pathological config name cannot exceed the filesystem's
#: 255-byte limit for a single directory entry. Truncation could collide, so the
#: caller's 409 is what prevents two views sharing a directory — see `view_key`,
#: which compares the ORIGINAL values and never the truncated ones.
_MAX_COMPONENT = 64


def _safe(part: str) -> str:
    return _UNSAFE.sub("_", part)[:_MAX_COMPONENT]


def raw_dataset_dirname(
    repo_id: str, config: Optional[str] = None, split: Optional[str] = None
) -> str:
    """The directory name for one downloaded VIEW of a repo.

    `org/name` → `org_name`, then `__<config>` and `__<split>` when set. The double
    underscore separates components because a single one already appears inside
    `org_name`, so `a_b__train` is unambiguous where `a_b_train` is not.

    With config and split both None this is the pre-FR-2 name exactly.
    """
    name = _safe(repo_id.replace("/", "_"))
    if config:
        name = f"{name}__{_safe(config)}"
    if split:
        name = f"{name}__{_safe(split)}"
    return name


def hf_cache_dirname(repo_id: str) -> str:
    """HuggingFace's own arrow tree for a repo — `org___name`, SHARED BY EVERY CONFIG.

    ⚠ THIS IS WHY THE CANCEL PATH CHANGED. `download_dataset_task` deletes this
    directory when a download is cancelled, and its comment justified that with
    "one Dataset row per repo_id is enforced at `datasets.py`'s 409 — which is what
    makes deleting them safe". FR-2 REMOVES that invariant. Cancelling a download of
    config B would then delete the arrow tree holding config A's already-downloaded
    data, so the cancel path must now ask whether any other row shares the repo.
    """
    return repo_id.replace("/", "___")


def display_name_for(repo_id: str, config: Optional[str] = None) -> str:
    """The row's human name. Suffixed with the config so a list of seven views of
    one repo is readable; the split is left out because it is shown separately and
    the name is not an identifier."""
    base = repo_id.split("/")[-1]
    return f"{base} ({config})" if config else base


def view_key(
    repo_id: str, config: Optional[str] = None, split: Optional[str] = None
) -> Tuple[str, Optional[str], Optional[str]]:
    """The identity a duplicate check compares.

    Uses the ORIGINAL strings, never the sanitised path components: two configs
    whose sanitised names collide after truncation are still two different views,
    and treating them as one is how a download silently overwrites another.

    Empty strings normalise to None so `config=""` and `config=None` are one view —
    a form that submits a blank field must not create a second row.
    """
    return (repo_id, config or None, split or None)


def view_key_of_metadata(
    repo_id: str, metadata: Optional[Mapping[str, object]]
) -> Tuple[str, Optional[str], Optional[str]]:
    """The view key of an EXISTING row, read from its `extra_metadata`.

    Rows created before FR-2 carry `{"split": None, "config": None}` or no keys at
    all, both of which yield `(repo, None, None)` — so an old row still collides
    with a new plain download of the same repo, which is the behaviour that was
    there before.
    """
    meta = metadata or {}
    return view_key(repo_id, meta.get("config"), meta.get("split"))  # type: ignore[arg-type]
