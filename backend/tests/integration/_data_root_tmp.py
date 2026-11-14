"""A temp directory that IS the trusted data root for the duration of a test.

MIS-E2E-071 gave every deletion sink a containment guard: a stored path is only
deletable if it resolves at least two levels below `data_dir`, `run_dir` or
`hf_cache_dir`. That is the whole point of the fix — the database is not a trust
boundary while an unauthenticated update can write a path column.

The cleanup integration tests built their fixtures in a bare
`tempfile.TemporaryDirectory()`, i.e. `/tmp/...`, which the guard now correctly
refuses. Two ways to keep them meaningful, and only one is honest:

  * relax the guard so `/tmp` passes — that deletes the fix, and the test then
    certifies the vulnerable behaviour; or
  * move the trusted root onto the temp directory, so the fixture sits where a
    real model or dataset sits relative to the root.

This is the second. `settings.data_dir` is repointed for the life of the
context and restored afterwards, so the test exercises the production path
including the containment check, rather than a path production never takes.
The real `data/` tree is owned by the container uid and is not writable from a
test run, so writing into it was never an option either.
"""

import contextlib
import tempfile
from pathlib import Path

from src.core.config import settings


@contextlib.contextmanager
def data_root_tmpdir(subdir: str = "test_tmp"):
    """Yield `<tmp>/<subdir>` with `settings.data_dir` pointed at `<tmp>`.

    Depth matters: a fixture created one level inside the yielded directory sits
    two levels below the root and clears the guard's `min_depth=2`. Anything
    shallower is refused, which is correct — a top-level category directory is
    never a legitimate deletion target.
    """
    original = settings.data_dir
    with tempfile.TemporaryDirectory() as root:
        target = Path(root) / subdir
        target.mkdir(parents=True, exist_ok=True)
        try:
            settings.data_dir = Path(root)
            yield str(target)
        finally:
            settings.data_dir = original
