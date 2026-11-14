"""Path containment: the one check every filesystem sink fed by a request goes through.

A request reaches the disk here through an id, a slug or a path, and three things have
let one escape the directory it was meant for:

* ``..`` — ``Path.__truediv__`` does not normalise it, and ``open``, ``rmtree`` and
  ``rename`` all resolve it through the OS (MIS-E2E-070).
* an absolute segment — ``root / "/etc"`` IS ``/etc``.
* a symlink inside the root that points outside it — a string-only normaliser
  (``Settings.resolve_user_path``) cannot see it, and every sink follows it.

So containment is decided on the REAL path: ``os.path.realpath`` of the candidate must be
the root's real path or lie beneath it. The returned path is that real path, so the sink
operates on exactly what was checked, not on a string a later symlink swap can redirect.

``PathEscape`` subclasses ``ValueError`` because every caller already maps ValueError to a
refusal; nothing has to learn a new exception to stay safe.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, "os.PathLike[str]"]

__all__ = ["PathEscape", "contained", "contained_in_any", "child_of", "safe_segment"]


class PathEscape(ValueError):
    """A path resolved outside the directory it was required to stay in."""


def contained(root: PathLike, candidate: PathLike) -> Path:
    """The real path of ``candidate``, refused unless it is ``root`` or beneath it.

    Both sides are resolved with ``os.path.realpath``, so ``..``, absolute
    components and symlinks are all followed BEFORE the comparison. The prefix test
    carries the separator, so ``/data-evil`` is not inside ``/data``.
    """
    root_real = os.path.realpath(os.fspath(root))
    real = os.path.realpath(os.fspath(candidate))
    if real == root_real:
        return Path(root_real)
    if not real.startswith(root_real.rstrip(os.sep) + os.sep):
        raise PathEscape(f"{os.fspath(candidate)!r} resolves outside {root_real!r}")
    return Path(real)


def contained_in_any(roots: Iterable[PathLike], candidate: PathLike) -> Path:
    """``contained`` against the first of ``roots`` that holds ``candidate``; else refused."""
    roots = list(roots)
    for root in roots:
        try:
            return contained(root, candidate)
        except PathEscape:
            continue
    raise PathEscape(
        f"{os.fspath(candidate)!r} resolves outside every allowed root "
        f"({', '.join(repr(os.fspath(r)) for r in roots)})"
    )


def safe_segment(segment: str) -> str:
    """``segment`` if it names exactly one directory entry; refused otherwise.

    An id or slug that reaches a join must be ONE component: not empty, not ``.`` or
    ``..``, and free of separators and NUL. This is the shape check; ``child_of`` adds
    the real-path check on top, because a valid name can still be a symlink.
    """
    if not isinstance(segment, str) or segment in ("", ".", ".."):
        raise PathEscape(f"{segment!r} is not a single path segment")
    if "/" in segment or "\\" in segment or "\x00" in segment or (
        os.altsep and os.altsep in segment
    ):
        raise PathEscape(f"{segment!r} is not a single path segment")
    return segment


def child_of(root: PathLike, *segments: str) -> Path:
    """``root/segment/...`` with every segment shape-checked and the result contained."""
    for segment in segments:
        safe_segment(segment)
    return contained(root, os.path.join(os.fspath(root), *segments))
