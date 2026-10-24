"""MIS-E2E-071 — a stored path must not become an arbitrary-deletion primitive.

`raw_path`, `file_path`, `quantized_path`, `tokenized_path`, `local_path`,
`output_path` and `training_dir` are all writable through create/update
endpoints that blind-`setattr` onto the ORM row. Every one of them is then read
back on delete and handed to `shutil.rmtree`. `resolve_data_path` — which the
sinks used — is documented for system-constructed paths and returns an existing
absolute path verbatim, so the database was silently acting as a trust boundary
it is not.

These tests pin BOTH halves of the guard:

  * the containment rule itself (`test_guard_refuses_*`), and
  * that every sink actually routes through it (`test_sink_*`).

The second half is the one that matters. This audit found five separate
instances of a correct guard sitting off the path it was written to protect,
and a test of the guard alone passes just as happily in that case.
"""

import inspect
from pathlib import Path

import pytest

from src.core.config import settings


# ── The guard itself ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "bad",
    [
        "",           # normalises to data_dir itself
        "/",
        ".",
        "/data",      # the docker-style root
        "data",
        "datasets",   # a top-level category dir: every dataset at once
        "models",
        "activations",
        "/data/datasets",
        "..",
        "../../etc",
        "/data/../..",
    ],
)
def test_guard_refuses_roots_and_traversal(bad):
    """A trusted root, a category directory, or an escape is never deletable.

    Containment alone would ACCEPT the first eight of these — `resolve_user_path`
    maps "" to data_dir and "datasets" to the directory holding every dataset.
    Depth is what makes this a deletion guard rather than a containment check.
    """
    with pytest.raises(ValueError):
        settings.resolve_deletable_path(bad)


@pytest.mark.parametrize(
    "good",
    [
        "/data/datasets/ds_abc",
        "datasets/ds_abc",
        "/data/models/m1/quantized",
        "activations/extr_20260726_174056_sae_d1a4_002/shard0",
        "trainings/train_969e90af",
    ],
)
def test_guard_accepts_legitimate_targets(good):
    """The guard must not break the paths the workers actually write.

    A refusal-only test would pass against a guard that refuses everything, and
    that guard would silently strand every real deletion as a logged error.
    """
    resolved = settings.resolve_deletable_path(good)
    assert resolved.is_absolute()
    assert str(resolved).startswith(str(Path(settings.data_dir).resolve()))


@pytest.mark.parametrize("bad", ["", "/", ".", None])
def test_guard_refuses_empty_and_root_even_with_depth_disabled(bad):
    """The empty/root refusal must hold independently of the depth check.

    Mutation control C5 originally SURVIVED: deleting this branch changed
    nothing, because `""`, `"/"` and `"."` all normalise to `data_dir` and the
    depth check rejects them anyway. That makes the branch redundant TODAY and
    load-bearing for any caller that relaxes `min_depth` — a containment-only
    caller passing `min_depth=0` would otherwise be handed `data_dir` itself and
    hand it to rmtree.

    So the branch is pinned by the contract it uniquely provides, rather than by
    a test that would pass with it deleted.
    """
    with pytest.raises(ValueError):
        settings.resolve_deletable_path(bad, min_depth=0)


def test_guard_refuses_symlink_escape(tmp_path, monkeypatch):
    """An existing path that resolves outside the roots via a symlink is refused.

    `resolve_user_path` is deliberately string-only — correct, because it must
    not touch the filesystem before containment succeeds — but `rmtree` does
    traverse symlinked components, so the deletion guard re-checks realpath.
    """
    root = tmp_path / "data"
    (root / "datasets").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "datasets" / "evil").symlink_to(outside, target_is_directory=True)

    # run_dir and hf_cache_dir are read-only properties derived from data_dir,
    # so patching data_dir moves all three trusted roots together.
    monkeypatch.setattr(settings, "data_dir", root)

    with pytest.raises(ValueError, match="symlink"):
        settings.resolve_deletable_path("datasets/evil")


# ── The sinks are ON the guard ─────────────────────────────────────────────
#
# Read from the live module source rather than a hand-list of files: a new sink
# added to one of these modules is covered automatically, which is the failure
# mode `REQUIRED_TABLES` and `EXPECTED_CALLS` both had.

_SINK_MODULES = [
    "src.workers.dataset_tasks",
    "src.workers.model_tasks",
    "src.workers.training_tasks",
    "src.services.dataset_service",
    "src.services.sae_manager_service",
    "src.api.v1.endpoints.datasets",
    "src.api.v1.endpoints.models",
    "src.api.v1.endpoints.neuronpedia",
]

# A stored, API-influenceable path column.
_TAINTED = {
    "raw_path", "file_path", "quantized_path", "tokenized_path",
    "local_path", "output_path", "training_dir",
}


def _deleted_names_resolved_unsafely(modname: str) -> list[str]:
    """Names handed to rmtree/unlink that were resolved by `resolve_data_path`.

    An AST walk, not a regex over source. A regex would have to guess at
    formatting and would flag the many *read* sites that legitimately resolve a
    stored path to load a model or a dataset — this asks the narrower question
    the finding is actually about: was a value resolved by the NON-containing
    helper, and is that same value then deleted?
    """
    import ast
    import importlib
    import inspect as _inspect

    tree = ast.parse(_inspect.getsource(importlib.import_module(modname)))

    def calls(node, name):
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == name
            for n in ast.walk(node)
        )

    def mentions_tainted(node):
        return any(
            (isinstance(n, ast.Attribute) and n.attr in _TAINTED)
            or (isinstance(n, ast.Name) and n.id in _TAINTED)
            for n in ast.walk(node)
        )

    def scan_scope(scope) -> set[str]:
        """Unsafely-resolved names that are deleted WITHIN THE SAME function.

        Scoping matters: `raw_path` names a legitimate read in one function and
        a guarded delete in another, and a module-wide walk conflates the two
        into a false positive. Cross-function flow is not modelled — that is a
        deliberate limit, and the reason the guard also lives in the callee.
        """
        unsafe = set()
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and node.value is not None:
                if calls(node.value, "resolve_data_path") and mentions_tainted(node.value):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            unsafe.add(tgt.id)

        deleted = set()
        for node in ast.walk(scope):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "rmtree" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Name):
                    deleted.add(arg.id)
                elif isinstance(arg, ast.Call) and arg.args and isinstance(arg.args[0], ast.Name):
                    deleted.add(arg.args[0].id)   # rmtree(Path(x))
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "unlink"
                and isinstance(func.value, ast.Name)
            ):
                deleted.add(func.value.id)

        return unsafe & deleted

    offenders: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            offenders |= scan_scope(node)
    return sorted(offenders)


@pytest.mark.parametrize("modname", _SINK_MODULES)
def test_no_sink_deletes_a_path_resolved_by_resolve_data_path(modname):
    """The defect, stated exactly: resolve with the non-containing helper, then delete.

    `resolve_data_path` is documented for "paths read from the database or
    constructed by the system itself" and returns an existing absolute path
    verbatim. That is correct for a READ. For a DELETE it made every one of
    these endpoints an arbitrary-deletion primitive, because the database is
    not a trust boundary while an unauthenticated update can write the column.
    """
    offenders = _deleted_names_resolved_unsafely(modname)
    assert not offenders, (
        f"{modname}: {offenders} are resolved by resolve_data_path and then "
        f"deleted. Use settings.resolve_deletable_path."
    )


def test_the_ast_scan_actually_bites():
    """Negative control for the scan — it must flag the pre-fix shape.

    A source-derived guard fails OPEN when the layout it assumes changes, and
    then asserts nothing forever. This audit hit that twice, once inside a
    reachability guard itself. So the scan is run here against a synthetic
    module containing exactly the code that was removed.
    """
    import ast
    import sys
    import types

    pre_fix = (
        "import shutil\n"
        "def f(dataset, settings):\n"
        "    raw_path = settings.resolve_data_path(dataset.raw_path)\n"
        "    shutil.rmtree(raw_path)\n"
    )
    mod = types.ModuleType("_prefix_sink")
    mod.__dict__["__source__"] = pre_fix
    sys.modules["_prefix_sink"] = mod

    import inspect as _inspect

    original = _inspect.getsource
    try:
        _inspect.getsource = lambda m: pre_fix if m is mod else original(m)
        flagged = _deleted_names_resolved_unsafely("_prefix_sink")
    finally:
        _inspect.getsource = original
        del sys.modules["_prefix_sink"]

    assert flagged == ["raw_path"], (
        f"the scan no longer detects the original defect (got {flagged}) — "
        f"it would pass vacuously on real modules too"
    )
    assert ast  # keep the import meaningful if the body is ever trimmed


def test_every_tainted_sink_module_imports_the_guard():
    """Each module that deletes a stored path must reference the guard by name."""
    import importlib

    missing = []
    for modname in _SINK_MODULES:
        src = inspect.getsource(importlib.import_module(modname))
        if "rmtree" in src or ".unlink()" in src:
            if "resolve_deletable_path" not in src:
                missing.append(modname)
    assert not missing, f"deletes a path but never calls the guard: {missing}"
