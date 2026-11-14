"""Task 8 — behaviour that is load-bearing and was pinned by nothing.

Every test here corresponds to a mutation that SURVIVED during the audit: the
line was broken, the suite stayed green, and the line was restored. A surviving
mutation is a test finding, not a code finding, so this file is the fix.

Where a guard was too narrow, it is widened by DERIVING from a registry rather
than extending a hand-list — three guards in this audit had scope narrower than
their claim (BR-002's two modules, `EXPECTED_CALLS`' 16 of 116,
`REQUIRED_TABLES`' 17 of 36), and each was maintained by hand.
"""

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

import httpx
import pytest


# ── 8.1 · MIS-E2E-078 — the steering hook target ───────────────────────────
#
# Hooking the discovered "residual" module lands on a post-attention RMSNorm on
# LFM2, which RENORMALISES the steering vector away: steered == unsteered at
# every dial, silently. Cost a hardware round to find. The correct target is the
# whole decoder layer output (resid_post), which is the point miLLM serves from.

_STEERING_IMPLEMENTATIONS = [
    ("src.services.steering_core", None, "build_steer_generator"),
    ("src.services.steering_service", "SteeringService", "_get_target_module"),
]


@pytest.mark.parametrize("modname, clsname, funcname", _STEERING_IMPLEMENTATIONS)
def test_steering_hooks_the_whole_decoder_layer(modname, clsname, funcname):
    """BOTH implementations. Fixing one and not the other is how this recurs."""
    mod = importlib.import_module(modname)
    owner = getattr(mod, clsname) if clsname else mod
    src = inspect.getsource(getattr(owner, funcname))
    assert "layers_module[" in src, (
        f"{modname}.{funcname} does not resolve its hook target from "
        f"`layers_module` — hooking the inner 'residual' module renormalises "
        f"the steering vector away and produces steered == unsteered"
    )


def test_no_steering_path_hooks_the_inner_residual_module():
    """The specific wrong target, named — as a CALL, not as prose.

    Both modules mention "residual" in comments explaining why that target was
    wrong; a substring scan cannot tell the explanation from the mistake. So
    this looks for an actual `get_hookable_module(..., "residual", ...)` call.

    That call is legitimate in the ATTRIBUTION and FAITHFULNESS paths, which
    read rather than steer. It is not legitimate where a vector is added.
    """
    for modname in ("src.services.steering_core", "src.services.steering_service"):
        tree = ast.parse(inspect.getsource(importlib.import_module(modname)))
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "get_hookable_module"
            and any(
                isinstance(a, ast.Constant) and a.value == "residual"
                for a in node.args
            )
        ]
        assert not offenders, (
            f"{modname} hooks the inner 'residual' module at lines {offenders}; "
            f"additive steering must target the decoder layer output, or the "
            f"RMSNorm renormalises the vector away and steered == unsteered"
        )


# ── 8.2 · MIS-E2E-077 — server-side sensitivity ────────────────────────────

def _sensitive_keys():
    from src.services.app_setting_service import _SENSITIVE_KEYS

    return sorted(_SENSITIVE_KEYS)


@pytest.mark.parametrize("key", _sensitive_keys())
def test_a_known_secret_is_encrypted_even_when_the_client_says_otherwise(key):
    """Parametrized off the registry, so a new secret is covered on the day it
    is added rather than the day someone remembers to extend a list."""
    from src.services.app_setting_service import _SENSITIVE_KEYS

    src = inspect.getsource(
        importlib.import_module("src.services.app_setting_service")
    )
    assert "data.key in _SENSITIVE_KEYS or data.is_sensitive" in src, (
        "sensitivity must be decided server-side; a client-supplied "
        "`is_sensitive: false` would otherwise downgrade a secret to plaintext"
    )
    assert key in _SENSITIVE_KEYS


def test_the_sensitive_key_registry_is_not_empty():
    """Negative control: an empty registry makes the parametrize above vacuous."""
    assert len(_sensitive_keys()) >= 3


# ── 8.3 · MIS-E2E-091 — artifact load is the RCE boundary ──────────────────

def _torch_load_calls():
    """Every `torch.load` in the source tree, with its keywords."""
    root = Path(inspect.getsourcefile(importlib.import_module("src.core.config"))).parents[1]
    found = []
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and getattr(node.func.value, "id", None) == "torch"
            ):
                kw = {k.arg: k.value for k in node.keywords}
                found.append((path.relative_to(root), node.lineno, kw))
    return found


def test_every_torch_load_is_weights_only():
    """A J-lens artifact is an UNTRUSTED FILE — mounted from a directory, or
    pulled from HuggingFace. The unrestricted loader executes pickled code, so
    this keyword is the only thing between an artifact and RCE."""
    calls = _torch_load_calls()
    assert calls, "found no torch.load calls — the scan broke and asserts nothing"

    unsafe = [
        f"{p}:{line}"
        for p, line, kw in calls
        if not (
            isinstance(kw.get("weights_only"), ast.Constant)
            and kw["weights_only"].value is True
        )
    ]
    assert not unsafe, f"torch.load without weights_only=True: {unsafe}"


# ── 8.4 · MIS-E2E-093/-097 — queue routing is by TASK NAME ─────────────────

def test_every_registered_task_routes_to_its_intended_queue():
    """`task_routes` globs match the TASK NAME, not the module path.

    A task registered under a short name silently uses the default queue — which
    is how `train_sae`, a GPU job, landed on `datasets`. Driven off the live
    registry, so a new task is covered without editing a list.
    """
    from src.core.celery_app import celery_app

    registered = [
        name for name in celery_app.tasks
        if name.startswith("src.workers.") or not name.startswith("celery.")
    ]
    assert registered, "no tasks registered — the harness would pass vacuously"

    # The invariant is ROUTING COVERAGE, not name shape: a short name is fine
    # if an explicit entry exists for it. What is not fine is a task no pattern
    # matches, which lands on the default queue silently.
    #
    # This found three live: `train_sae`, `resume_training` (both GPU training
    # jobs) and `delete_extraction`. Thirteen other short-named tasks already
    # had explicit entries — so the list was being maintained, and these were
    # simply missed, with nothing to say so.
    import fnmatch

    routes = celery_app.conf.task_routes or {}
    unrouted = [
        n for n in registered
        if not n.startswith("celery.")
        and not any(fnmatch.fnmatch(n, pattern) for pattern in routes)
    ]
    assert not unrouted, (
        f"no `task_routes` pattern matches these tasks, so they fall to the "
        f"default queue: {sorted(unrouted)}. Remember the globs match the TASK "
        f"NAME, not the module path."
    )


def test_gpu_tasks_are_not_on_the_datasets_queue():
    """The specific regression: a GPU job on an I/O queue starves both."""
    from src.core.celery_app import celery_app

    router = celery_app.conf.task_routes or {}
    for name in celery_app.tasks:
        if "train" in name or "steering" in name:
            for pattern, route in router.items():
                if pattern.endswith("*") and name.startswith(pattern[:-1]):
                    assert route.get("queue") != "datasets", (
                        f"{name} routes to the datasets queue"
                    )


# ── 8.9 · MIS-E2E-137/-142 — which emit failures are retried ───────────────

@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("slow"),
        httpx.ConnectTimeout("slow"),
        httpx.RemoteProtocolError("half-closed keep-alive"),
        httpx.ConnectError("connection refused"),
    ],
)
def test_transport_failures_are_retried(monkeypatch, exc):
    """All four may mean the request never arrived.

    The events configured with retries are the TERMINAL ones —
    `steering:completed` and friends — where a silent drop leaves the UI showing
    a finished job as still running, forever.
    """
    from src.workers import websocket_emitter as we

    attempts = {"n": 0}

    class _Resp:
        status_code = 200

    class _PooledClient:
        """Stands in for the module's POOLED client, which is the real seam.

        `emit_progress` calls `_get_http_client().post(...)`, not
        `httpx.Client(...)`, so patching the constructor patches nothing — the
        pooled instance was created at import.
        """

        def post(self, *a, **k):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise exc
            return _Resp()

    monkeypatch.setattr(we, "_get_http_client", lambda: _PooledClient())

    # `time` is imported INSIDE emit_progress, so patch the stdlib module the
    # function will import rather than an attribute this module does not have.
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_: None)

    assert we.emit_progress("steering/t1", "steering:completed", {}, retries=3) is True
    assert attempts["n"] == 2, (
        f"{type(exc).__name__} was not retried — it was abandoned on the first "
        f"attempt and the terminal event was dropped"
    )


def test_an_http_error_status_is_NOT_retried(monkeypatch):
    """The other direction. The server answered; repeating a rejected request
    is not a retry, it is a loop."""
    from src.workers import websocket_emitter as we

    attempts = {"n": 0}

    class _Resp:
        status_code = 500

    class _PooledClient:
        def post(self, *a, **k):
            attempts["n"] += 1
            return _Resp()

    monkeypatch.setattr(we, "_get_http_client", lambda: _PooledClient())

    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_: None)

    assert we.emit_progress("steering/t1", "steering:completed", {}, retries=3) is False
    assert attempts["n"] == 1


# ── 8.7 · MIS-E2E-090 — BR-002's guard was two modules of a package ────────

_FORBIDDEN_BAND_CONSTANTS = {38, 40, 90, 92}


def _jlens_modules():
    """Every module in the jlens surface, DERIVED — not a hand-written tuple.

    BR-002 says "no band constant ANYWHERE, by construction". The guard scanned
    a hardcoded two-module tuple, so injecting the literals into
    `jlens_band_service` — a sibling in the same package, and a plausible place
    for someone to add a default — left the suite green (mutation M13).
    """
    import src.ml as ml_pkg
    import src.services as svc_pkg

    names = []
    for pkg in (ml_pkg, svc_pkg):
        for mod in pkgutil.iter_modules(pkg.__path__):
            if "jlens" in mod.name:
                names.append(f"{pkg.__name__}.{mod.name}")
    return sorted(names)


def test_the_jlens_module_scan_finds_the_whole_surface():
    """Negative control for the derivation. A scan that discovers nothing
    asserts nothing, and that is how the original guard failed."""
    mods = _jlens_modules()
    assert len(mods) >= 5, f"only discovered {mods} — the derivation broke"
    assert any("band" in m for m in mods)


@pytest.mark.parametrize("modname", _jlens_modules())
def test_no_band_constant_anywhere_in_the_jlens_surface(modname):
    """BR-002, enforced across the package rather than two chosen files."""
    mod = importlib.import_module(modname)
    src = inspect.getsource(mod)
    tree = ast.parse(src)

    # Docstrings legitimately NAME the published figures in order to forbid
    # them; only executable constants are violations.
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                doc_lines.update(range(node.body[0].lineno,
                                       (node.body[0].end_lineno or node.body[0].lineno) + 1))

    offenders = [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and node.value in _FORBIDDEN_BAND_CONSTANTS
        and node.lineno not in doc_lines
    ]
    assert not offenders, (
        f"{modname} contains band constants {offenders}. BR-002: the published "
        f"boundaries were measured on ONE model; miStudio draws no bands unless "
        f"a band report exists for the model in front of you."
    )


# ── 8.5 · MIS-E2E-112 — IDOR on checkpoint deletion ────────────────────────

def test_the_checkpoint_delete_route_verifies_ownership():
    """A checkpoint must not be deletable through an unrelated training's URL.

    The guard exists; nothing asserted it, so removing it would have been
    invisible. `DELETE /trainings/{tid}/checkpoints/{cid}` takes both ids and
    the checkpoint id is globally unique — so without the cross-check, any
    training id at all would authorise deleting any checkpoint.
    """
    import ast

    from src.api.v1.endpoints import trainings

    tree = ast.parse(inspect.getsource(trainings))
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == "delete_checkpoint"
        ):
            target = node
            break
    assert target is not None, "delete_checkpoint route not found"

    body = ast.unparse(target)
    assert "checkpoint.training_id != training_id" in body, (
        "the route does not verify the checkpoint belongs to the training in "
        "the URL — any training id would authorise deleting any checkpoint"
    )
    # And it must REFUSE, not merely notice.
    assert "raise HTTPException" in body


def test_the_ownership_check_precedes_the_delete():
    """Order matters: noticing after the row is gone is not a guard."""
    import ast

    from src.api.v1.endpoints import trainings

    tree = ast.parse(inspect.getsource(trainings))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == "delete_checkpoint"
        ):
            lines = ast.unparse(node).splitlines()
            check = next(
                i for i, l in enumerate(lines)
                if "checkpoint.training_id != training_id" in l
            )
            delete = next(
                i for i, l in enumerate(lines)
                if "CheckpointService.delete_checkpoint" in l
            )
            assert check < delete, (
                "the ownership check runs after the deletion — it reports a "
                "breach instead of preventing one"
            )
            return
    raise AssertionError("delete_checkpoint route not found")
