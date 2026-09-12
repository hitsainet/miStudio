"""MIS-E2E-106 — a PATCH body must not carry worker-owned fields.

Three `*Update` schemas exposed the row's lifecycle `status` and were
blind-`setattr`'d onto the ORM. `TrainingUpdate` additionally exposed every
metric the worker owns.

    PATCH /api/trainings/{id} {"status": "completed"}

against a running job did three things at once:

  1. Unlocked SAE import from a partial checkpoint. `sae_manager_service`
     gates solely on `status != COMPLETED`, so an SAE is built from whatever
     step the run reached and imported as a finished artifact — with **no
     `finalized_from_step` marker**, the one signal Feature 21 added to
     distinguish a salvaged run from a complete one.
  2. Made the job uncancellable. `cancel_training` returns None for any
     terminal status, so it silently no-ops while the worker keeps the GPU.
  3. Falsified the record: `progress: 100`, `current_loss: 0.01`,
     `current_dead_neurons: 0` were writable in the same request.

The fields could not simply be deleted — internal callers write them through
these very schemas (`datasets.py` sets `status=PROCESSING` when it queues
tokenization). So the REQUEST has its own narrow model and the route binds that,
while the service enforces an allow-list of its own. Both halves are pinned
here: a narrow route with an open sink is one new caller away from the bug.
"""

import inspect

import pytest
from pydantic import ValidationError

from src.schemas.dataset import DatasetPatchRequest, DatasetUpdate
from src.schemas.model import ModelPatchRequest, ModelUpdate


# ── The request schemas are narrow ─────────────────────────────────────────

@pytest.mark.parametrize(
    "field, value",
    [
        ("status", "ready"),
        ("progress", 100.0),
        ("raw_path", "/data"),
        ("num_samples", 999_999),
        ("size_bytes", 1),
        ("error_message", "none"),
        ("metadata", {"task_id": "hijacked"}),
    ],
)
def test_dataset_patch_refuses_worker_owned_fields(field, value):
    with pytest.raises(ValidationError):
        DatasetPatchRequest(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("status", "ready"),
        ("progress", 100.0),
        ("file_path", "/data"),
        ("quantized_path", "/data"),
        ("error_message", "none"),
        ("architecture", "llama"),
        ("params_count", 1),
        ("metadata", {}),
    ],
)
def test_model_patch_refuses_worker_owned_fields(field, value):
    with pytest.raises(ValidationError):
        ModelPatchRequest(**{field: value})


def test_the_patch_schemas_still_allow_what_a_user_may_edit():
    """A refusal-only test would pass against a schema that refuses everything."""
    assert DatasetPatchRequest(name="My dataset").name == "My dataset"
    assert DatasetPatchRequest(tokenization_filter_enabled=True).tokenization_filter_enabled
    assert ModelPatchRequest(name="My model").name == "My model"


def test_the_internal_schemas_deliberately_keep_those_fields():
    """The workers write through `*Update`; narrowing THEM breaks the writers.

    Negative control for the direction of the fix. An over-eager cleanup that
    strips these from the internal schema too would fail 37 tests, and this
    records why it must not happen.
    """
    assert "status" in DatasetUpdate.model_fields
    assert "raw_path" in DatasetUpdate.model_fields
    assert "status" in ModelUpdate.model_fields
    assert "file_path" in ModelUpdate.model_fields


# ── The routes bind the narrow model ───────────────────────────────────────

def test_patch_routes_bind_the_request_schema_not_the_internal_one():
    """A narrow schema that no route uses protects nothing."""
    import typing

    from src.api.v1.endpoints.datasets import update_dataset
    from src.api.v1.endpoints.models import update_model

    for fn, expected, forbidden in [
        (update_dataset, DatasetPatchRequest, DatasetUpdate),
        (update_model, ModelPatchRequest, ModelUpdate),
    ]:
        annotations = typing.get_type_hints(fn)
        bound = set(annotations.values())
        assert expected in bound, f"{fn.__name__} does not bind {expected.__name__}"
        assert forbidden not in bound, (
            f"{fn.__name__} still binds the internal {forbidden.__name__} — "
            f"a request body can set every worker-owned field"
        )


def test_patch_training_route_is_gone():
    """`TrainingUpdate` is entirely worker-owned, so nothing was left of it.

    The route had no caller: the frontend issues no PATCH, no MCP tool wraps it,
    no test exercised it.
    """
    from src.api.v1.endpoints import trainings

    assert not hasattr(trainings, "update_training"), (
        "PATCH /trainings/{id} accepted only lifecycle and metric fields; "
        "removing the unsafe ones leaves nothing to bind"
    )
    routes = [
        r for r in trainings.router.routes
        if "PATCH" in getattr(r, "methods", set())
    ]
    assert not routes, f"a PATCH route survives on trainings: {routes}"


# ── The sinks enforce their own allow-list ─────────────────────────────────

@pytest.mark.parametrize(
    "module",
    [
        "src.services.training_service",
        "src.services.model_service",
        "src.services.dataset_service",
    ],
)
def test_the_sink_has_an_allow_list_not_a_blind_loop(module):
    """Closes the NEXT hole — a new field or a new caller, not today's body."""
    import importlib

    body = inspect.getsource(importlib.import_module(module))
    assert "_WRITABLE" in body, f"{module} still applies a blind setattr loop"
    assert "refused unknown fields" in body, (
        f"{module} computes an allow-list but does not act on it"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service, method, rogue",
    [
        ("src.services.dataset_service", "DatasetService", {"name": "ok", "id": "x"}),
        ("src.services.model_service", "ModelService", {"name": "ok", "id": "x"}),
        (
            "src.services.training_service",
            "TrainingService",
            {"status": None, "celery_task_id": "x"},
        ),
    ],
)
async def test_the_sink_refuses_a_field_outside_its_allow_list(
    service, method, rogue, monkeypatch
):
    """Exercise the guard, not merely its presence in the source.

    The row lookup is stubbed so the check is reached without a database; if the
    guard were absent the call would proceed to `setattr` and this test would
    fail with something other than the expected ValueError.
    """
    import importlib

    mod = importlib.import_module(service)
    svc = getattr(mod, method)
    fn = getattr(svc, {"DatasetService": "update_dataset",
                       "ModelService": "update_model",
                       "TrainingService": "update_training"}[method])

    class _Payload:
        def model_dump(self, **kw):
            return dict(rogue)

    class _Row:
        extra_metadata = None

    async def _get(*a, **k):
        return _Row()

    class _Result:
        @staticmethod
        def scalar_one_or_none():
            return _Row()

    class _DB:
        """Enough of a session to reach the guard, and no further.

        `commit` raises: if the allow-list were absent the call would fall
        through to `setattr` and then here, so this test cannot pass by
        accident on a missing guard — it would fail with the wrong exception.
        """

        async def execute(self, *a, **k):
            return _Result()

        async def commit(self):
            raise AssertionError("reached commit — the allow-list did not fire")

        async def refresh(self, *a, **k):
            raise AssertionError("reached refresh — the allow-list did not fire")

    # Sinks that go through a service getter rather than db.execute.
    # monkeypatch, NOT setattr: a bare setattr here leaked into every later
    # test in the session and turned `get_dataset` into a stub returning a
    # blank row — two unrelated tests went red and only when run together.
    for getter in ("get_dataset", "get_model", "get_training"):
        if hasattr(svc, getter):
            monkeypatch.setattr(svc, getter, staticmethod(_get))

    with pytest.raises(ValueError, match="refused unknown fields"):
        await fn(_DB(), "some-id", _Payload())
