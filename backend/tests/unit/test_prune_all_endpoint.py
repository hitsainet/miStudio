"""
The all-trainings sweep must be reachable from the UI, not only the scheduler.

Before this, the only prune routes were per-training:
    GET  /{training_id}/checkpoints/prune-preview
    POST /{training_id}/checkpoints/prune
so reclaiming space meant previewing and pruning one training at a time — "I
don't want to have to delete them one by one manually" (2026-08-28). 84 GB of
prunable checkpoints had accumulated because the sweep only ever ran from the
daily schedule, and that schedule no-ops while `checkpoint_prune_enabled` is
false, which is the shipped default.
"""

import pytest

from src.workers.prune_checkpoints import prune_checkpoints_task


def _reachable():
    """(path, method) pairs reachable through the ASSEMBLED api_router.

    `api_router.routes` holds `_IncludedRouter` wrappers in this FastAPI
    version rather than expanded routes, and `src.main.app` does not expand
    them either — enumerating `app.routes` returns 10 entries and NO endpoint
    paths at all, which reads as "every route is missing". Reach the
    sub-routers through `original_router`, the accessor
    test_jlens_reachable.py already had to work out.
    """
    from src.api.v1.router import api_router

    pairs = set()
    for included in api_router.routes:
        origin = getattr(included, "original_router", None)
        if origin is None:
            continue
        for route in getattr(origin, "routes", []):
            path = getattr(route, "path", None)
            for method in getattr(route, "methods", set()) or set():
                if path:
                    pairs.add((path, method))
    return pairs


def _ordered_paths():
    from src.api.v1.endpoints import trainings

    return [getattr(r, "path", "") for r in trainings.router.routes]


class TestTheRoutesAreRegistered:
    """Reachability: a route nothing serves is not a capability."""

    def test_the_sweep_trigger_is_registered(self):
        assert ("/trainings/checkpoints/prune-all", "POST") in _reachable(), (
            "no route sweeps every training, so the UI can only prune one at a time"
        )

    def test_the_sweep_preview_is_registered(self):
        assert ("/trainings/checkpoints/prune-preview-all", "GET") in _reachable()

    def test_the_literal_routes_precede_the_parameterised_one(self):
        """Declaration order decides matching; a parameterised route declared
        first would swallow "checkpoints" as a training_id."""
        paths = _ordered_paths()
        literal = paths.index("/trainings/checkpoints/prune-all")
        parameterised = next(
            i for i, p in enumerate(paths)
            if p == "/trainings/{training_id}/checkpoints/{checkpoint_id}"
        )
        assert literal < parameterised


class TestForceBypassesOnlyTheEnabledFlag:
    """"Run now" must work while the daily sweep is off, and no further."""

    def test_it_no_ops_when_disabled_and_not_forced(self, monkeypatch):
        from src.services import checkpoint_retention as cr
        from src.workers import prune_checkpoints as mod

        policy = cr.RetentionPolicy(enabled=False, dry_run=True)
        monkeypatch.setattr(mod, "load_policy", lambda db: policy)
        monkeypatch.setattr(
            prune_checkpoints_task, "get_db", lambda: _NullCtx(), raising=False
        )

        result = prune_checkpoints_task.run()
        assert result == {"enabled": False, "trainings_scanned": 0}

    def test_force_runs_even_when_disabled(self, monkeypatch):
        from src.services import checkpoint_retention as cr
        from src.workers import prune_checkpoints as mod

        policy = cr.RetentionPolicy(enabled=False, dry_run=True)
        monkeypatch.setattr(mod, "load_policy", lambda db: policy)
        monkeypatch.setattr(mod, "iter_prunable_trainings", lambda db: [])
        monkeypatch.setattr(
            prune_checkpoints_task, "get_db", lambda: _NullCtx(), raising=False
        )

        result = prune_checkpoints_task.run(force=True)
        assert result.get("enabled") is True, (
            "force did not bypass the enabled flag, so 'Run now' does nothing "
            "on a default installation"
        )

    def test_force_does_NOT_bypass_dry_run(self, monkeypatch):
        """The safety that matters most: forcing must not start deleting."""
        from src.services import checkpoint_retention as cr
        from src.workers import prune_checkpoints as mod

        policy = cr.RetentionPolicy(enabled=False, dry_run=True)
        monkeypatch.setattr(mod, "load_policy", lambda db: policy)
        monkeypatch.setattr(mod, "iter_prunable_trainings", lambda db: [])
        monkeypatch.setattr(
            prune_checkpoints_task, "get_db", lambda: _NullCtx(), raising=False
        )

        result = prune_checkpoints_task.run(force=True)
        assert result.get("dry_run") is True
        assert result.get("deleted", 0) == 0


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
